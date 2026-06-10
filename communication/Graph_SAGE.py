# -*- coding: utf-8 -*-
"""
Graph_SAGE backend (unified interface with Graph_GAT.GraphSAGE_sup)
Fixed:
1. Node initialization (fixes NetworkXError)
2. Output dimension padding (20 -> 32) (fixes ReplayMemory shape error)
"""
from __future__ import annotations

import os
from typing import List, Tuple, Optional

import numpy as np
import networkx as nx
from tf_runtime import configure_tensorflow_runtime


configure_tensorflow_runtime()
import tensorflow as tf
configure_tensorflow_runtime(tf)

from Environment import * # noqa: F401
from model_Graph import GraphModel


class GraphSAGE_sup(object):
    def __init__(self,
                 environment,
                 distance_threshold: float = 150.0,
                 lr: float = 5e-4,
                 gat_train_interval: Optional[int] = None, 
                 grad_clip: float = 5.0):
        self.env = environment
        self.weight_dir = 'weight'
        os.makedirs(self.weight_dir, exist_ok=True)

        try:
            from tensorflow.keras import mixed_precision as mp
            mp.set_global_policy('float32')
        except Exception:
            pass

        self.distance_threshold = float(distance_threshold)
        self.lr = float(lr)
        self.grad_clip = float(grad_clip)
        self.gat_train_interval = int(gat_train_interval) if gat_train_interval is not None else 50

        # === 内部依然使用 20 维，以匹配 channel_reward ===
        self.G_model = GraphModel(sample_num=5, depth=2, dims=20, gcn=True, concat=True)
        self.G_model_target = GraphModel(sample_num=5, depth=2, dims=20, gcn=True, concat=True)
        self.update_target_network()

        # === 显式定义 out_dim = 32，虽然内部是 20，但我们会填充到 32 ===
        self.out_dim = 32 

        self.num_vehicle = len(getattr(self.env, "vehicles", []))
        self.features = np.zeros((max(1, 3 * self.num_vehicle), 60), dtype=np.float32)

        self.order_nodes: List[int] = list(range(max(1, 3 * self.num_vehicle)))
        self.link = np.zeros((max(1, 3 * self.num_vehicle), 2), dtype=np.int32)
        self.graph = nx.Graph()
        self.gs_losses: List[float] = []
        self.gat_loss_history: List[Tuple[int, float]] = []
        self.tb_writer = None 

        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 2000
        self.learning_rate_minimum = 1e-4

        self.compile_model()

        self.num_V2V_list = np.zeros((self.num_vehicle, self.num_vehicle), dtype=np.float32)

        # === [FIX 1] 初始化时立即添加节点，防止 warmup 报错 ===
        if self.graph is None or len(self.graph.nodes) == 0:
            self.graph.add_nodes_from(self.order_nodes)

    def compile_model(self):
        base = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.lr,
            decay_steps=int(self.learning_rate_decay_step),
            decay_rate=float(self.learning_rate_decay),
            staircase=True,
        )

        class ClippedSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, inner, min_lr: float):
                self.inner = inner
                self.min_lr = tf.constant(min_lr, dtype=tf.float32)

            def __call__(self, step):
                return tf.maximum(self.inner(step), self.min_lr)

            def get_config(self):
                return {
                    "inner": tf.keras.saving.serialize_keras_object(self.inner),
                    "min_lr": float(self.min_lr.numpy())
                }

        lr_schedule = ClippedSchedule(base, float(self.learning_rate_minimum))
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, rho=0.95, epsilon=1e-7)
        self.G_model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

    def update_target_network(self):
        try:
            self.G_model_target.set_weights(self.G_model.get_weights())
        except Exception:
            pass

    def build_graph(self, num_V2V_list: np.ndarray) -> Tuple[nx.Graph, List[int], None]:
        vehicles = getattr(self.env, "vehicles", [])
        nveh = len(vehicles)
        n_nodes = 3 * nveh
        G = nx.Graph()
        order_nodes = list(range(n_nodes))
        G.add_nodes_from(order_nodes)

        pos = [np.asarray(getattr(v, "position", (0.0, 0.0)), dtype=np.float32) for v in vehicles]

        for i in range(nveh):
            for k in range(i + 1, nveh):
                d = float(np.linalg.norm(pos[i] - pos[k]))
                if d <= self.distance_threshold:
                    for j in range(3):
                        for t in range(3):
                            u = 3 * i + j
                            v = 3 * k + t
                            G.add_edge(u, v)
        return G, order_nodes, None

    def load_graph(self, graph: nx.Graph, order_nodes: List[int]):
        self.graph = graph
        self.order_nodes = list(order_nodes)
        n_nodes = len(self.order_nodes)
        link = np.zeros((n_nodes, 2), dtype=np.int32)
        for nid in range(n_nodes):
            link[nid, 0] = nid // 3
            link[nid, 1] = nid % 3
        self.link = link

    def fetch_batch(self, order_nodes: List[int], idx_list: List[int]):
        S1 = getattr(self.G_model, "sample_num", 5) or 5
        S2 = S1 

        index_map = {n: i for i, n in enumerate(order_nodes)}
        B = len(idx_list)
        f1 = np.zeros((B, S1), dtype=np.int32)
        f2 = np.zeros((B, S1, S2), dtype=np.int32)
        w1 = np.zeros((B, S1), dtype=np.float32)
        w2 = np.zeros((B, S1, S2), dtype=np.float32)

        for bi, nid in enumerate(idx_list):
            node_label = order_nodes[nid]
            
            # [FIX 1] 安全检查
            if self.graph.has_node(node_label):
                neighs = list(nx.neighbors(self.graph, node_label))
            else:
                neighs = []

            neigh_idx = [index_map.get(x, nid) for x in neighs] or [nid]

            if len(neigh_idx) >= S1:
                chosen = neigh_idx[:S1]
            else:
                chosen = neigh_idx + [nid] * (S1 - len(neigh_idx))
            f1[bi, :] = np.asarray(chosen, dtype=np.int32)
            w1_count = max(1, len(neighs))
            w1_val = 1.0 / float(w1_count)
            w1[bi, :] = w1_val

            for s, n1 in enumerate(chosen):
                n1_label = order_nodes[n1]
                # [FIX 1] 安全检查
                if self.graph.has_node(n1_label):
                    neighs2 = list(nx.neighbors(self.graph, n1_label))
                else:
                    neighs2 = []

                neigh2_idx = [index_map.get(x, n1) for x in neighs2] or [n1]
                if len(neigh2_idx) >= S2:
                    chosen2 = neigh2_idx[:S2]
                else:
                    chosen2 = neigh2_idx + [n1] * (S2 - len(neigh2_idx))
                f2[bi, s, :] = np.asarray(chosen2, dtype=np.int32)
                w2_count = max(1, len(neighs2))
                w2_val = 1.0 / float(w2_count)
                w2[bi, s, :] = w2_val

        return f1, f2, w1, w2

    def _forward_all(self, training: bool = False) -> tf.Tensor:
        if self.features is None: raise ValueError("features is None")
        if self.graph is None or self.order_nodes is None: raise ValueError("graph/order_nodes is None")
        N = int(self.features.shape[0])
        idx_all = list(range(N))
        f1, f2, w1, w2 = self.fetch_batch(self.order_nodes, idx_all)
        inputs = (
            tf.convert_to_tensor(self.features, dtype=tf.float32),
            tf.convert_to_tensor(idx_all, dtype=tf.int32),
            tf.convert_to_tensor(f1, dtype=tf.int32),
            tf.convert_to_tensor(f2, dtype=tf.int32),
            tf.convert_to_tensor(w1, dtype=tf.float32),
            tf.convert_to_tensor(w2, dtype=tf.float32),
        )
        out = self.G_model(inputs, training=training) # Shape: [N, 20]
        out = tf.cast(out, tf.float32)

        # === [FIX 2] 填充维度 20 -> 32 ===
        # ReplayMemory 期望 32 维，所以我们补 12 个 0
        paddings = [[0, 0], [0, 12]] # 在第二个维度补 12 列
        out_padded = tf.pad(out, paddings, "CONSTANT") # Shape: [N, 32]
        
        return out_padded

    def use_GraphSAGE(self,
                      channel_reward: np.ndarray,
                      step: int,
                      idx: List[int],
                      train_flag: bool = True):
        first_order_neighs, second_order_neighs, s1_weights, s2_weights = \
            self.fetch_batch(self.order_nodes, idx)

        inputs = (
            tf.convert_to_tensor(self.features, dtype=tf.float32),
            tf.convert_to_tensor(idx, dtype=tf.int32),
            tf.convert_to_tensor(first_order_neighs, dtype=tf.int32),
            tf.convert_to_tensor(second_order_neighs, dtype=tf.int32),
            tf.convert_to_tensor(s1_weights, dtype=tf.float32),
            tf.convert_to_tensor(s2_weights, dtype=tf.float32),
        )

        if train_flag and step % self.gat_train_interval == 0 and step > 0:
            with tf.GradientTape() as tape:
                emb = self.G_model(inputs, training=True)
                emb_t = self.G_model_target(inputs, training=False)
                emb = tf.cast(emb, tf.float32)
                emb_t = tf.cast(emb_t, tf.float32)

                labels = tf.gather(tf.convert_to_tensor(channel_reward, tf.float32),
                                   tf.convert_to_tensor(idx, tf.int32), axis=0)
                # 训练 Loss 依然使用原始的 20 维，因为 labels 也是 20 维
                target = 0.5 * emb_t + 0.5 * labels
                loss = tf.reduce_mean(tf.square(emb - target))

            vars_ = self.G_model.trainable_variables
            grads = tape.gradient(loss, vars_)
            if self.grad_clip and self.grad_clip > 0:
                grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
            self.G_model.optimizer.apply_gradients(zip(grads, vars_))

            lv = float(loss.numpy())
            self.gs_losses.append(lv)
            self.gat_loss_history.append((int(step), lv))

            if self.tb_writer is not None and (step % 100 == 0):
                with self.tb_writer.as_default():
                    tf.summary.scalar('SAGE/loss', lv, step=step)

            emb_out = emb
        else:
            emb_out = self.G_model(inputs, training=False)
            emb_out = tf.cast(emb_out, tf.float32)

        if step % 100 == 99:
            self.update_target_network()
            try:
                self.G_model.save_weights(os.path.join(self.weight_dir, 'GNN_weights_sage.h5'))
            except Exception:
                pass

        # === [FIX 2] 返回给 Agent 前，填充 Numpy 数组 20 -> 32 ===
        result = emb_out.numpy() # Shape [B, 20]
        # 在轴 1 (列) 上补 0：左边补0个，右边补12个
        result_padded = np.pad(result, ((0, 0), (0, 12)), mode='constant') # Shape [B, 32]
        
        return result_padded
