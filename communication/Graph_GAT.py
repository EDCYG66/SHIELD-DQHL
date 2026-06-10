#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph_GAT.py (Fixed GATv2 + InputNorm + Trainable + Repruning)
修复了 use_low_rank 参数传递错误，并整合了 GATv2 强力注意力机制。
已修复自环边丢失与dst目标注意力错位Bug。
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from tf_runtime import configure_tensorflow_runtime


configure_tensorflow_runtime()
import tensorflow as tf
configure_tensorflow_runtime(tf)
try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None

from graph_edges import build_link_edge_index

def glorot(shape, name=None):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    return tf.Variable(tf.random.uniform(shape, -init_range, init_range), name=name)

class MultiHeadAttention(tf.keras.layers.Layer):
    """
    GATv2 Attention Implementation
    Paper: "How Attentive are Graph Attention Networks?" (ICLR 2022)
    Logic: a^T LeakyReLU( W_src * h_i + W_dst * h_j )
    """
    def __init__(self, in_dim, out_dim, heads=2, attn_dropout=0.0, **kwargs):
        # --- 关键修复：从 kwargs 中移除旧版参数，防止传给 super() 报错 ---
        kwargs.pop('use_low_rank', None)
        kwargs.pop('low_rank_k', None)
        
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.attn_dropout = attn_dropout
        
        # GATv2 参数
        self.W_src = glorot((in_dim, heads * out_dim), "W_src")
        self.W_dst = glorot((in_dim, heads * out_dim), "W_dst")
        self.W_val = glorot((in_dim, heads * out_dim), "W_val") # Value projection
        self.attn_vec = glorot((1, heads, out_dim), "attn_vec") # Attention vector
        
        self.leaky_alpha = 0.2

    def call(self, x, edge_index, training=False):
        # x shape: [N, in_dim]
        N = tf.shape(x)[0]
        H = self.heads
        D = self.out_dim
        
        # 1. Projections
        h_s = tf.reshape(tf.matmul(x, self.W_src), (N, H, D))
        h_d = tf.reshape(tf.matmul(x, self.W_dst), (N, H, D))
        h_v = tf.reshape(tf.matmul(x, self.W_val), (N, H, D))
        
        src = edge_index[0]
        dst = edge_index[1]
        
        # 2. Gather pairs
        feat_s = tf.gather(h_s, src) # [E, H, D]
        feat_d = tf.gather(h_d, dst) # [E, H, D]
        
        # 3. GATv2 Score: a^T * LeakyReLU(h_s + h_d)
        middle = tf.nn.leaky_relu(feat_s + feat_d, alpha=self.leaky_alpha)
        attn_scores = tf.reduce_sum(middle * self.attn_vec, axis=-1) # [E, H]
        
        # 4. Softmax (Numerical Stable)
        attn_scores_t = tf.transpose(attn_scores, [1, 0]) # [H, E]
        
        seg_ids = tf.tile(tf.expand_dims(dst, 0), [H, 1])
        head_offsets = tf.range(H, dtype=dst.dtype)[:, None] * N
        seg_ids = tf.reshape(seg_ids + head_offsets, [-1])
        scores_flat = tf.reshape(attn_scores_t, [-1])
        seg_count = H * N
        seg_max = tf.math.unsorted_segment_max(scores_flat, seg_ids, seg_count)
        exp_scores = tf.exp(scores_flat - tf.gather(seg_max, seg_ids))
        seg_sum = tf.math.unsorted_segment_sum(exp_scores, seg_ids, seg_count)
        attn = tf.transpose(tf.reshape(exp_scores / (tf.gather(seg_sum, seg_ids) + 1e-9), [H, -1]), [1, 0])
        
        if training and self.attn_dropout > 0.0:
            attn = tf.nn.dropout(attn, rate=self.attn_dropout)
            
        # 5. Aggregate
        v_src = tf.gather(h_v, src) # [E, H, D]
        messages = tf.expand_dims(attn, -1) * v_src # [E, H, D]
        messages_flat = tf.reshape(messages, (tf.shape(messages)[0], H * D))
        out_sum = tf.math.unsorted_segment_sum(messages_flat, dst, N)
        
        return out_sum, attn

class GraphGAT(tf.keras.Model):
    def __init__(self,
                 num_nodes: int,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 heads: int = 2,
                 top_k: int = 6,
                 prune_mode: str = "distance",
                 add_self_loop: bool = True,
                 attn_dropout: float = 0.0,
                 use_low_rank: bool = False,
                 low_rank_k: int = 16,
                 lr: float = 5e-4,
                 gat_train_interval: int = 20,
                 grad_clip: float = 5.0,
                 # CLI 可透传参数
                 reprune_every: int = 300,
                 hysteresis_keep: float = 0.5,
                 reprune_start_step: int = 600,
                 reg_attn_w: float = 1e-3,
                 enable_reprune: bool = True,
                 use_proximity_edges: bool = True,
                 proximity_radius: float = 180.0,
                 max_proximity_neighbors: int = 6):
        super().__init__()
        # Shapes & params
        self.num_nodes = int(num_nodes)
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.heads = int(heads)
        self.top_k = int(top_k)
        self.prune_mode = prune_mode
        self.add_self_loop = bool(add_self_loop)
        self.grad_clip = float(grad_clip)
        self.use_proximity_edges = bool(use_proximity_edges)
        self.proximity_radius = float(proximity_radius)
        self.max_proximity_neighbors = int(max_proximity_neighbors)

        # Runtime state
        self.features = np.zeros((self.num_nodes, self.in_dim), dtype=np.float32)
        self.edge_index = None
        self.edge_index_tf = None
        self.order_nodes: List[str] = [str(i) for i in range(self.num_nodes)]
        self.node_positions: Optional[np.ndarray] = None
        self.link = np.zeros((self.num_nodes, 2), dtype=np.int32)
        self._cache_emb = None
        self._cache_emb_np = None
        self.gat_loss_history: List[Tuple[int, float]] = []

        # Repruning state
        self._prev_keep: Optional[np.ndarray] = None
        self._last_reprune_step: Optional[int] = None
        self.reprune_every = int(reprune_every)
        self.reprune_start_step = int(reprune_start_step)
        self.hysteresis_keep = float(hysteresis_keep)
        self.enable_reprune = bool(enable_reprune)

        # Layers
        # --- 关键修复：不再传递 use_low_rank 等旧参数 ---
        self.attn1 = MultiHeadAttention(in_dim, hidden_dim, heads=heads,
                                        attn_dropout=attn_dropout)
        
        self.attn2 = MultiHeadAttention(hidden_dim * heads, out_dim, heads=1,
                                        attn_dropout=attn_dropout)
        
        self.act = tf.keras.layers.ELU()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.skip1 = tf.keras.layers.Dense(hidden_dim * heads, use_bias=False)
        self.skip2 = tf.keras.layers.Dense(out_dim, use_bias=False)
        # 新增 Input Norm
        self.input_norm = tf.keras.layers.LayerNormalization()

        # Prediction Head
        self.head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.out_dim, activation='elu'),
            tf.keras.layers.Dense(20, activation=None)
        ])
        self.opt = tf.keras.optimizers.Adam(learning_rate=float(lr))
        self.reg_attn_w = float(reg_attn_w)
        self.gat_train_interval = int(gat_train_interval)

        # Conflict target
        self._conflict_map: Optional[np.ndarray] = None
        self._conflict_w: float = 0.02

    # -------- Graph build/load (保持不变) --------

    def build_graph(self, num_V2V_list: np.ndarray) -> Tuple[np.ndarray, List[str], np.ndarray]:
        edge_index, order_nodes, link_array = build_link_edge_index(
            num_V2V_list,
            node_positions=self.node_positions,
            use_proximity_edges=self.use_proximity_edges,
            proximity_radius=self.proximity_radius,
            max_proximity_neighbors=self.max_proximity_neighbors,
            add_self_loop=self.add_self_loop,
        )
        self.link = link_array.copy()
        self.order_nodes = order_nodes
        self.load_graph(edge_index, order_nodes)
        return edge_index, order_nodes, link_array

    def load_graph(self, nx_graph, node_order: List[str]):
        self.order_nodes = node_order
        if isinstance(nx_graph, np.ndarray):
            edge_array = np.asarray(nx_graph, dtype=np.int32).reshape(2, -1)
        else:
            id_map = {lab: i for i, lab in enumerate(node_order)}
            edges: List[Tuple[int, int]] = []
            for u, v in nx_graph.edges():
                if u in id_map and v in id_map:
                    su = id_map[u]; tv = id_map[v]
                    edges.append((su, tv)); edges.append((tv, su))
            if self.add_self_loop:
                for i in range(len(node_order)):
                    edges.append((i, i))
            edge_array = np.array(edges, dtype=np.int32).T
        if self.prune_mode == "distance" and self.node_positions is not None and self.top_k > 0:
            edge_array = self._prune_top_k_distance(edge_array, self.top_k)
        self.edge_index = edge_array
        self.edge_index_tf = tf.convert_to_tensor(self.edge_index, tf.int32)
        self._prev_keep = None
        self._last_reprune_step = None
        self._cache_emb = None
        self._cache_emb_np = None

    def update_positions(self, positions: np.ndarray):
        pos = np.asarray(positions, dtype=np.float32)
        if pos.shape[0] != self.num_nodes:
            return
        self.node_positions = pos
        self._cache_emb = None
        self._cache_emb_np = None

    def _pairwise_distance(self):
        pos = self.node_positions
        diff = pos[:, None, :] - pos[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))

    def _ensure_self_loops(self, edge_index: np.ndarray) -> np.ndarray:
        if not self.add_self_loop:
            return edge_index
        N = self.num_nodes
        existing = set(map(tuple, edge_index.T.tolist()))
        extra = []
        for i in range(N):
            if (i, i) not in existing:
                extra.append((i, i))
        if extra:
            extra_arr = np.array(extra, dtype=np.int32).T
            edge_index = np.concatenate([edge_index, extra_arr], axis=1)
        return edge_index

    def _prune_top_k_distance(self, edge_index: np.ndarray, k: int):
        if self.node_positions is None:
            return edge_index
        dist = self._pairwise_distance()
        keep_mask = np.zeros(edge_index.shape[1], dtype=bool)
        by_src: Dict[int, List[Tuple[float, int]]] = {}
        for eidx in range(edge_index.shape[1]):
            s = int(edge_index[0, eidx]); t = int(edge_index[1, eidx])
            by_src.setdefault(s, []).append((float(dist[s, t]), eidx))
        for s, lst in by_src.items():
            lst.sort(key=lambda x: x[0])
            for _, e in lst[:k]:
                keep_mask[e] = True
        pruned = edge_index[:, keep_mask]
        pruned = self._ensure_self_loops(pruned)
        return pruned

    # -------- adaptive attention pruning --------

    def adaptive_reprune(self, step: int, k: int = 6, hysteresis_keep: float = None):
        if hysteresis_keep is None:
            hysteresis_keep = self.hysteresis_keep

        # 关键修复：让 CLI 的 --disable-reprune / enable_reprune 开关真正生效
        if not self.enable_reprune:
            return

        if self.top_k <= 0 or self.edge_index is None:
            return
        if step < self.reprune_start_step:
            return
        if self._last_reprune_step is not None and (step - self._last_reprune_step) < self.reprune_every:
            return

        feats = tf.convert_to_tensor(self.features, tf.float32)
        edge_ix = self.edge_index_tf
        if edge_ix is None:
            edge_ix = tf.convert_to_tensor(self.edge_index, tf.int32)
            self.edge_index_tf = edge_ix
        _, attn = self._forward_tf_infer(feats, edge_ix)
        attn_mean = tf.reduce_mean(attn, axis=1).numpy()

        src = self.edge_index[0]
        dst = self.edge_index[1]
        keep = np.zeros(self.edge_index.shape[1], dtype=bool)
        
        # 1. 强制无条件保留自环边，防止节点在消息传递中丢失自身状态
        for eidx in range(len(src)):
            if src[eidx] == dst[eidx]:
                keep[eidx] = True

        from collections import defaultdict
        bucket: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
        
        # 2. 按照目标节点（dst，即信号接收方）进行分组
        # 因为注意力分数是在 dst 维度做 Softmax 的，只有同一 dst 的入边才具有可比性
        for eidx, d in enumerate(dst):
            # 排除自环边参与竞争，为干扰边腾出 Top-K 的名额
            if src[eidx] != dst[eidx]:
                bucket[int(d)].append((float(attn_mean[eidx]), int(eidx)))
                
        # 3. 筛选对当前接收方影响最大的 Top-K 干扰源
        for d, lst in bucket.items():
            lst.sort(key=lambda x: x[0], reverse=True)
            for _, e in lst[:k]:
                keep[e] = True

        # 直接应用筛选后的掩码更新边索引
        self.edge_index = self.edge_index[:, keep]
        # 再次确保自环边结构完整
        self.edge_index = self._ensure_self_loops(self.edge_index)
        self.edge_index_tf = tf.convert_to_tensor(self.edge_index, tf.int32)
        self._last_reprune_step = int(step)

    # -------- forward & train --------

    def _forward_impl(self, feats, edge_index, training: bool):
        # 核心优化：输入归一化，解决 Attention 对数值敏感问题
        feats = self.input_norm(feats)

        h1_skip = self.skip1(feats)
        h1, attn1 = self.attn1(feats, edge_index, training=training)
        h1 = self.act(h1 + h1_skip)
        h1 = self.layer_norm(h1)
        h2_skip = self.skip2(h1)
        h2, _ = self.attn2(h1, edge_index, training=training)
        h2 = h2 + h2_skip
        return h2, attn1

    @tf.function(
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(shape=[None, 60], dtype=tf.float32),
            tf.TensorSpec(shape=[2, None], dtype=tf.int32),
        ],
    )
    def _forward_tf_infer(self, feats, edge_index):
        return self._forward_impl(feats, edge_index, training=False)

    @tf.function(
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(shape=[None, 60], dtype=tf.float32),
            tf.TensorSpec(shape=[2, None], dtype=tf.int32),
        ],
    )
    def _forward_tf_train(self, feats, edge_index):
        return self._forward_impl(feats, edge_index, training=True)

    def _forward_all(self, training: bool = False, adaptive_prune: bool = False):
        if self.edge_index is None:
            out = tf.convert_to_tensor(self.features[:, : self.out_dim], dtype=tf.float32)
            self._cache_emb = out
        else:
            feats = tf.convert_to_tensor(self.features, tf.float32)
            edge_ix = self.edge_index_tf
            if edge_ix is None:
                edge_ix = tf.convert_to_tensor(self.edge_index, tf.int32)
                self.edge_index_tf = edge_ix
            if training:
                h, _ = self._forward_tf_train(feats, edge_ix)
            else:
                h, _ = self._forward_tf_infer(feats, edge_ix)
            self._cache_emb = h
        self._cache_emb_np = None
        return self._cache_emb

    def get_cached(self):
        return self._cache_emb

    def get_cached_np(self):
        return self._cache_emb_np

    def forward_cached_embeddings_np(self, training: bool = False) -> np.ndarray:
        h = self._forward_all(training=training)
        h_np = h.numpy() if hasattr(h, "numpy") else np.asarray(h)
        h_np = np.asarray(h_np, dtype=np.float32)
        scale = float(np.max(np.abs(h_np))) + 1e-4
        self._cache_emb_np = h_np / scale
        return self._cache_emb_np

    def _attn_entropy(self, attn: tf.Tensor) -> tf.Tensor:
        p = tf.clip_by_value(attn, 1e-8, 1.0)
        ent_per_head = -tf.reduce_mean(tf.reduce_sum(p * tf.math.log(p), axis=0))
        return ent_per_head

    def set_conflict_map(self, conflict_map: np.ndarray, weight: float = 0.02):
        cm = np.asarray(conflict_map, dtype=np.float32)
        if cm.shape[0] != 20:
            return
        self._conflict_map = cm.copy()
        self._conflict_w = float(weight)

    def _blend_target(self, target: tf.Tensor) -> tf.Tensor:
        if self._conflict_map is None:
            return target
        cm = tf.convert_to_tensor(self._conflict_map, tf.float32)
        scale = 1.0 - self._conflict_w * cm
        return target * scale

    @tf.function(
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(shape=[None, 60], dtype=tf.float32),
            tf.TensorSpec(shape=[2, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None, 20], dtype=tf.float32),
        ],
    )
    def train_on_batch(self, feats: tf.Tensor, edge_index: tf.Tensor, target: tf.Tensor):
        target_blended = self._blend_target(target)
        
        with tf.GradientTape() as tape:
            h2, attn1 = self._forward_tf_train(feats, edge_index)
            pred = self.head(h2)
            loss_main = tf.reduce_mean(tf.square(pred - target_blended))
            ent = self._attn_entropy(attn1)
            loss = loss_main + self.reg_attn_w * ent
            
        vars_ = self.trainable_variables
        grads = tape.gradient(loss, vars_)
        grads = [tf.clip_by_value(g, -self.grad_clip, self.grad_clip) if g is not None else None for g in grads]
        self.opt.apply_gradients(zip(grads, vars_))
        return loss

    def use_GraphSAGE(self, channel_reward, step, idx_list, Graph_SAGE_label: bool):
        feats = tf.convert_to_tensor(self.features, tf.float32)
        if Graph_SAGE_label and channel_reward is not None:
            try:
                tgt = tf.convert_to_tensor(channel_reward, tf.float32)
                edge_ix = self.edge_index_tf
                if edge_ix is None:
                    edge_ix = tf.convert_to_tensor(self.edge_index, tf.int32)
                    self.edge_index_tf = edge_ix
                loss = self.train_on_batch(feats, edge_ix, tgt)
                loss_val = float(loss.numpy()) if hasattr(loss, "numpy") else float(loss)
                self.gat_loss_history.append((int(step), loss_val))
            except Exception:
                self.gat_loss_history.append((int(step), float('nan')))
            try:
                self.adaptive_reprune(int(step), k=max(1, int(self.top_k)))
            except Exception:
                pass
        h_np = self.forward_cached_embeddings_np(training=False)
        idx_list = list(idx_list)
        if len(idx_list) == 0:
            return np.zeros((0, self.out_dim), dtype=np.float32)
        return h_np[idx_list]

# 兼容别名
GraphSAGE_sup = GraphGAT
GAT_sup = GraphGAT
