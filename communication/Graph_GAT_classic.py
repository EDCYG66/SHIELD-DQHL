#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph_GAT_classic.py
Classic GAT implementation with the same external API as Graph_GAT.py.

Compared with Graph_GAT.py (your current GATv2 file), the only essential
change is the attention scoring function:

Classic GAT:
    e_ij = LeakyReLU(a_src^T W h_i + a_dst^T W h_j)

Current GATv2 file:
    e_ij = a^T LeakyReLU(W_src h_i + W_dst h_j)

This file keeps the rest of the pipeline as consistent as possible:
- same GraphGAT-style model interface
- same repruning logic
- same auxiliary head / training interface
- same use_GraphSAGE() compatibility entry
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


class MultiHeadAttentionClassic(tf.keras.layers.Layer):
    """
    Classic GAT attention.

    Paper logic:
        e_ij = LeakyReLU(a_src^T W h_i + a_dst^T W h_j)

    Implementation notes:
    - one shared linear projection W for source / destination / value
    - separate attention vectors for source and destination terms
    - aggregation / softmax logic kept aligned with the current GATv2 file
    """

    def __init__(self, in_dim, out_dim, heads=2, attn_dropout=0.0, **kwargs):
        kwargs.pop('use_low_rank', None)
        kwargs.pop('low_rank_k', None)
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.leaky_alpha = 0.2

        # Shared projection for classic GAT.
        self.W = glorot((in_dim, heads * out_dim), "W")
        self.attn_src = glorot((1, heads, out_dim), "attn_src")
        self.attn_dst = glorot((1, heads, out_dim), "attn_dst")

    def call(self, x, edge_index, training=False):
        # x: [N, in_dim]
        N = tf.shape(x)[0]
        H = self.heads
        D = self.out_dim

        # 1. Shared projection.
        h = tf.reshape(tf.matmul(x, self.W), (N, H, D))

        src = edge_index[0]
        dst = edge_index[1]

        # 2. Gather source/destination node features.
        feat_s = tf.gather(h, src)  # [E, H, D]
        feat_d = tf.gather(h, dst)  # [E, H, D]

        # 3. Classic GAT score.
        score_s = tf.reduce_sum(feat_s * self.attn_src, axis=-1)  # [E, H]
        score_d = tf.reduce_sum(feat_d * self.attn_dst, axis=-1)  # [E, H]
        attn_scores = tf.nn.leaky_relu(score_s + score_d, alpha=self.leaky_alpha)

        # 4. Edge-wise softmax over incoming edges of each destination node.
        attn_scores_t = tf.transpose(attn_scores, [1, 0])  # [H, E]

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

        # 5. Aggregate messages using the same projected features h.
        v_src = tf.gather(h, src)  # [E, H, D]
        messages = tf.expand_dims(attn, -1) * v_src
        messages_flat = tf.reshape(messages, (tf.shape(messages)[0], H * D))
        out_sum = tf.math.unsorted_segment_sum(messages_flat, dst, N)

        return out_sum, attn


class GraphGATClassic(tf.keras.Model):
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
        self.gat_loss_history: List[Tuple[int, float]] = []

        # Repruning state
        self._prev_keep: Optional[np.ndarray] = None
        self._last_reprune_step: Optional[int] = None
        self.reprune_every = int(reprune_every)
        self.reprune_start_step = int(reprune_start_step)
        self.hysteresis_keep = float(hysteresis_keep)
        self.enable_reprune = bool(enable_reprune)

        # Layers
        self.attn1 = MultiHeadAttentionClassic(
            in_dim, hidden_dim, heads=heads, attn_dropout=attn_dropout
        )
        self.attn2 = MultiHeadAttentionClassic(
            hidden_dim * heads, out_dim, heads=1, attn_dropout=attn_dropout
        )

        self.act = tf.keras.layers.ELU()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.input_norm = tf.keras.layers.LayerNormalization()

        # Prediction head
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

    # -------- Graph build/load --------

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

    def update_positions(self, positions: np.ndarray):
        pos = np.asarray(positions, dtype=np.float32)
        if pos.shape[0] != self.num_nodes:
            return
        self.node_positions = pos

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

        if (not self.enable_reprune) or self.top_k <= 0 or self.edge_index is None:
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
        _, attn = self._forward_tf(feats, edge_ix)
        attn_mean = tf.reduce_mean(attn, axis=1).numpy()

        src = self.edge_index[0]
        keep = np.zeros(self.edge_index.shape[1], dtype=bool)
        from collections import defaultdict
        bucket: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
        for eidx, s in enumerate(src):
            bucket[int(s)].append((float(attn_mean[eidx]), int(eidx)))
        for s, lst in bucket.items():
            lst.sort(key=lambda x: x[0], reverse=True)
            for _, e in lst[:k]:
                keep[e] = True

        if self._prev_keep is not None and self._prev_keep.size == keep.size:
            prev_idx = np.where(self._prev_keep)[0]
            if prev_idx.size > 0 and hysteresis_keep > 0:
                n_keep_prev = max(1, int(hysteresis_keep * prev_idx.size))
                keep[prev_idx[:n_keep_prev]] = True

        self._prev_keep = keep.copy()
        self.edge_index = self.edge_index[:, keep]
        self.edge_index = self._ensure_self_loops(self.edge_index)
        self.edge_index_tf = tf.convert_to_tensor(self.edge_index, tf.int32)
        self._last_reprune_step = int(step)

    # -------- forward & train --------

    @tf.function(
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(shape=[None, 60], dtype=tf.float32),
            tf.TensorSpec(shape=[2, None], dtype=tf.int32),
        ],
    )
    def _forward_tf(self, feats, edge_index):
        feats = self.input_norm(feats)
        h1, attn1 = self.attn1(feats, edge_index, training=False)
        h1 = self.act(h1)
        h1 = self.layer_norm(h1)
        h2, _ = self.attn2(h1, edge_index, training=False)
        return h2, attn1

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
            h, _ = self._forward_tf(feats, edge_ix)
            self._cache_emb = h
        return self._cache_emb

    def get_cached(self):
        return self._cache_emb

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
            feats_norm = self.input_norm(feats)
            h1, attn1 = self.attn1(feats_norm, edge_index, training=True)
            h1 = self.act(h1)
            h1 = self.layer_norm(h1)
            h2, _ = self.attn2(h1, edge_index, training=True)
            pred = self.head(h2)
            loss_main = tf.reduce_mean(tf.square(pred - target_blended))
            ent = self._attn_entropy(attn1)
            loss = loss_main + self.reg_attn_w * ent

        vars_ = self.trainable_variables + self.head.trainable_variables
        grads = tape.gradient(loss, vars_)
        grads = [tf.clip_by_value(g, -5.0, 5.0) if g is not None else None for g in grads]
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
        h = self._forward_all(training=False)
        h_np = h.numpy() if hasattr(h, "numpy") else np.asarray(h)
        idx_list = list(idx_list)
        if len(idx_list) == 0:
            return np.zeros((0, self.out_dim), dtype=np.float32)
        return h_np[idx_list]


# Compatibility aliases
GraphGAT = GraphGATClassic
GraphSAGE_sup = GraphGATClassic
GAT_sup = GraphGATClassic
