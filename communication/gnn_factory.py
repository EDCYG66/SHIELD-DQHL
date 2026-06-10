#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gnn_factory.py
Factory to build GNN implementations for the Agent.
"""

from tf_runtime import configure_tensorflow_runtime


configure_tensorflow_runtime()
import tensorflow as tf
configure_tensorflow_runtime(tf)
import numpy as np
import sys

# === 关键修改：使用 tf.keras 避免 IDE 报错 ===
layers = tf.keras.layers
Model = tf.keras.Model

# ==========================================
# 1. 尝试导入您现有的 GAT/SAGE 模型
# ==========================================
try:
    # 从 Graph_GAT.py 导入 GraphGAT 类
    from Graph_GAT import GraphGAT
except ImportError:
    GraphGAT = None
    print("[Warning] Graph_GAT.py not found or GraphGAT class missing.")

try:
    # 从 Graph_SAGE.py 导入 GraphSAGE_sup 类
    from Graph_SAGE import GraphSAGE_sup
except ImportError:
    GraphSAGE_sup = None
    print("[Warning] Graph_SAGE.py not found or GraphSAGE_sup class missing.")


try:
    # 从 Graph_GAT_classic.py 导入 GraphGATClassic 类
    from Graph_GAT_classic import GraphGATClassic
except ImportError:
    GraphGATClassic = None
    print("[Warning] Graph_GAT_classic.py not found or GraphGATClassic class missing.")



# ==========================================
# 2. 定义纯全连接网络 (FCNet) - 用于基线对比
# ==========================================
class FCNet(Model):
    def __init__(self, n_features=60, n_hidden=64, n_embedding=32, max_nodes=200):
        super(FCNet, self).__init__()
        # 定义网络结构：简单的三层全连接网络
        # 输入 -> [Dense 64] -> [Dense 64] -> [Dense 32] -> 输出 Embedding
        self.dense1 = layers.Dense(n_hidden, activation='relu', name="fc_1")
        self.dense2 = layers.Dense(n_hidden, activation='relu', name="fc_2")
        self.out = layers.Dense(n_embedding, activation=None, name="fc_out") 
        
        # === 特征缓存区 ===
        # Agent 会直接修改这个数组来更新状态
        # 60: 输入特征维度 (与 Agent 中的 state 维度一致)
        # max_nodes: 预估的最大节点数 (车辆数 * 3)，设大一点防止越界
        self.features = np.zeros((max_nodes, n_features), dtype=np.float32)

    def call(self, inputs):
        """前向传播逻辑"""
        x = self.dense1(inputs)
        x = self.dense2(x)
        output = self.out(x)
        return output

    def _forward_all(self, training=False):
        """
        专用接口：Agent 调用此方法获取所有节点的 Embedding。
        它将 numpy 缓存转为 tensor 并通过网络。
        """
        # 将当前的 features 缓存转为 Tensor
        inp = tf.convert_to_tensor(self.features, dtype=tf.float32)
        return self(inp)


# ==========================================
# 3. 工厂函数：统一构建入口
# ==========================================
def build_gnn(env, 
              gnn_type, 
              distance_threshold=150.0, 
              lr=5e-4, 
              gat_train_interval=20,
              grad_clip=5.0,
              gat_hidden_dim=32,
              gat_out_dim=32,
              gat_heads=2,
              gat_attn_dropout=0.0,
              gat_top_k=6,
              gat_use_proximity_edges=True,
              gat_proximity_radius=180.0,
              gat_max_proximity_neighbors=6,
              # 下面是透传给 GAT 的参数
              reprune_every=300,
              hysteresis_keep=0.5,
              reprune_start_step=600,
              reg_attn_w=1e-3,
              enable_reprune=True,
              **kwargs):
    """
    根据 gnn_type 参数构建对应的模型实例。
    """
    tag = gnn_type.lower()
    
    # --- 情况 A: 构建 FC Baseline ---
    if tag == 'fc':
        print(f"[Factory] Building FCNet (Baseline) with input_dim=60...")
        # max_nodes 预设为 env.n_Veh * 3 的一点余量，或者直接 200
        n_max = getattr(env, 'n_Veh', 50) * 3 + 50
        return FCNet(n_features=60, max_nodes=n_max)
    
    # --- 情况 B: 构建 GATv2 模型 ---
    elif tag == 'gat':
        print(f"[Factory] Building GATv2 model (GraphGAT)...")
        if GraphGAT is None:
            raise ImportError("无法构建 GATv2 模型：Graph_GAT.py 缺失或导入失败。")

        num_nodes = env.n_Veh * 3
        return GraphGAT(
            num_nodes=num_nodes,
            in_dim=60,
            hidden_dim=int(gat_hidden_dim),
            out_dim=int(gat_out_dim),
            heads=int(gat_heads),
            top_k=int(gat_top_k),
            attn_dropout=float(gat_attn_dropout),
            lr=float(lr),
            gat_train_interval=int(gat_train_interval),
            grad_clip=float(grad_clip),
            reprune_every=reprune_every,
            hysteresis_keep=hysteresis_keep,
            reprune_start_step=reprune_start_step,
            reg_attn_w=reg_attn_w,
            enable_reprune=enable_reprune,
            use_proximity_edges=bool(gat_use_proximity_edges),
            proximity_radius=float(gat_proximity_radius),
            max_proximity_neighbors=int(gat_max_proximity_neighbors),
        )

    # --- 情况 C: 构建 classic GAT 模型 ---
    elif tag == 'gatclassic':
        print(f"[Factory] Building classic GAT model (GraphGATClassic)...")
        if GraphGATClassic is None:
            raise ImportError("无法构建 classic GAT 模型：Graph_GAT_classic.py 缺失或导入失败。")

        num_nodes = env.n_Veh * 3
        return GraphGATClassic(
            num_nodes=num_nodes,
            in_dim=60,
            hidden_dim=int(gat_hidden_dim),
            out_dim=int(gat_out_dim),
            heads=int(gat_heads),
            top_k=int(gat_top_k),
            attn_dropout=float(gat_attn_dropout),
            lr=float(lr),
            gat_train_interval=int(gat_train_interval),
            reprune_every=reprune_every,
            hysteresis_keep=hysteresis_keep,
            reprune_start_step=reprune_start_step,
            reg_attn_w=reg_attn_w,
            enable_reprune=enable_reprune,
            use_proximity_edges=bool(gat_use_proximity_edges),
            proximity_radius=float(gat_proximity_radius),
            max_proximity_neighbors=int(gat_max_proximity_neighbors),
        )

    # --- 情况 D: 构建 GraphSAGE 模型 ---
    elif tag == 'sage' or tag == 'graphsage':
        print(f"[Factory] Building GraphSAGE model (GraphSAGE_sup)...")
        if GraphSAGE_sup is None:
            raise ImportError("无法构建 GraphSAGE 模型：Graph_SAGE.py 缺失或导入失败。")
            
        # 适配 GraphSAGE_sup 的参数签名
        # def __init__(self, environment, distance_threshold=150.0, lr=5e-4, ...)
        return GraphSAGE_sup(
            environment=env,
            distance_threshold=distance_threshold,
            lr=lr,
            gat_train_interval=gat_train_interval,
            grad_clip=grad_clip
        )
            
    else:
        raise ValueError(f"Unknown gnn_type: {gnn_type}. Supported: 'fc', 'gat', 'gatclassic', 'sage'")
