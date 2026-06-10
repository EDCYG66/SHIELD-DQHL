# -*- coding: utf-8 -*-
"""
Agent（GPU加速版 + 平滑版 + 合并补丁 + 紧急度功率奖励 + 邻居RB冲突目标混合
      + RB去集中化惩罚 + 动作阶段RB热度软掩码 + 决策时间记录）

修改说明：
1. 新增 predict_batch 方法：将串行预测改为矩阵批量预测，大幅提升 GPU 利用率。
2. 重构 train_loop_step：先通过 get_state_all 一次性收集所有链路状态，再批量送入 DQN。
3. 在 predict_batch 中记录一次前向推理的耗时到 inference_time.csv。
4. 新增 dynamic_test_for_boxplot：在动态环境中跑一段时间，用当前策略生成更“活跃”的
   时序数据，用于画类似论文图 10 / 图 11 的时间序列和箱线图。
5. 在 _export_results 中，基于 dynamic_test_for_boxplot 的结果，额外导出
   timeseries_<tag>_dynamic.csv / timeseries_<tag>_dynamic.png。
"""

from __future__ import print_function, division
import os
import random
import json
import time
from datetime import datetime

import numpy as np
from tf_runtime import configure_tensorflow_runtime


configure_tensorflow_runtime()
import tensorflow as tf
configure_tensorflow_runtime(tf)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Environment import *
from base import BaseModel
from dqn_model import DQNModel
from replay_memory import ReplayMemory
from gnn_factory import build_gnn

__all__ = ["Agent"]


class Agent(BaseModel):
    def __init__(self,
                 config,
                 environment,
                 gnn_type: str = "gat",
                 warmup_steps: int = 1000,
                 epsilon_min: float = 0.05,
                 epsilon_decay_steps: int = 30000,
                 speed_mode: bool = False,
                 gat_train_interval: int = None,
                 plot_dpi: int = 224,
                 replay_size: int = 200000,
                 power_log_stride: int = 4,
                 power_log_max: int = 200000,
                 # --- 补丁参数 ---
                 soft_update_tau: float = 0.005,
                 power_cost_weight: float = 0.01,
                 conflict_cost_weight: float = 0.02,
                 skip_embedding_steps: int = 9,
                 batch_decay_step: int = None,
                 batch_decay_factor: float = 0.5,
                 # --- 紧急度/冲突图 ---
                 beta_urgency_pos: float = 0.02,
                 beta_urgency_neg: float = 0.02,
                 urgency_threshold: float = 0.25,
                 conflict_penalty_weight: float = 0.02,
                 conflict_window_steps: int = 50,
                 # --- RB去集中化与软掩码 ---
                 rb_anti_conc_alpha: float = 0.01,
                 rb_hot_threshold: float = 0.20,
                 rb_softmask_alpha: float = 0.15,
                 rb_softmask_window: int = 50,
                 dynamic_graph_refresh: bool = True,
                 graph_refresh_steps: int = 100,
                 # --- 新增：外部指定运行目录 ---
                 run_dir: str = None):
        super().__init__(config)
        self.weight_dir = 'weight'
        os.makedirs(self.weight_dir, exist_ok=True)
        self.env = environment

        self.gnn_type = (gnn_type or "gat").lower()
        self.G = build_gnn(
            environment,
            gnn_type=self.gnn_type,
            distance_threshold=150.0,
            lr=5e-4,
            gat_train_interval=gat_train_interval if gat_train_interval else 20,
            grad_clip=5.0,
        )

        # DQN: 输入为 32 维 GNN Embedding + 82 维原状态 = 114
        self.dqn = DQNModel(input_dim=114, output_dim=60,
                            learning_rate=0.001, decay_steps=500000,
                            decay_rate=0.96, min_lr=0.0005,
                            grad_clip_norm=5.0)

        model_dir = './Model/a.model'
        self.memory = ReplayMemory(model_dir, memory_size=int(replay_size), state_dim=114, batch_size=512)

        self.max_step = 100000
        self.RB_number = 20

        self.num_vehicle = getattr(self.env, 'n_Veh', len(getattr(self.env, 'vehicles', [])) or 20)
        self.action_all_with_power = np.zeros([self.num_vehicle, 3, 2], dtype='int32')
        self.action_all_with_power_training = np.zeros([self.num_vehicle, 3, 2], dtype='int32')

        self.discount = 0.9
        self.double_q = True
        self.training = True
        self.GraphSAGE = True

        self.channel_reward = np.zeros((3 * self.num_vehicle, self.RB_number), dtype=np.float32)
        self.neighbor_nodes = []
        self._neighbor_src_links = np.zeros((0,), dtype=np.int32)
        self._neighbor_dst_links = np.zeros((0,), dtype=np.int32)

        self.train_every_n_steps = 25
        self.target_q_update_step = 200

        self.warmup_steps = warmup_steps
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps

        if speed_mode:
            self.warmup_steps = min(self.warmup_steps, 800)
            self.train_every_n_steps = 10
            self.memory.batch_size = 256

        # ----- 日志 / 导出目录 -----
        # 如果外部通过 run_dir 指定了目录，就全部写到 run_dir 里；
        # 否则保持原有行为（runs/v2v_<model>_<timestamp>）
        if run_dir is not None:
            self.logdir = str(run_dir)
        else:
            ts = datetime.now().strftime('%Y%m%d-%H%M%S')
            self.logdir = os.path.join('runs', f'v2v_{self.gnn_type}_{ts}')

        os.makedirs(self.logdir, exist_ok=True)

        # TensorBoard 日志仍单独放在子目录
        self.tb_dqn = tf.summary.create_file_writer(os.path.join(self.logdir, 'dqn'))
        self.tb_gnn = tf.summary.create_file_writer(os.path.join(self.logdir, 'gnn'))

        # 关键：导出目录直接等于 logdir，不再额外嵌套 exports/
        self.export_dir = self.logdir

        if hasattr(self.G, "tb_writer"):
            self.G.tb_writer = self.tb_gnn

        # DQN 训练统计
        self.dqn_loss_history = []   # (step, loss)
        self.dqn_qmean_history = []  # (step, q_mean)

        self.used_blocks_history = []
        self.test_history = []
        self.power_log = []
        self._last_test_detailed = None
        self.POWER_DB = {0: 23, 1: 10, 2: 5}
        self.plot_dpi = int(plot_dpi)

        self.power_log_stride = max(1, int(power_log_stride)) if int(power_log_stride) > 0 else 0
        self.power_log_max = max(1000, int(power_log_max))
        self.last_exported_weights = {}

        # 补丁参数存储
        self.soft_update_tau = float(soft_update_tau)
        self.power_cost_weight = float(power_cost_weight)
        self.conflict_cost_weight = float(conflict_cost_weight)
        self.skip_embedding_steps = int(skip_embedding_steps)
        self.batch_decay_step = batch_decay_step
        self.batch_decay_factor = float(batch_decay_factor)
        self._cached_emb_step = -1

        # 紧急度/冲突图
        self.beta_urgency_pos = float(beta_urgency_pos)
        self.beta_urgency_neg = float(beta_urgency_neg)
        self.urgency_threshold = float(urgency_threshold)
        self.conflict_penalty_weight = float(conflict_penalty_weight)
        self.conflict_window_steps = int(conflict_window_steps)
        self._rb_neighbor_hits = np.zeros(self.RB_number, dtype=np.int32)
        self._rb_neighbor_hist_buffer = []

        # 新增：RB去集中化与软掩码
        self.rb_anti_conc_alpha = float(rb_anti_conc_alpha)
        self.rb_hot_threshold = float(rb_hot_threshold)
        self.rb_softmask_alpha = float(rb_softmask_alpha)
        self.rb_softmask_window = int(rb_softmask_window)
        self._rb_softmask_hist = []
        self.dynamic_graph_refresh = bool(dynamic_graph_refresh)
        self.graph_refresh_steps = max(0, int(graph_refresh_steps))

    # ----------------- 基础工具 -----------------

    def _ensure_action_buffers(self):
        n = getattr(self.env, 'n_Veh', len(getattr(self.env, 'vehicles', [])))
        if n and n != self.num_vehicle:
            self.num_vehicle = n
            self.action_all_with_power = np.zeros([n, 3, 2], dtype='int32')
            self.action_all_with_power_training = np.zeros([n, 3, 2], dtype='int32')

    def _epsilon(self, step: int) -> float:
        if step < self.warmup_steps:
            return 1.0
        decay_progress = (step - self.warmup_steps) / max(1, self.epsilon_decay_steps)
        return max(self.epsilon_min, 1.0 - decay_progress)

    def merge_action(self, idx, action: int):
        a = int(action)
        self.action_all_with_power[idx[0], idx[1], 0] = a % self.RB_number
        self.action_all_with_power[idx[0], idx[1], 1] = a // self.RB_number

    # ==================== 状态接口 ====================

    def _topology_destinations(self, n: int) -> np.ndarray:
        topo_fn = getattr(self.env, "topology_state_arrays", None)
        if callable(topo_fn):
            dests = np.asarray(topo_fn().get("destinations", np.full((n, 3), -1)), dtype=np.int32)
            if dests.shape[0] >= n and dests.shape[1] >= 3:
                return dests[:n, :3]
        return np.asarray(
            [[int(self.env.vehicles[i].destinations[j]) for j in range(3)] for i in range(n)],
            dtype=np.int32,
        )

    def _vehicle_positions_xy(self, n: int) -> np.ndarray:
        state_fn = getattr(self.env, "vehicle_state_arrays", None)
        if callable(state_fn):
            positions = np.asarray(state_fn().get("positions_xy", np.zeros((n, 2))), dtype=np.float32)
            if positions.shape[0] >= n and positions.shape[1] >= 2:
                return positions[:n, :2]
        return np.asarray(
            [getattr(veh, "position", (0.0, 0.0)) for veh in self.env.vehicles[:n]],
            dtype=np.float32,
        ).reshape(n, 2)

    def _refresh_neighbor_link_pairs(self, link_count: int) -> None:
        edge_index = getattr(self.G, "edge_index", None)
        if edge_index is not None:
            edge_index = np.asarray(edge_index, dtype=np.int32).reshape(2, -1)
            src = edge_index[0]
            dst = edge_index[1]
            valid = (src != dst) & (src >= 0) & (src < link_count) & (dst >= 0) & (dst < link_count)
            self._neighbor_src_links = src[valid].astype(np.int32, copy=False)
            self._neighbor_dst_links = dst[valid].astype(np.int32, copy=False)
            return
        if not self.neighbor_nodes:
            self._neighbor_src_links = np.zeros((0,), dtype=np.int32)
            self._neighbor_dst_links = np.zeros((0,), dtype=np.int32)
            return
        gnn_link_flat = (3 * self.G.link[:, 0] + (self.G.link[:, 1] % 3)).astype(np.int32, copy=False)
        src_chunks = []
        dst_chunks = []
        for link_flat in range(min(link_count, len(self.neighbor_nodes))):
            neighs_list = self.neighbor_nodes[link_flat][0]
            if not neighs_list:
                continue
            neigh_idx = np.asarray(neighs_list, dtype=np.int32)
            neigh_flat = gnn_link_flat[neigh_idx]
            valid = (neigh_flat >= 0) & (neigh_flat < link_count)
            if np.any(valid):
                dst = neigh_flat[valid]
                src_chunks.append(np.full(dst.shape, link_flat, dtype=np.int32))
                dst_chunks.append(dst.astype(np.int32, copy=False))
        if src_chunks:
            self._neighbor_src_links = np.concatenate(src_chunks).astype(np.int32, copy=False)
            self._neighbor_dst_links = np.concatenate(dst_chunks).astype(np.int32, copy=False)
        else:
            self._neighbor_src_links = np.zeros((0,), dtype=np.int32)
            self._neighbor_dst_links = np.zeros((0,), dtype=np.int32)

    def _neighbor_rb_selection_matrix(self, link_count: int) -> np.ndarray:
        nei_selection = np.zeros((link_count, self.RB_number), dtype=np.float32)
        src = getattr(self, "_neighbor_src_links", np.zeros((0,), dtype=np.int32))
        dst = getattr(self, "_neighbor_dst_links", np.zeros((0,), dtype=np.int32))
        if src.size == 0 or dst.size == 0:
            return nei_selection
        flat_rb = self.action_all_with_power_training[:, :, 0].reshape(-1)
        rb_vals = flat_rb[dst]
        valid = (rb_vals >= 0) & (rb_vals < self.RB_number)
        if np.any(valid):
            nei_selection[src[valid], rb_vals[valid]] = 1.0
        return nei_selection

    def get_state(self, idx):
        """单条链路的 82 维原始状态（不含 GNN Embedding）"""
        n = len(self.env.vehicles)
        dests = self._topology_destinations(n)
        dst = int(dests[idx[0], idx[1]])
        V2V_channel = (self.env.V2V_channels_with_fastfading[idx[0], dst, :] - 80) / 60
        V2I_channel = (self.env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
        V2V_interference = (-self.env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60
        link_flat = 3 * idx[0] + idx[1]
        if getattr(self, "_neighbor_src_links", np.zeros((0,), dtype=np.int32)).size == 0 and self.neighbor_nodes:
            self._refresh_neighbor_link_pairs(3 * n)
        NeiSelection = self._neighbor_rb_selection_matrix(3 * n)[link_flat]
        time_remaining = np.asarray(
            [self.env.demand[idx[0], idx[1]] / self.env.demand_amount], dtype=np.float32)
        load_remaining = np.asarray(
            [self.env.individual_time_limit[idx[0], idx[1]] / self.env.V2V_limit], dtype=np.float32)
        return np.concatenate((V2I_channel, V2V_interference, V2V_channel,
                               NeiSelection, time_remaining, load_remaining))

    def get_state_all(self, return_indices: bool = True):
        """
        一次性获取所有链路的 82 维原始状态。
        返回:
          - states: (3 * n_Veh, 82)
          - indices: list[(i,j)]，与 states 的行一一对应
        """
        n = len(self.env.vehicles)
        if n == 0:
            return np.zeros((0, 82), dtype=np.float32), []
        link_count = 3 * n
        veh_idx = np.repeat(np.arange(n, dtype=np.int32), 3)
        link_idx = np.tile(np.arange(3, dtype=np.int32), n)

        dests = self._topology_destinations(n)
        dests_flat = dests.reshape(-1)

        v2i = (self.env.V2I_channels_with_fastfading[:n, :] - 80.0) / 60.0
        v2i = np.repeat(v2i.astype(np.float32, copy=False), 3, axis=0)
        v2v = (self.env.V2V_channels_with_fastfading[veh_idx, dests_flat, :] - 80.0) / 60.0
        v2v = v2v.astype(np.float32, copy=False)
        interference = (-self.env.V2V_Interference_all[:n, :3, :].reshape(link_count, -1) - 60.0) / 60.0
        interference = interference.astype(np.float32, copy=False)

        if self.neighbor_nodes and getattr(self, "_neighbor_src_links", np.zeros((0,), dtype=np.int32)).size == 0:
            self._refresh_neighbor_link_pairs(link_count)
        nei_selection = self._neighbor_rb_selection_matrix(link_count)

        time_remaining = (
            self.env.demand[:n, :3].reshape(link_count, 1) / float(self.env.demand_amount)
        ).astype(np.float32, copy=False)
        load_remaining = (
            self.env.individual_time_limit[:n, :3].reshape(link_count, 1) / float(self.env.V2V_limit)
        ).astype(np.float32, copy=False)

        states = np.concatenate((v2i, interference, v2v, nei_selection, time_remaining, load_remaining), axis=1)
        if not return_indices:
            return states, None
        indices = [(int(i), int(j)) for i in range(n) for j in range(3)]
        return states, indices

    def _link_flat_indices(self, n_links: int) -> np.ndarray:
        return np.arange(int(n_links), dtype=np.int32)

    def _compose_full_states(self, emb_all: np.ndarray, raw_states: np.ndarray) -> np.ndarray:
        raw_states = np.asarray(raw_states, dtype=np.float32)
        n_links = raw_states.shape[0]
        if n_links == 0:
            return np.zeros((0, 114), dtype=np.float32)
        emb = np.asarray(emb_all, dtype=np.float32)[:n_links, :32]
        return np.concatenate((emb, raw_states), axis=1).astype(np.float32, copy=False)

    def _write_flat_actions(self, actions_list: np.ndarray) -> None:
        actions_arr = np.asarray(actions_list, dtype=np.int32).reshape(-1)
        n_links = min(actions_arr.size, self.action_all_with_power_training[:, :, 0].size)
        if n_links <= 0:
            return
        flat = self.action_all_with_power_training.reshape(-1, 2)
        acts = actions_arr[:n_links]
        flat[:n_links, 0] = acts % self.RB_number
        flat[:n_links, 1] = acts // self.RB_number

    def _refresh_gnn_features_from_states(self, raw_states: np.ndarray) -> None:
        n_links = min(raw_states.shape[0], self.G.features.shape[0])
        if n_links > 0:
            self.G.features[:n_links, :] = raw_states[:n_links, :60]

    def _collect_node_positions(self):
        n = len(self.env.vehicles)
        if n <= 0:
            return None
        return np.repeat(self._vehicle_positions_xy(n), 3, axis=0).astype(np.float32, copy=False)

    def _sync_graph_topology(self, force: bool = False):
        if self.gnn_type not in ("gat", "gatclassic", "sage"):
            return
        if not self.dynamic_graph_refresh:
            return
        if not hasattr(self.G, "build_graph") or not hasattr(self.G, "load_graph"):
            return

        step = int(getattr(self, "step", 0))
        if (not force) and self.graph_refresh_steps > 0:
            if step <= 0 or (step % self.graph_refresh_steps != 0):
                return
        elif (not force) and self.graph_refresh_steps == 0:
            return

        nveh = len(self.env.vehicles)
        if nveh <= 0:
            return

        adjacency_fn = getattr(self.env, "v2v_adjacency_matrix", None)
        if callable(adjacency_fn):
            self.G.num_V2V_list = np.asarray(adjacency_fn(), dtype=np.float32)
        else:
            self.G.num_V2V_list = np.zeros((nveh, nveh), dtype=np.float32)
            dests = self._topology_destinations(nveh)
            src = np.repeat(np.arange(nveh, dtype=np.int32), 3)
            dst = dests.reshape(-1)
            valid = (dst >= 0) & (dst < nveh)
            self.G.num_V2V_list[src[valid], dst[valid]] = 1.0

        if hasattr(self.G, "update_positions"):
            pos_nodes = self._collect_node_positions()
            if pos_nodes is not None:
                self.G.update_positions(pos_nodes)

        graph, order_nodes, _ = self.G.build_graph(self.G.num_V2V_list)
        if not isinstance(graph, np.ndarray):
            self.G.load_graph(graph, order_nodes)
        self.neighbor_nodes = []
        self._refresh_neighbor_link_pairs(3 * nveh)

    # ----------------- 动作阶段RB热度软掩码 -----------------

    def _compute_softmask_vector(self):
        """计算当前全局的 Softmask 向量 (60,)"""
        mask_vector = np.ones(60, dtype=np.float32)
        if len(self._rb_softmask_hist) == 0:
            return mask_vector

        hist = np.stack(self._rb_softmask_hist, axis=0)  # [T, RB]
        avg_hits = hist.mean(axis=0)                     # [RB]
        total = avg_hits.sum()
        if total <= 0:
            return mask_vector

        rb_ratio = avg_hits / (total + 1e-8)
        hot_mask = (rb_ratio > self.rb_hot_threshold)    # [RB] bool
        rb_for_actions = np.arange(mask_vector.size, dtype=np.int32) % self.RB_number
        mask_vector[hot_mask[rb_for_actions]] = (1.0 - self.rb_softmask_alpha)
        return mask_vector

    # ----------------- 批量预测 (加速核心 + 计时) -----------------

    def predict_batch(self, states_batch: np.ndarray, step: int, test_ep=False):
        """
        批量预测：大幅减少 CPU-GPU 通信开销
        states_batch: shape (N_samples, 114)
        """
        num_samples = states_batch.shape[0]
        if num_samples == 0:
            return np.array([], dtype=np.int32)

        ep = 0.0 if test_ep else self._epsilon(step)

        # 1. 探索逻辑 (Epsilon-Greedy) - 向量化生成
        random_mask = np.random.random(num_samples) < ep

        # 2. 利用逻辑 (GPU Batch Forward) + 计时
        start_t = time.time()
        q_values_tensor = self.dqn.forward(states_batch)  # -> Tensor (N, 60)
        q_values = q_values_tensor.numpy()
        end_t = time.time()
        infer_ms = (end_t - start_t) * 1000.0
        try:
            with open("inference_time.csv", "a") as f:
                f.write(f"{self.gnn_type},{infer_ms:.4f}\n")
        except Exception:
            pass

        # 3. 应用 Softmask
        mask_vec = self._compute_softmask_vector()  # (60,)
        q_values_masked = q_values * mask_vec[None, :]

        greedy_actions = np.argmax(q_values_masked, axis=1)  # (N,)

        # 4. 生成随机动作
        random_actions = np.random.randint(0, 60, size=num_samples)

        # 5. 组合
        final_actions = np.where(random_mask, random_actions, greedy_actions)
        return final_actions.astype(np.int32)

    def predict(self, s_t, step: int, test_ep=False):
        return int(self.predict_batch(s_t[None, :], step, test_ep)[0])

    # ----------------- 经验回放 / 通道奖励 -----------------

    def observe(self, prestate, poststate, reward, action, step):
        """只写入经验池；DQN 更新在全局 step 末尾统一触发一次。"""
        self.memory.add(prestate, poststate, reward, action)

    def _update_channel_reward(self):
        raw = self.G.features[:, 0:20] + self.G.features[:, 20:40] - self.G.features[:, 40:60]
        scale = float(np.max(np.abs(raw))) + 1e-6
        self.channel_reward = raw / scale

    # ----------------- 初始化 Embedding -----------------

    def initial_better_state(self, step, Graph_SAGE_label=True):
        self._ensure_action_buffers()
        self.neighbor_nodes = []
        state_old, indices = self.get_state_all()
        idx_list = list(range(state_old.shape[0]))
        self.G.features = np.zeros((3 * len(self.env.vehicles), 60), dtype=np.float32)
        self._sync_graph_topology(force=True)
        self._refresh_gnn_features_from_states(state_old)

        self._update_channel_reward()
        node_embeddings = self.G.use_GraphSAGE(self.channel_reward, step, idx_list, Graph_SAGE_label)
        emb_scale = float(np.max(np.abs(node_embeddings))) + 1e-4
        node_embeddings = node_embeddings / emb_scale
        return self._compose_full_states(node_embeddings, state_old)

    # ----------------- Warmup：已向量化 -----------------

    def warmup(self):
        if self.memory.size() >= self.warmup_steps:
            return
        self._ensure_action_buffers()
        print(f"[WarmUp] collecting {self.warmup_steps} transitions ...")
        self.env.new_random_game()
        self.initial_better_state(0, True)

        while self.memory.size() < self.warmup_steps:
            print(f"Warmup Progress: {self.memory.size()} / {self.warmup_steps}", end='\r')
            self._sync_graph_topology(force=True)

            # 1. 一次性收集所有链路状态 (82 维)
            s_old_all, _ = self.get_state_all(return_indices=False)
            n_links = s_old_all.shape[0]
            if n_links == 0:
                break

            # 2. 更新 GNN 特征，并前向一次得到 Embedding (32 维)
            self._refresh_gnn_features_from_states(s_old_all)
            self._update_channel_reward()
            idx_all = list(range(3 * len(self.env.vehicles)))
            emb_all = self.G.use_GraphSAGE(self.channel_reward, 0, idx_all, False)
            emb_all = emb_all / (np.max(np.abs(emb_all)) + 1e-4)

            # 3. 拼接 Embedding + 状态，随机动作
            full_old_batch = self._compose_full_states(emb_all, s_old_all)
            actions_batch = np.random.randint(0, 60, size=n_links, dtype=np.int32)
            self._write_flat_actions(actions_batch)

            # 4. 环境步进
            reward_matrix, _, _ = self.env.batch_reward_all(self.action_all_with_power_training)

            # 5. 计算 next state（简化：复用当前 Embedding）
            s_new_all, _ = self.get_state_all(return_indices=False)
            full_new_batch = self._compose_full_states(emb_all, s_new_all)
            reward_flat = reward_matrix.reshape(-1)
            needed = max(0, int(self.warmup_steps) - int(self.memory.size()))
            take = min(n_links, needed)
            if take > 0:
                self.memory.add_batch(
                    full_old_batch[:take],
                    full_new_batch[:take],
                    reward_flat[:take],
                    actions_batch[:take],
                )
        print("\n[WarmUp] done.")

    # ----------------- 紧急度奖励/冲突图/惩罚/缓存/衰减 -----------------

    def _extract_time_left_ratio(self, i: int, j: int) -> float:
        s = self.get_state([i, j])
        time_left_ratio = float(s[-2])
        return max(0.0, min(1.0, time_left_ratio))

    def apply_urgency_power_shaping(self):
        actions = self.action_all_with_power_training
        pw = actions[:, :, 1]
        n = len(self.env.vehicles)
        if n <= 0:
            return
        time_left_ratio = np.asarray(self.env.individual_time_limit[:n, :3], dtype=np.float32)
        time_left_ratio = np.clip(time_left_ratio / float(self.env.V2V_limit), 0.0, 1.0)
        urgent = time_left_ratio < self.urgency_threshold
        score = (
            self.beta_urgency_pos * float(np.sum(urgent & (pw[:n, :3] == 0)))
            - self.beta_urgency_neg * float(np.sum(urgent & (pw[:n, :3] == 2)))
        )
        shaping_factor = np.exp(score / (3.0 * n + 1e-6))
        self.channel_reward *= shaping_factor

    def compute_neighbor_rb_conflict_map(self) -> np.ndarray:
        hits = np.zeros(self.RB_number, dtype=np.int32)
        dst = getattr(self, "_neighbor_dst_links", np.zeros((0,), dtype=np.int32))
        if dst.size > 0:
            flat_rb = self.action_all_with_power_training[:, :, 0].reshape(-1)
            rb_vals = flat_rb[dst]
            valid = rb_vals[(rb_vals >= 0) & (rb_vals < self.RB_number)]
            if valid.size > 0:
                hits += np.bincount(valid, minlength=self.RB_number).astype(np.int32)
        self._rb_neighbor_hist_buffer.append(hits)
        if len(self._rb_neighbor_hist_buffer) > self.conflict_window_steps:
            self._rb_neighbor_hist_buffer = self._rb_neighbor_hist_buffer[-self.conflict_window_steps:]
        agg = np.sum(np.stack(self._rb_neighbor_hist_buffer, axis=0), axis=0) if self._rb_neighbor_hist_buffer else hits
        self._rb_neighbor_hits = agg.astype(np.int32)
        m = float(np.max(self._rb_neighbor_hits)) if np.max(self._rb_neighbor_hits) > 0 else 1.0
        return (self._rb_neighbor_hits.astype(np.float32) / m)

    def soft_update_target(self):
        for tw, ow in zip(self.dqn.target_model.weights, self.dqn.model.weights):
            tw.assign((1.0 - self.soft_update_tau) * tw + self.soft_update_tau * ow)

    def compute_penalty(self) -> float:
        actions = self.action_all_with_power_training
        rb = actions[:, :, 0]
        pw = actions[:, :, 1]
        penalty_total = 0.0
        penalty_total += self.power_cost_weight * float(np.sum(pw))
        src = getattr(self, "_neighbor_src_links", np.zeros((0,), dtype=np.int32))
        dst = getattr(self, "_neighbor_dst_links", np.zeros((0,), dtype=np.int32))
        if src.size > 0 and dst.size > 0:
            flat_rb = rb.reshape(-1)
            valid = flat_rb[src] == flat_rb[dst]
            penalty_total += self.conflict_cost_weight * float(np.sum(valid))
        return float(penalty_total)

    # ----------------- Embedding 前向缓存 -----------------

    def forward_embeddings(self, force=False):
        need_update = force or (getattr(self.env, "n_step", 0) % (self.skip_embedding_steps + 1) == 0) or (self._cached_emb_step < 0)
        if need_update and hasattr(self.G, "forward_cached_embeddings_np"):
            emb_all = self.G.forward_cached_embeddings_np(training=False)
            self._cached_emb_step = getattr(self.env, "n_step", 0)
        elif need_update and hasattr(self.G, "_forward_all"):
            emb_all_t = self.G._forward_all(training=False)
            emb_all = emb_all_t.numpy() if hasattr(emb_all_t, "numpy") else np.asarray(emb_all_t)
            emb_all = np.asarray(emb_all, dtype=np.float32)
            emb_all = emb_all / (np.max(np.abs(emb_all)) + 1e-4)
            self._cached_emb_step = getattr(self.env, "n_step", 0)
            self.G._cache_emb = emb_all
        else:
            emb_all = getattr(self.G, "_cache_emb_np", None)
            if emb_all is None:
                emb_all = getattr(self.G, "_cache_emb", None)
        if emb_all is None:
            N = 3 * len(self.env.vehicles)
            idx_all = list(range(N))
            emb_all = self.G.use_GraphSAGE(self.channel_reward, getattr(self, "step", 0), idx_all, False)
            self._cached_emb_step = getattr(self.env, "n_step", 0)
        return np.asarray(emb_all, dtype=np.float32)

    def maybe_decay_batch_size(self):
        if self.batch_decay_step is not None and getattr(self, "step", 0) >= self.batch_decay_step:
            self.memory.batch_size = int(max(64, int(self.memory.batch_size * self.batch_decay_factor)))

    def _rb_anti_concentration_penalty(self):
        rb = self.action_all_with_power_training[:, :, 0].reshape(-1)
        total_links = rb.shape[0]
        counts = np.bincount(rb, minlength=self.RB_number).astype(np.float32)
        max_rb = float(np.max(counts))
        hot_ratio = max_rb / (float(total_links) + 1e-6)
        factor = np.exp(-self.rb_anti_conc_alpha * hot_ratio)
        self.channel_reward *= factor
        self._rb_softmask_hist.append(counts)
        if len(self._rb_softmask_hist) > self.rb_softmask_window:
            self._rb_softmask_hist = self._rb_softmask_hist[-self.rb_softmask_window:]

    # ----------------- 训练单步（使用 get_state_all） -----------------

    def train_loop_step(self, gnn_train_interval=20, base_batch_size=512):
        self._sync_graph_topology(force=(self.step == 1))

        # 1. 使用 get_state_all 一次性收集所有链路状态，并刷新 GNN 特征
        s_old_all, _ = self.get_state_all(return_indices=False)
        n_links = s_old_all.shape[0]
        if n_links == 0:
            return
        self._refresh_gnn_features_from_states(s_old_all)
        self._update_channel_reward()

        # 2. 低频 GNN 训练
        if self.GraphSAGE and (self.step % gnn_train_interval == 0) and self.training:
            idx_all = list(range(3 * len(self.env.vehicles)))
            try:
                conflict_map = self.compute_neighbor_rb_conflict_map()
                if hasattr(self.G, "set_conflict_map"):
                    self.G.set_conflict_map(conflict_map, weight=self.conflict_penalty_weight)
            except Exception:
                pass
            _ = self.G.use_GraphSAGE(self.channel_reward, self.step, idx_all, True)

        # 3. 前向 Embedding（所有节点）
        emb_all = self.forward_embeddings(force=(self.step == 1))

        # 4. 拼接 Embedding + 状态，构造 batch_states
        batch_states = self._compose_full_states(emb_all, s_old_all)

        # 5. 执行批量预测
        actions_list = self.predict_batch(batch_states, self.step)

        # 6. 写回动作 & Power Log
        self._write_flat_actions(actions_list)
        if self.power_log_stride and (self.step % self.power_log_stride == 0):
            pw_flat = self.action_all_with_power_training[:, :, 1].reshape(-1)[:n_links]
            time_left_s = s_old_all[:n_links, -1].astype(np.float32) * float(self.env.V2V_limit)
            self.power_log.extend((float(t), int(pw)) for t, pw in zip(time_left_s, pw_flat))
            if len(self.power_log) > self.power_log_max:
                self.power_log = self.power_log[-self.power_log_max:]

        # 7. 奖励 + 修正
        reward_matrix, v2i_rate_total, fail_percent = self.env.batch_reward_all(self.action_all_with_power_training)
        self.apply_urgency_power_shaping()
        self._rb_anti_concentration_penalty()
        # === 新增：轻微加强 V2V 成功的权重 ===
        v2v_succ = 1.0 - float(fail_percent)
        # 成功越多越好，失败要惩罚；V2I 只给一个小权重
        R_v2v = 1.0 * v2v_succ - 1.0 * (1.0 - v2v_succ)   # 范围大致 [-1, +1]
        R_v2i = 0.005 * float(v2i_rate_total)             # V2I 较小权重
        extra = (R_v2v + R_v2i) / max(1.0, 3.0 * len(self.env.vehicles))
        reward_matrix = reward_matrix + extra
        # === 新增部分结束 ===
        # 8. 存入 Replay Memory（使用新的 get_state_all）
        self.maybe_decay_batch_size()
        s_new_all, _ = self.get_state_all(return_indices=False)  # 当前 step 后的状态
        full_new_batch = self._compose_full_states(emb_all, s_new_all)
        actions_flat = self.action_all_with_power_training.reshape(-1, 2)[:n_links]
        act_ints = (actions_flat[:, 0] + self.RB_number * actions_flat[:, 1]).astype(np.int32, copy=False)
        reward_flat = reward_matrix.reshape(-1)
        self.memory.add_batch(batch_states[:n_links], full_new_batch[:n_links], reward_flat[:n_links], act_ints)

        # 9. 每个全局 step 只做一次 DQN 更新，再做一次软更新
        if self.step >= self.warmup_steps and self.step % self.train_every_n_steps == 0:
            self.q_learning_mini_batch()
            self.soft_update_target()

        # 10. 统计 & penalty
        used_blocks = np.unique(self.action_all_with_power_training[:, :, 0])
        self.used_blocks_history.append((self.step, len(used_blocks)))
        penalty_val = self.compute_penalty()
        self.channel_reward *= np.exp(-0.001 * penalty_val)

        if self.tb_dqn is not None and self.step % 25 == 0:
            with self.tb_dqn.as_default():
                tf.summary.scalar('Env/used_blocks', float(len(used_blocks)), step=self.step)
                tf.summary.scalar('Env/epsilon', float(self._epsilon(self.step)), step=self.step)
                tf.summary.scalar('Patch/penalty', float(penalty_val), step=self.step)

    # ----------------- 训练主循环 -----------------

    def train(self, max_steps=50000, test_every_steps=2000, test_sample=200):
        self.dqn.update_target_network()
        self.warmup()

        self.env.new_random_game()
        _ = self.initial_better_state(0, True)

        for self.step in range(1, max_steps + 1):
            is_test_step = (test_every_steps > 0 and self.step % test_every_steps == 0)

            if is_test_step:
                self.training = False
                mean_v2i, fail = self.test_environment(test_sample=test_sample, detailed=False)
                self.test_history.append((self.step, float(mean_v2i), float(fail)))
                print(f"[TEST] step={self.step} v2i={mean_v2i:.3f} fail={fail:.3f} eps={self._epsilon(self.step):.3f}")
            self.training = True

            # 单步训练（向量化）
            self.train_loop_step(gnn_train_interval=getattr(self.G, "gat_train_interval", 20),
                                 base_batch_size=self.memory.batch_size)

            if self.step % self.target_q_update_step == 0:
                self.dqn.update_target_network()
                print(f"[Target] updated at step {self.step}")

        final_v2i, final_fail = self.test_environment(test_sample=test_sample, detailed=True)
        self.test_history.append((self.step, float(final_v2i), float(final_fail)))
        self._export_results(final_v2i, final_fail)
        print(f"[TRAIN DONE] steps={self.step} final_v2i={final_v2i:.4f} fail={final_fail:.4f}")

    # ----------------- 测试 / DQN 更新 / 导出 -----------------
    def test_environment(self, test_sample=200, detailed=False):
        """
        用当前策略重新决策 test_sample 步，统计平均 V2I / V2V。
        不再复用训练过程中的旧动作。
        """
        V2I_Rate_list = []
        fail_list = []
        inst_v2i, inst_v2v, inst_nveh, inst_t = [], [], [], []
        t_idx = 0

        for _ in range(test_sample):
            self._sync_graph_topology(force=True)
            # 1) 根据当前策略重新选择一整步动作
            s_old_all, _ = self.get_state_all(return_indices=False)
            n_links = s_old_all.shape[0]
            if n_links == 0:
                break

            # 刷新 GNN 特征
            self._refresh_gnn_features_from_states(s_old_all)
            self._update_channel_reward()
            emb_all = self.forward_embeddings(force=False)

            batch_states = self._compose_full_states(emb_all, s_old_all)

            # 测试阶段不探索（ep=0）
            actions_list = self.predict_batch(
                batch_states,
                step=self.step if hasattr(self, "step") else 0,
                test_ep=True
            )
            self._write_flat_actions(actions_list)

            # 2) 推进环境一步
            reward_vec, fail = self.env.act_asyn(self.action_all_with_power_training)
            V2I_Rate_list.append(float(np.sum(reward_vec)))
            fail_list.append(float(fail))

            if detailed:
                inst_v2i.append(float(np.sum(reward_vec)))
                inst_v2v.append(float(1.0 - fail))
                inst_nveh.append(int(len(self.env.vehicles)))
                inst_t.append(t_idx)
                t_idx += 1

        mean_v2i = float(np.mean(V2I_Rate_list)) if V2I_Rate_list else 0.0
        fail_percent = float(np.mean(fail_list)) if fail_list else 0.0
        if detailed:
            self._last_test_detailed = dict(
                t=inst_t,
                v2i=inst_v2i,
                v2v_succ=inst_v2v,
                nveh=inst_nveh
            )
        return mean_v2i, fail_percent

    # ---------- 新增：动态测试，用于生成箱线图数据 ----------

    def dynamic_test_for_boxplot(self, T=250, refresh_every=50):
        """
        使用当前策略在一个“动态环境”中跑 T 步，收集 (t, v2i_rate, v2v_success, nveh)。
        """
        inst_t, inst_v2i, inst_v2v, inst_nveh = [], [], [], []

        # 从一个新的随机场景开始
        self.env.new_random_game()
        self._ensure_action_buffers()
        # 刷新一次邻居图/特征
        _ = self.initial_better_state(0, True)

        for t in range(T):
            if (t > 0) and (t % refresh_every == 0):
                # 周期性重置场景，制造更多多样性
                self.env.new_random_game()
                self._ensure_action_buffers()
                _ = self.initial_better_state(0, True)

            self._sync_graph_topology(force=True)

            # 1) 收集当前状态 + 刷新 GNN 特征
            s_old_all, _ = self.get_state_all(return_indices=False)
            n_links = s_old_all.shape[0]
            if n_links == 0:
                continue
            self._refresh_gnn_features_from_states(s_old_all)
            self._update_channel_reward()

            # 2) 前向 Embedding + 批量动作选择（不探索）
            emb_all = self.forward_embeddings(force=(t == 0))
            batch_states = self._compose_full_states(emb_all, s_old_all)

            actions_list = self.predict_batch(batch_states, step=self.step if hasattr(self, "step") else 0, test_ep=True)

            self._write_flat_actions(actions_list)

            # 3) 推进环境一次，记录即时 V2I/V2V
            reward_vec, fail = self.env.act_asyn(self.action_all_with_power_training)
            v2i_rate = float(np.sum(reward_vec))
            v2v_succ = float(1.0 - fail)
            nveh_now = float(len(self.env.vehicles))

            inst_t.append(t)
            inst_v2i.append(v2i_rate)
            inst_v2v.append(v2v_succ)
            inst_nveh.append(nveh_now)

        return dict(t=inst_t, v2i=inst_v2i, v2v_succ=inst_v2v, nveh=inst_nveh)

    def q_learning_mini_batch(self):
        if self.memory.size() < self.memory.batch_size:
            return
        s_t, s_tp1, actions, rewards = self.memory.sample()
        s_t = tf.convert_to_tensor(s_t, tf.float32)
        s_tp1 = tf.convert_to_tensor(s_tp1, tf.float32)
        actions = tf.convert_to_tensor(actions, tf.int32)
        rewards = tf.convert_to_tensor(rewards, tf.float32)

        if self.double_q:
            q_next_online = self.dqn.forward(s_tp1)
            next_actions = tf.argmax(q_next_online, axis=1, output_type=tf.int32)
            q_next_target = self.dqn.forward_target(s_tp1)
            idxs = tf.stack([tf.range(tf.shape(next_actions)[0], dtype=tf.int32), next_actions], axis=1)
            q_tp1 = tf.gather_nd(q_next_target, idxs)
        else:
            q_next_target = self.dqn.forward_target(s_tp1)
            q_tp1 = tf.reduce_max(q_next_target, axis=1)

        q_tp1 = tf.cast(q_tp1, rewards.dtype)
        target = rewards + tf.cast(self.discount, rewards.dtype) * q_tp1

        loss, q_values = self.dqn.train_step(s_t, target, actions)
        loss_v = float(loss.numpy())
        q_mean = float(tf.reduce_mean(tf.cast(q_values, tf.float32)).numpy())

        # 记录 DQN 训练指标
        self.dqn_loss_history.append((self.step, loss_v))
        self.dqn_qmean_history.append((self.step, q_mean))

        if self.tb_dqn is not None and self.step % 25 == 0:
            with self.tb_dqn.as_default():
                tf.summary.scalar('DQN/loss', loss_v, step=self.step)
                tf.summary.scalar('DQN/q_mean', q_mean, step=self.step)

    def _export_results(self, final_mean_rate: float, final_fail_percent: float):
        os.makedirs(self.export_dir, exist_ok=True)
        tag = self.gnn_type
        weight_exports = self._export_model_weights()

        # ----- DQN loss / q_mean -----
        if self.dqn_loss_history:
            steps = [s for s, _ in self.dqn_loss_history]
            losses = [l for _, l in self.dqn_loss_history]
            qmeans = [qm for (_, qm) in self.dqn_qmean_history]
            with open(os.path.join(self.export_dir, f'dqn_metrics_{tag}.csv'), 'w') as f:
                f.write('step,loss,q_mean\n')
                for s, l, qm in zip(steps, losses, qmeans):
                    f.write(f'{s},{l},{qm}\n')
            plt.figure(figsize=(7, 4))
            try:
                import pandas as pd
                df = pd.DataFrame({'step': steps, 'loss': losses})
                smooth = df['loss'].rolling(5, min_periods=1).mean()
                plt.plot(steps, smooth, label=f'DQN loss ({tag})')
            except Exception:
                plt.plot(steps, losses, label=f'DQN loss ({tag})')
            plt.xlabel('step'); plt.ylabel('loss'); plt.title('DQN Loss over Steps')
            plt.grid(alpha=0.3); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.export_dir, f'dqn_loss_{tag}.png'), dpi=self.plot_dpi)
            plt.close()

        # ----- 环境指标：RB 使用 -----
        if self.used_blocks_history:
            with open(os.path.join(self.export_dir, f'env_metrics_{tag}.csv'), 'w') as f:
                f.write('step,used_blocks\n')
                for s, ub in self.used_blocks_history:
                    f.write(f'{s},{ub}\n')
            steps = [s for s, _ in self.used_blocks_history]
            ubs = [ub for _, ub in self.used_blocks_history]
            plt.figure(figsize=(7, 4))
            plt.plot(steps, ubs, label=f'Used RB Blocks ({tag})')
            plt.xlabel('step'); plt.ylabel('num_used_blocks')
            plt.title('Used RB Blocks over Steps')
            plt.grid(alpha=0.3); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.export_dir, f'used_blocks_{tag}.png'), dpi=self.plot_dpi)
            plt.close()

        # ----- GNN loss -----
        steps_gl, losses_gl = [], []
        if hasattr(self.G, 'gat_loss_history') and self.G.gat_loss_history:
            steps_gl = [s for s, _ in self.G.gat_loss_history]
            losses_gl = [l for _, l in self.G.gat_loss_history]

        if losses_gl:
            with open(os.path.join(self.export_dir, f'gnn_loss_{tag}.csv'), 'w') as f:
                f.write('step,loss\n')
                for s, l in zip(steps_gl, losses_gl):
                    f.write(f'{s},{l}\n')
            plt.figure(figsize=(7, 4))
            plt.plot(steps_gl, losses_gl, label=f'{self.gnn_type.upper()} loss')
            plt.xlabel('step'); plt.ylabel('loss'); plt.title(f'{self.gnn_type.upper()} Training Loss over Steps')
            plt.grid(alpha=0.3); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.export_dir, f'gnn_loss_{tag}.png'), dpi=self.plot_dpi)
            plt.close()

        # ----- Test history (V2I / V2V) -----
        if self.test_history:
            th = self.test_history
            with open(os.path.join(self.export_dir, f'test_history_{tag}.csv'), 'w') as f:
                f.write('step,v2i_mean,fail_percent,v2v_success\n')
                for s, v2i, fail in th:
                    f.write(f'{s},{v2i},{fail},{1.0 - fail}\n')
            steps = [s for s, _, _ in th]
            v2i_vals = [v for _, v, _ in th]
            v2v_succ = [1.0 - f for _, _, f in th]
            fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
            ln1 = ax1.plot(steps, v2v_succ, '-o', label='V2V Success Rate', color='tab:blue')
            ax1.set_xlabel('step'); ax1.set_ylabel('V2V Success Rate', color='tab:blue')
            ax2 = ax1.twinx()
            ln2 = ax2.plot(steps, v2i_vals, '-s', label='V2I Rate', color='tab:orange')
            ax2.set_ylabel('V2I Rate', color='tab:orange')
            lines = ln1 + ln2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='best')
            ax1.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.export_dir, f'training_effect_{tag}.png'), dpi=self.plot_dpi)
            plt.close()

        # ----- Power 选择分布 -----
        if self.power_log:
            arr = np.array(self.power_log, dtype=np.float32)
            times = arr[:, 0]
            pidx = arr[:, 1].astype(int)
            limit = self.env.V2V_limit if hasattr(self.env, 'V2V_limit') else max(0.1, times.max())
            times = np.clip(times, 0, limit)
            nbins = 12
            edges = np.linspace(0, limit, nbins + 1)
            mids = 0.5 * (edges[:-1] + edges[1:])
            probs = np.zeros((nbins, 3), dtype=np.float32)
            for b in range(nbins):
                mask = (times >= edges[b]) & (times < edges[b + 1])
                cnt = mask.sum()
                if cnt > 0:
                    for k in [0, 1, 2]:
                        probs[b, k] = np.sum(mask & (pidx == k)) / cnt
            with open(os.path.join(self.export_dir, f'power_select_{tag}.csv'), 'w') as f:
                f.write('time_left,prob_p0,prob_p1,prob_p2\n')
                for m, (p0, p1, p2) in zip(mids, probs):
                    f.write(f'{m},{p0},{p1},{p2}\n')
            plt.figure(figsize=(7.2, 4.6))
            plt.plot(mids, probs[:, 0], '-o', label=f'Power {self.POWER_DB.get(0,"0")} dB')
            plt.plot(mids, probs[:, 1], '-s', label=f'Power {self.POWER_DB.get(1,"1")} dB')
            plt.plot(mids, probs[:, 2], '-^', label=f'Power {self.POWER_DB.get(2,"2")} dB')
            plt.xlabel('Time left for V2V transmission (s)')
            plt.ylabel('Probability of power selection')
            plt.grid(alpha=0.3); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.export_dir, f'power_select_{tag}.png'), dpi=self.plot_dpi)
            plt.close()

        # === 原有：导出 slot 级静态测试时序 ===
        if getattr(self, "_last_test_detailed", None):
            ts = self._last_test_detailed
            t = np.asarray(ts.get("t", []), dtype=float)
            v2i_ts = np.asarray(ts.get("v2i", []), dtype=float)
            v2v_ts = np.asarray(ts.get("v2v_succ", []), dtype=float)
            nveh_ts = np.asarray(ts.get("nveh", []), dtype=float)

            if t.size > 0:
                # 1) CSV
                csv_ts = os.path.join(self.export_dir, f"timeseries_{tag}.csv")
                with open(csv_ts, "w", encoding="utf-8") as f:
                    f.write("t,v2i_rate,v2v_success,nveh\n")
                    for ti, a, b, c in zip(t, v2i_ts, v2v_ts, nveh_ts):
                        f.write(f"{ti},{a},{b},{c}\n")

                # 2) PNG：三条曲线，共享 x 轴，不同 y 轴
                fig, ax1 = plt.subplots(figsize=(7.5, 4.6))

                # 左轴：V2V 成功率
                ln1 = ax1.plot(t, v2v_ts, "-", color="tab:blue",
                               label="V2V Success Rate")
                ax1.set_xlabel("Time index")
                ax1.set_ylabel("V2V Success Rate", color="tab:blue")
                ax1.tick_params(axis="y", labelcolor="tab:blue")
                ax1.grid(alpha=0.3)

                # 右轴 1：V2I 速率
                ax2 = ax1.twinx()
                ln2 = ax2.plot(t, v2i_ts, "-", color="tab:orange",
                               label="V2I Rate")
                ax2.set_ylabel("V2I Rate", color="tab:orange")
                ax2.tick_params(axis="y", labelcolor="tab:orange")

                # 右轴 2：车辆数（第二个共享的 y 轴）
                ax3 = ax1.twinx()
                ax3.spines["right"].set_position(("outward", 55))
                ln3 = ax3.plot(t, nveh_ts, "-", color="tab:green",
                               label="Vehicle Count")
                ax3.set_ylabel("Vehicle Count", color="tab:green")
                ax3.tick_params(axis="y", labelcolor="tab:green")

                # 合并 legend
                lines = ln1 + ln2 + ln3
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc="upper left")

                plt.title(f"Instantaneous V2V/V2I and Vehicle Count ({tag})")
                plt.tight_layout()
                fig.savefig(os.path.join(self.export_dir,
                                         f"timeseries_{tag}.png"),
                            dpi=self.plot_dpi)
                plt.close(fig)

        # === 新增：动态测试时序，用于箱线图 ===
        try:
            dyn = self.dynamic_test_for_boxplot(T=250, refresh_every=50)
            t_dyn = np.asarray(dyn.get("t", []), dtype=float)
            v2i_dyn = np.asarray(dyn.get("v2i", []), dtype=float)
            v2v_dyn = np.asarray(dyn.get("v2v_succ", []), dtype=float)
            nveh_dyn = np.asarray(dyn.get("nveh", []), dtype=float)

            if t_dyn.size > 0:
                csv_dyn = os.path.join(self.export_dir, f"timeseries_{tag}_dynamic.csv")
                with open(csv_dyn, "w", encoding="utf-8") as f:
                    f.write("t,v2i_rate,v2v_success,nveh\n")
                    for ti, a, b, c in zip(t_dyn, v2i_dyn, v2v_dyn, nveh_dyn):
                        f.write(f"{ti},{a},{b},{c}\n")

                # 画一张简单的动态时序图（类似图10）
                fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
                ln1 = ax1.plot(t_dyn, v2v_dyn, "-", color="tab:blue",
                               label="V2V Success Rate")
                ax1.set_xlabel("Time index")
                ax1.set_ylabel("V2V Success Rate", color="tab:blue")
                ax1.tick_params(axis="y", labelcolor="tab:blue")
                ax1.grid(alpha=0.3)

                ax2 = ax1.twinx()
                ln2 = ax2.plot(t_dyn, v2i_dyn, "-", color="tab:orange",
                               label="V2I Rate")
                ax2.set_ylabel("V2I Rate", color="tab:orange")
                ax2.tick_params(axis="y", labelcolor="tab:orange")

                ax3 = ax1.twinx()
                ax3.spines["right"].set_position(("outward", 55))
                ln3 = ax3.plot(t_dyn, nveh_dyn, "-", color="tab:green",
                               label="Vehicle Count")
                ax3.set_ylabel("Vehicle Count", color="tab:green")
                ax3.tick_params(axis="y", labelcolor="tab:green")

                lines = ln1 + ln2 + ln3
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc="upper left")
                plt.title(f"Dynamic V2V/V2I and Vehicle Count ({tag})")
                plt.tight_layout()
                fig.savefig(os.path.join(self.export_dir,
                                         f"timeseries_{tag}_dynamic.png"),
                            dpi=self.plot_dpi)
                plt.close(fig)
        except Exception as e:
            print(f"[WARN] dynamic_test_for_boxplot failed: {e}")

        summary = {
            'model': self.gnn_type,
            'final_mean_v2i_rate': float(final_mean_rate),
            'final_fail_percent': float(final_fail_percent),
            'steps_trained': int(getattr(self, 'step', -1)),
            'epsilon_final': float(self._epsilon(getattr(self, 'step', 0))),
            'weight_exports': weight_exports,
            'files': sorted(os.listdir(self.export_dir)),
            'last_timeseries': self._last_test_detailed if getattr(self, '_last_test_detailed', None) else None,
        }
        with open(os.path.join(self.export_dir, f'summary_{tag}.json'), 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[Export] Files saved -> {self.export_dir} (tag={tag})")

    def _export_model_weights(self):
        os.makedirs(self.export_dir, exist_ok=True)
        tag = self.gnn_type
        exports = {
            "dqn_weights": "",
            "gnn_weights": "",
            "export_dir": self.export_dir,
        }

        dqn_path = os.path.join(self.export_dir, f"dqn_weights_{tag}.weights.h5")
        try:
            self.dqn.model.save_weights(dqn_path)
            exports["dqn_weights"] = dqn_path
        except Exception as exc:
            exports["dqn_error"] = str(exc)

        try:
            if hasattr(self.G, "_forward_all"):
                try:
                    _ = self.G._forward_all(training=False)
                except TypeError:
                    _ = self.G._forward_all()

            gnn_path = os.path.join(self.export_dir, f"gnn_weights_{tag}.weights.h5")
            if self.gnn_type in {"sage", "graphsage"} and hasattr(self.G, "G_model"):
                self.G.G_model.save_weights(gnn_path)
                exports["gnn_weights"] = gnn_path
                legacy_path = os.path.join(self.export_dir, "GNN_weights_sage.h5")
                try:
                    self.G.G_model.save_weights(legacy_path)
                    exports["gnn_weights_legacy"] = legacy_path
                except Exception:
                    pass
            elif hasattr(self.G, "save_weights"):
                self.G.save_weights(gnn_path)
                exports["gnn_weights"] = gnn_path
            else:
                exports["gnn_skipped"] = f"Model type '{tag}' does not expose save_weights."
        except Exception as exc:
            exports["gnn_error"] = str(exc)

        self.last_exported_weights = dict(exports)
        return exports
