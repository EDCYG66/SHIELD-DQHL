#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare GATv2 vs classic GAT vs GraphSAGE vs FC with flat output.

新增特性：
- 通过 --models 参数灵活选择要运行的模型集合：
  * --models gat
  * --models sage
  * --models fc
  * --models gat,sage
  * --models gat,sage,fc
  * --models gatclassic
  * --models gatclassic,gat,sage,fc
- 在单场景和车辆数扫描 (veh-list) 两种模式下，都用同一套 --models 逻辑。
"""
import time
import os

# --- 关键：确保 CUDA 路径和 XLA 标志 ---
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/environment/miniconda3/envs/tf212'

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- GPU 自检 ---
print("=" * 40)
print("[GPU Check]")
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f"✅ Found {len(gpus)} GPU(s): {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Memory growth set to True")
    except RuntimeError as e:
        print(f"⚠️ Memory growth setting failed: {e}")
else:
    print("❌ NO GPU FOUND! Check CUDA/Driver.")
print("=" * 40)

from highway_environment import HighwayTopoEnv
from agent import Agent
from gnn_factory import build_gnn


# ================= 工具函数 =================

def compute_entropy(prob: np.ndarray) -> float:
    prob = np.asarray(prob, dtype=np.float64)
    s = prob.sum()
    if s <= 0:
        return 0.0
    p = prob / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def compute_gini(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.size == 0:
        return 0.0
    s = counts.sum()
    if s <= 0:
        return 0.0
    diffs = np.abs(counts[:, None] - counts[None, :])
    return float(diffs.sum() / (2 * counts.size * s))


def compute_convergence_step(steps: List[int], values: List[float], ratio: float = 0.9) -> int:
    if not steps or not values:
        return -1
    final_v = values[-1]
    thr = ratio * final_v
    for s, v in zip(steps, values):
        if v >= thr:
            return int(s)
    return int(steps[-1])


def ensure_env_difficulty(env: HighwayTopoEnv, demand_amount: float, v2v_limit: float):
    env.demand_amount = float(demand_amount)
    env.demand = env.demand_amount * np.ones((env.n_Veh, 3))
    env.V2V_limit = float(v2v_limit)
    env.individual_time_limit = env.V2V_limit * np.ones((env.n_Veh, 3))
    if hasattr(env, "activate_links"):
        env.activate_links[:] = False


def parse_list_int(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_list_int_strict(s: str) -> List[int]:
    try:
        return parse_list_int(s)
    except Exception:
        return []


def parse_models(s: str) -> List[str]:
    """
    把 --models 参数解析成 ['gat','sage',...] 形式，并做一下合法性检查。
    """
    if not s:
        return ["gat", "sage"]  # 默认还是比较 GAT vs GraphSAGE
    raw = [x.strip().lower() for x in s.split(",") if x.strip()]
    allowed = {"gat", "gatclassic", "sage", "fc"}
    models = [m for m in raw if m in allowed]
    if not models:
        models = ["gat", "sage"]
    # 去重并保持原顺序
    seen = set()
    uniq = []
    for m in models:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq


# ================ 带统计的 Agent ================

class InstrumentedAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = 0
        self.rb_usage_counts = np.zeros(self.RB_number, dtype=np.int64)
        self.full_decision_time_acc = []
        self.gnn_only_time_acc = []
        self.v2i_curve_steps = []
        self.v2i_curve_vals = []
        self.v2v_curve_vals = []
        self.last_timeseries = None
        self._profile_step = False
        self._step_pred_time = 0.0
        self._step_gnn_time = 0.0

    def forward_embeddings(self, force=False):
        t0 = time.perf_counter()
        emb = super().forward_embeddings(force=force)
        if self._profile_step:
            self._step_gnn_time += (time.perf_counter() - t0)
        return emb

    def predict_batch(self, states_batch: np.ndarray, step: int, test_ep=False):
        t0 = time.perf_counter()
        actions = super().predict_batch(states_batch, step, test_ep=test_ep)
        if self._profile_step:
            self._step_pred_time += (time.perf_counter() - t0)
        return actions

    def _record_test_point(self, step_iter, test_sample: int):
        self.training = False
        mean_v2i, fail = self.test_environment(test_sample=test_sample, detailed=False)
        self.test_history.append((self.step, float(mean_v2i), float(fail)))
        self.v2i_curve_steps.append(self.step)
        self.v2i_curve_vals.append(float(mean_v2i))
        self.v2v_curve_vals.append(float(1.0 - fail))
        self.training = True
        if step_iter is not None:
            step_iter.set_postfix({"V2I": f"{mean_v2i:.2f}", "V2V": f"{1.0 - fail:.2f}"})

    def _collect_batch_stats(self):
        rb_flat = self.action_all_with_power_training[:, :, 0].reshape(-1)
        for rb in rb_flat:
            if 0 <= int(rb) < self.RB_number:
                self.rb_usage_counts[int(rb)] += 1
        used_blocks = np.unique(self.action_all_with_power_training[:, :, 0])
        self.used_blocks_history.append((self.step, len(used_blocks)))

    def _fc_train_loop_step_fast(self):
        s_old_all, indices = self.get_state_all()
        n_links = len(indices)
        if n_links == 0:
            return

        in_dim = self.dqn.input_dim
        batch_states = np.zeros((n_links, in_dim), dtype=np.float32)
        for k, (i, j) in enumerate(indices):
            s_old = s_old_all[k]
            copy_len = min(in_dim, s_old.shape[0])
            batch_states[k, :copy_len] = s_old[:copy_len]

        actions_list = self.predict_batch(batch_states, self.step)
        for k, (i, j) in enumerate(indices):
            action = int(actions_list[k])
            rb = action % self.RB_number
            pw = action // self.RB_number
            self.action_all_with_power_training[i, j, 0] = rb
            self.action_all_with_power_training[i, j, 1] = pw
            if self.power_log_stride and (self.step % self.power_log_stride == 0):
                time_left_s = float(s_old_all[k][-1]) * float(self.env.V2V_limit)
                self.power_log.append((time_left_s, pw))
                if len(self.power_log) > self.power_log_max:
                    self.power_log = self.power_log[-self.power_log_max:]

        reward_matrix, _, _ = self.env.batch_reward_all(self.action_all_with_power_training)
        s_new_all, _ = self.get_state_all()
        for k, (i, j) in enumerate(indices):
            post = np.zeros((in_dim,), dtype=np.float32)
            s_new = s_new_all[k]
            copy_len = min(in_dim, s_new.shape[0])
            post[:copy_len] = s_new[:copy_len]
            self.observe(batch_states[k], post, float(reward_matrix[i, j]), int(actions_list[k]), self.step)

        if self.step >= self.warmup_steps and self.step % self.train_every_n_steps == 0:
            self.q_learning_mini_batch()
        if self.step % self.target_q_update_step == 0:
            self.dqn.update_target_network()

        self._collect_batch_stats()

    def train(self, max_steps=1000, test_every_steps=200, test_sample=80):
        """保留 tqdm 和统计，但训练内核改成批量版。"""
        self.dqn.update_target_network()
        if self.gnn_type != "fc":
            self.warmup()

        self.env.new_random_game()
        if self.gnn_type != "fc":
            _ = self.initial_better_state(0, True)

        step_iter = tqdm(range(1, max_steps + 1), desc=f"Training {self.gnn_type}", ncols=80, leave=True)

        for step in step_iter:
            self.step = step
            if test_every_steps > 0 and self.step % test_every_steps == 0:
                self._record_test_point(step_iter, test_sample)

            self._step_pred_time = 0.0
            self._step_gnn_time = 0.0
            self._profile_step = True
            try:
                if self.gnn_type == "fc":
                    self._fc_train_loop_step_fast()
                else:
                    for i in range(len(self.env.vehicles)):
                        for j in range(3):
                            s_all = self.get_state([i, j])
                            self.G.features[3 * i + j, :] = s_all[:60]
                    self._update_channel_reward()
                    self.train_loop_step(
                        gnn_train_interval=getattr(self.G, "gat_train_interval", 20),
                        base_batch_size=self.memory.batch_size,
                    )
                    if self.step % self.target_q_update_step == 0:
                        self.dqn.update_target_network()
                        print(f"[Target] updated at step {self.step}")
            finally:
                self._profile_step = False

            n_links = max(1, 3 * len(self.env.vehicles))
            self.full_decision_time_acc.append((self._step_pred_time + self._step_gnn_time) / n_links)
            if self.gnn_type != "fc":
                self.gnn_only_time_acc.append(self._step_gnn_time / n_links)

        final_v2i, final_fail = self.test_environment(test_sample=test_sample, detailed=True)
        self.test_history.append((self.step, float(final_v2i), float(final_fail)))
        self.v2i_curve_steps.append(self.step)
        self.v2i_curve_vals.append(float(final_v2i))
        self.v2v_curve_vals.append(float(1.0 - final_fail))
        if hasattr(self, "_last_test_detailed") and self._last_test_detailed:
            self.last_timeseries = self._last_test_detailed.copy()
        self._export_results(final_v2i, final_fail)


# ================= 单次运行 =================

def run_one(model_type: str, args: argparse.Namespace, run_dir: Path) -> Dict:
    tag = model_type.lower()

    # 环境
    env = HighwayTopoEnv(
        n_up=args.n_up, n_down=args.n_down, lanes_per_dir=args.lanes, spacing=args.spacing,
        base_y=args.base_y, height=args.height, topology_type=args.topo,
        leader_at_front=bool(args.leader_front), leader_lane_up=args.leader_lane_up,
        leader_lane_down=args.leader_lane_down, leader_dynamic=bool(args.leader_dynamic),
        move_speed=float(args.use_move_speed), v2i_mode=args.v2i_mode,
        bs_layout=args.bs_layout, bs_spacing=args.bs_spacing,
        bs_min_stay_steps=args.bs_min_stay, bs_handover_hyst_m=args.bs_hyst,
        seed=args.seed
    )
    env.new_random_game()
    ensure_env_difficulty(env, args.demand_amount, args.v2v_limit)

    # 奖励相关超参
    env.beta_urgency_pos = float(args.beta_urgency_pos)
    env.beta_urgency_neg = float(args.beta_urgency_neg)
    env.urgency_threshold = float(args.urgency_threshold)
    env.rb_anti_conc_alpha = float(args.rb_anti_conc_alpha)
    env.rb_hot_threshold = float(args.rb_hot_threshold)
    env.rb_softmask_alpha = float(args.rb_softmask_alpha)
    env.rb_softmask_window = int(args.rb_softmask_window)

    # GNN
    G = build_gnn(
        env,
        gnn_type=tag,
        distance_threshold=150.0,
        lr=args.gat_lr,
        gat_train_interval=args.gat_train_interval,
        grad_clip=5.0,
        gat_hidden_dim=args.gat_hidden_dim,
        gat_out_dim=args.gat_out_dim,
        gat_heads=args.gat_heads,
        gat_attn_dropout=args.gat_attn_dropout,
        gat_top_k=args.gat_top_k,
        gat_use_proximity_edges=bool(args.enable_gat_proximity_edges),
        gat_proximity_radius=args.gat_proximity_radius,
        gat_max_proximity_neighbors=args.gat_max_proximity_neighbors,
        reprune_every=args.reprune_every,
        hysteresis_keep=args.hysteresis_keep,
        reprune_start_step=args.reprune_start_step,
        reg_attn_w=args.reg_attn_w,
        enable_reprune=(not args.disable_reprune),
    )

    # Agent：关键是 run_dir=str(run_dir)
    agent = InstrumentedAgent(
        [],
        env,
        gnn_type=tag,
        warmup_steps=args.warmup_steps,
        epsilon_decay_steps=args.epsilon_decay_steps,
        epsilon_min=args.epsilon_min,
        speed_mode=False,
        plot_dpi=args.dpi,
        beta_urgency_pos=args.beta_urgency_pos,
        beta_urgency_neg=args.beta_urgency_neg,
        urgency_threshold=args.urgency_threshold,
        rb_anti_conc_alpha=args.rb_anti_conc_alpha,
        rb_hot_threshold=args.rb_hot_threshold,
        rb_softmask_alpha=args.rb_softmask_alpha,
        rb_softmask_window=args.rb_softmask_window,
        dynamic_graph_refresh=(not args.disable_dynamic_graph_refresh),
        graph_refresh_steps=args.graph_refresh_steps,
        run_dir=str(run_dir),
    )
    agent.G = G

    agent.train(max_steps=args.train_steps, test_every_steps=args.test_every, test_sample=args.test_sample)

    # 统计 + 导出（这些 csv 会写到 run_dir 下）
    power_counts = np.zeros(3, dtype=np.int64)
    for _, pw in agent.power_log:
        if 0 <= pw < 3:
            power_counts[pw] += 1
    power_entropy = compute_entropy(power_counts)
    rb_gini = compute_gini(agent.rb_usage_counts)
    unique_rb_ratio = float(np.mean([c / agent.RB_number for _, c in agent.used_blocks_history])) if agent.used_blocks_history else 0.0
    conv_step_v2i = compute_convergence_step(agent.v2i_curve_steps, agent.v2i_curve_vals, 0.9)
    full_mean = float(np.mean(agent.full_decision_time_acc)) if agent.full_decision_time_acc else 0.0
    gnn_only_mean = float(np.mean(agent.gnn_only_time_acc)) if agent.gnn_only_time_acc else 0.0
    v2i_final = float(agent.v2i_curve_vals[-1]) if agent.v2i_curve_vals else 0.0
    v2v_final = float(agent.v2v_curve_vals[-1]) if agent.v2v_curve_vals else 0.0

    summary = dict(
        model=tag,
        v2i_final=v2i_final,
        v2v_final=v2v_final,
        v2i_conv_step=int(conv_step_v2i),
        power_entropy=float(power_entropy),
        power_counts=power_counts.tolist(),
        rb_unique_ratio=float(unique_rb_ratio),
        rb_gini=float(rb_gini),
        decision_time_full_s=full_mean,
        decision_time_gnn_only_s=gnn_only_mean,
        steps=len(agent.v2i_curve_steps),
        v2i_curve=list(zip(agent.v2i_curve_steps, agent.v2i_curve_vals)),
        v2v_curve=list(zip(agent.v2i_curve_steps, agent.v2v_curve_vals)),
        rb_usage_counts=agent.rb_usage_counts.tolist(),
        last_timeseries=agent.last_timeseries if agent.last_timeseries else None,
        weight_exports=dict(getattr(agent, "last_exported_weights", {})),
        agent_ref=agent,
    )

    # 额外：再在 run_dir 下保存一份简化 v2i/v2v 曲线 CSV
    csv_path_v2v = run_dir / f"{tag}_v2v_curve.csv"
    with csv_path_v2v.open("w", encoding="utf-8") as f:
        f.write("step,v2v_success\n")
        for s, v in summary["v2v_curve"]:
            f.write(f"{s},{v}\n")
    csv_path_v2i = run_dir / f"{tag}_v2i_curve.csv"
    with csv_path_v2i.open("w", encoding="utf-8") as f:
        f.write("step,v2i_rate\n")
        for s, v in summary["v2i_curve"]:
            f.write(f"{s},{v}\n")

    return summary


# ================= 车辆规模扫描 =================

def run_sweep_vehicles(args: argparse.Namespace, out_dir: Path, models: List[str]) -> Dict[str, Dict[str, List[float]]]:
    veh_list = parse_list_int_strict(getattr(args, "veh_list", ""))
    if not veh_list:
        return {}

    print(f"\n>>> [Sweep Start] Vehicles: {veh_list} | Models: {models}")
    results = {m: {"veh": [], "v2i": [], "v2v": []} for m in models}

    for nveh in veh_list:
        n_up = int(np.ceil(nveh / 2))
        n_down = int(nveh - n_up)

        for model in models:
            print(f"   -> Running: Veh={nveh}, Model={model} ...")
            env = HighwayTopoEnv(
                n_up=n_up, n_down=n_down, lanes_per_dir=args.lanes, spacing=args.spacing,
                base_y=args.base_y, height=args.height, topology_type=args.topo,
                v2i_mode=args.v2i_mode, bs_layout=args.bs_layout, bs_spacing=args.bs_spacing,
                seed=args.seed
            )
            env.new_random_game()
            ensure_env_difficulty(env, args.demand_amount, args.v2v_limit)

            env.beta_urgency_pos = float(args.beta_urgency_pos)
            env.beta_urgency_neg = float(args.beta_urgency_neg)
            env.urgency_threshold = float(args.urgency_threshold)
            env.rb_anti_conc_alpha = float(args.rb_anti_conc_alpha)
            env.rb_hot_threshold = float(args.rb_hot_threshold)
            env.rb_softmask_alpha = float(args.rb_softmask_alpha)
            env.rb_softmask_window = int(args.rb_softmask_window)

            G = build_gnn(
                env,
                gnn_type=model,
                distance_threshold=150.0,
                lr=args.gat_lr,
                gat_train_interval=args.gat_train_interval,
                grad_clip=5.0,
                gat_hidden_dim=args.gat_hidden_dim,
                gat_out_dim=args.gat_out_dim,
                gat_heads=args.gat_heads,
                gat_attn_dropout=args.gat_attn_dropout,
                gat_top_k=args.gat_top_k,
                gat_use_proximity_edges=bool(args.enable_gat_proximity_edges),
                gat_proximity_radius=args.gat_proximity_radius,
                gat_max_proximity_neighbors=args.gat_max_proximity_neighbors,
                reprune_every=args.reprune_every,
                hysteresis_keep=args.hysteresis_keep,
                reprune_start_step=args.reprune_start_step,
                reg_attn_w=args.reg_attn_w,
                enable_reprune=(not args.disable_reprune),
            )

            ag = InstrumentedAgent(
                [], env, gnn_type=model,
                warmup_steps=args.warmup_steps,
                epsilon_decay_steps=args.epsilon_decay_steps,
                epsilon_min=args.epsilon_min,
                speed_mode=False,
                plot_dpi=args.dpi,
                beta_urgency_pos=args.beta_urgency_pos,
                beta_urgency_neg=args.beta_urgency_neg,
                urgency_threshold=args.urgency_threshold,
                rb_anti_conc_alpha=args.rb_anti_conc_alpha,
                rb_hot_threshold=args.rb_hot_threshold,
                rb_softmask_alpha=args.rb_softmask_alpha,
                rb_softmask_window=args.rb_softmask_window,
                dynamic_graph_refresh=(not args.disable_dynamic_graph_refresh),
                graph_refresh_steps=args.graph_refresh_steps,
            )
            ag.G = G

            sweep_steps = args.train_steps
            ag.train(max_steps=sweep_steps, test_every_steps=args.test_every, test_sample=args.test_sample)

            v2i_final = ag.v2i_curve_vals[-1] if ag.v2i_curve_vals else 0.0
            v2v_final = ag.v2v_curve_vals[-1] if ag.v2v_curve_vals else 0.0

            results[model]["veh"].append(nveh)
            results[model]["v2i"].append(float(v2i_final))
            results[model]["v2v"].append(float(v2v_final))

    csv_path = out_dir / "sweep_vehicles.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("model,n_veh,v2i_mean,v2v_success\n")
        for model in models:
            for n, a, b in zip(results[model]["veh"], results[model]["v2i"], results[model]["v2v"]):
                f.write(f"{model},{n},{a},{b}\n")

    print(f">>> Sweep complete. Data saved to {csv_path}")
    return results


# ================= 主流程 =================

def main():
    ap = argparse.ArgumentParser(description="Compare GATv2 vs classic GAT vs GraphSAGE vs FC.")
    ap.add_argument("--n-up", type=int, default=12)
    ap.add_argument("--n-down", type=int, default=20)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--spacing", type=float, default=25.0)
    ap.add_argument("--base-y", type=float, default=0.0)
    ap.add_argument("--height", type=float, default=1200.0)
    ap.add_argument("--topo", type=str, default="tree", choices=["star", "tree"])
    ap.add_argument("--leader-front", action="store_true")
    ap.add_argument("--leader-lane-up", type=int, default=None)
    ap.add_argument("--leader-lane-down", type=int, default=None)
    ap.add_argument("--leader-dynamic", action="store_true")
    ap.add_argument("--use-move-speed", type=float, default=0.0)
    ap.add_argument("--v2i-mode", type=str, default="rsu", choices=["rsu", "single"])
    ap.add_argument("--bs-layout", type=str, default="median", choices=["median", "dual-roadside"])
    ap.add_argument("--bs-spacing", type=float, default=250.0)
    ap.add_argument("--bs-min-stay", type=int, default=5)
    ap.add_argument("--bs-hyst", type=float, default=15.0)
    ap.add_argument("--train-steps", type=int, default=2400)
    ap.add_argument("--test-every", type=int, default=200)
    ap.add_argument("--test-sample", type=int, default=80)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--epsilon-decay-steps", type=int, default=800)
    ap.add_argument("--epsilon-min", type=float, default=0.01)

    ap.add_argument("--demand-amount", type=float, default=130.0)
    ap.add_argument("--v2v-limit", type=float, default=0.045)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--dpi", type=int, default=224)
    ap.add_argument("--out-dir", type=str, default="runs")

    ap.add_argument("--models", type=str, default="gat,sage",
                    help="Comma-separated list of models to run: gat,gatclassic,sage,fc")
    ap.add_argument("--gnn-type", type=str, choices=["gat", "gatclassic", "sage", "fc"], default="gat")

    ap.add_argument("--reprune-every", type=int, default=300)
    ap.add_argument("--hysteresis-keep", type=float, default=0.5)
    ap.add_argument("--reprune-start-step", type=int, default=600)
    ap.add_argument("--reg-attn-w", type=float, default=1e-3)
    ap.add_argument("--gat-lr", type=float, default=5e-4)
    ap.add_argument("--gat-train-interval", type=int, default=20)
    ap.add_argument("--gat-hidden-dim", type=int, default=32)
    ap.add_argument("--gat-out-dim", type=int, default=32)
    ap.add_argument("--gat-heads", type=int, default=2)
    ap.add_argument("--gat-attn-dropout", type=float, default=0.0)
    ap.add_argument("--gat-top-k", type=int, default=6)
    ap.add_argument("--gat-proximity-radius", type=float, default=180.0)
    ap.add_argument("--gat-max-proximity-neighbors", type=int, default=6)
    ap.add_argument("--enable-gat-proximity-edges", action="store_true")
    ap.add_argument("--graph-refresh-steps", type=int, default=0)
    ap.add_argument("--disable-dynamic-graph-refresh", action="store_true")
    ap.add_argument("--disable-reprune", action="store_true",
                    help="Disable adaptive reprune explicitly for GAT/GATv2.")
    ap.add_argument("--beta-urgency-pos", type=float, default=0.018)
    ap.add_argument("--beta-urgency-neg", type=float, default=0.025)
    ap.add_argument("--urgency-threshold", type=float, default=0.25)
    ap.add_argument("--rb-anti-conc-alpha", type=float, default=0.012)
    ap.add_argument("--rb-hot-threshold", type=float, default=0.22)
    ap.add_argument("--rb-softmask-alpha", type=float, default=0.15)
    ap.add_argument("--rb-softmask-window", type=int, default=50)
    ap.add_argument("--veh-list", type=str, default="")
    ap.add_argument("--seeds", type=str, default="")

    args = ap.parse_args()
    np.random.seed(args.seed)

    base_out = Path(args.out_dir).expanduser().resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    models = parse_models(args.models)
    print(f"[Config] Models to run: {models}")

    combo_summary = {"settings": vars(args), "models": models}

    # 北京时间（UTC+8）
    from datetime import datetime, timezone, timedelta
    bj_tz = timezone(timedelta(hours=8))

    # 模式选择
    if getattr(args, "veh_list", ""):
        print(">>> Mode: Vehicle Density Sweep")
        sweep_results = run_sweep_vehicles(args, base_out, models)
        combo_summary["sweep"] = sweep_results
    else:
        print(">>> Mode: Single-scenario run")
        task_name = "train"
        for m in models:
            now_bj = datetime.now(bj_tz)
            time_str = now_bj.strftime("%Y-%m-%d-%H-%M-%S")
            run_dir_name = f"{m}-{task_name}-{time_str}"
            run_dir = base_out / run_dir_name
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"  -> Running model: {m.upper()}, outputs -> {run_dir}")
            sum_res = run_one(m, args, run_dir)
            combo_summary[m] = {k: v for k, v in sum_res.items() if k != "agent_ref"}

    with (base_out / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(combo_summary, f, indent=2, ensure_ascii=False)

    print("[OK] Run complete.")
    print("Results saved to:", base_out.resolve())


if __name__ == "__main__":
    main()
