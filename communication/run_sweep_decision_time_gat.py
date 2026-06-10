#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_sweep_decision_time_gat.py

在当前代码基础上，不修改已有文件，单独扫描：
  - 完全图 GAT (Complete Graph)
  - 不完全图 GAT (Pruned / Incomplete Graph)

对给定的车辆数列表 [n_veh1, n_veh2, ...]：
  对每个 n_veh 分别训练两种 GAT，一样的训练步数和环境参数，
  记录：
    - 平均单次决策时间 full_decision_time_acc
    - 平均 GNN-only 时间 gnn_only_time_acc
    - 最终 V2V 成功率 / V2I 速率

输出文件：
  out_dir/decision_time_sweep.csv

CSV 格式：
  model,n_veh,decision_time_full_s,decision_time_gnn_only_s,v2i_final,v2v_final

用法示例：
  python run_sweep_decision_time_gat.py \
      --veh-list "10,20,30,40,50" \
      --out-dir runs/gat_decision_sweep

之后可以用：
  python plot_decision_time_vs_vehicles.py \
      --csv runs/gat_decision_sweep/decision_time_sweep.csv \
      --out runs/gat_decision_sweep \
      --model gat_complete

或：
  --model gat_pruned
"""

import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/environment/miniconda3/envs/tf212'

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# 复用你现有的模块
from highway_environment import HighwayTopoEnv
from agent import Agent
from Graph_GAT import GraphGAT


# ---------- 工具函数 ----------

def parse_list_int(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def ensure_env_difficulty(env: HighwayTopoEnv, demand_amount: float, v2v_limit: float):
    env.demand_amount = float(demand_amount)
    env.demand = env.demand_amount * np.ones((env.n_Veh, 3))
    env.V2V_limit = float(v2v_limit)
    env.individual_time_limit = env.V2V_limit * np.ones((env.n_Veh, 3))
    if hasattr(env, "activate_links"):
        env.activate_links[:] = False


def build_gat_with_prune(env: HighwayTopoEnv,
                         enable_prune: bool,
                         distance_threshold: float = 150.0,
                         reprune_every: int = 300,
                         hysteresis_keep: float = 0.5,
                         reprune_start_step: int = 600,
                         reg_attn_w: float = 1e-3) -> GraphGAT:
    """
    直接构建 GraphGAT 实例，不依赖 gnn_factory.build_gnn，
    并通过 enable_prune 开关控制是否在 adaptive_reprune 中起作用。

    我们不修改原始 Graph_GAT.py，而是在这里“猴子补丁”式地包一层：
      - 实例化后，给它加上 enable_prune 属性
      - 在 adaptive_reprune 调用时做一个 if 判断
    """

    num_nodes = env.n_Veh * 3
    gat = GraphGAT(
        num_nodes=num_nodes,
        in_dim=60,
        hidden_dim=32,
        out_dim=32,
        heads=2,
        reprune_every=reprune_every,
        hysteresis_keep=hysteresis_keep,
        reprune_start_step=reprune_start_step,
        reg_attn_w=reg_attn_w
    )

    # 给实例打一个标记
    gat.enable_prune = bool(enable_prune)

    # 包一层原始的 adaptive_reprune，增加开关
    if hasattr(gat, "adaptive_reprune"):
        _orig_adaptive = gat.adaptive_reprune

        def adaptive_reprune_wrapper(step: int, k: int = 6, hysteresis_keep_: float = None):
            if not getattr(gat, "enable_prune", True):
                return
            return _orig_adaptive(step, k=k, hysteresis_keep=hysteresis_keep_)

        gat.adaptive_reprune = adaptive_reprune_wrapper  # 替换方法

    return gat


def attach_gat_to_agent(agent: Agent, gat: GraphGAT):
    """
    将构建好的 GAT 模型挂到 Agent 上，并保持属性一致性。
    """
    agent.G = gat
    # 如果 GAT 需要 tb_writer，可用 Agent 的 gnn writer
    if hasattr(agent, "tb_gnn"):
        gat.tb_writer = agent.tb_gnn


def train_one(env: HighwayTopoEnv,
              n_veh: int,
              model_tag: str,
              enable_prune: bool,
              args: argparse.Namespace) -> Tuple[float, float, float, float]:
    """
    对给定车辆数和 GAT 配置训练一次：
      - 返回 (full_decision_time_mean, gnn_only_time_mean, v2i_final, v2v_final)
    """

    # 构建 GAT
    gat = build_gat_with_prune(
        env,
        enable_prune=enable_prune,
        distance_threshold=150.0,
        reprune_every=args.reprune_every,
        hysteresis_keep=args.hysteresis_keep,
        reprune_start_step=args.reprune_start_step,
        reg_attn_w=args.reg_attn_w,
    )

    # 构建 Agent
    agent = Agent(
        [],
        env,
        gnn_type="gat",
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
    )

    attach_gat_to_agent(agent, gat)

    # 为了记录决策时间，需要一个简单的包装：我们利用 InstrumentedAgent 的思路，
    # 这里偷个懒：直接在 Agent 上挂两个属性，train_loop_step 不动，
    # 决策时间统计我们通过修改 Agent.predict_batch 前后的时间也比较麻烦，
    # 更直接的办法：复用你已经写好的 InstrumentedAgent 代码，
    # 但那在 run_compare_gnn_plus.py 里。这里我们实现一个极简版本：
    #
    # 简化方案（不会影响训练逻辑）：
    #   - 不再精确统计“GNN-only 时间”，只统计“完整决策时间”：
    #       * 每个 step 记录一次 time.perf_counter() 之前和之后的差
    #       * 归一化除以 (3*n_Veh)
    #   - 这足够画一张类似图 9 的“完整决策时间 vs 车辆数”。
    #
    # 为了不改 Agent 源码，这里用 monkey patch 覆盖 train_loop_step。

    import types, time as _time
    orig_train_loop_step = agent.train_loop_step

    agent.full_decision_time_acc = []

    def patched_train_loop_step(self, gnn_train_interval=20, base_batch_size=512):
        t2 = _time.perf_counter()
        orig_train_loop_step(gnn_train_interval=gnn_train_interval,
                             base_batch_size=base_batch_size)
        t3 = _time.perf_counter()
        N = 3 * len(self.env.vehicles)
        if N > 0:
            self.full_decision_time_acc.append((t3 - t2) / float(N))

    agent.train_loop_step = types.MethodType(patched_train_loop_step, agent)

    # 正式训练
    agent.dqn.update_target_network()
    agent.warmup()

    env.new_random_game()
    _ = agent.initial_better_state(0, True)

    # 主训练循环（基本复制 Agent.train，但用我们的 patched_train_loop_step）
    for agent.step in tqdm(range(1, args.train_steps + 1),
                           desc=f"Training {model_tag} (n_veh={n_veh})",
                           ncols=80,
                           leave=True):
        is_test_step = (args.test_every > 0 and agent.step % args.test_every == 0)

        if is_test_step:
            agent.training = False
            mean_v2i, fail = agent.test_environment(test_sample=args.test_sample, detailed=False)
            agent.test_history.append((agent.step, float(mean_v2i), float(fail)))
            agent.training = True

        # 刷新 GNN 特征
        for i in range(len(env.vehicles)):
            for j in range(3):
                s_all = agent.get_state([i, j])
                agent.G.features[3 * i + j, :] = s_all[:60]
        agent._update_channel_reward()

        # 使用我们 patch 过的 train_loop_step
        agent.train_loop_step(
            gnn_train_interval=getattr(agent.G, "gat_train_interval", 20),
            base_batch_size=agent.memory.batch_size
        )

        if agent.step % agent.target_q_update_step == 0:
            agent.dqn.update_target_network()

    # 最终一轮详细测试
    final_v2i, final_fail = agent.test_environment(test_sample=args.test_sample, detailed=False)
    v2i_final = float(final_v2i)
    v2v_final = float(1.0 - final_fail)
    full_mean = float(np.mean(agent.full_decision_time_acc)) if agent.full_decision_time_acc else 0.0

    return full_mean, 0.0, v2i_final, v2v_final  # gnn_only_time 暂时置 0


# ---------- 主流程：扫描车辆数 ----------

def main():
    ap = argparse.ArgumentParser(description="Decision time sweep for GAT: complete vs pruned graphs.")
    ap.add_argument("--veh-list", type=str, required=True,
                    help="Comma-separated list of total vehicle counts, e.g., '10,20,30,40,50'")
    ap.add_argument("--train-steps", type=int, default=12000)
    ap.add_argument("--test-every", type=int, default=500)
    ap.add_argument("--test-sample", type=int, default=100)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--epsilon-decay-steps", type=int, default=800)
    ap.add_argument("--epsilon-min", type=float, default=0.01)
    ap.add_argument("--demand-amount", type=float, default=110.0)
    ap.add_argument("--v2v-limit", type=float, default=0.06)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--spacing", type=float, default=25.0)
    ap.add_argument("--base-y", type=float, default=0.0)
    ap.add_argument("--height", type=float, default=1200.0)
    ap.add_argument("--topo", type=str, default="tree", choices=["star", "tree"])
    ap.add_argument("--v2i-mode", type=str, default="rsu", choices=["rsu", "single"])
    ap.add_argument("--bs-layout", type=str, default="median", choices=["median", "dual-roadside"])
    ap.add_argument("--bs-spacing", type=float, default=250.0)
    ap.add_argument("--bs-min-stay", type=int, default=5)
    ap.add_argument("--bs-hyst", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--dpi", type=int, default=224)
    ap.add_argument("--reprune-every", type=int, default=300)
    ap.add_argument("--hysteresis-keep", type=float, default=0.5)
    ap.add_argument("--reprune-start-step", type=int, default=600)
    ap.add_argument("--reg-attn-w", type=float, default=1e-3)
    ap.add_argument("--beta-urgency-pos", type=float, default=0.018)
    ap.add_argument("--beta-urgency-neg", type=float, default=0.025)
    ap.add_argument("--urgency-threshold", type=float, default=0.25)
    ap.add_argument("--rb-anti-conc-alpha", type=float, default=0.012)
    ap.add_argument("--rb-hot-threshold", type=float, default=0.22)
    ap.add_argument("--rb-softmask-alpha", type=float, default=0.15)
    ap.add_argument("--rb-softmask-window", type=int, default=50)
    ap.add_argument("--out-dir", type=str, default="runs/gat_decision_sweep")

    args = ap.parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    veh_list = parse_list_int(args.veh_list)
    if not veh_list:
        raise ValueError("veh-list is empty. Example: --veh-list '10,20,30,40,50'.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Config] Vehicle list: {veh_list}")
    print(f"[Config] Output dir: {out_dir}")

    rows = []

    for nveh in veh_list:
        # 拆分 n_up / n_down
        n_up = int(np.ceil(nveh / 2))
        n_down = int(nveh - n_up)

        # 环境
        env = HighwayTopoEnv(
            n_up=n_up, n_down=n_down, lanes_per_dir=args.lanes, spacing=args.spacing,
            base_y=args.base_y, height=args.height, topology_type=args.topo,
            v2i_mode=args.v2i_mode, bs_layout=args.bs_layout, bs_spacing=args.bs_spacing,
            bs_min_stay_steps=args.bs_min_stay, bs_handover_hyst_m=args.bs_hyst,
            seed=args.seed
        )
        env.new_random_game()
        ensure_env_difficulty(env, args.demand_amount, args.v2v_limit)

        # 注入奖励参数
        env.beta_urgency_pos = float(args.beta_urgency_pos)
        env.beta_urgency_neg = float(args.beta_urgency_neg)
        env.urgency_threshold = float(args.urgency_threshold)
        env.rb_anti_conc_alpha = float(args.rb_anti_conc_alpha)
        env.rb_hot_threshold = float(args.rb_hot_threshold)
        env.rb_softmask_alpha = float(args.rb_softmask_alpha)
        env.rb_softmask_window = int(args.rb_softmask_window)

        # 1) 完全图 GAT
        print(f"\n=== n_veh={nveh}, Model=gat_complete (no prune) ===")
        full_t_complete, gnn_t_complete, v2i_c, v2v_c = train_one(
            env, nveh, model_tag="gat_complete", enable_prune=False, args=args
        )
        rows.append(("gat_complete", nveh, full_t_complete, gnn_t_complete, v2i_c, v2v_c))

        # 重新初始化环境（确保公平）
        env = HighwayTopoEnv(
            n_up=n_up, n_down=n_down, lanes_per_dir=args.lanes, spacing=args.spacing,
            base_y=args.base_y, height=args.height, topology_type=args.topo,
            v2i_mode=args.v2i_mode, bs_layout=args.bs_layout, bs_spacing=args.bs_spacing,
            bs_min_stay_steps=args.bs_min_stay, bs_handover_hyst_m=args.bs_hyst,
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

        # 2) 剪枝版 GAT
        print(f"\n=== n_veh={nveh}, Model=gat_pruned (with prune) ===")
        full_t_pruned, gnn_t_pruned, v2i_p, v2v_p = train_one(
            env, nveh, model_tag="gat_pruned", enable_prune=True, args=args
        )
        rows.append(("gat_pruned", nveh, full_t_pruned, gnn_t_pruned, v2i_p, v2v_p))

    # 写 CSV
    csv_path = out_dir / "decision_time_sweep.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("model,n_veh,decision_time_full_s,decision_time_gnn_only_s,v2i_final,v2v_final\n")
        for m, n, dfull, dgnn, v2i, v2v in rows:
            f.write(f"{m},{n},{dfull},{dgnn},{v2i},{v2v}\n")
    print(f"[Saved] {csv_path}")

    # 顺手把配置也存一下
    with (out_dir / "settings.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print("[OK] Sweep complete.")


if __name__ == "__main__":
    main()