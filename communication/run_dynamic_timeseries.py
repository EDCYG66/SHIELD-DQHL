#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用已经训练好的 GAT 模型，在一系列不同车辆数的场景下
生成类似论文图 10/11 的动态时序数据：
  - 时间轴 0..T-1
  - nveh 从 nveh_min 线性增加到 nveh_max
  - 每隔 refresh_every 步，重建一个新的 HighwayTopoEnv (n_up,n_down 不同)
  - 每步用当前策略重新决策一次，记录 v2i / v2v / nveh

生成:
  runs/v2v_gat_xxx_dynamic/timeseries_gat_dynamic.csv
  然后用 plot_timeseries_and_box.py 画图 10 / 箱线图。
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from highway_environment import HighwayTopoEnv
from agent import Agent
from run_compare_gnn_plus import ensure_env_difficulty


def build_env_with_nveh(nveh: int, args) -> HighwayTopoEnv:
    n_up = int(np.ceil(nveh / 2))
    n_down = int(nveh - n_up)
    env = HighwayTopoEnv(
        n_up=n_up,
        n_down=n_down,
        lanes_per_dir=args.lanes,
        spacing=args.spacing,
        base_y=args.base_y,
        height=args.height,
        topology_type=args.topo,
        v2i_mode=args.v2i_mode,
        bs_layout=args.bs_layout,
        bs_spacing=args.bs_spacing,
        bs_min_stay_steps=args.bs_min_stay,
        bs_handover_hyst_m=args.bs_hyst,
        seed=args.seed
    )
    env.new_random_game()
    ensure_env_difficulty(env, args.demand_amount, args.v2v_limit)

    # 奖励相关参数保持和训练时一致
    env.beta_urgency_pos = float(args.beta_urgency_pos)
    env.beta_urgency_neg = float(args.beta_urgency_neg)
    env.urgency_threshold = float(args.urgency_threshold)
    env.rb_anti_conc_alpha = float(args.rb_anti_conc_alpha)
    env.rb_hot_threshold = float(args.rb_hot_threshold)
    env.rb_softmask_alpha = float(args.rb_softmask_alpha)
    env.rb_softmask_window = int(args.rb_softmask_window)

    return env


def dynamic_timeseries(args):
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 先构建一个“初始 env + Agent”，加载训练好的权重（如果有的话）
    env = build_env_with_nveh(args.nveh_min, args)
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

    # 如果你以后把 DQN/GNN 权重保存到文件，这里可以 load；目前先使用刚训练好的内存模型也可以

    inst_t, inst_v2i, inst_v2v, inst_nveh = [], [], [], []

    def target_nveh(t_idx: int) -> int:
        if args.T <= 1:
            return args.nveh_min
        r = t_idx / float(args.T - 1)
        return int(args.nveh_min + r * (args.nveh_max - args.nveh_min))

    cur_env = env

    for t in range(args.T):
        # 当需要切换车辆数时，重建 env 并绑定到 agent
        if (t % args.refresh_every == 0) or (len(cur_env.vehicles) == 0):
            nveh = target_nveh(t)
            cur_env = build_env_with_nveh(nveh, args)
            agent.env = cur_env
            agent._ensure_action_buffers()
            _ = agent.initial_better_state(0, True)

        # 1) 当前状态 + GNN 特征
        s_old_all, indices = agent.get_state_all()
        n_links = len(indices)
        if n_links == 0:
            continue
        for k, (i, j) in enumerate(indices):
            agent.G.features[3 * i + j, :] = s_old_all[k, :60]
        agent._update_channel_reward()
        emb_all = agent.forward_embeddings(force=(t == 0))

        batch_states = np.zeros((n_links, 114), dtype=np.float32)
        for k, (i, j) in enumerate(indices):
            emb = emb_all[3 * i + j]
            s_old = s_old_all[k]
            batch_states[k] = np.concatenate((emb, s_old), axis=0)

        actions_list = agent.predict_batch(batch_states,
                                           step=getattr(agent, "step", 0),
                                           test_ep=True)
        for k, (i, j) in enumerate(indices):
            act = int(actions_list[k])
            rb = act % agent.RB_number
            pw = act // agent.RB_number
            agent.action_all_with_power_training[i, j, 0] = rb
            agent.action_all_with_power_training[i, j, 1] = pw

        reward_vec, fail = agent.env.act_asyn(agent.action_all_with_power_training)
        v2i_rate = float(np.sum(reward_vec))
        v2v_succ = float(1.0 - fail)
        nveh_now = float(len(agent.env.vehicles))

        inst_t.append(t)
        inst_v2i.append(v2i_rate)
        inst_v2v.append(v2v_succ)
        inst_nveh.append(nveh_now)

    # 写 CSV
    csv_path = out_dir / "timeseries_gat_dynamic.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("t,v2i_rate,v2v_success,nveh\n")
        for ti, a, b, c in zip(inst_t, inst_v2i, inst_v2v, inst_nveh):
            f.write(f"{ti},{a},{b},{c}\n")
    print("[Saved]", csv_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nveh-min", type=int, default=20)
    ap.add_argument("--nveh-max", type=int, default=160)
    ap.add_argument("--T", type=int, default=250)
    ap.add_argument("--refresh-every", type=int, default=25)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--spacing", type=float, default=25.0)
    ap.add_argument("--base-y", type=float, default=0.0)
    ap.add_argument("--height", type=float, default=1200.0)
    ap.add_argument("--topo", type=str, default="tree")
    ap.add_argument("--v2i-mode", type=str, default="rsu")
    ap.add_argument("--bs-layout", type=str, default="median")
    ap.add_argument("--bs-spacing", type=float, default=250.0)
    ap.add_argument("--bs-min-stay", type=int, default=5)
    ap.add_argument("--bs-hyst", type=float, default=15.0)
    ap.add_argument("--demand-amount", type=float, default=110.0)
    ap.add_argument("--v2v-limit", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--dpi", type=int, default=224)

    # 这些和训练时保持一致即可
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--epsilon-decay-steps", type=int, default=6000)
    ap.add_argument("--epsilon-min", type=float, default=0.05)
    ap.add_argument("--beta-urgency-pos", type=float, default=0.018)
    ap.add_argument("--beta-urgency-neg", type=float, default=0.025)
    ap.add_argument("--urgency-threshold", type=float, default=0.25)
    ap.add_argument("--rb-anti-conc-alpha", type=float, default=0.012)
    ap.add_argument("--rb-hot-threshold", type=float, default=0.22)
    ap.add_argument("--rb-softmask-alpha", type=float, default=0.15)
    ap.add_argument("--rb-softmask-window", type=int, default=50)

    ap.add_argument("--out-dir", type=str, default="runs/gat_dynamic_timeseries")

    args = ap.parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    dynamic_timeseries(args)


if __name__ == "__main__":
    main()