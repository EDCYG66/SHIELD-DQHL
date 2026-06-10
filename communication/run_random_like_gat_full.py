#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random baseline with full-length evaluation, using the SAME environment
parameters and测试频率 as GAT/SAGE/FC.

- 环境参数、demand_amount、V2V_limit 与 GAT 一致
- 模拟 max_steps 步训练，每步用随机动作推进环境（不学习）
- 每隔 test_every 步，用同样的 test_sample 次随机策略评估一次
- 输出:
    runs/v2v_random_full/exports/random_v2i_curve.csv
    runs/v2v_random_full/exports/random_v2v_curve.csv
    runs/v2v_random_full/exports/training_effect_random.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from highway_environment import HighwayTopoEnv
from run_compare_gnn_plus import ensure_env_difficulty


def main():
    ap = argparse.ArgumentParser(description="Full-length Random baseline run.")
    ap.add_argument("--n-up", type=int, default=8)
    ap.add_argument("--n-down", type=int, default=12)
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

    ap.add_argument("--train-steps", type=int, default=12000)
    ap.add_argument("--test-every", type=int, default=500)
    ap.add_argument("--test-sample", type=int, default=100)

    ap.add_argument("--demand-amount", type=float, default=110.0)
    ap.add_argument("--v2v-limit", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--dpi", type=int, default=224)
    ap.add_argument("--out-dir", type=str, default="runs/v2v_random_full")

    args = ap.parse_args()
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir).expanduser().resolve()
    export_dir = out_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1) 环境配置与 GAT/SAGE/FC 完全一致
    env = HighwayTopoEnv(
        n_up=args.n_up,
        n_down=args.n_down,
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
        seed=args.seed,
    )
    env.new_random_game()
    ensure_env_difficulty(env, args.demand_amount, args.v2v_limit)

    n_veh = env.n_Veh
    n_rb = env.n_RB if hasattr(env, "n_RB") else 20
    actions = np.zeros((n_veh, 3, 2), dtype=np.int32)

    v2i_steps = []
    v2i_vals = []
    v2v_vals = []

    # 2) 主循环：每步随机动作推进环境，定期评估
    for step in range(1, args.train_steps + 1):
        # 随机动作推进一次环境
        actions[:, :, 0] = np.random.randint(0, n_rb, size=(n_veh, 3))
        actions[:, :, 1] = np.random.randint(0, 3, size=(n_veh, 3))
        _, _ = env.act_asyn(actions)

        if args.test_every > 0 and step % args.test_every == 0:
            v2i_list = []
            v2v_list = []
            for _ in range(args.test_sample):
                actions[:, :, 0] = np.random.randint(0, n_rb, size=(n_veh, 3))
                actions[:, :, 1] = np.random.randint(0, 3, size=(n_veh, 3))
                reward_vec, fail = env.act_asyn(actions)
                v2i_list.append(float(np.sum(reward_vec)))
                v2v_list.append(float(1.0 - fail))
            mean_v2i = float(np.mean(v2i_list)) if v2i_list else 0.0
            mean_v2v = float(np.mean(v2v_list)) if v2v_list else 0.0

            v2i_steps.append(step)
            v2i_vals.append(mean_v2i)
            v2v_vals.append(mean_v2v)
            print(f"[RANDOM TEST] step={step} v2i={mean_v2i:.3f} v2v={mean_v2v:.3f}")

    # 3) 保存曲线 CSV（结构跟 *_v2i_curve.csv 一样）
    csv_v2i = export_dir / "random_v2i_curve.csv"
    with csv_v2i.open("w", encoding="utf-8") as f:
        f.write("step,v2i_rate\n")
        for s, v in zip(v2i_steps, v2i_vals):
            f.write(f"{s},{v}\n")

    csv_v2v = export_dir / "random_v2v_curve.csv"
    with csv_v2v.open("w", encoding="utf-8") as f:
        f.write("step,v2v_success\n")
        for s, v in zip(v2i_steps, v2v_vals):
            f.write(f"{s},{v}\n")

    # 4) 画 training_effect_random.png，风格和 GAT/SAGE 一致
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ln1 = ax1.plot(v2i_steps, v2v_vals, "-o",
                   label="V2V Success Rate (Random)", color="tab:blue")
    ax1.set_xlabel("step")
    ax1.set_ylabel("V2V Success Rate", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ln2 = ax2.plot(v2i_steps, v2i_vals, "-s",
                   label="V2I Rate (Random)", color="tab:orange")
    ax2.set_ylabel("V2I Rate", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title("Training Effect (Random)")
    plt.tight_layout()
    fig.savefig(export_dir / "training_effect_random.png", dpi=args.dpi)
    plt.close(fig)

    print("[Saved]", csv_v2i)
    print("[Saved]", csv_v2v)
    print("[Saved]", export_dir / "training_effect_random.png")


if __name__ == "__main__":
    main()