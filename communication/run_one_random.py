#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run ONE random baseline scenario (n_veh) and output a single-line CSV.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")

from highway_environment import HighwayTopoEnv
from run_compare_gnn_plus import ensure_env_difficulty


def eval_random(env, test_sample=200):
    n_veh = env.n_Veh
    n_rb = env.n_RB if hasattr(env, "n_RB") else 20
    actions = np.zeros((n_veh, 3, 2), dtype=np.int32)

    v2i_list, v2v_list = [], []
    for _ in range(test_sample):
        actions[:, :, 0] = np.random.randint(0, n_rb, size=(n_veh, 3))
        actions[:, :, 1] = np.random.randint(0, 3, size=(n_veh, 3))
        reward_vec, fail = env.act_asyn(actions)
        v2i_list.append(float(np.sum(reward_vec)))
        v2v_list.append(float(1.0 - fail))
    return float(np.mean(v2i_list)), float(np.mean(v2v_list))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-veh", type=int, required=True)
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
    ap.add_argument("--test-sample", type=int, default=200)
    ap.add_argument("--out-csv", type=str, required=True)
    args = ap.parse_args()

    np.random.seed(args.seed)
    n_up = int(np.ceil(args.n_veh / 2))
    n_down = args.n_veh - n_up

    env = HighwayTopoEnv(
        n_up=n_up, n_down=n_down, lanes_per_dir=args.lanes, spacing=args.spacing,
        base_y=args.base_y, height=args.height, topology_type=args.topo,
        v2i_mode=args.v2i_mode, bs_layout=args.bs_layout, bs_spacing=args.bs_spacing,
        bs_min_stay_steps=args.bs_min_stay, bs_handover_hyst_m=args.bs_hyst,
        seed=args.seed
    )
    env.new_random_game()
    ensure_env_difficulty(env, args.demand_amount, args.v2v_limit)

    v2i, v2v = eval_random(env, test_sample=args.test_sample)

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("model,n_veh,v2i_mean,v2v_success\n")
        f.write(f"random,{args.n_veh},{v2i},{v2v}\n")
    print("[DONE]", out_csv, "=> random", args.n_veh, v2i, v2v)


if __name__ == "__main__":
    main()