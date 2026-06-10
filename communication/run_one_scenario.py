#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run ONE scenario (one model + one vehicle count) and write a single-line CSV.

用法示例：
  python run_one_scenario.py --model gat  --n-veh 20 --out-csv runs/veh_sweep/one_gat_20.csv
  python run_one_scenario.py --model fc   --n-veh 40 --out-csv runs/veh_sweep/one_fc_40.csv
  python run_one_scenario.py --model sage --n-veh 60 --out-csv runs/veh_sweep/one_sage_60.csv
"""

import argparse
from pathlib import Path
import numpy as np

from highway_environment import HighwayTopoEnv
from run_compare_gnn_plus import ensure_env_difficulty
from run_compare_gnn_plus import InstrumentedAgent
from gnn_factory import build_gnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, choices=["gat", "sage", "fc"], required=True)
    ap.add_argument("--n-veh", type=int, required=True, help="total number of vehicles (up+down)")
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

    ap.add_argument("--beta-urgency-pos", type=float, default=0.018)
    ap.add_argument("--beta-urgency-neg", type=float, default=0.025)
    ap.add_argument("--urgency-threshold", type=float, default=0.25)
    ap.add_argument("--rb-anti-conc-alpha", type=float, default=0.012)
    ap.add_argument("--rb-hot-threshold", type=float, default=0.22)
    ap.add_argument("--rb-softmask-alpha", type=float, default=0.15)
    ap.add_argument("--rb-softmask-window", type=int, default=50)

    ap.add_argument("--reprune-every", type=int, default=300)
    ap.add_argument("--hysteresis-keep", type=float, default=0.5)
    ap.add_argument("--reprune-start-step", type=int, default=600)
    ap.add_argument("--reg-attn-w", type=float, default=1e-3)
    ap.add_argument("--disable-reprune", action="store_true",
                    help="Disable adaptive reprune explicitly for GAT/GATv2.")

    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--epsilon-decay-steps", type=int, default=6000)
    ap.add_argument("--epsilon-min", type=float, default=0.05)

    ap.add_argument("--out-dir", type=str, default="runs/veh_sweep_single")
    ap.add_argument("--out-csv", type=str, required=True,
                    help="path to a one-line csv, e.g. runs/veh_sweep/one_gat_20.csv")

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

    env.beta_urgency_pos = float(args.beta_urgency_pos)
    env.beta_urgency_neg = float(args.beta_urgency_neg)
    env.urgency_threshold = float(args.urgency_threshold)
    env.rb_anti_conc_alpha = float(args.rb_anti_conc_alpha)
    env.rb_hot_threshold = float(args.rb_hot_threshold)
    env.rb_softmask_alpha = float(args.rb_softmask_alpha)
    env.rb_softmask_window = int(args.rb_softmask_window)

    tag = args.model.lower()
    G = build_gnn(
        env,
        gnn_type=tag,
        distance_threshold=150.0,
        lr=5e-4,
        gat_train_interval=20,
        grad_clip=5.0,
        reprune_every=args.reprune_every,
        hysteresis_keep=args.hysteresis_keep,
        reprune_start_step=args.reprune_start_step,
        reg_attn_w=args.reg_attn_w,
        enable_reprune=(not args.disable_reprune),
    )

    # 这里不用 run_dir，只是为了拿到 summary
    ag = InstrumentedAgent(
        [], env, gnn_type=tag,
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
    ag.G = G

    ag.train(max_steps=args.train_steps, test_every_steps=args.test_every, test_sample=args.test_sample)

    v2i_final = ag.v2i_curve_vals[-1] if ag.v2i_curve_vals else 0.0
    v2v_final = ag.v2v_curve_vals[-1] if ag.v2v_curve_vals else 0.0

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("model,n_veh,v2i_mean,v2v_success\n")
        f.write(f"{tag},{args.n_veh},{v2i_final},{v2v_final}\n")
    print("[DONE]", out_csv, "=>", tag, args.n_veh, v2i_final, v2v_final)


if __name__ == "__main__":
    main()