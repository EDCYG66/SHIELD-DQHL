#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Plot decision time vs vehicles for two GAT variants (complete vs pruned).")
    ap.add_argument("--csv", type=str, required=True,
                    help="Path to decision_time_sweep.csv")
    ap.add_argument("--out", type=str, default="",
                    help="Output directory (default: same as csv)")
    ap.add_argument("--dpi", type=int, default=224)
    ap.add_argument("--model-complete", type=str, default="gat_complete",
                    help="Name of complete-graph model in csv")
    ap.add_argument("--model-pruned", type=str, default="gat_pruned",
                    help="Name of pruned-graph model in csv")
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir = Path(args.out).expanduser().resolve() if args.out else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # 只取这两个模型
    df_c = df[df["model"] == args.model_complete].sort_values("n_veh")
    df_p = df[df["model"] == args.model_pruned].sort_values("n_veh")

    if df_c.empty or df_p.empty:
        print("[WARN] No rows for one of the models.")
        return

    # 要求 n_veh 集合一致（为了对齐）
    n_veh_c = df_c["n_veh"].to_numpy()
    n_veh_p = df_p["n_veh"].to_numpy()
    if not np.array_equal(n_veh_c, n_veh_p):
        print("[WARN] n_veh sets differ between complete and pruned. Will align by intersection.")
    veh_common = np.intersect1d(n_veh_c, n_veh_p)

    df_c = df_c[df_c["n_veh"].isin(veh_common)]
    df_p = df_p[df_p["n_veh"].isin(veh_common)]

    n_veh = df_c["n_veh"].to_numpy(dtype=float)
    t_c = df_c["decision_time_full_s"].to_numpy(dtype=float)
    t_p = df_p["decision_time_full_s"].to_numpy(dtype=float)

    x = np.arange(len(n_veh))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars1 = ax.bar(x - width / 2, t_c, width=width, label="Complete Graph", color="tab:blue")
    bars2 = ax.bar(x + width / 2, t_p, width=width, label="Incomplete Graph", color="tab:red")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in n_veh])
    ax.set_xlabel("Number of Participating Vehicles")
    ax.set_ylabel("Decision Time (s)")
    ax.set_title("Decision Time Comparison: Complete vs Incomplete Graph (GAT)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left")

    plt.tight_layout()
    out_path = out_dir / "decision_time_vs_vehicles_gat_pair.png"
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()