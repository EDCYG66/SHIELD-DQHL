#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot V2V success rate vs vehicle density (line chart).

Input CSV (from run_compare_gnn_plus.py sweep):
  sweep_vehicles.csv with columns:
    model,n_veh,v2i_mean,v2v_success
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "gat": "Proposed GATv2-DDQN",
    "sage": "GraphSAGE-DDQN",
    "fc": "FC-DQN (No Graph)",
    "gatclassic": "GATclassic-DDQN",
}

STYLES = {
    "gat": dict(color="tab:red", linestyle="-", marker="o"),
    "sage": dict(color="tab:green", linestyle="--", marker="^"),
    "fc": dict(color="tab:blue", linestyle="-.", marker="s"),
    "gatclassic": dict(color="tab:orange", linestyle="--", marker="D"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot V2V success vs vehicles (line).")
    ap.add_argument("--csv", required=True, help="Path to sweep_vehicles.csv")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty CSV: {csv_path}")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for model, sub in df.groupby("model"):
        sub = sub.sort_values("n_veh")
        label = LABELS.get(model, model)
        style = STYLES.get(model, {})
        ax.plot(
            sub["n_veh"].to_numpy(),
            sub["v2v_success"].to_numpy(),
            label=label,
            linewidth=1.8,
            markersize=5,
            **style,
        )

    ax.set_title("Robustness Analysis: V2V Performance vs. Vehicle Density")
    ax.set_xlabel("Number of Vehicles (Density)")
    ax.set_ylabel("V2V Link Success Rate")
    ax.grid(alpha=0.25)
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "v2v_success_vs_vehicles_line.png"
    fig.savefig(out_path, dpi=224)
    plt.close(fig)
    print("[Saved]", out_path)


if __name__ == "__main__":
    main()
