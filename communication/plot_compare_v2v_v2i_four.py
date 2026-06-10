#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create comparison plots for four models:
1) V2V success rate only
2) Combined V2V + V2I in a single figure (two stacked panels)

Default expects outputs from:
  runs_cmp_fair_seed123_gatv2/gat-train-*/gat_v2v_curve.csv
  runs_cmp_fair_seed123_gatv2/gat-train-*/gat_v2i_curve.csv
  runs_cmp_fair_seed123_gat1/gatclassic-train-*/gatclassic_v2v_curve.csv
  runs_cmp_fair_seed123_gat1/gatclassic-train-*/gatclassic_v2i_curve.csv
  runs_cmp_fair_seed123_sage/sage-train-*/sage_v2v_curve.csv
  runs_cmp_fair_seed123_sage/sage-train-*/sage_v2i_curve.csv
  runs_cmp_fair_seed123_fc/fc-train-*/fc_v2v_curve.csv
  runs_cmp_fair_seed123_fc/fc-train-*/fc_v2i_curve.csv

Outputs:
  runs_cmp_fair_seed123_compare/compare_v2v_four.png
  runs_cmp_fair_seed123_compare/compare_v2v_v2i_four.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EXPORT_DPI = 1000


def find_latest_run_dir(root: Path, prefix: str) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def load_curve(csv_path: Path, value_col: str) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        print(f"[WARN] Missing curve: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    if "step" not in df.columns or value_col not in df.columns:
        print(f"[WARN] Bad columns in: {csv_path}")
        return None
    return df


def main() -> None:
    models = [
        {
            "key": "gatv2",
            "name": "GATv2",
            "root": Path("runs_cmp_fair_seed123_gatv2"),
            "prefix": "gat-train-",
            "v2v": "gat_v2v_curve.csv",
            "v2i": "gat_v2i_curve.csv",
            "color": "tab:red",
            "style": "-",
            "marker": "o",
        },
        {
            "key": "gatclassic",
            "name": "GATclassic",
            "root": Path("runs_cmp_fair_seed123_gat1"),
            "prefix": "gatclassic-train-",
            "v2v": "gatclassic_v2v_curve.csv",
            "v2i": "gatclassic_v2i_curve.csv",
            "color": "tab:orange",
            "style": "-",
            "marker": "s",
        },
        {
            "key": "sage",
            "name": "GraphSAGE",
            "root": Path("runs_cmp_fair_seed123_sage"),
            "prefix": "sage-train-",
            "v2v": "sage_v2v_curve.csv",
            "v2i": "sage_v2i_curve.csv",
            "color": "tab:green",
            "style": "-.",
            "marker": "^",
        },
        {
            "key": "fc",
            "name": "FC",
            "root": Path("runs_cmp_fair_seed123_fc"),
            "prefix": "fc-train-",
            "v2v": "fc_v2v_curve.csv",
            "v2i": "fc_v2i_curve.csv",
            "color": "tab:blue",
            "style": "--",
            "marker": "D",
        },
    ]

    curves: List[Dict] = []
    for m in models:
        run_dir = find_latest_run_dir(m["root"], m["prefix"])
        if not run_dir:
            print(f"[WARN] Missing run dir for {m['name']}")
            continue
        df_v2v = load_curve(run_dir / m["v2v"], "v2v_success")
        df_v2i = load_curve(run_dir / m["v2i"], "v2i_rate")
        if df_v2v is None or df_v2i is None:
            continue
        curves.append(
            {
                "key": m["key"],
                "name": m["name"],
                "steps": df_v2v["step"].to_numpy(),
                "v2v": df_v2v["v2v_success"].to_numpy(),
                "v2i_steps": df_v2i["step"].to_numpy(),
                "v2i": df_v2i["v2i_rate"].to_numpy(),
                "color": m["color"],
                "style": m["style"],
                "marker": m["marker"],
            }
        )

    if not curves:
        print("[ERROR] No curves found. Please check run directories.")
        return

    out_dir = Path("runs_cmp_fair_seed123_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) V2V success rate comparison
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for c in curves:
        ax.plot(
            c["steps"],
            c["v2v"],
            c["style"],
            color=c["color"],
            marker=c["marker"],
            markersize=3.2,
            linewidth=1.6,
            label=c["name"],
        )
    ax.set_title("V2V Success Rate Comparison (Four Models)")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("V2V Success Rate")
    ax.grid(alpha=0.25)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()
    v2v_path = out_dir / "compare_v2v_four.png"
    fig.savefig(v2v_path, dpi=EXPORT_DPI)
    plt.close(fig)
    print("[Saved]", v2v_path)

    # 2) Combined figure: V2V + V2I (stacked)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.6, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]}
    )

    for c in curves:
        ax1.plot(
            c["steps"],
            c["v2v"],
            c["style"],
            color=c["color"],
            marker=c["marker"],
            markersize=3.0,
            linewidth=1.5,
            label=c["name"],
        )

    for c in curves:
        ax2.plot(
            c["v2i_steps"],
            c["v2i"],
            c["style"],
            color=c["color"],
            marker=c["marker"],
            markersize=3.0,
            linewidth=1.5,
            label=c["name"],
        )

    ax1.set_title("V2V & V2I Performance (Four Models)")
    ax1.set_ylabel("V2V Success Rate")
    ax1.grid(alpha=0.25)
    ax1.set_ylim(0.0, 1.02)

    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("V2I Rate")
    ax2.grid(alpha=0.25)

    ax1.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()
    combo_path = out_dir / "compare_v2v_v2i_four.png"
    fig.savefig(combo_path, dpi=EXPORT_DPI)
    plt.close(fig)
    print("[Saved]", combo_path)


if __name__ == "__main__":
    main()
