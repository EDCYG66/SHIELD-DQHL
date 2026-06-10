#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot V2V success curves for GATv2 / GATclassic / GraphSAGE / FC in a
"performance comparison" style figure similar to the paper example.

Default expects outputs from:
  runs_cmp_fair_seed123_gatv2/gat-train-*/gat_v2v_curve.csv
  runs_cmp_fair_seed123_gat1/gatclassic-train-*/gatclassic_v2v_curve.csv
  runs_cmp_fair_seed123_sage/sage-train-*/sage_v2v_curve.csv
  runs_cmp_fair_seed123_fc/fc-train-*/fc_v2v_curve.csv

Usage:
  python plot_compare_v2v_attention.py
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
    # names contain timestamps, so lexicographic order works
    return sorted(candidates, key=lambda p: p.name)[-1]


def load_curve(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        print(f"[WARN] Missing curve: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    if "step" not in df.columns or "v2v_success" not in df.columns:
        print(f"[WARN] Bad columns in: {csv_path}")
        return None
    return df


def fmt_delta(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f}%"


def main() -> None:
    models = [
        {
            "key": "gatv2",
            "name": "GATv2",
            "desc": "Attention",
            "root": Path("runs_cmp_fair_seed123_gatv2"),
            "prefix": "gat-train-",
            "curve": "gat_v2v_curve.csv",
            "color": "tab:red",
            "style": "-",
            "marker": "o",
        },
        {
            "key": "gatclassic",
            "name": "GATclassic",
            "desc": "Attention (v1)",
            "root": Path("runs_cmp_fair_seed123_gat1"),
            "prefix": "gatclassic-train-",
            "curve": "gatclassic_v2v_curve.csv",
            "color": "tab:orange",
            "style": "-",
            "marker": "s",
        },
        {
            "key": "sage",
            "name": "GraphSAGE",
            "desc": "Graph",
            "root": Path("runs_cmp_fair_seed123_sage"),
            "prefix": "sage-train-",
            "curve": "sage_v2v_curve.csv",
            "color": "tab:green",
            "style": "-.",
            "marker": "^",
        },
        {
            "key": "fc",
            "name": "FC",
            "desc": "No Graph",
            "root": Path("runs_cmp_fair_seed123_fc"),
            "prefix": "fc-train-",
            "curve": "fc_v2v_curve.csv",
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
        df = load_curve(run_dir / m["curve"])
        if df is None or df.empty:
            continue
        curves.append(
            {
                "key": m["key"],
                "name": m["name"],
                "desc": m["desc"],
                "steps": df["step"].to_numpy(),
                "vals": df["v2v_success"].to_numpy(),
                "color": m["color"],
                "style": m["style"],
                "marker": m["marker"],
            }
        )

    if not curves:
        print("[ERROR] No curves found. Please check run directories.")
        return

    # Compute range for y-axis.
    min_val = min(c["vals"].min() for c in curves)
    max_val = max(c["vals"].max() for c in curves)
    y_min = max(0.0, min_val - 0.02)
    y_max = min(1.02, max_val + 0.005)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for c in curves:
        final = float(c["vals"][-1])
        label = f"{c['name']} ({final*100:.2f}%)"
        if c["desc"]:
            label = f"{c['name']} ({c['desc']}) {final*100:.2f}%"
        ax.plot(
            c["steps"],
            c["vals"],
            c["style"],
            color=c["color"],
            marker=c["marker"],
            markersize=3.5,
            linewidth=1.6,
            label=label,
        )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("V2V Link Success Rate")
    ax.grid(alpha=0.25)
    ax.set_ylim(y_min, y_max)
    ax.set_title("")
    fig.suptitle("")

    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()

    out_dir = Path("runs_cmp_fair_seed123_compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "compare_v2v_attention.png"
    fig.savefig(out_path, dpi=EXPORT_DPI)
    plt.close(fig)
    print("[Saved]", out_path)


if __name__ == "__main__":
    main()
