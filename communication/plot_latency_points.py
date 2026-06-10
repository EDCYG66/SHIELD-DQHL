#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot decision latency comparison in two styles:
  - total+gnn (line)
  - total-only (bar)

Data source (comparison summaries):
  runs_cmp_fair_seed123_fc/comparison_summary.json
  runs_cmp_fair_seed123_sage/comparison_summary.json
  runs_cmp_fair_seed123_gatv2/comparison_summary.json

Outputs:
  fig14_latency.png
  fig14_latency_bar.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


def load_latency(path: Path, key: str) -> tuple[float, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    info = data[key]
    total_ms = float(info["decision_time_full_s"]) * 1000.0
    gnn_ms = float(info["decision_time_gnn_only_s"]) * 1000.0
    return total_ms, gnn_ms


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    models = [
        {
            "name": "FC-\nDQN",
            "key": "fc",
            "path": base_dir / "experiments" / "single_model" / "fc" / "runs_cmp_fair_seed123_fc" / "comparison_summary.json",
        },
        {
            "name": "GraphSAGE-\nDDQN",
            "key": "sage",
            "path": base_dir / "experiments" / "single_model" / "sage" / "runs_cmp_fair_seed123_sage" / "comparison_summary.json",
        },
        {
            "name": "GATclassic-\nDDQN",
            "key": "gatclassic",
            "path": base_dir / "experiments" / "single_model" / "gatclassic" / "runs_cmp_fair_seed123_gat1" / "comparison_summary.json",
        },
        {
            "name": "GATv2-\nDDQN",
            "key": "gat",
            "path": base_dir / "experiments" / "single_model" / "gatv2" / "runs_cmp_fair_seed123_gatv2" / "comparison_summary.json",
        },
    ]

    labels = []
    totals = []
    gnns = []
    for m in models:
        if not m["path"].exists():
            print(f"[WARN] Missing: {m['path']}")
            continue
        total_ms, gnn_ms = load_latency(m["path"], m["key"])
        labels.append(m["name"])
        totals.append(total_ms)
        gnns.append(gnn_ms)

    if not labels:
        print("[ERROR] No latency data found.")
        return

    x = np.arange(len(labels))
    totals = np.array(totals)
    gnns = np.array(gnns)
    y_ref = float(np.max(totals))
    total_label_offset = max(0.006, 0.03 * y_ref)
    gnn_label_offset = max(0.003, 0.015 * y_ref)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    ax.plot(x, totals, "-o", color="tab:blue", linewidth=1.8, markersize=6,
            label="Total latency")
    ax.plot(x, gnns, "--^", color="tab:orange", linewidth=1.4, markersize=5,
            label="GNN-only latency")

    # Numeric labels
    for i, v in enumerate(totals):
        ax.text(x[i], v + total_label_offset, f"{v:.2f}", ha="center", va="bottom",
                fontsize=9, color="tab:blue")
    for i, v in enumerate(gnns):
        ax.text(x[i], v + gnn_label_offset, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, color="tab:orange")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average single-step latency (ms)")
    ax.set_xlabel("Algorithm")
    ax.grid(alpha=0.25)
    y_top = max(
        float(np.max(totals) + total_label_offset + 0.02),
        float(np.max(gnns) + gnn_label_offset + 0.02),
    )
    ax.set_ylim(0.0, y_top)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)

    plt.tight_layout()
    out_path = base_dir / "fig14_latency.png"
    fig.savefig(out_path, dpi=224)
    plt.close(fig)
    print("[Saved]", out_path)

    # Total-only bar chart (clean paper view)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(x, totals, color="#A7C7F2", edgecolor="#6B8FC7", linewidth=1.0)
    bar_label_offset = max(0.004, 0.02 * y_ref)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + bar_label_offset,
                f"{totals[i]:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average single-step latency (ms)")
    ax.set_xlabel("Algorithm")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0.0, float(np.max(totals) + bar_label_offset + 0.03))
    plt.tight_layout()
    out_path = base_dir / "fig14_latency_bar.png"
    fig.savefig(out_path, dpi=224)
    plt.close(fig)
    print("[Saved]", out_path)

    # Defense-friendly bar chart: keep true latency values, but emphasize
    # that all methods are far below the 100 ms real-time budget.
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    threshold_ms = 100.0
    colors = ["#CFD8E3", "#F2C38B", "#5B8FD9", "#B9C2CC"]
    edge_colors = ["#94A3B8", "#CC8A3D", "#2F5EA8", "#8D99A8"]
    bars = ax.bar(x, totals, color=colors, edgecolor=edge_colors, linewidth=1.2, width=0.62)

    # Highlight GATv2 for presentation use without changing the true value.
    if len(bars) >= 4:
        bars[3].set_color("#2F6DB2")
        bars[3].set_edgecolor("#1F4E86")
        bars[3].set_linewidth(1.4)

    y_ref = float(np.max(totals))
    bar_label_offset = max(0.004, 0.03 * y_ref)
    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + bar_label_offset,
            f"{totals[i]:.3f} ms",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold" if i == 3 else "normal",
            color="#1F1F1F",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average single-step latency (ms)")
    ax.set_xlabel("Algorithm")
    ax.set_title("Decision Latency Comparison", fontsize=13, pad=10)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0.0, float(np.max(totals) * 1.55 + 0.01))

    max_ratio = float(np.max(totals) / threshold_ms * 100.0)
    gatv2_ms = float(totals[3]) if len(totals) >= 4 else float(totals[-1])
    gatv2_ratio = gatv2_ms / threshold_ms * 100.0

    note = (
        "Real-time budget: 100 ms\n"
        f"Max observed latency: {np.max(totals):.3f} ms ({max_ratio:.3f}% of budget)\n"
        f"GATv2-DDQN: {gatv2_ms:.3f} ms ({gatv2_ratio:.3f}% of budget)"
    )
    ax.text(
        0.98,
        0.96,
        note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF7DB", edgecolor="#D6B656", linewidth=1.0),
    )

    ax.text(
        x[3],
        totals[3] / 2.0,
        "Best trade-off\namong graph models",
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        weight="bold",
    )

    out_path = base_dir / "fig14_latency_bar_defense.png"
    fig.savefig(out_path, dpi=224, bbox_inches="tight")
    plt.close(fig)
    print("[Saved]", out_path)


if __name__ == "__main__":
    main()
