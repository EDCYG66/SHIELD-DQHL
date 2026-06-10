#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate paper figures as true vector PDFs directly from source CSV/JSON data.

Figures covered:
  - fig5_test_curve.pdf
  - fig7a_ddqn_power_distribution.pdf
  - fig7b_gat_power_distribution.pdf
  - fig9_robustness.pdf
  - fig10_latency.pdf

Notes:
  - fig7 is already exported as a vector PDF by plot_progressive_ablation_summary.py
  - fig9 is intentionally not regenerated here because the currently available
    comparison summaries do not match the values described in the manuscript.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
EXPORT_DPI = 1000


def find_latest_run_dir(root: Path, prefix: str) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def save_pdf(fig: plt.Figure, out_name: str, *, also_png: bool = False) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / out_name
    fig.savefig(out_path, dpi=EXPORT_DPI, bbox_inches="tight")
    if also_png:
        png_path = out_path.with_suffix(".png")
        fig.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight")
        print(f"[Saved] {png_path}")
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_fig6_test_curve() -> None:
    run_root = ROOT / "runs_cmp_fair_seed123_gatv2"
    run_dir = find_latest_run_dir(run_root, "gat-train-")
    if run_dir is None:
        raise FileNotFoundError("Missing GATv2 fair-comparison run directory for fig6.")

    df = pd.read_csv(run_dir / "timeseries_gat.csv")

    fig, ax1 = plt.subplots(figsize=(7.0, 4.6))
    color_v2v = "tab:blue"
    color_v2i = "tab:orange"

    ax1.plot(
        df["t"],
        df["v2v_success"],
        color=color_v2v,
        linewidth=1.6,
        marker="o",
        markersize=4.0,
        label="V2V Success Rate",
    )
    ax1.set_xlabel("t")
    ax1.set_ylabel("V2V Success Rate", color=color_v2v)
    ax1.tick_params(axis="y", labelcolor=color_v2v)
    ax1.grid(alpha=0.25)
    ax1.set_ylim(
        max(0.0, float(df["v2v_success"].min()) - 0.001),
        min(1.02, float(df["v2v_success"].max()) + 0.001),
    )

    ax2 = ax1.twinx()
    ax2.plot(
        df["t"],
        df["v2i_rate"],
        color=color_v2i,
        linewidth=1.6,
        marker="s",
        markersize=4.0,
        label="V2I Sum Rate",
    )
    ax2.set_ylabel("V2I Sum Rate", color=color_v2i)
    ax2.tick_params(axis="y", labelcolor=color_v2i)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True)

    fig.tight_layout()
    save_pdf(fig, "fig5_test_curve.pdf", also_png=True)


def _plot_power_distribution(csv_path: Path, out_name: str) -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6.2, 3.9))

    ax.plot(df["time_left"], df["prob_p0"], color="#A93226", marker="o", markersize=3.2, linewidth=1.7, label="23 dBm")
    ax.plot(df["time_left"], df["prob_p1"], color="#2874A6", marker="s", markersize=3.0, linewidth=1.7, label="10 dBm")
    ax.plot(df["time_left"], df["prob_p2"], color="#1E8449", marker="^", markersize=3.2, linewidth=1.7, label="5 dBm")

    ax.set_xlabel("Remaining TTL (s)")
    ax.set_ylabel("Selection probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True, fontsize=9)

    fig.tight_layout()
    save_pdf(fig, out_name)


def plot_fig11_and_fig12_power_distribution() -> None:
    fc_root = ROOT / "runs_cmp_fair_seed123_fc"
    gat_root = ROOT / "runs_cmp_fair_seed123_gatv2"
    fc_run = find_latest_run_dir(fc_root, "fc-train-")
    gat_run = find_latest_run_dir(gat_root, "gat-train-")
    if fc_run is None or gat_run is None:
        raise FileNotFoundError("Missing FC/GATv2 fair-comparison runs for fig11/fig12.")

    _plot_power_distribution(fc_run / "power_select_fc.csv", "fig7a_ddqn_power_distribution.pdf")
    _plot_power_distribution(gat_run / "power_select_gat.csv", "fig7b_gat_power_distribution.pdf")


def plot_fig13_robustness() -> None:
    model_specs = [
        ("GATv2-DDQN", ROOT / "runs/sweep_veh_20to100_gatv2/sweep_vehicles.csv", "#E15759"),
        ("GATclassic-DDQN", ROOT / "runs/sweep_veh_20to100_gatclassic/sweep_vehicles.csv", "#F39C12"),
        ("GraphSAGE-DDQN", ROOT / "runs/sweep_veh_20to100_sage/sweep_vehicles.csv", "#59A14F"),
        ("FC-DQN", ROOT / "runs/sweep_veh_20to100_fc/sweep_vehicles.csv", "#CFCFCF"),
    ]
    metrics = [
        ("v2v_success", "Average V2V success rate", "V2V"),
        ("v2i_mean", "Average V2I rate", "V2I"),
    ]

    data = {}
    for label, path, _ in model_specs:
        df = pd.read_csv(path).copy()
        df["n_veh"] = df["n_veh"].astype(int)
        data[label] = df.set_index("n_veh").sort_index()

    vehicle_counts = [20, 40, 60, 80, 100]
    x = np.arange(len(vehicle_counts), dtype=float)
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))

    for ax, (metric_key, ylabel, title) in zip(axes, metrics):
        all_values = []
        for idx, n_veh in enumerate(vehicle_counts):
            for offset, (label, _, color) in zip(offsets, model_specs):
                value = float(data[label].loc[n_veh, metric_key])
                ax.bar(
                    x[idx] + offset,
                    value,
                    width=width,
                    color=color,
                    edgecolor="none",
                )
                all_values.append(value)

        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xlabel("Number of vehicles")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in vehicle_counts])
        ax.grid(axis="y", alpha=0.22, linestyle="--")
        ax.set_axisbelow(True)

        val_min = min(all_values)
        val_max = max(all_values)
        if metric_key == "v2v_success":
            ax.set_ylim(max(0.0, val_min - 0.03), min(1.02, val_max + 0.008))
        else:
            padding = max(8.0, (val_max - val_min) * 0.10)
            ax.set_ylim(max(0.0, val_min - padding), val_max + padding)

    legend_handles = [Patch(facecolor=color, edgecolor="none", label=label) for label, _, color in model_specs]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        fontsize=8.0,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    save_pdf(fig, "fig9_robustness.pdf", also_png=True)


def plot_fig14_latency() -> None:
    model_specs = [
        ("FC-\nDQN", "fc", ROOT / "runs_cmp_fair_seed123_fc/comparison_summary.json"),
        ("GraphSAGE-\nDDQN", "sage", ROOT / "runs_cmp_fair_seed123_sage/comparison_summary.json"),
        ("GATclassic-\nDDQN", "gatclassic", ROOT / "runs_cmp_fair_seed123_gat1/comparison_summary.json"),
        ("GATv2-\nDDQN", "gat", ROOT / "runs_cmp_fair_seed123_gatv2/comparison_summary.json"),
    ]

    labels = []
    total_ms = []
    gnn_ms = []
    for display_name, key, path in model_specs:
        data = json.load(path.open("r", encoding="utf-8"))
        info = data[key]
        labels.append(display_name)
        total_ms.append(float(info["decision_time_full_s"]) * 1000.0)
        gnn_ms.append(float(info["decision_time_gnn_only_s"]) * 1000.0)

    fig, ax = plt.subplots(figsize=(6.9, 4.2))
    x = range(len(labels))

    ax.plot(x, total_ms, color="#2C7FB8", linewidth=1.8, marker="o", markersize=5.5, label="Total latency")
    ax.plot(x, gnn_ms, color="#D95F0E", linewidth=1.6, marker="^", markersize=5.0, linestyle="--", label="GNN-only latency")

    for i, value in enumerate(total_ms):
        ax.text(i, value + max(total_ms) * 0.03, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5, color="#2C7FB8")
    for i, value in enumerate(gnn_ms):
        ax.text(i, value + max(total_ms) * 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8.0, color="#D95F0E")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Average single-step latency (ms)")
    ax.grid(alpha=0.25)
    ax.set_ylim(0.0, max(total_ms) * 1.25)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.tight_layout()
    save_pdf(fig, "fig10_latency.pdf")


def main() -> None:
    plot_fig6_test_curve()
    plot_fig11_and_fig12_power_distribution()
    plot_fig13_robustness()
    plot_fig14_latency()
    print("[Done] Vector paper figures regenerated.")


if __name__ == "__main__":
    main()
