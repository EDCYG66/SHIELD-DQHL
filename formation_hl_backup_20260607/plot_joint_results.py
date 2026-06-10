"""Paper-style plotting helpers for joint formation/communication experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

EXPORT_DPI = 1000
BG = "white"
GRID_ALPHA = 0.25
SCENARIO_BAR = "#7FA6D8"
SCENARIO_EDGE = "#4E6EA9"
ABLATION_COLORS = {
    "full_joint": "#7FA6D8",
    "no_communication": "#F3B577",
    "no_safety_shield": "#CFCFCF",
    "no_reconfiguration": "#8FC9A8",
}
ABLATION_EDGE = {
    "full_joint": "#4E6EA9",
    "no_communication": "#C9823A",
    "no_safety_shield": "#8E8E8E",
    "no_reconfiguration": "#4C8E62",
}


def _save_png_pdf(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def _scenario_order(df: pd.DataFrame) -> List[str]:
    preferred = ["mild_mid", "moderate_mid", "severe_mid", "upstream_bottleneck", "downstream_bottleneck"]
    existing = [item for item in preferred if item in set(df["scenario"].astype(str))]
    remaining = sorted(set(df["scenario"].astype(str)) - set(existing))
    return existing + remaining


def plot_scenario_sweep_summary(summary_csv: str | Path, out_dir: str | Path) -> None:
    summary_csv = Path(summary_csv)
    out_dir = Path(out_dir)
    df = pd.read_csv(summary_csv)
    if df.empty:
        return
    if "scenario" not in df.columns:
        raise ValueError("scenario_sweep_summary.csv missing 'scenario' column.")

    order = _scenario_order(df)
    df = df.set_index("scenario").loc[order].reset_index()
    x = np.arange(len(df), dtype=float)

    metrics = [
        ("avg_reward", "Average reward"),
        ("avg_comm_v2v_success", "Average V2V success"),
        ("worst_min_gap_up", "Worst min gap up (m)"),
        ("safe_recovery_step", "Safe recovery step"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), facecolor=BG)
    axes = axes.reshape(-1)
    for ax, (metric, ylabel) in zip(axes, metrics):
        values = df[metric].astype(float).to_numpy()
        bars = ax.bar(x, values, color=SCENARIO_BAR, edgecolor=SCENARIO_EDGE, linewidth=1.0, width=0.62)
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace("_", "\n") for name in df["scenario"]], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=GRID_ALPHA, linestyle="--")
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            y = value
            offset = (np.max(values) - np.min(values) + 1e-6) * 0.03
            ax.text(bar.get_x() + bar.get_width() / 2.0, y + offset, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Joint Scenario Sweep Summary", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    _save_png_pdf(fig, out_dir / "scenario_sweep_summary_plot")


def plot_joint_ablation_summary(summary_csv: str | Path, out_dir: str | Path) -> None:
    summary_csv = Path(summary_csv)
    out_dir = Path(out_dir)
    df = pd.read_csv(summary_csv)
    if df.empty:
        return
    if "preset" not in df.columns:
        raise ValueError("joint_ablation_summary.csv missing 'preset' column.")

    preferred = ["full_joint", "no_communication", "no_safety_shield", "no_reconfiguration"]
    order = [item for item in preferred if item in set(df["preset"].astype(str))]
    remaining = sorted(set(df["preset"].astype(str)) - set(order))
    order += remaining
    df = df.set_index("preset").loc[order].reset_index()
    x = np.arange(len(df), dtype=float)

    metrics = [
        ("avg_reward", "Average reward"),
        ("avg_comm_v2v_success", "Average V2V success"),
        ("worst_min_gap_up", "Worst min gap up (m)"),
        ("total_shield_interventions", "Shield interventions"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), facecolor=BG)
    axes = axes.reshape(-1)
    for ax, (metric, ylabel) in zip(axes, metrics):
        values = df[metric].astype(float).to_numpy()
        colors = [ABLATION_COLORS.get(name, "#BDBDBD") for name in df["preset"]]
        edges = [ABLATION_EDGE.get(name, "#666666") for name in df["preset"]]
        bars = ax.bar(x, values, color=colors, edgecolor=edges, linewidth=1.0, width=0.62)
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace("_", "\n") for name in df["preset"]], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=GRID_ALPHA, linestyle="--")
        ax.set_axisbelow(True)
        if np.min(values) < 0.0:
            ax.axhline(0.0, color="#666666", linewidth=1.0)
        for bar, value in zip(bars, values):
            span = np.max(values) - np.min(values) + 1e-6
            offset = span * 0.03
            if value >= 0:
                ax.text(bar.get_x() + bar.get_width() / 2.0, value + offset, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2.0, value - offset, f"{value:.3f}", ha="center", va="top", fontsize=8)

    fig.suptitle("Joint Ablation Summary", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    _save_png_pdf(fig, out_dir / "joint_ablation_summary_plot")
