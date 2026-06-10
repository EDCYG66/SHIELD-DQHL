#!/usr/bin/env python3
"""
Generate two grouped-bar figures from four sweep CSV files:
1) Mean V2I rate vs number of vehicles
2) Mean V2V success rate vs number of vehicles
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_SPECS = [
    ("gat", "GATv2-DDQN", "#1f77b4"),
    ("fc", "FC-DQN", "#ff7f0e"),
    ("sage", "GraphSAGE-DDQN", "#2ca02c"),
    ("gatclassic", "GATclassic-DDQN", "#d62728"),
]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    required = {"model", "n_veh", "v2i_mean", "v2v_success"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Bad columns in {path}: missing {sorted(missing)}")
    return df


def make_grouped_bar(
    x_values: List[int],
    y_by_model: Dict[str, np.ndarray],
    ylabel: str,
    title: str,
    output_path: Path,
    y_lim: tuple[float, float] | None = None,
) -> None:
    bg = "white"
    fig, ax = plt.subplots(figsize=(8.2, 5.6), facecolor=bg)
    ax.set_facecolor(bg)

    x = np.arange(len(x_values), dtype=float)
    n_models = len(MODEL_SPECS)
    width = 0.18
    offsets = np.linspace(
        -((n_models - 1) * width) / 2.0,
        ((n_models - 1) * width) / 2.0,
        n_models,
    )

    for i, (key, label, color) in enumerate(MODEL_SPECS):
        ax.bar(
            x + offsets[i],
            y_by_model[key],
            width=width,
            color=color,
            label=label,
            edgecolor="none",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x_values])
    ax.set_xlabel("Number of Participating Vehicles")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    if y_lim is not None:
        ax.set_ylim(*y_lim)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot grouped bars from sweep CSV files.")
    parser.add_argument("--gat", default="runs/sweep_veh_gatv2/sweep_vehicles.csv")
    parser.add_argument("--gatclassic", default="runs/sweep_veh_gatclassic/sweep_vehicles.csv")
    parser.add_argument("--sage", default="runs/sweep_veh_sage/sweep_vehicles.csv")
    parser.add_argument("--fc", default="runs/sweep_veh_fc/sweep_vehicles.csv")
    parser.add_argument("--out-dir", default="runs/sweep_veh_compare_figs")
    args = parser.parse_args()

    paths = {
        "gat": Path(args.gat),
        "gatclassic": Path(args.gatclassic),
        "sage": Path(args.sage),
        "fc": Path(args.fc),
    }

    data = {k: load_csv(p) for k, p in paths.items()}

    veh_sets = [set(df["n_veh"].astype(int).tolist()) for df in data.values()]
    common_veh = sorted(set.intersection(*veh_sets))
    if not common_veh:
        raise ValueError("No common n_veh values across all models.")

    y_v2i: Dict[str, np.ndarray] = {}
    y_v2v: Dict[str, np.ndarray] = {}
    for key in data:
        df = data[key].copy()
        df["n_veh"] = df["n_veh"].astype(int)
        df = df.set_index("n_veh").sort_index()
        y_v2i[key] = np.array([float(df.loc[v, "v2i_mean"]) for v in common_veh], dtype=float)
        y_v2v[key] = np.array([float(df.loc[v, "v2v_success"]) for v in common_veh], dtype=float)

    out_dir = Path(args.out_dir)
    fig_v2i = out_dir / "fig6_v2i_vs_vehicles.png"
    fig_v2v = out_dir / "fig7_v2v_vs_vehicles.png"

    make_grouped_bar(
        x_values=common_veh,
        y_by_model=y_v2i,
        ylabel="V2I Communication Rate",
        title="Average V2I Rate vs Number of Participating Vehicles",
        output_path=fig_v2i,
    )
    make_grouped_bar(
        x_values=common_veh,
        y_by_model=y_v2v,
        ylabel="V2V Communication Success Rate",
        title="Average V2V Success Rate vs Number of Participating Vehicles",
        output_path=fig_v2v,
        y_lim=(0.8, 1.0),
    )

    print(f"[Saved] {fig_v2i}")
    print(f"[Saved] {fig_v2v}")


if __name__ == "__main__":
    main()
