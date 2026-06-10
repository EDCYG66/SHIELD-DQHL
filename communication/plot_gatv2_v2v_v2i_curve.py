#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot GATv2 evaluation curves:
  - V2V success rate (left y-axis)
  - V2I rate (right y-axis)

Default (curve) expects:
  runs_cmp_fair_seed123_gatv2/gat-train-*/gat_v2v_curve.csv
  runs_cmp_fair_seed123_gatv2/gat-train-*/gat_v2i_curve.csv

Timeseries option expects:
  runs_cmp_fair_seed123_gatv2/gat-train-*/timeseries_gat.csv

Outputs:
  curve      -> gatv2_v2v_v2i_curve.png
  timeseries -> gatv2_v2v_v2i_timeseries.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def find_latest_run_dir(root: Path, prefix: str) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def load_curve_data(run_dir: Path) -> Optional[Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
    v2v_path = run_dir / "gat_v2v_curve.csv"
    v2i_path = run_dir / "gat_v2i_curve.csv"
    if not v2v_path.exists() or not v2i_path.exists():
        return None
    df_v2v = pd.read_csv(v2v_path)
    df_v2i = pd.read_csv(v2i_path)
    if "step" not in df_v2v.columns or "v2v_success" not in df_v2v.columns:
        return None
    if "step" not in df_v2i.columns or "v2i_rate" not in df_v2i.columns:
        return None
    return (
        df_v2v["step"],
        df_v2v["v2v_success"],
        df_v2i["step"],
        df_v2i["v2i_rate"],
    )


def load_timeseries_data(run_dir: Path) -> Optional[Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
    ts_path = run_dir / "timeseries_gat.csv"
    if not ts_path.exists():
        return None
    df = pd.read_csv(ts_path)
    if "t" not in df.columns or "v2v_success" not in df.columns or "v2i_rate" not in df.columns:
        return None
    return (df["t"], df["v2v_success"], df["t"], df["v2i_rate"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GATv2 V2V/V2I curves.")
    parser.add_argument(
        "--source",
        choices=["curve", "timeseries"],
        default="curve",
        help="Use evaluation curve (default) or instantaneous timeseries.",
    )
    args = parser.parse_args()

    root = Path("runs_cmp_fair_seed123_gatv2")
    run_dir = find_latest_run_dir(root, "gat-train-")
    if not run_dir:
        print("[ERROR] No GATv2 run directory found.")
        return
    if args.source == "timeseries":
        data = load_timeseries_data(run_dir)
        if data is None:
            print(f"[ERROR] Missing timeseries file under: {run_dir}")
            return
        steps_v2v, v2v, steps_v2i, v2i = data
        out_path = run_dir / "gatv2_v2v_v2i_timeseries.png"
        xlabel = "t"
        ylim_v2v = (
            max(0.0, float(v2v.min()) - 0.001),
            min(1.02, float(v2v.max()) + 0.001),
        )
    else:
        data = load_curve_data(run_dir)
        if data is None:
            print(f"[ERROR] Missing curve files under: {run_dir}")
            return
        steps_v2v, v2v, steps_v2i, v2i = data
        out_path = run_dir / "gatv2_v2v_v2i_curve.png"
        xlabel = "step"
        ylim_v2v = (0.0, 1.02)

    fig, ax1 = plt.subplots(figsize=(7.0, 4.6))

    color_v2v = "tab:blue"
    color_v2i = "tab:orange"

    l1, = ax1.plot(
        steps_v2v,
        v2v,
        "-o",
        color=color_v2v,
        markersize=4.0,
        linewidth=1.6,
        label="V2V Success Rate",
    )
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("V2V Success Rate", color=color_v2v)
    ax1.tick_params(axis="y", labelcolor=color_v2v)
    ax1.grid(alpha=0.25)
    ax1.set_ylim(*ylim_v2v)

    ax2 = ax1.twinx()
    l2, = ax2.plot(
        steps_v2i,
        v2i,
        "-s",
        color=color_v2i,
        markersize=4.0,
        linewidth=1.6,
        label="V2I Rate",
    )
    ax2.set_ylabel("V2I Rate", color=color_v2i)
    ax2.tick_params(axis="y", labelcolor=color_v2i)

    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper left", frameon=True)
    plt.tight_layout()

    fig.savefig(out_path, dpi=224)
    plt.close(fig)
    print("[Saved]", out_path)


if __name__ == "__main__":
    main()
