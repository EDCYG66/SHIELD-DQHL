"""Render a paper-style reward/platoon training figure from saved CSV logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "STIXGeneral"

EXPORT_DPI = 1000
PANEL_BG = "#DCE6CE"
SHORT_COLOR = "#1F77B4"
LONG_COLOR = "#F39C12"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot reward/platoon training pair in paper style")
    parser.add_argument("--run-dir", type=str, required=True, help="Directory containing high_level_training_history.csv")
    parser.add_argument("--out-name", type=str, default="training_reward_platoon_pair_paper")
    parser.add_argument("--window-short", type=int, default=10)
    parser.add_argument("--window-long", type=int, default=100)
    return parser


def moving_average(values: List[float], window: int) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64)
    if series.size == 0:
        return np.asarray([], dtype=np.float64)
    window = max(1, min(int(window), int(series.size)))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    valid = np.convolve(series, kernel, mode="valid")
    prefix = np.asarray([np.mean(series[: idx + 1]) for idx in range(window - 1)], dtype=np.float64)
    return np.concatenate([prefix, valid])


def load_training_history(csv_path: Path) -> tuple[List[float], List[float], List[float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    episodes = [float(row["episode"]) for row in rows]
    rewards = [float(row["episode_reward"]) for row in rows]
    platoon_rates = [float(row.get("avg_platoon_rate", 0.0)) for row in rows]
    return episodes, rewards, platoon_rates


def render_pair(
    out_base: Path,
    episodes: List[float],
    rewards: List[float],
    platoon_rates: List[float],
    *,
    window_short: int,
    window_long: int,
) -> None:
    reward_short = moving_average(rewards, window_short)
    reward_long = moving_average(rewards, window_long)
    platoon_short = moving_average(platoon_rates, window_short)
    platoon_long = moving_average(platoon_rates, window_long)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), facecolor=PANEL_BG)
    for ax, short_values, long_values, ylabel, raw_values in (
        (axes[0], reward_short, reward_long, "Reward", rewards),
        (axes[1], platoon_short, platoon_long, "Platoon Rate", platoon_rates),
    ):
        ax.set_facecolor(PANEL_BG)
        ax.plot(episodes, short_values, color=SHORT_COLOR, linewidth=1.25, solid_capstyle="round")
        ax.plot(episodes, long_values, color=LONG_COLOR, linewidth=2.05, solid_capstyle="round")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_xlim(float(episodes[0]), float(episodes[-1]))
        y_values = np.asarray(raw_values, dtype=np.float64)
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))
        if y_max > y_min:
            pad = 0.06 * (y_max - y_min)
            ax.set_ylim(y_min - pad, y_max + pad)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color("#6B7A61")
            spine.set_linewidth(1.0)
        ax.tick_params(colors="#364032", labelsize=10, width=0.8, length=3.5)

    fig.tight_layout(pad=0.65, w_pad=1.5)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    csv_path = run_dir / "high_level_training_history.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Training history not found: {csv_path}")
    episodes, rewards, platoon_rates = load_training_history(csv_path)
    render_pair(
        run_dir / args.out_name,
        episodes,
        rewards,
        platoon_rates,
        window_short=args.window_short,
        window_long=args.window_long,
    )
    print(f"Paper-style training pair saved to: {run_dir / args.out_name}")


if __name__ == "__main__":
    main()
