"""Plotting helpers kept out of training and metrics hot paths."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "STIXGeneral"

EXPORT_DPI = 1000
PAPER_COLORS = {
    "reward": "#2C7FB8",
    "loss": "#D95F0E",
    "comm": "#1E8449",
    "gap": "#7B6FD0",
    "eval": "#A93226",
}
PAIR_BG = "#DCE6CE"
PAIR_AXIS = "#6B7A61"
PAIR_TICK = "#364032"
PAIR_SHORT = "#1F77B4"
PAIR_LONG = "#F39C12"


def _save_line_plot(
    out_path: Path,
    x_values,
    y_values,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    color: str,
    eval_x=None,
    eval_y=None,
    eval_label: str = "Eval",
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(
        x_values,
        y_values,
        color=color,
        linewidth=1.8,
        marker="o",
        markersize=4.2,
        label="Train",
    )
    if eval_x is not None and eval_y is not None and len(eval_x) > 0:
        ax.plot(
            eval_x,
            eval_y,
            color=PAPER_COLORS["eval"],
            linewidth=1.6,
            marker="s",
            markersize=4.0,
            linestyle="--",
            label=eval_label,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--")
    if eval_x is not None and eval_y is not None and len(eval_x) > 0:
        ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def _moving_average(values: List[float], window: int) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64)
    if series.size == 0:
        return np.asarray([], dtype=np.float64)
    window = max(1, min(int(window), int(series.size)))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    valid = np.convolve(series, kernel, mode="valid")
    prefix = np.asarray([np.mean(series[: idx + 1]) for idx in range(window - 1)], dtype=np.float64)
    return np.concatenate([prefix, valid])


def _save_training_reward_platoon_pair(
    out_path: Path,
    episodes: List[float],
    rewards: List[float],
    platoon_rates: List[float],
) -> None:
    if not episodes:
        return
    short_window = 10
    long_window = 100
    reward_ma_short = _moving_average(rewards, short_window)
    reward_ma_long = _moving_average(rewards, long_window)
    platoon_ma_short = _moving_average(platoon_rates, short_window)
    platoon_ma_long = _moving_average(platoon_rates, long_window)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), facecolor=PAIR_BG)
    for ax, short_values, long_values, ylabel, raw_values in (
        (axes[0], reward_ma_short, reward_ma_long, "Reward", rewards),
        (axes[1], platoon_ma_short, platoon_ma_long, "Platoon Rate", platoon_rates),
    ):
        ax.set_facecolor(PAIR_BG)
        ax.plot(episodes, short_values, color=PAIR_SHORT, linewidth=1.25, solid_capstyle="round")
        ax.plot(episodes, long_values, color=PAIR_LONG, linewidth=2.05, solid_capstyle="round")
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
            spine.set_color(PAIR_AXIS)
            spine.set_linewidth(1.0)
        ax.tick_params(colors=PAIR_TICK, labelsize=10, width=0.8, length=3.5)

    fig.tight_layout(pad=0.65, w_pad=1.5)
    fig.savefig(out_path.with_suffix(".png"), dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def _save_record_line_plot(out_path: Path, records: list[dict], x_key: str, y_key: str, *, title: str, xlabel: str, ylabel: str, color: str) -> None:
    if not records:
        return
    xs = [row.get(x_key, 0.0) for row in records]
    ys = [row.get(y_key, 0.0) for row in records]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(xs, ys, color=color, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def _save_record_bar_plot(out_path: Path, labels: List[str], values: List[float], *, title: str, ylabel: str, color: str) -> None:
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(labels, values, color=color, width=0.7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def export_eval_plots(out_dir: Path, records: list[dict]) -> None:
    out_dir = Path(out_dir)
    _save_record_line_plot(out_dir / "reward_curve", records, "step", "reward", title="Formation Reward Curve", xlabel="Step", ylabel="Reward", color="#2C7FB8")
    _save_record_line_plot(out_dir / "min_gap_curve", records, "step", "min_gap_up", title="Minimum Gap (Up Direction)", xlabel="Step", ylabel="Gap (m)", color="#7B6FD0")
    _save_record_line_plot(out_dir / "comm_success_curve", records, "step", "comm_v2v_success", title="V2V Success Rate", xlabel="Step", ylabel="Success Rate", color="#1E8449")
    _save_record_line_plot(out_dir / "platoon_rate_curve", records, "step", "platoon_rate", title="CAV Platoon Rate", xlabel="Step", ylabel="Rate", color="#2874A6")
    _save_record_line_plot(out_dir / "speed_cav_curve", records, "step", "mean_speed_cav", title="Average CAV Speed", xlabel="Step", ylabel="Speed (m/s)", color="#117A65")
    _save_record_line_plot(out_dir / "speed_hv_curve", records, "step", "mean_speed_hv", title="Average HV Speed", xlabel="Step", ylabel="Speed (m/s)", color="#7D6608")
    _save_record_line_plot(out_dir / "energy_curve", records, "step", "energy_step_kj", title="Step Energy Consumption", xlabel="Step", ylabel="Energy (kJ)", color="#2471A3")
    _save_record_line_plot(out_dir / "shield_intervention_curve", records, "step", "shield_interventions", title="Safety Shield Interventions", xlabel="Step", ylabel="Interventions", color="#D95F0E")
    _save_record_line_plot(out_dir / "cbf_adjustment_curve", records, "step", "shield_cbf_adjustments", title="CBF Safety Adjustments", xlabel="Step", ylabel="Adjustments", color="#AF601A")
    _save_record_line_plot(out_dir / "cbf_barrier_curve", records, "step", "shield_cbf_min_barrier", title="Minimum Longitudinal Barrier", xlabel="Step", ylabel="Barrier (m)", color="#6C5CE7")
    _save_record_line_plot(out_dir / "collision_curve", records, "step", "collision_pairs_active", title="Active Collision Pairs", xlabel="Step", ylabel="Pair Count", color="#C0392B")
    _save_record_line_plot(out_dir / "topology_switch_curve", records, "step", "topology_switches", title="Topology Switches", xlabel="Step", ylabel="Switch Count", color="#A93226")

    labels = ["keep", "eco_keep", "gap_recover", "compact", "expand", "split", "merge", "emergency"]
    action_values = [
        float(np.mean([row.get(f"action_{label}", 0.0) for row in records]))
        for label in labels
    ]
    _save_record_bar_plot(out_dir / "action_distribution", labels, action_values, title="Average High-Level Action Distribution", ylabel="Ratio", color="#2C7FB8")


def export_training_plots(out_dir: Path, training_rows: List[Dict[str, float]], eval_rows: List[Dict[str, float]]) -> None:
    if not training_rows:
        return

    train_ep = [row["episode"] for row in training_rows]
    train_reward = [row["episode_reward"] for row in training_rows]
    train_loss = [row["avg_loss"] for row in training_rows]
    train_comm = [row["avg_comm_v2v_success"] for row in training_rows]
    train_gap = [row["avg_min_gap"] for row in training_rows]
    train_platoon = [row.get("avg_platoon_rate", 0.0) for row in training_rows]
    train_pressure = [row.get("avg_event_pressure", 0.0) for row in training_rows]
    train_blocked = [row.get("avg_blocked_lane_ratio", 0.0) for row in training_rows]
    train_switch = [row.get("avg_topology_switch_indicator", 0.0) for row in training_rows]

    eval_ep = []
    eval_reward = []
    eval_comm = []
    eval_gap = []
    eval_platoon = []
    for row in eval_rows:
        tag = str(row.get("episode_tag", ""))
        if tag.startswith("ep"):
            try:
                eval_ep.append(float(int(tag[2:])))
            except Exception:
                continue
            eval_reward.append(float(row.get("avg_reward", 0.0)))
            eval_comm.append(float(row.get("avg_comm_v2v_success", 0.0)))
            gap_up = float(row.get("worst_min_gap_up", 0.0))
            gap_down = float(row.get("worst_min_gap_down", 0.0))
            eval_gap.append(min(gap_up, gap_down))
            eval_platoon.append(float(row.get("avg_platoon_rate", 0.0)))

    _save_line_plot(out_dir / "training_reward_curve", train_ep, train_reward, xlabel="Episode", ylabel="Episode Reward", title="High-Level Joint Training Reward", color=PAPER_COLORS["reward"], eval_x=eval_ep, eval_y=eval_reward, eval_label="Eval Reward")
    _save_line_plot(out_dir / "training_loss_curve", train_ep, train_loss, xlabel="Episode", ylabel="Average Loss", title="High-Level Joint Training Loss", color=PAPER_COLORS["loss"])
    _save_line_plot(out_dir / "training_comm_success_curve", train_ep, train_comm, xlabel="Episode", ylabel="Average V2V Success", title="Communication Success During Joint Training", color=PAPER_COLORS["comm"], eval_x=eval_ep, eval_y=eval_comm, eval_label="Eval V2V Success")
    _save_line_plot(out_dir / "training_min_gap_curve", train_ep, train_gap, xlabel="Episode", ylabel="Average Min Gap (m)", title="Minimum Gap During Joint Training", color=PAPER_COLORS["gap"], eval_x=eval_ep, eval_y=eval_gap, eval_label="Eval Worst Min Gap")
    _save_line_plot(out_dir / "training_platoon_rate_curve", train_ep, train_platoon, xlabel="Episode", ylabel="Average Platoon Rate", title="Platoon Rate During Joint Training", color="#2874A6", eval_x=eval_ep, eval_y=eval_platoon, eval_label="Eval Platoon Rate")
    _save_line_plot(out_dir / "training_event_pressure_curve", train_ep, train_pressure, xlabel="Episode", ylabel="Average Event Pressure", title="Event Pressure During Joint Training", color="#8E44AD")
    _save_line_plot(out_dir / "training_blocked_lane_curve", train_ep, train_blocked, xlabel="Episode", ylabel="Blocked Lane Ratio", title="Blocked Lane Ratio During Joint Training", color="#566573")
    _save_line_plot(out_dir / "training_topology_switch_curve", train_ep, train_switch, xlabel="Episode", ylabel="Topology Switch Indicator", title="Topology Switch Indicator During Joint Training", color="#B9770E")
    _save_training_reward_platoon_pair(out_dir / "training_reward_platoon_pair", train_ep, train_reward, train_platoon)
