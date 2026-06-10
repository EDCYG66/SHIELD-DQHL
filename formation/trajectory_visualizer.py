"""Trajectory and time-series plotting helpers for pure formation simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .scenario_renderer import (
        BLOCKED_ZONE,
        DOWN_COLOR,
        EVENT_ZONE,
        EXPORT_DPI,
        LEADER_EDGE,
        SimulationHistory,
        UP_COLOR,
        _draw_event_regions,
        _draw_segment_boundaries,
        _draw_static_road,
        _scene_legend_handles,
    )
except ImportError:  # pragma: no cover
    from scenario_renderer import (
        BLOCKED_ZONE,
        DOWN_COLOR,
        EVENT_ZONE,
        EXPORT_DPI,
        LEADER_EDGE,
        SimulationHistory,
        UP_COLOR,
        _draw_event_regions,
        _draw_segment_boundaries,
        _draw_static_road,
        _scene_legend_handles,
    )


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def _save_png_pdf(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def _event_spans(history: SimulationHistory) -> List[tuple[float, float]]:
    spans = []
    for event in history.events:
        spans.append((float(event.start_step), float(event.end_step)))
    return spans


def _shade_event_windows(ax: plt.Axes, history: SimulationHistory) -> None:
    for start, end in _event_spans(history):
        ax.axvspan(start, end, color=EVENT_ZONE, alpha=0.16, lw=0.0, zorder=0)


def plot_trajectory_overview(history: SimulationHistory, out_dir: str | Path) -> Path:
    """Plot full vehicle trajectories over the road/event background."""

    out_dir = Path(out_dir)
    if not history.frames:
        raise ValueError("Simulation history is empty.")

    fig, ax = plt.subplots(figsize=(7.0, 9.6), facecolor="white")
    _draw_static_road(ax, history)
    _draw_segment_boundaries(ax, history)
    _draw_event_regions(ax, history)

    x_min = min(history.up_lanes + history.down_lanes) - 2.2 * history.lane_width
    x_max = max(history.up_lanes + history.down_lanes) + 2.2 * history.lane_width

    nveh = max((len(frame.vehicles) for frame in history.frames), default=0)
    for veh_idx in range(nveh):
        xs: List[float] = []
        ys: List[float] = []
        direction = None
        is_leader = False
        for frame in history.frames:
            if veh_idx >= len(frame.vehicles):
                continue
            veh = frame.vehicles[veh_idx]
            xs.append(veh.x)
            ys.append(veh.y)
            direction = veh.direction
            is_leader = is_leader or veh.is_leader
        if not xs:
            continue
        color = UP_COLOR if direction == "u" else DOWN_COLOR
        lw = 1.9 if is_leader else 1.05
        alpha = 0.92 if is_leader else 0.55
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, zorder=2)
        ax.scatter(xs[0], ys[0], color=color, s=18, alpha=0.55, marker="o", zorder=3)
        ax.scatter(xs[-1], ys[-1], color=color, s=26, alpha=0.95, marker="D", edgecolors=LEADER_EDGE, linewidths=0.6, zorder=4)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, history.height)
    ax.set_xlabel("Lateral position (m)")
    ax.set_ylabel("Longitudinal position (m)")
    ax.set_title("Platoon trajectory evolution under the incident scenario")
    ax.grid(axis="y", alpha=0.18, linestyle="--")

    legend_handles = [
        plt.Line2D([0], [0], color=UP_COLOR, linewidth=1.8, label="Up-direction trajectories"),
        plt.Line2D([0], [0], color=DOWN_COLOR, linewidth=1.8, label="Down-direction trajectories"),
        *_scene_legend_handles(history)[2:],
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=9)

    out_base = out_dir / "trajectory_overview"
    _save_png_pdf(fig, out_base)
    return out_base.with_suffix(".png")


def plot_timeseries_dashboard(history: SimulationHistory, out_dir: str | Path) -> Path:
    """Plot reward, communication, safety, and control time series in one figure."""

    out_dir = Path(out_dir)
    if not history.frames:
        raise ValueError("Simulation history is empty.")

    steps = np.asarray([frame.step for frame in history.frames], dtype=float)
    rewards = np.asarray([frame.reward for frame in history.frames], dtype=float)
    v2v = np.asarray([frame.state_fields.get("comm_v2v_success", 0.0) for frame in history.frames], dtype=float)
    v2i = np.asarray([frame.state_fields.get("comm_v2i_rate_total", 0.0) for frame in history.frames], dtype=float)
    min_gap_up = np.asarray([frame.state_fields.get("min_gap_up", 0.0) for frame in history.frames], dtype=float)
    min_gap_down = np.asarray([frame.state_fields.get("min_gap_down", 0.0) for frame in history.frames], dtype=float)
    speed_up = np.asarray([frame.state_fields.get("mean_speed_up", 0.0) for frame in history.frames], dtype=float)
    speed_down = np.asarray([frame.state_fields.get("mean_speed_down", 0.0) for frame in history.frames], dtype=float)
    lane_changes = np.asarray([frame.info.get("lane_changes", 0.0) for frame in history.frames], dtype=float)
    shield_interventions = np.asarray([frame.info.get("shield_interventions", 0.0) for frame in history.frames], dtype=float)
    cbf_adjustments = np.asarray([frame.info.get("shield_cbf_adjustments", 0.0) for frame in history.frames], dtype=float)
    cacc_ratio = np.asarray([frame.info.get("cacc_cmd_ratio", 0.0) for frame in history.frames], dtype=float)
    hybrid_ratio = np.asarray([frame.info.get("hybrid_cmd_ratio", 0.0) for frame in history.frames], dtype=float)
    cidm_ratio = np.asarray([frame.info.get("cidm_cmd_ratio", 0.0) for frame in history.frames], dtype=float)

    fig, axes = plt.subplots(3, 2, figsize=(10.4, 9.2), facecolor="white")
    axes = axes.reshape(-1)

    axes[0].plot(steps, rewards, color="#2C7FB8", linewidth=1.7)
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Reward evolution")

    axes[1].plot(steps, v2v, color="#1E8449", linewidth=1.7, label="V2V success")
    ax1b = axes[1].twinx()
    ax1b.plot(steps, v2i, color="#D95F0E", linewidth=1.4, linestyle="--", label="V2I rate")
    axes[1].set_ylabel("V2V success")
    ax1b.set_ylabel("V2I rate")
    axes[1].set_title("Communication metrics")
    handles = axes[1].get_lines() + ax1b.get_lines()
    labels = [line.get_label() for line in handles]
    axes[1].legend(handles, labels, loc="upper right", fontsize=8, frameon=True)

    axes[2].plot(steps, min_gap_up, color=UP_COLOR, linewidth=1.6, label="Up direction")
    axes[2].plot(steps, min_gap_down, color=DOWN_COLOR, linewidth=1.6, linestyle="--", label="Down direction")
    axes[2].set_ylabel("Min gap (m)")
    axes[2].set_title("Safety gap evolution")
    axes[2].legend(loc="upper right", fontsize=8, frameon=True)

    axes[3].plot(steps, speed_up, color=UP_COLOR, linewidth=1.6, label="Up direction")
    axes[3].plot(steps, speed_down, color=DOWN_COLOR, linewidth=1.6, linestyle="--", label="Down direction")
    axes[3].set_ylabel("Mean speed (m/s)")
    axes[3].set_title("Mean speed evolution")
    axes[3].legend(loc="upper right", fontsize=8, frameon=True)

    axes[4].plot(steps, lane_changes, color="#8E44AD", linewidth=1.6, label="Lane changes")
    axes[4].plot(steps, shield_interventions, color="#7F8C8D", linewidth=1.4, linestyle="--", label="Shield interventions")
    axes[4].plot(steps, cbf_adjustments, color="#AF601A", linewidth=1.4, linestyle="-.", label="CBF adjustments")
    axes[4].set_ylabel("Count")
    axes[4].set_title("Reconfiguration activity")
    axes[4].legend(loc="upper right", fontsize=8, frameon=True)

    axes[5].plot(steps, cacc_ratio, color="#2874A6", linewidth=1.6, label="CACC ratio")
    axes[5].plot(steps, hybrid_ratio, color="#7D3C98", linewidth=1.6, linestyle="-.", label="Hybrid ratio")
    axes[5].plot(steps, cidm_ratio, color="#AF601A", linewidth=1.6, linestyle="--", label="C-IDM ratio")
    axes[5].set_ylabel("Ratio")
    axes[5].set_title("Low-level control mode ratio")
    axes[5].legend(loc="upper right", fontsize=8, frameon=True)

    for ax in axes:
        _shade_event_windows(ax, history)
        ax.set_xlabel("Step")
        ax.grid(alpha=0.22, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle("Pure-simulation joint platoon experiment dashboard", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    out_base = out_dir / "timeseries_dashboard"
    _save_png_pdf(fig, out_base)
    return out_base.with_suffix(".png")
