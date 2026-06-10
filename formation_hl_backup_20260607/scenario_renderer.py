"""Pure-simulation renderers for platoon reconfiguration scenes and animations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle

try:
    from .events import TrafficEvent
except ImportError:  # pragma: no cover
    from events import TrafficEvent


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

EXPORT_DPI = 1000
ANIMATION_DPI = 180
LANE_COLOR = "#B8BDC5"
MEDIAN_COLOR = "#E8EAF0"
ROAD_BG = "#FAFAFA"
ROAD_SURFACE = "#E6E9EE"
SHOULDER_COLOR = "#C9CED6"
EVENT_ZONE = "#F9D9A7"
BLOCKED_ZONE = "#F2A65A"
UP_COLOR = "#D55E5E"
DOWN_COLOR = "#4C7FB8"
LEADER_EDGE = "#1F1F1F"
MEDIAN_GREEN = "#DCE7D2"
GUARDRAIL_COLOR = "#7D8793"
BS_COLOR = "#3D8C54"
V2I_COLOR = "#9AA0A6"
TEXT_COLOR = "#222222"
WARNING_SIGN_EDGE = "#BB2E2E"
WARNING_SIGN_FILL = "#FFF8F4"
INCIDENT_LABEL_BG = "#FFF3C8"
SPEED_NORM = Normalize(vmin=0.0, vmax=30.0)
SPEED_CMAP = plt.get_cmap("RdYlBu")
COMM_CMAP = plt.get_cmap("RdYlGn")
VEHICLE_DRAW_LENGTH_SCALE = 1.22
VEHICLE_DRAW_WIDTH_SCALE = 1.16


@dataclass
class VehicleSnapshot:
    idx: int
    x: float
    y: float
    velocity: float
    yaw: float
    vehicle_length: float
    vehicle_width: float
    lane_idx: int
    direction: str
    is_leader: bool
    is_cav: bool = True
    veh_type: str = "cav"
    serving_bs_idx: int = -1
    neighbors: Tuple[int, ...] = ()
    destinations: Tuple[int, ...] = ()


@dataclass
class FrameSnapshot:
    step: int
    time_s: float
    action: str
    reward: float
    topology_type: str
    state_fields: Dict[str, float]
    info: Dict[str, float]
    vehicles: List[VehicleSnapshot]


@dataclass
class SimulationHistory:
    width: float
    height: float
    base_y: float
    lane_width: float
    up_lanes: List[float]
    down_lanes: List[float]
    bs_positions: List[Tuple[float, float]]
    events: List[TrafficEvent]
    metadata: Dict[str, object] = field(default_factory=dict)
    frames: List[FrameSnapshot] = field(default_factory=list)

    @classmethod
    def from_env(cls, env_wrapper, **metadata) -> "SimulationHistory":
        road_env = env_wrapper.env
        return cls(
            width=float(getattr(road_env, "width", 0.0)),
            height=float(getattr(road_env, "height", 0.0)),
            base_y=float(getattr(road_env, "base_y", 0.0)),
            lane_width=float(getattr(road_env, "lane_width", 3.5)),
            up_lanes=[float(value) for value in getattr(road_env, "true_up_lanes", [])],
            down_lanes=[float(value) for value in getattr(road_env, "true_down_lanes", [])],
            bs_positions=[(float(x), float(y)) for x, y in getattr(road_env, "bs_positions", [])],
            events=list(getattr(env_wrapper.scheduler, "events", [])),
            metadata=dict(metadata),
        )


def capture_frame(
    env_wrapper,
    *,
    step: int,
    action: str,
    reward: float = 0.0,
    state_fields: Optional[Dict[str, float]] = None,
    info: Optional[Dict[str, object]] = None,
) -> FrameSnapshot:
    """Capture a lightweight snapshot of the current simulation state."""

    road_env = env_wrapper.env
    state_fields = {key: float(value) for key, value in (state_fields or {}).items() if isinstance(value, (int, float))}
    info = info or {}
    flat_info = {
        key: float(value)
        for key, value in info.items()
        if key not in {"shield", "communication"} and isinstance(value, (int, float))
    }
    shield = info.get("shield", {}) if isinstance(info, dict) else {}
    if isinstance(shield, dict):
        flat_info["shield_interventions"] = float(shield.get("interventions", 0.0))
        flat_info["shield_lane_change_blocks"] = float(shield.get("lane_change_blocks", 0.0))
        flat_info["shield_emergency_brakes"] = float(shield.get("emergency_brakes", 0.0))
        flat_info["shield_cbf_adjustments"] = float(shield.get("cbf_adjustments", 0.0))
        flat_info["shield_cbf_longitudinal_clips"] = float(shield.get("cbf_longitudinal_clips", 0.0))
        flat_info["shield_cbf_lateral_clips"] = float(shield.get("cbf_lateral_clips", 0.0))
        flat_info["shield_cbf_blocked_lane_avoid"] = float(shield.get("cbf_blocked_lane_avoid", 0.0))
        flat_info["shield_cbf_min_barrier"] = float(shield.get("cbf_min_barrier", 0.0))
        flat_info["shield_cbf_lane_barrier"] = float(shield.get("cbf_lane_barrier", 0.0))

    leader_up = int(getattr(road_env, "leader_idx_up", -1))
    leader_down = int(getattr(road_env, "leader_idx_down", -1))
    serving_idx = np.asarray(getattr(road_env, "v2i_serving_idx", np.full(len(road_env.vehicles), -1, dtype=int)), dtype=int)
    vehicles: List[VehicleSnapshot] = []
    for idx, veh in enumerate(getattr(road_env, "vehicles", [])):
        is_leader = idx in {leader_up, leader_down}
        vehicles.append(
            VehicleSnapshot(
                idx=int(idx),
                x=float(veh.position[0]),
                y=float(veh.position[1]),
                velocity=float(getattr(veh, "velocity", 0.0)),
                yaw=float(getattr(veh, "yaw", 0.0)),
                vehicle_length=float(getattr(veh, "vehicle_length", 4.8)),
                vehicle_width=float(getattr(veh, "vehicle_width", 1.85)),
                lane_idx=int(road_env._lane_idx[idx]),
                direction=str(getattr(veh, "direction", "u")),
                is_leader=bool(is_leader),
                is_cav=bool(getattr(veh, "is_cav", True)),
                veh_type=str(getattr(veh, "veh_type", "cav")),
                serving_bs_idx=int(serving_idx[idx]) if idx < len(serving_idx) else -1,
                neighbors=tuple(int(v) for v in getattr(veh, "neighbors", [])),
                destinations=tuple(int(v) for v in getattr(veh, "destinations", [])),
            )
        )

    timestep = float(getattr(road_env, "timestep", 0.01))
    return FrameSnapshot(
        step=int(step),
        time_s=float(step) * timestep,
        action=str(action),
        reward=float(reward),
        topology_type=str(getattr(road_env, "topology_type", "star")),
        state_fields=state_fields,
        info=flat_info,
        vehicles=vehicles,
    )


def _save_png_pdf(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def _lane_group_bounds(lanes: Sequence[float], lane_width: float) -> Tuple[float, float]:
    if not lanes:
        return 0.0, 0.0
    return min(lanes) - 0.65 * lane_width, max(lanes) + 0.65 * lane_width


def _roadway_bounds(lanes: Sequence[float], lane_width: float) -> Tuple[float, float]:
    if not lanes:
        return 0.0, 0.0
    return min(lanes) - 0.5 * lane_width, max(lanes) + 0.5 * lane_width


def _road_x_bounds(history: SimulationHistory) -> Tuple[float, float]:
    lanes = history.up_lanes + history.down_lanes
    if not lanes:
        return -2.0 * history.lane_width, 2.0 * history.lane_width
    return min(lanes) - 2.2 * history.lane_width, max(lanes) + 2.2 * history.lane_width


def _draw_lane_family(
    ax: plt.Axes,
    lanes: Sequence[float],
    lane_width: float,
    *,
    shoulder_side: str,
) -> None:
    if not lanes:
        return
    lane_left, lane_right = _roadway_bounds(lanes, lane_width)
    shoulder_width = 0.42 * lane_width

    ax.add_patch(
        Rectangle(
            (lane_left, 0.0),
            lane_right - lane_left,
            ax.get_ylim()[1],
            facecolor=ROAD_SURFACE,
            edgecolor="none",
            zorder=0.02,
        )
    )

    if shoulder_side == "left":
        shoulder_x = lane_left - shoulder_width
        ax.add_patch(
            Rectangle(
                (shoulder_x, 0.0),
                shoulder_width,
                ax.get_ylim()[1],
                facecolor=SHOULDER_COLOR,
                edgecolor="none",
                zorder=0.03,
            )
        )
        ax.plot([shoulder_x, shoulder_x], [0.0, ax.get_ylim()[1]], color="#F8FBFF", linewidth=2.0, zorder=1.1)
    else:
        ax.add_patch(
            Rectangle(
                (lane_right, 0.0),
                shoulder_width,
                ax.get_ylim()[1],
                facecolor=SHOULDER_COLOR,
                edgecolor="none",
                zorder=0.03,
            )
        )
        ax.plot([lane_right + shoulder_width, lane_right + shoulder_width], [0.0, ax.get_ylim()[1]], color="#F8FBFF", linewidth=2.0, zorder=1.1)

    lane_positions = sorted(float(x) for x in lanes)
    outer_edges = [lane_positions[0] - 0.5 * lane_width, lane_positions[-1] + 0.5 * lane_width]
    for x in outer_edges:
        ax.plot([x, x], [0.0, ax.get_ylim()[1]], color="#FBFCFE", linewidth=2.0, zorder=1.15)

    for left_center, right_center in zip(lane_positions[:-1], lane_positions[1:]):
        divider_x = 0.5 * (left_center + right_center)
        ax.plot(
            [divider_x, divider_x],
            [0.0, ax.get_ylim()[1]],
            color="#F7FAFD",
            linewidth=1.2,
            linestyle=(0, (8, 7)),
            alpha=0.92,
            zorder=1.12,
        )


def _draw_guardrail(ax: plt.Axes, x_center: float, height: float, lane_width: float) -> None:
    y_positions = np.arange(18.0, max(height - 18.0, 18.0), max(24.0, 6.0 * lane_width))
    for y0 in y_positions:
        ax.plot(
            [x_center, x_center],
            [y0 - 4.5, y0 + 4.5],
            color=GUARDRAIL_COLOR,
            linewidth=1.0,
            alpha=0.8,
            zorder=1.22,
        )
        ax.scatter([x_center], [y0], s=8, c=GUARDRAIL_COLOR, alpha=0.9, zorder=1.25)


def _draw_event_signage(ax: plt.Axes, history: SimulationHistory) -> None:
    if not history.events:
        return
    lane_width = history.lane_width
    up_left = min(history.up_lanes) - 0.72 * lane_width if history.up_lanes else None
    down_right = max(history.down_lanes) + 0.72 * lane_width if history.down_lanes else None
    for event in history.events:
        advisory_kmh = int(max(30.0, round(80.0 * float(getattr(event, "speed_limit_ratio", 0.4)))))
        if up_left is not None:
            y_sign = max(28.0, float(event.y_start) - 42.0)
            _draw_speed_sign(ax, up_left, y_sign, advisory_kmh)
            _draw_incident_badge(ax, up_left + 1.15 * lane_width, y_sign + 19.0, "INCIDENT")
        if down_right is not None:
            y_sign = min(history.height - 28.0, float(event.y_end) + 42.0)
            _draw_speed_sign(ax, down_right, y_sign, advisory_kmh)
            _draw_incident_badge(ax, down_right - 1.15 * lane_width, y_sign - 19.0, "INCIDENT")


def _draw_speed_sign(ax: plt.Axes, x: float, y: float, speed_kmh: int) -> None:
    radius = 1.02
    ax.plot([x, x], [y - 6.0, y - 1.1], color="#717985", linewidth=1.1, zorder=1.6)
    ax.add_patch(
        Circle(
            (x, y),
            radius=radius,
            facecolor=WARNING_SIGN_FILL,
            edgecolor=WARNING_SIGN_EDGE,
            linewidth=1.3,
            zorder=1.7,
        )
    )
    ax.text(x, y, str(int(speed_kmh)), ha="center", va="center", fontsize=6.4, color="#8D1F1F", zorder=1.8)


def _draw_incident_badge(ax: plt.Axes, x: float, y: float, text: str) -> None:
    width = 2.3 + 0.18 * len(text)
    height = 2.0
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.5 * width, y - 0.5 * height),
            width,
            height,
            boxstyle="round,pad=0.12,rounding_size=0.22",
            facecolor=INCIDENT_LABEL_BG,
            edgecolor="#B8871A",
            linewidth=0.9,
            alpha=0.95,
            zorder=1.75,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=5.6, color="#6A4A06", zorder=1.8)


def _draw_static_road(ax: plt.Axes, history: SimulationHistory, *, show_titles: bool = True) -> None:
    ax.set_facecolor(ROAD_BG)
    x_min, x_max = _road_x_bounds(history)
    ax.add_patch(
        Rectangle(
            (x_min, 0.0),
            x_max - x_min,
            history.height,
            facecolor=ROAD_BG,
            edgecolor="none",
            zorder=0,
        )
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, history.height)

    _draw_lane_family(ax, history.up_lanes, history.lane_width, shoulder_side="left")
    _draw_lane_family(ax, history.down_lanes, history.lane_width, shoulder_side="right")

    if history.up_lanes and history.down_lanes:
        median_left = max(history.up_lanes) + 0.5 * history.lane_width
        median_right = min(history.down_lanes) - 0.5 * history.lane_width
        ax.add_patch(
            Rectangle(
                (median_left, 0.0),
                max(0.0, median_right - median_left),
                history.height,
                facecolor=MEDIAN_GREEN,
                edgecolor="none",
                zorder=0.12,
            )
        )
        _draw_guardrail(ax, 0.5 * (median_left + median_right), history.height, history.lane_width)

    if show_titles:
        up_left, up_right = _lane_group_bounds(history.up_lanes, history.lane_width)
        down_left, down_right = _lane_group_bounds(history.down_lanes, history.lane_width)
        if history.up_lanes:
            ax.text(0.5 * (up_left + up_right), history.height - 22.0, "Upstream platoon", ha="center", va="center", fontsize=9, color="#6A1B1B")
        if history.down_lanes:
            ax.text(0.5 * (down_left + down_right), history.height - 22.0, "Downstream platoon", ha="center", va="center", fontsize=9, color="#1F4D7A")
    _draw_event_signage(ax, history)


def _draw_segment_boundaries(ax: plt.Axes, history: SimulationHistory) -> None:
    segments = history.metadata.get("segments", []) if isinstance(history.metadata, dict) else []
    if not segments:
        return
    x_min, x_max = _road_x_bounds(history)
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        y0 = float(seg.get("start_y", 0.0))
        y1 = float(seg.get("end_y", 0.0))
        name = str(seg.get("name", "segment"))
        lane_count = seg.get("lane_count", None)
        speed_limit_ratio = seg.get("speed_limit_ratio", None)
        if bool(seg.get("ramp_in", False)):
            band_color = "#D8EAD3"
        elif bool(seg.get("ramp_out", False)):
            band_color = "#D8E2F0"
        elif bool(seg.get("merge_zone", False)) and bool(seg.get("split_zone", False)):
            band_color = "#E8D8F5"
        elif bool(seg.get("split_zone", False)):
            band_color = "#F6E5D9"
        elif bool(seg.get("merge_zone", False)):
            band_color = "#D9F0E1"
        else:
            band_color = "#EEF2F7"
        ax.add_patch(
            Rectangle(
                (x_min, y0),
                x_max - x_min,
                max(0.0, y1 - y0),
                facecolor=band_color,
                edgecolor="#C8CFD8",
                linewidth=0.7,
                alpha=0.14,
                zorder=0.08,
            )
        )
        ax.axhline(y0, color="#7E8A97", linewidth=0.8, linestyle="--", alpha=0.55, zorder=0.35)
        label_bits = [name.replace("_", " ").title()]
        if lane_count is not None:
            label_bits.append(f"lanes={lane_count}")
        if speed_limit_ratio is not None:
            label_bits.append(f"v={float(speed_limit_ratio):.2f}")
        ax.text(
            x_min + 0.4 * history.lane_width,
            0.5 * (y0 + y1),
            " | ".join(label_bits),
            ha="left",
            va="center",
            fontsize=7.5,
            color="#52606D",
            zorder=0.5,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#D0D5DD", "alpha": 0.82},
        )


def _draw_event_regions(ax: plt.Axes, history: SimulationHistory, active_step: Optional[int] = None) -> None:
    x_min, x_max = _road_x_bounds(history)
    x_min -= 0.85 * history.lane_width
    x_max += 0.85 * history.lane_width
    for event in history.events:
        is_active = active_step is not None and event.active(active_step)
        ax.add_patch(
            Rectangle(
                (x_min, event.y_start),
                x_max - x_min,
                event.y_end - event.y_start,
                facecolor=EVENT_ZONE,
                edgecolor="#D88D2F",
                linewidth=1.0 if is_active else 0.8,
                alpha=0.30 if is_active else 0.16,
                zorder=0.2,
            )
        )
        for direction, lanes in event.blocked_lanes.items():
            lane_positions = history.up_lanes if direction == "u" else history.down_lanes
            for lane_idx in lanes:
                if not (0 <= int(lane_idx) < len(lane_positions)):
                    continue
                lane_x = lane_positions[int(lane_idx)] - 0.5 * history.lane_width
                ax.add_patch(
                    Rectangle(
                        (lane_x, event.y_start),
                        history.lane_width,
                        event.y_end - event.y_start,
                        facecolor=BLOCKED_ZONE,
                        edgecolor="#B85C00",
                        linewidth=1.0,
                        alpha=0.35 if is_active else 0.22,
                        hatch="////",
                        zorder=0.3,
                    )
                )


def _draw_base_stations(ax: plt.Axes, history: SimulationHistory) -> None:
    if not history.bs_positions:
        return
    xs = [item[0] for item in history.bs_positions]
    ys = [item[1] for item in history.bs_positions]
    ax.scatter(xs, ys, s=76, marker="s", c=BS_COLOR, edgecolors="#1E4028", linewidths=0.8, zorder=4)


def _draw_v2i_links(ax: plt.Axes, history: SimulationHistory, frame: FrameSnapshot) -> None:
    if not history.bs_positions:
        return
    for veh in frame.vehicles:
        if not (0 <= veh.serving_bs_idx < len(history.bs_positions)):
            continue
        bx, by = history.bs_positions[veh.serving_bs_idx]
        ax.plot([veh.x, bx], [veh.y, by], color=V2I_COLOR, linestyle="--", linewidth=0.6, alpha=0.32, zorder=2)


def _draw_neighbor_links(ax: plt.Axes, frame: FrameSnapshot) -> None:
    veh_map = {veh.idx: veh for veh in frame.vehicles}
    drawn = set()
    for veh in frame.vehicles:
        for nb in veh.neighbors:
            edge = tuple(sorted((veh.idx, int(nb))))
            if edge in drawn or nb not in veh_map:
                continue
            drawn.add(edge)
            other = veh_map[nb]
            ax.plot([veh.x, other.x], [veh.y, other.y], color="#9096A0", linewidth=0.55, alpha=0.22, zorder=2)


def _front_gap_map(frame: FrameSnapshot) -> Dict[int, float]:
    gap_map: Dict[int, float] = {}
    lane_buckets: Dict[tuple[str, int], List[VehicleSnapshot]] = {}
    for veh in frame.vehicles:
        lane_buckets.setdefault((veh.direction, int(veh.lane_idx)), []).append(veh)

    for (direction, _lane_idx), bucket in lane_buckets.items():
        bucket.sort(key=lambda item: item.y, reverse=(direction == "u"))
        for front, follower in zip(bucket[:-1], bucket[1:]):
            gap = abs(float(front.y) - float(follower.y))
            gap_map[int(follower.idx)] = float(gap)
    return gap_map


def _vehicle_style(frame: FrameSnapshot, veh: VehicleSnapshot) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float], float]:
    speed_ratio = float(np.clip(float(veh.velocity) / max(SPEED_NORM.vmax, 1e-6), 0.0, 1.0))
    face_color = SPEED_CMAP(speed_ratio)
    comm_quality = float(np.clip(frame.state_fields.get("comm_v2v_success", 0.0), 0.0, 1.0))
    if not veh.is_cav:
        gray = np.asarray(matplotlib.colors.to_rgba("#A0A5AD"))
        face = 0.55 * np.asarray(face_color) + 0.45 * gray
        face_color = tuple(face.tolist())
        edge_color = matplotlib.colors.to_rgba("#5F6770")
        line_width = 0.9 + (0.40 if veh.is_leader else 0.0)
    else:
        edge_color = COMM_CMAP(comm_quality)
        line_width = 0.8 + 1.2 * comm_quality + (0.45 if veh.is_leader else 0.0)
    return face_color, edge_color, float(line_width)


def _draw_group_overlays(ax: plt.Axes, frame: FrameSnapshot) -> None:
    group_specs = [
        ("u", "#F6B8B8", "#B54C4C", "Upstream platoon"),
        ("d", "#B9D2F2", "#3F74A8", "Downstream platoon"),
    ]
    for direction, fill_color, edge_color, label in group_specs:
        group = [veh for veh in frame.vehicles if veh.direction == direction]
        if not group:
            continue
        xs = np.asarray([veh.x for veh in group], dtype=float)
        ys = np.asarray([veh.y for veh in group], dtype=float)
        x_span = float(np.max(xs) - np.min(xs)) if xs.size else 0.0
        y_span = float(np.max(ys) - np.min(ys)) if ys.size else 0.0
        width = max(3.4 * float(np.std(xs) + 1e-6), x_span + 2.6)
        height = max(58.0, y_span + 32.0)
        ax.add_patch(
            Ellipse(
                (float(np.mean(xs)), float(np.mean(ys))),
                width=width,
                height=height,
                facecolor=fill_color,
                edgecolor=edge_color,
                linewidth=1.2,
                linestyle=(0, (5, 4)),
                alpha=0.12,
                zorder=1.9,
            )
        )
        label_y = float(np.max(ys) + 12.0)
        va = "bottom"
        if direction == "d":
            label_y = float(np.min(ys) - 12.0)
            va = "top"
        ax.text(
            float(np.mean(xs)),
            label_y,
            label,
            ha="center",
            va=va,
            fontsize=8.2,
            color=edge_color,
            zorder=2.0,
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": edge_color, "alpha": 0.72},
        )


def _vehicle_geometry(veh: VehicleSnapshot) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    length = max(float(veh.vehicle_length) * VEHICLE_DRAW_LENGTH_SCALE, 4.8)
    width = max(float(veh.vehicle_width) * VEHICLE_DRAW_WIDTH_SCALE, 1.85)
    center = np.asarray([float(veh.x), float(veh.y)], dtype=np.float64)
    forward = np.asarray([np.cos(float(veh.yaw)), np.sin(float(veh.yaw))], dtype=np.float64)
    lateral = np.asarray([-np.sin(float(veh.yaw)), np.cos(float(veh.yaw))], dtype=np.float64)
    half_length = 0.5 * length
    half_width = 0.5 * width
    corners = np.asarray(
        [
            center + half_length * forward + half_width * lateral,
            center + half_length * forward - half_width * lateral,
            center - half_length * forward - half_width * lateral,
            center - half_length * forward + half_width * lateral,
        ],
        dtype=np.float64,
    )
    return corners, center, forward, lateral, float(length), float(width)


def _local_to_world(center: np.ndarray, forward: np.ndarray, lateral: np.ndarray, local_points: Sequence[Tuple[float, float]]) -> np.ndarray:
    return np.asarray(
        [center + lon * forward + lat * lateral for lon, lat in local_points],
        dtype=np.float64,
    )


def _draw_vehicle_stylized(
    ax: plt.Axes,
    center: np.ndarray,
    forward: np.ndarray,
    lateral: np.ndarray,
    draw_length: float,
    draw_width: float,
    *,
    halo_face,
    halo_edge,
    line_width: float,
    is_hv: bool,
    zorder: float,
) -> None:
    base_length = draw_length
    base_width = draw_width

    body_local = [
        (0.50 * base_length, 0.00 * base_width),
        (0.42 * base_length, 0.42 * base_width),
        (0.10 * base_length, 0.52 * base_width),
        (-0.34 * base_length, 0.50 * base_width),
        (-0.52 * base_length, 0.30 * base_width),
        (-0.56 * base_length, 0.00 * base_width),
        (-0.52 * base_length, -0.30 * base_width),
        (-0.34 * base_length, -0.50 * base_width),
        (0.10 * base_length, -0.52 * base_width),
        (0.42 * base_length, -0.42 * base_width),
    ]
    roof_local = [
        (0.24 * base_length, 0.25 * base_width),
        (-0.05 * base_length, 0.30 * base_width),
        (-0.22 * base_length, 0.20 * base_width),
        (-0.26 * base_length, 0.00 * base_width),
        (-0.22 * base_length, -0.20 * base_width),
        (-0.05 * base_length, -0.30 * base_width),
        (0.24 * base_length, -0.25 * base_width),
        (0.30 * base_length, 0.00 * base_width),
    ]
    windshield_local = [
        (0.40 * base_length, 0.14 * base_width),
        (0.28 * base_length, 0.22 * base_width),
        (0.28 * base_length, -0.22 * base_width),
        (0.40 * base_length, -0.14 * base_width),
    ]
    rear_window_local = [
        (-0.18 * base_length, 0.18 * base_width),
        (-0.34 * base_length, 0.24 * base_width),
        (-0.40 * base_length, 0.00 * base_width),
        (-0.34 * base_length, -0.24 * base_width),
        (-0.18 * base_length, -0.18 * base_width),
    ]

    tire_specs = [
        (-0.10 * base_length, 0.57 * base_width),
        (0.28 * base_length, 0.54 * base_width),
        (-0.10 * base_length, -0.57 * base_width),
        (0.28 * base_length, -0.54 * base_width),
    ]

    body_pts = _local_to_world(center, forward, lateral, body_local)
    roof_pts = _local_to_world(center, forward, lateral, roof_local)
    windshield_pts = _local_to_world(center, forward, lateral, windshield_local)
    rear_window_pts = _local_to_world(center, forward, lateral, rear_window_local)

    body_fill = "#F8F9FA" if not is_hv else "#D3D7DC"
    roof_fill = "#3B4048" if not is_hv else "#6F7781"
    glass_fill = "#DCE9F3" if not is_hv else "#E3E7EC"
    tire_fill = "#1C1F24"

    ax.add_patch(
        Polygon(
            body_pts,
            closed=True,
            facecolor=halo_face,
            edgecolor=halo_edge,
            linewidth=max(1.0, line_width),
            alpha=0.36,
            joinstyle="round",
            zorder=zorder - 0.2,
        )
    )
    ax.add_patch(
        Polygon(
            body_pts,
            closed=True,
            facecolor=body_fill,
            edgecolor=halo_edge,
            linewidth=max(0.9, 0.9 * line_width),
            joinstyle="round",
            zorder=zorder,
        )
    )
    ax.add_patch(
        Polygon(
            roof_pts,
            closed=True,
            facecolor=roof_fill,
            edgecolor="#1F1F1F",
            linewidth=0.55,
            joinstyle="round",
            zorder=zorder + 0.05,
        )
    )
    ax.add_patch(
        Polygon(
            windshield_pts,
            closed=True,
            facecolor=glass_fill,
            edgecolor="#5F6976",
            linewidth=0.42,
            joinstyle="round",
            zorder=zorder + 0.06,
        )
    )
    ax.add_patch(
        Polygon(
            rear_window_pts,
            closed=True,
            facecolor=glass_fill,
            edgecolor="#5F6976",
            linewidth=0.42,
            joinstyle="round",
            zorder=zorder + 0.06,
        )
    )

    for lon, lat in tire_specs:
        tire_center = center + lon * forward + lat * lateral
        tire_half_lon = 0.12 * base_length
        tire_half_lat = 0.06 * base_width
        tire_pts = _local_to_world(
            tire_center,
            forward,
            lateral,
            [
                (tire_half_lon, tire_half_lat),
                (tire_half_lon, -tire_half_lat),
                (-tire_half_lon, -tire_half_lat),
                (-tire_half_lon, tire_half_lat),
            ],
        )
        ax.add_patch(
            Polygon(
                tire_pts,
                closed=True,
                facecolor=tire_fill,
                edgecolor=tire_fill,
                linewidth=0.2,
                zorder=zorder + 0.04,
            )
        )


def _draw_vehicles(ax: plt.Axes, frame: FrameSnapshot, *, show_ids: bool = False, show_gaps: bool = False) -> None:
    gap_map = _front_gap_map(frame) if show_gaps else {}
    for veh in frame.vehicles:
        face_color, edge_color, line_width = _vehicle_style(frame, veh)
        _, center, forward, lateral, _, _ = _vehicle_geometry(veh)
        marker = "^" if veh.direction == "u" else "v"
        size = 58 if veh.is_cav else 44

        if veh.is_leader:
            ax.scatter(
                [center[0]],
                [center[1]],
                s=260,
                marker="o",
                facecolors="none",
                edgecolors=edge_color,
                linewidths=1.2,
                alpha=0.22,
                zorder=4.6,
            )

        ax.scatter(
            [center[0]],
            [center[1]],
            s=size,
            marker=marker,
            c=[face_color],
            edgecolors=[edge_color],
            linewidths=max(0.9, 0.9 * line_width),
            alpha=0.95 if veh.is_cav else 0.88,
            zorder=5.2,
        )

        if veh.is_leader:
            star_pos = center + 0.95 * forward + 0.10 * lateral
            ax.scatter(
                [star_pos[0]],
                [star_pos[1]],
                s=86,
                marker="*",
                c="#FFD166",
                edgecolors=LEADER_EDGE,
                linewidths=0.6,
                zorder=6.1,
            )
        if show_ids:
            label = f"#{veh.idx}"
            if show_gaps and veh.idx in gap_map:
                label = f"#{veh.idx}\nd={gap_map[veh.idx]:.1f}m"
            label_anchor = center + (1.05 + 0.10 * float(veh.lane_idx)) * lateral - 0.25 * forward
            ha = "left" if lateral[0] >= 0.0 else "right"
            if veh.direction == "d":
                label_anchor = label_anchor - 0.55 * forward
            ax.text(
                label_anchor[0],
                label_anchor[1],
                label,
                fontsize=6.0,
                color=TEXT_COLOR,
                ha=ha,
                va="center",
                zorder=6,
                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "#D5DADF", "alpha": 0.82},
            )


def _add_info_box(ax: plt.Axes, frame: FrameSnapshot) -> None:
    comm_v2v = frame.state_fields.get("comm_v2v_success", 0.0)
    min_gap = min(frame.state_fields.get("min_gap_up", 0.0), frame.state_fields.get("min_gap_down", 0.0))
    mean_speed = 0.5 * (frame.state_fields.get("mean_speed_up", 0.0) + frame.state_fields.get("mean_speed_down", 0.0))
    platoon_rate = frame.state_fields.get("platoon_rate", 0.0)
    mpr_cav = frame.state_fields.get("mpr_cav", 0.0)
    summary = (
        f"step={frame.step} | action={frame.action}\n"
        f"reward={frame.reward:.3f} | topology={frame.topology_type}\n"
        f"mean speed={mean_speed:.2f} m/s | min gap={min_gap:.2f} m\n"
        f"MPR(CAV)={mpr_cav:.2f} | platoon rate={platoon_rate:.2f}\n"
        f"V2V success={comm_v2v:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=TEXT_COLOR,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#AAB2BD", "alpha": 0.92},
        zorder=7,
    )


def draw_scene(
    ax: plt.Axes,
    history: SimulationHistory,
    frame: FrameSnapshot,
    *,
    show_v2i: bool = True,
    show_ids: bool = False,
    show_gaps: bool = False,
    show_neighbors: bool = False,
    show_groups: bool = True,
    annotate_box: bool = True,
    title: Optional[str] = None,
) -> None:
    """Draw a single scene frame."""

    _draw_static_road(ax, history, show_titles=False)
    _draw_segment_boundaries(ax, history)
    _draw_event_regions(ax, history, active_step=frame.step)
    _draw_base_stations(ax, history)
    if show_v2i:
        _draw_v2i_links(ax, history, frame)
    if show_neighbors:
        _draw_neighbor_links(ax, frame)
    if show_groups:
        _draw_group_overlays(ax, frame)
    _draw_vehicles(ax, frame, show_ids=show_ids, show_gaps=show_gaps)
    if annotate_box:
        _add_info_box(ax, frame)

    x_min, x_max = _road_x_bounds(history)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, history.height)
    ax.set_xlabel("Lateral position (m)")
    ax.set_ylabel("Longitudinal position (m)")
    ax.grid(alpha=0.10, linestyle="--", axis="y")
    if title:
        ax.set_title(title, fontsize=11)


def _key_frame_indices(history: SimulationHistory) -> List[int]:
    if not history.frames:
        return []
    initial_idx = 0
    active = [idx for idx, frame in enumerate(history.frames) if frame.state_fields.get("active_event", 0.0) > 0.5]
    if active:
        event_idx = active[len(active) // 2]
    else:
        event_idx = min(len(history.frames) - 1, max(0, len(history.frames) // 2))
    final_idx = len(history.frames) - 1
    deduped = []
    for idx in [initial_idx, event_idx, final_idx]:
        if idx not in deduped:
            deduped.append(idx)
    return deduped


def _scene_legend_handles(history: SimulationHistory) -> List[object]:
    handles = [
        Line2D([0], [0], marker="^", color="w", label="Up-direction vehicle", markerfacecolor=SPEED_CMAP(0.25), markeredgecolor=COMM_CMAP(0.8), markeredgewidth=1.2, markersize=8),
        Line2D([0], [0], marker="v", color="w", label="Down-direction vehicle", markerfacecolor=SPEED_CMAP(0.65), markeredgecolor=COMM_CMAP(0.8), markeredgewidth=1.2, markersize=8),
        Line2D([0], [0], marker="*", color="w", label="Leader highlight", markerfacecolor="#FFD166", markeredgecolor=LEADER_EDGE, markersize=10),
        Rectangle((0, 0), 1, 1, facecolor=EVENT_ZONE, edgecolor="#D88D2F", alpha=0.30, label="Incident region"),
        Rectangle((0, 0), 1, 1, facecolor=BLOCKED_ZONE, edgecolor="#B85C00", alpha=0.28, hatch="////", label="Blocked lane"),
        Rectangle((0, 0), 1, 1, facecolor=SHOULDER_COLOR, edgecolor="#A9AFB8", alpha=0.95, label="Shoulder / emergency lane"),
    ]
    if history.bs_positions:
        handles.append(Line2D([0], [0], marker="s", color="w", label="RSU / BS", markerfacecolor=BS_COLOR, markeredgecolor="#1E4028", markersize=8))
    return handles


def _render_scene_sequence(
    history: SimulationHistory,
    frames: Sequence[FrameSnapshot],
    titles: Sequence[str],
    out_base: Path,
    *,
    show_v2i: bool = True,
    show_ids: bool = True,
    show_gaps: bool = True,
    show_groups: bool = True,
    annotate_box: bool = True,
    show_speedbar: bool = True,
    show_legend: bool = True,
) -> Path:
    fig, axes = plt.subplots(1, len(frames), figsize=(4.65 * len(frames), 5.35), facecolor="white")
    if len(frames) == 1:
        axes = [axes]
    for panel_idx, (ax, frame, title) in enumerate(zip(axes, frames, titles)):
        draw_scene(
            ax,
            history,
            frame,
            show_v2i=show_v2i,
            show_ids=show_ids,
            show_gaps=show_gaps,
            show_neighbors=False,
            show_groups=show_groups,
            annotate_box=annotate_box,
            title=title,
        )
        ax.set_xlabel("Lateral position (m)", fontsize=8.5)
        ax.set_ylabel("Longitudinal position (m)" if panel_idx == 0 else "", fontsize=8.5)
        ax.tick_params(axis="both", labelsize=8)
        ax.title.set_fontsize(10.5)
    if show_speedbar:
        speed_bar = fig.colorbar(
            ScalarMappable(norm=SPEED_NORM, cmap=SPEED_CMAP),
            ax=list(axes),
            fraction=0.024,
            pad=0.02,
        )
        speed_bar.set_label("Vehicle speed (m/s)", fontsize=9)
        speed_bar.ax.tick_params(labelsize=8)
    if show_legend:
        fig.legend(
            handles=_scene_legend_handles(history),
            loc="lower center",
            ncol=min(6, len(history.bs_positions) + 5),
            frameon=True,
            fontsize=8.2,
            bbox_to_anchor=(0.5, -0.015),
        )
    bottom = 0.12 if show_legend else 0.08
    right = 0.925 if show_speedbar else 0.985
    fig.subplots_adjust(left=0.06, right=right, top=0.91, bottom=bottom, wspace=0.16)
    _save_png_pdf(fig, out_base)
    return out_base.with_suffix(".png")


def save_scene_triptych(
    history: SimulationHistory,
    out_dir: str | Path,
    *,
    show_v2i: bool = True,
    show_ids: bool = False,
    show_gaps: bool = False,
    annotate_box: bool = False,
    show_groups: bool = False,
    show_speedbar: bool = False,
    show_legend: bool = False,
    filename: str = "scene_triptych",
) -> Path:
    """Export a three-panel scene summary: initial, incident phase, final."""

    out_dir = Path(out_dir)
    indices = _key_frame_indices(history)
    frames = [history.frames[idx] for idx in indices]
    titles = []
    for idx, frame in zip(indices, frames):
        if idx == 0:
            titles.append("Initial platoon layout")
        elif idx == len(history.frames) - 1:
            titles.append("Final platoon layout")
        else:
            titles.append("Incident-phase reconfiguration")

    out_base = out_dir / filename
    return _render_scene_sequence(
        history,
        frames,
        titles,
        out_base,
        show_v2i=show_v2i,
        show_ids=show_ids,
        show_gaps=show_gaps,
        show_groups=show_groups,
        annotate_box=annotate_box,
        show_speedbar=show_speedbar,
        show_legend=show_legend,
    )


def save_scene_compare(history: SimulationHistory, out_dir: str | Path, *, show_v2i: bool = True) -> Path:
    """Export a concise compare figure with initial, mid, and final scenes."""

    out_dir = Path(out_dir)
    if not history.frames:
        raise ValueError("Simulation history is empty.")

    indices = [0, len(history.frames) // 2, len(history.frames) - 1]
    frames: List[FrameSnapshot] = []
    titles: List[str] = []
    seen = set()
    for idx in indices:
        if idx in seen or idx < 0 or idx >= len(history.frames):
            continue
        seen.add(idx)
        frames.append(history.frames[idx])
        if idx == 0:
            titles.append("Initial")
        elif idx == len(history.frames) - 1:
            titles.append("Final")
        else:
            titles.append("Mid / Event")

    out_base = out_dir / "scene_compare"
    return _render_scene_sequence(
        history,
        frames,
        titles,
        out_base,
        show_v2i=False,
        show_ids=False,
        show_gaps=False,
        annotate_box=True,
    )


def _figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()
    plt.close(fig)
    return image


class AnimationCanvas:
    """Reusable Matplotlib canvas for incremental animation rendering."""

    def __init__(self, history: SimulationHistory, *, show_v2i: bool = True):
        self.history = history
        self.show_v2i = bool(show_v2i)
        self.fig, self.ax = plt.subplots(figsize=(6.0, 8.8), facecolor="white")
        self._configure_axes()

    def _configure_axes(self) -> None:
        x_min, x_max = _road_x_bounds(self.history)
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(0.0, self.history.height)
        self.ax.set_xlabel("Lateral position (m)")
        self.ax.set_ylabel("Longitudinal position (m)")
        self.ax.grid(alpha=0.10, linestyle="--", axis="y")
        self.ax.set_title("Platoon reconfiguration evolution", fontsize=11)

    def update(self, frame: FrameSnapshot, *, show_ids: bool = False, show_neighbors: bool = False, annotate_box: bool = True) -> plt.Figure:
        self.ax.clear()
        self._configure_axes()
        draw_scene(
            self.ax,
            self.history,
            frame,
            show_v2i=self.show_v2i,
            show_ids=show_ids,
            show_gaps=False,
            show_neighbors=show_neighbors,
            show_groups=True,
            annotate_box=annotate_box,
            title="Platoon reconfiguration evolution",
        )
        return self.fig


def _animation_frame_figure(
    history: SimulationHistory,
    frame: FrameSnapshot,
    *,
    show_v2i: bool = True,
    canvas: Optional[AnimationCanvas] = None,
    show_ids: bool = False,
    show_neighbors: bool = False,
    annotate_box: bool = True,
) -> plt.Figure:
    canvas = canvas or AnimationCanvas(history, show_v2i=show_v2i)
    return canvas.update(frame, show_ids=show_ids, show_neighbors=show_neighbors, annotate_box=annotate_box)


def export_animation(
    history: SimulationHistory,
    out_dir: str | Path,
    *,
    show_v2i: bool = True,
    frame_stride: int = 2,
    gif_fps: int = 6,
    mp4_fps: int = 8,
    write_gif: bool = True,
    write_mp4: bool = True,
) -> Dict[str, str]:
    """Export GIF/MP4 animation from the recorded simulation history."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sampled_frames = history.frames[:: max(1, int(frame_stride))]
    if not sampled_frames:
        return {}
    if sampled_frames[-1] is not history.frames[-1]:
        sampled_frames.append(history.frames[-1])

    outputs: Dict[str, str] = {}
    gif_path = out_dir / "formation_evolution.gif"
    mp4_path = out_dir / "formation_evolution.mp4"

    gif_writer = imageio.get_writer(str(gif_path), mode="I", fps=max(1, int(gif_fps))) if write_gif else None
    mp4_writer = (
        imageio.get_writer(
            str(mp4_path),
            fps=max(1, int(mp4_fps)),
            codec="libx264",
            quality=7,
            macro_block_size=1,
        )
        if write_mp4
        else None
    )

    canvas = AnimationCanvas(history, show_v2i=show_v2i)
    try:
        for frame in sampled_frames:
            fig = _animation_frame_figure(history, frame, show_v2i=show_v2i, canvas=canvas)
            image = _figure_to_rgb(fig)
            if gif_writer is not None:
                gif_writer.append_data(image)
            if mp4_writer is not None:
                mp4_writer.append_data(image)
    finally:
        if gif_writer is not None:
            gif_writer.close()
            outputs["gif"] = str(gif_path)
        if mp4_writer is not None:
            mp4_writer.close()
            outputs["mp4"] = str(mp4_path)

    return outputs
