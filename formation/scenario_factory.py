"""Scenario definitions for batch platoon experiments and road-segment layouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .events import EventScheduler, TrafficEvent, build_bottleneck_event, build_staged_incident_events
except ImportError:  # pragma: no cover
    from events import EventScheduler, TrafficEvent, build_bottleneck_event, build_staged_incident_events


@dataclass(frozen=True)
class RoadSegment:
    """A longitudinal highway segment with local geometry and event semantics."""

    name: str
    start_y: float
    end_y: float
    lane_count: int
    speed_limit_ratio: float = 1.0
    blocked_lanes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    ramp_in: bool = False
    ramp_out: bool = False
    merge_zone: bool = False
    split_zone: bool = False
    lane_drop_to: Optional[int] = None
    description: str = ""

    @property
    def length(self) -> float:
        return float(self.end_y - self.start_y)

    def midpoint(self) -> float:
        return 0.5 * (float(self.start_y) + float(self.end_y))


@dataclass(frozen=True)
class HighwayScenarioSpec:
    """Scenario spec composed from multiple road segments and derived events."""

    name: str
    segments: Sequence[RoadSegment]
    event_start: int
    event_duration: int
    event_center: float
    event_length: float
    event_speed_scale: float
    blocked_lanes: Dict[str, Tuple[int, ...]]
    description: str = ""
    events: Sequence[TrafficEvent] = ()


# Backward-compatible alias used by the existing scripts.
ScenarioSpec = HighwayScenarioSpec


def _normalize_blocked_lanes(blocked_lanes: Dict[str, Sequence[int]] | None) -> Dict[str, Tuple[int, ...]]:
    return {direction: tuple(sorted(int(lane) for lane in lanes)) for direction, lanes in (blocked_lanes or {}).items()}


def _merge_segment_blocked_lanes(segments: Sequence[RoadSegment]) -> Dict[str, Tuple[int, ...]]:
    merged: Dict[str, set[int]] = {}
    for segment in segments:
        for direction, lanes in segment.blocked_lanes.items():
            merged.setdefault(direction, set()).update(int(lane) for lane in lanes)
    return {direction: tuple(sorted(lanes)) for direction, lanes in merged.items()}


def build_scheduler_from_spec(spec: HighwayScenarioSpec) -> EventScheduler:
    if getattr(spec, "events", ()):
        return EventScheduler(list(spec.events))
    events: List[TrafficEvent] = []
    for seg in spec.segments:
        if seg.speed_limit_ratio < 1.0 or seg.blocked_lanes:
            events.append(
                TrafficEvent(
                    name=seg.name,
                    start_step=spec.event_start,
                    end_step=int(spec.event_start + spec.event_duration),
                    y_start=float(seg.start_y),
                    y_end=float(seg.end_y),
                    speed_limit_ratio=float(seg.speed_limit_ratio),
                    severity=1.0,
                    affected_directions=("u", "d"),
                    blocked_lanes={k: tuple(v) for k, v in seg.blocked_lanes.items()},
                )
            )
    if not events:
        events = [
            build_bottleneck_event(
                start_step=spec.event_start,
                duration=spec.event_duration,
                y_center=spec.event_center,
                zone_length=spec.event_length,
                speed_limit_ratio=spec.event_speed_scale,
                blocked_lanes=spec.blocked_lanes,
            )
        ]
    return EventScheduler(events)


def build_segment_adjustments(spec: HighwayScenarioSpec) -> Dict[str, object]:
    """Return environment hints derived from RoadSegment layout."""

    lane_drop = None
    for seg in spec.segments:
        if seg.lane_drop_to is not None:
            lane_drop = int(seg.lane_drop_to)
            break
    segments = [
        {
            "name": seg.name,
            "start_y": float(seg.start_y),
            "end_y": float(seg.end_y),
            "lane_count": int(seg.lane_count),
            "speed_limit_ratio": float(seg.speed_limit_ratio),
            "ramp_in": bool(seg.ramp_in),
            "ramp_out": bool(seg.ramp_out),
            "merge_zone": bool(seg.merge_zone),
            "split_zone": bool(seg.split_zone),
            "lane_drop_to": int(seg.lane_drop_to) if seg.lane_drop_to is not None else None,
            "blocked_lanes": {k: list(v) for k, v in seg.blocked_lanes.items()},
            "description": seg.description,
        }
        for seg in spec.segments
    ]
    return {
        "segments": segments,
        "lane_drop_to": lane_drop,
        "lane_count_sequence": [int(seg.lane_count) for seg in spec.segments],
        "has_ramp_in": any(seg.ramp_in for seg in spec.segments),
        "has_ramp_out": any(seg.ramp_out for seg in spec.segments),
        "has_weaving_zone": any(seg.merge_zone and seg.split_zone for seg in spec.segments),
    }


def _highway_segments(height: float) -> List[RoadSegment]:
    h = float(height)
    return [
        RoadSegment(
            name="cruise_entry",
            start_y=0.0,
            end_y=0.18 * h,
            lane_count=4,
            speed_limit_ratio=1.0,
            description="Stable cruising and platoon formation",
        ),
        RoadSegment(
            name="ramp_merge",
            start_y=0.18 * h,
            end_y=0.34 * h,
            lane_count=4,
            speed_limit_ratio=0.95,
            ramp_in=True,
            merge_zone=True,
            description="On-ramp merging and pre-reconfiguration",
        ),
        RoadSegment(
            name="lane_drop_pre",
            start_y=0.34 * h,
            end_y=0.46 * h,
            lane_count=4,
            speed_limit_ratio=0.90,
            split_zone=True,
            lane_drop_to=3,
            description="Advance lane-drop warning and pre-positioning",
        ),
        RoadSegment(
            name="bottleneck",
            start_y=0.46 * h,
            end_y=0.60 * h,
            lane_count=3,
            speed_limit_ratio=0.42,
            blocked_lanes={"u": (1,), "d": (1,)},
            split_zone=True,
            lane_drop_to=3,
            description="Incident or lane closure bottleneck",
        ),
        RoadSegment(
            name="weaving_zone",
            start_y=0.60 * h,
            end_y=0.80 * h,
            lane_count=3,
            speed_limit_ratio=0.86,
            ramp_out=True,
            merge_zone=True,
            split_zone=True,
            description="Weaving section with frequent lane changes",
        ),
        RoadSegment(
            name="recovery",
            start_y=0.80 * h,
            end_y=h,
            lane_count=4,
            speed_limit_ratio=1.0,
            ramp_out=True,
            description="Recovery and re-stabilization",
        ),
    ]


def highway_scenario_spec(height: float = 1400.0) -> HighwayScenarioSpec:
    segments = _highway_segments(height)
    bottleneck = next(seg for seg in segments if seg.name == "bottleneck")
    blocked = _merge_segment_blocked_lanes(segments)
    return HighwayScenarioSpec(
        name="highway_mixed_flow",
        segments=segments,
        event_start=40,
        event_duration=90,
        event_center=bottleneck.midpoint(),
        event_length=float(bottleneck.length),
        event_speed_scale=float(bottleneck.speed_limit_ratio),
        blocked_lanes=blocked,
        description="Realistic highway with merge, lane-drop, bottleneck, weaving, and recovery sections",
    )


def build_staged_incident_spec(
    name: str,
    *,
    height: float = 1400.0,
    event_start: int = 60,
    event_duration: int = 110,
    event_center: Optional[float] = None,
    event_length: float = 180.0,
    event_speed_scale: float = 0.36,
    blocked_lanes: Optional[Dict[str, Sequence[int]]] = None,
    description: str = "",
) -> HighwayScenarioSpec:
    segments = _highway_segments(height)
    if event_center is None:
        event_center = 0.5 * float(height)
    blocked_map = _normalize_blocked_lanes(blocked_lanes or {"u": (1,), "d": (1,)})
    events = build_staged_incident_events(
        start_step=event_start,
        duration=event_duration,
        y_center=float(event_center),
        zone_length=event_length,
        speed_limit_ratio=event_speed_scale,
        blocked_lanes=blocked_map,
    )
    return HighwayScenarioSpec(
        name=name,
        segments=segments,
        event_start=int(event_start),
        event_duration=int(event_duration),
        event_center=float(event_center),
        event_length=float(event_length),
        event_speed_scale=float(event_speed_scale),
        blocked_lanes=blocked_map,
        description=description or "Three-phase staged incident with warning, main blockage, and recovery.",
        events=tuple(events),
    )


def default_scenarios(height: float = 1400.0) -> List[HighwayScenarioSpec]:
    spec = highway_scenario_spec(height=height)
    center_mid = float(height) * 0.50
    center_up = float(height) * 0.35
    center_down = float(height) * 0.65
    return [
        spec,
        HighwayScenarioSpec("mild_mid", spec.segments, 40, 70, center_mid, 120.0, 0.55, {"u": (1,), "d": (1,)}, description="Mild mid-road bottleneck"),
        HighwayScenarioSpec("moderate_mid", spec.segments, 40, 90, center_mid, 160.0, 0.40, {"u": (1,), "d": (1,)}, description="Moderate mid-road bottleneck"),
        HighwayScenarioSpec("severe_mid", spec.segments, 35, 110, center_mid, 190.0, 0.28, {"u": (1, 2), "d": (1, 2)}, description="Severe multi-lane bottleneck"),
        HighwayScenarioSpec("upstream_bottleneck", spec.segments, 45, 80, center_up, 150.0, 0.38, {"u": (1,), "d": (2,)}, description="Upstream-dominant bottleneck"),
        HighwayScenarioSpec("downstream_bottleneck", spec.segments, 45, 80, center_down, 150.0, 0.38, {"u": (2,), "d": (1,)}, description="Downstream-dominant bottleneck"),
        build_staged_incident_spec(
            "staged_mid",
            height=height,
            event_start=55,
            event_duration=110,
            event_center=center_mid,
            event_length=180.0,
            event_speed_scale=0.36,
            blocked_lanes={"u": (1,), "d": (1,)},
            description="Three-phase staged incident with symmetric single-lane closure.",
        ),
        build_staged_incident_spec(
            "staged_severe",
            height=height,
            event_start=50,
            event_duration=130,
            event_center=center_mid,
            event_length=210.0,
            event_speed_scale=0.26,
            blocked_lanes={"u": (1, 2), "d": (1,)},
            description="Three-phase staged incident with asymmetric multi-lane disruption.",
        ),
    ]


def select_scenarios(names: Iterable[str], *, height: float = 1400.0) -> List[HighwayScenarioSpec]:
    all_specs = {spec.name: spec for spec in default_scenarios(height=height)}
    selected = []
    for name in names:
        key = str(name).strip()
        if not key:
            continue
        if key not in all_specs:
            raise ValueError(f"Unknown scenario name: {key}")
        selected.append(all_specs[key])
    return selected
