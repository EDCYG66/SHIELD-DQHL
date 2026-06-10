"""Traffic event definitions for emergency bottleneck and incident experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


DirectionMap = Mapping[str, Sequence[int]]


@dataclass(frozen=True)
class TrafficEvent:
    """A longitudinal event region that alters lane availability and speed."""

    name: str
    start_step: int
    end_step: int
    y_start: float
    y_end: float
    speed_limit_ratio: float = 0.45
    severity: float = 1.0
    affected_directions: Tuple[str, ...] = ("u", "d")
    blocked_lanes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    phase_tag: str = "incident"
    advisory_only: bool = False

    def active(self, step: int) -> bool:
        return self.start_step <= int(step) <= self.end_step

    def contains_y(self, y: float, margin: float = 0.0) -> bool:
        return (self.y_start - margin) <= float(y) <= (self.y_end + margin)

    def applies_to(self, direction: str, step: int) -> bool:
        return self.active(step) and direction in self.affected_directions

    def blocks_lane(self, direction: str, lane_idx: int, y: float, step: int, margin: float = 0.0) -> bool:
        if not self.applies_to(direction, step):
            return False
        if not self.contains_y(y, margin=margin):
            return False
        if self.advisory_only:
            return False
        blocked = self.blocked_lanes.get(direction, ())
        return int(lane_idx) in blocked

    def speed_scale(self, direction: str, y: float, step: int, margin: float = 0.0) -> float:
        if not self.applies_to(direction, step):
            return 1.0
        if not self.contains_y(y, margin=margin):
            return 1.0
        return float(self.speed_limit_ratio)


class EventScheduler:
    """Keeps track of active events and provides query helpers."""

    def __init__(self, events: Iterable[TrafficEvent] | None = None):
        self.events: List[TrafficEvent] = list(events or [])

    def add_event(self, event: TrafficEvent) -> None:
        self.events.append(event)

    def active_events(self, step: int) -> List[TrafficEvent]:
        return [event for event in self.events if event.active(step)]

    def has_active_event(self, step: int) -> bool:
        return any(event.active(step) for event in self.events)

    def lane_blocked(self, direction: str, lane_idx: int, y: float, step: int, margin: float = 0.0) -> bool:
        return any(
            event.blocks_lane(direction, lane_idx, y, step, margin=margin)
            for event in self.events
        )

    def speed_scale(self, direction: str, lane_idx: int, y: float, step: int, margin: float = 0.0) -> float:
        scale = 1.0
        for event in self.events:
            if event.blocks_lane(direction, lane_idx, y, step, margin=margin):
                scale = min(scale, event.speed_limit_ratio)
            else:
                scale = min(scale, event.speed_scale(direction, y, step, margin=margin))
        return float(scale)

    def blocked_lanes(self, direction: str, step: int, y: float | None = None) -> Tuple[int, ...]:
        lanes = set()
        for event in self.events:
            if not event.applies_to(direction, step):
                continue
            if y is not None and not event.contains_y(y, margin=25.0):
                continue
            lanes.update(int(lane) for lane in event.blocked_lanes.get(direction, ()))
        return tuple(sorted(lanes))

    def nearest_event_distance(self, direction: str, y: float, step: int) -> float | None:
        distances: List[float] = []
        for event in self.active_events(step):
            if direction not in event.affected_directions:
                continue
            if event.contains_y(y):
                distances.append(0.0)
                continue
            if direction == "u" and y < event.y_start:
                distances.append(event.y_start - y)
            elif direction == "d" and y > event.y_end:
                distances.append(y - event.y_end)
        if not distances:
            return None
        return float(min(distances))

    def event_pressure(self, direction: str, y: float, step: int) -> float:
        pressure = 0.0
        for event in self.active_events(step):
            if direction not in event.affected_directions:
                continue
            distance = 0.0
            if y < event.y_start:
                distance = event.y_start - y
            elif y > event.y_end:
                distance = y - event.y_end
            scale = max(0.0, 1.0 - distance / max(1.0, event.y_end - event.y_start))
            pressure = max(pressure, event.severity * scale)
        return float(pressure)

    @property
    def horizon_end(self) -> int:
        if not self.events:
            return 0
        return max(event.end_step for event in self.events)


def _normalize_lane_map(blocked_lanes: DirectionMap | None) -> Dict[str, Tuple[int, ...]]:
    result: Dict[str, Tuple[int, ...]] = {}
    for direction, lanes in (blocked_lanes or {}).items():
        result[direction] = tuple(sorted(int(lane) for lane in lanes))
    return result


def build_bottleneck_event(
    *,
    start_step: int,
    duration: int,
    y_center: float,
    zone_length: float = 140.0,
    speed_limit_ratio: float = 0.40,
    severity: float = 1.0,
    blocked_lanes: DirectionMap | None = None,
    affected_directions: Sequence[str] = ("u", "d"),
    name: str = "bottleneck",
) -> TrafficEvent:
    half = float(zone_length) / 2.0
    return TrafficEvent(
        name=name,
        start_step=int(start_step),
        end_step=int(start_step + duration),
        y_start=float(y_center - half),
        y_end=float(y_center + half),
        speed_limit_ratio=float(speed_limit_ratio),
        severity=float(severity),
        affected_directions=tuple(affected_directions),
        blocked_lanes=_normalize_lane_map(blocked_lanes),
        phase_tag="incident",
        advisory_only=False,
    )


def build_staged_incident_events(
    *,
    start_step: int,
    duration: int,
    y_center: float,
    zone_length: float = 160.0,
    speed_limit_ratio: float = 0.40,
    blocked_lanes: DirectionMap | None = None,
    affected_directions: Sequence[str] = ("u", "d"),
    warning_length_scale: float = 1.35,
    recovery_length_scale: float = 1.10,
) -> List[TrafficEvent]:
    """Build a three-phase warning/incident/recovery event chain."""

    blocked = _normalize_lane_map(blocked_lanes)
    duration = max(12, int(duration))
    main_start = int(start_step)
    main_end = int(start_step + duration)
    warning_span = max(8, int(round(0.28 * duration)))
    recovery_span = max(10, int(round(0.25 * duration)))
    half = 0.5 * float(zone_length)
    warning_half = 0.5 * float(zone_length) * float(warning_length_scale)
    recovery_half = 0.5 * float(zone_length) * float(recovery_length_scale)

    return [
        TrafficEvent(
            name="pre_warning",
            start_step=max(0, main_start - warning_span),
            end_step=main_start - 1,
            y_start=float(y_center - warning_half),
            y_end=float(y_center + warning_half),
            speed_limit_ratio=min(0.92, 0.72 + 0.25 * float(speed_limit_ratio)),
            severity=0.45,
            affected_directions=tuple(affected_directions),
            blocked_lanes={},
            phase_tag="warning",
            advisory_only=True,
        ),
        TrafficEvent(
            name="main_incident",
            start_step=main_start,
            end_step=main_end,
            y_start=float(y_center - half),
            y_end=float(y_center + half),
            speed_limit_ratio=float(speed_limit_ratio),
            severity=1.0,
            affected_directions=tuple(affected_directions),
            blocked_lanes=blocked,
            phase_tag="incident",
            advisory_only=False,
        ),
        TrafficEvent(
            name="recovery_zone",
            start_step=main_end + 1,
            end_step=main_end + recovery_span,
            y_start=float(y_center - recovery_half),
            y_end=float(y_center + recovery_half),
            speed_limit_ratio=min(0.95, 0.84 + 0.18 * float(speed_limit_ratio)),
            severity=0.35,
            affected_directions=tuple(affected_directions),
            blocked_lanes={},
            phase_tag="recovery",
            advisory_only=True,
        ),
    ]
