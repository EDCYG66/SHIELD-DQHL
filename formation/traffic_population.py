"""Traffic population generation utilities for mixed CAV/HV microsimulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class VehicleProfile:
    """Lightweight vehicle/driver attribute bundle used by the simulator."""

    veh_type: str
    is_cav: bool
    driver_style: str
    desired_speed: float
    desired_headway: float
    desired_standstill_gap: float
    comfortable_brake: float
    politeness: float
    vehicle_length: float
    accel_limit: float
    lane_change_cooldown_s: float
    mobil_accel_threshold: float
    mobil_safe_brake: float
    mobil_right_bias: float
    perception_head_m: float
    perception_tail_m: float


@dataclass(frozen=True)
class TrafficCompositionConfig:
    """Mixed-traffic composition and behavior priors."""

    mpr_cav: float = 0.50
    exact_cav_count: int = -1
    hv_conservative_ratio: float = 0.30
    hv_aggressive_ratio: float = 0.20
    cav_desired_speed_mean: float = 24.0
    cav_desired_speed_std: float = 1.8
    hv_regular_speed_mean: float = 15.4
    hv_regular_speed_std: float = 1.1
    hv_conservative_speed_mean: float = 13.8
    hv_conservative_speed_std: float = 0.9
    hv_aggressive_speed_mean: float = 17.0
    hv_aggressive_speed_std: float = 1.2
    truck_ratio: float = 0.10


@dataclass(frozen=True)
class SpawnConfig:
    """Spatial initialization options for synthetic highway traffic."""

    random_spawn: bool = False
    spawn_y_min: float = 0.0
    spawn_y_max: float = 260.0
    spacing: float = 20.0
    lane_density_jitter: float = 0.35
    min_spawn_gap: float = 11.0


def even_split(total: int, parts: int) -> List[int]:
    """Split an integer total into nearly equal integer buckets."""

    total = max(0, int(total))
    parts = max(1, int(parts))
    q, r = divmod(total, parts)
    return [q + (1 if idx < r else 0) for idx in range(parts)]


def sample_vehicle_types(total: int, config: TrafficCompositionConfig, rng: np.random.Generator) -> List[bool]:
    """Sample whether each generated vehicle is a CAV."""

    total = max(0, int(total))
    requested_exact = int(getattr(config, "exact_cav_count", -1))
    if requested_exact >= 0:
        ncav = min(total, max(0, requested_exact))
    else:
        ncav = int(round(float(np.clip(config.mpr_cav, 0.0, 1.0)) * total))
    tags = np.zeros(total, dtype=bool)
    tags[:ncav] = True
    if total > 1:
        rng.shuffle(tags)
    return [bool(value) for value in tags.tolist()]


def _sample_hv_style(config: TrafficCompositionConfig, rng: np.random.Generator) -> str:
    conservative = float(np.clip(config.hv_conservative_ratio, 0.0, 1.0))
    aggressive = float(np.clip(config.hv_aggressive_ratio, 0.0, 1.0))
    regular = max(0.0, 1.0 - conservative - aggressive)
    probs = np.asarray([conservative, regular, aggressive], dtype=float)
    if float(np.sum(probs)) <= 1e-8:
        probs = np.asarray([0.3, 0.5, 0.2], dtype=float)
    probs = probs / np.sum(probs)
    return str(rng.choice(["conservative", "regular", "aggressive"], p=probs))


def build_vehicle_profile(is_cav: bool, config: TrafficCompositionConfig, rng: np.random.Generator) -> VehicleProfile:
    """Sample a vehicle profile from the configured traffic composition."""

    truck_ratio = float(np.clip(config.truck_ratio, 0.0, 0.6))
    is_truck = bool(rng.random() < truck_ratio)
    vehicle_length = 10.5 if is_truck else 4.8

    if is_cav:
        desired_speed = max(12.0, float(rng.normal(config.cav_desired_speed_mean, config.cav_desired_speed_std)))
        return VehicleProfile(
            veh_type="cav",
            is_cav=True,
            driver_style="cooperative",
            desired_speed=desired_speed,
            desired_headway=1.00 if not is_truck else 1.15,
            desired_standstill_gap=5.5 if not is_truck else 6.0,
            comfortable_brake=2.8 if not is_truck else 2.4,
            politeness=0.30,
            vehicle_length=vehicle_length,
            accel_limit=2.4 if not is_truck else 1.8,
            lane_change_cooldown_s=5.0,
            mobil_accel_threshold=0.20,
            mobil_safe_brake=0.80,
            mobil_right_bias=0.20,
            perception_head_m=100.0,
            perception_tail_m=100.0,
        )

    style = _sample_hv_style(config, rng)
    if style == "conservative":
        desired_speed = max(10.0, float(rng.normal(config.hv_conservative_speed_mean, config.hv_conservative_speed_std)))
        headway = 1.18
        standstill_gap = 6.4
        comfortable_brake = 3.10
        politeness = 0.12
        accel_limit = 1.36
        cooldown = 8.0
    elif style == "aggressive":
        desired_speed = max(12.0, float(rng.normal(config.hv_aggressive_speed_mean, config.hv_aggressive_speed_std)))
        headway = 0.92
        standstill_gap = 5.6
        comfortable_brake = 3.35
        politeness = 0.07
        accel_limit = 1.72
        cooldown = 8.0
    else:
        desired_speed = max(11.0, float(rng.normal(config.hv_regular_speed_mean, config.hv_regular_speed_std)))
        headway = 1.02
        standstill_gap = 6.0
        comfortable_brake = 3.24
        politeness = 0.10
        accel_limit = 1.52
        cooldown = 8.0

    if is_truck:
        desired_speed = min(desired_speed, 18.0)
        headway += 0.12
        standstill_gap += 0.6
        comfortable_brake = max(2.7, comfortable_brake - 0.25)
        accel_limit = min(accel_limit, 1.35)

    return VehicleProfile(
        veh_type="hv",
        is_cav=False,
        driver_style=style,
        desired_speed=desired_speed,
        desired_headway=headway,
        desired_standstill_gap=standstill_gap,
        comfortable_brake=comfortable_brake,
        politeness=politeness,
        vehicle_length=vehicle_length,
        accel_limit=accel_limit,
        lane_change_cooldown_s=cooldown,
        mobil_accel_threshold=0.20,
        mobil_safe_brake=0.80,
        mobil_right_bias=0.20,
        perception_head_m=100.0,
        perception_tail_m=100.0,
    )


def _direction_spawn_bounds(
    *,
    direction: str,
    base_y: float,
    height: float,
    config: SpawnConfig,
) -> tuple[float, float]:
    y_min = float(max(0.0, config.spawn_y_min))
    y_max = float(max(y_min + 1.0, config.spawn_y_max))
    road_min = float(base_y)
    road_max = float(height - base_y)
    if direction == "u":
        lower = min(road_max, road_min + y_min)
        upper = min(road_max, road_min + y_max)
    else:
        upper = max(road_min, road_max - y_min)
        lower = max(road_min, road_max - y_max)
    if upper <= lower:
        lower = road_min
        upper = road_max
    return float(lower), float(upper)


def _sample_lane_counts(total: int, lanes_per_dir: int, config: SpawnConfig, rng: np.random.Generator) -> List[int]:
    total = max(0, int(total))
    lanes_per_dir = max(1, int(lanes_per_dir))
    if total == 0:
        return [0 for _ in range(lanes_per_dir)]
    if not config.random_spawn:
        return even_split(total, lanes_per_dir)
    jitter = max(0.0, float(config.lane_density_jitter))
    weights = np.exp(rng.normal(0.0, jitter, size=lanes_per_dir))
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    counts = rng.multinomial(total, weights)
    return [int(value) for value in counts.tolist()]


def _sample_positions(count: int, lower: float, upper: float, config: SpawnConfig, rng: np.random.Generator) -> List[float]:
    count = max(0, int(count))
    lower = float(lower)
    upper = float(max(lower, upper))
    if count == 0:
        return []
    if count == 1:
        return [0.5 * (lower + upper)] if not config.random_spawn else [float(rng.uniform(lower, upper))]

    if not config.random_spawn:
        step = max(float(config.spacing), 1.0)
        values = [lower + step * idx for idx in range(count)]
        return [float(min(value, upper)) for value in values]

    min_gap = max(0.0, float(config.min_spawn_gap))
    available = max(0.0, upper - lower)
    required = min_gap * float(max(0, count - 1))
    if available <= required + 1e-6:
        return [float(value) for value in np.linspace(lower, upper, num=count)]

    slack = available - required
    offsets = np.sort(rng.uniform(0.0, slack, size=count))
    positions = lower + offsets + min_gap * np.arange(count, dtype=float)
    return [float(value) for value in positions.tolist()]


def generate_random_population(
    *,
    n_up: int,
    n_down: int,
    lanes_per_dir: int,
    lane_positions_up: Sequence[float],
    lane_positions_down: Sequence[float],
    height: float,
    base_y: float,
    composition: TrafficCompositionConfig,
    spawn: SpawnConfig,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    """Generate a mixed-traffic initial population for the highway scenario."""

    records: List[Dict[str, object]] = []
    total_vehicles = int(n_up) + int(n_down)
    global_type_tags: List[bool] | None = None
    if int(getattr(composition, "exact_cav_count", -1)) >= 0:
        global_type_tags = sample_vehicle_types(total_vehicles, composition, rng)
    global_cursor = 0
    for direction, total, lane_positions in (
        ("u", int(n_up), lane_positions_up),
        ("d", int(n_down), lane_positions_down),
    ):
        lower, upper = _direction_spawn_bounds(direction=direction, base_y=base_y, height=height, config=spawn)
        lane_counts = _sample_lane_counts(total, lanes_per_dir, spawn, rng)
        type_tags = sample_vehicle_types(total, composition, rng) if global_type_tags is None else global_type_tags[global_cursor: global_cursor + total]
        global_cursor += total
        profile_cursor = 0
        for lane_idx, lane_count in enumerate(lane_counts):
            positions = _sample_positions(lane_count, lower, upper, spawn, rng)
            if direction == "d":
                positions = sorted(positions, reverse=True)
            else:
                positions = sorted(positions)
            for pos_y in positions:
                is_cav = bool(type_tags[profile_cursor]) if profile_cursor < len(type_tags) else False
                profile_cursor += 1
                profile = build_vehicle_profile(is_cav, composition, rng)
                records.append(
                    {
                        "direction": direction,
                        "lane_idx": int(lane_idx),
                        "x": float(lane_positions[lane_idx]),
                        "y": float(pos_y),
                        "profile": profile,
                    }
                )
    return records
