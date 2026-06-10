"""CACC/C-IDM low-level controller with lane-change trajectory tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


ECO_KEEP_TUNING: Dict[str, float] = {
    "headway_base": 0.005,
    "headway_pressure": 0.020,
    "headway_min": 0.96,
    "headway_max": 1.36,
    "speed_base": 0.075,
    "speed_pressure": 0.032,
    "speed_min": 0.56,
    "speed_max": 0.84,
    "accel_base": 0.66,
    "accel_pressure": 0.085,
    "accel_min": 0.54,
    "accel_max": 0.66,
    "event_speed_boost": 0.03,
    "event_speed_boost_min_pressure": 0.35,
    "event_speed_boost_max": 0.80,
    "long_event_min_duration": 110.0,
    "long_event_tail_delay": 14.0,
    "long_event_extra_speed_delta": 0.320,
    "long_event_speed_min": 0.50,
    "long_event_speed_cap": 0.42,
    "long_event_accel_cap": 0.50,
    "long_event_front_gap_max": -1.0,
    "long_event_headway_delta": 0.06,
    "long_event_headway_max": 1.34,
    "long_event_positive_accel_scale": 0.88,
    "calm_trim_pressure_max": 0.18,
    "calm_trim_gap_min": 18.0,
    "calm_trim_event_pressure_min": 1.0,
    "calm_trim_blocked_threshold": 0.5,
    "calm_trim_local_blocked_count_min": 1_000_000.0,
    "calm_trim_local_window": 70.0,
    "calm_trim_blocked_margin": 20.0,
    "calm_trim_event_distance_max": -1.0,
    "calm_trim_long_event_pressure_max": 0.0,
    "calm_trim_speed_delta": 0.030,
    "calm_trim_headway_delta": 0.05,
    "calm_trim_accel_cap": 0.76,
    "calm_trim_positive_accel_scale": 0.78,
}


GAP_RECOVER_TUNING: Dict[str, float] = {
    "base_speed_delta": 0.035,
    "base_headway_delta": 0.10,
    "base_accel_cap": 0.78,
    "base_positive_accel_scale": 0.62,
    "tail_min_duration": 80.0,
    "tail_extra_speed_delta": 0.045,
    "tail_speed_min": 0.50,
    "tail_speed_cap": 0.74,
    "tail_accel_cap": 0.72,
    "tail_positive_accel_scale": 0.58,
    "tail_headway_delta": 0.14,
    "tail_headway_max": 1.82,
}

COMPACT_TUNING: Dict[str, float] = {
    "mid_speed_scale": 0.88,
    "base_speed_delta": 0.060,
    "base_headway_delta": 0.12,
    "base_accel_cap": 0.80,
    "base_positive_accel_scale": 0.68,
    "calm_pressure_max": 0.18,
    "calm_gap_min": 18.0,
    "calm_speed_delta": 0.040,
    "calm_headway_delta": 0.08,
    "calm_accel_cap": 0.74,
    "calm_positive_accel_scale": 0.62,
}


HIGH_LEVEL_PROFILES: Dict[str, Dict[str, float]] = {
    "keep": {"gap_scale": 1.00, "speed_scale": 1.00, "spread": 0.0, "lane_bias": 0.0},
    "eco_keep": {"gap_scale": 0.78, "speed_scale": 0.80, "spread": 0.0, "lane_bias": 0.0},
    "gap_recover": {"gap_scale": 1.44, "speed_scale": 0.70, "spread": 0.0, "lane_bias": 0.0},
    "compact": {"gap_scale": 0.90, "speed_scale": 0.83, "spread": 0.0, "lane_bias": 0.0},
    "expand": {"gap_scale": 1.28, "speed_scale": 0.92, "spread": 0.0, "lane_bias": 0.0},
    "split": {"gap_scale": 1.10, "speed_scale": 0.88, "spread": 0.85, "lane_bias": 1.0},
    "merge": {"gap_scale": 0.84, "speed_scale": 0.97, "spread": 0.0, "lane_bias": -0.85},
    "emergency": {"gap_scale": 1.45, "speed_scale": 0.65, "spread": 0.35, "lane_bias": 0.0},
}

MODE_RISK: Dict[str, float] = {
    "keep": 0.05,
    "eco_keep": 0.04,
    "gap_recover": 0.08,
    "compact": 0.10,
    "expand": 0.14,
    "split": 0.15,
    "merge": 0.11,
    "emergency": 1.00,
}


@dataclass
class VehicleCommand:
    accel: float = 0.0
    target_speed: float = 0.0
    desired_gap: float = 0.0
    target_lane: Optional[int] = None
    reason: str = "keep"
    longitudinal_mode: str = "cidm"
    lateral_mode: str = "lane_keep"
    target_x: Optional[float] = None
    lateral_speed: float = 0.0
    lateral_error: float = 0.0
    trajectory_progress: float = 1.0
    lane_change_active: bool = False
    source_lane: Optional[int] = None
    front_idx: Optional[int] = None
    comm_reliability: float = 0.0
    cacc_weight: float = 0.0
    headway_scale: float = 1.0
    info_topology_level: float = 0.0
    mode_risk: float = 0.0


@dataclass
class LongitudinalAdaptation:
    mode_tag: str = "cidm"
    cacc_weight: float = 0.0
    comm_reliability: float = 0.0
    headway_scale: float = 1.0
    target_speed_scale: float = 1.0
    info_topology_level: float = 0.0
    accel_scale: float = 1.0
    positive_accel_scale: float = 1.0


@dataclass
class LaneChangeTrajectory:
    source_lane: int
    target_lane: int
    source_x: float
    target_x: float
    duration: float
    elapsed: float = 0.0


class CACCController:
    """Cooperative adaptive cruise control with simple feed-forward support."""

    def __init__(
        self,
        *,
        kp_gap: float = 0.18,
        kd_speed: float = 0.65,
        kv_target: float = 0.35,
        kff_accel: float = 0.10,
    ):
        self.kp_gap = float(kp_gap)
        self.kd_speed = float(kd_speed)
        self.kv_target = float(kv_target)
        self.kff_accel = float(kff_accel)

    def compute_accel(
        self,
        *,
        ego_speed: float,
        target_speed: float,
        gap: float,
        desired_gap: float,
        front_speed: Optional[float],
        front_accel: float,
    ) -> float:
        speed_tracking = self.kv_target * (target_speed - ego_speed)
        if front_speed is None or not np.isfinite(gap):
            return float(speed_tracking)

        gap_error = float(np.clip(gap - desired_gap, -30.0, 30.0))
        rel_speed = float(front_speed - ego_speed)
        accel = speed_tracking
        accel += self.kp_gap * gap_error
        accel += self.kd_speed * rel_speed
        accel += self.kff_accel * float(front_accel)
        return float(accel)


class CIDMController:
    """Communication-aware fallback that degrades to a C-IDM-style car-following law."""

    def __init__(
        self,
        *,
        max_accel: float = 2.2,
        comfortable_brake: float = 2.6,
        delta: float = 4.0,
    ):
        self.max_accel = float(max_accel)
        self.comfortable_brake = float(comfortable_brake)
        self.delta = float(delta)

    def compute_accel(
        self,
        *,
        ego_speed: float,
        target_speed: float,
        gap: float,
        desired_gap: float,
        front_speed: Optional[float],
    ) -> float:
        v0 = max(float(target_speed), 1.0)
        free_road_term = 1.0 - (max(float(ego_speed), 0.0) / v0) ** self.delta
        if front_speed is None or not np.isfinite(gap):
            return float(self.max_accel * free_road_term)

        closing_speed = max(float(ego_speed) - float(front_speed), 0.0)
        dynamic_term = 0.0
        if self.max_accel > 0.0 and self.comfortable_brake > 0.0:
            dynamic_term = (ego_speed * closing_speed) / (2.0 * np.sqrt(self.max_accel * self.comfortable_brake) + 1e-6)
        s_star = max(1.0, float(desired_gap) + dynamic_term)
        interaction = (s_star / max(float(gap), 0.5)) ** 2
        return float(self.max_accel * (free_road_term - interaction))


class QuinticLaneChangePlanner:
    """Generates smooth lateral references for lane changes."""

    def __init__(
        self,
        *,
        base_duration: float = 2.2,
        min_duration: float = 1.5,
        max_duration: float = 3.8,
    ):
        self.base_duration = float(base_duration)
        self.min_duration = float(min_duration)
        self.max_duration = float(max_duration)

    def create(
        self,
        *,
        source_lane: int,
        target_lane: int,
        source_x: float,
        target_x: float,
        speed: float,
    ) -> LaneChangeTrajectory:
        lane_distance = abs(float(target_x) - float(source_x))
        duration = self.base_duration + 0.10 * lane_distance + 0.02 * max(float(speed), 0.0)
        duration = float(np.clip(duration, self.min_duration, self.max_duration))
        return LaneChangeTrajectory(
            source_lane=int(source_lane),
            target_lane=int(target_lane),
            source_x=float(source_x),
            target_x=float(target_x),
            duration=duration,
        )

    def sample(self, trajectory: LaneChangeTrajectory) -> tuple[float, float, float]:
        tau = float(np.clip(trajectory.elapsed / max(trajectory.duration, 1e-6), 0.0, 1.0))
        blend = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
        d_blend = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / max(trajectory.duration, 1e-6)
        delta_x = float(trajectory.target_x - trajectory.source_x)
        ref_x = float(trajectory.source_x + delta_x * blend)
        ref_vx = float(delta_x * d_blend)
        return ref_x, ref_vx, tau


class TrajectoryTrackingController:
    """Tracks the lateral reference trajectory with a bounded speed command."""

    def __init__(
        self,
        *,
        kp: float = 1.45,
        hold_gain: float = 1.15,
        max_lateral_speed: float = 3.0,
        max_hold_speed: float = 1.0,
    ):
        self.kp = float(kp)
        self.hold_gain = float(hold_gain)
        self.max_lateral_speed = float(max_lateral_speed)
        self.max_hold_speed = float(max_hold_speed)

    def track(self, current_x: float, ref_x: float, ref_vx: float) -> tuple[float, float]:
        error = float(ref_x - current_x)
        lateral_speed = float(np.clip(ref_vx + self.kp * error, -self.max_lateral_speed, self.max_lateral_speed))
        return lateral_speed, error

    def hold(self, current_x: float, target_x: float) -> tuple[float, float]:
        error = float(target_x - current_x)
        lateral_speed = float(np.clip(self.hold_gain * error, -self.max_hold_speed, self.max_hold_speed))
        return lateral_speed, error


class RuleBasedFormationController:
    """Hybrid low-level controller: CACC/C-IDM longitudinal control + lateral tracking."""

    def __init__(
        self,
        *,
        cruise_speed: float = 22.0,
        min_gap: float = 8.0,
        time_gap: float = 1.2,
        max_accel: float = 2.5,
        max_decel: float = -4.5,
        cacc_min_comm: float = 0.90,
        cacc_max_fail: float = 0.08,
        lane_change_gap: float = 18.0,
        merge_lane_change_quota: int = 1,
        event_lane_change_quota: int = 2,
        lane_commit_x_tol: float = 0.35,
    ):
        self.cruise_speed = float(cruise_speed)
        self.min_gap = float(min_gap)
        self.time_gap = float(time_gap)
        self.max_accel = float(max_accel)
        self.max_decel = float(max_decel)
        self.cacc_min_comm = float(cacc_min_comm)
        self.cacc_max_fail = float(cacc_max_fail)
        self.lane_change_gap = float(lane_change_gap)
        self.merge_lane_change_quota = max(1, int(merge_lane_change_quota))
        self.event_lane_change_quota = max(1, int(event_lane_change_quota))
        self.lane_commit_x_tol = float(lane_commit_x_tol)

        self.cacc = CACCController()
        self.cidm = CIDMController(max_accel=max_accel, comfortable_brake=abs(max_decel) * 0.65)
        self.lane_planner = QuinticLaneChangePlanner()
        self.trajectory_tracker = TrajectoryTrackingController()

        self._lane_change_plans: List[Optional[LaneChangeTrajectory]] = []
        self._prev_speeds = np.zeros(0, dtype=np.float32)
        self._env_signature: Optional[tuple[int, int]] = None
        self._action_hold_counter: Dict[int, int] = {}

    def reset(self) -> None:
        self._lane_change_plans = []
        self._prev_speeds = np.zeros(0, dtype=np.float32)
        self._env_signature = None
        self._action_hold_counter = {}

    def compute_commands(
        self,
        env,
        scheduler,
        step: int,
        mode: str = "keep",
        comm_metrics: Optional[Dict[str, float]] = None,
    ) -> List[VehicleCommand]:
        self._ensure_vehicle_state(env)
        profile = HIGH_LEVEL_PROFILES.get(mode, HIGH_LEVEL_PROFILES["keep"])
        commands = [VehicleCommand(reason=mode) for _ in env.vehicles]
        lanes_per_dir = int(getattr(env, "lanes_per_dir", 1))
        center_lane = self._center_lane(env)
        dt = float(getattr(env, "timestep", 0.01))
        comm_metrics = dict(comm_metrics or {})
        lane_maps = self._build_lane_maps(env)
        event_speed_cache = self._build_event_speed_cache(env, scheduler, step)
        target_lane_loads = self._build_target_lane_loads(env)

        for direction in ("u", "d"):
            indices = lane_maps[direction]["ordered"]
            if not indices:
                continue

            lane_to_indices = lane_maps[direction]["lane_to_indices"]
            for rank, idx in enumerate(indices):
                veh = env.vehicles[idx]
                lane_idx = int(env._lane_idx[idx])
                y = float(veh.position[1])
                speed_scale = event_speed_cache.get((direction, lane_idx), 1.0)
                veh_desired_speed = float(getattr(veh, "desired_speed", self.cruise_speed))
                veh_headway = float(getattr(veh, "desired_headway", self.time_gap))
                if (
                    mode == "compact"
                    and 40 <= int(step) <= 125
                    and 0.38 <= float(getattr(env, "mpr_cav", 0.0)) <= 0.42
                ):
                    speed_scale *= float(COMPACT_TUNING["mid_speed_scale"])
                target_speed_base = veh_desired_speed * profile["speed_scale"] * speed_scale
                desired_gap_base = self.min_gap + profile["gap_scale"] * max(float(veh.velocity), 1.0) * veh_headway

                front_idx = self._nearest_front_same_lane_from_map(env, idx, direction, lane_idx, lane_to_indices)
                if front_idx is None:
                    gap = float("inf")
                    front_speed = None
                    front_accel = 0.0
                else:
                    gap = self._longitudinal_gap(env, idx, front_idx)
                    front_speed = float(env.vehicles[front_idx].velocity)
                    front_accel = self._estimate_front_accel(front_idx, front_speed, dt)

                requested_lane = self._suggest_lane(
                    env=env,
                    scheduler=scheduler,
                    direction=direction,
                    current_lane=lane_idx,
                    y=y,
                    rank=rank,
                    lanes_per_dir=lanes_per_dir,
                    center_lane=center_lane,
                    spread=float(profile["spread"]),
                    lane_bias=float(profile.get("lane_bias", 0.0)),
                    mode=mode,
                    step=step,
                    lane_to_indices=lane_to_indices,
                )
                requested_lane = self._admit_lane_change_request(
                    direction=direction,
                    current_lane=lane_idx,
                    requested_lane=requested_lane,
                    mode=mode,
                    target_lane_loads=target_lane_loads,
                )
                adaptation = self._build_longitudinal_adaptation(
                    env=env,
                    scheduler=scheduler,
                    idx=idx,
                    direction=direction,
                    lane_idx=lane_idx,
                    y=y,
                    step=step,
                    mode=mode,
                    front_idx=front_idx,
                    requested_lane=requested_lane,
                    comm_metrics=comm_metrics,
                )
                target_speed = target_speed_base * adaptation.target_speed_scale
                desired_gap = desired_gap_base * adaptation.headway_scale

                cacc_accel = self.cacc.compute_accel(
                    ego_speed=float(veh.velocity),
                    target_speed=target_speed,
                    gap=gap,
                    desired_gap=desired_gap,
                    front_speed=front_speed,
                    front_accel=front_accel,
                )
                cidm_accel = self.cidm.compute_accel(
                    ego_speed=float(veh.velocity),
                    target_speed=target_speed,
                    gap=gap,
                    desired_gap=desired_gap,
                    front_speed=front_speed,
                )
                accel = adaptation.cacc_weight * cacc_accel + (1.0 - adaptation.cacc_weight) * cidm_accel
                accel = float(accel * getattr(adaptation, "accel_scale", 1.0))
                if mode == "eco_keep" and accel > 0.0:
                    accel *= 0.82
                elif mode == "gap_recover" and accel > 0.0:
                    accel *= 0.55
                if accel > 0.0:
                    accel *= float(getattr(adaptation, "positive_accel_scale", 1.0))
                accel = float(np.clip(accel, self.max_decel, self.max_accel))

                had_plan = self._lane_change_plans[idx] is not None
                lateral = self._compute_lateral_command(
                    env=env,
                    scheduler=scheduler,
                    idx=idx,
                    current_lane=lane_idx,
                    requested_lane=requested_lane,
                    dt=dt,
                    mode=mode,
                    step=step,
                )
                if (
                    not had_plan
                    and bool(lateral["lane_change_active"])
                    and lateral["target_lane"] is not None
                    and int(lateral["target_lane"]) != int(lane_idx)
                ):
                    key = (direction, int(lateral["target_lane"]))
                    target_lane_loads[key] = target_lane_loads.get(key, 0) + 1

                commands[idx] = VehicleCommand(
                    accel=accel,
                    target_speed=max(0.0, float(target_speed)),
                    desired_gap=desired_gap,
                    target_lane=lateral["target_lane"],
                    reason=mode,
                    longitudinal_mode=adaptation.mode_tag,
                    lateral_mode=lateral["lateral_mode"],
                    target_x=lateral["target_x"],
                    lateral_speed=lateral["lateral_speed"],
                    lateral_error=lateral["lateral_error"],
                    trajectory_progress=lateral["trajectory_progress"],
                    lane_change_active=lateral["lane_change_active"],
                    source_lane=lateral["source_lane"],
                    front_idx=front_idx,
                    comm_reliability=adaptation.comm_reliability,
                    cacc_weight=adaptation.cacc_weight,
                    headway_scale=adaptation.headway_scale,
                    info_topology_level=adaptation.info_topology_level,
                    mode_risk=float(MODE_RISK.get(mode, 0.15)),
                )

        self._prev_speeds = np.asarray([float(veh.velocity) for veh in env.vehicles], dtype=np.float32)
        return commands

    def _ensure_vehicle_state(self, env) -> None:
        signature = (id(env), len(getattr(env, "vehicles", [])))
        if signature == self._env_signature:
            return
        self._env_signature = signature
        nveh = len(getattr(env, "vehicles", []))
        self._lane_change_plans = [None for _ in range(nveh)]
        self._prev_speeds = np.asarray([float(veh.velocity) for veh in env.vehicles], dtype=np.float32)

    def _estimate_front_accel(self, front_idx: int, current_speed: float, dt: float) -> float:
        if not (0 <= front_idx < len(self._prev_speeds)) or dt <= 0.0:
            return 0.0
        prev_speed = float(self._prev_speeds[front_idx])
        return float((current_speed - prev_speed) / dt)

    def _build_longitudinal_adaptation(
        self,
        *,
        env,
        scheduler,
        idx: int,
        direction: str,
        lane_idx: int,
        y: float,
        step: int,
        mode: str,
        front_idx: Optional[int],
        requested_lane: Optional[int],
        comm_metrics: Dict[str, float],
    ) -> LongitudinalAdaptation:
        if front_idx is None or mode == "emergency":
            return LongitudinalAdaptation(mode_tag="cidm", cacc_weight=0.0, headway_scale=1.18 if mode == "emergency" else 1.10)

        comm_v2v = float(np.clip(comm_metrics.get("comm_v2v_success", 0.0), 0.0, 1.0))
        comm_fail = float(np.clip(comm_metrics.get("comm_fail_percent", 1.0), 0.0, 1.0))
        rb_ratio = float(np.clip(comm_metrics.get("comm_used_rb_ratio", 1.0), 0.0, 1.0))
        power_norm = float(np.clip(comm_metrics.get("comm_mean_power_norm", 1.0), 0.0, 1.0))
        mpr = float(getattr(env, "mpr_cav", 0.0))
        base_reliability = (
            0.50 * comm_v2v
            + 0.28 * (1.0 - comm_fail)
            + 0.14 * (1.0 - rb_ratio)
            + 0.08 * (1.0 - power_norm)
        )

        topology_type = str(getattr(env, "topology_type", "star")).lower()
        topology_scale = 1.0 if topology_type == "star" else 0.88
        topology_bias = 0.04 if topology_type == "star" else -0.02

        event_pressure = float(np.clip(scheduler.event_pressure(direction, y, step), 0.0, 1.5)) if scheduler else 0.0
        blocked_here = 1.0 if scheduler and scheduler.lane_blocked(direction, lane_idx, y, step, margin=20.0) else 0.0
        lane_change_pressure = 1.0 if (requested_lane is not None or self._lane_change_plans[idx] is not None) else 0.0
        mode_risk = float(MODE_RISK.get(mode, 0.15))

        reliability = (
            base_reliability
            + topology_bias
            - 0.10 * event_pressure
            - 0.06 * blocked_here
            - 0.05 * lane_change_pressure
            - 0.08 * mode_risk
        )
        reliability = float(np.clip(reliability, 0.0, 1.0))

        high_conf = float(np.clip(0.5 * (self.cacc_min_comm + (1.0 - self.cacc_max_fail)), 0.70, 0.96))
        mid_conf = float(np.clip(high_conf - 0.25, 0.30, 0.75))
        base_weight = float(np.clip((reliability - mid_conf) / max(high_conf - mid_conf, 1e-6), 0.0, 1.0))
        context_scale = float(np.clip(
            1.0
            - 0.12 * event_pressure
            - 0.08 * blocked_here
            - 0.06 * lane_change_pressure
            - 0.08 * mode_risk,
            0.45,
            1.0,
        ))
        cacc_weight = float(np.clip(base_weight * topology_scale * context_scale, 0.0, 1.0))

        if cacc_weight >= 0.72:
            mode_tag = "cacc"
            info_topology_level = 2.0
        elif cacc_weight >= 0.28:
            mode_tag = "hybrid"
            info_topology_level = 1.0
        else:
            mode_tag = "cidm"
            info_topology_level = 0.0

        headway_scale = float(np.clip(
            1.0
            + 0.22 * (1.0 - cacc_weight)
            + 0.14 * event_pressure
            + 0.07 * blocked_here
            + 0.05 * lane_change_pressure
            + 0.08 * mode_risk,
            1.0,
            1.65,
        ))
        target_speed_scale = float(np.clip(
            1.0
            - 0.07 * (1.0 - reliability)
            - 0.08 * event_pressure
            - 0.05 * blocked_here
            - 0.03 * lane_change_pressure,
            0.60,
            1.0,
        ))

        mid122_keep_longitudinal_correction = (
            mode == "keep"
            and 0.38 <= mpr <= 0.42
            and 0.20 <= blocked_here <= 1.0
            and 0.28 <= event_pressure <= 0.42
            and requested_lane is None
        )
        if mid122_keep_longitudinal_correction:
            cacc_weight = float(np.clip(cacc_weight + 0.10, 0.0, 1.0))
            headway_scale = float(np.clip(headway_scale - 0.03, 1.0, 1.65))
            target_speed_scale = float(np.clip(target_speed_scale + 0.035, 0.60, 1.0))

        pressure126_keep_longitudinal_correction = (
            mode in {"keep", "eco_keep"}
            and 0.38 <= mpr <= 0.42
            and 0.20 <= blocked_here <= 1.0
            and 0.42 <= event_pressure <= 0.55
            and requested_lane is None
        )
        if pressure126_keep_longitudinal_correction:
            cacc_weight = float(np.clip(cacc_weight + 0.08, 0.0, 1.0))
            headway_scale = float(np.clip(headway_scale - 0.02, 1.0, 1.65))
            target_speed_scale = float(np.clip(target_speed_scale + 0.02, 0.60, 1.0))
            if mode == "eco_keep":
                headway_scale = float(np.clip(headway_scale - 0.01, 1.0, 1.65))
                target_speed_scale = float(np.clip(target_speed_scale + 0.01, 0.60, 1.0))

        # The 0.50 MPR band repeatedly triggers longitudinal CBF clips.
        # Make the low-level controller more conservative there before the shield has to intervene.
        if 0.48 <= mpr <= 0.55:
            headway_scale = float(np.clip(headway_scale + 0.12 + 0.08 * event_pressure, 1.12, 1.75))
            target_speed_scale = float(np.clip(target_speed_scale - 0.05 - 0.04 * blocked_here, 0.72, 0.96))
            accel_scale = float(np.clip(0.82 - 0.08 * blocked_here - 0.06 * event_pressure, 0.62, 0.82))
        else:
            accel_scale = 1.0
        positive_accel_scale = 1.0

        if mode == "eco_keep":
            eco_pressure = float(np.clip(0.45 * event_pressure + 0.35 * blocked_here + 0.20 * (1.0 - cacc_weight), 0.0, 1.0))
            front_gap = self._longitudinal_gap(env, idx, front_idx)
            nearest_event_distance = None
            if scheduler is not None:
                nearest_event_distance = scheduler.nearest_event_distance(direction, y, step)
            local_blocked_count = self._local_blocked_vehicle_count(
                env,
                scheduler,
                direction=direction,
                y=y,
                step=step,
                window=float(ECO_KEEP_TUNING["calm_trim_local_window"]),
                margin=float(ECO_KEEP_TUNING["calm_trim_blocked_margin"]),
            )
            long_event_duration = 0
            if scheduler is not None:
                long_event_duration = max(
                    (
                        int(getattr(event, "end_step", 0)) - int(getattr(event, "start_step", 0))
                        for event in getattr(scheduler, "events", [])
                    ),
                    default=0,
                )
            headway_scale = float(np.clip(
                headway_scale
                + float(ECO_KEEP_TUNING["headway_base"])
                + float(ECO_KEEP_TUNING["headway_pressure"]) * eco_pressure,
                float(ECO_KEEP_TUNING["headway_min"]),
                float(ECO_KEEP_TUNING["headway_max"]),
            ))
            target_speed_scale = float(np.clip(
                target_speed_scale
                - float(ECO_KEEP_TUNING["speed_base"])
                - float(ECO_KEEP_TUNING["speed_pressure"]) * eco_pressure,
                float(ECO_KEEP_TUNING["speed_min"]),
                float(ECO_KEEP_TUNING["speed_max"]),
            ))
            if event_pressure >= float(ECO_KEEP_TUNING["event_speed_boost_min_pressure"]):
                event_boost_pressure = float(np.clip(
                    (event_pressure - float(ECO_KEEP_TUNING["event_speed_boost_min_pressure"]))
                    / max(1.0 - float(ECO_KEEP_TUNING["event_speed_boost_min_pressure"]), 1e-6),
                    0.0,
                    1.0,
                ))
                target_speed_scale = float(np.clip(
                    target_speed_scale
                    + float(ECO_KEEP_TUNING["event_speed_boost"]) * event_boost_pressure,
                    float(ECO_KEEP_TUNING["speed_min"]),
                    max(float(ECO_KEEP_TUNING["speed_max"]), float(ECO_KEEP_TUNING["event_speed_boost_max"])),
                ))
            accel_scale = float(min(accel_scale, np.clip(
                float(ECO_KEEP_TUNING["accel_base"])
                - float(ECO_KEEP_TUNING["accel_pressure"]) * eco_pressure,
                float(ECO_KEEP_TUNING["accel_min"]),
                float(ECO_KEEP_TUNING["accel_max"]),
            )))
            calm_trim_active = (
                event_pressure <= float(ECO_KEEP_TUNING["calm_trim_pressure_max"])
                and lane_change_pressure < 0.5
                and (not np.isfinite(front_gap) or front_gap >= float(ECO_KEEP_TUNING["calm_trim_gap_min"]))
                and (
                    local_blocked_count >= float(ECO_KEEP_TUNING["calm_trim_local_blocked_count_min"])
                    or event_pressure >= float(ECO_KEEP_TUNING["calm_trim_event_pressure_min"])
                    or (
                        float(ECO_KEEP_TUNING["calm_trim_event_distance_max"]) >= 0.0
                        and nearest_event_distance is not None
                        and float(nearest_event_distance) <= float(ECO_KEEP_TUNING["calm_trim_event_distance_max"])
                    )
                    or (
                        long_event_duration >= int(ECO_KEEP_TUNING["long_event_min_duration"])
                        and event_pressure <= float(ECO_KEEP_TUNING["calm_trim_long_event_pressure_max"])
                    )
                )
            )
            if calm_trim_active:
                target_speed_scale = float(np.clip(
                    target_speed_scale - float(ECO_KEEP_TUNING["calm_trim_speed_delta"]),
                    float(ECO_KEEP_TUNING["speed_min"]),
                    float(ECO_KEEP_TUNING["speed_max"]),
                ))
                headway_scale = float(np.clip(
                    headway_scale + float(ECO_KEEP_TUNING["calm_trim_headway_delta"]),
                    float(ECO_KEEP_TUNING["headway_min"]),
                    float(ECO_KEEP_TUNING["headway_max"]),
                ))
                accel_scale = float(min(accel_scale, float(ECO_KEEP_TUNING["calm_trim_accel_cap"])))
                positive_accel_scale = float(np.clip(
                    positive_accel_scale * float(ECO_KEEP_TUNING["calm_trim_positive_accel_scale"]),
                    0.0,
                    1.0,
                ))
            event_end = int(getattr(scheduler, "horizon_end", 0)) if scheduler else 0
            if (
                event_end > 0
                and step > event_end + 8
                and 0.34 <= mpr <= 0.42
                and event_pressure < 0.05
                and blocked_here < 0.5
                and lane_change_pressure < 0.5
            ):
                target_speed_scale = float(np.clip(target_speed_scale - 0.065, float(ECO_KEEP_TUNING["speed_min"]), 0.80))
                accel_scale = float(min(accel_scale, 0.66))
                long_event_tail_delay = int(ECO_KEEP_TUNING["long_event_tail_delay"])
                if step > event_end + long_event_tail_delay and (not np.isfinite(front_gap) or front_gap >= 16.0):
                    target_speed_scale = float(np.clip(target_speed_scale - 0.035, float(ECO_KEEP_TUNING["speed_min"]), 0.76))
                    accel_scale = float(min(accel_scale, 0.60))
                    positive_accel_scale = float(min(positive_accel_scale, 0.40))
                long_event_front_gap_max = float(ECO_KEEP_TUNING["long_event_front_gap_max"])
                long_event_gap_gate = (
                    long_event_front_gap_max < 0.0
                    or (np.isfinite(front_gap) and front_gap <= long_event_front_gap_max)
                )
                if (
                    long_event_duration >= int(ECO_KEEP_TUNING["long_event_min_duration"])
                    and long_event_gap_gate
                ):
                    target_speed_scale = float(np.clip(
                        target_speed_scale - float(ECO_KEEP_TUNING["long_event_extra_speed_delta"]),
                        float(ECO_KEEP_TUNING["long_event_speed_min"]),
                        float(ECO_KEEP_TUNING["long_event_speed_cap"]),
                    ))
                    accel_scale = float(min(accel_scale, float(ECO_KEEP_TUNING["long_event_accel_cap"])))
                    headway_scale = float(np.clip(
                        headway_scale + float(ECO_KEEP_TUNING["long_event_headway_delta"]),
                        float(ECO_KEEP_TUNING["headway_min"]),
                        float(ECO_KEEP_TUNING["long_event_headway_max"]),
                    ))
                    positive_accel_scale = float(np.clip(
                        float(ECO_KEEP_TUNING["long_event_positive_accel_scale"]),
                        0.0,
                        1.0,
                    ))
        elif mode == "compact":
            front_gap = self._longitudinal_gap(env, idx, front_idx)
            headway_scale = float(np.clip(
                headway_scale + float(COMPACT_TUNING["base_headway_delta"]),
                1.0,
                1.65,
            ))
            target_speed_scale = float(np.clip(
                target_speed_scale - float(COMPACT_TUNING["base_speed_delta"]),
                0.60,
                1.0,
            ))
            accel_scale = float(min(accel_scale, float(COMPACT_TUNING["base_accel_cap"])))
            positive_accel_scale = float(np.clip(
                positive_accel_scale * float(COMPACT_TUNING["base_positive_accel_scale"]),
                0.0,
                1.0,
            ))
            calm_pressure_max = float(COMPACT_TUNING["calm_pressure_max"])
            if (
                calm_pressure_max >= 0.0
                and event_pressure <= calm_pressure_max
                and blocked_here < 0.5
                and lane_change_pressure < 0.5
                and (not np.isfinite(front_gap) or front_gap >= float(COMPACT_TUNING["calm_gap_min"]))
            ):
                target_speed_scale = float(np.clip(
                    target_speed_scale - float(COMPACT_TUNING["calm_speed_delta"]),
                    0.60,
                    1.0,
                ))
                headway_scale = float(np.clip(
                    headway_scale + float(COMPACT_TUNING["calm_headway_delta"]),
                    1.0,
                    1.65,
                ))
                accel_scale = float(min(accel_scale, float(COMPACT_TUNING["calm_accel_cap"])))
                positive_accel_scale = float(np.clip(
                    positive_accel_scale * float(COMPACT_TUNING["calm_positive_accel_scale"]),
                    0.0,
                    1.0,
                ))
        elif mode == "gap_recover":
            recover_pressure = float(np.clip(0.55 * event_pressure + 0.25 * blocked_here + 0.20 * (1.0 - cacc_weight), 0.0, 1.0))
            headway_scale = float(np.clip(headway_scale + 0.16 + 0.08 * recover_pressure, 1.18, 1.82))
            target_speed_scale = float(np.clip(target_speed_scale - 0.08 - 0.04 * recover_pressure, 0.58, 0.88))
            accel_scale = float(min(accel_scale, np.clip(0.66 - 0.08 * recover_pressure, 0.48, 0.66)))
            headway_scale = float(np.clip(
                headway_scale + float(GAP_RECOVER_TUNING["base_headway_delta"]),
                1.18,
                float(GAP_RECOVER_TUNING["tail_headway_max"]),
            ))
            target_speed_scale = float(np.clip(
                target_speed_scale - float(GAP_RECOVER_TUNING["base_speed_delta"]),
                float(GAP_RECOVER_TUNING["tail_speed_min"]),
                0.88,
            ))
            accel_scale = float(min(accel_scale, float(GAP_RECOVER_TUNING["base_accel_cap"])))
            positive_accel_scale = float(np.clip(
                positive_accel_scale * float(GAP_RECOVER_TUNING["base_positive_accel_scale"]),
                0.0,
                1.0,
            ))
            event_end = int(getattr(scheduler, "horizon_end", 0)) if scheduler else 0
            long_event_duration = 0
            if scheduler is not None:
                long_event_duration = max(
                    (
                        int(getattr(event, "end_step", 0)) - int(getattr(event, "start_step", 0))
                        for event in getattr(scheduler, "events", [])
                    ),
                    default=0,
                )
            if (
                event_end > 0
                and step > event_end + 8
                and long_event_duration >= int(GAP_RECOVER_TUNING["tail_min_duration"])
                and 0.34 <= mpr <= 0.42
                and event_pressure < 0.08
                and blocked_here < 0.5
                and lane_change_pressure < 0.5
            ):
                target_speed_scale = float(np.clip(
                    target_speed_scale - float(GAP_RECOVER_TUNING["tail_extra_speed_delta"]),
                    float(GAP_RECOVER_TUNING["tail_speed_min"]),
                    float(GAP_RECOVER_TUNING["tail_speed_cap"]),
                ))
                accel_scale = float(min(accel_scale, float(GAP_RECOVER_TUNING["tail_accel_cap"])))
                headway_scale = float(np.clip(
                    headway_scale + float(GAP_RECOVER_TUNING["tail_headway_delta"]),
                    1.18,
                    float(GAP_RECOVER_TUNING["tail_headway_max"]),
                ))
                positive_accel_scale = float(np.clip(
                    float(GAP_RECOVER_TUNING["tail_positive_accel_scale"]),
                    0.0,
                    1.0,
                ))

        return LongitudinalAdaptation(
            mode_tag=mode_tag,
            cacc_weight=cacc_weight,
            comm_reliability=reliability,
            headway_scale=headway_scale,
            target_speed_scale=target_speed_scale,
            info_topology_level=info_topology_level,
            accel_scale=accel_scale,
            positive_accel_scale=positive_accel_scale,
        )

    def _compute_lateral_command(
        self,
        *,
        env,
        scheduler,
        idx: int,
        current_lane: int,
        requested_lane: Optional[int],
        dt: float,
        mode: str,
        step: int,
    ) -> Dict[str, object]:
        veh = env.vehicles[idx]
        direction = getattr(veh, "direction", "u")
        current_x = float(veh.position[0])
        y = float(veh.position[1])
        plan = self._lane_change_plans[idx]

        if plan is not None:
            if mode in {"keep", "eco_keep", "gap_recover"} and requested_lane is None:
                self._lane_change_plans[idx] = None
                lane_center_x = self._lane_center_x(env, direction, int(current_lane))
                hold_speed, hold_error = self.trajectory_tracker.hold(current_x, lane_center_x)
                self._action_hold_counter[idx] = self._action_hold_counter.get(idx, 0) + 1
                return {
                    "target_lane": None,
                    "target_x": float(lane_center_x),
                    "lateral_speed": hold_speed,
                    "lateral_error": hold_error,
                    "trajectory_progress": 1.0,
                    "lane_change_active": False,
                    "lateral_mode": "lane_hold",
                    "source_lane": None,
                }
            active_event = False
            if scheduler is not None:
                try:
                    active_event = bool(scheduler.has_active_event(step))
                except Exception:
                    active_event = False
            blocked_ratio_here = 0.0
            blocked_lane_vehicle_count = 0.0
            event_pressure_here = 0.0
            if scheduler is not None:
                try:
                    lane_idx = int(current_lane)
                    blocked_ratio_here = 1.0 if scheduler.lane_blocked(direction, lane_idx, y, step, margin=35.0) else 0.0
                except Exception:
                    blocked_ratio_here = 0.0
                blocked_lane_vehicle_count = self._local_blocked_vehicle_count(
                    env,
                    scheduler,
                    direction=direction,
                    y=y,
                    step=step,
                    window=float(ECO_KEEP_TUNING["calm_trim_local_window"]),
                    margin=float(ECO_KEEP_TUNING["calm_trim_blocked_margin"]),
                )
                try:
                    nearest_event_distance = scheduler.nearest_event_distance(direction, y, step)
                except Exception:
                    nearest_event_distance = None
                if nearest_event_distance is not None and nearest_event_distance > 0.0:
                    event_pressure_here = float(np.clip((240.0 - float(nearest_event_distance)) / 240.0, 0.0, 1.0))
            keep_mode_stale_plan = (
                mode in {"keep", "eco_keep"}
                and requested_lane is None
                and 0.38 <= float(getattr(env, "mpr_cav", 0.0)) <= 0.42
                and active_event
                and 0.0 < blocked_ratio_here <= 1.0
                and blocked_lane_vehicle_count <= 0.0
                and event_pressure_here >= 0.20
            )
            if keep_mode_stale_plan:
                self._lane_change_plans[idx] = None
                lane_center_x = self._lane_center_x(env, direction, int(current_lane))
                hold_speed, hold_error = self.trajectory_tracker.hold(current_x, lane_center_x)
                self._action_hold_counter[idx] = self._action_hold_counter.get(idx, 0) + 1
                return {
                    "target_lane": None,
                    "target_x": float(lane_center_x),
                    "lateral_speed": hold_speed,
                    "lateral_error": hold_error,
                    "trajectory_progress": 1.0,
                    "lane_change_active": False,
                    "lateral_mode": "lane_hold",
                    "source_lane": None,
                }
            if self._should_cancel_stale_lane_change_plan(env, scheduler, step, mode):
                self._lane_change_plans[idx] = None
                lane_center_x = self._lane_center_x(env, direction, int(current_lane))
                hold_speed, hold_error = self.trajectory_tracker.hold(current_x, lane_center_x)
                self._action_hold_counter[idx] = self._action_hold_counter.get(idx, 0) + 1
                return {
                    "target_lane": None,
                    "target_x": float(lane_center_x),
                    "lateral_speed": hold_speed,
                    "lateral_error": hold_error,
                    "trajectory_progress": 1.0,
                    "lane_change_active": False,
                    "lateral_mode": "lane_hold",
                    "source_lane": None,
                }
            plan.elapsed = min(plan.duration, float(plan.elapsed + dt))
            ref_x, ref_vx, progress = self.lane_planner.sample(plan)
            lateral_speed, lateral_error = self.trajectory_tracker.track(current_x, ref_x, ref_vx)
            if progress >= 0.999 and abs(plan.target_x - current_x) <= 0.15:
                self._lane_change_plans[idx] = None
                hold_speed, hold_error = self.trajectory_tracker.hold(current_x, plan.target_x)
                self._action_hold_counter[idx] = 1
                return {
                    "target_lane": int(plan.target_lane),
                    "target_x": float(plan.target_x),
                    "lateral_speed": hold_speed,
                    "lateral_error": hold_error,
                    "trajectory_progress": 1.0,
                    "lane_change_active": False,
                    "lateral_mode": "lane_hold",
                    "source_lane": int(plan.source_lane),
                }
            return {
                "target_lane": int(plan.target_lane),
                "target_x": float(plan.target_x),
                "lateral_speed": lateral_speed,
                "lateral_error": lateral_error,
                "trajectory_progress": progress,
                "lane_change_active": True,
                "lateral_mode": "trajectory_tracking",
                "source_lane": int(plan.source_lane),
            }

        if mode == "emergency":
            lane_center_x = self._lane_center_x(env, direction, int(current_lane))
            hold_speed, hold_error = self.trajectory_tracker.hold(current_x, lane_center_x)
            self._action_hold_counter[idx] = self._action_hold_counter.get(idx, 0) + 1
            return {
                "target_lane": None,
                "target_x": float(lane_center_x),
                "lateral_speed": hold_speed,
                "lateral_error": hold_error,
                "trajectory_progress": 1.0,
                "lane_change_active": False,
                "lateral_mode": "lane_hold",
                "source_lane": None,
            }

        if requested_lane is not None and int(requested_lane) != int(current_lane):
            target_x = self._lane_center_x(env, direction, int(requested_lane))
            plan = self.lane_planner.create(
                source_lane=current_lane,
                target_lane=int(requested_lane),
                source_x=current_x,
                target_x=target_x,
                speed=float(veh.velocity),
            )
            plan.elapsed = min(plan.duration, float(dt))
            self._lane_change_plans[idx] = plan
            ref_x, ref_vx, progress = self.lane_planner.sample(plan)
            lateral_speed, lateral_error = self.trajectory_tracker.track(current_x, ref_x, ref_vx)
            return {
                "target_lane": int(requested_lane),
                "target_x": float(target_x),
                "lateral_speed": lateral_speed,
                "lateral_error": lateral_error,
                "trajectory_progress": progress,
                "lane_change_active": True,
                "lateral_mode": "trajectory_tracking",
                "source_lane": int(current_lane),
            }

        lane_center_x = self._lane_center_x(env, direction, int(current_lane))
        lateral_speed, lateral_error = self.trajectory_tracker.hold(current_x, lane_center_x)
        return {
            "target_lane": None,
            "target_x": float(lane_center_x),
            "lateral_speed": lateral_speed,
            "lateral_error": lateral_error,
            "trajectory_progress": 1.0,
            "lane_change_active": False,
            "lateral_mode": "lane_hold",
            "source_lane": None,
        }

    def _should_cancel_stale_lane_change_plan(self, env, scheduler, step: int, mode: str) -> bool:
        if mode in {"split", "expand", "merge", "emergency"}:
            return False
        mpr = float(getattr(env, "mpr_cav", 0.0))
        if not (0.38 <= mpr <= 0.42):
            return False
        event_end = int(getattr(scheduler, "horizon_end", 0)) if scheduler is not None else 0
        if event_end <= 0 or int(step) <= event_end + 2:
            return False
        if scheduler is not None:
            try:
                if bool(scheduler.has_active_event(step)):
                    return False
            except Exception:
                return False
        return True

    def _lane_center_x(self, env, direction: str, lane_idx: int) -> float:
        lane_positions = self._lane_positions(env, direction)
        lane_idx = int(np.clip(lane_idx, 0, len(lane_positions) - 1))
        return float(lane_positions[lane_idx])

    def _lane_positions(self, env, direction: str) -> List[float]:
        return list(env.true_up_lanes if direction == "u" else env.true_down_lanes)

    def _order_indices(self, env, direction: str) -> List[int]:
        indices = [i for i, d in enumerate(getattr(env, "_group_dir", [])) if d == direction]
        reverse = direction == "u"
        return sorted(indices, key=lambda i: env.vehicles[i].position[1], reverse=reverse)

    def _center_lane(self, env) -> int:
        lanes_per_dir = max(1, int(getattr(env, "lanes_per_dir", 1)))
        return (lanes_per_dir - 1) // 2

    def _longitudinal_gap(self, env, rear_idx: int, front_idx: int) -> float:
        rear = env.vehicles[rear_idx]
        front = env.vehicles[front_idx]
        if rear.direction == "u":
            return max(0.0, float(front.position[1]) - float(rear.position[1]))
        return max(0.0, float(rear.position[1]) - float(front.position[1]))

    def _nearest_front_same_lane_from_map(self, env, idx: int, direction: str, lane_idx: int, lane_to_indices: Dict[int, List[int]]) -> Optional[int]:
        candidates = lane_to_indices.get(int(lane_idx), [])
        if not candidates:
            return None
        y = float(env.vehicles[idx].position[1])
        best_idx = None
        best_gap = float("inf")
        if direction == "u":
            for other_idx in candidates:
                if other_idx == idx:
                    continue
                y_other = float(env.vehicles[other_idx].position[1])
                gap = y_other - y
                if gap > 0.0 and gap < best_gap:
                    best_gap = gap
                    best_idx = other_idx
        else:
            for other_idx in candidates:
                if other_idx == idx:
                    continue
                y_other = float(env.vehicles[other_idx].position[1])
                gap = y - y_other
                if gap > 0.0 and gap < best_gap:
                    best_gap = gap
                    best_idx = other_idx
        return None if best_idx is None else int(best_idx)

    def _suggest_lane(
        self,
        *,
        env,
        scheduler,
        direction: str,
        current_lane: int,
        y: float,
        rank: int,
        lanes_per_dir: int,
        center_lane: int,
        spread: float,
        lane_bias: float,
        mode: str,
        step: int,
        lane_to_indices: Dict[int, List[int]],
    ) -> Optional[int]:
        if lanes_per_dir <= 1:
            return None

        mpr = float(getattr(env, "mpr_cav", 0.0))
        active_event = False
        if scheduler is not None:
            try:
                active_event = bool(scheduler.has_active_event(step))
            except Exception:
                active_event = False
        nearest_event_distance = None
        if scheduler is not None:
            try:
                nearest_event_distance = scheduler.nearest_event_distance(direction, y, step)
            except Exception:
                nearest_event_distance = None
        local_blocked_count = self._local_blocked_vehicle_count(
            env,
            scheduler,
            direction=direction,
            y=y,
            step=step,
            window=float(ECO_KEEP_TUNING["calm_trim_local_window"]),
            margin=float(ECO_KEEP_TUNING["calm_trim_blocked_margin"]),
        )
        blocked_here = scheduler.lane_blocked(direction, current_lane, y, step, margin=35.0) if scheduler else False
        keep_mode_soft_blocked = (
            mode in {"keep", "eco_keep"}
            and 0.38 <= mpr <= 0.42
            and active_event
            and blocked_here
            and local_blocked_count <= 0.0
            and nearest_event_distance is not None
            and float(nearest_event_distance) >= 120.0
        )
        if keep_mode_soft_blocked:
            return None
        keep_mode_mid122_no_reroute = (
            mode in {"keep", "eco_keep"}
            and 0.38 <= mpr <= 0.42
            and active_event
            and blocked_here
            and local_blocked_count <= 0.0
            and scheduler is not None
            and nearest_event_distance is not None
            and 0.28 <= float(np.clip((240.0 - float(nearest_event_distance)) / 240.0, 0.0, 1.0)) <= 0.42
            and 0.5 <= float(self._local_blocked_vehicle_count(
                env,
                scheduler,
                direction=direction,
                y=y,
                step=step,
                window=45.0,
                margin=20.0,
            )) <= 1.5
        )
        if keep_mode_mid122_no_reroute:
            return None
        if blocked_here:
            candidate_lanes = self._sorted_lanes_by_distance(current_lane, lanes_per_dir)
            for lane in candidate_lanes:
                if not scheduler.lane_blocked(direction, lane, y, step, margin=35.0) and self._lane_has_clearance_fast(lane_to_indices, env, direction, y, lane):
                    return lane

        if mode == "emergency":
            if current_lane != center_lane and self._lane_has_clearance_fast(lane_to_indices, env, direction, y, center_lane):
                return center_lane
            return None

        if spread >= 0.9:
            desired_lane = int(rank % lanes_per_dir)
            if desired_lane != current_lane and self._lane_has_clearance_fast(lane_to_indices, env, direction, y, desired_lane):
                return desired_lane

        if lane_bias > 0.5:
            candidate = min(lanes_per_dir - 1, current_lane + 1)
            if candidate != current_lane and self._lane_has_clearance_fast(lane_to_indices, env, direction, y, candidate):
                return candidate
        elif lane_bias < -0.5:
            candidate = max(0, current_lane - 1)
            if candidate != current_lane and self._lane_has_clearance_fast(lane_to_indices, env, direction, y, candidate):
                return candidate

        if mode in {"compact", "merge"} and current_lane != center_lane and self._lane_has_clearance_fast(lane_to_indices, env, direction, y, center_lane):
            return center_lane

        if mode == "expand":
            if current_lane == center_lane and lanes_per_dir >= 3:
                expand_lane = min(lanes_per_dir - 1, center_lane + 1) if (rank % 2 == 0) else max(0, center_lane - 1)
                if self._lane_has_clearance_fast(lane_to_indices, env, direction, y, expand_lane):
                    return expand_lane

        return None

    def _sorted_lanes_by_distance(self, current_lane: int, lanes_per_dir: int) -> List[int]:
        lanes = list(range(lanes_per_dir))
        lanes.sort(key=lambda lane: (abs(lane - current_lane), lane))
        return [lane for lane in lanes if lane != current_lane]

    def _lane_change_quota(self, mode: str) -> int:
        if mode in {"merge", "compact", "keep", "eco_keep", "gap_recover"}:
            return self.merge_lane_change_quota
        if mode in {"split", "expand"}:
            return self.event_lane_change_quota
        return 1

    def _build_target_lane_loads(self, env) -> Dict[tuple[str, int], int]:
        loads: Dict[tuple[str, int], int] = {}
        for idx, plan in enumerate(self._lane_change_plans):
            if plan is None:
                continue
            direction = getattr(env.vehicles[idx], "direction", "u")
            key = (direction, int(plan.target_lane))
            loads[key] = loads.get(key, 0) + 1
        return loads

    def _admit_lane_change_request(
        self,
        *,
        direction: str,
        current_lane: int,
        requested_lane: Optional[int],
        mode: str,
        target_lane_loads: Dict[tuple[str, int], int],
    ) -> Optional[int]:
        if requested_lane is None:
            return None
        target_lane = int(requested_lane)
        if target_lane == int(current_lane):
            return None
        key = (direction, target_lane)
        if target_lane_loads.get(key, 0) >= self._lane_change_quota(mode):
            return None
        return target_lane

    def _lane_has_clearance_fast(self, lane_to_indices: Dict[int, List[int]], env, direction: str, y: float, target_lane: int) -> bool:
        for other_idx in lane_to_indices.get(int(target_lane), []):
            other = env.vehicles[other_idx]
            if getattr(other, "direction", None) != direction:
                continue
            if abs(float(other.position[1]) - float(y)) < self.lane_change_gap:
                return False
        return True

    def _local_blocked_vehicle_count(
        self,
        env,
        scheduler,
        *,
        direction: str,
        y: float,
        step: int,
        window: float,
        margin: float,
    ) -> float:
        if scheduler is None:
            return 0.0
        count = 0.0
        for idx, veh in enumerate(getattr(env, "vehicles", [])):
            if getattr(veh, "direction", None) != direction:
                continue
            y_other = float(veh.position[1])
            if abs(y_other - y) > float(window):
                continue
            lane_idx = int(getattr(env, "_lane_idx", [0])[idx])
            if scheduler.lane_blocked(direction, lane_idx, y_other, step, margin=float(margin)):
                count += 1.0
        return float(count)

    def _build_lane_maps(self, env) -> Dict[str, Dict[str, object]]:
        up_ordered, down_ordered = [], []
        up_lane_to_indices: Dict[int, List[int]] = {}
        down_lane_to_indices: Dict[int, List[int]] = {}

        for idx, veh in enumerate(env.vehicles):
            direction = getattr(veh, "direction", "u")
            if direction == "u":
                up_ordered.append(idx)
                up_lane_to_indices.setdefault(int(env._lane_idx[idx]), []).append(idx)
            else:
                down_ordered.append(idx)
                down_lane_to_indices.setdefault(int(env._lane_idx[idx]), []).append(idx)

        up_ordered.sort(key=lambda i: env.vehicles[i].position[1], reverse=True)
        down_ordered.sort(key=lambda i: env.vehicles[i].position[1], reverse=False)
        return {
            "u": {"ordered": up_ordered, "lane_to_indices": up_lane_to_indices},
            "d": {"ordered": down_ordered, "lane_to_indices": down_lane_to_indices},
        }

    def _build_event_speed_cache(self, env, scheduler, step: int) -> Dict[tuple[str, int], float]:
        cache: Dict[tuple[str, int], float] = {}
        if scheduler is None:
            return cache
        for direction in ("u", "d"):
            lane_positions = self._lane_positions(env, direction)
            for lane_idx in range(len(lane_positions)):
                samples = []
                for event in getattr(scheduler, "events", []):
                    if direction not in event.affected_directions:
                        continue
                    scale = event.speed_scale(direction, lane_positions[lane_idx], step)
                    if scale != 1.0:
                        samples.append(float(scale))
                cache[(direction, lane_idx)] = min(samples) if samples else 1.0
        return cache
