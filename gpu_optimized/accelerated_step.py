"""Monkey-patch motion_models and low_level_controller to eliminate np.clip scalar overhead.

Replaces ~28 np.clip scalar calls per vehicle per step with pure Python _clamp/math.
At 24 vehicles, saves ~672 np.clip dispatches per step → ~20ms saving.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, Tuple, Optional


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


# ============================================================
# KinematicBicycleRoadModel.step_vehicle replacement
# ============================================================

def _step_vehicle_fast(
    self,
    veh,
    *,
    accel: float,
    desired_lateral_speed: float,
    dt: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
) -> Dict[str, float]:
    """Optimized step_vehicle: replaces np.clip/np.arcsin/np.sin/np.cos with math builtins."""

    self.initialize_vehicle(veh)
    dt = max(float(dt), 1e-4)
    direction = str(getattr(veh, "direction", "u"))
    sign = 1.0 if direction == "u" else -1.0
    wheelbase = max(float(getattr(veh, "wheelbase", self.default_wheelbase)), 1e-3)
    max_steer = abs(float(getattr(veh, "max_steer", self.default_max_steer)))
    max_steer_rate = abs(float(getattr(veh, "max_steer_rate", self.default_max_steer_rate)))
    max_heading_error = abs(float(getattr(veh, "max_heading_error", self.default_max_heading_error)))

    speed_prev = max(float(getattr(veh, "velocity", 0.0)), 0.0)
    speed_next = max(0.0, speed_prev + float(accel) * dt)
    speed_mid = 0.5 * (speed_prev + speed_next)

    heading_prev = float(getattr(veh, "heading_error", 0.0))
    steer_prev = float(getattr(veh, "steering_angle", 0.0))
    control_speed = max(speed_mid, self.min_speed_for_control)
    max_lateral_speed = control_speed * math.sin(max_heading_error)

    lateral_speed_cmd = _clamp(float(desired_lateral_speed), -max_lateral_speed, max_lateral_speed)
    asin_arg = _clamp(lateral_speed_cmd / max(control_speed, 1e-6), -1.0, 1.0)
    desired_heading = math.asin(asin_arg)
    desired_heading_rate = _clamp(
        self.heading_response * (desired_heading - heading_prev),
        -self.max_heading_rate, self.max_heading_rate,
    )
    steer_target = math.atan((wheelbase * desired_heading_rate) / max(control_speed, 1e-6))
    steer_target = _clamp(steer_target, -max_steer, max_steer)
    steer_step = max_steer_rate * dt
    steer_next = _clamp(steer_target, steer_prev - steer_step, steer_prev + steer_step)

    heading_rate = (speed_mid / wheelbase) * math.tan(steer_next)
    heading_next = _clamp(heading_prev + heading_rate * dt, -max_heading_error, max_heading_error)

    x_prev = float(veh.position[0])
    y_prev = float(veh.position[1])
    x_next = x_prev + speed_mid * math.sin(heading_next) * dt
    y_next = y_prev + sign * speed_mid * math.cos(heading_next) * dt

    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    x_clipped = _clamp(x_next, x_min, x_max)
    y_clipped = _clamp(y_next, y_min, y_max)

    if abs(x_clipped - x_next) > 1e-9:
        heading_next *= 0.5
        steer_next *= 0.5
    if abs(y_clipped - y_next) > 1e-9:
        speed_next = 0.0
        heading_next = 0.0
        steer_next = 0.0

    veh.position[0] = x_clipped
    veh.position[1] = y_clipped
    veh.velocity = float(speed_next)
    veh.heading_error = float(heading_next)
    veh.steering_angle = float(steer_next)
    veh.yaw = self.global_yaw(direction, heading_next)

    return {
        "speed": float(speed_next),
        "heading_error": float(heading_next),
        "steering_angle": float(steer_next),
        "desired_heading": float(desired_heading),
        "desired_lateral_speed": float(lateral_speed_cmd),
        "x": float(x_clipped),
        "y": float(y_clipped),
    }


# ============================================================
# CACCController.compute_accel replacement
# ============================================================

def _cacc_compute_accel_fast(
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
    if front_speed is None or not math.isfinite(gap):
        return speed_tracking

    gap_error = _clamp(gap - desired_gap, -30.0, 30.0)
    rel_speed = front_speed - ego_speed
    accel = speed_tracking
    accel += self.kp_gap * gap_error
    accel += self.kd_speed * rel_speed
    accel += self.kff_accel * float(front_accel)
    return accel


# ============================================================
# QuinticLaneChangePlanner.create replacement
# ============================================================

def _lane_planner_create_fast(self, *, source_lane, target_lane, source_x, target_x, speed):
    from formation.low_level_controller import LaneChangeTrajectory
    lane_distance = abs(float(target_x) - float(source_x))
    duration = self.base_duration + 0.10 * lane_distance + 0.02 * max(float(speed), 0.0)
    duration = _clamp(duration, self.min_duration, self.max_duration)
    return LaneChangeTrajectory(
        source_lane=int(source_lane),
        target_lane=int(target_lane),
        source_x=float(source_x),
        target_x=float(target_x),
        duration=duration,
    )


# ============================================================
# QuinticLaneChangePlanner.sample replacement
# ============================================================

def _lane_planner_sample_fast(self, trajectory):
    tau = _clamp(trajectory.elapsed / max(trajectory.duration, 1e-6), 0.0, 1.0)
    blend = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    d_blend = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / max(trajectory.duration, 1e-6)
    delta_x = float(trajectory.target_x - trajectory.source_x)
    ref_x = float(trajectory.source_x + delta_x * blend)
    ref_vx = float(delta_x * d_blend)
    return ref_x, ref_vx, tau


# ============================================================
# TrajectoryTrackingController.track and .hold replacements
# ============================================================

def _tracker_track_fast(self, current_x: float, ref_x: float, ref_vx: float):
    error = ref_x - current_x
    lateral_speed = _clamp(ref_vx + self.kp * error, -self.max_lateral_speed, self.max_lateral_speed)
    return lateral_speed, error


def _tracker_hold_fast(self, current_x: float, target_x: float):
    error = target_x - current_x
    lateral_speed = _clamp(self.hold_gain * error, -self.max_hold_speed, self.max_hold_speed)
    return lateral_speed, error


# ============================================================
# RuleBasedFormationController._build_longitudinal_adaptation replacement
# ============================================================

def _build_longitudinal_adaptation_fast(
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
    front_idx,
    requested_lane,
    comm_metrics,
):
    from formation.low_level_controller import LongitudinalAdaptation, MODE_RISK

    if front_idx is None or mode == "emergency":
        return LongitudinalAdaptation(
            mode_tag="cidm", cacc_weight=0.0,
            headway_scale=1.18 if mode == "emergency" else 1.10,
        )

    comm_v2v = _clamp(float(comm_metrics.get("comm_v2v_success", 0.0)), 0.0, 1.0)
    comm_fail = _clamp(float(comm_metrics.get("comm_fail_percent", 1.0)), 0.0, 1.0)
    rb_ratio = _clamp(float(comm_metrics.get("comm_used_rb_ratio", 1.0)), 0.0, 1.0)
    power_norm = _clamp(float(comm_metrics.get("comm_mean_power_norm", 1.0)), 0.0, 1.0)
    base_reliability = (
        0.50 * comm_v2v
        + 0.28 * (1.0 - comm_fail)
        + 0.14 * (1.0 - rb_ratio)
        + 0.08 * (1.0 - power_norm)
    )

    topology_type = str(getattr(env, "topology_type", "star")).lower()
    topology_scale = 1.0 if topology_type == "star" else 0.88
    topology_bias = 0.04 if topology_type == "star" else -0.02

    event_pressure = _clamp(float(scheduler.event_pressure(direction, y, step)), 0.0, 1.5) if scheduler else 0.0
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
    reliability = _clamp(reliability, 0.0, 1.0)

    high_conf = _clamp(0.5 * (self.cacc_min_comm + (1.0 - self.cacc_max_fail)), 0.70, 0.96)
    mid_conf = _clamp(high_conf - 0.25, 0.30, 0.75)
    base_weight = _clamp((reliability - mid_conf) / max(high_conf - mid_conf, 1e-6), 0.0, 1.0)
    context_scale = _clamp(
        1.0
        - 0.12 * event_pressure
        - 0.08 * blocked_here
        - 0.06 * lane_change_pressure
        - 0.08 * mode_risk,
        0.45,
        1.0,
    )
    cacc_weight = _clamp(base_weight * topology_scale * context_scale, 0.0, 1.0)

    if cacc_weight >= 0.72:
        mode_tag = "cacc"
        info_topology_level = 2.0
    elif cacc_weight >= 0.28:
        mode_tag = "hybrid"
        info_topology_level = 1.0
    else:
        mode_tag = "cidm"
        info_topology_level = 0.0

    headway_scale = _clamp(
        1.0
        + 0.22 * (1.0 - cacc_weight)
        + 0.14 * event_pressure
        + 0.07 * blocked_here
        + 0.05 * lane_change_pressure
        + 0.08 * mode_risk,
        1.0,
        1.65,
    )
    target_speed_scale = _clamp(
        1.0
        - 0.07 * (1.0 - reliability)
        - 0.08 * event_pressure
        - 0.05 * blocked_here
        - 0.03 * lane_change_pressure,
        0.60,
        1.0,
    )

    return LongitudinalAdaptation(
        mode_tag=mode_tag,
        cacc_weight=cacc_weight,
        comm_reliability=reliability,
        headway_scale=headway_scale,
        target_speed_scale=target_speed_scale,
        info_topology_level=info_topology_level,
    )


# ============================================================
# _lane_center_x replacement (1 np.clip call)
# ============================================================

def _lane_center_x_fast(self, env, direction: str, lane_idx: int) -> float:
    lane_positions = self._lane_positions(env, direction)
    lane_idx = _clamp(int(lane_idx), 0, len(lane_positions) - 1)
    return float(lane_positions[lane_idx])


# ============================================================
# Install / Uninstall
# ============================================================

_STEP_PATCHED = False


def install_step_patches() -> None:
    """Monkey-patch motion_models and low_level_controller to eliminate np.clip overhead."""
    global _STEP_PATCHED
    if _STEP_PATCHED:
        return

    from formation.motion_models import KinematicBicycleRoadModel
    from formation.low_level_controller import (
        CACCController,
        QuinticLaneChangePlanner,
        TrajectoryTrackingController,
        RuleBasedFormationController,
    )

    # motion_models
    KinematicBicycleRoadModel._orig_step_vehicle = KinematicBicycleRoadModel.step_vehicle
    KinematicBicycleRoadModel.step_vehicle = _step_vehicle_fast

    # low_level_controller
    CACCController._orig_compute_accel = CACCController.compute_accel
    CACCController.compute_accel = _cacc_compute_accel_fast

    QuinticLaneChangePlanner._orig_create = QuinticLaneChangePlanner.create
    QuinticLaneChangePlanner.create = _lane_planner_create_fast

    QuinticLaneChangePlanner._orig_sample = QuinticLaneChangePlanner.sample
    QuinticLaneChangePlanner.sample = _lane_planner_sample_fast

    TrajectoryTrackingController._orig_track = TrajectoryTrackingController.track
    TrajectoryTrackingController.track = _tracker_track_fast

    TrajectoryTrackingController._orig_hold = TrajectoryTrackingController.hold
    TrajectoryTrackingController.hold = _tracker_hold_fast

    RuleBasedFormationController._orig_build_longitudinal_adaptation = (
        RuleBasedFormationController._build_longitudinal_adaptation
    )
    RuleBasedFormationController._build_longitudinal_adaptation = _build_longitudinal_adaptation_fast

    RuleBasedFormationController._orig_lane_center_x = RuleBasedFormationController._lane_center_x
    RuleBasedFormationController._lane_center_x = _lane_center_x_fast

    _STEP_PATCHED = True


def uninstall_step_patches() -> None:
    """Restore original methods."""
    global _STEP_PATCHED
    if not _STEP_PATCHED:
        return

    from formation.motion_models import KinematicBicycleRoadModel
    from formation.low_level_controller import (
        CACCController,
        QuinticLaneChangePlanner,
        TrajectoryTrackingController,
        RuleBasedFormationController,
    )

    if hasattr(KinematicBicycleRoadModel, "_orig_step_vehicle"):
        KinematicBicycleRoadModel.step_vehicle = KinematicBicycleRoadModel._orig_step_vehicle

    if hasattr(CACCController, "_orig_compute_accel"):
        CACCController.compute_accel = CACCController._orig_compute_accel

    if hasattr(QuinticLaneChangePlanner, "_orig_create"):
        QuinticLaneChangePlanner.create = QuinticLaneChangePlanner._orig_create

    if hasattr(QuinticLaneChangePlanner, "_orig_sample"):
        QuinticLaneChangePlanner.sample = QuinticLaneChangePlanner._orig_sample

    if hasattr(TrajectoryTrackingController, "_orig_track"):
        TrajectoryTrackingController.track = TrajectoryTrackingController._orig_track

    if hasattr(TrajectoryTrackingController, "_orig_hold"):
        TrajectoryTrackingController.hold = TrajectoryTrackingController._orig_hold

    if hasattr(RuleBasedFormationController, "_orig_build_longitudinal_adaptation"):
        RuleBasedFormationController._build_longitudinal_adaptation = (
            RuleBasedFormationController._orig_build_longitudinal_adaptation
        )

    if hasattr(RuleBasedFormationController, "_orig_lane_center_x"):
        RuleBasedFormationController._lane_center_x = RuleBasedFormationController._orig_lane_center_x

    _STEP_PATCHED = False
