"""Rule-based human-driver longitudinal and lane-change models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from .low_level_controller import (
        LaneChangeTrajectory,
        QuinticLaneChangePlanner,
        TrajectoryTrackingController,
        VehicleCommand,
    )
except ImportError:  # pragma: no cover
    from low_level_controller import (
        LaneChangeTrajectory,
        QuinticLaneChangePlanner,
        TrajectoryTrackingController,
        VehicleCommand,
    )


@dataclass
class IDMParams:
    max_accel: float = 1.52
    comfortable_brake: float = 3.24
    delta: float = 4.0
    min_gap: float = 6.0
    headway: float = 1.02
    desired_speed: float = 15.4


@dataclass
class MOBILParams:
    politeness: float = 0.10
    accel_threshold: float = 0.20
    safe_brake_limit: float = -0.80
    right_bias: float = 0.20
    min_lane_change_gap: float = 6.0
    lane_change_cooldown_s: float = 8.0
    perception_head_m: float = 100.0
    perception_tail_m: float = 100.0


class HumanDriverController:
    """IDM + MOBIL mixed-traffic baseline for human-driven vehicles."""

    def __init__(
        self,
        *,
        default_desired_speed: float = 15.4,
        default_headway: float = 1.02,
        min_gap: float = 6.0,
    ):
        self.default_desired_speed = float(default_desired_speed)
        self.default_headway = float(default_headway)
        self.default_min_gap = float(min_gap)
        self.max_speed = 33.0
        self._lane_change_plans: List[Optional[LaneChangeTrajectory]] = []
        self.lane_planner = QuinticLaneChangePlanner(base_duration=2.4, min_duration=1.7, max_duration=4.2)
        self.trajectory_tracker = TrajectoryTrackingController(kp=1.30, hold_gain=1.05, max_lateral_speed=2.6, max_hold_speed=0.9)

    def reset(self) -> None:
        self._lane_change_plans = []

    def _ensure_state(self, env) -> None:
        nveh = len(getattr(env, "vehicles", []))
        if len(self._lane_change_plans) != nveh:
            self._lane_change_plans = [None for _ in range(nveh)]

    def compute_commands(self, env, step: int, controlled_indices: Sequence[int]) -> Dict[int, VehicleCommand]:
        """Compute commands only for the requested human-driven vehicle indices."""

        self._ensure_state(env)
        commands: Dict[int, VehicleCommand] = {}
        dt = float(getattr(env, "timestep", 0.01))
        lanes_per_dir = int(getattr(env, "lanes_per_dir", 1))
        for idx in controlled_indices:
            veh = env.vehicles[idx]
            lane_idx = int(env._lane_idx[idx])
            direction = str(getattr(veh, "direction", "u"))

            idm = self._idm_params(veh)
            mobil = self._mobil_params(veh)
            front_idx, front_gap = self._nearest_front_vehicle(env, idx, lane_idx)
            front_speed = float(env.vehicles[front_idx].velocity) if front_idx is not None else None
            accel = self._idm_accel(
                ego_speed=float(veh.velocity),
                front_speed=front_speed,
                gap=front_gap,
                params=idm,
            )
            requested_lane = self._select_lane(env, idx, lane_idx, direction, lanes_per_dir, mobil, idm, step)
            lateral = self._compute_lateral_command(env, idx, lane_idx, requested_lane, dt)
            desired_gap = idm.min_gap + max(float(veh.velocity), 0.0) * idm.headway
            commands[int(idx)] = VehicleCommand(
                accel=float(np.clip(accel, -max(6.0, idm.comfortable_brake), idm.max_accel)),
                target_speed=float(np.clip(idm.desired_speed, 0.0, self.max_speed)),
                desired_gap=float(desired_gap),
                target_lane=lateral["target_lane"],
                reason="hv_mobil",
                longitudinal_mode="hv_idm",
                lateral_mode=lateral["lateral_mode"],
                target_x=lateral["target_x"],
                lateral_speed=lateral["lateral_speed"],
                lateral_error=lateral["lateral_error"],
                trajectory_progress=lateral["trajectory_progress"],
                lane_change_active=lateral["lane_change_active"],
                source_lane=lateral["source_lane"],
                front_idx=front_idx,
                comm_reliability=0.0,
                cacc_weight=0.0,
                headway_scale=max(0.7, float(idm.headway) / max(self.default_headway, 1e-6)),
                info_topology_level=0.0,
            )
        return commands

    def _idm_params(self, veh) -> IDMParams:
        return IDMParams(
            max_accel=float(getattr(veh, "accel_limit", 1.52)),
            comfortable_brake=float(getattr(veh, "comfortable_brake", 3.24)),
            delta=4.0,
            min_gap=float(getattr(veh, "desired_standstill_gap", self.default_min_gap)),
            headway=float(getattr(veh, "desired_headway", self.default_headway)),
            desired_speed=float(getattr(veh, "desired_speed", self.default_desired_speed)),
        )

    def _mobil_params(self, veh) -> MOBILParams:
        return MOBILParams(
            politeness=float(getattr(veh, "politeness", 0.10)),
            accel_threshold=float(getattr(veh, "mobil_accel_threshold", 0.20)),
            safe_brake_limit=-abs(float(getattr(veh, "mobil_safe_brake", 0.80))),
            right_bias=float(getattr(veh, "mobil_right_bias", 0.20)),
            min_lane_change_gap=max(5.0, float(getattr(veh, "desired_standstill_gap", self.default_min_gap))),
            lane_change_cooldown_s=float(getattr(veh, "lane_change_cooldown_s", 8.0)),
            perception_head_m=float(getattr(veh, "perception_head_m", 100.0)),
            perception_tail_m=float(getattr(veh, "perception_tail_m", 100.0)),
        )

    def _idm_accel(
        self,
        *,
        ego_speed: float,
        front_speed: Optional[float],
        gap: float,
        params: IDMParams,
    ) -> float:
        v0 = max(float(params.desired_speed), 1.0)
        free_term = 1.0 - (max(float(ego_speed), 0.0) / v0) ** float(params.delta)
        if front_speed is None or not np.isfinite(gap):
            return float(params.max_accel * free_term)

        delta_v = max(float(ego_speed) - float(front_speed), 0.0)
        s_star = float(params.min_gap + max(0.0, params.headway * float(ego_speed) + (ego_speed * delta_v) / (2.0 * np.sqrt(max(params.max_accel * params.comfortable_brake, 1e-6)))))
        interaction = (s_star / max(float(gap), 0.5)) ** 2
        return float(params.max_accel * (free_term - interaction))

    def _select_lane(
        self,
        env,
        idx: int,
        current_lane: int,
        direction: str,
        lanes_per_dir: int,
        mobil: MOBILParams,
        idm: IDMParams,
        step: int,
    ) -> Optional[int]:
        veh = env.vehicles[idx]
        if self._lane_change_plans[idx] is not None:
            return None

        cooldown_steps = int(max(1.0, mobil.lane_change_cooldown_s / max(float(getattr(env, "timestep", 0.01)), 1e-6)))
        last_change_step = int(getattr(veh, "last_lane_change_step", -10**9))
        if int(step) - last_change_step < cooldown_steps:
            return None
        if self._nearby_lane_change_recent(env, idx, current_lane, mobil, step):
            return None

        current_front_idx, current_gap = self._nearest_front_vehicle(
            env,
            idx,
            current_lane,
            max_distance=mobil.perception_head_m,
        )
        current_front_speed = float(env.vehicles[current_front_idx].velocity) if current_front_idx is not None else None
        current_accel = self._idm_accel(
            ego_speed=float(veh.velocity),
            front_speed=current_front_speed,
            gap=current_gap,
            params=idm,
        )

        candidates = [lane for lane in (current_lane - 1, current_lane + 1) if 0 <= lane < lanes_per_dir]
        best_lane = None
        best_utility = 0.0
        for target_lane in candidates:
            utility = self._mobil_utility(
                env=env,
                idx=idx,
                current_lane=current_lane,
                target_lane=target_lane,
                direction=direction,
                lanes_per_dir=lanes_per_dir,
                current_accel=current_accel,
                mobil=mobil,
                idm=idm,
            )
            if utility is not None and utility > best_utility:
                best_utility = utility
                best_lane = int(target_lane)
        return best_lane

    def _mobil_utility(
        self,
        *,
        env,
        idx: int,
        current_lane: int,
        target_lane: int,
        direction: str,
        lanes_per_dir: int,
        current_accel: float,
        mobil: MOBILParams,
        idm: IDMParams,
    ) -> Optional[float]:
        veh = env.vehicles[idx]
        target_front_idx, target_front_gap = self._nearest_front_vehicle(
            env,
            idx,
            target_lane,
            max_distance=mobil.perception_head_m,
        )
        target_rear_idx, target_rear_gap = self._nearest_rear_vehicle(
            env,
            idx,
            target_lane,
            max_distance=mobil.perception_tail_m,
        )
        target_front_speed = float(env.vehicles[target_front_idx].velocity) if target_front_idx is not None else None
        target_accel = self._idm_accel(
            ego_speed=float(veh.velocity),
            front_speed=target_front_speed,
            gap=target_front_gap,
            params=idm,
        )
        if target_front_gap < mobil.min_lane_change_gap or target_rear_gap < mobil.min_lane_change_gap:
            return None

        current_rear_term = 0.0
        if self._include_old_rear_term(direction, current_lane, lanes_per_dir):
            current_rear_idx, current_rear_gap = self._nearest_rear_vehicle(
                env,
                idx,
                current_lane,
                max_distance=mobil.perception_tail_m,
            )
            if current_rear_idx is not None:
                current_rear = env.vehicles[current_rear_idx]
                current_rear_params = self._idm_params(current_rear)
                current_rear_accel_before = self._idm_accel(
                    ego_speed=float(current_rear.velocity),
                    front_speed=float(veh.velocity),
                    gap=current_rear_gap,
                    params=current_rear_params,
                )
                current_rear_front_after_idx, current_rear_front_after_gap = self._nearest_front_vehicle(
                    env,
                    current_rear_idx,
                    current_lane,
                    max_distance=mobil.perception_head_m,
                    exclude_indices=(idx,),
                )
                current_rear_front_after_speed = (
                    float(env.vehicles[current_rear_front_after_idx].velocity)
                    if current_rear_front_after_idx is not None
                    else None
                )
                current_rear_accel_after = self._idm_accel(
                    ego_speed=float(current_rear.velocity),
                    front_speed=current_rear_front_after_speed,
                    gap=current_rear_front_after_gap,
                    params=current_rear_params,
                )
                current_rear_term = float(current_rear_accel_before - current_rear_accel_after)

        target_rear_term = 0.0
        if target_rear_idx is not None:
            rear = env.vehicles[target_rear_idx]
            rear_params = self._idm_params(rear)
            rear_front_before_idx, rear_front_before_gap = self._nearest_front_vehicle(
                env,
                target_rear_idx,
                target_lane,
                max_distance=mobil.perception_head_m,
            )
            rear_front_before_speed = float(env.vehicles[rear_front_before_idx].velocity) if rear_front_before_idx is not None else None
            rear_accel_before = self._idm_accel(
                ego_speed=float(rear.velocity),
                front_speed=rear_front_before_speed,
                gap=rear_front_before_gap,
                params=rear_params,
            )
            rear_accel_after = self._idm_accel(
                ego_speed=float(rear.velocity),
                front_speed=float(veh.velocity),
                gap=target_rear_gap,
                params=rear_params,
            )
            if rear_accel_after < float(mobil.safe_brake_limit):
                return None
            target_rear_term = float(rear_accel_before - rear_accel_after)

        politeness_term = float(mobil.politeness) * float(current_rear_term + target_rear_term)
        incentive_gain = float(target_accel - current_accel)
        if incentive_gain <= politeness_term + float(mobil.accel_threshold):
            return None

        utility = incentive_gain + politeness_term
        utility += self._right_lane_bias(direction, current_lane, target_lane, mobil.right_bias)
        return utility if utility > 0.0 else None

    def _include_old_rear_term(self, direction: str, current_lane: int, lanes_per_dir: int) -> bool:
        lane_rank_from_right = self._lane_rank_from_right(direction, current_lane, lanes_per_dir)
        return lane_rank_from_right > 1

    def _lane_rank_from_right(self, direction: str, lane_idx: int, lanes_per_dir: int) -> int:
        lane_idx = int(np.clip(lane_idx, 0, max(0, lanes_per_dir - 1)))
        if direction == "u":
            return int(lane_idx + 1)
        return int(lanes_per_dir - lane_idx)

    def _nearby_lane_change_recent(
        self,
        env,
        idx: int,
        current_lane: int,
        mobil: MOBILParams,
        step: int,
    ) -> bool:
        veh = env.vehicles[idx]
        direction = str(getattr(veh, "direction", "u"))
        y = float(veh.position[1])
        dt = max(float(getattr(env, "timestep", 0.01)), 1e-6)
        cooldown_steps = int(max(1.0, mobil.lane_change_cooldown_s / dt))
        lane_low = max(0, int(current_lane) - 1)
        lane_high = min(int(getattr(env, "lanes_per_dir", 1)) - 1, int(current_lane) + 1)
        proximity_m = max(float(mobil.perception_head_m), float(mobil.perception_tail_m))
        ego_progress = self._progress_coord(direction, y)
        for other_idx, other in enumerate(env.vehicles):
            if other_idx == idx:
                continue
            if str(getattr(other, "direction", "u")) != direction:
                continue
            other_lane = int(env._lane_idx[other_idx])
            if other_lane < lane_low or other_lane > lane_high:
                continue
            other_progress = self._progress_coord(direction, float(other.position[1]))
            if abs(other_progress - ego_progress) > proximity_m:
                continue
            if 0 <= other_idx < len(self._lane_change_plans) and self._lane_change_plans[other_idx] is not None:
                return True
            if int(step) - int(getattr(other, "last_lane_change_step", -10**9)) < cooldown_steps:
                return True
        return False

    def _right_lane_bias(self, direction: str, current_lane: int, target_lane: int, value: float) -> float:
        if direction == "u" and target_lane < current_lane:
            return float(value)
        if direction == "d" and target_lane > current_lane:
            return float(value)
        return 0.0

    def _compute_lateral_command(
        self,
        env,
        idx: int,
        current_lane: int,
        requested_lane: Optional[int],
        dt: float,
    ) -> Dict[str, object]:
        veh = env.vehicles[idx]
        direction = getattr(veh, "direction", "u")
        current_x = float(veh.position[0])
        plan = self._lane_change_plans[idx]

        if plan is not None:
            plan.elapsed = min(plan.duration, float(plan.elapsed + dt))
            ref_x, ref_vx, progress = self.lane_planner.sample(plan)
            lateral_speed, lateral_error = self.trajectory_tracker.track(current_x, ref_x, ref_vx)
            if progress >= 0.999 and abs(plan.target_x - current_x) <= 0.16:
                self._lane_change_plans[idx] = None
                hold_speed, hold_error = self.trajectory_tracker.hold(current_x, plan.target_x)
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

        if requested_lane is not None and int(requested_lane) != int(current_lane):
            target_x = self._lane_center_x(env, direction, int(requested_lane))
            plan = self.lane_planner.create(
                source_lane=int(current_lane),
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

    def _lane_center_x(self, env, direction: str, lane_idx: int) -> float:
        positions = list(env.true_up_lanes if direction == "u" else env.true_down_lanes)
        lane_idx = int(np.clip(lane_idx, 0, len(positions) - 1))
        return float(positions[lane_idx])

    def _progress_coord(self, direction: str, y: float) -> float:
        return float(y) if direction == "u" else float(-y)

    def _nearest_front_vehicle(
        self,
        env,
        idx: int,
        lane_idx: int,
        *,
        max_distance: float = float("inf"),
        exclude_indices: Optional[Sequence[int]] = None,
    ) -> tuple[Optional[int], float]:
        veh = env.vehicles[idx]
        direction = getattr(veh, "direction", "u")
        y = float(veh.position[1])
        excluded = {int(idx)}
        if exclude_indices is not None:
            excluded.update(int(other_idx) for other_idx in exclude_indices)
        max_distance = float(max_distance)
        best_idx = None
        best_gap = float("inf")
        ego_progress = self._progress_coord(direction, y)
        for other_idx, other in enumerate(env.vehicles):
            if other_idx in excluded:
                continue
            if getattr(other, "direction", None) != direction:
                continue
            if int(env._lane_idx[other_idx]) != int(lane_idx):
                continue
            gap = float(self._progress_coord(direction, float(other.position[1])) - ego_progress)
            if np.isfinite(max_distance) and gap > max_distance:
                continue
            if gap > 0.0 and gap < best_gap:
                best_gap = gap
                best_idx = other_idx
        return (None if best_idx is None else int(best_idx), float(best_gap))

    def _nearest_rear_vehicle(
        self,
        env,
        idx: int,
        lane_idx: int,
        *,
        max_distance: float = float("inf"),
        exclude_indices: Optional[Sequence[int]] = None,
    ) -> tuple[Optional[int], float]:
        veh = env.vehicles[idx]
        direction = getattr(veh, "direction", "u")
        y = float(veh.position[1])
        excluded = {int(idx)}
        if exclude_indices is not None:
            excluded.update(int(other_idx) for other_idx in exclude_indices)
        max_distance = float(max_distance)
        best_idx = None
        best_gap = float("inf")
        ego_progress = self._progress_coord(direction, y)
        for other_idx, other in enumerate(env.vehicles):
            if other_idx in excluded:
                continue
            if getattr(other, "direction", None) != direction:
                continue
            if int(env._lane_idx[other_idx]) != int(lane_idx):
                continue
            gap = float(ego_progress - self._progress_coord(direction, float(other.position[1])))
            if np.isfinite(max_distance) and gap > max_distance:
                continue
            if gap > 0.0 and gap < best_gap:
                best_gap = gap
                best_idx = other_idx
        return (None if best_idx is None else int(best_idx), float(best_gap))
