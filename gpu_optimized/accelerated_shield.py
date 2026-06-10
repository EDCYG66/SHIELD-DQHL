"""Accelerated SafetyShield with O(log n) neighbor search + scalar clip elimination."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from formation.safety_shield import SafetyShield
from formation.low_level_controller import VehicleCommand

from .cupy_kernels import (
    build_sorted_lane_index,
    extract_vehicle_arrays,
    nearest_front_from_sorted,
    nearest_rear_from_sorted,
)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


class AcceleratedSafetyShield(SafetyShield):
    """SafetyShield with sorted-index neighbor search and no np.clip overhead."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lane_index: Optional[Dict] = None
        self._veh_arrays: Optional[Dict[str, np.ndarray]] = None

    def enforce(
        self,
        env,
        commands: List[VehicleCommand],
        scheduler=None,
        step: int = 0,
    ) -> Tuple[List[VehicleCommand], Dict[str, float]]:
        n = len(env.vehicles)
        lanes = int(getattr(env, "lanes_per_dir", 4))
        veh_arrays = extract_vehicle_arrays(env.vehicles, env._lane_idx)
        self._lane_index = build_sorted_lane_index(
            veh_arrays["positions"][:, 1],
            veh_arrays["directions"],
            veh_arrays["lane_idx"],
            lanes,
        )
        self._veh_arrays = veh_arrays

        safe_commands = [VehicleCommand(**vars(command)) for command in commands]
        interventions = 0
        accel_clips = 0
        speed_clips = 0
        lane_change_blocks = 0
        emergency_brakes = 0
        event_speed_reductions = 0
        cbf_adjustments = 0
        cbf_longitudinal_clips = 0
        cbf_lateral_clips = 0
        cbf_blocked_lane_avoid = 0
        longitudinal_barriers: List[float] = []
        lateral_barriers: List[float] = []

        max_decel = self.max_decel
        max_accel = self.max_accel
        max_speed = self.max_speed

        for idx, command in enumerate(safe_commands):
            veh = env.vehicles[idx]
            lane_idx = int(env._lane_idx[idx])
            direction = getattr(veh, "direction", "u")
            y = float(veh.position[1])
            ego_speed = float(getattr(veh, "velocity", 0.0))

            original_accel = float(command.accel)
            bounded_accel = _clamp(original_accel, max_decel, max_accel)
            if abs(bounded_accel - original_accel) > 1e-9:
                interventions += 1
                accel_clips += 1
                command.accel = bounded_accel
            else:
                command.accel = original_accel

            original_speed = float(command.target_speed)
            bounded_speed = _clamp(original_speed, 0.0, max_speed)
            if abs(bounded_speed - original_speed) > 1e-9:
                interventions += 1
                speed_clips += 1
                command.target_speed = bounded_speed
            else:
                command.target_speed = original_speed

            front_idx, gap = self._nearest_front_vehicle(env, idx, lane_idx)
            if front_idx is not None and gap < 1e15:
                front_speed = float(getattr(env.vehicles[front_idx], "velocity", 0.0))
                projected_accel, projected_speed, barrier, adjusted, hard_brake = self._project_longitudinal_command(
                    command=command,
                    ego_speed=ego_speed,
                    front_speed=front_speed,
                    gap=gap,
                )
                longitudinal_barriers.append(barrier)
                if adjusted:
                    command.accel = projected_accel
                    command.target_speed = projected_speed
                    interventions += 1
                    cbf_adjustments += 1
                    cbf_longitudinal_clips += 1
                    if hard_brake:
                        emergency_brakes += 1

            if scheduler and scheduler.has_active_event(step):
                if self._apply_event_speed_guard(
                    command=command,
                    scheduler=scheduler,
                    direction=direction,
                    lane_idx=lane_idx,
                    y=y,
                    step=step,
                    ego_speed=ego_speed,
                ):
                    interventions += 1
                    event_speed_reductions += 1
                    cbf_blocked_lane_avoid += 1

            if command.target_lane is not None:
                target_lane = int(command.target_lane)
                lane_eval = self._evaluate_lane_change_barrier(
                    env=env,
                    idx=idx,
                    target_lane=target_lane,
                    scheduler=scheduler,
                    step=step,
                )
                if np.isfinite(float(lane_eval["barrier"])):
                    lateral_barriers.append(float(lane_eval["barrier"]))
                if not bool(lane_eval["allow"]):
                    command.target_lane = None
                    command.lane_change_active = False
                    command.lateral_speed = 0.0
                    command.lateral_mode = "lane_keep"
                    interventions += 1
                    lane_change_blocks += 1
                    cbf_adjustments += 1
                    cbf_lateral_clips += 1
                    if bool(lane_eval["blocked_lane"]):
                        cbf_blocked_lane_avoid += 1
                else:
                    speed_scale = float(lane_eval["speed_scale"])
                    if speed_scale < 0.999:
                        command.lateral_speed = float(command.lateral_speed * speed_scale)
                        interventions += 1
                        cbf_adjustments += 1
                        cbf_lateral_clips += 1
                    target_front_idx = lane_eval["front_idx"]
                    target_front_gap = float(lane_eval["front_gap"])
                    if target_front_idx is not None and np.isfinite(target_front_gap):
                        target_front_speed = float(getattr(env.vehicles[int(target_front_idx)], "velocity", 0.0))
                        projected_accel, projected_speed, barrier, adjusted, hard_brake = self._project_longitudinal_command(
                            command=command,
                            ego_speed=ego_speed,
                            front_speed=target_front_speed,
                            gap=target_front_gap,
                        )
                        longitudinal_barriers.append(float(barrier))
                        if adjusted:
                            command.accel = projected_accel
                            command.target_speed = projected_speed
                            interventions += 1
                            cbf_adjustments += 1
                            cbf_longitudinal_clips += 1
                            if hard_brake:
                                emergency_brakes += 1

        self._lane_index = None
        self._veh_arrays = None

        summary = {
            "interventions": float(interventions),
            "accel_clips": float(accel_clips),
            "speed_clips": float(speed_clips),
            "lane_change_blocks": float(lane_change_blocks),
            "emergency_brakes": float(emergency_brakes),
            "event_speed_reductions": float(event_speed_reductions),
            "cbf_adjustments": float(cbf_adjustments),
            "cbf_longitudinal_clips": float(cbf_longitudinal_clips),
            "cbf_lateral_clips": float(cbf_lateral_clips),
            "cbf_blocked_lane_avoid": float(cbf_blocked_lane_avoid),
            "cbf_min_barrier": self._finite_min(longitudinal_barriers, default=0.0),
            "cbf_lane_barrier": self._finite_min(lateral_barriers, default=0.0),
        }
        return safe_commands, summary

    def _project_longitudinal_command(
        self,
        *,
        command: VehicleCommand,
        ego_speed: float,
        front_speed: float,
        gap: float,
    ) -> Tuple[float, float, float, bool, bool]:
        headway_time = self._effective_time_headway(command, ego_speed)
        closing_speed = max(ego_speed - front_speed, 0.0)
        min_gap = self.min_gap + self.closing_gap_gain * closing_speed
        barrier = gap - (min_gap + headway_time * max(ego_speed, 0.0))
        accel_upper = (front_speed - ego_speed + self.cbf_alpha * barrier) / max(headway_time, 1e-6)
        projected_accel = _clamp(min(float(command.accel), accel_upper), self.max_decel, self.max_accel)
        speed_cap = _clamp(front_speed + self.cbf_alpha * barrier, 0.0, self.max_speed)
        projected_speed = _clamp(min(float(command.target_speed), speed_cap), 0.0, self.max_speed)
        adjusted = (projected_accel < float(command.accel) - 1e-6) or (projected_speed < float(command.target_speed) - 1e-6)
        hard_brake = barrier < -4.0 or projected_accel <= min(-4.0, 0.70 * self.max_decel)
        return projected_accel, projected_speed, barrier, adjusted, hard_brake

    def _apply_event_speed_guard(
        self,
        *,
        command: VehicleCommand,
        scheduler,
        direction: str,
        lane_idx: int,
        y: float,
        step: int,
        ego_speed: float,
    ) -> bool:
        if not scheduler.lane_blocked(direction, lane_idx, y, step, margin=self.event_margin):
            return False
        speed_scale = _clamp(float(scheduler.speed_scale(direction, lane_idx, y, step, margin=self.event_margin)), 0.0, 1.0)
        pressure = _clamp(float(scheduler.event_pressure(direction, y, step)), 0.0, 1.5)
        speed_cap = _clamp(max(self.event_speed_floor, self.max_speed * max(speed_scale, 0.18)), 0.0, self.max_speed)
        accel_cap = _clamp(-1.2 - 1.5 * pressure - 0.25 * max(ego_speed - speed_cap, 0.0), self.max_decel, self.max_accel)
        adjusted = False
        if speed_cap < float(command.target_speed) - 1e-6:
            command.target_speed = speed_cap
            adjusted = True
        if accel_cap < float(command.accel) - 1e-6:
            command.accel = accel_cap
            adjusted = True
        return adjusted

    def _effective_time_headway(self, command: VehicleCommand, ego_speed: float) -> float:
        desired_gap = max(float(getattr(command, "desired_gap", self.min_gap)), self.min_gap)
        tau_from_gap = 0.45 * (desired_gap - self.min_gap) / max(ego_speed, 1.0)
        tau_from_scale = self.nominal_time_headway * max(float(getattr(command, "headway_scale", 1.0)), 0.7)
        headway_time = max(self.min_time_headway, tau_from_gap, tau_from_scale)
        return _clamp(headway_time, self.min_time_headway, self.max_time_headway)

    def _evaluate_lane_change_barrier(
        self,
        *,
        env,
        idx: int,
        target_lane: int,
        scheduler,
        step: int,
    ) -> Dict[str, object]:
        lanes_per_dir = int(getattr(env, "lanes_per_dir", 1))
        if target_lane < 0 or target_lane >= lanes_per_dir:
            return {
                "allow": False,
                "barrier": -float(self.lane_change_gap),
                "blocked_lane": False,
                "speed_scale": 0.0,
                "front_idx": None,
                "front_gap": float("inf"),
            }

        veh = env.vehicles[idx]
        direction = getattr(veh, "direction", "u")
        y = float(veh.position[1])
        ego_speed = float(getattr(veh, "velocity", 0.0))
        if scheduler and scheduler.lane_blocked(direction, target_lane, y, step, margin=self.event_lane_margin):
            return {
                "allow": False,
                "barrier": -float(self.lane_change_gap),
                "blocked_lane": True,
                "speed_scale": 0.0,
                "front_idx": None,
                "front_gap": float("inf"),
            }

        front_idx, front_gap = self._nearest_front_vehicle(env, idx, target_lane)
        rear_idx, rear_gap = self._nearest_rear_vehicle(env, idx, target_lane)
        front_speed = float(getattr(env.vehicles[front_idx], "velocity", ego_speed)) if front_idx is not None else ego_speed
        rear_speed = float(getattr(env.vehicles[rear_idx], "velocity", ego_speed)) if rear_idx is not None else ego_speed

        front_required = self.lane_change_gap + self.lane_front_time_headway * max(ego_speed, 0.0)
        front_required += self.lane_speed_gain * max(ego_speed - front_speed, 0.0)
        rear_required = 0.75 * self.lane_change_gap + self.lane_rear_time_headway * max(rear_speed, 0.0)
        rear_required += self.lane_speed_gain * max(rear_speed - ego_speed, 0.0)

        front_barrier = float(front_gap - front_required) if front_idx is not None and np.isfinite(front_gap) else float("inf")
        rear_barrier = float(rear_gap - rear_required) if rear_idx is not None and np.isfinite(rear_gap) else float("inf")
        barrier = min(front_barrier, rear_barrier)
        allow = barrier >= 0.0
        speed_scale = 1.0
        if allow and np.isfinite(barrier):
            speed_scale = _clamp(0.40 + self.lane_cbf_alpha * barrier / max(self.lane_change_gap, 1e-6), 0.40, 1.0)
        return {
            "allow": bool(allow),
            "barrier": barrier,
            "blocked_lane": False,
            "speed_scale": speed_scale,
            "front_idx": front_idx,
            "front_gap": float(front_gap),
        }

    def _nearest_front_vehicle(self, env, idx: int, lane_idx: int) -> Tuple[Optional[int], float]:
        if self._lane_index is None or self._veh_arrays is None:
            return super()._nearest_front_vehicle(env, idx, lane_idx)
        direction = str(self._veh_arrays["directions"][idx])
        y = float(self._veh_arrays["positions"][idx, 1])
        key = (direction, int(lane_idx))
        if key not in self._lane_index:
            return None, float("inf")
        sorted_y, sorted_idx = self._lane_index[key]
        return nearest_front_from_sorted(sorted_y, sorted_idx, y, idx, direction)

    def _nearest_rear_vehicle(self, env, idx: int, lane_idx: int) -> Tuple[Optional[int], float]:
        if self._lane_index is None or self._veh_arrays is None:
            return super()._nearest_rear_vehicle(env, idx, lane_idx)
        direction = str(self._veh_arrays["directions"][idx])
        y = float(self._veh_arrays["positions"][idx, 1])
        key = (direction, int(lane_idx))
        if key not in self._lane_index:
            return None, float("inf")
        sorted_y, sorted_idx = self._lane_index[key]
        return nearest_rear_from_sorted(sorted_y, sorted_idx, y, idx, direction)
