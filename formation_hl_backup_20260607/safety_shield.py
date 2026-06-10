"""CBF-style safety post-processor for vehicle commands."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from .low_level_controller import VehicleCommand
except ImportError:  # pragma: no cover
    from low_level_controller import VehicleCommand


class SafetyShield:
    """Projects reference commands onto a CBF-style safe control set."""

    def __init__(
        self,
        *,
        min_gap: float = 8.0,
        lane_change_gap: float = 14.0,
        max_speed: float = 33.0,
        max_accel: float = 2.5,
        max_decel: float = -6.0,
        nominal_time_headway: float = 0.80,
        min_time_headway: float = 0.50,
        max_time_headway: float = 1.80,
        cbf_alpha: float = 0.65,
        closing_gap_gain: float = 0.35,
        lane_front_time_headway: float = 0.12,
        lane_rear_time_headway: float = 0.15,
        lane_speed_gain: float = 0.20,
        lane_cbf_alpha: float = 0.60,
        event_margin: float = 12.0,
        event_lane_margin: float = 18.0,
        event_speed_floor: float = 4.0,
    ):
        self.min_gap = float(min_gap)
        self.lane_change_gap = float(lane_change_gap)
        self.max_speed = float(max_speed)
        self.max_accel = float(max_accel)
        self.max_decel = float(max_decel)
        self.nominal_time_headway = float(nominal_time_headway)
        self.min_time_headway = float(min_time_headway)
        self.max_time_headway = float(max_time_headway)
        self.cbf_alpha = float(cbf_alpha)
        self.closing_gap_gain = float(closing_gap_gain)
        self.lane_front_time_headway = float(lane_front_time_headway)
        self.lane_rear_time_headway = float(lane_rear_time_headway)
        self.lane_speed_gain = float(lane_speed_gain)
        self.lane_cbf_alpha = float(lane_cbf_alpha)
        self.event_margin = float(event_margin)
        self.event_lane_margin = float(event_lane_margin)
        self.event_speed_floor = float(event_speed_floor)

    def enforce(
        self,
        env,
        commands: List[VehicleCommand],
        scheduler=None,
        step: int = 0,
    ) -> Tuple[List[VehicleCommand], Dict[str, float]]:
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

        for idx, command in enumerate(safe_commands):
            veh = env.vehicles[idx]
            lane_idx = int(env._lane_idx[idx])
            direction = getattr(veh, "direction", "u")
            y = float(veh.position[1])
            ego_speed = float(getattr(veh, "velocity", 0.0))

            original_accel = float(command.accel)
            bounded_accel = float(np.clip(original_accel, self.max_decel, self.max_accel))
            if not np.isclose(bounded_accel, original_accel):
                interventions += 1
                accel_clips += 1
                command.accel = bounded_accel
            else:
                command.accel = original_accel

            original_speed = float(command.target_speed)
            bounded_speed = float(np.clip(original_speed, 0.0, self.max_speed))
            if not np.isclose(bounded_speed, original_speed):
                interventions += 1
                speed_clips += 1
                command.target_speed = bounded_speed
            else:
                command.target_speed = original_speed

            front_idx, gap = self._nearest_front_vehicle(env, idx, lane_idx)
            if front_idx is not None and np.isfinite(gap):
                front_speed = float(getattr(env.vehicles[front_idx], "velocity", 0.0))
                projected_accel, projected_speed, barrier, adjusted, hard_brake = self._project_longitudinal_command(
                    command=command,
                    ego_speed=ego_speed,
                    front_speed=front_speed,
                    gap=gap,
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
        closing_speed = max(float(ego_speed) - float(front_speed), 0.0)
        min_gap = self.min_gap + self.closing_gap_gain * closing_speed
        barrier = float(gap - (min_gap + headway_time * max(float(ego_speed), 0.0)))
        accel_upper = float((float(front_speed) - float(ego_speed) + self.cbf_alpha * barrier) / max(headway_time, 1e-6))
        projected_accel = float(np.clip(min(float(command.accel), accel_upper), self.max_decel, self.max_accel))
        speed_cap = float(np.clip(float(front_speed) + self.cbf_alpha * barrier, 0.0, self.max_speed))
        projected_speed = float(np.clip(min(float(command.target_speed), speed_cap), 0.0, self.max_speed))
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
        speed_scale = float(np.clip(scheduler.speed_scale(direction, lane_idx, y, step, margin=self.event_margin), 0.0, 1.0))
        pressure = float(np.clip(scheduler.event_pressure(direction, y, step), 0.0, 1.5))
        speed_cap = float(np.clip(max(self.event_speed_floor, self.max_speed * max(speed_scale, 0.18)), 0.0, self.max_speed))
        accel_cap = float(np.clip(-1.2 - 1.5 * pressure - 0.25 * max(ego_speed - speed_cap, 0.0), self.max_decel, self.max_accel))
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
        tau_from_gap = 0.45 * (desired_gap - self.min_gap) / max(float(ego_speed), 1.0)
        tau_from_scale = self.nominal_time_headway * max(float(getattr(command, "headway_scale", 1.0)), 0.7)
        headway_time = max(self.min_time_headway, tau_from_gap, tau_from_scale)
        return float(np.clip(headway_time, self.min_time_headway, self.max_time_headway))

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
        barrier = float(min(front_barrier, rear_barrier))
        allow = barrier >= 0.0
        speed_scale = 1.0
        if allow and np.isfinite(barrier):
            speed_scale = float(np.clip(0.40 + self.lane_cbf_alpha * barrier / max(self.lane_change_gap, 1e-6), 0.40, 1.0))
        return {
            "allow": bool(allow),
            "barrier": barrier,
            "blocked_lane": False,
            "speed_scale": speed_scale,
            "front_idx": front_idx,
            "front_gap": float(front_gap),
        }

    def _nearest_front_vehicle(self, env, idx: int, lane_idx: int) -> Tuple[Optional[int], float]:
        veh = env.vehicles[idx]
        direction = getattr(veh, "direction", "u")
        y = float(veh.position[1])
        candidates: List[Tuple[float, int]] = []
        for other_idx, other in enumerate(env.vehicles):
            if other_idx == idx:
                continue
            if getattr(other, "direction", None) != direction:
                continue
            if int(env._lane_idx[other_idx]) != lane_idx:
                continue
            y_other = float(other.position[1])
            if direction == "u" and y_other > y:
                candidates.append((y_other - y, other_idx))
            elif direction == "d" and y_other < y:
                candidates.append((y - y_other, other_idx))
        if not candidates:
            return None, float("inf")
        gap, other_idx = min(candidates, key=lambda item: item[0])
        return int(other_idx), float(gap)

    def _nearest_rear_vehicle(self, env, idx: int, lane_idx: int) -> Tuple[Optional[int], float]:
        veh = env.vehicles[idx]
        direction = getattr(veh, "direction", "u")
        y = float(veh.position[1])
        candidates: List[Tuple[float, int]] = []
        for other_idx, other in enumerate(env.vehicles):
            if other_idx == idx:
                continue
            if getattr(other, "direction", None) != direction:
                continue
            if int(env._lane_idx[other_idx]) != lane_idx:
                continue
            y_other = float(other.position[1])
            if direction == "u" and y_other < y:
                candidates.append((y - y_other, other_idx))
            elif direction == "d" and y_other > y:
                candidates.append((y_other - y, other_idx))
        if not candidates:
            return None, float("inf")
        gap, other_idx = min(candidates, key=lambda item: item[0])
        return int(other_idx), float(gap)

    def _finite_min(self, values: List[float], default: float = 0.0) -> float:
        finite = [float(value) for value in values if np.isfinite(value)]
        if not finite:
            return float(default)
        return float(min(finite))


class NoOpSafetyShield(SafetyShield):
    """A no-op shield used for ablation experiments."""

    def enforce(self, env, commands: List[VehicleCommand], scheduler=None, step: int = 0):  # type: ignore[override]
        summary = {
            "interventions": 0.0,
            "accel_clips": 0.0,
            "speed_clips": 0.0,
            "lane_change_blocks": 0.0,
            "emergency_brakes": 0.0,
            "event_speed_reductions": 0.0,
            "cbf_adjustments": 0.0,
            "cbf_longitudinal_clips": 0.0,
            "cbf_lateral_clips": 0.0,
            "cbf_blocked_lane_avoid": 0.0,
            "cbf_min_barrier": 0.0,
            "cbf_lane_barrier": 0.0,
        }
        return [VehicleCommand(**vars(command)) for command in commands], summary
