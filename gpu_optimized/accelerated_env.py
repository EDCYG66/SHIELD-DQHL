"""AcceleratedFormationEnv — subclass with vectorized energy/collision + clip elimination."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from formation.formation_env import FormationExperimentEnv

from .cupy_kernels import broad_phase_collision_pairs, vectorized_energy_kj


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


class AcceleratedFormationEnv(FormationExperimentEnv):
    """FormationExperimentEnv with vectorized hot paths and no np.clip overhead."""

    def _apply_commands_point_mass(self, commands) -> Dict[str, int]:
        lane_changes = 0
        lane_changes_cav = 0
        lane_changes_hv = 0
        dt = float(getattr(self.env, "timestep", 0.01))
        lane_positions_up = self.env.true_up_lanes
        lane_positions_down = self.env.true_down_lanes
        clip_margin = 0.5
        lane_commit_x_tol = float(getattr(self.controller, "lane_commit_x_tol", 0.35))
        max_speed = self.shield.max_speed

        for idx, command in enumerate(commands):
            veh = self.env.vehicles[idx]
            veh.velocity = _clamp(veh.velocity + command.accel * dt, 0.0, max_speed)
            current_lane = int(self.env._lane_idx[idx])

            direction = getattr(veh, "direction", "u")
            lane_positions = lane_positions_up if direction == "u" else lane_positions_down
            lane_target_x = float(lane_positions[current_lane])
            min_lane_x = float(min(lane_positions)) - clip_margin
            max_lane_x = float(max(lane_positions)) + clip_margin

            if str(getattr(command, "lateral_mode", "")) == "lane_keep":
                self._abort_lane_change(idx, current_lane)
                hold_speed, hold_error = self.controller.trajectory_tracker.hold(float(veh.position[0]), lane_target_x)
                command.target_lane = None
                command.target_x = lane_target_x
                command.lateral_speed = hold_speed
                command.lateral_error = hold_error
                command.trajectory_progress = 1.0
                command.lane_change_active = False

            target_lane = command.target_lane
            if bool(getattr(command, "lane_change_active", False)) and target_lane is not None:
                target_lane = int(target_lane)
                if not (0 <= target_lane < len(lane_positions)):
                    target_lane = None
                else:
                    lane_target_x = float(lane_positions[target_lane])
            elif target_lane is not None:
                target_lane = int(target_lane)
                if not (0 <= target_lane < len(lane_positions)):
                    target_lane = None
                else:
                    lane_target_x = float(lane_positions[target_lane])

            target_x = lane_target_x if command.target_x is None else float(command.target_x)
            current_x = float(veh.position[0])
            proposed_x = current_x + float(getattr(command, "lateral_speed", 0.0)) * dt
            if (target_x - current_x) * (target_x - proposed_x) <= 0.0:
                proposed_x = target_x
            elif abs(target_x - proposed_x) < 0.03:
                proposed_x = target_x
            veh.position[0] = _clamp(proposed_x, min_lane_x, max_lane_x)

            if target_lane is not None and target_lane != current_lane:
                progress = float(getattr(command, "trajectory_progress", 0.0))
                near_target = abs(float(veh.position[0]) - float(lane_target_x)) <= lane_commit_x_tol
                if progress >= 0.999 and near_target:
                    self.env._lane_idx[idx] = target_lane
                    setattr(veh, "last_lane_change_step", int(self.step_count))
                    lane_changes += 1
                    if bool(getattr(veh, "is_cav", False)):
                        lane_changes_cav += 1
                    else:
                        lane_changes_hv += 1

        self.lane_change_count += lane_changes
        self.lane_change_count_cav += lane_changes_cav
        self.lane_change_count_hv += lane_changes_hv
        return {"total": lane_changes, "cav": lane_changes_cav, "hv": lane_changes_hv}

    def _compute_reward(self, fields, info):
        reward = super()._compute_reward(fields, info)
        return reward

    def _collect_energy_stats(self, commands) -> Dict[str, float]:
        vehicles = getattr(self.env, "vehicles", [])
        n = len(vehicles)
        if n == 0:
            return super()._collect_energy_stats(commands)

        dt = max(float(getattr(self.env, "timestep", 0.01)), 1e-6)

        speeds = np.empty(n, dtype=np.float64)
        accels = np.empty(n, dtype=np.float64)
        lengths = np.empty(n, dtype=np.float64)
        is_cav = np.empty(n, dtype=bool)
        for i, veh in enumerate(vehicles):
            speeds[i] = max(float(getattr(veh, "velocity", 0.0)), 0.0)
            cmd = commands[i] if i < len(commands) else None
            accels[i] = float(getattr(cmd, "accel", 0.0)) if cmd is not None else 0.0
            lengths[i] = float(getattr(veh, "vehicle_length", 4.8))
            is_cav[i] = bool(getattr(veh, "is_cav", False))

        total_kj, cav_kj, hv_kj = vectorized_energy_kj(speeds, accels, lengths, is_cav, dt)

        self.total_energy_kj += total_kj
        self.total_energy_cav_kj += cav_kj
        self.total_energy_hv_kj += hv_kj
        nveh = max(1, n)
        return {
            "energy_step_kj": float(total_kj),
            "energy_step_cav_kj": float(cav_kj),
            "energy_step_hv_kj": float(hv_kj),
            "energy_cum_kj": float(self.total_energy_kj),
            "energy_cum_cav_kj": float(self.total_energy_cav_kj),
            "energy_cum_hv_kj": float(self.total_energy_hv_kj),
            "energy_step_per_vehicle_kj": float(total_kj) / float(nveh),
        }

    def _collect_collision_stats(self) -> Dict[str, float]:
        vehicles = getattr(self.env, "vehicles", [])
        if len(vehicles) < 2:
            self._prev_collision_pairs = set()
            return {
                "collision_count": 0.0,
                "collision_pairs_active": 0.0,
                "collision_vehicles_active": 0.0,
                "collision_occurred": 0.0,
                "collision_count_cum": float(self.total_collision_count),
                "collision_step_count_cum": float(self.total_collision_steps),
            }

        lane_width = float(getattr(self.env, "lane_width", 3.5))
        default_width = float(self.collision_width) if self.collision_width is not None else min(2.2, 0.65 * lane_width)
        default_length = float(max(self.collision_length, 0.1))

        collision_boxes = [self._vehicle_collision_box(veh, default_length, default_width) for veh in vehicles]

        n = len(collision_boxes)
        centers = np.empty((n, 2), dtype=np.float64)
        radii = np.empty(n, dtype=np.float64)
        for i, box in enumerate(collision_boxes):
            centers[i] = box["center"]
            radii[i] = box["radius"]

        candidate_pairs = broad_phase_collision_pairs(centers, radii)

        active_pairs: set = set()
        for i, j in candidate_pairs:
            if self._oriented_boxes_overlap(collision_boxes[i], collision_boxes[j]):
                active_pairs.add((i, j))

        new_pairs = active_pairs.difference(self._prev_collision_pairs)
        if active_pairs:
            self.total_collision_steps += 1
        self.total_collision_count += len(new_pairs)
        self._prev_collision_pairs = active_pairs

        collision_vehicles = set()
        for i, j in active_pairs:
            collision_vehicles.add(int(i))
            collision_vehicles.add(int(j))

        return {
            "collision_count": float(len(new_pairs)),
            "collision_pairs_active": float(len(active_pairs)),
            "collision_vehicles_active": float(len(collision_vehicles)),
            "collision_occurred": 1.0 if active_pairs else 0.0,
            "collision_count_cum": float(self.total_collision_count),
            "collision_step_count_cum": float(self.total_collision_steps),
        }

    def _collect_event_stats(self) -> Dict[str, float]:
        vehicles = getattr(self.env, "vehicles", [])
        n = len(vehicles)
        if n == 0:
            return super()._collect_event_stats()

        dirs = np.asarray([str(getattr(veh, "direction", "u")) for veh in vehicles], dtype=object)
        ys = np.asarray([float(veh.position[1]) for veh in vehicles], dtype=np.float64)
        lanes = np.asarray(getattr(self.env, "_lane_idx", np.zeros(n, dtype=int)), dtype=np.int32)
        step = int(self.step_count)
        active_events = self.scheduler.active_events(step) if self.scheduler.has_active_event(step) else ()

        event_zone = np.zeros(n, dtype=bool)
        blocked = np.zeros(n, dtype=bool)
        dist_up = []
        dist_down = []
        for event in active_events:
            y_mask = (ys >= float(event.y_start)) & (ys <= float(event.y_end))
            for direction in event.affected_directions:
                dir_mask = dirs == direction
                mask = y_mask & dir_mask
                event_zone |= mask
                if not bool(getattr(event, "advisory_only", False)):
                    blocked_lanes = np.asarray(event.blocked_lanes.get(direction, ()), dtype=np.int32)
                    if blocked_lanes.size:
                        margin_mask = (
                            (ys >= float(event.y_start) - 15.0)
                            & (ys <= float(event.y_end) + 15.0)
                            & dir_mask
                            & np.isin(lanes, blocked_lanes)
                        )
                        blocked |= margin_mask

                direction_ys = ys[dir_mask]
                if direction_ys.size == 0:
                    continue
                inside = (direction_ys >= float(event.y_start)) & (direction_ys <= float(event.y_end))
                distances = np.full(direction_ys.shape, np.inf, dtype=np.float64)
                distances[inside] = 0.0
                if direction == "u":
                    ahead = direction_ys < float(event.y_start)
                    distances[ahead] = float(event.y_start) - direction_ys[ahead]
                    finite = distances[np.isfinite(distances)]
                    if finite.size:
                        dist_up.extend(finite.tolist())
                elif direction == "d":
                    ahead = direction_ys > float(event.y_end)
                    distances[ahead] = direction_ys[ahead] - float(event.y_end)
                    finite = distances[np.isfinite(distances)]
                    if finite.size:
                        dist_down.extend(finite.tolist())

        bounds_up = self._direction_event_bounds("u")
        bounds_down = self._direction_event_bounds("d")
        up_mask = dirs == "u"
        down_mask = dirs == "d"
        if bounds_up is not None:
            entered = up_mask & (ys >= bounds_up[0]) & (ys <= bounds_up[1])
            self._event_entered_up[:n] |= entered
            self._event_passed_up[:n] |= self._event_entered_up[:n] & up_mask & (ys >= bounds_up[1])
        if bounds_down is not None:
            entered = down_mask & (ys >= bounds_down[0]) & (ys <= bounds_down[1])
            self._event_entered_down[:n] |= entered
            self._event_passed_down[:n] |= self._event_entered_down[:n] & down_mask & (ys <= bounds_down[0])

        n_up = max(1, int(np.sum(up_mask)))
        n_down = max(1, int(np.sum(down_mask)))
        return {
            "event_zone_vehicle_count": float(np.sum(event_zone)),
            "blocked_lane_vehicle_count": float(np.sum(blocked)),
            "avg_event_distance_up": float(np.mean(dist_up)) if dist_up else 0.0,
            "avg_event_distance_down": float(np.mean(dist_down)) if dist_down else 0.0,
            "passed_event_count_up": float(np.sum(self._event_passed_up)),
            "passed_event_count_down": float(np.sum(self._event_passed_down)),
            "passed_event_ratio_up": float(np.sum(self._event_passed_up)) / float(n_up),
            "passed_event_ratio_down": float(np.sum(self._event_passed_down)) / float(n_down),
        }
