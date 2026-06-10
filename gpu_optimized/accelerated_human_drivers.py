"""Accelerated HumanDriverController with sorted-index neighbor search + clip elimination."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from formation.human_driver_models import HumanDriverController
from formation.low_level_controller import VehicleCommand

from .cupy_kernels import (
    build_sorted_lane_index,
    extract_vehicle_arrays,
    nearest_front_from_sorted,
    nearest_rear_from_sorted,
)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


class AcceleratedHumanDriverController(HumanDriverController):
    """HumanDriverController with O(log n) neighbor search and no np.clip overhead."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lane_index: Optional[Dict] = None
        self._veh_arrays: Optional[Dict[str, np.ndarray]] = None

    def compute_commands(self, env, step: int, controlled_indices):
        n_lanes = int(getattr(env, "lanes_per_dir", 4))
        veh_arrays = extract_vehicle_arrays(env.vehicles, env._lane_idx)
        self._lane_index = build_sorted_lane_index(
            veh_arrays["positions"][:, 1],
            veh_arrays["directions"],
            veh_arrays["lane_idx"],
            n_lanes,
        )
        self._veh_arrays = veh_arrays

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
                accel=_clamp(float(accel), -max(6.0, idm.comfortable_brake), idm.max_accel),
                target_speed=_clamp(float(idm.desired_speed), 0.0, self.max_speed),
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

        self._lane_index = None
        self._veh_arrays = None
        return commands

    def _nearest_front_vehicle(
        self,
        env,
        idx: int,
        lane_idx: int,
        *,
        max_distance: float = float("inf"),
        exclude_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[Optional[int], float]:
        if self._lane_index is None or self._veh_arrays is None:
            return super()._nearest_front_vehicle(
                env, idx, lane_idx, max_distance=max_distance, exclude_indices=exclude_indices,
            )

        direction = str(self._veh_arrays["directions"][idx])
        y = float(self._veh_arrays["positions"][idx, 1])
        key = (direction, int(lane_idx))
        if key not in self._lane_index:
            return None, float("inf")

        sorted_y, sorted_idx = self._lane_index[key]
        excluded = {int(idx)}
        if exclude_indices is not None:
            excluded.update(int(i) for i in exclude_indices)

        if direction == "u":
            pos = np.searchsorted(sorted_y, y, side="right")
            for k in range(pos, len(sorted_y)):
                cand = int(sorted_idx[k])
                if cand in excluded:
                    continue
                gap = float(sorted_y[k] - y)
                if np.isfinite(max_distance) and gap > max_distance:
                    break
                return cand, gap
        else:
            pos = np.searchsorted(sorted_y, y, side="left")
            for k in range(pos - 1, -1, -1):
                cand = int(sorted_idx[k])
                if cand in excluded:
                    continue
                gap = float(y - sorted_y[k])
                if np.isfinite(max_distance) and gap > max_distance:
                    break
                return cand, gap
        return None, float("inf")

    def _nearest_rear_vehicle(
        self,
        env,
        idx: int,
        lane_idx: int,
        *,
        max_distance: float = float("inf"),
        exclude_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[Optional[int], float]:
        if self._lane_index is None or self._veh_arrays is None:
            return super()._nearest_rear_vehicle(
                env, idx, lane_idx, max_distance=max_distance, exclude_indices=exclude_indices,
            )

        direction = str(self._veh_arrays["directions"][idx])
        y = float(self._veh_arrays["positions"][idx, 1])
        key = (direction, int(lane_idx))
        if key not in self._lane_index:
            return None, float("inf")

        sorted_y, sorted_idx = self._lane_index[key]
        excluded = {int(idx)}
        if exclude_indices is not None:
            excluded.update(int(i) for i in exclude_indices)

        if direction == "u":
            pos = np.searchsorted(sorted_y, y, side="left")
            for k in range(pos - 1, -1, -1):
                cand = int(sorted_idx[k])
                if cand in excluded:
                    continue
                gap = float(y - sorted_y[k])
                if np.isfinite(max_distance) and gap > max_distance:
                    break
                return cand, gap
        else:
            pos = np.searchsorted(sorted_y, y, side="right")
            for k in range(pos, len(sorted_y)):
                cand = int(sorted_idx[k])
                if cand in excluded:
                    continue
                gap = float(sorted_y[k] - y)
                if np.isfinite(max_distance) and gap > max_distance:
                    break
                return cand, gap
        return None, float("inf")
