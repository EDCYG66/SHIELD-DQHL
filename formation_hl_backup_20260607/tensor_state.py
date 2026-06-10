"""Tensor-friendly snapshots of the object-based highway environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


def _pad_ragged(indices: list[list[int]], fill_value: int = -1) -> np.ndarray:
    width = max((len(row) for row in indices), default=0)
    if width == 0:
        return np.empty((len(indices), 0), dtype=np.int32)
    out = np.full((len(indices), width), fill_value, dtype=np.int32)
    for row_idx, row in enumerate(indices):
        if row:
            out[row_idx, : len(row)] = np.asarray(row, dtype=np.int32)
    return out


@dataclass(slots=True)
class TensorizedTrafficState:
    """Structured tensor snapshot extracted from the current object graph."""

    positions_xy: np.ndarray
    velocities: np.ndarray
    lane_indices: np.ndarray
    direction_sign: np.ndarray
    is_cav: np.ndarray
    vehicle_length: np.ndarray
    desired_speed: np.ndarray
    heading_error: np.ndarray
    yaw: np.ndarray
    destinations: np.ndarray
    neighbors: np.ndarray
    topology_degrees: np.ndarray
    topology_depth: np.ndarray
    topology_struct_features: np.ndarray
    leader_indices: np.ndarray
    leader_flags: np.ndarray
    v2i_serving_idx: np.ndarray
    bs_positions_xy: np.ndarray
    metadata: Dict[str, Any]

    @property
    def n_vehicles(self) -> int:
        return int(self.positions_xy.shape[0])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "positions_xy": self.positions_xy.copy(),
            "velocities": self.velocities.copy(),
            "lane_indices": self.lane_indices.copy(),
            "direction_sign": self.direction_sign.copy(),
            "is_cav": self.is_cav.copy(),
            "vehicle_length": self.vehicle_length.copy(),
            "desired_speed": self.desired_speed.copy(),
            "heading_error": self.heading_error.copy(),
            "yaw": self.yaw.copy(),
            "destinations": self.destinations.copy(),
            "neighbors": self.neighbors.copy(),
            "topology_degrees": self.topology_degrees.copy(),
            "topology_depth": self.topology_depth.copy(),
            "topology_struct_features": self.topology_struct_features.copy(),
            "leader_indices": self.leader_indices.copy(),
            "leader_flags": self.leader_flags.copy(),
            "v2i_serving_idx": self.v2i_serving_idx.copy(),
            "bs_positions_xy": self.bs_positions_xy.copy(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_env(cls, env) -> "TensorizedTrafficState":
        vehicles = list(getattr(env, "vehicles", []))
        n_vehicles = len(vehicles)

        positions_xy = np.asarray(
            [[float(v.position[0]), float(v.position[1])] for v in vehicles],
            dtype=np.float32,
        ).reshape(n_vehicles, 2)
        velocities = np.asarray([float(getattr(v, "velocity", 0.0)) for v in vehicles], dtype=np.float32)
        lane_indices = np.asarray(getattr(env, "_lane_idx", np.zeros(n_vehicles, dtype=int)), dtype=np.int32).copy()
        direction_sign = np.asarray(
            [1.0 if str(getattr(v, "direction", "u")).lower() == "u" else -1.0 for v in vehicles],
            dtype=np.float32,
        )
        is_cav = np.asarray([bool(getattr(v, "is_cav", False)) for v in vehicles], dtype=np.bool_)
        vehicle_length = np.asarray([float(getattr(v, "vehicle_length", 4.8)) for v in vehicles], dtype=np.float32)
        desired_speed = np.asarray([float(getattr(v, "desired_speed", getattr(v, "velocity", 0.0))) for v in vehicles], dtype=np.float32)
        heading_error = np.asarray([float(getattr(v, "heading_error", 0.0)) for v in vehicles], dtype=np.float32)
        yaw = np.asarray([float(getattr(v, "yaw", 0.0)) for v in vehicles], dtype=np.float32)

        topology_arrays = getattr(env, "topology_state_arrays", None)
        if callable(topology_arrays):
            topology = topology_arrays()
            destinations = np.asarray(topology.get("destinations", np.full((n_vehicles, 0), -1)), dtype=np.int32)
            neighbors = np.asarray(topology.get("neighbors", np.full((n_vehicles, 0), -1)), dtype=np.int32)
            topology_degrees = np.asarray(topology.get("degrees", np.zeros((n_vehicles,))), dtype=np.int32)
            topology_depth = np.asarray(topology.get("depth", np.zeros((n_vehicles,))), dtype=np.int32)
            topology_struct_features = np.asarray(topology.get("struct_features", np.zeros((n_vehicles, 2))), dtype=np.float32)
            leader_indices = np.asarray(topology.get("leader_indices", np.full((2,), -1)), dtype=np.int32)
            leader_flags = np.asarray(topology.get("leader_flags", np.zeros((n_vehicles, 2), dtype=np.bool_)), dtype=np.bool_)
        else:
            destinations = _pad_ragged(
                [list(int(x) for x in getattr(v, "destinations", [])) for v in vehicles],
                fill_value=-1,
            )
            neighbors = _pad_ragged(
                [list(int(x) for x in getattr(v, "neighbors", [])) for v in vehicles],
                fill_value=-1,
            )
            topology_degrees = np.asarray([len(getattr(v, "neighbors", [])) for v in vehicles], dtype=np.int32)
            topology_depth = np.asarray(getattr(env, "depth", np.zeros(n_vehicles, dtype=int)), dtype=np.int32).copy()
            topology_struct_features = np.asarray(getattr(env, "struct_features_per_vehicle", np.zeros((n_vehicles, 2))), dtype=np.float32).reshape(n_vehicles, 2)
            leader_indices = np.asarray([
                -1 if getattr(env, "leader_idx_up", None) is None else int(getattr(env, "leader_idx_up")),
                -1 if getattr(env, "leader_idx_down", None) is None else int(getattr(env, "leader_idx_down")),
            ], dtype=np.int32)
            leader_flags = np.zeros((n_vehicles, 2), dtype=np.bool_)
            if 0 <= leader_indices[0] < n_vehicles:
                leader_flags[leader_indices[0], 0] = True
            if 0 <= leader_indices[1] < n_vehicles:
                leader_flags[leader_indices[1], 1] = True

        v2i_serving_idx = np.asarray(getattr(env, "v2i_serving_idx", np.full(n_vehicles, -1, dtype=int)), dtype=np.int32).copy()
        bs_positions_xy = np.asarray(getattr(env, "bs_positions", []), dtype=np.float32).reshape(-1, 2)

        metadata: Dict[str, Any] = {
            "topology_type": str(getattr(env, "topology_type", "")),
            "topology_epoch": int(getattr(env, "topology_epoch", 0)),
            "n_up": int(getattr(env, "n_up", 0)),
            "n_down": int(getattr(env, "n_down", 0)),
            "lanes_per_dir": int(getattr(env, "lanes_per_dir", 0)),
            "width": float(getattr(env, "width", 0.0)),
            "height": float(getattr(env, "height", 0.0)),
            "base_y": float(getattr(env, "base_y", 0.0)),
            "timestep": float(getattr(env, "timestep", 0.0)),
        }
        return cls(
            positions_xy=positions_xy,
            velocities=velocities,
            lane_indices=lane_indices,
            direction_sign=direction_sign,
            is_cav=is_cav,
            vehicle_length=vehicle_length,
            desired_speed=desired_speed,
            heading_error=heading_error,
            yaw=yaw,
            destinations=destinations,
            neighbors=neighbors,
            topology_degrees=topology_degrees,
            topology_depth=topology_depth,
            topology_struct_features=topology_struct_features,
            leader_indices=leader_indices,
            leader_flags=leader_flags,
            v2i_serving_idx=v2i_serving_idx,
            bs_positions_xy=bs_positions_xy,
            metadata=metadata,
        )
