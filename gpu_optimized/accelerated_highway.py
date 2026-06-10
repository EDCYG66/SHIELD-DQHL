"""Monkey-patch HighwayTopoEnv with numpy-vectorized hot methods.

Pure numpy broadcasting for n=30 vehicles (no CuPy — transfer overhead > compute at this scale).
"""

from __future__ import annotations

import numpy as np

from .cupy_kernels import (
    build_sorted_lane_index,
    extract_vehicle_arrays,
    vectorized_position_update,
)

_PATCHED = False


def _update_distance_numpy(self) -> None:
    """Numpy broadcasting pairwise distance matrix."""
    n = self.n_Veh
    if n == 0:
        return
    positions = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        positions[i, 0] = float(self.vehicles[i].position[0])
        positions[i, 1] = float(self.vehicles[i].position[1])
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    np.copyto(self.Distance[:n, :n], dist)


def _renew_positions_vectorized(self) -> None:
    """Vectorized position update + numpy distance matrix."""
    n = self.n_Veh
    if n == 0:
        return

    positions_y = np.array([float(v.position[1]) for v in self.vehicles], dtype=np.float64)
    directions = np.array([str(getattr(v, "direction", "u")) for v in self.vehicles], dtype="U1")
    velocities = np.array([float(getattr(v, "velocity", 0.0)) for v in self.vehicles], dtype=np.float64)

    new_y, any_moved = vectorized_position_update(
        positions_y=positions_y,
        directions=directions,
        velocities=velocities,
        move_speed=float(self.move_speed),
        timestep=float(self.timestep),
        height=float(self.height),
        base_y=float(self.base_y),
        jitter_std=float(self.jitter_std),
        rng=getattr(self, "rng", None),
    )

    for i, v in enumerate(self.vehicles):
        v.position[1] = float(new_y[i])

    if any_moved:
        self._update_distance()
        if self.leader_dynamic:
            old_up, old_dn = self.leader_idx_up, self.leader_idx_down
            self._pick_leaders_by_policy()
            if self.leader_idx_up != old_up or self.leader_idx_down != old_dn:
                self._build_topology_two_clusters()
        self._assoc_v2i(initial=False)


def install_highway_patches() -> None:
    """Patch HighwayTopoEnv class methods with numpy-optimized versions."""
    global _PATCHED
    if _PATCHED:
        return

    from communication.highway_environment import HighwayTopoEnv

    HighwayTopoEnv._orig_update_distance = HighwayTopoEnv._update_distance
    HighwayTopoEnv._update_distance = _update_distance_numpy

    HighwayTopoEnv._orig_renew_positions = HighwayTopoEnv.renew_positions
    HighwayTopoEnv.renew_positions = _renew_positions_vectorized

    _PATCHED = True


def uninstall_highway_patches() -> None:
    """Restore original HighwayTopoEnv methods."""
    global _PATCHED
    if not _PATCHED:
        return

    from communication.highway_environment import HighwayTopoEnv

    if hasattr(HighwayTopoEnv, "_orig_update_distance"):
        HighwayTopoEnv._update_distance = HighwayTopoEnv._orig_update_distance
    if hasattr(HighwayTopoEnv, "_orig_renew_positions"):
        HighwayTopoEnv.renew_positions = HighwayTopoEnv._orig_renew_positions

    _PATCHED = False
