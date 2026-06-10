"""CuPy GPU kernels with numpy fallbacks for vehicle simulation hot paths."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import GPU_TRANSFER_THRESHOLD

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


def _init_cupy_pool(limit_bytes: int = 4 * 1024 ** 3) -> None:
    if HAS_CUPY:
        cp.get_default_memory_pool().set_limit(size=limit_bytes)


# ---------------------------------------------------------------------------
# 1. Pairwise distance matrix  (replaces highway_environment._update_distance)
# ---------------------------------------------------------------------------

def pairwise_distance_gpu(positions_xy: np.ndarray) -> np.ndarray:
    """(n,2) positions -> (n,n) Euclidean distance matrix."""
    n = positions_xy.shape[0]
    if HAS_CUPY and n >= GPU_TRANSFER_THRESHOLD:
        pos = cp.asarray(positions_xy, dtype=cp.float32)
        dx = pos[:, 0:1] - pos[:, 0:1].T
        dy = pos[:, 1:2] - pos[:, 1:2].T
        dist = cp.sqrt(dx * dx + dy * dy)
        return cp.asnumpy(dist)
    diff = positions_xy[:, None, :] - positions_xy[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1)).astype(np.float64)


# ---------------------------------------------------------------------------
# 2. Vectorized energy computation  (replaces _collect_energy_stats loop)
# ---------------------------------------------------------------------------

_GRAVITY = 9.81
_RHO_AIR = 1.225


def vectorized_energy_kj(
    speeds: np.ndarray,
    accels: np.ndarray,
    lengths: np.ndarray,
    is_cav: np.ndarray,
    dt: float,
) -> Tuple[float, float, float]:
    """Batch energy computation. Returns (total_kj, cav_kj, hv_kj)."""
    is_truck = lengths > 7.5
    mass = np.where(is_truck, 8500.0, 1600.0)
    c_rr = np.where(is_truck, 0.010, 0.012)
    cd_area = np.where(is_truck, 5.40, 0.72)

    rolling = c_rr * mass * _GRAVITY
    drag = 0.5 * _RHO_AIR * cd_area * speeds * speeds
    inertial = mass * np.maximum(accels, 0.0)
    power_w = np.maximum((rolling + drag + inertial) * speeds, 0.0)
    energy_kj = (power_w * dt) / 1000.0

    total = float(energy_kj.sum())
    cav_kj = float(energy_kj[is_cav].sum()) if is_cav.any() else 0.0
    hv_kj = float(energy_kj[~is_cav].sum()) if (~is_cav).any() else 0.0
    return total, cav_kj, hv_kj


# ---------------------------------------------------------------------------
# 3. Collision broad-phase  (replaces O(n^2) pair loop)
# ---------------------------------------------------------------------------

def broad_phase_collision_pairs(
    centers: np.ndarray, radii: np.ndarray
) -> List[Tuple[int, int]]:
    """Return candidate (i,j) pairs that pass bounding-sphere broad phase."""
    n = centers.shape[0]
    if n < 2:
        return []

    if HAS_CUPY and n >= GPU_TRANSFER_THRESHOLD:
        c = cp.asarray(centers, dtype=cp.float32)
        r = cp.asarray(radii, dtype=cp.float32)
        dx = c[:, 0:1] - c[:, 0:1].T
        dy = c[:, 1:2] - c[:, 1:2].T
        dist_sq = dx * dx + dy * dy
        thresh_sq = (r[:, None] + r[None, :]) ** 2
        mask = dist_sq <= thresh_sq
        mask_np = cp.asnumpy(mask)
    else:
        diff = centers[:, None, :] - centers[None, :, :]
        dist_sq = (diff * diff).sum(axis=-1)
        thresh_sq = (radii[:, None] + radii[None, :]) ** 2
        mask_np = dist_sq <= thresh_sq

    pairs = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if mask_np[i, j]:
                pairs.append((i, j))
    return pairs


# ---------------------------------------------------------------------------
# 4. Vectorized position update  (replaces renew_positions per-vehicle loop)
# ---------------------------------------------------------------------------

def vectorized_position_update(
    positions_y: np.ndarray,
    directions: np.ndarray,
    velocities: np.ndarray,
    move_speed: float,
    timestep: float,
    height: float,
    base_y: float,
    jitter_std: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, bool]:
    """Batch update Y positions. Returns (new_y, any_moved)."""
    n = len(positions_y)
    if move_speed > 0:
        dy = np.full(n, move_speed, dtype=np.float64)
    else:
        dy = velocities * timestep

    skip = (dy <= 0.0) & (jitter_std <= 0)
    dy[skip] = 0.0

    new_y = positions_y.copy()
    up = directions == "u"
    down = directions == "d"
    other = ~up & ~down

    y_max = height - base_y
    new_y[up] = np.minimum(new_y[up] + dy[up], y_max)
    new_y[down] = np.maximum(new_y[down] - dy[down], base_y)
    new_y[other] = np.minimum(new_y[other] + dy[other], y_max)

    if jitter_std > 0 and rng is not None:
        jitter = rng.normal(0.0, jitter_std, size=n)
        new_y += jitter
        np.clip(new_y, base_y, y_max, out=new_y)

    any_moved = bool(np.any(new_y != positions_y))
    return new_y, any_moved


# ---------------------------------------------------------------------------
# 5. Sorted lane index + O(log n) neighbor search
# ---------------------------------------------------------------------------

def build_sorted_lane_index(
    positions_y: np.ndarray,
    directions: np.ndarray,
    lane_indices: np.ndarray,
    n_lanes: int,
) -> Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]]:
    """Build sorted (y, vehicle_idx) arrays per (direction, lane)."""
    index: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
    for d in ("u", "d"):
        d_mask = directions == d
        for lane in range(n_lanes):
            l_mask = lane_indices == lane
            both = d_mask & l_mask
            idxs = np.where(both)[0]
            if len(idxs) == 0:
                index[(d, lane)] = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64))
                continue
            ys = positions_y[idxs]
            order = np.argsort(ys)
            index[(d, lane)] = (ys[order], idxs[order])
    return index


def nearest_front_from_sorted(
    sorted_y: np.ndarray,
    sorted_idx: np.ndarray,
    query_y: float,
    query_idx: int,
    direction: str,
) -> Tuple[Optional[int], float]:
    """O(log n) nearest-front lookup from sorted lane arrays."""
    if len(sorted_y) == 0:
        return None, float("inf")

    if direction == "u":
        pos = np.searchsorted(sorted_y, query_y, side="right")
        for k in range(pos, len(sorted_y)):
            if int(sorted_idx[k]) != query_idx:
                return int(sorted_idx[k]), float(sorted_y[k] - query_y)
    else:
        pos = np.searchsorted(sorted_y, query_y, side="left")
        for k in range(pos - 1, -1, -1):
            if int(sorted_idx[k]) != query_idx:
                return int(sorted_idx[k]), float(query_y - sorted_y[k])
    return None, float("inf")


def nearest_rear_from_sorted(
    sorted_y: np.ndarray,
    sorted_idx: np.ndarray,
    query_y: float,
    query_idx: int,
    direction: str,
) -> Tuple[Optional[int], float]:
    """O(log n) nearest-rear lookup from sorted lane arrays."""
    if len(sorted_y) == 0:
        return None, float("inf")

    if direction == "u":
        pos = np.searchsorted(sorted_y, query_y, side="left")
        for k in range(pos - 1, -1, -1):
            if int(sorted_idx[k]) != query_idx:
                return int(sorted_idx[k]), float(query_y - sorted_y[k])
    else:
        pos = np.searchsorted(sorted_y, query_y, side="right")
        for k in range(pos, len(sorted_y)):
            if int(sorted_idx[k]) != query_idx:
                return int(sorted_idx[k]), float(sorted_y[k] - query_y)
    return None, float("inf")


# ---------------------------------------------------------------------------
# 6. Vehicle array extraction utility
# ---------------------------------------------------------------------------

def extract_vehicle_arrays(vehicles, lane_idx_array) -> Dict[str, np.ndarray]:
    """Extract vehicle object attributes into numpy arrays (CPU)."""
    n = len(vehicles)
    positions = np.empty((n, 2), dtype=np.float64)
    velocities = np.empty(n, dtype=np.float64)
    directions = np.empty(n, dtype="U1")
    is_cav = np.empty(n, dtype=bool)
    lengths = np.empty(n, dtype=np.float64)
    for i, v in enumerate(vehicles):
        positions[i, 0] = float(v.position[0])
        positions[i, 1] = float(v.position[1])
        velocities[i] = float(getattr(v, "velocity", 0.0))
        directions[i] = str(getattr(v, "direction", "u"))
        is_cav[i] = bool(getattr(v, "is_cav", False))
        lengths[i] = float(getattr(v, "vehicle_length", 4.8))
    return {
        "positions": positions,
        "velocities": velocities,
        "directions": directions,
        "is_cav": is_cav,
        "lengths": lengths,
        "lane_idx": np.asarray(lane_idx_array, dtype=np.int32),
    }
