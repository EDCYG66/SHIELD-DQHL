"""Tensorized stepping primitives for the experimental highway environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class LongitudinalStepResult:
    positions_xy: np.ndarray
    moved_mask: np.ndarray
    step_distance: np.ndarray


def step_longitudinal_positions(
    positions_xy: np.ndarray,
    velocities: np.ndarray,
    direction_sign: np.ndarray,
    *,
    move_speed: float,
    timestep: float,
    base_y: float,
    height: float,
    jitter_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> LongitudinalStepResult:
    """Advance highway vehicles along the road axis using batched array math."""

    positions_xy = np.asarray(positions_xy, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32).reshape(-1)
    direction_sign = np.asarray(direction_sign, dtype=np.float32).reshape(-1)
    n_vehicles = int(positions_xy.shape[0])
    if n_vehicles == 0:
        return LongitudinalStepResult(
            positions_xy=np.empty((0, 2), dtype=np.float32),
            moved_mask=np.empty((0,), dtype=bool),
            step_distance=np.empty((0,), dtype=np.float32),
        )

    if positions_xy.shape != (n_vehicles, 2):
        raise ValueError("positions_xy must have shape (N, 2)")
    if velocities.shape[0] != n_vehicles or direction_sign.shape[0] != n_vehicles:
        raise ValueError("velocities and direction_sign must match positions_xy length")

    if float(move_speed) > 0.0:
        step_distance = np.full((n_vehicles,), float(move_speed), dtype=np.float32)
    else:
        step_distance = np.maximum(velocities * float(timestep), 0.0).astype(np.float32, copy=False)

    y_prev = positions_xy[:, 1]
    y_next = y_prev + direction_sign * step_distance
    y_next = np.clip(y_next, float(base_y), float(height - base_y))

    if float(jitter_std) > 0.0:
        rng = rng or np.random.default_rng()
        y_next = y_next + rng.normal(0.0, float(jitter_std), size=n_vehicles).astype(np.float32)
        y_next = np.clip(y_next, float(base_y), float(height - base_y))

    positions_next = positions_xy.copy()
    positions_next[:, 1] = y_next.astype(np.float32, copy=False)
    moved_mask = np.abs(positions_next[:, 1] - y_prev) > 1e-6
    return LongitudinalStepResult(
        positions_xy=positions_next,
        moved_mask=moved_mask,
        step_distance=step_distance,
    )


def pairwise_distance_matrix(positions_xy: np.ndarray) -> np.ndarray:
    """Compute an NxN Euclidean distance matrix from batched XY positions."""

    positions_xy = np.asarray(positions_xy, dtype=np.float32)
    n_vehicles = int(positions_xy.shape[0])
    if n_vehicles == 0:
        return np.zeros((0, 0), dtype=np.float32)
    diff = positions_xy[:, None, :] - positions_xy[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32, copy=False)


def associate_v2i_single(
    positions_xy: np.ndarray,
    bs_position_xy: np.ndarray,
    *,
    stay_steps: np.ndarray | None = None,
    initial: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Associate every vehicle to one BS in single-BS mode."""

    positions_xy = np.asarray(positions_xy, dtype=np.float32)
    n_vehicles = int(positions_xy.shape[0])
    if n_vehicles == 0:
        empty_i = np.zeros((0,), dtype=np.int32)
        empty_f = np.zeros((0,), dtype=np.float32)
        return empty_i, empty_i.copy(), empty_f

    bs_xy = np.asarray(bs_position_xy, dtype=np.float32).reshape(2)
    delta = positions_xy - bs_xy[None, :]
    dist = np.sqrt(np.sum(delta * delta, axis=1)).astype(np.float32, copy=False)
    serving = np.zeros((n_vehicles,), dtype=np.int32)
    if stay_steps is None or initial:
        stay = np.zeros((n_vehicles,), dtype=np.int32)
    else:
        stay = np.asarray(stay_steps, dtype=np.int32).reshape(-1) + 1
    return serving, stay, dist


def associate_v2i_rsu(
    positions_xy: np.ndarray,
    bs_positions_xy: np.ndarray,
    *,
    current_serving_idx: np.ndarray | None = None,
    current_dist_m: np.ndarray | None = None,
    stay_steps: np.ndarray | None = None,
    hysteresis_m: float,
    min_stay_steps: int,
    initial: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Associate every vehicle to the best RSU with hysteresis and minimum stay."""

    positions_xy = np.asarray(positions_xy, dtype=np.float32)
    bs_positions_xy = np.asarray(bs_positions_xy, dtype=np.float32).reshape(-1, 2)
    n_vehicles = int(positions_xy.shape[0])
    n_bs = int(bs_positions_xy.shape[0])
    if n_vehicles == 0:
        empty_i = np.zeros((0,), dtype=np.int32)
        empty_f = np.zeros((0,), dtype=np.float32)
        return empty_i, empty_i.copy(), empty_f
    if n_bs == 0:
        return (
            np.full((n_vehicles,), -1, dtype=np.int32),
            np.zeros((n_vehicles,), dtype=np.int32),
            np.zeros((n_vehicles,), dtype=np.float32),
        )

    diff = positions_xy[:, None, :] - bs_positions_xy[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    best_idx = np.argmin(d2, axis=1).astype(np.int32, copy=False)
    best_dist = np.sqrt(d2[np.arange(n_vehicles), best_idx]).astype(np.float32, copy=False)

    if initial or current_serving_idx is None or stay_steps is None or current_dist_m is None:
        return best_idx.copy(), np.zeros((n_vehicles,), dtype=np.int32), best_dist.copy()

    current_serving_idx = np.asarray(current_serving_idx, dtype=np.int32).reshape(-1)
    current_dist_m = np.asarray(current_dist_m, dtype=np.float32).reshape(-1)
    stay_steps = np.asarray(stay_steps, dtype=np.int32).reshape(-1)
    new_serving = current_serving_idx.copy()
    new_dist = current_dist_m.copy()
    new_stay = stay_steps.copy()

    for i in range(n_vehicles):
        curr = int(current_serving_idx[i]) if i < current_serving_idx.shape[0] else -1
        cand = int(best_idx[i])
        cand_dist = float(best_dist[i])
        curr_dist = float(np.sqrt(d2[i, curr])) if 0 <= curr < n_bs else float("inf")
        if curr == -1:
            new_serving[i] = cand
            new_dist[i] = cand_dist
            new_stay[i] = 0
            continue
        do_switch = (
            cand != curr
            and cand_dist + float(hysteresis_m) < curr_dist
            and int(stay_steps[i]) >= int(min_stay_steps)
        )
        if do_switch:
            new_serving[i] = cand
            new_dist[i] = cand_dist
            new_stay[i] = 0
        else:
            new_serving[i] = curr
            new_dist[i] = curr_dist
            new_stay[i] = int(stay_steps[i]) + 1
    return new_serving, new_stay, new_dist
