"""Correctness validation — compare GPU-optimized kernels against originals."""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def validate_distance_matrix(n_vehicles: int = 32, seed: int = 42) -> bool:
    """Compare CuPy pairwise distance vs naive Python loop."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, 1200, size=(n_vehicles, 2))

    from gpu_optimized.cupy_kernels import pairwise_distance_gpu

    result_gpu = pairwise_distance_gpu(positions)

    expected = np.zeros((n_vehicles, n_vehicles), dtype=np.float64)
    for i in range(n_vehicles):
        for j in range(i, n_vehicles):
            d = float(np.hypot(positions[i, 0] - positions[j, 0], positions[i, 1] - positions[j, 1]))
            expected[i, j] = expected[j, i] = d

    ok = np.allclose(result_gpu, expected, atol=1e-4)
    print(f"  distance_matrix (n={n_vehicles}): {'PASS' if ok else 'FAIL'} "
          f"max_diff={np.max(np.abs(result_gpu - expected)):.2e}")
    return ok


def validate_energy_stats(seed: int = 42) -> bool:
    """Compare vectorized energy vs per-vehicle loop."""
    rng = np.random.default_rng(seed)
    n = 32
    speeds = rng.uniform(0, 30, n)
    accels = rng.uniform(-3, 3, n)
    lengths = rng.choice([4.8, 5.0, 12.0], n)
    is_cav = rng.choice([True, False], n)
    dt = 0.1

    from gpu_optimized.cupy_kernels import vectorized_energy_kj

    total, cav, hv = vectorized_energy_kj(speeds, accels, lengths, is_cav, dt)

    total_ref = 0.0
    cav_ref = 0.0
    hv_ref = 0.0
    for i in range(n):
        speed = max(speeds[i], 0.0)
        is_truck = lengths[i] > 7.5
        mass = 8500.0 if is_truck else 1600.0
        c_rr = 0.010 if is_truck else 0.012
        cd_area = 5.40 if is_truck else 0.72
        rolling = c_rr * mass * 9.81
        drag = 0.5 * 1.225 * cd_area * speed * speed
        inertial = mass * max(accels[i], 0.0)
        power = max(0.0, (rolling + drag + inertial) * speed)
        e = (power * dt) / 1000.0
        total_ref += e
        if is_cav[i]:
            cav_ref += e
        else:
            hv_ref += e

    ok = abs(total - total_ref) < 1e-3 and abs(cav - cav_ref) < 1e-3 and abs(hv - hv_ref) < 1e-3
    print(f"  energy_stats: {'PASS' if ok else 'FAIL'} "
          f"diff_total={abs(total - total_ref):.2e} diff_cav={abs(cav - cav_ref):.2e}")
    return ok


def validate_broad_phase(seed: int = 42) -> bool:
    """Compare vectorized broad-phase collision vs naive loop."""
    rng = np.random.default_rng(seed)
    n = 20
    centers = rng.uniform(0, 100, (n, 2))
    radii = rng.uniform(2, 10, n)

    from gpu_optimized.cupy_kernels import broad_phase_collision_pairs

    pairs_gpu = set(broad_phase_collision_pairs(centers, radii))

    pairs_ref = set()
    for i in range(n - 1):
        for j in range(i + 1, n):
            d2 = (centers[i, 0] - centers[j, 0]) ** 2 + (centers[i, 1] - centers[j, 1]) ** 2
            if d2 <= (radii[i] + radii[j]) ** 2:
                pairs_ref.add((i, j))

    ok = pairs_gpu == pairs_ref
    print(f"  broad_phase (n={n}): {'PASS' if ok else 'FAIL'} "
          f"gpu={len(pairs_gpu)} ref={len(pairs_ref)} diff={len(pairs_gpu.symmetric_difference(pairs_ref))}")
    return ok


def validate_sorted_neighbor_search(seed: int = 42) -> bool:
    """Compare sorted-index neighbor search vs naive O(n) scan."""
    rng = np.random.default_rng(seed)
    n = 24
    positions_y = rng.uniform(100, 1000, n)
    directions = np.array(["u"] * 12 + ["d"] * 12)
    lane_indices = rng.integers(0, 4, n).astype(np.int32)

    from gpu_optimized.cupy_kernels import (
        build_sorted_lane_index,
        nearest_front_from_sorted,
        nearest_rear_from_sorted,
    )

    lane_index = build_sorted_lane_index(positions_y, directions, lane_indices, 4)

    errors = 0
    for idx in range(n):
        d = directions[idx]
        lane = int(lane_indices[idx])
        y = float(positions_y[idx])

        front_sorted, gap_sorted = nearest_front_from_sorted(
            *lane_index.get((d, lane), (np.empty(0), np.empty(0, dtype=np.int64))),
            y, idx, d,
        )

        front_naive = None
        gap_naive = float("inf")
        for j in range(n):
            if j == idx or directions[j] != d or lane_indices[j] != lane:
                continue
            yj = float(positions_y[j])
            if d == "u" and yj > y:
                g = yj - y
                if g < gap_naive:
                    gap_naive = g
                    front_naive = j
            elif d == "d" and yj < y:
                g = y - yj
                if g < gap_naive:
                    gap_naive = g
                    front_naive = j

        if front_sorted != front_naive or (front_sorted is not None and abs(gap_sorted - gap_naive) > 1e-6):
            errors += 1

    ok = errors == 0
    print(f"  neighbor_search (n={n}): {'PASS' if ok else 'FAIL'} errors={errors}")
    return ok


def validate_position_update(seed: int = 42) -> bool:
    """Compare vectorized position update vs per-vehicle loop."""
    rng = np.random.default_rng(seed)
    n = 24
    positions_y = rng.uniform(100, 1000, n).astype(np.float64)
    directions = np.array(["u"] * 12 + ["d"] * 12, dtype="U1")
    velocities = rng.uniform(15, 30, n).astype(np.float64)
    height = 1200.0
    base_y = 0.0
    timestep = 0.1
    move_speed = 0.0
    jitter_std = 0.0

    from gpu_optimized.cupy_kernels import vectorized_position_update

    new_y, _ = vectorized_position_update(
        positions_y, directions, velocities, move_speed, timestep, height, base_y, jitter_std,
    )

    ref_y = positions_y.copy()
    for i in range(n):
        dy = velocities[i] * timestep
        if directions[i] == "u":
            ref_y[i] = min(ref_y[i] + dy, height - base_y)
        elif directions[i] == "d":
            ref_y[i] = max(ref_y[i] - dy, base_y)
        else:
            ref_y[i] = min(ref_y[i] + dy, height - base_y)

    ok = np.allclose(new_y, ref_y, atol=1e-10)
    print(f"  position_update (n={n}): {'PASS' if ok else 'FAIL'} max_diff={np.max(np.abs(new_y - ref_y)):.2e}")
    return ok


def validate_v2v_pathloss(seed: int = 42) -> bool:
    """Compare vectorized V2V pathloss vs original Python loop."""
    import math

    rng = np.random.default_rng(seed)
    n = 30
    positions = [[float(rng.uniform(0, 750)), float(rng.uniform(0, 1300))] for _ in range(n)]

    from communication.Environment import V2Vchannels

    ch_orig = V2Vchannels(n, 20)
    ch_orig.update_positions(positions)
    ch_orig.update_pathloss()
    expected = ch_orig.PathLoss.copy()

    from gpu_optimized.accelerated_channel import _v2v_update_pathloss_vectorized

    ch_vec = V2Vchannels(n, 20)
    ch_vec.update_positions(positions)
    _v2v_update_pathloss_vectorized(ch_vec)
    result = ch_vec.PathLoss

    max_diff = float(np.max(np.abs(result - expected)))
    ok = max_diff < 0.1
    print(f"  v2v_pathloss (n={n}): {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4e}")
    if not ok:
        worst = np.unravel_index(np.argmax(np.abs(result - expected)), result.shape)
        print(f"    worst pair: {worst} orig={expected[worst]:.4f} vec={result[worst]:.4f}")
    return ok


def validate_v2i_pathloss(seed: int = 42) -> bool:
    """Compare vectorized V2I pathloss vs original Python loop."""
    rng = np.random.default_rng(seed)
    n = 30
    positions = [[float(rng.uniform(0, 750)), float(rng.uniform(0, 1300))] for _ in range(n)]

    from communication.Environment import V2Ichannels

    ch_orig = V2Ichannels(n, 20)
    ch_orig.update_positions(positions)
    ch_orig.update_pathloss()
    expected = ch_orig.PathLoss.copy()

    from gpu_optimized.accelerated_channel import _v2i_update_pathloss_vectorized

    ch_vec = V2Ichannels(n, 20)
    ch_vec.update_positions(positions)
    _v2i_update_pathloss_vectorized(ch_vec)
    result = ch_vec.PathLoss

    max_diff = float(np.max(np.abs(result - expected)))
    ok = max_diff < 0.01
    print(f"  v2i_pathloss (n={n}): {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4e}")
    return ok


def validate_clamp_equivalence() -> bool:
    """Verify _clamp matches np.clip for representative values."""
    from gpu_optimized.accelerated_shield import _clamp

    test_cases = [
        (5.0, 0.0, 10.0, 5.0),
        (-1.0, 0.0, 10.0, 0.0),
        (15.0, 0.0, 10.0, 10.0),
        (-7.0, -6.0, 2.5, -6.0),
        (0.5, -6.0, 2.5, 0.5),
        (3.0, -6.0, 2.5, 2.5),
    ]
    errors = 0
    for x, lo, hi, expected in test_cases:
        got = _clamp(x, lo, hi)
        np_got = float(np.clip(x, lo, hi))
        if abs(got - expected) > 1e-15 or abs(got - np_got) > 1e-15:
            errors += 1
    ok = errors == 0
    print(f"  clamp_equivalence: {'PASS' if ok else 'FAIL'} errors={errors}")
    return ok


def benchmark_pathloss(n: int = 30, repeats: int = 200) -> None:
    """Benchmark V2V pathloss: original vs vectorized."""
    positions = [[float(np.random.uniform(0, 750)), float(np.random.uniform(0, 1300))] for _ in range(n)]

    from communication.Environment import V2Vchannels
    from gpu_optimized.accelerated_channel import _v2v_update_pathloss_vectorized

    ch = V2Vchannels(n, 20)
    ch.update_positions(positions)

    t0 = time.perf_counter()
    for _ in range(repeats):
        ch.update_pathloss()
    t_orig = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        _v2v_update_pathloss_vectorized(ch)
    t_vec = (time.perf_counter() - t0) / repeats

    speedup = t_orig / max(t_vec, 1e-9)
    print(f"    V2V pathloss n={n}: orig={t_orig*1000:.3f}ms  vec={t_vec*1000:.3f}ms  speedup={speedup:.1f}x")


def benchmark_distance_matrix(sizes=(16, 32, 64, 100)) -> None:
    """Benchmark pairwise distance: original vs GPU."""
    from gpu_optimized.cupy_kernels import pairwise_distance_gpu

    print("\n  Benchmark: pairwise distance matrix")
    for n in sizes:
        positions = np.random.uniform(0, 1200, (n, 2))

        t0 = time.perf_counter()
        for _ in range(100):
            expected = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    d = float(np.hypot(positions[i, 0] - positions[j, 0], positions[i, 1] - positions[j, 1]))
                    expected[i, j] = expected[j, i] = d
        t_orig = (time.perf_counter() - t0) / 100

        pairwise_distance_gpu(positions)
        t0 = time.perf_counter()
        for _ in range(100):
            _ = pairwise_distance_gpu(positions)
        t_gpu = (time.perf_counter() - t0) / 100

        speedup = t_orig / max(t_gpu, 1e-9)
        print(f"    n={n:4d}: orig={t_orig*1000:.3f}ms  gpu={t_gpu*1000:.3f}ms  speedup={speedup:.1f}x")


def main():
    print("=" * 60)
    print("GPU Optimization Validation Suite")
    print("=" * 60)

    results = {}
    print("\n1. Correctness tests:")
    results["distance_matrix"] = validate_distance_matrix()
    results["energy_stats"] = validate_energy_stats()
    results["broad_phase"] = validate_broad_phase()
    results["neighbor_search"] = validate_sorted_neighbor_search()
    results["position_update"] = validate_position_update()
    results["v2v_pathloss"] = validate_v2v_pathloss()
    results["v2i_pathloss"] = validate_v2i_pathloss()
    results["clamp_equivalence"] = validate_clamp_equivalence()

    print("\n2. Performance benchmarks:")
    benchmark_pathloss()
    benchmark_distance_matrix()

    all_pass = all(results.values())
    print("\n" + "=" * 60)
    print(f"Results: {sum(results.values())}/{len(results)} passed")
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"FAILED: {', '.join(failed)}")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
