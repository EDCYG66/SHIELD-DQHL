"""Platoon extraction helpers for mixed-traffic formation experiments."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def extract_platoons(env, *, max_gap: float = 35.0, max_speed_delta: float = 4.0) -> List[List[int]]:
    """Extract simple same-lane CAV platoons from the current simulator state."""

    vehicles = getattr(env, "vehicles", [])
    group_dir = getattr(env, "_group_dir", [])
    lane_indices = getattr(env, "_lane_idx", [])
    platoons: List[List[int]] = []
    lane_groups: Dict[tuple[str, int], List[int]] = {}

    for idx, veh in enumerate(vehicles):
        if idx >= len(group_dir) or idx >= len(lane_indices):
            continue
        if not bool(getattr(veh, "is_cav", False)):
            continue
        lane_groups.setdefault((str(group_dir[idx]), int(lane_indices[idx])), []).append(idx)

    for (direction, _lane_idx), indices in lane_groups.items():
        ordered = sorted(indices, key=lambda item: float(vehicles[item].position[1]), reverse=(direction == "u"))
        current: List[int] = []
        for idx in ordered:
            if not current:
                current = [int(idx)]
                continue
            prev = current[-1]
            if direction == "u":
                gap = float(vehicles[prev].position[1] - vehicles[idx].position[1])
            else:
                gap = float(vehicles[idx].position[1] - vehicles[prev].position[1])
            speed_delta = abs(float(vehicles[prev].velocity) - float(vehicles[idx].velocity))
            if gap <= float(max_gap) and speed_delta <= float(max_speed_delta):
                current.append(int(idx))
            else:
                if len(current) >= 2:
                    platoons.append(list(current))
                current = [int(idx)]
        if len(current) >= 2:
            platoons.append(list(current))
    return platoons


def platoon_statistics(env, *, max_gap: float = 35.0, max_speed_delta: float = 4.0) -> Dict[str, float]:
    """Compute basic platoon-quality metrics for the current environment."""

    platoons = extract_platoons(env, max_gap=max_gap, max_speed_delta=max_speed_delta)
    cav_indices = [idx for idx, veh in enumerate(getattr(env, "vehicles", [])) if bool(getattr(veh, "is_cav", False))]
    total_cav = len(cav_indices)
    platooned = sum(len(group) for group in platoons)
    lengths = np.asarray([len(group) for group in platoons], dtype=np.float32)
    return {
        "n_cav": float(total_cav),
        "n_hv": float(max(0, len(getattr(env, "vehicles", [])) - total_cav)),
        "mpr_cav": float(total_cav) / float(max(1, len(getattr(env, "vehicles", [])))),
        "platoon_count": float(len(platoons)),
        "platoon_rate": float(platooned) / float(max(1, total_cav)),
        "mean_platoon_length": float(np.mean(lengths)) if lengths.size else 0.0,
        "max_platoon_length": float(np.max(lengths)) if lengths.size else 0.0,
    }
