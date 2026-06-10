"""Builds compact high-level state vectors for platoon experiments."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

try:
    from .platoon_analysis import platoon_statistics
except ImportError:  # pragma: no cover
    from platoon_analysis import platoon_statistics


class PlatoonStateBuilder:
    """Extracts high-level formation state from the wrapped highway environment."""

    def __init__(self, cruise_speed: float = 22.0):
        self.cruise_speed = float(cruise_speed)

    def build(self, env, scheduler, step: int) -> Dict[str, object]:
        vehicles = getattr(env, "vehicles", [])
        group_dir = getattr(env, "_group_dir", [])
        lanes_per_dir = max(1, int(getattr(env, "lanes_per_dir", 1)))
        height = max(1.0, float(getattr(env, "height", 1.0)))

        up_indices, down_indices = self._split_indices(group_dir)
        dir_cache = {
            "u": self._prepare_direction_cache(env, vehicles, up_indices, "u", lanes_per_dir),
            "d": self._prepare_direction_cache(env, vehicles, down_indices, "d", lanes_per_dir),
        }
        platoon_stats = platoon_statistics(env)
        cav_indices = [idx for idx, veh in enumerate(vehicles) if bool(getattr(veh, "is_cav", False))]
        hv_indices = [idx for idx, veh in enumerate(vehicles) if not bool(getattr(veh, "is_cav", False))]

        active_event = 1.0 if scheduler and scheduler.has_active_event(step) else 0.0
        topology_star = 1.0 if getattr(env, "topology_type", "star") == "star" else 0.0
        topology_switch_indicator = 1.0 if getattr(env, "topology_type", "star") == "tree" else 0.0
        blocked_lane_ratio_up = self._blocked_lane_ratio_cached(scheduler, "u", step, lanes_per_dir)
        blocked_lane_ratio_down = self._blocked_lane_ratio_cached(scheduler, "d", step, lanes_per_dir)
        event_pressure_up = self._event_pressure_cached(scheduler, vehicles, up_indices, "u", step)
        event_pressure_down = self._event_pressure_cached(scheduler, vehicles, down_indices, "d", step)

        fields = {
            "n_up": float(len(up_indices)),
            "n_down": float(len(down_indices)),
            "mean_speed_up": dir_cache["u"]["mean_speed"],
            "mean_speed_down": dir_cache["d"]["mean_speed"],
            "min_gap_up": dir_cache["u"]["min_gap"],
            "min_gap_down": dir_cache["d"]["min_gap"],
            "mean_gap_up": dir_cache["u"]["mean_gap"],
            "mean_gap_down": dir_cache["d"]["mean_gap"],
            "mean_v2i_distance": self._mean_v2i_distance(env),
            "mean_v2v_link_distance": self._mean_v2v_distance(env),
            "topology_star": topology_star,
            "active_event": active_event,
            "blocked_lane_ratio_up": blocked_lane_ratio_up,
            "blocked_lane_ratio_down": blocked_lane_ratio_down,
            "event_pressure_up": event_pressure_up,
            "event_pressure_down": event_pressure_down,
            "topology_switch_indicator": topology_switch_indicator,
            "lane_occupancy_std_up": dir_cache["u"]["lane_occupancy_std"],
            "lane_occupancy_std_down": dir_cache["d"]["lane_occupancy_std"],
            "n_cav": float(len(cav_indices)),
            "n_hv": float(len(hv_indices)),
            "mpr_cav": float(platoon_stats["mpr_cav"]),
            "mean_speed_cav": self._mean_speed_for_indices(vehicles, cav_indices),
            "mean_speed_hv": self._mean_speed_for_indices(vehicles, hv_indices),
            "platoon_count": float(platoon_stats["platoon_count"]),
            "platoon_rate": float(platoon_stats["platoon_rate"]),
            "mean_platoon_length": float(platoon_stats["mean_platoon_length"]),
            "max_platoon_length": float(platoon_stats["max_platoon_length"]),
        }
        vector = np.asarray([
            fields["mean_speed_up"] / self.cruise_speed,
            fields["mean_speed_down"] / self.cruise_speed,
            fields["min_gap_up"] / 30.0,
            fields["min_gap_down"] / 30.0,
            fields["mean_gap_up"] / 30.0,
            fields["mean_gap_down"] / 30.0,
            fields["mean_v2i_distance"] / height,
            fields["mean_v2v_link_distance"] / height,
            fields["topology_star"],
            fields["active_event"],
            fields["blocked_lane_ratio_up"],
            fields["blocked_lane_ratio_down"],
            fields["event_pressure_up"],
            fields["event_pressure_down"],
            fields["topology_switch_indicator"],
            fields["lane_occupancy_std_up"],
            fields["lane_occupancy_std_down"],
            fields["mpr_cav"],
            fields["mean_speed_cav"] / self.cruise_speed,
            fields["mean_speed_hv"] / self.cruise_speed,
            fields["platoon_rate"],
            fields["mean_platoon_length"] / max(1.0, fields["n_cav"]),
        ], dtype=np.float32)
        return {"vector": vector, "fields": fields}

    def _split_indices(self, group_dir: List[str]) -> Tuple[List[int], List[int]]:
        up_indices = [i for i, direction in enumerate(group_dir) if direction == "u"]
        down_indices = [i for i, direction in enumerate(group_dir) if direction == "d"]
        return up_indices, down_indices

    def _prepare_direction_cache(self, env, vehicles, indices: List[int], direction: str, lanes_per_dir: int) -> Dict[str, float]:
        if not indices:
            return {"mean_speed": 0.0, "min_gap": 0.0, "mean_gap": 0.0, "lane_occupancy_std": 0.0}

        speeds = np.fromiter((float(vehicles[idx].velocity) for idx in indices), dtype=np.float32, count=len(indices))
        lane_counts = np.zeros(lanes_per_dir, dtype=np.float32)
        lane_buckets: Dict[int, List[Tuple[float, int]]] = {}
        for idx in indices:
            lane_idx = int(env._lane_idx[idx])
            if 0 <= lane_idx < lanes_per_dir:
                lane_counts[lane_idx] += 1.0
                y = float(vehicles[idx].position[1])
                lane_buckets.setdefault(lane_idx, []).append((y, idx))

        gaps: List[float] = []
        for lane_idx, bucket in lane_buckets.items():
            bucket.sort(key=lambda item: item[0], reverse=(direction == "u"))
            for (y_prev, _), (y_next, _) in zip(bucket[:-1], bucket[1:]):
                gap = (y_prev - y_next) if direction == "u" else (y_next - y_prev)
                gaps.append(max(0.0, float(gap)))

        return {
            "mean_speed": float(np.mean(speeds)) if speeds.size else 0.0,
            "min_gap": float(min(gaps)) if gaps else 0.0,
            "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
            "lane_occupancy_std": float(np.std(lane_counts)),
        }

    def _mean_v2i_distance(self, env) -> float:
        distances = getattr(env, "v2i_dist_m", None)
        if distances is None or len(distances) == 0:
            return 0.0
        return float(np.mean(distances))

    def _mean_v2v_distance(self, env) -> float:
        values: List[float] = []
        distance_matrix = getattr(env, "Distance", None)
        if distance_matrix is None:
            return 0.0
        for tx_idx, vehicle in enumerate(getattr(env, "vehicles", [])):
            destinations = getattr(vehicle, "destinations", [])
            for rx_idx in destinations:
                if 0 <= rx_idx < len(env.vehicles):
                    values.append(float(distance_matrix[tx_idx, rx_idx]))
        return float(np.mean(values)) if values else 0.0

    def _mean_speed_for_indices(self, vehicles, indices: List[int]) -> float:
        if not indices:
            return 0.0
        values = np.asarray([float(vehicles[idx].velocity) for idx in indices], dtype=np.float32)
        return float(np.mean(values)) if values.size else 0.0

    def _blocked_lane_ratio_cached(self, scheduler, direction: str, step: int, lanes_per_dir: int) -> float:
        if not scheduler:
            return 0.0
        blocked = scheduler.blocked_lanes(direction, step)
        return float(len(blocked)) / float(max(1, lanes_per_dir))

    def _event_pressure_cached(self, scheduler, vehicles, indices: List[int], direction: str, step: int) -> float:
        if not scheduler or not indices:
            return 0.0
        values = [scheduler.event_pressure(direction, float(vehicles[idx].position[1]), step) for idx in indices]
        return float(np.mean(values)) if values else 0.0
