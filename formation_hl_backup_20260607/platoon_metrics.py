"""Metrics tracker for formation reconfiguration experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


class PlatoonMetricsTracker:
    """Stores step-wise records and exports summary statistics."""

    def __init__(self):
        self.records: List[Dict[str, float]] = []

    def update(self, *, step: int, state_fields: Dict[str, float], info: Dict[str, float], action: str, reward: float) -> None:
        row = {
            "step": float(step),
            "reward": float(reward),
            "action_keep": 1.0 if action == "keep" else 0.0,
            "action_compact": 1.0 if action == "compact" else 0.0,
            "action_expand": 1.0 if action == "expand" else 0.0,
            "action_split": 1.0 if action == "split" else 0.0,
            "action_merge": 1.0 if action == "merge" else 0.0,
            "action_emergency": 1.0 if action == "emergency" else 0.0,
        }
        row.update({key: float(value) for key, value in state_fields.items() if isinstance(value, (int, float))})
        row.update({
            key: float(value)
            for key, value in info.items()
            if key not in {"shield", "communication"} and isinstance(value, (int, float))
        })
        row["lane_changes"] = float(info.get("lane_changes", 0.0))
        row["handovers"] = float(info.get("handovers", 0.0))
        row["topology_switches"] = float(info.get("topology_switches", 0.0))
        row["policy_score"] = float(info.get("policy_score", 0.0))
        reward_components = info.get("reward_components", {})
        if isinstance(reward_components, dict):
            for key, value in reward_components.items():
                if isinstance(value, (int, float)):
                    row[f"reward_{key}"] = float(value)
        shield = info.get("shield", {})
        row["shield_interventions"] = float(shield.get("interventions", 0.0))
        row["shield_lane_blocks"] = float(shield.get("lane_change_blocks", 0.0))
        row["shield_emergency_brakes"] = float(shield.get("emergency_brakes", 0.0))
        row["shield_cbf_adjustments"] = float(shield.get("cbf_adjustments", 0.0))
        row["shield_cbf_longitudinal_clips"] = float(shield.get("cbf_longitudinal_clips", 0.0))
        row["shield_cbf_lateral_clips"] = float(shield.get("cbf_lateral_clips", 0.0))
        row["shield_cbf_blocked_lane_avoid"] = float(shield.get("cbf_blocked_lane_avoid", 0.0))
        row["shield_cbf_min_barrier"] = float(shield.get("cbf_min_barrier", 0.0))
        row["shield_cbf_lane_barrier"] = float(shield.get("cbf_lane_barrier", 0.0))
        self.records.append(row)

    def summary(self) -> Dict[str, float]:
        if not self.records:
            return {}

        def avg(key: str) -> float:
            return float(np.mean([row.get(key, 0.0) for row in self.records]))

        def total(key: str) -> float:
            return float(np.sum([row.get(key, 0.0) for row in self.records]))

        def first_step_where(predicate) -> float:
            for row in self.records:
                if predicate(row):
                    return float(row.get("step", 0.0))
            return -1.0

        def first_sustained_step(predicate, sustain_count: int, *, start_row: int = 0) -> float:
            count = 0
            first_step = -1.0
            for row in self.records[start_row:]:
                if predicate(row):
                    if count == 0:
                        first_step = float(row.get("step", 0.0))
                    count += 1
                    if count >= sustain_count:
                        return first_step
                else:
                    count = 0
                    first_step = -1.0
            return -1.0

        def step_to_time(step_value: float) -> float:
            if step_value < 0.0:
                return -1.0
            for row in self.records:
                if float(row.get("step", -1.0)) >= step_value:
                    return float(row.get("time_s", row.get("step", 0.0)))
            return float(self.records[-1].get("time_s", self.records[-1].get("step", 0.0)))

        event_active_steps = [float(row.get("step", 0.0)) for row in self.records if row.get("active_event", 0.0) > 0.0]
        event_start_step = min(event_active_steps) if event_active_steps else None
        event_start_time_s = step_to_time(float(event_start_step)) if event_start_step is not None else -1.0
        sustain_count = max(1, min(5, len(self.records)))
        peak_platoon_rate = float(np.max([row.get("platoon_rate", 0.0) for row in self.records]))
        peak_platoon_len = float(np.max([row.get("max_platoon_length", 0.0) for row in self.records]))
        platoon_target = 0.85 * peak_platoon_rate if peak_platoon_rate > 1e-6 else float("inf")
        formation_length_target = 2.0 if peak_platoon_len >= 2.0 else max(1.0, peak_platoon_len)
        formation_step = first_sustained_step(
            lambda row: (
                peak_platoon_rate > 1e-6
                and row.get("platoon_rate", 0.0) >= platoon_target
                and row.get("max_platoon_length", 0.0) >= formation_length_target
            ),
            sustain_count,
        )

        post_event_start_row = 0
        if event_start_step is not None:
            for idx, row in enumerate(self.records):
                if float(row.get("step", -1.0)) >= float(event_start_step):
                    post_event_start_row = idx
                    break
        post_event_rows = self.records[post_event_start_row:] if event_start_step is not None else []
        post_event_peak_pr = float(np.max([row.get("platoon_rate", 0.0) for row in post_event_rows])) if post_event_rows else 0.0
        post_event_target = 0.85 * post_event_peak_pr if post_event_peak_pr > 1e-6 else float("inf")
        reconfig_step = first_sustained_step(
            lambda row: (
                event_start_step is not None
                and row.get("step", -1.0) >= event_start_step
                and row.get("platoon_rate", 0.0) >= post_event_target
                and row.get("blocked_lane_vehicle_count", 1.0) <= 0.0
                and row.get("collision_occurred", 0.0) <= 0.0
                and min(row.get("min_gap_up", 0.0), row.get("min_gap_down", 0.0)) >= 12.0
            ),
            sustain_count,
            start_row=post_event_start_row,
        )
        reconfig_time_s = -1.0
        if event_start_step is not None and reconfig_step >= 0.0:
            reconfig_time_s = max(0.0, step_to_time(reconfig_step) - step_to_time(event_start_step))

        event_clearance_step = first_step_where(
            lambda row: (
                event_start_step is not None
                and row.get("step", -1) >= event_start_step
                and row.get("event_zone_vehicle_count", 1.0) <= 0.0
                and row.get("active_event", 0.0) <= 0.0
            )
        )
        safe_recovery_step = first_step_where(
            lambda row: (
                event_start_step is not None
                and row.get("step", -1) >= event_start_step
                and min(row.get("min_gap_up", 0.0), row.get("min_gap_down", 0.0)) >= 12.0
                and row.get("comm_fail_percent", 1.0) <= 0.05
            )
        )
        event_clearance_time_s = -1.0
        safe_recovery_time_s = -1.0
        if event_start_step is not None and event_clearance_step >= 0.0:
            event_clearance_time_s = max(0.0, step_to_time(event_clearance_step) - event_start_time_s)
        if event_start_step is not None and safe_recovery_step >= 0.0:
            safe_recovery_time_s = max(0.0, step_to_time(safe_recovery_step) - event_start_time_s)

        return {
            "steps": float(len(self.records)),
            "avg_reward": avg("reward"),
            "avg_speed_up": avg("mean_speed_up"),
            "avg_speed_down": avg("mean_speed_down"),
            "avg_speed_all": 0.5 * (avg("mean_speed_up") + avg("mean_speed_down")),
            "avg_speed_cav": avg("mean_speed_cav"),
            "avg_speed_hv": avg("mean_speed_hv"),
            "avg_comm_v2i_rate_total": avg("comm_v2i_rate_total"),
            "avg_comm_v2v_success": avg("comm_v2v_success"),
            "avg_comm_used_rb_ratio": avg("comm_used_rb_ratio"),
            "avg_comm_mean_power_norm": avg("comm_mean_power_norm"),
            "avg_comm_fail_percent": avg("comm_fail_percent"),
            "avg_policy_score": avg("policy_score"),
            "avg_abs_accel_cmd": avg("avg_abs_accel_cmd"),
            "max_abs_accel_cmd": float(np.max([row.get("max_abs_accel_cmd", 0.0) for row in self.records])),
            "avg_target_speed_cmd": avg("avg_target_speed_cmd"),
            "avg_desired_gap_cmd": avg("avg_desired_gap_cmd"),
            "avg_lateral_speed_cmd": avg("avg_lateral_speed_cmd"),
            "avg_lateral_error_cmd": avg("avg_lateral_error_cmd"),
            "max_lateral_speed_cmd": float(np.max([row.get("max_lateral_speed_cmd", 0.0) for row in self.records])),
            "avg_active_lane_change_cmds": avg("active_lane_change_cmds"),
            "avg_cacc_cmd_ratio": avg("cacc_cmd_ratio"),
            "avg_hybrid_cmd_ratio": avg("hybrid_cmd_ratio"),
            "avg_cidm_cmd_ratio": avg("cidm_cmd_ratio"),
            "avg_comm_reliability_cmd": avg("avg_comm_reliability_cmd"),
            "avg_cacc_weight_cmd": avg("avg_cacc_weight_cmd"),
            "avg_headway_scale_cmd": avg("avg_headway_scale_cmd"),
            "avg_info_topology_level_cmd": avg("avg_info_topology_level_cmd"),
            "avg_action_keep": avg("action_keep"),
            "avg_action_compact": avg("action_compact"),
            "avg_action_expand": avg("action_expand"),
            "avg_action_split": avg("action_split"),
            "avg_action_merge": avg("action_merge"),
            "avg_action_emergency": avg("action_emergency"),
            "worst_min_gap_up": float(np.min([row.get("min_gap_up", 0.0) for row in self.records])),
            "worst_min_gap_down": float(np.min([row.get("min_gap_down", 0.0) for row in self.records])),
            "avg_event_pressure_up": avg("event_pressure_up"),
            "avg_event_pressure_down": avg("event_pressure_down"),
            "avg_event_distance_up": avg("avg_event_distance_up"),
            "avg_event_distance_down": avg("avg_event_distance_down"),
            "avg_event_zone_vehicle_count": avg("event_zone_vehicle_count"),
            "peak_event_zone_vehicle_count": float(np.max([row.get("event_zone_vehicle_count", 0.0) for row in self.records])),
            "avg_blocked_lane_vehicle_count": avg("blocked_lane_vehicle_count"),
            "peak_blocked_lane_vehicle_count": float(np.max([row.get("blocked_lane_vehicle_count", 0.0) for row in self.records])),
            "avg_mpr_cav": avg("mpr_cav"),
            "avg_platoon_rate": avg("platoon_rate"),
            "avg_mean_platoon_length": avg("mean_platoon_length"),
            "peak_max_platoon_length": float(np.max([row.get("max_platoon_length", 0.0) for row in self.records])),
            "avg_v2i_distance": avg("mean_v2i_distance"),
            "avg_v2v_link_distance": avg("mean_v2v_link_distance"),
            "avg_energy_step_kj": avg("energy_step_kj"),
            "avg_energy_step_per_vehicle_kj": avg("energy_step_per_vehicle_kj"),
            "total_energy_kj": total("energy_step_kj"),
            "total_energy_cav_kj": total("energy_step_cav_kj"),
            "total_energy_hv_kj": total("energy_step_hv_kj"),
            "energy_per_vehicle_kj": total("energy_step_kj") / float(max(1, int(round(avg("n_up") + avg("n_down"))))),
            "avg_collision_pairs_active": avg("collision_pairs_active"),
            "peak_collision_pairs_active": float(np.max([row.get("collision_pairs_active", 0.0) for row in self.records])),
            "avg_collision_vehicles_active": avg("collision_vehicles_active"),
            "peak_platoon_rate": peak_platoon_rate,
            "event_start_step": float(event_start_step) if event_start_step is not None else -1.0,
            "event_start_time_s": event_start_time_s,
            "platoon_formation_step": formation_step,
            "platoon_formation_time_s": step_to_time(formation_step),
            "reconfiguration_completion_step": reconfig_step,
            "reconfiguration_time_s": reconfig_time_s,
            "final_passed_event_ratio_up": float(self.records[-1].get("passed_event_ratio_up", 0.0)),
            "final_passed_event_ratio_down": float(self.records[-1].get("passed_event_ratio_down", 0.0)),
            "final_passed_event_count_up": float(self.records[-1].get("passed_event_count_up", 0.0)),
            "final_passed_event_count_down": float(self.records[-1].get("passed_event_count_down", 0.0)),
            "total_lane_changes": total("lane_changes"),
            "total_lane_changes_cav": total("lane_changes_cav"),
            "total_lane_changes_hv": total("lane_changes_hv"),
            "total_handovers": total("handovers"),
            "total_topology_switches": total("topology_switches"),
            "total_collision_count": total("collision_count"),
            "total_collision_steps": total("collision_occurred"),
            "has_collision": 1.0 if total("collision_count") > 0.0 else 0.0,
            "total_shield_interventions": total("shield_interventions"),
            "total_shield_lane_blocks": total("shield_lane_blocks"),
            "total_emergency_brakes": total("shield_emergency_brakes"),
            "total_shield_cbf_adjustments": total("shield_cbf_adjustments"),
            "total_shield_cbf_longitudinal_clips": total("shield_cbf_longitudinal_clips"),
            "total_shield_cbf_lateral_clips": total("shield_cbf_lateral_clips"),
            "total_shield_cbf_blocked_lane_avoid": total("shield_cbf_blocked_lane_avoid"),
            "avg_shield_cbf_min_barrier": avg("shield_cbf_min_barrier"),
            "avg_shield_cbf_lane_barrier": avg("shield_cbf_lane_barrier"),
            "event_clearance_step": event_clearance_step,
            "safe_recovery_step": safe_recovery_step,
            "event_clearance_time_s": event_clearance_time_s,
            "safe_recovery_time_s": safe_recovery_time_s,
        }

    def save(
        self,
        out_dir: str | Path,
        *,
        save_plots: bool = True,
        save_csv: bool = True,
        record_stride: int = 1,
    ) -> Dict[str, float]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if self.records and save_csv:
            stride = max(1, int(record_stride))
            records_to_write = self.records[::stride]
            if records_to_write[-1].get("step") != self.records[-1].get("step"):
                records_to_write = records_to_write + [self.records[-1]]
            fieldnames: list[str] = []
            seen = set()
            for row in records_to_write:
                for key in row.keys():
                    if key not in seen:
                        seen.add(key)
                        fieldnames.append(key)
            csv_path = out_path / "formation_step_metrics.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records_to_write)

        if self.records and save_plots:
            try:
                from .plotting import export_eval_plots
            except ImportError:  # pragma: no cover
                from plotting import export_eval_plots

            export_eval_plots(out_path, self.records)

        summary = self.summary()
        summary_path = out_path / "formation_summary.json"
        with summary_path.open("w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, indent=2)
        return summary
