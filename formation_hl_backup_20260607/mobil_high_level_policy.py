"""External rule baseline: MOBIL-style lane-change pressure with C-IDM/CACC recovery."""

from __future__ import annotations

from typing import Dict

import numpy as np

try:
    from .high_level_policy import FormationHighLevelPolicy, PolicyDecision
except ImportError:  # pragma: no cover
    from high_level_policy import FormationHighLevelPolicy, PolicyDecision


class MOBILCIDMCACCHighLevelPolicy(FormationHighLevelPolicy):
    """Rule baseline approximating classical MOBIL + IDM/CACC reconfiguration behavior."""

    name = "mobil_cidm_cacc"

    def select_action(self, step: int, scheduler, state_fields: Dict[str, float], state_vector=None, training: bool = False) -> PolicyDecision:
        del state_vector, training
        mode = self._event_mode(step, scheduler, state_fields)
        min_gap = min(float(state_fields.get("min_gap_up", 0.0)), float(state_fields.get("min_gap_down", 0.0)))
        blocked_ratio = max(
            float(state_fields.get("blocked_lane_ratio_up", 0.0)),
            float(state_fields.get("blocked_lane_ratio_down", 0.0)),
        )
        event_pressure = max(
            float(state_fields.get("event_pressure_up", 0.0)),
            float(state_fields.get("event_pressure_down", 0.0)),
        )
        lane_imbalance = max(
            float(state_fields.get("lane_occupancy_std_up", 0.0)),
            float(state_fields.get("lane_occupancy_std_down", 0.0)),
        )
        platoon_rate = float(state_fields.get("platoon_rate", 0.0))
        comm_fail = float(state_fields.get("comm_fail_percent", 0.0))
        pressure = self._safety_pressure(state_fields)

        mobil_utility = (
            0.55 * blocked_ratio
            + 0.35 * float(np.clip((lane_imbalance - 0.8) / 1.5, 0.0, 1.0))
            + 0.30 * float(np.clip((12.0 - min_gap) / 8.0, 0.0, 1.0))
            + 0.20 * float(np.clip(event_pressure, 0.0, 1.0))
        )

        if mode == "event":
            if blocked_ratio > 0.0 and mobil_utility >= 0.45:
                return PolicyDecision("split", score=mobil_utility, reason="MOBIL lane redistribution under blocked lanes")
            if pressure > 0.85 or min_gap < 9.5:
                return PolicyDecision("expand", score=max(pressure, 10.0 - min_gap), reason="IDM/CACC safety expansion")
            if lane_imbalance >= 1.0:
                return PolicyDecision("split", score=lane_imbalance, reason="MOBIL incentive due to lane imbalance")
            if platoon_rate < 0.30:
                return PolicyDecision("compact", score=1.0 - platoon_rate, reason="restore compact platoon after disturbance")
            return PolicyDecision("keep", score=1.0 - pressure, reason="maintain stable rule-based topology")

        if mode == "recovery":
            if min_gap < 16.0 or pressure > 0.65:
                return PolicyDecision("keep", score=min_gap, reason="hold recovery under residual safety pressure")
            if comm_fail <= 0.10 and platoon_rate < 0.55:
                return PolicyDecision("merge", score=1.0 - platoon_rate, reason="merge platoon with rule-based recovery")
            if pressure > 0.45:
                return PolicyDecision("expand", score=pressure, reason="residual disturbance after incident")
            return PolicyDecision("merge", score=1.0, reason="return to compact structure")

        if min_gap < 12.0:
            return PolicyDecision("expand", score=12.0 - min_gap, reason="prevent overly tight cruising spacing")
        if platoon_rate < 0.25 and lane_imbalance < 0.8:
            return PolicyDecision("compact", score=1.0 - platoon_rate, reason="rule-based compact cruising")
        return PolicyDecision("keep", score=1.0 - 0.2 * pressure, reason="steady MOBIL/C-IDM baseline")
