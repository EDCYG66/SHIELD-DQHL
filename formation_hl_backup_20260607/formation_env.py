"""Formation experiment environment built on top of the existing highway simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from .communication_adapter import CommunicationCoordinator
    from .events import EventScheduler
    from .human_driver_models import HumanDriverController
    from .low_level_controller import RuleBasedFormationController
    from .motion_models import KinematicBicycleRoadModel
    from .path_utils import ensure_communication_on_path
    from .platoon_analysis import platoon_statistics
    from .platoon_state_builder import PlatoonStateBuilder
    from .safety_shield import SafetyShield
    from .scenario_factory import HighwayScenarioSpec, build_segment_adjustments
    from .tensor_state import TensorizedTrafficState
except ImportError:  # pragma: no cover
    from communication_adapter import CommunicationCoordinator
    from events import EventScheduler
    from human_driver_models import HumanDriverController
    from low_level_controller import RuleBasedFormationController
    from motion_models import KinematicBicycleRoadModel
    from path_utils import ensure_communication_on_path
    from platoon_analysis import platoon_statistics
    from platoon_state_builder import PlatoonStateBuilder
    from safety_shield import SafetyShield
    from scenario_factory import HighwayScenarioSpec, build_segment_adjustments
    from tensor_state import TensorizedTrafficState

ensure_communication_on_path()
from highway_environment import HighwayTopoEnv  # type: ignore  # noqa: E402


HIGH_LEVEL_ACTIONS = ("keep", "compact", "expand", "split", "merge", "emergency")


class FormationExperimentEnv:
    """Wraps HighwayTopoEnv with event handling and platoon-control utilities."""

    def __init__(
        self,
        *,
        scheduler: Optional[EventScheduler] = None,
        controller: Optional[RuleBasedFormationController] = None,
        human_controller: Optional[HumanDriverController] = None,
        shield: Optional[SafetyShield] = None,
        state_builder: Optional[PlatoonStateBuilder] = None,
        communication_enabled: bool = True,
        communication_gnn_type: str = "gat",
        communication_policy_mode: str = "auto",
        communication_dqn_weights: Optional[str] = None,
        communication_gnn_weights: Optional[str] = None,
        communication_weights_dir: Optional[str] = None,
        communication_run_dir: Optional[str] = None,
        communication_decision_interval_steps: int = 2,
        communication_fastfading_update_interval_steps: int = 2,
        communication_metrics_update_interval_steps: int = 2,
        communication_agent_kwargs: Optional[Dict[str, object]] = None,
        road_segments: Optional[list] = None,
        scenario_spec: Optional[HighwayScenarioSpec] = None,
        collision_length: float = 4.8,
        collision_width: Optional[float] = None,
        motion_model: str = "kinematic_bicycle",
        bicycle_model: Optional[KinematicBicycleRoadModel] = None,
        **env_kwargs,
    ):
        self.env_kwargs = dict(env_kwargs)
        self.road_segments = list(road_segments or [])
        self.scenario_spec = scenario_spec
        if self.scenario_spec is not None:
            self.segment_adjustments = build_segment_adjustments(self.scenario_spec)
        else:
            self.segment_adjustments = None
        self.scheduler = scheduler or EventScheduler()
        self.controller = controller or RuleBasedFormationController()
        self.human_controller = human_controller or HumanDriverController()
        self.shield = shield or SafetyShield()
        self.state_builder = state_builder or PlatoonStateBuilder()
        self.communication_enabled = bool(communication_enabled)
        self.communication_gnn_type = str(communication_gnn_type)
        self.communication_policy_mode = str(communication_policy_mode)
        self.communication_dqn_weights = communication_dqn_weights
        self.communication_gnn_weights = communication_gnn_weights
        self.communication_weights_dir = communication_weights_dir
        self.communication_run_dir = communication_run_dir
        self.communication_decision_interval_steps = max(1, int(communication_decision_interval_steps))
        self.communication_fastfading_update_interval_steps = max(1, int(communication_fastfading_update_interval_steps))
        self.communication_metrics_update_interval_steps = max(1, int(communication_metrics_update_interval_steps))
        self.communication_agent_kwargs = dict(communication_agent_kwargs or {})
        self.collision_length = float(collision_length)
        self.collision_width = collision_width
        self.motion_model = str(motion_model).strip().lower()
        if self.motion_model not in {"point_mass", "kinematic_bicycle"}:
            raise ValueError("motion_model must be 'point_mass' or 'kinematic_bicycle'")
        self.bicycle_model = bicycle_model or KinematicBicycleRoadModel()
        self.communication: Optional[CommunicationCoordinator] = None
        self.last_comm_metrics: Dict[str, float] = {
            "comm_v2i_rate_total": 0.0,
            "comm_v2v_success": 0.0,
            "comm_used_rb_ratio": 0.0,
            "comm_mean_power_norm": 0.0,
            "comm_fail_percent": 0.0,
        }
        self._event_cache: Dict[str, object] = {}
        self._state_cache: Optional[Dict[str, object]] = None
        self._tensor_state_cache: Optional[TensorizedTrafficState] = None
        self.reset()

    def reset(self):
        self.env = HighwayTopoEnv(**self.env_kwargs)
        self.env.new_random_game()
        self._apply_segment_adjustments()
        self._initialize_vehicle_dynamics()
        if hasattr(self.controller, "reset"):
            self.controller.reset()
        if hasattr(self.human_controller, "reset"):
            self.human_controller.reset()
        self.step_count = 0
        self.lane_change_count = 0
        self.lane_change_count_cav = 0
        self.lane_change_count_hv = 0
        self.total_handovers = 0
        self.topology_switch_count = 0
        self.total_collision_count = 0
        self.total_collision_steps = 0
        self.total_energy_kj = 0.0
        self._last_hl_action: str = "keep"
        self._hl_action_streak: int = 0
        self._platoon_stability_streak: int = 0
        self.total_energy_cav_kj = 0.0
        self.total_energy_hv_kj = 0.0
        self._prev_collision_pairs: set[tuple[int, int]] = set()
        self._event_passed_up = np.zeros(len(self.env.vehicles), dtype=bool)
        self._event_passed_down = np.zeros(len(self.env.vehicles), dtype=bool)
        self._event_entered_up = np.zeros(len(self.env.vehicles), dtype=bool)
        self._event_entered_down = np.zeros(len(self.env.vehicles), dtype=bool)
        self.last_serving_idx = np.asarray(getattr(self.env, "v2i_serving_idx", np.zeros(0, dtype=int)), dtype=int).copy()
        self._event_cache = {}
        self._state_cache = None
        self._tensor_state_cache = None
        if self.communication_enabled:
            if self.communication is None:
                self.communication = CommunicationCoordinator(
                    self.env,
                    gnn_type=self.communication_gnn_type,
                    policy_mode=self.communication_policy_mode,
                    dqn_weights=self.communication_dqn_weights,
                    gnn_weights=self.communication_gnn_weights,
                    weights_dir=self.communication_weights_dir,
                    run_dir=self.communication_run_dir,
                    decision_interval_steps=self.communication_decision_interval_steps,
                    agent_kwargs=self.communication_agent_kwargs,
                )
            else:
                self.communication.rebind_env(self.env)
            self.last_comm_metrics = dict(self.communication.last_metrics)
        else:
            self.communication = None
            self.last_comm_metrics = {
                "comm_v2i_rate_total": 0.0,
                "comm_v2v_success": 0.0,
                "comm_used_rb_ratio": 0.0,
                "comm_mean_power_norm": 0.0,
                "comm_fail_percent": 0.0,
            }
        return self.current_state()

    def _initialize_vehicle_dynamics(self) -> None:
        if self.motion_model != "kinematic_bicycle":
            return
        for veh in getattr(self.env, "vehicles", []):
            self.bicycle_model.initialize_vehicle(veh)

    def current_state(self):
        cached = self._state_cache
        if cached is not None and cached.get("step_count") == self.step_count:
            return {"vector": cached["vector"].copy(), "fields": dict(cached["fields"])}

        state = self.state_builder.build(self.env, self.scheduler, self.step_count)
        comm_metrics = self.last_comm_metrics if self.communication_enabled else {
            "comm_v2i_rate_total": 0.0,
            "comm_v2v_success": 0.0,
            "comm_used_rb_ratio": 0.0,
            "comm_mean_power_norm": 0.0,
            "comm_fail_percent": 0.0,
        }
        state["fields"].update(comm_metrics)
        vector = np.empty(state["vector"].shape[0] + 5, dtype=np.float32)
        vector[: state["vector"].shape[0]] = state["vector"]
        vector[-5:] = np.asarray([
            comm_metrics["comm_v2i_rate_total"] / 200.0,
            comm_metrics["comm_v2v_success"],
            comm_metrics["comm_used_rb_ratio"],
            comm_metrics["comm_mean_power_norm"],
            comm_metrics["comm_fail_percent"],
        ], dtype=np.float32)
        state["vector"] = vector
        self._state_cache = {"step_count": self.step_count, "vector": state["vector"].copy(), "fields": dict(state["fields"]), "comm_metrics": dict(comm_metrics)}
        state["comm_metrics"] = dict(comm_metrics)
        return state

    def tensor_state(self, *, refresh: bool = False) -> TensorizedTrafficState:
        cached = self._tensor_state_cache
        if cached is not None and not refresh:
            return cached
        self._tensor_state_cache = TensorizedTrafficState.from_env(self.env)
        return self._tensor_state_cache

    def tensor_state_dict(self, *, refresh: bool = False) -> Dict[str, object]:
        return self.tensor_state(refresh=refresh).as_dict()

    def set_tensorized_position_step(self, enabled: bool = True) -> None:
        if hasattr(self.env, "set_tensorized_position_step"):
            self.env.set_tensorized_position_step(enabled)

    def step(self, high_level_action: str = "keep") -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        action = high_level_action if high_level_action in HIGH_LEVEL_ACTIONS else "keep"
        if action == self._last_hl_action:
            self._hl_action_streak += 1
        else:
            self._hl_action_streak = 1
        self._last_hl_action = action
        topology_before = int(self.topology_switch_count)
        self._state_cache = None
        self._tensor_state_cache = None
        self._apply_high_level_action(action)
        commands = self.controller.compute_commands(
            self.env,
            self.scheduler,
            self.step_count,
            mode=action,
            comm_metrics=self.last_comm_metrics,
        )
        human_indices = [int(idx) for idx in getattr(self.env, "hv_indices", np.zeros(0, dtype=int)).tolist()]
        if human_indices:
            human_commands = self.human_controller.compute_commands(self.env, self.step_count, human_indices)
            for idx, command in human_commands.items():
                if 0 <= int(idx) < len(commands):
                    commands[int(idx)] = command
        command_stats = self._summarize_commands(commands)
        safe_commands, shield_summary = self.shield.enforce(self.env, commands, scheduler=self.scheduler, step=self.step_count)
        lane_changes = self._apply_commands(safe_commands)

        if self.motion_model == "kinematic_bicycle":
            self.env.post_position_update()
        else:
            self.env.renew_positions()
        refresh_fast_fading = (
            self.step_count == 0
            or (self.step_count % self.communication_fastfading_update_interval_steps) == 0
        )
        self.env.renew_channels_fastfading(
            update_fast_fading=refresh_fast_fading,
            update_slow_channel=True,
        )

        if lane_changes["total"] > 0 or action in {"split", "merge"}:
                self.env._pick_leaders_by_policy()
                self.env._build_topology_two_clusters()

        handovers = self._count_handovers()
        self.total_handovers += handovers
        topology_step_delta = int(self.topology_switch_count - topology_before)
        event_stats = self._collect_event_stats()
        collision_stats = self._collect_collision_stats()
        energy_stats = self._collect_energy_stats(safe_commands)
        if self.communication is not None and (
            self.step_count == 0
            or (self.step_count % self.communication_metrics_update_interval_steps) == 0
            or topology_step_delta > 0
        ):
            self.last_comm_metrics = self.communication.evaluate_step(self.step_count)
        self.step_count += 1

        state = self.current_state()
        info: Dict[str, object] = {
            "lane_changes": float(lane_changes["total"]),
            "lane_changes_cav": float(lane_changes["cav"]),
            "lane_changes_hv": float(lane_changes["hv"]),
            "lane_changes_cum": float(self.lane_change_count),
            "lane_changes_cav_cum": float(self.lane_change_count_cav),
            "lane_changes_hv_cum": float(self.lane_change_count_hv),
            "handovers": float(handovers),
            "handovers_cum": float(self.total_handovers),
            "topology_switches": float(topology_step_delta),
            "topology_switches_cum": float(self.topology_switch_count),
            "shield": shield_summary,
            "topology_type": self.env.topology_type,
            "communication": dict(self.last_comm_metrics),
            "action": action,
            "action_streak": float(self._hl_action_streak),
            "motion_model": self.motion_model,
            "time_s": float(self.step_count) * float(getattr(self.env, "timestep", 0.01)),
            **command_stats,
            **event_stats,
            **collision_stats,
            **energy_stats,
        }
        reward_components = self._reward_components(state["fields"], info)
        reward = float(sum(reward_components.values()))
        done = bool(float(info.get("collision_occurred", 0.0)) > 0.0)
        info["reward_components"] = reward_components
        return state, reward, done, info

    def export_snapshot(self, path: str | Path, show_v2i: bool = True) -> None:
        self.env.visualize(save_path=str(path), show_destinations=True, annotate_power=False, show_v2i=show_v2i)

    def _apply_segment_adjustments(self) -> None:
        if self.scenario_spec is None:
            return
        adjustments = self.segment_adjustments or build_segment_adjustments(self.scenario_spec)
        lane_drop_to = adjustments.get("lane_drop_to")
        if lane_drop_to is not None:
            self.env.lanes_per_dir = min(int(self.env.lanes_per_dir), int(lane_drop_to))
        segments = adjustments.get("segments", [])
        if segments:
            self.env.segment_profile = segments
            self.env.segment_lane_drop_to = lane_drop_to
            self.env.segment_has_ramp_in = bool(adjustments.get("has_ramp_in", False))
            self.env.segment_has_ramp_out = bool(adjustments.get("has_ramp_out", False))
            self.env.segment_has_weaving_zone = bool(adjustments.get("has_weaving_zone", False))

    def _apply_high_level_action(self, action: str) -> None:
        topology_target = None
        if action == "split":
            topology_target = "tree"
        elif action == "merge":
            topology_target = "star"

        if topology_target is not None and topology_target != self.env.topology_type:
            self.env.topology_type = topology_target
            self.topology_switch_count += 1

    def _apply_commands(self, commands) -> Dict[str, int]:
        if self.motion_model == "kinematic_bicycle":
            return self._apply_commands_bicycle(commands)
        return self._apply_commands_point_mass(commands)

    def _apply_commands_point_mass(self, commands) -> Dict[str, int]:
        lane_changes = 0
        lane_changes_cav = 0
        lane_changes_hv = 0
        dt = float(getattr(self.env, "timestep", 0.01))
        lane_positions_up = self.env.true_up_lanes
        lane_positions_down = self.env.true_down_lanes
        clip_margin = 0.5
        lane_commit_x_tol = float(getattr(self.controller, "lane_commit_x_tol", 0.35))

        for idx, command in enumerate(commands):
            veh = self.env.vehicles[idx]
            veh.velocity = float(np.clip(veh.velocity + command.accel * dt, 0.0, self.shield.max_speed))
            current_lane = int(self.env._lane_idx[idx])

            direction = getattr(veh, "direction", "u")
            lane_positions = lane_positions_up if direction == "u" else lane_positions_down
            lane_target_x = float(lane_positions[current_lane])
            min_lane_x = float(min(lane_positions)) - clip_margin
            max_lane_x = float(max(lane_positions)) + clip_margin

            if str(getattr(command, "lateral_mode", "")) == "lane_keep":
                self._abort_lane_change(idx, current_lane)
                hold_speed, hold_error = self.controller.trajectory_tracker.hold(float(veh.position[0]), lane_target_x)
                command.target_lane = None
                command.target_x = lane_target_x
                command.lateral_speed = hold_speed
                command.lateral_error = hold_error
                command.trajectory_progress = 1.0
                command.lane_change_active = False

            target_lane = command.target_lane
            if bool(getattr(command, "lane_change_active", False)) and target_lane is not None:
                target_lane = int(target_lane)
                if not (0 <= target_lane < len(lane_positions)):
                    target_lane = None
                else:
                    lane_target_x = float(lane_positions[target_lane])
            elif target_lane is not None:
                target_lane = int(target_lane)
                if not (0 <= target_lane < len(lane_positions)):
                    target_lane = None
                else:
                    lane_target_x = float(lane_positions[target_lane])

            target_x = lane_target_x if command.target_x is None else float(command.target_x)
            current_x = float(veh.position[0])
            proposed_x = current_x + float(getattr(command, "lateral_speed", 0.0)) * dt
            if (target_x - current_x) * (target_x - proposed_x) <= 0.0:
                proposed_x = target_x
            elif abs(target_x - proposed_x) < 0.03:
                proposed_x = target_x
            veh.position[0] = float(np.clip(proposed_x, min_lane_x, max_lane_x))

            if target_lane is not None and target_lane != current_lane:
                progress = float(getattr(command, "trajectory_progress", 0.0))
                near_target = abs(float(veh.position[0]) - float(lane_target_x)) <= lane_commit_x_tol
                if progress >= 0.999 and near_target:
                    self.env._lane_idx[idx] = target_lane
                    setattr(veh, "last_lane_change_step", int(self.step_count))
                    lane_changes += 1
                    if bool(getattr(veh, "is_cav", False)):
                        lane_changes_cav += 1
                    else:
                        lane_changes_hv += 1

        self.lane_change_count += lane_changes
        self.lane_change_count_cav += lane_changes_cav
        self.lane_change_count_hv += lane_changes_hv
        return {"total": lane_changes, "cav": lane_changes_cav, "hv": lane_changes_hv}

    def _apply_commands_bicycle(self, commands) -> Dict[str, int]:
        lane_changes = 0
        lane_changes_cav = 0
        lane_changes_hv = 0
        dt = float(getattr(self.env, "timestep", 0.01))
        lane_positions_up = self.env.true_up_lanes
        lane_positions_down = self.env.true_down_lanes
        clip_margin = 0.5
        lane_commit_x_tol = float(getattr(self.controller, "lane_commit_x_tol", 0.35))
        y_bounds = (float(getattr(self.env, "base_y", 0.0)), float(self.env.height - getattr(self.env, "base_y", 0.0)))

        for idx, command in enumerate(commands):
            veh = self.env.vehicles[idx]
            current_lane = int(self.env._lane_idx[idx])
            direction = getattr(veh, "direction", "u")
            lane_positions = lane_positions_up if direction == "u" else lane_positions_down
            lane_target_x = float(lane_positions[current_lane])
            min_lane_x = float(min(lane_positions)) - clip_margin
            max_lane_x = float(max(lane_positions)) + clip_margin

            if str(getattr(command, "lateral_mode", "")) == "lane_keep":
                self._abort_lane_change(idx, current_lane)
                hold_speed, hold_error = self.controller.trajectory_tracker.hold(float(veh.position[0]), lane_target_x)
                command.target_lane = None
                command.target_x = lane_target_x
                command.lateral_speed = hold_speed
                command.lateral_error = hold_error
                command.trajectory_progress = 1.0
                command.lane_change_active = False

            target_lane = command.target_lane
            if bool(getattr(command, "lane_change_active", False)) and target_lane is not None:
                target_lane = int(target_lane)
                if not (0 <= target_lane < len(lane_positions)):
                    target_lane = None
                else:
                    lane_target_x = float(lane_positions[target_lane])
            elif target_lane is not None:
                target_lane = int(target_lane)
                if not (0 <= target_lane < len(lane_positions)):
                    target_lane = None
                else:
                    lane_target_x = float(lane_positions[target_lane])

            target_x = lane_target_x if command.target_x is None else float(command.target_x)
            self.bicycle_model.step_vehicle(
                veh,
                accel=float(command.accel),
                desired_lateral_speed=float(getattr(command, "lateral_speed", 0.0)),
                dt=dt,
                x_bounds=(min_lane_x, max_lane_x),
                y_bounds=y_bounds,
            )

            if target_lane is not None and target_lane != current_lane:
                progress = float(getattr(command, "trajectory_progress", 0.0))
                near_target = abs(float(veh.position[0]) - float(target_x)) <= lane_commit_x_tol
                if progress >= 0.999 and near_target:
                    self.env._lane_idx[idx] = target_lane
                    setattr(veh, "last_lane_change_step", int(self.step_count))
                    lane_changes += 1
                    if bool(getattr(veh, "is_cav", False)):
                        lane_changes_cav += 1
                    else:
                        lane_changes_hv += 1

        self.lane_change_count += lane_changes
        self.lane_change_count_cav += lane_changes_cav
        self.lane_change_count_hv += lane_changes_hv
        return {"total": lane_changes, "cav": lane_changes_cav, "hv": lane_changes_hv}

    def _abort_lane_change(self, idx: int, current_lane: int) -> None:
        plans = getattr(self.controller, "_lane_change_plans", None)
        if isinstance(plans, list) and 0 <= idx < len(plans):
            plans[idx] = None
        hold_counter = getattr(self.controller, "_action_hold_counter", None)
        if isinstance(hold_counter, dict):
            hold_counter[idx] = 1

    def _count_handovers(self) -> int:
        current = np.asarray(getattr(self.env, "v2i_serving_idx", np.zeros(0, dtype=int)), dtype=int)
        if current.size == 0:
            self.last_serving_idx = current.copy()
            return 0
        if self.last_serving_idx.shape != current.shape:
            self.last_serving_idx = current.copy()
            return 0
        handovers = int(np.sum(current != self.last_serving_idx))
        self.last_serving_idx = current.copy()
        return handovers

    def _summarize_commands(self, commands) -> Dict[str, float]:
        if not commands:
            return {
                "avg_abs_accel_cmd": 0.0,
                "max_abs_accel_cmd": 0.0,
                "avg_target_speed_cmd": 0.0,
                "avg_desired_gap_cmd": 0.0,
                "avg_lateral_speed_cmd": 0.0,
                "avg_lateral_error_cmd": 0.0,
                "max_lateral_speed_cmd": 0.0,
                "active_lane_change_cmds": 0.0,
                "cacc_cmd_ratio": 0.0,
                "hybrid_cmd_ratio": 0.0,
                "cidm_cmd_ratio": 0.0,
                "avg_comm_reliability_cmd": 0.0,
                "avg_cacc_weight_cmd": 0.0,
                "avg_headway_scale_cmd": 0.0,
                "avg_info_topology_level_cmd": 0.0,
                "avg_mode_risk_cmd": 0.0,
            }
        accels = np.asarray([abs(float(cmd.accel)) for cmd in commands], dtype=np.float32)
        target_speeds = np.asarray([float(cmd.target_speed) for cmd in commands], dtype=np.float32)
        desired_gaps = np.asarray([float(cmd.desired_gap) for cmd in commands], dtype=np.float32)
        lateral_speeds = np.asarray([abs(float(getattr(cmd, "lateral_speed", 0.0))) for cmd in commands], dtype=np.float32)
        lateral_errors = np.asarray([abs(float(getattr(cmd, "lateral_error", 0.0))) for cmd in commands], dtype=np.float32)
        comm_reliability = np.asarray([float(getattr(cmd, "comm_reliability", 0.0)) for cmd in commands], dtype=np.float32)
        cacc_weights = np.asarray([float(getattr(cmd, "cacc_weight", 0.0)) for cmd in commands], dtype=np.float32)
        headway_scales = np.asarray([float(getattr(cmd, "headway_scale", 1.0)) for cmd in commands], dtype=np.float32)
        info_levels = np.asarray([float(getattr(cmd, "info_topology_level", 0.0)) for cmd in commands], dtype=np.float32)
        mode_risks = np.asarray([float(getattr(cmd, "mode_risk", 0.0)) for cmd in commands], dtype=np.float32)
        long_modes = np.asarray([str(getattr(cmd, "longitudinal_mode", "")).lower() for cmd in commands], dtype=object)
        cacc_count = float(np.sum(long_modes == "cacc"))
        hybrid_count = float(np.sum(long_modes == "hybrid"))
        cidm_count = float(np.sum(long_modes == "cidm"))
        return {
            "avg_abs_accel_cmd": float(np.mean(accels)),
            "max_abs_accel_cmd": float(np.max(accels)),
            "avg_target_speed_cmd": float(np.mean(target_speeds)),
            "avg_desired_gap_cmd": float(np.mean(desired_gaps)),
            "avg_lateral_speed_cmd": float(np.mean(lateral_speeds)),
            "avg_lateral_error_cmd": float(np.mean(lateral_errors)),
            "max_lateral_speed_cmd": float(np.max(lateral_speeds)),
            "active_lane_change_cmds": float(np.sum([1.0 if getattr(cmd, "lane_change_active", False) else 0.0 for cmd in commands])),
            "cacc_cmd_ratio": float(cacc_count) / float(max(1, len(commands))),
            "hybrid_cmd_ratio": float(hybrid_count) / float(max(1, len(commands))),
            "cidm_cmd_ratio": float(cidm_count) / float(max(1, len(commands))),
            "avg_comm_reliability_cmd": float(np.mean(comm_reliability)),
            "avg_cacc_weight_cmd": float(np.mean(cacc_weights)),
            "avg_headway_scale_cmd": float(np.mean(headway_scales)),
            "avg_info_topology_level_cmd": float(np.mean(info_levels)),
            "avg_mode_risk_cmd": float(np.mean(mode_risks)),
        }

    def _direction_event_bounds(self, direction: str):
        cache_key = f"bounds_{direction}"
        if cache_key in self._event_cache:
            return self._event_cache[cache_key]
        events = [event for event in getattr(self.scheduler, "events", []) if direction in event.affected_directions]
        core_events = [
            event
            for event in events
            if (not bool(getattr(event, "advisory_only", False)))
            or bool(getattr(event, "blocked_lanes", {}))
            or str(getattr(event, "phase_tag", "")).lower() == "incident"
        ]
        relevant = core_events or events
        if not relevant:
            self._event_cache[cache_key] = None
            return None
        y_start = min(event.y_start for event in relevant)
        y_end = max(event.y_end for event in relevant)
        bounds = (float(y_start), float(y_end))
        self._event_cache[cache_key] = bounds
        return bounds

    def _collect_event_stats(self) -> Dict[str, float]:
        event_zone_count = 0
        blocked_lane_vehicle_count = 0
        event_distance_up = []
        event_distance_down = []
        bounds_up = self._direction_event_bounds("u")
        bounds_down = self._direction_event_bounds("d")
        active_events = self.scheduler.active_events(self.step_count) if self.scheduler.has_active_event(self.step_count) else ()

        for idx, veh in enumerate(self.env.vehicles):
            direction = getattr(veh, "direction", "u")
            lane_idx = int(self.env._lane_idx[idx])
            y = float(veh.position[1])
            if active_events:
                if any(event.contains_y(y) and direction in event.affected_directions for event in active_events):
                    event_zone_count += 1
                if self.scheduler.lane_blocked(direction, lane_idx, y, self.step_count, margin=15.0):
                    blocked_lane_vehicle_count += 1
            dist = self.scheduler.nearest_event_distance(direction, y, self.step_count)
            if dist is not None:
                if direction == "u":
                    event_distance_up.append(float(dist))
                elif direction == "d":
                    event_distance_down.append(float(dist))

            if bounds_up is not None and direction == "u":
                if bounds_up[0] <= y <= bounds_up[1]:
                    self._event_entered_up[idx] = True
                if self._event_entered_up[idx] and y >= bounds_up[1]:
                    self._event_passed_up[idx] = True
            if bounds_down is not None and direction == "d":
                if bounds_down[0] <= y <= bounds_down[1]:
                    self._event_entered_down[idx] = True
                if self._event_entered_down[idx] and y <= bounds_down[0]:
                    self._event_passed_down[idx] = True

        n_up = max(1, sum(1 for d in getattr(self.env, "_group_dir", []) if d == "u"))
        n_down = max(1, sum(1 for d in getattr(self.env, "_group_dir", []) if d == "d"))
        return {
            "event_zone_vehicle_count": float(event_zone_count),
            "blocked_lane_vehicle_count": float(blocked_lane_vehicle_count),
            "avg_event_distance_up": float(np.mean(event_distance_up)) if event_distance_up else 0.0,
            "avg_event_distance_down": float(np.mean(event_distance_down)) if event_distance_down else 0.0,
            "passed_event_count_up": float(np.sum(self._event_passed_up)),
            "passed_event_count_down": float(np.sum(self._event_passed_down)),
            "passed_event_ratio_up": float(np.sum(self._event_passed_up)) / float(n_up),
            "passed_event_ratio_down": float(np.sum(self._event_passed_down)) / float(n_down),
        }

    def _collect_collision_stats(self) -> Dict[str, float]:
        vehicles = getattr(self.env, "vehicles", [])
        if len(vehicles) < 2:
            self._prev_collision_pairs = set()
            return {
                "collision_count": 0.0,
                "collision_pairs_active": 0.0,
                "collision_vehicles_active": 0.0,
                "collision_occurred": 0.0,
                "collision_count_cum": float(self.total_collision_count),
                "collision_step_count_cum": float(self.total_collision_steps),
            }

        lane_width = float(getattr(self.env, "lane_width", 3.5))
        default_width = float(self.collision_width) if self.collision_width is not None else min(2.2, 0.65 * lane_width)
        default_length = float(max(self.collision_length, 0.1))
        active_pairs: set[tuple[int, int]] = set()
        collision_boxes = [self._vehicle_collision_box(veh, default_length, default_width) for veh in vehicles]

        for idx in range(len(vehicles) - 1):
            box_i = collision_boxes[idx]
            for jdx in range(idx + 1, len(vehicles)):
                box_j = collision_boxes[jdx]
                if not self._collision_broad_phase(box_i, box_j):
                    continue
                if self._oriented_boxes_overlap(box_i, box_j):
                    active_pairs.add((idx, jdx))

        new_pairs = active_pairs.difference(self._prev_collision_pairs)
        if active_pairs:
            self.total_collision_steps += 1
        self.total_collision_count += len(new_pairs)
        self._prev_collision_pairs = active_pairs

        collision_vehicles = set()
        for idx, jdx in active_pairs:
            collision_vehicles.add(int(idx))
            collision_vehicles.add(int(jdx))

        return {
            "collision_count": float(len(new_pairs)),
            "collision_pairs_active": float(len(active_pairs)),
            "collision_vehicles_active": float(len(collision_vehicles)),
            "collision_occurred": 1.0 if active_pairs else 0.0,
            "collision_count_cum": float(self.total_collision_count),
            "collision_step_count_cum": float(self.total_collision_steps),
        }

    def _collect_energy_stats(self, commands) -> Dict[str, float]:
        dt = max(float(getattr(self.env, "timestep", 0.01)), 1e-6)
        total_kj = 0.0
        cav_kj = 0.0
        hv_kj = 0.0
        for idx, veh in enumerate(getattr(self.env, "vehicles", [])):
            command = commands[idx] if idx < len(commands) else None
            accel = float(getattr(command, "accel", 0.0)) if command is not None else 0.0
            energy_kj = self._vehicle_step_energy_kj(veh, accel=accel, dt=dt)
            total_kj += energy_kj
            if bool(getattr(veh, "is_cav", False)):
                cav_kj += energy_kj
            else:
                hv_kj += energy_kj

        self.total_energy_kj += total_kj
        self.total_energy_cav_kj += cav_kj
        self.total_energy_hv_kj += hv_kj
        nveh = max(1, len(getattr(self.env, "vehicles", [])))
        return {
            "energy_step_kj": float(total_kj),
            "energy_step_cav_kj": float(cav_kj),
            "energy_step_hv_kj": float(hv_kj),
            "energy_cum_kj": float(self.total_energy_kj),
            "energy_cum_cav_kj": float(self.total_energy_cav_kj),
            "energy_cum_hv_kj": float(self.total_energy_hv_kj),
            "energy_step_per_vehicle_kj": float(total_kj) / float(nveh),
        }

    def _vehicle_step_energy_kj(self, veh, *, accel: float, dt: float) -> float:
        speed = max(float(getattr(veh, "velocity", 0.0)), 0.0)
        length = float(getattr(veh, "vehicle_length", 4.8))
        is_truck = length > 7.5
        mass = 8500.0 if is_truck else 1600.0
        c_rr = 0.012 if not is_truck else 0.010
        cd_area = 5.40 if is_truck else 0.72
        rho_air = 1.225
        gravity = 9.81
        rolling_force = c_rr * mass * gravity
        drag_force = 0.5 * rho_air * cd_area * speed * speed
        inertial_force = mass * max(float(accel), 0.0)
        traction_power_w = max(0.0, (rolling_force + drag_force + inertial_force) * speed)
        return float((traction_power_w * dt) / 1000.0)

    def _vehicle_collision_box(self, veh, default_length: float, default_width: float) -> Dict[str, np.ndarray | float]:
        length = max(float(default_length), float(getattr(veh, "vehicle_length", default_length)), 0.1)
        if self.collision_width is not None:
            width = max(float(self.collision_width), float(getattr(veh, "vehicle_width", self.collision_width)), 0.1)
        else:
            width = max(float(getattr(veh, "vehicle_width", default_width)), 0.1)

        yaw = float(getattr(veh, "yaw", self._default_vehicle_yaw(getattr(veh, "direction", "u"))))
        center = np.asarray([float(veh.position[0]), float(veh.position[1])], dtype=np.float64)
        forward = np.asarray([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
        lateral = np.asarray([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)
        half_length = 0.5 * length
        half_width = 0.5 * width
        return {
            "center": center,
            "forward": forward,
            "lateral": lateral,
            "half_length": float(half_length),
            "half_width": float(half_width),
            "radius": float(np.hypot(half_length, half_width)),
        }

    def _default_vehicle_yaw(self, direction: str) -> float:
        direction = str(direction).lower()
        if direction == "d":
            return float(-0.5 * np.pi)
        return float(0.5 * np.pi)

    def _collision_broad_phase(self, box_i: Dict[str, np.ndarray | float], box_j: Dict[str, np.ndarray | float]) -> bool:
        center_delta = np.asarray(box_j["center"], dtype=np.float64) - np.asarray(box_i["center"], dtype=np.float64)
        radius_sum = float(box_i["radius"]) + float(box_j["radius"])
        return float(np.dot(center_delta, center_delta)) <= radius_sum * radius_sum

    def _oriented_boxes_overlap(self, box_i: Dict[str, np.ndarray | float], box_j: Dict[str, np.ndarray | float]) -> bool:
        center_delta = np.asarray(box_j["center"], dtype=np.float64) - np.asarray(box_i["center"], dtype=np.float64)
        axes = (
            np.asarray(box_i["forward"], dtype=np.float64),
            np.asarray(box_i["lateral"], dtype=np.float64),
            np.asarray(box_j["forward"], dtype=np.float64),
            np.asarray(box_j["lateral"], dtype=np.float64),
        )
        for axis in axes:
            projection_gap = abs(float(np.dot(center_delta, axis)))
            radius_i = self._box_projection_radius(box_i, axis)
            radius_j = self._box_projection_radius(box_j, axis)
            if projection_gap > radius_i + radius_j:
                return False
        return True

    def _box_projection_radius(self, box: Dict[str, np.ndarray | float], axis: np.ndarray) -> float:
        forward = np.asarray(box["forward"], dtype=np.float64)
        lateral = np.asarray(box["lateral"], dtype=np.float64)
        half_length = float(box["half_length"])
        half_width = float(box["half_width"])
        return float(abs(half_length * np.dot(forward, axis)) + abs(half_width * np.dot(lateral, axis)))

    def _reward_components(self, fields: Dict[str, float], info: Dict[str, object]) -> Dict[str, float]:
        min_gap = min(fields["min_gap_up"], fields["min_gap_down"])
        mean_speed = 0.5 * (fields["mean_speed_up"] + fields["mean_speed_down"])
        shield = info.get("shield", {})
        interventions = float(shield.get("interventions", 0.0))
        emergency_brakes = float(shield.get("emergency_brakes", 0.0))
        collision_count = float(info.get("collision_count", 0.0))
        collision_active = float(info.get("collision_occurred", 0.0))
        platoon_rate = float(fields.get("platoon_rate", 0.0))
        comm_v2v = float(fields.get("comm_v2v_success", 0.0))
        comm_fail = float(fields.get("comm_fail_percent", 0.0))
        event_pressure = 0.5 * (fields["event_pressure_up"] + fields["event_pressure_down"])
        blocked_ratio = max(float(fields.get("blocked_lane_ratio_up", 0.0)), float(fields.get("blocked_lane_ratio_down", 0.0)))
        action = str(info.get("action", "keep")).lower()
        platoon_active = float(platoon_rate >= 0.10 and min_gap >= 9.0 and comm_fail <= 0.05)
        if platoon_active > 0.0:
            self._platoon_stability_streak = min(self._platoon_stability_streak + 1, 50)
        else:
            self._platoon_stability_streak = max(self._platoon_stability_streak - 1, 0)
        platoon_stability = min(float(self._platoon_stability_streak) / 12.0, 1.0)

        safety = (
            -3.0 * collision_count
            - 1.5 * collision_active
            - 0.30 * emergency_brakes
            - 0.05 * interventions
            - 0.20 * max(0.0, 8.0 - min_gap) / 8.0
        )
        if min_gap >= 10.0 and interventions <= 0.0 and emergency_brakes <= 0.0:
            safety += 0.08

        platoon = 0.50 * platoon_rate * (0.6 + 0.4 * platoon_stability)

        comm_platoon_synergy = 0.15 * platoon_rate * comm_v2v
        communication = 0.20 * comm_v2v - 0.20 * comm_fail + comm_platoon_synergy

        speed_ratio = float(mean_speed / max(1.0, self.controller.cruise_speed))
        efficiency = 0.10 * speed_ratio - 0.03 * event_pressure

        action_penalty = 0.0
        emergency_needed = self._emergency_action_needed(
            fields, info, min_gap=min_gap, event_pressure=event_pressure, blocked_ratio=blocked_ratio
        )
        if action == "emergency" and not emergency_needed:
            action_penalty = -0.35
        if action == "compact" and min_gap < 10.0:
            action_penalty += -0.10

        return {
            "safety": float(safety),
            "platoon": float(platoon),
            "communication": float(communication),
            "efficiency": float(efficiency),
            "action_appropriateness": float(action_penalty),
        }

    def _compute_reward(self, fields: Dict[str, float], info: Dict[str, object]) -> float:
        components = self._reward_components(fields, info)
        return float(sum(components.values()))

    def _emergency_action_needed(
        self,
        fields: Dict[str, float],
        info: Dict[str, object],
        *,
        min_gap: float,
        event_pressure: float,
        blocked_ratio: float,
    ) -> bool:
        shield = info.get("shield", {})
        interventions = float(shield.get("interventions", 0.0))
        comm_fail = float(fields.get("comm_fail_percent", 0.0))
        comm_v2v = float(fields.get("comm_v2v_success", 1.0))
        collision_risk = float(info.get("collision_occurred", 0.0)) > 0.0
        tight_gap = min_gap > 1e-6 and min_gap < 9.0
        event_tight_gap = min_gap > 1e-6 and min_gap < 11.0 and event_pressure >= 0.35
        blocked_incident = event_pressure >= 0.65 and blocked_ratio > 0.0
        degraded_links = event_pressure >= 0.50 and (comm_fail >= 0.12 or comm_v2v <= 0.75)
        shield_pressure = interventions >= 2.0 and event_pressure >= 0.25
        return bool(collision_risk or tight_gap or event_tight_gap or blocked_incident or degraded_links or shield_pressure)

    def _split_action_needed(self, fields: Dict[str, float], *, event_pressure: float, blocked_ratio: float) -> bool:
        lane_imbalance = max(
            float(fields.get("lane_occupancy_std_up", 0.0)),
            float(fields.get("lane_occupancy_std_down", 0.0)),
        )
        mpr = float(fields.get("mpr_cav", 0.0))
        incident_split = event_pressure >= 0.45 and blocked_ratio > 0.0
        mixed_flow_split = 0.25 <= mpr <= 0.55 and event_pressure >= 0.30 and lane_imbalance >= 1.0
        return bool(incident_split or mixed_flow_split)
