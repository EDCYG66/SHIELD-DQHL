"""High-level platoon reconfiguration policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:
    from .events import EventScheduler
except ImportError:  # pragma: no cover
    from events import EventScheduler


@dataclass
class PolicyDecision:
    action: str
    score: float = 0.0
    reason: str = ""


class FormationHighLevelPolicy:
    """Base class for high-level platoon decision policies."""

    name = "base"

    def select_action(
        self,
        step: int,
        scheduler: EventScheduler,
        state_fields: Dict[str, float],
        state_vector: Optional[np.ndarray] = None,
        training: bool = False,
    ) -> PolicyDecision:
        raise NotImplementedError

    def _event_mode(self, step: int, scheduler: EventScheduler, state_fields: Dict[str, float]) -> str:
        if scheduler.has_active_event(step):
            return "event"
        if scheduler.events and scheduler.horizon_end < int(step) <= scheduler.horizon_end + 25:
            return "recovery"
        return "cruise"

    def _safety_pressure(self, state_fields: Dict[str, float]) -> float:
        min_gap = min(float(state_fields.get("min_gap_up", 0.0)), float(state_fields.get("min_gap_down", 0.0)))
        blocked = max(float(state_fields.get("blocked_lane_ratio_up", 0.0)), float(state_fields.get("blocked_lane_ratio_down", 0.0)))
        pressure = 0.0
        if min_gap < 12.0:
            pressure += (12.0 - min_gap) / 12.0
        pressure += blocked
        pressure += max(float(state_fields.get("event_pressure_up", 0.0)), float(state_fields.get("event_pressure_down", 0.0)))
        return float(np.clip(pressure, 0.0, 3.0))


class HeuristicFormationPolicy(FormationHighLevelPolicy):
    """Event-driven heuristic baseline."""

    name = "heuristic"

    def select_action(self, step: int, scheduler: EventScheduler, state_fields: Dict[str, float], state_vector=None, training: bool = False) -> PolicyDecision:
        mode = self._event_mode(step, scheduler, state_fields)
        pressure = self._safety_pressure(state_fields)
        blocked_up = float(state_fields.get("blocked_lane_ratio_up", 0.0))
        blocked_down = float(state_fields.get("blocked_lane_ratio_down", 0.0))
        min_gap = min(state_fields["min_gap_up"], state_fields["min_gap_down"])

        if mode == "event":
            if blocked_up > 0.0 or blocked_down > 0.0:
                return PolicyDecision("split", score=max(blocked_up, blocked_down), reason="blocked lane under active event")
            if pressure > 0.75 or min_gap < 10.0:
                return PolicyDecision("expand", score=pressure, reason="safety pressure too high")
            if pressure < 0.35:
                return PolicyDecision("compact", score=1.0 - pressure, reason="active event but safe headway")
            return PolicyDecision("keep", score=1.0 - pressure, reason="maintain stability during event")

        if mode == "recovery":
            if min_gap < 18.0:
                return PolicyDecision("keep", score=min_gap, reason="hold recovery until spacing rebuilds")
            if pressure > 0.5:
                return PolicyDecision("expand", score=pressure, reason="recovery with residual pressure")
            return PolicyDecision("merge", score=1.0 - pressure, reason="post-event recovery")
        return PolicyDecision("keep", reason="normal cruising")


class CommunicationAwareFormationPolicy(FormationHighLevelPolicy):
    """Adds communication quality into the reconfiguration logic."""

    name = "comm_aware"

    def select_action(self, step: int, scheduler: EventScheduler, state_fields: Dict[str, float], state_vector=None, training: bool = False) -> PolicyDecision:
        mode = self._event_mode(step, scheduler, state_fields)
        comm_v2v = float(state_fields.get("comm_v2v_success", 0.0))
        comm_fail = float(state_fields.get("comm_fail_percent", 0.0))
        blocked_ratio = max(
            float(state_fields.get("blocked_lane_ratio_up", 0.0)),
            float(state_fields.get("blocked_lane_ratio_down", 0.0)),
        )
        min_gap = min(float(state_fields.get("min_gap_up", 0.0)), float(state_fields.get("min_gap_down", 0.0)))
        pressure = self._safety_pressure(state_fields)

        if mode == "event":
            if blocked_ratio > 0.0 and comm_v2v >= 0.92:
                return PolicyDecision("split", score=comm_v2v, reason="event + strong communication support")
            if comm_fail > 0.08 or pressure > 0.85:
                return PolicyDecision("expand", score=max(1.0 - comm_fail, pressure), reason="communication degradation or safety pressure")
            if min_gap < 10.0:
                return PolicyDecision("expand", score=min_gap, reason="tight spacing near event")
            return PolicyDecision("compact", score=comm_v2v, reason="event handling with stable links")

        if mode == "recovery":
            if min_gap < 18.0:
                return PolicyDecision("keep", score=comm_v2v, reason="hold recovery until spacing rebuilds")
            if comm_v2v >= 0.95 and pressure < 0.5:
                return PolicyDecision("merge", score=comm_v2v, reason="recover to compact structure")
            if pressure > 0.75:
                return PolicyDecision("expand", score=pressure, reason="recovery with residual safety pressure")
            return PolicyDecision("keep", score=comm_v2v, reason="wait for communication recovery")
        return PolicyDecision("keep", score=comm_v2v, reason="steady cruise")


class ConservativeFormationPolicy(FormationHighLevelPolicy):
    """More cautious policy that prioritizes safety margins."""

    name = "conservative"

    def select_action(self, step: int, scheduler: EventScheduler, state_fields: Dict[str, float], state_vector=None, training: bool = False) -> PolicyDecision:
        min_gap = min(float(state_fields.get("min_gap_up", 0.0)), float(state_fields.get("min_gap_down", 0.0)))
        event_pressure = max(
            float(state_fields.get("event_pressure_up", 0.0)),
            float(state_fields.get("event_pressure_down", 0.0)),
        )
        blocked_ratio = max(
            float(state_fields.get("blocked_lane_ratio_up", 0.0)),
            float(state_fields.get("blocked_lane_ratio_down", 0.0)),
        )
        pressure = self._safety_pressure(state_fields)
        mode = self._event_mode(step, scheduler, state_fields)

        if mode == "event":
            if blocked_ratio > 0.0:
                return PolicyDecision("split", score=blocked_ratio, reason="explicit bottleneck blocking")
            if min_gap < 12.0 or event_pressure > 0.25 or pressure > 0.8:
                return PolicyDecision("expand", score=max(event_pressure, 12.0 - min_gap, pressure), reason="safety-first reaction")
            return PolicyDecision("keep", score=min_gap, reason="stable spacing during event")

        if mode == "recovery":
            if min_gap < 18.0:
                return PolicyDecision("keep", score=min_gap, reason="hold recovery until spacing rebuilds")
            if pressure > 0.45:
                return PolicyDecision("expand", score=pressure, reason="gradual recovery with residual pressure")
            return PolicyDecision("merge", score=1.0, reason="gradual recovery after event")
        return PolicyDecision("keep", score=min_gap, reason="normal operation")


def build_policy(name: str, **kwargs) -> FormationHighLevelPolicy:
    tag = str(name or "").strip().lower()
    if tag in {"heuristic", "baseline"}:
        return HeuristicFormationPolicy()
    if tag in {"comm", "comm_aware", "communication", "communication_aware"}:
        return CommunicationAwareFormationPolicy()
    if tag in {"conservative", "safe"}:
        return ConservativeFormationPolicy()
    if tag in {"mobil_cidm_cacc", "mobil", "mobil_rule"}:
        try:
            from .mobil_high_level_policy import MOBILCIDMCACCHighLevelPolicy
        except ImportError:  # pragma: no cover
            from mobil_high_level_policy import MOBILCIDMCACCHighLevelPolicy
        return MOBILCIDMCACCHighLevelPolicy()
    if tag in {"vanilla_ddqn", "vanilla_dqn", "plain_ddqn", "plain_dqn"}:
        try:
            from .vanilla_high_level_policy import VanillaDDQNHighLevelPolicy
        except ImportError:  # pragma: no cover
            from vanilla_high_level_policy import VanillaDDQNHighLevelPolicy
        policy = VanillaDDQNHighLevelPolicy(
            state_dim=kwargs.get("state_dim"),
            hidden_dims=tuple(kwargs.get("hidden_dims", (128, 96))),
            gamma=float(kwargs.get("gamma", 0.95)),
            learning_rate=float(kwargs.get("learning_rate", 1e-3)),
            replay_size=int(kwargs.get("replay_size", 20000)),
            batch_size=int(kwargs.get("batch_size", 128)),
            target_update_interval=int(kwargs.get("target_update_interval", 100)),
            min_buffer_before_train=int(kwargs.get("min_buffer_before_train", 256)),
            epsilon_start=float(kwargs.get("epsilon_start", 1.0)),
            epsilon_end=float(kwargs.get("epsilon_end", 0.05)),
            epsilon_decay_steps=int(kwargs.get("epsilon_decay_steps", 6000)),
            seed=int(kwargs.get("seed", 123)),
        )
        weights_path = kwargs.get("weights_path")
        meta_path = kwargs.get("meta_path")
        if weights_path:
            policy.load(weights_path, meta_path=meta_path)
        return policy
    if tag in {"ppo"}:
        try:
            from .ppo_high_level_policy import PPOHighLevelPolicy
        except ImportError:  # pragma: no cover
            from ppo_high_level_policy import PPOHighLevelPolicy
        policy = PPOHighLevelPolicy(
            state_dim=kwargs.get("state_dim"),
            hidden_dims=tuple(kwargs.get("hidden_dims", (128, 96))),
            gamma=float(kwargs.get("gamma", 0.99)),
            gae_lambda=float(kwargs.get("gae_lambda", 0.95)),
            learning_rate=float(kwargs.get("learning_rate", 3e-4)),
            clip_ratio=float(kwargs.get("clip_ratio", 0.2)),
            entropy_coef=float(kwargs.get("entropy_coef", 0.01)),
            value_coef=float(kwargs.get("value_coef", 0.5)),
            train_epochs=int(kwargs.get("train_epochs", 4)),
            minibatch_size=int(kwargs.get("minibatch_size", 128)),
            rollout_capacity=int(kwargs.get("rollout_capacity", 4096)),
            seed=int(kwargs.get("seed", 123)),
        )
        weights_path = kwargs.get("weights_path")
        meta_path = kwargs.get("meta_path")
        if weights_path:
            policy.load(weights_path, meta_path=meta_path)
        return policy
    if tag in {"learned", "trainable", "dqn"}:
        try:
            from .trainable_high_level_policy import TrainableHighLevelPolicy
        except ImportError:  # pragma: no cover
            from trainable_high_level_policy import TrainableHighLevelPolicy
        policy = TrainableHighLevelPolicy(
            state_dim=kwargs.get("state_dim"),
            hidden_dims=tuple(kwargs.get("hidden_dims", (128, 96))),
            gamma=float(kwargs.get("gamma", 0.95)),
            learning_rate=float(kwargs.get("learning_rate", 1e-3)),
            replay_size=int(kwargs.get("replay_size", 20000)),
            batch_size=int(kwargs.get("batch_size", 128)),
            target_update_interval=int(kwargs.get("target_update_interval", 100)),
            min_buffer_before_train=int(kwargs.get("min_buffer_before_train", 256)),
            epsilon_start=float(kwargs.get("epsilon_start", 1.0)),
            epsilon_end=float(kwargs.get("epsilon_end", 0.05)),
            epsilon_decay_steps=int(kwargs.get("epsilon_decay_steps", 6000)),
            max_split_streak=int(kwargs.get("max_split_streak", 24)),
            max_reconfig_streak=int(kwargs.get("max_reconfig_streak", 36)),
            split_cooldown_steps=int(kwargs.get("split_cooldown_steps", 18)),
            merge_cooldown_steps=int(kwargs.get("merge_cooldown_steps", 16)),
            max_merge_streak=int(kwargs.get("max_merge_streak", 18)),
            seed=int(kwargs.get("seed", 123)),
            mc_dropout_samples=int(kwargs.get("mc_dropout_samples", 10)),
            uncertainty_beta_base=float(kwargs.get("uncertainty_beta_base", 0.30)),
            uncertainty_beta_max=float(kwargs.get("uncertainty_beta_max", 1.20)),
        )
        weights_path = kwargs.get("weights_path")
        meta_path = kwargs.get("meta_path")
        if weights_path:
            policy.load(weights_path, meta_path=meta_path)
        return policy
    raise ValueError(f"Unknown formation policy: {name}")
