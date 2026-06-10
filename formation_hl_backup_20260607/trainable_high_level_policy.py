"""Trainable high-level platoon policy with replay buffer and TF backend."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import json
import random

import numpy as np

try:
    from .high_level_policy import FormationHighLevelPolicy, PolicyDecision
except ImportError:  # pragma: no cover
    from high_level_policy import FormationHighLevelPolicy, PolicyDecision


TRAINABLE_ACTIONS: Tuple[str, ...] = ("keep", "compact", "expand", "split", "merge", "emergency")


@dataclass
class HighLevelTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: float


class HighLevelReplayBuffer:
    """Simple replay buffer for high-level platoon decisions."""

    def __init__(self, capacity: int = 20000):
        self.capacity = int(capacity)
        self.buffer: Deque[HighLevelTransition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, state, action: int, reward: float, next_state, done: bool) -> None:
        self.buffer.append(
            HighLevelTransition(
                state=np.asarray(state, dtype=np.float32).copy(),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32).copy(),
                done=float(done),
            )
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, int(batch_size))
        states = np.stack([item.state for item in batch], axis=0)
        actions = np.asarray([item.action for item in batch], dtype=np.int32)
        rewards = np.asarray([item.reward for item in batch], dtype=np.float32)
        next_states = np.stack([item.next_state for item in batch], axis=0)
        dones = np.asarray([item.done for item in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones


class TrainableHighLevelPolicy(FormationHighLevelPolicy):
    """DQN-style high-level policy for formation reconfiguration."""

    name = "learned"

    def __init__(
        self,
        *,
        state_dim: Optional[int] = None,
        hidden_dims: Tuple[int, int] = (128, 96),
        gamma: float = 0.95,
        learning_rate: float = 1e-3,
        replay_size: int = 20000,
        batch_size: int = 128,
        target_update_interval: int = 100,
        min_buffer_before_train: int = 256,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 15000,
        max_split_streak: int = 24,
        max_reconfig_streak: int = 36,
        split_cooldown_steps: int = 18,
        merge_cooldown_steps: int = 16,
        emergency_cooldown_steps: int = 14,
        max_merge_streak: int = 18,
        max_emergency_streak: int = 8,
        max_expand_streak: int = 16,
        seed: int = 123,
        mc_dropout_samples: int = 10,
        uncertainty_beta_base: float = 0.30,
        uncertainty_beta_max: float = 1.20,
    ):
        self.state_dim = state_dim
        self.hidden_dims = tuple(int(v) for v in hidden_dims)
        self.mc_dropout_samples = int(mc_dropout_samples)
        self.uncertainty_beta_base = float(uncertainty_beta_base)
        self.uncertainty_beta_max = float(uncertainty_beta_max)
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.target_update_interval = int(target_update_interval)
        self.min_buffer_before_train = int(min_buffer_before_train)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.max_split_streak = int(max_split_streak)
        self.max_reconfig_streak = int(max_reconfig_streak)
        self.split_cooldown_steps = int(split_cooldown_steps)
        self.merge_cooldown_steps = int(merge_cooldown_steps)
        self.emergency_cooldown_steps = int(emergency_cooldown_steps)
        self.max_merge_streak = int(max_merge_streak)
        self.max_emergency_streak = int(max_emergency_streak)
        self.max_expand_streak = int(max_expand_streak)
        self.seed = int(seed)
        self.num_actions = len(TRAINABLE_ACTIONS)
        self.buffer = HighLevelReplayBuffer(capacity=replay_size)
        self.training_step = 0
        self.episode_count = 0
        self.loss_history: List[Tuple[int, float]] = []
        self.action_history: List[Tuple[int, int]] = []
        self.last_action_counts: Dict[str, int] = {name: 0 for name in TRAINABLE_ACTIONS}
        self._rng = np.random.default_rng(self.seed)
        self._tf = None
        self.model = None
        self.target_model = None
        self._compiled_train_step = None
        self._last_action_idx: Optional[int] = None
        self._switch_count = 0
        self._governor_last_action_idx: Optional[int] = None
        self._governor_action_streak = 0
        self._governor_split_cooldown = 0
        self._governor_merge_cooldown = 0
        self._governor_emergency_cooldown = 0
        if state_dim is not None:
            self._ensure_models(int(state_dim))

    # ---------- Model helpers ----------

    def _ensure_tf(self):
        if self._tf is None:
            try:
                from tf_runtime import configure_tensorflow_runtime
                configure_tensorflow_runtime()
                import tensorflow as tf  # pylint: disable=import-error
                configure_tensorflow_runtime(tf)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "TensorFlow is required for TrainableHighLevelPolicy. "
                    "Please run this policy inside the tf2.12 environment."
                ) from exc
            self._tf = tf
        return self._tf

    def _ensure_models(self, state_dim: int):
        if self.model is not None and self.target_model is not None and self.state_dim == state_dim:
            return
        tf = self._ensure_tf()
        self.state_dim = int(state_dim)

        def build_net():
            he = tf.keras.initializers.HeNormal(seed=self.seed)
            return tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(self.state_dim,)),
                tf.keras.layers.Dense(self.hidden_dims[0], activation="relu", kernel_initializer=he),
                tf.keras.layers.Dropout(0.10),
                tf.keras.layers.Dense(self.hidden_dims[1], activation="relu", kernel_initializer=he),
                tf.keras.layers.Dropout(0.10),
                tf.keras.layers.Dense(self.num_actions, activation=None),
            ])

        self.model = build_net()
        self.target_model = build_net()
        _ = self.model(tf.zeros((1, self.state_dim), dtype=tf.float32), training=False)
        _ = self.target_model(tf.zeros((1, self.state_dim), dtype=tf.float32), training=False)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = tf.keras.losses.Huber()

        @tf.function(reduce_retracing=True)
        def compiled_train_step(states_t, actions_t, rewards_t, next_states_t, dones_t):
            next_q_online = self.model(next_states_t, training=False)
            next_actions = tf.argmax(next_q_online, axis=1, output_type=tf.int32)
            next_q_target = self.target_model(next_states_t, training=False)
            gather_idx = tf.stack([tf.range(tf.shape(next_actions)[0], dtype=tf.int32), next_actions], axis=1)
            next_values = tf.gather_nd(next_q_target, gather_idx)
            targets = rewards_t + self.gamma * (1.0 - dones_t) * next_values

            with tf.GradientTape() as tape:
                q_values = self.model(states_t, training=True)
                action_mask = tf.one_hot(actions_t, self.num_actions, dtype=q_values.dtype)
                q_taken = tf.reduce_sum(q_values * action_mask, axis=1)
                loss = self.loss_fn(targets, q_taken)

            grads = tape.gradient(loss, self.model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

        self._compiled_train_step = compiled_train_step

    def action_index(self, action_name: str) -> int:
        return int(TRAINABLE_ACTIONS.index(action_name))

    def action_name(self, action_index: int) -> str:
        return TRAINABLE_ACTIONS[int(action_index)]

    def epsilon(self, global_step: int) -> float:
        progress = min(1.0, max(0.0, float(global_step) / max(1, self.epsilon_decay_steps)))
        return float(self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress)

    def q_values(self, state_vector, mc_samples: int = 1) -> np.ndarray:
        state = np.asarray(state_vector, dtype=np.float32).reshape(1, -1)
        self._ensure_models(state.shape[1])
        if mc_samples <= 1:
            q = self.model(state, training=False).numpy()[0]
            return np.asarray(q, dtype=np.float32)
        # MC Dropout: forward multiple times with training=True to get distribution
        samples = []
        for _ in range(mc_samples):
            q_sample = self.model(state, training=True).numpy()[0]
            samples.append(q_sample)
        stacked = np.stack(samples, axis=0)  # (mc_samples, num_actions)
        mean_q = np.mean(stacked, axis=0)
        return np.asarray(mean_q, dtype=np.float32)

    def q_distribution(self, state_vector, mc_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean_Q, std_Q) via MC Dropout."""
        mc_samples = mc_samples or self.mc_dropout_samples
        state = np.asarray(state_vector, dtype=np.float32).reshape(1, -1)
        self._ensure_models(state.shape[1])
        samples = []
        for _ in range(mc_samples):
            q_sample = self.model(state, training=True).numpy()[0]
            samples.append(q_sample)
        stacked = np.stack(samples, axis=0)
        mean_q = np.mean(stacked, axis=0)
        std_q = np.std(stacked, axis=0)
        return np.asarray(mean_q, dtype=np.float32), np.asarray(std_q, dtype=np.float32)

    @staticmethod
    def compute_uncertainty_beta(comm_metrics: Optional[Dict[str, float]]) -> float:
        """Compute risk-aversion coefficient beta from communication quality.
        Lower comm quality → higher beta → more conservative action selection.
        """
        if comm_metrics is None:
            return 0.0
        topology_scale = float(comm_metrics.get("topology_scale", 1.0))
        context_scale = float(comm_metrics.get("context_scale", 1.0))
        v2v_success = float(comm_metrics.get("v2v_success", 1.0))
        # Communication uncertainty = 1 - quality; scale to [0, 1]
        comm_uncertainty = 1.0 - 0.35 * topology_scale - 0.35 * context_scale - 0.30 * v2v_success
        comm_uncertainty = float(np.clip(comm_uncertainty, 0.0, 1.0))
        # Base beta for non-comm factors ( inherent model uncertainty )
        beta = 0.15 + 0.85 * comm_uncertainty
        return float(np.clip(beta, 0.0, 1.5))

    # ---------- Policy interface ----------

    def select_action(
        self,
        step: int,
        scheduler,
        state_fields: Dict[str, float],
        state_vector: Optional[np.ndarray] = None,
        training: bool = False,
        comm_metrics: Optional[Dict[str, float]] = None,
    ) -> PolicyDecision:
        if state_vector is None:
            raise ValueError("TrainableHighLevelPolicy requires state_vector for action selection.")
        state_vector = np.asarray(state_vector, dtype=np.float32)
        allowed_indices = self._allowed_action_indices(step, scheduler, state_fields)
        eps = self.epsilon(step) if training else 0.0
        masked = len(allowed_indices) < self.num_actions

        if training and self._rng.random() < eps:
            # Epsilon-greedy exploration: use deterministic Q (no MC dropout overhead)
            q = self.q_values(state_vector, mc_samples=1)
            action_idx = int(self._rng.choice(allowed_indices))
            score = float(q[action_idx])
            reason = f"epsilon={eps:.3f}"
            if masked:
                reason += ", emergency masked"
        else:
            # Uncertainty-aware action selection via MC Dropout
            beta = self.compute_uncertainty_beta(comm_metrics)
            if beta > 0.05 and self.mc_dropout_samples > 1:
                mean_q, std_q = self.q_distribution(state_vector, mc_samples=self.mc_dropout_samples)
                # Risk-sensitive decision: penalize high-variance actions
                ucb_q = mean_q - beta * std_q
                q_for_selection = ucb_q
                q_for_score = mean_q
                reason = f"uncertainty-aware (beta={beta:.2f})"
            else:
                # Deterministic Q (no communication uncertainty or disabled)
                q_for_selection = self.q_values(state_vector, mc_samples=1)
                q_for_score = q_for_selection
                reason = "greedy"

            masked_q = np.full_like(q_for_selection, -np.inf, dtype=np.float32)
            masked_q[allowed_indices] = q_for_selection[allowed_indices]

            if training:
                total_actions = max(1, sum(self.last_action_counts.values()))
                ucb_bonus = np.zeros_like(masked_q, dtype=np.float32)
                for idx in allowed_indices:
                    count = max(1, self.last_action_counts.get(self.action_name(int(idx)), 1))
                    ucb_bonus[int(idx)] = 0.15 * float(np.sqrt(np.log(total_actions + 1) / count))
                masked_q[allowed_indices] = masked_q[allowed_indices] + ucb_bonus[allowed_indices]

            action_idx = int(np.argmax(masked_q))
            score = float(q_for_score[action_idx])
            if masked:
                reason += ", emergency masked"

        # Get raw Q for governor (deterministic)
        raw_q = self.q_values(state_vector, mc_samples=1)
        action_idx, governor_reason = self._apply_soft_governor(
            action_idx=action_idx,
            q_values=raw_q,
            allowed_indices=allowed_indices,
            step=step,
            scheduler=scheduler,
            state_fields=state_fields,
        )
        if governor_reason:
            reason = f"{reason}, {governor_reason}"
        decision = PolicyDecision(self.action_name(action_idx), score=score, reason=reason)

        self._record_action(step, action_idx)
        return decision

    # ---------- Training helpers ----------

    def _allowed_action_indices(self, step: int, scheduler, state_fields: Dict[str, float]) -> np.ndarray:
        keep_idx = self.action_index("keep")
        compact_idx = self.action_index("compact")
        expand_idx = self.action_index("expand")
        split_idx = self.action_index("split")
        merge_idx = self.action_index("merge")
        emergency_idx = self.action_index("emergency")

        blocked_ratio = max(
            float(state_fields.get("blocked_lane_ratio_up", 0.0)),
            float(state_fields.get("blocked_lane_ratio_down", 0.0)),
        )
        event_pressure = max(
            float(state_fields.get("event_pressure_up", 0.0)),
            float(state_fields.get("event_pressure_down", 0.0)),
        )
        min_gap = min(
            float(state_fields.get("min_gap_up", 0.0)),
            float(state_fields.get("min_gap_down", 0.0)),
        )
        mpr = float(state_fields.get("mpr_cav", 0.0))
        platoon_rate = float(state_fields.get("platoon_rate", 0.0))
        comm_fail = float(state_fields.get("comm_fail_percent", 0.0))
        lane_imbalance = max(
            float(state_fields.get("lane_occupancy_std_up", 0.0)),
            float(state_fields.get("lane_occupancy_std_down", 0.0)),
        )
        split_pressure = self._split_pressure(step, scheduler, state_fields)

        active_event = bool(state_fields.get("active_event", 0.0) >= 0.5)
        if scheduler is not None:
            try:
                active_event = active_event or bool(scheduler.has_active_event(step))
            except Exception:
                pass

        calm_window = blocked_ratio <= 0.0 and event_pressure < 0.22 and min_gap >= 12.0
        low_mpr = mpr <= 0.30
        mid_mpr = 0.30 < mpr < 0.50
        mid_band_focus = 0.34 <= mpr <= 0.42
        high_mpr = mpr >= 0.50
        target_mpr_band = 0.48 <= mpr <= 0.55
        very_high_mpr = mpr >= 0.58
        split_ready = (
            (blocked_ratio > 0.0 or event_pressure >= 0.32 or lane_imbalance >= 1.05)
            and split_pressure >= 0.55
            and min_gap >= 8.0
        )
        merge_ready = (
            blocked_ratio <= 0.0
            and event_pressure < 0.40
            and min_gap >= 10.5
            and comm_fail <= 0.10
            and (platoon_rate >= 0.10 or high_mpr)
        )

        if low_mpr:
            allowed = {keep_idx, expand_idx}
            if min_gap < 9.5 or blocked_ratio > 0.0:
                allowed.add(compact_idx)
        elif mid_band_focus:
            allowed = {compact_idx, keep_idx}
            if calm_window and min_gap >= 16.0:
                allowed.add(expand_idx)
        elif mid_mpr:
            allowed = {keep_idx, compact_idx, expand_idx}
        elif target_mpr_band:
            allowed = {keep_idx}
            if blocked_ratio > 0.0 or event_pressure >= 0.30 or min_gap < 10.0:
                allowed.add(compact_idx)
            if blocked_ratio > 0.0 and split_pressure >= 0.85 and min_gap >= 8.5:
                allowed.add(split_idx)
        elif very_high_mpr:
            allowed = {keep_idx, compact_idx}
        else:
            allowed = {keep_idx, compact_idx}

        if split_ready:
            allowed.add(split_idx)
        if merge_ready:
            allowed.add(merge_idx)
        if calm_window:
            allowed.discard(split_idx)
        if low_mpr and blocked_ratio <= 0.0 and event_pressure < 0.30:
            allowed.discard(split_idx)
            if platoon_rate < 0.08:
                allowed.discard(merge_idx)
        if mid_band_focus and blocked_ratio <= 0.0 and event_pressure < 0.30:
            allowed.discard(merge_idx)
        if high_mpr and min_gap < 8.0:
            allowed.discard(compact_idx)
            allowed.discard(merge_idx)
        if high_mpr and blocked_ratio <= 0.0 and event_pressure < 0.25 and platoon_rate < 0.08:
            allowed.discard(split_idx)
        if target_mpr_band and blocked_ratio <= 0.0:
            allowed.discard(split_idx)
            allowed.discard(merge_idx)
            allowed.discard(expand_idx)
        if very_high_mpr:
            allowed.discard(split_idx)
            allowed.discard(merge_idx)
            allowed.discard(expand_idx)
        emergency_idx = self.action_index("emergency")
        if self._emergency_allowed(step, scheduler, state_fields):
            allowed.add(emergency_idx)
        if not allowed:
            allowed = {keep_idx}
        return np.asarray(sorted(allowed), dtype=np.int32)

    def reset(self) -> None:
        self._last_action_idx = None
        self._governor_last_action_idx = None
        self._governor_action_streak = 0
        self._governor_split_cooldown = 0
        self._governor_merge_cooldown = 0
        self._governor_emergency_cooldown = 0

    def _split_pressure(self, step: int, scheduler, state_fields: Dict[str, float]) -> float:
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
        mpr = float(state_fields.get("mpr_cav", 0.0))
        active_event = bool(state_fields.get("active_event", 0.0) >= 0.5)
        if scheduler is not None:
            try:
                active_event = active_event or bool(scheduler.has_active_event(step))
            except Exception:
                pass

        pressure = 0.0
        if active_event:
            pressure += 0.30
        pressure += 0.65 * float(np.clip(blocked_ratio, 0.0, 1.0))
        pressure += 0.55 * float(np.clip((event_pressure - 0.20) / 0.80, 0.0, 1.0))
        pressure += 0.35 * float(np.clip((lane_imbalance - 0.75) / 1.75, 0.0, 1.0))
        if 0.25 <= mpr <= 0.55:
            pressure += 0.20
        return float(np.clip(pressure, 0.0, 2.0))

    def _reconfig_pressure(self, step: int, scheduler, state_fields: Dict[str, float]) -> float:
        split_pressure = self._split_pressure(step, scheduler, state_fields)
        min_gap_up = float(state_fields.get("min_gap_up", 0.0))
        min_gap_down = float(state_fields.get("min_gap_down", 0.0))
        min_gap = min(min_gap_up, min_gap_down)
        comm_fail = float(state_fields.get("comm_fail_percent", 0.0))
        pressure = split_pressure
        if min_gap > 1e-6 and min_gap < 18.0:
            pressure += float(np.clip((18.0 - min_gap) / 12.0, 0.0, 1.0))
        pressure += 0.50 * float(np.clip(comm_fail / 0.25, 0.0, 1.0))
        return float(np.clip(pressure, 0.0, 3.0))

    def _candidate_ranking(self, q_values: np.ndarray, allowed_indices: np.ndarray) -> List[int]:
        if allowed_indices.size <= 0:
            return [self.action_index("keep")]
        ranked = sorted((int(idx) for idx in allowed_indices.tolist()), key=lambda idx: float(q_values[idx]), reverse=True)
        return ranked

    def _fallback_action(
        self,
        q_values: np.ndarray,
        allowed_indices: np.ndarray,
        blocked: set[int],
        *,
        state_fields: Optional[Dict[str, float]] = None,
    ) -> int:
        ranked = self._candidate_ranking(q_values, allowed_indices)
        preferred: List[int] = []
        if state_fields is not None:
            blocked_ratio = max(
                float(state_fields.get("blocked_lane_ratio_up", 0.0)),
                float(state_fields.get("blocked_lane_ratio_down", 0.0)),
            )
            event_pressure = max(
                float(state_fields.get("event_pressure_up", 0.0)),
                float(state_fields.get("event_pressure_down", 0.0)),
            )
            min_gap = min(
                float(state_fields.get("min_gap_up", 0.0)),
                float(state_fields.get("min_gap_down", 0.0)),
            )
            mpr = float(state_fields.get("mpr_cav", 0.0))
            calm_window = blocked_ratio <= 0.0 and event_pressure < 0.25 and min_gap >= 10.0
            if mpr <= 0.30:
                preferred = [
                    self.action_index("keep"),
                    self.action_index("expand"),
                    self.action_index("compact"),
                ]
            elif 0.34 <= mpr <= 0.42:
                preferred = [
                    self.action_index("compact"),
                    self.action_index("keep"),
                    self.action_index("expand"),
                ]
            elif 0.48 <= mpr <= 0.55:
                preferred = [
                    self.action_index("keep"),
                    self.action_index("compact"),
                    self.action_index("split"),
                ]
            elif mpr >= 0.58:
                preferred = [
                    self.action_index("keep"),
                    self.action_index("compact"),
                ]
            elif calm_window:
                preferred = [
                    self.action_index("keep"),
                    self.action_index("expand"),
                    self.action_index("compact"),
                ]
            else:
                preferred = [
                    self.action_index("compact"),
                    self.action_index("keep"),
                    self.action_index("expand"),
                ]
        for idx in preferred:
            if idx in ranked and idx not in blocked:
                return int(idx)
        for idx in ranked:
            if idx not in blocked:
                return int(idx)
        keep_idx = self.action_index("keep")
        if keep_idx in ranked:
            return keep_idx
        return int(ranked[0]) if ranked else keep_idx

    def _apply_soft_governor(
        self,
        *,
        action_idx: int,
        q_values: np.ndarray,
        allowed_indices: np.ndarray,
        step: int,
        scheduler,
        state_fields: Dict[str, float],
    ) -> Tuple[int, str]:
        split_idx = self.action_index("split")
        compact_idx = self.action_index("compact")
        merge_idx = self.action_index("merge")
        expand_idx = self.action_index("expand")
        emergency_idx = self.action_index("emergency")
        reconfig_actions = {split_idx, compact_idx, merge_idx}
        action_name = self.action_name(action_idx)
        split_pressure = self._split_pressure(step, scheduler, state_fields)
        reconfig_pressure = self._reconfig_pressure(step, scheduler, state_fields)
        high_split_pressure = split_pressure >= 0.90
        high_reconfig_pressure = reconfig_pressure >= 1.15
        severe_emergency_pressure = self._emergency_allowed(step, scheduler, state_fields)
        platoon_rate = float(state_fields.get("platoon_rate", 0.0))
        comm_fail = float(state_fields.get("comm_fail_percent", 0.0))
        low_mpr = float(state_fields.get("mpr_cav", 0.0)) <= 0.35
        high_mpr = float(state_fields.get("mpr_cav", 0.0)) >= 0.48
        blocked_ratio = max(
            float(state_fields.get("blocked_lane_ratio_up", 0.0)),
            float(state_fields.get("blocked_lane_ratio_down", 0.0)),
        )
        event_pressure = max(
            float(state_fields.get("event_pressure_up", 0.0)),
            float(state_fields.get("event_pressure_down", 0.0)),
        )
        min_gap = min(
            float(state_fields.get("min_gap_up", 0.0)),
            float(state_fields.get("min_gap_down", 0.0)),
        )
        merge_recovery_ready = (
            blocked_ratio <= 0.0
            and event_pressure < 0.45
            and min_gap >= 10.0
            and comm_fail <= 0.10
            and (platoon_rate >= 0.12 or high_mpr)
        )
        blocked: set[int] = set()
        reasons: List[str] = []

        if action_idx == expand_idx:
            if self._governor_last_action_idx == expand_idx and self._governor_action_streak >= self.max_expand_streak:
                if min_gap >= 10.0 and event_pressure < 0.30:
                    blocked.add(expand_idx)
                    reasons.append(f"governor expand streak>={self.max_expand_streak} gap safe")
            if min_gap >= 14.0 and event_pressure < 0.20 and blocked_ratio <= 0.0:
                blocked.add(expand_idx)
                reasons.append("governor expand unnecessary (gap>=14, calm)")

        if action_idx == split_idx:
            if self._governor_split_cooldown > 0 and not high_split_pressure:
                blocked.add(split_idx)
                reasons.append(f"governor split cooldown={self._governor_split_cooldown}")
            elif self._governor_last_action_idx == split_idx and self._governor_action_streak >= self.max_split_streak and not high_split_pressure:
                blocked.add(split_idx)
                reasons.append(f"governor split streak>={self.max_split_streak}")

        if action_idx == merge_idx:
            if self._governor_merge_cooldown > 0 and not merge_recovery_ready:
                blocked.add(merge_idx)
                reasons.append(f"governor merge cooldown={self._governor_merge_cooldown}")
            elif self._governor_last_action_idx == merge_idx and self._governor_action_streak >= self.max_merge_streak and not high_reconfig_pressure:
                blocked.add(merge_idx)
                reasons.append(f"governor merge streak>={self.max_merge_streak}")
            elif min_gap < 10.0:
                blocked.add(merge_idx)
                reasons.append("governor merge gap guard")
            elif event_pressure >= 0.55 and blocked_ratio > 0.0:
                blocked.add(merge_idx)
                reasons.append("governor merge blocked-event guard")
            elif low_mpr and platoon_rate < 0.08 and event_pressure < 0.30:
                blocked.add(merge_idx)
                reasons.append("governor low-mpr merge unready")

        if action_idx in reconfig_actions:
            same_reconfig_family = self._governor_last_action_idx in reconfig_actions and action_idx in reconfig_actions
            if same_reconfig_family and self._governor_action_streak >= self.max_reconfig_streak and not high_reconfig_pressure:
                blocked.add(action_idx)
                reasons.append(f"governor reconfig streak>={self.max_reconfig_streak}")
            if low_mpr and blocked_ratio <= 0.0 and event_pressure < 0.35 and action_idx == split_idx:
                blocked.add(action_idx)
                reasons.append("governor low-mpr split reroute")
            if low_mpr and event_pressure < 0.30 and blocked_ratio <= 0.0 and min_gap < 12.0 and action_idx == compact_idx:
                blocked.add(action_idx)
                reasons.append("governor low-mpr compact guard")
            if high_mpr and event_pressure < 0.30 and blocked_ratio <= 0.0 and action_idx == split_idx:
                blocked.add(action_idx)
                reasons.append("governor high-mpr split guard")
            if high_mpr and min_gap < 8.0 and action_idx in {compact_idx, merge_idx}:
                blocked.add(action_idx)
                reasons.append("governor high-mpr safety guard")

        if action_idx == emergency_idx:
            if self._governor_emergency_cooldown > 0 and not severe_emergency_pressure:
                blocked.add(emergency_idx)
                reasons.append(f"governor emergency cooldown={self._governor_emergency_cooldown}")
            elif self._governor_last_action_idx == emergency_idx and self._governor_action_streak >= self.max_emergency_streak and not severe_emergency_pressure:
                blocked.add(emergency_idx)
                reasons.append(f"governor emergency streak>={self.max_emergency_streak}")

        final_idx = action_idx
        if blocked:
            final_idx = self._fallback_action(
                q_values,
                allowed_indices,
                blocked,
                state_fields=state_fields,
            )
        reason = ", ".join(reasons)
        self._update_governor_state(final_idx)
        if final_idx != action_idx:
            reason = f"{reason}, reroute {action_name}->{self.action_name(final_idx)}" if reason else f"reroute {action_name}->{self.action_name(final_idx)}"
        return int(final_idx), reason

    def _update_governor_state(self, action_idx: int) -> None:
        split_idx = self.action_index("split")
        merge_idx = self.action_index("merge")
        emergency_idx = self.action_index("emergency")
        reconfig_actions = {
            self.action_index("compact"),
            split_idx,
            merge_idx,
        }
        if self._governor_last_action_idx == action_idx:
            self._governor_action_streak += 1
        else:
            self._governor_action_streak = 1
        if self._governor_split_cooldown > 0:
            self._governor_split_cooldown -= 1
        if self._governor_merge_cooldown > 0:
            self._governor_merge_cooldown -= 1
        if self._governor_emergency_cooldown > 0:
            self._governor_emergency_cooldown -= 1
        if self._governor_last_action_idx == split_idx and action_idx != split_idx:
            self._governor_split_cooldown = max(self._governor_split_cooldown, self.split_cooldown_steps)
        if self._governor_last_action_idx == merge_idx and action_idx != merge_idx:
            self._governor_merge_cooldown = max(self._governor_merge_cooldown, self.merge_cooldown_steps)
        if self._governor_last_action_idx == emergency_idx and action_idx != emergency_idx:
            self._governor_emergency_cooldown = max(self._governor_emergency_cooldown, self.emergency_cooldown_steps)
        if action_idx == split_idx:
            self._governor_split_cooldown = 0
        if action_idx == merge_idx:
            self._governor_merge_cooldown = 0
        if action_idx == emergency_idx:
            self._governor_emergency_cooldown = 0
        elif action_idx not in reconfig_actions:
            self._governor_split_cooldown = max(0, self._governor_split_cooldown)
            self._governor_merge_cooldown = max(0, self._governor_merge_cooldown)
        self._governor_last_action_idx = int(action_idx)

    def _emergency_allowed(self, step: int, scheduler, state_fields: Dict[str, float]) -> bool:
        min_gap_up = float(state_fields.get("min_gap_up", 0.0))
        min_gap_down = float(state_fields.get("min_gap_down", 0.0))
        min_gap = min(min_gap_up, min_gap_down)
        has_valid_gap = min_gap > 1e-6
        event_pressure = max(
            float(state_fields.get("event_pressure_up", 0.0)),
            float(state_fields.get("event_pressure_down", 0.0)),
        )
        blocked_ratio = max(
            float(state_fields.get("blocked_lane_ratio_up", 0.0)),
            float(state_fields.get("blocked_lane_ratio_down", 0.0)),
        )
        comm_fail = float(state_fields.get("comm_fail_percent", 0.0))
        comm_v2v = float(state_fields.get("comm_v2v_success", 1.0))
        active_event = bool(state_fields.get("active_event", 0.0) >= 0.5)
        if scheduler is not None:
            try:
                active_event = active_event or bool(scheduler.has_active_event(step))
            except Exception:
                pass

        tight_gap = has_valid_gap and min_gap < 9.0
        event_tight_gap = has_valid_gap and min_gap < 11.0 and event_pressure >= 0.35
        blocked_incident = active_event and blocked_ratio > 0.0 and event_pressure >= 0.65
        degraded_links = active_event and event_pressure >= 0.50 and (comm_fail >= 0.12 or comm_v2v <= 0.75)
        return bool(tight_gap or event_tight_gap or blocked_incident or degraded_links)

    def _record_action(self, step: int, action_idx: int) -> None:
        action_name = self.action_name(action_idx)
        self.last_action_counts[action_name] = self.last_action_counts.get(action_name, 0) + 1
        if self._last_action_idx is not None and self._last_action_idx != action_idx:
            self._switch_count += 1
        self._last_action_idx = int(action_idx)
        self.action_history.append((int(step), int(action_idx)))

    @property
    def switch_count(self) -> int:
        return int(self._switch_count)

    def action_distribution(self) -> Dict[str, float]:
        total = max(1, len(self.action_history))
        return {name: float(count) / float(total) for name, count in self.last_action_counts.items()}

    def store_transition(self, state, action_name: str, reward: float, next_state, done: bool) -> None:
        action_idx = self.action_index(action_name)
        self.buffer.add(state, action_idx, reward, next_state, done)

    def can_train(self) -> bool:
        return len(self.buffer) >= max(self.batch_size, self.min_buffer_before_train)

    def train_step(self) -> Optional[float]:
        if not self.can_train():
            return None
        tf = self._ensure_tf()
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        self._ensure_models(states.shape[1])

        states_t = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_t = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards_t = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states_t = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones_t = tf.convert_to_tensor(dones, dtype=tf.float32)
        loss = self._compiled_train_step(states_t, actions_t, rewards_t, next_states_t, dones_t)

        self.training_step += 1
        loss_value = float(loss.numpy())
        self.loss_history.append((self.training_step, loss_value))
        if self.training_step % self.target_update_interval == 0:
            self.target_model.set_weights(self.model.get_weights())
        return loss_value

    # ---------- Persistence ----------

    def save(self, out_dir: str | Path) -> Dict[str, str]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        if self.model is None:
            raise RuntimeError("Cannot save an uninitialized trainable policy.")
        self.model.save_weights(str(out_path / "high_level_dqn.weights.h5"))
        meta = {
            "state_dim": int(self.state_dim or 0),
            "hidden_dims": list(self.hidden_dims),
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "max_split_streak": int(self.max_split_streak),
            "max_reconfig_streak": int(self.max_reconfig_streak),
            "split_cooldown_steps": int(self.split_cooldown_steps),
            "merge_cooldown_steps": int(self.merge_cooldown_steps),
            "emergency_cooldown_steps": int(self.emergency_cooldown_steps),
            "max_merge_streak": int(self.max_merge_streak),
            "max_emergency_streak": int(self.max_emergency_streak),
            "num_actions": self.num_actions,
            "action_names": list(TRAINABLE_ACTIONS),
            "training_step": int(self.training_step),
            "episode_count": int(self.episode_count),
            "switch_count": int(self.switch_count),
            "action_distribution": self.action_distribution(),
            "mc_dropout_samples": self.mc_dropout_samples,
            "uncertainty_beta_base": self.uncertainty_beta_base,
            "uncertainty_beta_max": self.uncertainty_beta_max,
        }
        with (out_path / "high_level_policy_meta.json").open("w", encoding="utf-8") as file_obj:
            json.dump(meta, file_obj, indent=2)
        if self.loss_history:
            with (out_path / "high_level_loss.csv").open("w", encoding="utf-8") as file_obj:
                file_obj.write("step,loss\n")
                for step, loss in self.loss_history:
                    file_obj.write(f"{step},{loss}\n")
        if self.action_history:
            with (out_path / "high_level_action_history.csv").open("w", encoding="utf-8") as file_obj:
                file_obj.write("step,action_idx,action_name\n")
                for step, action_idx in self.action_history:
                    file_obj.write(f"{step},{action_idx},{self.action_name(action_idx)}\n")
        return {
            "weights": str(out_path / "high_level_dqn.weights.h5"),
            "meta": str(out_path / "high_level_policy_meta.json"),
        }

    def load(self, weights_path: str | Path, meta_path: str | Path | None = None) -> None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"High-level policy weights not found: {weights_path}")
        if meta_path is not None:
            meta_path = Path(meta_path)
            if meta_path.exists():
                with meta_path.open(encoding="utf-8") as file_obj:
                    meta = json.load(file_obj)
                state_dim = int(meta.get("state_dim", 0))
                if state_dim > 0:
                    self.hidden_dims = tuple(int(v) for v in meta.get("hidden_dims", list(self.hidden_dims)))
                    self.max_split_streak = int(meta.get("max_split_streak", self.max_split_streak))
                    self.max_reconfig_streak = int(meta.get("max_reconfig_streak", self.max_reconfig_streak))
                    self.split_cooldown_steps = int(meta.get("split_cooldown_steps", self.split_cooldown_steps))
                    self.merge_cooldown_steps = int(meta.get("merge_cooldown_steps", self.merge_cooldown_steps))
                    self.emergency_cooldown_steps = int(meta.get("emergency_cooldown_steps", self.emergency_cooldown_steps))
                    self.max_merge_streak = int(meta.get("max_merge_streak", self.max_merge_streak))
                    self.max_emergency_streak = int(meta.get("max_emergency_streak", self.max_emergency_streak))
                    self.mc_dropout_samples = int(meta.get("mc_dropout_samples", self.mc_dropout_samples))
                    self.uncertainty_beta_base = float(meta.get("uncertainty_beta_base", self.uncertainty_beta_base))
                    self.uncertainty_beta_max = float(meta.get("uncertainty_beta_max", self.uncertainty_beta_max))
                    self._ensure_models(state_dim)
        if self.model is None:
            raise RuntimeError("State dimension is unknown. Load metadata first or initialize with state_dim.")
        self.model.load_weights(str(weights_path))
        self.target_model.set_weights(self.model.get_weights())
