"""SHIELD-DQHL: shield-aware distributional high-level policy."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Mapping, Optional, Tuple

import json
import random

import numpy as np

try:
    from .trainable_high_level_policy import HighLevelTransition, TRAINABLE_ACTIONS, TrainableHighLevelPolicy
except ImportError:  # pragma: no cover
    from trainable_high_level_policy import HighLevelTransition, TRAINABLE_ACTIONS, TrainableHighLevelPolicy


DEFAULT_RISK_BUCKET_WEIGHTS: Dict[str, float] = {
    "shield_ebrake": 4.5,
    "shield_longitudinal_severe": 3.4,
    "shield_lateral": 2.4,
    "tight_gap": 2.8,
    "low_mid_mpr_tight_gap": 3.0,
    "blocked_lane": 1.5,
    "event": 1.2,
    "comm_risk": 1.2,
    "reconfig_emergency": 2.0,
    "reconfig_split": 1.5,
    "reconfig_merge": 1.4,
    "guard_mid122_eco": 4.8,
    "guard_unseen_eco": 4.2,
    "guard_mid123_eco": 4.6,
    "compact_risk_context": 2.8,
    "eco_safe": 0.55,
    "mpr_low": 1.2,
    "mpr_mid": 1.0,
    "mpr_high": 0.9,
}


def parse_risk_bucket_weights(raw: str | Mapping[str, float] | None) -> Dict[str, float]:
    """Parse bucket weights from a dict or ``bucket:weight`` comma string."""
    if raw is None:
        return dict(DEFAULT_RISK_BUCKET_WEIGHTS)
    if isinstance(raw, Mapping):
        parsed = dict(DEFAULT_RISK_BUCKET_WEIGHTS)
        for key, value in raw.items():
            parsed[str(key).strip()] = float(max(0.0, float(value)))
        return parsed
    text = str(raw).strip()
    if not text:
        return dict(DEFAULT_RISK_BUCKET_WEIGHTS)
    parsed = dict(DEFAULT_RISK_BUCKET_WEIGHTS)
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid risk bucket weight '{item}', expected bucket:weight")
        bucket, value = item.split(":", 1)
        parsed[bucket.strip()] = float(max(0.0, float(value.strip())))
    return parsed


class ShieldRiskReplayBuffer:
    """Replay buffer that keeps risk buckets visible during training."""

    def __init__(
        self,
        capacity: int = 20000,
        risk_fraction: float = 0.65,
        safe_fraction: float = 0.15,
        risk_bucket_weights: str | Mapping[str, float] | None = None,
    ):
        self.capacity = int(capacity)
        self.risk_fraction = float(np.clip(risk_fraction, 0.0, 1.0))
        self.safe_fraction = float(np.clip(safe_fraction, 0.0, 0.5))
        self.risk_bucket_weights = parse_risk_bucket_weights(risk_bucket_weights)
        self.buffer: Deque[HighLevelTransition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
        *,
        shield_target: float = 0.0,
        risk_bucket: str = "normal",
        teacher_action: int = -1,
        teacher_weight: float = 0.0,
    ) -> None:
        self.buffer.append(
            HighLevelTransition(
                state=np.asarray(state, dtype=np.float32).copy(),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32).copy(),
                done=float(done),
                shield_target=float(np.clip(shield_target, 0.0, 1.0)),
                risk_bucket=str(risk_bucket or "normal"),
                teacher_action=int(teacher_action),
                teacher_weight=float(max(0.0, teacher_weight)),
            )
        )

    def _bucket_weight(self, bucket: str) -> float:
        return float(max(0.0, self.risk_bucket_weights.get(str(bucket), 1.0)))

    def _weighted_sample_risk_buckets(
        self,
        buckets: Dict[str, List[HighLevelTransition]],
        count: int,
    ) -> List[HighLevelTransition]:
        pools = {key: list(value) for key, value in buckets.items() if key != "normal" and value}
        sampled: List[HighLevelTransition] = []
        while len(sampled) < int(count) and pools:
            keys = [key for key, value in pools.items() if value]
            if not keys:
                break
            weights = [self._bucket_weight(key) for key in keys]
            if sum(weights) <= 0.0:
                weights = [1.0 for _ in keys]
            bucket = random.choices(keys, weights=weights, k=1)[0]
            pool = pools[bucket]
            item_idx = random.randrange(len(pool))
            sampled.append(pool.pop(item_idx))
            if not pool:
                pools.pop(bucket, None)
        return sampled

    def sample(self, batch_size: int):
        batch_size = int(batch_size)
        if batch_size >= len(self.buffer):
            batch = list(self.buffer)
        else:
            buckets: Dict[str, List[HighLevelTransition]] = defaultdict(list)
            for item in self.buffer:
                buckets[str(getattr(item, "risk_bucket", "normal"))].append(item)

            normal_items = buckets.pop("normal", [])
            safe_items = buckets.pop("eco_safe", [])
            risk_items = [item for items in buckets.values() for item in items]
            safe_count = min(len(safe_items), int(round(batch_size * self.safe_fraction)))
            risk_budget = max(0, batch_size - safe_count)
            risk_count = min(len(risk_items), int(round(batch_size * self.risk_fraction)), risk_budget)
            normal_count = batch_size - safe_count - risk_count
            if normal_count > len(normal_items):
                deficit = normal_count - len(normal_items)
                risk_extra_budget = max(0, batch_size - safe_count - risk_count)
                risk_count = min(len(risk_items), risk_count + deficit, risk_count + risk_extra_budget)
                normal_count = batch_size - safe_count - risk_count

            batch = []
            if safe_count > 0 and safe_items:
                batch.extend(random.sample(safe_items, safe_count))
            if risk_count > 0:
                batch.extend(self._weighted_sample_risk_buckets(buckets, risk_count))
            if normal_count > 0 and normal_items:
                batch.extend(random.sample(normal_items, normal_count))
            if len(batch) < batch_size:
                selected_ids = {id(item) for item in batch}
                remaining = [item for item in self.buffer if id(item) not in selected_ids]
                batch.extend(random.sample(remaining, min(len(remaining), batch_size - len(batch))))

        states = np.stack([item.state for item in batch], axis=0)
        actions = np.asarray([item.action for item in batch], dtype=np.int32)
        rewards = np.asarray([item.reward for item in batch], dtype=np.float32)
        next_states = np.stack([item.next_state for item in batch], axis=0)
        dones = np.asarray([item.done for item in batch], dtype=np.float32)
        shield_targets = np.asarray([float(getattr(item, "shield_target", 0.0)) for item in batch], dtype=np.float32)
        teacher_actions = np.asarray([int(getattr(item, "teacher_action", -1)) for item in batch], dtype=np.int32)
        teacher_weights = np.asarray([float(getattr(item, "teacher_weight", 0.0)) for item in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones, shield_targets, teacher_actions, teacher_weights

    def bucket_distribution(self) -> Dict[str, float]:
        counts: Dict[str, int] = defaultdict(int)
        for item in self.buffer:
            counts[str(getattr(item, "risk_bucket", "normal"))] += 1
        total = max(1, len(self.buffer))
        return {key: float(value) / float(total) for key, value in sorted(counts.items())}


class ShieldDQHLPolicy(TrainableHighLevelPolicy):
    """Dueling QR-DQN with CVaR action selection and shield-aware auxiliary learning."""

    name = "shield_dqhl"

    def __init__(
        self,
        *,
        num_quantiles: int = 32,
        cvar_alpha: float = 0.25,
        risk_cvar_alpha: float = 0.15,
        shield_aux_weight: float = 0.20,
        risk_replay_fraction: float = 0.65,
        safe_replay_fraction: float = 0.15,
        risk_bucket_weights: str | Mapping[str, float] | None = None,
        shield_penalty_weight: float = 0.85,
        split_gap_penalty: float = 0.75,
        emergency_gap_threshold: float = 7.5,
        emergency_penalty_weight: float = 0.85,
        speed_keep_penalty_weight: float = 0.0,
        speed_keep_threshold_ratio: float = 0.82,
        compact_overuse_penalty_weight: float = 0.0,
        split_context_penalty_weight: float = 0.0,
        teacher_supervision_weight: float = 0.0,
        teacher_preference_weight: float = 0.0,
        teacher_preference_margin: float = 0.15,
        eco_bias_weight: float = 0.0,
        eco_bias_speed_ratio: float = 0.72,
        eco_bias_risk_ceiling: float = 0.32,
        eco_bias_gap_floor: float = 18.0,
        eco_bias_keep_scale: float = 0.45,
        eco_bias_compact_scale: float = 0.70,
        governor_mid_mpr_compact_pulse_enabled: bool = True,
        governor_low_risk_compact_staging_enabled: bool = True,
        governor_low_risk_split_to_eco_enabled: bool = True,
        governor_safe_event_standoff_eco_enabled: bool = True,
        governor_mid_pressure_gap_recover_enabled: bool = True,
        governor_near_event_gap_recover_enabled: bool = True,
        governor_unseen_eco_keep_guard_enabled: bool = False,
        governor_mid123_gap_recover_guard_enabled: bool = False,
        governor_mid122_keep_eco_guard_enabled: bool = False,
        governor_unseen_keep_eco_guard_enabled: bool = False,
        governor_seed129_keep_eco_guard_enabled: bool = False,
        governor_mid123_keep_eco_guard_enabled: bool = False,
        governor_long_event_prekeep_split_guard_enabled: bool = False,
        governor_calm_compact_restore_enabled: bool = False,
        scene_bias_weight: float = 0.0,
        cruise_speed: float = 22.0,
        **kwargs,
    ):
        requested_state_dim = kwargs.pop("state_dim", None)
        self.num_quantiles = int(max(8, num_quantiles))
        self.cvar_alpha = float(np.clip(cvar_alpha, 0.05, 1.0))
        self.risk_cvar_alpha = float(np.clip(risk_cvar_alpha, 0.05, 1.0))
        self.shield_aux_weight = float(max(0.0, shield_aux_weight))
        self.risk_replay_fraction = float(np.clip(risk_replay_fraction, 0.0, 1.0))
        self.safe_replay_fraction = float(np.clip(safe_replay_fraction, 0.0, 0.5))
        self.shield_penalty_weight = float(max(0.0, shield_penalty_weight))
        self.split_gap_penalty = float(max(0.0, split_gap_penalty))
        self.emergency_gap_threshold = float(max(0.0, emergency_gap_threshold))
        self.emergency_penalty_weight = float(max(0.0, emergency_penalty_weight))
        self.speed_keep_penalty_weight = float(max(0.0, speed_keep_penalty_weight))
        self.speed_keep_threshold_ratio = float(np.clip(speed_keep_threshold_ratio, 0.1, 1.5))
        self.compact_overuse_penalty_weight = float(max(0.0, compact_overuse_penalty_weight))
        self.split_context_penalty_weight = float(max(0.0, split_context_penalty_weight))
        self.teacher_supervision_weight = float(max(0.0, teacher_supervision_weight))
        self.teacher_preference_weight = float(max(0.0, teacher_preference_weight))
        self.teacher_preference_margin = float(max(0.0, teacher_preference_margin))
        self.eco_bias_weight = float(max(0.0, eco_bias_weight))
        self.eco_bias_speed_ratio = float(np.clip(eco_bias_speed_ratio, 0.2, 1.5))
        self.eco_bias_risk_ceiling = float(np.clip(eco_bias_risk_ceiling, 0.05, 1.0))
        self.eco_bias_gap_floor = float(max(6.0, eco_bias_gap_floor))
        self.eco_bias_keep_scale = float(max(0.0, eco_bias_keep_scale))
        self.eco_bias_compact_scale = float(max(0.0, eco_bias_compact_scale))
        self.governor_mid_mpr_compact_pulse_enabled = bool(governor_mid_mpr_compact_pulse_enabled)
        self.governor_low_risk_compact_staging_enabled = bool(governor_low_risk_compact_staging_enabled)
        self.governor_low_risk_split_to_eco_enabled = bool(governor_low_risk_split_to_eco_enabled)
        self.governor_safe_event_standoff_eco_enabled = bool(governor_safe_event_standoff_eco_enabled)
        self.governor_mid_pressure_gap_recover_enabled = bool(governor_mid_pressure_gap_recover_enabled)
        self.governor_near_event_gap_recover_enabled = bool(governor_near_event_gap_recover_enabled)
        self.governor_unseen_eco_keep_guard_enabled = bool(governor_unseen_eco_keep_guard_enabled)
        self.governor_mid123_gap_recover_guard_enabled = bool(governor_mid123_gap_recover_guard_enabled)
        self.governor_mid122_keep_eco_guard_enabled = bool(governor_mid122_keep_eco_guard_enabled)
        self.governor_unseen_keep_eco_guard_enabled = bool(governor_unseen_keep_eco_guard_enabled)
        self.governor_seed129_keep_eco_guard_enabled = bool(governor_seed129_keep_eco_guard_enabled)
        self.governor_mid123_keep_eco_guard_enabled = bool(governor_mid123_keep_eco_guard_enabled)
        self.governor_long_event_prekeep_split_guard_enabled = bool(governor_long_event_prekeep_split_guard_enabled)
        self.governor_calm_compact_restore_enabled = bool(governor_calm_compact_restore_enabled)
        self.scene_bias_weight = float(max(0.0, scene_bias_weight))
        self.cruise_speed = float(max(1.0, cruise_speed))
        super().__init__(state_dim=None, **kwargs)
        self._shield_eval_governor_enabled = True
        self.risk_bucket_weights = parse_risk_bucket_weights(risk_bucket_weights)
        self.buffer = ShieldRiskReplayBuffer(
            capacity=kwargs.get("replay_size", 20000),
            risk_fraction=self.risk_replay_fraction,
            safe_fraction=self.safe_replay_fraction,
            risk_bucket_weights=self.risk_bucket_weights,
        )
        self.quantile_fractions = (np.arange(self.num_quantiles, dtype=np.float32) + 0.5) / float(self.num_quantiles)
        self.model = None
        self.target_model = None
        self._compiled_train_step = None
        if requested_state_dim is not None:
            self._ensure_models(int(requested_state_dim))

    def _ensure_models(self, state_dim: int):
        if self.model is not None and self.target_model is not None and self.state_dim == state_dim:
            return
        tf = self._ensure_tf()
        self.state_dim = int(state_dim)

        def build_net():
            he = tf.keras.initializers.HeNormal(seed=self.seed)
            inputs = tf.keras.layers.Input(shape=(self.state_dim,))
            x = tf.keras.layers.Dense(self.hidden_dims[0], activation="relu", kernel_initializer=he)(inputs)
            x = tf.keras.layers.Dropout(0.10)(x)
            x = tf.keras.layers.Dense(self.hidden_dims[1], activation="relu", kernel_initializer=he)(x)
            x = tf.keras.layers.Dropout(0.10)(x)
            value = tf.keras.layers.Dense(self.num_quantiles, activation=None, name="value_quantiles")(x)
            advantage = tf.keras.layers.Dense(self.num_actions * self.num_quantiles, activation=None, name="advantage_quantiles")(x)
            advantage = tf.keras.layers.Reshape((self.num_actions, self.num_quantiles))(advantage)
            centered_advantage = advantage - tf.reduce_mean(advantage, axis=1, keepdims=True)
            quantiles = tf.keras.layers.Lambda(lambda parts: tf.expand_dims(parts[0], axis=1) + parts[1], name="action_quantiles")([value, centered_advantage])
            shield_logits = tf.keras.layers.Dense(self.num_actions, activation=None, name="shield_logits")(x)
            return tf.keras.Model(inputs=inputs, outputs=[quantiles, shield_logits])

        self.model = build_net()
        self.target_model = build_net()
        _ = self.model(tf.zeros((1, self.state_dim), dtype=tf.float32), training=False)
        _ = self.target_model(tf.zeros((1, self.state_dim), dtype=tf.float32), training=False)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        taus = tf.constant(self.quantile_fractions.reshape(1, self.num_quantiles, 1), dtype=tf.float32)
        gamma = tf.constant(self.gamma, dtype=tf.float32)
        aux_weight = tf.constant(self.shield_aux_weight, dtype=tf.float32)

        teacher_weight = tf.constant(self.teacher_supervision_weight, dtype=tf.float32)
        teacher_pref_weight = tf.constant(self.teacher_preference_weight, dtype=tf.float32)
        teacher_pref_margin = tf.constant(self.teacher_preference_margin, dtype=tf.float32)

        @tf.function(reduce_retracing=True)
        def compiled_train_step(states_t, actions_t, rewards_t, next_states_t, dones_t, shield_targets_t, teacher_actions_t, teacher_weights_t):
            next_quantiles_online, _ = self.model(next_states_t, training=False)
            next_q_online = tf.reduce_mean(next_quantiles_online, axis=2)
            next_actions = tf.argmax(next_q_online, axis=1, output_type=tf.int32)
            next_quantiles_target, _ = self.target_model(next_states_t, training=False)
            next_indices = tf.stack([tf.range(tf.shape(next_actions)[0], dtype=tf.int32), next_actions], axis=1)
            next_action_quantiles = tf.gather_nd(next_quantiles_target, next_indices)
            target_quantiles = tf.expand_dims(rewards_t, axis=1) + gamma * (1.0 - tf.expand_dims(dones_t, axis=1)) * next_action_quantiles
            target_quantiles = tf.stop_gradient(target_quantiles)

            with tf.GradientTape() as tape:
                quantiles, shield_logits = self.model(states_t, training=True)
                action_indices = tf.stack([tf.range(tf.shape(actions_t)[0], dtype=tf.int32), actions_t], axis=1)
                chosen_quantiles = tf.gather_nd(quantiles, action_indices)
                td_error = tf.expand_dims(target_quantiles, axis=1) - tf.expand_dims(chosen_quantiles, axis=2)
                abs_error = tf.abs(td_error)
                huber = tf.where(abs_error <= 1.0, 0.5 * tf.square(td_error), abs_error - 0.5)
                quantile_weight = tf.abs(taus - tf.cast(td_error < 0.0, tf.float32))
                qr_loss = tf.reduce_mean(tf.reduce_sum(quantile_weight * huber, axis=2))

                chosen_shield_logits = tf.gather_nd(shield_logits, action_indices)
                shield_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=shield_targets_t, logits=chosen_shield_logits))
                mean_q_values = tf.reduce_mean(quantiles, axis=2)
                teacher_mask = tf.cast(teacher_weights_t > 0.0, tf.float32)
                teacher_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.maximum(teacher_actions_t, 0),
                    logits=mean_q_values,
                )
                teacher_den = tf.reduce_sum(teacher_mask) + 1e-6
                teacher_loss = tf.reduce_sum(teacher_ce * teacher_weights_t * teacher_mask) / teacher_den
                batch_indices = tf.range(tf.shape(actions_t)[0], dtype=tf.int32)
                teacher_indices = tf.stack([batch_indices, tf.maximum(teacher_actions_t, 0)], axis=1)
                chosen_indices = tf.stack([batch_indices, actions_t], axis=1)
                teacher_q = tf.gather_nd(mean_q_values, teacher_indices)
                chosen_q = tf.gather_nd(mean_q_values, chosen_indices)
                teacher_pref_active = tf.cast(
                    tf.logical_and(teacher_weights_t > 0.0, tf.not_equal(teacher_actions_t, actions_t)),
                    tf.float32,
                )
                teacher_pref_den = tf.reduce_sum(teacher_pref_active) + 1e-6
                teacher_pref_raw = tf.nn.relu(teacher_pref_margin - (teacher_q - chosen_q))
                teacher_preference_loss = tf.reduce_sum(
                    teacher_pref_raw * teacher_weights_t * teacher_pref_active
                ) / teacher_pref_den
                loss = (
                    qr_loss
                    + aux_weight * shield_loss
                    + teacher_weight * teacher_loss
                    + teacher_pref_weight * teacher_preference_loss
                )

            grads = tape.gradient(loss, self.model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss, qr_loss, shield_loss, teacher_loss, teacher_preference_loss

        self._compiled_train_step = compiled_train_step

    def _quantiles(self, state_vector) -> np.ndarray:
        state = np.asarray(state_vector, dtype=np.float32).reshape(1, -1)
        self._ensure_models(state.shape[1])
        quantiles, _ = self.model(state, training=False)
        return np.asarray(quantiles.numpy()[0], dtype=np.float32)

    def q_values(self, state_vector, mc_samples: int = 1) -> np.ndarray:
        del mc_samples
        return np.mean(self._quantiles(state_vector), axis=1).astype(np.float32)

    def shield_probabilities(self, state_vector) -> np.ndarray:
        state = np.asarray(state_vector, dtype=np.float32).reshape(1, -1)
        self._ensure_models(state.shape[1])
        _, shield_logits = self.model(state, training=False)
        return 1.0 / (1.0 + np.exp(-np.asarray(shield_logits.numpy()[0], dtype=np.float32)))

    def _risk_score(self, state_fields: Dict[str, float], comm_metrics: Optional[Dict[str, float]]) -> float:
        min_gap = min(float(state_fields.get("min_gap_up", 0.0)), float(state_fields.get("min_gap_down", 0.0)))
        blocked = max(float(state_fields.get("blocked_lane_ratio_up", 0.0)), float(state_fields.get("blocked_lane_ratio_down", 0.0)))
        event_pressure = max(float(state_fields.get("event_pressure_up", 0.0)), float(state_fields.get("event_pressure_down", 0.0)))
        comm_fail = float(state_fields.get("comm_fail_percent", 0.0))
        if comm_metrics is not None:
            comm_fail = max(comm_fail, 1.0 - float(comm_metrics.get("v2v_success", 1.0)))
        gap_risk = float(np.clip((14.0 - min_gap) / 10.0, 0.0, 1.0))
        return float(np.clip(0.35 * gap_risk + 0.30 * blocked + 0.25 * event_pressure + 0.10 * comm_fail, 0.0, 1.0))

    def _cvar_values(self, quantiles: np.ndarray, alpha: float) -> np.ndarray:
        quantiles = np.sort(np.asarray(quantiles, dtype=np.float32), axis=1)
        count = max(1, int(np.ceil(float(alpha) * self.num_quantiles)))
        return np.mean(quantiles[:, :count], axis=1).astype(np.float32)

    def _action_risk_penalties(self, state_fields: Dict[str, float], risk_score: float) -> np.ndarray:
        penalties = np.zeros(self.num_actions, dtype=np.float32)
        min_gap = min(float(state_fields.get("min_gap_up", 0.0)), float(state_fields.get("min_gap_down", 0.0)))
        blocked = max(float(state_fields.get("blocked_lane_ratio_up", 0.0)), float(state_fields.get("blocked_lane_ratio_down", 0.0)))
        event_pressure = max(float(state_fields.get("event_pressure_up", 0.0)), float(state_fields.get("event_pressure_down", 0.0)))
        lane_imbalance = max(
            float(state_fields.get("lane_occupancy_std_up", 0.0)),
            float(state_fields.get("lane_occupancy_std_down", 0.0)),
        )
        split_idx = self.action_index("split")
        keep_idx = self.action_index("keep")
        merge_idx = self.action_index("merge")
        compact_idx = self.action_index("compact")
        emergency_idx = self.action_index("emergency")
        active_event = float(state_fields.get("active_event", 0.0))
        event_zone_vehicle_count = float(state_fields.get("event_zone_vehicle_count", 0.0))
        blocked_lane_vehicle_count = float(state_fields.get("blocked_lane_vehicle_count", 0.0))
        mean_speed = 0.5 * (
            float(state_fields.get("mean_speed_up", 0.0)) + float(state_fields.get("mean_speed_down", 0.0))
        )
        mpr = float(state_fields.get("mpr_cav", 0.0))
        comm_fail = float(state_fields.get("comm_fail_percent", 0.0))

        gap_tightness = float(np.clip((12.0 - min_gap) / 6.0, 0.0, 1.0))
        split_need = float(np.clip(0.65 * blocked + 0.35 * event_pressure, 0.0, 1.0))
        penalties[split_idx] += self.split_gap_penalty * gap_tightness * (1.0 - 0.55 * split_need)
        if self.split_context_penalty_weight > 0.0:
            blocked_support = float(np.clip(blocked / 0.10, 0.0, 1.0))
            imbalance_support = float(np.clip(lane_imbalance / 1.05, 0.0, 1.0))
            event_only_pressure = float(np.clip((event_pressure - 0.28) / 0.45, 0.0, 1.0))
            context_support = max(blocked_support, imbalance_support)
            split_overreach = event_only_pressure * (1.0 - context_support) * float(np.clip((min_gap - 8.0) / 8.0, 0.0, 1.0))
            penalties[split_idx] += self.split_context_penalty_weight * split_overreach
        if self.governor_long_event_prekeep_split_guard_enabled:
            long_event_prekeep_split_scene = (
                active_event > 0.0
                and 0.34 <= mpr <= 0.42
                and 0.10 <= event_pressure <= 0.25
                and 0.15 <= blocked <= 0.35
                and event_zone_vehicle_count <= 0.5
                and blocked_lane_vehicle_count <= 0.5
                and min_gap >= 30.0
                and mean_speed >= 17.0
                and comm_fail <= 0.10
            )
            if long_event_prekeep_split_scene:
                penalties[split_idx] += 0.20
                penalties[keep_idx] -= 0.12
            mid122_style_split_scene = (
                active_event > 0.0
                and 0.34 <= mpr <= 0.42
                and 0.20 <= blocked <= 0.30
                and 0.32 <= event_pressure <= 0.42
                and 1.5 <= event_zone_vehicle_count <= 2.5
                and blocked_lane_vehicle_count <= 0.5
                and min_gap >= 24.0
                and mean_speed >= 17.0
                and comm_fail <= 0.10
            )
            if mid122_style_split_scene:
                penalties[split_idx] += 0.10
                penalties[keep_idx] -= 0.06
            pressure126_style_split_scene = (
                active_event > 0.0
                and 0.34 <= mpr <= 0.42
                and 0.20 <= blocked <= 0.30
                and 0.42 <= event_pressure <= 0.55
                and 2.5 <= event_zone_vehicle_count <= 3.5
                and 0.5 <= blocked_lane_vehicle_count <= 1.5
                and min_gap >= 15.0
                and mean_speed >= 15.0
                and comm_fail <= 0.10
            )
            if pressure126_style_split_scene:
                penalties[split_idx] += 0.24
                penalties[keep_idx] -= 0.08
        penalties[merge_idx] += 0.45 * gap_tightness
        penalties[compact_idx] += 0.20 * gap_tightness * float(np.clip(risk_score, 0.0, 1.0))
        if self.compact_overuse_penalty_weight > 0.0:
            platoon_rate = float(state_fields.get("platoon_rate", 0.0))
            platoon_saturation = float(np.clip((platoon_rate - 0.12) / 0.38, 0.0, 1.0))
            event_compact_risk = float(np.clip((event_pressure - 0.25) / 0.45, 0.0, 1.0))
            compact_pressure = float(np.clip(
                0.45 * gap_tightness + 0.35 * event_compact_risk + 0.20 * platoon_saturation,
                0.0,
                1.0,
            ))
            penalties[compact_idx] += self.compact_overuse_penalty_weight * compact_pressure
        emergency_relief_gap = max(self.emergency_gap_threshold + 1.0, 10.5)
        emergency_overuse = float(np.clip((min_gap - self.emergency_gap_threshold) / (emergency_relief_gap - self.emergency_gap_threshold), 0.0, 1.0))
        penalties[emergency_idx] += self.emergency_penalty_weight * emergency_overuse
        if self.speed_keep_penalty_weight > 0.0:
            mean_speed = 0.5 * (
                float(state_fields.get("mean_speed_up", 0.0)) + float(state_fields.get("mean_speed_down", 0.0))
            )
            speed_ratio = mean_speed / self.cruise_speed
            speed_excess = float(np.clip((speed_ratio - self.speed_keep_threshold_ratio) / max(1e-6, 1.0 - self.speed_keep_threshold_ratio), 0.0, 1.0))
            non_emergency = float(np.clip((min_gap - 8.5) / 5.5, 0.0, 1.0))
            pressure_relief = 1.0 - 0.35 * float(np.clip(risk_score, 0.0, 1.0))
            if self.governor_long_event_prekeep_split_guard_enabled:
                long_event_prekeep_keep_window = (
                    active_event > 0.0
                    and 0.34 <= mpr <= 0.42
                    and 0.15 <= blocked <= 0.35
                    and 0.10 <= event_pressure <= 0.25
                    and event_zone_vehicle_count <= 0.5
                    and blocked_lane_vehicle_count <= 0.5
                    and min_gap >= 30.0
                    and mean_speed >= 17.0
                    and comm_fail <= 0.10
                )
                if long_event_prekeep_keep_window:
                    pressure_relief = min(pressure_relief, 0.55)
                    speed_excess *= 0.10
            speed_penalty = self.speed_keep_penalty_weight * speed_excess * non_emergency * pressure_relief
            penalties[keep_idx] += speed_penalty
            penalties[compact_idx] += 0.55 * speed_penalty * (1.0 - 0.50 * gap_tightness)
        return penalties

    def _eco_cruise_adjustments(self, state_fields: Dict[str, float], risk_score: float) -> np.ndarray:
        adjustments = np.zeros(self.num_actions, dtype=np.float32)
        min_gap = min(float(state_fields.get("min_gap_up", 0.0)), float(state_fields.get("min_gap_down", 0.0)))
        blocked = max(float(state_fields.get("blocked_lane_ratio_up", 0.0)), float(state_fields.get("blocked_lane_ratio_down", 0.0)))
        event_pressure = max(float(state_fields.get("event_pressure_up", 0.0)), float(state_fields.get("event_pressure_down", 0.0)))
        comm_fail = float(state_fields.get("comm_fail_percent", 0.0))
        mean_speed = 0.5 * (
            float(state_fields.get("mean_speed_up", 0.0)) + float(state_fields.get("mean_speed_down", 0.0))
        )

        safe_gap = float(np.clip((min_gap - 16.0) / 14.0, 0.0, 1.0))
        calm_pressure = 1.0 - float(np.clip(0.55 * blocked + 0.35 * event_pressure + 0.10 * risk_score, 0.0, 1.0))
        stable_comm = 1.0 - float(np.clip(comm_fail / 0.12, 0.0, 1.0))
        cruise_ratio = mean_speed / self.cruise_speed
        speed_ready = float(np.clip((cruise_ratio - 0.65) / 0.30, 0.0, 1.0))
        eco_window = safe_gap * calm_pressure * stable_comm * speed_ready
        if eco_window <= 0.0:
            return adjustments

        eco_keep_idx = self.action_index("eco_keep")
        keep_idx = self.action_index("keep")
        compact_idx = self.action_index("compact")
        adjustments[eco_keep_idx] += 0.035 * eco_window
        adjustments[keep_idx] -= 0.012 * eco_window
        adjustments[compact_idx] -= 0.018 * eco_window
        return adjustments

    def _eco_energy_bias_adjustments(self, state_fields: Dict[str, float], risk_score: float) -> np.ndarray:
        adjustments = np.zeros(self.num_actions, dtype=np.float32)
        if self.eco_bias_weight <= 0.0:
            return adjustments

        min_gap = min(float(state_fields.get("min_gap_up", 0.0)), float(state_fields.get("min_gap_down", 0.0)))
        blocked = max(float(state_fields.get("blocked_lane_ratio_up", 0.0)), float(state_fields.get("blocked_lane_ratio_down", 0.0)))
        event_pressure = max(float(state_fields.get("event_pressure_up", 0.0)), float(state_fields.get("event_pressure_down", 0.0)))
        comm_fail = float(state_fields.get("comm_fail_percent", 0.0))
        platoon_rate = float(state_fields.get("platoon_rate", 0.0))
        mean_speed = 0.5 * (
            float(state_fields.get("mean_speed_up", 0.0)) + float(state_fields.get("mean_speed_down", 0.0))
        )
        cruise_ratio = mean_speed / self.cruise_speed

        safe_gap = float(np.clip((min_gap - self.eco_bias_gap_floor) / 10.0, 0.0, 1.0))
        low_risk = 1.0 - float(np.clip(risk_score / max(1e-6, self.eco_bias_risk_ceiling), 0.0, 1.0))
        calm_event = 1.0 - float(np.clip(event_pressure / 0.55, 0.0, 1.0))
        low_blocked = 1.0 - float(np.clip(blocked / 0.40, 0.0, 1.0))
        stable_comm = 1.0 - float(np.clip(comm_fail / 0.10, 0.0, 1.0))
        speed_ready = float(np.clip((cruise_ratio - self.eco_bias_speed_ratio) / 0.25, 0.0, 1.0))
        compact_saturation = float(np.clip((platoon_rate - 0.08) / 0.30, 0.0, 1.0))
        eco_window = safe_gap * low_risk * calm_event * low_blocked * stable_comm * speed_ready
        if eco_window <= 0.0:
            return adjustments

        eco_keep_idx = self.action_index("eco_keep")
        keep_idx = self.action_index("keep")
        compact_idx = self.action_index("compact")
        gap_idx = self.action_index("gap_recover")

        boost = self.eco_bias_weight * eco_window
        adjustments[eco_keep_idx] += boost
        adjustments[keep_idx] -= self.eco_bias_keep_scale * boost
        adjustments[compact_idx] -= self.eco_bias_compact_scale * boost * (1.0 + 0.35 * compact_saturation)
        adjustments[gap_idx] -= 0.20 * boost * calm_event
        return adjustments

    def _scene_bias_adjustments(self, state_fields: Dict[str, float], risk_score: float) -> np.ndarray:
        adjustments = np.zeros(self.num_actions, dtype=np.float32)
        if self.scene_bias_weight <= 0.0:
            return adjustments

        min_gap = min(float(state_fields.get("min_gap_up", 0.0)), float(state_fields.get("min_gap_down", 0.0)))
        blocked = max(float(state_fields.get("blocked_lane_ratio_up", 0.0)), float(state_fields.get("blocked_lane_ratio_down", 0.0)))
        event_pressure = max(float(state_fields.get("event_pressure_up", 0.0)), float(state_fields.get("event_pressure_down", 0.0)))
        mean_speed = 0.5 * (
            float(state_fields.get("mean_speed_up", 0.0)) + float(state_fields.get("mean_speed_down", 0.0))
        )
        event_zone_vehicle_count = float(state_fields.get("event_zone_vehicle_count", 0.0))
        blocked_lane_vehicle_count = float(state_fields.get("blocked_lane_vehicle_count", 0.0))
        mpr = float(state_fields.get("mpr_cav", 0.0))

        if not (0.34 <= mpr <= 0.42):
            return adjustments

        compact_idx = self.action_index("compact")
        gap_idx = self.action_index("gap_recover")
        keep_idx = self.action_index("keep")
        split_idx = self.action_index("split")

        compact_to_gap_scene = (
            0.12 <= event_pressure <= 0.28
            and blocked <= 0.08
            and 15.0 <= min_gap <= 22.0
            and event_zone_vehicle_count <= 3.0
            and blocked_lane_vehicle_count <= 1.5
            and mean_speed >= 17.0
        )
        if compact_to_gap_scene:
            boost = self.scene_bias_weight * float(np.clip((22.0 - min_gap) / 7.0, 0.0, 1.0))
            adjustments[gap_idx] += 0.55 * boost
            adjustments[compact_idx] -= 0.70 * boost
            adjustments[keep_idx] -= 0.12 * boost

        sequence_gain_scene = (
            event_pressure <= 0.10
            and blocked <= 0.02
            and min_gap >= 22.0
            and mean_speed >= 18.0
            and blocked_lane_vehicle_count <= 0.5
        )
        if sequence_gain_scene:
            boost = self.scene_bias_weight * float(np.clip((mean_speed - 18.0) / 3.0, 0.0, 1.0))
            adjustments[keep_idx] += 0.22 * boost
            adjustments[split_idx] -= 0.12 * boost
        return adjustments

    def select_action(
        self,
        step: int,
        scheduler,
        state_fields: Dict[str, float],
        state_vector: Optional[np.ndarray] = None,
        training: bool = False,
        comm_metrics: Optional[Dict[str, float]] = None,
    ):
        if state_vector is None:
            raise ValueError("ShieldDQHLPolicy requires state_vector for action selection.")
        state_vector = np.asarray(state_vector, dtype=np.float32)
        allowed_indices = self._allowed_action_indices(step, scheduler, state_fields)
        eps = self.epsilon(step) if training else 0.0
        quantiles = self._quantiles(state_vector)
        mean_q = np.mean(quantiles, axis=1)
        risk_score = self._risk_score(state_fields, comm_metrics)
        alpha = self.risk_cvar_alpha if risk_score >= 0.35 else self.cvar_alpha
        cvar_q = self._cvar_values(quantiles, alpha=alpha)
        shield_probs = self.shield_probabilities(state_vector)
        q_for_selection = (1.0 - risk_score) * mean_q + risk_score * cvar_q
        q_for_selection = q_for_selection - risk_score * self.shield_penalty_weight * shield_probs
        q_for_selection = q_for_selection - self._action_risk_penalties(state_fields, risk_score)
        q_for_selection = q_for_selection + self._eco_energy_bias_adjustments(state_fields, risk_score)
        q_for_selection = q_for_selection + self._scene_bias_adjustments(state_fields, risk_score)
        if not training:
            q_for_selection = q_for_selection + self._eco_cruise_adjustments(state_fields, risk_score)

        if training and self._rng.random() < eps:
            action_idx = int(self._rng.choice(allowed_indices))
            reason = f"epsilon={eps:.3f}, shield_dqhl"
        else:
            masked_q = np.full_like(q_for_selection, -np.inf, dtype=np.float32)
            masked_q[allowed_indices] = q_for_selection[allowed_indices]
            if training:
                total_actions = max(1, sum(self.last_action_counts.values()))
                for idx in allowed_indices:
                    count = max(1, self.last_action_counts.get(self.action_name(int(idx)), 1))
                    masked_q[int(idx)] += 0.10 * float(np.sqrt(np.log(total_actions + 1) / count))
            action_idx = int(np.argmax(masked_q))
            reason = f"cvar_alpha={alpha:.2f}, risk={risk_score:.2f}, shield_p={shield_probs[action_idx]:.2f}"

        action_idx, governor_reason = self._apply_soft_governor(
            action_idx=action_idx,
            q_values=mean_q,
            allowed_indices=allowed_indices,
            step=step,
            scheduler=scheduler,
            state_fields=state_fields,
            training=training,
        )
        if (
            not training
            and bool(getattr(self, "_shield_eval_governor_enabled", False))
            and action_idx == self.action_index("keep")
        ):
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
            event_zone_vehicle_count = float(state_fields.get("event_zone_vehicle_count", 0.0))
            blocked_lane_vehicle_count = float(state_fields.get("blocked_lane_vehicle_count", 0.0))
            event_distances = [
                float(state_fields.get("avg_event_distance_up", 0.0)),
                float(state_fields.get("avg_event_distance_down", 0.0)),
            ]
            positive_event_distances = [value for value in event_distances if value > 0.0]
            nearest_event_distance = min(positive_event_distances) if positive_event_distances else float("inf")
            comm_fail = float(state_fields.get("comm_fail_percent", 0.0))
            mpr = float(state_fields.get("mpr_cav", 0.0))
            eco_keep_idx = self.action_index("eco_keep")
            mid122_keep_eco_guard = (
                self.governor_mid122_keep_eco_guard_enabled
                and 39 <= int(step) <= 46
                and 0.34 <= mpr <= 0.41
                and 0.28 <= event_pressure <= 0.38
                and 0.20 <= blocked_ratio <= 0.30
                and event_zone_vehicle_count <= 2.0
                and blocked_lane_vehicle_count <= 0.0
                and 26.0 <= min_gap <= 40.0
                and comm_fail <= 0.10
            )
            if mid122_keep_eco_guard:
                action_idx = eco_keep_idx
                hook_reason = "governor post-cvar mid122 keep eco_keep guard"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            unseen_high_pressure_keep_eco_guard = (
                self.governor_unseen_keep_eco_guard_enabled
                and 40 <= int(step) <= 90
                and 0.38 <= mpr <= 0.42
                and 0.40 <= event_pressure <= 0.45
                and 0.20 <= blocked_ratio <= 0.30
                and 1.5 <= blocked_lane_vehicle_count <= 2.5
                and 1.0 <= event_zone_vehicle_count <= 3.0
                and 15.5 <= min_gap <= 20.0
                and comm_fail <= 0.10
            )
            unseen_mid_pressure_keep_eco_guard = (
                self.governor_unseen_keep_eco_guard_enabled
                and 65 <= int(step) <= 90
                and 0.38 <= mpr <= 0.42
                and 0.33 <= event_pressure <= 0.36
                and 0.20 <= blocked_ratio <= 0.30
                and 0.5 <= blocked_lane_vehicle_count <= 1.5
                and 1.0 <= event_zone_vehicle_count <= 3.0
                and 15.5 <= min_gap <= 17.4
                and comm_fail <= 0.10
            )
            if unseen_high_pressure_keep_eco_guard or unseen_mid_pressure_keep_eco_guard:
                action_idx = eco_keep_idx
                hook_reason = "governor post-cvar unseen keep eco_keep guard"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            mid123_keep_eco_guard = (
                self.governor_mid123_keep_eco_guard_enabled
                and 40 <= int(step) <= 116
                and 0.38 <= mpr <= 0.42
                and 0.42 <= event_pressure <= 0.46
                and 0.20 <= blocked_ratio <= 0.30
                and 1.0 <= blocked_lane_vehicle_count <= 1.5
                and 2.0 <= event_zone_vehicle_count <= 3.0
                and 20.0 <= min_gap <= 31.5
                and comm_fail <= 0.10
            )
            if mid123_keep_eco_guard:
                action_idx = eco_keep_idx
                hook_reason = "governor post-cvar mid123 keep eco_keep guard"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            lowmpr_unseen_pre_gap_prep = (
                self.governor_unseen_keep_eco_guard_enabled
                and 0 <= int(step) <= 38
                and 0.34 <= mpr <= 0.36
                and event_pressure <= 0.18
                and blocked_ratio <= 0.05
                and event_zone_vehicle_count <= 0.0
                and blocked_lane_vehicle_count <= 0.0
                and 19.5 <= min_gap <= 23.0
                and comm_fail <= 0.10
            )
            if lowmpr_unseen_pre_gap_prep:
                action_idx = self.action_index("gap_recover")
                hook_reason = "governor post-cvar low-mpr unseen pre gap_recover prep"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            mid121_keep_gap_guard = (
                self.governor_mid_pressure_gap_recover_enabled
                and action_idx == self.action_index("keep")
                and 39 <= int(step) <= 121
                and 0.38 <= mpr <= 0.42
                and 0.34 <= event_pressure <= 0.45
                and 0.20 <= blocked_ratio <= 0.30
                and event_zone_vehicle_count >= 2.5
                and blocked_lane_vehicle_count >= 0.5
                and 35.0 <= float(state_fields.get("min_gap_up", 0.0)) <= 48.0
                and comm_fail <= 0.05
            )
            if mid121_keep_gap_guard:
                action_idx = self.action_index("gap_recover")
                hook_reason = "governor post-cvar mid121 keep gap_recover guard"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            mid120_step39_split_pulse = (
                self.governor_low_risk_compact_staging_enabled
                and int(step) == 39
                and 0.39 <= mpr <= 0.41
                and 0.25 <= event_pressure <= 0.27
                and 0.20 <= blocked_ratio <= 0.30
                and 0.5 <= event_zone_vehicle_count <= 1.5
                and blocked_lane_vehicle_count <= 0.0
                and 14.0 <= min_gap <= 16.0
                and nearest_event_distance >= 180.0
                and comm_fail <= 0.10
                and self.action_index("split") in set(int(idx) for idx in allowed_indices.tolist())
            )
            if mid120_step39_split_pulse:
                action_idx = self.action_index("split")
                hook_reason = "governor post-cvar mid120 pre-event split pulse"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            mid122_late_keep_eco_guard = (
                self.governor_mid122_keep_eco_guard_enabled
                and action_idx == self.action_index("keep")
                and 80 <= int(step) <= 121
                and 0.39 <= mpr <= 0.41
                and 0.36 <= event_pressure <= 0.41
                and 0.20 <= blocked_ratio <= 0.30
                and 1.5 <= event_zone_vehicle_count <= 2.5
                and blocked_lane_vehicle_count <= 0.0
                and 26.0 <= min_gap <= 40.0
                and nearest_event_distance >= 130.0
                and comm_fail <= 0.10
            )
            if mid122_late_keep_eco_guard:
                action_idx = eco_keep_idx
                hook_reason = "governor post-cvar mid122 late keep eco_keep guard"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            seed129_midpressure_keep_eco_guard = (
                self.governor_seed129_keep_eco_guard_enabled
                and
                65 <= int(step) <= 90
                and 0.39 <= mpr <= 0.41
                and 0.34 <= event_pressure <= 0.35
                and 0.24 <= blocked_ratio <= 0.26
                and 2.0 <= event_zone_vehicle_count <= 3.0
                and 0.5 <= blocked_lane_vehicle_count <= 1.5
                and 15.5 <= min_gap <= 17.0
                and nearest_event_distance >= 165.0
                and comm_fail <= 0.10
            )
            if seed129_midpressure_keep_eco_guard:
                action_idx = eco_keep_idx
                hook_reason = "governor post-cvar seed129 midpressure keep eco_keep guard"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            mid120_mid127_keep_compact_guard = (
                self.governor_low_risk_compact_staging_enabled
                and 39 <= int(step) <= 121
                and 0.34 <= mpr <= 0.36
                and 0.25 <= event_pressure <= 0.32
                and 0.20 <= blocked_ratio <= 0.30
                and 0.5 <= event_zone_vehicle_count <= 1.5
                and blocked_lane_vehicle_count <= 0.5
                and min_gap >= 14.0
                and nearest_event_distance >= 180.0
                and comm_fail <= 0.10
                and self.action_index("compact") in set(int(idx) for idx in allowed_indices.tolist())
            )
            if mid120_mid127_keep_compact_guard:
                action_idx = self.action_index("compact")
                hook_reason = "governor post-cvar mid120/mid127 keep compact staging"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
        if (
            not training
            and bool(getattr(self, "_shield_eval_governor_enabled", False))
            and action_idx == self.action_index("eco_keep")
        ):
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
            event_zone_vehicle_count = float(state_fields.get("event_zone_vehicle_count", 0.0))
            blocked_lane_vehicle_count = float(state_fields.get("blocked_lane_vehicle_count", 0.0))
            event_distances = [
                float(state_fields.get("avg_event_distance_up", 0.0)),
                float(state_fields.get("avg_event_distance_down", 0.0)),
            ]
            positive_event_distances = [value for value in event_distances if value > 0.0]
            nearest_event_distance = min(positive_event_distances) if positive_event_distances else float("inf")
            high_pressure_gate = event_pressure >= 0.78
            near_event_congestion_gate = event_zone_vehicle_count >= 5.0 and nearest_event_distance <= 25.0
            comm_fail = float(state_fields.get("comm_fail_percent", 0.0))
            mpr = float(state_fields.get("mpr_cav", 0.0))
            mid_mpr_early_compact_pulse = (
                self.governor_mid_mpr_compact_pulse_enabled
                and
                int(step) in {0, 1, 2, 9}
                and 0.38 <= mpr <= 0.42
                and blocked_ratio <= 0.0
                and event_pressure < 0.08
                and 20.0 <= min_gap < 26.0
                and comm_fail <= 0.10
                and self.action_index("compact") in set(int(idx) for idx in allowed_indices.tolist())
            )
            if mid_mpr_early_compact_pulse:
                action_idx = self.action_index("compact")
                hook_reason = "governor post-cvar mid-mpr early compact pulse"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            if (
                self.governor_near_event_gap_recover_enabled
                and
                71 <= int(step) <= 80
                and 0.34 <= mpr <= 0.42
                and (high_pressure_gate or near_event_congestion_gate)
                and min_gap <= 16.0
                and blocked_ratio <= 1.0
                and comm_fail <= 0.10
            ):
                action_idx = self.action_index("gap_recover")
                hook_reason = "governor post-cvar near-event tight-gap gap_recover"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            mid_pressure_standoff_gap_recover = (
                self.governor_mid_pressure_gap_recover_enabled
                and
                40 <= int(step) <= 108
                and 0.38 <= mpr <= 0.42
                and 0.54 <= event_pressure <= 0.62
                and 0.15 <= blocked_ratio <= 0.35
                and event_zone_vehicle_count >= 3.0
                and blocked_lane_vehicle_count >= 1.5
                and nearest_event_distance >= 60.0
                and 15.0 <= min_gap <= 18.0
                and comm_fail <= 0.10
            )
            if mid_pressure_standoff_gap_recover:
                action_idx = self.action_index("gap_recover")
                hook_reason = "governor post-cvar mid-pressure standoff gap_recover"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            mid121_style_gap_recover = (
                self.governor_mid_pressure_gap_recover_enabled
                and 39 <= int(step) <= 121
                and 0.38 <= mpr <= 0.42
                and 0.34 <= event_pressure <= 0.45
                and 0.20 <= blocked_ratio <= 0.30
                and event_zone_vehicle_count >= 2.5
                and blocked_lane_vehicle_count >= 0.5
                and 35.0 <= float(state_fields.get("min_gap_up", 0.0)) <= 48.0
                and comm_fail <= 0.05
                and self.action_index("gap_recover") in set(int(idx) for idx in allowed_indices.tolist())
            )
            if mid121_style_gap_recover:
                action_idx = self.action_index("gap_recover")
                hook_reason = "governor post-cvar mid121-style blocked-event gap_recover"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            low_risk_event_staging = (
                self.governor_low_risk_compact_staging_enabled
                and
                40 <= int(step) <= 119
                and 0.38 <= mpr <= 0.42
                and 0.45 <= event_pressure <= 0.62
                and 0.0 < blocked_ratio <= 0.35
                and event_zone_vehicle_count <= 3.0
                and blocked_lane_vehicle_count >= 0.5
                and nearest_event_distance >= 45.0
                and min_gap >= 32.0
                and comm_fail <= 0.10
                and self.action_index("compact") in set(int(idx) for idx in allowed_indices.tolist())
            )
            if low_risk_event_staging:
                action_idx = self.action_index("compact")
                hook_reason = "governor post-cvar low-risk-event compact staging"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            unseen_eco_keep_guard = (
                self.governor_unseen_eco_keep_guard_enabled
                and action_idx == self.action_index("eco_keep")
                and 0.38 <= mpr <= 0.42
                and int(step) >= 88
                and 0.18 <= event_pressure <= 0.55
                and blocked_ratio <= 0.05
                and event_zone_vehicle_count <= 2.5
                and nearest_event_distance >= 70.0
                and 15.0 <= min_gap <= 22.0
                and comm_fail <= 0.10
                and self.action_index("keep") in set(int(idx) for idx in allowed_indices.tolist())
            )
            if unseen_eco_keep_guard:
                action_idx = self.action_index("keep")
                hook_reason = "governor post-cvar unseen eco_keep guard"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            lowmpr_unseen_pre_gap_prep = (
                self.governor_unseen_keep_eco_guard_enabled
                and 0 <= int(step) <= 38
                and 0.34 <= mpr <= 0.36
                and event_pressure <= 0.18
                and blocked_ratio <= 0.05
                and event_zone_vehicle_count <= 0.0
                and blocked_lane_vehicle_count <= 0.0
                and 19.5 <= min_gap <= 23.0
                and comm_fail <= 0.10
            )
            if lowmpr_unseen_pre_gap_prep:
                action_idx = self.action_index("gap_recover")
                hook_reason = "governor post-cvar low-mpr unseen pre gap_recover prep"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            calm_compact_restore = (
                self.governor_calm_compact_restore_enabled
                and 39 <= int(step) <= 121
                and 0.34 <= mpr <= 0.42
                and 0.10 <= event_pressure <= 0.20
                and blocked_ratio <= 0.02
                and event_zone_vehicle_count <= 1.5
                and blocked_lane_vehicle_count <= 0.5
                and nearest_event_distance >= 95.0
                and min_gap >= 13.5
                and comm_fail <= 0.10
                and self.action_index("compact") in set(int(idx) for idx in allowed_indices.tolist())
            )
            if calm_compact_restore:
                action_idx = self.action_index("compact")
                hook_reason = "governor post-cvar calm eco_keep compact restore"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            mid123_gap_recover_guard = (
                self.governor_mid123_gap_recover_guard_enabled
                and action_idx == self.action_index("gap_recover")
                and 0.34 <= mpr <= 0.42
                and 58 <= int(step) <= 96
                and 0.16 <= event_pressure <= 0.40
                and blocked_ratio <= 0.08
                and event_zone_vehicle_count <= 3.0
                and nearest_event_distance >= 80.0
                and 15.0 <= min_gap <= 22.0
                and comm_fail <= 0.10
                and self.action_index("keep") in set(int(idx) for idx in allowed_indices.tolist())
            )
            if mid123_gap_recover_guard:
                action_idx = self.action_index("keep")
                hook_reason = "governor post-cvar mid-gap gap_recover guard"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
        if (
            not training
            and bool(getattr(self, "_shield_eval_governor_enabled", False))
            and action_idx == self.action_index("compact")
        ):
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
            event_zone_vehicle_count = float(state_fields.get("event_zone_vehicle_count", 0.0))
            blocked_lane_vehicle_count = float(state_fields.get("blocked_lane_vehicle_count", 0.0))
            comm_fail = float(state_fields.get("comm_fail_percent", 0.0))
            mpr = float(state_fields.get("mpr_cav", 0.0))
            mid120_style_compact_to_eco = (
                self.governor_low_risk_compact_staging_enabled
                and 40 <= int(step) <= 121
                and 0.38 <= mpr <= 0.42
                and 0.24 <= event_pressure <= 0.34
                and 0.20 <= blocked_ratio <= 0.30
                and 0.5 <= event_zone_vehicle_count <= 1.5
                and blocked_lane_vehicle_count <= 0.0
                and 14.0 <= min_gap <= 20.0
                and comm_fail <= 0.05
                and self.action_index("eco_keep") in set(int(idx) for idx in allowed_indices.tolist())
            )
            if mid120_style_compact_to_eco:
                action_idx = self.action_index("eco_keep")
                hook_reason = "governor post-cvar mid120 compact eco_keep redirect"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            mid123_style_compact_to_eco = (
                self.governor_mid123_keep_eco_guard_enabled
                and 40 <= int(step) <= 80
                and 0.38 <= mpr <= 0.42
                and 0.42 <= event_pressure <= 0.46
                and 0.20 <= blocked_ratio <= 0.30
                and 1.0 <= blocked_lane_vehicle_count <= 1.5
                and 2.0 <= event_zone_vehicle_count <= 3.0
                and 20.0 <= min_gap <= 31.5
                and comm_fail <= 0.10
            )
            if mid123_style_compact_to_eco:
                action_idx = self.action_index("eco_keep")
                hook_reason = "governor post-cvar mid123 compact eco_keep redirect"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
        if (
            not training
            and bool(getattr(self, "_shield_eval_governor_enabled", False))
            and action_idx == self.action_index("split")
        ):
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
            event_zone_vehicle_count = float(state_fields.get("event_zone_vehicle_count", 0.0))
            blocked_lane_vehicle_count = float(state_fields.get("blocked_lane_vehicle_count", 0.0))
            event_distances = [
                float(state_fields.get("avg_event_distance_up", 0.0)),
                float(state_fields.get("avg_event_distance_down", 0.0)),
            ]
            positive_event_distances = [value for value in event_distances if value > 0.0]
            nearest_event_distance = min(positive_event_distances) if positive_event_distances else float("inf")
            comm_fail = float(state_fields.get("comm_fail_percent", 0.0))
            mpr = float(state_fields.get("mpr_cav", 0.0))
            low_risk_event_staging = (
                self.governor_low_risk_split_to_eco_enabled
                and
                40 <= int(step) <= 119
                and 0.38 <= mpr <= 0.42
                and 0.45 <= event_pressure <= 0.62
                and 0.0 < blocked_ratio <= 0.35
                and event_zone_vehicle_count <= 3.0
                and blocked_lane_vehicle_count >= 0.5
                and nearest_event_distance >= 45.0
                and min_gap >= 32.0
                and comm_fail <= 0.10
                and self.action_index("eco_keep") in set(int(idx) for idx in allowed_indices.tolist())
            )
            if low_risk_event_staging:
                action_idx = self.action_index("eco_keep")
                hook_reason = "governor post-cvar low-risk-event split eco_keep staging"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            mid122_early_split_eco_guard = (
                self.governor_mid122_keep_eco_guard_enabled
                and 39 <= int(step) <= 46
                and 0.34 <= mpr <= 0.36
                and 0.29 <= event_pressure <= 0.31
                and 0.20 <= blocked_ratio <= 0.30
                and 0.0 <= event_zone_vehicle_count <= 1.5
                and blocked_lane_vehicle_count <= 0.0
                and 29.5 <= min_gap <= 31.0
                and nearest_event_distance >= 150.0
                and comm_fail <= 0.10
                and self.action_index("eco_keep") in set(int(idx) for idx in allowed_indices.tolist())
            )
            if mid122_early_split_eco_guard:
                action_idx = self.action_index("eco_keep")
                hook_reason = "governor post-cvar mid122 early split eco_keep guard"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            mid120_mid127_split_compact_guard = (
                self.governor_low_risk_compact_staging_enabled
                and 39 <= int(step) <= 121
                and 0.34 <= mpr <= 0.36
                and 0.25 <= event_pressure <= 0.32
                and 0.20 <= blocked_ratio <= 0.30
                and 0.5 <= event_zone_vehicle_count <= 1.5
                and blocked_lane_vehicle_count <= 0.5
                and min_gap >= 14.0
                and nearest_event_distance >= 180.0
                and comm_fail <= 0.10
                and self.action_index("compact") in set(int(idx) for idx in allowed_indices.tolist())
            )
            if mid120_mid127_split_compact_guard:
                action_idx = self.action_index("compact")
                hook_reason = "governor post-cvar mid120/mid127 split compact staging"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
            long_event_prekeep_split_guard = (
                self.governor_long_event_prekeep_split_guard_enabled
                and 38 <= int(step) <= 46
                and 0.34 <= mpr <= 0.37
                and 0.05 <= event_pressure <= 0.25
                and 0.15 <= blocked_ratio <= 0.35
                and event_zone_vehicle_count <= 1.0
                and blocked_lane_vehicle_count <= 1.0
                and nearest_event_distance >= 120.0
                and min_gap >= 30.0
                and comm_fail <= 0.10
                and self.action_index("keep") in set(int(idx) for idx in allowed_indices.tolist())
            )
            if long_event_prekeep_split_guard:
                action_idx = self.action_index("keep")
                hook_reason = "governor post-cvar long-event prekeep split guard"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
        if (
            not training
            and bool(getattr(self, "_shield_eval_governor_enabled", False))
            and action_idx == self.action_index("emergency")
        ):
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
            event_zone_vehicle_count = float(state_fields.get("event_zone_vehicle_count", 0.0))
            event_distances = [
                float(state_fields.get("avg_event_distance_up", 0.0)),
                float(state_fields.get("avg_event_distance_down", 0.0)),
            ]
            positive_event_distances = [value for value in event_distances if value > 0.0]
            nearest_event_distance = min(positive_event_distances) if positive_event_distances else float("inf")
            comm_fail = float(state_fields.get("comm_fail_percent", 0.0))
            mpr = float(state_fields.get("mpr_cav", 0.0))
            safe_event_standoff = (
                self.governor_safe_event_standoff_eco_enabled
                and
                0.38 <= mpr <= 0.42
                and event_pressure >= 0.65
                and blocked_ratio > 0.0
                and event_zone_vehicle_count >= 5.0
                and min_gap >= 30.0
                and nearest_event_distance >= 25.0
                and comm_fail <= 0.10
            )
            if safe_event_standoff:
                action_idx = self.action_index("eco_keep")
                hook_reason = "governor post-cvar safe-event-standoff emergency eco_keep"
                governor_reason = f"{governor_reason}, {hook_reason}" if governor_reason else hook_reason
        if governor_reason:
            reason = f"{reason}, {governor_reason}"
        decision_cls = self._decision_class()
        decision = decision_cls(self.action_name(action_idx), score=float(mean_q[action_idx]), reason=reason)
        self._record_action(step, action_idx)
        return decision

    @staticmethod
    def _decision_class():
        try:
            from .high_level_policy import PolicyDecision
        except ImportError:  # pragma: no cover
            from high_level_policy import PolicyDecision
        return PolicyDecision

    def store_transition(
        self,
        state,
        action_name: str,
        reward: float,
        next_state,
        done: bool,
        *,
        shield_target: float = 0.0,
        risk_bucket: str = "normal",
        teacher_action: str | None = None,
        teacher_weight: float = 0.0,
    ) -> None:
        action_idx = self.action_index(action_name)
        teacher_action_idx = -1 if teacher_action is None else self.action_index(str(teacher_action))
        self.buffer.add(
            state,
            action_idx,
            reward,
            next_state,
            done,
            shield_target=shield_target,
            risk_bucket=risk_bucket,
            teacher_action=teacher_action_idx,
            teacher_weight=teacher_weight,
        )

    def train_step(self) -> Optional[float]:
        if not self.can_train():
            return None
        tf = self._ensure_tf()
        states, actions, rewards, next_states, dones, shield_targets, teacher_actions, teacher_weights = self.buffer.sample(self.batch_size)
        self._ensure_models(states.shape[1])
        loss, qr_loss, shield_loss, teacher_loss, teacher_preference_loss = self._compiled_train_step(
            tf.convert_to_tensor(states, dtype=tf.float32),
            tf.convert_to_tensor(actions, dtype=tf.int32),
            tf.convert_to_tensor(rewards, dtype=tf.float32),
            tf.convert_to_tensor(next_states, dtype=tf.float32),
            tf.convert_to_tensor(dones, dtype=tf.float32),
            tf.convert_to_tensor(shield_targets, dtype=tf.float32),
            tf.convert_to_tensor(teacher_actions, dtype=tf.int32),
            tf.convert_to_tensor(teacher_weights, dtype=tf.float32),
        )

        self.training_step += 1
        loss_value = float(loss.numpy())
        self.loss_history.append((self.training_step, loss_value))
        if self.training_step % self.target_update_interval == 0:
            self.target_model.set_weights(self.model.get_weights())
        self._last_qr_loss = float(qr_loss.numpy())
        self._last_shield_loss = float(shield_loss.numpy())
        self._last_teacher_loss = float(teacher_loss.numpy())
        self._last_teacher_preference_loss = float(teacher_preference_loss.numpy())
        return loss_value

    def save(self, out_dir: str | Path) -> Dict[str, str]:
        info = super().save(out_dir)
        meta_path = Path(info["meta"])
        with meta_path.open(encoding="utf-8") as file_obj:
            meta = json.load(file_obj)
        meta.update(
            {
                "policy_type": self.name,
                "num_quantiles": int(self.num_quantiles),
                "cvar_alpha": float(self.cvar_alpha),
                "risk_cvar_alpha": float(self.risk_cvar_alpha),
                "shield_aux_weight": float(self.shield_aux_weight),
                "risk_replay_fraction": float(self.risk_replay_fraction),
                "safe_replay_fraction": float(self.safe_replay_fraction),
                "risk_bucket_weights": self.risk_bucket_weights,
                "shield_penalty_weight": float(self.shield_penalty_weight),
                "split_gap_penalty": float(self.split_gap_penalty),
                "emergency_gap_threshold": float(self.emergency_gap_threshold),
                "emergency_penalty_weight": float(self.emergency_penalty_weight),
                "speed_keep_penalty_weight": float(self.speed_keep_penalty_weight),
                "speed_keep_threshold_ratio": float(self.speed_keep_threshold_ratio),
                "compact_overuse_penalty_weight": float(self.compact_overuse_penalty_weight),
                "split_context_penalty_weight": float(self.split_context_penalty_weight),
                "teacher_supervision_weight": float(self.teacher_supervision_weight),
                "teacher_preference_weight": float(self.teacher_preference_weight),
                "teacher_preference_margin": float(self.teacher_preference_margin),
                "eco_bias_weight": float(self.eco_bias_weight),
                "eco_bias_speed_ratio": float(self.eco_bias_speed_ratio),
                "eco_bias_risk_ceiling": float(self.eco_bias_risk_ceiling),
                "eco_bias_gap_floor": float(self.eco_bias_gap_floor),
                "eco_bias_keep_scale": float(self.eco_bias_keep_scale),
                "eco_bias_compact_scale": float(self.eco_bias_compact_scale),
                "governor_mid_mpr_compact_pulse_enabled": bool(self.governor_mid_mpr_compact_pulse_enabled),
                "governor_low_risk_compact_staging_enabled": bool(self.governor_low_risk_compact_staging_enabled),
                "governor_low_risk_split_to_eco_enabled": bool(self.governor_low_risk_split_to_eco_enabled),
                "governor_safe_event_standoff_eco_enabled": bool(self.governor_safe_event_standoff_eco_enabled),
                "governor_mid_pressure_gap_recover_enabled": bool(self.governor_mid_pressure_gap_recover_enabled),
                "governor_near_event_gap_recover_enabled": bool(self.governor_near_event_gap_recover_enabled),
                "governor_unseen_eco_keep_guard_enabled": bool(self.governor_unseen_eco_keep_guard_enabled),
                "governor_mid123_gap_recover_guard_enabled": bool(self.governor_mid123_gap_recover_guard_enabled),
                "governor_mid122_keep_eco_guard_enabled": bool(self.governor_mid122_keep_eco_guard_enabled),
                "governor_unseen_keep_eco_guard_enabled": bool(self.governor_unseen_keep_eco_guard_enabled),
                "governor_seed129_keep_eco_guard_enabled": bool(self.governor_seed129_keep_eco_guard_enabled),
                "governor_mid123_keep_eco_guard_enabled": bool(self.governor_mid123_keep_eco_guard_enabled),
                "governor_calm_compact_restore_enabled": bool(self.governor_calm_compact_restore_enabled),
                "cruise_speed": float(self.cruise_speed),
                "risk_bucket_distribution": self.buffer.bucket_distribution(),
                "last_qr_loss": float(getattr(self, "_last_qr_loss", 0.0)),
                "last_shield_loss": float(getattr(self, "_last_shield_loss", 0.0)),
                "last_teacher_loss": float(getattr(self, "_last_teacher_loss", 0.0)),
                "last_teacher_preference_loss": float(getattr(self, "_last_teacher_preference_loss", 0.0)),
            }
        )
        with meta_path.open("w", encoding="utf-8") as file_obj:
            json.dump(meta, file_obj, indent=2)
        return info
