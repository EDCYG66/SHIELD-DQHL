"""PPO high-level baseline for platoon reconfiguration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
from collections import deque

import numpy as np

try:
    from .high_level_policy import FormationHighLevelPolicy, PolicyDecision
    from .trainable_high_level_policy import TRAINABLE_ACTIONS
except ImportError:  # pragma: no cover
    from high_level_policy import FormationHighLevelPolicy, PolicyDecision
    from trainable_high_level_policy import TRAINABLE_ACTIONS


@dataclass
class PPORolloutStep:
    state: np.ndarray
    action: int
    reward: float
    done: float
    log_prob: float
    value: float


class PPOHighLevelPolicy(FormationHighLevelPolicy):
    """Discrete-action PPO baseline over the same high-level action set."""

    name = "ppo"

    def __init__(
        self,
        *,
        state_dim: Optional[int] = None,
        hidden_dims: Tuple[int, int] = (128, 96),
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        train_epochs: int = 4,
        minibatch_size: int = 128,
        rollout_capacity: int = 4096,
        seed: int = 123,
    ):
        self.state_dim = state_dim
        self.hidden_dims = tuple(int(v) for v in hidden_dims)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.learning_rate = float(learning_rate)
        self.clip_ratio = float(clip_ratio)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.train_epochs = int(train_epochs)
        self.minibatch_size = int(minibatch_size)
        self.rollout_capacity = int(rollout_capacity)
        self.seed = int(seed)
        self.num_actions = len(TRAINABLE_ACTIONS)
        self._rng = np.random.default_rng(self.seed)
        self._tf = None
        self.actor = None
        self.critic = None
        self.optimizer = None
        self.rollout: List[PPORolloutStep] = []
        self.buffer = type("RolloutBufferView", (), {"buffer": deque(maxlen=self.rollout_capacity), "__len__": lambda self_: len(self_.buffer)})()
        self.training_step = 0
        self.episode_count = 0
        self.loss_history: List[Tuple[int, float]] = []
        self.action_history: List[Tuple[int, int]] = []
        self.last_action_counts: Dict[str, int] = {name: 0 for name in TRAINABLE_ACTIONS}
        self._switch_count = 0
        self._last_action_idx: Optional[int] = None
        if state_dim is not None:
            self._ensure_models(int(state_dim))

    def _ensure_tf(self):
        if self._tf is None:
            try:
                import tensorflow as tf  # pylint: disable=import-error
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "TensorFlow is required for PPOHighLevelPolicy. "
                    "Please run this policy inside the tf2.12 environment."
                ) from exc
            self._tf = tf
        return self._tf

    def _ensure_models(self, state_dim: int):
        if self.actor is not None and self.critic is not None and self.state_dim == state_dim:
            return
        tf = self._ensure_tf()
        self.state_dim = int(state_dim)

        def build_backbone():
            he = tf.keras.initializers.HeNormal(seed=self.seed)
            return tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(self.state_dim,)),
                tf.keras.layers.Dense(self.hidden_dims[0], activation="relu", kernel_initializer=he),
                tf.keras.layers.Dense(self.hidden_dims[1], activation="relu", kernel_initializer=he),
            ])

        actor_backbone = build_backbone()
        critic_backbone = build_backbone()
        actor_input = tf.keras.Input(shape=(self.state_dim,), dtype=tf.float32)
        actor_logits = tf.keras.layers.Dense(self.num_actions, activation=None)(actor_backbone(actor_input))
        self.actor = tf.keras.Model(actor_input, actor_logits)
        critic_input = tf.keras.Input(shape=(self.state_dim,), dtype=tf.float32)
        critic_value = tf.keras.layers.Dense(1, activation=None)(critic_backbone(critic_input))
        self.critic = tf.keras.Model(critic_input, critic_value)
        _ = self.actor(tf.zeros((1, self.state_dim), dtype=tf.float32), training=False)
        _ = self.critic(tf.zeros((1, self.state_dim), dtype=tf.float32), training=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def action_index(self, action_name: str) -> int:
        return int(TRAINABLE_ACTIONS.index(action_name))

    def action_name(self, action_index: int) -> str:
        return TRAINABLE_ACTIONS[int(action_index)]

    def epsilon(self, global_step: int) -> float:
        del global_step
        return 0.0

    def reset(self) -> None:
        self._last_action_idx = None

    def _allowed_action_indices(self, step: int, scheduler, state_fields: Dict[str, float]) -> np.ndarray:
        del step, scheduler
        allowed = list(range(self.num_actions))
        emergency_idx = self.action_index("emergency")
        min_gap = min(float(state_fields.get("min_gap_up", 0.0)), float(state_fields.get("min_gap_down", 0.0)))
        blocked_ratio = max(
            float(state_fields.get("blocked_lane_ratio_up", 0.0)),
            float(state_fields.get("blocked_lane_ratio_down", 0.0)),
        )
        event_pressure = max(
            float(state_fields.get("event_pressure_up", 0.0)),
            float(state_fields.get("event_pressure_down", 0.0)),
        )
        emergency_needed = (min_gap > 1e-6 and min_gap < 9.0) or (blocked_ratio > 0.0 and event_pressure >= 0.65)
        if emergency_idx in allowed and not emergency_needed:
            allowed.remove(emergency_idx)
        return np.asarray(allowed if allowed else [self.action_index("keep")], dtype=np.int32)

    def _policy_value(self, state_vector: np.ndarray) -> tuple[np.ndarray, float]:
        state = np.asarray(state_vector, dtype=np.float32).reshape(1, -1)
        self._ensure_models(state.shape[1])
        logits = self.actor(state, training=False).numpy()[0]
        value = float(self.critic(state, training=False).numpy()[0, 0])
        return np.asarray(logits, dtype=np.float32), value

    def select_action(
        self,
        step: int,
        scheduler,
        state_fields: Dict[str, float],
        state_vector: Optional[np.ndarray] = None,
        training: bool = False,
    ) -> PolicyDecision:
        if state_vector is None:
            raise ValueError("PPOHighLevelPolicy requires state_vector for action selection.")
        logits, value = self._policy_value(np.asarray(state_vector, dtype=np.float32))
        allowed = self._allowed_action_indices(step, scheduler, state_fields)
        masked_logits = np.full_like(logits, -1e9, dtype=np.float32)
        masked_logits[allowed] = logits[allowed]
        probs = self._softmax(masked_logits)
        if training:
            action_idx = int(self._rng.choice(np.arange(self.num_actions), p=probs))
            reason = "ppo-sample"
        else:
            action_idx = int(np.argmax(probs))
            reason = "ppo-greedy"
        self._record_action(step, action_idx)
        return PolicyDecision(self.action_name(action_idx), score=float(value), reason=reason)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        denom = np.sum(exp)
        if denom <= 0.0 or not np.isfinite(denom):
            out = np.zeros_like(logits, dtype=np.float32)
            out[self.action_index("keep")] = 1.0
            return out
        return np.asarray(exp / denom, dtype=np.float32)

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
        logits, value = self._policy_value(np.asarray(state, dtype=np.float32))
        probs = self._softmax(logits)
        action_idx = self.action_index(action_name)
        log_prob = float(np.log(max(1e-8, probs[action_idx])))
        self.rollout.append(
            PPORolloutStep(
                state=np.asarray(state, dtype=np.float32).copy(),
                action=int(action_idx),
                reward=float(reward),
                done=float(done),
                log_prob=log_prob,
                value=float(value),
            )
        )
        self.buffer.buffer.append(self.rollout[-1])
        if len(self.rollout) > self.rollout_capacity:
            self.rollout = self.rollout[-self.rollout_capacity:]
            while len(self.buffer.buffer) > self.rollout_capacity:
                self.buffer.buffer.popleft()

    def can_train(self) -> bool:
        return len(self.rollout) >= max(32, self.minibatch_size)

    def train_step(self) -> Optional[float]:
        if not self.can_train():
            return None
        tf = self._ensure_tf()
        states = np.stack([item.state for item in self.rollout], axis=0).astype(np.float32)
        actions = np.asarray([item.action for item in self.rollout], dtype=np.int32)
        rewards = np.asarray([item.reward for item in self.rollout], dtype=np.float32)
        dones = np.asarray([item.done for item in self.rollout], dtype=np.float32)
        old_log_probs = np.asarray([item.log_prob for item in self.rollout], dtype=np.float32)
        values = np.asarray([item.value for item in self.rollout], dtype=np.float32)
        next_values = np.concatenate([values[1:], np.zeros(1, dtype=np.float32)], axis=0)
        deltas = rewards + self.gamma * (1.0 - dones) * next_values - values
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        adv_mean = float(np.mean(advantages))
        adv_std = float(np.std(advantages) + 1e-8)
        advantages = (advantages - adv_mean) / adv_std

        ds_size = states.shape[0]
        indices = np.arange(ds_size)
        losses: List[float] = []
        for _ in range(self.train_epochs):
            self._rng.shuffle(indices)
            for start in range(0, ds_size, self.minibatch_size):
                batch_idx = indices[start:start + self.minibatch_size]
                if batch_idx.size == 0:
                    continue
                s_t = tf.convert_to_tensor(states[batch_idx], dtype=tf.float32)
                a_t = tf.convert_to_tensor(actions[batch_idx], dtype=tf.int32)
                old_lp_t = tf.convert_to_tensor(old_log_probs[batch_idx], dtype=tf.float32)
                ret_t = tf.convert_to_tensor(returns[batch_idx], dtype=tf.float32)
                adv_t = tf.convert_to_tensor(advantages[batch_idx], dtype=tf.float32)
                with tf.GradientTape() as tape:
                    logits = self.actor(s_t, training=True)
                    values_pred = tf.squeeze(self.critic(s_t, training=True), axis=1)
                    log_probs = tf.nn.log_softmax(logits, axis=1)
                    action_mask = tf.one_hot(a_t, self.num_actions, dtype=log_probs.dtype)
                    selected_log_probs = tf.reduce_sum(log_probs * action_mask, axis=1)
                    ratios = tf.exp(selected_log_probs - old_lp_t)
                    clipped = tf.clip_by_value(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                    actor_loss = -tf.reduce_mean(tf.minimum(ratios * adv_t, clipped * adv_t))
                    value_loss = tf.reduce_mean(tf.square(ret_t - values_pred))
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(log_probs) * log_probs, axis=1))
                    loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                variables = self.actor.trainable_variables + self.critic.trainable_variables
                grads = tape.gradient(loss, variables)
                grads, _ = tf.clip_by_global_norm(grads, 5.0)
                self.optimizer.apply_gradients(zip(grads, variables))
                losses.append(float(loss.numpy()))

        self.training_step += 1
        mean_loss = float(np.mean(losses)) if losses else 0.0
        self.loss_history.append((self.training_step, mean_loss))
        self.rollout.clear()
        self.buffer.buffer.clear()
        return mean_loss

    def save(self, out_dir: str | Path) -> Dict[str, str]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        if self.actor is None or self.critic is None:
            raise RuntimeError("Cannot save an uninitialized PPO policy.")
        self.actor.save_weights(str(out_path / "ppo_actor.weights.h5"))
        self.critic.save_weights(str(out_path / "ppo_critic.weights.h5"))
        meta = {
            "state_dim": int(self.state_dim or 0),
            "hidden_dims": list(self.hidden_dims),
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "learning_rate": self.learning_rate,
            "clip_ratio": self.clip_ratio,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "train_epochs": self.train_epochs,
            "minibatch_size": self.minibatch_size,
            "rollout_capacity": self.rollout_capacity,
            "num_actions": self.num_actions,
            "action_names": list(TRAINABLE_ACTIONS),
            "training_step": int(self.training_step),
            "episode_count": int(self.episode_count),
            "switch_count": int(self.switch_count),
            "action_distribution": self.action_distribution(),
        }
        with (out_path / "ppo_policy_meta.json").open("w", encoding="utf-8") as file_obj:
            json.dump(meta, file_obj, indent=2)
        if self.loss_history:
            with (out_path / "ppo_loss.csv").open("w", encoding="utf-8") as file_obj:
                file_obj.write("step,loss\n")
                for step, loss in self.loss_history:
                    file_obj.write(f"{step},{loss}\n")
        if self.action_history:
            with (out_path / "ppo_action_history.csv").open("w", encoding="utf-8") as file_obj:
                file_obj.write("step,action_idx,action_name\n")
                for step, action_idx in self.action_history:
                    file_obj.write(f"{step},{action_idx},{self.action_name(action_idx)}\n")
        return {
            "weights": str(out_path / "ppo_actor.weights.h5"),
            "critic_weights": str(out_path / "ppo_critic.weights.h5"),
            "meta": str(out_path / "ppo_policy_meta.json"),
        }

    def load(self, weights_path: str | Path, meta_path: str | Path | None = None) -> None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"PPO actor weights not found: {weights_path}")
        critic_path = weights_path.parent / "ppo_critic.weights.h5"
        if meta_path is not None:
            meta_path = Path(meta_path)
            if meta_path.exists():
                with meta_path.open(encoding="utf-8") as file_obj:
                    meta = json.load(file_obj)
                state_dim = int(meta.get("state_dim", 0))
                if state_dim > 0:
                    self.hidden_dims = tuple(int(v) for v in meta.get("hidden_dims", list(self.hidden_dims)))
                    self.gamma = float(meta.get("gamma", self.gamma))
                    self.gae_lambda = float(meta.get("gae_lambda", self.gae_lambda))
                    self.learning_rate = float(meta.get("learning_rate", self.learning_rate))
                    self.clip_ratio = float(meta.get("clip_ratio", self.clip_ratio))
                    self.entropy_coef = float(meta.get("entropy_coef", self.entropy_coef))
                    self.value_coef = float(meta.get("value_coef", self.value_coef))
                    self.train_epochs = int(meta.get("train_epochs", self.train_epochs))
                    self.minibatch_size = int(meta.get("minibatch_size", self.minibatch_size))
                    self.rollout_capacity = int(meta.get("rollout_capacity", self.rollout_capacity))
                    self._ensure_models(state_dim)
        if self.actor is None or self.critic is None:
            raise RuntimeError("State dimension is unknown. Load metadata first or initialize with state_dim.")
        self.actor.load_weights(str(weights_path))
        if critic_path.exists():
            self.critic.load_weights(str(critic_path))
