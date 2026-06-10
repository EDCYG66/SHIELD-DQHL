"""Optimized replay buffer and training parameter overrides."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from formation.trainable_high_level_policy import HighLevelTransition, TrainableHighLevelPolicy

from .config import BATCH_SIZE, REPLAY_CAPACITY, TRAIN_EVERY_K, TRAIN_UPDATES_PER_TRIGGER


class NumpyReplayBuffer:
    """Pre-allocated numpy replay buffer — no Python object overhead on sample()."""

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.size = 0
        self.pos = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.teacher_actions = np.full(capacity, -1, dtype=np.int32)
        self.teacher_weights = np.zeros(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done,
        *,
        teacher_action: int = -1,
        teacher_weight: float = 0.0,
    ) -> None:
        idx = self.pos % self.capacity
        self.states[idx] = np.asarray(state, dtype=np.float32).ravel()
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.next_states[idx] = np.asarray(next_state, dtype=np.float32).ravel()
        self.dones[idx] = float(done)
        self.teacher_actions[idx] = int(teacher_action)
        self.teacher_weights[idx] = float(max(0.0, teacher_weight))
        self.pos += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.randint(0, self.size, size=int(batch_size))
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.teacher_actions[indices],
            self.teacher_weights[indices],
        )


class NumpyReplayBufferAdapter:
    """Adapter that matches HighLevelReplayBuffer interface for TrainableHighLevelPolicy."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._inner: Optional[NumpyReplayBuffer] = None
        self.buffer = self

    def __len__(self) -> int:
        return len(self._inner) if self._inner is not None else 0

    def __iter__(self):
        if self._inner is None:
            return
        for i in range(self._inner.size):
            yield (
                self._inner.states[i],
                self._inner.actions[i],
                self._inner.rewards[i],
                self._inner.next_states[i],
                self._inner.dones[i],
                self._inner.teacher_actions[i],
                self._inner.teacher_weights[i],
            )

    def append(self, transition) -> None:
        if self._inner is None:
            state_dim = len(np.asarray(transition.state).ravel())
            self._inner = NumpyReplayBuffer(self.capacity, state_dim)
        self._inner.add(
            transition.state,
            int(transition.action),
            float(transition.reward),
            transition.next_state,
            float(transition.done),
            teacher_action=int(getattr(transition, "teacher_action", -1)),
            teacher_weight=float(getattr(transition, "teacher_weight", 0.0)),
        )

    def add(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
        *,
        teacher_action: int = -1,
        teacher_weight: float = 0.0,
    ) -> None:
        if self._inner is None:
            state_dim = len(np.asarray(state).ravel())
            self._inner = NumpyReplayBuffer(self.capacity, state_dim)
        self._inner.add(
            state,
            action,
            reward,
            next_state,
            done,
            teacher_action=teacher_action,
            teacher_weight=teacher_weight,
        )

    def sample(self, batch_size: int):
        if self._inner is None or self._inner.size < batch_size:
            raise ValueError("Not enough samples in buffer")
        return self._inner.sample(batch_size)


class OptimizedShieldReplayBuffer:
    """Preallocated shield-aware replay buffer with weighted bucket sampling."""

    def __init__(
        self,
        capacity: int,
        *,
        risk_fraction: float,
        safe_fraction: float,
        risk_bucket_weights: Dict[str, float],
    ):
        self.capacity = int(capacity)
        self.risk_fraction = float(np.clip(risk_fraction, 0.0, 1.0))
        self.safe_fraction = float(np.clip(safe_fraction, 0.0, 0.5))
        self.risk_bucket_weights = {str(k): float(max(0.0, v)) for k, v in dict(risk_bucket_weights).items()}
        self._inner: Optional[NumpyReplayBuffer] = None
        self.buffer = self
        self.shield_targets = np.zeros(self.capacity, dtype=np.float32)
        self.teacher_actions = np.full(self.capacity, -1, dtype=np.int32)
        self.teacher_weights = np.zeros(self.capacity, dtype=np.float32)
        self.risk_buckets: List[str] = ["normal"] * self.capacity

    def __len__(self) -> int:
        return len(self._inner) if self._inner is not None else 0

    def __iter__(self):
        if self._inner is None:
            return
        for i in range(self._inner.size):
            yield HighLevelTransition(
                state=self._inner.states[i].copy(),
                action=int(self._inner.actions[i]),
                reward=float(self._inner.rewards[i]),
                next_state=self._inner.next_states[i].copy(),
                done=float(self._inner.dones[i]),
                shield_target=float(self.shield_targets[i]),
                risk_bucket=str(self.risk_buckets[i]),
                teacher_action=int(self.teacher_actions[i]),
                teacher_weight=float(self.teacher_weights[i]),
            )

    def append(self, transition) -> None:
        self.add(
            transition.state,
            int(transition.action),
            float(transition.reward),
            transition.next_state,
            float(transition.done),
            shield_target=float(getattr(transition, "shield_target", 0.0)),
            risk_bucket=str(getattr(transition, "risk_bucket", "normal")),
            teacher_action=int(getattr(transition, "teacher_action", -1)),
            teacher_weight=float(getattr(transition, "teacher_weight", 0.0)),
        )

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
        if self._inner is None:
            state_dim = len(np.asarray(state, dtype=np.float32).ravel())
            self._inner = NumpyReplayBuffer(self.capacity, state_dim)
        idx = self._inner.pos % self.capacity
        self._inner.add(state, action, reward, next_state, done)
        self.shield_targets[idx] = float(np.clip(shield_target, 0.0, 1.0))
        self.teacher_actions[idx] = int(teacher_action)
        self.teacher_weights[idx] = float(max(0.0, teacher_weight))
        self.risk_buckets[idx] = str(risk_bucket or "normal")

    def _bucket_weight(self, bucket: str) -> float:
        return float(max(0.0, self.risk_bucket_weights.get(str(bucket), 1.0)))

    def _weighted_sample_risk_indices(
        self,
        buckets: Dict[str, List[int]],
        count: int,
    ) -> List[int]:
        pools = {key: list(values) for key, values in buckets.items() if key != "normal" and values}
        sampled: List[int] = []
        while len(sampled) < int(count) and pools:
            keys = [key for key, values in pools.items() if values]
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
        if self._inner is None or self._inner.size < batch_size:
            raise ValueError("Not enough samples in buffer")

        batch_size = int(batch_size)
        if batch_size >= self._inner.size:
            indices = list(range(self._inner.size))
        else:
            buckets: Dict[str, List[int]] = defaultdict(list)
            for idx in range(self._inner.size):
                buckets[str(self.risk_buckets[idx] or "normal")].append(idx)

            normal_items = buckets.pop("normal", [])
            safe_items = buckets.pop("eco_safe", [])
            risk_items = [idx for values in buckets.values() for idx in values]
            safe_count = min(len(safe_items), int(round(batch_size * self.safe_fraction)))
            risk_budget = max(0, batch_size - safe_count)
            risk_count = min(len(risk_items), int(round(batch_size * self.risk_fraction)), risk_budget)
            normal_count = batch_size - safe_count - risk_count
            if normal_count > len(normal_items):
                deficit = normal_count - len(normal_items)
                risk_extra_budget = max(0, batch_size - safe_count - risk_count)
                risk_count = min(len(risk_items), risk_count + deficit, risk_count + risk_extra_budget)
                normal_count = batch_size - safe_count - risk_count

            indices = []
            if safe_count > 0 and safe_items:
                indices.extend(random.sample(safe_items, safe_count))
            if risk_count > 0:
                indices.extend(self._weighted_sample_risk_indices(buckets, risk_count))
            if normal_count > 0 and normal_items:
                indices.extend(random.sample(normal_items, normal_count))
            if len(indices) < batch_size:
                selected = set(indices)
                remaining = [idx for idx in range(self._inner.size) if idx not in selected]
                indices.extend(random.sample(remaining, min(len(remaining), batch_size - len(indices))))

        indices_arr = np.asarray(indices, dtype=np.int64)
        return (
            self._inner.states[indices_arr],
            self._inner.actions[indices_arr],
            self._inner.rewards[indices_arr],
            self._inner.next_states[indices_arr],
            self._inner.dones[indices_arr],
            self.shield_targets[indices_arr],
            self.teacher_actions[indices_arr],
            self.teacher_weights[indices_arr],
        )

    def bucket_distribution(self) -> Dict[str, float]:
        if self._inner is None or self._inner.size <= 0:
            return {}
        counts: Dict[str, int] = defaultdict(int)
        for idx in range(self._inner.size):
            counts[str(self.risk_buckets[idx] or "normal")] += 1
        total = max(1, self._inner.size)
        return {key: float(value) / float(total) for key, value in sorted(counts.items())}


def install_optimized_training(policy: TrainableHighLevelPolicy) -> None:
    """Replace policy's replay buffer and update training hyperparams."""
    if str(getattr(policy, "name", "")) == "shield_dqhl":
        policy.buffer = OptimizedShieldReplayBuffer(
            capacity=REPLAY_CAPACITY,
            risk_fraction=float(getattr(policy, "risk_replay_fraction", 0.65)),
            safe_fraction=float(getattr(policy, "safe_replay_fraction", 0.15)),
            risk_bucket_weights=dict(getattr(policy, "risk_bucket_weights", {}) or {}),
        )
    else:
        policy.buffer = NumpyReplayBufferAdapter(capacity=REPLAY_CAPACITY)
    policy.batch_size = BATCH_SIZE
