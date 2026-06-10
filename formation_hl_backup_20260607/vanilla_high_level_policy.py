"""Vanilla DDQN high-level baseline without communication-aware governor shaping."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

try:
    from .trainable_high_level_policy import TrainableHighLevelPolicy
except ImportError:  # pragma: no cover
    from trainable_high_level_policy import TrainableHighLevelPolicy


class VanillaDDQNHighLevelPolicy(TrainableHighLevelPolicy):
    """Plain DDQN baseline over the same discrete action space."""

    name = "vanilla_ddqn"

    def _apply_soft_governor(
        self,
        *,
        action_idx: int,
        q_values: np.ndarray,
        allowed_indices: np.ndarray,
        step: int,
        scheduler,
        state_fields: Dict[str, float],
    ):
        del q_values, allowed_indices, step, scheduler, state_fields
        self._update_governor_state(action_idx)
        return int(action_idx), ""

    def q_values(self, state_vector) -> np.ndarray:
        state = np.asarray(state_vector, dtype=np.float32).copy()
        # Remove communication augmentation so the baseline only learns from traffic state.
        if state.ndim == 1 and state.shape[0] >= 5:
            state = state.copy()
            state[-5:] = 0.0
        return super().q_values(state)
