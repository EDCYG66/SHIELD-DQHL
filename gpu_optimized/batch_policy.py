"""BatchInferencePolicy — wraps high-level policies for N-env inference."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from tf_runtime import configure_tensorflow_runtime


configure_tensorflow_runtime()

import tensorflow as tf  # noqa: E402

configure_tensorflow_runtime(tf)

from formation.high_level_policy import PolicyDecision  # noqa: E402
from formation.trainable_high_level_policy import TRAINABLE_ACTIONS, TrainableHighLevelPolicy  # noqa: E402


class BatchInferencePolicy:
    """Wrapper that batches inference where safe and falls back otherwise."""

    def __init__(self, policy):
        self.policy = policy
        self._batch_model_compiled = False

    @property
    def model(self):
        return self.policy.model

    def batch_q_values(self, state_vectors_batch: np.ndarray) -> np.ndarray:
        """(N, state_dim) -> (N, n_actions) Q-values via single GPU forward pass."""
        batch = np.asarray(state_vectors_batch, dtype=np.float32)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        self.policy._ensure_models(batch.shape[1])
        model_out = self.policy.model(batch, training=False)
        if isinstance(model_out, (tuple, list)):
            q_batch = model_out[0]
            if len(np.asarray(q_batch).shape) == 3:
                q_batch = np.mean(np.asarray(q_batch), axis=2)
        else:
            q_batch = model_out
        q_batch = np.asarray(q_batch)
        return np.asarray(q_batch, dtype=np.float32)

    def _supports_batched_dqn_inference(self) -> bool:
        return isinstance(self.policy, TrainableHighLevelPolicy) and str(getattr(self.policy, "name", "")) in {
            "learned",
            "vanilla_ddqn",
        }

    def _select_actions_fallback(
        self,
        step: int,
        schedulers: List,
        state_fields_list: List[Dict[str, float]],
        state_vectors_batch: np.ndarray,
        training: bool,
    ) -> List[PolicyDecision]:
        decisions: List[PolicyDecision] = []
        for i, state_vector in enumerate(state_vectors_batch):
            decision = self.policy.select_action(
                step=step,
                scheduler=schedulers[i],
                state_fields=state_fields_list[i],
                state_vector=state_vector,
                training=training,
            )
            decisions.append(decision)
        return decisions

    def batch_select_actions(
        self,
        step: int,
        schedulers: List,
        state_fields_list: List[Dict[str, float]],
        state_vectors_batch: np.ndarray,
        training: bool = True,
    ) -> List[PolicyDecision]:
        """Batch inference + per-env masking/governor."""
        if not self._supports_batched_dqn_inference():
            return self._select_actions_fallback(
                step=step,
                schedulers=schedulers,
                state_fields_list=state_fields_list,
                state_vectors_batch=state_vectors_batch,
                training=training,
            )

        n = state_vectors_batch.shape[0]
        q_all = self.batch_q_values(state_vectors_batch)
        decisions: List[PolicyDecision] = []

        eps = self.policy.epsilon(step) if training else 0.0

        for i in range(n):
            q = q_all[i]
            fields = state_fields_list[i]
            scheduler = schedulers[i]

            allowed = self.policy._allowed_action_indices(step, scheduler, fields)
            masked = len(allowed) < self.policy.num_actions

            if training and self.policy._rng.random() < eps:
                action_idx = int(self.policy._rng.choice(allowed))
                reason = f"epsilon={eps:.3f}"
            else:
                masked_q = np.full_like(q, -np.inf, dtype=np.float32)
                masked_q[allowed] = q[allowed]
                if training:
                    total_actions = max(1, sum(self.policy.last_action_counts.values()))
                    ucb_bonus = np.zeros_like(q, dtype=np.float32)
                    for idx in allowed:
                        count = max(1, self.policy.last_action_counts.get(
                            self.policy.action_name(int(idx)), 1))
                        ucb_bonus[int(idx)] = 0.15 * float(
                            np.sqrt(np.log(total_actions + 1) / count))
                    masked_q[allowed] = masked_q[allowed] + ucb_bonus[allowed]
                action_idx = int(np.argmax(masked_q))
                reason = "greedy"

            if masked:
                reason += ", emergency masked"

            action_idx, gov_reason = self.policy._apply_soft_governor(
                action_idx=action_idx,
                q_values=q,
                allowed_indices=allowed,
                step=step,
                scheduler=scheduler,
                state_fields=fields,
            )
            score = float(q[action_idx])
            if gov_reason:
                reason = f"{reason}, {gov_reason}"

            self.policy._record_action(step, action_idx)
            decisions.append(PolicyDecision(
                self.policy.action_name(action_idx),
                score=score,
                reason=reason,
            ))
        return decisions
