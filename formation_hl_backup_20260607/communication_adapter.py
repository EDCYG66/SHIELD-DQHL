"""Adapter that plugs the communication module into the formation environment."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    from .path_utils import ensure_communication_on_path
except ImportError:  # pragma: no cover
    from path_utils import ensure_communication_on_path

ensure_communication_on_path()
try:  # pragma: no cover - runtime availability depends on the local environment
    from agent import Agent  # type: ignore  # noqa: E402
    _AGENT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    Agent = None  # type: ignore
    _AGENT_IMPORT_ERROR = exc


class CommunicationCoordinator:
    """Runs the existing communication policy on the shared highway environment."""

    def __init__(
        self,
        env,
        *,
        gnn_type: str = "gat",
        policy_mode: str = "auto",
        dqn_weights: Optional[str] = None,
        gnn_weights: Optional[str] = None,
        weights_dir: Optional[str] = None,
        run_dir: Optional[str] = None,
        decision_interval_steps: int = 2,
        agent_kwargs: Optional[Dict[str, object]] = None,
    ):
        self.env = env
        self.gnn_type = str(gnn_type)
        self.run_dir = run_dir
        self.decision_interval_steps = max(1, int(decision_interval_steps))
        self.agent_kwargs = dict(agent_kwargs or {})
        self.policy_mode = self._resolve_policy_mode(policy_mode)
        self.agent = None
        self._last_action_step = -10**9
        self._last_action_signature: tuple[int, int, int] | None = None
        self._cached_actions: Optional[np.ndarray] = None
        self._last_topology_epoch = int(getattr(self.env, "topology_epoch", 0))
        if self.policy_mode == "agent":
            self.agent = Agent(
                [],
                env,
                gnn_type=self.gnn_type,
                warmup_steps=1,
                epsilon_min=0.0,
                epsilon_decay_steps=1,
                speed_mode=True,
                run_dir=run_dir,
                **self.agent_kwargs,
            )
            self.agent.training = False
        self.last_metrics: Dict[str, float] = self._empty_metrics()
        self._force_links_active()
        if self.agent is not None:
            self._build_agent_networks()
        resolved_dqn, resolved_gnn = self._resolve_weight_paths(
            dqn_weights=dqn_weights,
            gnn_weights=gnn_weights,
            weights_dir=weights_dir,
        )
        if resolved_dqn and self.agent is not None:
            self.load_dqn_weights(resolved_dqn)
        if resolved_gnn and self.agent is not None:
            self.load_gnn_weights(resolved_gnn)

    def _empty_metrics(self) -> Dict[str, float]:
        return {
            "comm_v2i_rate_total": 0.0,
            "comm_v2v_success": 0.0,
            "comm_used_rb_ratio": 0.0,
            "comm_mean_power_norm": 0.0,
            "comm_fail_percent": 0.0,
        }

    def _resolve_policy_mode(self, policy_mode: str) -> str:
        requested = str(policy_mode or "auto").lower()
        if requested not in {"auto", "agent", "heuristic"}:
            requested = "auto"
        if requested == "heuristic":
            return "heuristic"
        if requested == "agent":
            if Agent is None:
                raise RuntimeError(
                    f"Communication agent requested but unavailable: {_AGENT_IMPORT_ERROR}"
                )
            return "agent"
        return "agent" if Agent is not None else "heuristic"

    def rebind_env(self, env) -> None:
        self.env = env
        if self.agent is not None:
            self.agent.env = env
            self.agent._ensure_action_buffers()
            self._build_agent_networks()
        self._force_links_active()
        self._last_action_step = -10**9
        self._last_action_signature = None
        self._cached_actions = None
        self._last_topology_epoch = int(getattr(self.env, "topology_epoch", 0))
        self.last_metrics = self._empty_metrics()

    def _build_agent_networks(self) -> None:
        if self.agent is None:
            return
        try:
            self.agent._ensure_action_buffers()
            self.agent._sync_graph_topology(force=True)
            # In rollout/eval mode we only need to materialize the graph and embeddings,
            # not run an on-the-fly GAT training step during initialization/reset.
            _ = self.agent.initial_better_state(0, False)
            self._last_topology_epoch = int(getattr(self.env, "topology_epoch", 0))
        except Exception:
            # Fallback: DQN is already built in __init__; GNN may be built lazily later.
            pass

    def _env_signature(self) -> tuple[int, int]:
        return (
            len(getattr(self.env, "vehicles", [])),
            int(getattr(self.env, "topology_epoch", 0)),
        )

    def _topology_changed(self) -> bool:
        topology_epoch = int(getattr(self.env, "topology_epoch", 0))
        changed = topology_epoch != self._last_topology_epoch
        self._last_topology_epoch = topology_epoch
        return changed

    def _can_reuse_actions(self, step: int) -> bool:
        if self._cached_actions is None:
            return False
        if (int(step) - int(self._last_action_step)) < self.decision_interval_steps:
            return True
        return False

    def _resolve_weight_paths(
        self,
        *,
        dqn_weights: Optional[str],
        gnn_weights: Optional[str],
        weights_dir: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        resolved_dqn = str(dqn_weights) if dqn_weights else None
        resolved_gnn = str(gnn_weights) if gnn_weights else None
        if weights_dir:
            base = Path(weights_dir)
            if base.is_file():
                if resolved_dqn is None:
                    resolved_dqn = str(base)
            elif base.is_dir():
                if resolved_dqn is None:
                    resolved_dqn = self._pick_first_existing(base, [
                        f"dqn_weights_{self.gnn_type}.weights.h5",
                        "dqn_weights.weights.h5",
                        "dqn_weights.h5",
                        "DQN_weights.h5",
                    ])
                if resolved_gnn is None:
                    resolved_gnn = self._pick_first_existing(base, [
                        f"gnn_weights_{self.gnn_type}.weights.h5",
                        f"gnn_weights_{self.gnn_type}.h5",
                        "gnn_weights.weights.h5",
                        "gnn_weights.h5",
                        "GNN_weights_sage.h5",
                    ])
        return resolved_dqn, resolved_gnn

    def _pick_first_existing(self, base: Path, candidates: list[str]) -> Optional[str]:
        for name in candidates:
            path = base / name
            if path.exists():
                return str(path)
        return None

    def load_dqn_weights(self, dqn_weights: str) -> None:
        if self.agent is None:
            raise RuntimeError("Cannot load DQN weights when the communication agent is unavailable.")
        weights_path = Path(dqn_weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"DQN weights not found: {weights_path}")
        self.agent.dqn.model.load_weights(str(weights_path))
        self.agent.dqn.target_model.load_weights(str(weights_path))

    def load_gnn_weights(self, gnn_weights: str) -> None:
        if self.agent is None:
            raise RuntimeError("Cannot load GNN weights when the communication agent is unavailable.")
        weights_path = Path(gnn_weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"GNN weights not found: {weights_path}")
        self._build_agent_networks()
        if self.gnn_type in {"sage", "graphsage"} and hasattr(self.agent.G, "G_model"):
            self.agent.G.G_model.load_weights(str(weights_path))
            if hasattr(self.agent.G, "update_target_network"):
                self.agent.G.update_target_network()
        elif hasattr(self.agent.G, "load_weights"):
            self.agent.G.load_weights(str(weights_path))
        else:
            raise RuntimeError(f"GNN type '{self.gnn_type}' does not expose a load_weights interface.")

    def _force_links_active(self) -> None:
        if hasattr(self.env, "activate_links"):
            self.env.activate_links[:] = False
            cav_indices = np.asarray(getattr(self.env, "cav_indices", np.arange(len(getattr(self.env, "vehicles", [])))), dtype=int)
            if cav_indices.size > 0:
                self.env.activate_links[cav_indices, :] = True
        if hasattr(self.env, "individual_time_interval") and np.size(self.env.individual_time_interval) > 0:
            self.env.individual_time_interval[:] = 1e9
            cav_indices = np.asarray(getattr(self.env, "cav_indices", np.arange(len(getattr(self.env, "vehicles", [])))), dtype=int)
            if cav_indices.size > 0:
                self.env.individual_time_interval[cav_indices, :] = 0.0

    def _select_actions_heuristic(self):
        nveh = len(getattr(self.env, "vehicles", []))
        if nveh == 0:
            empty = np.zeros((0, 3, 2), dtype=np.int32)
            return empty, self._empty_metrics()

        actions = np.zeros((nveh, 3, 2), dtype=np.int32)
        occupancy = np.zeros(int(getattr(self.env, "n_RB", 20)), dtype=np.float32)
        link_order = []
        for i in range(nveh):
            for j in range(3):
                time_left = float(self.env.individual_time_limit[i, j])
                link_order.append((time_left, i, j))
        link_order.sort(key=lambda item: item[0])

        for _, i, j in link_order:
            reward_table, penalty, _ = self.env.Compute_Performance_Reward_Batch(actions, (i, j))
            score = reward_table.copy() + float(penalty)
            score -= 0.03 * occupancy[:, None]
            flat_idx = int(np.argmax(score))
            rb, power_idx = np.unravel_index(flat_idx, score.shape)

            urgency = float(self.env.individual_time_limit[i, j] / max(self.env.V2V_limit, 1e-8))
            if urgency < 0.30:
                power_idx = 0
            elif urgency > 0.75 and power_idx == 0:
                power_idx = 1

            actions[i, j, 0] = int(rb)
            actions[i, j, 1] = int(power_idx)
            occupancy[int(rb)] += 1.0
        return actions, None

    def _select_actions(self, step: int):
        if self.policy_mode == "heuristic" or self.agent is None:
            return self._select_actions_heuristic()

        self.env.n_step = int(step)
        self.agent._ensure_action_buffers()
        self.agent.step = int(step)
        topology_changed = self._topology_changed()
        current_signature = self._env_signature()
        if self._can_reuse_actions(step) and self._last_action_signature == current_signature:
            return self._cached_actions.copy(), None

        self.agent._sync_graph_topology(force=topology_changed)
        s_old_all, indices = self.agent.get_state_all()
        n_links = s_old_all.shape[0]
        if n_links == 0:
            return self.agent.action_all_with_power_training.copy(), self._empty_metrics()

        self.agent.G.features[:n_links, :] = s_old_all[:, :60]

        self.agent._update_channel_reward()
        emb_all = self.agent.forward_embeddings(force=topology_changed)
        batch_states = np.empty((n_links, 114), dtype=np.float32)
        batch_states[:, :32] = emb_all[:n_links]
        batch_states[:, 32:] = s_old_all

        actions_list = self.agent.predict_batch(batch_states, max(1, int(step)), test_ep=True)
        flat_actions = self.agent.action_all_with_power_training.reshape(-1, 2)
        flat_actions[:n_links, 0] = actions_list % self.agent.RB_number
        flat_actions[:n_links, 1] = actions_list // self.agent.RB_number

        actions_out = self.agent.action_all_with_power_training.copy()
        self._cached_actions = actions_out.copy()
        self._last_action_step = int(step)
        self._last_action_signature = current_signature
        return actions_out, None

    def evaluate_step(self, step: int) -> Dict[str, float]:
        self._force_links_active()
        actions, empty_metrics = self._select_actions(step)
        if empty_metrics is not None:
            self.last_metrics = empty_metrics
            return dict(self.last_metrics)

        v2i_rate, fail_percent = self.env.Compute_Performance_Reward_fast_fading_with_power_asyn(actions)
        self.env.Compute_Interference(actions)

        used_rb = np.unique(actions[:, :, 0])
        rb_number = int(getattr(self.agent, "RB_number", getattr(self.env, "n_RB", 20)))
        used_rb = used_rb[(used_rb >= 0) & (used_rb < rb_number)]
        mean_power_index = float(np.mean(actions[:, :, 1])) if actions.size else 0.0

        self.last_metrics = {
            "comm_v2i_rate_total": float(np.sum(v2i_rate)),
            "comm_v2v_success": float(1.0 - fail_percent),
            "comm_used_rb_ratio": float(len(used_rb)) / float(max(1, rb_number)),
            "comm_mean_power_norm": mean_power_index / max(1.0, len(getattr(self.env, "V2V_power_dB_List", [23, 10, 5])) - 1),
            "comm_fail_percent": float(fail_percent),
        }
        return dict(self.last_metrics)
