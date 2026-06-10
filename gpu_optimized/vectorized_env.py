"""VectorizedFormationEnv — N environments managed in a single process."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .accelerated_env import AcceleratedFormationEnv
from .accelerated_human_drivers import AcceleratedHumanDriverController
from .accelerated_shield import AcceleratedSafetyShield
from .config import (
    COMM_AGENT_GRAPH_REFRESH_STEPS,
    COMM_AGENT_SKIP_EMBEDDING_STEPS,
    COMM_DECISION_INTERVAL_STEPS,
    COMM_FASTFADING_UPDATE_INTERVAL_STEPS,
    COMM_METRICS_UPDATE_INTERVAL_STEPS,
    THREAD_POOL_WORKERS,
)


def build_accelerated_env(args, scheduler, out_dir: Path) -> AcceleratedFormationEnv:
    """Mirror of build_env() but returns AcceleratedFormationEnv."""
    resolved_n_up = int(getattr(args, "_resolved_n_up", args.n_up))
    resolved_n_down = int(getattr(args, "_resolved_n_down", args.n_down))
    env = AcceleratedFormationEnv(
        scheduler=scheduler,
        shield=AcceleratedSafetyShield(),
        human_controller=AcceleratedHumanDriverController(),
        n_up=resolved_n_up,
        n_down=resolved_n_down,
        lanes_per_dir=args.lanes,
        spacing=args.spacing,
        height=args.height,
        mpr_cav=args.mpr_cav,
        random_spawn=True,
        spawn_y_min=args.spawn_y_min,
        spawn_y_max=args.spawn_y_max,
        lane_density_jitter=args.lane_density_jitter,
        topology_type=args.topology,
        leader_dynamic=True,
        seed=args.seed,
        v2i_mode="rsu",
        bs_layout="median",
        bs_spacing=250.0,
        communication_enabled=not args.disable_communication,
        communication_gnn_type=args.comm_gnn,
        communication_policy_mode=args.comm_policy,
        communication_dqn_weights=args.comm_dqn_weights or None,
        communication_gnn_weights=args.comm_gnn_weights or None,
        communication_weights_dir=args.comm_weights_dir or None,
        communication_run_dir=str(out_dir / "comm_agent"),
        communication_decision_interval_steps=COMM_DECISION_INTERVAL_STEPS,
        communication_fastfading_update_interval_steps=COMM_FASTFADING_UPDATE_INTERVAL_STEPS,
        communication_metrics_update_interval_steps=COMM_METRICS_UPDATE_INTERVAL_STEPS,
        communication_agent_kwargs={
            "skip_embedding_steps": COMM_AGENT_SKIP_EMBEDDING_STEPS,
            "graph_refresh_steps": COMM_AGENT_GRAPH_REFRESH_STEPS,
        },
    )
    backend = getattr(args, "comm_backend", None)
    if backend and getattr(env, "env", None) is not None:
        try:
            env.env.comm_backend = str(backend)
        except Exception:
            pass
    return env


class VectorizedFormationEnv:
    """Manages N formation environments for batch inference."""

    def __init__(
        self,
        n_envs: int,
        args,
        make_scheduler_fn: Callable,
        sample_mpr_fn: Callable,
        out_dir: Path,
        rng: np.random.Generator,
    ):
        self.n_envs = int(n_envs)
        self.args = args
        self.make_scheduler_fn = make_scheduler_fn
        self.sample_mpr_fn = sample_mpr_fn
        self.out_dir = Path(out_dir)
        self.rng = rng

        self.envs: List[AcceleratedFormationEnv] = []
        self.schedulers: List[Any] = []
        self.episode_mprs: List[float] = []
        self.dones = np.zeros(n_envs, dtype=bool)
        self._pool = ThreadPoolExecutor(max_workers=min(n_envs, THREAD_POOL_WORKERS))

        for i in range(n_envs):
            self._init_env(i)

    def _init_env(self, idx: int) -> None:
        mpr = float(self.sample_mpr_fn(self.args, self.rng))
        import copy
        args_copy = copy.copy(self.args)
        args_copy.mpr_cav = mpr
        scheduler = self.make_scheduler_fn(args_copy)
        env_dir = self.out_dir / f"vec_env_{idx:03d}"
        env_dir.mkdir(parents=True, exist_ok=True)
        env = build_accelerated_env(args_copy, scheduler, env_dir)

        if idx < len(self.envs):
            self.envs[idx] = env
            self.schedulers[idx] = scheduler
            self.episode_mprs[idx] = mpr
        else:
            self.envs.append(env)
            self.schedulers.append(scheduler)
            self.episode_mprs.append(mpr)
        self.dones[idx] = False

    def _reset_env(self, idx: int) -> None:
        mpr = float(self.sample_mpr_fn(self.args, self.rng))
        import copy
        args_copy = copy.copy(self.args)
        args_copy.mpr_cav = mpr
        scheduler = self.make_scheduler_fn(args_copy)

        env = self.envs[idx]
        env.scheduler = scheduler
        env.env_kwargs["mpr_cav"] = mpr
        env.reset()

        self.schedulers[idx] = scheduler
        self.episode_mprs[idx] = mpr
        self.dones[idx] = False

    def collect_states(self) -> Tuple[np.ndarray, List[Dict]]:
        """Gather state vectors from all envs into (N, state_dim) array."""
        states = [env.current_state() for env in self.envs]
        vectors = np.stack([s["vector"] for s in states], axis=0)
        fields_list = [dict(s["fields"]) for s in states]
        return vectors, fields_list

    def collect_states_threaded(self) -> Tuple[np.ndarray, List[Dict]]:
        futures = [self._pool.submit(env.current_state) for env in self.envs]
        states = [future.result() for future in futures]
        vectors = np.stack([s["vector"] for s in states], axis=0)
        fields_list = [dict(s["fields"]) for s in states]
        return vectors, fields_list

    def step_one(self, idx: int, action: str) -> Tuple[Dict, float, bool, Dict]:
        return self.envs[idx].step(action)

    def step_all_threaded(self, actions: List[str]) -> List[Tuple[Dict, float, bool, Dict]]:
        """Run env.step() for all envs using ThreadPoolExecutor."""
        futures = [
            self._pool.submit(self.step_one, i, actions[i])
            for i in range(self.n_envs)
        ]
        return [f.result() for f in futures]

    def step_all_sequential(self, actions: List[str]) -> List[Tuple[Dict, float, bool, Dict]]:
        return [self.step_one(i, actions[i]) for i in range(self.n_envs)]

    def auto_reset(self) -> np.ndarray:
        """Reset done environments with fresh MPR. Returns bool mask of resets."""
        reset_mask = self.dones.copy()
        for i in range(self.n_envs):
            if self.dones[i]:
                self._reset_env(i)
        return reset_mask

    def mark_done(self, idx: int) -> None:
        self.dones[idx] = True

    def close(self) -> None:
        self._pool.shutdown(wait=False)
