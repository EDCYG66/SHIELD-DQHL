"""Process-backed vectorized environments for CPU-bound rollout."""

from __future__ import annotations

import copy
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


def _worker_loop(conn, args, env_idx: int, out_dir: str, make_scheduler_fn: Callable, seed_offset: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    from .vectorized_env import build_accelerated_env

    rng = np.random.default_rng(int(getattr(args, "seed", 123)) + int(seed_offset))
    env = None
    scheduler = None
    episode_mpr = float(getattr(args, "mpr_cav", 0.0))

    def sample_mpr() -> float:
        raw = str(getattr(args, "train_mpr_values", "") or "")
        values = []
        for item in raw.split(","):
            item = item.strip()
            if item:
                values.append(float(np.clip(float(item), 0.0, 1.0)))
        if not values:
            return float(np.clip(float(getattr(args, "mpr_cav", 0.0)), 0.0, 1.0))
        mid = [v for v in values if 0.25 <= v <= 0.45]
        focus = float(getattr(args, "mid_mpr_focus_ratio", 0.0))
        pool = mid if mid and rng.random() < focus else values
        return float(rng.choice(pool))

    def make_env():
        nonlocal env, scheduler, episode_mpr
        args_copy = copy.copy(args)
        episode_mpr = sample_mpr()
        args_copy.mpr_cav = episode_mpr
        scheduler = make_scheduler_fn(args_copy)
        env_dir = Path(out_dir) / f"proc_env_{env_idx:03d}"
        env_dir.mkdir(parents=True, exist_ok=True)
        env = build_accelerated_env(args_copy, scheduler, env_dir)

    make_env()
    while True:
        cmd, payload = conn.recv()
        if cmd == "state":
            state = env.current_state()
            conn.send((state["vector"], dict(state["fields"]), episode_mpr))
        elif cmd == "step":
            next_state, reward, done, info = env.step(str(payload))
            conn.send((next_state["vector"], dict(next_state["fields"]), float(reward), bool(done), info))
        elif cmd == "reset":
            make_env()
            state = env.current_state()
            conn.send((state["vector"], dict(state["fields"]), episode_mpr))
        elif cmd == "close":
            conn.close()
            break
        else:
            raise ValueError(f"Unknown worker command: {cmd}")


class ProcessVectorizedFormationEnv:
    """N rollout environments hosted in child processes."""

    def __init__(
        self,
        n_envs: int,
        args,
        make_scheduler_fn: Callable,
        sample_mpr_fn: Callable,
        out_dir: Path,
        rng: np.random.Generator,
    ):
        del sample_mpr_fn, rng
        self.n_envs = int(n_envs)
        self.args = args
        self.make_scheduler_fn = make_scheduler_fn
        self.out_dir = Path(out_dir)
        self.dones = np.zeros(self.n_envs, dtype=bool)
        self.schedulers: List[Any] = [None] * self.n_envs
        self.episode_mprs: List[float] = [float(getattr(args, "mpr_cav", 0.0))] * self.n_envs
        self._ctx = mp.get_context("spawn")
        self._parents = []
        self._processes = []
        for idx in range(self.n_envs):
            parent, child = self._ctx.Pipe()
            proc = self._ctx.Process(
                target=_worker_loop,
                args=(child, args, idx, str(self.out_dir), make_scheduler_fn, 2026 + idx),
                daemon=True,
            )
            proc.start()
            child.close()
            self._parents.append(parent)
            self._processes.append(proc)
        self.collect_states()

    def _make_parent_scheduler(self, episode_mpr: float):
        args_copy = copy.copy(self.args)
        args_copy.mpr_cav = float(episode_mpr)
        return self.make_scheduler_fn(args_copy)

    def collect_states(self) -> Tuple[np.ndarray, List[Dict]]:
        for parent in self._parents:
            parent.send(("state", None))
        vectors = []
        fields_list = []
        for idx, parent in enumerate(self._parents):
            vector, fields, episode_mpr = parent.recv()
            vectors.append(vector)
            fields_list.append(fields)
            self.episode_mprs[idx] = float(episode_mpr)
            if self.schedulers[idx] is None:
                self.schedulers[idx] = self._make_parent_scheduler(float(episode_mpr))
        return np.stack(vectors, axis=0), fields_list

    def step_all_process(self, actions: List[str]) -> List[Tuple[Dict, float, bool, Dict]]:
        for idx, parent in enumerate(self._parents):
            parent.send(("step", actions[idx]))
        results = []
        for parent in self._parents:
            vector, fields, reward, done, info = parent.recv()
            results.append(({"vector": vector, "fields": fields}, reward, done, info))
        return results

    def auto_reset(self) -> np.ndarray:
        reset_mask = self.dones.copy()
        for idx, parent in enumerate(self._parents):
            if self.dones[idx]:
                parent.send(("reset", None))
        for idx, parent in enumerate(self._parents):
            if self.dones[idx]:
                vector, fields, episode_mpr = parent.recv()
                del vector, fields
                self.episode_mprs[idx] = float(episode_mpr)
                self.schedulers[idx] = self._make_parent_scheduler(float(episode_mpr))
                self.dones[idx] = False
        return reset_mask

    def mark_done(self, idx: int) -> None:
        self.dones[idx] = True

    def close(self) -> None:
        for parent in self._parents:
            try:
                parent.send(("close", None))
            except Exception:
                pass
        for proc in self._processes:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
