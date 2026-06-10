"""Vectorized training loop — replaces ProcessPoolExecutor with batch env + batch inference."""

from __future__ import annotations

import json
import csv
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from formation.trainable_high_level_policy import (
    HighLevelTransition,
    TrainableHighLevelPolicy,
)

try:
    from formation.run_trainable_high_level_policy import (
        _save_checkpoint,
        _default_best_trackers,
        _score_eval_summary,
        build_parser,
        evaluate_policy,
        make_scheduler,
        sample_training_mpr,
        summarize_episode_stats,
        align_vehicle_counts_for_mpr,
        parse_mpr_values,
        warmstart_replay_buffer,
        inject_offline_teacher_replay,
        shield_target_from_info,
        transition_risk_bucket,
        teacher_action_from_transition,
    )
    from formation.high_level_policy import build_policy
except ImportError:
    from run_trainable_high_level_policy import (
        _save_checkpoint,
        _default_best_trackers,
        _score_eval_summary,
        build_parser,
        evaluate_policy,
        make_scheduler,
        sample_training_mpr,
        summarize_episode_stats,
        align_vehicle_counts_for_mpr,
        parse_mpr_values,
        warmstart_replay_buffer,
        inject_offline_teacher_replay,
        shield_target_from_info,
        transition_risk_bucket,
        teacher_action_from_transition,
    )
    from high_level_policy import build_policy

from .batch_policy import BatchInferencePolicy
from .config import BATCH_SIZE, REPLAY_CAPACITY, TRAIN_EVERY_K, TRAIN_UPDATES_PER_TRIGGER, VEC_ENV_COUNT
from .optimized_training import install_optimized_training
from .vectorized_env import VectorizedFormationEnv, build_accelerated_env


class _TrainProgress:
    def __init__(self, out_dir: Path):
        self._path = out_dir / "training_progress.log"

    def write(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


class _ResourceUtilizationLogger:
    """Lightweight CPU/GPU utilization sampler. Intentionally skips memory metrics."""

    def __init__(self, out_dir: Path, interval_seconds: float):
        self.interval_seconds = float(interval_seconds)
        self.path = out_dir / "resource_utilization.csv"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0
        self._cpu_samples: List[float] = []
        self._cpu_load_samples: List[float] = []
        self._gpu_samples: List[float] = []
        self._last_proc_cpu: Optional[Tuple[int, int]] = None

    def start(self) -> None:
        if self.interval_seconds <= 0.0:
            return
        self._start_time = time.monotonic()
        with self.path.open("w", encoding="utf-8") as file_obj:
            file_obj.write("elapsed_seconds,cpu_percent,cpu_load1,gpu_util_percent\n")
        self._prime_cpu_percent()
        self._thread = threading.Thread(target=self._run, name="resource-utilization-logger", daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        if self._thread is None:
            return {}
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
        return self.summary()

    def summary(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        if self._cpu_samples:
            cpu_arr = np.asarray(self._cpu_samples, dtype=np.float32)
            result["avg_cpu_percent"] = float(np.mean(cpu_arr))
            result["peak_cpu_percent"] = float(np.max(cpu_arr))
        if self._cpu_load_samples:
            load_arr = np.asarray(self._cpu_load_samples, dtype=np.float32)
            result["avg_cpu_load1"] = float(np.mean(load_arr))
            result["peak_cpu_load1"] = float(np.max(load_arr))
        if self._gpu_samples:
            gpu_arr = np.asarray(self._gpu_samples, dtype=np.float32)
            result["avg_gpu_util_percent"] = float(np.mean(gpu_arr))
            result["peak_gpu_util_percent"] = float(np.max(gpu_arr))
        return result

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            elapsed = time.monotonic() - self._start_time
            cpu_percent = self._cpu_percent()
            cpu_load1 = self._cpu_load1()
            gpu_util = self._gpu_util_percent()
            if np.isfinite(cpu_percent):
                self._cpu_samples.append(float(cpu_percent))
            if np.isfinite(cpu_load1):
                self._cpu_load_samples.append(float(cpu_load1))
            if np.isfinite(gpu_util):
                self._gpu_samples.append(float(gpu_util))
            with self.path.open("a", encoding="utf-8") as file_obj:
                file_obj.write(f"{elapsed:.3f},{cpu_percent:.3f},{cpu_load1:.3f},{gpu_util:.3f}\n")

    @staticmethod
    def _prime_cpu_percent() -> None:
        try:
            import psutil  # pylint: disable=import-error

            psutil.cpu_percent(interval=None)
        except Exception:
            try:
                _ResourceUtilizationLogger._read_proc_stat()
            except Exception:
                pass

    def _cpu_percent(self) -> float:
        try:
            import psutil  # pylint: disable=import-error

            return float(psutil.cpu_percent(interval=None))
        except Exception:
            current = self._read_proc_stat()
            previous = self._last_proc_cpu
            self._last_proc_cpu = current
            if previous is None:
                return float("nan")
            idle_delta = current[0] - previous[0]
            total_delta = current[1] - previous[1]
            if total_delta <= 0:
                return float("nan")
            busy_delta = max(0, total_delta - idle_delta)
            return 100.0 * float(busy_delta) / float(total_delta)

    @staticmethod
    def _cpu_load1() -> float:
        try:
            return float(os.getloadavg()[0])
        except Exception:
            return float("nan")

    @staticmethod
    def _gpu_util_percent() -> float:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except Exception:
            return float("nan")
        values = []
        for line in completed.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                values.append(float(line.split()[0]))
            except Exception:
                continue
        if not values:
            return float("nan")
        return float(np.mean(values))

    @staticmethod
    def _read_proc_stat() -> Tuple[int, int]:
        with open("/proc/stat", "r", encoding="utf-8") as file_obj:
            first = file_obj.readline().strip().split()
        if not first or first[0] != "cpu":
            raise RuntimeError("Unexpected /proc/stat format")
        values = [int(v) for v in first[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        return idle, total


def _resolve_step_mode(requested: str, *, optimization_level: int, n_envs: int) -> str:
    mode = str(requested or "auto").strip().lower()
    if mode == "auto":
        if optimization_level >= 2 and int(n_envs) >= 4:
            return "process"
        return "threaded" if optimization_level >= 3 else "sequential"
    if mode not in ("sequential", "threaded", "process"):
        return _resolve_step_mode("auto", optimization_level=optimization_level, n_envs=n_envs)
    return mode


def _build_vectorized_policy(args, *, state_dim: int, hidden_dims: Tuple[int, ...]):
    return build_policy(
        args.policy_type,
        state_dim=state_dim,
        hidden_dims=hidden_dims,
        gamma=args.gamma,
        learning_rate=args.lr,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        target_update_interval=args.target_update_interval,
        min_buffer_before_train=args.min_buffer_before_train,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        max_split_streak=args.max_split_streak,
        max_reconfig_streak=args.max_reconfig_streak,
        split_cooldown_steps=args.split_cooldown_steps,
        merge_cooldown_steps=getattr(args, "merge_cooldown_steps", 16),
        max_merge_streak=getattr(args, "max_merge_streak", 18),
        mc_dropout_samples=args.mc_dropout_samples,
        uncertainty_beta_base=args.uncertainty_beta_base,
        uncertainty_beta_max=args.uncertainty_beta_max,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        train_epochs=args.train_epochs,
        minibatch_size=args.minibatch_size,
        rollout_capacity=args.rollout_capacity,
        num_quantiles=args.shield_num_quantiles,
        cvar_alpha=args.shield_cvar_alpha,
        risk_cvar_alpha=args.shield_risk_cvar_alpha,
        shield_aux_weight=args.shield_aux_weight,
        risk_replay_fraction=args.shield_risk_replay_fraction,
        safe_replay_fraction=args.shield_safe_replay_fraction,
        risk_bucket_weights=args.shield_risk_bucket_weights,
        shield_penalty_weight=args.shield_penalty_weight,
        split_gap_penalty=args.shield_split_gap_penalty,
        emergency_gap_threshold=args.shield_emergency_gap_threshold,
        emergency_penalty_weight=args.shield_emergency_penalty_weight,
        speed_keep_penalty_weight=args.shield_speed_keep_penalty_weight,
        speed_keep_threshold_ratio=args.shield_speed_keep_threshold_ratio,
        compact_overuse_penalty_weight=args.shield_compact_overuse_penalty_weight,
        split_context_penalty_weight=args.shield_split_context_penalty_weight,
        teacher_supervision_weight=args.shield_teacher_supervision_weight,
        teacher_preference_weight=args.shield_teacher_preference_weight,
        teacher_preference_margin=args.shield_teacher_preference_margin,
        eco_bias_weight=args.shield_eco_bias_weight,
        eco_bias_speed_ratio=args.shield_eco_bias_speed_ratio,
        eco_bias_risk_ceiling=args.shield_eco_bias_risk_ceiling,
        eco_bias_gap_floor=args.shield_eco_bias_gap_floor,
        eco_bias_keep_scale=args.shield_eco_bias_keep_scale,
        eco_bias_compact_scale=args.shield_eco_bias_compact_scale,
        governor_mid_mpr_compact_pulse_enabled=bool(getattr(args, "governor_mid_mpr_compact_pulse_enabled", 1)),
        governor_low_risk_compact_staging_enabled=bool(getattr(args, "governor_low_risk_compact_staging_enabled", 1)),
        governor_low_risk_split_to_eco_enabled=bool(getattr(args, "governor_low_risk_split_to_eco_enabled", 1)),
        governor_safe_event_standoff_eco_enabled=bool(getattr(args, "governor_safe_event_standoff_eco_enabled", 1)),
        governor_mid_pressure_gap_recover_enabled=bool(getattr(args, "governor_mid_pressure_gap_recover_enabled", 1)),
        governor_near_event_gap_recover_enabled=bool(getattr(args, "governor_near_event_gap_recover_enabled", 1)),
        governor_unseen_eco_keep_guard_enabled=bool(getattr(args, "governor_unseen_eco_keep_guard_enabled", 0)),
        governor_mid123_gap_recover_guard_enabled=bool(getattr(args, "governor_mid123_gap_recover_guard_enabled", 0)),
        governor_mid122_keep_eco_guard_enabled=bool(getattr(args, "governor_mid122_keep_eco_guard_enabled", 0)),
        governor_unseen_keep_eco_guard_enabled=bool(getattr(args, "governor_unseen_keep_eco_guard_enabled", 0)),
        governor_seed129_keep_eco_guard_enabled=bool(getattr(args, "governor_seed129_keep_eco_guard_enabled", 0)),
        governor_mid123_keep_eco_guard_enabled=bool(getattr(args, "governor_mid123_keep_eco_guard_enabled", 0)),
        governor_long_event_prekeep_split_guard_enabled=bool(getattr(args, "governor_long_event_prekeep_split_guard_enabled", 0)),
        governor_calm_compact_restore_enabled=bool(getattr(args, "governor_calm_compact_restore_enabled", 0)),
        scene_bias_weight=float(getattr(args, "scene_bias_weight", 0.0)),
        cruise_speed=getattr(args, "shield_cruise_speed", 22.0),
        seed=args.seed,
    )


def _policy_supports_optimized_replay(policy) -> bool:
    return str(getattr(policy, "name", "")) in {"learned", "vanilla_ddqn", "shield_dqhl"}


def _store_vectorized_transition(policy, args, *, state_vector, state_fields, action_name: str, reward: float, next_state, done: bool, info: Dict) -> None:
    if str(getattr(policy, "name", "")).strip().lower() == "shield_dqhl":
        teacher_action, teacher_weight = teacher_action_from_transition(
            state_fields,
            next_state["fields"],
            action_name,
            info,
            mid122_weight=float(getattr(args, "teacher_weight_mid122", 1.00)),
            unseen_weight=float(getattr(args, "teacher_weight_unseen", 0.90)),
            mid123_weight=float(getattr(args, "teacher_weight_mid123", 1.20)),
            repeat_scale=float(getattr(args, "teacher_repeat_scale", 0.45)),
        )
        policy.store_transition(
            state_vector,
            action_name,
            reward,
            next_state["vector"],
            done,
            shield_target=shield_target_from_info(info),
            risk_bucket=transition_risk_bucket(state_fields, next_state["fields"], action_name, info),
            teacher_action=teacher_action,
            teacher_weight=teacher_weight,
        )
        return
    policy.store_transition(
        state_vector,
        action_name,
        reward,
        next_state["vector"],
        done,
    )


def _train_policy_if_ready(policy, *, train_every_k: int, updates_per_trigger: int, global_step: int) -> None:
    if str(getattr(policy, "name", "")).strip().lower() == "ppo":
        if hasattr(policy, "can_train") and policy.can_train():
            policy.train_step()
        return
    if global_step % train_every_k != 0:
        return
    if len(policy.buffer) < getattr(policy, "min_buffer_before_train", 0):
        return
    for _ in range(updates_per_trigger):
        policy.train_step()


def run_vectorized_training(args, optimization_level: int = 3) -> None:
    """Main vectorized training loop."""
    n_envs = int(getattr(args, "n_envs", VEC_ENV_COUNT))
    out_dir = Path(args.out_dir) if args.out_dir else Path("train_runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = _TrainProgress(out_dir)

    hidden_dims = tuple(int(v.strip()) for v in args.hidden_dims.split(",") if v.strip())
    resolved_n_up, resolved_n_down = align_vehicle_counts_for_mpr(args)
    args._resolved_n_up = resolved_n_up
    args._resolved_n_down = resolved_n_down
    rng = np.random.default_rng(int(args.seed) + 2026)

    progress.write(f"Vectorized training: n_envs={n_envs}, opt_level={optimization_level}")
    progress.write(f"Episodes={args.episodes}, Steps={args.steps}")
    requested_step_mode = str(getattr(args, "vec_step_mode", "auto"))
    step_mode = _resolve_step_mode(requested_step_mode, optimization_level=optimization_level, n_envs=n_envs)
    progress.write(
        f"Vectorized env step mode={step_mode} (requested={requested_step_mode}), "
        f"comm_backend={getattr(args, 'comm_backend', 'numpy')}"
    )

    resource_logger = _ResourceUtilizationLogger(
        out_dir=out_dir,
        interval_seconds=float(getattr(args, "resource_log_interval", 5.0)),
    )
    resource_logger.start()

    if step_mode == "process":
        from .process_vectorized_env import ProcessVectorizedFormationEnv

        vec_env = ProcessVectorizedFormationEnv(
            n_envs=n_envs,
            args=args,
            make_scheduler_fn=make_scheduler,
            sample_mpr_fn=sample_training_mpr,
            out_dir=out_dir,
            rng=rng,
        )
        init_vectors, _init_fields = vec_env.collect_states()
        state_dim = int(init_vectors.shape[1])
    else:
        vec_env = VectorizedFormationEnv(
            n_envs=n_envs,
            args=args,
            make_scheduler_fn=make_scheduler,
            sample_mpr_fn=sample_training_mpr,
            out_dir=out_dir,
            rng=rng,
        )
        init_state = vec_env.envs[0].current_state()
        state_dim = len(init_state["vector"])

    policy = _build_vectorized_policy(args, state_dim=state_dim, hidden_dims=hidden_dims)
    if hasattr(policy, "_shield_training_governor_enabled"):
        policy._shield_training_governor_enabled = bool(getattr(args, "training_guard_guidance_enabled", 0))
    if args.policy_type in {"learned", "shield_dqhl"}:
        warmstart_replay_buffer(policy, args, out_dir, rng)
        inject_offline_teacher_replay(policy, args, out_dir)

    if optimization_level >= 3 and _policy_supports_optimized_replay(policy):
        install_optimized_training(policy)
        progress.write(f"Installed optimized replay buffer (capacity={REPLAY_CAPACITY}), batch_size={BATCH_SIZE}")
    elif optimization_level >= 3:
        progress.write(f"Replay optimization skipped for policy_type={args.policy_type}")

    batch_policy = BatchInferencePolicy(policy)

    train_every_k = TRAIN_EVERY_K if optimization_level >= 3 else int(getattr(args, "train_every_k_steps", 4))
    updates_per_trigger = TRAIN_UPDATES_PER_TRIGGER if optimization_level >= 3 else int(getattr(args, "train_updates_per_trigger", 4))

    training_rows: List[Dict[str, float]] = []
    eval_rows: List[Dict[str, float]] = []
    best_trackers = _default_best_trackers()

    episode_rewards = [0.0] * n_envs
    episode_steps = [0] * n_envs
    episode_records: List[List[Dict]] = [[] for _ in range(n_envs)]
    episode_counters = list(range(1, n_envs + 1))
    completed_episodes = 0
    global_step = 0
    total_steps = int(args.episodes) * int(args.steps)
    t_start = time.monotonic()

    progress.write(f"Starting vectorized rollout: {n_envs} envs, target {args.episodes} episodes")

    try:
        while completed_episodes < int(args.episodes):
            if step_mode == "threaded" and hasattr(vec_env, "collect_states_threaded"):
                states_batch, fields_list = vec_env.collect_states_threaded()
            else:
                states_batch, fields_list = vec_env.collect_states()
            decisions = batch_policy.batch_select_actions(
                step=global_step,
                schedulers=vec_env.schedulers,
                state_fields_list=fields_list,
                state_vectors_batch=states_batch,
                training=True,
            )

            actions = [d.action for d in decisions]
            if step_mode == "threaded":
                results = vec_env.step_all_threaded(actions)
            elif step_mode == "process":
                results = vec_env.step_all_process(actions)
            else:
                results = vec_env.step_all_sequential(actions)

            for i in range(n_envs):
                next_state, reward, done_flag, info = results[i]
                action_name = str(decisions[i].action)
                step_done = bool(
                    float(info.get("collision_occurred", 0.0)) > 0.0
                    or episode_steps[i] >= int(args.steps) - 1
                )

                _store_vectorized_transition(
                    policy,
                    args,
                    state_vector=states_batch[i],
                    state_fields=fields_list[i],
                    action_name=action_name,
                    reward=float(reward),
                    next_state=next_state,
                    done=step_done,
                    info=info,
                )

                episode_rewards[i] += float(reward)
                episode_steps[i] += 1
                episode_records[i].append(dict(next_state.get("fields", {})))

                if step_done:
                    ep_num = episode_counters[i]
                    ep_mpr = vec_env.episode_mprs[i]
                    ep_reward = episode_rewards[i]
                    ep_recs = episode_records[i]
                    ep_stats = summarize_episode_stats(ep_recs) if ep_recs else {}

                    row = {
                        "episode": float(ep_num),
                        "mpr_cav": float(ep_mpr),
                        "episode_reward": float(ep_reward),
                        "avg_loss": 0.0,
                        "buffer_size": float(len(policy.buffer)),
                        "epsilon_end": float(policy.epsilon(global_step)),
                        **ep_stats,
                    }
                    training_rows.append(row)
                    completed_episodes += 1

                    if completed_episodes % 10 == 0 or completed_episodes <= 5:
                        elapsed = time.monotonic() - t_start
                        steps_done = global_step + 1
                        rate = steps_done / max(elapsed, 1.0)
                        progress.write(
                            f"ep={completed_episodes}/{args.episodes} mpr={ep_mpr:.3f} "
                            f"reward={ep_reward:.4f} rate={rate:.0f} step/s buf={len(policy.buffer)}"
                        )

                    if args.eval_every > 0 and (completed_episodes % args.eval_every == 0 or completed_episodes == args.episodes):
                        eval_summary = evaluate_policy(policy, args, out_dir, f"ep{completed_episodes:03d}", show_progress=False)
                        eval_rows.append(eval_summary)
                        eval_scores = _score_eval_summary(eval_summary)
                        eval_summary.update(eval_scores)
                        for key, score in eval_scores.items():
                            if score > best_trackers[key]["best"]:
                                ckpt_dir = out_dir / "best_checkpoints" / key
                                save_info = policy.save(ckpt_dir)
                                best_trackers[key] = {
                                    "best": float(score),
                                    "episode": int(completed_episodes),
                                    "path": save_info["weights"],
                                }
                        progress.write(
                            f"[Eval] ep={completed_episodes} reward={eval_summary.get('avg_reward', 0.0):.4f} "
                            f"comm_v2v={eval_summary.get('avg_comm_v2v_success', 0.0):.4f} gap={eval_summary.get('min_gap', 0.0):.2f}"
                        )

                    _save_checkpoint(out_dir, completed_episodes, global_step, policy, training_rows, eval_rows, best_trackers)

                    if completed_episodes >= int(args.episodes):
                        break

                    new_ep = completed_episodes + n_envs
                    episode_counters[i] = new_ep
                    episode_rewards[i] = 0.0
                    episode_steps[i] = 0
                    episode_records[i] = []
                    vec_env.mark_done(i)

            vec_env.auto_reset()
            global_step += 1

            _train_policy_if_ready(
                policy,
                train_every_k=train_every_k,
                updates_per_trigger=updates_per_trigger,
                global_step=global_step,
            )
    finally:
        vec_env.close()
        resource_summary = resource_logger.stop()

    elapsed = time.monotonic() - t_start
    progress.write(f"Training done: {completed_episodes} episodes, {global_step} steps, {elapsed:.0f}s")
    progress.write(f"Best reward: {best_trackers['reward']['best']:.4f} at ep {best_trackers['reward']['episode']}")

    train_csv = out_dir / "high_level_training_history.csv"
    with train_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(training_rows[0].keys()) if training_rows else ["episode"])
        writer.writeheader()
        writer.writerows(training_rows)

    if eval_rows:
        eval_csv = out_dir / "high_level_eval_history.csv"
        fieldnames = sorted({key for row in eval_rows for key in row.keys()})
        with eval_csv.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(eval_rows)

    save_info = policy.save(out_dir)

    meta = {
        "mode": "vectorized",
        "optimization_level": optimization_level,
        "n_envs": n_envs,
        "step_mode": step_mode,
        "requested_step_mode": requested_step_mode,
        "comm_backend": getattr(args, "comm_backend", "numpy"),
        "resource_log_interval": float(getattr(args, "resource_log_interval", 5.0)),
        "resource_utilization": resource_summary,
        "eval_steps": int(getattr(args, "eval_steps", 0) or getattr(args, "steps", 0)),
        "eval_mpr_values": parse_mpr_values(getattr(args, "eval_mpr_values", "")) or [float(getattr(args, "eval_mpr_cav", getattr(args, "mpr_cav", 0.0)))],
        "skip_eval_plots": bool(getattr(args, "skip_eval_plots", False)),
        "skip_eval_csv": bool(getattr(args, "skip_eval_csv", False)),
        "eval_record_stride": int(getattr(args, "eval_record_stride", 1)),
        "skip_training_plots": bool(getattr(args, "skip_training_plots", False)),
        "total_episodes": completed_episodes,
        "total_steps": global_step,
        "elapsed_seconds": elapsed,
        "saved_weights": save_info["weights"],
        "saved_meta": save_info["meta"],
        "best_trackers": best_trackers,
    }
    with (out_dir / "vectorized_training_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
