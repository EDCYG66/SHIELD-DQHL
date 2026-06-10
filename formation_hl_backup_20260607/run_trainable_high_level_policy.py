"""Train a high-level platoon reconfiguration policy on the formation environment."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import multiprocessing as mp
import os
import pickle
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

# Reduce TensorFlow C++ info/debug log noise when training.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

try:
    from .events import EventScheduler, build_bottleneck_event
    from .formation_env import FormationExperimentEnv
    from .high_level_policy import build_policy
    from .mpr_utils import resolve_exact_vehicle_counts
    from .platoon_metrics import PlatoonMetricsTracker
    from .progress_utils import create_progress, progress_range
    from .trainable_high_level_policy import HighLevelTransition, TrainableHighLevelPolicy, TRAINABLE_ACTIONS
except ImportError:  # pragma: no cover
    from events import EventScheduler, build_bottleneck_event
    from formation_env import FormationExperimentEnv
    from high_level_policy import build_policy
    from mpr_utils import resolve_exact_vehicle_counts
    from platoon_metrics import PlatoonMetricsTracker
    from progress_utils import create_progress, progress_range
    from trainable_high_level_policy import HighLevelTransition, TrainableHighLevelPolicy, TRAINABLE_ACTIONS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train high-level formation policy")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-up", type=int, default=12)
    parser.add_argument("--n-down", type=int, default=12)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--spacing", type=float, default=20.0)
    parser.add_argument("--height", type=float, default=1400.0)
    parser.add_argument("--mpr-cav", type=float, default=0.375)
    parser.add_argument(
        "--train-mpr-values",
        type=str,
        default="0.20,0.30,0.35,0.375,0.40,0.50,0.60",
        help="Comma-separated MPR values sampled during training. Empty string uses --mpr-cav only.",
    )
    parser.add_argument(
        "--mid-mpr-focus-ratio",
        type=float,
        default=0.55,
        help="Sampling probability assigned to MPR values in [0.25, 0.45].",
    )
    parser.add_argument(
        "--eval-mpr-cav",
        type=float,
        default=0.375,
        help="MPR used for periodic evaluation during training.",
    )
    parser.add_argument(
        "--eval-mpr-values",
        type=str,
        default="0.35,0.375,0.40",
        help="Comma-separated MPR values used for periodic evaluation. Empty string falls back to --eval-mpr-cav.",
    )
    parser.add_argument("--topology", type=str, default="star", choices=["star", "tree"])
    parser.add_argument("--event-start", type=int, default=120)
    parser.add_argument("--event-duration", type=int, default=260)
    parser.add_argument("--event-center", type=float, default=620.0)
    parser.add_argument("--event-length", type=float, default=180.0)
    parser.add_argument("--event-speed-scale", type=float, default=0.40)
    parser.add_argument("--spawn-y-min", type=float, default=0.0)
    parser.add_argument("--spawn-y-max", type=float, default=520.0)
    parser.add_argument("--lane-density-jitter", type=float, default=0.35)
    parser.add_argument("--comm-gnn", type=str, default="gat", choices=["gat", "gatclassic", "sage", "fc"])
    parser.add_argument("--comm-policy", type=str, default="auto", choices=["auto", "agent", "heuristic"])
    parser.add_argument("--comm-dqn-weights", type=str, default="")
    parser.add_argument("--comm-gnn-weights", type=str, default="")
    parser.add_argument("--comm-weights-dir", type=str, default="")
    parser.add_argument("--disable-communication", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=20000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=str, default="128,96")
    parser.add_argument("--policy-type", type=str, default="learned", choices=["learned", "vanilla_ddqn", "ppo"])
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--min-buffer-before-train", type=int, default=256)
    parser.add_argument(
        "--train-every-k-steps",
        type=int,
        default=1,
        help="Trigger replay-based DQN updates every K collected transitions.",
    )
    parser.add_argument(
        "--train-updates-per-trigger",
        type=int,
        default=1,
        help="Number of replay updates to run whenever the DQN trigger fires.",
    )
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=6000)
    parser.add_argument("--mc-dropout-samples", type=int, default=10, help="Number of MC Dropout forward passes for uncertainty estimation (0=disable).")
    parser.add_argument("--uncertainty-beta-base", type=float, default=0.30, help="Base risk-aversion coefficient for uncertainty-aware action selection.")
    parser.add_argument("--uncertainty-beta-max", type=float, default=1.20, help="Max risk-aversion coefficient when communication quality is worst.")
    parser.add_argument("--max-split-streak", type=int, default=24)
    parser.add_argument("--max-reconfig-streak", type=int, default=36)
    parser.add_argument("--split-cooldown-steps", type=int, default=18)
    parser.add_argument("--merge-cooldown-steps", type=int, default=16)
    parser.add_argument("--max-merge-streak", type=int, default=18)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--rollout-capacity", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=0,
        help="Steps per evaluation rollout. 0 uses --steps.",
    )
    parser.add_argument(
        "--skip-eval-plots",
        action="store_true",
        help="Skip per-evaluation PNG/PDF plot export and keep summary outputs only.",
    )
    parser.add_argument(
        "--skip-eval-csv",
        action="store_true",
        help="Skip per-step evaluation CSV export.",
    )
    parser.add_argument(
        "--eval-record-stride",
        type=int,
        default=1,
        help="Write every Nth evaluation step to CSV. Summary metrics still use every step.",
    )
    parser.add_argument(
        "--skip-training-plots",
        action="store_true",
        help="Skip final training PNG/PDF plot export.",
    )
    parser.add_argument("--warmstart-heuristic-episodes", type=int, default=8)
    parser.add_argument(
        "--parallel-env-workers",
        type=int,
        default=1,
        help="Number of rollout worker processes for high-level training. "
             "Only enabled for learned policy with communication disabled or comm-policy=heuristic.",
    )
    parser.add_argument(
        "--allow-agent-parallel-rollout",
        action="store_true",
        help="Allow parallel rollout even when comm-policy=agent. "
             "Workers are pinned to CPU to avoid GPU contention with the learner.",
    )
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument(
        "--resume-dir",
        type=str,
        default="",
        help="Path to a previous training output directory to resume from. "
             "Loads checkpoint, weights, replay buffer and continues training.",
    )
    return parser


def parse_mpr_values(raw: str) -> List[float]:
    values: List[float] = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(np.clip(float(item), 0.0, 1.0)))
    return values


def sample_training_mpr(args, rng: np.random.Generator) -> float:
    values = parse_mpr_values(getattr(args, "train_mpr_values", ""))
    if not values:
        return float(np.clip(float(args.mpr_cav), 0.0, 1.0))
    values_arr = np.asarray(values, dtype=np.float64)
    focus_mask = (values_arr >= 0.25) & (values_arr <= 0.45)
    if bool(np.any(focus_mask)) and bool(np.any(~focus_mask)):
        focus_ratio = float(np.clip(getattr(args, "mid_mpr_focus_ratio", 0.75), 0.0, 1.0))
        probs = np.zeros_like(values_arr, dtype=np.float64)
        probs[focus_mask] = focus_ratio / float(np.sum(focus_mask))
        probs[~focus_mask] = (1.0 - focus_ratio) / float(np.sum(~focus_mask))
    else:
        probs = np.ones_like(values_arr, dtype=np.float64) / float(len(values_arr))
    return float(rng.choice(values_arr, p=probs))


def align_vehicle_counts_for_mpr(args) -> tuple[int, int]:
    mpr_values = parse_mpr_values(getattr(args, "train_mpr_values", ""))
    mpr_values.extend(resolve_eval_mpr_values(args))
    if not mpr_values:
        mpr_values = [float(np.clip(float(args.mpr_cav), 0.0, 1.0))]
    n_up, n_down, _ = resolve_exact_vehicle_counts(args.n_up, args.n_down, mpr_values)
    return int(n_up), int(n_down)


def make_scheduler(args) -> EventScheduler:
    return EventScheduler([
        build_bottleneck_event(
            start_step=args.event_start,
            duration=args.event_duration,
            y_center=args.event_center,
            zone_length=args.event_length,
            speed_limit_ratio=args.event_speed_scale,
            blocked_lanes={"u": (1,), "d": (1,)},
        )
    ])


def build_env(args, scheduler: EventScheduler, out_dir: Path) -> FormationExperimentEnv:
    resolved_n_up = int(getattr(args, "_resolved_n_up", args.n_up))
    resolved_n_down = int(getattr(args, "_resolved_n_down", args.n_down))
    return FormationExperimentEnv(
        scheduler=scheduler,
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
    )


def _args_to_worker_payload(args) -> Dict[str, Any]:
    payload = {}
    for key, value in vars(args).items():
        if key.startswith("_") and key not in {"_resolved_n_up", "_resolved_n_down"}:
            continue
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def _worker_args_namespace(args_payload: Dict[str, Any]):
    return SimpleNamespace(**copy.deepcopy(args_payload))


@contextmanager
def _temporary_env_var(key: str, value: Optional[str]):
    previous = os.environ.get(key)
    try:
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _parallel_rollout_worker_initializer() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    try:
        from tf_runtime import configure_tensorflow_runtime
        configure_tensorflow_runtime()
        import tensorflow as tf  # pylint: disable=import-error
        configure_tensorflow_runtime(tf)
    except Exception:
        return
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def _policy_snapshot(policy: TrainableHighLevelPolicy) -> Dict[str, Any]:
    return {
        "state_dim": int(policy.state_dim or 0),
        "hidden_dims": tuple(int(v) for v in policy.hidden_dims),
        "gamma": float(policy.gamma),
        "learning_rate": float(policy.learning_rate),
        "replay_size": int(policy.buffer.capacity),
        "batch_size": int(policy.batch_size),
        "target_update_interval": int(policy.target_update_interval),
        "min_buffer_before_train": int(policy.min_buffer_before_train),
        "epsilon_start": float(policy.epsilon_start),
        "epsilon_end": float(policy.epsilon_end),
        "epsilon_decay_steps": int(policy.epsilon_decay_steps),
        "max_split_streak": int(policy.max_split_streak),
        "max_reconfig_streak": int(policy.max_reconfig_streak),
        "split_cooldown_steps": int(policy.split_cooldown_steps),
        "merge_cooldown_steps": int(policy.merge_cooldown_steps),
        "emergency_cooldown_steps": int(policy.emergency_cooldown_steps),
        "max_merge_streak": int(policy.max_merge_streak),
        "max_emergency_streak": int(policy.max_emergency_streak),
        "seed": int(policy.seed),
        "weights": policy.model.get_weights() if getattr(policy, "model", None) is not None else None,
    }


def _load_policy_from_snapshot(snapshot: Dict[str, Any]) -> TrainableHighLevelPolicy:
    policy = TrainableHighLevelPolicy(
        state_dim=int(snapshot["state_dim"]),
        hidden_dims=tuple(int(v) for v in snapshot["hidden_dims"]),
        gamma=float(snapshot["gamma"]),
        learning_rate=float(snapshot["learning_rate"]),
        replay_size=int(snapshot["replay_size"]),
        batch_size=int(snapshot["batch_size"]),
        target_update_interval=int(snapshot["target_update_interval"]),
        min_buffer_before_train=int(snapshot["min_buffer_before_train"]),
        epsilon_start=float(snapshot["epsilon_start"]),
        epsilon_end=float(snapshot["epsilon_end"]),
        epsilon_decay_steps=int(snapshot["epsilon_decay_steps"]),
        max_split_streak=int(snapshot["max_split_streak"]),
        max_reconfig_streak=int(snapshot["max_reconfig_streak"]),
        split_cooldown_steps=int(snapshot["split_cooldown_steps"]),
        merge_cooldown_steps=int(snapshot.get("merge_cooldown_steps", 16)),
        emergency_cooldown_steps=int(snapshot["emergency_cooldown_steps"]),
        max_merge_streak=int(snapshot.get("max_merge_streak", 18)),
        max_emergency_streak=int(snapshot["max_emergency_streak"]),
        seed=int(snapshot["seed"]),
    )
    weights = snapshot.get("weights")
    if weights is not None and policy.model is not None:
        policy.model.set_weights(weights)
        policy.target_model.set_weights(weights)
    return policy


def _rollout_episode(
    policy: TrainableHighLevelPolicy,
    env: FormationExperimentEnv,
    scheduler: EventScheduler,
    *,
    episode: int,
    args,
) -> Tuple[List[HighLevelTransition], List[Dict[str, float]], float]:
    policy.reset()
    state = env.current_state()
    ep_reward = 0.0
    episode_records: List[Dict[str, float]] = []
    transitions: List[HighLevelTransition] = []
    for step in range(int(args.steps)):
        global_step = (int(episode) - 1) * int(args.steps) + int(step)
        decision = policy.select_action(
            step=global_step,
            scheduler=scheduler,
            state_fields=state["fields"],
            state_vector=state["vector"],
            training=True,
            comm_metrics=state.get("comm_metrics"),
        )
        next_state, reward, _, info = env.step(decision.action)
        done = bool(float(info.get("collision_occurred", 0.0)) > 0.0 or (step == int(args.steps) - 1))
        transitions.append(
            HighLevelTransition(
                state=np.asarray(state["vector"], dtype=np.float32).copy(),
                action=policy.action_index(decision.action),
                reward=float(reward),
                next_state=np.asarray(next_state["vector"], dtype=np.float32).copy(),
                done=float(done),
            )
        )
        ep_reward += float(reward)
        episode_records.append(dict(next_state["fields"]))
        state = next_state
        if done:
            break
    return transitions, episode_records, float(ep_reward)


def _rollout_worker_episode(
    episode: int,
    args_payload: Dict[str, Any],
    policy_snapshot: Dict[str, Any],
    episode_mpr: float,
    run_dir: str,
) -> Dict[str, Any]:
    args = _worker_args_namespace(args_payload)
    args.mpr_cav = float(episode_mpr)
    args._resolved_n_up = int(getattr(args, "_resolved_n_up", args.n_up))
    args._resolved_n_down = int(getattr(args, "_resolved_n_down", args.n_down))
    scheduler = make_scheduler(args)
    env = build_env(args, scheduler, Path(run_dir))
    policy = _load_policy_from_snapshot(policy_snapshot)
    transitions, episode_records, ep_reward = _rollout_episode(
        policy,
        env,
        scheduler,
        episode=int(episode),
        args=args,
    )
    return {
        "episode": int(episode),
        "episode_mpr": float(episode_mpr),
        "episode_reward": float(ep_reward),
        "episode_records": episode_records,
        "transitions": transitions,
        "action_distribution": policy.action_distribution(),
        "switch_count": int(policy.switch_count),
    }


def _schedule_parallel_episode(
    executor: ProcessPoolExecutor,
    *,
    episode: int,
    episode_mpr: float,
    args_payload: Dict[str, Any],
    policy_snapshot: Dict[str, Any],
    out_dir: Path,
):
    future = executor.submit(
        _rollout_worker_episode,
        int(episode),
        args_payload,
        policy_snapshot,
        float(episode_mpr),
        str(out_dir / f"episode_{episode:03d}"),
    )
    return future


def _use_parallel_rollout(args) -> bool:
    workers = int(max(1, getattr(args, "parallel_env_workers", 1)))
    if workers <= 1:
        return False
    if str(getattr(args, "policy_type", "learned")).strip().lower() != "learned":
        return False
    if bool(getattr(args, "disable_communication", False)):
        return True
    comm_policy = str(getattr(args, "comm_policy", "auto")).strip().lower()
    if comm_policy == "heuristic":
        return True
    if comm_policy == "agent" and bool(getattr(args, "allow_agent_parallel_rollout", False)):
        return True
    return False


def summarize_episode_stats(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {
            "avg_comm_v2i_rate_total": 0.0,
            "avg_comm_v2v_success": 0.0,
            "avg_comm_fail_percent": 0.0,
            "avg_min_gap": 0.0,
            "avg_mean_speed": 0.0,
            "avg_platoon_rate": 0.0,
            "avg_mean_platoon_length": 0.0,
            "peak_max_platoon_length": 0.0,
            "avg_event_pressure": 0.0,
            "avg_blocked_lane_ratio": 0.0,
            "avg_event_zone_vehicle_count": 0.0,
            "avg_blocked_lane_vehicle_count": 0.0,
            "avg_passed_event_ratio": 0.0,
            "final_passed_event_ratio": 0.0,
            "avg_target_speed_cmd": 0.0,
            "avg_desired_gap_cmd": 0.0,
            "avg_active_lane_change_cmds": 0.0,
            "avg_topology_switch_indicator": 0.0,
        }
    return {
        "avg_comm_v2i_rate_total": float(np.mean([row.get("comm_v2i_rate_total", 0.0) for row in records])),
        "avg_comm_v2v_success": float(np.mean([row.get("comm_v2v_success", 0.0) for row in records])),
        "avg_comm_fail_percent": float(np.mean([row.get("comm_fail_percent", 0.0) for row in records])),
        "avg_min_gap": float(np.mean([
            min(row.get("min_gap_up", 0.0), row.get("min_gap_down", 0.0)) for row in records
        ])),
        "avg_mean_speed": float(np.mean([
            0.5 * (row.get("mean_speed_up", 0.0) + row.get("mean_speed_down", 0.0)) for row in records
        ])),
        "avg_platoon_rate": float(np.mean([row.get("platoon_rate", 0.0) for row in records])),
        "avg_mean_platoon_length": float(np.mean([row.get("mean_platoon_length", 0.0) for row in records])),
        "peak_max_platoon_length": float(np.max([row.get("max_platoon_length", 0.0) for row in records])),
        "avg_event_pressure": float(np.mean([
            0.5 * (row.get("event_pressure_up", 0.0) + row.get("event_pressure_down", 0.0)) for row in records
        ])),
        "avg_blocked_lane_ratio": float(np.mean([
            max(row.get("blocked_lane_ratio_up", 0.0), row.get("blocked_lane_ratio_down", 0.0)) for row in records
        ])),
        "avg_event_zone_vehicle_count": float(np.mean([row.get("event_zone_vehicle_count", 0.0) for row in records])),
        "avg_blocked_lane_vehicle_count": float(np.mean([row.get("blocked_lane_vehicle_count", 0.0) for row in records])),
        "avg_passed_event_ratio": float(np.mean([
            0.5 * (row.get("passed_event_ratio_up", 0.0) + row.get("passed_event_ratio_down", 0.0)) for row in records
        ])),
        "final_passed_event_ratio": float(
            0.5 * (records[-1].get("passed_event_ratio_up", 0.0) + records[-1].get("passed_event_ratio_down", 0.0))
        ),
        "avg_target_speed_cmd": float(np.mean([row.get("avg_target_speed_cmd", 0.0) for row in records])),
        "avg_desired_gap_cmd": float(np.mean([row.get("avg_desired_gap_cmd", 0.0) for row in records])),
        "avg_active_lane_change_cmds": float(np.mean([row.get("active_lane_change_cmds", 0.0) for row in records])),
        "avg_topology_switch_indicator": float(np.mean([row.get("topology_switch_indicator", 0.0) for row in records])),
    }


def resolve_eval_mpr_values(args) -> List[float]:
    values = parse_mpr_values(getattr(args, "eval_mpr_values", ""))
    if values:
        return values
    return [float(np.clip(float(getattr(args, "eval_mpr_cav", args.mpr_cav)), 0.0, 1.0))]


def aggregate_eval_summaries(summaries: List[Dict[str, float]], episode_tag: str, eval_mpr_values: List[float]) -> Dict[str, float]:
    if not summaries:
        return {
            "episode_tag": episode_tag,
            "eval_mpr_cav": -1.0,
            "eval_mpr_count": 0.0,
        }
    if len(summaries) == 1:
        summary = dict(summaries[0])
        summary["eval_mpr_count"] = 1.0
        return summary

    numeric_keys = sorted({
        key
        for row in summaries
        for key, value in row.items()
        if isinstance(value, (int, float, np.floating))
    })
    aggregate: Dict[str, float] = {}
    for key in numeric_keys:
        values = [float(row.get(key, 0.0)) for row in summaries]
        if key in {"worst_min_gap_up", "worst_min_gap_down", "avg_shield_cbf_min_barrier", "avg_shield_cbf_lane_barrier"}:
            aggregate[key] = float(np.min(values))
        elif key in {"has_collision", "peak_collision_pairs_active", "peak_event_zone_vehicle_count", "peak_blocked_lane_vehicle_count"}:
            aggregate[key] = float(np.max(values))
        else:
            aggregate[key] = float(np.mean(values))
    aggregate["episode_tag"] = episode_tag
    aggregate["eval_mpr_count"] = float(len(eval_mpr_values))
    aggregate["eval_mpr_cav"] = float(np.mean(eval_mpr_values))
    aggregate["eval_mpr_values_mean"] = float(np.mean(eval_mpr_values))
    aggregate["eval_mpr_values_min"] = float(np.min(eval_mpr_values))
    aggregate["eval_mpr_values_max"] = float(np.max(eval_mpr_values))
    return aggregate


def evaluate_policy(
    policy: TrainableHighLevelPolicy,
    args,
    out_dir: Path,
    episode_tag: str,
    *,
    show_progress: bool = False,
) -> Dict[str, float]:
    previous_mpr = float(args.mpr_cav)
    eval_mpr_values = resolve_eval_mpr_values(args)
    eval_summaries: List[Dict[str, float]] = []
    eval_steps = int(getattr(args, "eval_steps", 0) or args.steps)
    eval_steps = max(1, eval_steps)
    save_eval_plots = not bool(getattr(args, "skip_eval_plots", False))
    save_eval_csv = not bool(getattr(args, "skip_eval_csv", False))
    eval_record_stride = max(1, int(getattr(args, "eval_record_stride", 1)))

    for eval_idx, eval_mpr in enumerate(eval_mpr_values, start=1):
        args.mpr_cav = float(np.clip(float(eval_mpr), 0.0, 1.0))
        scheduler = make_scheduler(args)
        suffix = f"eval_{episode_tag}_mpr_{int(round(1000.0 * args.mpr_cav)):03d}"
        env = build_env(args, scheduler, out_dir / suffix)
        policy.reset()
        metrics = PlatoonMetricsTracker()
        state = env.current_state()
        step_bar = progress_range(
            eval_steps,
            desc=f"Eval {episode_tag} ({eval_idx}/{len(eval_mpr_values)})",
            unit="step",
            leave=False,
            disable=not show_progress,
        )
        for step in step_bar:
            decision = policy.select_action(step=step, scheduler=scheduler, state_fields=state["fields"], state_vector=state["vector"], training=False, comm_metrics=state.get("comm_metrics"))
            next_state, reward, _, info = env.step(decision.action)
            info["policy_score"] = float(decision.score)
            metrics.update(step=step, state_fields=next_state["fields"], info=info, action=decision.action, reward=reward)
            state = next_state
            if step == 0 or (step + 1) == eval_steps or (step + 1) % max(1, eval_steps // 8) == 0:
                step_bar.set_postfix(
                    mpr=f"{args.mpr_cav:.3f}",
                    reward=f"{reward:.3f}",
                    v2v=f"{next_state['fields'].get('comm_v2v_success', 0.0):.3f}",
                    gap=f"{min(next_state['fields'].get('min_gap_up', 0.0), next_state['fields'].get('min_gap_down', 0.0)):.1f}",
                )
        step_bar.close()
        summary = metrics.save(
            out_dir / suffix,
            save_plots=save_eval_plots,
            save_csv=save_eval_csv,
            record_stride=eval_record_stride,
        )
        summary["episode_tag"] = episode_tag
        summary["eval_mpr_cav"] = float(args.mpr_cav)
        eval_summaries.append(summary)

    args.mpr_cav = previous_mpr
    return aggregate_eval_summaries(eval_summaries, episode_tag, eval_mpr_values)


def warmstart_replay_buffer(policy, args, out_dir: Path, rng: np.random.Generator) -> None:
    warmstart_eps = int(max(0, getattr(args, "warmstart_heuristic_episodes", 0)))
    if warmstart_eps <= 0:
        return
    heuristic_policy = build_policy("heuristic")
    for warm_idx in range(1, warmstart_eps + 1):
        episode_mpr = sample_training_mpr(args, rng)
        previous_mpr = float(args.mpr_cav)
        args.mpr_cav = float(episode_mpr)
        scheduler = make_scheduler(args)
        env = build_env(args, scheduler, out_dir / f"warmstart_{warm_idx:03d}")
        state = env.current_state()
        for step in range(args.steps):
            decision = heuristic_policy.select_action(
                step=step,
                scheduler=scheduler,
                state_fields=state["fields"],
                state_vector=state["vector"],
                training=False,
            )
            next_state, reward, done, _ = env.step(decision.action)
            done = bool(done or (step == args.steps - 1))
            policy.store_transition(state["vector"], decision.action, reward, next_state["vector"], done)
            state = next_state
            if done:
                break
        args.mpr_cav = previous_mpr
    print(f"[WarmStart] filled replay buffer with {warmstart_eps} heuristic episodes, buffer={len(policy.buffer)}")


def _train_from_episode_transitions(
    policy: TrainableHighLevelPolicy,
    transitions: List[HighLevelTransition],
    *,
    train_every_k_steps: int = 1,
    train_updates_per_trigger: int = 1,
) -> List[float]:
    loss_values: List[float] = []
    trigger_every = max(1, int(train_every_k_steps))
    updates_per_trigger = max(1, int(train_updates_per_trigger))
    for step_idx, transition in enumerate(transitions, start=1):
        policy.buffer.buffer.append(transition)
        if step_idx % trigger_every != 0:
            continue
        for _ in range(updates_per_trigger):
            loss = policy.train_step()
            if loss is not None:
                loss_values.append(float(loss))
    if transitions and (len(transitions) % trigger_every) != 0:
        for _ in range(updates_per_trigger):
            loss = policy.train_step()
            if loss is not None:
                loss_values.append(float(loss))
    return loss_values


def _episode_action_stats(transitions: List[HighLevelTransition]) -> Tuple[Dict[str, float], int]:
    counts = {name: 0 for name in TRAINABLE_ACTIONS}
    switches = 0
    prev_action: Optional[int] = None
    for transition in transitions:
        action_idx = int(transition.action)
        action_name = TRAINABLE_ACTIONS[action_idx]
        counts[action_name] += 1
        if prev_action is not None and prev_action != action_idx:
            switches += 1
        prev_action = action_idx
    total = max(1, len(transitions))
    dist = {name: float(count) / float(total) for name, count in counts.items()}
    return dist, int(switches)


def _sync_policy_action_history(
    policy: TrainableHighLevelPolicy,
    transitions: List[HighLevelTransition],
    *,
    episode: int,
    steps_per_episode: int,
) -> None:
    base_step = max(0, (int(episode) - 1) * int(steps_per_episode))
    for offset, transition in enumerate(transitions):
        policy._record_action(base_step + int(offset), int(transition.action))


def _save_checkpoint(
    out_dir: Path,
    episode: int,
    global_step: int,
    policy: TrainableHighLevelPolicy,
    training_rows: List[Dict[str, float]],
    eval_rows: List[Dict[str, float]],
    best_trackers: Dict[str, Dict],
) -> None:
    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "episode": int(episode),
        "global_step": int(global_step),
        "switch_count": int(policy.switch_count),
        "training_rows": training_rows,
        "eval_rows": eval_rows,
        "best_trackers": best_trackers,
    }
    with (ckpt_dir / "checkpoint.json").open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    with (ckpt_dir / "replay_buffer.pkl").open("wb") as f:
        pickle.dump(list(policy.buffer.buffer), f)


def _load_checkpoint(
    resume_dir: Path,
) -> Tuple[int, int, int, List[Dict[str, float]], List[Dict[str, float]], Dict[str, Dict], list]:
    ckpt_dir = resume_dir / "checkpoint"
    if not (ckpt_dir / "checkpoint.json").exists():
        raise FileNotFoundError(f"Checkpoint not found in {ckpt_dir}")
    with (ckpt_dir / "checkpoint.json").open(encoding="utf-8") as f:
        state = json.load(f)
    buffer_data = []
    pkl_path = ckpt_dir / "replay_buffer.pkl"
    if pkl_path.exists():
        with pkl_path.open("rb") as f:
            buffer_data = pickle.load(f)
    return (
        int(state["episode"]),
        int(state["global_step"]),
        int(state.get("switch_count", 0)),
        state.get("training_rows", []),
        state.get("eval_rows", []),
        state.get("best_trackers", {}),
        buffer_data,
    )


def run_training(args) -> None:
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "train_runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden_dims = tuple(int(v.strip()) for v in args.hidden_dims.split(",") if v.strip())
    train_mpr_values = parse_mpr_values(getattr(args, "train_mpr_values", ""))
    resolved_n_up, resolved_n_down = align_vehicle_counts_for_mpr(args)
    args._resolved_n_up = resolved_n_up
    args._resolved_n_down = resolved_n_down
    rng = np.random.default_rng(int(args.seed) + 2026)

    scheduler = make_scheduler(args)
    env = build_env(args, scheduler, out_dir)
    init_state = env.current_state()
    policy = build_policy(
        args.policy_type,
        state_dim=len(init_state["vector"]),
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
        merge_cooldown_steps=args.merge_cooldown_steps,
        max_merge_streak=args.max_merge_streak,
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
        seed=args.seed,
    )
    if args.policy_type == "learned":
        warmstart_replay_buffer(policy, args, out_dir, rng)

    training_rows: List[Dict[str, float]] = []
    eval_rows: List[Dict[str, float]] = []
    best_trackers = {
        "reward": {"best": float("-inf"), "episode": -1, "path": ""},
        "comm_success": {"best": float("-inf"), "episode": -1, "path": ""},
        "min_gap": {"best": float("-inf"), "episode": -1, "path": ""},
        "safe_reward": {"best": float("-inf"), "episode": -1, "path": ""},
        "safety_first": {"best": float("-inf"), "episode": -1, "path": ""},
    }
    start_episode = 1
    initial_global_step = 0

    # --- Resume from checkpoint ---
    if args.resume_dir:
        resume_path = Path(args.resume_dir)
        print(f"[Resume] Loading checkpoint from {resume_path} ...")
        ckpt_ep, ckpt_step, ckpt_switch, ckpt_train_rows, ckpt_eval_rows, ckpt_best, buffer_data = _load_checkpoint(resume_path)
        # Load weights and meta
        weights_file = resume_path / "high_level_dqn.weights.h5"
        meta_file = resume_path / "high_level_policy_meta.json"
        if not weights_file.exists():
            # Fall back to best_checkpoints/reward
            weights_file = resume_path / "best_checkpoints" / "reward" / "high_level_dqn.weights.h5"
        policy.load(weights_file, meta_file)
        policy.episode_count = ckpt_ep
        policy.training_step = ckpt_step
        policy._switch_count = ckpt_switch
        # Restore replay buffer
        if buffer_data:
            for t in buffer_data[-policy.buffer.capacity:]:
                policy.buffer.buffer.append(t)
        training_rows = ckpt_train_rows
        eval_rows = ckpt_eval_rows
        if ckpt_best:
            best_trackers.update(ckpt_best)
        start_episode = ckpt_ep + 1
        initial_global_step = ckpt_step
        print(f"[Resume] Resuming from episode {start_episode}, global_step={ckpt_step}, buffer={len(policy.buffer)}")

    total_steps = args.episodes * args.steps
    parallel_rollout_enabled = _use_parallel_rollout(args)
    if int(getattr(args, "parallel_env_workers", 1)) > 1 and not parallel_rollout_enabled:
        print(
            "[ParallelRollout] disabled: only supported for policy_type=learned with "
            "disable_communication=True, comm_policy=heuristic, or "
            "comm_policy=agent with --allow-agent-parallel-rollout."
        )
    train_progress = create_progress(total=total_steps, desc="Joint training", unit="step")
    if initial_global_step > 0:
        train_progress.update(initial_global_step)

    episode_results: List[Tuple[int, float, List[HighLevelTransition], List[Dict[str, float]], float, Dict[str, float], int]] = []

    if parallel_rollout_enabled:
        max_workers = int(max(1, getattr(args, "parallel_env_workers", 1)))
        args_payload = _args_to_worker_payload(args)
        episode_plan = [
            (episode, float(sample_training_mpr(args, rng)))
            for episode in range(start_episode, args.episodes + 1)
        ]
        with _temporary_env_var("CUDA_VISIBLE_DEVICES", ""):
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_parallel_rollout_worker_initializer,
            ) as executor:
                in_flight = {}
                next_plan_idx = 0

                while next_plan_idx < len(episode_plan) and len(in_flight) < max_workers:
                    episode, episode_mpr = episode_plan[next_plan_idx]
                    future = _schedule_parallel_episode(
                        executor,
                        episode=int(episode),
                        episode_mpr=float(episode_mpr),
                        args_payload=args_payload,
                        policy_snapshot=_policy_snapshot(policy),
                        out_dir=out_dir,
                    )
                    in_flight[future] = (int(episode), float(episode_mpr))
                    next_plan_idx += 1

                while in_flight:
                    done_futures, _ = wait(tuple(in_flight.keys()), return_when=FIRST_COMPLETED)
                    for future in done_futures:
                        episode, episode_mpr = in_flight.pop(future)
                        result = future.result()
                        transitions = list(result["transitions"])
                        episode_records = list(result["episode_records"])
                        ep_reward = float(result["episode_reward"])
                        _sync_policy_action_history(
                            policy,
                            transitions,
                            episode=int(episode),
                            steps_per_episode=int(args.steps),
                        )
                        loss_values = _train_from_episode_transitions(
                            policy,
                            transitions,
                            train_every_k_steps=int(getattr(args, "train_every_k_steps", 1)),
                            train_updates_per_trigger=int(getattr(args, "train_updates_per_trigger", 1)),
                        )
                        train_progress.update(int(args.steps))
                        recent_loss = float(np.mean(loss_values[-20:])) if loss_values else 0.0
                        train_progress.set_postfix(
                            ep=episode,
                            reward=f"{ep_reward:.2f}",
                            eps=f"{policy.epsilon(episode * args.steps):.3f}",
                            loss=f"{recent_loss:.4f}",
                            inflight=len(in_flight),
                        )
                        episode_results.append((
                            int(episode),
                            float(episode_mpr),
                            transitions,
                            episode_records,
                            ep_reward,
                            *_episode_action_stats(transitions),
                        ))

                        if next_plan_idx < len(episode_plan):
                            next_episode, next_episode_mpr = episode_plan[next_plan_idx]
                            next_future = _schedule_parallel_episode(
                                executor,
                                episode=int(next_episode),
                                episode_mpr=float(next_episode_mpr),
                                args_payload=args_payload,
                                policy_snapshot=_policy_snapshot(policy),
                                out_dir=out_dir,
                            )
                            in_flight[next_future] = (int(next_episode), float(next_episode_mpr))
                            next_plan_idx += 1
    else:
        for episode in range(start_episode, args.episodes + 1):
            episode_mpr = sample_training_mpr(args, rng)
            args.mpr_cav = float(episode_mpr)
            scheduler = make_scheduler(args)
            env = build_env(args, scheduler, out_dir / f"episode_{episode:03d}")
            transitions, episode_records, ep_reward = _rollout_episode(
                policy,
                env,
                scheduler,
                episode=episode,
                args=args,
            )
            loss_values = _train_from_episode_transitions(
                policy,
                transitions,
                train_every_k_steps=int(getattr(args, "train_every_k_steps", 1)),
                train_updates_per_trigger=int(getattr(args, "train_updates_per_trigger", 1)),
            )
            train_progress.update(len(transitions))
            recent_loss = float(np.mean(loss_values[-20:])) if loss_values else 0.0
            train_progress.set_postfix(
                ep=episode,
                reward=f"{ep_reward:.2f}",
                eps=f"{policy.epsilon(episode * args.steps):.3f}",
                loss=f"{recent_loss:.4f}",
            )
            episode_results.append((
                int(episode),
                float(episode_mpr),
                transitions,
                episode_records,
                ep_reward,
                *_episode_action_stats(transitions),
            ))

    episode_results.sort(key=lambda item: int(item[0]))

    for episode, episode_mpr, transitions, episode_records, ep_reward, action_dist, switch_count in episode_results:
        policy.episode_count = int(episode)
        episode_loss_values: List[float] = []
        if transitions:
            recent_count = min(len(transitions), len(policy.loss_history))
            if recent_count > 0:
                episode_loss_values = [float(loss) for _, loss in policy.loss_history[-recent_count:]]
        episode_stats = summarize_episode_stats(episode_records)
        row = {
            "episode": float(episode),
            "mpr_cav": float(episode_mpr),
            "episode_reward": float(ep_reward),
            "avg_loss": float(np.mean(episode_loss_values)) if episode_loss_values else 0.0,
            "buffer_size": float(len(policy.buffer)),
            "epsilon_end": float(policy.epsilon(episode * args.steps)),
            "switch_count": float(switch_count),
            "action_keep_ratio": float(action_dist.get("keep", 0.0)),
            "action_compact_ratio": float(action_dist.get("compact", 0.0)),
            "action_expand_ratio": float(action_dist.get("expand", 0.0)),
            "action_split_ratio": float(action_dist.get("split", 0.0)),
            "action_merge_ratio": float(action_dist.get("merge", 0.0)),
            "action_emergency_ratio": float(action_dist.get("emergency", 0.0)),
            **episode_stats,
        }
        training_rows.append(row)
        train_progress.write(
            f"[Train] episode={episode} mpr={episode_mpr:.3f} reward={row['episode_reward']:.4f} "
            f"avg_loss={row['avg_loss']:.6f} buffer={int(row['buffer_size'])}"
        )

        if args.eval_every > 0 and (episode % args.eval_every == 0 or episode == args.episodes):
            eval_summary = evaluate_policy(policy, args, out_dir, f"ep{episode:03d}", show_progress=True)
            eval_rows.append(eval_summary)
            reward_score = float(eval_summary.get("avg_reward", 0.0))
            comm_score = float(eval_summary.get("avg_comm_v2v_success", 0.0))
            gap_score = min(float(eval_summary.get("worst_min_gap_up", 0.0)), float(eval_summary.get("worst_min_gap_down", 0.0)))
            safety_first_score = float("-inf")
            has_collision = float(eval_summary.get("has_collision", 1.0))
            collision_pairs_peak = float(eval_summary.get("peak_collision_pairs_active", 0.0))
            shield_load = float(eval_summary.get("total_shield_interventions", 0.0))
            emergency_brakes = float(eval_summary.get("total_emergency_brakes", 0.0))
            collision_penalty = 0.80 * has_collision + 0.15 * collision_pairs_peak
            safe_reward_score = reward_score + 0.05 * gap_score - collision_penalty
            if has_collision <= 0.0:
                safety_first_score = (
                    reward_score
                    + 0.08 * gap_score
                    + 0.12 * float(eval_summary.get("avg_platoon_rate", 0.0))
                    - 0.00015 * shield_load
                    - 0.00035 * emergency_brakes
                )
            else:
                safety_first_score = (
                    reward_score
                    - 1.50 * has_collision
                    - 0.25 * collision_pairs_peak
                    + 0.02 * gap_score
                    - 0.00015 * shield_load
                    - 0.00035 * emergency_brakes
                )
            for key, score in [
                ("reward", reward_score),
                ("comm_success", comm_score),
                ("min_gap", gap_score),
                ("safe_reward", safe_reward_score),
                ("safety_first", safety_first_score),
            ]:
                if score > best_trackers[key]["best"]:
                    ckpt_dir = out_dir / "best_checkpoints" / key
                    save_info_best = policy.save(ckpt_dir)
                    best_trackers[key] = {
                        "best": float(score),
                        "episode": int(episode),
                        "path": save_info_best["weights"],
                    }
            train_progress.write(
                f"[Eval] episode={episode} avg_reward={eval_summary.get('avg_reward', 0.0):.4f} "
                f"avg_comm_v2v_success={eval_summary.get('avg_comm_v2v_success', 0.0):.4f}"
            )

        # Save checkpoint after each episode (for resume support)
        _save_checkpoint(out_dir, episode, episode * args.steps, policy, training_rows, eval_rows, best_trackers)

    # Save training logs
    train_progress.close()
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
    if not bool(getattr(args, "skip_training_plots", False)):
        try:
            from .plotting import export_training_plots
        except ImportError:  # pragma: no cover
            from plotting import export_training_plots

        export_training_plots(out_dir, training_rows, eval_rows)
    with (out_dir / "train_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "episodes": args.episodes,
                "steps_per_episode": args.steps,
                "final_buffer_size": len(policy.buffer),
                "saved_weights": save_info["weights"],
                "saved_meta": save_info["meta"],
                "final_training_reward": training_rows[-1]["episode_reward"] if training_rows else 0.0,
                "hidden_dims": list(hidden_dims),
                "communication_enabled": not args.disable_communication,
                "policy_type": args.policy_type,
                "communication_policy_mode": args.comm_policy,
                "communication_gnn_type": args.comm_gnn,
                "train_mpr_values": train_mpr_values or [float(args.mpr_cav)],
                "eval_mpr_values": resolve_eval_mpr_values(args),
                "eval_steps": int(getattr(args, "eval_steps", 0) or args.steps),
                "skip_eval_plots": bool(getattr(args, "skip_eval_plots", False)),
                "skip_eval_csv": bool(getattr(args, "skip_eval_csv", False)),
                "eval_record_stride": int(getattr(args, "eval_record_stride", 1)),
                "skip_training_plots": bool(getattr(args, "skip_training_plots", False)),
                "resolved_n_up": int(args._resolved_n_up),
                "resolved_n_down": int(args._resolved_n_down),
                "resolved_total_vehicles": int(args._resolved_n_up + args._resolved_n_down),
                "mid_mpr_focus_ratio": float(getattr(args, "mid_mpr_focus_ratio", 0.0)),
                "spawn_y_min": float(args.spawn_y_min),
                "spawn_y_max": float(args.spawn_y_max),
                "lane_density_jitter": float(args.lane_density_jitter),
                "event_start": int(args.event_start),
                "event_duration": int(args.event_duration),
                "event_center": float(args.event_center),
                "event_length": float(args.event_length),
                "max_split_streak": int(args.max_split_streak),
                "max_reconfig_streak": int(args.max_reconfig_streak),
                "split_cooldown_steps": int(args.split_cooldown_steps),
                "merge_cooldown_steps": int(args.merge_cooldown_steps),
                "max_merge_streak": int(args.max_merge_streak),
                "train_every_k_steps": int(getattr(args, "train_every_k_steps", 1)),
                "train_updates_per_trigger": int(getattr(args, "train_updates_per_trigger", 1)),
                "best_checkpoints": best_trackers,
                "action_distribution": policy.action_distribution(),
                "switch_count": policy.switch_count,
            },
            file_obj,
            indent=2,
        )
    print(f"Trainable high-level policy saved to: {out_dir}")


def main() -> None:
    args = build_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
