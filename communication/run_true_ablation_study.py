#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict ablation runner for the full GATv2-DDQN training pipeline.

Why this script exists:
- The existing run_ablation_experiment.py compares GAT/SAGE/FC, which is a baseline comparison.
- For a thesis/paper "ablation study", we usually need to keep the backbone fixed
  (here: GATv2-DDQN) and remove one proposed component at a time.

This script is designed around the full Agent.train() path in agent.py, so it can
ablate the modules that actually appear in your proposed method:
    1) urgency-aware power shaping
    2) RB de-concentration + soft mask
    3) conflict guidance / conflict penalty
    4) attention entropy regularization
    5) adaptive repruning

Typical usage:
python run_true_ablation_study.py \
    --out-dir runs/ablation_true \
    --variants full,no_urgency,no_rb_balance,no_conflict,no_attn_reg,no_reprune \
    --seeds 123,124,125 \
    --train-steps 12000 --test-every 200 --test-sample 80 \
    --n-up 12 --n-down 20 --topo tree
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Any

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reduce TF log noise.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from highway_environment import HighwayTopoEnv
from agent import Agent
from gnn_factory import build_gnn


# -----------------------------
# Common helpers
# -----------------------------

def setup_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_int_list(text: str) -> List[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    seen = set()
    for item in text.split(","):
        item = item.strip()
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def convergence_step(test_history: List[List[float]], ratio: float = 0.9) -> int:
    """Return first step whose V2V success reaches ratio * final value."""
    if not test_history:
        return -1
    final_v2v = 1.0 - float(test_history[-1][2])
    threshold = ratio * final_v2v
    for step, _v2i, fail in test_history:
        v2v = 1.0 - float(fail)
        if v2v >= threshold:
            return int(step)
    return int(test_history[-1][0])


# -----------------------------
# Environment wrapper
# -----------------------------

class StableAblationEnv(HighwayTopoEnv):
    """
    HighwayTopoEnv.new_random_game() / _init_session() will reset demand_amount,
    V2V_limit and some reward-related attributes. Agent.train() internally calls
    env.new_random_game(), so paper-level ablation settings would otherwise be lost.

    This wrapper re-applies all experiment overrides after every reset.
    """

    def __init__(
        self,
        *args,
        demand_amount: float,
        v2v_limit: float,
        reward_cfg: Dict[str, Any],
        **kwargs,
    ):
        self._ablation_demand_amount = float(demand_amount)
        self._ablation_v2v_limit = float(v2v_limit)
        self._ablation_reward_cfg = dict(reward_cfg)
        super().__init__(*args, **kwargs)
        self.apply_ablation_overrides()

    def apply_ablation_overrides(self) -> None:
        self.demand_amount = float(self._ablation_demand_amount)
        self.demand = self.demand_amount * np.ones((self.n_Veh, 3), dtype=np.float32)
        self.V2V_limit = float(self._ablation_v2v_limit)
        self.individual_time_limit = self.V2V_limit * np.ones((self.n_Veh, 3), dtype=np.float32)

        # Keep any existing time interval initialization, but make sure shape is correct.
        if not hasattr(self, "individual_time_interval") or self.individual_time_interval.shape != (self.n_Veh, 3):
            self.individual_time_interval = np.random.exponential(0.05, (self.n_Veh, 3))

        for key, value in self._ablation_reward_cfg.items():
            setattr(self, key, value)

    def new_random_game(self, n_Veh: int = 0):
        super().new_random_game(n_Veh=n_Veh)
        self.apply_ablation_overrides()


# -----------------------------
# Ablation presets
# -----------------------------

BASE_PRESET: Dict[str, Dict[str, Any]] = {
    "env": {
        "beta_urgency_pos": 0.014,
        "beta_urgency_neg": 0.008,
        "urgency_threshold": 0.25,
        "rb_anti_conc_alpha": 0.018,
        "rb_hot_threshold": 0.20,
        "rb_softmask_alpha": 0.18,
        "rb_softmask_window": 50,
    },
    "agent": {
        "beta_urgency_pos": 0.014,
        "beta_urgency_neg": 0.008,
        "urgency_threshold": 0.25,
        "conflict_penalty_weight": 0.02,
        "conflict_cost_weight": 0.02,
        "rb_anti_conc_alpha": 0.018,
        "rb_hot_threshold": 0.20,
        "rb_softmask_alpha": 0.18,
        "rb_softmask_window": 50,
    },
    "gnn": {
        "reprune_every": 300,
        "hysteresis_keep": 0.50,
        "reprune_start_step": 600,
        "reg_attn_w": 1e-4,
        "enable_reprune": True,
    },
}


ABLATION_OVERRIDES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "full": {},
    # Remove urgency-aware power shaping from both environment-side reward and agent-side shaping.
    "no_urgency": {
        "env": {
            "beta_urgency_pos": 0.0,
            "beta_urgency_neg": 0.0,
        },
        "agent": {
            "beta_urgency_pos": 0.0,
            "beta_urgency_neg": 0.0,
        },
    },
    # Remove RB de-concentration and soft-mask behavior.
    "no_rb_balance": {
        "env": {
            "rb_anti_conc_alpha": 0.0,
            "rb_softmask_alpha": 0.0,
            "rb_hot_threshold": 1.1,
        },
        "agent": {
            "rb_anti_conc_alpha": 0.0,
            "rb_softmask_alpha": 0.0,
            "rb_hot_threshold": 1.1,
        },
    },
    # Remove conflict-aware guidance from the agent side.
    "no_conflict": {
        "agent": {
            "conflict_penalty_weight": 0.0,
            "conflict_cost_weight": 0.0,
        },
    },
    # Remove attention entropy regularization in GraphGAT training.
    "no_attn_reg": {
        "gnn": {
            "reg_attn_w": 0.0,
        },
    },
    # Remove adaptive repruning (leave graph structure fixed during training).
    "no_reprune": {
        "gnn": {
            "reprune_every": 10**9,
            "reprune_start_step": 10**9,
            "hysteresis_keep": 0.0,
            "enable_reprune": False,
        },
    },
}


def deep_merge(base: Dict[str, Dict[str, Any]], override: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = deepcopy(base)
    for block, values in override.items():
        out.setdefault(block, {})
        out[block].update(values)
    return out


# -----------------------------
# Single run
# -----------------------------

def build_env(args: argparse.Namespace, preset: Dict[str, Dict[str, Any]], seed: int) -> StableAblationEnv:
    env = StableAblationEnv(
        n_up=args.n_up,
        n_down=args.n_down,
        lanes_per_dir=args.lanes,
        spacing=args.spacing,
        base_y=args.base_y,
        height=args.height,
        topology_type=args.topo,
        leader_at_front=bool(args.leader_front),
        leader_lane_up=args.leader_lane_up,
        leader_lane_down=args.leader_lane_down,
        leader_dynamic=bool(args.leader_dynamic),
        move_speed=float(args.use_move_speed),
        v2i_mode=args.v2i_mode,
        bs_layout=args.bs_layout,
        bs_spacing=args.bs_spacing,
        bs_min_stay_steps=args.bs_min_stay,
        bs_handover_hyst_m=args.bs_hyst,
        seed=seed,
        demand_amount=args.demand_amount,
        v2v_limit=args.v2v_limit,
        reward_cfg=preset["env"],
    )
    env.new_random_game()
    return env


def build_agent(
    args: argparse.Namespace,
    env: StableAblationEnv,
    preset: Dict[str, Dict[str, Any]],
    run_dir: Path,
) -> Agent:
    # Agent will internally create a default GNN first; then we replace it with a custom one.
    agent = Agent(
        [],
        env,
        gnn_type="gat",
        warmup_steps=args.warmup_steps,
        epsilon_min=args.epsilon_min,
        epsilon_decay_steps=args.epsilon_decay_steps,
        speed_mode=bool(args.speed_mode),
        plot_dpi=args.dpi,
        replay_size=args.replay_size,
        power_log_stride=args.power_log_stride,
        power_log_max=args.power_log_max,
        soft_update_tau=args.soft_update_tau,
        power_cost_weight=args.power_cost_weight,
        conflict_cost_weight=preset["agent"].get("conflict_cost_weight", 0.02),
        skip_embedding_steps=args.skip_embedding_steps,
        batch_decay_step=args.batch_decay_step,
        batch_decay_factor=args.batch_decay_factor,
        beta_urgency_pos=preset["agent"].get("beta_urgency_pos", 0.018),
        beta_urgency_neg=preset["agent"].get("beta_urgency_neg", 0.025),
        urgency_threshold=preset["agent"].get("urgency_threshold", 0.25),
        conflict_penalty_weight=preset["agent"].get("conflict_penalty_weight", 0.02),
        conflict_window_steps=args.conflict_window_steps,
        rb_anti_conc_alpha=preset["agent"].get("rb_anti_conc_alpha", 0.012),
        rb_hot_threshold=preset["agent"].get("rb_hot_threshold", 0.22),
        rb_softmask_alpha=preset["agent"].get("rb_softmask_alpha", 0.15),
        rb_softmask_window=preset["agent"].get("rb_softmask_window", 50),
        dynamic_graph_refresh=(not args.disable_dynamic_graph_refresh),
        graph_refresh_steps=args.graph_refresh_steps,
        run_dir=str(run_dir),
    )

    custom_gnn = build_gnn(
        env,
        gnn_type="gat",
        distance_threshold=args.distance_threshold,
        lr=args.gnn_lr,
        gat_train_interval=args.gat_train_interval,
        grad_clip=args.grad_clip,
        gat_hidden_dim=args.gat_hidden_dim,
        gat_out_dim=args.gat_out_dim,
        gat_heads=args.gat_heads,
        gat_attn_dropout=args.gat_attn_dropout,
        gat_top_k=args.gat_top_k,
        gat_use_proximity_edges=bool(args.enable_gat_proximity_edges),
        gat_proximity_radius=args.gat_proximity_radius,
        gat_max_proximity_neighbors=args.gat_max_proximity_neighbors,
        reprune_every=preset["gnn"].get("reprune_every", 300),
        hysteresis_keep=preset["gnn"].get("hysteresis_keep", 0.5),
        reprune_start_step=preset["gnn"].get("reprune_start_step", 600),
        reg_attn_w=preset["gnn"].get("reg_attn_w", 1e-3),
        enable_reprune=preset["gnn"].get("enable_reprune", True),
    )

    agent.G = custom_gnn
    if hasattr(agent.G, "tb_writer"):
        agent.G.tb_writer = agent.tb_gnn
    return agent


def run_one_variant_seed(
    variant: str,
    seed: int,
    args: argparse.Namespace,
    out_root: Path,
) -> Dict[str, Any]:
    preset = deep_merge(BASE_PRESET, ABLATION_OVERRIDES[variant])
    run_dir = out_root / variant / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    tf.keras.backend.clear_session()
    gc.collect()

    env = build_env(args, preset, seed)
    agent = build_agent(args, env, preset, run_dir)

    meta = {
        "variant": variant,
        "seed": seed,
        "preset": preset,
        "settings": vars(args),
    }
    with (run_dir / "ablation_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    t0 = time.perf_counter()
    agent.train(
        max_steps=args.train_steps,
        test_every_steps=args.test_every,
        test_sample=args.test_sample,
    )
    train_seconds = time.perf_counter() - t0

    final_v2i = float(agent.test_history[-1][1]) if agent.test_history else 0.0
    final_fail = float(agent.test_history[-1][2]) if agent.test_history else 1.0
    final_v2v = 1.0 - final_fail
    conv_step = convergence_step(agent.test_history, ratio=args.conv_ratio)
    mean_used_rb = float(np.mean([x[1] for x in agent.used_blocks_history])) if agent.used_blocks_history else 0.0

    summary = {
        "variant": variant,
        "seed": seed,
        "final_v2i": final_v2i,
        "final_v2v": final_v2v,
        "final_fail": final_fail,
        "convergence_step": conv_step,
        "mean_used_rb": mean_used_rb,
        "train_seconds": float(train_seconds),
        "run_dir": str(run_dir),
    }
    return summary


# -----------------------------
# Aggregation and plotting
# -----------------------------

def aggregate_results(seed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in seed_results:
        grouped.setdefault(row["variant"], []).append(row)

    mean_rows: List[Dict[str, Any]] = []
    for variant, rows in grouped.items():
        v2i_vals = [float(r["final_v2i"]) for r in rows]
        v2v_vals = [float(r["final_v2v"]) for r in rows]
        conv_vals = [float(r["convergence_step"]) for r in rows if float(r["convergence_step"]) >= 0]
        rb_vals = [float(r["mean_used_rb"]) for r in rows]
        t_vals = [float(r["train_seconds"]) for r in rows]

        def _std(vals: List[float]) -> float:
            return float(pstdev(vals)) if len(vals) > 1 else 0.0

        mean_rows.append({
            "variant": variant,
            "runs": len(rows),
            "final_v2i_mean": float(mean(v2i_vals)) if v2i_vals else 0.0,
            "final_v2i_std": _std(v2i_vals),
            "final_v2v_mean": float(mean(v2v_vals)) if v2v_vals else 0.0,
            "final_v2v_std": _std(v2v_vals),
            "conv_step_mean": float(mean(conv_vals)) if conv_vals else -1.0,
            "conv_step_std": _std(conv_vals) if conv_vals else 0.0,
            "mean_used_rb_mean": float(mean(rb_vals)) if rb_vals else 0.0,
            "mean_used_rb_std": _std(rb_vals),
            "train_seconds_mean": float(mean(t_vals)) if t_vals else 0.0,
            "train_seconds_std": _std(t_vals),
        })

    return mean_rows


def compute_delta_results(
    seed_results: List[Dict[str, Any]],
    base_variant: str = "full",
) -> List[Dict[str, Any]]:
    base_by_seed: Dict[int, Dict[str, Any]] = {
        int(r["seed"]): r for r in seed_results if r["variant"] == base_variant
    }
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for row in seed_results:
        variant = row["variant"]
        if variant == base_variant:
            continue
        seed = int(row["seed"])
        base = base_by_seed.get(seed)
        if not base:
            continue
        grouped.setdefault(variant, []).append(
            {
                "delta_v2i": float(row["final_v2i"]) - float(base["final_v2i"]),
                "delta_v2v": float(row["final_v2v"]) - float(base["final_v2v"]),
                "delta_conv": float(row["convergence_step"]) - float(base["convergence_step"]),
            }
        )

    delta_rows: List[Dict[str, Any]] = []
    for variant, rows in grouped.items():
        dv2i = [r["delta_v2i"] for r in rows]
        dv2v = [r["delta_v2v"] for r in rows]
        dconv = [r["delta_conv"] for r in rows]

        def _std(vals: List[float]) -> float:
            return float(pstdev(vals)) if len(vals) > 1 else 0.0

        delta_rows.append({
            "variant": variant,
            "runs": len(rows),
            "delta_v2i_mean": float(mean(dv2i)) if dv2i else 0.0,
            "delta_v2i_std": _std(dv2i),
            "delta_v2v_mean": float(mean(dv2v)) if dv2v else 0.0,
            "delta_v2v_std": _std(dv2v),
            "delta_conv_mean": float(mean(dconv)) if dconv else 0.0,
            "delta_conv_std": _std(dconv),
        })

    return delta_rows


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(mean_rows: List[Dict[str, Any]], metric_key: str, err_key: str, ylabel: str, out_path: Path) -> None:
    if not mean_rows:
        return
    labels = [row["variant"] for row in mean_rows]
    values = [row[metric_key] for row in mean_rows]
    errors = [row[err_key] for row in mean_rows]

    x = np.arange(len(labels))
    plt.figure(figsize=(8, 4.8))
    plt.bar(x, values, yerr=errors, capsize=4)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict ablation study on full GATv2-DDQN pipeline.")

    # Scenario
    parser.add_argument("--n-up", type=int, default=12)
    parser.add_argument("--n-down", type=int, default=20)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--spacing", type=float, default=25.0)
    parser.add_argument("--base-y", type=float, default=0.0)
    parser.add_argument("--height", type=float, default=1200.0)
    parser.add_argument("--topo", type=str, default="tree", choices=["star", "tree"])
    parser.add_argument("--leader-front", action="store_true")
    parser.add_argument("--leader-lane-up", type=int, default=None)
    parser.add_argument("--leader-lane-down", type=int, default=None)
    parser.add_argument("--leader-dynamic", action="store_true")
    parser.add_argument("--use-move-speed", type=float, default=0.0)
    parser.add_argument("--v2i-mode", type=str, default="rsu", choices=["rsu", "single"])
    parser.add_argument("--bs-layout", type=str, default="median", choices=["median", "dual-roadside"])
    parser.add_argument("--bs-spacing", type=float, default=250.0)
    parser.add_argument("--bs-min-stay", type=int, default=5)
    parser.add_argument("--bs-hyst", type=float, default=15.0)

    # Training
    parser.add_argument("--train-steps", type=int, default=12000)
    parser.add_argument("--test-every", type=int, default=200)
    parser.add_argument("--test-sample", type=int, default=80)
    parser.add_argument("--warmup-steps", type=int, default=800)
    parser.add_argument("--epsilon-decay-steps", type=int, default=4000)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=123, help="Used when --seeds is empty.")
    parser.add_argument("--seeds", type=str, default="123,124,125")
    parser.add_argument("--out-dir", type=str, default="runs/ablation_true")
    parser.add_argument("--dpi", type=int, default=224)
    parser.add_argument("--speed-mode", action="store_true")
    parser.add_argument("--conv-ratio", type=float, default=0.9)

    # Traffic difficulty
    parser.add_argument("--demand-amount", type=float, default=130.0)
    parser.add_argument("--v2v-limit", type=float, default=0.045)

    # Agent misc
    parser.add_argument("--replay-size", type=int, default=200000)
    parser.add_argument("--power-log-stride", type=int, default=4)
    parser.add_argument("--power-log-max", type=int, default=200000)
    parser.add_argument("--soft-update-tau", type=float, default=0.005)
    parser.add_argument("--power-cost-weight", type=float, default=0.01)
    parser.add_argument("--skip-embedding-steps", type=int, default=9)
    parser.add_argument("--batch-decay-step", type=int, default=None)
    parser.add_argument("--batch-decay-factor", type=float, default=0.5)
    parser.add_argument("--conflict-window-steps", type=int, default=50)

    # GNN
    parser.add_argument("--distance-threshold", type=float, default=150.0)
    parser.add_argument("--gnn-lr", type=float, default=3e-4)
    parser.add_argument("--gat-train-interval", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--gat-hidden-dim", type=int, default=48)
    parser.add_argument("--gat-out-dim", type=int, default=32)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--gat-attn-dropout", type=float, default=0.05)
    parser.add_argument("--gat-top-k", type=int, default=8)
    parser.add_argument("--gat-proximity-radius", type=float, default=220.0)
    parser.add_argument("--gat-max-proximity-neighbors", type=int, default=8)
    parser.add_argument("--enable-gat-proximity-edges", action="store_true", default=True)
    parser.add_argument("--graph-refresh-steps", type=int, default=300)
    parser.add_argument("--disable-dynamic-graph-refresh", action="store_true")

    # Which ablations to run
    parser.add_argument(
        "--variants",
        type=str,
        default="full,no_urgency,no_rb_balance,no_conflict,no_attn_reg,no_reprune",
        help="Comma-separated variants. Available: " + ",".join(ABLATION_OVERRIDES.keys()),
    )

    args = parser.parse_args()
    setup_gpu()

    variants = parse_str_list(args.variants)
    unknown = [v for v in variants if v not in ABLATION_OVERRIDES]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Available: {list(ABLATION_OVERRIDES.keys())}")
    if not variants:
        variants = ["full"]

    seeds = parse_int_list(args.seeds)
    if not seeds:
        seeds = [args.seed]

    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    all_seed_rows: List[Dict[str, Any]] = []

    print("=" * 70)
    print("Running strict ablation study on full GATv2-DDQN pipeline")
    print(f"Variants: {variants}")
    print(f"Seeds   : {seeds}")
    print(f"Out dir : {out_root}")
    print("=" * 70)

    for variant in variants:
        for seed in seeds:
            print(f"\n[Run] variant={variant} | seed={seed}")
            result = run_one_variant_seed(variant, seed, args, out_root)
            all_seed_rows.append(result)
            print(
                "  -> final_v2v={:.4f}, final_v2i={:.4f}, conv_step={}, train_seconds={:.1f}".format(
                    result["final_v2v"],
                    result["final_v2i"],
                    result["convergence_step"],
                    result["train_seconds"],
                )
            )

    mean_rows = aggregate_results(all_seed_rows)
    delta_rows = compute_delta_results(all_seed_rows, base_variant="full")

    save_csv(out_root / "ablation_seed_results.csv", all_seed_rows)
    save_csv(out_root / "ablation_mean_std.csv", mean_rows)
    save_csv(out_root / "ablation_delta_mean_std.csv", delta_rows)

    with (out_root / "ablation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "variants": variants,
                "seeds": seeds,
                "seed_results": all_seed_rows,
                "mean_std": mean_rows,
                "delta_mean_std": delta_rows,
                "base_preset": BASE_PRESET,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    plot_metric(
        mean_rows,
        metric_key="final_v2v_mean",
        err_key="final_v2v_std",
        ylabel="Final V2V Success Rate",
        out_path=out_root / "ablation_v2v_bar.png",
    )
    plot_metric(
        mean_rows,
        metric_key="final_v2i_mean",
        err_key="final_v2i_std",
        ylabel="Final V2I Rate",
        out_path=out_root / "ablation_v2i_bar.png",
    )
    plot_metric(
        mean_rows,
        metric_key="conv_step_mean",
        err_key="conv_step_std",
        ylabel="Convergence Step",
        out_path=out_root / "ablation_convergence_bar.png",
    )

    plot_metric(
        delta_rows,
        metric_key="delta_v2v_mean",
        err_key="delta_v2v_std",
        ylabel="Delta V2V Success Rate vs Full",
        out_path=out_root / "ablation_delta_v2v_bar.png",
    )
    plot_metric(
        delta_rows,
        metric_key="delta_v2i_mean",
        err_key="delta_v2i_std",
        ylabel="Delta V2I Rate vs Full",
        out_path=out_root / "ablation_delta_v2i_bar.png",
    )

    print("\n[OK] All ablation runs are complete.")
    print(f"Seed-level CSV : {out_root / 'ablation_seed_results.csv'}")
    print(f"Mean/std  CSV  : {out_root / 'ablation_mean_std.csv'}")
    print(f"Summary JSON   : {out_root / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()
