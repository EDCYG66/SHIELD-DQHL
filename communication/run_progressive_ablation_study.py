#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Progressive ablation runner for GATv2-DDQN.

Goal:
- Start from "no extra module" backbone.
- Add modules one by one.
- End at full GATv2 setting.

Default progressive stages:
  1) none             : no urgency / no RB-balance / no conflict / no attn-reg / no reprune
  2) plus_urgency     : + urgency-aware shaping
  3) plus_rb_balance  : + RB de-concentration + soft-mask
  4) plus_conflict    : + conflict guidance
  5) plus_attn_reg    : + attention entropy regularization
  6) full             : + adaptive repruning (full GATv2 modules)

Example:
python run_progressive_ablation_study.py \
  --out-dir runs/ablation_progressive \
  --stages none,plus_urgency,plus_rb_balance,plus_conflict,plus_attn_reg,full \
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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from highway_environment import HighwayTopoEnv
from agent import Agent
from gnn_factory import build_gnn


# -----------------------------
# Helpers
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
    if not test_history:
        return -1
    final_v2v = 1.0 - float(test_history[-1][2])
    threshold = ratio * final_v2v
    for step, _v2i, fail in test_history:
        v2v = 1.0 - float(fail)
        if v2v >= threshold:
            return int(step)
    return int(test_history[-1][0])


def deep_merge(base: Dict[str, Dict[str, Any]], override: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = deepcopy(base)
    for block, values in override.items():
        out.setdefault(block, {})
        out[block].update(values)
    return out


# -----------------------------
# Env wrapper (keep settings stable across reset)
# -----------------------------

class StableAblationEnv(HighwayTopoEnv):
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

        if not hasattr(self, "individual_time_interval") or self.individual_time_interval.shape != (self.n_Veh, 3):
            self.individual_time_interval = np.random.exponential(0.05, (self.n_Veh, 3))

        for key, value in self._ablation_reward_cfg.items():
            setattr(self, key, value)

    def new_random_game(self, n_Veh: int = 0):
        super().new_random_game(n_Veh=n_Veh)
        self.apply_ablation_overrides()


# -----------------------------
# Progressive stage definitions
# -----------------------------

FULL_VALUES: Dict[str, Dict[str, Any]] = {
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


# "none" stage baseline: keep backbone, disable all proposed add-on modules.
NONE_VALUES: Dict[str, Dict[str, Any]] = {
    "env": {
        "beta_urgency_pos": 0.0,
        "beta_urgency_neg": 0.0,
        "urgency_threshold": 0.25,
        "rb_anti_conc_alpha": 0.0,
        "rb_hot_threshold": 1.1,
        "rb_softmask_alpha": 0.0,
        "rb_softmask_window": 50,
    },
    "agent": {
        "beta_urgency_pos": 0.0,
        "beta_urgency_neg": 0.0,
        "urgency_threshold": 0.25,
        "conflict_penalty_weight": 0.0,
        "conflict_cost_weight": 0.0,
        "rb_anti_conc_alpha": 0.0,
        "rb_hot_threshold": 1.1,
        "rb_softmask_alpha": 0.0,
        "rb_softmask_window": 50,
    },
    "gnn": {
        "reprune_every": 10**9,
        "hysteresis_keep": 0.0,
        "reprune_start_step": 10**9,
        "reg_attn_w": 0.0,
        "enable_reprune": False,
    },
}


MODULES = ["urgency", "rb_balance", "conflict", "attn_reg", "reprune"]

DEFAULT_STAGE_MODULES: Dict[str, List[str]] = {
    "none": [],
    "plus_urgency": ["urgency"],
    "plus_rb_balance": ["urgency", "rb_balance"],
    "plus_conflict": ["urgency", "rb_balance", "conflict"],
    "plus_attn_reg": ["urgency", "rb_balance", "conflict", "attn_reg"],
    "full": ["urgency", "rb_balance", "conflict", "attn_reg", "reprune"],
}


def build_stage_preset(enabled_modules: List[str]) -> Dict[str, Dict[str, Any]]:
    preset = deepcopy(NONE_VALUES)
    enabled = set(enabled_modules)

    if "urgency" in enabled:
        preset["env"]["beta_urgency_pos"] = FULL_VALUES["env"]["beta_urgency_pos"]
        preset["env"]["beta_urgency_neg"] = FULL_VALUES["env"]["beta_urgency_neg"]
        preset["agent"]["beta_urgency_pos"] = FULL_VALUES["agent"]["beta_urgency_pos"]
        preset["agent"]["beta_urgency_neg"] = FULL_VALUES["agent"]["beta_urgency_neg"]

    if "rb_balance" in enabled:
        preset["env"]["rb_anti_conc_alpha"] = FULL_VALUES["env"]["rb_anti_conc_alpha"]
        preset["env"]["rb_hot_threshold"] = FULL_VALUES["env"]["rb_hot_threshold"]
        preset["env"]["rb_softmask_alpha"] = FULL_VALUES["env"]["rb_softmask_alpha"]
        preset["agent"]["rb_anti_conc_alpha"] = FULL_VALUES["agent"]["rb_anti_conc_alpha"]
        preset["agent"]["rb_hot_threshold"] = FULL_VALUES["agent"]["rb_hot_threshold"]
        preset["agent"]["rb_softmask_alpha"] = FULL_VALUES["agent"]["rb_softmask_alpha"]

    if "conflict" in enabled:
        preset["agent"]["conflict_penalty_weight"] = FULL_VALUES["agent"]["conflict_penalty_weight"]
        preset["agent"]["conflict_cost_weight"] = FULL_VALUES["agent"]["conflict_cost_weight"]

    if "attn_reg" in enabled:
        preset["gnn"]["reg_attn_w"] = FULL_VALUES["gnn"]["reg_attn_w"]

    if "reprune" in enabled:
        preset["gnn"]["reprune_every"] = FULL_VALUES["gnn"]["reprune_every"]
        preset["gnn"]["reprune_start_step"] = FULL_VALUES["gnn"]["reprune_start_step"]
        preset["gnn"]["hysteresis_keep"] = FULL_VALUES["gnn"]["hysteresis_keep"]
        preset["gnn"]["enable_reprune"] = FULL_VALUES["gnn"]["enable_reprune"]

    return preset


# -----------------------------
# Build env / agent
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
        conflict_cost_weight=preset["agent"].get("conflict_cost_weight", 0.0),
        skip_embedding_steps=args.skip_embedding_steps,
        batch_decay_step=args.batch_decay_step,
        batch_decay_factor=args.batch_decay_factor,
        beta_urgency_pos=preset["agent"].get("beta_urgency_pos", 0.0),
        beta_urgency_neg=preset["agent"].get("beta_urgency_neg", 0.0),
        urgency_threshold=preset["agent"].get("urgency_threshold", 0.25),
        conflict_penalty_weight=preset["agent"].get("conflict_penalty_weight", 0.0),
        conflict_window_steps=args.conflict_window_steps,
        rb_anti_conc_alpha=preset["agent"].get("rb_anti_conc_alpha", 0.0),
        rb_hot_threshold=preset["agent"].get("rb_hot_threshold", 1.1),
        rb_softmask_alpha=preset["agent"].get("rb_softmask_alpha", 0.0),
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
        reprune_every=preset["gnn"].get("reprune_every", 10**9),
        hysteresis_keep=preset["gnn"].get("hysteresis_keep", 0.0),
        reprune_start_step=preset["gnn"].get("reprune_start_step", 10**9),
        reg_attn_w=preset["gnn"].get("reg_attn_w", 0.0),
        enable_reprune=preset["gnn"].get("enable_reprune", False),
    )

    agent.G = custom_gnn
    if hasattr(agent.G, "tb_writer"):
        agent.G.tb_writer = agent.tb_gnn
    return agent


# -----------------------------
# Single run
# -----------------------------

def run_one_stage_seed(
    stage: str,
    stage_idx: int,
    enabled_modules: List[str],
    seed: int,
    args: argparse.Namespace,
    out_root: Path,
) -> Dict[str, Any]:
    preset = build_stage_preset(enabled_modules)
    run_dir = out_root / stage / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    tf.keras.backend.clear_session()
    gc.collect()

    env = build_env(args, preset, seed)
    agent = build_agent(args, env, preset, run_dir)

    meta = {
        "stage": stage,
        "stage_index": stage_idx,
        "enabled_modules": enabled_modules,
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

    return {
        "stage": stage,
        "stage_index": stage_idx,
        "enabled_modules": ",".join(enabled_modules),
        "seed": seed,
        "final_v2i": final_v2i,
        "final_v2v": final_v2v,
        "final_fail": final_fail,
        "convergence_step": conv_step,
        "mean_used_rb": mean_used_rb,
        "train_seconds": float(train_seconds),
        "run_dir": str(run_dir),
    }


# -----------------------------
# Aggregate / delta / plotting
# -----------------------------

def aggregate_results(seed_results: List[Dict[str, Any]], stage_order: List[str]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {s: [] for s in stage_order}
    for row in seed_results:
        grouped.setdefault(row["stage"], []).append(row)

    mean_rows: List[Dict[str, Any]] = []
    for stage in stage_order:
        rows = grouped.get(stage, [])
        if not rows:
            continue

        v2i_vals = [float(r["final_v2i"]) for r in rows]
        v2v_vals = [float(r["final_v2v"]) for r in rows]
        conv_vals = [float(r["convergence_step"]) for r in rows if float(r["convergence_step"]) >= 0]
        rb_vals = [float(r["mean_used_rb"]) for r in rows]
        t_vals = [float(r["train_seconds"]) for r in rows]

        def _std(vals: List[float]) -> float:
            return float(pstdev(vals)) if len(vals) > 1 else 0.0

        mean_rows.append({
            "stage": stage,
            "stage_index": int(rows[0]["stage_index"]),
            "enabled_modules": rows[0]["enabled_modules"],
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


def compute_delta_vs_prev(seed_results: List[Dict[str, Any]], stage_order: List[str]) -> List[Dict[str, Any]]:
    by_stage_seed: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for row in seed_results:
        by_stage_seed.setdefault(row["stage"], {})[int(row["seed"])] = row

    delta_rows: List[Dict[str, Any]] = []

    for i in range(1, len(stage_order)):
        prev_stage = stage_order[i - 1]
        curr_stage = stage_order[i]

        prev_by_seed = by_stage_seed.get(prev_stage, {})
        curr_by_seed = by_stage_seed.get(curr_stage, {})
        seeds = sorted(set(prev_by_seed.keys()).intersection(curr_by_seed.keys()))
        if not seeds:
            continue

        dv2i = []
        dv2v = []
        dconv = []
        for s in seeds:
            prev = prev_by_seed[s]
            curr = curr_by_seed[s]
            dv2i.append(float(curr["final_v2i"]) - float(prev["final_v2i"]))
            dv2v.append(float(curr["final_v2v"]) - float(prev["final_v2v"]))
            dconv.append(float(curr["convergence_step"]) - float(prev["convergence_step"]))

        def _std(vals: List[float]) -> float:
            return float(pstdev(vals)) if len(vals) > 1 else 0.0

        delta_rows.append({
            "prev_stage": prev_stage,
            "stage": curr_stage,
            "runs": len(seeds),
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
    labels = [row["stage"] for row in mean_rows]
    values = [row[metric_key] for row in mean_rows]
    errors = [row[err_key] for row in mean_rows]

    x = np.arange(len(labels))
    plt.figure(figsize=(9, 4.8))
    plt.bar(x, values, yerr=errors, capsize=4)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_delta_prev(delta_rows: List[Dict[str, Any]], metric_key: str, err_key: str, ylabel: str, out_path: Path) -> None:
    if not delta_rows:
        return
    labels = [row["stage"] for row in delta_rows]
    values = [row[metric_key] for row in delta_rows]
    errors = [row[err_key] for row in delta_rows]
    x = np.arange(len(labels))

    plt.figure(figsize=(9, 4.8))
    plt.bar(x, values, yerr=errors, capsize=4)
    plt.axhline(0.0, color="#555555", linewidth=1.0)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run progressive (module-addition) ablation on GATv2-DDQN.")

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
    parser.add_argument("--out-dir", type=str, default="runs/ablation_progressive")
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

    # Progressive stage selection
    parser.add_argument(
        "--stages",
        type=str,
        default="none,plus_urgency,plus_rb_balance,plus_conflict,plus_attn_reg,full",
        help="Comma-separated progressive stages. Available: " + ",".join(DEFAULT_STAGE_MODULES.keys()),
    )

    args = parser.parse_args()
    setup_gpu()

    stages = parse_str_list(args.stages)
    unknown = [s for s in stages if s not in DEFAULT_STAGE_MODULES]
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}. Available: {list(DEFAULT_STAGE_MODULES.keys())}")
    if not stages:
        stages = ["none", "full"]

    # Validate monotonic addition (modules should not be removed once added).
    prev_set = set()
    for s in stages:
        cur = set(DEFAULT_STAGE_MODULES[s])
        if not prev_set.issubset(cur):
            raise ValueError(
                f"Stage order is not progressive at '{s}'. "
                f"Expected modules to contain previous stage modules {sorted(prev_set)}, got {sorted(cur)}."
            )
        prev_set = cur

    seeds = parse_int_list(args.seeds)
    if not seeds:
        seeds = [args.seed]

    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    stage_rows = []
    for i, s in enumerate(stages):
        stage_rows.append({
            "stage": s,
            "stage_index": i,
            "enabled_modules": ",".join(DEFAULT_STAGE_MODULES[s]),
        })
    save_csv(out_root / "progressive_stage_definition.csv", stage_rows)

    all_seed_rows: List[Dict[str, Any]] = []

    print("=" * 74)
    print("Running progressive ablation study (none -> full GATv2 modules)")
    print(f"Stages : {stages}")
    print(f"Seeds  : {seeds}")
    print(f"Out dir: {out_root}")
    print("=" * 74)

    for i, stage in enumerate(stages):
        enabled_modules = DEFAULT_STAGE_MODULES[stage]
        print(f"\n[Stage {i}] {stage} | modules={enabled_modules}")
        for seed in seeds:
            print(f"[Run] stage={stage} | seed={seed}")
            result = run_one_stage_seed(stage, i, enabled_modules, seed, args, out_root)
            all_seed_rows.append(result)
            print(
                "  -> final_v2v={:.4f}, final_v2i={:.4f}, conv_step={}, train_seconds={:.1f}".format(
                    result["final_v2v"],
                    result["final_v2i"],
                    result["convergence_step"],
                    result["train_seconds"],
                )
            )

    mean_rows = aggregate_results(all_seed_rows, stage_order=stages)
    delta_prev_rows = compute_delta_vs_prev(all_seed_rows, stage_order=stages)

    save_csv(out_root / "progressive_seed_results.csv", all_seed_rows)
    save_csv(out_root / "progressive_mean_std.csv", mean_rows)
    save_csv(out_root / "progressive_delta_prev_mean_std.csv", delta_prev_rows)

    with (out_root / "progressive_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "stages": stages,
                "stage_definition": stage_rows,
                "seeds": seeds,
                "modules": MODULES,
                "full_values": FULL_VALUES,
                "none_values": NONE_VALUES,
                "seed_results": all_seed_rows,
                "mean_std": mean_rows,
                "delta_prev_mean_std": delta_prev_rows,
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
        out_path=out_root / "progressive_v2v_bar.png",
    )
    plot_metric(
        mean_rows,
        metric_key="final_v2i_mean",
        err_key="final_v2i_std",
        ylabel="Final V2I Rate",
        out_path=out_root / "progressive_v2i_bar.png",
    )
    plot_metric(
        mean_rows,
        metric_key="conv_step_mean",
        err_key="conv_step_std",
        ylabel="Convergence Step",
        out_path=out_root / "progressive_convergence_bar.png",
    )

    plot_delta_prev(
        delta_prev_rows,
        metric_key="delta_v2v_mean",
        err_key="delta_v2v_std",
        ylabel="Delta V2V vs Previous Stage",
        out_path=out_root / "progressive_delta_prev_v2v_bar.png",
    )
    plot_delta_prev(
        delta_prev_rows,
        metric_key="delta_v2i_mean",
        err_key="delta_v2i_std",
        ylabel="Delta V2I vs Previous Stage",
        out_path=out_root / "progressive_delta_prev_v2i_bar.png",
    )

    print("\n[OK] Progressive ablation complete.")
    print(f"Stage definition : {out_root / 'progressive_stage_definition.csv'}")
    print(f"Seed-level CSV   : {out_root / 'progressive_seed_results.csv'}")
    print(f"Mean/std CSV     : {out_root / 'progressive_mean_std.csv'}")
    print(f"Delta-prev CSV   : {out_root / 'progressive_delta_prev_mean_std.csv'}")
    print(f"Summary JSON     : {out_root / 'progressive_summary.json'}")


if __name__ == "__main__":
    main()
