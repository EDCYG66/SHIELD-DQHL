"""Preset-driven ablations for the current SHIELD-DQHL training line."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class AblationPreset:
    name: str
    description: str
    category: str
    overrides: Dict[str, object]


STORE_TRUE_FLAGS = {
    "disable_communication",
    "skip_eval_plots",
    "skip_training_plots",
    "skip_eval_csv",
}

GOVERNOR_FLAGS = (
    "governor_mid_mpr_compact_pulse_enabled",
    "governor_low_risk_compact_staging_enabled",
    "governor_low_risk_split_to_eco_enabled",
    "governor_safe_event_standoff_eco_enabled",
    "governor_mid_pressure_gap_recover_enabled",
    "governor_near_event_gap_recover_enabled",
    "governor_unseen_eco_keep_guard_enabled",
    "governor_mid123_gap_recover_guard_enabled",
    "governor_mid122_keep_eco_guard_enabled",
    "governor_unseen_keep_eco_guard_enabled",
    "governor_seed129_keep_eco_guard_enabled",
    "governor_mid123_keep_eco_guard_enabled",
    "governor_long_event_prekeep_split_guard_enabled",
    "governor_calm_compact_restore_enabled",
    "training_guard_guidance_enabled",
)

BASE_PARAMS: Dict[str, object] = {
    "policy_type": "shield_dqhl",
    "comm_policy": "heuristic",
    "comm_gnn": "gat",
    "episodes": 200,
    "steps": 140,
    "seed": 126,
    "n_up": 10,
    "n_down": 10,
    "train_mpr_values": "0.35,0.40",
    "eval_mpr_values": "0.35,0.40",
    "eval_every": 5,
    "event_start": 39,
    "event_duration": 82,
    "event_center": 620,
    "event_length": 180,
    "batch_size": 128,
    "replay_size": 20000,
    "gamma": 0.95,
    "lr": 1e-3,
    "hidden_dims": "128,96",
    "target_update_interval": 100,
    "min_buffer_before_train": 256,
    "train_every_k_steps": 1,
    "train_updates_per_trigger": 1,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 6000,
    "mc_dropout_samples": 10,
    "uncertainty_beta_base": 0.30,
    "uncertainty_beta_max": 1.20,
    "shield_num_quantiles": 16,
    "shield_cvar_alpha": 0.25,
    "shield_risk_cvar_alpha": 0.15,
    "shield_aux_weight": 0.20,
    "shield_risk_replay_fraction": 0.82,
    "shield_safe_replay_fraction": 0.15,
    "shield_penalty_weight": 1.0,
    "shield_split_gap_penalty": 1.2,
    "shield_emergency_gap_threshold": 7.5,
    "shield_emergency_penalty_weight": 1.0,
    "shield_speed_keep_penalty_weight": 0.65,
    "shield_speed_keep_threshold_ratio": 0.82,
    "shield_compact_overuse_penalty_weight": 2.1,
    "shield_split_context_penalty_weight": 0.8,
    "shield_teacher_supervision_weight": 0.0,
    "shield_teacher_preference_weight": 0.0,
    "shield_teacher_preference_margin": 0.15,
    "shield_eco_bias_weight": 0.0,
    "shield_eco_bias_speed_ratio": 0.72,
    "shield_eco_bias_risk_ceiling": 0.32,
    "shield_eco_bias_gap_floor": 18.0,
    "shield_eco_bias_keep_scale": 0.45,
    "shield_eco_bias_compact_scale": 0.70,
    "governor_mid_mpr_compact_pulse_enabled": 1,
    "governor_low_risk_compact_staging_enabled": 1,
    "governor_low_risk_split_to_eco_enabled": 1,
    "governor_safe_event_standoff_eco_enabled": 1,
    "governor_mid_pressure_gap_recover_enabled": 1,
    "governor_near_event_gap_recover_enabled": 1,
    "governor_unseen_eco_keep_guard_enabled": 0,
    "governor_mid123_gap_recover_guard_enabled": 0,
    "governor_mid122_keep_eco_guard_enabled": 0,
    "governor_unseen_keep_eco_guard_enabled": 0,
    "governor_seed129_keep_eco_guard_enabled": 0,
    "governor_mid123_keep_eco_guard_enabled": 1,
    "governor_long_event_prekeep_split_guard_enabled": 0,
    "governor_calm_compact_restore_enabled": 0,
    "training_guard_guidance_enabled": 0,
    "reward_energy_weight": 0.043,
    "reward_speed_excess_weight": 0.0,
    "reward_accel_weight": 0.014,
    "reward_energy_reference_kj": 0.20,
    "reward_speed_target_ratio": 0.82,
    "reward_eco_keep_bonus": 0.068,
    "reward_gap_recover_bonus_weight": 0.44,
    "reward_eco_context_bonus_weight": 0.20,
    "reward_eco_postevent_bonus_weight": 0.24,
    "reward_keep_midpressure_penalty_weight": 0.0,
    "reward_compact_risk_weight": 0.0,
    "reward_compact_risk_gap_ref": 14.0,
    "reward_compact_risk_event_ref": 0.30,
    "reward_compact_risk_blocked_ref": 0.15,
    "reward_collision_count_weight": 3.0,
    "reward_collision_active_weight": 1.5,
    "reward_emergency_brake_weight": 0.30,
    "reward_shield_intervention_weight": 0.05,
    "reward_tight_gap_weight": 0.20,
    "reward_safe_gap_bonus": 0.08,
    "warmstart_heuristic_episodes": 8,
    "skip_eval_plots": True,
    "skip_training_plots": True,
}

PRESETS: Dict[str, AblationPreset] = {
    "full": AblationPreset(
        name="full",
        category="reference",
        description="Current seed126 current-config SHIELD-DQHL reference.",
        overrides={},
    ),
    "no_cvar": AblationPreset(
        name="no_cvar",
        category="method",
        description="Disable CVaR action selection and fall back to mean-quantile selection.",
        overrides={
            "shield_cvar_alpha": 1.0,
            "shield_risk_cvar_alpha": 1.0,
        },
    ),
    "no_risk_replay": AblationPreset(
        name="no_risk_replay",
        category="method",
        description="Turn off risk/safe bucket replay prioritization and use near-uniform replay.",
        overrides={
            "shield_risk_replay_fraction": 0.0,
            "shield_safe_replay_fraction": 0.0,
        },
    ),
    "no_shield_priors": AblationPreset(
        name="no_shield_priors",
        category="method",
        description="Remove shield auxiliary loss and policy-side shield penalty/priors.",
        overrides={
            "shield_aux_weight": 0.0,
            "shield_penalty_weight": 0.0,
            "shield_split_gap_penalty": 0.0,
            "shield_emergency_penalty_weight": 0.0,
            "shield_speed_keep_penalty_weight": 0.0,
            "shield_compact_overuse_penalty_weight": 0.0,
            "shield_split_context_penalty_weight": 0.0,
        },
    ),
    "no_governor": AblationPreset(
        name="no_governor",
        category="method",
        description="Disable all SHIELD-DQHL governor hooks and scene-specific post-processing guards.",
        overrides={flag: 0 for flag in GOVERNOR_FLAGS},
    ),
    "no_reward_shaping": AblationPreset(
        name="no_reward_shaping",
        category="method",
        description="Remove the extra eco/energy/action-shaping reward terms while keeping core safety losses.",
        overrides={
            "reward_energy_weight": 0.0,
            "reward_accel_weight": 0.0,
            "reward_eco_keep_bonus": 0.0,
            "reward_gap_recover_bonus_weight": 0.0,
            "reward_eco_context_bonus_weight": 0.0,
            "reward_eco_postevent_bonus_weight": 0.0,
            "reward_shield_intervention_weight": 0.0,
            "reward_tight_gap_weight": 0.0,
            "reward_safe_gap_bonus": 0.0,
        },
    ),
    "no_communication": AblationPreset(
        name="no_communication",
        category="system",
        description="System-level ablation that disables the communication module entirely.",
        overrides={
            "disable_communication": True,
        },
    ),
}


def build_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run preset SHIELD-DQHL ablations.")
    parser.add_argument("--preset", type=str, default="full", help="Preset name or 'all'.")
    parser.add_argument("--episodes", type=int, default=int(BASE_PARAMS["episodes"]))
    parser.add_argument("--steps", type=int, default=int(BASE_PARAMS["steps"]))
    parser.add_argument("--seed", type=int, default=int(BASE_PARAMS["seed"]))
    parser.add_argument("--eval-every", type=int, default=int(BASE_PARAMS["eval_every"]))
    parser.add_argument("--out-dir", type=str, default="", help="Directory for a single run, or root directory for '--preset all'.")
    parser.add_argument("--list-presets", action="store_true", help="Print available presets and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print the generated command(s) without running them.")
    parser.add_argument("--gpu-optimized", action="store_true", help="Use gpu_optimized.launcher instead of the serial training entrypoint.")
    parser.add_argument("--optimization-level", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--cupy-mem-limit-gb", type=float, default=1.0)
    parser.add_argument("--comm-backend", type=str, default="auto", choices=["numpy", "cupy", "auto"])
    parser.add_argument("--vec-step-mode", type=str, default="auto", choices=["auto", "sequential", "threaded", "process"])
    parser.add_argument("--resource-log-interval", type=float, default=5.0)
    return parser


def _param_to_cli(params: Dict[str, object]) -> List[str]:
    args: List[str] = []
    for key, value in params.items():
        cli_key = f"--{key.replace('_', '-')}"
        if key in STORE_TRUE_FLAGS:
            if bool(value):
                args.append(cli_key)
            continue
        args.extend([cli_key, str(value)])
    return args


def _build_params(preset: AblationPreset, cli_args: argparse.Namespace, out_dir: Path) -> Dict[str, object]:
    params = dict(BASE_PARAMS)
    params.update(
        episodes=int(cli_args.episodes),
        steps=int(cli_args.steps),
        seed=int(cli_args.seed),
        eval_every=int(cli_args.eval_every),
        out_dir=str(out_dir),
    )
    params.update(preset.overrides)
    return params


def _default_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("tmp") / "shield_dqhl_ablations" / f"seed126_currentcfg_{stamp}"


def _resolve_run_targets(cli_args: argparse.Namespace) -> List[tuple[AblationPreset, Path]]:
    if cli_args.preset == "all":
        root = Path(cli_args.out_dir) if cli_args.out_dir else _default_root()
        return [(preset, root / preset.name) for preset in PRESETS.values()]

    if cli_args.preset not in PRESETS:
        valid = ", ".join(["all", *PRESETS.keys()])
        raise SystemExit(f"Unknown preset '{cli_args.preset}'. Valid options: {valid}")

    preset = PRESETS[cli_args.preset]
    if cli_args.out_dir:
        run_dir = Path(cli_args.out_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("tmp") / "shield_dqhl_ablations" / f"{preset.name}_ep{cli_args.episodes}_{stamp}"
    return [(preset, run_dir)]


def _format_preset_lines(presets: Iterable[AblationPreset]) -> str:
    lines = []
    for preset in presets:
        lines.append(f"{preset.name}: [{preset.category}] {preset.description}")
    return "\n".join(lines)


def main() -> None:
    cli_args = build_wrapper_parser().parse_args()
    if cli_args.list_presets:
        print(_format_preset_lines(PRESETS.values()))
        return

    targets = _resolve_run_targets(cli_args)
    failures: List[str] = []

    for preset, out_dir in targets:
        params = _build_params(preset, cli_args, out_dir)
        if cli_args.gpu_optimized:
            cmd = [
                sys.executable,
                "-m",
                "gpu_optimized.launcher",
                "--optimization-level",
                str(cli_args.optimization_level),
                "--n-envs",
                str(cli_args.n_envs),
                "--cupy-mem-limit-gb",
                str(cli_args.cupy_mem_limit_gb),
                "--comm-backend",
                str(cli_args.comm_backend),
                "--vec-step-mode",
                str(cli_args.vec_step_mode),
                "--resource-log-interval",
                str(cli_args.resource_log_interval),
                *_param_to_cli(params),
            ]
        else:
            train_script = Path(__file__).with_name("run_trainable_high_level_policy.py")
            cmd = [sys.executable, str(train_script), *_param_to_cli(params)]
        print(f"[Ablation] {preset.name}: {preset.description}")
        print("[Ablation] Command:")
        print("  " + shlex.join(cmd))
        print(f"[Ablation] Output: {out_dir}")
        if cli_args.dry_run:
            continue
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures.append(preset.name)
            print(f"[Ablation] {preset.name} failed with exit code {result.returncode}", file=sys.stderr)

    if failures:
        raise SystemExit(f"Failed presets: {', '.join(failures)}")


if __name__ == "__main__":
    main()
