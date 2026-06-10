"""Batch evaluation across multiple event scenarios."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    from .formation_env import FormationExperimentEnv
    from .high_level_policy import build_policy
    from .plot_joint_results import plot_scenario_sweep_summary
    from .platoon_metrics import PlatoonMetricsTracker
    from .progress_utils import progress_iter, progress_range
    from .scenario_factory import build_scheduler_from_spec, default_scenarios, select_scenarios
except ImportError:  # pragma: no cover
    from formation_env import FormationExperimentEnv
    from high_level_policy import build_policy
    from plot_joint_results import plot_scenario_sweep_summary
    from platoon_metrics import PlatoonMetricsTracker
    from progress_utils import progress_iter, progress_range
    from scenario_factory import build_scheduler_from_spec, default_scenarios, select_scenarios


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batch joint evaluation over multiple scenarios")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-up", type=int, default=12)
    parser.add_argument("--n-down", type=int, default=12)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--spacing", type=float, default=20.0)
    parser.add_argument("--height", type=float, default=1400.0)
    parser.add_argument("--mpr-cav", type=float, default=0.50)
    parser.add_argument("--random-spawn", action="store_true")
    parser.add_argument("--spawn-y-min", type=float, default=0.0)
    parser.add_argument("--spawn-y-max", type=float, default=260.0)
    parser.add_argument("--lane-density-jitter", type=float, default=0.35)
    parser.add_argument("--topology", type=str, default="star", choices=["star", "tree"])
    parser.add_argument("--policy", type=str, default="heuristic", choices=["heuristic", "comm_aware", "conservative", "learned", "vanilla_ddqn", "mobil_cidm_cacc", "ppo"])
    parser.add_argument("--policy-weights", type=str, default="")
    parser.add_argument("--policy-meta", type=str, default="")
    parser.add_argument("--scenarios", type=str, default="")
    parser.add_argument("--comm-gnn", type=str, default="gat", choices=["gat", "gatclassic", "sage", "fc"])
    parser.add_argument("--comm-policy", type=str, default="agent", choices=["auto", "agent", "heuristic"])
    parser.add_argument("--comm-dqn-weights", type=str, default="")
    parser.add_argument("--comm-gnn-weights", type=str, default="")
    parser.add_argument("--comm-weights-dir", type=str, default="")
    parser.add_argument("--disable-communication", action="store_true")
    parser.add_argument("--out-dir", type=str, default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "scenario_sweeps" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scenarios.strip():
        scenario_specs = select_scenarios([item for item in args.scenarios.split(",") if item.strip()], height=args.height)
    else:
        scenario_specs = default_scenarios(height=args.height)

    summaries: List[Dict[str, float]] = []
    scenario_bar = progress_iter(scenario_specs, total=len(scenario_specs), desc="Scenario sweep", unit="scenario")
    for spec in scenario_bar:
        scenario_dir = out_dir / spec.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scheduler = build_scheduler_from_spec(spec)
        env = FormationExperimentEnv(
            scheduler=scheduler,
            n_up=args.n_up,
            n_down=args.n_down,
            lanes_per_dir=args.lanes,
            spacing=args.spacing,
            height=args.height,
            mpr_cav=args.mpr_cav,
            random_spawn=args.random_spawn,
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
            communication_run_dir=str(scenario_dir / "comm_agent"),
        )
        initial_state = env.current_state()
        policy = build_policy(
            args.policy,
            state_dim=len(initial_state["vector"]),
            seed=args.seed,
            weights_path=args.policy_weights or None,
            meta_path=args.policy_meta or None,
        )
        metrics = PlatoonMetricsTracker()
        env.export_snapshot(scenario_dir / "formation_initial.png", show_v2i=True)
        state = initial_state
        step_bar = progress_range(args.steps, desc=f"Scenario {spec.name}", unit="step", leave=False)
        for step in step_bar:
            decision = policy.select_action(step, scheduler, state["fields"], state_vector=state["vector"], training=False)
            next_state, reward, _, info = env.step(decision.action)
            info["policy_score"] = float(decision.score)
            metrics.update(step=step, state_fields=next_state["fields"], info=info, action=decision.action, reward=reward)
            state = next_state
            if step == 0 or (step + 1) == args.steps or (step + 1) % max(1, args.steps // 8) == 0:
                step_bar.set_postfix(
                    reward=f"{reward:.3f}",
                    event=int(info.get("event_zone_vehicle_count", 0.0)),
                    v2v=f"{next_state['fields'].get('comm_v2v_success', 0.0):.3f}",
                )
        step_bar.close()
        env.export_snapshot(scenario_dir / "formation_final.png", show_v2i=True)
        summary = metrics.save(scenario_dir)
        summary["scenario"] = spec.name
        summary["policy"] = args.policy
        summaries.append(summary)
        scenario_bar.set_postfix(
            scenario=spec.name,
            reward=f"{summary.get('avg_reward', 0.0):.3f}",
            v2v=f"{summary.get('avg_comm_v2v_success', 0.0):.3f}",
        )
        scenario_bar.write(
            f"[Scenario] {spec.name}: avg_reward={summary.get('avg_reward', 0.0):.4f}, "
            f"avg_comm={summary.get('avg_comm_v2v_success', 0.0):.4f}"
        )
    scenario_bar.close()

    fieldnames = sorted({key for row in summaries for key in row.keys()})
    csv_path = out_dir / "scenario_sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    with (out_dir / "scenario_sweep_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summaries, file_obj, indent=2)
    plot_scenario_sweep_summary(csv_path, out_dir)
    print(f"Scenario sweep saved to: {out_dir}")


if __name__ == "__main__":
    main()
