"""Compare multiple high-level formation policies under the same event scenario."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    from .events import EventScheduler, build_bottleneck_event
    from .formation_env import FormationExperimentEnv
    from .high_level_policy import build_policy
    from .platoon_metrics import PlatoonMetricsTracker
    from .progress_utils import progress_iter, progress_range
except ImportError:  # pragma: no cover
    from events import EventScheduler, build_bottleneck_event
    from formation_env import FormationExperimentEnv
    from high_level_policy import build_policy
    from platoon_metrics import PlatoonMetricsTracker
    from progress_utils import progress_iter, progress_range


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare formation policies")
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-up", type=int, default=12)
    parser.add_argument("--n-down", type=int, default=12)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--spacing", type=float, default=20.0)
    parser.add_argument("--height", type=float, default=1400.0)
    parser.add_argument("--topology", type=str, default="star", choices=["star", "tree"])
    parser.add_argument("--event-start", type=int, default=60)
    parser.add_argument("--event-duration", type=int, default=100)
    parser.add_argument("--event-center", type=float, default=700.0)
    parser.add_argument("--event-length", type=float, default=160.0)
    parser.add_argument("--event-speed-scale", type=float, default=0.40)
    parser.add_argument("--policies", type=str, default="heuristic,comm_aware,conservative")
    parser.add_argument("--policy-weights", type=str, default="")
    parser.add_argument("--policy-meta", type=str, default="")
    parser.add_argument("--comm-gnn", type=str, default="gat", choices=["gat", "gatclassic", "sage", "fc"])
    parser.add_argument("--comm-policy", type=str, default="auto", choices=["auto", "agent", "heuristic"])
    parser.add_argument("--comm-dqn-weights", type=str, default="")
    parser.add_argument("--comm-gnn-weights", type=str, default="")
    parser.add_argument("--comm-weights-dir", type=str, default="")
    parser.add_argument("--disable-communication", action="store_true")
    parser.add_argument("--out-dir", type=str, default="")
    return parser


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


def run_policy(policy_name: str, args, root_out_dir: Path, *, show_progress: bool = True) -> Dict[str, float]:
    scheduler = make_scheduler(args)
    run_dir = root_out_dir / policy_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = FormationExperimentEnv(
        scheduler=scheduler,
        n_up=args.n_up,
        n_down=args.n_down,
        lanes_per_dir=args.lanes,
        spacing=args.spacing,
        height=args.height,
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
        communication_run_dir=str(run_dir / "comm_agent"),
    )
    initial_state = env.current_state()
    policy = build_policy(
        policy_name,
        state_dim=len(initial_state["vector"]),
        seed=args.seed,
        weights_path=args.policy_weights or None,
        meta_path=args.policy_meta or None,
    )
    run_dir = root_out_dir / policy.name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = PlatoonMetricsTracker()
    env.export_snapshot(run_dir / "formation_initial.png", show_v2i=True)

    step_bar = progress_range(args.steps, desc=f"Policy {policy.name}", unit="step", leave=False, disable=not show_progress)
    for step in step_bar:
        state = env.current_state()
        decision = policy.select_action(step, scheduler, state["fields"], state_vector=state["vector"], training=False)
        state, reward, _, info = env.step(decision.action)
        info["policy_score"] = float(decision.score)
        metrics.update(step=step, state_fields=state["fields"], info=info, action=decision.action, reward=reward)
        if step == 0 or (step + 1) == args.steps or (step + 1) % max(1, args.steps // 8) == 0:
            step_bar.set_postfix(
                action=decision.action,
                reward=f"{reward:.3f}",
                gap=f"{min(state['fields'].get('min_gap_up', 0.0), state['fields'].get('min_gap_down', 0.0)):.1f}",
                v2v=f"{state['fields'].get('comm_v2v_success', 0.0):.3f}",
            )
    step_bar.close()

    env.export_snapshot(run_dir / "formation_final.png", show_v2i=True)
    summary = metrics.save(run_dir)
    summary["policy"] = policy.name
    summary["topology_initial"] = args.topology
    with (run_dir / "policy_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
    return summary


def save_compare_summary(results: List[Dict[str, float]], out_dir: Path) -> None:
    csv_path = out_dir / "policy_compare_summary.csv"
    fieldnames = sorted({key for row in results for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    json_path = out_dir / "policy_compare_summary.json"
    with json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2)


def main() -> None:
    args = build_parser().parse_args()
    policy_names = [name.strip() for name in args.policies.split(",") if name.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "compare_runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    policy_bar = progress_iter(policy_names, total=len(policy_names), desc="Policy compare", unit="policy")
    for policy_name in policy_bar:
        summary = run_policy(policy_name, args, out_dir, show_progress=True)
        results.append(summary)
        policy_bar.set_postfix(
            policy=policy_name,
            reward=f"{summary.get('avg_reward', 0.0):.3f}",
            v2v=f"{summary.get('avg_comm_v2v_success', 0.0):.3f}",
        )
        policy_bar.write(
            f"[Compare] {policy_name}: avg_reward={summary.get('avg_reward', 0.0):.4f}, "
            f"avg_comm_v2v_success={summary.get('avg_comm_v2v_success', 0.0):.4f}, "
            f"worst_min_gap_up={summary.get('worst_min_gap_up', 0.0):.4f}"
        )
    policy_bar.close()

    save_compare_summary(results, out_dir)
    print(f"Policy comparison saved to: {out_dir}")


if __name__ == "__main__":
    main()
