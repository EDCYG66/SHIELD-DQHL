"""Runs a baseline platoon-reconfiguration experiment on top of HighwayTopoEnv."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

try:
    from .events import EventScheduler, build_bottleneck_event, build_staged_incident_events
    from .formation_env import FormationExperimentEnv
    from .high_level_policy import build_policy
    from .platoon_metrics import PlatoonMetricsTracker
    from .progress_utils import progress_range
except ImportError:  # pragma: no cover
    from events import EventScheduler, build_bottleneck_event, build_staged_incident_events
    from formation_env import FormationExperimentEnv
    from high_level_policy import build_policy
    from platoon_metrics import PlatoonMetricsTracker
    from progress_utils import progress_range


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline formation experiment")
    parser.add_argument("--steps", type=int, default=220)
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
    parser.add_argument("--event-start", type=int, default=60)
    parser.add_argument("--event-duration", type=int, default=100)
    parser.add_argument("--event-center", type=float, default=700.0)
    parser.add_argument("--event-length", type=float, default=160.0)
    parser.add_argument("--event-speed-scale", type=float, default=0.40)
    parser.add_argument("--staged-event", action="store_true")
    parser.add_argument("--policy", type=str, default="heuristic", choices=["heuristic", "comm_aware", "conservative", "learned", "vanilla_ddqn", "mobil_cidm_cacc", "ppo"])
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


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.staged_event:
        scheduler = EventScheduler(
            build_staged_incident_events(
                start_step=args.event_start,
                duration=args.event_duration,
                y_center=args.event_center,
                zone_length=args.event_length,
                speed_limit_ratio=args.event_speed_scale,
                blocked_lanes={"u": (1,), "d": (1,)},
            )
        )
    else:
        scheduler = EventScheduler([
            build_bottleneck_event(
                start_step=args.event_start,
                duration=args.event_duration,
                y_center=args.event_center,
                zone_length=args.event_length,
                speed_limit_ratio=args.event_speed_scale,
                blocked_lanes={"u": (1,), "d": (1,)},
            )
        ])

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
        communication_run_dir=str(out_dir / "comm_agent"),
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

    env.export_snapshot(out_dir / "formation_initial.png", show_v2i=True)

    step_bar = progress_range(args.steps, desc=f"Baseline {policy.name}", unit="step")
    for step in step_bar:
        state = env.current_state()
        decision = policy.select_action(step, scheduler, state["fields"], state_vector=state["vector"], training=False)
        state, reward, _, info = env.step(decision.action)
        info["policy_score"] = float(decision.score)
        metrics.update(step=step, state_fields=state["fields"], info=info, action=decision.action, reward=reward)
        if step == 0 or (step + 1) == args.steps or (step + 1) % max(1, args.steps // 10) == 0:
            step_bar.set_postfix(
                action=decision.action,
                reward=f"{reward:.3f}",
                gap=f"{min(state['fields'].get('min_gap_up', 0.0), state['fields'].get('min_gap_down', 0.0)):.1f}",
                v2v=f"{state['fields'].get('comm_v2v_success', 0.0):.3f}",
            )
    step_bar.close()

    env.export_snapshot(out_dir / "formation_final.png", show_v2i=True)
    summary = metrics.save(out_dir)
    summary["policy"] = policy.name

    print("Formation baseline run complete.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
