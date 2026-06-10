"""Run a pure formation simulation and export paper-ready visualizations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    from .events import EventScheduler, build_bottleneck_event, build_staged_incident_events
    from .formation_env import FormationExperimentEnv
    from .high_level_policy import build_policy
    from .platoon_metrics import PlatoonMetricsTracker
    from .progress_utils import progress_range
    from .scenario_factory import build_scheduler_from_spec, build_segment_adjustments, select_scenarios
    from .scenario_renderer import SimulationHistory, capture_frame, export_animation, save_scene_compare, save_scene_triptych
    from .trajectory_visualizer import plot_timeseries_dashboard, plot_trajectory_overview
except ImportError:  # pragma: no cover
    from events import EventScheduler, build_bottleneck_event, build_staged_incident_events
    from formation_env import FormationExperimentEnv
    from high_level_policy import build_policy
    from platoon_metrics import PlatoonMetricsTracker
    from progress_utils import progress_range
    from scenario_factory import build_scheduler_from_spec, build_segment_adjustments, select_scenarios
    from scenario_renderer import SimulationHistory, capture_frame, export_animation, save_scene_compare, save_scene_triptych
    from trajectory_visualizer import plot_timeseries_dashboard, plot_trajectory_overview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pure-simulation formation visualization runner")
    parser.add_argument("--steps", type=int, default=180)
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
    parser.add_argument("--scenario", type=str, default="", help="Optional preset scenario name from scenario_factory.py")
    parser.add_argument("--event-start", type=int, default=60)
    parser.add_argument("--event-duration", type=int, default=100)
    parser.add_argument("--event-center", type=float, default=700.0)
    parser.add_argument("--event-length", type=float, default=160.0)
    parser.add_argument("--event-speed-scale", type=float, default=0.40)
    parser.add_argument("--staged-event", action="store_true")
    parser.add_argument("--policy", type=str, default="comm_aware", choices=["heuristic", "comm_aware", "conservative", "learned", "vanilla_ddqn", "mobil_cidm_cacc", "ppo"])
    parser.add_argument("--policy-weights", type=str, default="")
    parser.add_argument("--policy-meta", type=str, default="")
    parser.add_argument("--comm-gnn", type=str, default="gat", choices=["gat", "gatclassic", "sage", "fc"])
    parser.add_argument("--comm-policy", type=str, default="auto", choices=["auto", "agent", "heuristic"])
    parser.add_argument("--comm-dqn-weights", type=str, default="")
    parser.add_argument("--comm-gnn-weights", type=str, default="")
    parser.add_argument("--comm-weights-dir", type=str, default="")
    parser.add_argument("--disable-communication", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--gif-fps", type=int, default=6)
    parser.add_argument("--mp4-fps", type=int, default=8)
    parser.add_argument("--skip-gif", action="store_true")
    parser.add_argument("--skip-mp4", action="store_true")
    parser.add_argument("--hide-v2i", action="store_true")
    parser.add_argument("--triptych-only", action="store_true", help="Only export the three-panel scene summary.")
    parser.add_argument("--out-dir", type=str, default="")
    return parser


def make_scheduler(args) -> tuple[EventScheduler, Optional[str], Dict[str, object]]:
    if args.scenario:
        spec = select_scenarios([args.scenario], height=args.height)[0]
        scheduler = build_scheduler_from_spec(spec)
        meta = {
            "scenario_name": spec.name,
            "event_start": spec.event_start,
            "event_duration": spec.event_duration,
            "event_center": spec.event_center,
            "event_length": spec.event_length,
            "event_speed_scale": spec.event_speed_scale,
            "blocked_lanes": {key: list(value) for key, value in spec.blocked_lanes.items()},
        }
        return scheduler, spec.name, meta

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
    meta = {
        "scenario_name": "custom_staged" if args.staged_event else "custom",
        "event_start": args.event_start,
        "event_duration": args.event_duration,
        "event_center": args.event_center,
        "event_length": args.event_length,
        "event_speed_scale": args.event_speed_scale,
        "blocked_lanes": {"u": [1], "d": [1]},
        "staged_event": bool(args.staged_event),
    }
    return scheduler, None, meta


def run_visualization(args) -> Path:
    scheduler, scenario_name, scenario_meta = make_scheduler(args)
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "visual_runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_spec = None
    road_segments = None
    if scenario_name:
        scenario_spec = select_scenarios([scenario_name], height=args.height)[0]
        road_segments = list(scenario_spec.segments)
    else:
        from .scenario_factory import highway_scenario_spec
        scenario_spec = highway_scenario_spec(height=args.height)
        road_segments = list(scenario_spec.segments)
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
        road_segments=road_segments,
        scenario_spec=scenario_spec,
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
    history = SimulationHistory.from_env(
        env,
        policy=policy.name,
        scenario=scenario_name or "custom",
        communication_enabled=not args.disable_communication,
        comm_gnn=args.comm_gnn,
        comm_policy=args.comm_policy,
        topology_initial=args.topology,
    )
    history.frames.append(
        capture_frame(
            env,
            step=0,
            action="init",
            reward=0.0,
            state_fields=initial_state["fields"],
            info={},
        )
    )

    step_bar = progress_range(args.steps, desc="Formation visualization", unit="step")
    for step in step_bar:
        state = env.current_state()
        decision = policy.select_action(step, scheduler, state["fields"], state_vector=state["vector"], training=False)
        next_state, reward, _, info = env.step(decision.action)
        info["policy_score"] = float(decision.score)
        metrics.update(step=step, state_fields=next_state["fields"], info=info, action=decision.action, reward=reward)
        if step == 0 or (step + 1) == args.steps or (step + 1) % max(1, args.steps // 12) == 0:
            step_bar.set_postfix(
                action=decision.action,
                reward=f"{reward:.3f}",
                event=int(info.get("event_zone_vehicle_count", 0.0)),
                v2v=f"{next_state['fields'].get('comm_v2v_success', 0.0):.3f}",
            )
        history.frames.append(
            capture_frame(
                env,
                step=step + 1,
                action=decision.action,
                reward=reward,
                state_fields=next_state["fields"],
                info=info,
            )
        )
    step_bar.write("Simulation complete, exporting figures...")
    step_bar.close()

    summary = metrics.save(out_dir)
    outputs = {
        "scene_triptych": str(save_scene_triptych(history, out_dir, show_v2i=not args.hide_v2i)),
    }
    if not args.triptych_only:
        outputs.update(
            {
                "scene_compare": str(save_scene_compare(history, out_dir, show_v2i=not args.hide_v2i)),
                "trajectory_overview": str(plot_trajectory_overview(history, out_dir)),
                "timeseries_dashboard": str(plot_timeseries_dashboard(history, out_dir)),
            }
        )
        outputs.update(
            export_animation(
                history,
                out_dir,
                show_v2i=not args.hide_v2i,
                frame_stride=args.frame_stride,
                gif_fps=args.gif_fps,
                mp4_fps=args.mp4_fps,
                write_gif=not args.skip_gif,
                write_mp4=not args.skip_mp4,
            )
        )

    manifest = {
        "policy": policy.name,
        "steps": args.steps,
        "seed": args.seed,
        "scenario": scenario_meta,
        "summary": summary,
        "outputs": outputs,
    }
    with (out_dir / "visualization_manifest.json").open("w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, indent=2)

    print(f"Pure-simulation visualization suite saved to: {out_dir}")
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return out_dir


def main() -> None:
    args = build_parser().parse_args()
    run_visualization(args)


if __name__ == "__main__":
    main()
