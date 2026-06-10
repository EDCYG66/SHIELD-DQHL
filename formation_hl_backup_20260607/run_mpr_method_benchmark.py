"""Batch benchmark across MPR levels and high-level method baselines."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import multiprocessing as mp
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Set, Tuple

# Reduce TensorFlow C++ info/debug log noise when benchmarking.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .events import EventScheduler, build_bottleneck_event, build_staged_incident_events
    from .formation_env import FormationExperimentEnv
    from .high_level_policy import PolicyDecision, build_policy
    from .mpr_utils import resolve_exact_vehicle_counts
    from .platoon_metrics import PlatoonMetricsTracker
    from .progress_utils import create_progress, progress_iter, progress_range
    from .safety_shield import NoOpSafetyShield, SafetyShield
except ImportError:  # pragma: no cover
    from events import EventScheduler, build_bottleneck_event, build_staged_incident_events
    from formation_env import FormationExperimentEnv
    from high_level_policy import PolicyDecision, build_policy
    from mpr_utils import resolve_exact_vehicle_counts
    from platoon_metrics import PlatoonMetricsTracker
    from progress_utils import create_progress, progress_iter, progress_range
    from safety_shield import NoOpSafetyShield, SafetyShield


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

EXPORT_DPI = 1000
METHOD_COLORS = {
    "heuristic": "#7FA6D8",
    "comm_aware": "#5DAE8B",
    "conservative": "#E7A95A",
    "no_reconfiguration": "#B9BEC7",
    "no_communication": "#C98383",
    "learned": "#8667C7",
}
TIME_METRIC_LABELS = {
    "formation": "Formation Time [s]",
    "reconfiguration": "Reconfiguration Time [s]",
    "safe_recovery": "Safe Recovery Time [s]",
    "event_clearance": "Event Clearance Time [s]",
}


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    policy_name: str
    disable_communication: bool = False
    shield_mode: str = "on"
    fixed_action: str | None = None
    requires_weights: bool = False


METHOD_LIBRARY: Dict[str, MethodSpec] = {
    "heuristic": MethodSpec("heuristic", "Heuristic", "heuristic"),
    "comm_aware": MethodSpec("comm_aware", "Comm-Aware", "comm_aware"),
    "conservative": MethodSpec("conservative", "Conservative", "conservative"),
    "mobil_cidm_cacc": MethodSpec("mobil_cidm_cacc", "MOBIL+C-IDM/CACC", "mobil_cidm_cacc"),
    "no_reconfiguration": MethodSpec("no_reconfiguration", "No-Reconfig", "heuristic", fixed_action="keep"),
    "no_communication": MethodSpec("no_communication", "No-Communication", "heuristic", disable_communication=True),
    "learned": MethodSpec("learned", "Learned-HL", "learned", requires_weights=True),
    "vanilla_ddqn": MethodSpec("vanilla_ddqn", "Vanilla-DDQN", "vanilla_ddqn", requires_weights=True),
    "ppo": MethodSpec("ppo", "PPO-HL", "ppo", requires_weights=True),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmark tables across MPR levels and method baselines")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-up", type=int, default=12)
    parser.add_argument("--n-down", type=int, default=12)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--spacing", type=float, default=20.0)
    parser.add_argument("--height", type=float, default=1400.0)
    parser.add_argument("--topology", type=str, default="star", choices=["star", "tree"])
    parser.add_argument("--mpr-values", type=str, default="0.30,0.35,0.375,0.40,0.50")
    parser.add_argument("--methods", type=str, default="heuristic,comm_aware,conservative,mobil_cidm_cacc,no_reconfiguration,no_communication")
    parser.add_argument("--policy-weights", type=str, default="")
    parser.add_argument("--policy-meta", type=str, default="")
    parser.add_argument("--random-spawn", action="store_true")
    parser.add_argument("--spawn-y-min", type=float, default=0.0)
    parser.add_argument("--spawn-y-max", type=float, default=520.0)
    parser.add_argument("--lane-density-jitter", type=float, default=0.35)
    parser.add_argument("--staged-event", action="store_true")
    parser.add_argument("--event-start", type=int, default=120)
    parser.add_argument("--event-duration", type=int, default=260)
    parser.add_argument("--event-center", type=float, default=620.0)
    parser.add_argument("--event-length", type=float, default=180.0)
    parser.add_argument("--event-speed-scale", type=float, default=0.38)
    parser.add_argument("--comm-gnn", type=str, default="gat", choices=["gat", "gatclassic", "sage", "fc"])
    parser.add_argument("--comm-policy", type=str, default="auto", choices=["auto", "agent", "heuristic"])
    parser.add_argument("--comm-dqn-weights", type=str, default="")
    parser.add_argument("--comm-gnn-weights", type=str, default="")
    parser.add_argument("--comm-weights-dir", type=str, default="")
    parser.add_argument("--time-metric", type=str, default="formation", choices=["formation", "reconfiguration", "safe_recovery", "event_clearance"])
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument(
        "--force-rerun",
        type=str,
        default="",
        help="Comma-separated benchmark combos to rerun even if already recorded. "
             "Accepts run names like mpr_375_learned or keys like 0.375:learned.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Run benchmark combos in parallel worker processes. Defaults to 1 (disabled).",
    )
    parser.add_argument(
        "--parallel-worker-device",
        type=str,
        default="cpu",
        choices=["cpu", "inherit"],
        help="Device visibility inside parallel benchmark workers. "
             "'cpu' hides GPUs to avoid contention; 'inherit' keeps the parent's device view.",
    )
    return parser


def parse_mpr_values(raw: str) -> List[float]:
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("At least one MPR value is required.")
    return values


def resolve_methods(raw: str, *, policy_weights: str) -> List[MethodSpec]:
    requested = [token.strip().lower() for token in str(raw).split(",") if token.strip()]
    if not requested:
        raise ValueError("At least one benchmark method is required.")
    methods: List[MethodSpec] = []
    for key in requested:
        if key not in METHOD_LIBRARY:
            raise ValueError(f"Unknown benchmark method: {key}")
        spec = METHOD_LIBRARY[key]
        if spec.requires_weights and not policy_weights:
            raise ValueError("Method 'learned' requires --policy-weights.")
        methods.append(spec)
    return methods


def make_scheduler(args) -> EventScheduler:
    if args.staged_event:
        events = build_staged_incident_events(
            start_step=args.event_start,
            duration=args.event_duration,
            y_center=args.event_center,
            zone_length=args.event_length,
            speed_limit_ratio=args.event_speed_scale,
            blocked_lanes={"u": (1,), "d": (1,)},
        )
        return EventScheduler(events)
    return EventScheduler(
        [
            build_bottleneck_event(
                start_step=args.event_start,
                duration=args.event_duration,
                y_center=args.event_center,
                zone_length=args.event_length,
                speed_limit_ratio=args.event_speed_scale,
                blocked_lanes={"u": (1,), "d": (1,)},
            )
        ]
    )


def choose_action(policy, state, scheduler, step: int, fixed_action: str | None) -> PolicyDecision:
    if fixed_action is not None:
        return PolicyDecision(fixed_action, score=0.0, reason="fixed benchmark action")
    return policy.select_action(step, scheduler, state["fields"], state_vector=state["vector"], training=False)


def metric_time_value(summary: Dict[str, float], metric_name: str) -> float:
    key_map = {
        "formation": "platoon_formation_time_s",
        "reconfiguration": "reconfiguration_time_s",
        "safe_recovery": "safe_recovery_time_s",
        "event_clearance": "event_clearance_time_s",
    }
    return float(summary.get(key_map[metric_name], -1.0))


def metric_time_label(metric_name: str) -> str:
    return TIME_METRIC_LABELS[metric_name]


def _mpr_label(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _save_png_pdf(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def _render_metric_grid(df: pd.DataFrame, metrics: Sequence[tuple[str, str]], out_base: Path, title: str) -> None:
    if df.empty:
        return
    mpr_order = list(dict.fromkeys(df["MPR"].astype(str).tolist()))
    method_order = list(dict.fromkeys(df["Method"].astype(str).tolist()))
    x = np.arange(len(mpr_order), dtype=float)
    width = 0.78 / float(max(1, len(method_order)))

    n_metrics = len(metrics)
    ncols = 2 if n_metrics > 1 else 1
    nrows = int(np.ceil(n_metrics / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.9 * nrows), facecolor="white")
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, (column, ylabel) in zip(axes, metrics):
        for midx, method in enumerate(method_order):
            offsets = x + (midx - 0.5 * (len(method_order) - 1)) * width
            values = []
            for mpr in mpr_order:
                row = df[(df["MPR"] == mpr) & (df["Method"] == method)]
                values.append(float(row.iloc[0][column]) if not row.empty else np.nan)
            ax.bar(
                offsets,
                values,
                width=width * 0.92,
                color=METHOD_COLORS.get(method.lower().replace("-", "_"), "#B9BEC7"),
                edgecolor="#5D6670",
                linewidth=0.8,
                label=method,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(mpr_order)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)

    for ax in axes[n_metrics:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(method_order)), frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=12, y=0.99)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.96))
    _save_png_pdf(fig, out_base)


def _render_table(df: pd.DataFrame, out_base: Path, title: str) -> None:
    if df.empty:
        return
    display_df = df.copy()
    if "MPR" in display_df.columns:
        prev = None
        shown = []
        for value in display_df["MPR"].tolist():
            if value == prev:
                shown.append("")
            else:
                shown.append(value)
                prev = value
        display_df["MPR"] = shown

    formatted = []
    for _, row in display_df.iterrows():
        cells = []
        for col in display_df.columns:
            value = row[col]
            if isinstance(value, float):
                if value < 0.0 and ("Time" in col or col == "Time [s]"):
                    cells.append("N/A")
                elif abs(value) >= 100.0:
                    cells.append(f"{value:.1f}")
                else:
                    cells.append(f"{value:.2f}")
            else:
                cells.append(str(value))
        formatted.append(cells)

    fig_height = max(2.8, 0.45 * (len(formatted) + 2))
    fig, ax = plt.subplots(figsize=(8.8, fig_height), facecolor="white")
    ax.axis("off")
    table = ax.table(
        cellText=formatted,
        colLabels=list(display_df.columns),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#4D5966")
        cell.set_linewidth(0.7 if row_idx == 0 else 0.5)
        if row_idx == 0:
            cell.set_facecolor("#EAF0F8")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")
    ax.set_title(title, fontsize=12, pad=12)
    fig.tight_layout()
    _save_png_pdf(fig, out_base)


def _combo_run_name(mpr: float, method_key: str) -> str:
    return f"mpr_{int(round(float(mpr) * 1000)):03d}_{str(method_key).strip().lower()}"


def _combo_identity_keys(mpr: float, method_key: str) -> Set[str]:
    method_key = str(method_key).strip().lower()
    run_name = _combo_run_name(mpr, method_key)
    mpr_text = f"{float(mpr):.6f}".rstrip("0").rstrip(".")
    return {
        run_name,
        f"{mpr_text}:{method_key}",
    }


def _parse_force_rerun(raw: str) -> Set[str]:
    tokens: Set[str] = set()
    for item in str(raw or "").split(","):
        token = item.strip().lower()
        if not token:
            continue
        tokens.add(token)
        if ":" in token:
            left, right = token.split(":", 1)
            try:
                normalized_left = f"{float(left):.6f}".rstrip("0").rstrip(".")
                tokens.add(f"{normalized_left}:{right.strip()}")
            except Exception:
                pass
    return tokens


def _load_existing_summaries(out_dir: Path) -> List[Dict[str, float | str]]:
    json_path = out_dir / "mpr_method_benchmark.json"
    csv_path = out_dir / "mpr_method_benchmark.csv"
    if json_path.exists():
        try:
            with json_path.open(encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            if isinstance(payload, list):
                return [dict(row) for row in payload if isinstance(row, dict)]
        except Exception:
            pass
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            return [dict(row) for row in reader]
    return []


def _summary_combo_keys(summary_row: Dict[str, float | str]) -> Set[str]:
    method_key = str(summary_row.get("method_key", "")).strip().lower()
    mpr_value = summary_row.get("mpr", "")
    try:
        mpr_float = float(mpr_value)
    except Exception:
        return set()
    return _combo_identity_keys(mpr_float, method_key)


def _normalize_summary_rows(
    rows: List[Dict[str, float | str]],
    *,
    method_rank: Dict[str, int],
    mpr_rank: Dict[float, int],
    time_metric_name: str,
    time_metric_label: str,
) -> List[Dict[str, float | str]]:
    normalized: List[Dict[str, float | str]] = []
    for raw_row in rows:
        row = dict(raw_row)
        try:
            mpr = float(row.get("mpr", row.get("mpr_percent", 0.0)) if row.get("mpr", "") != "" else float(row.get("mpr_percent", 0.0)) / 100.0)
        except Exception:
            continue
        method_key = str(row.get("method_key", "")).strip().lower()
        if not method_key:
            continue
        row["mpr"] = float(mpr)
        row["mpr_percent"] = 100.0 * float(mpr)
        row["MPR"] = _mpr_label(mpr)
        row["method_key"] = method_key
        if "Method" not in row or not str(row.get("Method", "")).strip():
            row["Method"] = METHOD_LIBRARY.get(method_key, MethodSpec(method_key, method_key, method_key)).label
        row["method_rank"] = float(method_rank.get(method_key, 10**6))
        row["mpr_rank"] = float(mpr_rank.get(float(mpr), 10**6))
        row["time_metric_name"] = str(row.get("time_metric_name", time_metric_name) or time_metric_name)
        row["time_metric_label"] = str(row.get("time_metric_label", time_metric_label) or time_metric_label)
        if "time_metric_s" in row:
            try:
                row["time_metric_s"] = float(row["time_metric_s"])
            except Exception:
                row["time_metric_s"] = -1.0
        normalized.append(row)
    return normalized


def _persist_summaries(out_dir: Path, summaries: List[Dict[str, float | str]]) -> None:
    full_df = pd.DataFrame(summaries)
    if not full_df.empty:
        full_df = full_df.sort_values(by=["mpr_rank", "method_rank"], ascending=[True, True]).reset_index(drop=True)
    full_csv = out_dir / "mpr_method_benchmark.csv"
    full_json = out_dir / "mpr_method_benchmark.json"
    full_df.to_csv(full_csv, index=False)
    with full_json.open("w", encoding="utf-8") as file_obj:
        json.dump(full_df.to_dict(orient="records"), file_obj, indent=2)


def _upsert_summary_row(
    summaries: List[Dict[str, float | str]],
    summary_row: Dict[str, float | str],
) -> List[Dict[str, float | str]]:
    target_keys = _summary_combo_keys(summary_row)
    if not target_keys:
        return summaries + [summary_row]
    filtered = [row for row in summaries if not (_summary_combo_keys(row) & target_keys)]
    filtered.append(summary_row)
    return filtered


def _temporary_env_var(key: str, value: str | None):
    class _EnvVarContext:
        def __enter__(self_inner):
            self_inner.previous = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            if self_inner.previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = self_inner.previous
            return False

    return _EnvVarContext()


def _benchmark_worker_initializer(disable_gpu: bool) -> None:
    if disable_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import tensorflow as tf  # pylint: disable=import-error
    except Exception:
        return
    if not disable_gpu:
        return
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def _args_to_worker_payload(args) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def _worker_args_namespace(args_payload: Dict[str, Any]):
    return SimpleNamespace(**copy.deepcopy(args_payload))


def _make_summary_row(
    summary: Dict[str, float],
    *,
    mpr: float,
    method: MethodSpec,
    resolved_total: int,
    resolved_n_up: int,
    resolved_n_down: int,
    method_rank: Dict[str, int],
    mpr_rank: Dict[float, int],
    time_metric_name: str,
    time_metric_label: str,
) -> Dict[str, float | str]:
    summary_row: Dict[str, float | str] = dict(summary)
    summary_row["mpr"] = float(mpr)
    summary_row["mpr_percent"] = 100.0 * float(mpr)
    summary_row["MPR"] = _mpr_label(mpr)
    summary_row["resolved_total_vehicles"] = float(resolved_total)
    summary_row["resolved_n_up"] = float(resolved_n_up)
    summary_row["resolved_n_down"] = float(resolved_n_down)
    summary_row["method_key"] = method.key
    summary_row["Method"] = method.label
    summary_row["method_rank"] = float(method_rank[method.key])
    summary_row["mpr_rank"] = float(mpr_rank[float(mpr)])
    summary_row["time_metric_name"] = time_metric_name
    summary_row["time_metric_label"] = time_metric_label
    summary_row["time_metric_s"] = metric_time_value(summary, time_metric_name)
    return summary_row


def _run_single_combo(
    args,
    *,
    mpr: float,
    method: MethodSpec,
    resolved_n_up: int,
    resolved_n_down: int,
    resolved_total: int,
    method_rank: Dict[str, int],
    mpr_rank: Dict[float, int],
    time_col: str,
    run_dir: Path,
    show_step_progress: bool,
) -> Dict[str, float | str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    scheduler = make_scheduler(args)
    shield = SafetyShield() if method.shield_mode == "on" else NoOpSafetyShield()
    env = FormationExperimentEnv(
        scheduler=scheduler,
        shield=shield,
        n_up=resolved_n_up,
        n_down=resolved_n_down,
        lanes_per_dir=args.lanes,
        spacing=args.spacing,
        height=args.height,
        mpr_cav=mpr,
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
        communication_enabled=not method.disable_communication,
        communication_gnn_type=args.comm_gnn,
        communication_policy_mode=args.comm_policy,
        communication_dqn_weights=args.comm_dqn_weights or None,
        communication_gnn_weights=args.comm_gnn_weights or None,
        communication_weights_dir=args.comm_weights_dir or None,
        communication_run_dir=str(run_dir / "comm_agent"),
    )
    initial_state = env.current_state()
    policy = build_policy(
        method.policy_name,
        state_dim=len(initial_state["vector"]),
        seed=args.seed,
        weights_path=args.policy_weights or None,
        meta_path=args.policy_meta or None,
    )
    metrics = PlatoonMetricsTracker()
    state = initial_state

    step_bar = (
        progress_range(args.steps, desc=f"{_mpr_label(mpr)}-{method.key}", unit="step", leave=False)
        if show_step_progress
        else None
    )
    step_iter = step_bar if step_bar is not None else range(args.steps)
    for step in step_iter:
        decision = choose_action(policy, state, scheduler, step, method.fixed_action)
        next_state, reward, _, info = env.step(decision.action)
        info["policy_score"] = float(decision.score)
        metrics.update(step=step, state_fields=next_state["fields"], info=info, action=decision.action, reward=reward)
        state = next_state
        if step_bar is not None and (
            step == 0 or (step + 1) == args.steps or (step + 1) % max(1, args.steps // 6) == 0
        ):
            step_bar.set_postfix(
                reward=f"{reward:.3f}",
                platoon=f"{next_state['fields'].get('platoon_rate', 0.0):.3f}",
                v2v=f"{next_state['fields'].get('comm_v2v_success', 0.0):.3f}",
            )
    if step_bar is not None:
        step_bar.close()

    summary = metrics.save(run_dir)
    return _make_summary_row(
        summary,
        mpr=float(mpr),
        method=method,
        resolved_total=resolved_total,
        resolved_n_up=resolved_n_up,
        resolved_n_down=resolved_n_down,
        method_rank=method_rank,
        mpr_rank=mpr_rank,
        time_metric_name=args.time_metric,
        time_metric_label=time_col,
    )


def _run_combo_worker(
    args_payload: Dict[str, Any],
    *,
    mpr: float,
    method: MethodSpec,
    resolved_n_up: int,
    resolved_n_down: int,
    resolved_total: int,
    method_rank: Dict[str, int],
    mpr_rank: Dict[float, int],
    time_col: str,
    run_dir: str,
) -> Dict[str, float | str]:
    args = _worker_args_namespace(args_payload)
    return _run_single_combo(
        args,
        mpr=float(mpr),
        method=method,
        resolved_n_up=int(resolved_n_up),
        resolved_n_down=int(resolved_n_down),
        resolved_total=int(resolved_total),
        method_rank=method_rank,
        mpr_rank=mpr_rank,
        time_col=time_col,
        run_dir=Path(run_dir),
        show_step_progress=False,
    )


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "mpr_benchmarks" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    mpr_values = parse_mpr_values(args.mpr_values)
    resolved_n_up, resolved_n_down, resolved_total = resolve_exact_vehicle_counts(args.n_up, args.n_down, mpr_values)
    methods = resolve_methods(args.methods, policy_weights=args.policy_weights)
    time_col = metric_time_label(args.time_metric)
    method_rank = {method.key: idx for idx, method in enumerate(methods)}
    mpr_rank = {float(mpr): idx for idx, mpr in enumerate(mpr_values)}
    force_rerun_tokens = _parse_force_rerun(args.force_rerun)
    summaries = _normalize_summary_rows(
        _load_existing_summaries(out_dir),
        method_rank=method_rank,
        mpr_rank=mpr_rank,
        time_metric_name=args.time_metric,
        time_metric_label=time_col,
    )
    completed_combo_keys: Set[str] = set()
    for row in summaries:
        completed_combo_keys.update(_summary_combo_keys(row))

    combo_plan = [(mpr, method) for mpr in mpr_values for method in methods]
    skipped_run_names: List[str] = []
    pending_run_names: List[str] = []
    for mpr, method in combo_plan:
        combo_keys = _combo_identity_keys(mpr, method.key)
        run_name = _combo_run_name(mpr, method.key)
        forced = bool(combo_keys & force_rerun_tokens)
        if (combo_keys & completed_combo_keys) and not forced:
            skipped_run_names.append(run_name)
        else:
            pending_run_names.append(run_name)

    print(
        f"[Resume] out_dir={out_dir} total={len(combo_plan)} "
        f"completed={len(skipped_run_names)} pending={len(pending_run_names)}"
    )
    if skipped_run_names:
        preview = ", ".join(skipped_run_names[:8])
        if len(skipped_run_names) > 8:
            preview += ", ..."
        print(f"[Resume] skipping existing combos: {preview}")
    if force_rerun_tokens:
        forced_preview = ", ".join(sorted(force_rerun_tokens))
        print(f"[Resume] force-rerun enabled for: {forced_preview}")
    parallel_workers = max(1, int(getattr(args, "parallel_workers", 1)))
    parallel_enabled = parallel_workers > 1
    parallel_device = str(getattr(args, "parallel_worker_device", "cpu")).strip().lower()
    if parallel_enabled:
        print(f"[Parallel] workers={parallel_workers} device={parallel_device}")

    combo_bar = create_progress(total=len(combo_plan), desc="MPR benchmark", unit="run")
    if not parallel_enabled:
        for mpr, method in combo_plan:
            run_name = _combo_run_name(mpr, method.key)
            combo_keys = _combo_identity_keys(mpr, method.key)
            forced = bool(combo_keys & force_rerun_tokens)
            if (combo_keys & completed_combo_keys) and not forced:
                combo_bar.write(f"[Resume] skip {run_name}")
                combo_bar.set_postfix(
                    mpr=_mpr_label(mpr),
                    method=method.key,
                    status="skip",
                )
                combo_bar.update(1)
                continue
            run_dir = out_dir / run_name
            summary_row = _run_single_combo(
                args,
                mpr=float(mpr),
                method=method,
                resolved_n_up=resolved_n_up,
                resolved_n_down=resolved_n_down,
                resolved_total=resolved_total,
                method_rank=method_rank,
                mpr_rank=mpr_rank,
                time_col=time_col,
                run_dir=run_dir,
                show_step_progress=True,
            )
            summaries = _upsert_summary_row(summaries, summary_row)
            completed_combo_keys.update(combo_keys)
            _persist_summaries(out_dir, summaries)
            combo_bar.set_postfix(
                mpr=_mpr_label(mpr),
                method=method.key,
                reward=f"{float(summary_row.get('avg_reward', 0.0)):.3f}",
                platoon=f"{float(summary_row.get('avg_platoon_rate', 0.0)):.3f}",
            )
            combo_bar.update(1)
    else:
        args_payload = _args_to_worker_payload(args)
        pending_plan: List[Tuple[float, MethodSpec]] = []
        for mpr, method in combo_plan:
            combo_keys = _combo_identity_keys(mpr, method.key)
            run_name = _combo_run_name(mpr, method.key)
            forced = bool(combo_keys & force_rerun_tokens)
            if (combo_keys & completed_combo_keys) and not forced:
                combo_bar.write(f"[Resume] skip {run_name}")
                combo_bar.set_postfix(
                    mpr=_mpr_label(mpr),
                    method=method.key,
                    status="skip",
                )
                combo_bar.update(1)
                continue
            pending_plan.append((float(mpr), method))

        worker_initializer = None
        worker_initargs: Tuple[Any, ...] = ()
        env_ctx = nullcontext()
        if parallel_device == "cpu":
            env_ctx = _temporary_env_var("CUDA_VISIBLE_DEVICES", "")
            worker_initializer = _benchmark_worker_initializer
            worker_initargs = (True,)

        with env_ctx:
            with ProcessPoolExecutor(
                max_workers=parallel_workers,
                mp_context=mp.get_context("spawn"),
                initializer=worker_initializer,
                initargs=worker_initargs,
            ) as executor:
                in_flight = {}
                next_plan_idx = 0

                while next_plan_idx < len(pending_plan) and len(in_flight) < parallel_workers:
                    mpr, method = pending_plan[next_plan_idx]
                    run_name = _combo_run_name(mpr, method.key)
                    future = executor.submit(
                        _run_combo_worker,
                        args_payload,
                        mpr=float(mpr),
                        method=method,
                        resolved_n_up=resolved_n_up,
                        resolved_n_down=resolved_n_down,
                        resolved_total=resolved_total,
                        method_rank=method_rank,
                        mpr_rank=mpr_rank,
                        time_col=time_col,
                        run_dir=str(out_dir / run_name),
                    )
                    in_flight[future] = (float(mpr), method)
                    next_plan_idx += 1

                while in_flight:
                    done_futures, _ = wait(tuple(in_flight.keys()), return_when=FIRST_COMPLETED)
                    for future in done_futures:
                        mpr, method = in_flight.pop(future)
                        summary_row = future.result()
                        combo_keys = _combo_identity_keys(mpr, method.key)
                        summaries = _upsert_summary_row(summaries, summary_row)
                        completed_combo_keys.update(combo_keys)
                        _persist_summaries(out_dir, summaries)
                        combo_bar.set_postfix(
                            mpr=_mpr_label(mpr),
                            method=method.key,
                            reward=f"{float(summary_row.get('avg_reward', 0.0)):.3f}",
                            platoon=f"{float(summary_row.get('avg_platoon_rate', 0.0)):.3f}",
                        )
                        combo_bar.update(1)

                        if next_plan_idx < len(pending_plan):
                            next_mpr, next_method = pending_plan[next_plan_idx]
                            next_run_name = _combo_run_name(next_mpr, next_method.key)
                            next_future = executor.submit(
                                _run_combo_worker,
                                args_payload,
                                mpr=float(next_mpr),
                                method=next_method,
                                resolved_n_up=resolved_n_up,
                                resolved_n_down=resolved_n_down,
                                resolved_total=resolved_total,
                                method_rank=method_rank,
                                mpr_rank=mpr_rank,
                                time_col=time_col,
                                run_dir=str(out_dir / next_run_name),
                            )
                            in_flight[next_future] = (float(next_mpr), next_method)
                            next_plan_idx += 1
    combo_bar.close()

    _persist_summaries(out_dir, summaries)
    full_df = pd.DataFrame(summaries)
    if not full_df.empty:
        full_df = full_df.sort_values(by=["mpr_rank", "method_rank"], ascending=[True, True]).reset_index(drop=True)

    platoon_table = full_df[["MPR", "Method"]].copy()
    platoon_table["Platoon Rate"] = full_df["avg_platoon_rate"].astype(float)
    platoon_table["Max Length"] = full_df["peak_max_platoon_length"].astype(float)
    platoon_table[time_col] = full_df["time_metric_s"].astype(float)
    platoon_table["LC"] = full_df["total_lane_changes"].astype(float)

    efficiency_table = full_df[["MPR", "Method"]].copy()
    efficiency_table["Average Speed (m/s)"] = full_df["avg_speed_all"].astype(float)
    efficiency_table["Total Energy (kJ)"] = full_df["total_energy_kj"].astype(float)

    platoon_csv = out_dir / "mpr_platoon_table.csv"
    efficiency_csv = out_dir / "mpr_efficiency_table.csv"
    platoon_table.to_csv(platoon_csv, index=False)
    efficiency_table.to_csv(efficiency_csv, index=False)

    _render_table(platoon_table, out_dir / "mpr_platoon_table", title="MPR vs Method Benchmark: Platoon Metrics")
    _render_table(efficiency_table, out_dir / "mpr_efficiency_table", title="MPR vs Method Benchmark: Speed and Energy")

    _render_metric_grid(
        platoon_table,
        [
            ("Platoon Rate", "Platoon Rate"),
            ("Max Length", "Maximum Platoon Length"),
            (time_col, time_col),
            ("LC", "Lane Changes"),
        ],
        out_dir / "mpr_platoon_metrics_plot",
        title="MPR Benchmark: Platoon-Oriented Metrics",
    )
    _render_metric_grid(
        efficiency_table,
        [
            ("Average Speed (m/s)", "Average Speed (m/s)"),
            ("Total Energy (kJ)", "Energy Consumption (kJ)"),
        ],
        out_dir / "mpr_efficiency_metrics_plot",
        title="MPR Benchmark: Efficiency Metrics",
    )
    print(f"MPR benchmark saved to: {out_dir}")


if __name__ == "__main__":
    main()
