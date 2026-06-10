#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge progressive ablation outputs produced by run_progressive_ablation_study.py.

Typical use:
python merge_progressive_ablation.py \
  --inputs runs/ablation_progressive_seed123,runs/ablation_progressive_seed124,runs/ablation_progressive_seed125 \
  --out-dir runs/ablation_progressive_merged
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_int(row: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except Exception:
        return int(default)


def as_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def normalize_seed_row(row: Dict[str, str], source_dir: str) -> Dict[str, Any]:
    return {
        "stage": str(row.get("stage", "")).strip(),
        "stage_index": as_int(row, "stage_index", 0),
        "enabled_modules": str(row.get("enabled_modules", "")).strip(),
        "seed": as_int(row, "seed", -1),
        "final_v2i": as_float(row, "final_v2i", 0.0),
        "final_v2v": as_float(row, "final_v2v", 0.0),
        "final_fail": as_float(row, "final_fail", 1.0),
        "convergence_step": as_float(row, "convergence_step", -1.0),
        "mean_used_rb": as_float(row, "mean_used_rb", 0.0),
        "train_seconds": as_float(row, "train_seconds", 0.0),
        "run_dir": str(row.get("run_dir", "")).strip(),
        "source_dir": source_dir,
    }


def rows_close(a: Dict[str, Any], b: Dict[str, Any], tol: float = 1e-9) -> bool:
    same_scalar = (
        a["stage"] == b["stage"]
        and int(a["stage_index"]) == int(b["stage_index"])
        and a["enabled_modules"] == b["enabled_modules"]
        and int(a["seed"]) == int(b["seed"])
        and a["run_dir"] == b["run_dir"]
    )
    if not same_scalar:
        return False
    keys = ["final_v2i", "final_v2v", "final_fail", "convergence_step", "mean_used_rb", "train_seconds"]
    for k in keys:
        if abs(float(a[k]) - float(b[k])) > tol:
            return False
    return True


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


def infer_stage_order(
    stage_def_rows: List[Dict[str, Any]],
    seed_rows: List[Dict[str, Any]],
    user_stages: List[str],
) -> List[str]:
    if user_stages:
        return user_stages

    if stage_def_rows:
        tmp = sorted(stage_def_rows, key=lambda r: int(r["stage_index"]))
        out = []
        seen = set()
        for r in tmp:
            s = str(r["stage"])
            if s not in seen:
                out.append(s)
                seen.add(s)
        if out:
            return out

    by_stage: Dict[str, int] = {}
    for r in seed_rows:
        s = str(r["stage"])
        idx = int(r["stage_index"])
        if s in by_stage and by_stage[s] != idx:
            raise ValueError(f"Inconsistent stage_index for stage '{s}': {by_stage[s]} vs {idx}")
        by_stage[s] = idx
    return [k for k, _ in sorted(by_stage.items(), key=lambda x: x[1])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge progressive ablation results from multiple output folders.")
    parser.add_argument("--inputs", type=str, required=True, help="Comma-separated run directories to merge.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for merged results.")
    parser.add_argument("--stages", type=str, default="", help="Optional stage order override.")
    parser.add_argument("--allow-missing-stage-def", action="store_true", help="Allow missing progressive_stage_definition.csv.")

    args = parser.parse_args()

    input_dirs = [Path(x).expanduser().resolve() for x in parse_str_list(args.inputs)]
    if not input_dirs:
        raise ValueError("No input directories provided.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stage_defs: List[Dict[str, Any]] = []
    raw_seed_rows: List[Dict[str, Any]] = []

    for d in input_dirs:
        seed_csv = d / "progressive_seed_results.csv"
        stage_csv = d / "progressive_stage_definition.csv"
        if not seed_csv.exists():
            raise FileNotFoundError(f"Missing file: {seed_csv}")
        if (not stage_csv.exists()) and (not args.allow_missing_stage_def):
            raise FileNotFoundError(f"Missing file: {stage_csv} (or pass --allow-missing-stage-def)")

        for r in read_csv(stage_csv):
            all_stage_defs.append({
                "stage": str(r.get("stage", "")).strip(),
                "stage_index": as_int(r, "stage_index", 0),
                "enabled_modules": str(r.get("enabled_modules", "")).strip(),
                "source_dir": str(d),
            })

        for r in read_csv(seed_csv):
            raw_seed_rows.append(normalize_seed_row(r, str(d)))

    stage_order = infer_stage_order(
        stage_def_rows=all_stage_defs,
        seed_rows=raw_seed_rows,
        user_stages=parse_str_list(args.stages),
    )

    # Stage metadata map from stage definition and seed rows.
    stage_meta: Dict[str, Dict[str, Any]] = {}
    for r in all_stage_defs:
        s = r["stage"]
        idx = int(r["stage_index"])
        em = r["enabled_modules"]
        if s in stage_meta:
            if stage_meta[s]["stage_index"] != idx:
                raise ValueError(f"Stage index conflict for '{s}': {stage_meta[s]['stage_index']} vs {idx}")
            if stage_meta[s]["enabled_modules"] != em:
                raise ValueError(f"enabled_modules conflict for '{s}' across inputs.")
        else:
            stage_meta[s] = {"stage": s, "stage_index": idx, "enabled_modules": em}

    for r in raw_seed_rows:
        s = r["stage"]
        if s not in stage_meta:
            stage_meta[s] = {"stage": s, "stage_index": int(r["stage_index"]), "enabled_modules": r["enabled_modules"]}

    # Deduplicate by (stage, seed).
    dedup: Dict[Tuple[str, int], Dict[str, Any]] = {}
    duplicates_info: List[Dict[str, Any]] = []
    for r in raw_seed_rows:
        key = (r["stage"], int(r["seed"]))
        if key not in dedup:
            dedup[key] = r
            continue
        if rows_close(dedup[key], r):
            duplicates_info.append({"key": f"{key}", "kept_from": dedup[key]["source_dir"], "dup_from": r["source_dir"]})
            continue
        raise ValueError(
            f"Conflicting duplicate row for stage={key[0]}, seed={key[1]}.\n"
            f"Existing source: {dedup[key]['source_dir']}\n"
            f"New source     : {r['source_dir']}"
        )

    seed_rows = list(dedup.values())
    seed_rows.sort(key=lambda x: (stage_order.index(x["stage"]) if x["stage"] in stage_order else 10**9, int(x["seed"])))

    # Keep output columns aligned with producer script.
    seed_rows_out = []
    for r in seed_rows:
        seed_rows_out.append({
            "stage": r["stage"],
            "stage_index": int(r["stage_index"]),
            "enabled_modules": r["enabled_modules"],
            "seed": int(r["seed"]),
            "final_v2i": float(r["final_v2i"]),
            "final_v2v": float(r["final_v2v"]),
            "final_fail": float(r["final_fail"]),
            "convergence_step": float(r["convergence_step"]),
            "mean_used_rb": float(r["mean_used_rb"]),
            "train_seconds": float(r["train_seconds"]),
            "run_dir": r["run_dir"],
        })

    stage_def_out = []
    for s in stage_order:
        meta = stage_meta.get(s, {"stage": s, "stage_index": stage_order.index(s), "enabled_modules": ""})
        stage_def_out.append({
            "stage": meta["stage"],
            "stage_index": int(meta["stage_index"]),
            "enabled_modules": meta["enabled_modules"],
        })
    stage_def_out.sort(key=lambda x: int(x["stage_index"]))

    mean_rows = aggregate_results(seed_rows_out, stage_order)
    delta_prev_rows = compute_delta_vs_prev(seed_rows_out, stage_order)

    save_csv(out_dir / "progressive_stage_definition.csv", stage_def_out)
    save_csv(out_dir / "progressive_seed_results.csv", seed_rows_out)
    save_csv(out_dir / "progressive_mean_std.csv", mean_rows)
    save_csv(out_dir / "progressive_delta_prev_mean_std.csv", delta_prev_rows)

    with (out_dir / "progressive_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "stages": stage_order,
                "stage_definition": stage_def_out,
                "inputs": [str(x) for x in input_dirs],
                "duplicates_ignored": duplicates_info,
                "num_seed_rows_raw": len(raw_seed_rows),
                "num_seed_rows_merged": len(seed_rows_out),
                "seed_results": seed_rows_out,
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
        out_path=out_dir / "progressive_v2v_bar.png",
    )
    plot_metric(
        mean_rows,
        metric_key="final_v2i_mean",
        err_key="final_v2i_std",
        ylabel="Final V2I Rate",
        out_path=out_dir / "progressive_v2i_bar.png",
    )
    plot_metric(
        mean_rows,
        metric_key="conv_step_mean",
        err_key="conv_step_std",
        ylabel="Convergence Step",
        out_path=out_dir / "progressive_convergence_bar.png",
    )
    plot_delta_prev(
        delta_prev_rows,
        metric_key="delta_v2v_mean",
        err_key="delta_v2v_std",
        ylabel="Delta V2V vs Previous Stage",
        out_path=out_dir / "progressive_delta_prev_v2v_bar.png",
    )
    plot_delta_prev(
        delta_prev_rows,
        metric_key="delta_v2i_mean",
        err_key="delta_v2i_std",
        ylabel="Delta V2I vs Previous Stage",
        out_path=out_dir / "progressive_delta_prev_v2i_bar.png",
    )

    print("[OK] Merge complete.")
    print(f"Inputs           : {[str(x) for x in input_dirs]}")
    print(f"Merged seed rows : {len(seed_rows_out)} (raw={len(raw_seed_rows)})")
    print(f"Output dir       : {out_dir}")
    print(f"Summary JSON     : {out_dir / 'progressive_summary.json'}")


if __name__ == "__main__":
    main()
