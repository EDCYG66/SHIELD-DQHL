#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine per-variant ablation results (separate folders) into
clean summary figures with error bars and delta vs full.

Expected structure:
  runs/ablation_true_formal/<variant>/ablation_mean_std.csv
  runs/ablation_true_formal/<variant>/ablation_seed_results.csv

Outputs (default):
  runs/ablation_true_formal/ablation_summary_abs.png
  runs/ablation_true_formal/ablation_summary_delta.png
  runs/ablation_true_formal/ablation_combined_mean_std.csv
  runs/ablation_true_formal/ablation_combined_delta.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path("runs/ablation_true_formal")
VARIANTS = [
    "full",
    "no_urgency",
    "no_rb_balance",
    "no_conflict",
    "no_attn_reg",
    "no_reprune",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_mean_std(base_dir: Path, variant: str) -> Dict[str, float]:
    rows = read_csv(base_dir / variant / "ablation_mean_std.csv")
    if not rows:
        return {}
    row = rows[0]
    return {
        "variant": row["variant"],
        "final_v2i_mean": float(row["final_v2i_mean"]),
        "final_v2i_std": float(row["final_v2i_std"]),
        "final_v2v_mean": float(row["final_v2v_mean"]),
        "final_v2v_std": float(row["final_v2v_std"]),
    }


def load_seed_results(base_dir: Path, variant: str) -> List[Dict[str, str]]:
    return read_csv(base_dir / variant / "ablation_seed_results.csv")


def compute_delta_vs_full(
    full_rows: List[Dict[str, str]],
    variant_rows: List[Dict[str, str]],
) -> Dict[str, float]:
    base_by_seed = {int(r["seed"]): r for r in full_rows}
    delta_v2i = []
    delta_v2v = []
    for r in variant_rows:
        seed = int(r["seed"])
        base = base_by_seed.get(seed)
        if not base:
            continue
        delta_v2i.append(float(r["final_v2i"]) - float(base["final_v2i"]))
        delta_v2v.append(float(r["final_v2v"]) - float(base["final_v2v"]))

    def _std(vals: List[float]) -> float:
        return float(pstdev(vals)) if len(vals) > 1 else 0.0

    return {
        "delta_v2i_mean": float(mean(delta_v2i)) if delta_v2i else 0.0,
        "delta_v2i_std": _std(delta_v2i),
        "delta_v2v_mean": float(mean(delta_v2v)) if delta_v2v else 0.0,
        "delta_v2v_std": _std(delta_v2v),
    }


def save_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_abs(mean_rows: List[Dict[str, float]], out_path: Path) -> None:
    labels = [r["variant"] for r in mean_rows]
    v2v = [r["final_v2v_mean"] for r in mean_rows]
    v2v_err = [r["final_v2v_std"] for r in mean_rows]
    v2i = [r["final_v2i_mean"] for r in mean_rows]
    v2i_err = [r["final_v2i_std"] for r in mean_rows]

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)

    ax1.bar(x, v2v, yerr=v2v_err, capsize=4, color="#7FA6D8", edgecolor="#4E6EA9")
    ax1.set_ylabel("Final V2V Success Rate")
    ax1.grid(axis="y", alpha=0.25)

    ax2.bar(x, v2i, yerr=v2i_err, capsize=4, color="#F3B577", edgecolor="#C9823A")
    ax2.set_ylabel("Final V2I Rate")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20)
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=224)
    plt.close(fig)


def plot_delta(delta_rows: List[Dict[str, float]], out_path: Path) -> None:
    labels = [r["variant"] for r in delta_rows]
    v2v = [r["delta_v2v_mean"] for r in delta_rows]
    v2v_err = [r["delta_v2v_std"] for r in delta_rows]
    v2i = [r["delta_v2i_mean"] for r in delta_rows]
    v2i_err = [r["delta_v2i_std"] for r in delta_rows]

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)

    ax1.bar(x, v2v, yerr=v2v_err, capsize=4, color="#7FA6D8", edgecolor="#4E6EA9")
    ax1.axhline(0.0, color="#555", linewidth=1.0)
    ax1.set_ylabel("Δ V2V (vs full)")
    ax1.grid(axis="y", alpha=0.25)

    ax2.bar(x, v2i, yerr=v2i_err, capsize=4, color="#F3B577", edgecolor="#C9823A")
    ax2.axhline(0.0, color="#555", linewidth=1.0)
    ax2.set_ylabel("Δ V2I (vs full)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20)
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=224)
    plt.close(fig)


def plot_combined(
    mean_rows: List[Dict[str, float]],
    delta_rows: List[Dict[str, float]],
    out_path: Path,
) -> None:
    labels = [r["variant"] for r in mean_rows]
    v2v = [r["final_v2v_mean"] for r in mean_rows]
    v2v_err = [r["final_v2v_std"] for r in mean_rows]
    v2i = [r["final_v2i_mean"] for r in mean_rows]
    v2i_err = [r["final_v2i_std"] for r in mean_rows]

    d_labels = [r["variant"] for r in delta_rows]
    dv2v = [r["delta_v2v_mean"] for r in delta_rows]
    dv2v_err = [r["delta_v2v_std"] for r in delta_rows]
    dv2i = [r["delta_v2i_mean"] for r in delta_rows]
    dv2i_err = [r["delta_v2i_std"] for r in delta_rows]

    x = np.arange(len(labels))
    xd = np.arange(len(d_labels))

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2))
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    ax1.bar(x, v2v, yerr=v2v_err, capsize=3, color="#7FA6D8", edgecolor="#4E6EA9")
    ax1.set_title("Final V2V")
    ax1.set_ylabel("V2V Success Rate")
    ax1.grid(axis="y", alpha=0.25)

    ax2.bar(x, v2i, yerr=v2i_err, capsize=3, color="#F3B577", edgecolor="#C9823A")
    ax2.set_title("Final V2I")
    ax2.set_ylabel("V2I Rate")
    ax2.grid(axis="y", alpha=0.25)

    ax3.bar(xd, dv2v, yerr=dv2v_err, capsize=3, color="#7FA6D8", edgecolor="#4E6EA9")
    ax3.axhline(0.0, color="#555", linewidth=1.0)
    ax3.set_title("Δ V2V vs Full")
    ax3.set_ylabel("Δ V2V")
    ax3.set_xticks(xd)
    ax3.set_xticklabels(d_labels, rotation=20)
    ax3.grid(axis="y", alpha=0.25)

    ax4.bar(xd, dv2i, yerr=dv2i_err, capsize=3, color="#F3B577", edgecolor="#C9823A")
    ax4.axhline(0.0, color="#555", linewidth=1.0)
    ax4.set_title("Δ V2I vs Full")
    ax4.set_ylabel("Δ V2I")
    ax4.set_xticks(xd)
    ax4.set_xticklabels(d_labels, rotation=20)
    ax4.grid(axis="y", alpha=0.25)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    mean_rows = []
    for v in VARIANTS:
        row = load_mean_std(BASE_DIR, v)
        if row:
            mean_rows.append(row)

    full_rows = load_seed_results(BASE_DIR, "full")
    delta_rows = []
    for v in VARIANTS:
        if v == "full":
            continue
        v_rows = load_seed_results(BASE_DIR, v)
        if not v_rows:
            continue
        delta = compute_delta_vs_full(full_rows, v_rows)
        delta_rows.append({"variant": v, **delta})

    save_csv(BASE_DIR / "ablation_combined_mean_std.csv", mean_rows)
    save_csv(BASE_DIR / "ablation_combined_delta.csv", delta_rows)

    plot_abs(mean_rows, BASE_DIR / "ablation_summary_abs.png")
    plot_delta(delta_rows, BASE_DIR / "ablation_summary_delta.png")
    plot_combined(mean_rows, delta_rows, BASE_DIR / "ablation_summary_combined.png")
    print("[Saved]", BASE_DIR / "ablation_summary_abs.png")
    print("[Saved]", BASE_DIR / "ablation_summary_delta.png")
    print("[Saved]", BASE_DIR / "ablation_summary_combined.png")


if __name__ == "__main__":
    main()
