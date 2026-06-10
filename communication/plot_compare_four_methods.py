#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare V2V and V2I curves of four methods: GAT, SAGE, FC, Random.

需要的文件（根据你的实际路径稍微改一下）：

- runs/v2v_gat_new/gat_v2v_curve.csv
- runs/v2v_gat_new/gat_v2i_curve.csv
- runs/v2v_sage_new/sage_v2v_curve.csv
- runs/v2v_sage_new/sage_v2i_curve.csv
- runs/v2v_fc_new/fc_v2v_curve.csv
- runs/v2v_fc_new/fc_v2i_curve.csv
- runs/v2v_random_full/exports/random_v2v_curve.csv
- runs/v2v_random_full/exports/random_v2i_curve.csv

输出：

- runs/compare_four/compare_v2v.png
- runs/compare_four/compare_v2i.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_curve(csv_path, step_col, value_col, tag):
    p = Path(csv_path).expanduser().resolve()
    if not p.exists():
        print(f"[WARN] not found: {p}")
        return None
    df = pd.read_csv(p)
    return dict(
        tag=tag,
        steps=df[step_col].values,
        vals=df[value_col].values,
    )


def main():
    base = Path("runs")

    curves_v2v = []
    curves_v2i = []

    # 根据你的实际目录调整这里的路径 --------------------------
    curves_v2v.append(load_curve(
        base / "v2v_gat_new/gat_v2v_curve.csv",
        "step", "v2v_success", "GAT"))
    curves_v2i.append(load_curve(
        base / "v2v_gat_new/gat_v2i_curve.csv",
        "step", "v2i_rate", "GAT"))

    curves_v2v.append(load_curve(
        base / "v2v_sage_new/sage_v2v_curve.csv",
        "step", "v2v_success", "SAGE"))
    curves_v2i.append(load_curve(
        base / "v2v_sage_new/sage_v2i_curve.csv",
        "step", "v2i_rate", "SAGE"))

    # 如果已经跑完 FC，就保留下面两行；还没跑就先注释掉
    curves_v2v.append(load_curve(
        base / "v2v_fc_new/fc_v2v_curve.csv",
        "step", "v2v_success", "FC"))
    curves_v2i.append(load_curve(
        base / "v2v_fc_new/fc_v2i_curve.csv",
        "step", "v2i_rate", "FC"))

    curves_v2v.append(load_curve(
        base / "v2v_random_full/exports/random_v2v_curve.csv",
        "step", "v2v_success", "Random"))
    curves_v2i.append(load_curve(
        base / "v2v_random_full/exports/random_v2i_curve.csv",
        "step", "v2i_rate", "Random"))
    # ---------------------------------------------------------

    # 过滤掉 None（比如 FC 还没跑）
    curves_v2v = [c for c in curves_v2v if c is not None]
    curves_v2i = [c for c in curves_v2i if c is not None]

    out_dir = base / "compare_four"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) V2V 对比图
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for c in curves_v2v:
        ax.plot(c["steps"], c["vals"], "-o", label=c["tag"])
    ax.set_xlabel("step")
    ax.set_ylabel("V2V Success Rate")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    plt.tight_layout()
    v2v_path = out_dir / "compare_v2v.png"
    fig.savefig(v2v_path, dpi=224)
    plt.close(fig)
    print("[Saved]", v2v_path)

    # 2) V2I 对比图
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for c in curves_v2i:
        ax.plot(c["steps"], c["vals"], "-o", label=c["tag"])
    ax.set_xlabel("step")
    ax.set_ylabel("V2I Rate")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    v2i_path = out_dir / "compare_v2i.png"
    fig.savefig(v2i_path, dpi=224)
    plt.close(fig)
    print("[Saved]", v2i_path)


if __name__ == "__main__":
    main()