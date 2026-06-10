#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_v2v_bar_vs_vehicles.py

读取 sweep_vehicles.csv，画“平均 V2V 成功率 vs 车辆数”的柱状图，
风格类似参考文献中的图 7（当前只有一个方法 GAT）。

CSV 格式示例：
    model,n_veh,v2i_mean,v2v_success
    gat,10, ..., 0.95
    gat,20, ..., 0.93
    ...

用法：
    python plot_v2v_bar_vs_vehicles.py \
        --csv runs/gat_sweep_veh_all/sweep_vehicles.csv \
        --out runs/gat_sweep_veh_all \
        --model gat
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Plot average V2V success vs vehicles (bar).")
    ap.add_argument("--csv", type=str, required=True,
                    help="Path to sweep_vehicles.csv")
    ap.add_argument("--out", type=str, default="",
                    help="Output directory (default: same as csv)")
    ap.add_argument("--dpi", type=int, default=224)
    ap.add_argument("--model", type=str, default="gat",
                    help="Model name to filter, e.g., gat/sage/fc")
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir = Path(args.out).expanduser().resolve() if args.out else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # 只取指定 model 的数据
    df = df[df["model"].str.lower() == args.model.lower()]
    if df.empty:
        print("[WARN] No rows for model:", args.model)
        return

    df = df.sort_values("n_veh")
    n_veh = df["n_veh"].to_numpy(dtype=float)
    v2v = df["v2v_success"].to_numpy(dtype=float)

    x = np.arange(len(n_veh))
    width = 0.6

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(x, v2v, width=width, color="tab:blue", label=args.model.upper())

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in n_veh])
    ax.set_xlabel("Number of Participating Vehicles")
    ax.set_ylabel("V2V Communication Success Rate")
    ax.set_ylim(0.0, 1.0)  # 根据需要可以改成 0.8~1.0 之类
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right")

    plt.tight_layout()
    out_path = out_dir / f"v2v_success_vs_vehicles_{args.model.lower()}_bar.png"
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()