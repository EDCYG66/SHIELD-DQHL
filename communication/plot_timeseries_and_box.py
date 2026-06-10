#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_timeseries_and_box.py

从 Agent 导出的 timeseries_<model>.csv（如 timeseries_gat.csv）中读取：
    t, v2i_rate, v2v_success, nveh

画两类图：
1) 图 10 风格的时间序列图：
   - V2V 成功率 (左轴)
   - V2I 速率 (右轴-1)
   - 车辆数 (右轴-2)

2) 图 11 风格的箱线图：
   - 将整个时间段切成 N 个 chunk（默认 5）
   - 对每个 chunk 分别画：
        * 车辆数的箱线
        * V2I 速率的箱线
        * V2V 成功率的箱线（映射到右 y 轴）

用法示例：
    python plot_timeseries_and_box.py \
        --csv runs/full_gat/.../exports/timeseries_gat.csv \
        --out runs/full_gat \
        --chunks 5 \
        --model-name GAT
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_timeseries(csv_path: Path):
    df = pd.read_csv(csv_path)
    # 期望列名：t,v2i_rate,v2v_success,nveh
    # 兼容老名字（如果有的话）
    col_t = "t"
    col_v2i = "v2i_rate" if "v2i_rate" in df.columns else "v2i"
    col_v2v = "v2v_success" if "v2v_success" in df.columns else "v2v_succ"
    col_nveh = "nveh"

    t = df[col_t].to_numpy(dtype=float)
    v2i = df[col_v2i].to_numpy(dtype=float)
    v2v = df[col_v2v].to_numpy(dtype=float)
    nveh = df[col_nveh].to_numpy(dtype=float)
    return t, v2i, v2v, nveh


def plot_timeseries(t, v2i, v2v, nveh, out_dir: Path, model_name: str, dpi: int = 224):
    """
    图 10 风格：三条时间序列共存在一张图，多 y 轴。
    """
    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))

    # 左轴：V2V 成功率
    ln1 = ax1.plot(t, v2v, "-", color="tab:blue",
                   label="Instantaneous V2V Success Rate")
    ax1.set_xlabel("Time index")
    ax1.set_ylabel("V2V Success Rate", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(alpha=0.3)

    # 右轴 1：V2I 速率
    ax2 = ax1.twinx()
    ln2 = ax2.plot(t, v2i, "-", color="tab:orange",
                   label="Instantaneous V2I Rate")
    ax2.set_ylabel("V2I Rate", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # 右轴 2：车辆数
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 55))
    ln3 = ax3.plot(t, nveh, "-", color="tab:green",
                   label="Number of Vehicles")
    ax3.set_ylabel("Vehicle Count", color="tab:green")
    ax3.tick_params(axis="y", labelcolor="tab:green")

    # 合并 legend
    lines = ln1 + ln2 + ln3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title(f"Instantaneous V2V/V2I and Vehicle Count ({model_name})")
    plt.tight_layout()

    out_path = out_dir / f"timeseries_{model_name.lower()}_paper.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[Saved] {out_path}")


def chunk_indices(n: int, num_chunks: int):
    """
    把 [0, n) 均匀分成 num_chunks 段，返回每段的 (start, end) 索引。
    """
    edges = np.linspace(0, n, num_chunks + 1, dtype=int)
    return [(edges[i], edges[i + 1]) for i in range(num_chunks)]


def plot_box_chunks(t, v2i, v2v, nveh,
                    out_dir: Path,
                    model_name: str,
                    num_chunks: int = 5,
                    dpi: int = 224):
    """
    图 11 风格：把时序分成若干 chunk，对每个 chunk 画车辆数 / V2I / V2V 的箱线图。

    左 y 轴：车辆数、V2I 盒须图
    右 y 轴：V2V 成功率盒须图
    """
    n = len(t)
    if n == 0:
        print("[WARN] Empty timeseries, skip box plot.")
        return

    chunks = chunk_indices(n, num_chunks)
    labels = [f"Chunk {i+1}" for i in range(len(chunks))]

    # 为了在一个图里区分三类 box，我们在 x 轴上做微小偏移
    x = np.arange(len(chunks))
    offset = 0.2

    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))

    # 保存每个 chunk 的数据
    veh_data = []
    v2i_data = []
    v2v_data = []

    for (s, e) in chunks:
        veh_data.append(nveh[s:e])
        v2i_data.append(v2i[s:e])
        v2v_data.append(v2v[s:e])

    # 1) 车辆数 box（左轴，偏左）
    bp1 = ax1.boxplot(
        veh_data,
        positions=x - offset,
        widths=0.2,
        patch_artist=True,
        boxprops=dict(facecolor="lightgray", color="black"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    # 2) V2I rate box（左轴，正中）
    bp2 = ax1.boxplot(
        v2i_data,
        positions=x,
        widths=0.2,
        patch_artist=True,
        boxprops=dict(facecolor="lightgreen", color="green"),
        medianprops=dict(color="green"),
        whiskerprops=dict(color="green"),
        capprops=dict(color="green"),
    )

    ax1.set_xlabel("Data Chunk")
    ax1.set_ylabel("V2I Rate and Vehicle Count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(alpha=0.3, axis="y")

    # 右轴：V2V success box（偏右）
    ax2 = ax1.twinx()
    bp3 = ax2.boxplot(
        v2v_data,
        positions=x + offset,
        widths=0.2,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="blue"),
        medianprops=dict(color="blue"),
        whiskerprops=dict(color="blue"),
        capprops=dict(color="blue"),
    )
    ax2.set_ylabel("V2V Success Rate")
    ax2.set_ylim(0.0, 1.05)

    # 手动画 legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="lightgray", edgecolor="black", label="Vehicle Count"),
        Patch(facecolor="lightgreen", edgecolor="green", label="V2I Rate"),
        Patch(facecolor="lightblue", edgecolor="blue", label="V2V Success Rate"),
    ]
    ax1.legend(handles=legend_elems, loc="upper right")

    plt.title(f"Algorithm Performance Boxplot over {num_chunks} Chunks ({model_name})")
    plt.tight_layout()

    out_path = out_dir / f"box_chunks_{model_name.lower()}_paper.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[Saved] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot timeseries and boxplots from timeseries_<model>.csv")
    ap.add_argument("--csv", type=str, required=True,
                    help="Path to timeseries_<model>.csv (e.g., timeseries_gat.csv)")
    ap.add_argument("--out", type=str, default="",
                    help="Output directory for figures (default: same dir as csv)")
    ap.add_argument("--chunks", type=int, default=5,
                    help="Number of chunks for boxplot (default: 5)")
    ap.add_argument("--dpi", type=int, default=224,
                    help="DPI for saved figures")
    ap.add_argument("--model-name", type=str, default="GAT",
                    help="Model name for titles and filenames, e.g., GAT/SAGE/FC")
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"timeseries csv not found: {csv_path}")

    out_dir = Path(args.out).expanduser().resolve() if args.out else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    t, v2i, v2v, nveh = load_timeseries(csv_path)

    plot_timeseries(t, v2i, v2v, nveh, out_dir, args.model_name, dpi=args.dpi)
    plot_box_chunks(t, v2i, v2v, nveh, out_dir, args.model_name,
                    num_chunks=args.chunks, dpi=args.dpi)

    print("[OK] timeseries & box plots generated.")


if __name__ == "__main__":
    main()