"""Append a concise defense-slide legend below the platoon triptych figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

try:
    from .scenario_renderer import BLOCKED_ZONE, EVENT_ZONE, LEADER_EDGE
except ImportError:  # pragma: no cover
    from scenario_renderer import BLOCKED_ZONE, EVENT_ZONE, LEADER_EDGE


def _configure_chinese_font() -> font_manager.FontProperties:
    font_paths = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ]
    for font_path in font_paths:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_prop = font_manager.FontProperties(fname=str(font_path))
            plt.rcParams["font.family"] = font_prop.get_name()
            plt.rcParams["font.sans-serif"] = [font_prop.get_name()]
            plt.rcParams["pdf.fonttype"] = 42
            plt.rcParams["ps.fonttype"] = 42
            plt.rcParams["axes.unicode_minus"] = False
            return font_prop
    plt.rcParams["axes.unicode_minus"] = False
    return font_manager.FontProperties()


def append_triptych_legend(input_path: Path, output_base: Path) -> Path:
    font_prop = _configure_chinese_font()
    image = plt.imread(input_path)

    fig = plt.figure(figsize=(13.6, 6.0), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=[5.05, 0.95], hspace=0.02)

    ax_img = fig.add_subplot(grid[0])
    ax_img.imshow(image)
    ax_img.axis("off")

    ax_leg = fig.add_subplot(grid[1])
    ax_leg.axis("off")

    handles = [
        Line2D([0], [0], marker="^", linestyle="none", label="上行车辆", markerfacecolor="#9BD77D", markeredgecolor="#08733D", markeredgewidth=1.8, markersize=10),
        Line2D([0], [0], marker="v", linestyle="none", label="下行车辆", markerfacecolor="#D8D39A", markeredgecolor="#08733D", markeredgewidth=1.8, markersize=10),
        Line2D([0], [0], marker="*", linestyle="none", label="星标：领航/协调车辆", markerfacecolor="#FFD166", markeredgecolor=LEADER_EDGE, markeredgewidth=0.9, markersize=12),
        Rectangle((0, 0), 1, 1, facecolor=EVENT_ZONE, edgecolor="#D88D2F", alpha=0.28, label="浅黄色：事故影响区/限速区"),
        Rectangle((0, 0), 1, 1, facecolor=BLOCKED_ZONE, edgecolor="#B85C00", alpha=0.30, hatch="////", label="斜线深黄色：受阻车道/瓶颈区域"),
        Line2D([0], [0], marker="_", linestyle="none", label="红色 34：局部限速", markerfacecolor="#C33A3A", markeredgecolor="#C33A3A", markeredgewidth=2.0, markersize=16),
    ]
    legend_font = font_prop.copy()
    legend_font.set_size(8.8)
    ax_leg.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="#D0D5DD",
        columnspacing=1.05,
        handletextpad=0.45,
        borderpad=0.42,
        prop=legend_font,
        bbox_to_anchor=(0.5, 0.94),
    )
    ax_leg.text(
        0.5,
        0.11,
        "说明：车辆出现在事故影响区附近表示正在通过该路段并执行避让/重构，并不表示该车辆本身发生故障。",
        ha="center",
        va="bottom",
        fontsize=8.4,
        fontproperties=font_prop,
        color="#4B5563",
        transform=ax_leg.transAxes,
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=450, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return output_base.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a PPT legend to a triptych figure.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output_base = args.output or args.input.with_name(args.input.stem + "_with_legend")
    print(append_triptych_legend(args.input, output_base))


if __name__ == "__main__":
    main()
