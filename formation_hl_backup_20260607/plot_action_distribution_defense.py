"""Create a PPT-friendly action-distribution figure for formation defense slides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


DEFAULT_RESULT_DIR = Path(
    "/home/edcyg/MyProject/formation/results/eval/_fix_validation_joint_3600_v2_triptych_ppt"
)


ACTIONS = [
    ("keep", "保持队形"),
    ("split", "拆分重构"),
    ("merge", "合并恢复"),
    ("compact", "紧凑"),
    ("expand", "扩展"),
    ("emergency", "应急避险"),
]

COLORS = {
    "keep": "#1F5AA6",
    "split": "#E07A2D",
    "merge": "#4E9A9A",
    "compact": "#C9CDD3",
    "expand": "#C9CDD3",
    "emergency": "#C9CDD3",
}

FONT_REGULAR_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_RESULT_DIR / "formation_summary.json",
        help="Path to formation_summary.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULT_DIR / "action_distribution_defense.png",
        help="Output PNG path. A PDF with the same basename is also written.",
    )
    return parser.parse_args()


def read_action_ratios(summary_path: Path) -> dict[str, float]:
    with summary_path.open("r", encoding="utf-8") as file_obj:
        summary = json.load(file_obj)
    return {key: float(summary.get(f"avg_action_{key}", 0.0)) for key, _ in ACTIONS}


def load_chinese_fonts() -> tuple[font_manager.FontProperties, font_manager.FontProperties]:
    for font_path in [FONT_REGULAR_PATH, FONT_BOLD_PATH]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))

    regular = font_manager.FontProperties(fname=str(FONT_REGULAR_PATH))
    bold = font_manager.FontProperties(fname=str(FONT_BOLD_PATH))
    plt.rcParams["font.family"] = regular.get_name()
    return regular, bold


def plot_action_distribution(ratios: dict[str, float], output_path: Path) -> None:
    font_regular, font_bold = load_chinese_fonts()
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    values = [ratios[key] * 100.0 for key, _ in ACTIONS]
    labels = [label for _, label in ACTIONS]
    colors = [COLORS[key] for key, _ in ACTIONS]

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=300)
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#F7F9FC")

    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=1.2, height=0.58)
    ax.invert_yaxis()

    max_value = max(values) if values else 100.0
    ax.set_xlim(0, max(80.0, max_value + 8.0))
    ax.set_xlabel("动作占比（%）", fontsize=13, color="#374151", labelpad=8, fontproperties=font_regular)
    ax.tick_params(axis="x", labelsize=11, colors="#4B5563")
    ax.tick_params(axis="y", labelsize=13, colors="#111827", length=0)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontproperties(font_regular)
    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.22, color="#64748B")
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.spines["bottom"].set_linewidth(1.0)

    for bar, value in zip(bars, values):
        x_pos = value + 1.2 if value >= 1.0 else 1.2
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            ha="left",
            fontsize=12,
            color="#111827" if value >= 1.0 else "#6B7280",
            fontproperties=font_bold if value >= 5.0 else font_regular,
            fontweight="bold" if value >= 5.0 else "normal",
        )

    keep_pct = ratios.get("keep", 0.0) * 100.0
    split_pct = ratios.get("split", 0.0) * 100.0
    fig.text(0.08, 0.93, "高层重构动作分布", fontsize=22, color="#0F172A", fontproperties=font_bold)
    fig.text(
        0.08,
        0.865,
        "以稳定保持为主，事故影响阶段触发拆分重构",
        fontsize=12.5,
        color="#475569",
        fontproperties=font_regular,
    )
    fig.text(
        0.72,
        0.885,
        f"保持 {keep_pct:.1f}%\n拆分 {split_pct:.1f}%",
        ha="center",
        va="center",
        fontsize=12,
        color="#0F172A",
        fontproperties=font_regular,
        bbox={
            "boxstyle": "round,pad=0.45,rounding_size=0.12",
            "facecolor": "#EAF2FF",
            "edgecolor": "#A7C7F9",
            "linewidth": 1.0,
        },
    )
    fig.text(
        0.08,
        0.055,
        "数据来源：3600 step 典型验证场景",
        fontsize=10.5,
        color="#64748B",
        fontproperties=font_regular,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.18, right=0.94, top=0.78, bottom=0.18)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ratios = read_action_ratios(args.summary)
    plot_action_distribution(ratios, args.output)
    print(args.output)
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
