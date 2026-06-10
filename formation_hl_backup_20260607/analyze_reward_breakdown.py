"""Summarize reward component contributions from formation_step_metrics.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze reward component breakdown from evaluation CSV")
    parser.add_argument("csv_path", type=Path, help="Path to formation_step_metrics.csv")
    parser.add_argument("--top", type=int, default=20, help="Number of components to print")
    args = parser.parse_args()

    csv_path = args.csv_path.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, raw in row.items():
                if not key.startswith("reward_") or raw in {"", None}:
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                totals[key] = totals.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1

    items = []
    for key, total in totals.items():
        count = max(1, counts.get(key, 1))
        items.append((key, total, total / count, abs(total)))

    items.sort(key=lambda x: x[3], reverse=True)

    print("component,total,mean")
    for key, total, mean, _abs_total in items[: max(1, int(args.top))]:
        print(f"{key},{total:.6f},{mean:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
