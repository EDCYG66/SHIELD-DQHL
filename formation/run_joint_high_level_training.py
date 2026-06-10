"""Convenience entry for joint high-level training with the communication Agent enabled."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Reduce TensorFlow C++ info/debug log noise when training.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from .run_trainable_high_level_policy import build_parser as build_base_parser, run_training
except ImportError:  # pragma: no cover
    from run_trainable_high_level_policy import build_parser as build_base_parser, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser()
    parser.description = "Joint high-level training with communication Agent enabled"
    parser.set_defaults(
        comm_policy="agent",
        comm_gnn="gat",
        disable_communication=False,
        eval_every=5,
    )
    parser.add_argument(
        "--auto-comm-weight-dir",
        action="store_true",
        help="If set and --comm-weights-dir is empty, try MyProject/communication/weight automatically.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.auto_comm_weight_dir and not args.comm_weights_dir:
        default_dir = Path(__file__).resolve().parents[1] / "communication" / "weight"
        args.comm_weights_dir = str(default_dir)
    run_training(args)


if __name__ == "__main__":
    main()
