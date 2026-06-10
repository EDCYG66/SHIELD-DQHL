"""Entry point — applies vectorized optimizations and runs training."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tf_runtime import configure_tensorflow_runtime


configure_tensorflow_runtime()

from gpu_optimized.cupy_kernels import _init_cupy_pool
from gpu_optimized.accelerated_highway import install_highway_patches
from gpu_optimized.accelerated_channel import install_channel_patches
from gpu_optimized.accelerated_step import install_step_patches
from gpu_optimized.config import CUPY_MEMPOOL_LIMIT_BYTES


def main():
    try:
        from formation.run_joint_high_level_training import build_parser
    except Exception:
        from formation.run_trainable_high_level_policy import build_parser
    from formation.run_trainable_high_level_policy import run_training

    parser = build_parser()
    parser.add_argument("--optimization-level", type=int, default=3, choices=[1, 2, 3],
                        help="1=CuPy only, 2=+vectorized env, 3=+enhanced training")
    parser.add_argument("--n-envs", type=int, default=16,
                        help="Number of parallel environments (level 2+)")
    parser.add_argument("--cupy-mem-limit-gb", type=float, default=4.0,
                        help="CuPy GPU memory pool limit in GB")
    parser.add_argument("--comm-backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"],
                        help="Communication tensor backend for experimental kernels")
    parser.add_argument("--vec-step-mode", type=str, default="auto", choices=["auto", "sequential", "threaded", "process"],
                        help="Step mode for vectorized rollout environments")
    parser.add_argument("--resource-log-interval", type=float, default=5.0,
                        help="Seconds between CPU/GPU utilization samples. Set <=0 to disable.")
    args = parser.parse_args()

    if getattr(args, "auto_comm_weight_dir", False) and not args.comm_weights_dir:
        from pathlib import Path
        default_dir = Path(__file__).resolve().parents[1] / "communication" / "weight"
        args.comm_weights_dir = str(default_dir)

    opt_level = int(args.optimization_level)

    _init_cupy_pool(int(args.cupy_mem_limit_gb * 1024 ** 3))
    install_highway_patches()
    install_channel_patches()
    install_step_patches()
    print(f"[GPU-Opt] Level {opt_level}: highway + channel + step patches installed", flush=True)

    if opt_level >= 2:
        from gpu_optimized.vectorized_rollout import run_vectorized_training
        print(f"[GPU-Opt] Level {opt_level}: vectorized training, n_envs={args.n_envs}", flush=True)
        run_vectorized_training(args, optimization_level=opt_level)
    else:
        print(f"[GPU-Opt] Level 1: optimized original training loop", flush=True)
        run_training(args)

    print("[GPU-Opt] Done.", flush=True)


if __name__ == "__main__":
    main()
