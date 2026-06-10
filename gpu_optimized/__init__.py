"""GPU-optimized training for vehicle platoon formation RL."""

from __future__ import annotations

import os

from tf_runtime import configure_tensorflow_runtime


configure_tensorflow_runtime()

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False
