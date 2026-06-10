"""Compatibility shim for training modules that import top-level tf_runtime."""

from __future__ import annotations


def configure_tensorflow_runtime(tf=None) -> None:
    """Best-effort TensorFlow runtime tuning for the isolated lab.

    The main `formation/` package expects a top-level `tf_runtime` module.
    This lab-local shim keeps that expectation satisfied without modifying the
    main code tree.
    """

    if tf is None:
        return
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        return
