"""Optional backend helpers for tensorized communication kernels."""

from __future__ import annotations

import os


def requested_backend(default: str = "numpy") -> str:
    return os.environ.get("TENSORIZED_COMM_BACKEND", default).strip().lower() or default


def cupy_available() -> bool:
    try:
        import cupy  # noqa: F401
    except Exception:
        return False
    return True


def resolve_backend(name: str | None = None) -> str:
    backend = requested_backend() if name is None else str(name).strip().lower()
    if backend in {"np", "numpy", "cpu"}:
        return "numpy"
    if backend in {"cp", "cupy", "gpu"}:
        return "cupy" if cupy_available() else "numpy"
    if backend == "auto":
        return "cupy" if cupy_available() else "numpy"
    return "numpy"


def should_use_cupy(name: str | None, *, n_links: int, default_min_links: int = 512) -> bool:
    backend = requested_backend() if name is None else str(name).strip().lower()
    if backend in {"cp", "cupy", "gpu"}:
        return cupy_available()
    if backend != "auto":
        return False
    try:
        min_links = int(os.environ.get("TENSORIZED_COMM_CUPY_MIN_LINKS", default_min_links))
    except Exception:
        min_links = int(default_min_links)
    return cupy_available() and int(n_links) >= min_links
