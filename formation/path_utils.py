"""Helpers for resolving project-local paths from the formation package."""

from __future__ import annotations

import sys
from pathlib import Path


def get_myproject_root() -> Path:
    """Return the current MyProject root.

    Expected layout:
      MyProject/
        formation/
        communication/   (preferred, if present)
        *.py
    """
    return Path(__file__).resolve().parents[1]


def get_formation_root() -> Path:
    return get_myproject_root() / "formation"


def get_communication_root() -> Path:
    """Return the communication code root.

    Prefer `MyProject/communication` if it exists.
    Otherwise fall back to `MyProject` itself, for compatibility with the
    original flat code layout where communication files lived at repo root.
    """
    project_root = get_myproject_root()
    communication_root = project_root / "communication"
    if communication_root.exists():
        return communication_root
    return project_root


def ensure_communication_on_path() -> Path:
    communication_root = get_communication_root()
    communication_str = str(communication_root)
    if communication_str not in sys.path:
        sys.path.insert(0, communication_str)
    return communication_root


def ensure_myproject_on_path() -> Path:
    project_root = get_myproject_root()
    project_str = str(project_root)
    if project_str not in sys.path:
        sys.path.insert(0, project_str)
    return project_root
