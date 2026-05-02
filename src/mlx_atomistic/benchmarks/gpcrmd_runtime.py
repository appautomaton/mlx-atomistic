"""Shared helpers for GPCRmd MLX runtime benchmark reporting."""

from __future__ import annotations

import resource
from pathlib import Path
from typing import Any

import numpy as np


def directory_size_bytes(path: str | Path) -> int:
    """Return total file bytes under a benchmark output directory."""

    root = Path(path)
    if not root.exists():
        return 0
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def max_rss_mb() -> float:
    """Return approximate max resident memory in MB for the current process."""

    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if rss > 10_000_000:
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def pme_mesh_summary(
    mesh_shape: np.ndarray,
    metadata_pme_config: dict[str, Any] | None = None,
) -> tuple[str | None, int | None]:
    """Return stable PME mesh shape text and total mesh points."""

    mesh = np.asarray(mesh_shape)
    if mesh.size == 0 and metadata_pme_config:
        mesh = np.asarray(metadata_pme_config.get("mesh_shape", ()))
    if mesh.size == 0:
        return None, None
    mesh = np.asarray(mesh, dtype=np.int32).reshape(-1)
    if mesh.shape != (3,):
        return "invalid", None
    return "x".join(str(int(item)) for item in mesh), int(np.prod(mesh))


def last_int(values: np.ndarray) -> int | None:
    """Return the last integer from an array-like diagnostic."""

    array = np.asarray(values)
    if array.size == 0:
        return None
    return int(array.reshape(-1)[-1])


def max_float(values: np.ndarray) -> float | None:
    """Return a finite max from an array-like diagnostic."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.max(array))


__all__ = [
    "directory_size_bytes",
    "last_int",
    "max_float",
    "max_rss_mb",
    "pme_mesh_summary",
]
