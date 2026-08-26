"""Internal accepted-step state for periodic geometry optimization."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class _PeriodicGeometryContinuationState:
    """Portable state captured after one accepted ionic step."""

    completed_step: int
    positions: np.ndarray
    density: mx.array
    energy: float
    forces: np.ndarray
    steps: tuple[dict[str, object], ...]
    s_history: tuple[np.ndarray, ...]
    y_history: tuple[np.ndarray, ...]
    scf_evaluations: int
    line_search_evaluations: int
    lineage: tuple[str, ...] = ()
