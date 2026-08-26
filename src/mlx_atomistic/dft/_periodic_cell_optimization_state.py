"""Internal accepted-step state for periodic cell optimization."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class _PeriodicCellContinuationState:
    """Portable state captured after one accepted cell step."""

    completed_step: int
    cell: np.ndarray
    positions: np.ndarray
    fractional_positions: np.ndarray
    density: mx.array
    energy: float
    stress: np.ndarray
    forces: np.ndarray | None
    steps: tuple[dict[str, object], ...]
    scf_evaluations: int
    stress_evaluations: int
    line_search_evaluations: int
    ionic_scf_evaluations: int
    lineage: tuple[str, ...] = ()
