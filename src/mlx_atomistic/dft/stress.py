"""Finite-difference stress diagnostics for orthorhombic DFT cells."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell
from mlx_atomistic.dft.scf import SCFConfig, run_scf
from mlx_atomistic.dft.system import DFTSystem
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional


@dataclass(frozen=True)
class StressResult:
    """Diagonal orthorhombic stress estimate."""

    stress: mx.array
    pressure: float
    displacement: float
    samples: tuple[dict, ...]

    def to_dict(self) -> dict:
        """Return a JSON-safe stress summary."""

        return {
            "stress": np.array(self.stress).tolist(),
            "pressure": self.pressure,
            "displacement": self.displacement,
            "samples": list(self.samples),
        }


def finite_difference_stress(
    system: DFTSystem,
    *,
    config: SCFConfig | None = None,
    xc_functional: ExchangeCorrelationFunctional | None = None,
    displacement: float = 1e-3,
) -> StressResult:
    """Estimate diagonal stress by finite-differencing orthorhombic cell lengths."""

    if displacement <= 0.0:
        msg = "displacement must be positive"
        raise ValueError(msg)
    config = SCFConfig(max_iterations=2, solver="dense", seed=41) if config is None else config
    lengths = np.array(system.cell.lengths, dtype=np.float64)
    volume = float(np.prod(lengths))
    stress = np.zeros(3, dtype=np.float64)
    samples: list[dict] = []
    for axis in range(3):
        plus_lengths = lengths.copy()
        minus_lengths = lengths.copy()
        plus_lengths[axis] += displacement
        minus_lengths[axis] -= displacement
        if minus_lengths[axis] <= 0.0:
            msg = "cell displacement produced a nonpositive length"
            raise ValueError(msg)
        plus = run_scf(
            system.with_cell(Cell.orthorhombic(plus_lengths), scale_centers=True),
            config=config,
            xc_functional=xc_functional,
        )
        minus = run_scf(
            system.with_cell(Cell.orthorhombic(minus_lengths), scale_centers=True),
            config=config,
            xc_functional=xc_functional,
        )
        derivative = (plus.total_energy - minus.total_energy) / (2.0 * displacement)
        stress[axis] = -lengths[axis] * derivative / volume
        samples.append(
            {
                "axis": axis,
                "energy_plus": plus.total_energy,
                "energy_minus": minus.total_energy,
                "dE_dL": float(derivative),
            }
        )
    return StressResult(
        stress=mx.array(stress.astype(np.float32)),
        pressure=float(-np.mean(stress)),
        displacement=displacement,
        samples=tuple(samples),
    )
