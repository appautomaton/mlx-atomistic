"""k-point meshes and band-structure diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft.operators import DenseHamiltonianReference, KohnShamOperator
from mlx_atomistic.dft.scf import SCFResult
from mlx_atomistic.dft.system import DFTSystem
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional


@dataclass(frozen=True)
class KPoint:
    """One reciprocal-space k point in Cartesian reciprocal units."""

    vector: tuple[float, float, float]
    weight: float = 1.0
    label: str | None = None

    def __init__(
        self,
        vector: Sequence[float],
        *,
        weight: float = 1.0,
        label: str | None = None,
    ):
        if len(vector) != 3:
            msg = "k-point vector must have three components"
            raise ValueError(msg)
        if weight <= 0.0:
            msg = "k-point weight must be positive"
            raise ValueError(msg)
        object.__setattr__(self, "vector", tuple(float(value) for value in vector))
        object.__setattr__(self, "weight", float(weight))
        object.__setattr__(self, "label", label)

    @classmethod
    def gamma(cls) -> KPoint:
        """Return the Γ point."""

        return cls((0.0, 0.0, 0.0), label="Γ")

    def to_dict(self) -> dict:
        """Return a JSON-safe representation."""

        return {
            "vector": list(self.vector),
            "weight": self.weight,
            "label": self.label,
        }


@dataclass(frozen=True)
class KPointMesh:
    """Weighted k-point mesh."""

    points: tuple[KPoint, ...]

    def __init__(self, points: Sequence[KPoint]):
        if not points:
            msg = "KPointMesh requires at least one point"
            raise ValueError(msg)
        weight_sum = sum(point.weight for point in points)
        normalized = tuple(
            KPoint(point.vector, weight=point.weight / weight_sum, label=point.label)
            for point in points
        )
        object.__setattr__(self, "points", normalized)

    @classmethod
    def gamma(cls) -> KPointMesh:
        """Return a one-point Γ mesh."""

        return cls([KPoint.gamma()])

    def to_dict(self) -> dict:
        """Return a JSON-safe representation."""

        return {"points": [point.to_dict() for point in self.points]}


@dataclass(frozen=True)
class MonkhorstPackGrid(KPointMesh):
    """Simple Γ-centered Monkhorst-Pack-style mesh."""

    size: tuple[int, int, int] = (1, 1, 1)

    def __init__(self, size: Sequence[int]):
        parsed = tuple(int(value) for value in size)
        if len(parsed) != 3 or any(value <= 0 for value in parsed):
            msg = "MonkhorstPackGrid size must contain three positive integers"
            raise ValueError(msg)
        points = []
        total = int(np.prod(parsed))
        for indices in np.ndindex(parsed):
            vector = tuple(
                (index - (count - 1) / 2.0) / count
                for index, count in zip(indices, parsed, strict=True)
            )
            points.append(KPoint(vector, weight=1.0 / total))
        KPointMesh.__init__(self, points)
        object.__setattr__(self, "size", parsed)


@dataclass(frozen=True)
class BandPath:
    """Explicit k-point path for non-SCF band diagnostics."""

    points: tuple[KPoint, ...]

    def __init__(self, points: Sequence[KPoint]):
        if not points:
            msg = "BandPath requires at least one point"
            raise ValueError(msg)
        object.__setattr__(self, "points", tuple(points))

    @classmethod
    def line(
        cls,
        start: Sequence[float],
        end: Sequence[float],
        *,
        count: int,
        start_label: str | None = None,
        end_label: str | None = None,
    ) -> BandPath:
        """Build a linear path between two k points."""

        if count <= 0:
            msg = "count must be positive"
            raise ValueError(msg)
        start_np = np.asarray(start, dtype=np.float64)
        end_np = np.asarray(end, dtype=np.float64)
        points = []
        for index in range(count):
            fraction = 0.0 if count == 1 else index / (count - 1)
            label = start_label if index == 0 else end_label if index == count - 1 else None
            points.append(KPoint((1.0 - fraction) * start_np + fraction * end_np, label=label))
        return cls(points)


@dataclass(frozen=True)
class BandStructureResult:
    """Non-SCF band energies along a path."""

    kpoints: tuple[KPoint, ...]
    eigenvalues: mx.array
    reused_density: bool

    def to_dict(self) -> dict:
        """Return JSON-safe band data."""

        return {
            "kpoints": [point.to_dict() for point in self.kpoints],
            "eigenvalues": np.array(self.eigenvalues).tolist(),
            "reused_density": self.reused_density,
        }


def run_band_structure(
    system: DFTSystem,
    scf_result: SCFResult,
    band_path: BandPath,
    *,
    n_bands: int = 1,
    xc_functional: ExchangeCorrelationFunctional | None = None,
) -> BandStructureResult:
    """Evaluate non-SCF bands on top of a converged density."""

    if n_bands <= 0:
        msg = "n_bands must be positive"
        raise ValueError(msg)
    v_local = system.pseudopotential.field(system.grid)
    values = []
    for point in band_path.points:
        operator = KohnShamOperator.from_density(
            system.grid,
            v_local,
            scf_result.density,
            xc_functional=xc_functional,
            kpoint=point.vector,
        )
        diagonalized = DenseHamiltonianReference(operator).diagonalize(n_bands)
        values.append(np.array(diagonalized.eigenvalues, dtype=np.float32))
    return BandStructureResult(
        kpoints=band_path.points,
        eigenvalues=mx.array(np.stack(values)),
        reused_density=True,
    )
