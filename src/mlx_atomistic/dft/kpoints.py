"""k-point meshes and band-structure diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft.nonlocal_pseudopotential import NonlocalPseudopotentialOperator
from mlx_atomistic.dft.operators import DenseHamiltonianReference, KohnShamOperator
from mlx_atomistic.dft.scf import SCFResult
from mlx_atomistic.dft.system import DFTSystem
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional


@dataclass(frozen=True)
class KPoint:
    """One reciprocal-space k point.

    Args:
        vector: Three-component k-point vector. Cartesian vectors are in the
            same reciprocal units as `ReciprocalGrid.vectors`; reduced vectors
            are fractional diagnostic coordinates and are not accepted by
            Hamiltonian evaluation.
        weight: Positive integration weight. Defaults to ``1.0``.
        label: Optional display label such as ``"Γ"``. Defaults to ``None``.
        coordinate_system: Either ``"cartesian"`` or ``"reduced"``. Defaults
            to ``"cartesian"``.
    """

    vector: tuple[float, float, float]
    weight: float = 1.0
    label: str | None = None
    coordinate_system: str = "cartesian"

    def __init__(
        self,
        vector: Sequence[float],
        *,
        weight: float = 1.0,
        label: str | None = None,
        coordinate_system: str = "cartesian",
    ):
        if len(vector) != 3:
            msg = "k-point vector must have three components"
            raise ValueError(msg)
        if weight <= 0.0:
            msg = "k-point weight must be positive"
            raise ValueError(msg)
        if coordinate_system not in {"cartesian", "reduced"}:
            msg = "coordinate_system must be 'cartesian' or 'reduced'"
            raise ValueError(msg)
        object.__setattr__(self, "vector", tuple(float(value) for value in vector))
        object.__setattr__(self, "weight", float(weight))
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "coordinate_system", coordinate_system)

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
            "coordinate_system": self.coordinate_system,
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
            KPoint(
                point.vector,
                weight=point.weight / weight_sum,
                label=point.label,
                coordinate_system=point.coordinate_system,
            )
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
            points.append(KPoint(vector, weight=1.0 / total, coordinate_system="reduced"))
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
    """Non-SCF band energies along a path.

    Args:
        kpoints: Cartesian k-points evaluated in path order.
        eigenvalues: Eigenvalue array with shape ``(n_kpoints, n_bands)``.
        reused_density: Whether the calculation reused the SCF density without
            another SCF cycle.
        nonlocal_available: Whether ion-backed nonlocal projector metadata was
            available on the system.
        nonlocal_applied: Whether nonlocal projectors were applied to the band
            Hamiltonian.
        nonlocal_projector_count: Number of projector channels included when
            nonlocal projectors were applied.
    """

    kpoints: tuple[KPoint, ...]
    eigenvalues: mx.array
    reused_density: bool
    nonlocal_available: bool = False
    nonlocal_applied: bool = False
    nonlocal_projector_count: int = 0

    def to_dict(self) -> dict:
        """Return JSON-safe band data."""

        return {
            "kpoints": [point.to_dict() for point in self.kpoints],
            "eigenvalues": np.array(self.eigenvalues).tolist(),
            "reused_density": self.reused_density,
            "nonlocal_available": self.nonlocal_available,
            "nonlocal_applied": self.nonlocal_applied,
            "nonlocal_projector_count": self.nonlocal_projector_count,
        }


def _is_gamma(point: KPoint, *, atol: float = 1e-12) -> bool:
    return all(abs(component) <= atol for component in point.vector)


def _validate_cartesian_band_path(band_path: BandPath) -> None:
    for point in band_path.points:
        if point.coordinate_system != "cartesian":
            msg = "run_band_structure requires cartesian k-points"
            raise ValueError(msg)


def run_band_structure(
    system: DFTSystem,
    scf_result: SCFResult,
    band_path: BandPath,
    *,
    n_bands: int = 1,
    xc_functional: ExchangeCorrelationFunctional | None = None,
    apply_nonlocal: bool | None = None,
) -> BandStructureResult:
    """Evaluate non-SCF bands on top of a converged density.

    Args:
        system: DFT system that supplied the SCF density.
        scf_result: Converged or diagnostic SCF result whose density is reused.
        band_path: Explicit k-point path.
        n_bands: Number of eigenvalues to report at each k-point. Defaults to ``1``.
        xc_functional: Exchange-correlation functional for the fixed-density operator;
            ``None`` uses LDA. Defaults to ``None``.
        apply_nonlocal: Whether to include ion-backed nonlocal pseudopotential
            projectors. ``None`` mirrors ``scf_result.nonlocal_applied``. Defaults to
            ``None``.

    Returns:
        Non-SCF band energies and pseudopotential diagnostics.
    """

    if n_bands <= 0:
        msg = "n_bands must be positive"
        raise ValueError(msg)
    if n_bands > system.grid.size:
        msg = "n_bands cannot exceed the real-space grid size"
        raise ValueError(msg)
    _validate_cartesian_band_path(band_path)
    v_local = system.pseudopotential.field(system.grid)
    nonlocal_available = bool(system.ions is not None and system.ions.nonlocal_available)
    should_apply_nonlocal = (
        scf_result.nonlocal_applied if apply_nonlocal is None else bool(apply_nonlocal)
    )
    if should_apply_nonlocal and nonlocal_available and any(
        not _is_gamma(point) for point in band_path.points
    ):
        msg = "nonlocal band diagnostics are currently limited to Γ-point paths"
        raise ValueError(msg)
    nonlocal_operator = None
    nonlocal_projector_count = 0
    if should_apply_nonlocal and nonlocal_available and system.ions is not None:
        nonlocal_operator = NonlocalPseudopotentialOperator.from_ions(system.ions, system.grid)
        nonlocal_projector_count = nonlocal_operator.projectors.count
    nonlocal_applied = bool(nonlocal_operator is not None and nonlocal_operator.available)
    values = []
    for point in band_path.points:
        operator = KohnShamOperator.from_density(
            system.grid,
            v_local,
            scf_result.density,
            xc_functional=xc_functional,
            nonlocal_operator=nonlocal_operator,
            kpoint=point.vector,
        )
        diagonalized = DenseHamiltonianReference(operator).diagonalize(n_bands)
        values.append(np.array(diagonalized.eigenvalues, dtype=np.float32))
    return BandStructureResult(
        kpoints=band_path.points,
        eigenvalues=mx.array(np.stack(values)),
        reused_density=True,
        nonlocal_available=nonlocal_available,
        nonlocal_applied=nonlocal_applied,
        nonlocal_projector_count=nonlocal_projector_count,
    )
