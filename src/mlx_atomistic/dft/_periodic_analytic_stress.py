"""Differentiable frozen-energy graph for periodic analytic stress."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import NamedTuple

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft._compact import _CompactLaneState
from mlx_atomistic.dft._periodic_density import _density_from_kpoints
from mlx_atomistic.dft._periodic_models import (
    PeriodicDFTSystem,
    PeriodicKPointResult,
    PeriodicSCFResult,
)
from mlx_atomistic.dft.gga import (
    PBEExchangeCorrelation,
    ProductionPBEExchangeCorrelation,
)
from mlx_atomistic.dft.grids import ReciprocalGrid
from mlx_atomistic.dft.periodic_electrostatics import (
    periodic_ewald_energy,
    periodic_ewald_stress,
)
from mlx_atomistic.dft.periodic_gth import (
    _GTH_OVERLAP_CHUNK_SIZE,
    _flattened_gth_coupling,
    _gth_radial,
    _per_ion_pseudopotentials,
    _real_spherical_harmonics,
)
from mlx_atomistic.dft.pseudopotentials import PseudopotentialData
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional

_ENERGY_TERM_NAMES = (
    "kinetic",
    "local_gth",
    "nonlocal_gth",
    "hartree",
    "xc",
    "ion_ewald",
    "entropy_correction",
    "total",
)
_MAX_STATIONARY_CALIBRATION_HARTREE_PER_ELECTRON = 1.0e-3


class _EnergyTerms(NamedTuple):
    kinetic: mx.array
    local_gth: mx.array
    nonlocal_gth: mx.array
    hartree: mx.array
    xc: mx.array
    ion_ewald: mx.array
    entropy_correction: mx.array
    total: mx.array


@dataclass(frozen=True)
class _AnalyticKPoint:
    """Frozen compact state and geometry-independent nonlocal metadata."""

    base_vectors: mx.array
    coefficients: mx.array
    occupations: mx.array
    integration_weight: float
    phases: tuple[mx.array, ...]


@dataclass(frozen=True)
class _PeriodicAnalyticStressEvaluation:
    """Internal analytic stress values before public result assembly."""

    stress: np.ndarray
    pressure: float
    energy_by_term: dict[str, float]
    base_energy_error: float


def _determinant_3x3(matrix: mx.array) -> mx.array:
    """Return a differentiable determinant for one 3 by 3 matrix."""

    return (
        matrix[0, 0]
        * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
        - matrix[0, 1]
        * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
        + matrix[0, 2]
        * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
    )


def _inverse_3x3(matrix: mx.array) -> mx.array:
    """Return a differentiable inverse for one nonsingular 3 by 3 matrix."""

    a, b, c = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    d, e, f = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    g, h, i = matrix[2, 0], matrix[2, 1], matrix[2, 2]
    adjugate = mx.stack(
        [
            mx.stack([e * i - f * h, c * h - b * i, b * f - c * e]),
            mx.stack([f * g - d * i, a * i - c * g, c * d - a * f]),
            mx.stack([d * h - e * g, b * g - a * h, a * e - b * d]),
        ]
    )
    return adjugate / _determinant_3x3(matrix)


def _occupations(point: PeriodicKPointResult, vector_count: int) -> mx.array:
    values = (
        np.full((vector_count,), 2.0, dtype=np.float32)
        if point.occupations is None
        else np.asarray(point.occupations, dtype=np.float32)
    )
    if values.shape != (vector_count,) or not np.all(np.isfinite(values)):
        raise ValueError("analytic stress requires one finite occupation per band")
    return mx.array(values)


def _local_single_coefficients(
    pseudopotential: PseudopotentialData,
    g2: mx.array,
    volume: mx.array,
) -> mx.array:
    rloc = float(pseudopotential.gth_rloc)
    coefficients = list(pseudopotential.gth_coefficients) + [0.0] * 4
    c1, c2, c3, c4 = coefficients[:4]
    zion = float(pseudopotential.valence_charge)
    rq2 = g2 * rloc * rloc
    gaussian = mx.exp(-0.5 * rq2)
    polynomial = (
        c1
        + c2 * (3.0 - rq2)
        + c3 * (15.0 - 10.0 * rq2 + rq2 * rq2)
        + c4 * (105.0 - rq2 * (105.0 - rq2 * (21.0 - rq2)))
    )
    safe_g2 = mx.where(g2 > 1.0e-14, g2, 1.0)
    nonzero = (
        4.0
        * pi
        * gaussian
        * (
            -zion / safe_g2
            + sqrt(pi / 2.0) * rloc**3 * polynomial
        )
        / volume
    )
    zero = (
        2.0 * pi * rloc * rloc * zion
        + (2.0 * pi) ** 1.5
        * rloc**3
        * (c1 + 3.0 * c2 + 15.0 * c3 + 105.0 * c4)
    ) / volume
    return mx.where(g2 > 1.0e-14, nonzero, zero)


def _local_gth_energy(
    density: mx.array,
    vectors: mx.array,
    volume: mx.array,
    grid_size: int,
    pseudopotentials: tuple[PseudopotentialData, ...],
    phases: tuple[mx.array, ...],
) -> mx.array:
    g2 = mx.sum(vectors * vectors, axis=-1)
    coefficients = mx.zeros(g2.shape, dtype=mx.complex64)
    for pseudopotential, phase in zip(pseudopotentials, phases, strict=True):
        single = _local_single_coefficients(pseudopotential, g2, volume)
        coefficients = coefficients + single * phase
    potential = mx.real(mx.fft.ifftn(coefficients) * grid_size)
    return mx.sum(density * potential) * volume / grid_size


def _hartree_energy(
    density: mx.array,
    vectors: mx.array,
    volume: mx.array,
    grid_size: int,
) -> mx.array:
    density_g = mx.fft.fftn(density)
    g2 = mx.sum(vectors * vectors, axis=-1)
    safe_g2 = mx.where(g2 > 1.0e-14, g2, 1.0)
    potential_g = mx.where(
        g2 > 1.0e-14,
        4.0 * pi * density_g / safe_g2,
        mx.zeros_like(density_g),
    )
    potential = mx.real(mx.fft.ifftn(potential_g))
    return 0.5 * mx.sum(density * potential) * volume / grid_size


def _xc_energy(
    functional: ExchangeCorrelationFunctional,
    density: mx.array,
    vectors: mx.array,
    volume: mx.array,
    grid_size: int,
) -> mx.array:
    if not isinstance(
        functional,
        (PBEExchangeCorrelation, ProductionPBEExchangeCorrelation),
    ):
        raise ValueError(
            "analytic periodic stress currently requires an MLX PBE functional"
        )
    density_g = mx.fft.fftn(density)
    gradient = mx.stack(
        [
            mx.real(mx.fft.ifftn(1j * vectors[..., axis] * density_g))
            for axis in range(3)
        ],
        axis=0,
    )
    energy_density = functional._energy_density_from_gradient(
        density,
        gradient,
        1.0e-12,
    )
    return mx.sum(energy_density) * volume / grid_size


def _nonlocal_projectors(
    point: _AnalyticKPoint,
    vectors: mx.array,
    volume: mx.array,
    pseudopotentials: tuple[PseudopotentialData, ...],
) -> mx.array:
    q2 = mx.sum(vectors * vectors, axis=-1)
    q = mx.sqrt(mx.maximum(q2, 1.0e-20))
    rows = []
    for ion_index, pseudopotential in enumerate(pseudopotentials):
        for channel in pseudopotential.gth_channels:
            harmonics = _real_spherical_harmonics(
                channel.angular_momentum,
                vectors,
                q,
            )
            angular_phase = (-1j) ** channel.angular_momentum
            prefactor = (
                4.0
                * pi
                * pi**0.25
                * mx.sqrt(
                    2.0 ** (channel.angular_momentum + 1)
                    * channel.radius ** (2 * channel.angular_momentum + 3)
                    / volume
                )
            )
            for harmonic in harmonics:
                rows.append(
                    mx.stack(
                        [
                            (
                                prefactor
                                * _gth_radial(channel, index, q)
                                * harmonic
                                * point.phases[ion_index]
                                * angular_phase
                            ).astype(mx.complex64)
                            for index in range(channel.projector_count)
                        ],
                        axis=0,
                    )
                )
    return mx.concatenate(rows, axis=0)


def _compensated_projector_overlaps(
    projectors: mx.array,
    coefficients: mx.array,
) -> mx.array:
    """Match the production GTH overlap accumulation for one k-point."""

    coefficient_matrix = mx.transpose(coefficients)
    overlaps = mx.zeros(
        (projectors.shape[0], coefficients.shape[0]),
        dtype=mx.complex64,
    )
    compensation = mx.zeros_like(overlaps)
    for start in range(0, int(projectors.shape[1]), _GTH_OVERLAP_CHUNK_SIZE):
        stop = min(start + _GTH_OVERLAP_CHUNK_SIZE, int(projectors.shape[1]))
        partial = mx.matmul(
            mx.conjugate(projectors[:, start:stop]),
            coefficient_matrix[start:stop],
        )
        adjusted = partial - compensation
        updated = overlaps + adjusted
        compensation = (updated - overlaps) - adjusted
        overlaps = updated
    return overlaps


class _PeriodicAnalyticStressGraph:
    """One immutable fixed-topology strain-energy graph."""

    def __init__(
        self,
        system: PeriodicDFTSystem,
        result: PeriodicSCFResult,
        xc_functional: ExchangeCorrelationFunctional,
    ) -> None:
        self.functional = xc_functional
        self.base_volume = float(system.grid.volume)
        self.grid_size = int(system.grid.size)
        density = _density_from_kpoints(
            result.owned_kpoints,
            occupation=2.0 if result.smearing_width_hartree is None else None,
        )
        density_count = float(mx.sum(density) * system.grid.dv)
        if not np.isfinite(density_count) or density_count <= 0.0:
            raise ValueError("analytic stress orbital density is non-finite or empty")
        self.base_density = (
            density * (system.electron_count / density_count)
        ).astype(mx.float32)
        mx.eval(self.base_density)
        reciprocal = ReciprocalGrid.from_real_space(system.grid)
        self.base_reciprocal_vectors = mx.array(reciprocal.vectors)
        self.pseudopotentials = _per_ion_pseudopotentials(
            system.pseudopotentials,
            len(system.positions),
        )
        positions = np.asarray(system.positions, dtype=np.float32)
        full_vectors = np.asarray(reciprocal.vectors, dtype=np.float32)
        self.local_phases = tuple(
            mx.array(np.exp(-1j * np.sum(full_vectors * center, axis=-1)).astype(np.complex64))
            for center in positions
        )
        points = []
        for point in result.owned_kpoints:
            state = point.eigen._compact_coefficients
            if not isinstance(state, _CompactLaneState):
                raise ValueError("analytic stress requires retained compact orbitals")
            base_vectors = np.asarray(
                point.basis.active_shifted_vectors,
                dtype=np.float32,
            )
            points.append(
                _AnalyticKPoint(
                    base_vectors=mx.array(base_vectors),
                    coefficients=state.values,
                    occupations=_occupations(point, state.vector_count),
                    integration_weight=float(point.integration_weight),
                    phases=tuple(
                        mx.array(
                            np.exp(-1j * (base_vectors @ center)).astype(np.complex64)
                        )
                        for center in positions
                    ),
                )
            )
        if not points:
            raise ValueError("analytic stress requires at least one owned k-point")
        self.points = tuple(points)
        self.nonlocal_coupling, _ = _flattened_gth_coupling(
            self.pseudopotentials
        )
        self.ewald_energy = periodic_ewald_energy(
            system.charges,
            system.positions,
            system.grid.cell,
        )
        self.entropy_correction = -(
            0.0
            if result.smearing_width_hartree is None
            else result.smearing_width_hartree * result.electronic_entropy
        )

    def energy_terms(self, strain: mx.array) -> _EnergyTerms:
        deformation = mx.eye(3) + strain
        volume_scale = _determinant_3x3(deformation)
        volume = self.base_volume * volume_scale
        reciprocal_deformation = mx.transpose(_inverse_3x3(deformation))
        reciprocal_vectors = self.base_reciprocal_vectors @ reciprocal_deformation
        density = self.base_density / volume_scale

        kinetic = mx.array(0.0, dtype=mx.float32)
        nonlocal_gth = mx.array(0.0, dtype=mx.float32)
        for point in self.points:
            vectors = point.base_vectors @ reciprocal_deformation
            q2 = mx.sum(vectors * vectors, axis=1)
            probabilities = mx.abs(point.coefficients) ** 2
            norms = mx.sum(probabilities, axis=1)
            kinetic = kinetic + point.integration_weight * mx.sum(
                point.occupations
                * mx.sum(probabilities * (0.5 * q2)[None, :], axis=1)
                / norms
            )
            beta = _nonlocal_projectors(
                point,
                vectors,
                volume,
                self.pseudopotentials,
            )
            overlaps = _compensated_projector_overlaps(
                beta,
                point.coefficients,
            )
            mixed = mx.matmul(self.nonlocal_coupling, overlaps)
            expectations = mx.real(mx.sum(mx.conjugate(overlaps) * mixed, axis=0))
            nonlocal_gth = nonlocal_gth + point.integration_weight * mx.sum(
                point.occupations * expectations / norms
            )

        local_gth = _local_gth_energy(
            density,
            reciprocal_vectors,
            volume,
            self.grid_size,
            self.pseudopotentials,
            self.local_phases,
        )
        hartree = _hartree_energy(
            density,
            reciprocal_vectors,
            volume,
            self.grid_size,
        )
        xc = _xc_energy(
            self.functional,
            density,
            reciprocal_vectors,
            volume,
            self.grid_size,
        )
        ion_ewald = mx.array(self.ewald_energy, dtype=mx.float32)
        entropy = mx.array(self.entropy_correction, dtype=mx.float32)
        total = kinetic + local_gth + nonlocal_gth + hartree + xc + ion_ewald + entropy
        return _EnergyTerms(
            kinetic,
            local_gth,
            nonlocal_gth,
            hartree,
            xc,
            ion_ewald,
            entropy,
            total,
        )


def _evaluate_periodic_analytic_stress(
    system: PeriodicDFTSystem,
    result: PeriodicSCFResult,
    *,
    xc_functional: ExchangeCorrelationFunctional,
    mode: str,
    variational_energy_tolerance: float,
    reference_energy: float,
) -> _PeriodicAnalyticStressEvaluation:
    """Differentiate one converged fixed-topology periodic free energy."""

    graph = _PeriodicAnalyticStressGraph(system, result, xc_functional)
    zero = mx.zeros((3, 3), dtype=mx.float32)

    base_terms = graph.energy_terms(zero)
    mx.eval(*base_terms)
    correction = float(reference_energy) - float(base_terms.total)
    correction_per_electron = abs(correction) / system.electron_count
    if correction_per_electron > _MAX_STATIONARY_CALIBRATION_HARTREE_PER_ELECTRON:
        raise ValueError(
            "analytic stationary reference correction is too large: "
            f"{correction_per_electron:.6g} Hartree/electron"
        )

    def total_energy(strain: mx.array) -> mx.array:
        return graph.energy_terms(strain).total + correction

    energy, derivative = mx.value_and_grad(total_energy)(zero)
    terms = base_terms
    mx.eval(energy, derivative)
    energy_by_term = {
        name: float(value) for name, value in zip(_ENERGY_TERM_NAMES, terms, strict=True)
    }
    energy_by_term["stationary_reference_correction"] = correction
    energy_by_term["total"] = float(energy)
    base_error = abs(float(energy) - float(result.total_energy))
    if base_error > variational_energy_tolerance:
        raise ValueError(
            "analytic frozen energy differs from the converged SCF by "
            f"{base_error:.6g} Hartree"
        )
    raw = -np.asarray(derivative, dtype=np.float64) / graph.base_volume
    raw += periodic_ewald_stress(
        system.charges,
        system.positions,
        system.grid.cell,
    )
    symmetric = 0.5 * (raw + raw.T)
    pressure = float(np.trace(symmetric) / 3.0)
    if mode == "isotropic":
        stress = np.eye(3, dtype=np.float64) * pressure
    elif mode == "diagonal":
        stress = np.diag(np.diag(symmetric))
    elif mode == "symmetric":
        stress = symmetric
    else:
        raise ValueError("stress mode must be 'isotropic', 'diagonal', or 'symmetric'")
    return _PeriodicAnalyticStressEvaluation(
        stress=stress,
        pressure=pressure,
        energy_by_term=energy_by_term,
        base_energy_error=base_error,
    )
