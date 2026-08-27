"""Analytic fixed-cell forces for converged periodic plane-wave DFT."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft._compact import _CompactLaneState
from mlx_atomistic.dft._memory import _bounded_dft_allocator
from mlx_atomistic.dft._periodic_models import (
    PeriodicDFTSystem,
    PeriodicKPointResult,
    PeriodicSCFResult,
)
from mlx_atomistic.dft._periodic_pseudopotential_runtime import (
    _periodic_local_forces,
    _periodic_nonlocal_operator,
    _periodic_pseudopotential_format,
)
from mlx_atomistic.dft.periodic_electrostatics import periodic_ewald_forces
from mlx_atomistic.dft.periodic_gth import _ProjectorCache

_FORCE_KPOINT_MATERIALIZATION_BATCH_SIZE = 8


@dataclass(frozen=True)
class PeriodicForceResult:
    """Hellmann--Feynman force decomposition at a converged SCF state.

    Args:
        forces: Total ionic forces in Hartree/bohr.
        local: Local pseudopotential electron-ion contribution.
        nonlocal_force: Nonlocal pseudopotential projector contribution.
        ion_ewald: Periodic ion-ion Ewald contribution.
        timings: Wall-clock phase timings in milliseconds.
        provenance: Method and stationarity metadata.
    """

    forces: mx.array
    local: mx.array
    nonlocal_force: mx.array
    ion_ewald: mx.array
    timings: dict[str, float]
    provenance: dict[str, str]

    @property
    def max_force(self) -> float:
        """Return the largest ionic force norm in Hartree/bohr."""

        norms = mx.sqrt(mx.sum(self.forces * self.forces, axis=1))
        return float(mx.max(norms))

    @property
    def net_force(self) -> tuple[float, float, float]:
        """Return the translational force residual in Hartree/bohr."""

        values = np.asarray(mx.sum(self.forces, axis=0), dtype=np.float64)
        return tuple(float(value) for value in values)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe force decomposition."""

        pseudopotential_format = self.provenance.get(
            "pseudopotential_format",
            "gth",
        )
        return {
            "forces_hartree_per_bohr": np.asarray(self.forces).tolist(),
            "force_by_term_hartree_per_bohr": {
                f"local_{pseudopotential_format}": np.asarray(self.local).tolist(),
                f"nonlocal_{pseudopotential_format}": np.asarray(
                    self.nonlocal_force
                ).tolist(),
                "ion_ewald": np.asarray(self.ion_ewald).tolist(),
            },
            "max_force_hartree_per_bohr": self.max_force,
            "net_force_hartree_per_bohr": list(self.net_force),
            "timings_ms": dict(self.timings),
            "provenance": dict(self.provenance),
        }


def _validated_periodic_force_state(
    system: PeriodicDFTSystem,
    result: PeriodicSCFResult,
) -> tuple[mx.array, tuple[PeriodicKPointResult, ...]]:
    if not result.converged:
        msg = "periodic forces require a converged SCF result"
        raise ValueError(msg)
    if result.point_group_symmetry_reduced:
        msg = "periodic forces do not yet reconstruct point-group symmetry orbits"
        raise ValueError(msg)
    if result.system_fingerprint != system.fingerprint:
        msg = "SCF result does not match the periodic force system"
        raise ValueError(msg)
    if result.density.shape != system.grid.shape:
        msg = "SCF density shape does not match the periodic force system"
        raise ValueError(msg)
    if not np.isclose(
        result.electron_count,
        system.electron_count,
        rtol=0.0,
        atol=1e-4,
    ):
        msg = "SCF electron count does not match the periodic force system"
        raise ValueError(msg)
    owned = result.owned_kpoints
    if not owned:
        msg = "periodic forces require retained occupied k-point states"
        raise ValueError(msg)
    if any(
        point.basis.grid.shape != system.grid.shape
        or not np.array_equal(
            np.asarray(point.basis.grid.cell.matrix, dtype=np.float64),
            np.asarray(system.grid.cell.matrix, dtype=np.float64),
        )
        for point in owned
    ):
        msg = "SCF k-point bases do not match the periodic force system"
        raise ValueError(msg)
    density = mx.real(mx.array(result.density)).astype(mx.float32)
    density_finite = mx.all(mx.isfinite(density))
    density_count = mx.sum(density) * system.grid.dv
    mx.eval(density, density_finite, density_count)
    if not bool(density_finite) or not np.isclose(
        float(density_count),
        system.electron_count,
        rtol=0.0,
        atol=1e-4,
    ):
        msg = "SCF density is non-finite or has the wrong electron count"
        raise ValueError(msg)
    return density, owned


def _periodic_nonlocal_forces(
    system: PeriodicDFTSystem,
    owned: Sequence[PeriodicKPointResult],
    *,
    projector_cache: _ProjectorCache,
) -> mx.array:
    force = mx.zeros((system.ion_count, 3), dtype=mx.float32)
    for point_index, point in enumerate(owned, start=1):
        compact = point.eigen._compact_coefficients
        if not isinstance(compact, _CompactLaneState):
            msg = "periodic forces require compact occupied k-point states"
            raise ValueError(msg)
        operator = _periodic_nonlocal_operator(
            system.pseudopotentials,
            point.basis,
            system.positions,
            cache=projector_cache,
        )
        occupations = point.occupations
        if occupations is None:
            occupations = (2.0,) * compact.vector_count
        if len(occupations) != compact.vector_count:
            msg = "periodic force occupations must match the retained band count"
            raise ValueError(msg)
        point_force = operator._forces_compact(
            compact,
            occupations=occupations,
            evaluate=False,
        )
        force = force + float(point.integration_weight) * point_force
        # Bound the lazy projector graph without serializing every k-point.
        if point_index % _FORCE_KPOINT_MATERIALIZATION_BATCH_SIZE == 0:
            mx.eval(force)
    if len(owned) % _FORCE_KPOINT_MATERIALIZATION_BATCH_SIZE:
        mx.eval(force)
    return force


def _validated_force_sum(
    local: mx.array,
    nonlocal_force: mx.array,
    ion_ewald: mx.array,
) -> mx.array:
    forces = (local + nonlocal_force + ion_ewald).astype(mx.float32)
    finite = (
        mx.all(mx.isfinite(local))
        & mx.all(mx.isfinite(nonlocal_force))
        & mx.all(mx.isfinite(ion_ewald))
        & mx.all(mx.isfinite(forces))
    )
    mx.eval(forces, finite)
    if not bool(finite):
        msg = "periodic force evaluation produced a non-finite value"
        raise ValueError(msg)
    return forces


def periodic_scf_forces(
    system: PeriodicDFTSystem,
    result: PeriodicSCFResult,
    *,
    ewald_tolerance: float = 1e-10,
) -> PeriodicForceResult:
    """Evaluate fixed-cell periodic forces from a converged SCF result.

    The local and nonlocal electron-ion terms use analytic derivatives of the
    pseudopotential phase factors. The ion-ion term uses the analytic Ewald
    derivative.
    There is no ionic Pulay term because the fixed-cell plane-wave basis does
    not depend on ion positions. Finite-temperature results use their resolved
    occupations, yielding the stationary electronic free-energy force.

    Args:
        system: Exact periodic system used for the SCF calculation.
        result: Converged periodic SCF result retaining compact occupied states.
        ewald_tolerance: Real/reciprocal Ewald truncation target. Defaults to
            ``1e-10``.

    Returns:
        Total forces, component forces, timings, and provenance.

    Raises:
        TypeError: If ``system`` or ``result`` has an unsupported type.
        ValueError: If the SCF state is unconverged, mismatched, incomplete, or
            non-finite.
    """

    if not isinstance(system, PeriodicDFTSystem):
        msg = "system must be PeriodicDFTSystem"
        raise TypeError(msg)
    if not isinstance(result, PeriodicSCFResult):
        msg = "result must be PeriodicSCFResult"
        raise TypeError(msg)
    density, owned = _validated_periodic_force_state(system, result)
    timings = {
        "local": 0.0,
        "nonlocal": 0.0,
        "ion_ewald": 0.0,
        "total": 0.0,
    }
    total_start = perf_counter()
    pseudopotential_format = _periodic_pseudopotential_format(
        system.pseudopotentials
    ).value
    with _bounded_dft_allocator(), _ProjectorCache() as projector_cache:
        phase_start = perf_counter()
        local = _periodic_local_forces(
            density,
            system.pseudopotentials,
            owned[0].basis,
            system.positions,
        )
        timings["local"] = (perf_counter() - phase_start) * 1000.0

        phase_start = perf_counter()
        nonlocal_force = _periodic_nonlocal_forces(
            system,
            owned,
            projector_cache=projector_cache,
        )
        timings["nonlocal"] = (perf_counter() - phase_start) * 1000.0

    phase_start = perf_counter()
    ion_ewald = mx.array(
        periodic_ewald_forces(
            system.charges,
            system.positions,
            system.grid.cell,
            tolerance=ewald_tolerance,
            method="analytic",
        ).astype(np.float32)
    )
    mx.eval(ion_ewald)
    timings["ion_ewald"] = (perf_counter() - phase_start) * 1000.0

    forces = _validated_force_sum(local, nonlocal_force, ion_ewald)
    timings["total"] = (perf_counter() - total_start) * 1000.0
    return PeriodicForceResult(
        forces=forces,
        local=local,
        nonlocal_force=nonlocal_force,
        ion_ewald=ion_ewald,
        timings=timings,
        provenance={
            "pseudopotential_format": pseudopotential_format,
            f"local_{pseudopotential_format}": (
                "analytic_reciprocal_density_phase_derivative"
            ),
            f"nonlocal_{pseudopotential_format}": (
                "analytic_projector_phase_derivative"
            ),
            "ion_ewald": "analytic_ewald_derivative",
            "pulay": "zero_for_fixed_cell_plane_wave_basis",
            "stationarity": "converged_scf_required",
        },
    )
