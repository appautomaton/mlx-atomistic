"""Frozen variational-state energies for periodic cell derivatives."""

from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft._compact import _CompactLaneState, _remap_initial_coefficients
from mlx_atomistic.dft._periodic_density import _density_from_kpoints
from mlx_atomistic.dft._periodic_hamiltonian import PeriodicKohnShamOperator
from mlx_atomistic.dft._periodic_models import (
    PeriodicDFTSystem,
    PeriodicEigenResult,
    PeriodicKPointResult,
    PeriodicSCFConfig,
    PeriodicSCFResult,
)
from mlx_atomistic.dft._runtime_observer import RuntimeObserver
from mlx_atomistic.dft.gga import ProductionPBEExchangeCorrelation
from mlx_atomistic.dft.grids import ReciprocalGrid
from mlx_atomistic.dft.kpoints import KPointMesh
from mlx_atomistic.dft.periodic_electrostatics import periodic_ewald_energy
from mlx_atomistic.dft.periodic_gth import (
    PeriodicGTHNonlocalOperator,
    _GTHProjectorCache,
    gth_local_potential_grid,
)
from mlx_atomistic.dft.plane_wave import PlaneWaveBasis
from mlx_atomistic.dft.potentials import hartree_potential
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional


@dataclass(frozen=True)
class _PeriodicFrozenEnergyResult:
    """One frozen variational-state energy on a replacement cell."""

    total_energy: float
    energy_by_term: dict[str, float]
    electron_count: float
    active_counts: tuple[int, ...]


def _transported_bases(
    system: PeriodicDFTSystem,
    source: PeriodicSCFResult,
    *,
    cutoff_hartree: float,
    kpoint_mesh: KPointMesh,
) -> tuple[PlaneWaveBasis, ...]:
    if len(source.kpoints) != len(kpoint_mesh.points):
        raise ValueError("source k-point count does not match the fixed mesh")
    reciprocal = ReciprocalGrid.from_real_space(system.grid)
    bases = []
    for index, (point, source_point) in enumerate(
        zip(kpoint_mesh.points, source.kpoints, strict=True)
    ):
        if not np.allclose(
            point.vector,
            source_point.reduced_kpoint,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("source k-point order does not match the fixed mesh")
        bases.append(
            PlaneWaveBasis.from_reduced_kpoint(
                system.grid,
                cutoff_hartree,
                point.vector,
                reciprocal_grid=reciprocal,
                lane_label=f"kpoint:{index}",
                active_integer_g=np.asarray(source_point.basis.active_integer_g),
            )
        )
    return tuple(bases)


def _transported_owned_results(
    source: PeriodicSCFResult,
    bases: tuple[PlaneWaveBasis, ...],
) -> tuple[PeriodicKPointResult, ...]:
    results = []
    for point in source.owned_kpoints:
        index = point.explicit_index
        state = point.eigen._compact_coefficients
        if index is None or not isinstance(state, _CompactLaneState):
            raise ValueError("source owned k-points require compact explicit states")
        basis = bases[index]
        transported = _remap_initial_coefficients(state, basis._layout)
        eigen = PeriodicEigenResult._from_compact(
            eigenvalues=point.eigen.eigenvalues,
            compact_coefficients=transported,
            basis=basis,
            residuals=point.eigen.residuals,
            orthonormality_error=point.eigen.orthonormality_error,
            iterations=0,
            converged=True,
            subspace_size=point.eigen.subspace_size,
            restart_count=0,
        )
        results.append(replace(point, basis=basis, eigen=eigen))
    return tuple(results)


def _frozen_band_energy(
    results: tuple[PeriodicKPointResult, ...],
    effective_potential: mx.array,
    system: PeriodicDFTSystem,
    *,
    config: PeriodicSCFConfig,
    observer: RuntimeObserver | None,
) -> float:
    policy = config._compact_batch_policy()
    total = 0.0
    with _GTHProjectorCache() as cache:
        operators = [
            PeriodicKohnShamOperator._from_shared_potential(
                point.basis,
                effective_potential,
                PeriodicGTHNonlocalOperator(
                    system.pseudopotentials,
                    point.basis,
                    system.positions,
                    cache=cache,
                ),
                observer,
            )
            for point in results
        ]
        states = [point.eigen._compact_coefficients for point in results]
        if any(not isinstance(state, _CompactLaneState) for state in states):
            raise ValueError("frozen energy requires compact coefficient states")
        for start in range(0, len(results), policy.batch_cap):
            stop = min(len(results), start + policy.batch_cap)
            group_states = states[start:stop]
            outcome = PeriodicKohnShamOperator._apply_compact_batch(
                operators[start:stop],
                group_states,
                observer=observer,
                policy=policy,
            )
            for offset, (point, state) in enumerate(
                zip(results[start:stop], group_states, strict=True)
            ):
                action = outcome.action_for(offset)
                numerator = mx.sum(mx.conjugate(state.values) * action.values, axis=1)
                denominator = mx.sum(mx.abs(state.values) ** 2, axis=1)
                rayleigh = np.asarray(mx.real(numerator / denominator), dtype=np.float64)
                occupations = point.occupations
                if occupations is None:
                    occupation_values = np.full(rayleigh.shape, 2.0, dtype=np.float64)
                else:
                    occupation_values = np.asarray(occupations, dtype=np.float64)
                total += point.integration_weight * float(
                    np.dot(occupation_values, rayleigh)
                )
    return total


def _evaluate_periodic_frozen_energy(
    system: PeriodicDFTSystem,
    source: PeriodicSCFResult,
    *,
    cutoff_hartree: float,
    kpoint_mesh: KPointMesh,
    config: PeriodicSCFConfig,
    xc_functional: ExchangeCorrelationFunctional | None,
    observer: RuntimeObserver | None,
) -> _PeriodicFrozenEnergyResult:
    """Evaluate the source variational state on a replacement periodic cell."""

    bases = _transported_bases(
        system,
        source,
        cutoff_hartree=cutoff_hartree,
        kpoint_mesh=kpoint_mesh,
    )
    owned = _transported_owned_results(source, bases)
    density = _density_from_kpoints(
        owned,
        occupation=2.0 if source.smearing_width_hartree is None else None,
        policy=config._compact_batch_policy(),
        observer=observer,
    )
    electron_count = float(mx.sum(density) * system.grid.dv)
    density = density * (system.electron_count / electron_count)
    electron_count = float(mx.sum(density) * system.grid.dv)
    xc = ProductionPBEExchangeCorrelation() if xc_functional is None else xc_functional
    hartree = hartree_potential(density, system.grid)
    xc_result = xc.evaluate(density, system.grid)
    gamma_basis = PlaneWaveBasis(
        system.grid,
        cutoff_hartree,
        lane_label="gamma-local-potential",
    )
    local = gth_local_potential_grid(
        system.pseudopotentials,
        gamma_basis,
        system.positions,
    )
    effective = mx.array(local + hartree + xc_result.potential)
    band_energy = _frozen_band_energy(
        owned,
        effective,
        system,
        config=config,
        observer=observer,
    )
    hartree_energy = 0.5 * float(mx.sum(density * hartree) * system.grid.dv)
    xc_energy = float(xc_result.total_energy)
    density_xc = float(mx.sum(density * xc_result.potential) * system.grid.dv)
    ewald = periodic_ewald_energy(system.charges, system.positions, system.grid.cell)
    internal = band_energy - hartree_energy + xc_energy - density_xc + ewald
    entropy_correction = -(
        0.0
        if source.smearing_width_hartree is None
        else source.smearing_width_hartree * source.electronic_entropy
    )
    total = internal + entropy_correction
    terms = {
        "band": band_energy,
        "hartree": hartree_energy,
        "xc": xc_energy,
        "density_xc_potential": density_xc,
        "ion_ewald": ewald,
        "total": total,
    }
    if source.smearing_width_hartree is not None:
        terms.update(
            {
                "internal_total": internal,
                "entropy_correction": entropy_correction,
            }
        )
    return _PeriodicFrozenEnergyResult(
        total_energy=total,
        energy_by_term=terms,
        electron_count=electron_count,
        active_counts=tuple(basis.active_count for basis in bases),
    )
