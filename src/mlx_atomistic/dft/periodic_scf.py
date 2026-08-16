"""Self-consistent weighted k-point plane-wave DFT."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft._compact import (
    _CompactLaneState,
)
from mlx_atomistic.dft._memory import _bounded_dft_allocator
from mlx_atomistic.dft._periodic_davidson import (
    solve_periodic_eigenproblem as solve_periodic_eigenproblem,
)
from mlx_atomistic.dft._periodic_hamiltonian import PeriodicKohnShamOperator
from mlx_atomistic.dft._periodic_models import (
    PeriodicBandPointResult,
    PeriodicBandStructureResult,
    PeriodicDavidsonConfig,
    PeriodicDFTSystem,
    PeriodicEigenResult,
    PeriodicFrozenDensity,
    PeriodicSCFConfig,
    PeriodicSCFResult,
)
from mlx_atomistic.dft._periodic_models import (
    PeriodicKPointResult as PeriodicKPointResult,
)
from mlx_atomistic.dft._periodic_models import (
    _eigensolve_provenance as _periodic_eigensolve_provenance,
)
from mlx_atomistic.dft._periodic_scf_engine import _run_periodic_scf_controlled
from mlx_atomistic.dft._runtime_observer import (
    RuntimeObserver,
)
from mlx_atomistic.dft.gga import ProductionPBEExchangeCorrelation
from mlx_atomistic.dft.grids import ReciprocalGrid
from mlx_atomistic.dft.kpoints import (
    BandPath,
    KPointMesh,
)
from mlx_atomistic.dft.periodic_gth import (
    PeriodicGTHNonlocalOperator,
    _GTHProjectorCache,
    gth_local_potential_grid,
)
from mlx_atomistic.dft.plane_wave import PlaneWaveBasis
from mlx_atomistic.dft.potentials import hartree_potential
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional


def _eigensolve_provenance() -> dict[str, str]:
    """Preserve the periodic SCF provenance-helper import contract."""

    return _periodic_eigensolve_provenance()


def run_periodic_scf(
    system: PeriodicDFTSystem,
    *,
    cutoff_hartree: float,
    kpoint_mesh: KPointMesh,
    n_bands: int | None = None,
    config: PeriodicSCFConfig | None = None,
    xc_functional: ExchangeCorrelationFunctional | None = None,
    initial_density: mx.array | None = None,
    initial_coefficients: Sequence[mx.array] | None = None,
    observer: RuntimeObserver | None = None,
) -> PeriodicSCFResult:
    """Run weighted self-consistent periodic plane-wave DFT.

    Args:
        system: Periodic GTH system.
        cutoff_hartree: Kinetic cutoff in Hartree.
        kpoint_mesh: Weighted reduced-coordinate k-point mesh.
        n_bands: Number of occupied bands. Defaults to half the electron count.
        config: SCF controls. Defaults to `PeriodicSCFConfig`.
        xc_functional: Exchange-correlation functional. Defaults to production PBE.
        initial_density: Optional starting density on the FFT grid.
        initial_coefficients: Optional orbital stack per k-point.
        observer: Optional progress, synchronized timing, and work observer.

    Returns:
        Periodic SCF result with complete weighted k-point diagnostics.
    """

    return _run_periodic_scf_controlled(
        system,
        cutoff_hartree=cutoff_hartree,
        kpoint_mesh=kpoint_mesh,
        n_bands=n_bands,
        config=config,
        xc_functional=xc_functional,
        initial_density=initial_density,
        initial_coefficients=initial_coefficients,
        observer=observer,
    )


def _periodic_band_density_source(
    system: PeriodicDFTSystem,
    source: PeriodicSCFResult | PeriodicFrozenDensity,
) -> tuple[mx.array, float, float, str]:
    """Validate and normalize one frozen-density source."""

    if isinstance(source, PeriodicSCFResult):
        if not source.converged:
            msg = "periodic band structure requires a converged SCF result"
            raise ValueError(msg)
        if not source.kpoints:
            msg = "periodic SCF result has no k-point basis metadata"
            raise ValueError(msg)
        if any(
            point.basis.grid.shape != system.grid.shape
            or not np.array_equal(
                np.asarray(point.basis.grid.cell.matrix, dtype=np.float64),
                np.asarray(system.grid.cell.matrix, dtype=np.float64),
            )
            for point in source.kpoints
        ):
            msg = "periodic SCF result basis grid and cell must match the system"
            raise ValueError(msg)
        cutoffs = np.asarray(
            [point.basis.cutoff_hartree for point in source.kpoints],
            dtype=np.float64,
        )
        if not np.allclose(cutoffs, cutoffs[0], rtol=0.0, atol=1e-12):
            msg = "periodic SCF result contains inconsistent plane-wave cutoffs"
            raise ValueError(msg)
        density = source.density
        cutoff_hartree = float(cutoffs[0])
        electron_count = float(source.electron_count)
        source_kind = "scf_result"
        source_fingerprint = source.system_fingerprint
    elif isinstance(source, PeriodicFrozenDensity):
        density = source.density
        cutoff_hartree = source.cutoff_hartree
        electron_count = source.electron_count
        source_kind = "frozen_density"
        source_fingerprint = source.system_fingerprint
    else:
        msg = "source must be PeriodicSCFResult or PeriodicFrozenDensity"
        raise TypeError(msg)

    if source_fingerprint is None:
        if not system.is_homogeneous:
            msg = "multi-element frozen density requires an exact system fingerprint"
            raise ValueError(msg)
    elif source_fingerprint != system.fingerprint:
        msg = "frozen density pseudopotential assignment must match the periodic system"
        raise ValueError(msg)

    if density.shape != system.grid.shape:
        msg = "frozen density shape must match system.grid.shape"
        raise ValueError(msg)
    if not np.isclose(
        electron_count,
        system.electron_count,
        rtol=0.0,
        atol=1e-4,
    ):
        msg = "frozen density electron count must match the periodic system"
        raise ValueError(msg)
    density_snapshot = mx.real(mx.array(density))
    finite = mx.all(mx.isfinite(density_snapshot))
    minimum = mx.min(density_snapshot)
    integrated = mx.sum(density_snapshot) * system.grid.dv
    mx.eval(density_snapshot, finite, minimum, integrated)
    if not bool(finite):
        msg = "frozen density must contain only finite values"
        raise ValueError(msg)
    if float(minimum) < -1e-6:
        msg = "frozen density must be nonnegative within numerical tolerance"
        raise ValueError(msg)
    count_tolerance = max(1e-4, 1e-5 * electron_count)
    if not np.isclose(
        float(integrated),
        electron_count,
        rtol=0.0,
        atol=count_tolerance,
    ):
        msg = "frozen density integral does not match its electron count"
        raise ValueError(msg)
    return density_snapshot, cutoff_hartree, electron_count, source_kind


def run_periodic_band_structure(
    system: PeriodicDFTSystem,
    source: PeriodicSCFResult | PeriodicFrozenDensity,
    band_path: BandPath,
    *,
    n_bands: int | None = None,
    guard_bands: int = 0,
    config: PeriodicDavidsonConfig | None = None,
    xc_functional: ExchangeCorrelationFunctional | None = None,
    observer: RuntimeObserver | None = None,
) -> PeriodicBandStructureResult:
    """Solve production periodic bands on top of a converged frozen density.

    This is a non-self-consistent calculation: the ionic, Hartree, and
    exchange-correlation potentials are built once from ``source`` and reused
    unchanged at every path point. Each k-point still receives its own
    cutoff-projected plane-wave basis and complete local plus nonlocal GTH
    Hamiltonian.

    Args:
        system: Periodic GTH system matching the source SCF calculation.
        source: Converged periodic SCF result or validated portable density.
        band_path: Explicit reduced-coordinate k-point path.
        n_bands: Lowest bands to return. Defaults to occupied bands plus eight.
        guard_bands: Extra unpublished states used to contain degeneracies at
            the requested eigenspace boundary. Defaults to zero.
        config: Davidson controls. Defaults to ``PeriodicDavidsonConfig``.
        xc_functional: Exchange-correlation functional. Defaults to production
            PBE, matching ``run_periodic_scf``.
        observer: Optional progress and work observer.

    Returns:
        Fixed-density band energies, residuals, bases, and compact eigenstates.

    Raises:
        RuntimeError: If a path-point Davidson solve does not converge.
        TypeError: If ``source`` has an unsupported type.
        ValueError: If source, path, density, or band metadata are inconsistent.
    """

    for point in band_path.points:
        if point.coordinate_system != "reduced":
            msg = "periodic band structure requires reduced-coordinate k-points"
            raise ValueError(msg)
    density, cutoff_hartree, _electron_count, source_kind = _periodic_band_density_source(
        system,
        source,
    )
    physical_electron_count = float(system.electron_count)
    occupied_bands = int(round(physical_electron_count / 2.0))
    if (
        occupied_bands <= 0
        or abs(2.0 * occupied_bands - physical_electron_count) > 1e-8
    ):
        msg = "periodic band structure requires two electrons per occupied band"
        raise ValueError(msg)
    requested_bands = occupied_bands + 8 if n_bands is None else n_bands
    if type(requested_bands) is not int or requested_bands < occupied_bands:
        msg = "n_bands must be an integer no smaller than the occupied band count"
        raise ValueError(msg)
    if type(guard_bands) is not int or guard_bands < 0:
        msg = "guard_bands must be a nonnegative integer"
        raise ValueError(msg)
    solved_bands = requested_bands + guard_bands

    solver_config = PeriodicDavidsonConfig() if config is None else config
    xc = ProductionPBEExchangeCorrelation() if xc_functional is None else xc_functional
    timings = {"setup": 0.0, "eigensolver": 0.0, "total": 0.0}
    total_start = perf_counter()
    with _bounded_dft_allocator(), _GTHProjectorCache() as projector_cache:
        setup_start = perf_counter()
        shared_reciprocal = ReciprocalGrid.from_real_space(system.grid)
        gamma_basis = PlaneWaveBasis(
            system.grid,
            cutoff_hartree,
            reciprocal_grid=shared_reciprocal,
            lane_label="band:local-potential",
        )
        local_potential = gth_local_potential_grid(
            system.pseudopotentials,
            gamma_basis,
            system.positions,
        )
        hartree = hartree_potential(density, system.grid)
        xc_result = xc.evaluate(density, system.grid)
        effective_snapshot = mx.array(local_potential + hartree + xc_result.potential)
        finite = (
            mx.all(mx.isfinite(xc_result.energy_density))
            & mx.all(mx.isfinite(xc_result.potential))
            & mx.isfinite(xc_result.total_energy)
            & mx.all(mx.isfinite(effective_snapshot))
        )
        mx.eval(effective_snapshot, finite)
        if not bool(finite):
            msg = "frozen-density effective potential is non-finite"
            raise ValueError(msg)
        timings["setup"] = (perf_counter() - setup_start) * 1000.0

        point_results: list[PeriodicBandPointResult] = []
        for point_index, point in enumerate(band_path.points):
            basis = PlaneWaveBasis.from_reduced_kpoint(
                system.grid,
                cutoff_hartree,
                point.vector,
                reciprocal_grid=shared_reciprocal,
                lane_label=f"band:kpoint:{point_index}",
            )
            if solved_bands > basis.active_count:
                msg = (
                    f"n_bands plus guard_bands={solved_bands} exceeds active basis size "
                    f"{basis.active_count} at path point {point_index}"
                )
                raise ValueError(msg)
            nonlocal_operator = PeriodicGTHNonlocalOperator(
                system.pseudopotentials,
                basis,
                system.positions,
                cache=projector_cache,
            )
            operator = PeriodicKohnShamOperator._from_shared_potential(
                basis,
                effective_snapshot,
                nonlocal_operator,
                observer,
            )
            if observer is not None:
                observer.emit(
                    "band_kpoint",
                    status="started",
                    point_index=point_index,
                    point_count=len(band_path.points),
                    reduced_kpoint=list(point.vector),
                )
            solve_start = perf_counter()
            eigen = solve_periodic_eigenproblem(
                operator,
                n_bands=solved_bands,
                config=solver_config,
                observer=observer,
            )
            timings["eigensolver"] += (perf_counter() - solve_start) * 1000.0
            if guard_bands:
                compact = eigen._compact_coefficients
                if not isinstance(compact, _CompactLaneState):
                    msg = "guard-band trimming requires compact eigenvectors"
                    raise RuntimeError(msg)
                requested_residuals = eigen.residuals[:requested_bands]
                requested_converged = bool(
                    mx.all(requested_residuals <= solver_config.tolerance)
                )
                eigen = PeriodicEigenResult._from_compact(
                    eigenvalues=eigen.eigenvalues[:requested_bands],
                    compact_coefficients=basis._state_from_compact(
                        compact.values[:requested_bands]
                    ),
                    basis=basis,
                    residuals=requested_residuals,
                    orthonormality_error=eigen.orthonormality_error,
                    iterations=eigen.iterations,
                    converged=requested_converged,
                    subspace_size=eigen.subspace_size,
                    restart_count=eigen.restart_count,
                )
            values_finite = mx.all(mx.isfinite(eigen.eigenvalues))
            residuals_finite = mx.all(mx.isfinite(eigen.residuals))
            mx.eval(values_finite, residuals_finite)
            if not bool(values_finite) or not bool(residuals_finite):
                msg = f"periodic band eigensolve is non-finite at path point {point_index}"
                raise RuntimeError(msg)
            if not eigen.converged:
                worst = float(mx.max(eigen.residuals))
                msg = (
                    f"periodic band eigensolve did not converge at path point "
                    f"{point_index}; worst residual={worst:.6e}"
                )
                raise RuntimeError(msg)
            point_results.append(
                PeriodicBandPointResult(
                    requested_kpoint=point,
                    basis=basis,
                    eigen=eigen,
                )
            )
            if observer is not None:
                observer.emit(
                    "band_kpoint",
                    status="completed",
                    point_index=point_index,
                    iterations=eigen.iterations,
                    worst_residual=float(mx.max(eigen.residuals)),
                )

    eigenvalues = mx.stack([point.eigen.eigenvalues for point in point_results])
    residuals = mx.stack([point.eigen.residuals for point in point_results])
    mx.eval(eigenvalues, residuals)
    timings["total"] = (perf_counter() - total_start) * 1000.0
    return PeriodicBandStructureResult(
        kpoints=band_path.points,
        eigenvalues=eigenvalues,
        residuals=residuals,
        points=tuple(point_results),
        occupied_band_count=occupied_bands,
        cutoff_hartree=cutoff_hartree,
        density_source=source_kind,
        timings=timings,
        guard_band_count=guard_bands,
    )
