"""Fixed-density periodic band-structure execution."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft._compact import _CompactLaneState
from mlx_atomistic.dft._memory import _bounded_dft_allocator
from mlx_atomistic.dft._periodic_davidson import solve_periodic_eigenproblem
from mlx_atomistic.dft._periodic_hamiltonian import PeriodicKohnShamOperator
from mlx_atomistic.dft._periodic_models import (
    PeriodicBandPointResult,
    PeriodicBandStructureResult,
    PeriodicDavidsonConfig,
    PeriodicDFTSystem,
    PeriodicEigenResult,
    PeriodicFrozenDensity,
    PeriodicSCFResult,
)
from mlx_atomistic.dft._runtime_observer import RuntimeObserver
from mlx_atomistic.dft.gga import ProductionPBEExchangeCorrelation
from mlx_atomistic.dft.grids import ReciprocalGrid
from mlx_atomistic.dft.kpoints import BandPath, KPoint
from mlx_atomistic.dft.periodic_gth import (
    PeriodicGTHNonlocalOperator,
    _GTHProjectorCache,
    gth_local_potential_grid,
)
from mlx_atomistic.dft.plane_wave import PlaneWaveBasis
from mlx_atomistic.dft.potentials import hartree_potential
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional


@dataclass(frozen=True)
class _PeriodicBandSource:
    density: mx.array
    cutoff_hartree: float
    electron_count: float
    kind: str
    system_fingerprint: str | None


def _source_from_scf(
    system: PeriodicDFTSystem,
    source: PeriodicSCFResult,
) -> _PeriodicBandSource:
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
    return _PeriodicBandSource(
        density=source.density,
        cutoff_hartree=float(cutoffs[0]),
        electron_count=float(source.electron_count),
        kind="scf_result",
        system_fingerprint=source.system_fingerprint,
    )


def _normalized_band_source(
    system: PeriodicDFTSystem,
    source: PeriodicSCFResult | PeriodicFrozenDensity,
) -> _PeriodicBandSource:
    if isinstance(source, PeriodicSCFResult):
        normalized = _source_from_scf(system, source)
    elif isinstance(source, PeriodicFrozenDensity):
        normalized = _PeriodicBandSource(
            density=source.density,
            cutoff_hartree=source.cutoff_hartree,
            electron_count=source.electron_count,
            kind="frozen_density",
            system_fingerprint=source.system_fingerprint,
        )
    else:
        msg = "source must be PeriodicSCFResult or PeriodicFrozenDensity"
        raise TypeError(msg)
    return _validated_band_source(system, normalized)


def _validated_band_source(
    system: PeriodicDFTSystem,
    source: _PeriodicBandSource,
) -> _PeriodicBandSource:
    if source.system_fingerprint is None:
        if not system.is_homogeneous:
            msg = "multi-element frozen density requires an exact system fingerprint"
            raise ValueError(msg)
    elif source.system_fingerprint != system.fingerprint:
        msg = "frozen density pseudopotential assignment must match the periodic system"
        raise ValueError(msg)
    if source.density.shape != system.grid.shape:
        msg = "frozen density shape must match system.grid.shape"
        raise ValueError(msg)
    if not np.isclose(
        source.electron_count,
        system.electron_count,
        rtol=0.0,
        atol=1e-4,
    ):
        msg = "frozen density electron count must match the periodic system"
        raise ValueError(msg)
    density = mx.real(mx.array(source.density))
    finite = mx.all(mx.isfinite(density))
    minimum = mx.min(density)
    integrated = mx.sum(density) * system.grid.dv
    mx.eval(density, finite, minimum, integrated)
    if not bool(finite):
        msg = "frozen density must contain only finite values"
        raise ValueError(msg)
    if float(minimum) < -1e-6:
        msg = "frozen density must be nonnegative within numerical tolerance"
        raise ValueError(msg)
    count_tolerance = max(1e-4, 1e-5 * source.electron_count)
    if not np.isclose(
        float(integrated),
        source.electron_count,
        rtol=0.0,
        atol=count_tolerance,
    ):
        msg = "frozen density integral does not match its electron count"
        raise ValueError(msg)
    return _PeriodicBandSource(
        density=density,
        cutoff_hartree=source.cutoff_hartree,
        electron_count=source.electron_count,
        kind=source.kind,
        system_fingerprint=source.system_fingerprint,
    )


def _periodic_band_density_source(
    system: PeriodicDFTSystem,
    source: PeriodicSCFResult | PeriodicFrozenDensity,
) -> tuple[mx.array, float, float, str]:
    """Validate and normalize one frozen-density source."""

    normalized = _normalized_band_source(system, source)
    return (
        normalized.density,
        normalized.cutoff_hartree,
        normalized.electron_count,
        normalized.kind,
    )


@dataclass(frozen=True)
class _PeriodicBandRequest:
    system: PeriodicDFTSystem
    source: _PeriodicBandSource
    band_path: BandPath
    occupied_bands: int
    requested_bands: int
    guard_bands: int
    solved_bands: int
    solver_config: PeriodicDavidsonConfig
    xc: ExchangeCorrelationFunctional
    observer: RuntimeObserver | None


def _validated_band_request(
    system: PeriodicDFTSystem,
    source: PeriodicSCFResult | PeriodicFrozenDensity,
    band_path: BandPath,
    *,
    n_bands: int | None,
    guard_bands: int,
    config: PeriodicDavidsonConfig | None,
    xc_functional: ExchangeCorrelationFunctional | None,
    observer: RuntimeObserver | None,
) -> _PeriodicBandRequest:
    for point in band_path.points:
        if point.coordinate_system != "reduced":
            msg = "periodic band structure requires reduced-coordinate k-points"
            raise ValueError(msg)
    normalized = _normalized_band_source(system, source)
    physical_electron_count = float(system.electron_count)
    occupied_bands = int(round(physical_electron_count / 2.0))
    if occupied_bands <= 0 or abs(2.0 * occupied_bands - physical_electron_count) > 1e-8:
        msg = "periodic band structure requires two electrons per occupied band"
        raise ValueError(msg)
    requested_bands = occupied_bands + 8 if n_bands is None else n_bands
    if type(requested_bands) is not int or requested_bands < occupied_bands:
        msg = "n_bands must be an integer no smaller than the occupied band count"
        raise ValueError(msg)
    if type(guard_bands) is not int or guard_bands < 0:
        msg = "guard_bands must be a nonnegative integer"
        raise ValueError(msg)
    return _PeriodicBandRequest(
        system=system,
        source=normalized,
        band_path=band_path,
        occupied_bands=occupied_bands,
        requested_bands=requested_bands,
        guard_bands=guard_bands,
        solved_bands=requested_bands + guard_bands,
        solver_config=PeriodicDavidsonConfig() if config is None else config,
        xc=(ProductionPBEExchangeCorrelation() if xc_functional is None else xc_functional),
        observer=observer,
    )


@dataclass
class _PeriodicBandController:
    """Own fixed-potential setup, path-point solves, and result assembly."""

    request: _PeriodicBandRequest
    timings: dict[str, float]

    @classmethod
    def create(
        cls,
        system: PeriodicDFTSystem,
        source: PeriodicSCFResult | PeriodicFrozenDensity,
        band_path: BandPath,
        *,
        n_bands: int | None,
        guard_bands: int,
        config: PeriodicDavidsonConfig | None,
        xc_functional: ExchangeCorrelationFunctional | None,
        observer: RuntimeObserver | None,
    ) -> _PeriodicBandController:
        return cls(
            request=_validated_band_request(
                system,
                source,
                band_path,
                n_bands=n_bands,
                guard_bands=guard_bands,
                config=config,
                xc_functional=xc_functional,
                observer=observer,
            ),
            timings={"setup": 0.0, "eigensolver": 0.0, "total": 0.0},
        )

    def run(self) -> PeriodicBandStructureResult:
        total_start = perf_counter()
        with _bounded_dft_allocator(), _GTHProjectorCache() as projector_cache:
            reciprocal, effective_potential = self._build_effective_potential()
            points = tuple(
                self._solve_point(
                    point_index,
                    point,
                    reciprocal=reciprocal,
                    effective_potential=effective_potential,
                    projector_cache=projector_cache,
                )
                for point_index, point in enumerate(self.request.band_path.points)
            )
        return self._finalize(points, total_start=total_start)

    def _build_effective_potential(self) -> tuple[ReciprocalGrid, mx.array]:
        start = perf_counter()
        request = self.request
        reciprocal = ReciprocalGrid.from_real_space(request.system.grid)
        gamma_basis = PlaneWaveBasis(
            request.system.grid,
            request.source.cutoff_hartree,
            reciprocal_grid=reciprocal,
            lane_label="band:local-potential",
        )
        local_potential = gth_local_potential_grid(
            request.system.pseudopotentials,
            gamma_basis,
            request.system.positions,
        )
        hartree = hartree_potential(request.source.density, request.system.grid)
        xc_result = request.xc.evaluate(request.source.density, request.system.grid)
        effective = mx.array(local_potential + hartree + xc_result.potential)
        finite = (
            mx.all(mx.isfinite(xc_result.energy_density))
            & mx.all(mx.isfinite(xc_result.potential))
            & mx.isfinite(xc_result.total_energy)
            & mx.all(mx.isfinite(effective))
        )
        mx.eval(effective, finite)
        if not bool(finite):
            msg = "frozen-density effective potential is non-finite"
            raise ValueError(msg)
        self.timings["setup"] = (perf_counter() - start) * 1000.0
        return reciprocal, effective

    def _solve_point(
        self,
        point_index: int,
        point: KPoint,
        *,
        reciprocal: ReciprocalGrid,
        effective_potential: mx.array,
        projector_cache: _GTHProjectorCache,
    ) -> PeriodicBandPointResult:
        request = self.request
        basis = PlaneWaveBasis.from_reduced_kpoint(
            request.system.grid,
            request.source.cutoff_hartree,
            point.vector,
            reciprocal_grid=reciprocal,
            lane_label=f"band:kpoint:{point_index}",
        )
        if request.solved_bands > basis.active_count:
            msg = (
                f"n_bands plus guard_bands={request.solved_bands} exceeds active "
                f"basis size {basis.active_count} at path point {point_index}"
            )
            raise ValueError(msg)
        nonlocal_operator = PeriodicGTHNonlocalOperator(
            request.system.pseudopotentials,
            basis,
            request.system.positions,
            cache=projector_cache,
        )
        operator = PeriodicKohnShamOperator._from_shared_potential(
            basis,
            effective_potential,
            nonlocal_operator,
            request.observer,
        )
        self._emit_point_started(point_index, point)
        start = perf_counter()
        eigen = solve_periodic_eigenproblem(
            operator,
            n_bands=request.solved_bands,
            config=request.solver_config,
            observer=request.observer,
        )
        self.timings["eigensolver"] += (perf_counter() - start) * 1000.0
        eigen = self._trim_guard_bands(eigen, basis)
        self._validate_point_result(point_index, eigen)
        self._emit_point_completed(point_index, eigen)
        return PeriodicBandPointResult(
            requested_kpoint=point,
            basis=basis,
            eigen=eigen,
        )

    def _trim_guard_bands(
        self,
        eigen: PeriodicEigenResult,
        basis: PlaneWaveBasis,
    ) -> PeriodicEigenResult:
        request = self.request
        if not request.guard_bands:
            return eigen
        compact = eigen._compact_coefficients
        if not isinstance(compact, _CompactLaneState):
            msg = "guard-band trimming requires compact eigenvectors"
            raise RuntimeError(msg)
        residuals = eigen.residuals[: request.requested_bands]
        converged = bool(mx.all(residuals <= request.solver_config.tolerance))
        return PeriodicEigenResult._from_compact(
            eigenvalues=eigen.eigenvalues[: request.requested_bands],
            compact_coefficients=basis._state_from_compact(
                compact.values[: request.requested_bands]
            ),
            basis=basis,
            residuals=residuals,
            orthonormality_error=eigen.orthonormality_error,
            iterations=eigen.iterations,
            converged=converged,
            subspace_size=eigen.subspace_size,
            restart_count=eigen.restart_count,
        )

    @staticmethod
    def _validate_point_result(
        point_index: int,
        eigen: PeriodicEigenResult,
    ) -> None:
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

    def _emit_point_started(self, point_index: int, point: KPoint) -> None:
        observer = self.request.observer
        if observer is not None:
            observer.emit(
                "band_kpoint",
                status="started",
                point_index=point_index,
                point_count=len(self.request.band_path.points),
                reduced_kpoint=list(point.vector),
            )

    def _emit_point_completed(
        self,
        point_index: int,
        eigen: PeriodicEigenResult,
    ) -> None:
        observer = self.request.observer
        if observer is not None:
            observer.emit(
                "band_kpoint",
                status="completed",
                point_index=point_index,
                iterations=eigen.iterations,
                worst_residual=float(mx.max(eigen.residuals)),
            )

    def _finalize(
        self,
        points: tuple[PeriodicBandPointResult, ...],
        *,
        total_start: float,
    ) -> PeriodicBandStructureResult:
        eigenvalues = mx.stack([point.eigen.eigenvalues for point in points])
        residuals = mx.stack([point.eigen.residuals for point in points])
        mx.eval(eigenvalues, residuals)
        self.timings["total"] = (perf_counter() - total_start) * 1000.0
        request = self.request
        return PeriodicBandStructureResult(
            kpoints=request.band_path.points,
            eigenvalues=eigenvalues,
            residuals=residuals,
            points=points,
            occupied_band_count=request.occupied_bands,
            cutoff_hartree=request.source.cutoff_hartree,
            density_source=request.source.kind,
            timings=self.timings,
            guard_band_count=request.guard_bands,
        )


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
    """Run fixed-density periodic bands through the structured controller."""

    return _PeriodicBandController.create(
        system,
        source,
        band_path,
        n_bands=n_bands,
        guard_bands=guard_bands,
        config=config,
        xc_functional=xc_functional,
        observer=observer,
    ).run()
