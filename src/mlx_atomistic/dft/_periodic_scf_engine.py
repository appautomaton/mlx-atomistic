"""Self-consistent periodic DFT execution engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from time import perf_counter

import mlx.core as mx
import numpy as np

import mlx_atomistic.dft._periodic_resume as _periodic_resume
from mlx_atomistic.dft._compact import (
    _CompactBatch,
    _CompactBatchPolicy,
    _CompactLaneState,
    _remap_initial_coefficients,
    _require_layout,
)
from mlx_atomistic.dft._memory import _bounded_dft_allocator
from mlx_atomistic.dft._periodic_davidson import (
    _DavidsonApplicationTicket,
    _DavidsonEngine,
    _DavidsonLaneRequest,
    _DavidsonScheduler,
    _initial_trial,
)
from mlx_atomistic.dft._periodic_density import _density_from_kpoints
from mlx_atomistic.dft._periodic_execution import _detached_failure
from mlx_atomistic.dft._periodic_hamiltonian import PeriodicKohnShamOperator
from mlx_atomistic.dft._periodic_models import (
    PeriodicDFTSystem,
    PeriodicEigenResult,
    PeriodicKPointResult,
    PeriodicSCFConfig,
    PeriodicSCFResult,
    _is_finite_positive_control,
    _time_reversed_compact_values,
    _TimeReversalContinuationSeed,
)
from mlx_atomistic.dft._periodic_state import _PeriodicSCFContinuationState
from mlx_atomistic.dft._runtime_observer import (
    RuntimeObserver,
    add_observed_work,
    observed_phase,
)
from mlx_atomistic.dft.gga import ProductionPBEExchangeCorrelation
from mlx_atomistic.dft.grids import RealSpaceGrid, ReciprocalGrid
from mlx_atomistic.dft.kpoints import (
    KPointMesh,
    TimeReversalOwnership,
    TimeReversalOwnershipEntry,
    _independent_pair,
    admit_time_reversal_bases,
    build_time_reversal_ownership,
)
from mlx_atomistic.dft.mixing import LinearMixer, PulayDIISMixer
from mlx_atomistic.dft.periodic_gth import (
    PeriodicGTHNonlocalOperator,
    _GTHProjectorCache,
    gth_local_potential_grid,
    periodic_ewald_energy,
)
from mlx_atomistic.dft.plane_wave import PlaneWaveBasis
from mlx_atomistic.dft.potentials import hartree_potential
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional, XCResult


def _next_scf_eigensolver_tolerance(
    config: PeriodicSCFConfig,
    current_tolerance: float,
    density_residual: float,
    electron_count: float,
) -> float:
    if not config.adaptive_eigensolver_tolerance:
        return float(config.davidson.tolerance)
    return max(
        float(config.davidson.tolerance),
        min(
            float(current_tolerance),
            float(config.eigensolver_tolerance_scale)
            * float(density_residual)
            / max(1.0, float(electron_count)),
        ),
    )


def _scf_eigensolver_tolerance(
    config: PeriodicSCFConfig,
    history: Sequence[Mapping[str, object]],
    electron_count: float,
) -> float:
    if not config.adaptive_eigensolver_tolerance:
        return float(config.davidson.tolerance)
    tolerance = float(config.initial_eigensolver_tolerance)
    for row in history:
        recorded = row.get("eigensolver_tolerance")
        residual = row.get("density_residual")
        if not _is_finite_positive_control(recorded) or not (
            not isinstance(residual, (bool, np.bool_))
            and isinstance(residual, (int, float, np.integer, np.floating))
            and np.isfinite(float(residual))
            and float(residual) >= 0.0
        ):
            msg = "adaptive periodic resume history has a malformed tolerance schedule"
            raise ValueError(msg)
        if not np.isclose(float(recorded), tolerance, rtol=1e-12, atol=0.0):
            msg = "adaptive periodic resume history has an inconsistent tolerance schedule"
            raise ValueError(msg)
        tolerance = _next_scf_eigensolver_tolerance(
            config,
            tolerance,
            float(residual),
            electron_count,
        )
    return tolerance


def _density_residual(current: mx.array, target: mx.array, grid: RealSpaceGrid) -> float:
    delta = target - current
    return float(mx.sqrt(mx.sum(delta * delta) * grid.dv))


def _pack_initial_states(
    bases: Sequence[PlaneWaveBasis],
    initial_coefficients: Sequence[mx.array],
) -> list[_CompactLaneState | _TimeReversalContinuationSeed]:
    states = []
    for basis, coefficients in zip(bases, initial_coefficients, strict=True):
        if isinstance(coefficients, _TimeReversalContinuationSeed):
            state = coefficients
        elif isinstance(coefficients, _CompactLaneState):
            try:
                _require_layout(coefficients, basis._layout)
                state = coefficients
            except ValueError:
                state = _remap_initial_coefficients(coefficients, basis._layout)
        else:
            state, _ = basis._state_from_full(coefficients)
        states.append(state)
    return states


def _time_reversal_subspaces_match(
    owner_state: _CompactLaneState,
    partner_state: _CompactLaneState,
    partner_basis: PlaneWaveBasis,
    permutation: np.ndarray,
    *,
    n_bands: int,
    atol: float = 3e-4,
) -> bool:
    if owner_state.vector_count < n_bands or partner_state.vector_count < n_bands:
        return False
    expected = _time_reversed_compact_values(
        owner_state.values[:n_bands],
        permutation,
    )
    partner_occupied = partner_state.values[:n_bands]
    try:
        expected_orthonormal = partner_basis._orthonormalize_compact(expected)
        partner_orthonormal = partner_basis._orthonormalize_compact(partner_occupied)
    except ValueError:
        return False
    overlap = expected_orthonormal @ mx.conjugate(mx.transpose(partner_orthonormal))
    singular_values = np.linalg.svd(
        np.asarray(overlap, dtype=np.complex128),
        compute_uv=False,
    )
    return bool(
        singular_values.shape == (n_bands,)
        and np.isfinite(singular_values).all()
        and np.all(np.abs(singular_values - 1.0) <= atol)
    )


def _admit_initial_time_reversal(
    ownership: TimeReversalOwnership,
    bases: Sequence[PlaneWaveBasis],
    initial_coefficients: Sequence[mx.array] | None,
    *,
    n_bands: int,
) -> tuple[TimeReversalOwnership, dict[int, _CompactLaneState | None]]:
    if initial_coefficients is None:
        return ownership, dict.fromkeys(ownership.owned_indices)
    states = _pack_initial_states(bases, initial_coefficients)
    admitted = ownership
    visited: set[int] = set()
    for entry in ownership.entries:
        if entry.explicit_index in visited or entry.role != "owner":
            continue
        partner_index = entry.partner_index
        if partner_index is None or partner_index == entry.explicit_index:
            visited.add(entry.explicit_index)
            continue
        owner_state = states[entry.explicit_index]
        partner_state = states[partner_index]
        descriptor_match = (
            isinstance(partner_state, _TimeReversalContinuationSeed)
            and partner_state.owner_index == entry.explicit_index
            and isinstance(owner_state, _CompactLaneState)
            and owner_state.vector_count >= n_bands
        )
        permutation = entry._time_reversal_permutation
        subspace_match = descriptor_match or (
            permutation is not None
            and isinstance(owner_state, _CompactLaneState)
            and isinstance(partner_state, _CompactLaneState)
            and _time_reversal_subspaces_match(
                owner_state,
                partner_state,
                bases[partner_index],
                permutation,
                n_bands=n_bands,
            )
        )
        if not subspace_match:
            admitted = _independent_pair(
                admitted,
                entry.explicit_index,
                "initial_coefficients_time_reversal_mismatch",
            )
        visited.update({entry.explicit_index, partner_index})
    return admitted, {
        index: (None if isinstance(states[index], _TimeReversalContinuationSeed) else states[index])
        for index in admitted.owned_indices
    }


def _owned_kpoint_result(
    *,
    entry: TimeReversalOwnershipEntry,
    basis: PlaneWaveBasis,
    eigen: PeriodicEigenResult,
) -> PeriodicKPointResult:
    return PeriodicKPointResult(
        reduced_kpoint=entry.reduced_kpoint,
        weight=entry.original_weight,
        basis=basis,
        eigen=eigen,
        explicit_index=entry.explicit_index,
        aggregated_weight=entry.aggregated_weight,
        ownership_role=entry.role,
        fallback_reason=entry.fallback_reason,
    )


def _publish_explicit_kpoints(
    ownership: TimeReversalOwnership,
    bases: Sequence[PlaneWaveBasis],
    owned_results: dict[int, PeriodicKPointResult],
    observer: RuntimeObserver | None,
) -> tuple[PeriodicKPointResult, ...]:
    explicit: list[PeriodicKPointResult] = []
    for entry, basis in zip(ownership.entries, bases, strict=True):
        if entry.owner_index == entry.explicit_index:
            explicit.append(owned_results[entry.explicit_index])
            continue
        owner_result = owned_results[entry.owner_index]
        owner_entry = ownership.entry_for(entry.owner_index)
        permutation = owner_entry._time_reversal_permutation
        if permutation is None:
            msg = "admitted time-reversal partner has no active-basis permutation"
            raise RuntimeError(msg)
        eigen = PeriodicEigenResult._from_time_reversal_owner(
            owner=owner_result.eigen,
            partner_basis=basis,
            permutation=permutation,
            observer=observer,
        )
        explicit.append(
            PeriodicKPointResult(
                reduced_kpoint=entry.reduced_kpoint,
                weight=entry.original_weight,
                basis=basis,
                eigen=eigen,
                explicit_index=entry.explicit_index,
                aggregated_weight=entry.aggregated_weight,
                ownership_role=entry.role,
                fallback_reason=entry.fallback_reason,
            )
        )
    return tuple(explicit)


@dataclass(frozen=True)
class _PeriodicSCFSetup:
    """Immutable system and execution resources shared by every iteration."""

    system: PeriodicDFTSystem
    kpoint_mesh: KPointMesh
    config: PeriodicSCFConfig
    compact_policy: _CompactBatchPolicy
    xc: ExchangeCorrelationFunctional
    occupied_bands: int
    observer: RuntimeObserver | None
    ownership: TimeReversalOwnership
    bases: tuple[PlaneWaveBasis, ...]
    owned_indices: tuple[int, ...]
    nonlocal_operators: dict[int, PeriodicGTHNonlocalOperator]
    local_potential: mx.array
    mixer: LinearMixer | PulayDIISMixer
    ewald_energy: float
    resumed: bool


@dataclass
class _PeriodicSCFProgress:
    """Mutable accepted-iteration state owned by one SCF controller."""

    density: mx.array
    previous_states: dict[int, _CompactLaneState | None]
    previous_energy: float | None
    history: list[dict[str, float | int | str | None]]
    energy_terms: dict[str, float]
    iteration_start: int
    lineage: tuple[str, ...]
    eigensolver_tolerance: float
    final_owned_results: tuple[PeriodicKPointResult, ...] = ()
    converged: bool = False
    stopped_for_checkpoint: bool = False
    final_checkpoint_state: _PeriodicSCFContinuationState | None = None
    density_residual: float = float("inf")
    energy_delta: float | None = None
    timings: dict[str, float] = field(
        default_factory=lambda: {
            "hartree": 0.0,
            "xc": 0.0,
            "eigensolver": 0.0,
            "total": 0.0,
        }
    )


@dataclass(frozen=True)
class _PeriodicSCFIterationResult:
    """One evaluated SCF iteration before convergence or mixing is accepted."""

    owned_results: tuple[PeriodicKPointResult, ...]
    target_density: mx.array
    target_count: float
    density_residual: float
    max_orbital_residual: float
    energy_delta: float | None
    energy_terms: dict[str, float]
    all_eigen_converged: bool
    eigen_residuals_directly_validated: bool


class _PeriodicSCFController:
    """Own setup, iteration transitions, checkpoints, and finalization."""

    def __init__(
        self,
        setup: _PeriodicSCFSetup,
        progress: _PeriodicSCFProgress,
        *,
        checkpoint_callback: Callable[[_PeriodicSCFContinuationState], bool] | None,
        checkpoint_iteration: int | None,
    ) -> None:
        self.setup = setup
        self.progress = progress
        self.checkpoint_callback = checkpoint_callback
        self.checkpoint_iteration = checkpoint_iteration

    @classmethod
    def create(
        cls,
        system: PeriodicDFTSystem,
        *,
        cutoff_hartree: float,
        kpoint_mesh: KPointMesh,
        n_bands: int | None,
        config: PeriodicSCFConfig | None,
        xc_functional: ExchangeCorrelationFunctional | None,
        initial_density: mx.array | None,
        initial_coefficients: Sequence[mx.array] | None,
        observer: RuntimeObserver | None,
        projector_cache: _GTHProjectorCache,
        resume_state: _PeriodicSCFContinuationState | None,
        checkpoint_callback: Callable[[_PeriodicSCFContinuationState], bool] | None,
        checkpoint_iteration: int | None,
    ) -> _PeriodicSCFController:
        scf_config = PeriodicSCFConfig() if config is None else config
        compact_policy = scf_config._compact_batch_policy()
        xc = ProductionPBEExchangeCorrelation() if xc_functional is None else xc_functional
        occupied_bands = int(round(system.electron_count / 2.0)) if n_bands is None else n_bands
        cls._validate_request(
            system,
            kpoint_mesh=kpoint_mesh,
            occupied_bands=occupied_bands,
            initial_density=initial_density,
            initial_coefficients=initial_coefficients,
            resume_state=resume_state,
            config=scf_config,
        )
        ownership = build_time_reversal_ownership(kpoint_mesh)
        if observer is not None:
            observer.emit(
                "setup",
                status="started",
                kpoint_count=len(kpoint_mesh.points),
                grid_shape=list(system.grid.shape),
            )
        with observed_phase(observer, "setup"):
            shared_reciprocal = ReciprocalGrid.from_real_space(system.grid)
            bases = tuple(
                PlaneWaveBasis.from_reduced_kpoint(
                    system.grid,
                    cutoff_hartree,
                    point.vector,
                    reciprocal_grid=shared_reciprocal,
                    lane_label=f"kpoint:{point_index}",
                )
                for point_index, point in enumerate(kpoint_mesh.points)
            )
            ownership = admit_time_reversal_bases(ownership, bases)
            if resume_state is None:
                ownership, previous_states = _admit_initial_time_reversal(
                    ownership,
                    bases,
                    initial_coefficients,
                    n_bands=occupied_bands,
                )
            else:
                ownership = _periodic_resume._resume_ownership(
                    ownership,
                    resume_state.ownership,
                )
            owned_indices = ownership.owned_indices
            gamma_basis = PlaneWaveBasis(
                system.grid,
                cutoff_hartree,
                reciprocal_grid=shared_reciprocal,
                lane_label="gamma-local-potential",
            )
            nonlocal_operators = {
                point_index: PeriodicGTHNonlocalOperator(
                    system.pseudopotentials,
                    bases[point_index],
                    system.positions,
                    cache=projector_cache,
                )
                for point_index in owned_indices
            }
            local_potential = gth_local_potential_grid(
                system.pseudopotentials,
                gamma_basis,
                system.positions,
            )
            mixer = (
                PulayDIISMixer(beta=scf_config.mixing_beta)
                if scf_config.mixer == "diis"
                else LinearMixer(beta=scf_config.mixing_beta)
            )
            if resume_state is None:
                density = cls._initial_density(system, initial_density)
                previous_energy: float | None = None
                history: list[dict[str, float | int | str | None]] = []
                energy_terms: dict[str, float] = {}
                iteration_start = 1
                lineage: tuple[str, ...] = ()
            else:
                (
                    density,
                    previous_states,
                    restored_energy,
                    history,
                    energy_terms,
                ) = _periodic_resume._restore_continuation_state(
                    resume_state,
                    bases=bases,
                    ownership=ownership,
                    occupied_bands=occupied_bands,
                    grid=system.grid,
                    electron_count=system.electron_count,
                    mixer=mixer,
                )
                previous_energy = restored_energy
                iteration_start = resume_state.completed_iteration + 1
                lineage = resume_state.lineage
            ewald = periodic_ewald_energy(
                system.charges,
                system.positions,
                np.asarray(system.grid.lengths),
            )
            eigensolver_tolerance = _scf_eigensolver_tolerance(
                scf_config,
                history,
                system.electron_count,
            )
        setup = _PeriodicSCFSetup(
            system=system,
            kpoint_mesh=kpoint_mesh,
            config=scf_config,
            compact_policy=compact_policy,
            xc=xc,
            occupied_bands=occupied_bands,
            observer=observer,
            ownership=ownership,
            bases=bases,
            owned_indices=owned_indices,
            nonlocal_operators=nonlocal_operators,
            local_potential=local_potential,
            mixer=mixer,
            ewald_energy=ewald,
            resumed=resume_state is not None,
        )
        progress = _PeriodicSCFProgress(
            density=density,
            previous_states=previous_states,
            previous_energy=previous_energy,
            history=history,
            energy_terms=energy_terms,
            iteration_start=iteration_start,
            lineage=lineage,
            eigensolver_tolerance=eigensolver_tolerance,
        )
        cls._emit_setup_completed(setup, progress)
        return cls(
            setup,
            progress,
            checkpoint_callback=checkpoint_callback,
            checkpoint_iteration=checkpoint_iteration,
        )

    @staticmethod
    def _validate_request(
        system: PeriodicDFTSystem,
        *,
        kpoint_mesh: KPointMesh,
        occupied_bands: int,
        initial_density: mx.array | None,
        initial_coefficients: Sequence[mx.array] | None,
        resume_state: _PeriodicSCFContinuationState | None,
        config: PeriodicSCFConfig,
    ) -> None:
        if occupied_bands <= 0 or abs(2.0 * occupied_bands - system.electron_count) > 1e-8:
            msg = "the bounded spin-unpolarized path requires two electrons per occupied band"
            raise ValueError(msg)
        if any(point.coordinate_system != "reduced" for point in kpoint_mesh.points):
            msg = "periodic SCF requires reduced-coordinate k-points"
            raise ValueError(msg)
        if initial_coefficients is not None and len(initial_coefficients) != len(
            kpoint_mesh.points
        ):
            msg = "initial_coefficients length must match the k-point mesh"
            raise ValueError(msg)
        if resume_state is not None and (
            initial_density is not None or initial_coefficients is not None
        ):
            msg = "periodic resume state is mutually exclusive with public initial guesses"
            raise ValueError(msg)
        if resume_state is not None and resume_state.completed_iteration >= config.max_iterations:
            msg = "periodic resume state has no remaining SCF iteration"
            raise ValueError(msg)

    @staticmethod
    def _initial_density(
        system: PeriodicDFTSystem,
        initial_density: mx.array | None,
    ) -> mx.array:
        if initial_density is None:
            return mx.full(
                system.grid.shape,
                system.electron_count / system.grid.volume,
            )
        density = mx.real(mx.array(initial_density))
        if density.shape != system.grid.shape:
            msg = "initial_density must have shape system.grid.shape"
            raise ValueError(msg)
        count = float(mx.sum(density) * system.grid.dv)
        if count <= 0.0:
            msg = "initial_density must integrate to a positive count"
            raise ValueError(msg)
        return density * (system.electron_count / count)

    @staticmethod
    def _emit_setup_completed(
        setup: _PeriodicSCFSetup,
        progress: _PeriodicSCFProgress,
    ) -> None:
        observer = setup.observer
        if observer is None:
            return
        observer.record_memory("shared_full_grid_bytes", setup.system.grid.size * 4 * 4)
        observer.record_memory("persistent_projector_bytes", 0)
        observer.emit(
            "setup",
            status="completed",
            active_counts=[basis.active_count for basis in setup.bases],
            owned_indices=list(setup.owned_indices),
            owned_active_counts=[setup.bases[index].active_count for index in setup.owned_indices],
            representative_count=len(setup.ownership.representative_indices),
            fallback_reasons=setup.ownership.fallback_reasons,
            batch_policy=setup.config.batch_policy(),
            resumed=setup.resumed,
            iteration_start=progress.iteration_start,
        )

    def run(self) -> PeriodicSCFResult:
        total_start = perf_counter()
        iteration = self.progress.iteration_start - 1
        for iteration in range(
            self.progress.iteration_start,
            self.setup.config.max_iterations + 1,
        ):
            self._emit_iteration_started(iteration)
            outcome = self._evaluate_iteration(iteration)
            self._record_iteration(iteration, outcome)
            if self._accept_converged(iteration, outcome):
                break
            self._advance_unconverged(outcome)
            if self._checkpoint(iteration):
                self.progress.stopped_for_checkpoint = True
                break
        self.progress.timings["total"] = (perf_counter() - total_start) * 1000.0
        return self._finalize(iteration)

    def _emit_iteration_started(self, iteration: int) -> None:
        observer = self.setup.observer
        if observer is not None:
            observer.emit(
                "scf_iteration",
                status="started",
                iteration=iteration,
                total_iterations=self.setup.config.max_iterations,
                eigensolver_tolerance=self.progress.eigensolver_tolerance,
            )

    def _evaluate_iteration(self, iteration: int) -> _PeriodicSCFIterationResult:
        hartree, xc_result, effective_snapshot = self._effective_potential()
        (
            owned_results,
            max_orbital_residual,
            orbital_densities,
            eigen_residuals_directly_validated,
        ) = self._solve_owned_kpoints(
            iteration,
            effective_snapshot,
        )
        with observed_phase(self.setup.observer, "density"):
            target_density = _density_from_kpoints(
                owned_results,
                occupation=2.0,
                policy=self.setup.compact_policy,
                observer=self.setup.observer,
                orbital_densities=orbital_densities,
            )
            target_count = float(mx.sum(target_density) * self.setup.system.grid.dv)
            target_density = target_density * (self.setup.system.electron_count / target_count)
            density_residual = _density_residual(
                self.progress.density,
                target_density,
                self.setup.system.grid,
            )
        band_energy = sum(
            result.integration_weight * 2.0 * float(mx.sum(result.eigen.eigenvalues))
            for result in owned_results
        )
        hartree_energy = 0.5 * float(
            mx.sum(self.progress.density * hartree) * self.setup.system.grid.dv
        )
        xc_energy = float(xc_result.total_energy)
        density_xc = float(
            mx.sum(self.progress.density * xc_result.potential) * self.setup.system.grid.dv
        )
        total_energy = (
            band_energy - hartree_energy + xc_energy - density_xc + self.setup.ewald_energy
        )
        energy_delta = (
            None
            if self.progress.previous_energy is None
            else total_energy - self.progress.previous_energy
        )
        energy_terms = {
            "band": band_energy,
            "hartree": hartree_energy,
            "xc": xc_energy,
            "density_xc_potential": density_xc,
            "ion_ewald": self.setup.ewald_energy,
            "total": total_energy,
        }
        return _PeriodicSCFIterationResult(
            owned_results=owned_results,
            target_density=target_density,
            target_count=target_count,
            density_residual=density_residual,
            max_orbital_residual=max_orbital_residual,
            energy_delta=energy_delta,
            energy_terms=energy_terms,
            all_eigen_converged=all(result.eigen.converged for result in owned_results),
            eigen_residuals_directly_validated=eigen_residuals_directly_validated,
        )

    def _effective_potential(self) -> tuple[mx.array, XCResult, mx.array]:
        start = perf_counter()
        hartree = hartree_potential(self.progress.density, self.setup.system.grid)
        self.progress.timings["hartree"] += (perf_counter() - start) * 1000.0
        start = perf_counter()
        xc_result = self.setup.xc.evaluate(
            self.progress.density,
            self.setup.system.grid,
        )
        self.progress.timings["xc"] += (perf_counter() - start) * 1000.0
        effective_snapshot = mx.array(self.setup.local_potential + hartree + xc_result.potential)
        xc_finite = (
            mx.all(mx.isfinite(xc_result.energy_density))
            & mx.all(mx.isfinite(xc_result.potential))
            & mx.isfinite(xc_result.total_energy)
        )
        effective_finite = mx.all(mx.isfinite(effective_snapshot))
        mx.eval(effective_snapshot, xc_finite, effective_finite)
        if not bool(xc_finite):
            msg = "SCF exchange-correlation result is non-finite"
            raise ValueError(msg)
        if not bool(effective_finite):
            msg = "SCF effective potential is non-finite"
            raise ValueError(msg)
        return hartree, xc_result, effective_snapshot

    def _solve_owned_kpoints(
        self,
        iteration: int,
        effective_snapshot: mx.array,
    ) -> tuple[
        tuple[PeriodicKPointResult, ...],
        float,
        tuple[mx.array, ...] | None,
        bool,
    ]:
        start = perf_counter()
        operators_by_index = {
            point_index: PeriodicKohnShamOperator._from_shared_potential(
                self.setup.bases[point_index],
                effective_snapshot,
                self.setup.nonlocal_operators[point_index],
                self.setup.observer,
            )
            for point_index in self.setup.owned_indices
        }
        lane_to_index = {
            self.setup.bases[point_index]._layout.lane_id: point_index
            for point_index in self.setup.owned_indices
        }
        iteration_davidson = replace(
            self.setup.config.davidson,
            tolerance=self.progress.eigensolver_tolerance,
        )
        require_direct_validation = (
            not self.setup.config.adaptive_eigensolver_tolerance
            or self.progress.eigensolver_tolerance
            <= float(self.setup.config.davidson.tolerance)
        )
        requests = tuple(
            _DavidsonLaneRequest(
                lane_id=self.setup.bases[point_index]._layout.lane_id,
                operator=operators_by_index[point_index],
                n_bands=self.setup.occupied_bands,
                config=iteration_davidson,
                trial=_initial_trial(
                    self.setup.bases[point_index],
                    self.setup.occupied_bands,
                    self.progress.previous_states.get(point_index),
                ),
                observer=self.setup.observer,
                trial_is_orthonormal=(
                    self.progress.previous_states.get(point_index) is None
                    or self.setup.resumed
                    or iteration > self.progress.iteration_start
                ),
                require_direct_validation=require_direct_validation,
                capture_orbital_density=require_direct_validation,
            )
            for point_index in self.setup.owned_indices
        )
        scheduler = _DavidsonScheduler(
            policy=self.setup.compact_policy,
            submission_callback=self._submission_callback(
                iteration,
                lane_to_index,
            ),
        )
        with observed_phase(
            self.setup.observer,
            "eigensolver_control",
            synchronize=False,
        ):
            eigen_outcome = _DavidsonEngine(scheduler=scheduler).solve(requests)
        if eigen_outcome.failures:
            self._raise_eigensolver_failure(
                iteration,
                requests,
                lane_to_index,
                eigen_outcome.failures,
            )
        owned_by_index: dict[int, PeriodicKPointResult] = {}
        orbital_density_by_index: dict[int, mx.array] | None = (
            {} if require_direct_validation else None
        )
        max_orbital_residual = 0.0
        for point_index in self.setup.owned_indices:
            basis = self.setup.bases[point_index]
            entry = self.setup.ownership.entry_for(point_index)
            eigen = eigen_outcome.result_for(basis._layout.lane_id)
            if orbital_density_by_index is not None:
                orbital_density_by_index[point_index] = (
                    eigen_outcome.orbital_density_for(basis._layout.lane_id)
                )
            add_observed_work(self.setup.observer, {"kpoint_lane_solves": 1})
            if entry.role == "owner":
                add_observed_work(
                    self.setup.observer,
                    {"representative_lane_solves": 1},
                )
            max_orbital_residual = max(
                max_orbital_residual,
                float(mx.max(eigen.residuals)),
            )
            owned_by_index[point_index] = _owned_kpoint_result(
                entry=entry,
                basis=basis,
                eigen=eigen,
            )
        self.progress.timings["eigensolver"] += (perf_counter() - start) * 1000.0
        return (
            tuple(owned_by_index[index] for index in self.setup.owned_indices),
            max_orbital_residual,
            (
                None
                if orbital_density_by_index is None
                else tuple(
                    orbital_density_by_index[index]
                    for index in self.setup.owned_indices
                )
            ),
            require_direct_validation,
        )

    def _submission_callback(
        self,
        iteration: int,
        lane_to_index: dict[str, int],
    ) -> Callable[
        [
            str,
            int,
            tuple[_DavidsonApplicationTicket, ...],
            _CompactBatch,
            dict[str, Exception],
        ],
        None,
    ]:
        def emit_submission(
            status: str,
            batch_index: int,
            tickets: tuple[_DavidsonApplicationTicket, ...],
            batch: _CompactBatch,
            failures: dict[str, Exception],
        ) -> None:
            observer = self.setup.observer
            if observer is None or not observer.detail_events:
                return
            explicit_indices = [lane_to_index[ticket.lane_id] for ticket in tickets]
            complete_transient_bytes = PeriodicKohnShamOperator._estimated_batch_transient_bytes(
                [ticket.operator for ticket in tickets],
                batch,
                captured_density_lanes=sum(ticket.capture_orbital_density for ticket in tickets),
            )
            fields: dict[str, object] = {
                "status": status,
                "scf_iteration": iteration,
                "batch_index": batch_index,
                "batch_size": len(tickets),
                "lane_capacity": batch.lane_capacity,
                "lane_ids": [ticket.lane_id for ticket in tickets],
                "purposes": [ticket.purpose for ticket in tickets],
                "reduced_kpoints": [
                    list(self.setup.kpoint_mesh.points[index].vector) for index in explicit_indices
                ],
                "explicit_indices": explicit_indices,
                "active_counts": list(batch.active_counts),
                "active_capacity": batch.bucket_size,
                "vector_count": batch.vector_count,
                "logical_vector_counts": list(batch.vector_counts),
                "padding_elements": batch.padding_elements,
                "lane_padding_elements": batch.lane_padding_elements,
                "vector_padding_elements": batch.vector_padding_elements,
                "estimated_transient_bytes": complete_transient_bytes,
                "compact_batch_transient_bytes": batch.estimated_transient_bytes,
                "batch_policy": self.setup.config.batch_policy(),
                "synchronized": observer.synchronize is not None,
            }
            if failures:
                fields["failed_explicit_indices"] = [lane_to_index[lane_id] for lane_id in failures]
                fields["failure_messages"] = {
                    lane_id: str(error) for lane_id, error in failures.items()
                }
            observer.emit("kpoint_batch", **fields)

        return emit_submission

    def _raise_eigensolver_failure(
        self,
        iteration: int,
        requests: Sequence[_DavidsonLaneRequest],
        lane_to_index: dict[str, int],
        failures: dict[str, Exception],
    ) -> None:
        observer = self.setup.observer
        if observer is not None:
            observer.emit(
                "failure",
                stage="eigensolver",
                scf_iteration=iteration,
                failed_explicit_indices=[lane_to_index[lane_id] for lane_id in failures],
                failure_messages={lane_id: str(error) for lane_id, error in failures.items()},
            )
        first_failed_lane = next(
            request.lane_id for request in requests if request.lane_id in failures
        )
        raise _detached_failure(failures[first_failed_lane]) from None

    def _record_iteration(
        self,
        iteration: int,
        outcome: _PeriodicSCFIterationResult,
    ) -> None:
        self.progress.final_owned_results = outcome.owned_results
        self.progress.density_residual = outcome.density_residual
        self.progress.energy_delta = outcome.energy_delta
        self.progress.energy_terms = outcome.energy_terms
        self.progress.history.append(
            {
                "iteration": iteration,
                "total_energy_hartree": outcome.energy_terms["total"],
                "energy_delta_hartree": outcome.energy_delta,
                "density_residual": outcome.density_residual,
                "electron_count": outcome.target_count,
                "max_orbital_residual": outcome.max_orbital_residual,
                "eigensolver_tolerance": self.progress.eigensolver_tolerance,
                "eigensolver_method": "davidson",
                "all_kpoints_converged": str(outcome.all_eigen_converged).lower(),
                "orbital_residual_source": (
                    "direct_operator"
                    if outcome.eigen_residuals_directly_validated
                    else "paired_subspace"
                ),
            }
        )
        observer = self.setup.observer
        if observer is not None:
            observer.emit(
                "scf_iteration",
                status="completed",
                iteration=iteration,
                total_energy_hartree=outcome.energy_terms["total"],
                energy_delta_hartree=outcome.energy_delta,
                density_residual=outcome.density_residual,
                max_orbital_residual=outcome.max_orbital_residual,
                eigensolver_tolerance=self.progress.eigensolver_tolerance,
                eigensolver_method="davidson",
                all_kpoints_converged=outcome.all_eigen_converged,
                orbital_residual_source=(
                    "direct_operator"
                    if outcome.eigen_residuals_directly_validated
                    else "paired_subspace"
                ),
            )

    def _accept_converged(
        self,
        iteration: int,
        outcome: _PeriodicSCFIterationResult,
    ) -> bool:
        config = self.setup.config
        converged = (
            iteration >= config.min_iterations
            and outcome.eigen_residuals_directly_validated
            and outcome.all_eigen_converged
            and outcome.density_residual <= config.density_tolerance
            and outcome.energy_delta is not None
            and abs(outcome.energy_delta) <= config.energy_tolerance
            and outcome.max_orbital_residual <= config.orbital_tolerance
        )
        if converged:
            self.progress.converged = True
            self.progress.density = outcome.target_density
        return converged

    def _advance_unconverged(
        self,
        outcome: _PeriodicSCFIterationResult,
    ) -> None:
        self.progress.eigensolver_tolerance = _next_scf_eigensolver_tolerance(
            self.setup.config,
            self.progress.eigensolver_tolerance,
            outcome.density_residual,
            self.setup.system.electron_count,
        )
        self._mix_density(outcome.target_density)
        self.progress.previous_energy = outcome.energy_terms["total"]
        self.progress.previous_states = {
            result.explicit_index: result.eigen._compact_coefficients
            for result in outcome.owned_results
            if result.explicit_index is not None
        }

    def _mix_density(self, target_density: mx.array) -> None:
        with observed_phase(self.setup.observer, "mixing"):
            mixed = self.setup.mixer.mix(
                self.progress.density,
                target_density,
            )
            mixed_finite = mx.all(mx.isfinite(mixed))
            mixed_minimum_array = mx.min(mixed)
            mixed_count_array = mx.sum(mixed) * self.setup.system.grid.dv
            mx.eval(
                mixed,
                mixed_finite,
                mixed_minimum_array,
                mixed_count_array,
            )
            mixed_minimum = float(mixed_minimum_array)
            mixed_count = float(mixed_count_array)
            if (
                not bool(mixed_finite)
                or not np.isfinite(mixed_minimum)
                or mixed_minimum < 0.0
                or not np.isfinite(mixed_count)
                or mixed_count <= 0.0
            ):
                msg = "SCF mixer produced a non-finite, negative, or empty density"
                raise ValueError(msg)
            normalized_density = mixed * (self.setup.system.electron_count / mixed_count)
            normalized_finite = mx.all(mx.isfinite(normalized_density))
            normalized_count_array = mx.sum(normalized_density) * self.setup.system.grid.dv
            mx.eval(
                normalized_density,
                normalized_finite,
                normalized_count_array,
            )
            normalized_count = float(normalized_count_array)
            if (
                not bool(normalized_finite)
                or not np.isfinite(normalized_count)
                or abs(normalized_count - self.setup.system.electron_count) > 1e-4
            ):
                msg = "SCF mixer density normalization failed"
                raise ValueError(msg)
            self.progress.density = normalized_density
            if self.setup.observer is not None:
                stored_history = int(self.setup.mixer.metadata().get("stored", 0))
                self.setup.observer.record_peak_memory(
                    "shared_full_grid_bytes",
                    (4 + 2 * stored_history) * self.setup.system.grid.size * 4,
                )

    def _checkpoint(self, iteration: int) -> bool:
        if self.checkpoint_callback is None or (
            self.checkpoint_iteration is not None and self.checkpoint_iteration != iteration
        ):
            return False
        observer = self.setup.observer
        if observer is not None:
            observer.emit(
                "persistence",
                status="started",
                iteration=iteration,
                resume_eligible=True,
            )
        try:
            with observed_phase(observer, "persistence"):
                checkpoint_state = _periodic_resume._continuation_state_from_boundary(
                    completed_iteration=iteration,
                    density=self.progress.density,
                    owned_results=self.progress.final_owned_results,
                    previous_energy=float(self.progress.energy_terms["total"]),
                    energy_by_term=self.progress.energy_terms,
                    history=self.progress.history,
                    mixer=self.setup.mixer,
                    ownership=self.setup.ownership,
                    lineage=self.progress.lineage,
                )
                self.progress.final_checkpoint_state = checkpoint_state
                stop_after_checkpoint = bool(self.checkpoint_callback(checkpoint_state))
        except Exception as error:
            if observer is not None:
                observer.emit(
                    "persistence",
                    status="failed",
                    iteration=iteration,
                    resume_eligible=True,
                    error=str(error),
                )
            raise
        if observer is not None:
            observer.emit(
                "persistence",
                status="completed",
                iteration=iteration,
                resume_eligible=True,
            )
        return stop_after_checkpoint

    def _finalize(self, iteration: int) -> PeriodicSCFResult:
        final_owned_by_index = {
            result.explicit_index: result
            for result in self.progress.final_owned_results
            if result.explicit_index is not None
        }
        final_results = _publish_explicit_kpoints(
            self.setup.ownership,
            self.setup.bases,
            final_owned_by_index,
            self.setup.observer,
        )
        electron_count = float(mx.sum(self.progress.density) * self.setup.system.grid.dv)
        self._emit_completion(iteration)
        result_status = self._result_status()
        timing_admission_status = (
            "ineligible_resumed_state"
            if self.setup.resumed
            else "ineligible_checkpointed"
            if self.progress.stopped_for_checkpoint
            else "fresh"
        )
        return PeriodicSCFResult(
            converged=self.progress.converged,
            status=result_status,
            iterations=iteration,
            total_energy=float(self.progress.energy_terms["total"]),
            electron_count=electron_count,
            density_residual=self.progress.density_residual,
            energy_delta=self.progress.energy_delta,
            density=self.progress.density,
            kpoints=final_results,
            energy_by_term=self.progress.energy_terms,
            history=tuple(self.progress.history),
            timings=self.progress.timings,
            batch_policy=self.setup.config.batch_policy(),
            time_reversal_ownership=self.setup.ownership,
            numerical_status=result_status,
            resume_integrity_status="validated" if self.setup.resumed else "fresh",
            timing_admission_status=timing_admission_status,
            lineage=self.progress.lineage,
            system_fingerprint=self.setup.system.fingerprint,
            _owned_kpoints=self.progress.final_owned_results,
            _checkpoint_state=(
                None if self.progress.converged else self.progress.final_checkpoint_state
            ),
        )

    def _emit_completion(self, iteration: int) -> None:
        observer = self.setup.observer
        if observer is None:
            return
        coefficient_bytes = sum(
            int(np.prod(result.eigen._compact_coefficients.values.shape)) * 8
            for result in self.progress.final_owned_results
            if isinstance(result.eigen._compact_coefficients, _CompactLaneState)
        )
        observer.record_memory("persistent_coefficient_bytes", coefficient_bytes)
        observer.record_memory("coefficient_payload_bytes", coefficient_bytes)
        observation = observer.snapshot()
        traffic_elements = int(observation["work_counters"]["projector_traffic_elements"])
        observer.record_memory("projector_traffic_bytes", traffic_elements * 8)
        observer.emit(
            "completion",
            stage="scf",
            status=self._result_status(),
            iterations=iteration,
            total_energy_hartree=float(self.progress.energy_terms["total"]),
        )

    def _result_status(self) -> str:
        if self.progress.converged:
            return "converged"
        if self.progress.stopped_for_checkpoint:
            return "checkpointed"
        return "max_iterations"


def _run_periodic_scf_with_projector_cache(
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
    projector_cache: _GTHProjectorCache,
    resume_state: _PeriodicSCFContinuationState | None = None,
    checkpoint_callback: Callable[[_PeriodicSCFContinuationState], bool] | None = None,
    checkpoint_iteration: int | None = None,
) -> PeriodicSCFResult:
    """Run periodic SCF inside a caller-owned projector-cache lifetime."""

    return _PeriodicSCFController.create(
        system,
        cutoff_hartree=cutoff_hartree,
        kpoint_mesh=kpoint_mesh,
        n_bands=n_bands,
        config=config,
        xc_functional=xc_functional,
        initial_density=initial_density,
        initial_coefficients=initial_coefficients,
        observer=observer,
        projector_cache=projector_cache,
        resume_state=resume_state,
        checkpoint_callback=checkpoint_callback,
        checkpoint_iteration=checkpoint_iteration,
    ).run()


def _run_periodic_scf_controlled(
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
    resume_state: _PeriodicSCFContinuationState | None = None,
    checkpoint_callback: Callable[[_PeriodicSCFContinuationState], bool] | None = None,
    checkpoint_iteration: int | None = None,
) -> PeriodicSCFResult:
    with _bounded_dft_allocator(), _GTHProjectorCache() as projector_cache:
        return _run_periodic_scf_with_projector_cache(
            system,
            cutoff_hartree=cutoff_hartree,
            kpoint_mesh=kpoint_mesh,
            n_bands=n_bands,
            config=config,
            xc_functional=xc_functional,
            initial_density=initial_density,
            initial_coefficients=initial_coefficients,
            observer=observer,
            projector_cache=projector_cache,
            resume_state=resume_state,
            checkpoint_callback=checkpoint_callback,
            checkpoint_iteration=checkpoint_iteration,
        )
