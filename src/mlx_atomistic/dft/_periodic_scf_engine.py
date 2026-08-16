"""Self-consistent periodic DFT execution engine."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from time import perf_counter

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft._compact import (
    _DEFAULT_COMPACT_BATCH_POLICY,
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
    _plan_compact_submissions,
    _stable_compact_capacity_groups,
)
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
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional


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


def _density_from_kpoints(
    results: Sequence[PeriodicKPointResult],
    *,
    occupation: float,
    policy: _CompactBatchPolicy = _DEFAULT_COMPACT_BATCH_POLICY,
    observer: RuntimeObserver | None = None,
) -> mx.array:
    if not results:
        msg = "density construction requires at least one k-point result"
        raise ValueError(msg)
    grid_shape = results[0].basis.grid.shape
    states: list[_CompactLaneState] = []
    for result in results:
        if result.basis.grid.shape != grid_shape:
            msg = "density k-point results must share one real-space grid"
            raise ValueError(msg)
        compact = result.eigen._compact_coefficients
        if not isinstance(compact, _CompactLaneState):
            msg = "density construction requires owned compact k-point states"
            raise ValueError(msg)
        states.append(compact)

    def estimate_density_batch(
        _indices: tuple[int, ...],
        batch: _CompactBatch,
    ) -> int:
        return (
            batch.estimated_transient_bytes
            + batch.lane_capacity * batch.grid_size * 4
            + batch.grid_size * 4
        )

    density = mx.zeros(grid_shape, dtype=mx.float32)
    compatible: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        compatible[(id(state.layout.reciprocal), state.layout.grid_shape)].append(index)
    for compatible_indices in compatible.values():
        vector_capacity = max(states[index].vector_count for index in compatible_indices)
        capacity_groups = _stable_compact_capacity_groups(
            states,
            compatible_indices,
            lane_capacity=policy.batch_cap,
            vector_capacity=vector_capacity,
            max_padding_fraction=policy.max_padding_fraction,
        )
        for capacity_indices, capacity in capacity_groups:
            capacity_states = [states[index] for index in capacity_indices]
            plan = _plan_compact_submissions(
                capacity_states,
                policy=policy,
                batch_byte_estimator=estimate_density_batch,
                capacity=capacity,
            )
            if plan.failures:
                failed_index = min(plan.failures)
                raise _detached_failure(plan.failures[failed_index]) from None
            for submission in plan.submissions:
                logical_indices = tuple(capacity_indices[index] for index in submission.indices)
                batch = _CompactBatch.from_states(
                    [states[index] for index in logical_indices],
                    policy=policy,
                    lane_capacity=capacity.lanes,
                    vector_capacity=capacity.vectors,
                    active_capacity=capacity.active,
                )
                estimated_transient_bytes = estimate_density_batch(
                    logical_indices,
                    batch,
                )
                if estimated_transient_bytes > policy.max_transient_bytes:
                    msg = "density batch exceeds the complete transient byte budget"
                    raise ValueError(msg)
                orbitals = batch.to_real()
                weights = mx.array(
                    np.asarray(
                        [
                            results[index].integration_weight * occupation
                            for index in logical_indices
                        ],
                        dtype=np.float32,
                    )
                )
                if batch.lane_capacity > batch.lane_count:
                    weights = mx.concatenate(
                        [
                            weights,
                            mx.zeros(
                                (batch.lane_capacity - batch.lane_count,),
                                dtype=mx.float32,
                            ),
                        ]
                    )
                weighted_density = weights[:, None, None, None] * mx.sum(
                    mx.abs(orbitals) ** 2,
                    axis=1,
                )
                density = density + mx.sum(weighted_density, axis=0)
                mx.eval(density)
                add_observed_work(
                    observer,
                    {
                        "fft_submissions": 1,
                        "fft_vector_equivalents": batch.logical_vector_count,
                        "padding_elements": batch.padding_elements,
                    },
                )
                if observer is not None:
                    observer.record_peak_memory(
                        "fft_workspace_bytes",
                        batch.lane_capacity * batch.vector_count * batch.grid_size * 8,
                    )
                    observer.record_peak_memory(
                        "peak_temporary_bytes",
                        estimated_transient_bytes,
                    )

    return mx.real(density)


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


def _owned_device_copy(values: mx.array, *, dtype: mx.Dtype) -> mx.array:
    source = mx.array(values).astype(dtype)
    copied = (source + mx.zeros_like(source)).astype(dtype)
    mx.eval(copied)
    return copied


def _continuation_state_from_boundary(
    *,
    completed_iteration: int,
    density: mx.array,
    owned_results: Sequence[PeriodicKPointResult],
    previous_energy: float,
    energy_by_term: dict[str, float],
    history: Sequence[dict[str, float | int | str | None]],
    mixer: LinearMixer | PulayDIISMixer,
    ownership: TimeReversalOwnership,
    lineage: tuple[str, ...],
) -> _PeriodicSCFContinuationState:
    owned_coefficients: list[tuple[int, mx.array]] = []
    owned_lanes: list[dict[str, object]] = []
    for result in owned_results:
        if result.explicit_index is None:
            msg = "checkpoint state requires explicit owner indices"
            raise RuntimeError(msg)
        compact = result.eigen._compact_coefficients
        if not isinstance(compact, _CompactLaneState):
            msg = "checkpoint state requires compact owner coefficients"
            raise RuntimeError(msg)
        owned_coefficients.append(
            (
                result.explicit_index,
                _owned_device_copy(compact.values, dtype=mx.complex64),
            )
        )
        owned_lanes.append(
            {
                "owner_index": result.explicit_index,
                "reduced_kpoint": list(result.reduced_kpoint),
                "active_count": result.basis.active_count,
                "basis_fingerprint": result.basis.basis_fingerprint,
                "basis_order_fingerprint": result.basis.order_fingerprint,
                "lane_id": result.basis.lane_id,
            }
        )
    return _PeriodicSCFContinuationState(
        completed_iteration=completed_iteration,
        density=_owned_device_copy(density, dtype=mx.float32),
        owned_coefficients=tuple(owned_coefficients),
        owned_lanes=tuple(owned_lanes),
        previous_energy=float(previous_energy),
        energy_by_term=dict(energy_by_term),
        history=tuple(dict(row) for row in history),
        mixer_state=mixer._checkpoint_state(),
        ownership=ownership.to_dict(),
        lineage=lineage,
    )


def _resume_ownership(
    rebuilt: TimeReversalOwnership,
    stored: dict[str, object],
) -> TimeReversalOwnership:
    if rebuilt.to_dict() == stored:
        return rebuilt
    stored_entries = stored.get("entries")
    if not isinstance(stored_entries, list) or len(stored_entries) != len(rebuilt.entries):
        msg = "periodic resume ownership payload is malformed"
        raise ValueError(msg)
    candidate = rebuilt
    for index, rebuilt_entry in enumerate(rebuilt.entries):
        stored_entry = stored_entries[index]
        if not isinstance(stored_entry, dict):
            msg = "periodic resume ownership entry is malformed"
            raise ValueError(msg)
        if (
            rebuilt_entry.role in {"owner", "partner"}
            and stored_entry.get("role") == "independent"
            and stored_entry.get("fallback_reason") == "initial_coefficients_time_reversal_mismatch"
        ):
            candidate = _independent_pair(
                candidate,
                index,
                "initial_coefficients_time_reversal_mismatch",
            )
    if candidate.to_dict() != stored:
        msg = "periodic resume ownership is not a valid stored fallback refinement"
        raise ValueError(msg)
    return candidate


def _restore_continuation_state(
    state: _PeriodicSCFContinuationState,
    *,
    bases: Sequence[PlaneWaveBasis],
    ownership: TimeReversalOwnership,
    occupied_bands: int,
    grid: RealSpaceGrid,
    electron_count: float,
    mixer: LinearMixer | PulayDIISMixer,
) -> tuple[
    mx.array,
    dict[int, _CompactLaneState],
    float,
    list[dict[str, float | int | str | None]],
    dict[str, float],
]:
    if state.completed_iteration <= 0:
        msg = "periodic resume iteration must be positive"
        raise ValueError(msg)
    if state.ownership != ownership.to_dict():
        msg = "periodic resume ownership does not match the rebuilt topology"
        raise ValueError(msg)
    if len(state.history) != state.completed_iteration or any(
        row.get("iteration") != index for index, row in enumerate(state.history, start=1)
    ):
        msg = "periodic resume history does not match its iteration cursor"
        raise ValueError(msg)
    if not np.isfinite(state.previous_energy):
        msg = "periodic resume energy must be finite"
        raise ValueError(msg)
    if (
        not state.history
        or not np.isclose(
            float(state.history[-1]["total_energy_hartree"]),
            state.previous_energy,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.isclose(
            float(state.energy_by_term.get("total", float("nan"))),
            state.previous_energy,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        msg = "periodic resume energy state is internally inconsistent"
        raise ValueError(msg)

    density = mx.array(state.density)
    if density.shape != grid.shape or density.dtype != mx.float32:
        msg = "periodic resume density has incompatible shape or dtype"
        raise ValueError(msg)
    density_finite = mx.all(mx.isfinite(density))
    density_minimum = mx.min(density)
    density_count = mx.sum(density) * grid.dv
    mx.eval(density, density_finite, density_minimum, density_count)
    if (
        not bool(density_finite)
        or float(density_minimum) < -1e-7
        or abs(float(density_count) - electron_count) > 1e-4
    ):
        msg = "periodic resume density is non-finite, negative, or misnormalized"
        raise ValueError(msg)
    density = _owned_device_copy(density, dtype=mx.float32)

    coefficient_map = state.coefficient_map
    if len(coefficient_map) != len(state.owned_coefficients) or set(coefficient_map) != set(
        ownership.owned_indices
    ):
        msg = "periodic resume owner coefficient inventory is inconsistent"
        raise ValueError(msg)
    lane_map = {
        int(lane["owner_index"]): lane
        for lane in state.owned_lanes
        if isinstance(lane, dict) and "owner_index" in lane
    }
    if len(lane_map) != len(state.owned_lanes) or set(lane_map) != set(ownership.owned_indices):
        msg = "periodic resume owner lane inventory is inconsistent"
        raise ValueError(msg)
    previous_states: dict[int, _CompactLaneState] = {}
    finite_checks: list[mx.array] = []
    for owner_index in ownership.owned_indices:
        basis = bases[owner_index]
        lane = lane_map[owner_index]
        expected_lane = {
            "owner_index": owner_index,
            "reduced_kpoint": list(ownership.entry_for(owner_index).reduced_kpoint),
            "active_count": basis.active_count,
            "basis_fingerprint": basis.basis_fingerprint,
            "basis_order_fingerprint": basis.order_fingerprint,
            "lane_id": basis.lane_id,
        }
        if lane != expected_lane:
            msg = "periodic resume owner lane identity does not match rebuilt bases"
            raise ValueError(msg)
        values = mx.array(coefficient_map[owner_index])
        if values.dtype != mx.complex64 or values.shape != (occupied_bands, basis.active_count):
            msg = "periodic resume owner coefficients have incompatible shape or dtype"
            raise ValueError(msg)
        copied = _owned_device_copy(values, dtype=mx.complex64)
        finite_checks.append(mx.all(mx.isfinite(copied)))
        previous_states[owner_index] = basis._state_from_compact(copied)
    mx.eval(*finite_checks)
    if not all(bool(finite) for finite in finite_checks):
        msg = "periodic resume owner coefficients must be finite"
        raise ValueError(msg)

    mixer._restore_checkpoint_state(state.mixer_state, expected_shape=grid.shape)
    return (
        density,
        previous_states,
        float(state.previous_energy),
        [dict(row) for row in state.history],
        dict(state.energy_by_term),
    )


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
    """Run periodic SCF inside a caller-owned projector-cache lifetime.

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
        projector_cache: Cache closed by the public runtime-context wrapper.
        resume_state: Validated internal next-iteration state. Defaults to fresh.
        checkpoint_callback: Optional accepted-iteration publisher returning
            whether execution should stop after publication.
        checkpoint_iteration: Optional single iteration at which to materialize
            callback state. Defaults to every accepted iteration when a callback
            is present.

    Returns:
        Periodic SCF result with complete weighted k-point diagnostics.
    """

    scf_config = PeriodicSCFConfig() if config is None else config
    compact_policy = scf_config._compact_batch_policy()
    xc = ProductionPBEExchangeCorrelation() if xc_functional is None else xc_functional
    occupied_bands = int(round(system.electron_count / 2.0)) if n_bands is None else n_bands
    if occupied_bands <= 0 or abs(2.0 * occupied_bands - system.electron_count) > 1e-8:
        msg = "the bounded spin-unpolarized path requires two electrons per occupied band"
        raise ValueError(msg)
    for point in kpoint_mesh.points:
        if point.coordinate_system != "reduced":
            msg = "periodic SCF requires reduced-coordinate k-points"
            raise ValueError(msg)
    if initial_coefficients is not None and len(initial_coefficients) != len(kpoint_mesh.points):
        msg = "initial_coefficients length must match the k-point mesh"
        raise ValueError(msg)
    if resume_state is not None and (
        initial_density is not None or initial_coefficients is not None
    ):
        msg = "periodic resume state is mutually exclusive with public initial guesses"
        raise ValueError(msg)
    if resume_state is not None and resume_state.completed_iteration >= scf_config.max_iterations:
        msg = "periodic resume state has no remaining SCF iteration"
        raise ValueError(msg)
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
        bases = [
            PlaneWaveBasis.from_reduced_kpoint(
                system.grid,
                cutoff_hartree,
                point.vector,
                reciprocal_grid=shared_reciprocal,
                lane_label=f"kpoint:{point_index}",
            )
            for point_index, point in enumerate(kpoint_mesh.points)
        ]
        ownership = admit_time_reversal_bases(ownership, bases)
        if resume_state is None:
            ownership, previous_states = _admit_initial_time_reversal(
                ownership,
                bases,
                initial_coefficients,
                n_bands=occupied_bands,
            )
        else:
            ownership = _resume_ownership(ownership, resume_state.ownership)
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
            if initial_density is None:
                density = mx.full(
                    system.grid.shape,
                    system.electron_count / system.grid.volume,
                )
            else:
                density = mx.real(mx.array(initial_density))
                if density.shape != system.grid.shape:
                    msg = "initial_density must have shape system.grid.shape"
                    raise ValueError(msg)
                count = float(mx.sum(density) * system.grid.dv)
                if count <= 0.0:
                    msg = "initial_density must integrate to a positive count"
                    raise ValueError(msg)
                density = density * (system.electron_count / count)
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
            ) = _restore_continuation_state(
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
    if observer is not None:
        observer.record_memory("shared_full_grid_bytes", system.grid.size * 4 * 4)
        observer.record_memory("persistent_projector_bytes", 0)
        observer.emit(
            "setup",
            status="completed",
            active_counts=[basis.active_count for basis in bases],
            owned_indices=list(owned_indices),
            owned_active_counts=[bases[index].active_count for index in owned_indices],
            representative_count=len(ownership.representative_indices),
            fallback_reasons=ownership.fallback_reasons,
            batch_policy=scf_config.batch_policy(),
            resumed=resume_state is not None,
            iteration_start=iteration_start,
        )
    final_owned_results: tuple[PeriodicKPointResult, ...] = ()
    converged = False
    stopped_for_checkpoint = False
    final_checkpoint_state: _PeriodicSCFContinuationState | None = None
    density_residual = float("inf")
    energy_delta: float | None = None
    timings = {"hartree": 0.0, "xc": 0.0, "eigensolver": 0.0, "total": 0.0}
    total_start = perf_counter()
    for iteration in range(iteration_start, scf_config.max_iterations + 1):
        if observer is not None:
            observer.emit(
                "scf_iteration",
                status="started",
                iteration=iteration,
                total_iterations=scf_config.max_iterations,
                eigensolver_tolerance=eigensolver_tolerance,
            )
        start = perf_counter()
        hartree = hartree_potential(density, system.grid)
        timings["hartree"] += (perf_counter() - start) * 1000.0
        start = perf_counter()
        xc_result = xc.evaluate(density, system.grid)
        timings["xc"] += (perf_counter() - start) * 1000.0
        effective = local_potential + hartree + xc_result.potential
        effective_snapshot = mx.array(effective)
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
        owned_by_index: dict[int, PeriodicKPointResult] = {}
        max_orbital_residual = 0.0
        start = perf_counter()
        operators_by_index = {
            point_index: PeriodicKohnShamOperator._from_shared_potential(
                bases[point_index],
                effective_snapshot,
                nonlocal_operators[point_index],
                observer,
            )
            for point_index in owned_indices
        }
        lane_to_index = {
            bases[point_index]._layout.lane_id: point_index for point_index in owned_indices
        }

        def emit_submission(
            status: str,
            batch_index: int,
            tickets: tuple[_DavidsonApplicationTicket, ...],
            batch: _CompactBatch,
            failures: dict[str, Exception],
            *,
            _iteration: int = iteration,
            _lane_to_index: dict[str, int] = lane_to_index,
        ) -> None:
            if observer is None or not observer.detail_events:
                return
            explicit_indices = [_lane_to_index[ticket.lane_id] for ticket in tickets]
            complete_transient_bytes = PeriodicKohnShamOperator._estimated_batch_transient_bytes(
                [ticket.operator for ticket in tickets],
                batch,
            )
            fields: dict[str, object] = {
                "status": status,
                "scf_iteration": _iteration,
                "batch_index": batch_index,
                "batch_size": len(tickets),
                "lane_capacity": batch.lane_capacity,
                "lane_ids": [ticket.lane_id for ticket in tickets],
                "reduced_kpoints": [
                    list(kpoint_mesh.points[index].vector) for index in explicit_indices
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
                "compact_batch_transient_bytes": (batch.estimated_transient_bytes),
                "batch_policy": scf_config.batch_policy(),
                "synchronized": observer.synchronize is not None,
            }
            if failures:
                fields["failed_explicit_indices"] = [
                    _lane_to_index[lane_id] for lane_id in failures
                ]
                fields["failure_messages"] = {
                    lane_id: str(error) for lane_id, error in failures.items()
                }
            observer.emit("kpoint_batch", **fields)

        iteration_davidson = replace(
            scf_config.davidson,
            tolerance=eigensolver_tolerance,
        )
        requests = tuple(
            _DavidsonLaneRequest(
                lane_id=bases[point_index]._layout.lane_id,
                operator=operators_by_index[point_index],
                n_bands=occupied_bands,
                config=iteration_davidson,
                trial=_initial_trial(
                    bases[point_index],
                    occupied_bands,
                    previous_states.get(point_index),
                ),
                observer=observer,
                trial_is_orthonormal=(
                    previous_states.get(point_index) is None
                    or resume_state is not None
                    or iteration > iteration_start
                ),
            )
            for point_index in owned_indices
        )
        def new_scheduler() -> _DavidsonScheduler:
            return _DavidsonScheduler(
                policy=compact_policy,
                submission_callback=emit_submission,
            )
        with observed_phase(
            observer,
            "eigensolver_control",
            synchronize=False,
        ):
            eigen_outcome = _DavidsonEngine(
                scheduler=new_scheduler(),
            ).solve(requests)
        if eigen_outcome.failures:
            if observer is not None:
                observer.emit(
                    "failure",
                    stage="eigensolver",
                    scf_iteration=iteration,
                    failed_explicit_indices=[
                        lane_to_index[lane_id] for lane_id in eigen_outcome.failures
                    ],
                    failure_messages={
                        lane_id: str(error)
                        for lane_id, error in eigen_outcome.failures.items()
                    },
                )
            first_failed_lane = next(
                request.lane_id
                for request in requests
                if request.lane_id in eigen_outcome.failures
            )
            raise _detached_failure(eigen_outcome.failures[first_failed_lane]) from None

        for point_index in owned_indices:
            basis = bases[point_index]
            entry = ownership.entry_for(point_index)
            eigen = eigen_outcome.result_for(basis._layout.lane_id)
            add_observed_work(observer, {"kpoint_lane_solves": 1})
            if entry.role == "owner":
                add_observed_work(observer, {"representative_lane_solves": 1})
            max_orbital_residual = max(
                max_orbital_residual,
                float(mx.max(eigen.residuals)),
            )
            owned_by_index[point_index] = _owned_kpoint_result(
                entry=entry,
                basis=basis,
                eigen=eigen,
            )
        timings["eigensolver"] += (perf_counter() - start) * 1000.0
        final_owned_results = tuple(owned_by_index[index] for index in owned_indices)
        with observed_phase(observer, "density"):
            target_density = _density_from_kpoints(
                final_owned_results,
                occupation=2.0,
                policy=compact_policy,
                observer=observer,
            )
            target_count = float(mx.sum(target_density) * system.grid.dv)
            target_density = target_density * (system.electron_count / target_count)
            density_residual = _density_residual(density, target_density, system.grid)

        band_energy = sum(
            result.integration_weight * 2.0 * float(mx.sum(result.eigen.eigenvalues))
            for result in final_owned_results
        )
        hartree_energy = 0.5 * float(mx.sum(density * hartree) * system.grid.dv)
        xc_energy = float(xc_result.total_energy)
        density_xc = float(mx.sum(density * xc_result.potential) * system.grid.dv)
        total_energy = band_energy - hartree_energy + xc_energy - density_xc + ewald
        energy_delta = None if previous_energy is None else total_energy - previous_energy
        energy_terms = {
            "band": band_energy,
            "hartree": hartree_energy,
            "xc": xc_energy,
            "density_xc_potential": density_xc,
            "ion_ewald": ewald,
            "total": total_energy,
        }
        history.append(
            {
                "iteration": iteration,
                "total_energy_hartree": total_energy,
                "energy_delta_hartree": energy_delta,
                "density_residual": density_residual,
                "electron_count": target_count,
                "max_orbital_residual": max_orbital_residual,
                "eigensolver_tolerance": eigensolver_tolerance,
                "eigensolver_method": "davidson",
                "all_kpoints_converged": str(
                    all(result.eigen.converged for result in final_owned_results)
                ).lower(),
            }
        )
        all_eigen_converged = all(result.eigen.converged for result in final_owned_results)
        if observer is not None:
            observer.emit(
                "scf_iteration",
                status="completed",
                iteration=iteration,
                total_energy_hartree=total_energy,
                energy_delta_hartree=energy_delta,
                density_residual=density_residual,
                max_orbital_residual=max_orbital_residual,
                eigensolver_tolerance=eigensolver_tolerance,
                eigensolver_method="davidson",
                all_kpoints_converged=all_eigen_converged,
            )
        if (
            iteration >= scf_config.min_iterations
            and all_eigen_converged
            and density_residual <= scf_config.density_tolerance
            and energy_delta is not None
            and abs(energy_delta) <= scf_config.energy_tolerance
            and max_orbital_residual <= scf_config.orbital_tolerance
        ):
            converged = True
            density = target_density
            break
        eigensolver_tolerance = _next_scf_eigensolver_tolerance(
            scf_config,
            eigensolver_tolerance,
            density_residual,
            system.electron_count,
        )
        with observed_phase(observer, "mixing"):
            mixed = mixer.mix(density, target_density)
            mixed_finite = mx.all(mx.isfinite(mixed))
            mixed_minimum_array = mx.min(mixed)
            mixed_count_array = mx.sum(mixed) * system.grid.dv
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
            normalized_density = mixed * (system.electron_count / mixed_count)
            normalized_finite = mx.all(mx.isfinite(normalized_density))
            normalized_count_array = mx.sum(normalized_density) * system.grid.dv
            mx.eval(
                normalized_density,
                normalized_finite,
                normalized_count_array,
            )
            normalized_count = float(normalized_count_array)
            if (
                not bool(normalized_finite)
                or not np.isfinite(normalized_count)
                or abs(normalized_count - system.electron_count) > 1e-4
            ):
                msg = "SCF mixer density normalization failed"
                raise ValueError(msg)
            density = normalized_density
            if observer is not None:
                stored_history = int(mixer.metadata().get("stored", 0))
                observer.record_peak_memory(
                    "shared_full_grid_bytes",
                    (4 + 2 * stored_history) * system.grid.size * 4,
                )
        previous_energy = total_energy
        previous_states = {
            result.explicit_index: result.eigen._compact_coefficients
            for result in final_owned_results
            if result.explicit_index is not None
        }
        capture_for_callback = checkpoint_callback is not None and (
            checkpoint_iteration is None or checkpoint_iteration == iteration
        )
        if capture_for_callback:
            if observer is not None:
                observer.emit(
                    "persistence",
                    status="started",
                    iteration=iteration,
                    resume_eligible=True,
                )
            try:
                with observed_phase(observer, "persistence"):
                    final_checkpoint_state = _continuation_state_from_boundary(
                        completed_iteration=iteration,
                        density=density,
                        owned_results=final_owned_results,
                        previous_energy=total_energy,
                        energy_by_term=energy_terms,
                        history=history,
                        mixer=mixer,
                        ownership=ownership,
                        lineage=lineage,
                    )
                    stop_after_checkpoint = bool(checkpoint_callback(final_checkpoint_state))
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
            if stop_after_checkpoint:
                stopped_for_checkpoint = True
                break

    timings["total"] = (perf_counter() - total_start) * 1000.0
    final_owned_by_index = {
        result.explicit_index: result
        for result in final_owned_results
        if result.explicit_index is not None
    }
    final_results = _publish_explicit_kpoints(
        ownership,
        bases,
        final_owned_by_index,
        observer,
    )
    electron_count = float(mx.sum(density) * system.grid.dv)
    if observer is not None:
        coefficient_bytes = sum(
            int(np.prod(result.eigen._compact_coefficients.values.shape)) * 8
            for result in final_owned_results
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
            status=(
                "converged"
                if converged
                else "checkpointed"
                if stopped_for_checkpoint
                else "max_iterations"
            ),
            iterations=iteration,
            total_energy_hartree=float(energy_terms["total"]),
        )
    result_status = (
        "converged" if converged else "checkpointed" if stopped_for_checkpoint else "max_iterations"
    )
    timing_admission_status = (
        "ineligible_resumed_state"
        if resume_state is not None
        else "ineligible_checkpointed"
        if stopped_for_checkpoint
        else "fresh"
    )
    return PeriodicSCFResult(
        converged=converged,
        status=result_status,
        iterations=iteration,
        total_energy=float(energy_terms["total"]),
        electron_count=electron_count,
        density_residual=density_residual,
        energy_delta=energy_delta,
        density=density,
        kpoints=final_results,
        energy_by_term=energy_terms,
        history=tuple(history),
        timings=timings,
        batch_policy=scf_config.batch_policy(),
        time_reversal_ownership=ownership,
        numerical_status=result_status,
        resume_integrity_status="validated" if resume_state is not None else "fresh",
        timing_admission_status=timing_admission_status,
        lineage=lineage,
        system_fingerprint=system.fingerprint,
        _owned_kpoints=final_owned_results,
        _checkpoint_state=None if converged else final_checkpoint_state,
    )


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

