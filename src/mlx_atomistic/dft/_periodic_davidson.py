"""Block-Davidson eigensolver and compact scheduling for periodic DFT."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft._compact import (
    _CompactLaneState,
    _remap_initial_coefficients,
    _require_layout,
)
from mlx_atomistic.dft._periodic_davidson_scheduler import (
    _DavidsonApplicationTicket,
    _DavidsonScheduler,
    _DavidsonScheduleResult,
    _FixedHamiltonianToken,
)
from mlx_atomistic.dft._periodic_davidson_subspace import (
    _DavidsonRitzCandidate,
    _DavidsonRitzPair,
    _PairedDavidsonState,
    _projected_eigh_batch,
    _ritz_candidate_from_projected_eigensystem,
    _ritz_candidate_with_direct_action,
    _ritz_pair,
    _ritz_pair_with_direct_action,
    _seal_ritz_candidate,
)
from mlx_atomistic.dft._periodic_execution import (
    _detached_failure,
    _materialize,
)
from mlx_atomistic.dft._periodic_hamiltonian import PeriodicKohnShamOperator
from mlx_atomistic.dft._periodic_models import PeriodicDavidsonConfig, PeriodicEigenResult
from mlx_atomistic.dft._periodic_orthonormalization import (
    _DAVIDSON_RANK_POLICY,
    _Complex64RankPolicy,
)
from mlx_atomistic.dft._runtime_observer import (
    RuntimeObserver,
    add_observed_work,
    observed_phase,
)
from mlx_atomistic.dft.plane_wave import PlaneWaveBasis


def _logical_hpsi_memory(
    *,
    vector_count: int,
    grid_count: int,
    projector_elements: int,
) -> tuple[int, int]:
    """Preserve the frozen full-grid Hpsi memory model for baseline audits."""

    fft_workspace_bytes = 2 * vector_count * grid_count * 8
    peak_temporary_bytes = fft_workspace_bytes + projector_elements * 8
    return fft_workspace_bytes, peak_temporary_bytes


def _initial_coefficients(basis: PlaneWaveBasis, count: int) -> _CompactLaneState:
    if count > basis.active_count:
        msg = "orbital count exceeds the active plane-wave basis"
        raise ValueError(msg)
    selected = mx.argsort(basis._layout._active_kinetic_energies)[:count]
    slots = mx.arange(basis.active_count, dtype=selected.dtype)[None, :]
    coefficients = (slots == selected[:, None]).astype(mx.complex64)
    return basis._state_from_compact(coefficients)


def _initial_trial(
    basis: PlaneWaveBasis,
    n_bands: int,
    initial_coefficients: object | None,
) -> _CompactLaneState:
    if isinstance(initial_coefficients, _PairedDavidsonState):
        msg = "paired Davidson H(V) cannot seed a new fixed-Hamiltonian solve"
        raise ValueError(msg)
    if initial_coefficients is None:
        return _initial_coefficients(basis, n_bands)
    if isinstance(initial_coefficients, _CompactLaneState):
        try:
            _require_layout(initial_coefficients, basis._layout)
            return initial_coefficients
        except ValueError:
            return _remap_initial_coefficients(
                initial_coefficients,
                basis._layout,
            )
    trial, _ = basis._state_from_full(initial_coefficients)
    return trial


@dataclass(frozen=True)
class _DavidsonLaneRequest:
    """One unpadded fixed-Hamiltonian lane submitted to the shared engine."""

    lane_id: str
    operator: PeriodicKohnShamOperator
    n_bands: int
    config: PeriodicDavidsonConfig
    trial: _CompactLaneState
    observer: RuntimeObserver | None
    rank_policy: _Complex64RankPolicy = _DAVIDSON_RANK_POLICY
    trial_is_orthonormal: bool = False
    require_direct_validation: bool = True
    capture_orbital_density: bool = False


@dataclass(frozen=True)
class _DavidsonPendingAction:
    """One scheduled Davidson action and the state needed to consume it."""

    purpose: Literal["correction", "direct_validation"]
    vectors: _CompactLaneState
    reused_width: int = 0
    ritz_pair: _DavidsonRitzPair | None = None
    terminal: bool = False

    def __post_init__(self) -> None:
        if self.purpose == "correction":
            if self.ritz_pair is not None or self.terminal:
                msg = "Davidson correction action has invalid validation state"
                raise ValueError(msg)
        elif self.purpose == "direct_validation":
            if self.ritz_pair is None or self.reused_width != 0:
                msg = "Davidson direct validation action is incomplete"
                raise ValueError(msg)
        else:
            msg = "Davidson pending action purpose is invalid"
            raise ValueError(msg)


@dataclass
class _DavidsonLaneProgress:
    """Mutable lane-local Davidson progression owned by the shared engine."""

    request: _DavidsonLaneRequest
    token: _FixedHamiltonianToken
    initial_vectors: _CompactLaneState
    paired: _PairedDavidsonState | None = None
    ritz_pair: _DavidsonRitzPair | None = None
    iteration_count: int = 0
    restart_count: int = 0
    correction_width: int = 0
    pending_action: _DavidsonPendingAction | None = None
    orbital_density: mx.array | None = None
    direct_validated: bool = False
    converged: bool = False
    done: bool = False
    failure: Exception | None = None


@dataclass(frozen=True)
class _DavidsonEngineResult:
    """Independent lane outcomes and actual shared-engine scheduling evidence."""

    results: dict[str, PeriodicEigenResult]
    orbital_densities: dict[str, mx.array]
    failures: dict[str, Exception]
    ready_rounds: tuple[tuple[str, ...], ...]
    compatibility_groups: tuple[tuple[str, ...], ...]
    submission_groups: tuple[tuple[str, ...], ...]
    scheduler_calls: int

    def result_for(self, lane_id: str) -> PeriodicEigenResult:
        failure = self.failures.get(lane_id)
        if failure is not None:
            raise _detached_failure(failure) from None
        try:
            return self.results[lane_id]
        except KeyError as error:
            msg = f"Davidson engine has no result for lane {lane_id!r}"
            raise ValueError(msg) from error

    def orbital_density_for(self, lane_id: str) -> mx.array:
        """Return the captured final-orbital density for one lane."""

        failure = self.failures.get(lane_id)
        if failure is not None:
            raise _detached_failure(failure) from None
        try:
            return self.orbital_densities[lane_id]
        except KeyError as error:
            msg = f"Davidson engine has no orbital density for lane {lane_id!r}"
            raise ValueError(msg) from error


class _DavidsonEngine:
    """Advance ragged Davidson lanes and collectively schedule ready H blocks."""

    def __init__(self, *, scheduler: _DavidsonScheduler | None = None) -> None:
        self.scheduler = _DavidsonScheduler() if scheduler is None else scheduler
        self._ready_rounds: list[tuple[str, ...]] = []
        self._compatibility_groups: list[tuple[str, ...]] = []
        self._submission_groups: list[tuple[str, ...]] = []
        self._scheduler_calls = 0

    @staticmethod
    def _validate_request(request: _DavidsonLaneRequest) -> None:
        operator = request.operator
        observer = request.observer
        if (
            observer is not None
            and operator.observer is not None
            and observer is not operator.observer
        ):
            msg = "operator and solver observers must be the same object"
            raise ValueError(msg)
        basis = operator.basis
        if (
            type(request.n_bands) is not int
            or request.n_bands <= 0
            or request.n_bands > basis.active_count
        ):
            msg = "n_bands must be a positive non-bool integer no larger than the active basis size"
            raise ValueError(msg)
        if request.config.max_subspace_size < request.n_bands:
            msg = "max_subspace_size cannot be smaller than n_bands"
            raise ValueError(msg)
        basis._validate_state(request.trial)
        if request.trial.kind != "coefficients":
            msg = "initial coefficients cannot be a cached Hamiltonian action"
            raise ValueError(msg)
        if request.trial.vector_count < request.n_bands:
            msg = "initial coefficients contain fewer vectors than requested bands"
            raise ValueError(msg)
        if type(request.trial_is_orthonormal) is not bool:
            msg = "trial_is_orthonormal must be a bool"
            raise ValueError(msg)
        if type(request.require_direct_validation) is not bool:
            msg = "require_direct_validation must be a bool"
            raise ValueError(msg)
        if type(request.capture_orbital_density) is not bool:
            msg = "capture_orbital_density must be a bool"
            raise ValueError(msg)
        if request.capture_orbital_density and not request.require_direct_validation:
            msg = "orbital density capture requires direct validation"
            raise ValueError(msg)

    @staticmethod
    def _ticket(
        progress: _DavidsonLaneProgress,
        vectors: _CompactLaneState,
        *,
        purpose: Literal["initial", "correction", "direct_validation"] = "initial",
    ) -> _DavidsonApplicationTicket:
        request = progress.request
        return _DavidsonApplicationTicket(
            lane_id=request.lane_id,
            operator=request.operator,
            config=request.config,
            n_bands=request.n_bands,
            rank_policy=request.rank_policy,
            token=progress.token,
            vectors=vectors,
            observer=request.observer,
            purpose=purpose,
            capture_orbital_density=(
                request.capture_orbital_density and purpose == "direct_validation"
            ),
        )

    def _schedule(
        self,
        tickets: Sequence[_DavidsonApplicationTicket],
    ) -> _DavidsonScheduleResult:
        self._ready_rounds.append(tuple(ticket.lane_id for ticket in tickets))
        scheduled = self.scheduler.apply(tickets)
        self._scheduler_calls += 1
        self._compatibility_groups.extend(scheduled.compatibility_groups)
        self._submission_groups.extend(scheduled.groups)
        return scheduled

    @staticmethod
    def _fail_lane(
        progress: _DavidsonLaneProgress,
        failures: dict[str, Exception],
        error: Exception,
    ) -> None:
        failure = _detached_failure(error)
        failures[progress.request.lane_id] = failure
        progress.failure = failure
        progress.done = True

    def _prepare_lane(
        self,
        request: _DavidsonLaneRequest,
    ) -> _DavidsonLaneProgress:
        self._validate_request(request)
        basis = request.operator.basis
        if request.trial_is_orthonormal:
            if request.trial.vector_count > min(
                request.config.max_subspace_size,
                basis.active_count,
            ):
                msg = "trusted initial coefficients exceed the Davidson subspace limit"
                raise ValueError(msg)
            initial_vectors = request.trial
        else:
            with observed_phase(
                request.observer,
                "orthogonalization",
                synchronize=False,
            ):
                initial_rank = request.rank_policy.orthonormalize(
                    request.trial.values,
                    required_count=request.n_bands,
                    max_count=min(
                        request.config.max_subspace_size,
                        basis.active_count,
                    ),
                )
                initial_vectors = basis._state_from_compact(initial_rank.values)
            add_observed_work(
                request.observer,
                {"orthogonalization_vectors": request.trial.vector_count},
            )
        token = _FixedHamiltonianToken.create(
            request.operator,
            request.config,
            request.n_bands,
            request.rank_policy,
        )
        return _DavidsonLaneProgress(
            request=request,
            token=token,
            initial_vectors=initial_vectors,
        )

    @staticmethod
    def _unconverged_indices(
        ritz_pair: _DavidsonRitzPair,
        tolerance: float,
    ) -> np.ndarray:
        residual_values = np.asarray(ritz_pair.residuals, dtype=np.float32)
        return np.flatnonzero(residual_values > tolerance).astype(np.int32)

    @staticmethod
    def _emit_iteration(
        progress: _DavidsonLaneProgress,
        ritz_pair: _DavidsonRitzPair,
        *,
        residual_source: str,
    ) -> None:
        request = progress.request
        unconverged = _DavidsonEngine._unconverged_indices(
            ritz_pair,
            request.config.tolerance,
        )
        if request.observer is not None and request.observer.detail_events:
            request.observer.emit(
                "davidson_iteration",
                lane_id=request.lane_id,
                iteration=progress.iteration_count,
                subspace_size=(0 if progress.paired is None else progress.paired.vector_count),
                max_residual=ritz_pair.max_residual,
                unconverged_band_count=int(unconverged.size),
                residual_source=residual_source,
                converged=ritz_pair.max_residual <= request.config.tolerance,
            )

    @staticmethod
    def _prepare_direct_validation(
        progress: _DavidsonLaneProgress,
        ritz_pair: _DavidsonRitzPair,
        *,
        terminal: bool,
    ) -> _DavidsonPendingAction:
        request = progress.request
        paired = progress.paired
        if paired is None:
            msg = "Davidson lane has no paired V/HV state"
            raise RuntimeError(msg)
        orthonormality = request.rank_policy.overlap_error(ritz_pair.vectors.values)
        if orthonormality > request.rank_policy.guard_tolerance(request.n_bands):
            with observed_phase(
                request.observer,
                "orthogonalization",
                synchronize=False,
            ):
                final_rank = request.rank_policy.orthonormalize(
                    ritz_pair.vectors.values,
                    required_count=request.n_bands,
                    max_count=request.n_bands,
                )
                paired = paired.rebase_ranked(
                    final_rank,
                    ritz_pair.applied,
                    token=progress.token,
                )
                ritz_pair = _ritz_pair(paired, request.n_bands)
            add_observed_work(
                request.observer,
                {"orthogonalization_vectors": request.n_bands},
            )
            progress.paired = paired
        request.rank_policy.validate(
            ritz_pair.vectors.values,
            required_count=request.n_bands,
        )
        pending = _DavidsonPendingAction(
            purpose="direct_validation",
            vectors=ritz_pair.vectors,
            ritz_pair=ritz_pair,
            terminal=terminal,
        )
        progress.ritz_pair = ritz_pair
        progress.pending_action = pending
        progress.direct_validated = False
        return pending

    @staticmethod
    def _raw_corrections(
        progress: _DavidsonLaneProgress,
        ritz_pair: _DavidsonRitzPair,
    ) -> tuple[mx.array, int]:
        request = progress.request
        unconverged = _DavidsonEngine._unconverged_indices(
            ritz_pair,
            request.config.tolerance,
        )
        if unconverged.size == 0:
            msg = "Davidson residual decision disagrees with its maximum"
            raise RuntimeError(msg)
        unconverged_indices = mx.array(unconverged)
        unconverged_eigenvalues = mx.take(
            ritz_pair.eigenvalues,
            unconverged_indices,
            axis=0,
        )
        unconverged_residuals = mx.take(
            ritz_pair.residual_stack,
            unconverged_indices,
            axis=0,
        )
        denominator = (
            request.operator.basis._layout._active_kinetic_energies[None, :]
            - unconverged_eigenvalues[:, None]
        )
        sign = mx.where(denominator < 0.0, -1.0, 1.0)
        safe = sign * mx.maximum(
            mx.abs(denominator),
            request.config.preconditioner_floor,
        )
        raw_corrections = -unconverged_residuals / safe
        return raw_corrections, int(unconverged.size)

    @staticmethod
    def _prepare_correction(
        progress: _DavidsonLaneProgress,
        ritz_pair: _DavidsonRitzPair,
    ) -> _DavidsonPendingAction | None:
        request = progress.request
        paired = progress.paired
        if paired is None:
            msg = "Davidson lane has no paired V/HV state"
            raise RuntimeError(msg)
        raw_corrections, unconverged_count = _DavidsonEngine._raw_corrections(
            progress,
            ritz_pair,
        )

        if paired.vector_count + unconverged_count > request.config.max_subspace_size:
            progress.restart_count += 1
            # The Ritz vectors and their H(V) values were formed together from
            # an orthonormal paired state by a unitary projected eigensolve.
            # Re-orthogonalizing them here changed both through an avoidable
            # complex64 round trip. Rebase directly onto that certified pair.
            paired = _PairedDavidsonState.initialize(
                ritz_pair.vectors,
                ritz_pair.applied,
                progress.token,
            )
            progress.paired = paired
            if request.config.max_subspace_size - paired.vector_count <= 0:
                if progress.direct_validated:
                    progress.done = True
                    return None
                return _DavidsonEngine._prepare_direct_validation(
                    progress,
                    ritz_pair,
                    terminal=True,
                )

        # A restart changes only the retained basis, not the Ritz residuals
        # from which this correction block was built. Orthogonalize those raw
        # corrections once against the authoritative current basis. The old
        # flow first processed them against the discarded basis and then
        # processed the resulting block again after every restart.
        with observed_phase(
            request.observer,
            "orthogonalization",
            synchronize=False,
        ):
            append_rank = request.rank_policy.orthonormalize(
                mx.concatenate([paired.vectors.values, raw_corrections], axis=0),
                locked_count=paired.vector_count,
                required_count=paired.vector_count,
                max_count=request.config.max_subspace_size,
                single_pass_tolerance=request.rank_policy.single_pass_tolerance(
                    residual_tolerance=request.config.tolerance,
                    vector_count=paired.vector_count + unconverged_count,
                ),
            )
        add_observed_work(
            request.observer,
            {"orthogonalization_vectors": unconverged_count},
        )
        correction_values = append_rank.values[paired.vector_count :]
        correction_count = int(correction_values.shape[0])
        progress.correction_width = correction_count
        if correction_count == 0:
            if progress.direct_validated:
                progress.done = True
                return None
            return _DavidsonEngine._prepare_direct_validation(
                progress,
                ritz_pair,
                terminal=True,
            )

        pending = _DavidsonPendingAction(
            purpose="correction",
            vectors=request.operator.basis._state_from_compact(correction_values),
            reused_width=paired.vector_count,
        )
        progress.pending_action = pending
        return pending

    @staticmethod
    def _batched_ritz_pairs(
        progresses: Sequence[_DavidsonLaneProgress],
    ) -> tuple[dict[str, _DavidsonRitzPair], dict[str, Exception]]:
        """Build equal-width lane Ritz pairs with one LAPACK bridge per group."""

        grouped: dict[tuple[int, int], list[_DavidsonLaneProgress]] = defaultdict(list)
        failures: dict[str, Exception] = {}
        pairs: dict[str, _DavidsonRitzPair] = {}
        for progress in progresses:
            paired = progress.paired
            if paired is None:
                failures[progress.request.lane_id] = RuntimeError(
                    "Davidson lane has no paired V/HV state"
                )
                continue
            grouped[(id(progress.request.observer), paired.vector_count)].append(progress)

        for compatible in grouped.values():
            observer = compatible[0].request.observer
            try:
                with observed_phase(
                    observer,
                    "rayleigh_ritz",
                    synchronize=False,
                ):
                    paired_states = tuple(
                        progress.paired for progress in compatible if progress.paired is not None
                    )
                    with observed_phase(
                        observer,
                        "cpu_small_solve",
                        synchronize=False,
                    ):
                        eigensystems = _projected_eigh_batch(
                            tuple(state.projected for state in paired_states),
                            observer=observer,
                        )
                    candidates = []
                    for progress, state, (values, eigenvectors) in zip(
                        compatible,
                        paired_states,
                        eigensystems,
                        strict=True,
                    ):
                        candidates.append(
                            (
                                progress,
                                _ritz_candidate_from_projected_eigensystem(
                                    state,
                                    progress.request.n_bands,
                                    values,
                                    eigenvectors,
                                ),
                            )
                        )
                    _materialize(
                        observer,
                        *(
                            value
                            for _progress, candidate in candidates
                            for value in (candidate.max_residual, candidate.finite)
                        ),
                    )
                    for progress, candidate in candidates:
                        pairs[progress.request.lane_id] = _seal_ritz_candidate(candidate)
            except Exception:
                # Preserve lane-local failure isolation. The single-lane path is
                # only a recovery path for malformed or numerically failed data.
                for progress in compatible:
                    try:
                        paired = progress.paired
                        if paired is None:
                            msg = "Davidson lane has no paired V/HV state"
                            raise RuntimeError(msg)
                        with observed_phase(
                            progress.request.observer,
                            "rayleigh_ritz",
                            synchronize=False,
                        ):
                            pairs[progress.request.lane_id] = _ritz_pair(
                                paired,
                                progress.request.n_bands,
                            )
                    except Exception as error:
                        failures[progress.request.lane_id] = _detached_failure(error)
        return pairs, failures

    @staticmethod
    def _advance_lane(
        progress: _DavidsonLaneProgress,
        *,
        ritz_pair: _DavidsonRitzPair | None = None,
    ) -> _DavidsonPendingAction | None:
        request = progress.request
        paired = progress.paired
        if paired is None:
            msg = "Davidson lane has no paired V/HV state"
            raise RuntimeError(msg)
        if progress.pending_action is not None:
            msg = "Davidson lane cannot advance with an unconsumed action"
            raise RuntimeError(msg)
        progress.iteration_count += 1
        progress.direct_validated = False
        if ritz_pair is None:
            with observed_phase(
                request.observer,
                "rayleigh_ritz",
                synchronize=False,
            ):
                ritz_pair = _ritz_pair(paired, request.n_bands)
        progress.ritz_pair = ritz_pair
        if (
            ritz_pair.max_residual <= request.config.tolerance
            or progress.iteration_count >= request.config.max_iterations
        ):
            if not request.require_direct_validation:
                progress.converged = ritz_pair.max_residual <= request.config.tolerance
                progress.done = True
                _DavidsonEngine._emit_iteration(
                    progress,
                    ritz_pair,
                    residual_source="paired_subspace",
                )
                return None
            return _DavidsonEngine._prepare_direct_validation(
                progress,
                ritz_pair,
                terminal=(progress.iteration_count >= request.config.max_iterations),
            )
        pending = _DavidsonEngine._prepare_correction(progress, ritz_pair)
        if pending is not None and pending.purpose == "correction":
            _DavidsonEngine._emit_iteration(
                progress,
                ritz_pair,
                residual_source="paired_subspace",
            )
        return pending

    @staticmethod
    def _batched_direct_pairs(
        tickets: Sequence[_DavidsonApplicationTicket],
        scheduled: _DavidsonScheduleResult,
        progress_by_lane: Mapping[str, _DavidsonLaneProgress],
    ) -> tuple[dict[str, _DavidsonRitzPair], dict[str, Exception]]:
        """Materialize scheduled direct residuals through one MLX boundary."""

        candidates: list[tuple[str, _DavidsonRitzCandidate]] = []
        failures: dict[str, Exception] = {}
        for ticket in tickets:
            lane_id = ticket.lane_id
            if ticket.purpose != "direct_validation" or lane_id in scheduled.failures:
                continue
            try:
                progress = progress_by_lane[lane_id]
                pending = progress.pending_action
                if pending is None or pending.ritz_pair is None:
                    msg = "Davidson direct validation lost its Ritz state"
                    raise RuntimeError(msg)
                candidates.append(
                    (
                        lane_id,
                        _ritz_candidate_with_direct_action(
                            pending.ritz_pair,
                            scheduled.action_for(lane_id),
                        ),
                    )
                )
            except Exception as error:
                failures[lane_id] = _detached_failure(error)
        if candidates:
            try:
                observer = tickets[0].observer if tickets else None
                _materialize(
                    observer,
                    *(
                        value
                        for _lane_id, candidate in candidates
                        for value in (candidate.max_residual, candidate.finite)
                    ),
                )
            except Exception as error:
                failure = _detached_failure(error)
                failures.update(
                    {
                        lane_id: failure
                        for lane_id, _candidate in candidates
                        if lane_id not in failures
                    }
                )
        pairs: dict[str, _DavidsonRitzPair] = {}
        for lane_id, candidate in candidates:
            if lane_id in failures:
                continue
            try:
                pairs[lane_id] = _seal_ritz_candidate(candidate)
            except Exception as error:
                failures[lane_id] = _detached_failure(error)
        return pairs, failures

    @staticmethod
    def _consume_action(
        progress: _DavidsonLaneProgress,
        ticket: _DavidsonApplicationTicket,
        applied: _CompactLaneState,
        *,
        direct_pair: _DavidsonRitzPair | None = None,
        orbital_density: mx.array | None = None,
    ) -> _DavidsonPendingAction | None:
        pending = progress.pending_action
        if pending is None or pending.vectors is not ticket.vectors:
            msg = "Davidson scheduled action does not match lane state"
            raise RuntimeError(msg)
        progress.pending_action = None
        if pending.purpose == "correction":
            if orbital_density is not None:
                msg = "Davidson correction unexpectedly captured orbital density"
                raise RuntimeError(msg)
            paired = progress.paired
            if paired is None:
                msg = "Davidson lane lost paired V/HV state"
                raise RuntimeError(msg)
            progress.paired = paired.append(
                pending.vectors,
                applied,
                token=progress.token,
            )
            add_observed_work(
                progress.request.observer,
                {"davidson_hv_reused_vectors": pending.reused_width},
            )
            progress.direct_validated = False
            return None

        candidate = pending.ritz_pair
        if candidate is None:
            msg = "Davidson direct validation lost its Ritz state"
            raise RuntimeError(msg)
        if ticket.capture_orbital_density:
            if orbital_density is None:
                msg = "Davidson direct validation lost its orbital density"
                raise RuntimeError(msg)
            progress.orbital_density = orbital_density
        elif orbital_density is not None:
            msg = "Davidson direct validation captured an unrequested orbital density"
            raise RuntimeError(msg)
        if direct_pair is None:
            direct_pair = _ritz_pair_with_direct_action(candidate, applied)
        elif direct_pair.applied is not applied:
            msg = "Davidson batched direct residual does not match its scheduled action"
            raise RuntimeError(msg)
        progress.ritz_pair = direct_pair
        progress.direct_validated = True
        progress.converged = direct_pair.max_residual <= progress.request.config.tolerance
        _DavidsonEngine._emit_iteration(
            progress,
            direct_pair,
            residual_source="direct_operator",
        )
        if progress.converged or pending.terminal:
            progress.done = True
            return None

        progress.orbital_density = None
        progress.paired = _PairedDavidsonState.initialize(
            candidate.vectors,
            applied,
            progress.token,
        )
        return _DavidsonEngine._prepare_correction(progress, direct_pair)

    @staticmethod
    def _finalize_lane(
        progress: _DavidsonLaneProgress,
    ) -> PeriodicEigenResult:
        request = progress.request
        paired = progress.paired
        ritz_pair = progress.ritz_pair
        if paired is None or ritz_pair is None:
            msg = "Davidson solver produced no Ritz state"
            raise RuntimeError(msg)
        if progress.pending_action is not None or (
            request.require_direct_validation and not progress.direct_validated
        ):
            msg = "Davidson result was not sealed by a direct residual"
            raise RuntimeError(msg)
        if request.capture_orbital_density and progress.orbital_density is None:
            msg = "Davidson result has no final-orbital density"
            raise RuntimeError(msg)
        orthonormality = request.rank_policy.validate(
            ritz_pair.vectors.values,
            required_count=request.n_bands,
        )
        final_max_residual = ritz_pair.max_residual
        if progress.converged != (final_max_residual <= request.config.tolerance):
            msg = "Davidson convergence disagrees with its final residual"
            raise RuntimeError(msg)
        return PeriodicEigenResult._from_compact(
            eigenvalues=ritz_pair.eigenvalues,
            compact_coefficients=ritz_pair.vectors,
            basis=request.operator.basis,
            residuals=ritz_pair.residuals,
            orthonormality_error=orthonormality,
            iterations=progress.iteration_count,
            converged=progress.converged,
            subspace_size=paired.vector_count,
            restart_count=progress.restart_count,
        )

    def solve(
        self,
        requests: Sequence[_DavidsonLaneRequest],
    ) -> _DavidsonEngineResult:
        self._reset_solve(requests)
        failures: dict[str, Exception] = {}
        progress_by_lane = self._initialize_lanes(requests, failures)
        while active := self._active_lanes(progress_by_lane, failures):
            pending_tickets = self._prepare_round(active, failures)
            self._consume_pending_waves(
                pending_tickets,
                progress_by_lane,
                failures,
            )
        results = self._finalize_lanes(progress_by_lane, failures)
        orbital_densities = {
            lane_id: progress.orbital_density
            for lane_id, progress in progress_by_lane.items()
            if lane_id in results and progress.orbital_density is not None
        }
        return _DavidsonEngineResult(
            results=results,
            orbital_densities=orbital_densities,
            failures=failures,
            ready_rounds=tuple(self._ready_rounds),
            compatibility_groups=tuple(self._compatibility_groups),
            submission_groups=tuple(self._submission_groups),
            scheduler_calls=self._scheduler_calls,
        )

    def _reset_solve(
        self,
        requests: Sequence[_DavidsonLaneRequest],
    ) -> None:
        self.scheduler.reset()
        self._ready_rounds.clear()
        self._compatibility_groups.clear()
        self._submission_groups.clear()
        self._scheduler_calls = 0
        if not requests:
            msg = "Davidson engine requires at least one lane"
            raise ValueError(msg)
        lane_ids = [request.lane_id for request in requests]
        if len(set(lane_ids)) != len(lane_ids):
            msg = "Davidson engine lane IDs must be unique"
            raise ValueError(msg)

    def _initialize_lanes(
        self,
        requests: Sequence[_DavidsonLaneRequest],
        failures: dict[str, Exception],
    ) -> dict[str, _DavidsonLaneProgress]:
        progress_by_lane: dict[str, _DavidsonLaneProgress] = {}
        initial_tickets: list[_DavidsonApplicationTicket] = []
        for request in requests:
            try:
                progress = self._prepare_lane(request)
                progress_by_lane[request.lane_id] = progress
                initial_tickets.append(self._ticket(progress, progress.initial_vectors))
            except Exception as error:
                failures[request.lane_id] = _detached_failure(error)
        if not initial_tickets:
            return progress_by_lane
        self.scheduler.bind(initial_tickets)
        initial_schedule = self._schedule(initial_tickets)
        for ticket in initial_tickets:
            progress = progress_by_lane[ticket.lane_id]
            try:
                progress.paired = _PairedDavidsonState.initialize(
                    progress.initial_vectors,
                    initial_schedule.action_for(ticket.lane_id),
                    progress.token,
                )
            except Exception as error:
                self._fail_lane(progress, failures, error)
        return progress_by_lane

    @staticmethod
    def _active_lanes(
        progress_by_lane: dict[str, _DavidsonLaneProgress],
        failures: dict[str, Exception],
    ) -> list[_DavidsonLaneProgress]:
        return [
            progress
            for lane_id, progress in progress_by_lane.items()
            if lane_id not in failures and not progress.done
        ]

    def _prepare_round(
        self,
        active: Sequence[_DavidsonLaneProgress],
        failures: dict[str, Exception],
    ) -> list[_DavidsonApplicationTicket]:
        ritz_pairs, ritz_failures = self._batched_ritz_pairs(active)
        self._emit_summarized_rounds(active, ritz_pairs)
        pending_tickets: list[_DavidsonApplicationTicket] = []
        for progress in active:
            lane_id = progress.request.lane_id
            failure = ritz_failures.get(lane_id)
            if failure is not None:
                self._fail_lane(progress, failures, failure)
                continue
            try:
                pending = self._advance_lane(
                    progress,
                    ritz_pair=ritz_pairs[lane_id],
                )
                if pending is not None:
                    pending_tickets.append(self._ticket_for_pending(progress, pending))
            except Exception as error:
                self._fail_lane(progress, failures, error)
        return pending_tickets

    @staticmethod
    def _emit_summarized_rounds(
        active: Sequence[_DavidsonLaneProgress],
        ritz_pairs: dict[str, _DavidsonRitzPair],
    ) -> None:
        summarized_observers = {
            id(progress.request.observer): progress.request.observer
            for progress in active
            if progress.request.observer is not None and not progress.request.observer.detail_events
        }
        for observer in summarized_observers.values():
            summarized_progress = [
                progress
                for progress in active
                if progress.request.observer is observer and progress.request.lane_id in ritz_pairs
            ]
            if not summarized_progress:
                continue
            observer.emit(
                "davidson_round",
                iteration_min=min(progress.iteration_count + 1 for progress in summarized_progress),
                iteration_max=max(progress.iteration_count + 1 for progress in summarized_progress),
                active_lane_count=len(summarized_progress),
                maximum_subspace_size=max(
                    progress.paired.vector_count
                    for progress in summarized_progress
                    if progress.paired is not None
                ),
                maximum_residual=max(
                    ritz_pairs[progress.request.lane_id].max_residual
                    for progress in summarized_progress
                ),
                converged_candidate_count=sum(
                    ritz_pairs[progress.request.lane_id].max_residual
                    <= progress.request.config.tolerance
                    for progress in summarized_progress
                ),
                residual_source="paired_subspace",
            )

    @staticmethod
    def _ticket_for_pending(
        progress: _DavidsonLaneProgress,
        pending: _DavidsonPendingAction,
    ) -> _DavidsonApplicationTicket:
        return _DavidsonEngine._ticket(
            progress,
            pending.vectors,
            purpose=pending.purpose,
        )

    def _consume_pending_waves(
        self,
        pending_tickets: list[_DavidsonApplicationTicket],
        progress_by_lane: dict[str, _DavidsonLaneProgress],
        failures: dict[str, Exception],
    ) -> None:
        current_tickets = pending_tickets
        while current_tickets:
            scheduled = self._schedule(current_tickets)
            direct_pairs, direct_failures = self._batched_direct_pairs(
                current_tickets,
                scheduled,
                progress_by_lane,
            )
            current_tickets = self._consume_scheduled_wave(
                current_tickets,
                scheduled,
                direct_pairs,
                direct_failures,
                progress_by_lane,
                failures,
            )

    def _consume_scheduled_wave(
        self,
        tickets: Sequence[_DavidsonApplicationTicket],
        scheduled: _DavidsonScheduleResult,
        direct_pairs: dict[str, _DavidsonRitzPair],
        direct_failures: dict[str, Exception],
        progress_by_lane: dict[str, _DavidsonLaneProgress],
        failures: dict[str, Exception],
    ) -> list[_DavidsonApplicationTicket]:
        followup_tickets: list[_DavidsonApplicationTicket] = []
        for ticket in tickets:
            lane_id = ticket.lane_id
            failure = scheduled.failures.get(lane_id) or direct_failures.get(lane_id)
            if failure is not None:
                self._fail_lane(
                    progress_by_lane[lane_id],
                    failures,
                    failure,
                )
                continue
            progress = progress_by_lane[lane_id]
            try:
                followup = self._consume_action(
                    progress,
                    ticket,
                    scheduled.action_for(lane_id),
                    direct_pair=direct_pairs.get(lane_id),
                    orbital_density=(
                        scheduled.orbital_density_for(lane_id)
                        if ticket.capture_orbital_density
                        else None
                    ),
                )
                if followup is not None:
                    followup_tickets.append(self._ticket_for_pending(progress, followup))
            except Exception as error:
                self._fail_lane(progress, failures, error)
        return followup_tickets

    def _finalize_lanes(
        self,
        progress_by_lane: dict[str, _DavidsonLaneProgress],
        failures: dict[str, Exception],
    ) -> dict[str, PeriodicEigenResult]:
        results: dict[str, PeriodicEigenResult] = {}
        for lane_id, progress in progress_by_lane.items():
            if lane_id in failures:
                continue
            try:
                results[lane_id] = self._finalize_lane(progress)
            except Exception as error:
                self._fail_lane(progress, failures, error)
        return results


def solve_periodic_eigenproblem(
    operator: PeriodicKohnShamOperator,
    *,
    n_bands: int,
    config: PeriodicDavidsonConfig | None = None,
    initial_coefficients: mx.array | None = None,
    observer: RuntimeObserver | None = None,
) -> PeriodicEigenResult:
    """Solve the lowest periodic eigenpairs with block Davidson/Rayleigh-Ritz.

    Args:
        operator: Fixed-density periodic Kohn-Sham operator.
        n_bands: Number of lowest states to return.
        config: Davidson controls. Defaults to `PeriodicDavidsonConfig`.
        initial_coefficients: Optional initial orbital stack. Defaults to the
            lowest kinetic plane waves.
        observer: Optional progress and work observer. Defaults to the
            observer carried by ``operator``.

    Returns:
        Converged or exhausted result sealed by direct final residuals.
    """

    runtime_observer = operator.observer if observer is None else observer
    solver_config = PeriodicDavidsonConfig() if config is None else config
    basis = operator.basis
    if type(n_bands) is not int or n_bands <= 0 or n_bands > basis.active_count:
        msg = "n_bands must be a positive non-bool integer no larger than the active basis size"
        raise ValueError(msg)
    trial = _initial_trial(basis, n_bands, initial_coefficients)
    lane_id = basis._layout.lane_id
    request = _DavidsonLaneRequest(
        lane_id=lane_id,
        operator=operator,
        n_bands=n_bands,
        config=solver_config,
        trial=trial,
        observer=runtime_observer,
        trial_is_orthonormal=initial_coefficients is None,
    )
    engine = _DavidsonEngine(scheduler=_DavidsonScheduler())
    return engine.solve([request]).result_for(lane_id)
