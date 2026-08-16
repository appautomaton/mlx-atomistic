"""Block-Davidson eigensolver and compact scheduling for periodic DFT."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

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
from mlx_atomistic.dft._periodic_execution import (
    _detached_failure,
    _materialize,
    _to_numpy,
)
from mlx_atomistic.dft._periodic_hamiltonian import PeriodicKohnShamOperator
from mlx_atomistic.dft._periodic_models import PeriodicDavidsonConfig, PeriodicEigenResult
from mlx_atomistic.dft._periodic_orthonormalization import (
    _DAVIDSON_RANK_POLICY,
    _Complex64RankPolicy,
    _RankResult,
)
from mlx_atomistic.dft._runtime_observer import (
    RuntimeObserver,
    add_observed_work,
    observed_phase,
)
from mlx_atomistic.dft.periodic_gth import PeriodicGTHNonlocalOperator
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


def _subspace_matrix(basis_vectors: mx.array, applied: mx.array) -> mx.array:
    matrix = mx.conjugate(basis_vectors) @ mx.transpose(applied)
    return 0.5 * (matrix + mx.conjugate(mx.transpose(matrix)))


def _hamiltonian_context(
    operator: PeriodicKohnShamOperator,
    config: PeriodicDavidsonConfig,
    n_bands: int,
    rank_policy: _Complex64RankPolicy,
) -> tuple[object, ...]:
    nonlocal_context = (
        None
        if operator.nonlocal_operator is None
        else (
            id(operator.nonlocal_operator),
            operator.nonlocal_operator._context_identity,
        )
    )
    potential = operator._effective_local_potential
    return (
        id(operator),
        id(potential),
        tuple(int(value) for value in potential.shape),
        str(potential.dtype),
        operator.basis.basis_fingerprint,
        operator.basis.order_fingerprint,
        operator.basis._layout.lane_id,
        operator.basis.reciprocal_grid.fingerprint,
        tuple(float(value) for value in operator.basis.kpoint_cartesian),
        nonlocal_context,
        "complex64-float32",
        str(mx.default_device()),
        config.max_iterations,
        config.tolerance,
        config.max_subspace_size,
        config.preconditioner_floor,
        n_bands,
        "complex64-adaptive-choleskyqr-cgs2-mgs2-rank-v6",
        rank_policy.relative_tolerance,
    )


@dataclass(frozen=True, eq=False)
class _FixedHamiltonianToken:
    """Solve-local identity that prevents paired H(V) from crossing contexts."""

    context: tuple[object, ...]
    nonce: object = field(default_factory=object, repr=False)

    @classmethod
    def create(
        cls,
        operator: PeriodicKohnShamOperator,
        config: PeriodicDavidsonConfig,
        n_bands: int,
        rank_policy: _Complex64RankPolicy = _DAVIDSON_RANK_POLICY,
    ) -> _FixedHamiltonianToken:
        return cls(_hamiltonian_context(operator, config, n_bands, rank_policy))

    def validate(
        self,
        operator: PeriodicKohnShamOperator,
        config: PeriodicDavidsonConfig,
        n_bands: int,
        rank_policy: _Complex64RankPolicy = _DAVIDSON_RANK_POLICY,
    ) -> None:
        if self.context != _hamiltonian_context(
            operator,
            config,
            n_bands,
            rank_policy,
        ):
            msg = "Davidson H(V) token does not match the fixed Hamiltonian"
            raise ValueError(msg)


@dataclass(frozen=True)
class _CompactBatchCapacity:
    """Solve-local physical shape shared by compatible logical submissions."""

    lanes: int
    vectors: int
    active: int


def _finite_lane_capacity(lane_count: int, maximum: int) -> int:
    """Return the smallest bounded power-of-two lane capacity."""

    if type(lane_count) is not int or type(maximum) is not int:
        msg = "finite lane capacities require non-bool integers"
        raise ValueError(msg)
    if lane_count <= 0 or maximum <= 0 or lane_count > maximum:
        msg = "finite lane capacity must be positive and no larger than its maximum"
        raise ValueError(msg)
    return min(1 << (lane_count - 1).bit_length(), maximum)


def _finite_vector_capacity(vector_count: int, maximum: int) -> int:
    """Return a 4/8/16-style vector bucket bounded by the solve maximum."""

    if type(vector_count) is not int or type(maximum) is not int:
        msg = "finite vector capacities require non-bool integers"
        raise ValueError(msg)
    if vector_count <= 0 or maximum <= 0 or vector_count > maximum:
        msg = "finite vector capacity must be positive and no larger than its maximum"
        raise ValueError(msg)
    capacity = min(4, maximum)
    while capacity < vector_count:
        capacity = min(2 * capacity, maximum)
    return capacity


@dataclass(frozen=True)
class _CompactSubmission:
    indices: tuple[int, ...]
    capacity: _CompactBatchCapacity | None = None


@dataclass(frozen=True)
class _CompactSubmissionPlan:
    submissions: tuple[_CompactSubmission, ...]
    failures: dict[int, Exception]
    compatibility_groups: tuple[tuple[int, ...], ...]


class _CompactSubmissionPlanner:
    """Build deterministic compact submissions under one bounded policy."""

    def __init__(
        self,
        states: Sequence[_CompactLaneState],
        *,
        policy: _CompactBatchPolicy,
        batch_byte_estimator: (
            Callable[[tuple[int, ...], _CompactBatch], int] | None
        ),
        capacity: _CompactBatchCapacity | None,
    ) -> None:
        self.states = tuple(states)
        self.policy = policy
        self.batch_byte_estimator = batch_byte_estimator
        self.capacity = capacity
        self.capacity_prototype: _CompactBatch | None = None
        self._validate_capacity()

    def _validate_capacity(self) -> None:
        capacity = self.capacity
        if capacity is None:
            return
        if (
            type(capacity.lanes) is not int
            or type(capacity.vectors) is not int
            or type(capacity.active) is not int
            or capacity.lanes <= 0
            or capacity.vectors <= 0
            or capacity.active <= 0
            or capacity.lanes > self.policy.batch_cap
        ):
            msg = "compact capacity must contain positive bounded integers"
            raise ValueError(msg)

    def plan(self) -> _CompactSubmissionPlan:
        submissions: list[_CompactSubmission] = []
        failures: dict[int, Exception] = {}
        compatibility_groups: list[tuple[int, ...]] = []
        for compatible_indices in self._compatibility_groups():
            compatibility_groups.append(compatible_indices)
            planned, rejected = self._plan_group(compatible_indices)
            submissions.extend(planned)
            failures.update(rejected)
        return _CompactSubmissionPlan(
            submissions=tuple(submissions),
            failures=failures,
            compatibility_groups=tuple(compatibility_groups),
        )

    def _compatibility_groups(self) -> tuple[tuple[int, ...], ...]:
        grouped: dict[tuple[object, ...], list[int]] = defaultdict(list)
        for index, state in enumerate(self.states):
            grouped[
                (
                    id(state.layout.reciprocal),
                    state.layout.grid_shape,
                    (
                        state.vector_count
                        if self.capacity is None
                        else self.capacity.vectors
                    ),
                )
            ].append(index)
        return tuple(tuple(indices) for indices in grouped.values())

    def _plan_group(
        self,
        compatible_indices: tuple[int, ...],
    ) -> tuple[list[_CompactSubmission], dict[int, Exception]]:
        ordered = sorted(
            compatible_indices,
            key=lambda index: (
                self.states[index].layout.active_count,
                index,
            ),
        )
        submissions: list[_CompactSubmission] = []
        failures: dict[int, Exception] = {}
        current: list[int] = []
        current_batch: _CompactBatch | None = None
        for index in ordered:
            if len(current) == self.policy.batch_cap:
                self._append_submission(
                    submissions,
                    current,
                    current_batch,
                    "active",
                )
                current = []
                current_batch = None
            current, current_batch = self._extend_group(
                current,
                current_batch,
                index,
                submissions,
                failures,
            )
        if current:
            self._append_submission(
                submissions,
                current,
                current_batch,
                "final",
            )
        return submissions, failures

    def _extend_group(
        self,
        current: list[int],
        current_batch: _CompactBatch | None,
        index: int,
        submissions: list[_CompactSubmission],
        failures: dict[int, Exception],
    ) -> tuple[list[int], _CompactBatch | None]:
        candidate = [*current, index]
        try:
            candidate_batch = self._build_candidate(candidate)
        except ValueError:
            if current:
                self._append_submission(
                    submissions,
                    current,
                    current_batch,
                    "bounded",
                )
            try:
                singleton_batch = self._build_candidate((index,))
            except ValueError as error:
                failures[index] = _detached_failure(error)
                return [], None
            return [index], singleton_batch
        return candidate, candidate_batch

    def _append_submission(
        self,
        submissions: list[_CompactSubmission],
        indices: Sequence[int],
        batch: _CompactBatch | None,
        stage: str,
    ) -> None:
        if batch is None:
            msg = f"compact submission planner lost its {stage} batch"
            raise RuntimeError(msg)
        submissions.append(
            _CompactSubmission(
                tuple(indices),
                self.capacity,
            )
        )

    def _build_candidate(
        self,
        indices: Sequence[int],
    ) -> _CompactBatch:
        selected = [self.states[index] for index in indices]
        capacity = self.capacity
        if capacity is None:
            candidate_batch = _CompactBatch.from_states(
                selected,
                policy=self.policy,
            )
        else:
            self._validate_selected_capacity(selected)
            if self.capacity_prototype is None:
                self.capacity_prototype = _CompactBatch.from_states(
                    selected[:1],
                    policy=self.policy,
                    lane_capacity=capacity.lanes,
                    vector_capacity=capacity.vectors,
                    active_capacity=capacity.active,
                )
            candidate_batch = self.capacity_prototype
        estimated_bytes = (
            candidate_batch.estimated_transient_bytes
            if self.batch_byte_estimator is None
            else self.batch_byte_estimator(
                tuple(indices),
                candidate_batch,
            )
        )
        if estimated_bytes > self.policy.max_transient_bytes:
            msg = "compact batch exceeds the complete transient byte budget"
            raise ValueError(msg)
        return candidate_batch

    def _validate_selected_capacity(
        self,
        selected: Sequence[_CompactLaneState],
    ) -> None:
        capacity = self.capacity
        if capacity is None:
            raise RuntimeError("compact capacity validation requires a capacity")
        if len(selected) > capacity.lanes:
            msg = "compact candidate exceeds its stable lane capacity"
            raise ValueError(msg)
        if any(
            state.vector_count > capacity.vectors
            or state.layout.active_count > capacity.active
            or (capacity.active - state.layout.active_count) / capacity.active
            > self.policy.max_padding_fraction
            for state in selected
        ):
            msg = "compact candidate exceeds its stable shape policy"
            raise ValueError(msg)


def _plan_compact_submissions(
    states: Sequence[_CompactLaneState],
    *,
    policy: _CompactBatchPolicy,
    batch_byte_estimator: Callable[[tuple[int, ...], _CompactBatch], int] | None = None,
    capacity: _CompactBatchCapacity | None = None,
) -> _CompactSubmissionPlan:
    """Build deterministic active-count buckets within hard batch bounds."""

    return _CompactSubmissionPlanner(
        states,
        policy=policy,
        batch_byte_estimator=batch_byte_estimator,
        capacity=capacity,
    ).plan()


def _stable_compact_capacity_groups(
    states: Sequence[_CompactLaneState],
    indices: Sequence[int],
    *,
    lane_capacity: int,
    vector_capacity: int,
    max_padding_fraction: float,
) -> tuple[tuple[tuple[int, ...], _CompactBatchCapacity], ...]:
    """Partition compatible lanes into stable active-width capacity groups."""

    ordered = sorted(indices, key=lambda index: states[index].layout.active_count)
    groups: list[tuple[tuple[int, ...], _CompactBatchCapacity]] = []
    current: list[int] = []
    for index in ordered:
        candidate = [*current, index]
        smallest = states[candidate[0]].layout.active_count
        largest = states[candidate[-1]].layout.active_count
        if current and (largest - smallest) / largest > max_padding_fraction:
            active_capacity = states[current[-1]].layout.active_count
            groups.append(
                (
                    tuple(current),
                    _CompactBatchCapacity(
                        lane_capacity,
                        vector_capacity,
                        active_capacity,
                    ),
                )
            )
            current = [index]
        else:
            current = candidate
    if current:
        groups.append(
            (
                tuple(current),
                _CompactBatchCapacity(
                    lane_capacity,
                    vector_capacity,
                    states[current[-1]].layout.active_count,
                ),
            )
        )
    return tuple(groups)


@dataclass(frozen=True)
class _DavidsonApplicationTicket:
    """One lane's coefficient block awaiting a scheduled H application."""

    lane_id: str
    operator: PeriodicKohnShamOperator
    config: PeriodicDavidsonConfig
    n_bands: int
    rank_policy: _Complex64RankPolicy
    token: _FixedHamiltonianToken
    vectors: _CompactLaneState
    observer: RuntimeObserver | None
    purpose: str = "basis"
    capture_orbital_density: bool = False


_DavidsonSubmissionCallback = Callable[
    [
        str,
        int,
        tuple[_DavidsonApplicationTicket, ...],
        _CompactBatch,
        dict[str, Exception],
    ],
    None,
]


@dataclass(frozen=True)
class _DavidsonScheduleResult:
    """Per-lane actions plus compatible and actually submitted groups."""

    actions: dict[str, _CompactLaneState]
    orbital_densities: dict[str, mx.array]
    failures: dict[str, Exception]
    groups: tuple[tuple[str, ...], ...]
    compatibility_groups: tuple[tuple[str, ...], ...]

    @property
    def submission_count(self) -> int:
        return len(self.groups)

    def action_for(self, lane_id: str) -> _CompactLaneState:
        failure = self.failures.get(lane_id)
        if failure is not None:
            raise _detached_failure(failure) from None
        try:
            return self.actions[lane_id]
        except KeyError as error:
            msg = f"Davidson scheduler has no result for lane {lane_id!r}"
            raise ValueError(msg) from error

    def orbital_density_for(self, lane_id: str) -> mx.array:
        """Return the captured final-orbital density for one lane."""

        failure = self.failures.get(lane_id)
        if failure is not None:
            raise _detached_failure(failure) from None
        try:
            return self.orbital_densities[lane_id]
        except KeyError as error:
            msg = f"Davidson scheduler has no orbital density for lane {lane_id!r}"
            raise ValueError(msg) from error


class _DavidsonScheduler:
    """Submit compatible ragged tickets under an explicit batch-cap policy."""

    def __init__(
        self,
        *,
        policy: _CompactBatchPolicy = _DEFAULT_COMPACT_BATCH_POLICY,
        submission_callback: _DavidsonSubmissionCallback | None = None,
    ) -> None:
        self._policy = policy
        self._submission_callback = submission_callback
        self._submission_index = 0
        self._capacity_by_lane: dict[str, _CompactBatchCapacity] = {}

    @property
    def batch_cap(self) -> int:
        return self._policy.batch_cap

    @property
    def max_padding_fraction(self) -> float:
        return self._policy.max_padding_fraction

    @property
    def shape_policy(self) -> str:
        return self._policy.shape_policy

    def reset(self) -> None:
        """Reset solve-local submission numbering."""

        self._submission_index = 0
        self._capacity_by_lane.clear()

    @staticmethod
    def _observer(ticket: _DavidsonApplicationTicket) -> RuntimeObserver | None:
        return ticket.operator.observer if ticket.observer is None else ticket.observer

    @staticmethod
    def _physical_group_key(
        ticket: _DavidsonApplicationTicket,
    ) -> tuple[object, ...]:
        layout = ticket.vectors.layout
        nonlocal_operator = ticket.operator.nonlocal_operator
        if nonlocal_operator is None:
            nonlocal_context: object = None
        elif isinstance(nonlocal_operator, PeriodicGTHNonlocalOperator):
            nonlocal_context = ("gth", nonlocal_operator._context_identity)
        else:
            nonlocal_context = ("custom", id(nonlocal_operator))
        return (
            id(layout.reciprocal),
            layout.grid_shape,
            id(_DavidsonScheduler._observer(ticket)),
            nonlocal_context,
        )

    def _group_key(
        self,
        ticket: _DavidsonApplicationTicket,
    ) -> tuple[object, ...]:
        capacity = self._capacity_by_lane.get(ticket.lane_id)
        if capacity is None:
            shape: object = ("dynamic", ticket.vectors.vector_count)
        elif self.shape_policy == "finite-buckets":
            shape = (
                "finite-buckets",
                _finite_vector_capacity(ticket.vectors.vector_count, capacity.vectors),
                capacity.active,
            )
        else:
            shape = (
                "stable",
                capacity.lanes,
                capacity.vectors,
                capacity.active,
            )
        return (*self._physical_group_key(ticket), shape)

    def bind(self, tickets: Sequence[_DavidsonApplicationTicket]) -> None:
        """Freeze compatible submission capacities for one Davidson solve."""

        self._capacity_by_lane.clear()
        grouped: dict[
            tuple[object, ...],
            list[_DavidsonApplicationTicket],
        ] = defaultdict(list)
        for ticket in tickets:
            grouped[self._physical_group_key(ticket)].append(ticket)
        for compatible in grouped.values():
            states = [ticket.vectors for ticket in compatible]
            vector_capacity = max(
                max(ticket.n_bands, ticket.vectors.vector_count) for ticket in compatible
            )
            for indices, capacity in _stable_compact_capacity_groups(
                states,
                range(len(states)),
                lane_capacity=self.batch_cap,
                vector_capacity=vector_capacity,
                max_padding_fraction=self.max_padding_fraction,
            ):
                for index in indices:
                    self._capacity_by_lane[compatible[index].lane_id] = capacity

    def apply(
        self,
        tickets: Sequence[_DavidsonApplicationTicket],
    ) -> _DavidsonScheduleResult:
        ready, failures = self._validate_tickets(tickets)
        actions: dict[str, _CompactLaneState] = {}
        orbital_densities: dict[str, mx.array] = {}
        groups: list[tuple[str, ...]] = []
        compatibility_groups: list[tuple[str, ...]] = []
        for compatible in self._compatible_batches(ready):
            compatibility_groups.append(
                tuple(ticket.lane_id for ticket in compatible)
            )
            plan = self._plan_compatible_group(compatible)
            for lane_index, error in plan.failures.items():
                failures[compatible[lane_index].lane_id] = error
            for planned in plan.submissions:
                submission = tuple(
                    compatible[index] for index in planned.indices
                )
                prepared_batch = self._prepare_submission(
                    submission,
                    planned,
                )
                lane_ids = tuple(ticket.lane_id for ticket in submission)
                groups.append(lane_ids)
                (
                    submission_index,
                    batch_actions,
                    batch_densities,
                    submission_failures,
                ) = self._submit(
                    submission,
                    prepared_batch,
                )
                actions.update(batch_actions)
                orbital_densities.update(batch_densities)
                failures.update(submission_failures)
                self._notify_submission(
                    "completed",
                    submission_index,
                    submission,
                    prepared_batch,
                    submission_failures,
                )
        return _DavidsonScheduleResult(
            actions=actions,
            orbital_densities=orbital_densities,
            failures=failures,
            groups=tuple(groups),
            compatibility_groups=tuple(compatibility_groups),
        )

    def _validate_tickets(
        self,
        tickets: Sequence[_DavidsonApplicationTicket],
    ) -> tuple[list[_DavidsonApplicationTicket], dict[str, Exception]]:
        if not tickets:
            msg = "Davidson scheduler requires at least one application ticket"
            raise ValueError(msg)
        seen: set[str] = set()
        failures: dict[str, Exception] = {}
        validated: list[tuple[_DavidsonApplicationTicket, mx.array]] = []
        for ticket in tickets:
            if ticket.lane_id in seen:
                msg = f"duplicate Davidson scheduler lane: {ticket.lane_id!r}"
                raise ValueError(msg)
            seen.add(ticket.lane_id)
            try:
                finite = self._validate_ticket(ticket)
            except Exception as error:
                failures[ticket.lane_id] = _detached_failure(error)
            else:
                validated.append((ticket, finite))
        ready: list[_DavidsonApplicationTicket] = []
        if not validated:
            return ready, failures
        try:
            mx.eval(*(finite for _, finite in validated))
        except Exception as error:
            failure = _detached_failure(error)
            failures.update(
                {ticket.lane_id: failure for ticket, _ in validated}
            )
            return ready, failures
        for ticket, finite in validated:
            if bool(finite):
                ready.append(ticket)
            else:
                failures[ticket.lane_id] = ValueError(
                    "Davidson application block must be finite"
                )
        return ready, failures

    @staticmethod
    def _validate_ticket(ticket: _DavidsonApplicationTicket) -> mx.array:
        if ticket.purpose not in {"basis", "direct_validation"}:
            msg = "Davidson application purpose is invalid"
            raise ValueError(msg)
        if type(ticket.capture_orbital_density) is not bool or (
            ticket.capture_orbital_density and ticket.purpose != "direct_validation"
        ):
            msg = "orbital density can only be captured for direct validation"
            raise ValueError(msg)
        ticket.token.validate(
            ticket.operator,
            ticket.config,
            ticket.n_bands,
            ticket.rank_policy,
        )
        ticket.operator.basis._validate_state(ticket.vectors)
        if ticket.vectors.kind != "coefficients":
            msg = "Davidson scheduler accepts coefficient blocks only"
            raise ValueError(msg)
        return mx.all(mx.isfinite(ticket.vectors.values))

    def _compatible_batches(
        self,
        tickets: Sequence[_DavidsonApplicationTicket],
    ) -> list[list[_DavidsonApplicationTicket]]:
        grouped: dict[
            tuple[object, ...],
            list[_DavidsonApplicationTicket],
        ] = defaultdict(list)
        for ticket in tickets:
            grouped[self._group_key(ticket)].append(ticket)
        batches: list[list[_DavidsonApplicationTicket]] = []
        for compatible in grouped.values():
            if self.shape_policy == "finite-buckets":
                batches.extend(
                    compatible[start : start + self.batch_cap]
                    for start in range(
                        0,
                        len(compatible),
                        self.batch_cap,
                    )
                )
            else:
                batches.append(compatible)
        return batches

    def _plan_compatible_group(
        self,
        compatible: list[_DavidsonApplicationTicket],
    ) -> _CompactSubmissionPlan:
        def estimate_submission(
            indices: tuple[int, ...],
            batch: _CompactBatch,
        ) -> int:
            return PeriodicKohnShamOperator._estimated_batch_transient_bytes(
                [compatible[index].operator for index in indices],
                batch,
                captured_density_lanes=sum(
                    compatible[index].capture_orbital_density for index in indices
                ),
            )

        bound_capacity = self._capacity_by_lane.get(compatible[0].lane_id)
        if bound_capacity is not None and any(
            self._capacity_by_lane.get(ticket.lane_id) != bound_capacity
            for ticket in compatible
        ):
            msg = "Davidson stable-shape group has inconsistent capacities"
            raise RuntimeError(msg)
        capacity = bound_capacity
        if capacity is not None and self.shape_policy == "finite-buckets":
            capacity = _CompactBatchCapacity(
                lanes=_finite_lane_capacity(
                    len(compatible),
                    capacity.lanes,
                ),
                vectors=_finite_vector_capacity(
                    compatible[0].vectors.vector_count,
                    capacity.vectors,
                ),
                active=capacity.active,
            )
        return _plan_compact_submissions(
            [ticket.vectors for ticket in compatible],
            policy=self._policy,
            batch_byte_estimator=estimate_submission,
            capacity=capacity,
        )

    def _prepare_submission(
        self,
        submission: tuple[_DavidsonApplicationTicket, ...],
        planned: _CompactSubmission,
    ) -> _CompactBatch:
        prepared_batch = _CompactBatch.from_states(
            [ticket.vectors for ticket in submission],
            policy=self._policy,
            lane_capacity=(
                None if planned.capacity is None else planned.capacity.lanes
            ),
            vector_capacity=(
                None if planned.capacity is None else planned.capacity.vectors
            ),
            active_capacity=(
                None if planned.capacity is None else planned.capacity.active
            ),
        )
        complete_transient_bytes = (
            PeriodicKohnShamOperator._estimated_batch_transient_bytes(
                [ticket.operator for ticket in submission],
                prepared_batch,
                captured_density_lanes=sum(
                    ticket.capture_orbital_density for ticket in submission
                ),
            )
        )
        runtime_observer = self._observer(submission[0])
        if runtime_observer is not None:
            runtime_observer.record_peak_memory(
                "peak_temporary_bytes",
                complete_transient_bytes,
            )
        return prepared_batch

    def _submit(
        self,
        submission: tuple[_DavidsonApplicationTicket, ...],
        prepared_batch: _CompactBatch,
    ) -> tuple[
        int,
        dict[str, _CompactLaneState],
        dict[str, mx.array],
        dict[str, Exception],
    ]:
        submission_index = self._submission_index
        self._submission_index += 1
        self._notify_submission(
            "started",
            submission_index,
            submission,
            prepared_batch,
            {},
        )
        submission_failures: dict[str, Exception] = {}
        accepted_actions: dict[str, _CompactLaneState] = {}
        accepted_densities: dict[str, mx.array] = {}
        try:
            (
                batch_actions,
                batch_densities,
                submission_failures,
            ) = self._apply_submission(
                submission,
                prepared_batch,
            )
            for ticket in submission:
                if (
                    ticket.capture_orbital_density
                    and ticket.lane_id in batch_actions
                    and ticket.lane_id not in batch_densities
                ):
                    msg = "direct validation did not capture its orbital density"
                    raise RuntimeError(msg)
            for ticket in submission:
                applied = batch_actions.get(ticket.lane_id)
                if applied is None:
                    continue
                accepted_actions[ticket.lane_id] = applied
                if ticket.purpose == "basis":
                    add_observed_work(
                        ticket.observer,
                        {
                            "davidson_hv_new_vectors": (
                                ticket.vectors.vector_count
                            )
                        },
                    )
                if ticket.capture_orbital_density:
                    accepted_densities[ticket.lane_id] = batch_densities[ticket.lane_id]
        except Exception as error:
            failure = _detached_failure(error)
            accepted_actions.clear()
            accepted_densities.clear()
            submission_failures.update(
                {ticket.lane_id: failure for ticket in submission}
            )
        return (
            submission_index,
            accepted_actions,
            accepted_densities,
            submission_failures,
        )

    def _apply_submission(
        self,
        submission: tuple[_DavidsonApplicationTicket, ...],
        prepared_batch: _CompactBatch,
    ) -> tuple[
        dict[str, _CompactLaneState],
        dict[str, mx.array],
        dict[str, Exception],
    ]:
        if len(submission) == 1 and not submission[0].capture_orbital_density:
            ticket = submission[0]
            applied = ticket.operator._apply_compact(
                ticket.vectors,
                observer=ticket.observer,
                policy=self._policy,
                prepared_batch=prepared_batch,
            )
            return {ticket.lane_id: applied}, {}, {}
        outcome = PeriodicKohnShamOperator._apply_compact_batch(
            tuple(ticket.operator for ticket in submission),
            tuple(ticket.vectors for ticket in submission),
            observer=self._observer(submission[0]),
            policy=self._policy,
            prepared_batch=prepared_batch,
            capture_orbital_densities=tuple(
                ticket.capture_orbital_density for ticket in submission
            ),
        )
        actions = {
            ticket.lane_id: outcome.actions[index]
            for index, ticket in enumerate(submission)
            if index in outcome.actions
        }
        failures = {
            submission[index].lane_id: error
            for index, error in outcome.failures.items()
        }
        orbital_densities = {
            submission[index].lane_id: density
            for index, density in outcome.orbital_densities.items()
        }
        return actions, orbital_densities, failures

    def _notify_submission(
        self,
        status: str,
        submission_index: int,
        submission: tuple[_DavidsonApplicationTicket, ...],
        prepared_batch: _CompactBatch,
        failures: dict[str, Exception],
    ) -> None:
        if self._submission_callback is not None:
            self._submission_callback(
                status,
                submission_index,
                submission,
                prepared_batch,
                failures,
            )


@dataclass(frozen=True)
class _PairedDavidsonState:
    """Unpadded lane-local V/HV pair and its incremental projection."""

    vectors: _CompactLaneState
    applied: _CompactLaneState
    projected: mx.array
    token: _FixedHamiltonianToken

    def __post_init__(self) -> None:
        _require_layout(self.vectors, self.applied.layout)
        if self.vectors.kind != "coefficients":
            msg = "Davidson V state must contain coefficients"
            raise ValueError(msg)
        if self.applied.kind != "hamiltonian_action":
            msg = "Davidson HV state must contain Hamiltonian actions"
            raise ValueError(msg)
        if self.vectors.vector_count != self.applied.vector_count:
            msg = "Davidson V and HV widths must match"
            raise ValueError(msg)
        width = self.vectors.vector_count
        matrix = mx.array(self.projected).astype(mx.complex64)
        if matrix.shape != (width, width):
            msg = "Davidson projected matrix must match the paired width"
            raise ValueError(msg)
        # Finiteness is checked collectively at the Rayleigh-Ritz boundary.
        # Materializing here would serialize every lane after each append.
        object.__setattr__(self, "projected", matrix)

    @classmethod
    def initialize(
        cls,
        vectors: _CompactLaneState,
        applied: _CompactLaneState,
        token: _FixedHamiltonianToken,
    ) -> _PairedDavidsonState:
        return cls(vectors, applied, _subspace_matrix(vectors.values, applied.values), token)

    @property
    def vector_count(self) -> int:
        return self.vectors.vector_count

    def require_token(self, token: _FixedHamiltonianToken) -> None:
        if token is not self.token:
            msg = "Davidson paired H(V) cannot cross a solve token"
            raise ValueError(msg)

    def append(
        self,
        vectors: _CompactLaneState,
        applied: _CompactLaneState,
        *,
        token: _FixedHamiltonianToken,
    ) -> _PairedDavidsonState:
        self.require_token(token)
        _require_layout(vectors, self.vectors.layout)
        _require_layout(applied, self.vectors.layout)
        if vectors.kind != "coefficients" or applied.kind != "hamiltonian_action":
            msg = "Davidson append requires paired C/H(C) state"
            raise ValueError(msg)
        if vectors.vector_count != applied.vector_count:
            msg = "Davidson C and H(C) widths must match"
            raise ValueError(msg)
        old_new = mx.conjugate(self.vectors.values) @ mx.transpose(applied.values)
        new_new = _subspace_matrix(vectors.values, applied.values)
        top = mx.concatenate([self.projected, old_new], axis=1)
        bottom = mx.concatenate(
            [mx.conjugate(mx.transpose(old_new)), new_new],
            axis=1,
        )
        return _PairedDavidsonState(
            _CompactLaneState(
                mx.concatenate([self.vectors.values, vectors.values], axis=0),
                self.vectors.layout,
            ),
            _CompactLaneState(
                mx.concatenate([self.applied.values, applied.values], axis=0),
                self.applied.layout,
                "hamiltonian_action",
            ),
            mx.concatenate([top, bottom], axis=0),
            token,
        )

    def transform(
        self,
        transform: mx.array,
        *,
        token: _FixedHamiltonianToken,
    ) -> _PairedDavidsonState:
        self.require_token(token)
        weights = mx.array(transform).astype(mx.complex64)
        if len(weights.shape) != 2 or int(weights.shape[1]) != self.vector_count:
            msg = "Davidson paired transform has the wrong source width"
            raise ValueError(msg)
        vectors = weights @ self.vectors.values
        applied = weights @ self.applied.values
        projected = mx.conjugate(weights) @ self.projected @ mx.transpose(weights)
        return _PairedDavidsonState(
            _CompactLaneState(vectors, self.vectors.layout),
            _CompactLaneState(
                applied,
                self.applied.layout,
                "hamiltonian_action",
            ),
            0.5 * (projected + mx.conjugate(mx.transpose(projected))),
            token,
        )

    def rebase_ranked(
        self,
        rank: _RankResult,
        source_applied: _CompactLaneState,
        *,
        token: _FixedHamiltonianToken,
    ) -> _PairedDavidsonState:
        """Rebase H(V) onto authoritative rank-filtered coefficient values."""

        self.require_token(token)
        _require_layout(source_applied, self.vectors.layout)
        if source_applied.kind != "hamiltonian_action":
            msg = "Davidson ranked rebase requires Hamiltonian actions"
            raise ValueError(msg)
        transform = mx.array(rank.transform).astype(mx.complex64)
        values = mx.array(rank.values).astype(mx.complex64)
        if (
            len(transform.shape) != 2
            or len(values.shape) != 2
            or int(transform.shape[0]) != int(values.shape[0])
            or int(transform.shape[1]) != source_applied.vector_count
        ):
            msg = "Davidson ranked rebase transform has incompatible dimensions"
            raise ValueError(msg)
        vectors = _CompactLaneState(values, self.vectors.layout)
        applied = _CompactLaneState(
            transform @ source_applied.values,
            self.applied.layout,
            "hamiltonian_action",
        )
        return _PairedDavidsonState.initialize(vectors, applied, token)


@dataclass(frozen=True)
class _DavidsonRitzPair:
    """Selected Ritz/H-Ritz values derived entirely from paired lane state."""

    eigenvalues: mx.array
    vectors: _CompactLaneState
    applied: _CompactLaneState
    residual_stack: mx.array
    residuals: mx.array
    max_residual: float
    transform: mx.array


@dataclass(frozen=True)
class _DavidsonRitzCandidate:
    """Lazy Ritz data awaiting one collective finite/residual materialization."""

    eigenvalues: mx.array
    vectors: _CompactLaneState
    applied: _CompactLaneState
    residual_stack: mx.array
    residuals: mx.array
    max_residual: mx.array
    finite: mx.array
    transform: mx.array


def _ritz_residual_arrays(
    eigenvalues: mx.array,
    vectors: _CompactLaneState,
    applied: _CompactLaneState,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    residual_stack = applied.values - eigenvalues[:, None] * vectors.values
    residuals = mx.sqrt(mx.sum(mx.abs(residual_stack) ** 2, axis=1))
    max_residual = mx.max(residuals)
    finite = mx.all(mx.isfinite(eigenvalues)) & mx.all(mx.isfinite(residuals))
    return residual_stack, residuals, max_residual, finite


def _ritz_residual_data(
    eigenvalues: mx.array,
    vectors: _CompactLaneState,
    applied: _CompactLaneState,
) -> tuple[mx.array, mx.array, float]:
    residual_stack, residuals, max_residual_array, finite = _ritz_residual_arrays(
        eigenvalues,
        vectors,
        applied,
    )
    mx.eval(max_residual_array, finite)
    if not bool(finite):
        msg = "Davidson Ritz data must be finite"
        raise ValueError(msg)
    return residual_stack, residuals, float(max_residual_array)


def _seal_ritz_candidate(candidate: _DavidsonRitzCandidate) -> _DavidsonRitzPair:
    if not bool(candidate.finite):
        msg = "Davidson Ritz data must be finite"
        raise ValueError(msg)
    return _DavidsonRitzPair(
        eigenvalues=candidate.eigenvalues,
        vectors=candidate.vectors,
        applied=candidate.applied,
        residual_stack=candidate.residual_stack,
        residuals=candidate.residuals,
        max_residual=float(candidate.max_residual),
        transform=candidate.transform,
    )


def _ritz_candidate_from_projected_eigensystem(
    state: _PairedDavidsonState,
    n_bands: int,
    values: mx.array,
    eigenvectors: mx.array,
) -> _DavidsonRitzCandidate:
    """Build lazy Ritz data from an already solved projected eigensystem."""

    selected_values = mx.real(values[:n_bands])
    selected_vectors = eigenvectors[:, :n_bands]
    transform = mx.transpose(selected_vectors)
    vectors = _CompactLaneState(
        transform @ state.vectors.values,
        state.vectors.layout,
    )
    applied = _CompactLaneState(
        transform @ state.applied.values,
        state.applied.layout,
        "hamiltonian_action",
    )
    residual_stack, residuals, max_residual, finite = _ritz_residual_arrays(
        selected_values,
        vectors,
        applied,
    )
    return _DavidsonRitzCandidate(
        eigenvalues=selected_values,
        vectors=vectors,
        applied=applied,
        residual_stack=residual_stack,
        residuals=residuals,
        max_residual=max_residual,
        finite=finite,
        transform=transform,
    )


def _ritz_pair_from_projected_eigensystem(
    state: _PairedDavidsonState,
    n_bands: int,
    values: mx.array,
    eigenvectors: mx.array,
) -> _DavidsonRitzPair:
    """Build one Ritz pair from an already solved projected eigensystem."""

    candidate = _ritz_candidate_from_projected_eigensystem(
        state,
        n_bands,
        values,
        eigenvectors,
    )
    mx.eval(candidate.max_residual, candidate.finite)
    return _seal_ritz_candidate(candidate)


def _ritz_pair(state: _PairedDavidsonState, n_bands: int) -> _DavidsonRitzPair:
    values, eigenvectors = _projected_eigh(state.projected)
    return _ritz_pair_from_projected_eigensystem(
        state,
        n_bands,
        values,
        eigenvectors,
    )


def _ritz_candidate_with_direct_action(
    candidate: _DavidsonRitzPair,
    applied: _CompactLaneState,
) -> _DavidsonRitzCandidate:
    """Build lazy Ritz data whose residual uses the exact scheduled H(X)."""

    _require_layout(applied, candidate.vectors.layout)
    if applied.kind != "hamiltonian_action":
        msg = "Davidson direct validation requires Hamiltonian actions"
        raise ValueError(msg)
    if applied.vector_count != candidate.vectors.vector_count:
        msg = "Davidson direct validation width does not match its Ritz vectors"
        raise ValueError(msg)
    residual_stack, residuals, max_residual, finite = _ritz_residual_arrays(
        candidate.eigenvalues,
        candidate.vectors,
        applied,
    )
    return _DavidsonRitzCandidate(
        eigenvalues=candidate.eigenvalues,
        vectors=candidate.vectors,
        applied=applied,
        residual_stack=residual_stack,
        residuals=residuals,
        max_residual=max_residual,
        finite=finite,
        transform=candidate.transform,
    )


def _ritz_pair_with_direct_action(
    candidate: _DavidsonRitzPair,
    applied: _CompactLaneState,
) -> _DavidsonRitzPair:
    """Return one Ritz pair whose residuals use the exact scheduled H(X)."""

    direct = _ritz_candidate_with_direct_action(candidate, applied)
    mx.eval(direct.max_residual, direct.finite)
    return _seal_ritz_candidate(direct)


def _projected_eigh(
    matrix: mx.array,
    *,
    observer: RuntimeObserver | None = None,
) -> tuple[mx.array, mx.array]:
    # Only the small projected Rayleigh-Ritz matrix crosses to the CPU. LAPACK's
    # complex128 solve avoids the complex64 convergence floor while every
    # full-grid operator, residual, and FFT remains on the default MLX device.
    projected = _to_numpy(matrix, dtype=np.complex128, observer=observer)
    if projected.ndim != 2 or projected.shape[0] == 0 or projected.shape[0] != projected.shape[1]:
        msg = "projected Rayleigh-Ritz matrix must be non-empty and square"
        raise ValueError(msg)
    if not np.all(np.isfinite(projected)):
        msg = "projected Rayleigh-Ritz matrix must be finite"
        raise ValueError(msg)
    values, vectors = np.linalg.eigh(projected)
    if (
        values.shape != (projected.shape[0],)
        or vectors.shape != projected.shape
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(vectors))
    ):
        msg = "projected Rayleigh-Ritz eigensolve returned invalid eigenpairs"
        raise ValueError(msg)
    return (
        mx.array(values.astype(np.float32)),
        mx.array(vectors.astype(np.complex64)),
    )


def _projected_eigh_batch(
    matrices: Sequence[mx.array],
    *,
    observer: RuntimeObserver | None = None,
) -> tuple[tuple[mx.array, mx.array], ...]:
    """Solve equal-width projected eigensystems through one device-to-CPU transfer."""

    if not matrices:
        msg = "projected Rayleigh-Ritz batch must be non-empty"
        raise ValueError(msg)
    shape = tuple(int(value) for value in matrices[0].shape)
    if len(shape) != 2 or shape[0] == 0 or shape[0] != shape[1]:
        msg = "projected Rayleigh-Ritz matrices must be non-empty and square"
        raise ValueError(msg)
    if any(tuple(int(value) for value in matrix.shape) != shape for matrix in matrices):
        msg = "projected Rayleigh-Ritz batch matrices must have equal shapes"
        raise ValueError(msg)
    projected_stack = mx.stack(
        [
            matrix if isinstance(matrix, mx.array) else mx.array(matrix)
            for matrix in matrices
        ],
        axis=0,
    )
    projected = _to_numpy(
        projected_stack,
        dtype=np.complex128,
        observer=observer,
    )
    if not np.all(np.isfinite(projected)):
        msg = "projected Rayleigh-Ritz batch must be finite"
        raise ValueError(msg)
    values, vectors = np.linalg.eigh(projected)
    expected_values = (len(matrices), shape[0])
    expected_vectors = (len(matrices), *shape)
    if (
        values.shape != expected_values
        or vectors.shape != expected_vectors
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(vectors))
    ):
        msg = "projected Rayleigh-Ritz batch returned invalid eigenpairs"
        raise ValueError(msg)
    return tuple(
        (
            mx.array(values[index].astype(np.float32)),
            mx.array(vectors[index].astype(np.complex64)),
        )
        for index in range(len(matrices))
    )


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
    capture_orbital_density: bool = False


@dataclass(frozen=True)
class _DavidsonPendingAction:
    """One scheduled Davidson action and the state needed to consume it."""

    purpose: str
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
        if type(request.capture_orbital_density) is not bool:
            msg = "capture_orbital_density must be a bool"
            raise ValueError(msg)

    @staticmethod
    def _ticket(
        progress: _DavidsonLaneProgress,
        vectors: _CompactLaneState,
        *,
        purpose: str = "basis",
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
                        progress.paired
                        for progress in compatible
                        if progress.paired is not None
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
                        )
                    )
                    for progress, candidate in candidates:
                        pairs[progress.request.lane_id] = _seal_ritz_candidate(
                            candidate
                        )
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
                    )
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
        if progress.pending_action is not None or not progress.direct_validated:
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
            msg = "Davidson convergence disagrees with its direct residual"
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
                initial_tickets.append(
                    self._ticket(progress, progress.initial_vectors)
                )
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
                    pending_tickets.append(
                        self._ticket_for_pending(progress, pending)
                    )
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
            if progress.request.observer is not None
            and not progress.request.observer.detail_events
        }
        for observer in summarized_observers.values():
            summarized_progress = [
                progress
                for progress in active
                if progress.request.observer is observer
                and progress.request.lane_id in ritz_pairs
            ]
            if not summarized_progress:
                continue
            observer.emit(
                "davidson_round",
                iteration_min=min(
                    progress.iteration_count + 1
                    for progress in summarized_progress
                ),
                iteration_max=max(
                    progress.iteration_count + 1
                    for progress in summarized_progress
                ),
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
            purpose=(
                "direct_validation"
                if pending.purpose == "direct_validation"
                else "basis"
            ),
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
            failure = (
                scheduled.failures.get(lane_id)
                or direct_failures.get(lane_id)
            )
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
                    followup_tickets.append(
                        self._ticket_for_pending(progress, followup)
                    )
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
