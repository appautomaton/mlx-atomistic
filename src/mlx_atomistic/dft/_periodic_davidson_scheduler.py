"""Hpsi dispatch scheduling for periodic Davidson lanes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

from mlx_atomistic.dft._compact import (
    _DEFAULT_COMPACT_BATCH_POLICY,
    _CompactBatch,
    _CompactBatchPolicy,
    _CompactLaneState,
)
from mlx_atomistic.dft._periodic_davidson_context import _FixedHamiltonianToken
from mlx_atomistic.dft._periodic_davidson_planner import (
    _CompactBatchCapacity,
    _CompactSubmission,
    _CompactSubmissionPlan,
    _finite_lane_capacity,
    _finite_vector_capacity,
    _plan_compact_submissions,
    _stable_compact_capacity_groups,
)
from mlx_atomistic.dft._periodic_execution import _detached_failure
from mlx_atomistic.dft._periodic_hamiltonian import PeriodicKohnShamOperator
from mlx_atomistic.dft._periodic_models import PeriodicDavidsonConfig
from mlx_atomistic.dft._periodic_orthonormalization import _Complex64RankPolicy
from mlx_atomistic.dft._runtime_observer import RuntimeObserver, add_observed_work


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
    purpose: Literal["initial", "correction", "direct_validation"] = "initial"
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
        return (
            id(layout.reciprocal),
            layout.grid_shape,
            id(_DavidsonScheduler._observer(ticket)),
            ticket.operator._nonlocal_batch_compatibility_key(),
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
            compatibility_groups.append(tuple(ticket.lane_id for ticket in compatible))
            plan = self._plan_compatible_group(compatible)
            for lane_index, error in plan.failures.items():
                failures[compatible[lane_index].lane_id] = error
            for planned in plan.submissions:
                submission = tuple(compatible[index] for index in planned.indices)
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
            failures.update({ticket.lane_id: failure for ticket, _ in validated})
            return ready, failures
        for ticket, finite in validated:
            if bool(finite):
                ready.append(ticket)
            else:
                failures[ticket.lane_id] = ValueError("Davidson application block must be finite")
        return ready, failures

    @staticmethod
    def _validate_ticket(ticket: _DavidsonApplicationTicket) -> mx.array:
        if ticket.purpose not in {"initial", "correction", "direct_validation"}:
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
            self._capacity_by_lane.get(ticket.lane_id) != bound_capacity for ticket in compatible
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
            lane_capacity=(None if planned.capacity is None else planned.capacity.lanes),
            vector_capacity=(None if planned.capacity is None else planned.capacity.vectors),
            active_capacity=(None if planned.capacity is None else planned.capacity.active),
        )
        complete_transient_bytes = PeriodicKohnShamOperator._estimated_batch_transient_bytes(
            [ticket.operator for ticket in submission],
            prepared_batch,
            captured_density_lanes=sum(ticket.capture_orbital_density for ticket in submission),
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
                if ticket.purpose != "direct_validation":
                    add_observed_work(
                        ticket.observer,
                        {"davidson_hv_new_vectors": (ticket.vectors.vector_count)},
                    )
                if ticket.capture_orbital_density:
                    accepted_densities[ticket.lane_id] = batch_densities[ticket.lane_id]
        except Exception as error:
            failure = _detached_failure(error)
            accepted_actions.clear()
            accepted_densities.clear()
            submission_failures.update({ticket.lane_id: failure for ticket in submission})
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
        failures = {submission[index].lane_id: error for index, error in outcome.failures.items()}
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
