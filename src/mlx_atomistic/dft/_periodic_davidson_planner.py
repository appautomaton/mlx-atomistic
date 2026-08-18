"""Physical shape and submission planning for compact Davidson batches."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from mlx_atomistic.dft._compact import (
    _CompactBatch,
    _CompactBatchPolicy,
    _CompactLaneState,
)
from mlx_atomistic.dft._periodic_execution import _detached_failure


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
        batch_byte_estimator: (Callable[[tuple[int, ...], _CompactBatch], int] | None),
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
                    (state.vector_count if self.capacity is None else self.capacity.vectors),
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
