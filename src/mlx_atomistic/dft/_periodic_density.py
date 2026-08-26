"""Periodic density construction from compact k-point states."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft._compact import (
    _DEFAULT_COMPACT_BATCH_POLICY,
    _CompactBatch,
    _CompactBatchPolicy,
    _CompactLaneState,
)
from mlx_atomistic.dft._periodic_davidson_planner import (
    _CompactBatchCapacity,
    _plan_compact_submissions,
    _stable_compact_capacity_groups,
)
from mlx_atomistic.dft._periodic_density_symmetry import _DensitySymmetryPlan
from mlx_atomistic.dft._periodic_execution import _detached_failure
from mlx_atomistic.dft._periodic_models import PeriodicKPointResult
from mlx_atomistic.dft._runtime_observer import RuntimeObserver, add_observed_work


def _density_batch_bytes(batch: _CompactBatch) -> int:
    return (
        batch.estimated_transient_bytes
        + batch.lane_capacity * batch.grid_size * 4
        + batch.grid_size * 4
    )


def _cached_density_batch_bytes(*, lane_capacity: int, grid_size: int) -> int:
    return (lane_capacity + 1) * grid_size * 4


def _validated_density_states(
    results: Sequence[PeriodicKPointResult],
) -> tuple[tuple[int, int, int], tuple[_CompactLaneState, ...]]:
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
    return grid_shape, tuple(states)


def _compatible_density_groups(
    states: Sequence[_CompactLaneState],
) -> tuple[tuple[int, ...], ...]:
    groups: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        groups[(id(state.layout.reciprocal), state.layout.grid_shape)].append(index)
    return tuple(tuple(indices) for indices in groups.values())


@dataclass
class _PeriodicDensityBuilder:
    """Build one density while preserving deterministic compact-batch order."""

    results: tuple[PeriodicKPointResult, ...]
    occupation: float | None
    policy: _CompactBatchPolicy
    observer: RuntimeObserver | None
    grid_shape: tuple[int, int, int]
    states: tuple[_CompactLaneState, ...]
    orbital_densities: tuple[mx.array, ...] | None
    symmetry_plan: _DensitySymmetryPlan | None
    density: mx.array

    @classmethod
    def create(
        cls,
        results: Sequence[PeriodicKPointResult],
        *,
        occupation: float | None,
        policy: _CompactBatchPolicy,
        observer: RuntimeObserver | None,
        orbital_densities: Sequence[mx.array] | None,
        symmetry_plan: _DensitySymmetryPlan | None,
    ) -> _PeriodicDensityBuilder:
        owned_results = tuple(results)
        grid_shape, states = _validated_density_states(owned_results)
        cached_densities = cls._validated_orbital_densities(
            orbital_densities,
            count=len(owned_results),
            grid_shape=grid_shape,
        )
        cls._validate_occupations(
            owned_results,
            states,
            occupation=occupation,
            orbital_densities=cached_densities,
        )
        cls._validate_symmetry_plan(
            owned_results,
            grid_shape=grid_shape,
            symmetry_plan=symmetry_plan,
        )
        return cls(
            results=owned_results,
            occupation=occupation,
            policy=policy,
            observer=observer,
            grid_shape=grid_shape,
            states=states,
            orbital_densities=cached_densities,
            symmetry_plan=symmetry_plan,
            density=mx.zeros(grid_shape, dtype=mx.float32),
        )

    @staticmethod
    def _validate_symmetry_plan(
        results: Sequence[PeriodicKPointResult],
        *,
        grid_shape: tuple[int, int, int],
        symmetry_plan: _DensitySymmetryPlan | None,
    ) -> None:
        if symmetry_plan is None:
            return
        if symmetry_plan.grid_shape != grid_shape:
            raise ValueError("symmetry density plan differs from the k-point grid")
        for result in results:
            explicit_index = result.explicit_index
            if explicit_index is None or not 0 <= explicit_index < len(
                symmetry_plan.terms_by_explicit_index
            ):
                raise ValueError("symmetry density plan requires explicit k-point indices")
            if not symmetry_plan.terms_by_explicit_index[explicit_index]:
                raise ValueError("symmetry density results must contain only owner k-points")

    @staticmethod
    def _validate_occupations(
        results: Sequence[PeriodicKPointResult],
        states: Sequence[_CompactLaneState],
        *,
        occupation: float | None,
        orbital_densities: Sequence[mx.array] | None,
    ) -> None:
        if occupation is not None:
            value = float(occupation)
            if not np.isfinite(value) or not 0.0 <= value <= 2.0:
                msg = "uniform periodic occupation must be finite and lie in [0, 2]"
                raise ValueError(msg)
            return
        if orbital_densities is not None:
            msg = "cached orbital-density sums require one uniform occupation"
            raise ValueError(msg)
        for result, state in zip(results, states, strict=True):
            values = result.occupations
            if values is None or len(values) != state.vector_count:
                msg = "resolved occupations must match every k-point band count"
                raise ValueError(msg)
            array = np.asarray(values, dtype=np.float64)
            if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 2.0):
                msg = "resolved periodic occupations must be finite and lie in [0, 2]"
                raise ValueError(msg)

    @staticmethod
    def _validated_orbital_densities(
        orbital_densities: Sequence[mx.array] | None,
        *,
        count: int,
        grid_shape: tuple[int, int, int],
    ) -> tuple[mx.array, ...] | None:
        if orbital_densities is None:
            return None
        densities = tuple(mx.array(density) for density in orbital_densities)
        if len(densities) != count:
            msg = "cached orbital densities must match the k-point results"
            raise ValueError(msg)
        if any(density.shape != grid_shape for density in densities):
            msg = "cached orbital densities must match the real-space grid"
            raise ValueError(msg)
        if any(density.dtype != mx.float32 for density in densities):
            msg = "cached orbital densities must use float32 storage"
            raise ValueError(msg)
        return densities

    def build(self) -> mx.array:
        """Accumulate every compatible state group into one real density."""

        for indices in _compatible_density_groups(self.states):
            self._accumulate_compatible_group(indices)
        return mx.real(self.density)

    def _accumulate_compatible_group(self, indices: tuple[int, ...]) -> None:
        vector_capacity = max(self.states[index].vector_count for index in indices)
        groups = _stable_compact_capacity_groups(
            self.states,
            indices,
            lane_capacity=self.policy.batch_cap,
            vector_capacity=vector_capacity,
            max_padding_fraction=self.policy.max_padding_fraction,
        )
        for capacity_indices, capacity in groups:
            capacity_states = [self.states[index] for index in capacity_indices]
            plan = _plan_compact_submissions(
                capacity_states,
                policy=self.policy,
                batch_byte_estimator=lambda planned_indices,
                batch,
                capacity_indices=capacity_indices: self._batch_bytes(
                    tuple(capacity_indices[index] for index in planned_indices),
                    batch,
                ),
                capacity=capacity,
            )
            if plan.failures:
                failed_index = min(plan.failures)
                raise _detached_failure(plan.failures[failed_index]) from None
            for submission in plan.submissions:
                logical_indices = tuple(capacity_indices[index] for index in submission.indices)
                self._accumulate_submission(logical_indices, capacity)

    def _batch_bytes(
        self,
        indices: tuple[int, ...],
        batch: _CompactBatch,
    ) -> int:
        if self.orbital_densities is None:
            return _density_batch_bytes(batch) + self._symmetry_expansion_bytes(indices)
        return (
            _cached_density_batch_bytes(
                lane_capacity=batch.lane_capacity,
                grid_size=batch.grid_size,
            )
            + self._symmetry_expansion_bytes(indices)
        )

    def _symmetry_expansion_bytes(self, indices: Sequence[int]) -> int:
        if self.symmetry_plan is None:
            return 0
        term_count = sum(
            len(
                self.symmetry_plan.terms_by_explicit_index[
                    self.results[index].explicit_index
                ]
            )
            for index in indices
        )
        return (term_count + len(indices)) * int(np.prod(self.grid_shape)) * 4

    def _accumulate_submission(
        self,
        indices: tuple[int, ...],
        capacity: _CompactBatchCapacity,
    ) -> None:
        if self.orbital_densities is not None:
            self._accumulate_cached_submission(indices, capacity)
            return
        batch = _CompactBatch.from_states(
            [self.states[index] for index in indices],
            policy=self.policy,
            lane_capacity=capacity.lanes,
            vector_capacity=capacity.vectors,
            active_capacity=capacity.active,
        )
        estimated_transient_bytes = (
            _density_batch_bytes(batch) + self._symmetry_expansion_bytes(indices)
        )
        if estimated_transient_bytes > self.policy.max_transient_bytes:
            msg = "density batch exceeds the complete transient byte budget"
            raise ValueError(msg)
        weights = self._padded_weights(
            indices,
            batch.lane_capacity,
            vector_capacity=batch.vector_count,
        )
        orbitals = batch.to_real()
        orbital_density = mx.abs(orbitals) ** 2
        if self.occupation is None:
            weighted_density = mx.sum(
                weights[:, :, None, None, None] * orbital_density,
                axis=1,
            )
        else:
            weighted_density = weights[:, None, None, None] * mx.sum(
                orbital_density,
                axis=1,
            )
        self._accumulate_lane_densities(indices, weighted_density)
        mx.eval(self.density)
        self._record_submission(batch, estimated_transient_bytes)

    def _accumulate_cached_submission(
        self,
        indices: tuple[int, ...],
        capacity: _CompactBatchCapacity,
    ) -> None:
        if self.orbital_densities is None:
            msg = "cached density submission requires captured orbitals"
            raise RuntimeError(msg)
        densities = mx.stack(
            [self.orbital_densities[index] for index in indices],
            axis=0,
        )
        padding = capacity.lanes - len(indices)
        if padding:
            densities = mx.concatenate(
                [
                    densities,
                    mx.zeros(
                        (padding, *self.grid_shape),
                        dtype=mx.float32,
                    ),
                ],
                axis=0,
            )
        weights = self._padded_weights(indices, capacity.lanes)
        weighted_density = weights[:, None, None, None] * densities
        self._accumulate_lane_densities(indices, weighted_density)
        mx.eval(self.density)
        if self.observer is not None:
            self.observer.record_peak_memory(
                "peak_temporary_bytes",
                _cached_density_batch_bytes(
                    lane_capacity=capacity.lanes,
                    grid_size=int(np.prod(self.grid_shape)),
                )
                + self._symmetry_expansion_bytes(indices),
            )

    def _padded_weights(
        self,
        indices: tuple[int, ...],
        lane_capacity: int,
        *,
        vector_capacity: int | None = None,
    ) -> mx.array:
        def integration_weight(index: int) -> float:
            if self.symmetry_plan is not None:
                return 1.0
            return self.results[index].integration_weight

        if self.occupation is None:
            if vector_capacity is None:
                msg = "band-resolved density weights require a vector capacity"
                raise RuntimeError(msg)
            weights = np.zeros((lane_capacity, vector_capacity), dtype=np.float32)
            for lane, index in enumerate(indices):
                occupations = self.results[index].occupations
                if occupations is None:
                    msg = "resolved periodic occupations are unavailable"
                    raise RuntimeError(msg)
                weights[lane, : len(occupations)] = (
                    integration_weight(index) * np.asarray(occupations, dtype=np.float32)
                )
            return mx.array(weights)
        weights = mx.array(
            np.asarray(
                [integration_weight(index) * self.occupation for index in indices],
                dtype=np.float32,
            )
        )
        padding = lane_capacity - len(indices)
        if padding:
            weights = mx.concatenate([weights, mx.zeros((padding,), dtype=mx.float32)])
        return weights

    def _accumulate_lane_densities(
        self,
        indices: tuple[int, ...],
        lane_densities: mx.array,
    ) -> None:
        if self.symmetry_plan is None:
            self.density = self.density + mx.sum(lane_densities, axis=0)
            return
        expanded = tuple(
            self.symmetry_plan.expand(
                self.results[index].explicit_index,
                lane_densities[lane],
            )
            for lane, index in enumerate(indices)
        )
        self.density = self.density + sum(expanded[1:], expanded[0])

    def _record_submission(
        self,
        batch: _CompactBatch,
        estimated_transient_bytes: int,
    ) -> None:
        add_observed_work(
            self.observer,
            {
                "fft_submissions": 1,
                "fft_vector_equivalents": batch.logical_vector_count,
                "padding_elements": batch.padding_elements,
            },
        )
        if self.observer is None:
            return
        self.observer.record_peak_memory(
            "fft_workspace_bytes",
            batch.lane_capacity * batch.vector_count * batch.grid_size * 8,
        )
        self.observer.record_peak_memory(
            "peak_temporary_bytes",
            estimated_transient_bytes,
        )


def _density_from_kpoints(
    results: Sequence[PeriodicKPointResult],
    *,
    occupation: float | None = None,
    policy: _CompactBatchPolicy = _DEFAULT_COMPACT_BATCH_POLICY,
    observer: RuntimeObserver | None = None,
    orbital_densities: Sequence[mx.array] | None = None,
    symmetry_plan: _DensitySymmetryPlan | None = None,
) -> mx.array:
    return _PeriodicDensityBuilder.create(
        results,
        occupation=occupation,
        policy=policy,
        observer=observer,
        orbital_densities=orbital_densities,
        symmetry_plan=symmetry_plan,
    ).build()
