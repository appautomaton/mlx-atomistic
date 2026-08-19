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
    occupation: float
    policy: _CompactBatchPolicy
    observer: RuntimeObserver | None
    grid_shape: tuple[int, int, int]
    states: tuple[_CompactLaneState, ...]
    orbital_densities: tuple[mx.array, ...] | None
    density: mx.array

    @classmethod
    def create(
        cls,
        results: Sequence[PeriodicKPointResult],
        *,
        occupation: float,
        policy: _CompactBatchPolicy,
        observer: RuntimeObserver | None,
        orbital_densities: Sequence[mx.array] | None,
    ) -> _PeriodicDensityBuilder:
        owned_results = tuple(results)
        grid_shape, states = _validated_density_states(owned_results)
        cached_densities = cls._validated_orbital_densities(
            orbital_densities,
            count=len(owned_results),
            grid_shape=grid_shape,
        )
        return cls(
            results=owned_results,
            occupation=occupation,
            policy=policy,
            observer=observer,
            grid_shape=grid_shape,
            states=states,
            orbital_densities=cached_densities,
            density=mx.zeros(grid_shape, dtype=mx.float32),
        )

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
                batch_byte_estimator=self._batch_bytes,
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
        _indices: tuple[int, ...],
        batch: _CompactBatch,
    ) -> int:
        if self.orbital_densities is None:
            return _density_batch_bytes(batch)
        return _cached_density_batch_bytes(
            lane_capacity=batch.lane_capacity,
            grid_size=batch.grid_size,
        )

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
        estimated_transient_bytes = _density_batch_bytes(batch)
        if estimated_transient_bytes > self.policy.max_transient_bytes:
            msg = "density batch exceeds the complete transient byte budget"
            raise ValueError(msg)
        weights = self._padded_weights(indices, batch.lane_capacity)
        orbitals = batch.to_real()
        weighted_density = weights[:, None, None, None] * mx.sum(
            mx.abs(orbitals) ** 2,
            axis=1,
        )
        self.density = self.density + mx.sum(weighted_density, axis=0)
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
        self.density = self.density + mx.sum(
            weights[:, None, None, None] * densities,
            axis=0,
        )
        mx.eval(self.density)
        if self.observer is not None:
            self.observer.record_peak_memory(
                "peak_temporary_bytes",
                _cached_density_batch_bytes(
                    lane_capacity=capacity.lanes,
                    grid_size=int(np.prod(self.grid_shape)),
                ),
            )

    def _padded_weights(self, indices: tuple[int, ...], lane_capacity: int) -> mx.array:
        weights = mx.array(
            np.asarray(
                [self.results[index].integration_weight * self.occupation for index in indices],
                dtype=np.float32,
            )
        )
        padding = lane_capacity - len(indices)
        if padding:
            weights = mx.concatenate([weights, mx.zeros((padding,), dtype=mx.float32)])
        return weights

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
    occupation: float,
    policy: _CompactBatchPolicy = _DEFAULT_COMPACT_BATCH_POLICY,
    observer: RuntimeObserver | None = None,
    orbital_densities: Sequence[mx.array] | None = None,
) -> mx.array:
    return _PeriodicDensityBuilder.create(
        results,
        occupation=occupation,
        policy=policy,
        observer=observer,
        orbital_densities=orbital_densities,
    ).build()
