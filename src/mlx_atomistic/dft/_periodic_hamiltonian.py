"""Compact fixed-density Hamiltonian execution for periodic DFT."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import mlx.core as mx

from mlx_atomistic.dft._compact import (
    _DEFAULT_COMPACT_BATCH_POLICY,
    _CompactBatch,
    _CompactBatchPolicy,
    _CompactLaneState,
)
from mlx_atomistic.dft._periodic_execution import _detached_failure, _materialize
from mlx_atomistic.dft._periodic_pseudopotential_runtime import (
    _PERIODIC_NONLOCAL_TYPES,
    _PeriodicNonlocalOperator,
)
from mlx_atomistic.dft._runtime_observer import (
    RuntimeObserver,
    add_observed_work,
    observed_phase,
)
from mlx_atomistic.dft.plane_wave import PlaneWaveBasis


def _empty_projector_metrics() -> dict[str, int]:
    return {
        "projector_payload_elements": 0,
        "projector_elements_generated": 0,
        "projector_elements_loaded": 0,
        "projector_traffic_elements": 0,
        "projector_cache_hits": 0,
        "projector_cache_misses": 0,
        "projector_cache_bytes": 0,
        "projector_peak_workspace_bytes": 0,
    }


@dataclass(frozen=True)
class _CompactHamiltonianBatchResult:
    """Lane-local outcomes from one physical compact Hamiltonian batch."""

    actions: dict[int, _CompactLaneState]
    orbital_densities: dict[int, mx.array]
    failures: dict[int, Exception]
    batch: _CompactBatch | None

    def action_for(self, lane_index: int) -> _CompactLaneState:
        failure = self.failures.get(lane_index)
        if failure is not None:
            raise _detached_failure(failure) from None
        try:
            return self.actions[lane_index]
        except KeyError as error:
            msg = f"compact Hamiltonian batch has no lane {lane_index}"
            raise ValueError(msg) from error

    def orbital_density_for(self, lane_index: int) -> mx.array:
        """Return one requested lane's captured real-space orbital density."""

        failure = self.failures.get(lane_index)
        if failure is not None:
            raise _detached_failure(failure) from None
        try:
            return self.orbital_densities[lane_index]
        except KeyError as error:
            msg = f"compact Hamiltonian batch has no orbital density for lane {lane_index}"
            raise ValueError(msg) from error


def _estimated_batch_transient_bytes(
    operators: Sequence[PeriodicKohnShamOperator],
    batch: _CompactBatch,
    *,
    captured_density_lanes: int = 0,
) -> int:
    """Return the complete logical transient bound for one Hpsi batch."""

    if len(operators) > batch.lane_capacity:
        msg = "Hamiltonian operator count exceeds the compact capacity"
        raise ValueError(msg)
    lane_count = batch.lane_capacity
    padded_complex_bytes = lane_count * batch.vector_count * batch.bucket_size * 8
    kinetic_index_bytes = lane_count * batch.bucket_size * 4
    if not 0 <= captured_density_lanes <= len(operators):
        msg = "captured density lane count must match the Hamiltonian operators"
        raise ValueError(msg)
    estimate = (
        batch.estimated_transient_bytes
        + 5 * padded_complex_bytes
        + kinetic_index_bytes
        + captured_density_lanes * batch.grid_size * 4
    )
    first_potential = operators[0]._effective_local_potential
    if not all(operator._effective_local_potential is first_potential for operator in operators):
        estimate += lane_count * batch.grid_size * 8

    grouped_nonlocal: dict[tuple[type, str], list[int]] = defaultdict(list)
    for index, operator in enumerate(operators):
        nonlocal_operator = operator.nonlocal_operator
        if isinstance(nonlocal_operator, _PERIODIC_NONLOCAL_TYPES):
            grouped_nonlocal[
                (type(nonlocal_operator), nonlocal_operator._context_identity)
            ].append(index)
        elif nonlocal_operator is not None:
            estimate += padded_complex_bytes // lane_count
    for indices in grouped_nonlocal.values():
        nonlocal_type = type(operators[indices[0]].nonlocal_operator)
        estimate += nonlocal_type._estimated_batch_transient_bytes(
            [operators[index].nonlocal_operator for index in indices],
            batch,
        )
    return estimate


class _CompactHamiltonianExecutor:
    """Own one bounded batch-first Hamiltonian submission."""

    def __init__(
        self,
        operators: Sequence[PeriodicKohnShamOperator],
        coefficients: Sequence[_CompactLaneState],
        *,
        observer: RuntimeObserver | None,
        policy: _CompactBatchPolicy,
        prepared_batch: _CompactBatch | None,
        capture_orbital_densities: Sequence[bool] | None,
    ) -> None:
        if not operators or len(operators) != len(coefficients):
            msg = "compact Hamiltonian batches require matching non-empty lanes"
            raise ValueError(msg)
        self.operators = tuple(operators)
        self.coefficients = tuple(coefficients)
        self.observer = self._resolve_observer(observer)
        self.policy = policy
        self.prepared_batch = prepared_batch
        self.capture_orbital_densities = (
            (False,) * len(self.operators)
            if capture_orbital_densities is None
            else tuple(capture_orbital_densities)
        )
        if len(self.capture_orbital_densities) != len(self.operators) or any(
            type(value) is not bool for value in self.capture_orbital_densities
        ):
            msg = "orbital-density capture flags must match the Hamiltonian operators"
            raise ValueError(msg)
        self.failures: dict[int, Exception] = {}
        self.actions: dict[int, _CompactLaneState] = {}
        self.orbital_densities: dict[int, mx.array] = {}
        self.ready_indices: list[int] = []
        self.projector_actions: dict[int, _CompactLaneState] = {}
        self.projector_metrics: dict[int, dict[str, int]] = {}
        self.batch: _CompactBatch | None = None
        self.estimated_transient_bytes = 0
        self.executed_fft = False

    def _resolve_observer(self, observer: RuntimeObserver | None) -> RuntimeObserver | None:
        operator_observers = {
            id(operator.observer): operator.observer
            for operator in self.operators
            if operator.observer is not None
        }
        if observer is not None:
            return observer
        if len(operator_observers) > 1:
            msg = "compact Hamiltonian batch operators must share one observer"
            raise ValueError(msg)
        return next(iter(operator_observers.values()), None)

    def execute(self) -> _CompactHamiltonianBatchResult:
        """Execute the batch while isolating lane-local failures."""

        with observed_phase(self.observer, "hpsi"):
            self._admit_lanes()
            if self.ready_indices:
                try:
                    self._execute_ready_lanes()
                except Exception as error:
                    failure = _detached_failure(error)
                    self.actions = {}
                    self.orbital_densities = {}
                    for lane_index in self.ready_indices:
                        self.failures[lane_index] = failure
        self._record_execution()
        return _CompactHamiltonianBatchResult(
            actions=self.actions,
            orbital_densities=self.orbital_densities,
            failures=self.failures,
            batch=self.batch if self.executed_fft else None,
        )

    def _admit_lanes(self) -> None:
        for lane_index, (operator, state) in enumerate(
            zip(self.operators, self.coefficients, strict=True)
        ):
            try:
                if (
                    self.observer is not None
                    and operator.observer is not None
                    and operator.observer is not self.observer
                ):
                    msg = "operator apply observers must be the same object"
                    raise ValueError(msg)
                operator.basis._validate_state(state)
                if state.kind != "coefficients":
                    msg = "Hamiltonian input must be coefficient state"
                    raise ValueError(msg)
                if operator._effective_local_potential.shape != state.layout.grid_shape:
                    msg = "effective local potential must match its lane grid"
                    raise ValueError(msg)
                self.projector_metrics[lane_index] = _empty_projector_metrics()
                if operator.nonlocal_operator is not None and not isinstance(
                    operator.nonlocal_operator,
                    _PERIODIC_NONLOCAL_TYPES,
                ):
                    action, metrics = operator.nonlocal_operator._apply_compact(
                        state,
                        evaluate=False,
                    )
                    self.projector_actions[lane_index] = action
                    self.projector_metrics[lane_index] = metrics
                self.ready_indices.append(lane_index)
            except Exception as error:
                self.failures[lane_index] = _detached_failure(error)

    def _execute_ready_lanes(self) -> None:
        ready_states = [self.coefficients[index] for index in self.ready_indices]
        self.batch = self._prepare_batch(ready_states)
        ready_operators = [self.operators[index] for index in self.ready_indices]
        self.estimated_transient_bytes = _estimated_batch_transient_bytes(
            ready_operators,
            self.batch,
            captured_density_lanes=sum(
                self.capture_orbital_densities[index] for index in self.ready_indices
            ),
        )
        if self.estimated_transient_bytes > self.policy.max_transient_bytes:
            msg = "compact Hpsi batch exceeds the complete transient byte budget"
            raise ValueError(msg)
        self._apply_separable_projectors()
        self._assemble_actions(ready_states)

    def _prepare_batch(self, ready_states: Sequence[_CompactLaneState]) -> _CompactBatch:
        prepared = self.prepared_batch
        if (
            prepared is not None
            and self.ready_indices == list(range(len(self.coefficients)))
            and len(prepared.layouts) == len(ready_states)
            and all(
                layout is state.layout
                for layout, state in zip(prepared.layouts, ready_states, strict=True)
            )
            and prepared.kinds == tuple(state.kind for state in ready_states)
            and prepared.vector_counts == tuple(state.vector_count for state in ready_states)
            and prepared.estimated_transient_bytes <= self.policy.max_transient_bytes
            and max(
                (prepared.bucket_size - count) / prepared.bucket_size
                for count in prepared.active_counts
            )
            <= self.policy.max_padding_fraction
        ):
            return prepared
        return _CompactBatch.from_states(ready_states, policy=self.policy)

    def _apply_separable_projectors(self) -> None:
        grouped_nonlocal: dict[tuple[type, str], list[int]] = defaultdict(list)
        for lane_index in self.ready_indices:
            nonlocal_operator = self.operators[lane_index].nonlocal_operator
            if isinstance(nonlocal_operator, _PERIODIC_NONLOCAL_TYPES):
                grouped_nonlocal[
                    (type(nonlocal_operator), nonlocal_operator._context_identity)
                ].append(lane_index)
        for lane_indices in grouped_nonlocal.values():
            states = [self.coefficients[index] for index in lane_indices]
            if lane_indices == self.ready_indices:
                nonlocal_batch = self.batch
            else:
                nonlocal_batch = _CompactBatch.from_states(states, policy=self.policy)
            nonlocal_type = type(self.operators[lane_indices[0]].nonlocal_operator)
            try:
                actions, metrics_by_lane = nonlocal_type._apply_compact_batch(
                    [self.operators[index].nonlocal_operator for index in lane_indices],
                    states,
                    batch=nonlocal_batch,
                    # The combined Hpsi result is materialized below; evaluating
                    # this contribution alone would add a redundant device barrier.
                    evaluate=False,
                )
            except Exception:
                self._apply_separable_projectors_individually(lane_indices)
            else:
                for lane_index, action, metrics in zip(
                    lane_indices,
                    actions,
                    metrics_by_lane,
                    strict=True,
                ):
                    self.projector_actions[lane_index] = action
                    self.projector_metrics[lane_index] = metrics

    def _apply_separable_projectors_individually(
        self,
        lane_indices: Sequence[int],
    ) -> None:
        for lane_index in lane_indices:
            nonlocal_operator = self.operators[lane_index].nonlocal_operator
            try:
                action, metrics = nonlocal_operator._apply_compact(
                    self.coefficients[lane_index],
                    evaluate=True,
                )
                self.projector_actions[lane_index] = action
                self.projector_metrics[lane_index] = metrics
            except Exception as error:
                self.failures[lane_index] = _detached_failure(error)

    def _assemble_actions(self, ready_states: Sequence[_CompactLaneState]) -> None:
        if self.batch is None:
            raise RuntimeError("compact Hamiltonian batch was not prepared")
        scattered = self.batch.scatter()
        kinetic_values, nonlocal_values = self._padded_kinetic_and_nonlocal(ready_states)
        kinetic_action = self.batch.values * kinetic_values[:, None, :]
        real_orbitals = self.batch.to_real(scattered=scattered)
        local_action = self.batch.apply_local(
            self._batched_potentials(),
            real_orbitals=real_orbitals,
        )
        self.executed_fft = True
        states = self.batch.unpad(
            kinetic_action + local_action + nonlocal_values,
            kind="hamiltonian_action",
        )
        finite = [mx.all(mx.isfinite(state.values)) for state in states]
        captured_densities = {
            lane_index: mx.sum(
                mx.abs(real_orbitals[batch_index, : state.vector_count]) ** 2,
                axis=0,
            ).astype(mx.float32)
            for batch_index, (lane_index, state) in enumerate(
                zip(self.ready_indices, ready_states, strict=True)
            )
            if self.capture_orbital_densities[lane_index] and lane_index not in self.failures
        }
        _materialize(
            self.observer,
            *(state.values for state in states),
            *captured_densities.values(),
            *finite,
        )
        for lane_index, state, is_finite in zip(
            self.ready_indices,
            states,
            finite,
            strict=True,
        ):
            if lane_index in self.failures:
                continue
            if bool(is_finite):
                self.actions[lane_index] = state
            else:
                self.failures[lane_index] = ValueError(
                    "Davidson Hamiltonian action must be finite"
                )
        self.orbital_densities = {
            lane_index: density
            for lane_index, density in captured_densities.items()
            if lane_index in self.actions
        }

    def _padded_kinetic_and_nonlocal(
        self,
        ready_states: Sequence[_CompactLaneState],
    ) -> tuple[mx.array, mx.array]:
        if self.batch is None:
            raise RuntimeError("compact Hamiltonian batch was not prepared")
        kinetic_rows = []
        nonlocal_rows = []
        for lane_index, state in zip(self.ready_indices, ready_states, strict=True):
            padding = self.batch.bucket_size - state.layout.active_count
            kinetic = state.layout._active_kinetic_energies
            nonlocal_values = (
                self.projector_actions[lane_index].values
                if lane_index in self.projector_actions
                else mx.zeros_like(state.values)
            )
            if padding:
                kinetic = mx.concatenate([kinetic, mx.zeros((padding,), dtype=mx.float32)])
                nonlocal_values = mx.concatenate(
                    [
                        nonlocal_values,
                        mx.zeros(
                            (state.vector_count, padding),
                            dtype=mx.complex64,
                        ),
                    ],
                    axis=1,
                )
            vector_padding = self.batch.vector_count - state.vector_count
            if vector_padding:
                nonlocal_values = mx.concatenate(
                    [
                        nonlocal_values,
                        mx.zeros(
                            (vector_padding, self.batch.bucket_size),
                            dtype=mx.complex64,
                        ),
                    ],
                    axis=0,
                )
            kinetic_rows.append(kinetic)
            nonlocal_rows.append(nonlocal_values)
        lane_padding = self.batch.lane_capacity - self.batch.lane_count
        kinetic_values = mx.stack(kinetic_rows, axis=0)
        nonlocal_values = mx.stack(nonlocal_rows, axis=0)
        if lane_padding:
            kinetic_values = mx.concatenate(
                [
                    kinetic_values,
                    mx.zeros(
                        (lane_padding, self.batch.bucket_size),
                        dtype=mx.float32,
                    ),
                ],
                axis=0,
            )
            nonlocal_values = mx.concatenate(
                [
                    nonlocal_values,
                    mx.zeros(
                        (
                            lane_padding,
                            self.batch.vector_count,
                            self.batch.bucket_size,
                        ),
                        dtype=mx.complex64,
                    ),
                ],
                axis=0,
            )
        return kinetic_values, nonlocal_values

    def _batched_potentials(self) -> mx.array:
        first_potential = self.operators[
            self.ready_indices[0]
        ]._effective_local_potential
        if all(
            self.operators[index]._effective_local_potential is first_potential
            for index in self.ready_indices
        ):
            return first_potential
        return mx.stack(
            [
                self.operators[index]._effective_local_potential
                for index in self.ready_indices
            ],
            axis=0,
        )

    def _record_execution(self) -> None:
        if self.batch is None or not self.executed_fft:
            return
        logical_vector_count = self.batch.logical_vector_count
        submitted_vector_count = self.batch.lane_capacity * self.batch.vector_count
        lane_padding_vector_count = (
            self.batch.lane_capacity - self.batch.lane_count
        ) * self.batch.vector_count
        vector_padding_count = sum(
            self.batch.vector_count - count for count in self.batch.vector_counts
        )
        if (
            logical_vector_count + lane_padding_vector_count + vector_padding_count
            != submitted_vector_count
        ):
            msg = "compact Hpsi physical-vector accounting is inconsistent"
            raise RuntimeError(msg)
        generated = self._projector_metric("projector_elements_generated")
        loaded = self._projector_metric("projector_elements_loaded")
        traffic = self._projector_metric("projector_traffic_elements")
        cache_hits = self._projector_metric("projector_cache_hits")
        cache_misses = self._projector_metric("projector_cache_misses")
        add_observed_work(
            self.observer,
            {
                "hpsi_calls": 1,
                "hpsi_vector_equivalents": logical_vector_count,
                "hpsi_submitted_vector_equivalents": submitted_vector_count,
                "hpsi_lane_padding_vector_equivalents": lane_padding_vector_count,
                "hpsi_vector_padding_equivalents": vector_padding_count,
                "fft_submissions": 2,
                "fft_vector_equivalents": 2 * logical_vector_count,
                "projector_elements_generated": generated,
                "projector_elements_loaded": loaded,
                "projector_traffic_elements": traffic,
                "padding_elements": self.batch.padding_elements,
                "projector_cache_hits": cache_hits,
                "projector_cache_misses": cache_misses,
            },
        )
        if self.observer is not None:
            self._record_memory()

    def _projector_metric(self, name: str) -> int:
        return sum(self.projector_metrics[index][name] for index in self.ready_indices)

    def _record_memory(self) -> None:
        if self.batch is None or self.observer is None:
            return
        self.observer.record_hpsi_shape(
            self.batch.lane_capacity,
            self.batch.vector_count,
        )
        fft_workspace_bytes = (
            2
            * self.batch.lane_capacity
            * self.batch.vector_count
            * self.batch.grid_size
            * 8
        )
        self.observer.record_peak_memory("fft_workspace_bytes", fft_workspace_bytes)
        self.observer.record_peak_memory("hpsi_fft_workspace_bytes", fft_workspace_bytes)
        self.observer.record_peak_memory(
            "peak_temporary_bytes",
            self.estimated_transient_bytes,
        )
        self.observer.record_peak_memory(
            "hpsi_peak_temporary_bytes",
            self.estimated_transient_bytes,
        )
        self.observer.record_peak_memory(
            "projector_payload_bytes",
            self._projector_metric("projector_payload_elements") * 8,
        )
        self.observer.record_memory(
            "persistent_projector_bytes",
            max(
                (
                    self.projector_metrics[index]["projector_cache_bytes"]
                    for index in self.ready_indices
                ),
                default=0,
            ),
        )


@dataclass(frozen=True, init=False)
class PeriodicKohnShamOperator:
    """Fixed-density periodic Kohn-Sham operator in coefficient space."""

    basis: PlaneWaveBasis
    _effective_local_potential: mx.array = field(repr=False)
    nonlocal_operator: _PeriodicNonlocalOperator | None = None
    observer: RuntimeObserver | None = None

    def __init__(
        self,
        basis: PlaneWaveBasis,
        effective_local_potential: mx.array,
        nonlocal_operator: _PeriodicNonlocalOperator | None = None,
        observer: RuntimeObserver | None = None,
    ) -> None:
        potential_snapshot = mx.array(effective_local_potential)
        # Materialize an owned device buffer now so later caller mutation cannot
        # alter this fixed Hamiltonian through a lazy dependency.
        mx.eval(potential_snapshot)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "_effective_local_potential", potential_snapshot)
        object.__setattr__(self, "nonlocal_operator", nonlocal_operator)
        object.__setattr__(self, "observer", observer)

    @classmethod
    def _from_shared_potential(
        cls,
        basis: PlaneWaveBasis,
        potential_snapshot: mx.array,
        nonlocal_operator: _PeriodicNonlocalOperator | None = None,
        observer: RuntimeObserver | None = None,
    ) -> PeriodicKohnShamOperator:
        """Bind a private SCF operator to one already evaluated potential."""

        if potential_snapshot.shape != basis.grid.shape:
            msg = "shared effective local potential must match the basis grid"
            raise ValueError(msg)
        result = object.__new__(cls)
        object.__setattr__(result, "basis", basis)
        object.__setattr__(result, "_effective_local_potential", potential_snapshot)
        object.__setattr__(result, "nonlocal_operator", nonlocal_operator)
        object.__setattr__(result, "observer", observer)
        return result

    @property
    def effective_local_potential(self) -> mx.array:
        """Return a fresh caller-owned copy of the fixed local potential."""

        potential = mx.array(self._effective_local_potential)
        mx.eval(potential)
        return potential

    def _nonlocal_batch_compatibility_key(self) -> object:
        nonlocal_operator = self.nonlocal_operator
        if nonlocal_operator is None:
            return None
        if isinstance(nonlocal_operator, _PERIODIC_NONLOCAL_TYPES):
            return (
                str(nonlocal_operator.pseudopotentials[0].format),
                nonlocal_operator._context_identity,
            )
        return ("custom", id(nonlocal_operator))

    def apply(
        self,
        coefficients: mx.array,
        *,
        observer: RuntimeObserver | None = None,
    ) -> mx.array:
        """Apply kinetic, local, and optional nonlocal terms.

        Args:
            coefficients: One coefficient grid or an orbital stack.
            observer: Optional observer overriding an absent operator observer.

        Returns:
            Hamiltonian action with matching shape.
        """

        if observer is not None and self.observer is not None and observer is not self.observer:
            msg = "operator apply observers must be the same object"
            raise ValueError(msg)
        state, was_single = self.basis._state_from_full(coefficients)
        applied = self._apply_compact(state, observer=observer)
        return self.basis._layout.unpack_fresh(applied.values, single=was_single)

    def _apply_compact(
        self,
        coefficients: _CompactLaneState,
        *,
        observer: RuntimeObserver | None = None,
        policy: _CompactBatchPolicy = _DEFAULT_COMPACT_BATCH_POLICY,
        prepared_batch: _CompactBatch | None = None,
    ) -> _CompactLaneState:
        outcome = self._apply_compact_batch(
            (self,),
            (coefficients,),
            observer=observer,
            policy=policy,
            prepared_batch=prepared_batch,
        )
        return outcome.action_for(0)

    @staticmethod
    def _estimated_batch_transient_bytes(
        operators: Sequence[PeriodicKohnShamOperator],
        batch: _CompactBatch,
        *,
        captured_density_lanes: int = 0,
    ) -> int:
        return _estimated_batch_transient_bytes(
            operators,
            batch,
            captured_density_lanes=captured_density_lanes,
        )

    @staticmethod
    def _apply_compact_batch(
        operators: Sequence[PeriodicKohnShamOperator],
        coefficients: Sequence[_CompactLaneState],
        *,
        observer: RuntimeObserver | None = None,
        policy: _CompactBatchPolicy = _DEFAULT_COMPACT_BATCH_POLICY,
        prepared_batch: _CompactBatch | None = None,
        capture_orbital_densities: Sequence[bool] | None = None,
    ) -> _CompactHamiltonianBatchResult:
        """Apply one bounded batch-first Hamiltonian submission."""

        return _CompactHamiltonianExecutor(
            operators,
            coefficients,
            observer=observer,
            policy=policy,
            prepared_batch=prepared_batch,
            capture_orbital_densities=capture_orbital_densities,
        ).execute()

    def rayleigh_quotients(
        self,
        coefficients: mx.array,
        *,
        observer: RuntimeObserver | None = None,
    ) -> mx.array:
        """Return one Rayleigh quotient per orbital.

        Args:
            coefficients: Orbital stack in coefficient space.
            observer: Optional runtime observer.

        Returns:
            Real energy estimates in Hartree.
        """

        state, _ = self.basis._state_from_full(coefficients)
        return self._rayleigh_quotients_compact(state, observer=observer)

    def _rayleigh_quotients_compact(
        self,
        coefficients: _CompactLaneState,
        *,
        observer: RuntimeObserver | None = None,
    ) -> mx.array:
        self.basis._validate_state(coefficients)
        applied = self._apply_compact(coefficients, observer=observer)
        numerator = mx.sum(mx.conjugate(coefficients.values) * applied.values, axis=1)
        denominator = mx.sum(mx.abs(coefficients.values) ** 2, axis=1)
        return mx.real(numerator / denominator)
