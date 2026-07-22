"""Frozen progress, synchronized phase, work, and memory observation contract."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from time import perf_counter

EVENT_SCHEMA = "mlx-atomistic.dft-runtime-event.v1"
OBSERVATION_SCHEMA = "mlx-atomistic.dft-runtime-observation.v1"

WORK_COUNTER_NAMES = (
    "hpsi_calls",
    "hpsi_vector_equivalents",
    "hpsi_submitted_vector_equivalents",
    "hpsi_lane_padding_vector_equivalents",
    "hpsi_vector_padding_equivalents",
    "fft_submissions",
    "fft_vector_equivalents",
    "projector_elements_generated",
    "projector_elements_loaded",
    "projector_traffic_elements",
    "davidson_hv_new_vectors",
    "davidson_hv_reused_vectors",
    "projected_old_old_rebuilds",
    "orthogonalization_vectors",
    "kpoint_lane_solves",
    "representative_lane_solves",
    "partner_reconstructions",
    "padding_elements",
    "projector_cache_hits",
    "projector_cache_misses",
    "device_materializations",
    "device_materialized_arrays",
    "cpu_bridge_calls",
    "cpu_bridge_elements",
)
MEMORY_FIELD_NAMES = (
    "persistent_coefficient_bytes",
    "persistent_projector_bytes",
    "coefficient_payload_bytes",
    "projector_payload_bytes",
    "projector_traffic_bytes",
    "shared_full_grid_bytes",
    "peak_temporary_bytes",
    "fft_workspace_bytes",
    "hpsi_fft_workspace_bytes",
    "hpsi_peak_temporary_bytes",
    "process_high_water_bytes",
    "unified_memory_high_water_bytes",
)
PHASE_NAMES = (
    "setup",
    "hpsi",
    "orthogonalization",
    "rayleigh_ritz",
    "density",
    "mixing",
    "persistence",
    "eigensolver_control",
    "cpu_small_solve",
)

EventCallback = Callable[[dict[str, object]], None]
Synchronize = Callable[[], None]
Clock = Callable[[], float]


@dataclass
class RuntimeObserver:
    """Collect one runtime's flushed events and exclusive synchronized timings.

    Args:
        callback: Optional callback invoked synchronously for every event.
        synchronize: Optional MLX synchronization callable around measured phases.
        clock: Monotonic clock callable. Defaults to ``perf_counter``.
        detail_events: Whether callers should emit lane- and batch-level detail.
            Defaults to ``True``; full production runs can disable it in favor of
            collective progress summaries.
    """

    callback: EventCallback | None = None
    synchronize: Synchronize | None = None
    clock: Clock = perf_counter
    detail_events: bool = True
    _started: float = field(init=False, repr=False)
    _sequence: int = field(default=0, init=False, repr=False)
    _events: list[dict[str, object]] = field(default_factory=list, init=False, repr=False)
    _work: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _memory: dict[str, int | None] = field(default_factory=dict, init=False, repr=False)
    _phases: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _hpsi_shapes: dict[tuple[int, int], int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _phase_stack: list[list[object]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.detail_events) is not bool:
            msg = "detail_events must be a bool"
            raise ValueError(msg)
        self._started = self.clock()
        self._work = dict.fromkeys(WORK_COUNTER_NAMES, 0)
        self._memory = dict.fromkeys(MEMORY_FIELD_NAMES)
        self._phases = dict.fromkeys(PHASE_NAMES, 0.0)

    def emit(self, event: str, **fields: object) -> dict[str, object]:
        """Record and synchronously deliver one ordered progress event.

        Args:
            event: Stable event name.
            **fields: JSON-safe event-specific fields.

        Returns:
            The delivered event record.
        """

        if not event:
            msg = "runtime event name must be non-empty"
            raise ValueError(msg)
        self._sequence += 1
        record: dict[str, object] = {
            "schema_version": EVENT_SCHEMA,
            "sequence": self._sequence,
            "event": event,
            "elapsed_seconds": max(self.clock() - self._started, 0.0),
            **fields,
        }
        self._events.append(record)
        if self.callback is not None:
            self.callback(dict(record))
        return record

    def add_work(self, counter: str, amount: int = 1) -> None:
        """Increment one non-negative algorithmic work counter.

        Args:
            counter: Counter name from the frozen schema.
            amount: Non-negative integer increment. Defaults to ``1``.
        """

        if counter not in self._work:
            msg = f"unknown DFT runtime work counter: {counter}"
            raise ValueError(msg)
        if not isinstance(amount, int) or amount < 0:
            msg = "runtime work increments must be non-negative integers"
            raise ValueError(msg)
        self._work[counter] += amount

    def record_memory(self, field_name: str, byte_count: int | None) -> None:
        """Record one logical or observed memory field.

        Args:
            field_name: Memory field from the frozen schema.
            byte_count: Non-negative bytes, or ``None`` when unavailable.
        """

        if field_name not in self._memory:
            msg = f"unknown DFT runtime memory field: {field_name}"
            raise ValueError(msg)
        if byte_count is not None and (not isinstance(byte_count, int) or byte_count < 0):
            msg = "runtime memory values must be non-negative integer bytes"
            raise ValueError(msg)
        self._memory[field_name] = byte_count

    def record_peak_memory(self, field_name: str, byte_count: int) -> None:
        """Retain the largest observed value for one memory field.

        Args:
            field_name: Memory field from the frozen schema.
            byte_count: Non-negative candidate peak in bytes.
        """

        if field_name not in self._memory:
            msg = f"unknown DFT runtime memory field: {field_name}"
            raise ValueError(msg)
        if not isinstance(byte_count, int) or byte_count < 0:
            msg = "runtime peak-memory values must be non-negative integer bytes"
            raise ValueError(msg)
        current = self._memory[field_name]
        self._memory[field_name] = (
            byte_count if current is None else max(current, byte_count)
        )

    def record_hpsi_shape(self, lane_capacity: int, vector_capacity: int) -> None:
        """Count one physically submitted Hpsi lane/vector capacity.

        Args:
            lane_capacity: Positive physical lane count.
            vector_capacity: Positive physical vectors per lane.
        """

        if (
            type(lane_capacity) is not int
            or type(vector_capacity) is not int
            or lane_capacity <= 0
            or vector_capacity <= 0
        ):
            msg = "Hpsi shape capacities must be positive non-bool integers"
            raise ValueError(msg)
        shape = (lane_capacity, vector_capacity)
        self._hpsi_shapes[shape] = self._hpsi_shapes.get(shape, 0) + 1

    @contextmanager
    def phase(self, name: str, *, synchronize: bool = True) -> Iterator[None]:
        """Measure one exclusive synchronized named phase.

        Args:
            name: Phase name from the frozen schema.
            synchronize: Whether to place device barriers around this phase.
                Lane-local phases that already materialize their results should
                disable these redundant barriers. Defaults to ``True``.

        Yields:
            Control to the measured operation.
        """

        if name not in self._phases:
            msg = f"unknown DFT runtime phase: {name}"
            raise ValueError(msg)
        should_synchronize = synchronize and self.synchronize is not None
        if should_synchronize:
            self.synchronize()
        frame: list[object] = [name, self.clock(), 0.0]
        self._phase_stack.append(frame)
        try:
            yield
        finally:
            if should_synchronize:
                self.synchronize()
            elapsed = max(self.clock() - float(frame[1]), 0.0)
            child_elapsed = float(frame[2])
            exclusive = max(elapsed - child_elapsed, 0.0)
            self._phases[name] += exclusive
            popped = self._phase_stack.pop()
            if popped is not frame:
                msg = "runtime phase nesting was corrupted"
                raise RuntimeError(msg)
            if self._phase_stack:
                self._phase_stack[-1][2] = float(self._phase_stack[-1][2]) + elapsed

    def snapshot(self) -> dict[str, object]:
        """Return reconciled events, work, memory, and exclusive phase timings."""

        if self._phase_stack:
            msg = "cannot snapshot while a runtime phase is active"
            raise RuntimeError(msg)
        elapsed = max(self.clock() - self._started, 0.0)
        accounted = sum(self._phases.values())
        total = max(elapsed, accounted)
        phases = dict(self._phases)
        phases["unaccounted"] = max(total - accounted, 0.0)
        return {
            "schema_version": OBSERVATION_SCHEMA,
            "total_elapsed_seconds": total,
            "phase_seconds": phases,
            "work_counters": dict(self._work),
            "memory": dict(self._memory),
            "hpsi_shapes": [
                {
                    "lane_capacity": lanes,
                    "vector_capacity": vectors,
                    "calls": calls,
                }
                for (lanes, vectors), calls in sorted(self._hpsi_shapes.items())
            ],
            "events": [dict(event) for event in self._events],
        }


def observed_phase(
    observer: RuntimeObserver | None,
    name: str,
    *,
    synchronize: bool = True,
):
    """Return an observer phase or a no-op context manager.

    Args:
        observer: Optional runtime observer.
        name: Frozen phase name.
        synchronize: Whether the observer should place device barriers around
            the phase. Defaults to ``True``.

    Returns:
        Context manager suitable for a ``with`` statement.
    """

    return (
        nullcontext()
        if observer is None
        else observer.phase(name, synchronize=synchronize)
    )


def add_observed_work(
    observer: RuntimeObserver | None,
    counters: Mapping[str, int],
) -> None:
    """Increment several counters only when observation is enabled.

    Args:
        observer: Optional runtime observer.
        counters: Counter increments keyed by frozen counter name.
    """

    if observer is None:
        return
    for name, amount in counters.items():
        observer.add_work(name, amount)
