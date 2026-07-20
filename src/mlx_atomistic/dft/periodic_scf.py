"""Self-consistent weighted k-point plane-wave DFT."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft._compact import (
    _CompactBatch,
    _CompactLaneState,
    _CompatibilityCoefficientState,
    _remap_initial_coefficients,
    _require_layout,
)
from mlx_atomistic.dft._memory import _bounded_dft_allocator
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
from mlx_atomistic.dft.mixing import (
    LinearMixer,
    PulayDIISMixer,
    _MixerCheckpointState,
)
from mlx_atomistic.dft.periodic_gth import (
    PeriodicGTHNonlocalOperator,
    _GTHProjectorCache,
    gth_local_potential_grid,
    periodic_ewald_energy,
)
from mlx_atomistic.dft.plane_wave import PlaneWaveBasis
from mlx_atomistic.dft.potentials import hartree_potential
from mlx_atomistic.dft.pseudopotentials import PseudopotentialData
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional


def _eigensolve_provenance() -> dict[str, str]:
    return {
        "full_grid_precision": "complex64/float32",
        "projected_eigensolve_device": "cpu",
        "projected_eigensolve_backend": "numpy-lapack-cpu-complex128",
        "projected_eigensolve_precision": "complex128",
        "projected_eigensolve_output_precision": "float32/complex64",
    }


def _is_finite_positive_control(value: object) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and np.isfinite(float(value))
        and float(value) > 0.0
    )


def _time_reversed_compact_values(
    values: mx.array,
    permutation: np.ndarray,
) -> mx.array:
    """Map source compact coefficients into target time-reversal order."""

    mapping = np.asarray(permutation, dtype=np.int32)
    inverse = np.empty_like(mapping)
    inverse[mapping] = np.arange(mapping.size, dtype=np.int32)
    return mx.take(
        mx.conjugate(values),
        mx.array(inverse),
        axis=1,
    ).astype(mx.complex64)


@dataclass(frozen=True)
class PeriodicDFTSystem:
    """Orthorhombic periodic DFT system for a shared GTH pseudopotential.

    Args:
        cell_lengths: Orthorhombic cell lengths in bohr.
        grid_shape: FFT grid shape.
        positions: Ionic Cartesian positions in bohr.
        pseudopotential: Shared GTH pseudopotential for every ion.
        electron_count: Total valence electron count. Defaults to the neutral
            pseudopotential charge sum.
    """

    grid: RealSpaceGrid
    positions: np.ndarray
    pseudopotential: PseudopotentialData
    electron_count: float

    def __init__(
        self,
        cell_lengths: Sequence[float],
        grid_shape: Sequence[int],
        positions: Sequence[Sequence[float]],
        pseudopotential: PseudopotentialData,
        electron_count: float | None = None,
    ):
        positions_np = np.asarray(positions, dtype=np.float64)
        if positions_np.ndim != 2 or positions_np.shape[1] != 3 or positions_np.shape[0] == 0:
            msg = "positions must have shape (n_ions, 3)"
            raise ValueError(msg)
        count = (
            float(pseudopotential.valence_charge * positions_np.shape[0])
            if electron_count is None
            else float(electron_count)
        )
        if count <= 0.0:
            msg = "electron_count must be positive"
            raise ValueError(msg)
        object.__setattr__(self, "grid", RealSpaceGrid(grid_shape, cell_lengths))
        object.__setattr__(self, "positions", positions_np)
        object.__setattr__(self, "pseudopotential", pseudopotential)
        object.__setattr__(self, "electron_count", count)

    @property
    def ion_count(self) -> int:
        """Number of ions in the periodic cell."""

        return int(self.positions.shape[0])

    @property
    def charges(self) -> tuple[float, ...]:
        """Valence point charges used by the periodic Ewald term."""

        return tuple(float(self.pseudopotential.valence_charge) for _ in range(self.ion_count))


@dataclass(frozen=True)
class PeriodicDavidsonConfig:
    """Controls for the incremental block Davidson/Rayleigh-Ritz eigensolver."""

    max_iterations: int = 30
    tolerance: float = 1e-5
    max_subspace_size: int = 64
    preconditioner_floor: float = 0.5

    def __post_init__(self) -> None:
        if type(self.max_iterations) is not int or self.max_iterations <= 0:
            msg = "max_iterations must be a positive non-bool integer"
            raise ValueError(msg)
        if not _is_finite_positive_control(self.tolerance):
            msg = "tolerance must be finite and positive"
            raise ValueError(msg)
        if type(self.max_subspace_size) is not int or self.max_subspace_size <= 1:
            msg = "max_subspace_size must be a non-bool integer exceeding one"
            raise ValueError(msg)
        if not _is_finite_positive_control(self.preconditioner_floor):
            msg = "preconditioner_floor must be finite and positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class PeriodicSCFConfig:
    """Controls for weighted k-point self-consistent field iteration."""

    max_iterations: int = 40
    density_tolerance: float = 1e-5
    energy_tolerance: float = 1e-6
    orbital_tolerance: float = 1e-5
    min_iterations: int = 2
    mixing_beta: float = 0.35
    mixer: str = "diis"
    davidson: PeriodicDavidsonConfig = field(default_factory=PeriodicDavidsonConfig)
    kpoint_batch_size: int = 6
    max_batch_padding_fraction: float = _CompactBatch._DEFAULT_MAX_PADDING_FRACTION
    max_batch_transient_bytes: int = _CompactBatch._DEFAULT_MAX_TRANSIENT_BYTES

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            msg = "max_iterations must be positive"
            raise ValueError(msg)
        if self.density_tolerance <= 0.0 or self.energy_tolerance <= 0.0:
            msg = "SCF tolerances must be positive"
            raise ValueError(msg)
        if self.orbital_tolerance <= 0.0:
            msg = "orbital_tolerance must be positive"
            raise ValueError(msg)
        if self.min_iterations <= 0:
            msg = "min_iterations must be positive"
            raise ValueError(msg)
        if not 0.0 < self.mixing_beta <= 1.0:
            msg = "mixing_beta must lie in (0, 1]"
            raise ValueError(msg)
        if self.mixer not in {"linear", "diis"}:
            msg = "mixer must be 'linear' or 'diis'"
            raise ValueError(msg)
        if type(self.kpoint_batch_size) is not int or self.kpoint_batch_size <= 0:
            msg = "kpoint_batch_size must be a positive non-bool integer"
            raise ValueError(msg)
        if (
            not isinstance(self.max_batch_padding_fraction, int | float)
            or isinstance(self.max_batch_padding_fraction, bool)
            or not np.isfinite(float(self.max_batch_padding_fraction))
            or not 0.0 <= float(self.max_batch_padding_fraction) < 1.0
        ):
            msg = "max_batch_padding_fraction must be finite and lie in [0, 1)"
            raise ValueError(msg)
        if type(self.max_batch_transient_bytes) is not int or self.max_batch_transient_bytes <= 0:
            msg = "max_batch_transient_bytes must be a positive non-bool integer"
            raise ValueError(msg)

    def batch_policy(self) -> dict[str, int | float]:
        """Return the exact bounded compact-batch policy."""

        return {
            "kpoint_batch_size": self.kpoint_batch_size,
            "max_batch_padding_fraction": float(self.max_batch_padding_fraction),
            "max_batch_transient_bytes": self.max_batch_transient_bytes,
        }


@dataclass(frozen=True)
class _PeriodicSCFContinuationState:
    completed_iteration: int
    density: mx.array
    owned_coefficients: tuple[tuple[int, mx.array], ...]
    owned_lanes: tuple[dict[str, object], ...]
    previous_energy: float
    energy_by_term: dict[str, float]
    history: tuple[dict[str, float | int | str | None], ...]
    mixer_state: _MixerCheckpointState
    ownership: dict[str, object]
    lineage: tuple[str, ...] = ()

    @property
    def coefficient_map(self) -> dict[int, mx.array]:
        return dict(self.owned_coefficients)


@dataclass(frozen=True, init=False)
class PeriodicEigenResult:
    """Lowest eigenspace result with compact runtime-owned coefficients.

    Public construction accepts full-grid coefficients only with an explicit
    basis. Runtime code uses the private compact factory, so no dense fallback
    is retained.

    Args:
        eigenvalues: Lowest eigenvalues in Hartree.
        coefficients: Public full-grid coefficient stack.
        residuals: Direct ``H(X) - epsilon X`` norm per returned eigenpair.
        orthonormality_error: Maximum overlap error.
        iterations: Davidson iteration count.
        converged: Whether the requested tolerance was reached.
        subspace_size: Final Davidson subspace width.
        restart_count: Number of Davidson restarts.
        basis: Optional basis used to pack the public full-grid coefficient
            input. When omitted, the legacy eight-argument constructor stores
            only the input's exact nonzero support for round-trip compatibility.
    """

    eigenvalues: mx.array
    _compact_coefficients: _CompactLaneState | _CompatibilityCoefficientState | None
    _basis: PlaneWaveBasis | None
    _time_reversal_owner: PeriodicEigenResult | None
    _time_reversal_permutation: np.ndarray | None
    _time_reversal_observer: RuntimeObserver | None
    residuals: mx.array
    orthonormality_error: float
    iterations: int
    converged: bool
    subspace_size: int
    restart_count: int

    def __init__(
        self,
        eigenvalues: mx.array,
        coefficients: mx.array,
        residuals: mx.array,
        orthonormality_error: float,
        iterations: int,
        converged: bool,
        subspace_size: int,
        restart_count: int,
        *,
        basis: PlaneWaveBasis | None = None,
    ) -> None:
        compact: _CompactLaneState | _CompatibilityCoefficientState
        if basis is None:
            compact = _CompatibilityCoefficientState.from_full(coefficients)
        else:
            compact, _ = basis._state_from_full(coefficients)
        self._set_fields(
            eigenvalues=eigenvalues,
            compact_coefficients=compact,
            basis=basis,
            residuals=residuals,
            orthonormality_error=orthonormality_error,
            iterations=iterations,
            converged=converged,
            subspace_size=subspace_size,
            restart_count=restart_count,
        )

    def _set_fields(
        self,
        *,
        eigenvalues: mx.array,
        compact_coefficients: _CompactLaneState | _CompatibilityCoefficientState,
        basis: PlaneWaveBasis | None,
        residuals: mx.array,
        orthonormality_error: float,
        iterations: int,
        converged: bool,
        subspace_size: int,
        restart_count: int,
    ) -> None:
        if basis is None:
            if not isinstance(compact_coefficients, _CompatibilityCoefficientState):
                msg = "basis-free public results require compatibility coefficient state"
                raise ValueError(msg)
        else:
            if not isinstance(compact_coefficients, _CompactLaneState):
                msg = "basis-bound results require compact lane state"
                raise ValueError(msg)
            basis._validate_state(compact_coefficients)
            if compact_coefficients.kind != "coefficients":
                msg = "periodic eigen results must own coefficient state"
                raise ValueError(msg)
        object.__setattr__(self, "eigenvalues", mx.array(eigenvalues))
        object.__setattr__(self, "_compact_coefficients", compact_coefficients)
        object.__setattr__(self, "_basis", basis)
        object.__setattr__(self, "residuals", mx.array(residuals))
        object.__setattr__(self, "orthonormality_error", float(orthonormality_error))
        object.__setattr__(self, "iterations", int(iterations))
        object.__setattr__(self, "converged", bool(converged))
        object.__setattr__(self, "subspace_size", int(subspace_size))
        object.__setattr__(self, "restart_count", int(restart_count))
        object.__setattr__(self, "_time_reversal_owner", None)
        object.__setattr__(self, "_time_reversal_permutation", None)
        object.__setattr__(self, "_time_reversal_observer", None)

    @classmethod
    def _from_compact(
        cls,
        *,
        eigenvalues: mx.array,
        compact_coefficients: _CompactLaneState,
        basis: PlaneWaveBasis,
        residuals: mx.array,
        orthonormality_error: float,
        iterations: int,
        converged: bool,
        subspace_size: int,
        restart_count: int,
    ) -> PeriodicEigenResult:
        result = object.__new__(cls)
        result._set_fields(
            eigenvalues=eigenvalues,
            compact_coefficients=compact_coefficients,
            basis=basis,
            residuals=residuals,
            orthonormality_error=orthonormality_error,
            iterations=iterations,
            converged=converged,
            subspace_size=subspace_size,
            restart_count=restart_count,
        )
        return result

    @classmethod
    def _from_time_reversal_owner(
        cls,
        *,
        owner: PeriodicEigenResult,
        partner_basis: PlaneWaveBasis,
        permutation: np.ndarray,
        observer: RuntimeObserver | None,
    ) -> PeriodicEigenResult:
        owner_state = owner._compact_coefficients
        if not isinstance(owner_state, _CompactLaneState) or owner._basis is None:
            msg = "time-reversal views require a compact basis-bound owner"
            raise ValueError(msg)
        mapping = np.array(permutation, dtype=np.int32, copy=True)
        if (
            mapping.shape != (owner_state.layout.active_count,)
            or partner_basis.active_count != mapping.size
            or not np.array_equal(
                np.sort(mapping),
                np.arange(mapping.size, dtype=np.int32),
            )
        ):
            msg = "time-reversal permutation must be a complete active-basis bijection"
            raise ValueError(msg)
        mapping.setflags(write=False)
        result = object.__new__(cls)
        eigenvalues = mx.array(owner.eigenvalues)
        residuals = mx.array(owner.residuals)
        mx.eval(eigenvalues, residuals)
        object.__setattr__(result, "eigenvalues", eigenvalues)
        object.__setattr__(result, "_compact_coefficients", None)
        object.__setattr__(result, "_basis", partner_basis)
        object.__setattr__(result, "residuals", residuals)
        object.__setattr__(
            result,
            "orthonormality_error",
            owner.orthonormality_error,
        )
        object.__setattr__(result, "iterations", owner.iterations)
        object.__setattr__(result, "converged", owner.converged)
        object.__setattr__(result, "subspace_size", owner.subspace_size)
        object.__setattr__(result, "restart_count", owner.restart_count)
        object.__setattr__(result, "_time_reversal_owner", owner)
        object.__setattr__(result, "_time_reversal_permutation", mapping)
        object.__setattr__(result, "_time_reversal_observer", observer)
        return result

    @property
    def is_time_reversal_view(self) -> bool:
        """Whether coefficients are an uncached time-reversed owner view."""

        return self._time_reversal_owner is not None

    @property
    def coefficients(self) -> mx.array:
        """Materialize a fresh full-grid coefficient stack.

        Returns:
            Caller-owned ``complex64`` coefficients with exact inactive zeros.
        """

        if self._time_reversal_owner is None:
            if self._compact_coefficients is None:
                msg = "periodic eigen result has no coefficient state"
                raise RuntimeError(msg)
            return self._compact_coefficients.full_grid_fresh()
        owner_state = self._time_reversal_owner._compact_coefficients
        if (
            not isinstance(owner_state, _CompactLaneState)
            or self._time_reversal_permutation is None
            or self._basis is None
        ):
            msg = "time-reversal owner state is unavailable"
            raise RuntimeError(msg)
        values = _time_reversed_compact_values(
            owner_state.values,
            self._time_reversal_permutation,
        )
        add_observed_work(
            self._time_reversal_observer,
            {"partner_reconstructions": 1},
        )
        return self._basis._layout.unpack_fresh(values)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe eigensolver summary.

        Returns:
            Eigenvalues, residuals, convergence, and subspace diagnostics.
        """

        return {
            "eigenvalues_hartree": np.asarray(self.eigenvalues).tolist(),
            "residuals": np.asarray(self.residuals).tolist(),
            "orthonormality_error": self.orthonormality_error,
            "iterations": self.iterations,
            "converged": self.converged,
            "subspace_size": self.subspace_size,
            "restart_count": self.restart_count,
            "solver": "block-davidson-rayleigh-ritz",
            "dense_full_hamiltonian": False,
            **_eigensolve_provenance(),
            "full_grid_device": "default-mlx-device",
        }


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


@dataclass(frozen=True, init=False)
class PeriodicKohnShamOperator:
    """Fixed-density periodic Kohn-Sham operator in coefficient space."""

    basis: PlaneWaveBasis
    _effective_local_potential: mx.array = field(repr=False)
    nonlocal_operator: PeriodicGTHNonlocalOperator | None = None
    observer: RuntimeObserver | None = None

    def __init__(
        self,
        basis: PlaneWaveBasis,
        effective_local_potential: mx.array,
        nonlocal_operator: PeriodicGTHNonlocalOperator | None = None,
        observer: RuntimeObserver | None = None,
    ) -> None:
        potential_snapshot = mx.array(effective_local_potential)
        # Materialize an owned device buffer now so later caller mutation cannot
        # alter this fixed Hamiltonian through a lazy dependency.
        mx.eval(potential_snapshot)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(
            self,
            "_effective_local_potential",
            potential_snapshot,
        )
        object.__setattr__(self, "nonlocal_operator", nonlocal_operator)
        object.__setattr__(self, "observer", observer)

    @classmethod
    def _from_shared_potential(
        cls,
        basis: PlaneWaveBasis,
        potential_snapshot: mx.array,
        nonlocal_operator: PeriodicGTHNonlocalOperator | None = None,
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
        return self.basis._layout.unpack_fresh(
            applied.values,
            single=was_single,
        )

    def _apply_compact(
        self,
        coefficients: _CompactLaneState,
        *,
        observer: RuntimeObserver | None = None,
        max_padding_fraction: float = _CompactBatch._DEFAULT_MAX_PADDING_FRACTION,
        max_transient_bytes: int = _CompactBatch._DEFAULT_MAX_TRANSIENT_BYTES,
        prepared_batch: _CompactBatch | None = None,
    ) -> _CompactLaneState:
        outcome = self._apply_compact_batch(
            (self,),
            (coefficients,),
            observer=observer,
            max_padding_fraction=max_padding_fraction,
            max_transient_bytes=max_transient_bytes,
            prepared_batch=prepared_batch,
        )
        return outcome.action_for(0)

    @staticmethod
    def _estimated_batch_transient_bytes(
        operators: Sequence[PeriodicKohnShamOperator],
        batch: _CompactBatch,
    ) -> int:
        """Return the complete logical transient bound for one Hpsi batch."""

        if len(operators) > batch.lane_capacity:
            msg = "Hamiltonian operator count exceeds the compact capacity"
            raise ValueError(msg)
        lane_count = batch.lane_capacity
        padded_complex_bytes = lane_count * batch.vector_count * batch.bucket_size * 8
        kinetic_index_bytes = lane_count * batch.bucket_size * 4
        estimate = batch.estimated_transient_bytes + 5 * padded_complex_bytes + kinetic_index_bytes
        first_potential = operators[0]._effective_local_potential
        if not all(
            operator._effective_local_potential is first_potential for operator in operators
        ):
            estimate += lane_count * batch.grid_size * 8

        grouped_gth: dict[str, list[int]] = defaultdict(list)
        for index, operator in enumerate(operators):
            nonlocal_operator = operator.nonlocal_operator
            if isinstance(nonlocal_operator, PeriodicGTHNonlocalOperator):
                grouped_gth[nonlocal_operator._context_identity].append(index)
            elif nonlocal_operator is not None:
                estimate += padded_complex_bytes // lane_count
        for indices in grouped_gth.values():
            estimate += PeriodicGTHNonlocalOperator._estimated_batch_transient_bytes(
                [operators[index].nonlocal_operator for index in indices],
                batch,
            )
        return estimate

    @staticmethod
    def _apply_compact_batch(
        operators: Sequence[PeriodicKohnShamOperator],
        coefficients: Sequence[_CompactLaneState],
        *,
        observer: RuntimeObserver | None = None,
        max_padding_fraction: float = _CompactBatch._DEFAULT_MAX_PADDING_FRACTION,
        max_transient_bytes: int = _CompactBatch._DEFAULT_MAX_TRANSIENT_BYTES,
        prepared_batch: _CompactBatch | None = None,
    ) -> _CompactHamiltonianBatchResult:
        """Apply one bounded batch-first Hamiltonian submission."""

        if not operators or len(operators) != len(coefficients):
            msg = "compact Hamiltonian batches require matching non-empty lanes"
            raise ValueError(msg)
        if (
            not isinstance(max_padding_fraction, int | float)
            or isinstance(max_padding_fraction, bool)
            or not np.isfinite(float(max_padding_fraction))
            or not 0.0 <= float(max_padding_fraction) < 1.0
        ):
            msg = "max_padding_fraction must be finite and lie in [0, 1)"
            raise ValueError(msg)
        if type(max_transient_bytes) is not int or max_transient_bytes <= 0:
            msg = "max_transient_bytes must be a positive non-bool integer"
            raise ValueError(msg)
        operator_observers = {
            id(operator.observer): operator.observer
            for operator in operators
            if operator.observer is not None
        }
        if observer is None:
            if len(operator_observers) > 1:
                msg = "compact Hamiltonian batch operators must share one observer"
                raise ValueError(msg)
            runtime_observer = next(iter(operator_observers.values()), None)
        else:
            runtime_observer = observer

        failures: dict[int, Exception] = {}
        ready_indices: list[int] = []
        projector_actions: dict[int, _CompactLaneState] = {}
        projector_metrics: dict[int, dict[str, int]] = {}
        estimated_transient_bytes = 0
        executed_fft = False
        with observed_phase(runtime_observer, "hpsi"):
            for lane_index, (operator, state) in enumerate(
                zip(operators, coefficients, strict=True)
            ):
                try:
                    if (
                        runtime_observer is not None
                        and operator.observer is not None
                        and operator.observer is not runtime_observer
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
                    projector_metrics[lane_index] = _empty_projector_metrics()
                    if operator.nonlocal_operator is not None and not isinstance(
                        operator.nonlocal_operator,
                        PeriodicGTHNonlocalOperator,
                    ):
                        action, metrics = operator.nonlocal_operator._apply_compact(
                            state,
                            evaluate=False,
                        )
                        projector_actions[lane_index] = action
                        projector_metrics[lane_index] = metrics
                    ready_indices.append(lane_index)
                except Exception as error:
                    failures[lane_index] = _detached_failure(error)

            batch: _CompactBatch | None = None
            if ready_indices:
                ready_states = [coefficients[index] for index in ready_indices]
                try:
                    if (
                        prepared_batch is not None
                        and ready_indices == list(range(len(coefficients)))
                        and len(prepared_batch.layouts) == len(ready_states)
                        and all(
                            layout is state.layout
                            for layout, state in zip(
                                prepared_batch.layouts,
                                ready_states,
                                strict=True,
                            )
                        )
                        and prepared_batch.kinds == tuple(state.kind for state in ready_states)
                        and prepared_batch.vector_counts
                        == tuple(state.vector_count for state in ready_states)
                        and prepared_batch.estimated_transient_bytes <= max_transient_bytes
                        and max(
                            (prepared_batch.bucket_size - count) / prepared_batch.bucket_size
                            for count in prepared_batch.active_counts
                        )
                        <= max_padding_fraction
                    ):
                        batch = prepared_batch
                    else:
                        batch = _CompactBatch.from_states(
                            ready_states,
                            max_padding_fraction=max_padding_fraction,
                            max_transient_bytes=max_transient_bytes,
                        )
                    ready_operators = [operators[index] for index in ready_indices]
                    estimated_transient_bytes = (
                        PeriodicKohnShamOperator._estimated_batch_transient_bytes(
                            ready_operators,
                            batch,
                        )
                    )
                    if estimated_transient_bytes > max_transient_bytes:
                        msg = "compact Hpsi batch exceeds the complete transient byte budget"
                        raise ValueError(msg)
                    grouped_gth: dict[str, list[int]] = defaultdict(list)
                    for lane_index in ready_indices:
                        nonlocal_operator = operators[lane_index].nonlocal_operator
                        if isinstance(
                            nonlocal_operator,
                            PeriodicGTHNonlocalOperator,
                        ):
                            grouped_gth[nonlocal_operator._context_identity].append(lane_index)
                    for gth_indices in grouped_gth.values():
                        gth_states = [coefficients[index] for index in gth_indices]
                        if gth_indices == ready_indices:
                            gth_batch = batch
                        else:
                            gth_batch = _CompactBatch.from_states(
                                gth_states,
                                max_padding_fraction=max_padding_fraction,
                                max_transient_bytes=max_transient_bytes,
                            )
                        try:
                            gth_actions, gth_metrics = (
                                PeriodicGTHNonlocalOperator._apply_compact_batch(
                                    [operators[index].nonlocal_operator for index in gth_indices],
                                    gth_states,
                                    batch=gth_batch,
                                    evaluate=True,
                                )
                            )
                        except Exception:
                            for lane_index in gth_indices:
                                nonlocal_operator = operators[lane_index].nonlocal_operator
                                try:
                                    action, metrics = nonlocal_operator._apply_compact(
                                        coefficients[lane_index],
                                        evaluate=True,
                                    )
                                    projector_actions[lane_index] = action
                                    projector_metrics[lane_index] = metrics
                                except Exception as error:
                                    failures[lane_index] = _detached_failure(error)
                        else:
                            for lane_index, action, metrics in zip(
                                gth_indices,
                                gth_actions,
                                gth_metrics,
                                strict=True,
                            ):
                                projector_actions[lane_index] = action
                                projector_metrics[lane_index] = metrics

                    scattered = batch.scatter()
                    kinetic_rows = []
                    nonlocal_rows = []
                    for lane_index, state in zip(
                        ready_indices,
                        ready_states,
                        strict=True,
                    ):
                        padding = batch.bucket_size - state.layout.active_count
                        kinetic = state.layout._active_kinetic_energies
                        nonlocal_values = (
                            projector_actions[lane_index].values
                            if lane_index in projector_actions
                            else mx.zeros_like(state.values)
                        )
                        if padding:
                            kinetic = mx.concatenate(
                                [kinetic, mx.zeros((padding,), dtype=mx.float32)]
                            )
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
                        vector_padding = batch.vector_count - state.vector_count
                        if vector_padding:
                            nonlocal_values = mx.concatenate(
                                [
                                    nonlocal_values,
                                    mx.zeros(
                                        (vector_padding, batch.bucket_size),
                                        dtype=mx.complex64,
                                    ),
                                ],
                                axis=0,
                            )
                        kinetic_rows.append(kinetic)
                        nonlocal_rows.append(nonlocal_values)
                    lane_padding = batch.lane_capacity - batch.lane_count
                    kinetic_values = mx.stack(kinetic_rows, axis=0)
                    nonlocal_values = mx.stack(nonlocal_rows, axis=0)
                    if lane_padding:
                        kinetic_values = mx.concatenate(
                            [
                                kinetic_values,
                                mx.zeros(
                                    (lane_padding, batch.bucket_size),
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
                                        batch.vector_count,
                                        batch.bucket_size,
                                    ),
                                    dtype=mx.complex64,
                                ),
                            ],
                            axis=0,
                        )
                    kinetic_action = batch.values * kinetic_values[:, None, :]
                    first_potential = operators[ready_indices[0]]._effective_local_potential
                    if all(
                        operators[index]._effective_local_potential is first_potential
                        for index in ready_indices
                    ):
                        potentials = first_potential
                    else:
                        potentials = mx.stack(
                            [
                                operators[index]._effective_local_potential
                                for index in ready_indices
                            ],
                            axis=0,
                        )
                    local_action = batch.apply_local(
                        potentials,
                        scattered=scattered,
                    )
                    executed_fft = True
                    applied_values = kinetic_action + local_action + nonlocal_values
                    states = batch.unpad(
                        applied_values,
                        kind="hamiltonian_action",
                    )
                    finite = [mx.all(mx.isfinite(state.values)) for state in states]
                    mx.eval(*(state.values for state in states), *finite)
                    actions = {}
                    for lane_index, state, is_finite in zip(
                        ready_indices,
                        states,
                        finite,
                        strict=True,
                    ):
                        if lane_index in failures:
                            continue
                        if bool(is_finite):
                            actions[lane_index] = state
                        else:
                            failures[lane_index] = ValueError(
                                "Davidson Hamiltonian action must be finite"
                            )
                except Exception as error:
                    failure = _detached_failure(error)
                    actions = {}
                    for lane_index in ready_indices:
                        failures[lane_index] = failure
            else:
                actions = {}

        if batch is not None and executed_fft:
            logical_vector_count = batch.logical_vector_count
            generated = sum(
                projector_metrics[index]["projector_elements_generated"] for index in ready_indices
            )
            loaded = sum(
                projector_metrics[index]["projector_elements_loaded"] for index in ready_indices
            )
            traffic = sum(
                projector_metrics[index]["projector_traffic_elements"] for index in ready_indices
            )
            cache_hits = sum(
                projector_metrics[index]["projector_cache_hits"] for index in ready_indices
            )
            cache_misses = sum(
                projector_metrics[index]["projector_cache_misses"] for index in ready_indices
            )
            add_observed_work(
                runtime_observer,
                {
                    "hpsi_calls": 1,
                    "hpsi_vector_equivalents": logical_vector_count,
                    "fft_submissions": 2,
                    "fft_vector_equivalents": 2 * logical_vector_count,
                    "projector_elements_generated": generated,
                    "projector_elements_loaded": loaded,
                    "projector_traffic_elements": traffic,
                    "padding_elements": batch.padding_elements,
                    "projector_cache_hits": cache_hits,
                    "projector_cache_misses": cache_misses,
                },
            )
            if runtime_observer is not None:
                fft_workspace_bytes = (
                    2 * batch.lane_capacity * batch.vector_count * batch.grid_size * 8
                )
                runtime_observer.record_peak_memory(
                    "fft_workspace_bytes",
                    fft_workspace_bytes,
                )
                runtime_observer.record_peak_memory(
                    "peak_temporary_bytes",
                    estimated_transient_bytes,
                )
                runtime_observer.record_peak_memory(
                    "projector_payload_bytes",
                    sum(
                        projector_metrics[index]["projector_payload_elements"]
                        for index in ready_indices
                    )
                    * 8,
                )
                runtime_observer.record_memory(
                    "persistent_projector_bytes",
                    max(
                        (
                            projector_metrics[index]["projector_cache_bytes"]
                            for index in ready_indices
                        ),
                        default=0,
                    ),
                )
        return _CompactHamiltonianBatchResult(
            actions=actions,
            failures=failures,
            batch=batch if executed_fft else None,
        )

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


def _subspace_matrix(basis_vectors: mx.array, applied: mx.array) -> mx.array:
    matrix = mx.conjugate(basis_vectors) @ mx.transpose(applied)
    return 0.5 * (matrix + mx.conjugate(mx.transpose(matrix)))


def _all_finite(values: mx.array) -> bool:
    """Check a device array through one scalar synchronization."""

    return bool(mx.all(mx.isfinite(values)))


def _detached_failure(error: Exception) -> Exception:
    """Copy failure type/message without retaining traceback frame state."""

    error_type: type[Exception]
    if type(error).__module__ == "builtins":
        error_type = type(error)
        prefix = ""
    else:
        error_type = RuntimeError
        prefix = f"{type(error).__name__}: "
    try:
        message = prefix + str(error)
    except Exception:
        message = prefix + "failure message unavailable"
    try:
        detached = error_type(message)
    except Exception:
        detached = RuntimeError(message)
    detached.__traceback__ = None
    detached.__context__ = None
    detached.__cause__ = None
    return detached


@dataclass(frozen=True)
class _RankResult:
    """Rank-filtered vectors and their row transform from the input stack."""

    values: mx.array
    transform: mx.array
    deflated_count: int


@dataclass(frozen=True)
class _Complex64RankPolicy:
    """One deterministic rank and orthogonality policy for Davidson state."""

    relative_tolerance: float = 32.0 * float(np.finfo(np.float32).eps)

    def orthonormality_tolerance(self, vector_count: int) -> float:
        return 64.0 * float(np.finfo(np.float32).eps) * max(vector_count, 1)

    def guard_tolerance(self, vector_count: int) -> float:
        return 8.0 * float(np.finfo(np.float32).eps) * max(vector_count, 1)

    def overlap_error(self, values: mx.array) -> float:
        count = int(values.shape[0])
        overlap = values @ mx.conjugate(mx.transpose(values))
        # This is O(subspace**2); no active plane-wave axis crosses to NumPy.
        overlap_np = np.asarray(overlap, dtype=np.complex64)
        if not np.all(np.isfinite(overlap_np)):
            msg = "Davidson overlap matrix must be finite"
            raise ValueError(msg)
        return float(np.max(np.abs(overlap_np - np.eye(count))))

    def validate(self, values: mx.array, *, required_count: int) -> float:
        stack = mx.array(values).astype(mx.complex64)
        if len(stack.shape) != 2 or int(stack.shape[0]) < required_count:
            msg = "Davidson state has insufficient rank"
            raise ValueError(msg)
        if not _all_finite(stack):
            msg = "Davidson state must be finite"
            raise ValueError(msg)
        error = self.overlap_error(stack)
        if error > self.orthonormality_tolerance(int(stack.shape[0])):
            msg = "Davidson state violates the complex64 rank policy"
            raise ValueError(msg)
        return error

    def _try_batched_choleskyqr2(
        self,
        stack: mx.array,
        *,
        original_norms: np.ndarray,
        locked_count: int,
        required_count: int,
        limit: int,
    ) -> _RankResult | None:
        """Admit a well-resolved row block through CholeskyQR2."""

        input_count = int(stack.shape[0])
        candidate_count = min(input_count - locked_count, limit - locked_count)
        retained_count = locked_count + candidate_count
        if (
            candidate_count <= 0
            or candidate_count != input_count - locked_count
            or retained_count < required_count
        ):
            return None

        identity = mx.eye(input_count, dtype=mx.float32).astype(mx.complex64)
        locked_values = stack[:locked_count]
        locked_transforms = identity[:locked_count]
        candidate_values = stack[locked_count:retained_count]
        candidate_transforms = identity[locked_count:retained_count]
        for _ in range(2):
            overlaps = mx.conjugate(locked_values) @ mx.transpose(candidate_values)
            candidate_values = candidate_values - mx.transpose(overlaps) @ locked_values
            candidate_transforms = candidate_transforms - mx.transpose(overlaps) @ locked_transforms

        candidate_norms = original_norms[locked_count:retained_count]
        if not np.all(np.isfinite(candidate_norms)) or np.any(candidate_norms <= 0.0):
            return None
        scale = mx.array(candidate_norms.astype(np.float32))[:, None]
        candidate_values = candidate_values / scale
        candidate_transforms = candidate_transforms / scale

        def cholesky_step(values: mx.array) -> tuple[mx.array, np.ndarray] | None:
            gram = values @ mx.conjugate(mx.transpose(values))
            gram_np = np.asarray(gram, dtype=np.complex64).astype(np.complex128)
            if not np.all(np.isfinite(gram_np)):
                return None
            gram_np = 0.5 * (gram_np + np.conjugate(gram_np.T))
            try:
                lower = np.linalg.cholesky(gram_np)
                solve = np.linalg.solve(
                    lower,
                    np.eye(candidate_count, dtype=np.complex128),
                )
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(solve)):
                return None
            return mx.array(solve.astype(np.complex64)), lower

        first = cholesky_step(candidate_values)
        if first is None:
            return None
        first_solve, first_lower = first
        pivot_floor = 8.0 * float(np.sqrt(np.finfo(np.float32).eps))
        first_pivots = np.real(np.diag(first_lower))
        first_condition = np.linalg.cond(first_lower)
        if (
            np.any(first_pivots <= pivot_floor)
            or not np.isfinite(first_condition)
            or first_condition > 256.0
        ):
            return None
        candidate_values = first_solve @ candidate_values
        candidate_transforms = first_solve @ candidate_transforms

        second = cholesky_step(candidate_values)
        if second is None:
            return None
        second_solve, _second_lower = second
        candidate_values = second_solve @ candidate_values
        candidate_transforms = second_solve @ candidate_transforms

        result_values = mx.concatenate(
            [locked_values, candidate_values],
            axis=0,
        )
        result_transform = mx.concatenate(
            [locked_transforms, candidate_transforms],
            axis=0,
        )
        if self.overlap_error(result_values) > self.orthonormality_tolerance(retained_count):
            return None
        return _RankResult(
            values=result_values,
            transform=result_transform,
            deflated_count=input_count - retained_count,
        )

    def _sequential_mgs2(
        self,
        stack: mx.array,
        *,
        original_norms: np.ndarray,
        locked_count: int,
        required_count: int,
        limit: int,
    ) -> _RankResult:
        """Recover a rare unstable block projection with ordered MGS2."""

        input_count = int(stack.shape[0])
        identity = mx.eye(input_count, dtype=mx.float32).astype(mx.complex64)
        accepted = [stack[index] for index in range(locked_count)]
        transforms = [identity[index] for index in range(locked_count)]
        for index in range(locked_count, input_count):
            if len(accepted) >= limit:
                break
            original_norm = float(original_norms[index])
            if original_norm == 0.0:
                continue
            vector = stack[index]
            transform = identity[index]
            for _ in range(2):
                for accepted_vector, accepted_transform in zip(
                    accepted,
                    transforms,
                    strict=True,
                ):
                    overlap = mx.sum(mx.conjugate(accepted_vector) * vector)
                    vector = vector - overlap * accepted_vector
                    transform = transform - overlap * accepted_transform
            norm = float(mx.sqrt(mx.real(mx.sum(mx.conjugate(vector) * vector))))
            if not np.isfinite(norm):
                msg = "Davidson rank norm must be finite"
                raise ValueError(msg)
            if norm <= self.relative_tolerance * original_norm:
                continue
            accepted.append(vector / norm)
            transforms.append(transform / norm)

        retained_count = len(accepted)
        if retained_count < required_count:
            msg = (
                "Davidson rank policy retained "
                f"{retained_count} vectors but {required_count} are required"
            )
            raise ValueError(msg)
        result_values = mx.stack(accepted, axis=0)
        result_transform = mx.stack(transforms, axis=0)
        self.validate(result_values, required_count=required_count)
        return _RankResult(
            values=result_values,
            transform=result_transform,
            deflated_count=input_count - retained_count,
        )

    def orthonormalize(
        self,
        values: mx.array,
        *,
        locked_count: int = 0,
        required_count: int = 0,
        max_count: int | None = None,
    ) -> _RankResult:
        """Twice reorthogonalize in complex64 and deterministically deflate."""

        stack = mx.array(values).astype(mx.complex64)
        if len(stack.shape) != 2 or int(stack.shape[0]) == 0:
            msg = "Davidson rank input must be a non-empty matrix"
            raise ValueError(msg)
        input_count = int(stack.shape[0])
        if not 0 <= locked_count <= input_count:
            msg = "locked Davidson rank must lie within the input stack"
            raise ValueError(msg)
        limit = input_count if max_count is None else int(max_count)
        if limit < locked_count or limit <= 0:
            msg = "Davidson rank limit cannot discard locked vectors"
            raise ValueError(msg)
        if required_count < 0 or required_count > limit:
            msg = "required Davidson rank exceeds the rank limit"
            raise ValueError(msg)

        candidate_stack = stack[locked_count:]
        candidate_norms = mx.sqrt(
            mx.real(mx.sum(mx.conjugate(candidate_stack) * candidate_stack, axis=1))
        )
        candidate_norms_np = np.asarray(candidate_norms, dtype=np.float32)
        if not np.all(np.isfinite(candidate_norms_np)):
            msg = "Davidson rank input must be finite"
            raise ValueError(msg)
        original_norms_np = np.ones((input_count,), dtype=np.float32)
        original_norms_np[locked_count:] = candidate_norms_np

        accepted_values = stack[:locked_count]

        batched = self._try_batched_choleskyqr2(
            stack,
            original_norms=original_norms_np,
            locked_count=locked_count,
            required_count=required_count,
            limit=limit,
        )
        if batched is not None:
            return batched

        identity = mx.eye(input_count, dtype=mx.float32).astype(mx.complex64)
        accepted_transforms = identity[:locked_count]

        deflated = 0
        for index in range(locked_count, input_count):
            if int(accepted_values.shape[0]) >= limit:
                deflated += input_count - index
                break
            original_norm = float(original_norms_np[index])
            if original_norm == 0.0:
                deflated += 1
                continue
            vector = stack[index : index + 1]
            transform = identity[index : index + 1]
            for _ in range(2):
                if int(accepted_values.shape[0]) == 0:
                    break
                overlaps = mx.conjugate(accepted_values) @ mx.transpose(vector)
                vector = vector - mx.transpose(overlaps) @ accepted_values
                transform = transform - mx.transpose(overlaps) @ accepted_transforms
            norm = float(mx.sqrt(mx.real(mx.sum(mx.conjugate(vector) * vector))))
            if not np.isfinite(norm):
                msg = "Davidson rank norm must be finite"
                raise ValueError(msg)
            if norm <= self.relative_tolerance * original_norm:
                deflated += 1
                continue
            accepted_values = mx.concatenate(
                [accepted_values, vector / norm],
                axis=0,
            )
            accepted_transforms = mx.concatenate(
                [accepted_transforms, transform / norm],
                axis=0,
            )

        retained_count = int(accepted_values.shape[0])
        if retained_count < required_count:
            return self._sequential_mgs2(
                stack,
                original_norms=original_norms_np,
                locked_count=locked_count,
                required_count=required_count,
                limit=limit,
            )

        result_transform = accepted_transforms
        # Preserve the incrementally reorthogonalized values. Collapsing the
        # accumulated transform into one complex64 matmul can reintroduce a
        # deflated component through cancellation for nearly dependent rows.
        result_values = accepted_values
        if not _all_finite(result_values):
            msg = "Davidson state must be finite"
            raise ValueError(msg)
        overlap_error = self.overlap_error(result_values)
        if overlap_error > self.orthonormality_tolerance(retained_count):
            return self._sequential_mgs2(
                stack,
                original_norms=original_norms_np,
                locked_count=locked_count,
                required_count=required_count,
                limit=limit,
            )
        return _RankResult(
            values=result_values,
            transform=result_transform,
            deflated_count=deflated,
        )


_DAVIDSON_RANK_POLICY = _Complex64RankPolicy()


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
        "complex64-choleskyqr2-cgs2-mgs2-rank-v5",
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


@dataclass(frozen=True)
class _CompactSubmission:
    indices: tuple[int, ...]
    capacity: _CompactBatchCapacity | None = None


@dataclass(frozen=True)
class _CompactSubmissionPlan:
    submissions: tuple[_CompactSubmission, ...]
    failures: dict[int, Exception]
    compatibility_groups: tuple[tuple[int, ...], ...]


def _plan_compact_submissions(
    states: Sequence[_CompactLaneState],
    *,
    batch_cap: int,
    max_padding_fraction: float,
    max_transient_bytes: int,
    batch_byte_estimator: Callable[[tuple[int, ...], _CompactBatch], int] | None = None,
    capacity: _CompactBatchCapacity | None = None,
) -> _CompactSubmissionPlan:
    """Build deterministic active-count buckets within hard batch bounds."""

    if type(batch_cap) is not int or batch_cap <= 0:
        msg = "compact batch_cap must be a positive non-bool integer"
        raise ValueError(msg)
    if (
        not isinstance(max_padding_fraction, int | float)
        or isinstance(max_padding_fraction, bool)
        or not np.isfinite(float(max_padding_fraction))
        or not 0.0 <= float(max_padding_fraction) < 1.0
    ):
        msg = "compact max_padding_fraction must be finite and lie in [0, 1)"
        raise ValueError(msg)
    if type(max_transient_bytes) is not int or max_transient_bytes <= 0:
        msg = "compact max_transient_bytes must be a positive non-bool integer"
        raise ValueError(msg)
    if capacity is not None and (
        type(capacity.lanes) is not int
        or type(capacity.vectors) is not int
        or type(capacity.active) is not int
        or capacity.lanes <= 0
        or capacity.vectors <= 0
        or capacity.active <= 0
        or capacity.lanes > batch_cap
    ):
        msg = "compact capacity must contain positive bounded integers"
        raise ValueError(msg)

    capacity_prototype: _CompactBatch | None = None

    def build_candidate(indices: Sequence[int]) -> _CompactBatch:
        nonlocal capacity_prototype
        selected = [states[index] for index in indices]
        if capacity is None:
            candidate_batch = _CompactBatch.from_states(
                selected,
                max_padding_fraction=float(max_padding_fraction),
                max_transient_bytes=max_transient_bytes,
            )
        else:
            if len(selected) > capacity.lanes:
                msg = "compact candidate exceeds its stable lane capacity"
                raise ValueError(msg)
            if any(
                state.vector_count > capacity.vectors
                or state.layout.active_count > capacity.active
                or (capacity.active - state.layout.active_count) / capacity.active
                > max_padding_fraction
                for state in selected
            ):
                msg = "compact candidate exceeds its stable shape policy"
                raise ValueError(msg)
            if capacity_prototype is None:
                capacity_prototype = _CompactBatch.from_states(
                    selected[:1],
                    max_padding_fraction=float(max_padding_fraction),
                    max_transient_bytes=max_transient_bytes,
                    lane_capacity=capacity.lanes,
                    vector_capacity=capacity.vectors,
                    active_capacity=capacity.active,
                )
            candidate_batch = capacity_prototype
        estimated_bytes = (
            candidate_batch.estimated_transient_bytes
            if batch_byte_estimator is None
            else batch_byte_estimator(tuple(indices), candidate_batch)
        )
        if estimated_bytes > max_transient_bytes:
            msg = "compact batch exceeds the complete transient byte budget"
            raise ValueError(msg)
        return candidate_batch

    grouped: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        grouped[
            (
                id(state.layout.reciprocal),
                state.layout.grid_shape,
                (state.vector_count if capacity is None else capacity.vectors),
            )
        ].append(index)

    submissions: list[_CompactSubmission] = []
    failures: dict[int, Exception] = {}
    compatibility_groups: list[tuple[int, ...]] = []
    for compatible_indices in grouped.values():
        compatibility_groups.append(tuple(compatible_indices))
        ordered = sorted(
            compatible_indices,
            key=lambda index: (states[index].layout.active_count, index),
        )
        current: list[int] = []
        current_batch: _CompactBatch | None = None
        for index in ordered:
            if len(current) == batch_cap:
                if current_batch is None:
                    msg = "compact submission planner lost its active batch"
                    raise RuntimeError(msg)
                submissions.append(_CompactSubmission(tuple(current), capacity))
                current = []
                current_batch = None
            candidate = [*current, index]
            try:
                candidate_batch = build_candidate(candidate)
            except ValueError:
                if current:
                    if current_batch is None:
                        msg = "compact submission planner lost its bounded batch"
                        raise RuntimeError(msg) from None
                    submissions.append(_CompactSubmission(tuple(current), capacity))
                    current = []
                    current_batch = None
                try:
                    candidate_batch = build_candidate((index,))
                except ValueError as error:
                    failures[index] = _detached_failure(error)
                    continue
                current = [index]
                current_batch = candidate_batch
            else:
                current = candidate
                current_batch = candidate_batch
        if current:
            if current_batch is None:
                msg = "compact submission planner lost its final batch"
                raise RuntimeError(msg)
            submissions.append(_CompactSubmission(tuple(current), capacity))

    return _CompactSubmissionPlan(
        submissions=tuple(submissions),
        failures=failures,
        compatibility_groups=tuple(compatibility_groups),
    )


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


class _DavidsonScheduler:
    """Submit compatible ragged tickets under an explicit batch-cap policy."""

    def __init__(
        self,
        *,
        batch_cap: int = 1,
        max_padding_fraction: float = _CompactBatch._DEFAULT_MAX_PADDING_FRACTION,
        max_transient_bytes: int = _CompactBatch._DEFAULT_MAX_TRANSIENT_BYTES,
        submission_callback: _DavidsonSubmissionCallback | None = None,
    ) -> None:
        _plan_compact_submissions(
            (),
            batch_cap=batch_cap,
            max_padding_fraction=max_padding_fraction,
            max_transient_bytes=max_transient_bytes,
        )
        self._batch_cap = batch_cap
        self._max_padding_fraction = float(max_padding_fraction)
        self._max_transient_bytes = max_transient_bytes
        self._submission_callback = submission_callback
        self._submission_index = 0
        self._capacity_by_lane: dict[str, _CompactBatchCapacity] = {}

    @property
    def batch_cap(self) -> int:
        return self._batch_cap

    @property
    def max_padding_fraction(self) -> float:
        return self._max_padding_fraction

    @property
    def max_transient_bytes(self) -> int:
        return self._max_transient_bytes

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
        if not tickets:
            msg = "Davidson scheduler requires at least one application ticket"
            raise ValueError(msg)
        seen: set[str] = set()
        ready: list[_DavidsonApplicationTicket] = []
        failures: dict[str, Exception] = {}
        validated: list[tuple[_DavidsonApplicationTicket, mx.array]] = []
        for ticket in tickets:
            if ticket.lane_id in seen:
                msg = f"duplicate Davidson scheduler lane: {ticket.lane_id!r}"
                raise ValueError(msg)
            seen.add(ticket.lane_id)
            try:
                if ticket.purpose not in {"basis", "direct_validation"}:
                    msg = "Davidson application purpose is invalid"
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
                validated.append((ticket, mx.all(mx.isfinite(ticket.vectors.values))))
            except Exception as error:
                failures[ticket.lane_id] = _detached_failure(error)
        if validated:
            try:
                mx.eval(*(finite for _, finite in validated))
            except Exception as error:
                failure = _detached_failure(error)
                failures.update({ticket.lane_id: failure for ticket, _ in validated})
            else:
                for ticket, finite in validated:
                    if bool(finite):
                        ready.append(ticket)
                    else:
                        failures[ticket.lane_id] = ValueError(
                            "Davidson application block must be finite"
                        )

        grouped: dict[tuple[object, ...], list[_DavidsonApplicationTicket]] = defaultdict(list)
        for ticket in ready:
            grouped[self._group_key(ticket)].append(ticket)

        actions: dict[str, _CompactLaneState] = {}
        groups: list[tuple[str, ...]] = []
        compatibility_groups: list[tuple[str, ...]] = []
        for compatible in grouped.values():
            compatibility_groups.append(tuple(ticket.lane_id for ticket in compatible))

            def estimate_submission(
                indices: tuple[int, ...],
                batch: _CompactBatch,
                *,
                _compatible: list[_DavidsonApplicationTicket] = compatible,
            ) -> int:
                return PeriodicKohnShamOperator._estimated_batch_transient_bytes(
                    [_compatible[index].operator for index in indices],
                    batch,
                )

            capacity = self._capacity_by_lane.get(compatible[0].lane_id)
            if capacity is not None and any(
                self._capacity_by_lane.get(ticket.lane_id) != capacity for ticket in compatible
            ):
                msg = "Davidson stable-shape group has inconsistent capacities"
                raise RuntimeError(msg)
            plan = _plan_compact_submissions(
                [ticket.vectors for ticket in compatible],
                batch_cap=self.batch_cap,
                max_padding_fraction=self.max_padding_fraction,
                max_transient_bytes=self.max_transient_bytes,
                batch_byte_estimator=estimate_submission,
                capacity=capacity,
            )
            for lane_index, error in plan.failures.items():
                failures[compatible[lane_index].lane_id] = error
            for planned in plan.submissions:
                submission = tuple(compatible[index] for index in planned.indices)
                prepared_batch = _CompactBatch.from_states(
                    [ticket.vectors for ticket in submission],
                    max_padding_fraction=self.max_padding_fraction,
                    max_transient_bytes=self.max_transient_bytes,
                    lane_capacity=(None if planned.capacity is None else planned.capacity.lanes),
                    vector_capacity=(
                        None if planned.capacity is None else planned.capacity.vectors
                    ),
                    active_capacity=(None if planned.capacity is None else planned.capacity.active),
                )
                lane_ids = tuple(ticket.lane_id for ticket in submission)
                groups.append(lane_ids)
                submission_index = self._submission_index
                self._submission_index += 1
                if self._submission_callback is not None:
                    self._submission_callback(
                        "started",
                        submission_index,
                        submission,
                        prepared_batch,
                        {},
                    )
                submission_failures: dict[str, Exception] = {}
                try:
                    if len(submission) == 1:
                        ticket = submission[0]
                        apply_kwargs: dict[str, object] = {
                            "observer": ticket.observer,
                            "prepared_batch": prepared_batch,
                        }
                        if (
                            self.max_padding_fraction != _CompactBatch._DEFAULT_MAX_PADDING_FRACTION
                            or self.max_transient_bytes
                            != _CompactBatch._DEFAULT_MAX_TRANSIENT_BYTES
                        ):
                            apply_kwargs.update(
                                max_padding_fraction=self.max_padding_fraction,
                                max_transient_bytes=self.max_transient_bytes,
                            )
                        applied = ticket.operator._apply_compact(
                            ticket.vectors,
                            **apply_kwargs,
                        )
                        batch_actions = {ticket.lane_id: applied}
                    else:
                        outcome = PeriodicKohnShamOperator._apply_compact_batch(
                            tuple(ticket.operator for ticket in submission),
                            tuple(ticket.vectors for ticket in submission),
                            observer=self._observer(submission[0]),
                            max_padding_fraction=self.max_padding_fraction,
                            max_transient_bytes=self.max_transient_bytes,
                            prepared_batch=prepared_batch,
                        )
                        batch_actions = {
                            ticket.lane_id: outcome.actions[index]
                            for index, ticket in enumerate(submission)
                            if index in outcome.actions
                        }
                        submission_failures.update(
                            {
                                submission[index].lane_id: error
                                for index, error in outcome.failures.items()
                            }
                        )
                    for ticket in submission:
                        applied = batch_actions.get(ticket.lane_id)
                        if applied is None:
                            continue
                        actions[ticket.lane_id] = applied
                        if ticket.purpose == "basis":
                            add_observed_work(
                                ticket.observer,
                                {"davidson_hv_new_vectors": (ticket.vectors.vector_count)},
                            )
                except Exception as error:
                    failure = _detached_failure(error)
                    submission_failures.update({ticket.lane_id: failure for ticket in submission})
                failures.update(submission_failures)
                if self._submission_callback is not None:
                    self._submission_callback(
                        "completed",
                        submission_index,
                        submission,
                        prepared_batch,
                        submission_failures,
                    )
        return _DavidsonScheduleResult(
            actions=actions,
            failures=failures,
            groups=tuple(groups),
            compatibility_groups=tuple(compatibility_groups),
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
        if not _all_finite(matrix):
            msg = "Davidson projected matrix must be finite"
            raise ValueError(msg)
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


def _ritz_residual_data(
    eigenvalues: mx.array,
    vectors: _CompactLaneState,
    applied: _CompactLaneState,
) -> tuple[mx.array, mx.array, float]:
    residual_stack = applied.values - eigenvalues[:, None] * vectors.values
    residuals = mx.sqrt(mx.sum(mx.abs(residual_stack) ** 2, axis=1))
    max_residual_array = mx.max(residuals)
    finite = mx.all(mx.isfinite(eigenvalues)) & mx.all(mx.isfinite(residuals))
    mx.eval(max_residual_array, finite)
    if not bool(finite):
        msg = "Davidson Ritz data must be finite"
        raise ValueError(msg)
    return residual_stack, residuals, float(max_residual_array)


def _ritz_pair(state: _PairedDavidsonState, n_bands: int) -> _DavidsonRitzPair:
    values, eigenvectors = _projected_eigh(state.projected)
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
    residual_stack, residuals, max_residual = _ritz_residual_data(
        selected_values,
        vectors,
        applied,
    )
    return _DavidsonRitzPair(
        eigenvalues=selected_values,
        vectors=vectors,
        applied=applied,
        residual_stack=residual_stack,
        residuals=residuals,
        max_residual=max_residual,
        transform=transform,
    )


def _ritz_pair_with_direct_action(
    candidate: _DavidsonRitzPair,
    applied: _CompactLaneState,
) -> _DavidsonRitzPair:
    """Return one Ritz pair whose residuals use the exact scheduled H(X)."""

    _require_layout(applied, candidate.vectors.layout)
    if applied.kind != "hamiltonian_action":
        msg = "Davidson direct validation requires Hamiltonian actions"
        raise ValueError(msg)
    if applied.vector_count != candidate.vectors.vector_count:
        msg = "Davidson direct validation width does not match its Ritz vectors"
        raise ValueError(msg)
    residual_stack, residuals, max_residual = _ritz_residual_data(
        candidate.eigenvalues,
        candidate.vectors,
        applied,
    )
    return _DavidsonRitzPair(
        eigenvalues=candidate.eigenvalues,
        vectors=candidate.vectors,
        applied=applied,
        residual_stack=residual_stack,
        residuals=residuals,
        max_residual=max_residual,
        transform=candidate.transform,
    )


def _projected_eigh(matrix: mx.array) -> tuple[mx.array, mx.array]:
    # Only the small projected Rayleigh-Ritz matrix crosses to the CPU. LAPACK's
    # complex128 solve avoids the complex64 convergence floor while every
    # full-grid operator, residual, and FFT remains on the default MLX device.
    projected = np.asarray(matrix, dtype=np.complex128)
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
    direct_validated: bool = False
    converged: bool = False
    done: bool = False
    failure: Exception | None = None


@dataclass(frozen=True)
class _DavidsonEngineResult:
    """Independent lane outcomes and actual shared-engine scheduling evidence."""

    results: dict[str, PeriodicEigenResult]
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


class _DavidsonEngine:
    """Advance ragged Davidson lanes and collectively schedule ready H blocks."""

    def __init__(self, *, scheduler: _DavidsonScheduler | None = None) -> None:
        self.scheduler = _DavidsonScheduler(batch_cap=1) if scheduler is None else scheduler
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
        with observed_phase(request.observer, "orthogonalization"):
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
        if request.observer is not None:
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
            with observed_phase(request.observer, "orthogonalization"):
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
    def _prepare_correction(
        progress: _DavidsonLaneProgress,
        ritz_pair: _DavidsonRitzPair,
    ) -> _DavidsonPendingAction | None:
        request = progress.request
        paired = progress.paired
        if paired is None:
            msg = "Davidson lane has no paired V/HV state"
            raise RuntimeError(msg)
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

        with observed_phase(request.observer, "orthogonalization"):
            append_rank = request.rank_policy.orthonormalize(
                mx.concatenate([paired.vectors.values, raw_corrections], axis=0),
                locked_count=paired.vector_count,
                required_count=paired.vector_count,
            )
        add_observed_work(
            request.observer,
            {"orthogonalization_vectors": int(unconverged.size)},
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

        if paired.vector_count + correction_count > request.config.max_subspace_size:
            progress.restart_count += 1
            with observed_phase(request.observer, "orthogonalization"):
                restart_rank = request.rank_policy.orthonormalize(
                    ritz_pair.vectors.values,
                    required_count=request.n_bands,
                    max_count=request.n_bands,
                )
                paired = paired.rebase_ranked(
                    restart_rank,
                    ritz_pair.applied,
                    token=progress.token,
                )
                progress.paired = paired
            add_observed_work(
                request.observer,
                {"orthogonalization_vectors": request.n_bands},
            )
            if request.config.max_subspace_size - paired.vector_count <= 0:
                if progress.direct_validated:
                    progress.done = True
                    return None
                return _DavidsonEngine._prepare_direct_validation(
                    progress,
                    ritz_pair,
                    terminal=True,
                )
            with observed_phase(request.observer, "orthogonalization"):
                restarted_append = request.rank_policy.orthonormalize(
                    mx.concatenate(
                        [paired.vectors.values, correction_values],
                        axis=0,
                    ),
                    locked_count=paired.vector_count,
                    required_count=paired.vector_count,
                    max_count=request.config.max_subspace_size,
                )
            add_observed_work(
                request.observer,
                {"orthogonalization_vectors": correction_count},
            )
            correction_values = restarted_append.values[paired.vector_count :]
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
    def _advance_lane(
        progress: _DavidsonLaneProgress,
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
        with observed_phase(request.observer, "rayleigh_ritz"):
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
    def _consume_action(
        progress: _DavidsonLaneProgress,
        ticket: _DavidsonApplicationTicket,
        applied: _CompactLaneState,
    ) -> _DavidsonPendingAction | None:
        pending = progress.pending_action
        if pending is None or pending.vectors is not ticket.vectors:
            msg = "Davidson scheduled action does not match lane state"
            raise RuntimeError(msg)
        progress.pending_action = None
        if pending.purpose == "correction":
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
        direct_pair = _ritz_pair_with_direct_action(candidate, applied)
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

        progress_by_lane: dict[str, _DavidsonLaneProgress] = {}
        failures: dict[str, Exception] = {}
        initial_tickets: list[_DavidsonApplicationTicket] = []
        for request in requests:
            try:
                progress = self._prepare_lane(request)
                progress_by_lane[request.lane_id] = progress
                initial_tickets.append(self._ticket(progress, progress.initial_vectors))
            except Exception as error:
                failures[request.lane_id] = _detached_failure(error)

        if initial_tickets:
            self.scheduler.bind(initial_tickets)
            initial_schedule = self._schedule(initial_tickets)
            for ticket in initial_tickets:
                lane_id = ticket.lane_id
                progress = progress_by_lane[lane_id]
                try:
                    progress.paired = _PairedDavidsonState.initialize(
                        progress.initial_vectors,
                        initial_schedule.action_for(lane_id),
                        progress.token,
                    )
                except Exception as error:
                    self._fail_lane(progress, failures, error)

        while True:
            active = [
                progress
                for lane_id, progress in progress_by_lane.items()
                if lane_id not in failures and not progress.done
            ]
            if not active:
                break
            pending_tickets: list[_DavidsonApplicationTicket] = []
            for progress in active:
                try:
                    pending = self._advance_lane(progress)
                    if pending is not None:
                        pending_tickets.append(
                            self._ticket(
                                progress,
                                pending.vectors,
                                purpose=(
                                    "direct_validation"
                                    if pending.purpose == "direct_validation"
                                    else "basis"
                                ),
                            )
                        )
                except Exception as error:
                    self._fail_lane(progress, failures, error)
            if not pending_tickets:
                continue

            while pending_tickets:
                scheduled = self._schedule(pending_tickets)
                followup_tickets: list[_DavidsonApplicationTicket] = []
                for ticket in pending_tickets:
                    lane_id = ticket.lane_id
                    failure = scheduled.failures.get(lane_id)
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
                        )
                        if followup is not None:
                            followup_tickets.append(
                                self._ticket(
                                    progress,
                                    followup.vectors,
                                    purpose=(
                                        "direct_validation"
                                        if followup.purpose == "direct_validation"
                                        else "basis"
                                    ),
                                )
                            )
                    except Exception as error:
                        self._fail_lane(progress, failures, error)
                pending_tickets = followup_tickets

        results: dict[str, PeriodicEigenResult] = {}
        for lane_id, progress in progress_by_lane.items():
            if lane_id in failures:
                continue
            try:
                results[lane_id] = self._finalize_lane(progress)
            except Exception as error:
                self._fail_lane(progress, failures, error)
        return _DavidsonEngineResult(
            results=results,
            failures=failures,
            ready_rounds=tuple(self._ready_rounds),
            compatibility_groups=tuple(self._compatibility_groups),
            submission_groups=tuple(self._submission_groups),
            scheduler_calls=self._scheduler_calls,
        )


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
    )
    engine = _DavidsonEngine(scheduler=_DavidsonScheduler(batch_cap=1))
    return engine.solve([request]).result_for(lane_id)


@dataclass(frozen=True)
class PeriodicKPointResult:
    """One weighted k-point result in a periodic SCF calculation."""

    reduced_kpoint: tuple[float, float, float]
    weight: float
    basis: PlaneWaveBasis
    eigen: PeriodicEigenResult
    explicit_index: int | None = None
    aggregated_weight: float | None = None
    ownership_role: str = "independent"
    fallback_reason: str | None = None

    @property
    def integration_weight(self) -> float:
        """Return the owner-aggregated or original integration weight."""

        return self.weight if self.aggregated_weight is None else self.aggregated_weight

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe k-point summary.

        Returns:
            Reduced k-point, weight, basis metadata, and eigensolver summary.
        """

        return {
            "reduced_kpoint": list(self.reduced_kpoint),
            "weight": self.weight,
            "basis": self.basis.to_dict(),
            "eigensolver": self.eigen.to_dict(),
        }


@dataclass(frozen=True)
class _TimeReversalContinuationSeed:
    owner_index: int


@dataclass(frozen=True)
class PeriodicSCFResult:
    """Result bundle for a weighted periodic plane-wave SCF calculation."""

    converged: bool
    status: str
    iterations: int
    total_energy: float
    electron_count: float
    density_residual: float
    energy_delta: float | None
    density: mx.array
    kpoints: tuple[PeriodicKPointResult, ...]
    energy_by_term: dict[str, float]
    history: tuple[dict[str, float | int | str | None], ...]
    timings: dict[str, float]
    time_reversal_ownership: TimeReversalOwnership | None = None
    batch_policy: dict[str, int | float] = field(default_factory=dict)
    numerical_status: str = "not_evaluated"
    resume_integrity_status: str = "fresh"
    timing_admission_status: str = "fresh"
    lineage: tuple[str, ...] = ()
    _owned_kpoints: tuple[PeriodicKPointResult, ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _checkpoint_state: _PeriodicSCFContinuationState | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _artifact_execution_contract_fingerprint: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _artifact_calculation_fingerprint: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def owned_kpoints(self) -> tuple[PeriodicKPointResult, ...]:
        """Return the compact-state-owning k-point results.

        Legacy manually constructed results without ownership metadata return
        their explicit k-point tuple unchanged.
        """

        return self.kpoints if self._owned_kpoints is None else self._owned_kpoints

    @property
    def continuation_coefficients(self) -> tuple[object, ...]:
        """Return an explicit owner-aware initial-coefficient sequence.

        Owner and independent entries reference their compact state. Admitted
        partners use lightweight time-reversal descriptors, so constructing the
        sequence neither materializes nor retains partner coefficients.
        """

        if self.time_reversal_ownership is None:
            return tuple(item.eigen._compact_coefficients for item in self.kpoints)
        owned = {
            item.explicit_index: item.eigen._compact_coefficients for item in self.owned_kpoints
        }
        return tuple(
            owned[entry.explicit_index]
            if entry.owner_index == entry.explicit_index
            else _TimeReversalContinuationSeed(entry.owner_index)
            for entry in self.time_reversal_ownership.entries
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe periodic SCF summary.

        Returns:
            Convergence, energy, k-point, history, and timing diagnostics without
            dense orbital or density payloads.
        """

        return {
            "converged": self.converged,
            "status": self.status,
            "iterations": self.iterations,
            "total_energy_hartree": self.total_energy,
            "electron_count": self.electron_count,
            "density_residual": self.density_residual,
            "energy_delta_hartree": self.energy_delta,
            "kpoints": [result.to_dict() for result in self.kpoints],
            "energy_by_term_hartree": dict(self.energy_by_term),
            "history": [dict(row) for row in self.history],
            "timings_ms": dict(self.timings),
            "batch_policy": dict(self.batch_policy),
            "numerical_status": self.numerical_status,
            "resume_integrity_status": self.resume_integrity_status,
            "timing_admission_status": self.timing_admission_status,
            "lineage": list(self.lineage),
            "dense_full_hamiltonian": False,
        }


def _density_from_kpoints(
    results: Sequence[PeriodicKPointResult],
    *,
    occupation: float,
    batch_cap: int = 1,
    max_padding_fraction: float = _CompactBatch._DEFAULT_MAX_PADDING_FRACTION,
    max_transient_bytes: int = _CompactBatch._DEFAULT_MAX_TRANSIENT_BYTES,
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
            lane_capacity=batch_cap,
            vector_capacity=vector_capacity,
            max_padding_fraction=max_padding_fraction,
        )
        for capacity_indices, capacity in capacity_groups:
            capacity_states = [states[index] for index in capacity_indices]
            plan = _plan_compact_submissions(
                capacity_states,
                batch_cap=batch_cap,
                max_padding_fraction=max_padding_fraction,
                max_transient_bytes=max_transient_bytes,
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
                    max_padding_fraction=max_padding_fraction,
                    max_transient_bytes=max_transient_bytes,
                    lane_capacity=capacity.lanes,
                    vector_capacity=capacity.vectors,
                    active_capacity=capacity.active,
                )
                estimated_transient_bytes = estimate_density_batch(
                    logical_indices,
                    batch,
                )
                if estimated_transient_bytes > max_transient_bytes:
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
                system.pseudopotential,
                bases[point_index],
                system.positions,
                cache=projector_cache,
            )
            for point_index in owned_indices
        }
        local_potential = gth_local_potential_grid(
            system.pseudopotential,
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
            if observer is None:
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

        requests = tuple(
            _DavidsonLaneRequest(
                lane_id=bases[point_index]._layout.lane_id,
                operator=operators_by_index[point_index],
                n_bands=occupied_bands,
                config=scf_config.davidson,
                trial=_initial_trial(
                    bases[point_index],
                    occupied_bands,
                    previous_states.get(point_index),
                ),
                observer=observer,
            )
            for point_index in owned_indices
        )
        engine = _DavidsonEngine(
            scheduler=_DavidsonScheduler(
                batch_cap=scf_config.kpoint_batch_size,
                max_padding_fraction=scf_config.max_batch_padding_fraction,
                max_transient_bytes=scf_config.max_batch_transient_bytes,
                submission_callback=emit_submission,
            )
        )
        eigen_outcome = engine.solve(requests)
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
                        lane_id: str(error) for lane_id, error in eigen_outcome.failures.items()
                    },
                )
            first_failed_lane = next(
                request.lane_id for request in requests if request.lane_id in eigen_outcome.failures
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
                batch_cap=scf_config.kpoint_batch_size,
                max_padding_fraction=scf_config.max_batch_padding_fraction,
                max_transient_bytes=scf_config.max_batch_transient_bytes,
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


def run_periodic_scf(
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
) -> PeriodicSCFResult:
    """Run weighted self-consistent periodic plane-wave DFT.

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

    Returns:
        Periodic SCF result with complete weighted k-point diagnostics.
    """

    return _run_periodic_scf_controlled(
        system,
        cutoff_hartree=cutoff_hartree,
        kpoint_mesh=kpoint_mesh,
        n_bands=n_bands,
        config=config,
        xc_functional=xc_functional,
        initial_density=initial_density,
        initial_coefficients=initial_coefficients,
        observer=observer,
    )
