"""Self-consistent weighted k-point plane-wave DFT."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
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
from mlx_atomistic.dft._runtime_observer import (
    RuntimeObserver,
    add_observed_work,
    observed_phase,
)
from mlx_atomistic.dft.gga import ProductionPBEExchangeCorrelation
from mlx_atomistic.dft.grids import RealSpaceGrid, ReciprocalGrid
from mlx_atomistic.dft.kpoints import KPointMesh
from mlx_atomistic.dft.mixing import LinearMixer, PulayDIISMixer
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


@dataclass(frozen=True, init=False)
class PeriodicEigenResult:
    """Lowest eigenspace result with compact runtime-owned coefficients.

    Public construction accepts full-grid coefficients only with an explicit
    basis. Runtime code uses the private compact factory, so no dense fallback
    is retained.

    Args:
        eigenvalues: Lowest eigenvalues in Hartree.
        coefficients: Public full-grid coefficient stack.
        residuals: Residual norm per eigenpair.
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
    _compact_coefficients: _CompactLaneState | _CompatibilityCoefficientState
    _basis: PlaneWaveBasis | None
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

    @property
    def coefficients(self) -> mx.array:
        """Materialize a fresh full-grid coefficient stack.

        Returns:
            Caller-owned ``complex64`` coefficients with exact inactive zeros.
        """

        return self._compact_coefficients.full_grid_fresh()

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


def _compact_hpsi_memory(
    *,
    vector_count: int,
    grid_count: int,
    projector_workspace_bytes: int,
) -> tuple[int, int]:
    """Return compact FFT and GTH workspace bytes for one Hpsi batch."""

    fft_workspace_bytes = 2 * vector_count * grid_count * 8
    peak_temporary_bytes = fft_workspace_bytes + projector_workspace_bytes
    return fft_workspace_bytes, peak_temporary_bytes


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
        return self.basis._layout.unpack_fresh(applied.values, single=was_single)

    def _apply_compact(
        self,
        coefficients: _CompactLaneState,
        *,
        observer: RuntimeObserver | None = None,
    ) -> _CompactLaneState:
        if observer is not None and self.observer is not None and observer is not self.observer:
            msg = "operator apply observers must be the same object"
            raise ValueError(msg)
        self.basis._validate_state(coefficients)
        if coefficients.kind != "coefficients":
            msg = "Hamiltonian input must be coefficient state"
            raise ValueError(msg)
        runtime_observer = self.observer if observer is None else observer
        vector_count = coefficients.vector_count
        projector_metrics = {
            "projector_payload_elements": 0,
            "projector_elements_generated": 0,
            "projector_elements_loaded": 0,
            "projector_traffic_elements": 0,
            "projector_cache_hits": 0,
            "projector_cache_misses": 0,
            "projector_cache_bytes": 0,
            "projector_peak_workspace_bytes": 0,
        }
        if runtime_observer is not None:
            fft_workspace_bytes, _ = _compact_hpsi_memory(
                vector_count=vector_count,
                grid_count=self.basis.grid.size,
                projector_workspace_bytes=0,
            )
            runtime_observer.record_peak_memory(
                "fft_workspace_bytes",
                fft_workspace_bytes,
            )
        with observed_phase(runtime_observer, "hpsi"):
            batch = _CompactBatch.from_states([coefficients])
            scattered = batch.scatter()
            kinetic = (
                coefficients.values
                * self.basis._layout._active_kinetic_energies[None, :]
            )
            local = batch.unpad(
                batch.apply_local(
                    self._effective_local_potential,
                    scattered=scattered,
                ),
                kind="hamiltonian_action",
            )[0]
            applied_values = kinetic + local.values
            if self.nonlocal_operator is not None:
                nonlocal_action, projector_metrics = (
                    self.nonlocal_operator._apply_compact(coefficients)
                )
                applied_values = applied_values + nonlocal_action.values
            result = self.basis._state_from_compact(
                applied_values,
                kind="hamiltonian_action",
            )
        add_observed_work(
            runtime_observer,
            {
                "hpsi_calls": 1,
                "hpsi_vector_equivalents": vector_count,
                "fft_submissions": 2,
                "fft_vector_equivalents": 2 * vector_count,
                "projector_elements_generated": projector_metrics[
                    "projector_elements_generated"
                ],
                "projector_elements_loaded": projector_metrics[
                    "projector_elements_loaded"
                ],
                "projector_traffic_elements": projector_metrics[
                    "projector_traffic_elements"
                ],
                "projector_cache_hits": projector_metrics["projector_cache_hits"],
                "projector_cache_misses": projector_metrics[
                    "projector_cache_misses"
                ],
            },
        )
        if runtime_observer is not None:
            _, peak_temporary_bytes = _compact_hpsi_memory(
                vector_count=vector_count,
                grid_count=self.basis.grid.size,
                projector_workspace_bytes=projector_metrics[
                    "projector_peak_workspace_bytes"
                ],
            )
            runtime_observer.record_peak_memory(
                "peak_temporary_bytes",
                peak_temporary_bytes,
            )
            runtime_observer.record_memory(
                "projector_payload_bytes",
                projector_metrics["projector_payload_elements"] * 8,
            )
            runtime_observer.record_memory(
                "persistent_projector_bytes",
                projector_metrics["projector_cache_bytes"],
            )
        return result

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
        if not _all_finite(stack):
            msg = "Davidson rank input must be finite"
            raise ValueError(msg)

        accepted = [stack[index] for index in range(locked_count)]
        identity = mx.eye(input_count, dtype=mx.float32).astype(mx.complex64)
        transforms = [identity[index] for index in range(locked_count)]
        if locked_count:
            self.validate(mx.stack(accepted, axis=0), required_count=locked_count)

        deflated = 0
        for index in range(locked_count, input_count):
            if len(accepted) >= limit:
                deflated += input_count - index
                break
            vector = stack[index]
            transform = identity[index]
            original_norm = float(
                mx.sqrt(mx.real(mx.sum(mx.conjugate(vector) * vector)))
            )
            if not np.isfinite(original_norm):
                msg = "Davidson rank norm must be finite"
                raise ValueError(msg)
            if original_norm == 0.0:
                deflated += 1
                continue
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
                deflated += 1
                continue
            accepted.append(vector / norm)
            transforms.append(transform / norm)

        if len(accepted) < required_count:
            msg = (
                "Davidson rank policy retained "
                f"{len(accepted)} vectors but {required_count} are required"
            )
            raise ValueError(msg)
        result_values = mx.stack(accepted, axis=0)
        result_transform = mx.stack(transforms, axis=0)
        self.validate(result_values, required_count=required_count)
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
        "complex64-mgs2-rank-v1",
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
class _DavidsonApplicationTicket:
    """One lane's newly accepted block awaiting its only H application."""

    lane_id: str
    operator: PeriodicKohnShamOperator
    config: PeriodicDavidsonConfig
    n_bands: int
    rank_policy: _Complex64RankPolicy
    token: _FixedHamiltonianToken
    vectors: _CompactLaneState
    observer: RuntimeObserver | None


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

    def __init__(self, *, batch_cap: int = 1) -> None:
        if type(batch_cap) is not int or batch_cap != 1:
            msg = (
                "Davidson cross-lane Hpsi lowering is unavailable; "
                "batch_cap must be one"
            )
            raise ValueError(msg)
        self._batch_cap = batch_cap

    @property
    def batch_cap(self) -> int:
        return self._batch_cap

    @staticmethod
    def _group_key(ticket: _DavidsonApplicationTicket) -> tuple[object, ...]:
        layout = ticket.vectors.layout
        return (
            id(layout.reciprocal),
            layout.grid_shape,
            layout.active_count,
            ticket.vectors.vector_count,
            id(ticket.observer),
        )

    def apply(
        self,
        tickets: Sequence[_DavidsonApplicationTicket],
    ) -> _DavidsonScheduleResult:
        if not tickets:
            msg = "Davidson scheduler requires at least one application ticket"
            raise ValueError(msg)
        if self.batch_cap != 1:
            msg = "Davidson scheduler batch-cap policy was mutated"
            raise RuntimeError(msg)
        seen: set[str] = set()
        ready: list[_DavidsonApplicationTicket] = []
        failures: dict[str, Exception] = {}
        for ticket in tickets:
            if ticket.lane_id in seen:
                msg = f"duplicate Davidson scheduler lane: {ticket.lane_id!r}"
                raise ValueError(msg)
            seen.add(ticket.lane_id)
            try:
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
                if not _all_finite(ticket.vectors.values):
                    msg = "Davidson application block must be finite"
                    raise ValueError(msg)
                ready.append(ticket)
            except Exception as error:
                failures[ticket.lane_id] = _detached_failure(error)

        grouped: dict[tuple[object, ...], list[_DavidsonApplicationTicket]] = (
            defaultdict(list)
        )
        for ticket in ready:
            grouped[self._group_key(ticket)].append(ticket)

        actions: dict[str, _CompactLaneState] = {}
        groups: list[tuple[str, ...]] = []
        compatibility_groups: list[tuple[str, ...]] = []
        for compatible in grouped.values():
            compatibility_groups.append(
                tuple(ticket.lane_id for ticket in compatible)
            )
            for start in range(0, len(compatible), self.batch_cap):
                submission = compatible[start : start + self.batch_cap]
                groups.append(tuple(ticket.lane_id for ticket in submission))
                # The explicit cap-one fallback is one real Hpsi submission.
                ticket = submission[0]
                try:
                    applied = ticket.operator._apply_compact(
                        ticket.vectors,
                        observer=ticket.observer,
                    )
                    if not _all_finite(applied.values):
                        msg = "Davidson Hamiltonian action must be finite"
                        raise ValueError(msg)
                    actions[ticket.lane_id] = applied
                    add_observed_work(
                        ticket.observer,
                        {"davidson_hv_new_vectors": ticket.vectors.vector_count},
                    )
                except Exception as error:  # lane-local fail-closed result
                    failures[ticket.lane_id] = _detached_failure(error)
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
        projected = (
            mx.conjugate(weights)
            @ self.projected
            @ mx.transpose(weights)
        )
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


@dataclass(frozen=True)
class _DavidsonRitzPair:
    """Selected Ritz/H-Ritz values derived entirely from paired lane state."""

    eigenvalues: mx.array
    vectors: _CompactLaneState
    applied: _CompactLaneState
    residual_stack: mx.array
    residuals: mx.array
    transform: mx.array


def _ritz_pair(state: _PairedDavidsonState, n_bands: int) -> _DavidsonRitzPair:
    values, eigenvectors = _projected_eigh(state.projected)
    selected_values = mx.real(values[:n_bands])
    selected_vectors = eigenvectors[:, :n_bands]
    transform = mx.transpose(selected_vectors)
    ritz_values = transform @ state.vectors.values
    h_ritz_values = transform @ state.applied.values
    residual_stack = h_ritz_values - selected_values[:, None] * ritz_values
    residuals = mx.sqrt(mx.sum(mx.abs(residual_stack) ** 2, axis=1))
    if not _all_finite(selected_values) or not _all_finite(residuals):
        msg = "Davidson Ritz data must be finite"
        raise ValueError(msg)
    return _DavidsonRitzPair(
        eigenvalues=selected_values,
        vectors=_CompactLaneState(ritz_values, state.vectors.layout),
        applied=_CompactLaneState(
            h_ritz_values,
            state.applied.layout,
            "hamiltonian_action",
        ),
        residual_stack=residual_stack,
        residuals=residuals,
        transform=transform,
    )


def _projected_eigh(matrix: mx.array) -> tuple[mx.array, mx.array]:
    # Only the small projected Rayleigh-Ritz matrix crosses to the CPU. LAPACK's
    # complex128 solve avoids the complex64 convergence floor while every
    # full-grid operator, residual, and FFT remains on the default MLX device.
    projected = np.asarray(matrix, dtype=np.complex128)
    if (
        projected.ndim != 2
        or projected.shape[0] == 0
        or projected.shape[0] != projected.shape[1]
    ):
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
    pending_vectors: _CompactLaneState | None = None
    pending_reused_width: int = 0
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
        self.scheduler = (
            _DavidsonScheduler(batch_cap=1) if scheduler is None else scheduler
        )
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
            msg = (
                "n_bands must be a positive non-bool integer no larger "
                "than the active basis size"
            )
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
    def _advance_lane(
        progress: _DavidsonLaneProgress,
    ) -> tuple[_CompactLaneState, int] | None:
        request = progress.request
        paired = progress.paired
        if paired is None:
            msg = "Davidson lane has no paired V/HV state"
            raise RuntimeError(msg)
        progress.iteration_count += 1
        iteration = progress.iteration_count
        with observed_phase(request.observer, "rayleigh_ritz"):
            ritz_pair = _ritz_pair(paired, request.n_bands)
        progress.ritz_pair = ritz_pair
        max_residual = float(mx.max(ritz_pair.residuals))
        if not np.isfinite(max_residual):
            msg = "Davidson maximum residual must be finite"
            raise ValueError(msg)
        if request.observer is not None:
            request.observer.emit(
                "davidson_iteration",
                lane_id=request.lane_id,
                iteration=iteration,
                subspace_size=paired.vector_count,
                max_residual=max_residual,
                converged=max_residual <= request.config.tolerance,
            )
        progress.converged = max_residual <= request.config.tolerance
        if progress.converged or iteration >= request.config.max_iterations:
            progress.done = True
            return None

        basis = request.operator.basis
        denominator = (
            basis._layout._active_kinetic_energies[None, :]
            - ritz_pair.eigenvalues[:, None]
        )
        sign = mx.where(denominator < 0.0, -1.0, 1.0)
        safe = sign * mx.maximum(
            mx.abs(denominator),
            request.config.preconditioner_floor,
        )
        raw_corrections = -ritz_pair.residual_stack / safe
        if not _all_finite(raw_corrections):
            msg = "Davidson preconditioned corrections must be finite"
            raise ValueError(msg)

        with observed_phase(request.observer, "orthogonalization"):
            append_rank = request.rank_policy.orthonormalize(
                mx.concatenate([paired.vectors.values, raw_corrections], axis=0),
                locked_count=paired.vector_count,
                required_count=paired.vector_count,
            )
        add_observed_work(
            request.observer,
            {"orthogonalization_vectors": request.n_bands},
        )
        correction_values = append_rank.values[paired.vector_count :]
        correction_count = int(correction_values.shape[0])
        progress.correction_width = correction_count
        if correction_count == 0:
            progress.done = True
            return None

        if paired.vector_count + correction_count > request.config.max_subspace_size:
            progress.restart_count += 1
            with observed_phase(request.observer, "orthogonalization"):
                restart_rank = request.rank_policy.orthonormalize(
                    ritz_pair.vectors.values,
                    required_count=request.n_bands,
                    max_count=request.n_bands,
                )
                paired = paired.transform(
                    restart_rank.transform @ ritz_pair.transform,
                    token=progress.token,
                )
                progress.paired = paired
            add_observed_work(
                request.observer,
                {"orthogonalization_vectors": request.n_bands},
            )
            capacity = request.config.max_subspace_size - paired.vector_count
            if capacity <= 0:
                progress.done = True
                return None
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
                progress.done = True
                return None

        progress.pending_vectors = basis._state_from_compact(correction_values)
        progress.pending_reused_width = paired.vector_count
        return progress.pending_vectors, progress.pending_reused_width

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
        orthonormality = request.rank_policy.overlap_error(
            ritz_pair.vectors.values
        )
        if orthonormality > request.rank_policy.guard_tolerance(request.n_bands):
            with observed_phase(request.observer, "orthogonalization"):
                final_rank = request.rank_policy.orthonormalize(
                    ritz_pair.vectors.values,
                    required_count=request.n_bands,
                    max_count=request.n_bands,
                )
                ritz_state = paired.transform(
                    ritz_pair.transform,
                    token=progress.token,
                ).transform(final_rank.transform, token=progress.token)
                ritz_pair = _ritz_pair(ritz_state, request.n_bands)
            add_observed_work(
                request.observer,
                {"orthogonalization_vectors": request.n_bands},
            )
            orthonormality = request.rank_policy.overlap_error(
                ritz_pair.vectors.values
            )
        request.rank_policy.validate(
            ritz_pair.vectors.values,
            required_count=request.n_bands,
        )
        final_max_residual = float(mx.max(ritz_pair.residuals))
        if not np.isfinite(final_max_residual):
            msg = "Davidson final residual must be finite"
            raise ValueError(msg)
        return PeriodicEigenResult._from_compact(
            eigenvalues=ritz_pair.eigenvalues,
            compact_coefficients=ritz_pair.vectors,
            basis=request.operator.basis,
            residuals=ritz_pair.residuals,
            orthonormality_error=orthonormality,
            iterations=progress.iteration_count,
            converged=final_max_residual <= request.config.tolerance,
            subspace_size=paired.vector_count,
            restart_count=progress.restart_count,
        )

    def solve(
        self,
        requests: Sequence[_DavidsonLaneRequest],
    ) -> _DavidsonEngineResult:
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
                initial_tickets.append(
                    self._ticket(progress, progress.initial_vectors)
                )
            except Exception as error:
                failures[request.lane_id] = _detached_failure(error)

        if initial_tickets:
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
            correction_tickets: list[_DavidsonApplicationTicket] = []
            for progress in active:
                try:
                    correction = self._advance_lane(progress)
                    if correction is not None:
                        correction_tickets.append(
                            self._ticket(progress, correction[0])
                        )
                except Exception as error:
                    self._fail_lane(progress, failures, error)
            if not correction_tickets:
                continue

            correction_schedule = self._schedule(correction_tickets)
            for ticket in correction_tickets:
                lane_id = ticket.lane_id
                failure = correction_schedule.failures.get(lane_id)
                if failure is not None:
                    self._fail_lane(progress_by_lane[lane_id], failures, failure)
                    continue
                progress = progress_by_lane[lane_id]
                paired = progress.paired
                if paired is None:
                    self._fail_lane(
                        progress,
                        failures,
                        RuntimeError("Davidson lane lost paired V/HV state"),
                    )
                    continue
                corrections = progress.pending_vectors
                reused_width = progress.pending_reused_width
                if corrections is None:
                    self._fail_lane(
                        progress,
                        failures,
                        RuntimeError(
                            "Davidson lane lost its pending correction block"
                        ),
                    )
                    continue
                try:
                    progress.paired = paired.append(
                        corrections,
                        correction_schedule.actions[lane_id],
                        token=progress.token,
                    )
                    progress.request.rank_policy.validate(
                        progress.paired.vectors.values,
                        required_count=progress.request.n_bands,
                    )
                    add_observed_work(
                        progress.request.observer,
                        {"davidson_hv_reused_vectors": reused_width},
                    )
                    progress.pending_vectors = None
                    progress.pending_reused_width = 0
                except Exception as error:
                    self._fail_lane(progress, failures, error)

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
        Converged or exhausted periodic eigensolver result.
    """

    runtime_observer = operator.observer if observer is None else observer
    solver_config = PeriodicDavidsonConfig() if config is None else config
    basis = operator.basis
    if type(n_bands) is not int or n_bands <= 0 or n_bands > basis.active_count:
        msg = (
            "n_bands must be a positive non-bool integer no larger than "
            "the active basis size"
        )
        raise ValueError(msg)
    if isinstance(initial_coefficients, _PairedDavidsonState):
        msg = "paired Davidson H(V) cannot seed a new fixed-Hamiltonian solve"
        raise ValueError(msg)
    if initial_coefficients is None:
        trial = _initial_coefficients(basis, n_bands)
    elif isinstance(initial_coefficients, _CompactLaneState):
        try:
            _require_layout(initial_coefficients, basis._layout)
            trial = initial_coefficients
        except ValueError:
            trial = _remap_initial_coefficients(initial_coefficients, basis._layout)
    else:
        trial, _ = basis._state_from_full(initial_coefficients)
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
            "dense_full_hamiltonian": False,
        }


def _density_from_kpoints(
    results: Sequence[PeriodicKPointResult],
    *,
    occupation: float,
) -> mx.array:
    density = mx.zeros(results[0].basis.grid.shape, dtype=mx.float32)
    for result in results:
        orbitals = result.basis._to_real_compact(result.eigen._compact_coefficients)
        density = density + float(result.weight * occupation) * mx.sum(
            mx.abs(orbitals) ** 2,
            axis=0,
        )
    return mx.real(density)


def _density_residual(current: mx.array, target: mx.array, grid: RealSpaceGrid) -> float:
    delta = target - current
    return float(mx.sqrt(mx.sum(delta * delta) * grid.dv))


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
        gamma_basis = PlaneWaveBasis(
            system.grid,
            cutoff_hartree,
            reciprocal_grid=shared_reciprocal,
            lane_label="gamma-local-potential",
        )
        nonlocal_operators = [
            PeriodicGTHNonlocalOperator(
                system.pseudopotential,
                basis,
                system.positions,
                cache=projector_cache,
            )
            for basis in bases
        ]
        local_potential = gth_local_potential_grid(
            system.pseudopotential,
            gamma_basis,
            system.positions,
        )
        if initial_density is None:
            density = mx.full(system.grid.shape, system.electron_count / system.grid.volume)
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
        mixer = (
            PulayDIISMixer(beta=scf_config.mixing_beta)
            if scf_config.mixer == "diis"
            else LinearMixer(beta=scf_config.mixing_beta)
        )
        ewald = periodic_ewald_energy(
            system.charges,
            system.positions,
            np.asarray(system.grid.lengths),
        )
        previous_energy: float | None = None
        previous_states = (
            list(initial_coefficients)
            if initial_coefficients is not None
            else [None] * len(bases)
        )
    if observer is not None:
        observer.record_memory("shared_full_grid_bytes", system.grid.size * 4 * 4)
        observer.record_memory("persistent_projector_bytes", 0)
        observer.emit(
            "setup",
            status="completed",
            active_counts=[basis.active_count for basis in bases],
        )
    history: list[dict[str, float | int | str | None]] = []
    final_results: tuple[PeriodicKPointResult, ...] = ()
    energy_terms: dict[str, float] = {}
    converged = False
    density_residual = float("inf")
    energy_delta: float | None = None
    timings = {"hartree": 0.0, "xc": 0.0, "eigensolver": 0.0, "total": 0.0}
    total_start = perf_counter()
    for iteration in range(1, scf_config.max_iterations + 1):
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
        results = []
        max_orbital_residual = 0.0
        start = perf_counter()
        for point_index, (point, basis, nonlocal_operator) in enumerate(
            zip(
                kpoint_mesh.points,
                bases,
                nonlocal_operators,
                strict=True,
            )
        ):
            if observer is not None:
                observer.emit(
                    "kpoint_batch",
                    status="started",
                    scf_iteration=iteration,
                    batch_index=point_index,
                    batch_size=1,
                    reduced_kpoints=[list(point.vector)],
                )
            operator = PeriodicKohnShamOperator(
                basis,
                effective,
                nonlocal_operator,
                observer,
            )
            eigen = solve_periodic_eigenproblem(
                operator,
                n_bands=occupied_bands,
                config=scf_config.davidson,
                initial_coefficients=previous_states[point_index],
                observer=observer,
            )
            add_observed_work(observer, {"kpoint_lane_solves": 1})
            max_orbital_residual = max(
                max_orbital_residual,
                float(mx.max(eigen.residuals)),
            )
            results.append(
                PeriodicKPointResult(
                    reduced_kpoint=tuple(float(value) for value in point.vector),
                    weight=float(point.weight),
                    basis=basis,
                    eigen=eigen,
                )
            )
            if observer is not None:
                observer.emit(
                    "kpoint_batch",
                    status="completed",
                    scf_iteration=iteration,
                    batch_index=point_index,
                    batch_size=1,
                    reduced_kpoints=[list(point.vector)],
                    converged=eigen.converged,
                )
        timings["eigensolver"] += (perf_counter() - start) * 1000.0
        final_results = tuple(results)
        with observed_phase(observer, "density"):
            target_density = _density_from_kpoints(final_results, occupation=2.0)
            add_observed_work(
                observer,
                {
                    "fft_submissions": len(final_results),
                    "fft_vector_equivalents": sum(occupied_bands for _ in final_results),
                },
            )
            target_count = float(mx.sum(target_density) * system.grid.dv)
            target_density = target_density * (system.electron_count / target_count)
            density_residual = _density_residual(density, target_density, system.grid)

        band_energy = sum(
            result.weight * 2.0 * float(mx.sum(result.eigen.eigenvalues))
            for result in final_results
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
                    all(result.eigen.converged for result in final_results)
                ).lower(),
            }
        )
        all_eigen_converged = all(result.eigen.converged for result in final_results)
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
            mixed = mx.maximum(mixer.mix(density, target_density), 0.0)
            mixed_count = float(mx.sum(mixed) * system.grid.dv)
            density = mixed * (system.electron_count / mixed_count)
        previous_energy = total_energy
        previous_states = [
            result.eigen._compact_coefficients for result in final_results
        ]

    timings["total"] = (perf_counter() - total_start) * 1000.0
    electron_count = float(mx.sum(density) * system.grid.dv)
    if observer is not None:
        coefficient_bytes = sum(
            int(np.prod(result.eigen._compact_coefficients.values.shape)) * 8
            for result in final_results
        )
        observer.record_memory("persistent_coefficient_bytes", coefficient_bytes)
        observer.record_memory("coefficient_payload_bytes", coefficient_bytes)
        observation = observer.snapshot()
        traffic_elements = int(observation["work_counters"]["projector_traffic_elements"])
        observer.record_memory("projector_traffic_bytes", traffic_elements * 8)
        observer.emit(
            "completion",
            stage="scf",
            status="converged" if converged else "max_iterations",
            iterations=iteration,
            total_energy_hartree=float(energy_terms["total"]),
        )
    return PeriodicSCFResult(
        converged=converged,
        status="converged" if converged else "max_iterations",
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

    with _GTHProjectorCache() as projector_cache:
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
        )
