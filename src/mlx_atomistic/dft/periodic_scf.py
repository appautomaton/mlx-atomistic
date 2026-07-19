"""Self-consistent weighted k-point plane-wave DFT."""

from __future__ import annotations

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
    """Controls for the full block Davidson/Rayleigh-Ritz eigensolver."""

    max_iterations: int = 30
    tolerance: float = 1e-5
    max_subspace_size: int = 64
    preconditioner_floor: float = 0.5

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            msg = "max_iterations must be positive"
            raise ValueError(msg)
        if self.tolerance <= 0.0:
            msg = "tolerance must be positive"
            raise ValueError(msg)
        if self.max_subspace_size <= 1:
            msg = "max_subspace_size must exceed one"
            raise ValueError(msg)
        if self.preconditioner_floor <= 0.0:
            msg = "preconditioner_floor must be positive"
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


@dataclass(frozen=True)
class PeriodicKohnShamOperator:
    """Fixed-density periodic Kohn-Sham operator in coefficient space."""

    basis: PlaneWaveBasis
    effective_local_potential: mx.array
    nonlocal_operator: PeriodicGTHNonlocalOperator | None = None
    observer: RuntimeObserver | None = None

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
                * self.basis.active_kinetic_energies[None, :]
            )
            local = batch.unpad(
                batch.apply_local(
                    self.effective_local_potential,
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


def _combine(weights: mx.array, vectors: mx.array) -> mx.array:
    return mx.transpose(weights) @ vectors


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
    kinetic = np.asarray(basis.active_kinetic_energies, dtype=np.float64)
    selected = np.argsort(kinetic, kind="stable")[:count]
    coefficients = np.zeros((count, basis.active_count), dtype=np.complex64)
    coefficients[np.arange(count), selected] = 1.0
    return basis._state_from_compact(mx.array(coefficients))


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

    if observer is not None and operator.observer is not None and observer is not operator.observer:
        msg = "operator and solver observers must be the same object"
        raise ValueError(msg)
    runtime_observer = operator.observer if observer is None else observer
    solver_config = PeriodicDavidsonConfig() if config is None else config
    basis = operator.basis
    if n_bands <= 0 or n_bands > basis.active_count:
        msg = "n_bands must be between one and the active basis size"
        raise ValueError(msg)
    if solver_config.max_subspace_size < n_bands:
        msg = "max_subspace_size cannot be smaller than n_bands"
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
    if trial.kind != "coefficients":
        msg = "initial coefficients cannot be a cached Hamiltonian action"
        raise ValueError(msg)
    with observed_phase(runtime_observer, "orthogonalization"):
        subspace = basis._state_from_compact(
            basis._orthonormalize_compact(trial.values)
        )
    add_observed_work(
        runtime_observer,
        {"orthogonalization_vectors": trial.vector_count},
    )
    restart_count = 0
    converged = False
    eigenvalues = operator._rayleigh_quotients_compact(
        subspace,
        observer=runtime_observer,
    )
    residuals = mx.full((n_bands,), float("inf"), dtype=mx.float32)
    ritz = basis._state_from_compact(subspace.values[:n_bands])
    iteration_count = 0
    for _iteration in range(1, solver_config.max_iterations + 1):
        iteration_count = _iteration
        applied = operator._apply_compact(subspace, observer=runtime_observer)
        subspace_size = subspace.vector_count
        add_observed_work(
            runtime_observer,
            {
                "davidson_hv_new_vectors": subspace_size,
                "projected_old_old_rebuilds": subspace_size * subspace_size,
            },
        )
        with observed_phase(runtime_observer, "rayleigh_ritz"):
            projected = _subspace_matrix(subspace.values, applied.values)
            values, vectors = _projected_eigh(projected)
            selected_values = values[:n_bands]
            selected_vectors = vectors[:, :n_bands]
            ritz = basis._state_from_compact(
                _combine(selected_vectors, subspace.values)
            )
            h_ritz = basis._state_from_compact(
                _combine(selected_vectors, applied.values),
                kind="hamiltonian_action",
            )
            residual_stack = h_ritz.values - selected_values[:, None] * ritz.values
            residuals = mx.sqrt(mx.sum(mx.abs(residual_stack) ** 2, axis=1))
            eigenvalues = mx.real(selected_values)
        max_residual = float(mx.max(residuals))
        if runtime_observer is not None:
            runtime_observer.emit(
                "davidson_iteration",
                iteration=_iteration,
                subspace_size=subspace_size,
                max_residual=max_residual,
                converged=max_residual <= solver_config.tolerance,
            )
        if max_residual <= solver_config.tolerance:
            converged = True
            break

        corrections = []
        for band_index in range(n_bands):
            denominator = basis.active_kinetic_energies - eigenvalues[band_index]
            sign = mx.where(denominator < 0.0, -1.0, 1.0)
            safe = sign * mx.maximum(mx.abs(denominator), solver_config.preconditioner_floor)
            correction = -residual_stack[band_index] / safe
            candidate_vectors = [*list(subspace.values), *corrections]
            for _ in range(2):
                for accepted in candidate_vectors:
                    overlap = mx.sum(mx.conjugate(accepted) * correction)
                    correction = correction - overlap * accepted
            norm = mx.sqrt(mx.real(mx.sum(mx.conjugate(correction) * correction)))
            if float(norm) > 1e-10:
                corrections.append(correction / norm)
        if not corrections:
            break
        correction_stack = mx.stack(corrections, axis=0)
        if subspace.vector_count + len(corrections) > solver_config.max_subspace_size:
            restart_count += 1
            with observed_phase(runtime_observer, "orthogonalization"):
                subspace = basis._state_from_compact(
                    basis._orthonormalize_compact(
                        mx.concatenate([ritz.values, correction_stack], axis=0)
                    )
                )
        else:
            with observed_phase(runtime_observer, "orthogonalization"):
                subspace = basis._state_from_compact(
                    basis._orthonormalize_compact(
                        mx.concatenate([subspace.values, correction_stack], axis=0)
                    )
                )
        add_observed_work(
            runtime_observer,
            {"orthogonalization_vectors": subspace.vector_count},
        )

    with observed_phase(runtime_observer, "orthogonalization"):
        ritz = basis._state_from_compact(
            basis._orthonormalize_compact(ritz.values)
        )
    add_observed_work(
        runtime_observer,
        {"orthogonalization_vectors": ritz.vector_count},
    )
    final_applied = operator._apply_compact(ritz, observer=runtime_observer)
    eigenvalues = operator._rayleigh_quotients_compact(
        ritz,
        observer=runtime_observer,
    )
    residual_stack = final_applied.values - eigenvalues[:, None] * ritz.values
    residuals = mx.sqrt(mx.sum(mx.abs(residual_stack) ** 2, axis=1))
    overlap = np.asarray(basis._overlap_matrix_compact(ritz.values))
    orthonormality = float(np.max(np.abs(overlap - np.eye(n_bands))))
    return PeriodicEigenResult._from_compact(
        eigenvalues=eigenvalues,
        compact_coefficients=ritz,
        basis=basis,
        residuals=residuals,
        orthonormality_error=orthonormality,
        iterations=iteration_count,
        converged=converged or float(mx.max(residuals)) <= solver_config.tolerance,
        subspace_size=subspace.vector_count,
        restart_count=restart_count,
    )


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
