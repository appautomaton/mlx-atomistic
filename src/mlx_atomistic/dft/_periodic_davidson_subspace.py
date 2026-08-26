"""Paired subspace state and Rayleigh-Ritz mathematics for Davidson."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft._compact import _CompactLaneState, _require_layout
from mlx_atomistic.dft._periodic_davidson_context import _FixedHamiltonianToken
from mlx_atomistic.dft._periodic_execution import _to_numpy
from mlx_atomistic.dft._periodic_orthonormalization import _RankResult
from mlx_atomistic.dft._runtime_observer import RuntimeObserver


def _subspace_matrix(basis_vectors: mx.array, applied: mx.array) -> mx.array:
    matrix = mx.conjugate(basis_vectors) @ mx.transpose(applied)
    return 0.5 * (matrix + mx.conjugate(mx.transpose(matrix)))


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
    """Build lazy Ritz data from the exact scheduled H(X)."""

    _require_layout(applied, candidate.vectors.layout)
    if applied.kind != "hamiltonian_action":
        msg = "Davidson direct validation requires Hamiltonian actions"
        raise ValueError(msg)
    if applied.vector_count != candidate.vectors.vector_count:
        msg = "Davidson direct validation width does not match its Ritz vectors"
        raise ValueError(msg)
    # The scheduled H(X) is authoritative. Refreshing each Rayleigh quotient
    # removes the residual component parallel to its vector without another
    # Hamiltonian application or changing the orthogonal convergence error.
    vector_norms = mx.real(
        mx.sum(mx.conjugate(candidate.vectors.values) * candidate.vectors.values, axis=1)
    )
    eigenvalues = mx.real(
        mx.sum(mx.conjugate(candidate.vectors.values) * applied.values, axis=1)
        / vector_norms
    )
    residual_stack, residuals, max_residual, finite = _ritz_residual_arrays(
        eigenvalues,
        candidate.vectors,
        applied,
    )
    return _DavidsonRitzCandidate(
        eigenvalues=eigenvalues,
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
        [matrix if isinstance(matrix, mx.array) else mx.array(matrix) for matrix in matrices],
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
