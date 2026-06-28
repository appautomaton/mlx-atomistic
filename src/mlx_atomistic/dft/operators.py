"""Kohn-Sham operator and tiny-grid numerical references."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft.fft import fft3, ifft3
from mlx_atomistic.dft.grids import RealSpaceGrid, ReciprocalGrid
from mlx_atomistic.dft.nonlocal_pseudopotential import NonlocalPseudopotentialOperator
from mlx_atomistic.dft.potentials import hartree_potential
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional, LDAExchangeCorrelation


def _as_orbital_stack(orbitals: mx.array, grid: RealSpaceGrid) -> tuple[mx.array, bool]:
    array = mx.array(orbitals)
    if array.shape == grid.shape:
        return mx.reshape(array, (1, *grid.shape)), True
    if len(array.shape) == 4 and array.shape[1:] == grid.shape:
        return array, False
    msg = "orbitals must have shape grid.shape or (n_orbitals, *grid.shape)"
    raise ValueError(msg)


def _restore_orbital_shape(stack: mx.array, was_single: bool) -> mx.array:
    return stack[0] if was_single else stack


def _orthonormalize_numpy(orbitals: np.ndarray, grid: RealSpaceGrid) -> np.ndarray:
    stack = np.asarray(orbitals, dtype=np.complex64)
    if stack.shape == grid.shape:
        stack = stack.reshape((1, *grid.shape))
    flat = stack.reshape(stack.shape[0], grid.size)
    weighted = flat.T * np.sqrt(grid.dv)
    q, _ = np.linalg.qr(weighted)
    return (q[:, : stack.shape[0]].T / np.sqrt(grid.dv)).reshape(stack.shape).astype(np.complex64)


def apply_kinetic(
    orbitals: mx.array,
    grid: RealSpaceGrid,
    *,
    kpoint: Sequence[float] | None = None,
) -> mx.array:
    """Apply the plane-wave kinetic operator ``-1/2 ∇²``.

    Args:
        orbitals: Real-space orbital(s); a single orbital of shape ``grid.shape`` or a
            stack ``(n_orbitals, *grid.shape)``.
        grid: Real-space grid defining the FFT mesh.
        kpoint: Optional k-point 3-vector added to the reciprocal-grid G-vectors;
            ``None`` uses the Γ point. Defaults to ``None``.

    Returns:
        The kinetic operator applied to ``orbitals``, same shape as ``orbitals``.
    """

    stack, was_single = _as_orbital_stack(orbitals, grid)
    reciprocal = ReciprocalGrid.from_real_space(grid)
    if kpoint is None:
        kinetic_g2 = reciprocal.g2
    else:
        k = mx.reshape(mx.array(kpoint, dtype=mx.float32), (1, 1, 1, 3))
        shifted = reciprocal.vectors + k
        kinetic_g2 = mx.sum(shifted * shifted, axis=-1)
    applied = mx.stack(
        [ifft3(0.5 * kinetic_g2 * fft3(stack[index])) for index in range(stack.shape[0])],
        axis=0,
    )
    return _restore_orbital_shape(applied, was_single)


def apply_local_potential(orbitals: mx.array, local_potential: mx.array) -> mx.array:
    """Apply a local multiplicative potential to orbitals.

    Args:
        orbitals: Real-space orbital(s); a single orbital of shape ``grid.shape`` or a
            stack ``(n_orbitals, *grid.shape)``.
        local_potential: Local potential on the grid, shape ``grid.shape``.

    Returns:
        The elementwise product ``V · ψ``, same shape as ``orbitals``.
    """

    return mx.array(orbitals) * mx.array(local_potential)


def apply_hartree_xc_potential(
    orbitals: mx.array,
    hartree: mx.array,
    xc_potential: mx.array,
) -> mx.array:
    """Apply Hartree plus exchange-correlation potential to orbitals.

    Args:
        orbitals: Real-space orbital(s); a single orbital of shape ``grid.shape`` or a
            stack ``(n_orbitals, *grid.shape)``.
        hartree: Hartree potential on the grid, shape ``grid.shape``.
        xc_potential: Exchange-correlation potential on the grid, shape ``grid.shape``.

    Returns:
        The product ``(V_H + V_xc) · ψ``, same shape as ``orbitals``.
    """

    return mx.array(orbitals) * (mx.array(hartree) + mx.array(xc_potential))


def apply_hamiltonian(
    orbitals: mx.array,
    effective_potential: mx.array,
    grid: RealSpaceGrid,
) -> mx.array:
    """Apply ``T + V_eff`` using an explicit effective potential.

    Args:
        orbitals: Real-space orbital(s); a single orbital of shape ``grid.shape`` or a
            stack ``(n_orbitals, *grid.shape)``.
        effective_potential: Combined local effective potential on the grid, shape
            ``grid.shape``.
        grid: Real-space grid defining the FFT mesh.

    Returns:
        ``(T + V_eff) ψ``, same shape as ``orbitals``.
    """

    return apply_kinetic(orbitals, grid) + apply_local_potential(orbitals, effective_potential)


@dataclass(frozen=True)
class KohnShamOperator:
    """Kohn-Sham Hamiltonian operator for a fixed density and local potential."""

    grid: RealSpaceGrid
    local_potential: mx.array
    density: mx.array
    hartree: mx.array
    xc_potential: mx.array
    effective_potential: mx.array
    nonlocal_operator: NonlocalPseudopotentialOperator | None = None
    kpoint: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_density(
        cls,
        grid: RealSpaceGrid,
        local_potential: mx.array,
        density: mx.array,
        *,
        xc_functional: ExchangeCorrelationFunctional | None = None,
        density_floor: float = 1e-12,
        nonlocal_operator: NonlocalPseudopotentialOperator | None = None,
        kpoint: Sequence[float] | None = None,
    ) -> KohnShamOperator:
        """Build an operator from ``ρ`` and a local potential.

        Args:
            grid: Real-space grid defining the FFT mesh.
            local_potential: External local (ionic) potential on the grid, shape ``grid.shape``.
            density: Electron density ``ρ`` on the grid, shape ``grid.shape``.
            xc_functional: Exchange-correlation functional; ``None`` uses
                `LDAExchangeCorrelation`. Defaults to ``None``.
            density_floor: Lower clamp on the density for XC numerical stability.
                Defaults to ``1e-12``.
            nonlocal_operator: Optional nonlocal pseudopotential operator. Defaults to ``None``.
            kpoint: Optional Bloch k-point; ``None`` uses Γ. Defaults to ``None``.

        Returns:
            A `KohnShamOperator` with the Hartree, XC, and effective potentials
                precomputed.
        """

        xc_functional = LDAExchangeCorrelation() if xc_functional is None else xc_functional
        hartree = hartree_potential(density, grid)
        xc = xc_functional.evaluate(density, grid, density_floor=density_floor)
        effective = mx.array(local_potential) + hartree + xc.potential
        return cls(
            grid=grid,
            local_potential=mx.array(local_potential),
            density=mx.array(density),
            hartree=hartree,
            xc_potential=xc.potential,
            effective_potential=effective,
            nonlocal_operator=nonlocal_operator,
            kpoint=(0.0, 0.0, 0.0)
            if kpoint is None
            else tuple(float(value) for value in kpoint),
        )

    def apply_kinetic(self, orbitals: mx.array) -> mx.array:
        """Apply only the kinetic part.

        Args:
            orbitals: Real-space orbital(s); a single orbital of shape ``grid.shape`` or a
                stack ``(n_orbitals, *grid.shape)``.

        Returns:
            ``T ψ`` at this operator's k-point, same shape as ``orbitals``.
        """

        return apply_kinetic(orbitals, self.grid, kpoint=self.kpoint)

    def apply_local_potential(self, orbitals: mx.array) -> mx.array:
        """Apply only the external local potential.

        Args:
            orbitals: Real-space orbital(s); a single orbital of shape ``grid.shape`` or a
                stack ``(n_orbitals, *grid.shape)``.

        Returns:
            ``V_local · ψ``, same shape as ``orbitals``.
        """

        return apply_local_potential(orbitals, self.local_potential)

    def apply_hartree_xc_potential(self, orbitals: mx.array) -> mx.array:
        """Apply only the Hartree plus exchange-correlation potential.

        Args:
            orbitals: Real-space orbital(s); a single orbital of shape ``grid.shape`` or a
                stack ``(n_orbitals, *grid.shape)``.

        Returns:
            ``(V_H + V_xc) · ψ``, same shape as ``orbitals``.
        """

        return apply_hartree_xc_potential(orbitals, self.hartree, self.xc_potential)

    def apply_hamiltonian(self, orbitals: mx.array) -> mx.array:
        """Apply ``H = T + V_eff``.

        Args:
            orbitals: Real-space orbital(s); a single orbital of shape ``grid.shape`` or a
                stack ``(n_orbitals, *grid.shape)``.

        Returns:
            ``H ψ`` (kinetic + effective local + any nonlocal term), same shape as ``orbitals``.
        """

        applied = self.apply_kinetic(orbitals) + apply_local_potential(
            orbitals,
            self.effective_potential,
        )
        if self.nonlocal_operator is not None and self.nonlocal_operator.available:
            applied = applied + self.nonlocal_operator.apply(orbitals)
        return applied

    def rayleigh_quotients(self, orbitals: mx.array) -> mx.array:
        """Return ``<ψᵢ|H|ψᵢ>/<ψᵢ|ψᵢ>`` for each orbital.

        Args:
            orbitals: Real-space orbital stack ``(n_orbitals, *grid.shape)``.

        Returns:
            The per-orbital Rayleigh quotients (energy estimates), shape
                ``(n_orbitals,)``.
        """

        stack, _ = _as_orbital_stack(orbitals, self.grid)
        applied, _ = _as_orbital_stack(self.apply_hamiltonian(stack), self.grid)
        values = []
        for index in range(stack.shape[0]):
            numerator = mx.sum(mx.conjugate(stack[index]) * applied[index]) * self.grid.dv
            denominator = mx.sum(mx.conjugate(stack[index]) * stack[index]) * self.grid.dv
            values.append(mx.real(numerator / denominator))
        return mx.stack(values)


def orthonormality_error(orbitals: mx.array, grid: RealSpaceGrid) -> float:
    """Return max absolute overlap error from the identity matrix.

    Args:
        orbitals: Real-space orbital stack ``(n_orbitals, *grid.shape)``.
        grid: Real-space grid; its ``dv`` weights the overlap integral.

    Returns:
        The maximum absolute deviation of the overlap matrix ``⟨ψᵢ|ψⱼ⟩`` from the
            identity.
    """

    stack, _ = _as_orbital_stack(orbitals, grid)
    flat = np.array(stack, dtype=np.complex128).reshape(stack.shape[0], grid.size)
    overlap = flat @ flat.conjugate().T * grid.dv
    return float(np.max(np.abs(overlap - np.eye(stack.shape[0]))))


def orbital_residuals(
    orbitals: mx.array,
    operator: KohnShamOperator,
    eigenvalues: mx.array | None = None,
) -> mx.array:
    """Return ``||Hψᵢ - εᵢψᵢ||`` for each orbital.

    Args:
        orbitals: Real-space orbital stack ``(n_orbitals, *grid.shape)``.
        operator: The `KohnShamOperator` supplying ``H``.
        eigenvalues: Optional per-orbital eigenvalues εᵢ; ``None`` uses the Rayleigh
            quotients. Defaults to ``None``.

    Returns:
        The per-orbital residual norms, shape ``(n_orbitals,)``.
    """

    stack, _ = _as_orbital_stack(orbitals, operator.grid)
    if eigenvalues is None:
        eigenvalues = operator.rayleigh_quotients(stack)
    else:
        eigenvalues = mx.array(eigenvalues)
    applied, _ = _as_orbital_stack(operator.apply_hamiltonian(stack), operator.grid)
    residuals = []
    for index in range(stack.shape[0]):
        delta = applied[index] - eigenvalues[index] * stack[index]
        residuals.append(mx.sqrt(mx.sum(mx.abs(delta) ** 2) * operator.grid.dv))
    return mx.stack(residuals)


@dataclass(frozen=True)
class DiagonalizationResult:
    """Small eigenproblem result bundle."""

    eigenvalues: mx.array
    orbitals: mx.array
    residuals: mx.array
    orthonormality_error: float
    iterations: int
    converged: bool
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-safe summary.

        Returns:
            The eigenvalues, residuals, orthonormality error, iteration count,
                convergence flag, and metadata as a JSON-serializable dict.
        """

        return {
            "eigenvalues": np.array(self.eigenvalues).tolist(),
            "residuals": np.array(self.residuals).tolist(),
            "orthonormality_error": self.orthonormality_error,
            "iterations": self.iterations,
            "converged": self.converged,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DenseHamiltonianReference:
    """Explicit dense Hamiltonian for tiny-grid validation."""

    operator: KohnShamOperator

    def matrix(self) -> np.ndarray:
        """Build the dense matrix by applying the operator to basis vectors.

        Returns:
            The Hermitized dense Hamiltonian, shape ``(grid.size, grid.size)``.
        """

        size = self.operator.grid.size
        columns = []
        for index in range(size):
            basis = np.zeros(size, dtype=np.complex64)
            basis[index] = 1.0
            applied = self.operator.apply_hamiltonian(basis.reshape(self.operator.grid.shape))
            columns.append(np.array(applied, dtype=np.complex128).reshape(size))
        matrix = np.column_stack(columns)
        return 0.5 * (matrix + matrix.conjugate().T)

    def matvec(self, orbital: mx.array) -> mx.array:
        """Apply the dense matrix to a single orbital.

        Args:
            orbital: A single real-space orbital of shape ``grid.shape``.

        Returns:
            ``H ψ`` as a grid-shaped array.
        """

        flat = np.array(orbital, dtype=np.complex128).reshape(self.operator.grid.size)
        applied = self.matrix() @ flat
        return mx.array(applied.reshape(self.operator.grid.shape).astype(np.complex64))

    def diagonalize(self, n_orbitals: int) -> DiagonalizationResult:
        """Diagonalize the dense reference matrix.

        Args:
            n_orbitals: Number of lowest eigenpairs to return.

        Returns:
            A `DiagonalizationResult` with the lowest ``n_orbitals`` eigenpairs
                (orthonormalized).
        """

        if n_orbitals <= 0:
            msg = "n_orbitals must be positive"
            raise ValueError(msg)
        if n_orbitals > self.operator.grid.size:
            msg = "n_orbitals cannot exceed the real-space grid size"
            raise ValueError(msg)
        matrix = self.matrix()
        values, vectors = np.linalg.eigh(matrix)
        orbitals = vectors[:, :n_orbitals].T.reshape((n_orbitals, *self.operator.grid.shape))
        orbitals = _orthonormalize_numpy(orbitals, self.operator.grid)
        orbitals_mx = mx.array(orbitals)
        eigenvalues = mx.array(values[:n_orbitals].astype(np.float32))
        residuals = orbital_residuals(orbitals_mx, self.operator, eigenvalues)
        return DiagonalizationResult(
            eigenvalues=eigenvalues,
            orbitals=orbitals_mx,
            residuals=residuals,
            orthonormality_error=orthonormality_error(orbitals_mx, self.operator.grid),
            iterations=1,
            converged=True,
            metadata={"solver": "dense-reference"},
        )


@dataclass(frozen=True)
class SubspaceDiagonalizer:
    """Operator-backed Rayleigh-Ritz diagonalizer for tiny prototype grids."""

    max_iterations: int = 8
    tolerance: float = 1e-6

    def solve(self, operator: KohnShamOperator, *, n_orbitals: int) -> DiagonalizationResult:
        """Solve the occupied subspace.

        For this milestone the implementation intentionally uses the explicit
        tiny-grid reference matrix as the Rayleigh-Ritz space. The public solver
        boundary lets us replace this with an iterative Davidson path later.

        Args:
            operator: The `KohnShamOperator` to diagonalize.
            n_orbitals: Number of occupied orbitals to solve for.

        Returns:
            A `DiagonalizationResult` for the lowest ``n_orbitals`` states.
        """

        result = DenseHamiltonianReference(operator).diagonalize(n_orbitals)
        max_residual = float(np.max(np.array(result.residuals)))
        return DiagonalizationResult(
            eigenvalues=result.eigenvalues,
            orbitals=result.orbitals,
            residuals=result.residuals,
            orthonormality_error=result.orthonormality_error,
            iterations=1,
            converged=max_residual <= self.tolerance,
            metadata={"solver": "subspace-dense-reference"},
        )


@dataclass(frozen=True)
class EigensolverConfig:
    """Configuration for iterative Kohn-Sham eigensolvers."""

    max_iterations: int = 16
    tolerance: float = 1e-6
    max_subspace_size: int = 12
    dense_fallback_size: int = 512

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            msg = "max_iterations must be positive"
            raise ValueError(msg)
        if self.tolerance <= 0.0:
            msg = "tolerance must be positive"
            raise ValueError(msg)
        if self.max_subspace_size <= 0:
            msg = "max_subspace_size must be positive"
            raise ValueError(msg)
        if self.dense_fallback_size <= 0:
            msg = "dense_fallback_size must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class KineticPreconditioner:
    """Simple reciprocal-space kinetic preconditioner."""

    shift: float = 0.5

    def apply(self, residual: mx.array, operator: KohnShamOperator) -> mx.array:
        """Apply ``1 / (0.5|G+k|² + shift)`` to a residual.

        Args:
            residual: A real-space residual to precondition.
            operator: The `KohnShamOperator` supplying the grid and k-point.

        Returns:
            The preconditioned residual, same shape as ``residual``.
        """

        reciprocal = ReciprocalGrid.from_real_space(operator.grid)
        k = mx.reshape(mx.array(operator.kpoint, dtype=mx.float32), (1, 1, 1, 3))
        shifted = reciprocal.vectors + k
        denominator = 0.5 * mx.sum(shifted * shifted, axis=-1) + self.shift
        return ifft3(fft3(residual) / denominator)


@dataclass(frozen=True)
class DavidsonDiagonalizer:
    """Alpha Davidson-style eigensolver.

    The dense reference path is still used for tiny grids. Above that cutoff,
    this applies a conservative preconditioned residual iteration that avoids
    building the dense Hamiltonian. It is not yet a production Rayleigh-Ritz
    Davidson implementation.
    """

    config: EigensolverConfig = field(default_factory=EigensolverConfig)
    preconditioner: KineticPreconditioner = field(default_factory=KineticPreconditioner)

    def solve(
        self,
        operator: KohnShamOperator,
        *,
        n_orbitals: int,
        initial_orbitals: mx.array | None = None,
    ) -> DiagonalizationResult:
        """Solve for the lowest occupied orbitals.

        Args:
            operator: The `KohnShamOperator` to diagonalize.
            n_orbitals: Number of occupied orbitals to solve for.
            initial_orbitals: Optional starting orbital stack; ``None`` uses a
                deterministic random guess. Defaults to ``None``.

        Returns:
            A `DiagonalizationResult` for the lowest ``n_orbitals`` states (dense
                reference below the fallback size, alpha preconditioned iteration
                above).
        """

        if operator.grid.size <= self.config.dense_fallback_size:
            result = DenseHamiltonianReference(operator).diagonalize(n_orbitals)
            return DiagonalizationResult(
                eigenvalues=result.eigenvalues,
                orbitals=result.orbitals,
                residuals=result.residuals,
                orthonormality_error=result.orthonormality_error,
                iterations=result.iterations,
                converged=result.converged,
                metadata={
                    "solver": "davidson-dense-reference",
                    "subspace_size": operator.grid.size,
                    "restart_count": 0,
                },
            )

        if initial_orbitals is None:
            rng = np.random.default_rng(17)
            trial = rng.normal(size=(n_orbitals, *operator.grid.shape)).astype(np.float32)
            orbitals = mx.array(_orthonormalize_numpy(trial, operator.grid))
        else:
            orbitals = mx.array(_orthonormalize_numpy(np.array(initial_orbitals), operator.grid))
        eigenvalues = operator.rayleigh_quotients(orbitals)
        converged = False
        restart_count = 0
        for iteration in range(1, self.config.max_iterations + 1):
            applied, _ = _as_orbital_stack(operator.apply_hamiltonian(orbitals), operator.grid)
            stack, _ = _as_orbital_stack(orbitals, operator.grid)
            updates = []
            residual_values = []
            for index in range(n_orbitals):
                residual = applied[index] - eigenvalues[index] * stack[index]
                residual_values.append(mx.sqrt(mx.sum(mx.abs(residual) ** 2) * operator.grid.dv))
                updates.append(stack[index] - 0.35 * self.preconditioner.apply(residual, operator))
            orbitals = mx.array(_orthonormalize_numpy(np.array(mx.stack(updates)), operator.grid))
            eigenvalues = operator.rayleigh_quotients(orbitals)
            residuals = mx.stack(residual_values)
            if float(mx.max(residuals)) <= self.config.tolerance:
                converged = True
                break
            if iteration % self.config.max_subspace_size == 0:
                restart_count += 1
        residuals = orbital_residuals(orbitals, operator, eigenvalues)
        return DiagonalizationResult(
            eigenvalues=eigenvalues,
            orbitals=orbitals,
            residuals=residuals,
            orthonormality_error=orthonormality_error(orbitals, operator.grid),
            iterations=iteration,
            converged=converged,
            metadata={
                "solver": "davidson-preconditioned-residual",
                "subspace_size": min(
                    self.config.max_subspace_size,
                    self.config.max_iterations,
                ),
                "restart_count": restart_count,
            },
        )
