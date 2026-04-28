"""Kohn-Sham operator and tiny-grid numerical references."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft.fft import fft3, ifft3
from mlx_atomistic.dft.grids import RealSpaceGrid, ReciprocalGrid
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


def apply_kinetic(orbitals: mx.array, grid: RealSpaceGrid) -> mx.array:
    """Apply the plane-wave kinetic operator ``-1/2 ∇²``."""

    stack, was_single = _as_orbital_stack(orbitals, grid)
    reciprocal = ReciprocalGrid.from_real_space(grid)
    applied = mx.stack(
        [ifft3(0.5 * reciprocal.g2 * fft3(stack[index])) for index in range(stack.shape[0])],
        axis=0,
    )
    return _restore_orbital_shape(applied, was_single)


def apply_local_potential(orbitals: mx.array, local_potential: mx.array) -> mx.array:
    """Apply a local multiplicative potential to orbitals."""

    return mx.array(orbitals) * mx.array(local_potential)


def apply_hartree_xc_potential(
    orbitals: mx.array,
    hartree: mx.array,
    xc_potential: mx.array,
) -> mx.array:
    """Apply Hartree plus exchange-correlation potential to orbitals."""

    return mx.array(orbitals) * (mx.array(hartree) + mx.array(xc_potential))


def apply_hamiltonian(
    orbitals: mx.array,
    effective_potential: mx.array,
    grid: RealSpaceGrid,
) -> mx.array:
    """Apply ``T + V_eff`` using an explicit effective potential."""

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

    @classmethod
    def from_density(
        cls,
        grid: RealSpaceGrid,
        local_potential: mx.array,
        density: mx.array,
        *,
        xc_functional: ExchangeCorrelationFunctional | None = None,
        density_floor: float = 1e-12,
    ) -> KohnShamOperator:
        """Build an operator from ``ρ`` and a local potential."""

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
        )

    def apply_kinetic(self, orbitals: mx.array) -> mx.array:
        """Apply only the kinetic part."""

        return apply_kinetic(orbitals, self.grid)

    def apply_local_potential(self, orbitals: mx.array) -> mx.array:
        """Apply only the external local potential."""

        return apply_local_potential(orbitals, self.local_potential)

    def apply_hartree_xc_potential(self, orbitals: mx.array) -> mx.array:
        """Apply only the Hartree plus exchange-correlation potential."""

        return apply_hartree_xc_potential(orbitals, self.hartree, self.xc_potential)

    def apply_hamiltonian(self, orbitals: mx.array) -> mx.array:
        """Apply ``H = T + V_eff``."""

        return self.apply_kinetic(orbitals) + apply_local_potential(
            orbitals,
            self.effective_potential,
        )

    def rayleigh_quotients(self, orbitals: mx.array) -> mx.array:
        """Return ``<ψᵢ|H|ψᵢ>/<ψᵢ|ψᵢ>`` for each orbital."""

        stack, _ = _as_orbital_stack(orbitals, self.grid)
        applied, _ = _as_orbital_stack(self.apply_hamiltonian(stack), self.grid)
        values = []
        for index in range(stack.shape[0]):
            numerator = mx.sum(mx.conjugate(stack[index]) * applied[index]) * self.grid.dv
            denominator = mx.sum(mx.conjugate(stack[index]) * stack[index]) * self.grid.dv
            values.append(mx.real(numerator / denominator))
        return mx.stack(values)


def orthonormality_error(orbitals: mx.array, grid: RealSpaceGrid) -> float:
    """Return max absolute overlap error from the identity matrix."""

    stack, _ = _as_orbital_stack(orbitals, grid)
    flat = np.array(stack, dtype=np.complex128).reshape(stack.shape[0], grid.size)
    overlap = flat @ flat.conjugate().T * grid.dv
    return float(np.max(np.abs(overlap - np.eye(stack.shape[0]))))


def orbital_residuals(
    orbitals: mx.array,
    operator: KohnShamOperator,
    eigenvalues: mx.array | None = None,
) -> mx.array:
    """Return ``||Hψᵢ - εᵢψᵢ||`` for each orbital."""

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

    def to_dict(self) -> dict:
        """Return a JSON-safe summary."""

        return {
            "eigenvalues": np.array(self.eigenvalues).tolist(),
            "residuals": np.array(self.residuals).tolist(),
            "orthonormality_error": self.orthonormality_error,
            "iterations": self.iterations,
            "converged": self.converged,
        }


@dataclass(frozen=True)
class DenseHamiltonianReference:
    """Explicit dense Hamiltonian for tiny-grid validation."""

    operator: KohnShamOperator

    def matrix(self) -> np.ndarray:
        """Build the dense matrix by applying the operator to basis vectors."""

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
        """Apply the dense matrix to a single orbital."""

        flat = np.array(orbital, dtype=np.complex128).reshape(self.operator.grid.size)
        applied = self.matrix() @ flat
        return mx.array(applied.reshape(self.operator.grid.shape).astype(np.complex64))

    def diagonalize(self, n_orbitals: int) -> DiagonalizationResult:
        """Diagonalize the dense reference matrix."""

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
        )
