"""Cutoff-projected fixed-shape plane-wave bases for periodic DFT."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import pi, sqrt

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft.grids import RealSpaceGrid, ReciprocalGrid


def _as_stack(array: mx.array, shape: tuple[int, int, int]) -> tuple[mx.array, bool]:
    values = mx.array(array)
    if values.shape == shape:
        return mx.reshape(values, (1, *shape)), True
    if len(values.shape) == 4 and values.shape[1:] == shape:
        return values, False
    msg = "plane-wave arrays must have shape grid.shape or (n, *grid.shape)"
    raise ValueError(msg)


def _restore(stack: mx.array, was_single: bool) -> mx.array:
    return stack[0] if was_single else stack


@dataclass(frozen=True)
class PlaneWaveBasis:
    """Fixed FFT-grid plane-wave basis with a per-k-point kinetic cutoff.

    Coefficients use unit Euclidean norm. Real-space orbitals use the periodic
    normalization ``psi(r) = sum_G c_G exp(i G.r) / sqrt(volume)`` so the
    real-space integral and coefficient-space norm are identical.

    Args:
        grid: Uniform orthorhombic real-space grid.
        cutoff_hartree: Kinetic energy cutoff in Hartree.
        kpoint_cartesian: Bloch k-point in inverse bohr. Defaults to Gamma.
    """

    grid: RealSpaceGrid
    cutoff_hartree: float
    kpoint_cartesian: tuple[float, float, float]
    reciprocal_vectors: mx.array
    shifted_vectors: mx.array
    kinetic_energies: mx.array
    mask: mx.array
    active_count: int

    def __init__(
        self,
        grid: RealSpaceGrid,
        cutoff_hartree: float,
        kpoint_cartesian: Sequence[float] = (0.0, 0.0, 0.0),
    ):
        if cutoff_hartree <= 0.0:
            msg = "cutoff_hartree must be positive"
            raise ValueError(msg)
        if len(kpoint_cartesian) != 3:
            msg = "kpoint_cartesian must have three components"
            raise ValueError(msg)
        reciprocal = ReciprocalGrid.from_real_space(grid)
        vectors_np = np.asarray(reciprocal.vectors, dtype=np.float64)
        kpoint = np.asarray(kpoint_cartesian, dtype=np.float64)
        shifted_np = vectors_np + kpoint.reshape((1, 1, 1, 3))
        kinetic_np = 0.5 * np.sum(shifted_np * shifted_np, axis=-1)
        mask_np = kinetic_np <= float(cutoff_hartree) + 1e-12
        active_count = int(np.count_nonzero(mask_np))
        if active_count == 0:
            msg = "plane-wave cutoff admits no reciprocal coefficients"
            raise ValueError(msg)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "cutoff_hartree", float(cutoff_hartree))
        object.__setattr__(self, "kpoint_cartesian", tuple(float(value) for value in kpoint))
        object.__setattr__(self, "reciprocal_vectors", mx.array(vectors_np.astype(np.float32)))
        object.__setattr__(self, "shifted_vectors", mx.array(shifted_np.astype(np.float32)))
        object.__setattr__(self, "kinetic_energies", mx.array(kinetic_np.astype(np.float32)))
        object.__setattr__(self, "mask", mx.array(mask_np))
        object.__setattr__(self, "active_count", active_count)

    @classmethod
    def from_reduced_kpoint(
        cls,
        grid: RealSpaceGrid,
        cutoff_hartree: float,
        reduced_kpoint: Sequence[float],
    ) -> PlaneWaveBasis:
        """Build a basis from fractional reciprocal coordinates.

        Args:
            grid: Uniform orthorhombic real-space grid.
            cutoff_hartree: Kinetic energy cutoff in Hartree.
            reduced_kpoint: Fractional coordinates along reciprocal cell axes.

        Returns:
            A basis whose Cartesian k-point is ``2*pi*k_i/L_i``.
        """

        if len(reduced_kpoint) != 3:
            msg = "reduced_kpoint must have three components"
            raise ValueError(msg)
        lengths = np.asarray(grid.lengths, dtype=np.float64)
        reduced = np.asarray(reduced_kpoint, dtype=np.float64)
        cartesian = 2.0 * pi * reduced / lengths
        return cls(grid, cutoff_hartree, cartesian)

    @property
    def volume(self) -> float:
        """Cell volume in bohr cubed."""

        return self.grid.volume

    def project(self, coefficients: mx.array) -> mx.array:
        """Zero coefficients outside the admitted kinetic cutoff.

        Args:
            coefficients: One coefficient grid or a stack of grids.

        Returns:
            Coefficients with every inactive entry set exactly to zero.
        """

        stack, was_single = _as_stack(coefficients, self.grid.shape)
        projected = mx.where(self.mask[None, ...], stack, mx.zeros_like(stack))
        return _restore(projected, was_single)

    def to_real(self, coefficients: mx.array) -> mx.array:
        """Transform reciprocal coefficients to normalized real-space orbitals.

        Args:
            coefficients: One coefficient grid or a stack of grids.

        Returns:
            Real-space complex orbital values with matching leading shape.
        """

        stack, was_single = _as_stack(self.project(coefficients), self.grid.shape)
        scale = self.grid.size / sqrt(self.volume)
        orbitals = mx.stack([mx.fft.ifftn(item) * scale for item in stack], axis=0)
        return _restore(orbitals, was_single)

    def to_coefficients(self, orbitals: mx.array) -> mx.array:
        """Transform real-space orbitals into the admitted coefficient basis.

        Args:
            orbitals: One real-space orbital grid or a stack of grids.

        Returns:
            Cutoff-projected reciprocal coefficients.
        """

        stack, was_single = _as_stack(orbitals, self.grid.shape)
        scale = sqrt(self.volume) / self.grid.size
        coefficients = mx.stack([mx.fft.fftn(item) * scale for item in stack], axis=0)
        projected = self.project(coefficients)
        projected_stack, _ = _as_stack(projected, self.grid.shape)
        return _restore(projected_stack, was_single)

    def coefficient_norms(self, coefficients: mx.array) -> mx.array:
        """Return coefficient-space norms for one orbital or a stack.

        Args:
            coefficients: One coefficient grid or a stack of grids.

        Returns:
            One norm per orbital.
        """

        stack, _ = _as_stack(self.project(coefficients), self.grid.shape)
        return mx.sqrt(mx.sum(mx.abs(stack) ** 2, axis=(1, 2, 3)))

    def real_norms(self, orbitals: mx.array) -> mx.array:
        """Return real-space integral norms for one orbital or a stack.

        Args:
            orbitals: One real-space orbital grid or a stack of grids.

        Returns:
            One ``sqrt(integral |psi|^2)`` value per orbital.
        """

        stack, _ = _as_stack(orbitals, self.grid.shape)
        return mx.sqrt(mx.sum(mx.abs(stack) ** 2, axis=(1, 2, 3)) * self.grid.dv)

    def normalize(self, coefficients: mx.array) -> mx.array:
        """Normalize each orbital in coefficient space.

        Args:
            coefficients: One coefficient grid or a stack of grids.

        Returns:
            Cutoff-projected unit-norm coefficients.

        Raises:
            ValueError: If any orbital has zero norm.
        """

        stack, was_single = _as_stack(self.project(coefficients), self.grid.shape)
        norms = self.coefficient_norms(stack)
        if bool(mx.any(norms <= 0.0)):
            msg = "cannot normalize a zero plane-wave orbital"
            raise ValueError(msg)
        normalized = stack / mx.reshape(norms, (-1, 1, 1, 1))
        return _restore(normalized, was_single)

    def overlap_matrix(self, coefficients: mx.array) -> mx.array:
        """Return the orbital overlap matrix in coefficient space.

        Args:
            coefficients: Orbital stack with shape ``(n, *grid.shape)``.

        Returns:
            Complex Hermitian overlap matrix with shape ``(n, n)``.
        """

        stack, _ = _as_stack(self.project(coefficients), self.grid.shape)
        flat = mx.reshape(stack, (stack.shape[0], self.grid.size))
        return flat @ mx.conjugate(mx.transpose(flat))

    def orthonormalize(self, coefficients: mx.array) -> mx.array:
        """Orthonormalize an admitted orbital stack with MLX QR.

        Args:
            coefficients: Orbital stack with shape ``(n, *grid.shape)``.

        Returns:
            Orthonormal cutoff-projected coefficients with the same shape.

        Raises:
            ValueError: If the stack is empty or wider than the active basis.
        """

        stack, _ = _as_stack(self.project(coefficients), self.grid.shape)
        orbital_count = int(stack.shape[0])
        if orbital_count <= 0 or orbital_count > self.active_count:
            msg = "orbital count must be between one and the active basis size"
            raise ValueError(msg)
        flat = mx.reshape(stack, (orbital_count, self.grid.size))
        vectors: list[mx.array] = []
        for index in range(orbital_count):
            vector = flat[index]
            # Reorthogonalized modified Gram-Schmidt stays on MLX and supports
            # complex arrays, unlike the current MLX QR implementation.
            for _ in range(2):
                for accepted in vectors:
                    overlap = mx.sum(mx.conjugate(accepted) * vector)
                    vector = vector - overlap * accepted
            norm = mx.sqrt(mx.real(mx.sum(mx.conjugate(vector) * vector)))
            if float(norm) <= 1e-12:
                msg = "plane-wave orbital stack is linearly dependent"
                raise ValueError(msg)
            vectors.append(vector / norm)
        orthonormal = mx.stack(vectors, axis=0)
        return mx.reshape(orthonormal, (orbital_count, *self.grid.shape))

    def apply_kinetic(self, coefficients: mx.array) -> mx.array:
        """Apply ``0.5 |G+k|^2`` in coefficient space.

        Args:
            coefficients: One coefficient grid or a stack of grids.

        Returns:
            Kinetic operator action with inactive entries kept at zero.
        """

        stack, was_single = _as_stack(self.project(coefficients), self.grid.shape)
        applied = stack * self.kinetic_energies[None, ...]
        return _restore(applied, was_single)

    def apply_local(self, coefficients: mx.array, potential: mx.array) -> mx.array:
        """Apply a periodic local potential and project back to the basis.

        Args:
            coefficients: One coefficient grid or a stack of grids.
            potential: Real local potential with shape ``grid.shape``.

        Returns:
            Coefficient-space local-potential action.

        Raises:
            ValueError: If the potential shape does not match the grid.
        """

        field = mx.array(potential)
        if field.shape != self.grid.shape:
            msg = "local potential must have shape grid.shape"
            raise ValueError(msg)
        stack, was_single = _as_stack(coefficients, self.grid.shape)
        real_stack, _ = _as_stack(self.to_real(stack), self.grid.shape)
        applied = self.to_coefficients(real_stack * field[None, ...])
        applied_stack, _ = _as_stack(applied, self.grid.shape)
        return _restore(applied_stack, was_single)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe basis metadata.

        Returns:
            Cutoff, k-point, FFT shape, active count, and normalization metadata.
        """

        return {
            "cutoff_hartree": self.cutoff_hartree,
            "kpoint_cartesian_bohr_inverse": list(self.kpoint_cartesian),
            "fft_shape": list(self.grid.shape),
            "active_count": self.active_count,
            "normalization": "unit-coefficients__real-integral-unit",
        }
