"""Compact cutoff-projected plane-wave bases for periodic DFT."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from math import pi

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft._compact import (
    _CompactBasisLayout,
    _CompactBatch,
    _CompactLaneState,
    _require_layout,
)
from mlx_atomistic.dft.grids import RealSpaceGrid, ReciprocalGrid

_RECIPROCAL_CACHE_LIMIT = 8
_RECIPROCAL_CACHE: OrderedDict[tuple[object, ...], ReciprocalGrid] = OrderedDict()


def _reciprocal_cache_key(grid: RealSpaceGrid) -> tuple[object, ...]:
    matrix = np.asarray(grid.cell.matrix, dtype=np.float64)
    return (
        grid.shape,
        tuple(float(value) for value in matrix.reshape(-1)),
        str(mx.default_device()),
    )


def _shared_reciprocal_grid(grid: RealSpaceGrid) -> ReciprocalGrid:
    key = _reciprocal_cache_key(grid)
    cached = _RECIPROCAL_CACHE.get(key)
    if cached is not None:
        _RECIPROCAL_CACHE.move_to_end(key)
        return cached
    reciprocal = ReciprocalGrid.from_real_space(grid)
    _RECIPROCAL_CACHE[key] = reciprocal
    _RECIPROCAL_CACHE.move_to_end(key)
    while len(_RECIPROCAL_CACHE) > _RECIPROCAL_CACHE_LIMIT:
        _RECIPROCAL_CACHE.popitem(last=False)
    return reciprocal


def _real_stack(
    array: mx.array,
    shape: tuple[int, int, int],
) -> tuple[mx.array, bool]:
    values = mx.array(array).astype(mx.complex64)
    if values.shape == shape:
        return mx.reshape(values, (1, *shape)), True
    if len(values.shape) == 4 and values.shape[1:] == shape:
        return values, False
    msg = "plane-wave arrays must have shape grid.shape or (n, *grid.shape)"
    raise ValueError(msg)


@dataclass(frozen=True)
class PlaneWaveBasis:
    """Compact plane-wave basis with a public full-grid compatibility facade.

    Runtime-owned coefficients use canonical ascending FFT-index order with
    shape ``(vectors, active_count)``. Public methods continue to accept and
    return full FFT grids, materializing them only at the call boundary.

    Args:
        grid: Uniform orthorhombic real-space grid.
        cutoff_hartree: Kinetic energy cutoff in Hartree.
        kpoint_cartesian: Bloch k-point in inverse bohr. Defaults to Gamma.
        reciprocal_grid: Optional shared reciprocal descriptor. Compatible
            bases created for one SCF calculation should share this object.
        lane_label: Stable label used to distinguish equal-basis runtime lanes.
    """

    grid: RealSpaceGrid
    cutoff_hartree: float
    kpoint_cartesian: tuple[float, float, float]
    _layout: _CompactBasisLayout

    def __init__(
        self,
        grid: RealSpaceGrid,
        cutoff_hartree: float,
        kpoint_cartesian: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        reciprocal_grid: ReciprocalGrid | None = None,
        lane_label: str = "lane:0",
    ):
        if len(kpoint_cartesian) != 3:
            msg = "kpoint_cartesian must have three components"
            raise ValueError(msg)
        reciprocal = (
            _shared_reciprocal_grid(grid)
            if reciprocal_grid is None
            else reciprocal_grid
        )
        if reciprocal.real_grid.shape != grid.shape or not np.array_equal(
            np.asarray(reciprocal.real_grid.cell.matrix, dtype=np.float64),
            np.asarray(grid.cell.matrix, dtype=np.float64),
        ):
            msg = "reciprocal_grid must match the basis real-space grid"
            raise ValueError(msg)
        layout = _CompactBasisLayout.build(
            reciprocal,
            cutoff_hartree,
            kpoint_cartesian,
            lane_label=lane_label,
        )
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "cutoff_hartree", float(cutoff_hartree))
        object.__setattr__(self, "kpoint_cartesian", layout.kpoint_cartesian)
        object.__setattr__(self, "_layout", layout)

    @classmethod
    def from_reduced_kpoint(
        cls,
        grid: RealSpaceGrid,
        cutoff_hartree: float,
        reduced_kpoint: Sequence[float],
        *,
        reciprocal_grid: ReciprocalGrid | None = None,
        lane_label: str = "lane:0",
    ) -> PlaneWaveBasis:
        """Build a basis from fractional reciprocal coordinates.

        Args:
            grid: Uniform orthorhombic real-space grid.
            cutoff_hartree: Kinetic energy cutoff in Hartree.
            reduced_kpoint: Fractional coordinates along reciprocal cell axes.
            reciprocal_grid: Optional shared reciprocal descriptor.
            lane_label: Stable runtime lane label.

        Returns:
            A basis whose Cartesian k-point is ``2*pi*k_i/L_i``.
        """

        if len(reduced_kpoint) != 3:
            msg = "reduced_kpoint must have three components"
            raise ValueError(msg)
        lengths = np.asarray(grid.lengths, dtype=np.float64)
        reduced = np.asarray(reduced_kpoint, dtype=np.float64)
        cartesian = 2.0 * pi * reduced / lengths
        return cls(
            grid,
            cutoff_hartree,
            cartesian,
            reciprocal_grid=reciprocal_grid,
            lane_label=lane_label,
        )

    @property
    def volume(self) -> float:
        """Cell volume in bohr cubed."""

        return self.grid.volume

    @property
    def reciprocal_grid(self) -> ReciprocalGrid:
        """Shared reciprocal descriptor for this basis."""

        return self._layout.reciprocal

    @property
    def reciprocal_vectors(self) -> mx.array:
        """Shared full-grid unshifted reciprocal vectors."""

        return self._layout.reciprocal.vectors

    @property
    def shifted_vectors(self) -> mx.array:
        """Fresh full-grid shifted vectors for compatibility callers."""

        return self._layout.shifted_vectors_fresh()

    @property
    def kinetic_energies(self) -> mx.array:
        """Fresh full-grid kinetic energies for compatibility callers."""

        return self._layout.kinetic_energies_fresh()

    @property
    def mask(self) -> mx.array:
        """Fresh full-grid cutoff mask for compatibility callers."""

        return self._layout.mask_fresh()

    @property
    def active_count(self) -> int:
        """Number of active plane waves."""

        return self._layout.active_count

    @property
    def active_flat_indices(self) -> mx.array:
        """Canonical ascending flat FFT indices."""

        return self._layout.active_flat_indices

    @property
    def active_integer_g(self) -> mx.array:
        """Ordered exact integer ``G`` coordinates."""

        return self._layout.active_integer_g

    @property
    def active_shifted_vectors(self) -> mx.array:
        """Shifted reciprocal vectors in compact order."""

        return self._layout.active_shifted_vectors

    @property
    def active_kinetic_energies(self) -> mx.array:
        """Kinetic energies in compact order."""

        return self._layout.active_kinetic_energies

    @property
    def order_fingerprint(self) -> str:
        """Canonical active-order fingerprint."""

        return self._layout.order_fingerprint

    @property
    def basis_fingerprint(self) -> str:
        """Complete compact basis fingerprint."""

        return self._layout.basis_fingerprint

    @property
    def lane_id(self) -> str:
        """Immutable identity of this runtime lane."""

        return self._layout.lane_id

    def _state_from_full(self, coefficients: mx.array) -> tuple[_CompactLaneState, bool]:
        values, was_single = self._layout.pack_full(coefficients)
        return _CompactLaneState(values, self._layout), was_single

    def _state_from_compact(
        self,
        coefficients: mx.array,
        *,
        kind: str = "coefficients",
    ) -> _CompactLaneState:
        return _CompactLaneState(coefficients, self._layout, kind)

    def _validate_state(self, state: _CompactLaneState) -> None:
        _require_layout(state, self._layout)

    def _to_real_compact(self, state: _CompactLaneState) -> mx.array:
        self._validate_state(state)
        batch = _CompactBatch.from_states([state])
        return batch.to_real()[0]

    def _from_real_compact(self, orbitals: mx.array) -> _CompactLaneState:
        stack, _ = _real_stack(orbitals, self.grid.shape)
        empty = self._state_from_compact(
            mx.zeros((stack.shape[0], self.active_count), dtype=mx.complex64)
        )
        batch = _CompactBatch.from_states([empty])
        return batch.unpad(batch.from_real(stack[None, ...]))[0]

    def _coefficient_norms_compact(self, values: mx.array) -> mx.array:
        return mx.sqrt(mx.sum(mx.abs(values) ** 2, axis=1))

    def _normalize_compact(self, values: mx.array) -> mx.array:
        norms = self._coefficient_norms_compact(values)
        if bool(mx.any(norms <= 0.0)):
            msg = "cannot normalize a zero plane-wave orbital"
            raise ValueError(msg)
        return values / norms[:, None]

    def _overlap_matrix_compact(self, values: mx.array) -> mx.array:
        return values @ mx.conjugate(mx.transpose(values))

    def _orthonormalize_compact(self, values: mx.array) -> mx.array:
        stack = mx.array(values).astype(mx.complex64)
        if len(stack.shape) != 2 or stack.shape[1] != self.active_count:
            msg = "compact orbitals must have shape (vectors, active_count)"
            raise ValueError(msg)
        orbital_count = int(stack.shape[0])
        if orbital_count <= 0 or orbital_count > self.active_count:
            msg = "orbital count must be between one and the active basis size"
            raise ValueError(msg)
        vectors: list[mx.array] = []
        for index in range(orbital_count):
            vector = stack[index]
            for _ in range(2):
                for accepted in vectors:
                    overlap = mx.sum(mx.conjugate(accepted) * vector)
                    vector = vector - overlap * accepted
            norm = mx.sqrt(mx.real(mx.sum(mx.conjugate(vector) * vector)))
            if float(norm) <= 1e-12:
                msg = "plane-wave orbital stack is linearly dependent"
                raise ValueError(msg)
            vectors.append(vector / norm)
        return mx.stack(vectors, axis=0)

    def project(self, coefficients: mx.array) -> mx.array:
        """Zero coefficients outside the admitted kinetic cutoff.

        Args:
            coefficients: One coefficient grid or a stack of grids.

        Returns:
            A fresh full-grid value with every inactive entry exactly zero.
        """

        state, was_single = self._state_from_full(coefficients)
        return self._layout.unpack_fresh(state.values, single=was_single)

    def to_real(self, coefficients: mx.array) -> mx.array:
        """Transform reciprocal coefficients to normalized real-space orbitals.

        Args:
            coefficients: One coefficient grid or a stack of grids.

        Returns:
            Real-space complex orbital values with matching leading shape.
        """

        state, was_single = self._state_from_full(coefficients)
        orbitals = self._to_real_compact(state)
        return orbitals[0] if was_single else orbitals

    def to_coefficients(self, orbitals: mx.array) -> mx.array:
        """Transform real-space orbitals into the admitted coefficient basis.

        Args:
            orbitals: One real-space orbital grid or a stack of grids.

        Returns:
            Fresh cutoff-projected reciprocal coefficients.
        """

        _, was_single = _real_stack(orbitals, self.grid.shape)
        state = self._from_real_compact(orbitals)
        return self._layout.unpack_fresh(state.values, single=was_single)

    def coefficient_norms(self, coefficients: mx.array) -> mx.array:
        """Return coefficient-space norms for one orbital or a stack.

        Args:
            coefficients: One coefficient grid or a stack of grids.

        Returns:
            One norm per orbital.
        """

        state, _ = self._state_from_full(coefficients)
        return self._coefficient_norms_compact(state.values)

    def real_norms(self, orbitals: mx.array) -> mx.array:
        """Return real-space integral norms for one orbital or a stack.

        Args:
            orbitals: One real-space orbital grid or a stack of grids.

        Returns:
            One ``sqrt(integral |psi|^2)`` value per orbital.
        """

        stack, _ = _real_stack(orbitals, self.grid.shape)
        return mx.sqrt(mx.sum(mx.abs(stack) ** 2, axis=(1, 2, 3)) * self.grid.dv)

    def normalize(self, coefficients: mx.array) -> mx.array:
        """Normalize each orbital in compact coefficient space.

        Args:
            coefficients: One coefficient grid or a stack of grids.

        Returns:
            Fresh full-grid unit-norm coefficients.

        Raises:
            ValueError: If any orbital has zero norm.
        """

        state, was_single = self._state_from_full(coefficients)
        normalized = self._normalize_compact(state.values)
        return self._layout.unpack_fresh(normalized, single=was_single)

    def overlap_matrix(self, coefficients: mx.array) -> mx.array:
        """Return the orbital overlap matrix in compact coefficient space.

        Args:
            coefficients: Orbital stack with shape ``(n, *grid.shape)``.

        Returns:
            Complex Hermitian overlap matrix with shape ``(n, n)``.
        """

        state, _ = self._state_from_full(coefficients)
        return self._overlap_matrix_compact(state.values)

    def orthonormalize(self, coefficients: mx.array) -> mx.array:
        """Orthonormalize an admitted orbital stack in compact space.

        Args:
            coefficients: Orbital stack with shape ``(n, *grid.shape)``.

        Returns:
            Fresh orthonormal full-grid coefficients.

        Raises:
            ValueError: If the stack is empty, too wide, or rank deficient.
        """

        state, _ = self._state_from_full(coefficients)
        orthonormal = self._orthonormalize_compact(state.values)
        return self._layout.unpack_fresh(orthonormal)

    def apply_kinetic(self, coefficients: mx.array) -> mx.array:
        """Apply ``0.5 |G+k|^2`` in compact coefficient space.

        Args:
            coefficients: One coefficient grid or a stack of grids.

        Returns:
            Fresh full-grid kinetic action with exact inactive zeros.
        """

        state, was_single = self._state_from_full(coefficients)
        applied = state.values * self.active_kinetic_energies[None, :]
        return self._layout.unpack_fresh(applied, single=was_single)

    def apply_local(self, coefficients: mx.array, potential: mx.array) -> mx.array:
        """Apply a periodic local potential through one batched FFT pair.

        Args:
            coefficients: One coefficient grid or a stack of grids.
            potential: Real local potential with shape ``grid.shape``.

        Returns:
            Fresh full-grid local-potential action.

        Raises:
            ValueError: If the potential shape does not match the grid.
        """

        state, was_single = self._state_from_full(coefficients)
        batch = _CompactBatch.from_states([state])
        compact = batch.unpad(batch.apply_local(potential))[0]
        return self._layout.unpack_fresh(compact.values, single=was_single)

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
