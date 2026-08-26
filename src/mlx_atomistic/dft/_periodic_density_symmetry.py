"""Point-group reconstruction for periodic SCF densities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft.kpoints import KPointMesh, TimeReversalOwnership

_OperationKey = tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class _DensitySymmetryTerm:
    weight: float
    permutation_index: int


@dataclass(frozen=True)
class _DensitySymmetryPlan:
    """Reusable device permutations that expand representative densities."""

    grid_shape: tuple[int, int, int]
    permutations: tuple[mx.array, ...] = field(repr=False, compare=False)
    terms_by_explicit_index: tuple[tuple[_DensitySymmetryTerm, ...], ...]
    full_point_count: int

    @property
    def persistent_bytes(self) -> int:
        """Return storage retained by the grid permutations."""

        return sum(permutation.size * 4 for permutation in self.permutations)

    def expand(self, explicit_index: int | None, density: mx.array) -> mx.array:
        """Expand one owner density over its point-group and time-reversal orbits."""

        if explicit_index is None or not 0 <= explicit_index < len(
            self.terms_by_explicit_index
        ):
            raise ValueError("symmetry density expansion requires a valid explicit index")
        values = mx.real(mx.array(density))
        if values.shape != self.grid_shape:
            raise ValueError("symmetry density expansion shape differs from its grid")
        terms = self.terms_by_explicit_index[explicit_index]
        if not terms:
            raise ValueError("symmetry density expansion requires an owner k-point")
        flat = mx.reshape(values, (-1,))
        expanded = [
            term.weight
            * mx.reshape(
                mx.take(flat, self.permutations[term.permutation_index]),
                self.grid_shape,
            )
            for term in terms
        ]
        return sum(expanded[1:], expanded[0])


def _grid_source_permutation(
    grid_shape: Sequence[int],
    reciprocal_operation: Sequence[Sequence[int]],
) -> np.ndarray:
    """Map each target FFT index to its symmetry-related source index."""

    shape = np.asarray(tuple(grid_shape), dtype=np.int64)
    operation = np.asarray(reciprocal_operation, dtype=np.int64)
    if shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError("symmetry density grid shape must contain three positive values")
    if operation.shape != (3, 3):
        raise ValueError("symmetry density operation must be a 3 x 3 matrix")
    target = np.indices(tuple(int(value) for value in shape), dtype=np.int64)
    target = target.reshape(3, -1).T
    source_scaled = (target.astype(np.float64) / shape) @ operation * shape
    source_indices = np.rint(source_scaled).astype(np.int64)
    if not np.allclose(source_scaled, source_indices, rtol=0.0, atol=1.0e-10):
        raise ValueError("symmetry operation is incompatible with the FFT grid shape")
    source_indices = np.remainder(source_indices, shape)
    source_flat = np.ravel_multi_index(source_indices.T, tuple(shape)).astype(np.int32)
    if np.unique(source_flat).size != source_flat.size:
        raise ValueError("symmetry operation is not bijective on the FFT grid")
    return source_flat


def _operation_key(operation: Sequence[Sequence[int]]) -> _OperationKey:
    return tuple(tuple(int(value) for value in row) for row in operation)


def _build_density_symmetry_plan(
    kpoint_mesh: KPointMesh,
    ownership: TimeReversalOwnership,
    grid_shape: Sequence[int],
) -> _DensitySymmetryPlan | None:
    """Build exact grid permutations for a point-group-reduced k-point mesh."""

    reduction = kpoint_mesh._symmetry_reduction
    if reduction is None:
        return None
    if len(ownership.entries) != len(reduction.orbits):
        raise ValueError("symmetry density ownership differs from the reduced mesh")

    weights_by_owner: list[dict[_OperationKey, float]] = [
        {} for _ in ownership.entries
    ]
    for explicit_index, orbit in enumerate(reduction.orbits):
        owner_index = ownership.entry_for(explicit_index).owner_index
        owner_weights = weights_by_owner[owner_index]
        for member in orbit.members:
            key = _operation_key(member.reciprocal_operation)
            owner_weights[key] = owner_weights.get(key, 0.0) + member.original_weight

    for entry in ownership.entries:
        weights = weights_by_owner[entry.explicit_index]
        if entry.owner_index != entry.explicit_index:
            if weights:
                raise ValueError("symmetry density partner unexpectedly owns orbit weights")
            continue
        if not np.isclose(
            sum(weights.values()),
            entry.aggregated_weight,
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise ValueError("symmetry density orbit weights differ from k-point ownership")

    operation_keys = tuple(
        sorted({key for weights in weights_by_owner for key in weights})
    )
    parsed_shape = tuple(int(value) for value in grid_shape)
    unique_permutations: list[np.ndarray] = []
    permutation_by_bytes: dict[bytes, int] = {}
    permutation_indices: dict[_OperationKey, int] = {}
    for key in operation_keys:
        permutation = _grid_source_permutation(parsed_shape, key)
        permutation_bytes = permutation.tobytes()
        permutation_index = permutation_by_bytes.get(permutation_bytes)
        if permutation_index is None:
            permutation_index = len(unique_permutations)
            permutation_by_bytes[permutation_bytes] = permutation_index
            unique_permutations.append(permutation)
        permutation_indices[key] = permutation_index
    permutations = tuple(
        mx.array(permutation, dtype=mx.int32) for permutation in unique_permutations
    )
    terms_by_explicit_index = []
    for weights in weights_by_owner:
        weights_by_permutation: dict[int, float] = {}
        for key, weight in weights.items():
            permutation_index = permutation_indices[key]
            weights_by_permutation[permutation_index] = (
                weights_by_permutation.get(permutation_index, 0.0) + weight
            )
        terms_by_explicit_index.append(
            tuple(
                _DensitySymmetryTerm(weight, permutation_index)
                for permutation_index, weight in sorted(weights_by_permutation.items())
            )
        )
    frozen_terms = tuple(terms_by_explicit_index)
    total_weight = sum(
        term.weight for terms in frozen_terms for term in terms
    )
    if not np.isclose(total_weight, 1.0, rtol=1.0e-12, atol=1.0e-15):
        raise ValueError("symmetry density plan does not cover the full k-point mesh")
    return _DensitySymmetryPlan(
        grid_shape=parsed_shape,
        permutations=permutations,
        terms_by_explicit_index=frozen_terms,
        full_point_count=reduction.full_point_count,
    )
