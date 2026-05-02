"""Pair-distance constraints for molecular dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell, as_mx_array


@dataclass(frozen=True)
class DistanceConstraints:
    """Fixed pair-distance constraints."""

    pairs: object
    distances: object
    tolerance: float = 1e-5
    max_iterations: int = 20

    def __post_init__(self) -> None:
        pairs = np.asarray(self.pairs, dtype=np.int32)
        if pairs.size == 0:
            pairs = np.empty((0, 2), dtype=np.int32)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            msg = "constraint pairs must have shape (n, 2)"
            raise ValueError(msg)
        if np.any(pairs < 0):
            msg = "constraint pairs must be non-negative"
            raise ValueError(msg)
        distances = np.asarray(self.distances, dtype=np.float32)
        if distances.ndim == 0:
            distances = np.full((pairs.shape[0],), float(distances), dtype=np.float32)
        if distances.shape != (pairs.shape[0],):
            msg = "constraint distances must be scalar or have shape (n_constraints,)"
            raise ValueError(msg)
        if np.any(distances <= 0.0):
            msg = "constraint distances must be positive"
            raise ValueError(msg)
        if self.tolerance <= 0.0:
            msg = "constraint tolerance must be positive"
            raise ValueError(msg)
        if self.max_iterations <= 0:
            msg = "max_iterations must be positive"
            raise ValueError(msg)
        max_pair_index = int(np.max(pairs)) if pairs.size else -1
        object.__setattr__(self, "pairs", mx.array(pairs, dtype=mx.int32))
        object.__setattr__(self, "distances", as_mx_array(distances))
        object.__setattr__(self, "_max_pair_index", max_pair_index)

    def _displacements(self, positions: mx.array, cell: Cell | None) -> mx.array:
        i = self.pairs[:, 0]
        j = self.pairs[:, 1]
        displacement = positions[i] - positions[j]
        if cell is not None:
            displacement = cell.minimum_image(displacement)
        return displacement

    def max_error(self, positions, cell: Cell | None = None) -> mx.array:
        """Return the maximum absolute distance error."""

        positions = as_mx_array(positions)
        if self.pairs.shape[0] == 0:
            return mx.sum(positions[:, 0] * 0.0)
        displacement = self._displacements(positions, cell)
        distances = mx.sqrt(mx.maximum(mx.sum(displacement * displacement, axis=-1), 1e-12))
        return mx.max(mx.abs(distances - self.distances))

    def apply_positions(
        self,
        positions,
        masses,
        cell: Cell | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Project positions onto the configured pair distances."""

        constrained = as_mx_array(positions)
        masses = as_mx_array(masses)
        if self.pairs.shape[0] == 0:
            return constrained, self.max_error(constrained, cell)
        if self._max_pair_index >= constrained.shape[0]:
            msg = "constraint pair index outside positions"
            raise ValueError(msg)

        i = self.pairs[:, 0]
        j = self.pairs[:, 1]
        inverse_masses = 1.0 / masses
        for _ in range(self.max_iterations):
            displacement = self._displacements(constrained, cell)
            distances = mx.sqrt(mx.maximum(mx.sum(displacement * displacement, axis=-1), 1e-12))
            errors = distances - self.distances
            unit = displacement / distances[:, None]
            weight_i = inverse_masses[i] / (inverse_masses[i] + inverse_masses[j])
            weight_j = inverse_masses[j] / (inverse_masses[i] + inverse_masses[j])
            correction = errors[:, None] * unit
            constrained = constrained.at[i].add(-weight_i[:, None] * correction)
            constrained = constrained.at[j].add(weight_j[:, None] * correction)
            if cell is not None:
                constrained = cell.wrap(constrained)
        return constrained, self.max_error(constrained, cell)

    def apply_velocities(
        self,
        positions,
        velocities,
        masses,
        cell: Cell | None = None,
    ) -> mx.array:
        """Remove constrained relative velocity components."""

        positions = as_mx_array(positions)
        constrained = as_mx_array(velocities)
        masses = as_mx_array(masses)
        if self.pairs.shape[0] == 0:
            return constrained
        if self._max_pair_index >= positions.shape[0]:
            msg = "constraint pair index outside positions"
            raise ValueError(msg)

        i = self.pairs[:, 0]
        j = self.pairs[:, 1]
        inverse_masses = 1.0 / masses
        displacement = self._displacements(positions, cell)
        distances = mx.sqrt(mx.maximum(mx.sum(displacement * displacement, axis=-1), 1e-12))
        unit = displacement / distances[:, None]
        relative_velocity = constrained[i] - constrained[j]
        relative_along_bond = mx.sum(relative_velocity * unit, axis=-1)
        weight_i = inverse_masses[i] / (inverse_masses[i] + inverse_masses[j])
        weight_j = inverse_masses[j] / (inverse_masses[i] + inverse_masses[j])
        correction = relative_along_bond[:, None] * unit
        constrained = constrained.at[i].add(-weight_i[:, None] * correction)
        constrained = constrained.at[j].add(weight_j[:, None] * correction)
        return constrained


__all__ = ["DistanceConstraints"]
