"""Pair-distance constraints for molecular dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell, as_mx_array


def _empty_pairs() -> mx.array:
    return mx.array(np.empty((0, 2), dtype=np.int32), dtype=mx.int32)


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


@dataclass(frozen=True)
class SettleWaterConstraints:
    """Analytical rigid-water constraints for `(oxygen, hydrogen, hydrogen)` triplets."""

    waters: object
    oh_distance: float = 1.0
    hh_distance: float = 1.6329932
    tolerance: float = 1e-5
    max_velocity_iterations: int = 200

    def __post_init__(self) -> None:
        waters = np.asarray(self.waters, dtype=np.int32)
        if waters.size == 0:
            waters = np.empty((0, 3), dtype=np.int32)
        if waters.ndim != 2 or waters.shape[1] != 3:
            msg = "SETTLE waters must have shape (n_waters, 3)"
            raise ValueError(msg)
        if np.any(waters < 0):
            msg = "SETTLE water atom indices must be non-negative"
            raise ValueError(msg)
        if any(len(set(row.tolist())) != 3 for row in waters):
            msg = "SETTLE water triplets must contain three distinct atom indices"
            raise ValueError(msg)
        if self.oh_distance <= 0.0 or self.hh_distance <= 0.0:
            msg = "SETTLE distances must be positive"
            raise ValueError(msg)
        if self.hh_distance >= 2.0 * self.oh_distance:
            msg = "SETTLE H-H distance must be shorter than twice the O-H distance"
            raise ValueError(msg)
        if self.tolerance <= 0.0:
            msg = "SETTLE tolerance must be positive"
            raise ValueError(msg)
        if self.max_velocity_iterations <= 0:
            msg = "SETTLE max_velocity_iterations must be positive"
            raise ValueError(msg)

        pair_rows = []
        for oxygen, hydrogen_a, hydrogen_b in waters:
            pair_rows.extend(
                [
                    (int(oxygen), int(hydrogen_a)),
                    (int(oxygen), int(hydrogen_b)),
                    (int(hydrogen_a), int(hydrogen_b)),
                ]
            )
        distances = np.tile(
            np.asarray([self.oh_distance, self.oh_distance, self.hh_distance], dtype=np.float32),
            waters.shape[0],
        )
        object.__setattr__(self, "waters", mx.array(waters, dtype=mx.int32))
        object.__setattr__(
            self,
            "_pair_constraints",
            DistanceConstraints(
                np.asarray(pair_rows, dtype=np.int32).reshape((-1, 2)),
                distances=distances,
                tolerance=self.tolerance,
                max_iterations=1,
            ),
        )
        object.__setattr__(self, "pairs", self._pair_constraints.pairs)
        object.__setattr__(self, "distances", self._pair_constraints.distances)
        max_index = int(np.max(waters)) if waters.size else -1
        object.__setattr__(self, "_max_pair_index", max_index)

    def max_error(self, positions, cell: Cell | None = None) -> mx.array:
        """Return the maximum absolute SETTLE distance error."""

        return self._pair_constraints.max_error(positions, cell)

    def apply_positions(
        self,
        positions,
        masses,
        cell: Cell | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Project water triplets onto the configured rigid geometry."""

        constrained = as_mx_array(positions)
        if self.pairs.shape[0] == 0:
            return constrained, self.max_error(constrained, cell)
        if self._max_pair_index >= constrained.shape[0]:
            msg = "SETTLE water atom index outside positions"
            raise ValueError(msg)
        waters = np.asarray(self.waters, dtype=np.int32)
        constrained_np = np.asarray(constrained, dtype=np.float32).copy()
        for oxygen, hydrogen_a, hydrogen_b in waters:
            origin = constrained_np[oxygen]
            displacements = constrained_np[[hydrogen_a, hydrogen_b]] - origin
            if cell is not None:
                displacements = np.asarray(cell.minimum_image(displacements), dtype=np.float32)
            first = displacements[0]
            second = displacements[1]
            bisector = _unit_or_fallback(first + second, first)
            difference = first - second
            difference = difference - np.dot(difference, bisector) * bisector
            difference = _unit_or_fallback(difference, _perpendicular_unit(bisector))

            half_hh = 0.5 * float(self.hh_distance)
            along_bisector = float(np.sqrt(float(self.oh_distance) ** 2 - half_hh * half_hh))
            constrained_np[hydrogen_a] = origin + along_bisector * bisector + half_hh * difference
            constrained_np[hydrogen_b] = origin + along_bisector * bisector - half_hh * difference
        constrained = as_mx_array(constrained_np)
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
        """Remove constrained relative velocity components for SETTLE pairs."""

        constrained = as_mx_array(velocities)
        for _ in range(self.max_velocity_iterations):
            constrained = self._pair_constraints.apply_velocities(
                positions,
                constrained,
                masses,
                cell,
            )
        return constrained


@dataclass(frozen=True)
class CompositeConstraints:
    """Apply multiple constraint objects through the standard constraint protocol."""

    constraints: tuple[object, ...]

    def __post_init__(self) -> None:
        if not self.constraints:
            msg = "CompositeConstraints requires at least one constraint object"
            raise ValueError(msg)
        pairs = []
        for constraint in self.constraints:
            constraint_pairs = getattr(constraint, "pairs", None)
            if constraint_pairs is None:
                msg = "constraint objects must expose pairs"
                raise ValueError(msg)
            pairs.append(np.asarray(constraint_pairs, dtype=np.int32).reshape((-1, 2)))
        object.__setattr__(
            self,
            "pairs",
            mx.array(np.concatenate(pairs, axis=0), dtype=mx.int32) if pairs else _empty_pairs(),
        )

    def max_error(self, positions, cell: Cell | None = None) -> mx.array:
        """Return the maximum absolute error across child constraints."""

        errors = [constraint.max_error(positions, cell) for constraint in self.constraints]
        return mx.max(mx.stack(errors))

    def apply_positions(
        self,
        positions,
        masses,
        cell: Cell | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Apply child position constraints in sequence."""

        constrained = as_mx_array(positions)
        for constraint in self.constraints:
            constrained, _ = constraint.apply_positions(constrained, masses, cell)
        return constrained, self.max_error(constrained, cell)

    def apply_velocities(
        self,
        positions,
        velocities,
        masses,
        cell: Cell | None = None,
    ) -> mx.array:
        """Apply child velocity constraints in sequence."""

        constrained = as_mx_array(velocities)
        for constraint in self.constraints:
            constrained = constraint.apply_velocities(positions, constrained, masses, cell)
        return constrained


def _unit_or_fallback(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8 and np.isfinite(norm):
        return (vector / norm).astype(np.float32)
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm > 1e-8 and np.isfinite(fallback_norm):
        return (fallback / fallback_norm).astype(np.float32)
    return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


def _perpendicular_unit(vector: np.ndarray) -> np.ndarray:
    axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(vector, axis))) > 0.9:
        axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    perpendicular = np.cross(vector, axis)
    return _unit_or_fallback(perpendicular, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))


__all__ = ["CompositeConstraints", "DistanceConstraints", "SettleWaterConstraints"]
