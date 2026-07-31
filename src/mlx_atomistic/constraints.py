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
        return constrained, self.max_error(constrained, cell)

    def apply_position_step(
        self,
        reference_positions,
        predicted_positions,
        masses,
        cell: Cell | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Apply a SHAKE position step from a constrained reference geometry."""

        reference = as_mx_array(reference_positions)
        constrained = as_mx_array(predicted_positions)
        masses = as_mx_array(masses)
        if self.pairs.shape[0] == 0:
            return constrained, self.max_error(constrained, cell)
        if reference.shape != constrained.shape:
            msg = "reference_positions and predicted_positions must have matching shapes"
            raise ValueError(msg)
        if self._max_pair_index >= constrained.shape[0]:
            msg = "constraint pair index outside positions"
            raise ValueError(msg)

        i = self.pairs[:, 0]
        j = self.pairs[:, 1]
        inverse_masses = 1.0 / masses
        reference_displacement = reference[i] - reference[j]
        if cell is not None:
            reference_displacement = cell.minimum_image(reference_displacement)
        inverse_mass_sum = inverse_masses[i] + inverse_masses[j]
        target_squared = self.distances * self.distances
        for _ in range(self.max_iterations):
            displacement = self._displacements(constrained, cell)
            error_squared = target_squared - mx.sum(displacement * displacement, axis=-1)
            denominator = 2.0 * inverse_mass_sum * mx.sum(
                displacement * reference_displacement,
                axis=-1,
            )
            safe_denominator = mx.where(
                mx.abs(denominator) > 1.0e-20,
                denominator,
                mx.where(denominator < 0.0, -1.0e-20, 1.0e-20),
            )
            multiplier = error_squared / safe_denominator
            correction = multiplier[:, None] * reference_displacement
            constrained = constrained.at[i].add(inverse_masses[i, None] * correction)
            constrained = constrained.at[j].add(-inverse_masses[j, None] * correction)
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
        weight_i = inverse_masses[i] / (inverse_masses[i] + inverse_masses[j])
        weight_j = inverse_masses[j] / (inverse_masses[i] + inverse_masses[j])
        for _ in range(self.max_iterations):
            relative_velocity = constrained[i] - constrained[j]
            relative_along_bond = mx.sum(relative_velocity * unit, axis=-1)
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
        masses = as_mx_array(masses)
        if self.pairs.shape[0] == 0:
            return constrained, self.max_error(constrained, cell)
        if self._max_pair_index >= constrained.shape[0]:
            msg = "SETTLE water atom index outside positions"
            raise ValueError(msg)
        if masses.shape != (constrained.shape[0],):
            msg = "masses must have shape (n_particles,)"
            raise ValueError(msg)

        oxygen = self.waters[:, 0]
        hydrogen_a = self.waters[:, 1]
        hydrogen_b = self.waters[:, 2]
        origins = constrained[oxygen]
        first = constrained[hydrogen_a] - origins
        second = constrained[hydrogen_b] - origins
        if cell is not None:
            first = cell.minimum_image(first)
            second = cell.minimum_image(second)

        bisector = _unit_vectors_mx(first + second, first)
        difference = first - second
        difference = difference - mx.sum(
            difference * bisector,
            axis=-1,
            keepdims=True,
        ) * bisector
        fallback_axis = mx.where(
            (mx.abs(bisector[:, :1]) > 0.9),
            mx.array([[0.0, 1.0, 0.0]], dtype=constrained.dtype),
            mx.array([[1.0, 0.0, 0.0]], dtype=constrained.dtype),
        )
        difference = _unit_vectors_mx(
            difference,
            _cross_vectors_mx(bisector, fallback_axis),
        )

        half_hh = 0.5 * float(self.hh_distance)
        along_bisector = float(
            np.sqrt(float(self.oh_distance) ** 2 - half_hh * half_hh)
        )
        target_a = (
            along_bisector * bisector + half_hh * difference
        )
        target_b = (
            along_bisector * bisector - half_hh * difference
        )
        oxygen_mass = masses[oxygen]
        hydrogen_a_mass = masses[hydrogen_a]
        hydrogen_b_mass = masses[hydrogen_b]
        total_mass = oxygen_mass + hydrogen_a_mass + hydrogen_b_mass
        current_center = origins + (
            hydrogen_a_mass[:, None] * first
            + hydrogen_b_mass[:, None] * second
        ) / total_mass[:, None]
        target_center_offset = (
            hydrogen_a_mass[:, None] * target_a
            + hydrogen_b_mass[:, None] * target_b
        ) / total_mass[:, None]
        projected_oxygen = current_center - target_center_offset
        projected_a = projected_oxygen + target_a
        projected_b = projected_oxygen + target_b
        constrained = (
            constrained.at[oxygen]
            .add(projected_oxygen - constrained[oxygen])
            .at[hydrogen_a]
            .add(projected_a - constrained[hydrogen_a])
            .at[hydrogen_b]
            .add(projected_b - constrained[hydrogen_b])
        )
        return constrained, self.max_error(constrained, cell)

    def apply_position_step(
        self,
        reference_positions,
        predicted_positions,
        masses,
        cell: Cell | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Apply the SETTLE dynamics projection from a constrained reference state."""

        reference = as_mx_array(reference_positions)
        predicted = as_mx_array(predicted_positions)
        masses = as_mx_array(masses)
        if self.pairs.shape[0] == 0:
            return predicted, self.max_error(predicted, cell)
        if reference.shape != predicted.shape:
            msg = "reference_positions and predicted_positions must have matching shapes"
            raise ValueError(msg)
        if self._max_pair_index >= predicted.shape[0]:
            msg = "SETTLE water atom index outside positions"
            raise ValueError(msg)
        if masses.shape != (predicted.shape[0],):
            msg = "masses must have shape (n_particles,)"
            raise ValueError(msg)

        oxygen = self.waters[:, 0]
        hydrogen_a = self.waters[:, 1]
        hydrogen_b = self.waters[:, 2]

        def displacement(first: mx.array, second: mx.array) -> mx.array:
            value = first - second
            return cell.minimum_image(value) if cell is not None else value

        old_b = displacement(reference[hydrogen_a], reference[oxygen])
        old_c = displacement(reference[hydrogen_b], reference[oxygen])
        step_o = displacement(predicted[oxygen], reference[oxygen])
        step_a = displacement(predicted[hydrogen_a], reference[hydrogen_a])
        step_b = displacement(predicted[hydrogen_b], reference[hydrogen_b])

        # Vectorized transcription of OpenMM's ReferenceSETTLEAlgorithm::apply:
        # preserve the predicted center of mass, solve the rigid geometry in a
        # frame defined by the prior constrained water, then rotate it back.
        mass_o = masses[oxygen]
        mass_a = masses[hydrogen_a]
        mass_b = masses[hydrogen_b]
        total_mass = mass_o + mass_a + mass_b
        center = (
            mass_o[:, None] * step_o
            + mass_a[:, None] * (old_b + step_a)
            + mass_b[:, None] * (old_c + step_b)
        ) / total_mass[:, None]
        centered_o = step_o - center
        centered_a = old_b + step_a - center
        centered_b = old_c + step_b - center

        axis_z = _cross_vectors_mx(old_b, old_c)
        axis_z = axis_z / mx.sqrt(
            mx.maximum(mx.sum(axis_z * axis_z, axis=-1), 1.0e-20)
        )[:, None]
        axis_x = _cross_vectors_mx(centered_o, axis_z)
        axis_x = axis_x / mx.sqrt(
            mx.maximum(mx.sum(axis_x * axis_x, axis=-1), 1.0e-20)
        )[:, None]
        axis_y = _cross_vectors_mx(axis_z, axis_x)
        axis_y = axis_y / mx.sqrt(
            mx.maximum(mx.sum(axis_y * axis_y, axis=-1), 1.0e-20)
        )[:, None]

        def component(values: mx.array, axis: mx.array) -> mx.array:
            return mx.sum(values * axis, axis=-1)

        old_b_x = component(old_b, axis_x)
        old_b_y = component(old_b, axis_y)
        old_c_x = component(old_c, axis_x)
        old_c_y = component(old_c, axis_y)
        centered_o_z = component(centered_o, axis_z)
        centered_a_x = component(centered_a, axis_x)
        centered_a_y = component(centered_a, axis_y)
        centered_a_z = component(centered_a, axis_z)
        centered_b_x = component(centered_b, axis_x)
        centered_b_y = component(centered_b, axis_y)
        centered_b_z = component(centered_b, axis_z)

        half_hh = 0.5 * float(self.hh_distance)
        oxygen_to_h_axis = float(
            np.sqrt(float(self.oh_distance) ** 2 - half_hh * half_hh)
        )
        oxygen_radius = oxygen_to_h_axis * (mass_a + mass_b) / total_mass
        hydrogen_radius = oxygen_to_h_axis - oxygen_radius
        sin_phi = mx.clip(centered_o_z / oxygen_radius, -1.0, 1.0)
        cos_phi = mx.sqrt(mx.maximum(1.0 - sin_phi * sin_phi, 0.0))
        sin_psi = mx.clip(
            (centered_a_z - centered_b_z)
            / mx.maximum(2.0 * half_hh * cos_phi, 1.0e-20),
            -1.0,
            1.0,
        )
        cos_psi = mx.sqrt(mx.maximum(1.0 - sin_psi * sin_psi, 0.0))

        oxygen_y = oxygen_radius * cos_phi
        hydrogen_x = -half_hh * cos_psi
        hydrogen_a_y = -hydrogen_radius * cos_phi - half_hh * sin_psi * sin_phi
        hydrogen_b_y = -hydrogen_radius * cos_phi + half_hh * sin_psi * sin_phi
        hydrogen_x_squared = hydrogen_x * hydrogen_x
        current_hh_squared = (
            4.0 * hydrogen_x_squared
            + (hydrogen_a_y - hydrogen_b_y) ** 2
            + (centered_a_z - centered_b_z) ** 2
        )
        delta_x = 2.0 * hydrogen_x + mx.sqrt(
            mx.maximum(
                4.0 * hydrogen_x_squared
                - current_hh_squared
                + float(self.hh_distance) ** 2,
                0.0,
            )
        )
        hydrogen_x = hydrogen_x - 0.5 * delta_x

        alpha = (
            hydrogen_x * (old_b_x - old_c_x)
            + old_b_y * hydrogen_a_y
            + old_c_y * hydrogen_b_y
        )
        beta = (
            hydrogen_x * (old_c_y - old_b_y)
            + old_b_x * hydrogen_a_y
            + old_c_x * hydrogen_b_y
        )
        gamma = (
            old_b_x * centered_a_y
            - centered_a_x * old_b_y
            + old_c_x * centered_b_y
            - centered_b_x * old_c_y
        )
        alpha_beta_squared = alpha * alpha + beta * beta
        sin_theta = (
            alpha * gamma
            - beta
            * mx.sqrt(mx.maximum(alpha_beta_squared - gamma * gamma, 0.0))
        ) / mx.maximum(alpha_beta_squared, 1.0e-20)
        sin_theta = mx.clip(sin_theta, -1.0, 1.0)
        cos_theta = mx.sqrt(mx.maximum(1.0 - sin_theta * sin_theta, 0.0))

        oxygen_x = -oxygen_y * sin_theta
        oxygen_y = oxygen_y * cos_theta
        hydrogen_a_x = hydrogen_x * cos_theta - hydrogen_a_y * sin_theta
        hydrogen_a_y = hydrogen_x * sin_theta + hydrogen_a_y * cos_theta
        hydrogen_b_x = -hydrogen_x * cos_theta - hydrogen_b_y * sin_theta
        hydrogen_b_y = -hydrogen_x * sin_theta + hydrogen_b_y * cos_theta

        def from_frame(x, y, z) -> mx.array:
            return x[:, None] * axis_x + y[:, None] * axis_y + z[:, None] * axis_z

        projected_o = reference[oxygen] + center + from_frame(
            oxygen_x,
            oxygen_y,
            centered_o_z,
        )
        projected_a = reference[oxygen] + center + from_frame(
            hydrogen_a_x,
            hydrogen_a_y,
            centered_a_z,
        )
        projected_b = reference[oxygen] + center + from_frame(
            hydrogen_b_x,
            hydrogen_b_y,
            centered_b_z,
        )
        constrained = (
            predicted.at[oxygen]
            .add(projected_o - predicted[oxygen])
            .at[hydrogen_a]
            .add(projected_a - predicted[hydrogen_a])
            .at[hydrogen_b]
            .add(projected_b - predicted[hydrogen_b])
        )
        return constrained, self.max_error(constrained, cell)

    def apply_velocities(
        self,
        positions,
        velocities,
        masses,
        cell: Cell | None = None,
    ) -> mx.array:
        """Remove constrained relative velocity components for SETTLE pairs."""

        positions = as_mx_array(positions)
        constrained = as_mx_array(velocities)
        masses = as_mx_array(masses)
        if self.pairs.shape[0] == 0:
            return constrained
        if self._max_pair_index >= positions.shape[0]:
            msg = "SETTLE water atom index outside positions"
            raise ValueError(msg)
        if masses.shape != (positions.shape[0],):
            msg = "masses must have shape (n_particles,)"
            raise ValueError(msg)

        oxygen = self.waters[:, 0]
        hydrogen_a = self.waters[:, 1]
        hydrogen_b = self.waters[:, 2]
        q_oh_a = positions[oxygen] - positions[hydrogen_a]
        q_oh_b = positions[oxygen] - positions[hydrogen_b]
        q_hh = positions[hydrogen_a] - positions[hydrogen_b]
        if cell is not None:
            q_oh_a = cell.minimum_image(q_oh_a)
            q_oh_b = cell.minimum_image(q_oh_b)
            q_hh = cell.minimum_image(q_hh)

        inverse_oxygen = 1.0 / masses[oxygen]
        inverse_hydrogen_a = 1.0 / masses[hydrogen_a]
        inverse_hydrogen_b = 1.0 / masses[hydrogen_b]
        dot_oh = mx.sum(q_oh_a * q_oh_b, axis=-1)
        dot_a_hh = mx.sum(q_oh_a * q_hh, axis=-1)
        dot_b_hh = mx.sum(q_oh_b * q_hh, axis=-1)
        matrix = mx.stack(
            [
                mx.stack(
                    [
                        (inverse_oxygen + inverse_hydrogen_a)
                        * mx.sum(q_oh_a * q_oh_a, axis=-1),
                        inverse_oxygen * dot_oh,
                        -inverse_hydrogen_a * dot_a_hh,
                    ],
                    axis=-1,
                ),
                mx.stack(
                    [
                        inverse_oxygen * dot_oh,
                        (inverse_oxygen + inverse_hydrogen_b)
                        * mx.sum(q_oh_b * q_oh_b, axis=-1),
                        inverse_hydrogen_b * dot_b_hh,
                    ],
                    axis=-1,
                ),
                mx.stack(
                    [
                        -inverse_hydrogen_a * dot_a_hh,
                        inverse_hydrogen_b * dot_b_hh,
                        (inverse_hydrogen_a + inverse_hydrogen_b)
                        * mx.sum(q_hh * q_hh, axis=-1),
                    ],
                    axis=-1,
                ),
            ],
            axis=-2,
        )
        rhs = -mx.stack(
            [
                mx.sum(
                    q_oh_a
                    * (constrained[oxygen] - constrained[hydrogen_a]),
                    axis=-1,
                ),
                mx.sum(
                    q_oh_b
                    * (constrained[oxygen] - constrained[hydrogen_b]),
                    axis=-1,
                ),
                mx.sum(
                    q_hh
                    * (
                        constrained[hydrogen_a]
                        - constrained[hydrogen_b]
                    ),
                    axis=-1,
                ),
            ],
            axis=-1,
        )
        row_0 = matrix[:, 0, :]
        row_1 = matrix[:, 1, :]
        row_2 = matrix[:, 2, :]
        cross_12 = _cross_vectors_mx(row_1, row_2)
        determinant = mx.sum(row_0 * cross_12, axis=-1)
        safe_determinant = mx.where(
            mx.abs(determinant) > 1.0e-20,
            determinant,
            1.0,
        )
        multipliers = (
            rhs[:, :1] * cross_12
            + rhs[:, 1:2] * _cross_vectors_mx(row_2, row_0)
            + rhs[:, 2:3] * _cross_vectors_mx(row_0, row_1)
        ) / safe_determinant[:, None]
        lambda_oh_a = multipliers[:, 0]
        lambda_oh_b = multipliers[:, 1]
        lambda_hh = multipliers[:, 2]
        oxygen_correction = inverse_oxygen[:, None] * (
            lambda_oh_a[:, None] * q_oh_a
            + lambda_oh_b[:, None] * q_oh_b
        )
        hydrogen_a_correction = inverse_hydrogen_a[:, None] * (
            -lambda_oh_a[:, None] * q_oh_a
            + lambda_hh[:, None] * q_hh
        )
        hydrogen_b_correction = inverse_hydrogen_b[:, None] * (
            -lambda_oh_b[:, None] * q_oh_b
            - lambda_hh[:, None] * q_hh
        )
        return (
            constrained.at[oxygen]
            .add(oxygen_correction)
            .at[hydrogen_a]
            .add(hydrogen_a_correction)
            .at[hydrogen_b]
            .add(hydrogen_b_correction)
        )

    def _apply_pre_force_velocities(
        self,
        positions,
        velocities,
        masses,
        cell: Cell | None = None,
    ) -> mx.array:
        """Skip the redundant first of two SETTLE velocity projections."""

        del positions, masses, cell
        return as_mx_array(velocities)


@dataclass(frozen=True)
class CompositeConstraints:
    """Apply multiple constraint objects through the standard constraint protocol."""

    constraints: tuple[object, ...]

    def __post_init__(self) -> None:
        if not self.constraints:
            msg = "CompositeConstraints requires at least one constraint object"
            raise ValueError(msg)
        pairs = []
        child_atoms: list[set[int]] = []
        for constraint in self.constraints:
            constraint_pairs = getattr(constraint, "pairs", None)
            if constraint_pairs is None:
                msg = "constraint objects must expose pairs"
                raise ValueError(msg)
            pair_array = np.asarray(
                constraint_pairs,
                dtype=np.int32,
            ).reshape((-1, 2))
            pairs.append(pair_array)
            child_atoms.append(set(int(value) for value in pair_array.reshape(-1)))
        object.__setattr__(
            self,
            "pairs",
            mx.array(np.concatenate(pairs, axis=0), dtype=mx.int32) if pairs else _empty_pairs(),
        )
        object.__setattr__(
            self,
            "_requires_iteration",
            any(
                bool(child_atoms[left] & child_atoms[right])
                for left in range(len(child_atoms))
                for right in range(left + 1, len(child_atoms))
            ),
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
        cycles = 8 if self._requires_iteration else 1
        for _ in range(cycles):
            for constraint in self.constraints:
                constrained, _ = constraint.apply_positions(
                    constrained,
                    masses,
                    cell,
                )
        return constrained, self.max_error(constrained, cell)

    def apply_position_step(
        self,
        reference_positions,
        predicted_positions,
        masses,
        cell: Cell | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Apply dynamics-aware child position constraints in sequence."""

        constrained = as_mx_array(predicted_positions)
        cycles = 8 if self._requires_iteration else 1
        for _ in range(cycles):
            for constraint in self.constraints:
                step_projector = getattr(constraint, "apply_position_step", None)
                if step_projector is None:
                    constrained, _ = constraint.apply_positions(
                        constrained,
                        masses,
                        cell,
                    )
                else:
                    constrained, _ = step_projector(
                        reference_positions,
                        constrained,
                        masses,
                        cell,
                    )
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
        cycles = 8 if self._requires_iteration else 1
        for _ in range(cycles):
            for constraint in self.constraints:
                constrained = constraint.apply_velocities(
                    positions,
                    constrained,
                    masses,
                    cell,
                )
        return constrained

    def _apply_pre_force_velocities(
        self,
        positions,
        velocities,
        masses,
        cell: Cell | None = None,
    ) -> mx.array:
        """Project non-SETTLE children before the final constrained kick."""

        if self._requires_iteration:
            return self.apply_velocities(positions, velocities, masses, cell)
        constrained = as_mx_array(velocities)
        for constraint in self.constraints:
            if isinstance(constraint, SettleWaterConstraints):
                continue
            constrained = constraint.apply_velocities(
                positions,
                constrained,
                masses,
                cell,
            )
        return constrained


def _unit_vectors_mx(values: mx.array, fallback: mx.array) -> mx.array:
    norm = mx.sqrt(mx.maximum(mx.sum(values * values, axis=-1), 1.0e-20))
    fallback_norm = mx.sqrt(
        mx.maximum(mx.sum(fallback * fallback, axis=-1), 1.0e-20)
    )
    normalized = values / norm[:, None]
    normalized_fallback = fallback / fallback_norm[:, None]
    default = mx.broadcast_to(
        mx.array([[1.0, 0.0, 0.0]], dtype=values.dtype),
        values.shape,
    )
    normalized_fallback = mx.where(
        (fallback_norm > 1.0e-8)[:, None],
        normalized_fallback,
        default,
    )
    return mx.where(
        (norm > 1.0e-8)[:, None],
        normalized,
        normalized_fallback,
    )


def _cross_vectors_mx(left: mx.array, right: mx.array) -> mx.array:
    return mx.stack(
        [
            left[:, 1] * right[:, 2] - left[:, 2] * right[:, 1],
            left[:, 2] * right[:, 0] - left[:, 0] * right[:, 2],
            left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0],
        ],
        axis=-1,
    )


__all__ = ["CompositeConstraints", "DistanceConstraints", "SettleWaterConstraints"]
