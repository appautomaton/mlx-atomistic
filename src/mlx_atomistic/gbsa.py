"""GB-OBC implicit solvent with ACE surface-area approximation."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell, as_mx_array

DEFAULT_SOLVENT_DIELECTRIC = 78.5
DEFAULT_SOLUTE_DIELECTRIC = 1.0
DEFAULT_SURFACE_AREA_ENERGY_KJ_MOL_A2 = 0.0225936
DEFAULT_PROBE_RADIUS_A = 1.4
DEFAULT_RADIUS_OFFSET_A = 0.09
DEFAULT_COULOMB_CONSTANT_KJ_MOL_ANGSTROM = 1389.3545764438198


def _zero_energy(positions: mx.array) -> mx.array:
    return mx.sum(positions[:, 0] * 0.0)


def _vector_parameter(value: object, *, count: int, name: str) -> mx.array:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = np.full((count,), float(array), dtype=np.float32)
    if array.shape != (count,):
        msg = f"{name} must be scalar or have shape ({count},)"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} values must be finite"
        raise ValueError(msg)
    return as_mx_array(array)


@dataclass(frozen=True)
class GBSAForcePotential:
    """Generalized-Born OBC implicit solvent plus ACE nonpolar term.

    Positions and radii are in Angstrom. Energies are in kJ/mol.
    The implementation follows the OpenMM CustomGBForce OBC example, with
    pairwise Born integral accumulation and the ACE surface-area term.
    """

    charges: object
    radius: object
    scale: object
    solvent_dielectric: float = DEFAULT_SOLVENT_DIELECTRIC
    solute_dielectric: float = DEFAULT_SOLUTE_DIELECTRIC
    surface_area_energy: float = DEFAULT_SURFACE_AREA_ENERGY_KJ_MOL_A2
    probe_radius: float = DEFAULT_PROBE_RADIUS_A
    radius_offset: float = DEFAULT_RADIUS_OFFSET_A
    coulomb_constant: float = DEFAULT_COULOMB_CONSTANT_KJ_MOL_ANGSTROM
    name: str = "gbsa"
    supports_virial: bool = True

    def __post_init__(self) -> None:
        charges = np.asarray(self.charges, dtype=np.float32)
        if charges.ndim != 1:
            msg = "charges must have shape (n_atoms,)"
            raise ValueError(msg)
        if not np.all(np.isfinite(charges)):
            msg = "charges values must be finite"
            raise ValueError(msg)
        count = charges.shape[0]
        radius = np.asarray(self.radius, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        if radius.shape != (count,):
            msg = f"radius must have shape ({count},)"
            raise ValueError(msg)
        if scale.shape != (count,):
            msg = f"scale must have shape ({count},)"
            raise ValueError(msg)
        if not np.all(np.isfinite(radius)) or np.any(radius <= 0.0):
            msg = "radius values must be finite and positive"
            raise ValueError(msg)
        if not np.all(np.isfinite(scale)) or np.any(scale < 0.0):
            msg = "scale values must be finite and non-negative"
            raise ValueError(msg)
        radius_offset = float(self.radius_offset)
        if not np.isfinite(radius_offset) or radius_offset < 0.0:
            msg = "radius_offset must be finite and non-negative"
            raise ValueError(msg)
        if np.any(radius <= radius_offset):
            msg = "radius values must be larger than radius_offset"
            raise ValueError(msg)
        for name in (
            "solvent_dielectric",
            "solute_dielectric",
            "surface_area_energy",
            "probe_radius",
            "coulomb_constant",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                msg = f"{name} must be finite"
                raise ValueError(msg)
            if name.endswith("dielectric") and value <= 0.0:
                msg = f"{name} must be positive"
                raise ValueError(msg)
            if name in {"probe_radius", "coulomb_constant"} and value < 0.0:
                msg = f"{name} must be non-negative"
                raise ValueError(msg)
        object.__setattr__(self, "charges", as_mx_array(charges))
        object.__setattr__(self, "radius", _vector_parameter(radius, count=count, name="radius"))
        object.__setattr__(self, "scale", _vector_parameter(scale, count=count, name="scale"))

    def _validate_positions(self, positions: object) -> mx.array:
        positions_np = np.asarray(positions, dtype=np.float32)
        if not np.all(np.isfinite(positions_np)):
            msg = "positions values must be finite"
            raise ValueError(msg)
        positions = as_mx_array(positions)
        if positions.ndim != 2 or positions.shape[1] != 3:
            msg = "positions must have shape (n_atoms, 3)"
            raise ValueError(msg)
        if positions.shape[0] != self.charges.shape[0]:
            msg = f"positions must contain {self.charges.shape[0]} atoms"
            raise ValueError(msg)
        return positions

    def _all_pairs(self, count: int) -> mx.array:
        pairs = np.asarray(
            [(i, j) for i in range(count) for j in range(i + 1, count)],
            dtype=np.int32,
        )
        if pairs.size == 0:
            pairs = np.empty((0, 2), dtype=np.int32)
        return mx.array(pairs, dtype=mx.int32)

    def _validated_pairs(self, pairs: object | None, count: int) -> mx.array:
        if pairs is None:
            return self._all_pairs(count)
        array = np.asarray(pairs, dtype=np.int32)
        if array.size == 0:
            array = np.empty((0, 2), dtype=np.int32)
        if array.ndim != 2 or array.shape[1] != 2:
            msg = "pairs must have shape (n, 2)"
            raise ValueError(msg)
        if array.size and (np.any(array < 0) or np.any(array >= count)):
            msg = "pairs contain atom indices outside [0, n_atoms)"
            raise ValueError(msg)
        return mx.array(array, dtype=mx.int32)

    def _born_radii(self, positions: mx.array, cell: Cell | None, pairs: mx.array) -> mx.array:
        offset_radius = self.radius - self.radius_offset
        if pairs.shape[0] == 0:
            psi = mx.zeros_like(offset_radius)
        else:
            i = pairs[:, 0]
            j = pairs[:, 1]
            displacement = positions[i] - positions[j]
            if cell is not None:
                displacement = cell.minimum_image(displacement)
            r = mx.sqrt(mx.maximum(mx.sum(displacement * displacement, axis=-1), 1e-12))
            i_to_j = self._born_integral(r, offset_radius[i], offset_radius[j], self.scale[j])
            j_to_i = self._born_integral(r, offset_radius[j], offset_radius[i], self.scale[i])
            integral = mx.zeros_like(offset_radius).at[i].add(i_to_j).at[j].add(j_to_i)
            psi = integral * offset_radius
        tanh_arg = psi - 0.8 * psi * psi + 4.85 * psi * psi * psi
        return 1.0 / (1.0 / offset_radius - mx.tanh(tanh_arg) / self.radius)

    def _born_integral(
        self,
        r: mx.array,
        offset_radius1: mx.array,
        offset_radius2: mx.array,
        scale2: mx.array,
    ) -> mx.array:
        scaled_radius2 = scale2 * offset_radius2
        u = r + scaled_radius2
        d = mx.abs(r - scaled_radius2)
        lower = mx.maximum(offset_radius1, d)
        c = (
            2.0
            * (1.0 / offset_radius1 - 1.0 / lower)
            * (scaled_radius2 - r - offset_radius1 >= 0.0)
        )
        radius_difference = r - scaled_radius2 * scaled_radius2 / r
        value = 0.5 * (
            1.0 / lower
            - 1.0 / u
            + 0.25 * (1.0 / (u * u) - 1.0 / (lower * lower)) * radius_difference
            + 0.5 * mx.log(lower / u) / r
            + c
        )
        return mx.where(r + scaled_radius2 - offset_radius1 >= 0.0, value, 0.0)

    def ace_surface_area_energy(self, positions: object, cell: Cell | None = None) -> mx.array:
        """Return the ACE nonpolar (surface-area) solvation energy.

        Args:
            positions: Atomic coordinates, shape ``(n_atoms, 3)``.
            cell: Optional periodic cell; when given, distances use the minimum-image
                convention. Defaults to ``None``.

        Returns:
            The nonpolar solvation energy as a scalar array.
        """

        positions = self._validate_positions(positions)
        pairs = self._all_pairs(positions.shape[0])
        born = self._born_radii(positions, cell, pairs)
        ratio = self.radius / born
        return mx.sum(
            4.0
            * np.pi
            * self.surface_area_energy
            * (self.radius + self.probe_radius)
            * (self.radius + self.probe_radius)
            * ratio
            * ratio
            * ratio
            * ratio
            * ratio
            * ratio
        )

    def _potential_energy_from_validated(
        self,
        positions: mx.array,
        cell: Cell | None,
        pairs: mx.array,
    ) -> mx.array:
        if positions.shape[0] == 0:
            return _zero_energy(positions)
        born = self._born_radii(positions, cell, pairs)
        dielectric_factor = (1.0 / self.solute_dielectric) - (1.0 / self.solvent_dielectric)
        self_energy = -0.5 * self.coulomb_constant * dielectric_factor * mx.sum(
            self.charges * self.charges / born
        )
        ratio = self.radius / born
        ratio6 = ratio * ratio * ratio * ratio * ratio * ratio
        surface_energy = mx.sum(
            4.0
            * np.pi
            * self.surface_area_energy
            * (self.radius + self.probe_radius)
            * (self.radius + self.probe_radius)
            * ratio6
        )
        if pairs.shape[0] == 0:
            return self_energy + surface_energy
        i = pairs[:, 0]
        j = pairs[:, 1]
        displacement = positions[i] - positions[j]
        if cell is not None:
            displacement = cell.minimum_image(displacement)
        r2 = mx.maximum(mx.sum(displacement * displacement, axis=-1), 1e-12)
        f = mx.sqrt(r2 + born[i] * born[j] * mx.exp(-r2 / (4.0 * born[i] * born[j])))
        charge_products = self.charges[i] * self.charges[j]
        pair_energy = -self.coulomb_constant * dielectric_factor * charge_products / f
        return self_energy + surface_energy + mx.sum(pair_energy)

    def potential_energy(
        self,
        positions: object,
        cell: Cell | None = None,
        pairs: object | None = None,
    ) -> mx.array:
        """Return the GB/SA implicit-solvent energy (polar + nonpolar terms).

        Args:
            positions: Atomic coordinates, shape ``(n_atoms, 3)``.
            cell: Optional periodic cell; when given, distances use the minimum-image
                convention. Defaults to ``None``.
            pairs: Accepted for interface uniformity and ignored; the term uses its
                stored index list. Defaults to ``None``.

        Returns:
            Total implicit-solvent energy as a scalar array.
        """

        del pairs
        positions = self._validate_positions(positions)
        validated_pairs = self._all_pairs(positions.shape[0])
        return self._potential_energy_from_validated(positions, cell, validated_pairs)

    def energy_forces(
        self,
        positions: object,
        cell: Cell | None = None,
        pairs: object | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Return the GB/SA implicit-solvent energy and per-atom forces.

        Args:
            positions: Atomic coordinates, shape ``(n_atoms, 3)``.
            cell: Optional periodic cell; when given, distances use the minimum-image
                convention. Defaults to ``None``.
            pairs: Accepted for interface uniformity and ignored; the term uses its
                stored index list. Defaults to ``None``.

        Returns:
            An ``(energy, forces)`` tuple: scalar energy and per-atom forces of shape
                ``(n_atoms, 3)``.
        """

        del pairs
        positions = self._validate_positions(positions)
        validated_pairs = self._all_pairs(positions.shape[0])

        def energy_fn(pos: mx.array) -> mx.array:
            return self._potential_energy_from_validated(pos, cell, validated_pairs)

        energy = energy_fn(positions)
        forces = -mx.grad(energy_fn)(positions)
        return energy, forces


__all__ = [
    "DEFAULT_PROBE_RADIUS_A",
    "DEFAULT_RADIUS_OFFSET_A",
    "DEFAULT_SOLUTE_DIELECTRIC",
    "DEFAULT_SOLVENT_DIELECTRIC",
    "DEFAULT_SURFACE_AREA_ENERGY_KJ_MOL_A2",
    "GBSAForcePotential",
]
