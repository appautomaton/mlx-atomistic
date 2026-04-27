"""Molecular mechanics force terms."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.topology import Topology


def _parameter_array(value, *, count: int, name: str) -> mx.array:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = np.full((count,), float(array), dtype=np.float32)
    if array.shape != (count,):
        msg = f"{name} must be scalar or have shape ({count},)"
        raise ValueError(msg)
    return as_mx_array(array)


def _zero_energy(positions: mx.array) -> mx.array:
    return mx.sum(positions[:, 0] * 0.0)


def _norm(vector: mx.array) -> mx.array:
    return mx.sqrt(mx.maximum(mx.sum(vector * vector, axis=-1), 1e-12))


def _cross(a: mx.array, b: mx.array) -> mx.array:
    return mx.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


@dataclass(frozen=True)
class HarmonicBondPotential:
    """Harmonic bond stretch potential."""

    bonds: object
    k: object
    length: object
    name: str = "bond"

    def __post_init__(self) -> None:
        bonds = np.asarray(self.bonds, dtype=np.int32)
        if bonds.size == 0:
            bonds = np.empty((0, 2), dtype=np.int32)
        if bonds.ndim != 2 or bonds.shape[1] != 2:
            msg = "bonds must have shape (n, 2)"
            raise ValueError(msg)
        count = bonds.shape[0]
        object.__setattr__(self, "bonds", mx.array(bonds, dtype=mx.int32))
        object.__setattr__(self, "k", _parameter_array(self.k, count=count, name="k"))
        object.__setattr__(
            self,
            "length",
            _parameter_array(self.length, count=count, name="length"),
        )

    def potential_energy(self, positions: mx.array, cell: Cell | None = None) -> mx.array:
        positions = as_mx_array(positions)
        if self.bonds.shape[0] == 0:
            return _zero_energy(positions)
        i = self.bonds[:, 0]
        j = self.bonds[:, 1]
        displacement = positions[i] - positions[j]
        if cell is not None:
            displacement = cell.minimum_image(displacement)
        distance = _norm(displacement)
        return 0.5 * mx.sum(self.k * (distance - self.length) * (distance - self.length))

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        del pairs
        positions = as_mx_array(positions)
        if self.bonds.shape[0] == 0:
            return _zero_energy(positions), mx.zeros_like(positions)
        energy, gradient = mx.value_and_grad(
            lambda pos: self.potential_energy(pos, cell)
        )(positions)
        return energy, -gradient


@dataclass(frozen=True)
class HarmonicAnglePotential:
    """Harmonic angle bend potential."""

    angles: object
    k: object
    angle: object
    name: str = "angle"

    def __post_init__(self) -> None:
        angles = np.asarray(self.angles, dtype=np.int32)
        if angles.size == 0:
            angles = np.empty((0, 3), dtype=np.int32)
        if angles.ndim != 2 or angles.shape[1] != 3:
            msg = "angles must have shape (n, 3)"
            raise ValueError(msg)
        count = angles.shape[0]
        object.__setattr__(self, "angles", mx.array(angles, dtype=mx.int32))
        object.__setattr__(self, "k", _parameter_array(self.k, count=count, name="k"))
        object.__setattr__(self, "angle", _parameter_array(self.angle, count=count, name="angle"))

    def potential_energy(self, positions: mx.array, cell: Cell | None = None) -> mx.array:
        positions = as_mx_array(positions)
        if self.angles.shape[0] == 0:
            return _zero_energy(positions)
        i = self.angles[:, 0]
        j = self.angles[:, 1]
        k = self.angles[:, 2]
        left = positions[i] - positions[j]
        right = positions[k] - positions[j]
        if cell is not None:
            left = cell.minimum_image(left)
            right = cell.minimum_image(right)
        cosine = mx.sum(left * right, axis=-1) / (_norm(left) * _norm(right))
        theta = mx.arccos(mx.clip(cosine, -0.999999, 0.999999))
        return 0.5 * mx.sum(self.k * (theta - self.angle) * (theta - self.angle))

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        del pairs
        positions = as_mx_array(positions)
        if self.angles.shape[0] == 0:
            return _zero_energy(positions), mx.zeros_like(positions)
        energy, gradient = mx.value_and_grad(
            lambda pos: self.potential_energy(pos, cell)
        )(positions)
        return energy, -gradient


@dataclass(frozen=True)
class PeriodicDihedralPotential:
    """Periodic torsion potential k * (1 + cos(n phi - phase))."""

    dihedrals: object
    k: object
    periodicity: object
    phase: object = 0.0
    name: str = "dihedral"

    def __post_init__(self) -> None:
        dihedrals = np.asarray(self.dihedrals, dtype=np.int32)
        if dihedrals.size == 0:
            dihedrals = np.empty((0, 4), dtype=np.int32)
        if dihedrals.ndim != 2 or dihedrals.shape[1] != 4:
            msg = "dihedrals must have shape (n, 4)"
            raise ValueError(msg)
        count = dihedrals.shape[0]
        object.__setattr__(self, "dihedrals", mx.array(dihedrals, dtype=mx.int32))
        object.__setattr__(self, "k", _parameter_array(self.k, count=count, name="k"))
        object.__setattr__(
            self,
            "periodicity",
            _parameter_array(self.periodicity, count=count, name="periodicity"),
        )
        object.__setattr__(self, "phase", _parameter_array(self.phase, count=count, name="phase"))

    def potential_energy(self, positions: mx.array, cell: Cell | None = None) -> mx.array:
        positions = as_mx_array(positions)
        if self.dihedrals.shape[0] == 0:
            return _zero_energy(positions)
        i = self.dihedrals[:, 0]
        j = self.dihedrals[:, 1]
        k = self.dihedrals[:, 2]
        m = self.dihedrals[:, 3]

        b0 = positions[i] - positions[j]
        b1 = positions[k] - positions[j]
        b2 = positions[m] - positions[k]
        if cell is not None:
            b0 = cell.minimum_image(b0)
            b1 = cell.minimum_image(b1)
            b2 = cell.minimum_image(b2)

        b1_unit = b1 / _norm(b1)[:, None]
        v = b0 - mx.sum(b0 * b1_unit, axis=-1)[:, None] * b1_unit
        w = b2 - mx.sum(b2 * b1_unit, axis=-1)[:, None] * b1_unit
        x = mx.sum(v * w, axis=-1)
        y = mx.sum(_cross(b1_unit, v) * w, axis=-1)
        phi = mx.arctan2(y, x)
        return mx.sum(self.k * (1.0 + mx.cos(self.periodicity * phi - self.phase)))

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        del pairs
        positions = as_mx_array(positions)
        if self.dihedrals.shape[0] == 0:
            return _zero_energy(positions), mx.zeros_like(positions)
        energy, gradient = mx.value_and_grad(
            lambda pos: self.potential_energy(pos, cell)
        )(positions)
        return energy, -gradient


@dataclass(frozen=True)
class CoulombPotential:
    """Direct pair Coulomb potential in reduced units."""

    charges: object | None = None
    coulomb_constant: float = 1.0
    cutoff: float | None = None
    shift: bool = False
    topology: Topology | None = None
    one_four_scale: float = 1.0
    name: str = "coulomb"

    def __post_init__(self) -> None:
        if self.topology is not None and self.charges is None:
            if self.topology.partial_charges is None:
                msg = "topology must define partial_charges when charges are not provided"
                raise ValueError(msg)
            charges = self.topology.partial_charges
        elif self.charges is None:
            msg = "charges are required"
            raise ValueError(msg)
        else:
            charges = as_mx_array(self.charges)

        if charges.ndim != 1:
            msg = "charges must have shape (n_atoms,)"
            raise ValueError(msg)
        if self.cutoff is not None and self.cutoff <= 0.0:
            msg = "cutoff must be positive"
            raise ValueError(msg)
        if self.one_four_scale < 0.0:
            msg = "one_four_scale must be non-negative"
            raise ValueError(msg)
        object.__setattr__(self, "charges", charges)

    def _pairs_and_scales(
        self,
        positions: mx.array,
        pairs: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        if self.topology is not None:
            filtered = self.topology.nonbonded_pairs(pairs)
            scales = self.topology.pair_scales(filtered, one_four_scale=self.one_four_scale)
            return filtered, scales

        if pairs is None:
            n_atoms = positions.shape[0]
            pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]
            if not pairs:
                return (
                    mx.array(np.empty((0, 2), dtype=np.int32), dtype=mx.int32),
                    as_mx_array([]),
                )
            return mx.array(pairs, dtype=mx.int32), as_mx_array([1.0] * len(pairs))
        pairs = mx.array(pairs, dtype=mx.int32)
        return pairs, as_mx_array([1.0] * pairs.shape[0])

    def potential_energy(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> mx.array:
        positions = as_mx_array(positions)
        pairs, scales = self._pairs_and_scales(positions, pairs)
        if pairs.shape[0] == 0:
            return _zero_energy(positions)

        i = pairs[:, 0]
        j = pairs[:, 1]
        displacement = positions[i] - positions[j]
        if cell is not None:
            displacement = cell.minimum_image(displacement)
        distance = _norm(displacement)
        mask = distance > 0.0
        if self.cutoff is not None:
            mask = mask & (distance < self.cutoff)
        safe_distance = mx.where(mask, distance, 1.0)
        qij = self.charges[i] * self.charges[j]
        pair_energy = self.coulomb_constant * qij / safe_distance
        if self.shift and self.cutoff is not None:
            pair_energy = pair_energy - self.coulomb_constant * qij / self.cutoff
        pair_energy = mx.where(mask, pair_energy * scales, 0.0)
        return mx.sum(pair_energy)

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        positions = as_mx_array(positions)
        energy, gradient = mx.value_and_grad(
            lambda pos: self.potential_energy(pos, cell=cell, pairs=pairs)
        )(positions)
        return energy, -gradient


@dataclass(frozen=True)
class NonbondedPotential:
    """Combined Lennard-Jones and Coulomb pair potential."""

    sigma: object
    epsilon: object
    charges: object
    coulomb_constant: float = 1.0
    cutoff: float | None = 2.5
    lj_shift: bool = True
    coulomb_shift: bool = False
    topology: Topology | None = None
    lj_one_four_scale: float = 1.0
    coulomb_one_four_scale: float = 1.0
    name: str = "nonbonded"

    def __post_init__(self) -> None:
        sigma = as_mx_array(self.sigma)
        epsilon = as_mx_array(self.epsilon)
        charges = as_mx_array(self.charges)
        if sigma.ndim != 1 or epsilon.ndim != 1 or charges.ndim != 1:
            msg = "sigma, epsilon, and charges must have shape (n_atoms,)"
            raise ValueError(msg)
        if sigma.shape != epsilon.shape or sigma.shape != charges.shape:
            msg = "sigma, epsilon, and charges must have matching shapes"
            raise ValueError(msg)
        if bool(np.any(np.asarray(sigma) <= 0.0)):
            msg = "sigma values must be positive"
            raise ValueError(msg)
        if bool(np.any(np.asarray(epsilon) < 0.0)):
            msg = "epsilon values must be non-negative"
            raise ValueError(msg)
        if self.topology is not None and self.topology.n_atoms != sigma.shape[0]:
            msg = "topology.n_atoms must match nonbonded parameter length"
            raise ValueError(msg)
        if self.cutoff is not None and self.cutoff <= 0.0:
            msg = "cutoff must be positive"
            raise ValueError(msg)
        if self.lj_one_four_scale < 0.0 or self.coulomb_one_four_scale < 0.0:
            msg = "1-4 scaling factors must be non-negative"
            raise ValueError(msg)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "charges", charges)

    def _pairs_and_scales(self, positions: mx.array, pairs) -> tuple[mx.array, mx.array, mx.array]:
        if self.topology is not None:
            filtered = self.topology.nonbonded_pairs(pairs)
            lj_scales = self.topology.pair_scales(
                filtered,
                one_four_scale=self.lj_one_four_scale,
            )
            coulomb_scales = self.topology.pair_scales(
                filtered,
                one_four_scale=self.coulomb_one_four_scale,
            )
            return filtered, lj_scales, coulomb_scales

        if pairs is None:
            n_atoms = positions.shape[0]
            pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]
            if not pairs:
                empty_pairs = mx.array(np.empty((0, 2), dtype=np.int32), dtype=mx.int32)
                return empty_pairs, as_mx_array([]), as_mx_array([])
            pair_array = mx.array(pairs, dtype=mx.int32)
        else:
            pair_array = mx.array(pairs, dtype=mx.int32)
        scales = as_mx_array([1.0] * pair_array.shape[0])
        return pair_array, scales, scales

    def mixed_pair_parameters(self, pairs) -> tuple[mx.array, mx.array]:
        """Return Lorentz-Berthelot mixed sigma and epsilon for pairs."""

        pair_array = mx.array(pairs, dtype=mx.int32)
        if pair_array.shape[0] == 0:
            return as_mx_array([]), as_mx_array([])
        i = pair_array[:, 0]
        j = pair_array[:, 1]
        sigma_ij = 0.5 * (self.sigma[i] + self.sigma[j])
        epsilon_ij = mx.sqrt(self.epsilon[i] * self.epsilon[j])
        return sigma_ij, epsilon_ij

    def _pair_components(
        self,
        positions: mx.array,
        cell: Cell | None,
        pairs,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        pairs, lj_scales, coulomb_scales = self._pairs_and_scales(positions, pairs)
        if pairs.shape[0] == 0:
            zero = _zero_energy(positions)
            return pairs, zero, zero, mx.zeros_like(positions), lj_scales, coulomb_scales

        i = pairs[:, 0]
        j = pairs[:, 1]
        displacement = positions[i] - positions[j]
        if cell is not None:
            displacement = cell.minimum_image(displacement)

        r2 = mx.sum(displacement * displacement, axis=-1)
        pair_mask = r2 > 0.0
        if self.cutoff is not None:
            pair_mask = pair_mask & (r2 < self.cutoff * self.cutoff)
        safe_r2 = mx.where(pair_mask, r2, 1.0)
        distance = mx.sqrt(safe_r2)

        sigma_ij, epsilon_ij = self.mixed_pair_parameters(pairs)
        sigma2_over_r2 = (sigma_ij * sigma_ij) / safe_r2
        inv_r6 = sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2
        inv_r12 = inv_r6 * inv_r6
        lj_pair_energy = 4.0 * epsilon_ij * (inv_r12 - inv_r6)
        if self.lj_shift and self.cutoff is not None:
            sigma2_over_rc2 = (sigma_ij * sigma_ij) / (self.cutoff * self.cutoff)
            inv_rc6 = sigma2_over_rc2 * sigma2_over_rc2 * sigma2_over_rc2
            inv_rc12 = inv_rc6 * inv_rc6
            lj_pair_energy = lj_pair_energy - 4.0 * epsilon_ij * (inv_rc12 - inv_rc6)
        lj_pair_energy = mx.where(pair_mask, lj_pair_energy * lj_scales, 0.0)

        qij = self.charges[i] * self.charges[j]
        coulomb_pair_energy = self.coulomb_constant * qij / distance
        if self.coulomb_shift and self.cutoff is not None:
            coulomb_pair_energy = coulomb_pair_energy - self.coulomb_constant * qij / self.cutoff
        coulomb_pair_energy = mx.where(
            pair_mask,
            coulomb_pair_energy * coulomb_scales,
            0.0,
        )

        lj_scalar = 24.0 * epsilon_ij * (2.0 * inv_r12 - inv_r6) / safe_r2
        coulomb_scalar = self.coulomb_constant * qij / (safe_r2 * distance)
        scalar = mx.where(
            pair_mask,
            lj_scalar * lj_scales + coulomb_scalar * coulomb_scales,
            0.0,
        )
        pair_forces = scalar[:, None] * displacement
        forces = mx.zeros_like(positions).at[i].add(pair_forces).at[j].add(-pair_forces)
        return (
            pairs,
            mx.sum(lj_pair_energy),
            mx.sum(coulomb_pair_energy),
            forces,
            lj_scales,
            coulomb_scales,
        )

    def component_energies(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> dict[str, mx.array]:
        """Return LJ and Coulomb energy components."""

        positions = as_mx_array(positions)
        _, lj_energy, coulomb_energy, _, _, _ = self._pair_components(positions, cell, pairs)
        return {"lj": lj_energy, "coulomb": coulomb_energy}

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Return total nonbonded energy and forces."""

        positions = as_mx_array(positions)
        if positions.ndim != 2 or positions.shape[1] != 3:
            msg = "positions must have shape (n_atoms, 3)"
            raise ValueError(msg)
        _, lj_energy, coulomb_energy, forces, _, _ = self._pair_components(positions, cell, pairs)
        return lj_energy + coulomb_energy, forces
