"""Molecular mechanics force terms."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic.charmm_terms import (
    CHARMMCMAPPotential as CHARMMCMAPPotential,
)
from mlx_atomistic.charmm_terms import (
    CHARMMForceSwitchNonbondedPotential as CHARMMForceSwitchNonbondedPotential,
)
from mlx_atomistic.charmm_terms import (
    CHARMMNBFIXPairOverridePotential as CHARMMNBFIXPairOverridePotential,
)
from mlx_atomistic.charmm_terms import (
    CHARMMUreyBradleyPotential as CHARMMUreyBradleyPotential,
)
from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.nonbonded import (
    DEFAULT_DENSE_MEMORY_BUDGET_BYTES,
    EwaldReferenceConfig,
    NonbondedBackend,
    NonbondedElectrostatics,
    NonbondedExecutionConfig,
    choose_nonbonded_backend,
    dense_combined_energy_forces,
    estimate_dense_nonbonded_bytes,
    ewald_reference_coulomb_energy_forces,
)
from mlx_atomistic.pme import PMEConfig, pme_coulomb_energy_forces
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


def _unit_pair_scale() -> mx.array:
    return mx.array(1.0, dtype=mx.float32)


def _topology_pair_scales(
    topology: Topology,
    pairs: mx.array,
    *,
    one_four_scale: float,
) -> mx.array:
    if float(one_four_scale) == 1.0:
        return _unit_pair_scale()
    return topology.pair_scales(pairs, one_four_scale=one_four_scale)


def _topology_nonbonded_pair_scales(
    topology: Topology,
    *,
    one_four_scale: float,
) -> mx.array:
    if float(one_four_scale) == 1.0:
        return _unit_pair_scale()
    return topology.nonbonded_pair_scales(one_four_scale=one_four_scale)


def _norm(vector: mx.array) -> mx.array:
    return mx.sqrt(mx.maximum(mx.sum(vector * vector, axis=-1), 1e-12))


def _norm2(vector: mx.array) -> mx.array:
    return mx.maximum(mx.sum(vector * vector, axis=-1), 1e-12)


def _encoded_pairs(pairs: set[tuple[int, int]], n_atoms: int) -> np.ndarray:
    if not pairs:
        return np.empty((0,), dtype=np.int64)
    array = np.asarray(tuple(pairs), dtype=np.int64)
    left = np.minimum(array[:, 0], array[:, 1])
    right = np.maximum(array[:, 0], array[:, 1])
    return np.sort(left * np.int64(n_atoms) + right)


def _isin_sorted_codes(codes: np.ndarray, sorted_codes: np.ndarray) -> np.ndarray:
    if codes.size == 0 or sorted_codes.size == 0:
        return np.zeros(codes.shape, dtype=bool)
    indices = np.searchsorted(sorted_codes, codes)
    matched = indices < sorted_codes.size
    result = np.zeros(codes.shape, dtype=bool)
    result[matched] = sorted_codes[indices[matched]] == codes[matched]
    return result


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
    supports_virial: bool = True

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
        i = self.bonds[:, 0]
        j = self.bonds[:, 1]
        displacement = positions[i] - positions[j]
        if cell is not None:
            displacement = cell.minimum_image(displacement)
        distance = _norm(displacement)
        delta = distance - self.length
        energy = 0.5 * mx.sum(self.k * delta * delta)
        scalar = -self.k * delta / distance
        pair_forces = scalar[:, None] * displacement
        forces = mx.zeros_like(positions).at[i].add(pair_forces).at[j].add(-pair_forces)
        return energy, forces


@dataclass(frozen=True)
class HarmonicAnglePotential:
    """Harmonic angle bend potential."""

    angles: object
    k: object
    angle: object
    name: str = "angle"
    supports_virial: bool = True

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
        i = self.angles[:, 0]
        j = self.angles[:, 1]
        k = self.angles[:, 2]
        left = positions[i] - positions[j]
        right = positions[k] - positions[j]
        if cell is not None:
            left = cell.minimum_image(left)
            right = cell.minimum_image(right)

        left_norm = _norm(left)
        right_norm = _norm(right)
        cosine = mx.sum(left * right, axis=-1) / (left_norm * right_norm)
        cosine = mx.clip(cosine, -0.999999, 0.999999)
        theta = mx.arccos(cosine)
        delta = theta - self.angle
        energy = 0.5 * mx.sum(self.k * delta * delta)

        sin_theta = mx.sqrt(mx.maximum(1.0 - cosine * cosine, 1e-12))
        prefactor = self.k * delta / sin_theta
        left_force = prefactor[:, None] * (
            right / (left_norm * right_norm)[:, None]
            - cosine[:, None] * left / (left_norm * left_norm)[:, None]
        )
        right_force = prefactor[:, None] * (
            left / (left_norm * right_norm)[:, None]
            - cosine[:, None] * right / (right_norm * right_norm)[:, None]
        )
        center_force = -(left_force + right_force)
        forces = (
            mx.zeros_like(positions)
            .at[i]
            .add(left_force)
            .at[j]
            .add(center_force)
            .at[k]
            .add(right_force)
        )
        return energy, forces


@dataclass(frozen=True)
class PositionalRestraintPotential:
    """Harmonic positional restraint for selected atoms."""

    reference_positions: object
    mask: object
    k: float
    name: str = "positional_restraint"
    supports_virial: bool = True

    def __post_init__(self) -> None:
        reference = as_mx_array(self.reference_positions)
        mask = np.asarray(self.mask, dtype=bool)
        if reference.ndim != 2 or reference.shape[1] != 3:
            msg = "reference_positions must have shape (n_atoms, 3)"
            raise ValueError(msg)
        if mask.shape != (reference.shape[0],):
            msg = "mask must have shape (n_atoms,)"
            raise ValueError(msg)
        if self.k < 0.0:
            msg = "restraint k must be non-negative"
            raise ValueError(msg)
        object.__setattr__(self, "reference_positions", reference)
        object.__setattr__(self, "mask", as_mx_array(mask.astype(np.float32)))

    def potential_energy(self, positions: mx.array, cell: Cell | None = None) -> mx.array:
        del cell
        positions = as_mx_array(positions)
        displacement = positions - self.reference_positions
        squared = mx.sum(displacement * displacement, axis=-1)
        return 0.5 * self.k * mx.sum(squared * self.mask)

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        del pairs
        positions = as_mx_array(positions)
        energy = self.potential_energy(positions, cell)
        forces = -self.k * self.mask[:, None] * (positions - self.reference_positions)
        return energy, forces


@dataclass(frozen=True)
class PeriodicDihedralPotential:
    """Periodic torsion potential with the package dihedral-angle convention."""

    dihedrals: object
    k: object
    periodicity: object
    phase: object = 0.0
    name: str = "dihedral"
    supports_virial: bool = True

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
        phi, _, _, _ = self._openmm_dihedral_components(positions, cell)
        return mx.sum(self.k * (1.0 + mx.cos(self.periodicity * phi + self.phase)))

    def _openmm_dihedral_components(
        self,
        positions: mx.array,
        cell: Cell | None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        i = self.dihedrals[:, 0]
        j = self.dihedrals[:, 1]
        k = self.dihedrals[:, 2]
        m = self.dihedrals[:, 3]

        delta_ab = positions[j] - positions[i]
        delta_bc = positions[j] - positions[k]
        delta_cd = positions[m] - positions[k]
        if cell is not None:
            delta_ab = cell.minimum_image(delta_ab)
            delta_bc = cell.minimum_image(delta_bc)
            delta_cd = cell.minimum_image(delta_cd)

        cross_ab_bc = _cross(delta_ab, delta_bc)
        cross_bc_cd = _cross(delta_bc, delta_cd)
        cosine = mx.sum(cross_ab_bc * cross_bc_cd, axis=-1) / (
            _norm(cross_ab_bc) * _norm(cross_bc_cd)
        )
        cosine = mx.clip(cosine, -0.999999, 0.999999)
        angle = mx.arccos(cosine)
        sign = mx.where(mx.sum(delta_ab * cross_bc_cd, axis=-1) < 0.0, -1.0, 1.0)
        return angle * sign, delta_ab, delta_bc, delta_cd

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
        i = self.dihedrals[:, 0]
        j = self.dihedrals[:, 1]
        k = self.dihedrals[:, 2]
        m = self.dihedrals[:, 3]

        phi, delta_ab, delta_bc, delta_cd = self._openmm_dihedral_components(positions, cell)
        delta_angle = self.periodicity * phi + self.phase
        energy = mx.sum(self.k * (1.0 + mx.cos(delta_angle)))
        d_energy_d_phi = self.k * self.periodicity * mx.sin(delta_angle)

        cross_ab_bc = _cross(delta_ab, delta_bc)
        cross_bc_cd = _cross(delta_bc, delta_cd)
        norm_cross_1 = _norm2(cross_ab_bc)
        norm_cross_2 = _norm2(cross_bc_cd)
        norm_bc = _norm(delta_bc)
        norm_bc2 = _norm2(delta_bc)

        factor_i = (-d_energy_d_phi * norm_bc / norm_cross_1)[:, None]
        factor_m = (d_energy_d_phi * norm_bc / norm_cross_2)[:, None]
        force_i = factor_i * cross_ab_bc
        force_m = factor_m * cross_bc_cd

        factor_j = (mx.sum(delta_ab * delta_bc, axis=-1) / norm_bc2)[:, None]
        factor_k = (mx.sum(delta_cd * delta_bc, axis=-1) / norm_bc2)[:, None]
        shared = factor_j * force_i - factor_k * force_m
        force_j = -(force_i - shared)
        force_k = -(force_m + shared)

        forces = (
            mx.zeros_like(positions)
            .at[i]
            .add(force_i)
            .at[j]
            .add(force_j)
            .at[k]
            .add(force_k)
            .at[m]
            .add(force_m)
        )
        return energy, forces


@dataclass(frozen=True)
class ImproperDihedralPotential(PeriodicDihedralPotential):
    """Periodic improper torsion potential using the same functional form."""

    name: str = "improper"


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
    supports_virial: bool = True

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
            if pairs is None and self.topology.nonbonded_pair_policy == "lazy":
                msg = (
                    "lazy topology requires a runtime nonbonded pair provider; "
                    "full dense pair materialization was not requested"
                )
                raise ValueError(msg)
            filtered = self.topology.nonbonded_pairs(pairs)
            scales = _topology_pair_scales(
                self.topology,
                filtered,
                one_four_scale=self.one_four_scale,
            )
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
        pairs, scales = self._pairs_and_scales(positions, pairs)
        if pairs.shape[0] == 0:
            return _zero_energy(positions), mx.zeros_like(positions)

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
        qij = self.charges[i] * self.charges[j]
        pair_energy = self.coulomb_constant * qij / distance
        if self.shift and self.cutoff is not None:
            pair_energy = pair_energy - self.coulomb_constant * qij / self.cutoff
        pair_energy = mx.where(pair_mask, pair_energy * scales, 0.0)

        scalar = self.coulomb_constant * qij / (safe_r2 * distance)
        scalar = mx.where(pair_mask, scalar * scales, 0.0)
        pair_forces = scalar[:, None] * displacement
        forces = mx.zeros_like(positions).at[i].add(pair_forces).at[j].add(-pair_forces)
        return mx.sum(pair_energy), forces


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
    electrostatics: NonbondedElectrostatics = "cutoff"
    switch_distance: float | None = None
    topology: Topology | None = None
    lj_one_four_scale: float = 1.0
    coulomb_one_four_scale: float = 1.0
    exception_pairs: object = ()
    exception_charge_products: object | None = None
    exception_sigma: object | None = None
    exception_epsilon: object | None = None
    atom_types: object | None = None
    nbfix_pairs: object = ()
    nbfix_sigma: object | None = None
    nbfix_epsilon: object | None = None
    nbfix_type_pairs: object = ()
    nbfix_type_sigma: object | None = None
    nbfix_type_epsilon: object | None = None
    backend: NonbondedBackend = "auto"
    ewald_config: EwaldReferenceConfig | None = None
    pme_config: PMEConfig | None = None
    tile_size: int = 512
    memory_budget_bytes: int | None = DEFAULT_DENSE_MEMORY_BUDGET_BYTES
    name: str = "nonbonded"
    supports_virial: bool = True

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
        if not np.isfinite(float(self.coulomb_constant)):
            msg = "coulomb_constant must be finite"
            raise ValueError(msg)
        if self.topology is not None and self.topology.n_atoms != sigma.shape[0]:
            msg = "topology.n_atoms must match nonbonded parameter length"
            raise ValueError(msg)
        if self.cutoff is not None and self.cutoff <= 0.0:
            msg = "cutoff must be positive"
            raise ValueError(msg)
        if self.switch_distance is not None:
            if self.cutoff is None:
                msg = "switch_distance requires a cutoff"
                raise ValueError(msg)
            if self.switch_distance < 0.0 or self.switch_distance >= self.cutoff:
                msg = "switch_distance must be non-negative and smaller than cutoff"
                raise ValueError(msg)
        if self.lj_one_four_scale < 0.0 or self.coulomb_one_four_scale < 0.0:
            msg = "1-4 scaling factors must be non-negative"
            raise ValueError(msg)
        exception_pairs = np.asarray(self.exception_pairs, dtype=np.int32)
        if exception_pairs.size == 0:
            exception_pairs = np.empty((0, 2), dtype=np.int32)
        if exception_pairs.ndim != 2 or exception_pairs.shape[1] != 2:
            msg = "exception_pairs must have shape (n, 2)"
            raise ValueError(msg)
        if exception_pairs.size and (
            np.any(exception_pairs < 0) or np.any(exception_pairs >= sigma.shape[0])
        ):
            msg = "exception_pairs contain atom indices outside [0, n_atoms)"
            raise ValueError(msg)
        exception_count = exception_pairs.shape[0]
        if self.exception_charge_products is None:
            charge_products = np.asarray([], dtype=np.float32)
        else:
            charge_products = np.asarray(self.exception_charge_products, dtype=np.float32)
        if self.exception_sigma is None:
            exception_sigma = np.asarray([], dtype=np.float32)
        else:
            exception_sigma = np.asarray(self.exception_sigma, dtype=np.float32)
        if self.exception_epsilon is None:
            exception_epsilon = np.asarray([], dtype=np.float32)
        else:
            exception_epsilon = np.asarray(self.exception_epsilon, dtype=np.float32)
        for name, values in [
            ("exception_charge_products", charge_products),
            ("exception_sigma", exception_sigma),
            ("exception_epsilon", exception_epsilon),
        ]:
            if exception_count == 0 and values.size == 0:
                values.resize((0,), refcheck=False)
            if values.shape != (exception_count,):
                msg = f"{name} must have shape ({exception_count},)"
                raise ValueError(msg)
        if np.any(exception_sigma < 0.0) or np.any(exception_epsilon < 0.0):
            msg = "exception sigma and epsilon values must be non-negative"
            raise ValueError(msg)
        nbfix_pairs = np.asarray(self.nbfix_pairs, dtype=np.int32)
        if nbfix_pairs.size == 0:
            nbfix_pairs = np.empty((0, 2), dtype=np.int32)
        if nbfix_pairs.ndim != 2 or nbfix_pairs.shape[1] != 2:
            msg = "nbfix_pairs must have shape (n, 2)"
            raise ValueError(msg)
        if nbfix_pairs.size and (
            np.any(nbfix_pairs < 0) or np.any(nbfix_pairs >= sigma.shape[0])
        ):
            msg = "nbfix_pairs contain atom indices outside [0, n_atoms)"
            raise ValueError(msg)
        nbfix_pair_set: set[tuple[int, int]] = set()
        for left, right in nbfix_pairs.tolist():
            if left == right:
                msg = "nbfix_pairs must not contain self pairs"
                raise ValueError(msg)
            pair = (min(int(left), int(right)), max(int(left), int(right)))
            if pair in nbfix_pair_set:
                msg = "nbfix_pairs must not contain duplicate pairs"
                raise ValueError(msg)
            nbfix_pair_set.add(pair)
        nbfix_pair_count = nbfix_pairs.shape[0]
        nbfix_sigma = (
            np.asarray([], dtype=np.float32)
            if self.nbfix_sigma is None
            else np.asarray(self.nbfix_sigma, dtype=np.float32)
        )
        nbfix_epsilon = (
            np.asarray([], dtype=np.float32)
            if self.nbfix_epsilon is None
            else np.asarray(self.nbfix_epsilon, dtype=np.float32)
        )
        for name, values in [
            ("nbfix_sigma", nbfix_sigma),
            ("nbfix_epsilon", nbfix_epsilon),
        ]:
            if nbfix_pair_count == 0 and values.size == 0:
                values.resize((0,), refcheck=False)
            if values.shape != (nbfix_pair_count,):
                msg = f"{name} must have shape ({nbfix_pair_count},)"
                raise ValueError(msg)
        if np.any(~np.isfinite(nbfix_sigma)) or np.any(nbfix_sigma <= 0.0):
            msg = "nbfix_sigma values must be finite and positive"
            raise ValueError(msg)
        if np.any(~np.isfinite(nbfix_epsilon)) or np.any(nbfix_epsilon < 0.0):
            msg = "nbfix_epsilon values must be finite and non-negative"
            raise ValueError(msg)

        nbfix_type_pairs = np.asarray(self.nbfix_type_pairs, dtype=str)
        if nbfix_type_pairs.size == 0:
            nbfix_type_pairs = np.empty((0, 2), dtype=str)
        if nbfix_type_pairs.ndim != 2 or nbfix_type_pairs.shape[1] != 2:
            msg = "nbfix_type_pairs must have shape (n, 2)"
            raise ValueError(msg)
        nbfix_type_count = nbfix_type_pairs.shape[0]
        nbfix_type_sigma = (
            np.asarray([], dtype=np.float32)
            if self.nbfix_type_sigma is None
            else np.asarray(self.nbfix_type_sigma, dtype=np.float32)
        )
        nbfix_type_epsilon = (
            np.asarray([], dtype=np.float32)
            if self.nbfix_type_epsilon is None
            else np.asarray(self.nbfix_type_epsilon, dtype=np.float32)
        )
        for name, values in [
            ("nbfix_type_sigma", nbfix_type_sigma),
            ("nbfix_type_epsilon", nbfix_type_epsilon),
        ]:
            if nbfix_type_count == 0 and values.size == 0:
                values.resize((0,), refcheck=False)
            if values.shape != (nbfix_type_count,):
                msg = f"{name} must have shape ({nbfix_type_count},)"
                raise ValueError(msg)
        if np.any(~np.isfinite(nbfix_type_sigma)) or np.any(nbfix_type_sigma <= 0.0):
            msg = "nbfix_type_sigma values must be finite and positive"
            raise ValueError(msg)
        if np.any(~np.isfinite(nbfix_type_epsilon)) or np.any(nbfix_type_epsilon < 0.0):
            msg = "nbfix_type_epsilon values must be finite and non-negative"
            raise ValueError(msg)
        if nbfix_type_count > 0 and np.any(np.char.str_len(nbfix_type_pairs) == 0):
            msg = "nbfix_type_pairs must not contain empty type names"
            raise ValueError(msg)
        seen_type_pairs: set[tuple[str, str]] = set()
        for left, right in nbfix_type_pairs.tolist():
            pair = tuple(sorted((str(left), str(right))))
            if pair in seen_type_pairs:
                msg = "nbfix_type_pairs must not contain duplicate type pairs"
                raise ValueError(msg)
            seen_type_pairs.add(pair)
        atom_type_ids = np.empty((0,), dtype=np.int32)
        nbfix_type_pair_ids = np.empty((0, 2), dtype=np.int32)
        if nbfix_type_count > 0:
            if self.atom_types is None:
                msg = "atom_types are required when nbfix_type_pairs are provided"
                raise ValueError(msg)
            atom_types = np.asarray(self.atom_types, dtype=str)
            if atom_types.shape != (sigma.shape[0],):
                msg = "atom_types must have shape (n_atoms,)"
                raise ValueError(msg)
            type_to_id = {
                atom_type: index for index, atom_type in enumerate(sorted(set(atom_types)))
            }
            missing_type_names = sorted(
                {
                    str(atom_type)
                    for pair in nbfix_type_pairs.tolist()
                    for atom_type in pair
                    if str(atom_type) not in type_to_id
                }
            )
            if missing_type_names:
                msg = (
                    "nbfix_type_pairs reference atom types absent from atom_types: "
                    + ", ".join(missing_type_names)
                )
                raise ValueError(msg)
            atom_type_ids = np.asarray(
                [type_to_id[atom_type] for atom_type in atom_types],
                dtype=np.int32,
            )
            nbfix_type_pair_ids = np.asarray(
                [
                    [type_to_id[str(left)], type_to_id[str(right)]]
                    for left, right in nbfix_type_pairs.tolist()
                ],
                dtype=np.int32,
            )
        exception_pair_set = {
            (min(int(i), int(j)), max(int(i), int(j))) for i, j in exception_pairs.tolist()
        }
        exceptions_excluded_by_topology = (
            self.topology is not None and exception_pair_set.issubset(self.topology.exclusion_set)
        )
        config = NonbondedExecutionConfig(
            backend=self.backend,
            electrostatics=self.electrostatics,
            tile_size=self.tile_size,
            memory_budget_bytes=self.memory_budget_bytes,
        )
        if config.electrostatics == "pme":
            if self.pme_config is None:
                msg = "PME electrostatics requires pme_config"
                raise ValueError(msg)
            if not np.isfinite(float(self.pme_config.alpha)) or self.pme_config.alpha <= 0.0:
                msg = "PME electrostatics requires finite positive pme_config.alpha"
                raise ValueError(msg)
            if self.pme_config.real_cutoff is not None and (
                not np.isfinite(float(self.pme_config.real_cutoff))
                or self.pme_config.real_cutoff <= 0.0
            ):
                msg = (
                    "PME electrostatics requires finite positive "
                    "pme_config.real_cutoff when provided"
                )
                raise ValueError(msg)
            if (
                not np.isfinite(float(self.pme_config.charge_tolerance))
                or self.pme_config.charge_tolerance < 0.0
            ):
                msg = "PME electrostatics requires finite non-negative pme_config.charge_tolerance"
                raise ValueError(msg)
            net_charge = float(np.sum(np.asarray(charges, dtype=np.float64), dtype=np.float64))
            if abs(net_charge) > self.pme_config.charge_tolerance:
                msg = (
                    "PME electrostatics requires a neutral system; non-neutral "
                    f"background policy is not implemented: net_charge={net_charge:g}"
                )
                raise ValueError(msg)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "charges", charges)
        object.__setattr__(self, "exception_pairs", mx.array(exception_pairs, dtype=mx.int32))
        object.__setattr__(self, "exception_charge_products", as_mx_array(charge_products))
        object.__setattr__(self, "exception_sigma", as_mx_array(exception_sigma))
        object.__setattr__(self, "exception_epsilon", as_mx_array(exception_epsilon))
        object.__setattr__(self, "nbfix_pairs", mx.array(nbfix_pairs, dtype=mx.int32))
        object.__setattr__(self, "nbfix_sigma", as_mx_array(nbfix_sigma))
        object.__setattr__(self, "nbfix_epsilon", as_mx_array(nbfix_epsilon))
        object.__setattr__(self, "nbfix_type_pairs", nbfix_type_pairs)
        object.__setattr__(self, "nbfix_type_sigma", as_mx_array(nbfix_type_sigma))
        object.__setattr__(self, "nbfix_type_epsilon", as_mx_array(nbfix_type_epsilon))
        object.__setattr__(self, "_atom_type_ids", mx.array(atom_type_ids, dtype=mx.int32))
        object.__setattr__(
            self,
            "_nbfix_type_pair_ids",
            mx.array(nbfix_type_pair_ids, dtype=mx.int32),
        )
        object.__setattr__(self, "_exception_pair_set", frozenset(exception_pair_set))
        object.__setattr__(
            self,
            "_exception_pair_codes",
            _encoded_pairs(exception_pair_set, sigma.shape[0]),
        )
        object.__setattr__(
            self,
            "_exceptions_excluded_by_topology",
            exceptions_excluded_by_topology,
        )
        object.__setattr__(self, "backend", config.backend)
        object.__setattr__(self, "electrostatics", config.electrostatics)
        object.__setattr__(self, "tile_size", config.tile_size)
        object.__setattr__(self, "memory_budget_bytes", config.memory_budget_bytes)
        object.__setattr__(self, "_pair_scale_cache", None)

    @property
    def has_exceptions(self) -> bool:
        """Whether explicit nonbonded pair overrides are active."""

        return int(self.exception_pairs.shape[0]) > 0

    @property
    def has_nbfix(self) -> bool:
        """Whether NBFIX LJ overrides are active."""

        return int(self.nbfix_pairs.shape[0]) > 0 or int(self.nbfix_type_pairs.shape[0]) > 0

    def _pairs_and_scales(self, positions: mx.array, pairs) -> tuple[mx.array, mx.array, mx.array]:
        if self.topology is not None:
            if pairs is None and self.topology.nonbonded_pair_policy == "lazy":
                msg = (
                    "lazy topology requires a runtime nonbonded pair provider; "
                    "full dense pair materialization was not requested"
                )
                raise ValueError(msg)
            if pairs is not None:
                cache_key = (id(pairs), self.lj_one_four_scale, self.coulomb_one_four_scale)
                cache = self._pair_scale_cache
                if cache is not None and cache[0] == cache_key:
                    return cache[1]
            filtered = self.topology.nonbonded_pairs(pairs)
            if not self._exceptions_excluded_by_topology:
                filtered = self._remove_exception_pairs(filtered)
            if pairs is None and self._exceptions_excluded_by_topology:
                lj_scales = _topology_nonbonded_pair_scales(
                    self.topology,
                    one_four_scale=self.lj_one_four_scale,
                )
                coulomb_scales = _topology_nonbonded_pair_scales(
                    self.topology,
                    one_four_scale=self.coulomb_one_four_scale,
                )
            else:
                lj_scales = _topology_pair_scales(
                    self.topology,
                    filtered,
                    one_four_scale=self.lj_one_four_scale,
                )
                coulomb_scales = _topology_pair_scales(
                    self.topology,
                    filtered,
                    one_four_scale=self.coulomb_one_four_scale,
                )
            if pairs is not None:
                object.__setattr__(
                    self,
                    "_pair_scale_cache",
                    (cache_key, (filtered, lj_scales, coulomb_scales)),
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
        pair_array = self._remove_exception_pairs(pair_array)
        scales = as_mx_array([1.0] * pair_array.shape[0])
        return pair_array, scales, scales

    def _remove_exception_pairs(self, pairs: mx.array) -> mx.array:
        if not self.has_exceptions or pairs.shape[0] == 0:
            return pairs
        pair_array = np.asarray(pairs, dtype=np.int32)
        left = np.minimum(pair_array[:, 0], pair_array[:, 1])
        right = np.maximum(pair_array[:, 0], pair_array[:, 1])
        normalized = np.stack((left, right), axis=1).astype(np.int32, copy=False)
        codes = left.astype(np.int64) * np.int64(self.sigma.shape[0]) + right.astype(np.int64)
        keep = ~_isin_sorted_codes(codes, self._exception_pair_codes)
        if not np.any(keep):
            return mx.array(np.empty((0, 2), dtype=np.int32), dtype=mx.int32)
        return mx.array(normalized[keep], dtype=mx.int32)

    def _switch(self, distance: mx.array) -> tuple[mx.array, mx.array]:
        if self.switch_distance is None or self.cutoff is None:
            return mx.ones_like(distance), mx.zeros_like(distance)
        width = self.cutoff - self.switch_distance
        x = mx.clip((distance - self.switch_distance) / width, 0.0, 1.0)
        smooth = 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5
        switch = 1.0 - smooth
        derivative = -(30.0 * x**2 - 60.0 * x**3 + 30.0 * x**4) / width
        derivative = mx.where(distance <= self.switch_distance, 0.0, derivative)
        derivative = mx.where(distance >= self.cutoff, 0.0, derivative)
        return switch, derivative

    def mixed_pair_parameters(self, pairs) -> tuple[mx.array, mx.array]:
        """Return mixed sigma and epsilon, with NBFIX LJ overrides substituted."""

        pair_array = mx.array(pairs, dtype=mx.int32)
        if pair_array.shape[0] == 0:
            return as_mx_array([]), as_mx_array([])
        i = pair_array[:, 0]
        j = pair_array[:, 1]
        sigma_ij = 0.5 * (self.sigma[i] + self.sigma[j])
        epsilon_ij = mx.sqrt(self.epsilon[i] * self.epsilon[j])
        if not self.has_nbfix:
            return sigma_ij, epsilon_ij
        if self.nbfix_type_pairs.shape[0] > 0:
            type_i = self._atom_type_ids[i]
            type_j = self._atom_type_ids[j]
            for index in range(int(self._nbfix_type_pair_ids.shape[0])):
                left = self._nbfix_type_pair_ids[index, 0]
                right = self._nbfix_type_pair_ids[index, 1]
                known_types = (left >= 0) & (right >= 0)
                matched = known_types & (
                    ((type_i == left) & (type_j == right))
                    | ((type_i == right) & (type_j == left))
                )
                sigma_ij = mx.where(matched, self.nbfix_type_sigma[index], sigma_ij)
                epsilon_ij = mx.where(matched, self.nbfix_type_epsilon[index], epsilon_ij)
        if self.nbfix_pairs.shape[0] > 0:
            for index in range(int(self.nbfix_pairs.shape[0])):
                left = self.nbfix_pairs[index, 0]
                right = self.nbfix_pairs[index, 1]
                matched = ((i == left) & (j == right)) | ((i == right) & (j == left))
                sigma_ij = mx.where(matched, self.nbfix_sigma[index], sigma_ij)
                epsilon_ij = mx.where(matched, self.nbfix_epsilon[index], epsilon_ij)
        return sigma_ij, epsilon_ij

    def _pair_components(
        self,
        positions: mx.array,
        cell: Cell | None,
        pairs,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        if (
            self.topology is not None
            and pairs is None
            and self.topology.nonbonded_pair_policy == "lazy"
        ):
            msg = (
                "lazy topology requires a runtime nonbonded pair provider; "
                "full dense pair materialization was not requested"
            )
            raise ValueError(msg)

        estimated_bytes = estimate_dense_nonbonded_bytes(
            positions.shape[0],
            components="combined",
        )
        concrete_backend = choose_nonbonded_backend(
            requested=self.backend,
            n_atoms=positions.shape[0],
            pairs_provided=pairs is not None,
            estimated_dense_bytes=estimated_bytes,
            memory_budget_bytes=self.memory_budget_bytes,
        )
        if (
            self.switch_distance is not None
            or self.has_exceptions
            and not self._exceptions_excluded_by_topology
            or self.has_nbfix
        ):
            concrete_backend = "mlx_pairs"
        if concrete_backend in {"mlx_dense", "mlx_tiled"}:
            lj_energy, coulomb_energy, forces = dense_combined_energy_forces(
                positions,
                sigma=self.sigma,
                epsilon=self.epsilon,
                charges=self.charges,
                coulomb_constant=self.coulomb_constant,
                cutoff=self.cutoff,
                lj_shift=self.lj_shift,
                coulomb_shift=self.coulomb_shift,
                cell=cell,
                topology=self.topology,
                lj_one_four_scale=self.lj_one_four_scale,
                coulomb_one_four_scale=self.coulomb_one_four_scale,
                tile_size=self.tile_size if concrete_backend == "mlx_tiled" else None,
            )
            empty_pairs = mx.array(np.empty((0, 2), dtype=np.int32), dtype=mx.int32)
            empty_scales = as_mx_array([])
            return empty_pairs, lj_energy, coulomb_energy, forces, empty_scales, empty_scales

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
        switch, switch_derivative = self._switch(distance)
        unswitched_lj_pair_energy = lj_pair_energy
        lj_pair_energy = lj_pair_energy * switch
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

        lj_scalar = (
            24.0 * epsilon_ij * (2.0 * inv_r12 - inv_r6) / safe_r2 * switch
            - unswitched_lj_pair_energy * switch_derivative / distance
        )
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

    def _regular_lj_components(
        self,
        positions: mx.array,
        cell: Cell | None,
    ) -> tuple[mx.array, mx.array]:
        pairs, lj_scales, _ = self._pairs_and_scales(positions, None)
        if pairs.shape[0] == 0:
            return _zero_energy(positions), mx.zeros_like(positions)

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
        pair_energy = 4.0 * epsilon_ij * (inv_r12 - inv_r6)
        if self.lj_shift and self.cutoff is not None:
            sigma2_over_rc2 = (sigma_ij * sigma_ij) / (self.cutoff * self.cutoff)
            inv_rc6 = sigma2_over_rc2 * sigma2_over_rc2 * sigma2_over_rc2
            inv_rc12 = inv_rc6 * inv_rc6
            pair_energy = pair_energy - 4.0 * epsilon_ij * (inv_rc12 - inv_rc6)
        switch, switch_derivative = self._switch(distance)
        unswitched_pair_energy = pair_energy
        pair_energy = pair_energy * switch
        pair_energy = mx.where(pair_mask, pair_energy * lj_scales, 0.0)

        scalar = (
            24.0 * epsilon_ij * (2.0 * inv_r12 - inv_r6) / safe_r2 * switch
            - unswitched_pair_energy * switch_derivative / distance
        )
        scalar = mx.where(pair_mask, scalar * lj_scales, 0.0)
        pair_forces = scalar[:, None] * displacement
        forces = mx.zeros_like(positions).at[i].add(pair_forces).at[j].add(-pair_forces)
        return mx.sum(pair_energy), forces

    def _exception_components(
        self,
        positions: mx.array,
        cell: Cell | None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        if not self.has_exceptions:
            zero = _zero_energy(positions)
            return zero, zero, mx.zeros_like(positions)
        pairs = self.exception_pairs
        i = pairs[:, 0]
        j = pairs[:, 1]
        displacement = positions[i] - positions[j]
        if cell is not None:
            displacement = cell.minimum_image(displacement)
        r2 = mx.sum(displacement * displacement, axis=-1)
        mask = r2 > 0.0
        safe_r2 = mx.where(mask, r2, 1.0)
        distance = mx.sqrt(safe_r2)

        sigma2_over_r2 = (self.exception_sigma * self.exception_sigma) / safe_r2
        inv_r6 = sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2
        inv_r12 = inv_r6 * inv_r6
        lj_pair_energy = 4.0 * self.exception_epsilon * (inv_r12 - inv_r6)
        coulomb_pair_energy = self.coulomb_constant * self.exception_charge_products / distance
        lj_pair_energy = mx.where(mask, lj_pair_energy, 0.0)
        coulomb_pair_energy = mx.where(mask, coulomb_pair_energy, 0.0)

        lj_scalar = 24.0 * self.exception_epsilon * (2.0 * inv_r12 - inv_r6) / safe_r2
        coulomb_scalar = self.coulomb_constant * self.exception_charge_products / (
            safe_r2 * distance
        )
        scalar = mx.where(mask, lj_scalar + coulomb_scalar, 0.0)
        pair_forces = scalar[:, None] * displacement
        forces = mx.zeros_like(positions).at[i].add(pair_forces).at[j].add(-pair_forces)
        return mx.sum(lj_pair_energy), mx.sum(coulomb_pair_energy), forces

    def _exception_lj_components(
        self,
        positions: mx.array,
        cell: Cell | None,
    ) -> tuple[mx.array, mx.array]:
        if not self.has_exceptions:
            return _zero_energy(positions), mx.zeros_like(positions)
        pairs = self.exception_pairs
        i = pairs[:, 0]
        j = pairs[:, 1]
        displacement = positions[i] - positions[j]
        if cell is not None:
            displacement = cell.minimum_image(displacement)
        r2 = mx.sum(displacement * displacement, axis=-1)
        mask = r2 > 0.0
        safe_r2 = mx.where(mask, r2, 1.0)

        sigma2_over_r2 = (self.exception_sigma * self.exception_sigma) / safe_r2
        inv_r6 = sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2
        inv_r12 = inv_r6 * inv_r6
        lj_pair_energy = 4.0 * self.exception_epsilon * (inv_r12 - inv_r6)
        lj_pair_energy = mx.where(mask, lj_pair_energy, 0.0)

        scalar = 24.0 * self.exception_epsilon * (2.0 * inv_r12 - inv_r6) / safe_r2
        scalar = mx.where(mask, scalar, 0.0)
        pair_forces = scalar[:, None] * displacement
        forces = mx.zeros_like(positions).at[i].add(pair_forces).at[j].add(-pair_forces)
        return mx.sum(lj_pair_energy), forces

    def _bare_coulomb_components(
        self,
        positions: mx.array,
        cell: Cell | None,
        pairs: mx.array,
        charge_products: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if pairs.shape[0] == 0:
            return _zero_energy(positions), mx.zeros_like(positions)
        i = pairs[:, 0]
        j = pairs[:, 1]
        displacement = positions[i] - positions[j]
        if cell is not None:
            displacement = cell.minimum_image(displacement)
        r2 = mx.sum(displacement * displacement, axis=-1)
        mask = r2 > 0.0
        safe_r2 = mx.where(mask, r2, 1.0)
        distance = mx.sqrt(safe_r2)

        pair_energy = self.coulomb_constant * charge_products / distance
        pair_energy = mx.where(mask, pair_energy, 0.0)
        scalar = self.coulomb_constant * charge_products / (safe_r2 * distance)
        scalar = mx.where(mask, scalar, 0.0)
        pair_forces = scalar[:, None] * displacement
        forces = mx.zeros_like(positions).at[i].add(pair_forces).at[j].add(-pair_forces)
        return mx.sum(pair_energy), forces

    def _ewald_correction_pairs(self) -> mx.array:
        pairs = set(self._exception_pair_set)
        if self.topology is not None:
            pairs.update(self.topology.exclusion_set)
        if not pairs:
            return mx.array(np.empty((0, 2), dtype=np.int32), dtype=mx.int32)
        return mx.array(tuple(sorted(pairs)), dtype=mx.int32)

    def _ewald_one_four_pairs(self) -> mx.array:
        if self.topology is None or self.coulomb_one_four_scale == 1.0:
            return mx.array(np.empty((0, 2), dtype=np.int32), dtype=mx.int32)
        excluded = set(self.topology.exclusion_set)
        excluded.update(self._exception_pair_set)
        pairs = [pair for pair in self.topology.one_four_set if pair not in excluded]
        if not pairs:
            return mx.array(np.empty((0, 2), dtype=np.int32), dtype=mx.int32)
        return mx.array(tuple(sorted(pairs)), dtype=mx.int32)

    def _ewald_energy_forces_with_components(
        self,
        positions: mx.array,
        cell: Cell | None,
        pairs: mx.array | None,
    ) -> tuple[mx.array, mx.array, dict[str, mx.array]]:
        if cell is None:
            msg = "Ewald reference electrostatics requires a periodic cell"
            raise ValueError(msg)
        if pairs is not None:
            msg = "Ewald reference electrostatics requires full-system evaluation"
            raise ValueError(msg)

        lj_energy, lj_forces = self._regular_lj_components(positions, cell)
        exception_lj, exception_lj_forces = self._exception_lj_components(positions, cell)
        lj_energy = lj_energy + exception_lj
        lj_forces = lj_forces + exception_lj_forces

        ewald_energy, ewald_forces, ewald_components = ewald_reference_coulomb_energy_forces(
            positions,
            self.charges,
            cell,
            coulomb_constant=self.coulomb_constant,
            config=self.ewald_config,
        )
        correction_pairs = self._ewald_correction_pairs()
        if correction_pairs.shape[0] == 0:
            zero = _zero_energy(positions)
            correction_energy = zero
            correction_forces = mx.zeros_like(positions)
        else:
            i = correction_pairs[:, 0]
            j = correction_pairs[:, 1]
            original_charge_products = self.charges[i] * self.charges[j]
            correction_energy, correction_forces = self._bare_coulomb_components(
                positions,
                cell,
                correction_pairs,
                -original_charge_products,
            )

        exception_coulomb, exception_coulomb_forces = self._bare_coulomb_components(
            positions,
            cell,
            self.exception_pairs,
            self.exception_charge_products,
        )
        one_four_pairs = self._ewald_one_four_pairs()
        if one_four_pairs.shape[0] == 0:
            one_four_energy = _zero_energy(positions)
            one_four_forces = mx.zeros_like(positions)
        else:
            i = one_four_pairs[:, 0]
            j = one_four_pairs[:, 1]
            one_four_charge_products = (
                (self.coulomb_one_four_scale - 1.0) * self.charges[i] * self.charges[j]
            )
            one_four_energy, one_four_forces = self._bare_coulomb_components(
                positions,
                cell,
                one_four_pairs,
                one_four_charge_products,
            )
        coulomb_energy = ewald_energy + correction_energy + exception_coulomb + one_four_energy
        coulomb_forces = (
            ewald_forces
            + correction_forces
            + exception_coulomb_forces
            + one_four_forces
        )
        components = {
            "lj": lj_energy,
            "coulomb": coulomb_energy,
            "coulomb_real": ewald_components["coulomb_real"],
            "coulomb_reciprocal": ewald_components["coulomb_reciprocal"],
            "coulomb_self": ewald_components["coulomb_self"],
            "coulomb_exclusion_correction": correction_energy,
            "coulomb_exception": exception_coulomb,
            "coulomb_one_four_correction": one_four_energy,
        }
        return lj_energy + coulomb_energy, lj_forces + coulomb_forces, components

    def _pme_energy_forces_with_components(
        self,
        positions: mx.array,
        cell: Cell | None,
        pairs: mx.array | None,
    ) -> tuple[mx.array, mx.array, dict[str, mx.array | object]]:
        if cell is None:
            msg = "PME electrostatics requires a periodic cell"
            raise ValueError(msg)
        if pairs is not None:
            msg = "PME electrostatics requires full-system evaluation"
            raise ValueError(msg)
        if self.pme_config is None:
            msg = "PME electrostatics requires pme_config"
            raise ValueError(msg)

        lj_energy, lj_forces = self._regular_lj_components(positions, cell)
        exception_lj, exception_lj_forces = self._exception_lj_components(positions, cell)
        lj_energy = lj_energy + exception_lj
        lj_forces = lj_forces + exception_lj_forces

        pme_energy, pme_forces, pme_components = pme_coulomb_energy_forces(
            positions,
            self.charges,
            cell,
            coulomb_constant=self.coulomb_constant,
            config=self.pme_config,
        )
        correction_pairs = self._ewald_correction_pairs()
        if correction_pairs.shape[0] == 0:
            correction_energy = _zero_energy(positions)
            correction_forces = mx.zeros_like(positions)
        else:
            i = correction_pairs[:, 0]
            j = correction_pairs[:, 1]
            original_charge_products = self.charges[i] * self.charges[j]
            correction_energy, correction_forces = self._bare_coulomb_components(
                positions,
                cell,
                correction_pairs,
                -original_charge_products,
            )

        exception_coulomb, exception_coulomb_forces = self._bare_coulomb_components(
            positions,
            cell,
            self.exception_pairs,
            self.exception_charge_products,
        )
        one_four_pairs = self._ewald_one_four_pairs()
        if one_four_pairs.shape[0] == 0:
            one_four_energy = _zero_energy(positions)
            one_four_forces = mx.zeros_like(positions)
        else:
            i = one_four_pairs[:, 0]
            j = one_four_pairs[:, 1]
            one_four_charge_products = (
                (self.coulomb_one_four_scale - 1.0) * self.charges[i] * self.charges[j]
            )
            one_four_energy, one_four_forces = self._bare_coulomb_components(
                positions,
                cell,
                one_four_pairs,
                one_four_charge_products,
            )
        coulomb_energy = pme_energy + correction_energy + exception_coulomb + one_four_energy
        coulomb_forces = (
            pme_forces
            + correction_forces
            + exception_coulomb_forces
            + one_four_forces
        )
        components = {
            "lj": lj_energy,
            "coulomb": coulomb_energy,
            "coulomb_real": pme_components["coulomb_real"],
            "coulomb_reciprocal": pme_components["coulomb_reciprocal"],
            "coulomb_self": pme_components["coulomb_self"],
            "coulomb_exclusion_correction": correction_energy,
            "coulomb_exception": exception_coulomb,
            "coulomb_one_four_correction": one_four_energy,
            "pme_diagnostics": pme_components["diagnostics"],
        }
        return lj_energy + coulomb_energy, lj_forces + coulomb_forces, components

    def component_energies(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> dict[str, mx.array | object]:
        """Return LJ and Coulomb energy components."""

        positions = as_mx_array(positions)
        if self.electrostatics in {"ewald_reference", "pme"}:
            _, _, components = self.energy_forces_with_components(positions, cell, pairs)
            return components
        _, lj_energy, coulomb_energy, _, _, _ = self._pair_components(positions, cell, pairs)
        exception_lj, exception_coulomb, _ = self._exception_components(positions, cell)
        lj_energy = lj_energy + exception_lj
        coulomb_energy = coulomb_energy + exception_coulomb
        return {"lj": lj_energy, "coulomb": coulomb_energy}

    def energy_forces_with_components(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, dict[str, mx.array | object]]:
        """Return total energy, forces, and LJ/Coulomb components in one pass."""

        positions = as_mx_array(positions)
        if positions.ndim != 2 or positions.shape[1] != 3:
            msg = "positions must have shape (n_atoms, 3)"
            raise ValueError(msg)
        if self.electrostatics == "ewald_reference":
            return self._ewald_energy_forces_with_components(positions, cell, pairs)
        if self.electrostatics == "pme":
            return self._pme_energy_forces_with_components(positions, cell, pairs)
        _, lj_energy, coulomb_energy, forces, _, _ = self._pair_components(
            positions,
            cell,
            pairs,
        )
        exception_lj, exception_coulomb, exception_forces = self._exception_components(
            positions,
            cell,
        )
        lj_energy = lj_energy + exception_lj
        coulomb_energy = coulomb_energy + exception_coulomb
        return (
            lj_energy + coulomb_energy,
            forces + exception_forces,
            {"lj": lj_energy, "coulomb": coulomb_energy},
        )

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
        energy, forces, _ = self.energy_forces_with_components(positions, cell, pairs)
        return energy, forces


@dataclass(frozen=True)
class PairRestrictedNonbondedPotential:
    """Nonbonded potential evaluated only on an explicit atom-pair list."""

    potential: NonbondedPotential
    pairs: object
    name: str = "pair_restricted_nonbonded"
    supports_virial: bool = True

    def __post_init__(self) -> None:
        pairs = np.asarray(self.pairs, dtype=np.int32)
        if pairs.size == 0:
            pairs = np.empty((0, 2), dtype=np.int32)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            msg = "pairs must have shape (n, 2)"
            raise ValueError(msg)
        atom_count = int(self.potential.sigma.shape[0])
        if pairs.size and (np.any(pairs < 0) or np.any(pairs >= atom_count)):
            msg = "pairs contain atom indices outside [0, n_atoms)"
            raise ValueError(msg)
        object.__setattr__(self, "pairs", mx.array(pairs, dtype=mx.int32))

    def component_energies(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> dict[str, mx.array]:
        del pairs
        return self.potential.component_energies(positions, cell=cell, pairs=self.pairs)

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        del pairs
        return self.potential.energy_forces(positions, cell=cell, pairs=self.pairs)

    def energy_forces_with_components(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, dict[str, mx.array]]:
        del pairs
        return self.potential.energy_forces_with_components(
            positions,
            cell=cell,
            pairs=self.pairs,
        )
