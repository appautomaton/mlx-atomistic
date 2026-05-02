"""CHARMM-specific molecular mechanics force-term primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell, as_mx_array


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


def _parameter_array(value, *, count: int, name: str) -> mx.array:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = np.full((count,), float(array), dtype=np.float32)
    if array.shape != (count,):
        msg = f"{name} must be scalar or have shape ({count},)"
        raise ValueError(msg)
    return as_mx_array(array)


def _finite_parameter_array(value, *, count: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = np.full((count,), float(array), dtype=np.float32)
    if array.shape != (count,):
        msg = f"{name} must be scalar or have shape ({count},)"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} values must be finite"
        raise ValueError(msg)
    return array


def _index_array(value, *, width: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.int32)
    if array.size == 0:
        array = np.empty((0, width), dtype=np.int32)
    if array.ndim != 2 or array.shape[1] != width:
        msg = f"{name} must have shape (n, {width})"
        raise ValueError(msg)
    if np.any(array < 0):
        msg = f"{name} atom indices must be non-negative"
        raise ValueError(msg)
    return array


def _pair_array(value, *, name: str) -> np.ndarray:
    pairs = _index_array(value, width=2, name=name)
    seen: set[tuple[int, int]] = set()
    for left, right in pairs.tolist():
        if left == right:
            msg = f"{name} must not contain self pairs"
            raise ValueError(msg)
        pair = (min(int(left), int(right)), max(int(left), int(right)))
        if pair in seen:
            msg = f"{name} must not contain duplicate pairs"
            raise ValueError(msg)
        seen.add(pair)
    return pairs


def _all_pairs(count: int) -> mx.array:
    if count < 2:
        return mx.array(np.empty((0, 2), dtype=np.int32), dtype=mx.int32)
    pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
    return mx.array(pairs, dtype=mx.int32)


def _dihedral_angle(
    positions: mx.array,
    atoms: tuple[int, int, int, int],
    cell: Cell | None,
) -> mx.array:
    i, j, k, m = atoms
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
    return angle * sign


def _catmull_rom(p0: mx.array, p1: mx.array, p2: mx.array, p3: mx.array, t: mx.array) -> mx.array:
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _periodic_cubic_grid_value(grid: mx.array, phi: mx.array, psi: mx.array) -> mx.array:
    size = int(grid.shape[0])
    phi_scaled = (phi + pi) * (size / (2.0 * pi))
    psi_scaled = (psi + pi) * (size / (2.0 * pi))
    phi_base = mx.floor(phi_scaled).astype(mx.int32)
    psi_base = mx.floor(psi_scaled).astype(mx.int32)
    phi_t = phi_scaled - mx.floor(phi_scaled)
    psi_t = psi_scaled - mx.floor(psi_scaled)

    row_values = []
    for phi_offset in (-1, 0, 1, 2):
        values = []
        phi_index = (phi_base + phi_offset) % size
        for psi_offset in (-1, 0, 1, 2):
            psi_index = (psi_base + psi_offset) % size
            values.append(grid[phi_index, psi_index])
        row_values.append(_catmull_rom(values[0], values[1], values[2], values[3], psi_t))
    return _catmull_rom(row_values[0], row_values[1], row_values[2], row_values[3], phi_t)


@dataclass(frozen=True)
class CHARMMUreyBradleyPotential:
    """CHARMM Urey-Bradley 1-3 distance term for angle triplets."""

    urey_bradley_terms: object
    k: object
    distance: object
    name: str = "urey_bradley"
    supports_virial: bool = True

    def __post_init__(self) -> None:
        terms = _index_array(self.urey_bradley_terms, width=3, name="urey_bradley_terms")
        count = terms.shape[0]
        k = _finite_parameter_array(self.k, count=count, name="k")
        distance = _finite_parameter_array(self.distance, count=count, name="distance")
        if np.any(k < 0.0):
            msg = "k values must be non-negative"
            raise ValueError(msg)
        if np.any(distance <= 0.0):
            msg = "distance values must be positive"
            raise ValueError(msg)
        object.__setattr__(self, "urey_bradley_terms", mx.array(terms, dtype=mx.int32))
        object.__setattr__(self, "k", as_mx_array(k))
        object.__setattr__(self, "distance", as_mx_array(distance))

    def potential_energy(self, positions: mx.array, cell: Cell | None = None) -> mx.array:
        positions = as_mx_array(positions)
        if self.urey_bradley_terms.shape[0] == 0:
            return _zero_energy(positions)
        i = self.urey_bradley_terms[:, 0]
        k = self.urey_bradley_terms[:, 2]
        displacement = positions[i] - positions[k]
        if cell is not None:
            displacement = cell.minimum_image(displacement)
        distance = _norm(displacement)
        delta = distance - self.distance
        return 0.5 * mx.sum(self.k * delta * delta)

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        del pairs
        positions = as_mx_array(positions)
        if self.urey_bradley_terms.shape[0] == 0:
            return _zero_energy(positions), mx.zeros_like(positions)
        i = self.urey_bradley_terms[:, 0]
        k = self.urey_bradley_terms[:, 2]
        displacement = positions[i] - positions[k]
        if cell is not None:
            displacement = cell.minimum_image(displacement)
        distance = _norm(displacement)
        delta = distance - self.distance
        energy = 0.5 * mx.sum(self.k * delta * delta)
        scalar = -self.k * delta / distance
        pair_forces = scalar[:, None] * displacement
        forces = mx.zeros_like(positions).at[i].add(pair_forces).at[k].add(-pair_forces)
        return energy, forces


@dataclass(frozen=True)
class CHARMMCMAPPotential:
    """CHARMM CMAP two-dihedral correction term with periodic bicubic interpolation."""

    charmm_cmap_terms: object
    cmap_grids: object
    cmap_indices: object | None = None
    name: str = "charmm_cmap_terms"
    supports_virial: bool = True
    _terms_np: np.ndarray = field(init=False, repr=False)
    _indices_np: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        terms = _index_array(self.charmm_cmap_terms, width=8, name="charmm_cmap_terms")
        grids = np.asarray(self.cmap_grids, dtype=np.float32)
        if grids.ndim == 2:
            grids = grids[None, :, :]
        if grids.ndim != 3 or grids.shape[1] != grids.shape[2]:
            msg = "cmap_grids must have shape (n_maps, grid, grid) or (grid, grid)"
            raise ValueError(msg)
        if grids.shape[1] < 4:
            msg = "cmap_grids must use at least a 4x4 periodic grid"
            raise ValueError(msg)
        if not np.all(np.isfinite(grids)):
            msg = "cmap_grids must be finite"
            raise ValueError(msg)
        if self.cmap_indices is None:
            indices = np.zeros((terms.shape[0],), dtype=np.int32)
        else:
            indices = np.asarray(self.cmap_indices, dtype=np.int32)
        if indices.shape != (terms.shape[0],):
            msg = f"cmap_indices must have shape ({terms.shape[0]},)"
            raise ValueError(msg)
        if np.any(indices < 0) or np.any(indices >= grids.shape[0]):
            msg = "cmap_indices contain map indices outside cmap_grids"
            raise ValueError(msg)

        object.__setattr__(self, "charmm_cmap_terms", mx.array(terms, dtype=mx.int32))
        object.__setattr__(self, "cmap_grids", as_mx_array(grids))
        object.__setattr__(self, "cmap_indices", mx.array(indices, dtype=mx.int32))
        object.__setattr__(self, "_terms_np", terms)
        object.__setattr__(self, "_indices_np", indices)

    def potential_energy(self, positions: mx.array, cell: Cell | None = None) -> mx.array:
        positions = as_mx_array(positions)
        if self._terms_np.shape[0] == 0:
            return _zero_energy(positions)
        energy = _zero_energy(positions)
        for term, map_index in zip(self._terms_np.tolist(), self._indices_np.tolist(), strict=True):
            phi = _dihedral_angle(positions, tuple(term[:4]), cell)
            psi = _dihedral_angle(positions, tuple(term[4:]), cell)
            energy = energy + _periodic_cubic_grid_value(self.cmap_grids[int(map_index)], phi, psi)
        return energy

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        del pairs
        positions = as_mx_array(positions)
        if self._terms_np.shape[0] == 0:
            return _zero_energy(positions), mx.zeros_like(positions)

        def energy_fn(current_positions: mx.array) -> mx.array:
            return self.potential_energy(current_positions, cell)

        energy = energy_fn(positions)
        forces = -mx.grad(energy_fn)(positions)
        return energy, forces


@dataclass(frozen=True)
class CHARMMForceSwitchNonbondedPotential:
    """CHARMM LJ force-switch nonbonded primitive."""

    sigma: object
    epsilon: object
    charges: object
    cutoff: float
    switch_distance: float
    coulomb_constant: float = 1.0
    lj_shift: bool = False
    coulomb_shift: bool = False
    topology: object | None = None
    lj_one_four_scale: float = 1.0
    coulomb_one_four_scale: float = 1.0
    exception_pairs: object = ()
    exception_charge_products: object | None = None
    exception_sigma: object | None = None
    exception_epsilon: object | None = None
    backend: str = "auto"
    tile_size: int = 512
    memory_budget_bytes: int | None = None
    name: str = "charmm_force_switch_nonbonded"
    supports_virial: bool = True
    _atom_count: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        sigma = as_mx_array(self.sigma)
        epsilon = as_mx_array(self.epsilon)
        charges = as_mx_array(self.charges)
        sigma_np = np.asarray(sigma)
        epsilon_np = np.asarray(epsilon)
        charges_np = np.asarray(charges)
        if sigma.ndim != 1 or epsilon.ndim != 1 or charges.ndim != 1:
            msg = "sigma, epsilon, and charges must have shape (n_atoms,)"
            raise ValueError(msg)
        if sigma.shape != epsilon.shape or sigma.shape != charges.shape:
            msg = "sigma, epsilon, and charges must have matching shapes"
            raise ValueError(msg)
        if not np.all(np.isfinite(sigma_np)) or np.any(sigma_np <= 0.0):
            msg = "sigma values must be finite and positive"
            raise ValueError(msg)
        if not np.all(np.isfinite(epsilon_np)) or np.any(epsilon_np < 0.0):
            msg = "epsilon values must be finite and non-negative"
            raise ValueError(msg)
        if not np.all(np.isfinite(charges_np)):
            msg = "charges values must be finite"
            raise ValueError(msg)
        if not np.isfinite(self.coulomb_constant):
            msg = "coulomb_constant must be finite"
            raise ValueError(msg)
        if not np.isfinite(self.cutoff) or self.cutoff <= 0.0:
            msg = "cutoff must be positive"
            raise ValueError(msg)
        if (
            not np.isfinite(self.switch_distance)
            or self.switch_distance <= 0.0
            or self.switch_distance >= self.cutoff
        ):
            msg = "switch_distance must be positive and smaller than cutoff"
            raise ValueError(msg)
        if self.lj_shift:
            msg = "CHARMM force-switch LJ is already zero at cutoff and does not support lj_shift"
            raise ValueError(msg)
        has_exception_pairs = np.asarray(self.exception_pairs, dtype=np.int32).size > 0
        if (
            self.topology is not None
            or self.lj_one_four_scale != 1.0
            or self.coulomb_one_four_scale != 1.0
            or has_exception_pairs
            or self.exception_charge_products is not None
            or self.exception_sigma is not None
            or self.exception_epsilon is not None
        ):
            msg = (
                "CHARMM force-switch primitive does not yet support "
                "topology or exception overrides"
            )
            raise ValueError(msg)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "charges", charges)
        object.__setattr__(self, "_atom_count", int(sigma.shape[0]))

    def _pairs(self, pairs: mx.array | None) -> mx.array:
        if pairs is None:
            return _all_pairs(self._atom_count)
        pair_array = _pair_array(pairs, name="pairs")
        if pair_array.size and np.any(pair_array >= self._atom_count):
            msg = "pairs contain atom indices outside [0, n_atoms)"
            raise ValueError(msg)
        return mx.array(pair_array, dtype=mx.int32)

    def _component_energies_for_pairs(
        self,
        positions: mx.array,
        cell: Cell | None,
        pairs: mx.array,
    ) -> dict[str, mx.array]:
        if pairs.shape[0] == 0:
            zero = _zero_energy(positions)
            return {"lj": zero, "coulomb": zero}
        i = pairs[:, 0]
        j = pairs[:, 1]
        displacement = positions[i] - positions[j]
        if cell is not None:
            displacement = cell.minimum_image(displacement)
        r2 = mx.sum(displacement * displacement, axis=-1)
        pair_mask = (r2 > 0.0) & (r2 < self.cutoff * self.cutoff)
        safe_r2 = mx.where(pair_mask, r2, 1.0)
        distance = mx.sqrt(safe_r2)

        sigma_ij = 0.5 * (self.sigma[i] + self.sigma[j])
        epsilon_ij = mx.sqrt(self.epsilon[i] * self.epsilon[j])
        inv_r = 1.0 / distance
        inv_r3 = inv_r * inv_r * inv_r
        inv_r6 = inv_r3 * inv_r3
        sigma6 = sigma_ij**6
        sigma12 = sigma6 * sigma6
        c12 = 4.0 * epsilon_ij * sigma12
        c6 = 4.0 * epsilon_ij * sigma6

        rc = self.cutoff
        ri = self.switch_distance
        rc3 = rc**3
        rc6 = rc3 * rc3
        ri3 = ri**3
        ri6 = ri3 * ri3
        rc3_inv = 1.0 / rc3
        rc6_inv = 1.0 / rc6
        ri3_inv = 1.0 / ri3
        ri6_inv = 1.0 / ri6

        inner_lj = c12 * (inv_r6 * inv_r6 - ri6_inv * rc6_inv) - c6 * (
            inv_r6 - ri3_inv * rc3_inv
        )
        switched_lj = c12 * rc6 / (rc6 - ri6) * (inv_r6 - rc6_inv) ** 2 - c6 * rc3 / (
            rc3 - ri3
        ) * (inv_r3 - rc3_inv) ** 2
        lj_pair_energy = mx.where(distance <= ri, inner_lj, switched_lj)
        lj_pair_energy = mx.where(pair_mask, lj_pair_energy, 0.0)

        qij = self.charges[i] * self.charges[j]
        coulomb_pair_energy = self.coulomb_constant * qij / distance
        if self.coulomb_shift:
            coulomb_pair_energy = coulomb_pair_energy - self.coulomb_constant * qij / self.cutoff
        coulomb_pair_energy = mx.where(pair_mask, coulomb_pair_energy, 0.0)
        return {"lj": mx.sum(lj_pair_energy), "coulomb": mx.sum(coulomb_pair_energy)}

    def potential_energy(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> mx.array:
        positions = as_mx_array(positions)
        pair_array = self._pairs(pairs)
        components = self._component_energies_for_pairs(positions, cell, pair_array)
        return components["lj"] + components["coulomb"]

    def component_energies(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> dict[str, mx.array]:
        positions = as_mx_array(positions)
        return self._component_energies_for_pairs(positions, cell, self._pairs(pairs))

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        positions = as_mx_array(positions)
        pair_array = self._pairs(pairs)

        def energy_fn(current_positions: mx.array) -> mx.array:
            components = self._component_energies_for_pairs(current_positions, cell, pair_array)
            return components["lj"] + components["coulomb"]

        energy = energy_fn(positions)
        forces = -mx.grad(energy_fn)(positions)
        return energy, forces

    def energy_forces_with_components(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, dict[str, mx.array]]:
        energy, forces = self.energy_forces(positions, cell=cell, pairs=pairs)
        components = self.component_energies(positions, cell=cell, pairs=pairs)
        return energy, forces, components


@dataclass(frozen=True)
class CHARMMNBFIXPairOverridePotential:
    """CHARMM NBFIX-style explicit-pair LJ override layered over regular nonbonded terms."""

    sigma: object
    epsilon: object
    charges: object
    nbfix_pairs: object
    nbfix_sigma: object
    nbfix_epsilon: object
    coulomb_constant: float = 1.0
    cutoff: float | None = 2.5
    switch_distance: float | None = None
    lj_shift: bool = False
    coulomb_shift: bool = False
    backend: str = "auto"
    name: str = "nbfix_pair_overrides"
    supports_virial: bool = True
    potential: object = field(init=False, repr=False)
    _pairs_np: np.ndarray = field(init=False, repr=False)

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
        sigma_np = np.asarray(sigma)
        epsilon_np = np.asarray(epsilon)
        charges_np = np.asarray(charges)
        if not np.all(np.isfinite(sigma_np)) or np.any(sigma_np <= 0.0):
            msg = "sigma values must be finite and positive"
            raise ValueError(msg)
        if not np.all(np.isfinite(epsilon_np)) or np.any(epsilon_np < 0.0):
            msg = "epsilon values must be finite and non-negative"
            raise ValueError(msg)
        if not np.all(np.isfinite(charges_np)):
            msg = "charges values must be finite"
            raise ValueError(msg)
        if not np.isfinite(self.coulomb_constant):
            msg = "coulomb_constant must be finite"
            raise ValueError(msg)
        if self.cutoff is not None and (not np.isfinite(self.cutoff) or self.cutoff <= 0.0):
            msg = "cutoff must be positive"
            raise ValueError(msg)
        if self.switch_distance is not None:
            if self.cutoff is None:
                msg = "switch_distance requires a cutoff"
                raise ValueError(msg)
            if (
                not np.isfinite(self.switch_distance)
                or self.switch_distance < 0.0
                or self.switch_distance >= self.cutoff
            ):
                msg = "switch_distance must be non-negative and smaller than cutoff"
                raise ValueError(msg)
        pairs = _pair_array(self.nbfix_pairs, name="nbfix_pairs")
        if pairs.size and np.any(pairs >= sigma.shape[0]):
            msg = "nbfix_pairs contain atom indices outside [0, n_atoms)"
            raise ValueError(msg)
        pair_count = pairs.shape[0]
        nbfix_sigma = _parameter_array(self.nbfix_sigma, count=pair_count, name="nbfix_sigma")
        nbfix_epsilon = _parameter_array(self.nbfix_epsilon, count=pair_count, name="nbfix_epsilon")
        nbfix_sigma_np = np.asarray(nbfix_sigma)
        nbfix_epsilon_np = np.asarray(nbfix_epsilon)
        if bool(np.any(~np.isfinite(nbfix_sigma_np))) or bool(np.any(nbfix_sigma_np <= 0.0)):
            msg = "nbfix_sigma values must be finite and positive"
            raise ValueError(msg)
        if bool(np.any(~np.isfinite(nbfix_epsilon_np))) or bool(np.any(nbfix_epsilon_np < 0.0)):
            msg = "nbfix_epsilon values must be finite and non-negative"
            raise ValueError(msg)

        from mlx_atomistic.forcefields import NonbondedPotential

        potential = NonbondedPotential(
            sigma=sigma,
            epsilon=epsilon,
            charges=charges,
            coulomb_constant=self.coulomb_constant,
            cutoff=self.cutoff,
            lj_shift=self.lj_shift,
            coulomb_shift=self.coulomb_shift,
            switch_distance=self.switch_distance,
            backend=self.backend,
        )
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "charges", charges)
        object.__setattr__(self, "nbfix_pairs", mx.array(pairs, dtype=mx.int32))
        object.__setattr__(self, "nbfix_sigma", nbfix_sigma)
        object.__setattr__(self, "nbfix_epsilon", nbfix_epsilon)
        object.__setattr__(self, "potential", potential)
        object.__setattr__(self, "_pairs_np", pairs)

    def _switch(self, distance: mx.array) -> tuple[mx.array, mx.array]:
        if self.switch_distance is None or self.cutoff is None:
            return mx.ones_like(distance), mx.zeros_like(distance)
        width = self.cutoff - self.switch_distance
        x = mx.clip((distance - self.switch_distance) / width, 0.0, 1.0)
        smooth = 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5
        return 1.0 - smooth, mx.zeros_like(distance)

    def _lj_energy_for_parameters(
        self,
        positions: mx.array,
        cell: Cell | None,
        sigma: mx.array,
        epsilon: mx.array,
    ) -> mx.array:
        if self.nbfix_pairs.shape[0] == 0:
            return _zero_energy(positions)
        i = self.nbfix_pairs[:, 0]
        j = self.nbfix_pairs[:, 1]
        displacement = positions[i] - positions[j]
        if cell is not None:
            displacement = cell.minimum_image(displacement)
        r2 = mx.sum(displacement * displacement, axis=-1)
        mask = r2 > 0.0
        if self.cutoff is not None:
            mask = mask & (r2 < self.cutoff * self.cutoff)
        safe_r2 = mx.where(mask, r2, 1.0)
        distance = mx.sqrt(safe_r2)
        sigma2_over_r2 = (sigma * sigma) / safe_r2
        inv_r6 = sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2
        inv_r12 = inv_r6 * inv_r6
        pair_energy = 4.0 * epsilon * (inv_r12 - inv_r6)
        if self.lj_shift and self.cutoff is not None:
            sigma2_over_rc2 = (sigma * sigma) / (self.cutoff * self.cutoff)
            inv_rc6 = sigma2_over_rc2 * sigma2_over_rc2 * sigma2_over_rc2
            inv_rc12 = inv_rc6 * inv_rc6
            pair_energy = pair_energy - 4.0 * epsilon * (inv_rc12 - inv_rc6)
        switch, _ = self._switch(distance)
        pair_energy = mx.where(mask, pair_energy * switch, 0.0)
        return mx.sum(pair_energy)

    def _correction_energy(self, positions: mx.array, cell: Cell | None) -> mx.array:
        if self.nbfix_pairs.shape[0] == 0:
            return _zero_energy(positions)
        i = self.nbfix_pairs[:, 0]
        j = self.nbfix_pairs[:, 1]
        mixed_sigma = 0.5 * (self.sigma[i] + self.sigma[j])
        mixed_epsilon = mx.sqrt(self.epsilon[i] * self.epsilon[j])
        override = self._lj_energy_for_parameters(
            positions,
            cell,
            self.nbfix_sigma,
            self.nbfix_epsilon,
        )
        regular = self._lj_energy_for_parameters(positions, cell, mixed_sigma, mixed_epsilon)
        return override - regular

    def potential_energy(self, positions: mx.array, cell: Cell | None = None) -> mx.array:
        positions = as_mx_array(positions)
        base_energy, _ = self.potential.energy_forces(positions, cell=cell)
        return base_energy + self._correction_energy(positions, cell)

    def component_energies(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> dict[str, mx.array]:
        if pairs is not None:
            msg = "CHARMM NBFIX pair overrides require full-system nonbonded evaluation"
            raise ValueError(msg)
        components = dict(self.potential.component_energies(positions, cell=cell, pairs=pairs))
        components["nbfix_lj_correction"] = self._correction_energy(as_mx_array(positions), cell)
        components["lj"] = components["lj"] + components["nbfix_lj_correction"]
        return components

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        if pairs is not None:
            msg = "CHARMM NBFIX pair overrides require full-system nonbonded evaluation"
            raise ValueError(msg)
        positions = as_mx_array(positions)
        base_energy, base_forces = self.potential.energy_forces(positions, cell=cell)

        def correction_fn(current_positions: mx.array) -> mx.array:
            return self._correction_energy(current_positions, cell)

        correction_energy = correction_fn(positions)
        correction_forces = -mx.grad(correction_fn)(positions)
        return base_energy + correction_energy, base_forces + correction_forces


__all__ = [
    "CHARMMCMAPPotential",
    "CHARMMForceSwitchNonbondedPotential",
    "CHARMMNBFIXPairOverridePotential",
    "CHARMMUreyBradleyPotential",
]
