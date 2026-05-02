"""MLX execution helpers for pairwise nonbonded interactions."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell
from mlx_atomistic.topology import Topology

NonbondedBackend = Literal["auto", "mlx_dense", "mlx_tiled", "mlx_pairs", "python_neighbor"]
NonbondedElectrostatics = Literal["cutoff", "ewald_reference", "pme"]

DEFAULT_DENSE_MEMORY_BUDGET_BYTES = 32 * 1024**3
_FLOAT_BYTES = 4
_ELECTROSTATICS_ALIASES = {
    "direct": "cutoff",
    "direct_cutoff": "cutoff",
    "minimum_image": "cutoff",
    "short_range": "cutoff",
    "short_range_cutoff": "cutoff",
    "short_range_electrostatics_prototype": "cutoff",
    "ewald": "ewald_reference",
    "reference_ewald": "ewald_reference",
    "particle_mesh_ewald": "pme",
    "pme_ewald": "pme",
    "pme_ewald_periodic_electrostatics": "pme",
}


@dataclass(frozen=True)
class NonbondedExecutionConfig:
    """Execution controls for MLX nonbonded pair evaluation."""

    backend: NonbondedBackend = "auto"
    electrostatics: NonbondedElectrostatics = "cutoff"
    tile_size: int = 512
    memory_budget_bytes: int | None = DEFAULT_DENSE_MEMORY_BUDGET_BYTES

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", validate_nonbonded_backend(self.backend))
        object.__setattr__(
            self,
            "electrostatics",
            validate_nonbonded_electrostatics(self.electrostatics),
        )
        if self.tile_size <= 0:
            msg = "tile_size must be positive"
            raise ValueError(msg)
        if self.memory_budget_bytes is not None and self.memory_budget_bytes <= 0:
            msg = "memory_budget_bytes must be positive when provided"
            raise ValueError(msg)


@dataclass(frozen=True)
class EwaldReferenceConfig:
    """Controls for the small-system Ewald reference electrostatics backend."""

    alpha: float = 0.35
    real_cutoff: float | None = None
    reciprocal_cutoff: int = 5
    charge_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            msg = "alpha must be positive"
            raise ValueError(msg)
        if self.real_cutoff is not None and self.real_cutoff <= 0.0:
            msg = "real_cutoff must be positive when provided"
            raise ValueError(msg)
        if self.reciprocal_cutoff < 0:
            msg = "reciprocal_cutoff must be non-negative"
            raise ValueError(msg)
        if self.charge_tolerance < 0.0:
            msg = "charge_tolerance must be non-negative"
            raise ValueError(msg)


def validate_nonbonded_backend(backend: str) -> NonbondedBackend:
    """Validate and normalize a nonbonded backend name."""

    allowed = {"auto", "mlx_dense", "mlx_tiled", "mlx_pairs", "python_neighbor"}
    if backend not in allowed:
        msg = f"unknown nonbonded backend {backend!r}; expected one of {sorted(allowed)}"
        raise ValueError(msg)
    return backend  # type: ignore[return-value]


def normalize_nonbonded_electrostatics(mode: str) -> NonbondedElectrostatics:
    """Normalize an electrostatics mode or known metadata alias."""

    normalized = mode.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _ELECTROSTATICS_ALIASES.get(normalized, normalized)
    allowed = {"cutoff", "ewald_reference", "pme"}
    if normalized not in allowed:
        msg = f"unknown electrostatics mode {mode!r}; expected one of {sorted(allowed)}"
        raise ValueError(msg)
    return normalized  # type: ignore[return-value]


def validate_nonbonded_electrostatics(mode: str) -> NonbondedElectrostatics:
    """Validate an electrostatics mode for executable nonbonded evaluation."""

    return normalize_nonbonded_electrostatics(mode)


def ewald_reference_coulomb_energy(
    positions: mx.array,
    charges: mx.array,
    cell: Cell,
    *,
    coulomb_constant: float = 1.0,
    config: EwaldReferenceConfig | None = None,
) -> tuple[mx.array, dict[str, mx.array]]:
    """Evaluate neutral orthorhombic Ewald Coulomb energy.

    This is a correctness reference for small periodic systems. It is not a
    particle-mesh implementation and is intentionally kept separate from the
    production nonbonded force path until force validation is complete.
    """

    config = EwaldReferenceConfig() if config is None else config
    positions = mx.array(positions, dtype=mx.float32)
    charges = mx.array(charges, dtype=mx.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    if charges.shape != (positions.shape[0],):
        msg = "charges must have shape (n_atoms,)"
        raise ValueError(msg)
    cell_lengths = np.asarray(cell.lengths, dtype=np.float32)
    if cell_lengths.shape != (3,) or np.any(cell_lengths <= 0.0):
        msg = "Ewald reference requires positive orthorhombic cell lengths"
        raise ValueError(msg)
    net_charge = float(np.asarray(mx.sum(charges)))
    if abs(net_charge) > config.charge_tolerance:
        msg = (
            "Ewald reference electrostatics requires a neutral system: "
            f"net_charge={net_charge:g}"
        )
        raise ValueError(msg)

    real_cutoff = config.real_cutoff
    if real_cutoff is None:
        real_cutoff = 0.5 * float(np.min(cell_lengths))
    real_shifts = _ewald_real_shifts(cell_lengths, real_cutoff)
    k_vectors = _ewald_k_vectors(cell_lengths, config.reciprocal_cutoff)

    real_energy = _ewald_real_energy(
        positions,
        charges,
        real_shifts,
        alpha=config.alpha,
        cutoff=real_cutoff,
        coulomb_constant=coulomb_constant,
    )
    reciprocal_energy = _ewald_reciprocal_energy(
        positions,
        charges,
        k_vectors,
        cell_lengths=cell_lengths,
        alpha=config.alpha,
        coulomb_constant=coulomb_constant,
    )
    self_energy = (
        -float(coulomb_constant)
        * config.alpha
        / float(np.sqrt(np.pi))
        * mx.sum(charges * charges)
    )
    total = real_energy + reciprocal_energy + self_energy
    return total, {
        "coulomb_real": real_energy,
        "coulomb_reciprocal": reciprocal_energy,
        "coulomb_self": self_energy,
    }


def ewald_reference_coulomb_energy_forces(
    positions: mx.array,
    charges: mx.array,
    cell: Cell,
    *,
    coulomb_constant: float = 1.0,
    config: EwaldReferenceConfig | None = None,
) -> tuple[mx.array, mx.array, dict[str, mx.array]]:
    """Evaluate neutral orthorhombic Ewald Coulomb energy and analytical forces."""

    config = EwaldReferenceConfig() if config is None else config
    positions = mx.array(positions, dtype=mx.float32)
    charges = mx.array(charges, dtype=mx.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    if charges.shape != (positions.shape[0],):
        msg = "charges must have shape (n_atoms,)"
        raise ValueError(msg)
    cell_lengths = np.asarray(cell.lengths, dtype=np.float32)
    if cell_lengths.shape != (3,) or np.any(cell_lengths <= 0.0):
        msg = "Ewald reference requires positive orthorhombic cell lengths"
        raise ValueError(msg)
    net_charge = float(np.asarray(mx.sum(charges)))
    if abs(net_charge) > config.charge_tolerance:
        msg = (
            "Ewald reference electrostatics requires a neutral system: "
            f"net_charge={net_charge:g}"
        )
        raise ValueError(msg)

    real_cutoff = config.real_cutoff
    if real_cutoff is None:
        real_cutoff = 0.5 * float(np.min(cell_lengths))
    real_shifts = _ewald_real_shifts(cell_lengths, real_cutoff)
    k_vectors = _ewald_k_vectors(cell_lengths, config.reciprocal_cutoff)

    real_energy, real_forces = _ewald_real_energy_forces(
        positions,
        charges,
        real_shifts,
        alpha=config.alpha,
        cutoff=real_cutoff,
        coulomb_constant=coulomb_constant,
    )
    reciprocal_energy, reciprocal_forces = _ewald_reciprocal_energy_forces(
        positions,
        charges,
        k_vectors,
        cell_lengths=cell_lengths,
        alpha=config.alpha,
        coulomb_constant=coulomb_constant,
    )
    self_energy = (
        -float(coulomb_constant)
        * config.alpha
        / float(np.sqrt(np.pi))
        * mx.sum(charges * charges)
    )
    total = real_energy + reciprocal_energy + self_energy
    return total, real_forces + reciprocal_forces, {
        "coulomb_real": real_energy,
        "coulomb_reciprocal": reciprocal_energy,
        "coulomb_self": self_energy,
    }


def estimate_dense_nonbonded_bytes(n_atoms: int, *, components: str = "combined") -> int:
    """Return a conservative estimate for dense MLX nonbonded temporaries."""

    if n_atoms < 0:
        msg = "n_atoms must be non-negative"
        raise ValueError(msg)
    # Pair displacement takes three dense float matrices. The remaining factors
    # cover r2, masks/scales, mixed parameters, energies, and force scalars.
    factor = 14 if components == "lj" else 22
    return int(n_atoms) * int(n_atoms) * factor * _FLOAT_BYTES


def choose_nonbonded_backend(
    *,
    requested: NonbondedBackend,
    n_atoms: int,
    pairs_provided: bool,
    estimated_dense_bytes: int,
    memory_budget_bytes: int | None,
) -> NonbondedBackend:
    """Choose the concrete backend for one nonbonded evaluation."""

    validate_nonbonded_backend(requested)
    if requested == "auto":
        if pairs_provided:
            return "mlx_pairs"
        if _within_budget(estimated_dense_bytes, memory_budget_bytes):
            return "mlx_dense"
        return "mlx_tiled"
    if requested == "python_neighbor":
        return "mlx_pairs" if pairs_provided else "mlx_tiled"
    if requested == "mlx_dense" and not _within_budget(estimated_dense_bytes, memory_budget_bytes):
        budget = "unbounded" if memory_budget_bytes is None else str(memory_budget_bytes)
        msg = (
            "mlx_dense nonbonded evaluation exceeds memory budget: "
            f"estimated_dense_bytes={estimated_dense_bytes}, memory_budget_bytes={budget}"
        )
        raise MemoryError(msg)
    return requested


def _within_budget(estimated_dense_bytes: int, memory_budget_bytes: int | None) -> bool:
    return memory_budget_bytes is None or estimated_dense_bytes <= memory_budget_bytes


def dense_lj_energy_forces(
    positions: mx.array,
    *,
    epsilon: float,
    sigma: float,
    cutoff: float | None,
    shift: bool,
    cell: Cell | None,
    topology: Topology | None,
    one_four_scale: float,
    tile_size: int | None = None,
) -> tuple[mx.array, mx.array]:
    """Evaluate uniform Lennard-Jones energy and forces with dense MLX arrays."""

    if tile_size is not None:
        return tiled_lj_energy_forces(
            positions,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            shift=shift,
            cell=cell,
            topology=topology,
            one_four_scale=one_four_scale,
            tile_size=tile_size,
        )
    n_atoms = int(positions.shape[0])
    if n_atoms <= 1:
        return _zero_energy(positions), mx.zeros_like(positions)
    displacement = positions[:, None, :] - positions[None, :, :]
    if cell is not None:
        displacement = cell.minimum_image(displacement)
    r2 = mx.sum(displacement * displacement, axis=-1)
    pair_mask, scales, _ = dense_pair_mask_and_scales(
        n_atoms,
        r2=r2,
        cutoff=cutoff,
        topology=topology,
        lj_one_four_scale=one_four_scale,
        coulomb_one_four_scale=one_four_scale,
    )
    energy, forces = _lj_from_displacement(
        displacement,
        r2,
        pair_mask,
        scales,
        epsilon=epsilon,
        sigma=sigma,
        cutoff=cutoff,
        shift=shift,
    )
    return energy, forces


def tiled_lj_energy_forces(
    positions: mx.array,
    *,
    epsilon: float,
    sigma: float,
    cutoff: float | None,
    shift: bool,
    cell: Cell | None,
    topology: Topology | None,
    one_four_scale: float,
    tile_size: int,
) -> tuple[mx.array, mx.array]:
    """Evaluate uniform Lennard-Jones energy and forces by row tiles."""

    n_atoms = int(positions.shape[0])
    if n_atoms <= 1:
        return _zero_energy(positions), mx.zeros_like(positions)
    topology_mask, lj_scales, _ = topology_dense_matrices(
        n_atoms,
        topology=topology,
        lj_one_four_scale=one_four_scale,
        coulomb_one_four_scale=one_four_scale,
    )
    force_blocks = []
    total_energy = _zero_energy(positions)
    cols = mx.arange(n_atoms)
    for start in range(0, n_atoms, tile_size):
        stop = min(start + tile_size, n_atoms)
        block = positions[start:stop]
        rows = mx.arange(stop - start) + start
        displacement = block[:, None, :] - positions[None, :, :]
        if cell is not None:
            displacement = cell.minimum_image(displacement)
        r2 = mx.sum(displacement * displacement, axis=-1)
        pair_mask = rows[:, None] != cols[None, :]
        if topology_mask is not None:
            pair_mask = pair_mask & topology_mask[start:stop]
        if cutoff is not None:
            pair_mask = pair_mask & (r2 < cutoff * cutoff)
        scales = mx.ones_like(r2)
        if lj_scales is not None:
            scales = lj_scales[start:stop]
        energy, forces = _lj_from_displacement(
            displacement,
            r2,
            pair_mask,
            scales,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            shift=shift,
        )
        total_energy = total_energy + energy
        force_blocks.append(forces)
    return total_energy, mx.concatenate(force_blocks, axis=0)


def dense_combined_energy_forces(
    positions: mx.array,
    *,
    sigma: mx.array,
    epsilon: mx.array,
    charges: mx.array,
    coulomb_constant: float,
    cutoff: float | None,
    lj_shift: bool,
    coulomb_shift: bool,
    cell: Cell | None,
    topology: Topology | None,
    lj_one_four_scale: float,
    coulomb_one_four_scale: float,
    tile_size: int | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Evaluate mixed LJ+Coulomb energy components and forces with MLX arrays."""

    if tile_size is not None:
        return tiled_combined_energy_forces(
            positions,
            sigma=sigma,
            epsilon=epsilon,
            charges=charges,
            coulomb_constant=coulomb_constant,
            cutoff=cutoff,
            lj_shift=lj_shift,
            coulomb_shift=coulomb_shift,
            cell=cell,
            topology=topology,
            lj_one_four_scale=lj_one_four_scale,
            coulomb_one_four_scale=coulomb_one_four_scale,
            tile_size=tile_size,
        )
    n_atoms = int(positions.shape[0])
    if n_atoms <= 1:
        zero = _zero_energy(positions)
        return zero, zero, mx.zeros_like(positions)
    displacement = positions[:, None, :] - positions[None, :, :]
    if cell is not None:
        displacement = cell.minimum_image(displacement)
    r2 = mx.sum(displacement * displacement, axis=-1)
    pair_mask, lj_scales, coulomb_scales = dense_pair_mask_and_scales(
        n_atoms,
        r2=r2,
        cutoff=cutoff,
        topology=topology,
        lj_one_four_scale=lj_one_four_scale,
        coulomb_one_four_scale=coulomb_one_four_scale,
    )
    sigma_ij = 0.5 * (sigma[:, None] + sigma[None, :])
    epsilon_ij = mx.sqrt(epsilon[:, None] * epsilon[None, :])
    qij = charges[:, None] * charges[None, :]
    lj_energy, coulomb_energy, forces = _combined_from_displacement(
        displacement,
        r2,
        pair_mask,
        lj_scales,
        coulomb_scales,
        sigma_ij=sigma_ij,
        epsilon_ij=epsilon_ij,
        qij=qij,
        coulomb_constant=coulomb_constant,
        cutoff=cutoff,
        lj_shift=lj_shift,
        coulomb_shift=coulomb_shift,
    )
    return lj_energy, coulomb_energy, forces


def tiled_combined_energy_forces(
    positions: mx.array,
    *,
    sigma: mx.array,
    epsilon: mx.array,
    charges: mx.array,
    coulomb_constant: float,
    cutoff: float | None,
    lj_shift: bool,
    coulomb_shift: bool,
    cell: Cell | None,
    topology: Topology | None,
    lj_one_four_scale: float,
    coulomb_one_four_scale: float,
    tile_size: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Evaluate mixed LJ+Coulomb energy components and forces by row tiles."""

    n_atoms = int(positions.shape[0])
    if n_atoms <= 1:
        zero = _zero_energy(positions)
        return zero, zero, mx.zeros_like(positions)
    topology_mask, lj_scale_matrix, coulomb_scale_matrix = topology_dense_matrices(
        n_atoms,
        topology=topology,
        lj_one_four_scale=lj_one_four_scale,
        coulomb_one_four_scale=coulomb_one_four_scale,
    )
    force_blocks = []
    total_lj = _zero_energy(positions)
    total_coulomb = _zero_energy(positions)
    cols = mx.arange(n_atoms)
    for start in range(0, n_atoms, tile_size):
        stop = min(start + tile_size, n_atoms)
        block = positions[start:stop]
        rows = mx.arange(stop - start) + start
        displacement = block[:, None, :] - positions[None, :, :]
        if cell is not None:
            displacement = cell.minimum_image(displacement)
        r2 = mx.sum(displacement * displacement, axis=-1)
        pair_mask = rows[:, None] != cols[None, :]
        if topology_mask is not None:
            pair_mask = pair_mask & topology_mask[start:stop]
        if cutoff is not None:
            pair_mask = pair_mask & (r2 < cutoff * cutoff)
        sigma_ij = 0.5 * (sigma[start:stop, None] + sigma[None, :])
        epsilon_ij = mx.sqrt(epsilon[start:stop, None] * epsilon[None, :])
        qij = charges[start:stop, None] * charges[None, :]
        lj_scales = mx.ones_like(r2)
        coulomb_scales = mx.ones_like(r2)
        if lj_scale_matrix is not None:
            lj_scales = lj_scale_matrix[start:stop]
        if coulomb_scale_matrix is not None:
            coulomb_scales = coulomb_scale_matrix[start:stop]
        lj_energy, coulomb_energy, forces = _combined_from_displacement(
            displacement,
            r2,
            pair_mask,
            lj_scales,
            coulomb_scales,
            sigma_ij=sigma_ij,
            epsilon_ij=epsilon_ij,
            qij=qij,
            coulomb_constant=coulomb_constant,
            cutoff=cutoff,
            lj_shift=lj_shift,
            coulomb_shift=coulomb_shift,
        )
        total_lj = total_lj + lj_energy
        total_coulomb = total_coulomb + coulomb_energy
        force_blocks.append(forces)
    return total_lj, total_coulomb, mx.concatenate(force_blocks, axis=0)


def dense_pair_mask_and_scales(
    n_atoms: int,
    *,
    r2: mx.array,
    cutoff: float | None,
    topology: Topology | None,
    lj_one_four_scale: float,
    coulomb_one_four_scale: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """Return dense pair mask and scale matrices."""

    indices = mx.arange(n_atoms)
    pair_mask = indices[:, None] != indices[None, :]
    topology_mask, lj_scales, coulomb_scales = topology_dense_matrices(
        n_atoms,
        topology=topology,
        lj_one_four_scale=lj_one_four_scale,
        coulomb_one_four_scale=coulomb_one_four_scale,
    )
    if topology_mask is not None:
        pair_mask = pair_mask & topology_mask
    if cutoff is not None:
        pair_mask = pair_mask & (r2 < cutoff * cutoff)
    if lj_scales is None:
        lj_scales = mx.ones_like(r2)
    if coulomb_scales is None:
        coulomb_scales = mx.ones_like(r2)
    return pair_mask, lj_scales, coulomb_scales


def topology_dense_matrices(
    n_atoms: int,
    *,
    topology: Topology | None,
    lj_one_four_scale: float,
    coulomb_one_four_scale: float,
) -> tuple[mx.array | None, mx.array | None, mx.array | None]:
    """Build dense topology masks/scales without creating explicit pair lists."""

    if topology is None:
        return None, None, None
    if topology.n_atoms != n_atoms:
        msg = "topology.n_atoms must match positions"
        raise ValueError(msg)
    enabled = np.ones((n_atoms, n_atoms), dtype=np.bool_)
    np.fill_diagonal(enabled, False)
    exclusions = np.asarray(topology.exclusions, dtype=np.int32)
    if exclusions.size:
        enabled[exclusions[:, 0], exclusions[:, 1]] = False
        enabled[exclusions[:, 1], exclusions[:, 0]] = False
    lj_scales = np.ones((n_atoms, n_atoms), dtype=np.float32)
    coulomb_scales = np.ones((n_atoms, n_atoms), dtype=np.float32)
    one_four = np.asarray(topology.one_four_pairs, dtype=np.int32)
    if one_four.size:
        lj_scales[one_four[:, 0], one_four[:, 1]] = float(lj_one_four_scale)
        lj_scales[one_four[:, 1], one_four[:, 0]] = float(lj_one_four_scale)
        coulomb_scales[one_four[:, 0], one_four[:, 1]] = float(coulomb_one_four_scale)
        coulomb_scales[one_four[:, 1], one_four[:, 0]] = float(coulomb_one_four_scale)
    return mx.array(enabled), mx.array(lj_scales), mx.array(coulomb_scales)


def _lj_from_displacement(
    displacement: mx.array,
    r2: mx.array,
    pair_mask: mx.array,
    scales: mx.array,
    *,
    epsilon: float,
    sigma: float,
    cutoff: float | None,
    shift: bool,
) -> tuple[mx.array, mx.array]:
    safe_r2 = mx.where(pair_mask, r2, 1.0)
    sigma2_over_r2 = (sigma * sigma) / safe_r2
    inv_r6 = sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2
    inv_r12 = inv_r6 * inv_r6
    pair_energy = 4.0 * epsilon * (inv_r12 - inv_r6)
    if shift and cutoff is not None:
        sigma2_over_rc2 = (sigma * sigma) / (cutoff * cutoff)
        inv_rc6 = sigma2_over_rc2 * sigma2_over_rc2 * sigma2_over_rc2
        inv_rc12 = inv_rc6 * inv_rc6
        pair_energy = pair_energy - 4.0 * epsilon * (inv_rc12 - inv_rc6)
    pair_energy = mx.where(pair_mask, pair_energy * scales, 0.0)
    scalar = 24.0 * epsilon * (2.0 * inv_r12 - inv_r6) / safe_r2
    scalar = mx.where(pair_mask, scalar * scales, 0.0)
    forces = mx.sum(scalar[:, :, None] * displacement, axis=1)
    return 0.5 * mx.sum(pair_energy), forces


def _combined_from_displacement(
    displacement: mx.array,
    r2: mx.array,
    pair_mask: mx.array,
    lj_scales: mx.array,
    coulomb_scales: mx.array,
    *,
    sigma_ij: mx.array,
    epsilon_ij: mx.array,
    qij: mx.array,
    coulomb_constant: float,
    cutoff: float | None,
    lj_shift: bool,
    coulomb_shift: bool,
) -> tuple[mx.array, mx.array, mx.array]:
    safe_r2 = mx.where(pair_mask, r2, 1.0)
    distance = mx.sqrt(safe_r2)

    sigma2_over_r2 = (sigma_ij * sigma_ij) / safe_r2
    inv_r6 = sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2
    inv_r12 = inv_r6 * inv_r6
    lj_pair_energy = 4.0 * epsilon_ij * (inv_r12 - inv_r6)
    if lj_shift and cutoff is not None:
        sigma2_over_rc2 = (sigma_ij * sigma_ij) / (cutoff * cutoff)
        inv_rc6 = sigma2_over_rc2 * sigma2_over_rc2 * sigma2_over_rc2
        inv_rc12 = inv_rc6 * inv_rc6
        lj_pair_energy = lj_pair_energy - 4.0 * epsilon_ij * (inv_rc12 - inv_rc6)
    lj_pair_energy = mx.where(pair_mask, lj_pair_energy * lj_scales, 0.0)

    coulomb_pair_energy = coulomb_constant * qij / distance
    if coulomb_shift and cutoff is not None:
        coulomb_pair_energy = coulomb_pair_energy - coulomb_constant * qij / cutoff
    coulomb_pair_energy = mx.where(pair_mask, coulomb_pair_energy * coulomb_scales, 0.0)

    lj_scalar = 24.0 * epsilon_ij * (2.0 * inv_r12 - inv_r6) / safe_r2
    coulomb_scalar = coulomb_constant * qij / (safe_r2 * distance)
    scalar = mx.where(
        pair_mask,
        lj_scalar * lj_scales + coulomb_scalar * coulomb_scales,
        0.0,
    )
    forces = mx.sum(scalar[:, :, None] * displacement, axis=1)
    return 0.5 * mx.sum(lj_pair_energy), 0.5 * mx.sum(coulomb_pair_energy), forces


def _ewald_real_shifts(cell_lengths: np.ndarray, cutoff: float) -> mx.array:
    ranges = [
        range(
            -int(ceil(float(cutoff) / float(length))) - 1,
            int(ceil(float(cutoff) / float(length))) + 2,
        )
        for length in cell_lengths
    ]
    shifts = [
        (
            nx * float(cell_lengths[0]),
            ny * float(cell_lengths[1]),
            nz * float(cell_lengths[2]),
        )
        for nx in ranges[0]
        for ny in ranges[1]
        for nz in ranges[2]
    ]
    return mx.array(np.asarray(shifts, dtype=np.float32))


def _ewald_k_vectors(cell_lengths: np.ndarray, reciprocal_cutoff: int) -> mx.array:
    if reciprocal_cutoff == 0:
        return mx.array(np.empty((0, 3), dtype=np.float32))
    vectors = []
    for nx in range(-reciprocal_cutoff, reciprocal_cutoff + 1):
        for ny in range(-reciprocal_cutoff, reciprocal_cutoff + 1):
            for nz in range(-reciprocal_cutoff, reciprocal_cutoff + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                vectors.append(
                    (
                        2.0 * np.pi * nx / float(cell_lengths[0]),
                        2.0 * np.pi * ny / float(cell_lengths[1]),
                        2.0 * np.pi * nz / float(cell_lengths[2]),
                    )
                )
    return mx.array(np.asarray(vectors, dtype=np.float32))


def _ewald_real_energy(
    positions: mx.array,
    charges: mx.array,
    shifts: mx.array,
    *,
    alpha: float,
    cutoff: float,
    coulomb_constant: float,
) -> mx.array:
    displacement = positions[:, None, None, :] - positions[None, :, None, :] + shifts
    r2 = mx.sum(displacement * displacement, axis=-1)
    distance = mx.sqrt(mx.where(r2 > 0.0, r2, 1.0))
    zero_shift = mx.sum(shifts * shifts, axis=-1) == 0.0
    atom_index = mx.arange(positions.shape[0])
    self_image = (atom_index[:, None, None] == atom_index[None, :, None]) & zero_shift
    pair_mask = (r2 > 0.0) & (distance < cutoff) & ~self_image
    qij = charges[:, None, None] * charges[None, :, None]
    pair_energy = float(coulomb_constant) * qij * (1.0 - mx.erf(alpha * distance)) / distance
    pair_energy = mx.where(pair_mask, pair_energy, 0.0)
    return 0.5 * mx.sum(pair_energy)


def _ewald_real_energy_forces(
    positions: mx.array,
    charges: mx.array,
    shifts: mx.array,
    *,
    alpha: float,
    cutoff: float,
    coulomb_constant: float,
) -> tuple[mx.array, mx.array]:
    displacement = positions[:, None, None, :] - positions[None, :, None, :] + shifts
    r2 = mx.sum(displacement * displacement, axis=-1)
    distance = mx.sqrt(mx.where(r2 > 0.0, r2, 1.0))
    zero_shift = mx.sum(shifts * shifts, axis=-1) == 0.0
    atom_index = mx.arange(positions.shape[0])
    self_image = (atom_index[:, None, None] == atom_index[None, :, None]) & zero_shift
    pair_mask = (r2 > 0.0) & (distance < cutoff) & ~self_image
    qij = charges[:, None, None] * charges[None, :, None]
    erfc = 1.0 - mx.erf(alpha * distance)
    pair_energy = float(coulomb_constant) * qij * erfc / distance
    pair_energy = mx.where(pair_mask, pair_energy, 0.0)

    safe_r2 = mx.where(pair_mask, r2, 1.0)
    safe_distance = mx.sqrt(safe_r2)
    scalar = float(coulomb_constant) * qij * (
        erfc / (safe_r2 * safe_distance)
        + (2.0 * alpha / float(np.sqrt(np.pi)))
        * mx.exp(-(alpha * alpha) * safe_r2)
        / safe_r2
    )
    scalar = mx.where(pair_mask, scalar, 0.0)
    forces = mx.sum(scalar[:, :, :, None] * displacement, axis=(1, 2))
    return 0.5 * mx.sum(pair_energy), forces


def _ewald_reciprocal_energy(
    positions: mx.array,
    charges: mx.array,
    k_vectors: mx.array,
    *,
    cell_lengths: np.ndarray,
    alpha: float,
    coulomb_constant: float,
) -> mx.array:
    if k_vectors.shape[0] == 0:
        return _zero_energy(positions)
    k_dot_r = positions @ mx.transpose(k_vectors)
    structure_cos = mx.sum(charges[:, None] * mx.cos(k_dot_r), axis=0)
    structure_sin = mx.sum(charges[:, None] * mx.sin(k_dot_r), axis=0)
    structure2 = structure_cos * structure_cos + structure_sin * structure_sin
    k2 = mx.sum(k_vectors * k_vectors, axis=-1)
    volume = float(np.prod(cell_lengths))
    coefficient = (
        float(coulomb_constant)
        * 2.0
        * float(np.pi)
        / volume
        * mx.exp(-k2 / (4.0 * alpha * alpha))
        / k2
    )
    return mx.sum(coefficient * structure2)


def _ewald_reciprocal_energy_forces(
    positions: mx.array,
    charges: mx.array,
    k_vectors: mx.array,
    *,
    cell_lengths: np.ndarray,
    alpha: float,
    coulomb_constant: float,
) -> tuple[mx.array, mx.array]:
    if k_vectors.shape[0] == 0:
        return _zero_energy(positions), mx.zeros_like(positions)
    k_dot_r = positions @ mx.transpose(k_vectors)
    cos_phase = mx.cos(k_dot_r)
    sin_phase = mx.sin(k_dot_r)
    structure_cos = mx.sum(charges[:, None] * cos_phase, axis=0)
    structure_sin = mx.sum(charges[:, None] * sin_phase, axis=0)
    structure2 = structure_cos * structure_cos + structure_sin * structure_sin
    k2 = mx.sum(k_vectors * k_vectors, axis=-1)
    volume = float(np.prod(cell_lengths))
    coefficient = (
        float(coulomb_constant)
        * 2.0
        * float(np.pi)
        / volume
        * mx.exp(-k2 / (4.0 * alpha * alpha))
        / k2
    )
    energy = mx.sum(coefficient * structure2)
    force_scale = (
        2.0
        * coefficient[None, :]
        * charges[:, None]
        * (structure_cos[None, :] * sin_phase - structure_sin[None, :] * cos_phase)
    )
    forces = mx.sum(force_scale[:, :, None] * k_vectors[None, :, :], axis=1)
    return energy, forces


def _zero_energy(positions: mx.array) -> mx.array:
    return mx.sum(positions[:, 0] * 0.0)
