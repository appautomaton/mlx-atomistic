"""Standalone particle-mesh Ewald electrostatics for small periodic fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import mlx.core as mx
import numpy as np
from scipy.special import erfc

from mlx_atomistic.core import Cell

PME_EXECUTION_BACKEND = "numpy_reference"
PME_PRODUCTION_EXECUTABLE = False


@dataclass(frozen=True)
class PMEConfig:
    """Controls for the standalone PME mesh backend."""

    mesh_shape: tuple[int, int, int] = (32, 32, 32)
    alpha: float = 0.35
    real_cutoff: float | None = None
    assignment_order: int = 2
    charge_tolerance: float = 1e-5
    deconvolve_assignment: bool = True

    def __post_init__(self) -> None:
        if len(self.mesh_shape) != 3:
            msg = "mesh_shape must contain exactly three dimensions"
            raise ValueError(msg)
        if any(int(size) != size or size < 4 for size in self.mesh_shape):
            msg = "mesh_shape dimensions must be integers >= 4"
            raise ValueError(msg)
        object.__setattr__(self, "mesh_shape", tuple(int(size) for size in self.mesh_shape))
        if self.alpha <= 0.0:
            msg = "alpha must be positive"
            raise ValueError(msg)
        if self.real_cutoff is not None and self.real_cutoff <= 0.0:
            msg = "real_cutoff must be positive when provided"
            raise ValueError(msg)
        if self.assignment_order != 2:
            msg = "standalone PME currently supports assignment_order=2 (CIC) only"
            raise ValueError(msg)
        if self.charge_tolerance < 0.0:
            msg = "charge_tolerance must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class PMEDiagnostics:
    """Diagnostics emitted by one standalone PME evaluation."""

    mesh_shape: tuple[int, int, int]
    assignment_order: int
    alpha: float
    real_cutoff: float
    net_charge: float
    volume: float
    charge_grid_sum: float
    reciprocal_modes: int
    max_charge_grid_abs: float

    def to_dict(self) -> dict[str, float | int | tuple[int, int, int]]:
        return {
            "mesh_shape": self.mesh_shape,
            "assignment_order": self.assignment_order,
            "alpha": self.alpha,
            "real_cutoff": self.real_cutoff,
            "net_charge": self.net_charge,
            "volume": self.volume,
            "charge_grid_sum": self.charge_grid_sum,
            "reciprocal_modes": self.reciprocal_modes,
            "max_charge_grid_abs": self.max_charge_grid_abs,
        }


def pme_readiness_report(
    *,
    atom_count: int,
    charges: object,
    cell_lengths: object,
    config: PMEConfig | None,
    nonbonded_cutoff: float | None,
    exclusion_count: int,
    one_four_count: int,
    explicit_exception_count: int,
) -> dict[str, object]:
    """Return fail-closed PME readiness metadata for production run gates."""

    checks: dict[str, bool] = {}
    blockers: list[str] = []

    checks["production_executable_backend"] = PME_PRODUCTION_EXECUTABLE
    if not PME_PRODUCTION_EXECUTABLE:
        blockers.append(
            "pme_backend_not_production_executable:current_backend=numpy_reference"
        )

    if config is None:
        checks["config"] = False
        blockers.append("pme_config:missing")
        charge_tolerance = 1e-5
    else:
        checks["config"] = True
        charge_tolerance = float(config.charge_tolerance)
        checks["mesh_shape"] = (
            len(config.mesh_shape) == 3
            and all(isinstance(size, int) and size >= 4 for size in config.mesh_shape)
        )
        checks["alpha"] = np.isfinite(float(config.alpha)) and float(config.alpha) > 0.0
        checks["cutoff"] = (
            config.real_cutoff is not None
            and np.isfinite(float(config.real_cutoff))
            and float(config.real_cutoff) > 0.0
            and nonbonded_cutoff is not None
            and np.isfinite(float(nonbonded_cutoff))
            and float(nonbonded_cutoff) > 0.0
        )
        for name in ("mesh_shape", "alpha", "cutoff"):
            if not checks[name]:
                blockers.append(f"pme_{name}:invalid")

    charge_values = np.asarray(charges, dtype=np.float64)
    net_charge = float(np.sum(charge_values, dtype=np.float64)) if charge_values.size else 0.0
    checks["neutrality"] = bool(
        charge_values.shape == (int(atom_count),)
        and np.all(np.isfinite(charge_values))
        and abs(net_charge) <= charge_tolerance
    )
    if not checks["neutrality"]:
        blockers.append(f"neutrality:net_charge={net_charge:g}")

    box = np.asarray(cell_lengths, dtype=np.float64)
    checks["box"] = bool(box.shape == (3,) and np.all(np.isfinite(box)) and np.all(box > 0.0))
    if not checks["box"]:
        blockers.append("box:missing_or_invalid")

    checks["exclusions"] = int(exclusion_count) >= 0
    checks["one_four_corrections"] = int(one_four_count) >= 0
    checks["explicit_exceptions"] = int(explicit_exception_count) >= 0
    for name in ("exclusions", "one_four_corrections", "explicit_exceptions"):
        if not checks[name]:
            blockers.append(f"{name}:invalid")

    return {
        "status": "ready" if not blockers else "blocked",
        "backend": PME_EXECUTION_BACKEND,
        "production_executable": PME_PRODUCTION_EXECUTABLE,
        "atom_count": int(atom_count),
        "net_charge": net_charge,
        "mesh_shape": None if config is None else config.mesh_shape,
        "alpha": None if config is None else float(config.alpha),
        "real_cutoff": None if config is None else config.real_cutoff,
        "nonbonded_cutoff": nonbonded_cutoff,
        "exclusion_count": int(exclusion_count),
        "one_four_count": int(one_four_count),
        "explicit_exception_count": int(explicit_exception_count),
        "checks": checks,
        "blockers": tuple(blockers),
    }


def pme_coulomb_energy_forces(
    positions: mx.array,
    charges: mx.array,
    cell: Cell,
    *,
    coulomb_constant: float = 1.0,
    config: PMEConfig | None = None,
) -> tuple[mx.array, mx.array, dict[str, mx.array | PMEDiagnostics]]:
    """Evaluate neutral orthorhombic Coulomb energy and forces with PME."""

    config = PMEConfig() if config is None else config
    positions_np, charges_np, cell_lengths = _validate_inputs(
        positions,
        charges,
        cell,
        charge_tolerance=config.charge_tolerance,
    )
    real_cutoff = config.real_cutoff
    if real_cutoff is None:
        real_cutoff = 0.5 * float(np.min(cell_lengths))

    real_energy, real_forces = _real_space_energy_forces(
        positions_np,
        charges_np,
        cell_lengths,
        alpha=config.alpha,
        cutoff=real_cutoff,
        coulomb_constant=coulomb_constant,
    )
    reciprocal_energy, reciprocal_forces, mesh_info = _mesh_reciprocal_energy_forces(
        positions_np,
        charges_np,
        cell_lengths,
        config=config,
        coulomb_constant=coulomb_constant,
    )
    self_energy = (
        -float(coulomb_constant)
        * config.alpha
        / float(np.sqrt(np.pi))
        * float(np.sum(charges_np * charges_np, dtype=np.float64))
    )

    total_energy = real_energy + reciprocal_energy + self_energy
    forces = real_forces + reciprocal_forces
    diagnostics = PMEDiagnostics(
        mesh_shape=config.mesh_shape,
        assignment_order=config.assignment_order,
        alpha=config.alpha,
        real_cutoff=real_cutoff,
        net_charge=float(np.sum(charges_np, dtype=np.float64)),
        volume=float(np.prod(cell_lengths, dtype=np.float64)),
        charge_grid_sum=mesh_info["charge_grid_sum"],
        reciprocal_modes=int(mesh_info["reciprocal_modes"]),
        max_charge_grid_abs=mesh_info["max_charge_grid_abs"],
    )
    return (
        mx.array(total_energy, dtype=mx.float32),
        mx.array(forces.astype(np.float32)),
        {
            "coulomb_real": mx.array(real_energy, dtype=mx.float32),
            "coulomb_reciprocal": mx.array(reciprocal_energy, dtype=mx.float32),
            "coulomb_self": mx.array(self_energy, dtype=mx.float32),
            "diagnostics": diagnostics,
        },
    )


def assign_charges_cic(
    positions: mx.array,
    charges: mx.array,
    cell: Cell,
    mesh_shape: tuple[int, int, int],
) -> mx.array:
    """Assign charges to a periodic mesh with cloud-in-cell weights."""

    positions_np, charges_np, cell_lengths = _validate_inputs(
        positions,
        charges,
        cell,
        charge_tolerance=np.inf,
    )
    mesh_shape = _validate_mesh_shape(mesh_shape)
    grid = _assign_charges_cic_np(positions_np, charges_np, cell_lengths, mesh_shape)
    return mx.array(grid.astype(np.float32))


def _validate_inputs(
    positions: mx.array,
    charges: mx.array,
    cell: Cell,
    *,
    charge_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(cell, Cell):
        msg = "PME requires an orthorhombic Cell"
        raise ValueError(msg)
    positions_np = np.asarray(positions, dtype=np.float64)
    charges_np = np.asarray(charges, dtype=np.float64)
    if positions_np.ndim != 2 or positions_np.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    if charges_np.shape != (positions_np.shape[0],):
        msg = "charges must have shape (n_atoms,)"
        raise ValueError(msg)
    if not np.all(np.isfinite(positions_np)):
        msg = "positions must be finite"
        raise ValueError(msg)
    if not np.all(np.isfinite(charges_np)):
        msg = "charges must be finite"
        raise ValueError(msg)
    cell_lengths = np.asarray(cell.lengths, dtype=np.float64)
    if cell_lengths.shape != (3,) or not np.all(np.isfinite(cell_lengths)):
        msg = "PME requires finite orthorhombic cell lengths with shape (3,)"
        raise ValueError(msg)
    if np.any(cell_lengths <= 0.0):
        msg = "PME requires positive orthorhombic cell lengths"
        raise ValueError(msg)
    net_charge = float(np.sum(charges_np, dtype=np.float64))
    if abs(net_charge) > charge_tolerance:
        msg = (
            "PME requires a neutral system; non-neutral background policy is not "
            f"implemented: net_charge={net_charge:g}"
        )
        raise ValueError(msg)
    return np.mod(positions_np, cell_lengths), charges_np, cell_lengths


def _validate_mesh_shape(mesh_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(mesh_shape) != 3:
        msg = "mesh_shape must contain exactly three dimensions"
        raise ValueError(msg)
    if any(int(size) != size or size < 4 for size in mesh_shape):
        msg = "mesh_shape dimensions must be integers >= 4"
        raise ValueError(msg)
    normalized = tuple(int(size) for size in mesh_shape)
    return normalized


def _real_space_energy_forces(
    positions: np.ndarray,
    charges: np.ndarray,
    cell_lengths: np.ndarray,
    *,
    alpha: float,
    cutoff: float,
    coulomb_constant: float,
) -> tuple[float, np.ndarray]:
    shifts = _real_space_shifts(cell_lengths, cutoff)
    displacement = positions[:, None, None, :] - positions[None, :, None, :] + shifts
    r2 = np.sum(displacement * displacement, axis=-1)
    distance = np.sqrt(np.where(r2 > 0.0, r2, 1.0))
    zero_shift = np.sum(shifts * shifts, axis=-1) == 0.0
    atom_index = np.arange(positions.shape[0])
    self_image = (atom_index[:, None, None] == atom_index[None, :, None]) & zero_shift
    pair_mask = (r2 > 0.0) & (distance < cutoff) & ~self_image
    qij = charges[:, None, None] * charges[None, :, None]
    erfc_term = erfc(alpha * distance)
    pair_energy = float(coulomb_constant) * qij * erfc_term / distance
    pair_energy = np.where(pair_mask, pair_energy, 0.0)

    safe_r2 = np.where(pair_mask, r2, 1.0)
    safe_distance = np.sqrt(safe_r2)
    scalar = float(coulomb_constant) * qij * (
        erfc_term / (safe_r2 * safe_distance)
        + (2.0 * alpha / float(np.sqrt(np.pi))) * np.exp(-(alpha * alpha) * safe_r2) / safe_r2
    )
    scalar = np.where(pair_mask, scalar, 0.0)
    forces = np.sum(scalar[:, :, :, None] * displacement, axis=(1, 2))
    return 0.5 * float(np.sum(pair_energy, dtype=np.float64)), forces


def _real_space_shifts(cell_lengths: np.ndarray, cutoff: float) -> np.ndarray:
    ranges = [
        range(
            -int(ceil(float(cutoff) / float(length))) - 1,
            int(ceil(float(cutoff) / float(length))) + 2,
        )
        for length in cell_lengths
    ]
    return np.asarray(
        [
            (
                nx * float(cell_lengths[0]),
                ny * float(cell_lengths[1]),
                nz * float(cell_lengths[2]),
            )
            for nx in ranges[0]
            for ny in ranges[1]
            for nz in ranges[2]
        ],
        dtype=np.float64,
    )


def _mesh_reciprocal_energy_forces(
    positions: np.ndarray,
    charges: np.ndarray,
    cell_lengths: np.ndarray,
    *,
    config: PMEConfig,
    coulomb_constant: float,
) -> tuple[float, np.ndarray, dict[str, float | int]]:
    charge_grid = _assign_charges_cic_np(positions, charges, cell_lengths, config.mesh_shape)
    rho_hat = np.fft.fftn(charge_grid)
    influence, k_components, mode_count = _influence_function(
        cell_lengths,
        config.mesh_shape,
        alpha=config.alpha,
        coulomb_constant=coulomb_constant,
        deconvolve_assignment=config.deconvolve_assignment,
    )
    phi_hat = influence * rho_hat
    grid_size = int(np.prod(config.mesh_shape))
    potential_grid = np.fft.ifftn(phi_hat).real * grid_size
    field_grids = [
        np.fft.ifftn((-1j * k_axis) * phi_hat).real * grid_size
        for k_axis in k_components
    ]
    field_grid = np.stack(field_grids, axis=-1)
    potential_at_atoms = _interpolate_cic_np(positions, potential_grid, cell_lengths)
    field_at_atoms = _interpolate_cic_np(positions, field_grid, cell_lengths)
    energy = 0.5 * float(np.sum(charges * potential_at_atoms, dtype=np.float64))
    forces = charges[:, None] * field_at_atoms
    return energy, forces, {
        "charge_grid_sum": float(np.sum(charge_grid, dtype=np.float64)),
        "max_charge_grid_abs": float(np.max(np.abs(charge_grid))) if charge_grid.size else 0.0,
        "reciprocal_modes": mode_count,
    }


def _influence_function(
    cell_lengths: np.ndarray,
    mesh_shape: tuple[int, int, int],
    *,
    alpha: float,
    coulomb_constant: float,
    deconvolve_assignment: bool,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], int]:
    k_components = []
    window = np.ones(mesh_shape, dtype=np.float64)
    for axis, (length, size) in enumerate(zip(cell_lengths, mesh_shape, strict=True)):
        frequencies = np.fft.fftfreq(size, d=float(length) / size)
        k_axis = 2.0 * np.pi * frequencies
        shape = [1, 1, 1]
        shape[axis] = size
        k_grid = k_axis.reshape(shape)
        k_components.append(np.broadcast_to(k_grid, mesh_shape))
        if deconvolve_assignment:
            window_axis = np.sinc(k_axis * float(length) / (2.0 * np.pi * size)) ** 2
            window *= np.broadcast_to(window_axis.reshape(shape), mesh_shape)

    kx, ky, kz = k_components
    k2 = kx * kx + ky * ky + kz * kz
    mask = k2 > 0.0
    volume = float(np.prod(cell_lengths, dtype=np.float64))
    influence = np.zeros(mesh_shape, dtype=np.float64)
    denominator = k2.copy()
    if deconvolve_assignment:
        denominator = denominator * np.maximum(window * window, 1e-12)
    influence[mask] = (
        float(coulomb_constant)
        * 4.0
        * np.pi
        / volume
        * np.exp(-k2[mask] / (4.0 * alpha * alpha))
        / denominator[mask]
    )
    return influence, (kx, ky, kz), int(np.count_nonzero(mask))


def _assign_charges_cic_np(
    positions: np.ndarray,
    charges: np.ndarray,
    cell_lengths: np.ndarray,
    mesh_shape: tuple[int, int, int],
) -> np.ndarray:
    grid = np.zeros(mesh_shape, dtype=np.float64)
    scaled = np.mod(positions, cell_lengths) / cell_lengths * np.asarray(mesh_shape)
    base = np.floor(scaled).astype(np.int64)
    fraction = scaled - base
    for atom_index, charge in enumerate(charges):
        for dx in (0, 1):
            wx = (1.0 - fraction[atom_index, 0]) if dx == 0 else fraction[atom_index, 0]
            ix = (base[atom_index, 0] + dx) % mesh_shape[0]
            for dy in (0, 1):
                wy = (1.0 - fraction[atom_index, 1]) if dy == 0 else fraction[atom_index, 1]
                iy = (base[atom_index, 1] + dy) % mesh_shape[1]
                for dz in (0, 1):
                    wz = (
                        (1.0 - fraction[atom_index, 2])
                        if dz == 0
                        else fraction[atom_index, 2]
                    )
                    iz = (base[atom_index, 2] + dz) % mesh_shape[2]
                    grid[ix, iy, iz] += charge * wx * wy * wz
    return grid


def _interpolate_cic_np(
    positions: np.ndarray,
    grid: np.ndarray,
    cell_lengths: np.ndarray,
) -> np.ndarray:
    mesh_shape = grid.shape[:3]
    trailing_shape = grid.shape[3:]
    values = np.zeros((positions.shape[0], *trailing_shape), dtype=np.float64)
    scaled = np.mod(positions, cell_lengths) / cell_lengths * np.asarray(mesh_shape)
    base = np.floor(scaled).astype(np.int64)
    fraction = scaled - base
    for atom_index in range(positions.shape[0]):
        for dx in (0, 1):
            wx = (1.0 - fraction[atom_index, 0]) if dx == 0 else fraction[atom_index, 0]
            ix = (base[atom_index, 0] + dx) % mesh_shape[0]
            for dy in (0, 1):
                wy = (1.0 - fraction[atom_index, 1]) if dy == 0 else fraction[atom_index, 1]
                iy = (base[atom_index, 1] + dy) % mesh_shape[1]
                for dz in (0, 1):
                    wz = (
                        (1.0 - fraction[atom_index, 2])
                        if dz == 0
                        else fraction[atom_index, 2]
                    )
                    iz = (base[atom_index, 2] + dz) % mesh_shape[2]
                    values[atom_index] += wx * wy * wz * grid[ix, iy, iz]
    return values
