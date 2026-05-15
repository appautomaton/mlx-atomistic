"""Standalone particle-mesh Ewald electrostatics for small periodic fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell

PME_EXECUTION_BACKEND = "mlx_fft_cic"
PME_PRODUCTION_EXECUTABLE = True
PME_PRODUCTION_MAX_ATOMS = 4096


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
            f"pme_backend_not_production_executable:current_backend={PME_EXECUTION_BACKEND}"
        )
    checks["atom_count"] = 0 <= int(atom_count) <= PME_PRODUCTION_MAX_ATOMS
    if not checks["atom_count"]:
        blockers.append(
            "atom_count:outside_pme_runtime_envelope:"
            f"atom_count={int(atom_count)},max_atoms={PME_PRODUCTION_MAX_ATOMS}"
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
        "runtime_envelope": {
            "max_atoms": PME_PRODUCTION_MAX_ATOMS,
            "cell": "orthorhombic",
            "assignment": "cloud-in-cell",
        },
        "virial": {
            "status": "finite_difference_cell_strain",
            "analytic_supported": False,
        },
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
    positions_mx, charges_mx, cell_lengths_mx, cell_lengths_np = _validate_inputs_mx(
        positions,
        charges,
        cell,
        charge_tolerance=config.charge_tolerance,
    )
    real_cutoff = config.real_cutoff
    if real_cutoff is None:
        real_cutoff = 0.5 * float(np.min(cell_lengths_np))

    real_energy, real_forces = _real_space_energy_forces_mx(
        positions_mx,
        charges_mx,
        cell_lengths_mx,
        cell_lengths_np,
        alpha=config.alpha,
        cutoff=real_cutoff,
        coulomb_constant=coulomb_constant,
    )
    reciprocal_energy, reciprocal_forces, mesh_info = _mesh_reciprocal_energy_forces_mx(
        positions_mx,
        charges_mx,
        cell_lengths_mx,
        cell_lengths_np,
        config=config,
        coulomb_constant=coulomb_constant,
    )
    self_energy = (
        -float(coulomb_constant)
        * config.alpha
        / float(np.sqrt(np.pi))
        * mx.sum(charges_mx * charges_mx)
    )

    total_energy = real_energy + reciprocal_energy + self_energy
    forces = real_forces + reciprocal_forces
    mx.eval(total_energy, forces, real_energy, reciprocal_energy, self_energy)
    diagnostics = PMEDiagnostics(
        mesh_shape=config.mesh_shape,
        assignment_order=config.assignment_order,
        alpha=config.alpha,
        real_cutoff=real_cutoff,
        net_charge=float(np.asarray(mx.sum(charges_mx))),
        volume=float(np.prod(cell_lengths_np, dtype=np.float64)),
        charge_grid_sum=mesh_info["charge_grid_sum"],
        reciprocal_modes=int(mesh_info["reciprocal_modes"]),
        max_charge_grid_abs=mesh_info["max_charge_grid_abs"],
    )
    return (
        total_energy.astype(mx.float32),
        forces.astype(mx.float32),
        {
            "coulomb_real": real_energy.astype(mx.float32),
            "coulomb_reciprocal": reciprocal_energy.astype(mx.float32),
            "coulomb_self": self_energy.astype(mx.float32),
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

    positions_mx, charges_mx, cell_lengths_mx, _ = _validate_inputs_mx(
        positions,
        charges,
        cell,
        charge_tolerance=np.inf,
    )
    mesh_shape = _validate_mesh_shape(mesh_shape)
    return _assign_charges_cic_mx(positions_mx, charges_mx, cell_lengths_mx, mesh_shape)


def _validate_inputs_mx(
    positions: mx.array,
    charges: mx.array,
    cell: Cell,
    *,
    charge_tolerance: float,
) -> tuple[mx.array, mx.array, mx.array, np.ndarray]:
    if not isinstance(cell, Cell):
        msg = "PME requires an orthorhombic Cell"
        raise ValueError(msg)
    positions_mx = mx.array(positions, dtype=mx.float32)
    charges_mx = mx.array(charges, dtype=mx.float32)
    if positions_mx.ndim != 2 or positions_mx.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    if charges_mx.shape != (positions_mx.shape[0],):
        msg = "charges must have shape (n_atoms,)"
        raise ValueError(msg)
    if not bool(np.asarray(mx.all(mx.isfinite(positions_mx)))):
        msg = "positions must be finite"
        raise ValueError(msg)
    if not bool(np.asarray(mx.all(mx.isfinite(charges_mx)))):
        msg = "charges must be finite"
        raise ValueError(msg)
    cell_lengths_mx = mx.array(cell.lengths, dtype=mx.float32)
    cell_lengths_np = np.asarray(cell_lengths_mx, dtype=np.float64)
    if cell_lengths_np.shape != (3,) or not np.all(np.isfinite(cell_lengths_np)):
        msg = "PME requires finite orthorhombic cell lengths with shape (3,)"
        raise ValueError(msg)
    if np.any(cell_lengths_np <= 0.0):
        msg = "PME requires positive orthorhombic cell lengths"
        raise ValueError(msg)
    net_charge = float(np.asarray(mx.sum(charges_mx)))
    if abs(net_charge) > charge_tolerance:
        msg = (
            "PME requires a neutral system; non-neutral background policy is not "
            f"implemented: net_charge={net_charge:g}"
        )
        raise ValueError(msg)
    wrapped_positions = positions_mx - mx.floor(positions_mx / cell_lengths_mx) * cell_lengths_mx
    return wrapped_positions, charges_mx, cell_lengths_mx, cell_lengths_np


def _validate_mesh_shape(mesh_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(mesh_shape) != 3:
        msg = "mesh_shape must contain exactly three dimensions"
        raise ValueError(msg)
    if any(int(size) != size or size < 4 for size in mesh_shape):
        msg = "mesh_shape dimensions must be integers >= 4"
        raise ValueError(msg)
    normalized = tuple(int(size) for size in mesh_shape)
    return normalized


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


def _real_space_energy_forces_mx(
    positions: mx.array,
    charges: mx.array,
    cell_lengths: mx.array,
    cell_lengths_np: np.ndarray,
    *,
    alpha: float,
    cutoff: float,
    coulomb_constant: float,
) -> tuple[mx.array, mx.array]:
    shifts = _real_space_shifts(cell_lengths_np, cutoff)
    n_atoms = int(positions.shape[0])
    atom_index = mx.arange(n_atoms)
    cutoff2 = float(cutoff) * float(cutoff)
    total_energy = mx.array(0.0, dtype=mx.float32)
    forces = mx.zeros_like(positions)
    qij = charges[:, None] * charges[None, :]
    for shift in shifts:
        shift_value = mx.array(shift, dtype=mx.float32)
        displacement = positions[:, None, :] - positions[None, :, :] + shift_value
        r2 = mx.sum(displacement * displacement, axis=-1)
        pair_mask = (r2 > 0.0) & (r2 < cutoff2)
        if bool(np.sum(shift * shift) == 0.0):
            pair_mask = pair_mask & (atom_index[:, None] != atom_index[None, :])
        safe_r2 = mx.where(pair_mask, r2, 1.0)
        distance = mx.sqrt(safe_r2)
        erfc_term = 1.0 - mx.erf(float(alpha) * distance)
        pair_energy = float(coulomb_constant) * qij * erfc_term / distance
        pair_energy = mx.where(pair_mask, pair_energy, 0.0)
        scalar = float(coulomb_constant) * qij * (
            erfc_term / (safe_r2 * distance)
            + (2.0 * float(alpha) / float(np.sqrt(np.pi)))
            * mx.exp(-(float(alpha) * float(alpha)) * safe_r2)
            / safe_r2
        )
        scalar = mx.where(pair_mask, scalar, 0.0)
        forces = forces + mx.sum(scalar[:, :, None] * displacement, axis=1)
        total_energy = total_energy + 0.5 * mx.sum(pair_energy)
    return total_energy, forces


def _mesh_reciprocal_energy_forces_mx(
    positions: mx.array,
    charges: mx.array,
    cell_lengths: mx.array,
    cell_lengths_np: np.ndarray,
    *,
    config: PMEConfig,
    coulomb_constant: float,
) -> tuple[mx.array, mx.array, dict[str, float | int]]:
    charge_grid = _assign_charges_cic_mx(positions, charges, cell_lengths, config.mesh_shape)
    rho_hat = mx.fft.fftn(charge_grid)
    influence, k_components, mode_count = _influence_function_mx(
        cell_lengths_np,
        config.mesh_shape,
        alpha=config.alpha,
        coulomb_constant=coulomb_constant,
        deconvolve_assignment=config.deconvolve_assignment,
    )
    phi_hat = influence * rho_hat
    grid_size = int(np.prod(config.mesh_shape))
    potential_grid = mx.real(mx.fft.ifftn(phi_hat)) * float(grid_size)
    field_grids = [
        mx.real(mx.fft.ifftn((-1j * k_axis) * phi_hat)) * float(grid_size)
        for k_axis in k_components
    ]
    field_grid = mx.stack(field_grids, axis=-1)
    potential_at_atoms = _interpolate_cic_mx(positions, potential_grid, cell_lengths)
    field_at_atoms = _interpolate_cic_mx(positions, field_grid, cell_lengths)
    energy = 0.5 * mx.sum(charges * potential_at_atoms)
    forces = charges[:, None] * field_at_atoms
    mx.eval(energy, forces, charge_grid)
    return energy, forces, {
        "charge_grid_sum": float(np.asarray(mx.sum(charge_grid))),
        "max_charge_grid_abs": float(np.asarray(mx.max(mx.abs(charge_grid))))
        if int(np.prod(config.mesh_shape)) > 0
        else 0.0,
        "reciprocal_modes": mode_count,
    }


def _influence_function_mx(
    cell_lengths: np.ndarray,
    mesh_shape: tuple[int, int, int],
    *,
    alpha: float,
    coulomb_constant: float,
    deconvolve_assignment: bool,
) -> tuple[mx.array, tuple[mx.array, mx.array, mx.array], int]:
    k_components = []
    window = mx.ones(mesh_shape, dtype=mx.float32)
    for axis, (length, size) in enumerate(zip(cell_lengths, mesh_shape, strict=True)):
        frequencies = mx.fft.fftfreq(size, d=float(length) / float(size))
        k_axis = 2.0 * float(np.pi) * frequencies
        shape = [1, 1, 1]
        shape[axis] = int(size)
        k_grid = mx.reshape(k_axis, tuple(shape))
        k_components.append(mx.broadcast_to(k_grid, mesh_shape))
        if deconvolve_assignment:
            window_axis = _sinc_mx(
                k_axis * float(length) / (2.0 * float(np.pi) * float(size))
            ) ** 2
            window = window * mx.broadcast_to(mx.reshape(window_axis, tuple(shape)), mesh_shape)

    kx, ky, kz = k_components
    k2 = kx * kx + ky * ky + kz * kz
    mask = k2 > 0.0
    denominator = k2
    if deconvolve_assignment:
        denominator = denominator * mx.maximum(window * window, mx.array(1e-12))
    safe_denominator = mx.where(mask, denominator, 1.0)
    volume = float(np.prod(cell_lengths, dtype=np.float64))
    influence = (
        float(coulomb_constant)
        * 4.0
        * float(np.pi)
        / volume
        * mx.exp(-k2 / (4.0 * float(alpha) * float(alpha)))
        / safe_denominator
    )
    influence = mx.where(mask, influence, 0.0)
    return influence, (kx, ky, kz), int(np.prod(mesh_shape) - 1)


def _sinc_mx(values: mx.array) -> mx.array:
    argument = float(np.pi) * values
    near_zero = mx.abs(argument) < 1e-7
    safe_argument = mx.where(near_zero, 1.0, argument)
    return mx.where(near_zero, 1.0, mx.sin(argument) / safe_argument)


def _assign_charges_cic_mx(
    positions: mx.array,
    charges: mx.array,
    cell_lengths: mx.array,
    mesh_shape: tuple[int, int, int],
) -> mx.array:
    mesh = mx.array(mesh_shape, dtype=mx.float32)
    scaled = (positions - mx.floor(positions / cell_lengths) * cell_lengths) / cell_lengths * mesh
    base = mx.floor(scaled).astype(mx.int32)
    fraction = scaled - base.astype(mx.float32)
    nx, ny, nz = mesh_shape
    grid = mx.zeros((nx * ny * nz,), dtype=mx.float32)
    for dx in (0, 1):
        wx = (1.0 - fraction[:, 0]) if dx == 0 else fraction[:, 0]
        ix = (base[:, 0] + dx) % nx
        for dy in (0, 1):
            wy = (1.0 - fraction[:, 1]) if dy == 0 else fraction[:, 1]
            iy = (base[:, 1] + dy) % ny
            for dz in (0, 1):
                wz = (1.0 - fraction[:, 2]) if dz == 0 else fraction[:, 2]
                iz = (base[:, 2] + dz) % nz
                flat_index = (ix * ny + iy) * nz + iz
                grid = grid.at[flat_index].add(charges * wx * wy * wz)
    return mx.reshape(grid, mesh_shape)


def _interpolate_cic_mx(
    positions: mx.array,
    grid: mx.array,
    cell_lengths: mx.array,
) -> mx.array:
    mesh_shape = grid.shape[:3]
    trailing_shape = grid.shape[3:]
    n_atoms = int(positions.shape[0])
    mesh = mx.array(mesh_shape, dtype=mx.float32)
    scaled = (positions - mx.floor(positions / cell_lengths) * cell_lengths) / cell_lengths * mesh
    base = mx.floor(scaled).astype(mx.int32)
    fraction = scaled - base.astype(mx.float32)
    values = mx.zeros((n_atoms, *trailing_shape), dtype=grid.dtype)
    nx, ny, nz = mesh_shape
    for dx in (0, 1):
        wx = (1.0 - fraction[:, 0]) if dx == 0 else fraction[:, 0]
        ix = (base[:, 0] + dx) % nx
        for dy in (0, 1):
            wy = (1.0 - fraction[:, 1]) if dy == 0 else fraction[:, 1]
            iy = (base[:, 1] + dy) % ny
            for dz in (0, 1):
                wz = (1.0 - fraction[:, 2]) if dz == 0 else fraction[:, 2]
                iz = (base[:, 2] + dz) % nz
                weight = wx * wy * wz
                corner_values = grid[ix, iy, iz]
                if trailing_shape:
                    weight = mx.reshape(weight, (n_atoms, *([1] * len(trailing_shape))))
                values = values + weight * corner_values
    return values

