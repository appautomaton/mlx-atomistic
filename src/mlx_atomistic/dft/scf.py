"""Minimal spin-unpolarized Γ-point plane-wave SCF driver."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft.density import density_from_orbitals
from mlx_atomistic.dft.fft import fft_backend
from mlx_atomistic.dft.grids import RealSpaceGrid, ReciprocalGrid
from mlx_atomistic.dft.potentials import (
    LocalGaussianPseudopotential,
    energy_decomposition,
    hartree_potential,
    lda_exchange_energy_potential,
)

SCFSolver = Literal["auto", "dense", "gradient"]


@dataclass(frozen=True)
class SCFConfig:
    """Configuration for the toy SCF driver."""

    max_iterations: int = 25
    tolerance: float = 1e-5
    mixing: float = 0.35
    step_size: float = 0.15
    solver: SCFSolver = "auto"
    max_dense_grid_points: int = 512
    seed: int = 0
    density_floor: float = 1e-12

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            msg = "max_iterations must be positive"
            raise ValueError(msg)
        if self.tolerance <= 0.0:
            msg = "tolerance must be positive"
            raise ValueError(msg)
        if not 0.0 < self.mixing <= 1.0:
            msg = "mixing must be in the interval (0, 1]"
            raise ValueError(msg)
        if self.step_size <= 0.0:
            msg = "step_size must be positive"
            raise ValueError(msg)
        if self.solver not in {"auto", "dense", "gradient"}:
            msg = "solver must be 'auto', 'dense', or 'gradient'"
            raise ValueError(msg)
        if self.max_dense_grid_points <= 0:
            msg = "max_dense_grid_points must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class SCFResult:
    """Result bundle for a toy DFT SCF calculation."""

    converged: bool
    iterations: int
    solver: str
    fft_backend: str
    electron_count: float
    total_energy: float
    residual: float
    density: mx.array
    orbitals: mx.array
    effective_potential: mx.array
    energy_by_term: dict[str, float]
    history: list[dict[str, float | int]]

    def to_dict(self) -> dict:
        """Return a JSON-safe summary without array payloads."""

        payload = asdict(self)
        payload.pop("density")
        payload.pop("orbitals")
        payload.pop("effective_potential")
        payload["grid_shape"] = list(self.density.shape)
        return payload


def _occupations(electron_count: float, n_orbitals: int | None = None) -> list[float]:
    if electron_count <= 0.0:
        msg = "electron_count must be positive"
        raise ValueError(msg)
    if n_orbitals is None:
        n_orbitals = int(np.ceil(electron_count / 2.0))
    if n_orbitals <= 0:
        msg = "n_orbitals must be positive"
        raise ValueError(msg)
    if electron_count > 2.0 * n_orbitals:
        msg = "spin-unpolarized occupations cannot exceed 2 electrons per orbital"
        raise ValueError(msg)
    remaining = float(electron_count)
    occupations: list[float] = []
    for _ in range(n_orbitals):
        occupation = min(2.0, remaining)
        occupations.append(occupation)
        remaining -= occupation
    return occupations


def _orthonormalize_numpy(orbitals: np.ndarray, grid: RealSpaceGrid) -> np.ndarray:
    stack = np.asarray(orbitals, dtype=np.complex64)
    if stack.shape == grid.shape:
        stack = stack.reshape((1, *grid.shape))
    n_orbitals = stack.shape[0]
    flat = stack.reshape(n_orbitals, grid.size)
    weighted = flat.T * np.sqrt(grid.dv)
    q, _ = np.linalg.qr(weighted)
    normalized = (q[:, :n_orbitals].T / np.sqrt(grid.dv)).reshape((n_orbitals, *grid.shape))
    return normalized.astype(np.complex64)


def _initial_orbitals(
    grid: RealSpaceGrid,
    *,
    n_orbitals: int,
    seed: int,
    initial_orbitals: mx.array | None,
) -> mx.array:
    if initial_orbitals is not None:
        orbitals_np = np.array(initial_orbitals, dtype=np.complex64)
    else:
        rng = np.random.default_rng(seed)
        coordinates = np.array(grid.coordinates(), dtype=np.float32)
        center = np.array(grid.lengths, dtype=np.float32) / 2.0
        r2 = np.sum((coordinates - center) ** 2, axis=-1)
        base = np.exp(-r2 / max(float(np.min(np.array(grid.lengths))) ** 2 / 16.0, 1e-6))
        orbitals = []
        for index in range(n_orbitals):
            noise = 0.05 * rng.normal(size=grid.shape)
            phase = rng.uniform(-np.pi, np.pi, size=grid.shape)
            orbital = base * (1.0 + noise) * np.exp(1j * 0.02 * (index + 1) * phase)
            orbitals.append(orbital)
        orbitals_np = np.stack(orbitals).astype(np.complex64)
    if orbitals_np.shape == grid.shape:
        orbitals_np = orbitals_np.reshape((1, *grid.shape))
    if orbitals_np.shape[1:] != grid.shape:
        msg = "initial_orbitals must have shape grid.shape or (n_orbitals, *grid.shape)"
        raise ValueError(msg)
    if orbitals_np.shape[0] != n_orbitals:
        msg = "initial_orbitals count must match n_orbitals"
        raise ValueError(msg)
    return mx.array(_orthonormalize_numpy(orbitals_np, grid))


def _local_field(
    local_potential: LocalGaussianPseudopotential | mx.array | Sequence[float],
    grid: RealSpaceGrid,
) -> mx.array:
    if isinstance(local_potential, LocalGaussianPseudopotential):
        field = local_potential.field(grid)
    else:
        field = mx.array(local_potential)
    if field.shape != grid.shape:
        msg = "local_potential field must have shape grid.shape"
        raise ValueError(msg)
    return mx.real(field)


def _choose_solver(config: SCFConfig, grid: RealSpaceGrid) -> str:
    if config.solver == "auto":
        return "dense" if grid.size <= config.max_dense_grid_points else "gradient"
    if config.solver == "dense" and grid.size > config.max_dense_grid_points:
        msg = "dense SCF solver requested for grid larger than max_dense_grid_points"
        raise ValueError(msg)
    return config.solver


def _dense_lowest_orbitals(
    potential: mx.array,
    grid: RealSpaceGrid,
    *,
    n_orbitals: int,
) -> mx.array:
    v_np = np.array(potential, dtype=np.float64).reshape(-1)
    reciprocal = ReciprocalGrid.from_real_space(grid)
    g2 = np.array(reciprocal.g2, dtype=np.float64)
    columns = []
    for index in range(grid.size):
        basis = np.zeros(grid.size, dtype=np.complex128)
        basis[index] = 1.0
        basis_grid = basis.reshape(grid.shape)
        kinetic = np.fft.ifftn(0.5 * g2 * np.fft.fftn(basis_grid)).reshape(-1)
        columns.append(kinetic)
    hamiltonian = np.column_stack(columns)
    hamiltonian += np.diag(v_np)
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conjugate().T)
    _, vectors = np.linalg.eigh(hamiltonian)
    selected = vectors[:, :n_orbitals].T.reshape((n_orbitals, *grid.shape))
    return mx.array(_orthonormalize_numpy(selected, grid))


def _gradient_step_orbitals(
    orbitals: mx.array,
    potential: mx.array,
    grid: RealSpaceGrid,
    *,
    step_size: float,
) -> mx.array:
    psi_np = np.array(orbitals, dtype=np.complex64)
    potential_np = np.array(potential, dtype=np.float32)
    reciprocal = ReciprocalGrid.from_real_space(grid)
    g2 = np.array(reciprocal.g2, dtype=np.float32)
    spectral_radius = float(np.max(0.5 * g2) + np.max(np.abs(potential_np)))
    dt = min(step_size, 0.2 / max(spectral_radius, 1e-6))
    real_damp = np.exp(-0.5 * dt * potential_np)
    kinetic_damp = np.exp(-dt * 0.5 * g2)
    updated = []
    for orbital in psi_np:
        propagated = real_damp * orbital
        propagated = np.fft.ifftn(kinetic_damp * np.fft.fftn(propagated))
        propagated = real_damp * propagated
        updated.append(propagated)
    return mx.array(_orthonormalize_numpy(np.stack(updated), grid))


def _density_residual(old_density: mx.array, new_density: mx.array, grid: RealSpaceGrid) -> float:
    delta = np.array(new_density - old_density, dtype=np.float64)
    return float(np.sqrt(np.sum(delta * delta) * grid.dv))


def _assert_finite(*fields: mx.array) -> None:
    for field in fields:
        if not bool(mx.all(mx.isfinite(field))):
            msg = "SCF produced NaN or infinite values"
            raise FloatingPointError(msg)


def _history_row(
    iteration: int,
    residual: float,
    energy_terms: dict[str, mx.array],
    density: mx.array,
    grid: RealSpaceGrid,
) -> dict[str, float | int]:
    return {
        "iteration": iteration,
        "residual": residual,
        "electron_count": float(mx.sum(density) * grid.dv),
        "kinetic": float(energy_terms["kinetic"]),
        "local": float(energy_terms["local"]),
        "hartree": float(energy_terms["hartree"]),
        "exchange": float(energy_terms["exchange"]),
        "total": float(energy_terms["total"]),
    }


def run_scf(
    grid: RealSpaceGrid,
    local_potential: LocalGaussianPseudopotential | mx.array | Sequence[float],
    *,
    electron_count: float = 2.0,
    n_orbitals: int | None = None,
    config: SCFConfig | None = None,
    initial_orbitals: mx.array | None = None,
) -> SCFResult:
    """Run a minimal spin-unpolarized Γ-point SCF calculation."""

    config = SCFConfig() if config is None else config
    occupation_values = _occupations(electron_count, n_orbitals)
    n_occ_orbitals = len(occupation_values)
    solver = _choose_solver(config, grid)
    v_local = _local_field(local_potential, grid)
    orbitals = _initial_orbitals(
        grid,
        n_orbitals=n_occ_orbitals,
        seed=config.seed,
        initial_orbitals=initial_orbitals,
    )
    density = density_from_orbitals(orbitals, grid, occupations=occupation_values)
    history: list[dict[str, float | int]] = []
    converged = False
    residual = float("inf")
    effective_potential = v_local
    energy_terms = energy_decomposition(
        orbitals,
        density,
        v_local,
        grid,
        occupations=occupation_values,
        density_floor=config.density_floor,
    )

    for iteration in range(1, config.max_iterations + 1):
        v_hartree = hartree_potential(density, grid)
        _, v_exchange = lda_exchange_energy_potential(
            density,
            grid,
            density_floor=config.density_floor,
        )
        effective_potential = v_local + v_hartree + v_exchange
        _assert_finite(effective_potential)

        if solver == "dense":
            next_orbitals = _dense_lowest_orbitals(
                effective_potential,
                grid,
                n_orbitals=n_occ_orbitals,
            )
        else:
            next_orbitals = _gradient_step_orbitals(
                orbitals,
                effective_potential,
                grid,
                step_size=config.step_size,
            )
        next_density = density_from_orbitals(
            next_orbitals,
            grid,
            occupations=occupation_values,
        )
        residual = _density_residual(density, next_density, grid)
        density = (1.0 - config.mixing) * density + config.mixing * next_density
        orbitals = next_orbitals
        energy_terms = energy_decomposition(
            orbitals,
            density,
            v_local,
            grid,
            occupations=occupation_values,
            density_floor=config.density_floor,
        )
        _assert_finite(density, orbitals, effective_potential, *energy_terms.values())
        history.append(_history_row(iteration, residual, energy_terms, density, grid))
        if residual <= config.tolerance:
            converged = True
            break

    energy_by_term = {name: float(value) for name, value in energy_terms.items()}
    return SCFResult(
        converged=converged,
        iterations=len(history),
        solver=solver,
        fft_backend=fft_backend(),
        electron_count=float(mx.sum(density) * grid.dv),
        total_energy=energy_by_term["total"],
        residual=residual,
        density=density,
        orbitals=orbitals,
        effective_potential=effective_potential,
        energy_by_term=energy_by_term,
        history=history,
    )
