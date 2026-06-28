"""Spin-unpolarized Γ-point plane-wave SCF driver."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft.density import density_from_orbitals
from mlx_atomistic.dft.fft import fft_backend
from mlx_atomistic.dft.grids import RealSpaceGrid, ReciprocalGrid
from mlx_atomistic.dft.mixing import LinearMixer, PulayDIISMixer
from mlx_atomistic.dft.nonlocal_pseudopotential import NonlocalPseudopotentialOperator
from mlx_atomistic.dft.operators import (
    DavidsonDiagonalizer,
    EigensolverConfig,
    KohnShamOperator,
    orbital_residuals,
    orthonormality_error,
)
from mlx_atomistic.dft.potentials import (
    LocalGaussianPseudopotential,
    apply_kinetic,
    hartree_potential,
    local_pseudopotential_forces,
)
from mlx_atomistic.dft.pseudopotentials import LocalPseudopotentialField
from mlx_atomistic.dft.system import DFTSystem, center_center_energy, center_center_forces
from mlx_atomistic.dft.xc import (
    DiracExchange,
    ExchangeCorrelationFunctional,
    LDAExchangeCorrelation,
)

SCFSolver = Literal["auto", "dense", "gradient", "davidson"]
SCFConvergenceMode = Literal["density", "energy", "either", "both"]
SCFMixer = Literal["linear", "diis"]


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
    mixer: SCFMixer | LinearMixer | PulayDIISMixer = "linear"
    convergence_mode: SCFConvergenceMode = "density"
    min_iterations: int = 1
    record_timing: bool = True
    potential_tolerance: float | None = None
    orbital_tolerance: float | None = None
    max_density_residual: float | None = 1e6
    max_orthonormality_error: float = 1e-3
    apply_nonlocal: bool = True
    eigensolver_config: EigensolverConfig = field(default_factory=EigensolverConfig)

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
        if self.solver not in {"auto", "dense", "gradient", "davidson"}:
            msg = "solver must be 'auto', 'dense', 'gradient', or 'davidson'"
            raise ValueError(msg)
        if self.max_dense_grid_points <= 0:
            msg = "max_dense_grid_points must be positive"
            raise ValueError(msg)
        if isinstance(self.mixer, str) and self.mixer not in {"linear", "diis"}:
            msg = "mixer must be 'linear', 'diis', or a mixer instance"
            raise ValueError(msg)
        if self.convergence_mode not in {"density", "energy", "either", "both"}:
            msg = "convergence_mode must be 'density', 'energy', 'either', or 'both'"
            raise ValueError(msg)
        if self.min_iterations <= 0:
            msg = "min_iterations must be positive"
            raise ValueError(msg)
        if self.potential_tolerance is not None and self.potential_tolerance <= 0.0:
            msg = "potential_tolerance must be positive when provided"
            raise ValueError(msg)
        if self.orbital_tolerance is not None and self.orbital_tolerance <= 0.0:
            msg = "orbital_tolerance must be positive when provided"
            raise ValueError(msg)
        if self.max_density_residual is not None and self.max_density_residual <= 0.0:
            msg = "max_density_residual must be positive when provided"
            raise ValueError(msg)
        if self.max_orthonormality_error <= 0.0:
            msg = "max_orthonormality_error must be positive"
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
    history: list[dict[str, float | int | str | None]]
    status: str
    convergence_reason: str
    failure_reason: str | None
    timings: dict[str, float]
    forces: mx.array | None = None
    mixer_metadata: dict | None = None
    orbital_eigenvalues: mx.array | None = None
    orbital_residuals: mx.array | None = None
    orthonormality_error: float = 0.0
    electronic_energy: float | None = None
    center_center_energy: float = 0.0
    force_consistency: dict | None = None
    pseudopotential_format: str | None = None
    ion_count: int | None = None
    valence_electron_count: float | None = None
    nonlocal_available: bool = False
    nonlocal_applied: bool = False
    nonlocal_projector_count: int = 0
    force_provenance: dict | None = None
    solver_metadata: dict | None = None

    def to_dict(self) -> dict:
        """Return a JSON-safe summary without dense array payloads.

        Returns:
            A JSON-serializable dict of the scalar results, energy terms, history, and
                timings (dense arrays such as the density are reduced to shapes/lists).
        """

        return {
            "converged": self.converged,
            "status": self.status,
            "convergence_reason": self.convergence_reason,
            "failure_reason": self.failure_reason,
            "iterations": self.iterations,
            "solver": self.solver,
            "fft_backend": self.fft_backend,
            "electron_count": self.electron_count,
            "total_energy": self.total_energy,
            "electronic_energy": self.electronic_energy,
            "center_center_energy": self.center_center_energy,
            "residual": self.residual,
            "energy_by_term": dict(self.energy_by_term),
            "history": list(self.history),
            "timings": dict(self.timings),
            "mixer": {} if self.mixer_metadata is None else dict(self.mixer_metadata),
            "grid_shape": list(self.density.shape),
            "forces": None if self.forces is None else np.array(self.forces).tolist(),
            "orbital_eigenvalues": (
                None
                if self.orbital_eigenvalues is None
                else np.array(self.orbital_eigenvalues).tolist()
            ),
            "orbital_residuals": (
                None
                if self.orbital_residuals is None
                else np.array(self.orbital_residuals).tolist()
            ),
            "orthonormality_error": self.orthonormality_error,
            "force_consistency": self.force_consistency,
            "pseudopotential_format": self.pseudopotential_format,
            "ion_count": self.ion_count,
            "valence_electron_count": self.valence_electron_count,
            "nonlocal_available": self.nonlocal_available,
            "nonlocal_applied": self.nonlocal_applied,
            "nonlocal_projector_count": self.nonlocal_projector_count,
            "force_provenance": self.force_provenance,
            "solver_metadata": {} if self.solver_metadata is None else dict(self.solver_metadata),
        }


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
    local_potential: LocalGaussianPseudopotential
    | LocalPseudopotentialField
    | mx.array
    | Sequence[float],
    grid: RealSpaceGrid,
) -> mx.array:
    if isinstance(local_potential, LocalGaussianPseudopotential | LocalPseudopotentialField):
        field = local_potential.field(grid)
    else:
        field = mx.array(local_potential)
    if field.shape != grid.shape:
        msg = "local_potential field must have shape grid.shape"
        raise ValueError(msg)
    return mx.real(field)


def _choose_solver(config: SCFConfig, grid: RealSpaceGrid) -> str:
    if config.solver == "auto":
        return "dense" if grid.size <= config.max_dense_grid_points else "davidson"
    if config.solver == "dense" and grid.size > config.max_dense_grid_points:
        msg = "dense SCF solver requested for grid larger than max_dense_grid_points"
        raise ValueError(msg)
    return config.solver


def _build_mixer(config: SCFConfig) -> LinearMixer | PulayDIISMixer:
    if isinstance(config.mixer, LinearMixer | PulayDIISMixer):
        config.mixer.reset()
        return config.mixer
    if config.mixer == "diis":
        return PulayDIISMixer(beta=config.mixing)
    return LinearMixer(beta=config.mixing)


def _dense_lowest_orbitals(
    potential: mx.array,
    grid: RealSpaceGrid,
    *,
    n_orbitals: int,
    nonlocal_operator: NonlocalPseudopotentialOperator | None = None,
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
        if nonlocal_operator is not None and nonlocal_operator.available:
            kinetic = kinetic + np.array(
                nonlocal_operator.apply(mx.array(basis_grid.astype(np.complex64))),
                dtype=np.complex128,
            ).reshape(-1)
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


def _field_residual(old_field: mx.array, new_field: mx.array, grid: RealSpaceGrid) -> float:
    delta = np.array(new_field - old_field, dtype=np.float64)
    return float(np.sqrt(np.sum(delta * delta) * grid.dv))


def _assert_finite(*fields: mx.array) -> None:
    for array in fields:
        if not bool(mx.all(mx.isfinite(array))):
            msg = "SCF produced NaN or infinite values"
            raise FloatingPointError(msg)


def _empty_timings() -> dict[str, float]:
    return {
        "hartree_ms": 0.0,
        "xc_ms": 0.0,
        "solver_ms": 0.0,
        "mixer_ms": 0.0,
        "kinetic_ms": 0.0,
        "energy_ms": 0.0,
        "force_ms": 0.0,
        "operator_ms": 0.0,
        "orthonormality_ms": 0.0,
        "diagonalization_ms": 0.0,
        "preconditioner_ms": 0.0,
        "nonlocal_ms": 0.0,
        "total_scf_ms": 0.0,
    }


def _add_timing(timings: dict[str, float], key: str, start: float, *, enabled: bool) -> None:
    if enabled:
        timings[key] += (perf_counter() - start) * 1000.0


def _kinetic_energy_timed(
    orbitals: mx.array,
    grid: RealSpaceGrid,
    *,
    occupations: Sequence[float],
) -> mx.array:
    stack = mx.array(orbitals)
    if stack.shape == grid.shape:
        stack = mx.reshape(stack, (1, *grid.shape))
    energy = mx.array(0.0, dtype=mx.float32)
    for index, occupation in enumerate(occupations):
        applied = apply_kinetic(stack[index], grid)
        expectation = mx.real(mx.sum(mx.conjugate(stack[index]) * applied))
        energy = energy + float(occupation) * expectation * grid.dv
    return energy


def _energy_terms(
    orbitals: mx.array,
    density: mx.array,
    v_local: mx.array,
    v_hartree: mx.array,
    xc_energy: mx.array,
    grid: RealSpaceGrid,
    *,
    occupations: Sequence[float],
    timings: dict[str, float],
    timing_enabled: bool,
    nonlocal_operator: NonlocalPseudopotentialOperator | None = None,
) -> dict[str, mx.array]:
    start = perf_counter()
    kinetic = _kinetic_energy_timed(orbitals, grid, occupations=occupations)
    _add_timing(timings, "kinetic_ms", start, enabled=timing_enabled)
    start = perf_counter()
    local = mx.sum(density * v_local) * grid.dv
    hartree = 0.5 * mx.sum(density * v_hartree) * grid.dv
    if nonlocal_operator is None or not nonlocal_operator.available:
        nonlocal_energy = mx.array(0.0, dtype=mx.float32)
    else:
        nonlocal_energy = nonlocal_operator.energy(orbitals, occupations=occupations)
    total = kinetic + local + nonlocal_energy + hartree + xc_energy
    _add_timing(timings, "energy_ms", start, enabled=timing_enabled)
    return {
        "kinetic": kinetic,
        "local": local,
        "nonlocal_pseudopotential": nonlocal_energy,
        "hartree": hartree,
        "xc": xc_energy,
        "total": total,
    }


def _add_xc_components(
    energy_terms: dict[str, mx.array],
    xc_functional: ExchangeCorrelationFunctional,
    density: mx.array,
    grid: RealSpaceGrid,
    *,
    density_floor: float,
) -> None:
    if isinstance(xc_functional, LDAExchangeCorrelation):
        energy_terms["exchange"] = xc_functional.exchange.evaluate(
            density,
            grid,
            density_floor=density_floor,
        ).total_energy
        energy_terms["correlation"] = xc_functional.correlation.evaluate(
            density,
            grid,
            density_floor=density_floor,
        ).total_energy
    elif isinstance(xc_functional, DiracExchange):
        energy_terms["exchange"] = energy_terms["xc"]


def _converged(
    *,
    iteration: int,
    config: SCFConfig,
    density_residual: float,
    potential_residual: float | None,
    energy_delta: float | None,
    orbital_residual: float | None,
) -> bool:
    if iteration < config.min_iterations:
        return False
    density_ok = density_residual <= config.tolerance
    energy_ok = energy_delta is not None and abs(energy_delta) <= config.tolerance
    potential_ok = True
    if config.potential_tolerance is not None:
        potential_ok = (
            potential_residual is not None and potential_residual <= config.potential_tolerance
        )
    orbital_ok = True
    if config.orbital_tolerance is not None:
        orbital_ok = orbital_residual is not None and orbital_residual <= config.orbital_tolerance
    if config.convergence_mode == "density":
        return density_ok and potential_ok and orbital_ok
    if config.convergence_mode == "energy":
        return energy_ok and potential_ok and orbital_ok
    if config.convergence_mode == "either":
        return (density_ok or energy_ok) and potential_ok and orbital_ok
    return density_ok and energy_ok and potential_ok and orbital_ok


def _resolve_inputs(
    system_or_grid: DFTSystem | RealSpaceGrid,
    local_potential: LocalGaussianPseudopotential
    | LocalPseudopotentialField
    | mx.array
    | Sequence[float]
    | None,
    electron_count: float | None,
) -> tuple[
    RealSpaceGrid,
    LocalGaussianPseudopotential | LocalPseudopotentialField | mx.array | Sequence[float],
    float,
    DFTSystem | None,
]:
    if isinstance(system_or_grid, DFTSystem):
        if local_potential is not None:
            msg = "local_potential must not be provided when running a DFTSystem"
            raise ValueError(msg)
        return (
            system_or_grid.grid,
            system_or_grid.pseudopotential,
            system_or_grid.electron_count if electron_count is None else electron_count,
            system_or_grid,
        )
    if local_potential is None:
        msg = "local_potential is required when running from a RealSpaceGrid"
        raise ValueError(msg)
    return system_or_grid, local_potential, 2.0 if electron_count is None else electron_count, None


def _initial_density_or_default(
    initial_density: mx.array | None,
    orbitals: mx.array,
    grid: RealSpaceGrid,
    *,
    occupations: Sequence[float],
    electron_count: float,
) -> mx.array:
    if initial_density is None:
        return density_from_orbitals(orbitals, grid, occupations=occupations)
    density = mx.real(mx.array(initial_density))
    if density.shape != grid.shape:
        msg = "initial_density must have shape grid.shape"
        raise ValueError(msg)
    if not bool(mx.all(density >= 0.0)):
        msg = "initial_density must be non-negative"
        raise ValueError(msg)
    count = float(mx.sum(density) * grid.dv)
    if count <= 0.0:
        msg = "initial_density must integrate to a positive electron count"
        raise ValueError(msg)
    return density * (electron_count / count)


def _nonlocal_force_correction(
    system: DFTSystem,
    grid: RealSpaceGrid,
    orbitals: mx.array,
    *,
    occupations: Sequence[float],
    displacement: float = 1e-3,
) -> mx.array:
    """Fixed-orbital finite-difference nonlocal force correction."""

    if system.ions is None:
        return mx.zeros((system.center_count, 3), dtype=mx.float32)
    centers = np.array(system.centers, dtype=np.float64)
    correction = np.zeros_like(centers)
    for center_index in range(system.center_count):
        for axis in range(3):
            plus = centers.copy()
            minus = centers.copy()
            plus[center_index, axis] += displacement
            minus[center_index, axis] -= displacement
            plus_operator = NonlocalPseudopotentialOperator.from_ions(
                system.ions.with_positions(plus),
                grid,
            )
            minus_operator = NonlocalPseudopotentialOperator.from_ions(
                system.ions.with_positions(minus),
                grid,
            )
            e_plus = float(plus_operator.energy(orbitals, occupations=occupations))
            e_minus = float(minus_operator.energy(orbitals, occupations=occupations))
            correction[center_index, axis] = -(e_plus - e_minus) / (2.0 * displacement)
    return mx.array(correction.astype(np.float32))


def _history_row(
    iteration: int,
    density_residual: float,
    potential_residual: float | None,
    energy_delta: float | None,
    energy_terms: dict[str, mx.array],
    density: mx.array,
    grid: RealSpaceGrid,
    center_energy: float,
    orbital_residual: float | None,
    orthonormality: float | None,
) -> dict[str, float | int | str | None]:
    electronic = float(energy_terms["total"])
    exchange = energy_terms.get("exchange")
    correlation = energy_terms.get("correlation")
    return {
        "iteration": iteration,
        "residual": density_residual,
        "density_residual": density_residual,
        "potential_residual": potential_residual,
        "energy_delta": energy_delta,
        "orbital_residual": orbital_residual,
        "orthonormality_error": orthonormality,
        "electron_count": float(mx.sum(density) * grid.dv),
        "kinetic": float(energy_terms["kinetic"]),
        "local": float(energy_terms["local"]),
        "local_pseudopotential_energy": float(energy_terms["local"]),
        "nonlocal_pseudopotential": float(
            energy_terms.get("nonlocal_pseudopotential", mx.array(0.0))
        ),
        "hartree": float(energy_terms["hartree"]),
        "xc": float(energy_terms["xc"]),
        "exchange": None if exchange is None else float(exchange),
        "correlation": None if correlation is None else float(correlation),
        "electronic": electronic,
        "center_center": center_energy,
        "total": electronic + center_energy,
    }


def run_scf(
    system_or_grid: DFTSystem | RealSpaceGrid,
    local_potential: LocalGaussianPseudopotential
    | LocalPseudopotentialField
    | mx.array
    | Sequence[float]
    | None = None,
    *,
    electron_count: float | None = None,
    n_orbitals: int | None = None,
    config: SCFConfig | None = None,
    initial_orbitals: mx.array | None = None,
    initial_density: mx.array | None = None,
    xc_functional: ExchangeCorrelationFunctional | None = None,
) -> SCFResult:
    """Run a minimal spin-unpolarized Γ-point SCF calculation.

    Args:
        system_or_grid: Either a `DFTSystem` (carrying grid, ions, and electron
            count) or a bare `RealSpaceGrid`.
        local_potential: External local potential — a Gaussian/field pseudopotential, a
            grid array, or ``None`` when supplied by ``system_or_grid``. Defaults to ``None``.
        electron_count: Total electron count; ``None`` takes it from the system.
            Defaults to ``None``.
        n_orbitals: Number of orbitals to solve; ``None`` derives it from the electron
            count. Defaults to ``None``.
        config: SCF controls (max iterations, tolerances, mixer, solver); ``None`` uses
            defaults. Defaults to ``None``.
        initial_orbitals: Optional starting orbitals; ``None`` uses a deterministic
            guess. Defaults to ``None``.
        initial_density: Optional starting density; ``None`` builds it from the initial
            orbitals. Defaults to ``None``.
        xc_functional: Exchange-correlation functional; ``None`` uses
            `LDAExchangeCorrelation`. Defaults to ``None``.

    Returns:
        An `SCFResult` with the converged density, orbitals, energy
            decomposition, and convergence/timing diagnostics.
    """

    config = SCFConfig() if config is None else config
    grid, local_input, electron_count_value, system = _resolve_inputs(
        system_or_grid,
        local_potential,
        electron_count,
    )
    xc_functional = LDAExchangeCorrelation() if xc_functional is None else xc_functional
    occupation_values = _occupations(electron_count_value, n_orbitals)
    n_occ_orbitals = len(occupation_values)
    solver = _choose_solver(config, grid)
    mixer = _build_mixer(config)
    timings = _empty_timings()
    total_start = perf_counter()

    v_local = _local_field(local_input, grid)
    orbitals = _initial_orbitals(
        grid,
        n_orbitals=n_occ_orbitals,
        seed=config.seed,
        initial_orbitals=initial_orbitals,
    )
    density = _initial_density_or_default(
        initial_density,
        orbitals,
        grid,
        occupations=occupation_values,
        electron_count=electron_count_value,
    )
    history: list[dict[str, float | int | str | None]] = []
    converged = False
    convergence_reason = "max_iterations"
    failure_reason: str | None = None
    density_residual = float("inf")
    potential_residual: float | None = None
    previous_potential: mx.array | None = None
    previous_energy: float | None = None
    effective_potential = v_local
    energy_terms: dict[str, mx.array] = {}
    center_energy = 0.0 if system is None else center_center_energy(system)
    orbital_eigenvalues = None
    orbital_residual_values = None
    final_orthonormality_error = 0.0
    max_orbital_residual: float | None = None
    pseudopotential_format = "array"
    ion_count = None
    valence_electron_count = None
    nonlocal_available = False
    if isinstance(local_input, LocalGaussianPseudopotential):
        pseudopotential_format = "gaussian"
    elif isinstance(local_input, LocalPseudopotentialField):
        formats = sorted(set(local_input.ions.formats))
        pseudopotential_format = ",".join(formats)
        ion_count = len(local_input.ions.ions)
        valence_electron_count = local_input.ions.valence_electron_count
        nonlocal_available = local_input.nonlocal_available
    nonlocal_operator = None
    nonlocal_projector_count = 0
    if (
        config.apply_nonlocal
        and isinstance(local_input, LocalPseudopotentialField)
        and local_input.nonlocal_available
    ):
        start = perf_counter()
        nonlocal_operator = NonlocalPseudopotentialOperator.from_ions(local_input.ions, grid)
        nonlocal_projector_count = nonlocal_operator.projectors.count
        _add_timing(timings, "nonlocal_ms", start, enabled=config.record_timing)
    nonlocal_applied = nonlocal_operator is not None and nonlocal_operator.available
    solver_metadata: dict = {"solver": solver}

    for iteration in range(1, config.max_iterations + 1):
        start = perf_counter()
        v_hartree = hartree_potential(density, grid)
        _add_timing(timings, "hartree_ms", start, enabled=config.record_timing)

        start = perf_counter()
        xc = xc_functional.evaluate(
            density,
            grid,
            density_floor=config.density_floor,
        )
        _add_timing(timings, "xc_ms", start, enabled=config.record_timing)

        effective_potential = v_local + v_hartree + xc.potential
        if previous_potential is not None:
            potential_residual = _field_residual(previous_potential, effective_potential, grid)
        previous_potential = effective_potential
        _assert_finite(effective_potential)

        start = perf_counter()
        if solver == "dense":
            next_orbitals = _dense_lowest_orbitals(
                effective_potential,
                grid,
                n_orbitals=n_occ_orbitals,
                nonlocal_operator=nonlocal_operator,
            )
        elif solver == "davidson":
            iteration_operator = KohnShamOperator.from_density(
                grid,
                v_local,
                density,
                xc_functional=xc_functional,
                density_floor=config.density_floor,
                nonlocal_operator=nonlocal_operator,
            )
            diagonalized = DavidsonDiagonalizer(config.eigensolver_config).solve(
                iteration_operator,
                n_orbitals=n_occ_orbitals,
                initial_orbitals=orbitals,
            )
            next_orbitals = diagonalized.orbitals
            solver_metadata = {
                **diagonalized.metadata,
                "eigensolver_converged": diagonalized.converged,
                "max_eigensolver_residual": float(mx.max(diagonalized.residuals)),
            }
        else:
            next_orbitals = _gradient_step_orbitals(
                orbitals,
                effective_potential,
                grid,
                step_size=config.step_size,
            )
        solver_elapsed_ms = (perf_counter() - start) * 1000.0
        if config.record_timing:
            timings["solver_ms"] += solver_elapsed_ms
            if solver == "dense":
                timings["diagonalization_ms"] += solver_elapsed_ms
            elif solver == "davidson":
                timings["preconditioner_ms"] += solver_elapsed_ms

        next_density = density_from_orbitals(
            next_orbitals,
            grid,
            occupations=occupation_values,
        )
        density_residual = _field_residual(density, next_density, grid)

        start = perf_counter()
        density = mixer.mix(density, next_density)
        _add_timing(timings, "mixer_ms", start, enabled=config.record_timing)
        orbitals = next_orbitals

        start = perf_counter()
        energy_v_hartree = hartree_potential(density, grid)
        _add_timing(timings, "hartree_ms", start, enabled=config.record_timing)
        start = perf_counter()
        energy_xc = xc_functional.evaluate(
            density,
            grid,
            density_floor=config.density_floor,
        )
        _add_timing(timings, "xc_ms", start, enabled=config.record_timing)
        effective_potential = v_local + energy_v_hartree + energy_xc.potential
        energy_terms = _energy_terms(
            orbitals,
            density,
            v_local,
            energy_v_hartree,
            energy_xc.total_energy,
            grid,
            occupations=occupation_values,
            timings=timings,
            timing_enabled=config.record_timing,
            nonlocal_operator=nonlocal_operator,
        )
        _add_xc_components(
            energy_terms,
            xc_functional,
            density,
            grid,
            density_floor=config.density_floor,
        )
        total_energy = float(energy_terms["total"])
        energy_delta = None if previous_energy is None else total_energy - previous_energy
        previous_energy = total_energy
        start = perf_counter()
        iteration_operator = KohnShamOperator.from_density(
            grid,
            v_local,
            density,
            xc_functional=xc_functional,
            density_floor=config.density_floor,
            nonlocal_operator=nonlocal_operator,
        )
        iteration_eigenvalues = iteration_operator.rayleigh_quotients(orbitals)
        iteration_residuals = orbital_residuals(orbitals, iteration_operator, iteration_eigenvalues)
        max_orbital_residual = float(mx.max(iteration_residuals))
        _add_timing(timings, "operator_ms", start, enabled=config.record_timing)
        start = perf_counter()
        iteration_orthonormality_error = orthonormality_error(orbitals, grid)
        _add_timing(timings, "orthonormality_ms", start, enabled=config.record_timing)
        _assert_finite(density, orbitals, effective_potential, *energy_terms.values())
        history.append(
            _history_row(
                iteration,
                density_residual,
                potential_residual,
                energy_delta,
                energy_terms,
                density,
                grid,
                center_energy,
                max_orbital_residual,
                iteration_orthonormality_error,
            )
        )
        if iteration_orthonormality_error > config.max_orthonormality_error:
            failure_reason = "orthonormality_loss"
            break
        if (
            config.max_density_residual is not None
            and density_residual > config.max_density_residual
        ):
            failure_reason = "diverged"
            break
        if _converged(
            iteration=iteration,
            config=config,
            density_residual=density_residual,
            potential_residual=potential_residual,
            energy_delta=energy_delta,
            orbital_residual=max_orbital_residual,
        ):
            converged = True
            convergence_reason = config.convergence_mode
            break

    density = density_from_orbitals(
        orbitals,
        grid,
        occupations=occupation_values,
    )
    start = perf_counter()
    final_v_hartree = hartree_potential(density, grid)
    _add_timing(timings, "hartree_ms", start, enabled=config.record_timing)
    start = perf_counter()
    final_xc = xc_functional.evaluate(
        density,
        grid,
        density_floor=config.density_floor,
    )
    _add_timing(timings, "xc_ms", start, enabled=config.record_timing)
    effective_potential = v_local + final_v_hartree + final_xc.potential
    energy_terms = _energy_terms(
        orbitals,
        density,
        v_local,
        final_v_hartree,
        final_xc.total_energy,
        grid,
        occupations=occupation_values,
        timings=timings,
        timing_enabled=config.record_timing,
        nonlocal_operator=nonlocal_operator,
    )
    _add_xc_components(
        energy_terms,
        xc_functional,
        density,
        grid,
        density_floor=config.density_floor,
    )
    _assert_finite(density, orbitals, effective_potential, *energy_terms.values())

    forces = None
    if isinstance(local_input, LocalGaussianPseudopotential | LocalPseudopotentialField):
        start = perf_counter()
        if isinstance(local_input, LocalGaussianPseudopotential):
            forces = local_pseudopotential_forces(density, grid, local_input)
            force_provenance = {
                "local_analytic": True,
                "nonlocal_finite_difference": False,
                "center_center": system is not None,
            }
        else:
            forces = local_input.forces(density, grid)
            if nonlocal_applied and system is not None:
                forces = forces + _nonlocal_force_correction(
                    system,
                    grid,
                    orbitals,
                    occupations=occupation_values,
                )
            force_provenance = {
                "local_analytic": True,
                "nonlocal_finite_difference": bool(nonlocal_applied),
                "center_center": system is not None,
            }
        if system is not None:
            forces = forces + mx.array(center_center_forces(system), dtype=forces.dtype)
        _add_timing(timings, "force_ms", start, enabled=config.record_timing)

    start = perf_counter()
    final_operator = KohnShamOperator.from_density(
        grid,
        v_local,
        density,
        xc_functional=xc_functional,
        density_floor=config.density_floor,
        nonlocal_operator=nonlocal_operator,
    )
    orbital_eigenvalues = final_operator.rayleigh_quotients(orbitals)
    orbital_residual_values = orbital_residuals(orbitals, final_operator, orbital_eigenvalues)
    _add_timing(timings, "operator_ms", start, enabled=config.record_timing)

    start = perf_counter()
    final_orthonormality_error = orthonormality_error(orbitals, grid)
    _add_timing(timings, "orthonormality_ms", start, enabled=config.record_timing)

    if config.record_timing:
        timings["total_scf_ms"] = (perf_counter() - total_start) * 1000.0
    if not history:
        failure_reason = "no_scf_iterations"
    elif not converged and failure_reason is None:
        failure_reason = "max_iterations_reached"

    energy_by_term = {name: float(value) for name, value in energy_terms.items()}
    electronic_energy = energy_by_term["total"]
    energy_by_term["electronic"] = electronic_energy
    energy_by_term["center_center"] = center_energy
    energy_by_term["total"] = electronic_energy + center_energy
    energy_by_term["local_pseudopotential"] = energy_by_term["local"]
    energy_by_term.setdefault("nonlocal_pseudopotential", 0.0)
    if converged:
        status = "converged"
    elif failure_reason == "max_iterations_reached":
        status = "max_iterations"
    else:
        status = "failed"
    return SCFResult(
        converged=converged,
        iterations=len(history),
        solver=solver,
        fft_backend=fft_backend(),
        electron_count=float(mx.sum(density) * grid.dv),
        total_energy=energy_by_term["total"],
        residual=density_residual,
        density=density,
        orbitals=orbitals,
        effective_potential=effective_potential,
        energy_by_term=energy_by_term,
        history=history,
        status=status,
        convergence_reason=convergence_reason,
        failure_reason=failure_reason,
        timings=timings,
        forces=forces,
        mixer_metadata=mixer.metadata(),
        orbital_eigenvalues=orbital_eigenvalues,
        orbital_residuals=orbital_residual_values,
        orthonormality_error=final_orthonormality_error,
        electronic_energy=electronic_energy,
        center_center_energy=center_energy,
        force_consistency=None,
        pseudopotential_format=pseudopotential_format,
        ion_count=ion_count,
        valence_electron_count=valence_electron_count,
        nonlocal_available=nonlocal_available,
        nonlocal_applied=nonlocal_applied,
        nonlocal_projector_count=nonlocal_projector_count,
        force_provenance=locals().get("force_provenance"),
        solver_metadata=solver_metadata,
    )
