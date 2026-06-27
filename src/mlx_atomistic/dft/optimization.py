"""Fixed-cell DFT geometry optimization workflows."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell
from mlx_atomistic.dft.scf import SCFConfig, SCFResult, run_scf
from mlx_atomistic.dft.system import DFTSystem
from mlx_atomistic.dft.xc import DiracExchange, ExchangeCorrelationFunctional

GeometryOptimizer = Literal["lbfgs", "steepest_descent"]
GeometryRelaxationMode = Literal["ions", "cell", "ions_cell"]
GeometryStatus = Literal[
    "converged",
    "max_steps",
    "line_search_failed",
    "scf_failed",
    "nonfinite",
]


@dataclass(frozen=True)
class GeometryOptimizationConfig:
    """Configuration for fixed-cell ion-position relaxation."""

    max_steps: int = 25
    force_tolerance: float = 1e-3
    energy_tolerance: float = 1e-6
    initial_step_size: float = 0.08
    max_step: float = 0.25
    line_search_shrink: float = 0.5
    line_search_min_step: float = 1e-4
    max_line_search_iterations: int = 8
    optimizer: GeometryOptimizer = "lbfgs"
    scf_config: SCFConfig | None = None
    reuse_scf_state: bool = True
    relaxation_mode: GeometryRelaxationMode = "ions"
    stress_tolerance: float = 1e-3
    cell_step_size: float = 0.02

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            msg = "max_steps must be positive"
            raise ValueError(msg)
        if self.force_tolerance <= 0.0:
            msg = "force_tolerance must be positive"
            raise ValueError(msg)
        if self.energy_tolerance <= 0.0:
            msg = "energy_tolerance must be positive"
            raise ValueError(msg)
        if self.initial_step_size <= 0.0:
            msg = "initial_step_size must be positive"
            raise ValueError(msg)
        if self.max_step <= 0.0:
            msg = "max_step must be positive"
            raise ValueError(msg)
        if not 0.0 < self.line_search_shrink < 1.0:
            msg = "line_search_shrink must be in the interval (0, 1)"
            raise ValueError(msg)
        if self.line_search_min_step <= 0.0:
            msg = "line_search_min_step must be positive"
            raise ValueError(msg)
        if self.max_line_search_iterations <= 0:
            msg = "max_line_search_iterations must be positive"
            raise ValueError(msg)
        if self.optimizer not in {"lbfgs", "steepest_descent"}:
            msg = "optimizer must be 'lbfgs' or 'steepest_descent'"
            raise ValueError(msg)
        if self.relaxation_mode not in {"ions", "cell", "ions_cell"}:
            msg = "relaxation_mode must be 'ions', 'cell', or 'ions_cell'"
            raise ValueError(msg)
        if self.stress_tolerance <= 0.0:
            msg = "stress_tolerance must be positive"
            raise ValueError(msg)
        if self.cell_step_size <= 0.0:
            msg = "cell_step_size must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class GeometryOptimizationStep:
    """One accepted geometry-optimization step."""

    index: int
    energy: float
    energy_delta: float
    max_force: float
    rms_force: float
    force_norm: float
    step_norm: float
    accepted_step_size: float
    line_search_iterations: int
    scf_status: str
    scf_iterations: int
    scf_residual: float
    electron_count: float
    timing_summary: dict[str, float]
    positions: np.ndarray
    forces: np.ndarray
    status: str = "accepted"
    stress_norm: float | None = None
    rejected_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe step summary."""

        return {
            "index": self.index,
            "status": self.status,
            "energy": self.energy,
            "energy_delta": self.energy_delta,
            "max_force": self.max_force,
            "rms_force": self.rms_force,
            "force_norm": self.force_norm,
            "step_norm": self.step_norm,
            "accepted_step_size": self.accepted_step_size,
            "line_search_iterations": self.line_search_iterations,
            "scf_status": self.scf_status,
            "scf_iterations": self.scf_iterations,
            "scf_residual": self.scf_residual,
            "electron_count": self.electron_count,
            "timing_summary": dict(self.timing_summary),
            "positions": self.positions.tolist(),
            "forces": self.forces.tolist(),
            "stress_norm": self.stress_norm,
            "rejected_reason": self.rejected_reason,
        }


@dataclass(frozen=True)
class GeometryOptimizationResult:
    """Result bundle for a fixed-cell DFT geometry optimization."""

    status: GeometryStatus
    convergence_reason: str
    initial_system: DFTSystem
    final_system: DFTSystem
    final_scf: SCFResult | None
    steps: tuple[GeometryOptimizationStep, ...]
    config: GeometryOptimizationConfig
    elapsed_ms: float

    @property
    def converged(self) -> bool:
        """Whether the geometry optimization met a convergence criterion."""

        return self.status == "converged"

    @property
    def final_energy(self) -> float | None:
        """Final total energy, if an SCF result is available."""

        return None if self.final_scf is None else self.final_scf.total_energy

    @property
    def final_positions(self) -> np.ndarray:
        """Final ion-center positions."""

        return np.array(self.final_system.centers, dtype=np.float64)

    @property
    def final_forces(self) -> np.ndarray | None:
        """Final ion-center forces, if available."""

        if self.final_scf is None or self.final_scf.forces is None:
            return None
        return np.array(self.final_scf.forces, dtype=np.float64)

    @property
    def final_max_force(self) -> float | None:
        """Final maximum per-center force norm."""

        forces = self.final_forces
        return None if forces is None else _max_force(forces)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary without dense SCF arrays."""

        final_forces = self.final_forces
        return {
            "status": self.status,
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
            "elapsed_ms": self.elapsed_ms,
            "step_count": len(self.steps),
            "final_energy": self.final_energy,
            "final_max_force": self.final_max_force,
            "final_positions": self.final_positions.tolist(),
            "final_forces": None if final_forces is None else final_forces.tolist(),
            "initial_system": _system_summary(self.initial_system),
            "final_system": _system_summary(self.final_system),
            "config": _config_summary(self.config),
            "steps": [step.to_dict() for step in self.steps],
            "history": [step.to_dict() for step in self.steps],
            "final_scf": None if self.final_scf is None else self.final_scf.to_dict(),
        }


@dataclass(frozen=True)
class GeometryOptimizationRecord:
    """Loaded NPZ geometry-optimization history."""

    positions: np.ndarray
    forces: np.ndarray
    energies: np.ndarray
    max_forces: np.ndarray
    statuses: tuple[str, ...]
    metadata: dict[str, Any]
    history: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary."""

        return {
            "positions": self.positions.tolist(),
            "forces": self.forces.tolist(),
            "energies": self.energies.tolist(),
            "max_forces": self.max_forces.tolist(),
            "statuses": list(self.statuses),
            "metadata": dict(self.metadata),
            "history": list(self.history),
        }


def optimize_geometry(
    system: DFTSystem,
    *,
    config: GeometryOptimizationConfig | None = None,
    xc_functional: ExchangeCorrelationFunctional | None = None,
) -> GeometryOptimizationResult:
    """Relax ion positions in a fixed orthorhombic cell using SCF forces."""

    config = GeometryOptimizationConfig() if config is None else config
    scf_config = _default_scf_config() if config.scf_config is None else config.scf_config
    xc_functional = DiracExchange() if xc_functional is None else xc_functional
    start = perf_counter()
    steps: list[GeometryOptimizationStep] = []

    current_system = system.with_centers(_wrapped_positions(system.centers, system.cell))
    try:
        current_scf = run_scf(current_system, config=scf_config, xc_functional=xc_functional)
    except (FloatingPointError, ValueError):
        return GeometryOptimizationResult(
            status="scf_failed",
            convergence_reason="initial_scf_failed",
            initial_system=system,
            final_system=current_system,
            final_scf=None,
            steps=tuple(steps),
            config=config,
            elapsed_ms=(perf_counter() - start) * 1000.0,
        )
    if not _scf_usable(current_scf):
        status: GeometryStatus = "nonfinite" if not _energy_finite(current_scf) else "scf_failed"
        return GeometryOptimizationResult(
            status=status,
            convergence_reason="initial_scf_unusable",
            initial_system=system,
            final_system=current_system,
            final_scf=current_scf,
            steps=tuple(steps),
            config=config,
            elapsed_ms=(perf_counter() - start) * 1000.0,
        )

    current_forces = _result_forces(current_scf)
    if _max_force(current_forces) <= config.force_tolerance:
        return GeometryOptimizationResult(
            status="converged",
            convergence_reason="force_tolerance",
            initial_system=system,
            final_system=current_system,
            final_scf=current_scf,
            steps=tuple(steps),
            config=config,
            elapsed_ms=(perf_counter() - start) * 1000.0,
        )

    previous_gradient: np.ndarray | None = None
    previous_positions: np.ndarray | None = None
    s_history: list[np.ndarray] = []
    y_history: list[np.ndarray] = []

    status: GeometryStatus = "max_steps"
    convergence_reason = "max_steps"
    for step_index in range(1, config.max_steps + 1):
        if config.reuse_scf_state:
            try:
                refreshed_scf = run_scf(
                    current_system,
                    config=scf_config,
                    initial_density=current_scf.density,
                    initial_orbitals=current_scf.orbitals,
                    xc_functional=xc_functional,
                )
            except (FloatingPointError, ValueError):
                status = "scf_failed"
                convergence_reason = "scf_continuation_failed"
                break
            if not _scf_usable(refreshed_scf):
                status = "nonfinite" if not _energy_finite(refreshed_scf) else "scf_failed"
                convergence_reason = "scf_continuation_unusable"
                current_scf = refreshed_scf
                break
            current_scf = refreshed_scf
            current_forces = _result_forces(current_scf)
            if _max_force(current_forces) <= config.force_tolerance:
                status = "converged"
                convergence_reason = "force_tolerance"
                break

        positions = np.array(current_system.centers, dtype=np.float64)
        gradient = -current_forces
        direction = _search_direction(
            gradient,
            current_forces,
            optimizer=config.optimizer,
            s_history=s_history,
            y_history=y_history,
        )
        direction = _clip_direction(direction, config.max_step)
        if not np.isfinite(direction).all() or np.linalg.norm(direction) <= 1e-14:
            direction = _clip_direction(current_forces, config.max_step)
        if not np.isfinite(direction).all() or np.linalg.norm(direction) <= 1e-14:
            status = "nonfinite"
            convergence_reason = "invalid_search_direction"
            break

        candidate = _backtracking_line_search(
            current_system=current_system,
            current_scf=current_scf,
            direction=direction,
            config=config,
            scf_config=scf_config,
            xc_functional=xc_functional,
        )
        if candidate is None:
            status = "line_search_failed"
            convergence_reason = "line_search_exhausted"
            break

        next_system, next_scf, accepted_step_size, line_search_iterations = candidate
        next_forces = _result_forces(next_scf)
        step_positions = np.array(next_system.centers, dtype=np.float64)
        displacement = _minimum_image_delta(step_positions - positions, current_system.cell)
        energy_delta = next_scf.total_energy - current_scf.total_energy
        step = GeometryOptimizationStep(
            index=step_index,
            energy=next_scf.total_energy,
            energy_delta=energy_delta,
            max_force=_max_force(next_forces),
            rms_force=_rms_force(next_forces),
            force_norm=float(np.linalg.norm(next_forces)),
            step_norm=_max_displacement(displacement),
            accepted_step_size=accepted_step_size,
            line_search_iterations=line_search_iterations,
            scf_status=next_scf.status,
            scf_iterations=next_scf.iterations,
            scf_residual=next_scf.residual,
            electron_count=next_scf.electron_count,
            timing_summary=dict(next_scf.timings),
            positions=step_positions,
            forces=next_forces,
        )
        steps.append(step)

        if previous_positions is not None and previous_gradient is not None:
            s_vector = _flatten_minimum_image(positions - previous_positions, current_system.cell)
            y_vector = (gradient - previous_gradient).reshape(-1)
            if _valid_lbfgs_pair(s_vector, y_vector):
                s_history.append(s_vector)
                y_history.append(y_vector)
                del s_history[:-5]
                del y_history[:-5]
        previous_positions = positions
        previous_gradient = gradient

        current_system = next_system
        current_scf = next_scf
        current_forces = next_forces
        if step.max_force <= config.force_tolerance:
            status = "converged"
            convergence_reason = "force_tolerance"
            break
        if abs(step.energy_delta) <= config.energy_tolerance:
            status = "converged"
            convergence_reason = "energy_tolerance"
            break

    return GeometryOptimizationResult(
        status=status,
        convergence_reason=convergence_reason,
        initial_system=system,
        final_system=current_system,
        final_scf=current_scf,
        steps=tuple(steps),
        config=config,
        elapsed_ms=(perf_counter() - start) * 1000.0,
    )


def save_geometry_optimization(
    path: str | Path,
    result: GeometryOptimizationResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a geometry-optimization history to compressed NPZ plus JSON metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = [step.to_dict() for step in result.steps]
    merged_metadata = {
        "format": "mlx-atomistic.dft.geometry-optimization.v1",
        "result": {
            "status": result.status,
            "convergence_reason": result.convergence_reason,
            "final_energy": result.final_energy,
            "final_max_force": result.final_max_force,
            "step_count": len(result.steps),
        },
        "user": {} if metadata is None else dict(metadata),
    }
    positions = np.asarray([step.positions for step in result.steps], dtype=np.float64)
    forces = np.asarray([step.forces for step in result.steps], dtype=np.float64)
    energies = np.asarray([step.energy for step in result.steps], dtype=np.float64)
    max_forces = np.asarray([step.max_force for step in result.steps], dtype=np.float64)
    statuses = np.asarray([step.status for step in result.steps], dtype="U32")
    if not result.steps:
        centers = np.array(result.final_system.centers, dtype=np.float64)
        positions = np.empty((0, *centers.shape), dtype=np.float64)
        forces = np.empty((0, *centers.shape), dtype=np.float64)
    np.savez_compressed(
        path,
        positions=positions,
        forces=forces,
        energies=energies,
        max_forces=max_forces,
        statuses=statuses,
        metadata_json=np.asarray(json.dumps(merged_metadata)),
        history_json=np.asarray(json.dumps(history)),
    )


def load_geometry_optimization(path: str | Path) -> GeometryOptimizationRecord:
    """Load a compressed geometry-optimization history."""

    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        history = json.loads(str(data["history_json"]))
        return GeometryOptimizationRecord(
            positions=np.array(data["positions"], dtype=np.float64),
            forces=np.array(data["forces"], dtype=np.float64),
            energies=np.array(data["energies"], dtype=np.float64),
            max_forces=np.array(data["max_forces"], dtype=np.float64),
            statuses=tuple(str(item) for item in data["statuses"]),
            metadata=metadata,
            history=history,
        )


def geometry_demo_system(
    name: str,
    *,
    grid_shape: Sequence[int] = (4, 4, 4),
) -> DFTSystem:
    """Build a compact built-in DFT geometry-optimization demo system."""

    shape = tuple(int(item) for item in grid_shape)
    if name == "gaussian-dimer":
        return DFTSystem.two_center(
            cell=(8.0, 8.0, 8.0),
            grid_shape=shape,
            centers=((3.15, 4.0, 4.0), (5.05, 4.0, 4.0)),
            electron_count=2.0,
            amplitudes=(-1.4, -1.4),
            widths=(0.85, 0.85),
            charges=(0.6, 0.6),
        )
    msg = "unknown demo system; expected gaussian-dimer"
    raise ValueError(msg)


def _default_scf_config() -> SCFConfig:
    return SCFConfig(
        max_iterations=20,
        tolerance=1e-8,
        mixing=0.45,
        solver="dense",
        seed=23,
        convergence_mode="either",
        record_timing=True,
    )


def _backtracking_line_search(
    *,
    current_system: DFTSystem,
    current_scf: SCFResult,
    direction: np.ndarray,
    config: GeometryOptimizationConfig,
    scf_config: SCFConfig,
    xc_functional: ExchangeCorrelationFunctional,
) -> tuple[DFTSystem, SCFResult, float, int] | None:
    alpha = config.initial_step_size
    initial_density = current_scf.density if config.reuse_scf_state else None
    initial_orbitals = current_scf.orbitals if config.reuse_scf_state else None
    current_energy = current_scf.total_energy
    positions = np.array(current_system.centers, dtype=np.float64)
    for iteration in range(1, config.max_line_search_iterations + 1):
        if alpha < config.line_search_min_step:
            break
        next_positions = _wrapped_positions(positions + alpha * direction, current_system.cell)
        next_system = current_system.with_centers(next_positions)
        try:
            next_scf = run_scf(
                next_system,
                config=scf_config,
                initial_density=initial_density,
                initial_orbitals=initial_orbitals,
                xc_functional=xc_functional,
            )
        except (FloatingPointError, ValueError):
            alpha *= config.line_search_shrink
            continue
        if _scf_usable(next_scf) and next_scf.total_energy <= current_energy + 1e-10:
            return next_system, next_scf, alpha, iteration
        alpha *= config.line_search_shrink
    return None


def _search_direction(
    gradient: np.ndarray,
    forces: np.ndarray,
    *,
    optimizer: GeometryOptimizer,
    s_history: Sequence[np.ndarray],
    y_history: Sequence[np.ndarray],
) -> np.ndarray:
    if optimizer == "steepest_descent" or not s_history:
        return forces.copy()
    direction = _lbfgs_direction(gradient.reshape(-1), s_history, y_history).reshape(
        forces.shape
    )
    if not np.isfinite(direction).all():
        return forces.copy()
    if float(np.sum(direction * forces)) <= 0.0:
        return forces.copy()
    return direction


def _lbfgs_direction(
    gradient: np.ndarray,
    s_history: Sequence[np.ndarray],
    y_history: Sequence[np.ndarray],
) -> np.ndarray:
    q = gradient.copy()
    alphas: list[float] = []
    rhos: list[float] = []
    for s_vector, y_vector in zip(reversed(s_history), reversed(y_history), strict=True):
        rho = 1.0 / float(np.dot(y_vector, s_vector))
        alpha = rho * float(np.dot(s_vector, q))
        q = q - alpha * y_vector
        alphas.append(alpha)
        rhos.append(rho)
    if s_history:
        s_last = s_history[-1]
        y_last = y_history[-1]
        scale = float(np.dot(s_last, y_last) / max(np.dot(y_last, y_last), 1e-20))
        r = scale * q
    else:
        r = q
    for s_vector, y_vector, alpha, rho in zip(
        s_history,
        y_history,
        reversed(alphas),
        reversed(rhos),
        strict=True,
    ):
        beta = rho * float(np.dot(y_vector, r))
        r = r + s_vector * (alpha - beta)
    return -r


def _valid_lbfgs_pair(s_vector: np.ndarray, y_vector: np.ndarray) -> bool:
    curvature = float(np.dot(s_vector, y_vector))
    return bool(np.isfinite(curvature) and curvature > 1e-12)


def _clip_direction(direction: np.ndarray, max_step: float) -> np.ndarray:
    clipped = np.array(direction, dtype=np.float64, copy=True)
    max_norm = _max_displacement(clipped)
    if max_norm > max_step:
        clipped *= max_step / max_norm
    return clipped


def _result_forces(result: SCFResult) -> np.ndarray:
    if result.forces is None:
        msg = "SCF result did not include forces"
        raise ValueError(msg)
    forces = np.array(result.forces, dtype=np.float64)
    if forces.ndim != 2 or forces.shape[1] != 3:
        msg = "SCF forces must have shape (n_centers, 3)"
        raise ValueError(msg)
    return forces


def _scf_usable(result: SCFResult) -> bool:
    return result.status != "failed" and _energy_finite(result) and result.forces is not None


def _energy_finite(result: SCFResult) -> bool:
    return bool(np.isfinite(result.total_energy) and np.isfinite(result.residual))


def _max_force(forces: np.ndarray) -> float:
    return _max_displacement(forces)


def _rms_force(forces: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(forces, dtype=np.float64) ** 2)))


def _max_displacement(displacement: np.ndarray) -> float:
    values = np.asarray(displacement, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(values.reshape((-1, 3)), axis=1)))


def _wrapped_positions(positions: Sequence[Sequence[float]] | mx.array, cell: Cell) -> np.ndarray:
    values = np.array(positions, dtype=np.float64)
    lengths = np.array(cell.lengths, dtype=np.float64)
    return np.mod(values, lengths)


def _minimum_image_delta(displacement: np.ndarray, cell: Cell) -> np.ndarray:
    values = np.asarray(displacement, dtype=np.float64)
    lengths = np.array(cell.lengths, dtype=np.float64)
    return values - lengths * np.round(values / lengths)


def _flatten_minimum_image(displacement: np.ndarray, cell: Cell) -> np.ndarray:
    return _minimum_image_delta(displacement, cell).reshape(-1)


def _system_summary(system: DFTSystem) -> dict[str, Any]:
    return {
        "grid_shape": list(system.grid_shape),
        "cell_lengths": np.array(system.cell.lengths, dtype=np.float64).tolist(),
        "center_count": system.center_count,
        "electron_count": system.electron_count,
        "charges": list(system.charges),
        "positions": np.array(system.centers, dtype=np.float64).tolist(),
        "pseudopotential_format": _pseudopotential_format(system),
        "nonlocal_available": False
        if system.ions is None
        else system.ions.nonlocal_available,
    }


def _pseudopotential_format(system: DFTSystem) -> str:
    if system.ions is None:
        return "gaussian"
    return ",".join(sorted(set(system.ions.formats)))


def _config_summary(config: GeometryOptimizationConfig) -> dict[str, Any]:
    return {
        "max_steps": config.max_steps,
        "force_tolerance": config.force_tolerance,
        "energy_tolerance": config.energy_tolerance,
        "initial_step_size": config.initial_step_size,
        "max_step": config.max_step,
        "line_search_shrink": config.line_search_shrink,
        "line_search_min_step": config.line_search_min_step,
        "max_line_search_iterations": config.max_line_search_iterations,
        "optimizer": config.optimizer,
        "reuse_scf_state": config.reuse_scf_state,
        "relaxation_mode": config.relaxation_mode,
        "stress_tolerance": config.stress_tolerance,
        "cell_step_size": config.cell_step_size,
        "scf_config": None if config.scf_config is None else _scf_config_summary(config.scf_config),
    }


def _scf_config_summary(config: SCFConfig) -> dict[str, Any]:
    return {
        "max_iterations": config.max_iterations,
        "tolerance": config.tolerance,
        "mixing": config.mixing,
        "step_size": config.step_size,
        "solver": config.solver,
        "max_dense_grid_points": config.max_dense_grid_points,
        "seed": config.seed,
        "density_floor": config.density_floor,
        "mixer": config.mixer if isinstance(config.mixer, str) else config.mixer.metadata(),
        "convergence_mode": config.convergence_mode,
        "min_iterations": config.min_iterations,
        "record_timing": config.record_timing,
        "potential_tolerance": config.potential_tolerance,
        "orbital_tolerance": config.orbital_tolerance,
        "max_density_residual": config.max_density_residual,
        "max_orthonormality_error": config.max_orthonormality_error,
        "apply_nonlocal": config.apply_nonlocal,
        "eigensolver_config": {
            "max_iterations": config.eigensolver_config.max_iterations,
            "tolerance": config.eigensolver_config.tolerance,
            "max_subspace_size": config.eigensolver_config.max_subspace_size,
            "dense_fallback_size": config.eigensolver_config.dense_fallback_size,
        },
    }
