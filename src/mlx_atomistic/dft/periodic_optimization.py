"""Fixed-cell ionic relaxation for the periodic plane-wave DFT runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft._geometry_math import (
    _clip_direction,
    _max_vector_norm,
    _rms_components,
    _search_direction,
    _valid_lbfgs_pair,
)
from mlx_atomistic.dft._periodic_artifact_contracts import (
    periodic_scf_calculation_contract,
)
from mlx_atomistic.dft._periodic_models import (
    PeriodicDFTSystem,
    PeriodicSCFConfig,
    PeriodicSCFResult,
)
from mlx_atomistic.dft._periodic_optimization_checkpoint import (
    _load_periodic_geometry_checkpoint,
    _publish_periodic_geometry_checkpoint,
)
from mlx_atomistic.dft._periodic_optimization_state import (
    _PeriodicGeometryContinuationState,
)
from mlx_atomistic.dft._runtime_observer import RuntimeObserver
from mlx_atomistic.dft.kpoints import KPointMesh
from mlx_atomistic.dft.periodic_forces import PeriodicForceResult, periodic_scf_forces
from mlx_atomistic.dft.periodic_scf import (
    _run_periodic_scf_fixed_topology,
    run_periodic_scf,
)
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional

PeriodicGeometryOptimizer = Literal["lbfgs", "steepest_descent"]
PeriodicGeometryStatus = Literal[
    "converged",
    "max_steps",
    "line_search_failed",
    "scf_failed",
    "nonfinite",
    "checkpointed",
]


@dataclass(frozen=True)
class PeriodicGeometryOptimizationConfig:
    """Controls for periodic fixed-cell ionic relaxation.

    Args:
        max_steps: Maximum accepted ionic steps.
        force_tolerance: Maximum per-ion force norm in Hartree/bohr.
        rms_force_tolerance: RMS Cartesian force in Hartree/bohr.
        displacement_tolerance: Maximum final per-ion displacement in bohr.
        initial_step_size: Initial inverse-Hessian scale in bohr squared/Hartree.
        max_step: Maximum per-ion trial displacement in bohr.
        line_search_shrink: Backtracking scale factor.
        line_search_min_step: Smallest inverse-Hessian scale to try.
        max_line_search_iterations: Maximum SCF trials per ionic step.
        armijo_constant: Sufficient-decrease coefficient.
        history_size: Maximum retained L-BFGS curvature pairs.
        optimizer: `lbfgs` or `steepest_descent`.
        reuse_scf_state: Reuse accepted density and compact eigenspaces.
        relaxation_mode: Must remain `ions`; cell modes fail closed.
    """

    max_steps: int = 25
    force_tolerance: float = 5.0e-4
    rms_force_tolerance: float = 3.0e-4
    displacement_tolerance: float = 3.0e-3
    initial_step_size: float = 1.0
    max_step: float = 0.25
    line_search_shrink: float = 0.5
    line_search_min_step: float = 1.0e-4
    max_line_search_iterations: int = 8
    armijo_constant: float = 1.0e-4
    history_size: int = 5
    optimizer: PeriodicGeometryOptimizer = "lbfgs"
    reuse_scf_state: bool = True
    relaxation_mode: Literal["ions"] = "ions"

    def __post_init__(self) -> None:
        positive = {
            "force_tolerance": self.force_tolerance,
            "rms_force_tolerance": self.rms_force_tolerance,
            "displacement_tolerance": self.displacement_tolerance,
            "initial_step_size": self.initial_step_size,
            "max_step": self.max_step,
            "line_search_min_step": self.line_search_min_step,
        }
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive non-bool integer")
        if any(
            isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0.0
            for value in positive.values()
        ):
            raise ValueError("periodic geometry tolerances and steps must be positive")
        if not 0.0 < self.line_search_shrink < 1.0:
            raise ValueError("line_search_shrink must lie in (0, 1)")
        if not 0.0 < self.armijo_constant < 1.0:
            raise ValueError("armijo_constant must lie in (0, 1)")
        if type(self.max_line_search_iterations) is not int or self.max_line_search_iterations <= 0:
            raise ValueError("max_line_search_iterations must be a positive non-bool integer")
        if type(self.history_size) is not int or self.history_size <= 0:
            raise ValueError("history_size must be a positive non-bool integer")
        if self.optimizer not in {"lbfgs", "steepest_descent"}:
            raise ValueError("optimizer must be 'lbfgs' or 'steepest_descent'")
        if type(self.reuse_scf_state) is not bool:
            raise ValueError("reuse_scf_state must be bool")
        if self.relaxation_mode != "ions":
            raise ValueError("periodic geometry optimization supports only fixed-cell ions")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-safe optimizer settings."""

        return {
            "max_steps": self.max_steps,
            "force_tolerance_hartree_per_bohr": self.force_tolerance,
            "rms_force_tolerance_hartree_per_bohr": self.rms_force_tolerance,
            "displacement_tolerance_bohr": self.displacement_tolerance,
            "initial_step_size_bohr2_per_hartree": self.initial_step_size,
            "max_step_bohr": self.max_step,
            "line_search_shrink": self.line_search_shrink,
            "line_search_min_step_bohr2_per_hartree": self.line_search_min_step,
            "max_line_search_iterations": self.max_line_search_iterations,
            "armijo_constant": self.armijo_constant,
            "history_size": self.history_size,
            "optimizer": self.optimizer,
            "reuse_scf_state": self.reuse_scf_state,
            "relaxation_mode": self.relaxation_mode,
        }


@dataclass(frozen=True)
class PeriodicGeometryOptimizationStep:
    """One accepted periodic ionic step."""

    index: int
    energy: float
    energy_delta: float
    armijo_limit: float
    max_force: float
    rms_force: float
    step_norm: float
    accepted_step_size: float
    line_search_iterations: int
    scf_iterations: int
    scf_wall_ms: float
    force_wall_ms: float
    used_density_continuation: bool
    used_coefficient_continuation: bool
    positions: np.ndarray
    forces: np.ndarray

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe accepted-step record."""

        return {
            "index": self.index,
            "energy_hartree": self.energy,
            "energy_delta_hartree": self.energy_delta,
            "armijo_limit_hartree": self.armijo_limit,
            "max_force_hartree_per_bohr": self.max_force,
            "rms_force_hartree_per_bohr": self.rms_force,
            "step_norm_bohr": self.step_norm,
            "accepted_step_size_bohr2_per_hartree": self.accepted_step_size,
            "line_search_iterations": self.line_search_iterations,
            "scf_iterations": self.scf_iterations,
            "scf_wall_ms": self.scf_wall_ms,
            "force_wall_ms": self.force_wall_ms,
            "used_density_continuation": self.used_density_continuation,
            "used_coefficient_continuation": self.used_coefficient_continuation,
            "positions_bohr": self.positions.tolist(),
            "forces_hartree_per_bohr": self.forces.tolist(),
        }

    @classmethod
    def _from_dict(cls, payload: Mapping[str, object]) -> PeriodicGeometryOptimizationStep:
        return cls(
            index=int(payload["index"]),
            energy=float(payload["energy_hartree"]),
            energy_delta=float(payload["energy_delta_hartree"]),
            armijo_limit=float(payload["armijo_limit_hartree"]),
            max_force=float(payload["max_force_hartree_per_bohr"]),
            rms_force=float(payload["rms_force_hartree_per_bohr"]),
            step_norm=float(payload["step_norm_bohr"]),
            accepted_step_size=float(payload["accepted_step_size_bohr2_per_hartree"]),
            line_search_iterations=int(payload["line_search_iterations"]),
            scf_iterations=int(payload["scf_iterations"]),
            scf_wall_ms=float(payload["scf_wall_ms"]),
            force_wall_ms=float(payload["force_wall_ms"]),
            used_density_continuation=bool(payload["used_density_continuation"]),
            used_coefficient_continuation=bool(payload["used_coefficient_continuation"]),
            positions=np.asarray(payload["positions_bohr"], dtype=np.float64),
            forces=np.asarray(payload["forces_hartree_per_bohr"], dtype=np.float64),
        )


@dataclass(frozen=True)
class PeriodicGeometryOptimizationResult:
    """Result of periodic fixed-cell ionic relaxation."""

    status: PeriodicGeometryStatus
    convergence_reason: str
    initial_system: PeriodicDFTSystem
    final_system: PeriodicDFTSystem
    final_scf: PeriodicSCFResult | None
    final_force: PeriodicForceResult | None
    steps: tuple[PeriodicGeometryOptimizationStep, ...]
    config: PeriodicGeometryOptimizationConfig
    elapsed_ms: float
    scf_evaluations: int
    line_search_evaluations: int
    continuation_density_uses: int
    continuation_coefficient_uses: int
    lineage: tuple[str, ...] = ()
    checkpoint_manifest: dict[str, object] | None = None

    @property
    def converged(self) -> bool:
        """Whether every configured ionic convergence gate passed."""

        return self.status == "converged"

    @property
    def final_energy(self) -> float | None:
        """Return the final periodic energy or free energy in Hartree."""

        return None if self.final_scf is None else float(self.final_scf.total_energy)

    @property
    def final_positions(self) -> np.ndarray:
        """Return final wrapped Cartesian positions in bohr."""

        return np.array(self.final_system.positions, dtype=np.float64, copy=True)

    @property
    def final_forces(self) -> np.ndarray | None:
        """Return final analytic periodic forces in Hartree/bohr."""

        if self.final_force is None:
            return None
        return np.asarray(self.final_force.forces, dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe summary without dense electronic arrays."""

        return {
            "status": self.status,
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
            "elapsed_ms": self.elapsed_ms,
            "scf_evaluations": self.scf_evaluations,
            "line_search_evaluations": self.line_search_evaluations,
            "continuation_density_uses": self.continuation_density_uses,
            "continuation_coefficient_uses": self.continuation_coefficient_uses,
            "accepted_step_count": len(self.steps),
            "final_energy_hartree": self.final_energy,
            "final_positions_bohr": self.final_positions.tolist(),
            "final_forces_hartree_per_bohr": (
                None if self.final_forces is None else self.final_forces.tolist()
            ),
            "config": self.config.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "lineage": list(self.lineage),
            "checkpoint_manifest_sha256": (
                None
                if self.checkpoint_manifest is None
                else self.checkpoint_manifest.get("manifest_sha256")
            ),
        }


@dataclass(frozen=True)
class _PeriodicEvaluation:
    system: PeriodicDFTSystem
    scf: PeriodicSCFResult
    force: PeriodicForceResult
    scf_wall_ms: float
    force_wall_ms: float
    used_density: bool
    used_coefficients: bool


class _EvaluationFailure(RuntimeError):
    pass


def _wrap_positions(system: PeriodicDFTSystem, positions: np.ndarray) -> np.ndarray:
    matrix = np.asarray(system.grid.cell.matrix, dtype=np.float64)
    fractional = np.asarray(positions, dtype=np.float64) @ np.linalg.inv(matrix)
    return (fractional - np.floor(fractional)) @ matrix


def _minimum_image(system: PeriodicDFTSystem, displacement: np.ndarray) -> np.ndarray:
    matrix = np.asarray(system.grid.cell.matrix, dtype=np.float64)
    fractional = np.asarray(displacement, dtype=np.float64) @ np.linalg.inv(matrix)
    return (fractional - np.rint(fractional)) @ matrix


def _optimization_contract(
    system: PeriodicDFTSystem,
    *,
    cutoff_hartree: float,
    kpoint_mesh: KPointMesh,
    n_bands: int | None,
    config: PeriodicGeometryOptimizationConfig,
    scf_config: PeriodicSCFConfig,
    xc_functional: ExchangeCorrelationFunctional | None,
) -> dict[str, object]:
    electronic = periodic_scf_calculation_contract(
        system,
        cutoff_hartree=cutoff_hartree,
        kpoint_mesh=kpoint_mesh,
        n_bands=n_bands,
        config=scf_config,
        xc_functional=xc_functional,
    )
    electronic.pop("selected_device", None)
    electronic.pop("eigensolve", None)
    return {
        "schema_version": "mlx-atomistic.periodic-geometry-calculation.v1",
        "electronic_calculation": electronic,
        "optimizer": config.to_dict(),
    }


def _evaluate_geometry(
    system: PeriodicDFTSystem,
    *,
    cutoff_hartree: float,
    kpoint_mesh: KPointMesh,
    n_bands: int | None,
    scf_config: PeriodicSCFConfig,
    xc_functional: ExchangeCorrelationFunctional | None,
    initial_density: object | None,
    initial_coefficients: object | None,
    basis_integer_g: Sequence[np.ndarray] | None,
    observer: RuntimeObserver | None,
) -> _PeriodicEvaluation:
    started = perf_counter()
    try:
        scf_kwargs = {
            "cutoff_hartree": cutoff_hartree,
            "kpoint_mesh": kpoint_mesh,
            "n_bands": n_bands,
            "config": scf_config,
            "xc_functional": xc_functional,
            "initial_density": initial_density,
            "initial_coefficients": initial_coefficients,
            "observer": observer,
        }
        if basis_integer_g is None:
            result = run_periodic_scf(system, **scf_kwargs)
        else:
            result = _run_periodic_scf_fixed_topology(
                system,
                basis_integer_g=basis_integer_g,
                **scf_kwargs,
            )
    except (FloatingPointError, RuntimeError, ValueError) as error:
        raise _EvaluationFailure(f"periodic SCF failed: {error}") from error
    scf_wall_ms = (perf_counter() - started) * 1000.0
    if (
        not result.converged
        or not np.isfinite(result.total_energy)
        or not np.isfinite(result.electron_count)
        or abs(result.electron_count - system.electron_count) > 1.0e-4
    ):
        raise _EvaluationFailure("periodic SCF result is unconverged or non-finite")
    started = perf_counter()
    try:
        force = periodic_scf_forces(system, result)
    except (FloatingPointError, RuntimeError, ValueError) as error:
        raise _EvaluationFailure(f"periodic force evaluation failed: {error}") from error
    force_wall_ms = (perf_counter() - started) * 1000.0
    forces = np.asarray(force.forces, dtype=np.float64)
    if forces.shape != system.positions.shape or not np.isfinite(forces).all():
        raise _EvaluationFailure("periodic forces are malformed or non-finite")
    return _PeriodicEvaluation(
        system=system,
        scf=result,
        force=force,
        scf_wall_ms=scf_wall_ms,
        force_wall_ms=force_wall_ms,
        used_density=initial_density is not None,
        used_coefficients=initial_coefficients is not None,
    )


@dataclass
class _PeriodicGeometryController:
    initial_system: PeriodicDFTSystem
    cutoff_hartree: float
    kpoint_mesh: KPointMesh
    n_bands: int | None
    config: PeriodicGeometryOptimizationConfig
    scf_config: PeriodicSCFConfig
    xc_functional: ExchangeCorrelationFunctional | None
    observer: RuntimeObserver | None
    calculation_contract: dict[str, object]
    checkpoint_to: Path | None
    checkpoint_step: int | None
    provenance: Mapping[str, object] | None
    start: float
    current: _PeriodicEvaluation | None
    steps: list[PeriodicGeometryOptimizationStep]
    s_history: list[np.ndarray]
    y_history: list[np.ndarray]
    next_step: int
    scf_evaluations: int
    line_search_evaluations: int
    continuation_density_uses: int
    continuation_coefficient_uses: int
    lineage: tuple[str, ...]
    status: PeriodicGeometryStatus
    convergence_reason: str
    checkpoint_manifest: dict[str, object] | None
    resume_density: mx.array | None
    resume_coefficients: Sequence[mx.array] | None
    basis_integer_g: Sequence[np.ndarray] | None

    @classmethod
    def create(
        cls,
        system: PeriodicDFTSystem,
        *,
        cutoff_hartree: float,
        kpoint_mesh: KPointMesh,
        n_bands: int | None,
        config: PeriodicGeometryOptimizationConfig,
        scf_config: PeriodicSCFConfig,
        xc_functional: ExchangeCorrelationFunctional | None,
        observer: RuntimeObserver | None,
        initial_density: mx.array | None,
        initial_coefficients: Sequence[mx.array] | None,
        basis_integer_g: Sequence[np.ndarray] | None,
        checkpoint_to: str | Path | None,
        checkpoint_step: int | None,
        resume_from: str | Path | None,
        provenance: Mapping[str, object] | None,
    ) -> _PeriodicGeometryController:
        contract = _optimization_contract(
            system,
            cutoff_hartree=cutoff_hartree,
            kpoint_mesh=kpoint_mesh,
            n_bands=n_bands,
            config=config,
            scf_config=scf_config,
            xc_functional=xc_functional,
        )
        if (checkpoint_to is None) != (checkpoint_step is None):
            raise ValueError("checkpoint_to and checkpoint_step must be supplied together")
        if (
            initial_density is not None or initial_coefficients is not None
        ) and resume_from is not None:
            raise ValueError("initial electronic state and resume_from are mutually exclusive")
        if (
            initial_density is not None or initial_coefficients is not None
        ) and not config.reuse_scf_state:
            raise ValueError("initial electronic state requires reuse_scf_state=True")
        if checkpoint_step is not None and (
            type(checkpoint_step) is not int
            or checkpoint_step <= 0
            or checkpoint_step > config.max_steps
        ):
            raise ValueError("checkpoint_step must be an accepted step within max_steps")
        steps: list[PeriodicGeometryOptimizationStep] = []
        s_history: list[np.ndarray] = []
        y_history: list[np.ndarray] = []
        next_step = 1
        scf_evaluations = 0
        line_search_evaluations = 0
        lineage: tuple[str, ...] = ()
        resume_density = None if initial_density is None else mx.array(initial_density)
        resume_coefficients = initial_coefficients
        if resume_from is not None:
            state = _load_periodic_geometry_checkpoint(
                resume_from,
                expected_calculation_contract=contract,
            )
            if state.completed_step >= config.max_steps:
                raise ValueError("periodic geometry checkpoint has no remaining step budget")
            if checkpoint_step is not None and checkpoint_step <= state.completed_step:
                raise ValueError("new checkpoint_step must follow the resumed step")
            steps = [PeriodicGeometryOptimizationStep._from_dict(item) for item in state.steps]
            s_history = [np.array(value, copy=True) for value in state.s_history]
            y_history = [np.array(value, copy=True) for value in state.y_history]
            next_step = state.completed_step + 1
            scf_evaluations = state.scf_evaluations
            line_search_evaluations = state.line_search_evaluations
            lineage = state.lineage
            resume_density = state.density
            resume_coefficients = None
        return cls(
            initial_system=system,
            cutoff_hartree=float(cutoff_hartree),
            kpoint_mesh=kpoint_mesh,
            n_bands=n_bands,
            config=config,
            scf_config=scf_config,
            xc_functional=xc_functional,
            observer=observer,
            calculation_contract=contract,
            checkpoint_to=None if checkpoint_to is None else Path(checkpoint_to),
            checkpoint_step=checkpoint_step,
            provenance=provenance,
            start=perf_counter(),
            current=None,
            steps=steps,
            s_history=s_history,
            y_history=y_history,
            next_step=next_step,
            scf_evaluations=scf_evaluations,
            line_search_evaluations=line_search_evaluations,
            continuation_density_uses=0,
            continuation_coefficient_uses=0,
            lineage=lineage,
            status="max_steps",
            convergence_reason="max_steps",
            checkpoint_manifest=None,
            resume_density=resume_density,
            resume_coefficients=resume_coefficients,
            basis_integer_g=basis_integer_g,
        )

    def run(self) -> PeriodicGeometryOptimizationResult:
        if not self._initialize():
            return self._result()
        if self._forces_converged(self._current_forces()):
            if not self.steps:
                self._stop("converged", "initial_force_tolerances")
                return self._result()
            if self.steps[-1].step_norm <= self.config.displacement_tolerance:
                self._stop("converged", "resumed_force_and_displacement_tolerances")
                return self._result()
        for step_index in range(self.next_step, self.config.max_steps + 1):
            direction = self._direction()
            if direction is None:
                break
            accepted = self._line_search(direction)
            if accepted is None:
                self._stop("line_search_failed", "line_search_exhausted")
                break
            evaluation, displacement, alpha, iteration, armijo_limit = accepted
            self._accept(
                step_index,
                evaluation,
                displacement,
                alpha,
                iteration,
                armijo_limit,
            )
            if self.checkpoint_step == step_index:
                self._publish_checkpoint(step_index)
                self._stop("checkpointed", "requested_accepted_step")
                break
            if self._step_converged(self.steps[-1]):
                self._stop("converged", "force_and_displacement_tolerances")
                break
        return self._result()

    def _initialize(self) -> bool:
        system = (
            self.initial_system.with_positions(
                _wrap_positions(self.initial_system, self.initial_system.positions)
            )
            if not self.steps
            else self.initial_system.with_positions(self.steps[-1].positions)
        )
        try:
            self.current = self._evaluate(
                system,
                density=(self.resume_density if self.config.reuse_scf_state else None),
                coefficients=(
                    self.resume_coefficients if self.config.reuse_scf_state else None
                ),
            )
        except _EvaluationFailure as error:
            self._stop("scf_failed", str(error))
            return False
        return True

    def _evaluate(
        self,
        system: PeriodicDFTSystem,
        *,
        density: object | None,
        coefficients: object | None,
    ) -> _PeriodicEvaluation:
        self.scf_evaluations += 1
        if density is not None:
            self.continuation_density_uses += 1
        if coefficients is not None:
            self.continuation_coefficient_uses += 1
        return _evaluate_geometry(
            system,
            cutoff_hartree=self.cutoff_hartree,
            kpoint_mesh=self.kpoint_mesh,
            n_bands=self.n_bands,
            scf_config=self.scf_config,
            xc_functional=self.xc_functional,
            initial_density=density,
            initial_coefficients=coefficients,
            basis_integer_g=self.basis_integer_g,
            observer=self.observer,
        )

    def _direction(self) -> np.ndarray | None:
        forces = self._current_forces()
        direction = _search_direction(
            -forces,
            forces,
            optimizer=self.config.optimizer,
            s_history=self.s_history,
            y_history=self.y_history,
        )
        if not np.isfinite(direction).all() or _max_vector_norm(direction) <= 1e-14:
            self._stop("nonfinite", "invalid_search_direction")
            return None
        return direction

    def _line_search(
        self,
        direction: np.ndarray,
    ) -> tuple[_PeriodicEvaluation, np.ndarray, float, int, float] | None:
        current = self._current()
        current_forces = np.asarray(current.force.forces, dtype=np.float64)
        current_positions = np.asarray(current.system.positions, dtype=np.float64)
        gradient = -current_forces
        density = current.scf.density if self.config.reuse_scf_state else None
        coefficients = (
            current.scf.continuation_coefficients if self.config.reuse_scf_state else None
        )
        alpha = self.config.initial_step_size
        for iteration in range(1, self.config.max_line_search_iterations + 1):
            if alpha < self.config.line_search_min_step:
                break
            displacement = _clip_direction(alpha * direction, self.config.max_step)
            directional_change = float(np.sum(gradient * displacement))
            if not np.isfinite(directional_change) or directional_change >= 0.0:
                return None
            positions = _wrap_positions(current.system, current_positions + displacement)
            system = current.system.with_positions(positions)
            self.line_search_evaluations += 1
            try:
                candidate = self._evaluate(
                    system,
                    density=density,
                    coefficients=coefficients,
                )
            except _EvaluationFailure:
                alpha *= self.config.line_search_shrink
                continue
            limit = current.scf.total_energy + self.config.armijo_constant * directional_change
            if candidate.scf.total_energy <= limit:
                accepted_displacement = _minimum_image(
                    current.system,
                    positions - current_positions,
                )
                return candidate, accepted_displacement, alpha, iteration, limit
            alpha *= self.config.line_search_shrink
        return None

    def _accept(
        self,
        step_index: int,
        candidate: _PeriodicEvaluation,
        displacement: np.ndarray,
        alpha: float,
        line_search_iterations: int,
        armijo_limit: float,
    ) -> None:
        current = self._current()
        current_forces = np.asarray(current.force.forces, dtype=np.float64)
        next_forces = np.asarray(candidate.force.forces, dtype=np.float64)
        s_vector = displacement.reshape(-1)
        y_vector = (current_forces - next_forces).reshape(-1)
        if _valid_lbfgs_pair(s_vector, y_vector):
            self.s_history.append(s_vector)
            self.y_history.append(y_vector)
            del self.s_history[: -self.config.history_size]
            del self.y_history[: -self.config.history_size]
        step = PeriodicGeometryOptimizationStep(
            index=step_index,
            energy=float(candidate.scf.total_energy),
            energy_delta=float(candidate.scf.total_energy - current.scf.total_energy),
            armijo_limit=float(armijo_limit),
            max_force=_max_vector_norm(next_forces),
            rms_force=_rms_components(next_forces),
            step_norm=_max_vector_norm(displacement),
            accepted_step_size=alpha,
            line_search_iterations=line_search_iterations,
            scf_iterations=int(candidate.scf.iterations),
            scf_wall_ms=candidate.scf_wall_ms,
            force_wall_ms=candidate.force_wall_ms,
            used_density_continuation=candidate.used_density,
            used_coefficient_continuation=candidate.used_coefficients,
            positions=np.asarray(candidate.system.positions, dtype=np.float64),
            forces=next_forces,
        )
        self.steps.append(step)
        self.current = candidate

    def _publish_checkpoint(self, completed_step: int) -> None:
        current = self._current()
        state = _PeriodicGeometryContinuationState(
            completed_step=completed_step,
            positions=np.asarray(current.system.positions, dtype=np.float64),
            density=current.scf.density,
            energy=float(current.scf.total_energy),
            forces=np.asarray(current.force.forces, dtype=np.float64),
            steps=tuple(step.to_dict() for step in self.steps),
            s_history=tuple(self.s_history),
            y_history=tuple(self.y_history),
            scf_evaluations=self.scf_evaluations,
            line_search_evaluations=self.line_search_evaluations,
            lineage=self.lineage,
        )
        self.checkpoint_manifest = _publish_periodic_geometry_checkpoint(
            self.checkpoint_to,
            state=state,
            calculation_contract=self.calculation_contract,
            provenance=self.provenance,
        )

    def _forces_converged(self, forces: np.ndarray) -> bool:
        return bool(
            _max_vector_norm(forces) <= self.config.force_tolerance
            and _rms_components(forces) <= self.config.rms_force_tolerance
        )

    def _step_converged(self, step: PeriodicGeometryOptimizationStep) -> bool:
        return bool(
            step.max_force <= self.config.force_tolerance
            and step.rms_force <= self.config.rms_force_tolerance
            and step.step_norm <= self.config.displacement_tolerance
        )

    def _current(self) -> _PeriodicEvaluation:
        if self.current is None:
            raise RuntimeError("periodic geometry optimization has no current state")
        return self.current

    def _current_forces(self) -> np.ndarray:
        return np.asarray(self._current().force.forces, dtype=np.float64)

    def _stop(self, status: PeriodicGeometryStatus, reason: str) -> None:
        self.status = status
        self.convergence_reason = reason

    def _result(self) -> PeriodicGeometryOptimizationResult:
        current = self.current
        return PeriodicGeometryOptimizationResult(
            status=self.status,
            convergence_reason=self.convergence_reason,
            initial_system=self.initial_system,
            final_system=(self.initial_system if current is None else current.system),
            final_scf=None if current is None else current.scf,
            final_force=None if current is None else current.force,
            steps=tuple(self.steps),
            config=self.config,
            elapsed_ms=(perf_counter() - self.start) * 1000.0,
            scf_evaluations=self.scf_evaluations,
            line_search_evaluations=self.line_search_evaluations,
            continuation_density_uses=self.continuation_density_uses,
            continuation_coefficient_uses=self.continuation_coefficient_uses,
            lineage=self.lineage,
            checkpoint_manifest=self.checkpoint_manifest,
        )


def optimize_periodic_geometry(
    system: PeriodicDFTSystem,
    *,
    cutoff_hartree: float,
    kpoint_mesh: KPointMesh,
    n_bands: int | None = None,
    config: PeriodicGeometryOptimizationConfig | None = None,
    scf_config: PeriodicSCFConfig | None = None,
    xc_functional: ExchangeCorrelationFunctional | None = None,
    observer: RuntimeObserver | None = None,
    initial_density: mx.array | None = None,
    initial_coefficients: Sequence[mx.array] | None = None,
    checkpoint_to: str | Path | None = None,
    checkpoint_step: int | None = None,
    resume_from: str | Path | None = None,
    provenance: Mapping[str, object] | None = None,
) -> PeriodicGeometryOptimizationResult:
    """Relax ions in a fixed periodic cell using converged analytic forces.

    Args:
        system: Initial periodic GTH system.
        cutoff_hartree: Plane-wave kinetic cutoff in Hartree.
        kpoint_mesh: Fixed weighted reduced-coordinate k-point mesh.
        n_bands: Fixed computed band count.
        config: Ionic optimizer controls.
        scf_config: Exact periodic SCF controls.
        xc_functional: Exchange-correlation functional. Defaults to production PBE.
        observer: Optional shared runtime observer.
        initial_density: Optional density seed for the initial periodic SCF.
        initial_coefficients: Optional k-point orbital seeds for the initial SCF.
        checkpoint_to: Previously absent accepted-step checkpoint destination.
        checkpoint_step: Accepted step at which to publish and stop.
        resume_from: Explicit accepted-step checkpoint to resume.
        provenance: Optional non-identity checkpoint provenance.

    Returns:
        Complete periodic optimization result and accepted-step history.

    Raises:
        TypeError: If the system or k-point mesh has an unsupported type.
        ValueError: If controls conflict or checkpoint settings are invalid.
        ArtifactIntegrityError: If an explicit checkpoint fails validation.
    """

    if not isinstance(system, PeriodicDFTSystem):
        raise TypeError("system must be PeriodicDFTSystem")
    if not isinstance(kpoint_mesh, KPointMesh):
        raise TypeError("kpoint_mesh must be KPointMesh")
    if (
        isinstance(cutoff_hartree, (bool, np.bool_))
        or not np.isfinite(cutoff_hartree)
        or cutoff_hartree <= 0.0
    ):
        raise ValueError("cutoff_hartree must be finite and positive")
    resolved_config = PeriodicGeometryOptimizationConfig() if config is None else config
    resolved_scf = PeriodicSCFConfig() if scf_config is None else scf_config
    return _PeriodicGeometryController.create(
        system,
        cutoff_hartree=cutoff_hartree,
        kpoint_mesh=kpoint_mesh,
        n_bands=n_bands,
        config=resolved_config,
        scf_config=resolved_scf,
        xc_functional=xc_functional,
        observer=observer,
        initial_density=initial_density,
        initial_coefficients=initial_coefficients,
        basis_integer_g=None,
        checkpoint_to=checkpoint_to,
        checkpoint_step=checkpoint_step,
        resume_from=resume_from,
        provenance=provenance,
    ).run()


def _optimize_periodic_geometry_fixed_topology(
    system: PeriodicDFTSystem,
    *,
    cutoff_hartree: float,
    kpoint_mesh: KPointMesh,
    n_bands: int | None,
    config: PeriodicGeometryOptimizationConfig,
    scf_config: PeriodicSCFConfig,
    xc_functional: ExchangeCorrelationFunctional | None,
    observer: RuntimeObserver | None,
    initial_density: mx.array | None,
    initial_coefficients: Sequence[mx.array] | None,
    basis_integer_g: Sequence[np.ndarray],
) -> PeriodicGeometryOptimizationResult:
    """Compose fixed-cell ionic relaxation inside one cell-trajectory basis."""

    return _PeriodicGeometryController.create(
        system,
        cutoff_hartree=cutoff_hartree,
        kpoint_mesh=kpoint_mesh,
        n_bands=n_bands,
        config=config,
        scf_config=scf_config,
        xc_functional=xc_functional,
        observer=observer,
        initial_density=initial_density,
        initial_coefficients=initial_coefficients,
        basis_integer_g=basis_integer_g,
        checkpoint_to=None,
        checkpoint_step=None,
        resume_from=None,
        provenance=None,
    ).run()
