"""Restart-ready variable-cell relaxation for periodic plane-wave DFT."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft._periodic_artifact_contracts import (
    periodic_scf_calculation_contract,
)
from mlx_atomistic.dft._periodic_cell_optimization_checkpoint import (
    _load_periodic_cell_checkpoint,
    _publish_periodic_cell_checkpoint,
)
from mlx_atomistic.dft._periodic_cell_optimization_models import (
    PeriodicCellOptimizationConfig,
    PeriodicCellOptimizationResult,
    PeriodicCellOptimizationStep,
    PeriodicCellStatus,
)
from mlx_atomistic.dft._periodic_cell_optimization_state import (
    _PeriodicCellContinuationState,
)
from mlx_atomistic.dft._periodic_models import (
    PeriodicDFTSystem,
    PeriodicSCFConfig,
    PeriodicSCFResult,
)
from mlx_atomistic.dft._runtime_observer import RuntimeObserver
from mlx_atomistic.dft.kpoints import KPointMesh
from mlx_atomistic.dft.periodic_forces import PeriodicForceResult
from mlx_atomistic.dft.periodic_optimization import (
    PeriodicGeometryOptimizationResult,
    _optimize_periodic_geometry_fixed_topology,
)
from mlx_atomistic.dft.periodic_scf import (
    _run_periodic_scf_fixed_topology,
    run_periodic_scf,
)
from mlx_atomistic.dft.periodic_stress import (
    PeriodicStressResult,
    periodic_finite_difference_stress,
)
from mlx_atomistic.dft.plane_wave import PlaneWaveBasis
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional


@dataclass(frozen=True)
class _CellState:
    system: PeriodicDFTSystem
    scf: PeriodicSCFResult
    stress: PeriodicStressResult
    force: PeriodicForceResult | None
    ionic: PeriodicGeometryOptimizationResult | None
    enthalpy: float


def _force_metrics(force: PeriodicForceResult | None) -> tuple[float | None, float | None]:
    if force is None:
        return None, None
    values = np.asarray(force.forces, dtype=np.float64)
    maximum = float(np.max(np.linalg.norm(values, axis=1)))
    rms = float(np.sqrt(np.mean(values * values)))
    return maximum, rms


def _stress_residual(state: _CellState, config: PeriodicCellOptimizationConfig) -> np.ndarray:
    stress = np.asarray(state.stress.stress, dtype=np.float64)
    residual = stress - config.external_pressure * np.eye(3)
    mode = config.stress_config.mode
    if mode == "isotropic":
        return np.eye(3) * float(np.trace(residual) / 3.0)
    if mode == "diagonal":
        return np.diag(np.diag(residual))
    return 0.5 * (residual + residual.T)


def _stress_norm(residual: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvalsh(residual))))


def _bounded_strain(
    residual: np.ndarray,
    config: PeriodicCellOptimizationConfig,
) -> np.ndarray:
    strain = config.cell_compliance * residual
    maximum = float(np.max(np.abs(np.linalg.eigvalsh(strain))))
    if maximum > config.max_strain:
        strain *= config.max_strain / maximum
    return strain


def _cutoff_basis_topology(
    system: PeriodicDFTSystem,
    *,
    cutoff_hartree: float,
    kpoint_mesh: KPointMesh,
) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(
            PlaneWaveBasis.from_reduced_kpoint(
                system.grid,
                cutoff_hartree,
                point.vector,
                lane_label=f"kpoint:{index}",
            ).active_integer_g,
            dtype=np.int32,
        )
        for index, point in enumerate(kpoint_mesh.points)
    )


def _cell_optimization_contract(
    system: PeriodicDFTSystem,
    *,
    cutoff_hartree: float,
    kpoint_mesh: KPointMesh,
    n_bands: int | None,
    config: PeriodicCellOptimizationConfig,
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
        "schema_version": "mlx-atomistic.periodic-cell-calculation.v1",
        "electronic_calculation": electronic,
        "optimizer": config.to_dict(),
    }


class _CellOptimizationFailure(RuntimeError):
    pass


class _StressEvaluationFailure(_CellOptimizationFailure):
    pass


@dataclass
class _PeriodicCellController:
    initial_system: PeriodicDFTSystem
    cutoff_hartree: float
    kpoint_mesh: KPointMesh
    n_bands: int | None
    config: PeriodicCellOptimizationConfig
    scf_config: PeriodicSCFConfig
    xc_functional: ExchangeCorrelationFunctional | None
    observer: RuntimeObserver | None
    started: float = field(default_factory=perf_counter)
    current: _CellState | None = None
    steps: list[PeriodicCellOptimizationStep] = field(default_factory=list)
    scf_evaluations: int = 0
    stress_evaluations: int = 0
    line_search_evaluations: int = 0
    ionic_scf_evaluations: int = 0
    status: PeriodicCellStatus = "max_steps"
    convergence_reason: str = "max_steps"
    calculation_contract: dict[str, object] = field(default_factory=dict)
    checkpoint_to: Path | None = None
    checkpoint_step: int | None = None
    provenance: Mapping[str, object] | None = None
    next_step: int = 1
    resume_system: PeriodicDFTSystem | None = None
    resume_density: mx.array | None = None
    lineage: tuple[str, ...] = ()
    checkpoint_manifest: dict[str, object] | None = None
    basis_integer_g: tuple[np.ndarray, ...] | None = None

    @classmethod
    def create(
        cls,
        system: PeriodicDFTSystem,
        *,
        cutoff_hartree: float,
        kpoint_mesh: KPointMesh,
        n_bands: int | None,
        config: PeriodicCellOptimizationConfig,
        scf_config: PeriodicSCFConfig,
        xc_functional: ExchangeCorrelationFunctional | None,
        observer: RuntimeObserver | None,
        checkpoint_to: str | Path | None,
        checkpoint_step: int | None,
        resume_from: str | Path | None,
        provenance: Mapping[str, object] | None,
    ) -> _PeriodicCellController:
        contract = _cell_optimization_contract(
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
        if checkpoint_step is not None and (
            type(checkpoint_step) is not int
            or checkpoint_step <= 0
            or checkpoint_step > config.max_steps
        ):
            raise ValueError("checkpoint_step must be an accepted step within max_steps")
        controller = cls(
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
        )
        if resume_from is None:
            return controller
        state = _load_periodic_cell_checkpoint(
            resume_from,
            expected_calculation_contract=contract,
        )
        if state.completed_step >= config.max_steps:
            raise ValueError("periodic cell checkpoint has no remaining step budget")
        if checkpoint_step is not None and checkpoint_step <= state.completed_step:
            raise ValueError("new checkpoint_step must follow the resumed step")
        controller.steps = [
            PeriodicCellOptimizationStep._from_dict(item) for item in state.steps
        ]
        controller.scf_evaluations = state.scf_evaluations
        controller.stress_evaluations = state.stress_evaluations
        controller.line_search_evaluations = state.line_search_evaluations
        controller.ionic_scf_evaluations = state.ionic_scf_evaluations
        controller.next_step = state.completed_step + 1
        controller.resume_system = system.with_cell(
            state.cell,
            scale_positions=True,
        ).with_positions(state.positions)
        controller.resume_density = state.density
        controller.lineage = state.lineage
        controller.basis_integer_g = _cutoff_basis_topology(
            system,
            cutoff_hartree=cutoff_hartree,
            kpoint_mesh=kpoint_mesh,
        )
        return controller

    def run(self) -> PeriodicCellOptimizationResult:
        starting_system = (
            self.initial_system if self.resume_system is None else self.resume_system
        )
        if (
            self.basis_integer_g is None
            and self.config.relaxation_mode == "ions_and_cell"
        ):
            self.basis_integer_g = _cutoff_basis_topology(
                self.initial_system,
                cutoff_hartree=self.cutoff_hartree,
                kpoint_mesh=self.kpoint_mesh,
            )
        try:
            self.current = self._evaluate_state(
                starting_system,
                density=self.resume_density,
            )
        except _StressEvaluationFailure as error:
            self._stop("stress_failed", str(error))
            return self._result()
        except _CellOptimizationFailure as error:
            self._stop("scf_failed", str(error))
            return self._result()
        if self.basis_integer_g is None:
            self.basis_integer_g = tuple(
                np.asarray(point.basis.active_integer_g, dtype=np.int32)
                for point in self.current.scf.kpoints
            )
        if self._converged(self.current):
            self._stop("converged", "initial_stress_and_force_tolerances")
            return self._result()
        initial_volume = float(self.initial_system.grid.volume)
        for index in range(self.next_step, self.config.max_steps + 1):
            current = self._current()
            residual = _stress_residual(current, self.config)
            direction = _bounded_strain(residual, self.config)
            accepted = self._line_search(direction, initial_volume=initial_volume)
            if accepted is None:
                self._stop("line_search_failed", "enthalpy_line_search_exhausted")
                break
            next_state, strain, alpha, iteration, armijo_limit = accepted
            self.current = next_state
            maximum_force, rms_force = _force_metrics(next_state.force)
            ionic_steps = 0 if next_state.ionic is None else len(next_state.ionic.steps)
            self.steps.append(
                PeriodicCellOptimizationStep(
                    index=index,
                    energy=float(next_state.scf.total_energy),
                    enthalpy=next_state.enthalpy,
                    volume=float(next_state.system.grid.volume),
                    pressure=next_state.stress.pressure,
                    stress_residual=_stress_norm(
                        _stress_residual(next_state, self.config)
                    ),
                    max_force=maximum_force,
                    rms_force=rms_force,
                    strain_norm=float(np.max(np.abs(np.linalg.eigvalsh(strain)))),
                    accepted_step_size=alpha,
                    line_search_iterations=iteration,
                    armijo_limit=armijo_limit,
                    ionic_steps=ionic_steps,
                    cell=np.asarray(next_state.system.grid.cell.matrix, dtype=np.float64),
                    positions=np.asarray(next_state.system.positions, dtype=np.float64),
                )
            )
            if self.checkpoint_step == index:
                self._publish_checkpoint(index)
                self._stop("checkpointed", "requested_accepted_step")
                break
            if self._converged(next_state):
                self._stop("converged", "stress_force_and_step_tolerances")
                break
        return self._result()

    def _evaluate_state(
        self,
        system: PeriodicDFTSystem,
        *,
        density: object | None,
        initial_scf: PeriodicSCFResult | None = None,
    ) -> _CellState:
        ionic = None
        force = None
        scf = initial_scf
        if self.config.relaxation_mode == "ions_and_cell":
            coefficient_seed = (
                None if initial_scf is None else initial_scf.continuation_coefficients
            )
            if self.basis_integer_g is None:
                raise _CellOptimizationFailure("cell trajectory basis is unavailable")
            ionic = _optimize_periodic_geometry_fixed_topology(
                system,
                cutoff_hartree=self.cutoff_hartree,
                kpoint_mesh=self.kpoint_mesh,
                n_bands=self.n_bands,
                config=self.config.ionic_config,
                scf_config=self.scf_config,
                xc_functional=self.xc_functional,
                observer=self.observer,
                initial_density=density,
                initial_coefficients=coefficient_seed,
                basis_integer_g=self.basis_integer_g,
            )
            self.ionic_scf_evaluations += ionic.scf_evaluations
            self.scf_evaluations += ionic.scf_evaluations
            if ionic.status not in {"converged", "max_steps"}:
                raise _CellOptimizationFailure(
                    f"ionic relaxation failed with status {ionic.status}"
                )
            if ionic.final_scf is None or ionic.final_force is None:
                raise _CellOptimizationFailure("ionic relaxation produced no final state")
            system = ionic.final_system
            scf = ionic.final_scf
            force = ionic.final_force
        elif scf is None:
            scf = self._run_scf(system, density=density)
        if scf is None or not scf.converged or scf.system_fingerprint != system.fingerprint:
            raise _CellOptimizationFailure("cell state SCF is unconverged or mismatched")
        try:
            stress = periodic_finite_difference_stress(
                system,
                cutoff_hartree=self.cutoff_hartree,
                kpoint_mesh=self.kpoint_mesh,
                n_bands=self.n_bands,
                config=self.config.stress_config,
                scf_config=self.scf_config,
                xc_functional=self.xc_functional,
                observer=self.observer,
                base_result=scf,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise _StressEvaluationFailure(f"stress evaluation failed: {error}") from error
        self.scf_evaluations += stress.scf_evaluations
        self.stress_evaluations += 1
        enthalpy = float(
            scf.total_energy + self.config.external_pressure * system.grid.volume
        )
        return _CellState(
            system=system,
            scf=scf,
            stress=stress,
            force=force,
            ionic=ionic,
            enthalpy=enthalpy,
        )

    def _run_scf(
        self,
        system: PeriodicDFTSystem,
        *,
        density: object | None,
        coefficients: object | None = None,
    ) -> PeriodicSCFResult:
        scf_kwargs = {
            "cutoff_hartree": self.cutoff_hartree,
            "kpoint_mesh": self.kpoint_mesh,
            "n_bands": self.n_bands,
            "config": self.scf_config,
            "xc_functional": self.xc_functional,
            "initial_density": density,
            "initial_coefficients": coefficients,
            "observer": self.observer,
        }
        if self.basis_integer_g is None:
            result = run_periodic_scf(system, **scf_kwargs)
        else:
            result = _run_periodic_scf_fixed_topology(
                system,
                basis_integer_g=self.basis_integer_g,
                **scf_kwargs,
            )
        self.scf_evaluations += 1
        if (
            not result.converged
            or not np.isfinite(result.total_energy)
            or not np.isfinite(result.electron_count)
            or abs(result.electron_count - system.electron_count) > 1.0e-4
            or result.system_fingerprint != system.fingerprint
        ):
            raise _CellOptimizationFailure("periodic SCF failed during cell optimization")
        return result

    def _line_search(
        self,
        direction: np.ndarray,
        *,
        initial_volume: float,
    ) -> tuple[_CellState, np.ndarray, float, int, float] | None:
        current = self._current()
        residual = _stress_residual(current, self.config)
        matrix = np.asarray(current.system.grid.cell.matrix, dtype=np.float64)
        alpha = 1.0
        for iteration in range(1, self.config.max_line_search_iterations + 1):
            if alpha < self.config.line_search_min_step:
                break
            strain = alpha * direction
            directional_derivative = -float(current.system.grid.volume) * float(
                np.sum(residual * strain)
            )
            if not np.isfinite(directional_derivative) or directional_derivative >= 0.0:
                return None
            try:
                candidate = current.system.with_cell(
                    matrix @ (np.eye(3) + strain),
                    scale_positions=True,
                )
            except ValueError:
                alpha *= self.config.line_search_shrink
                continue
            if candidate.grid.volume < self.config.minimum_volume_ratio * initial_volume:
                alpha *= self.config.line_search_shrink
                continue
            self.line_search_evaluations += 1
            try:
                scf = self._run_scf(
                    candidate,
                    density=current.scf.density,
                    coefficients=current.scf.continuation_coefficients,
                )
            except _CellOptimizationFailure:
                alpha *= self.config.line_search_shrink
                continue
            enthalpy = float(
                scf.total_energy + self.config.external_pressure * candidate.grid.volume
            )
            limit = (
                current.enthalpy
                + self.config.armijo_constant * directional_derivative
            )
            if enthalpy <= limit:
                try:
                    state = self._evaluate_state(
                        candidate,
                        density=scf.density,
                        initial_scf=scf,
                    )
                except _CellOptimizationFailure:
                    alpha *= self.config.line_search_shrink
                    continue
                if state.enthalpy <= limit:
                    return state, strain, alpha, iteration, limit
            alpha *= self.config.line_search_shrink
        return None

    def _publish_checkpoint(self, completed_step: int) -> None:
        current = self._current()
        matrix = np.asarray(current.system.grid.cell.matrix, dtype=np.float64)
        positions = np.asarray(current.system.positions, dtype=np.float64)
        state = _PeriodicCellContinuationState(
            completed_step=completed_step,
            cell=matrix,
            positions=positions,
            fractional_positions=positions @ np.linalg.inv(matrix),
            density=current.scf.density,
            energy=float(current.scf.total_energy),
            stress=np.asarray(current.stress.stress, dtype=np.float64),
            forces=(
                None
                if current.force is None
                else np.asarray(current.force.forces, dtype=np.float64)
            ),
            steps=tuple(step.to_dict() for step in self.steps),
            scf_evaluations=self.scf_evaluations,
            stress_evaluations=self.stress_evaluations,
            line_search_evaluations=self.line_search_evaluations,
            ionic_scf_evaluations=self.ionic_scf_evaluations,
            lineage=self.lineage,
        )
        if self.checkpoint_to is None:
            raise RuntimeError("periodic cell checkpoint destination is missing")
        self.checkpoint_manifest = _publish_periodic_cell_checkpoint(
            self.checkpoint_to,
            state=state,
            calculation_contract=self.calculation_contract,
            provenance=self.provenance,
        )

    def _converged(self, state: _CellState) -> bool:
        residual_ok = (
            _stress_norm(_stress_residual(state, self.config))
            <= self.config.stress_tolerance
        )
        force_ok = True
        if self.config.relaxation_mode == "ions_and_cell":
            maximum, rms = _force_metrics(state.force)
            force_ok = bool(
                maximum is not None
                and rms is not None
                and maximum <= self.config.ionic_config.force_tolerance
                and rms <= self.config.ionic_config.rms_force_tolerance
                and state.ionic is not None
                and state.ionic.converged
            )
        remaining_strain_ok = (
            _stress_norm(
                _bounded_strain(_stress_residual(state, self.config), self.config)
            )
            <= self.config.strain_tolerance
        )
        return residual_ok and force_ok and remaining_strain_ok

    def _current(self) -> _CellState:
        if self.current is None:
            raise RuntimeError("periodic cell optimization has no current state")
        return self.current

    def _stop(self, status: PeriodicCellStatus, reason: str) -> None:
        self.status = status
        self.convergence_reason = reason

    def _result(self) -> PeriodicCellOptimizationResult:
        current = self.current
        return PeriodicCellOptimizationResult(
            status=self.status,
            convergence_reason=self.convergence_reason,
            initial_system=self.initial_system,
            final_system=(self.initial_system if current is None else current.system),
            final_scf=None if current is None else current.scf,
            final_stress=None if current is None else current.stress,
            final_force=None if current is None else current.force,
            steps=tuple(self.steps),
            config=self.config,
            elapsed_ms=(perf_counter() - self.started) * 1000.0,
            scf_evaluations=self.scf_evaluations,
            stress_evaluations=self.stress_evaluations,
            line_search_evaluations=self.line_search_evaluations,
            ionic_scf_evaluations=self.ionic_scf_evaluations,
            lineage=self.lineage,
            checkpoint_manifest=self.checkpoint_manifest,
        )


def optimize_periodic_cell(
    system: PeriodicDFTSystem,
    *,
    cutoff_hartree: float,
    kpoint_mesh: KPointMesh,
    n_bands: int | None = None,
    config: PeriodicCellOptimizationConfig | None = None,
    scf_config: PeriodicSCFConfig | None = None,
    xc_functional: ExchangeCorrelationFunctional | None = None,
    observer: RuntimeObserver | None = None,
    checkpoint_to: str | Path | None = None,
    checkpoint_step: int | None = None,
    resume_from: str | Path | None = None,
    provenance: Mapping[str, object] | None = None,
) -> PeriodicCellOptimizationResult:
    """Relax a periodic cell at fixed scalar external pressure.

    Args:
        system: Initial periodic GTH system.
        cutoff_hartree: Plane-wave kinetic cutoff in Hartree.
        kpoint_mesh: Fixed reduced-coordinate k-point mesh.
        n_bands: Fixed computed band count.
        config: Cell, stress, pressure, and optional ionic controls.
        scf_config: Exact periodic SCF controls.
        xc_functional: Exchange-correlation functional.
        observer: Optional shared runtime observer.
        checkpoint_to: Previously absent accepted-step checkpoint destination.
        checkpoint_step: Accepted cell step at which to publish and stop.
        resume_from: Explicit accepted-cell checkpoint to resume.
        provenance: Optional non-identity checkpoint provenance.

    Returns:
        Complete accepted cell trajectory and final periodic state.

    Raises:
        TypeError: If public inputs have unsupported types.
        ValueError: If controls are invalid.
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
    resolved = PeriodicCellOptimizationConfig() if config is None else config
    resolved_scf = PeriodicSCFConfig() if scf_config is None else scf_config
    if not isinstance(resolved, PeriodicCellOptimizationConfig):
        raise TypeError("config must be PeriodicCellOptimizationConfig")
    if not isinstance(resolved_scf, PeriodicSCFConfig):
        raise TypeError("scf_config must be PeriodicSCFConfig")
    return _PeriodicCellController.create(
        system,
        cutoff_hartree=cutoff_hartree,
        kpoint_mesh=kpoint_mesh,
        n_bands=n_bands,
        config=resolved,
        scf_config=resolved_scf,
        xc_functional=xc_functional,
        observer=observer,
        checkpoint_to=checkpoint_to,
        checkpoint_step=checkpoint_step,
        resume_from=resume_from,
        provenance=provenance,
    ).run()
