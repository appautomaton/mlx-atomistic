"""Public data contracts for periodic cell optimization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from mlx_atomistic.dft._periodic_models import (
    PeriodicDFTSystem,
    PeriodicSCFResult,
)
from mlx_atomistic.dft.periodic_forces import PeriodicForceResult
from mlx_atomistic.dft.periodic_optimization import (
    PeriodicGeometryOptimizationConfig,
)
from mlx_atomistic.dft.periodic_stress import (
    PeriodicStressConfig,
    PeriodicStressResult,
)

PeriodicCellRelaxationMode = Literal["cell", "ions_and_cell"]
PeriodicCellStatus = Literal[
    "converged",
    "max_steps",
    "line_search_failed",
    "scf_failed",
    "stress_failed",
    "checkpointed",
]


def _default_ionic_config() -> PeriodicGeometryOptimizationConfig:
    return PeriodicGeometryOptimizationConfig(max_steps=4)


@dataclass(frozen=True)
class PeriodicCellOptimizationConfig:
    """Controls for periodic variable-cell relaxation.

    Args:
        max_steps: Maximum accepted cell steps.
        relaxation_mode: `cell` or alternating `ions_and_cell` relaxation.
        external_pressure: Scalar compression-positive pressure in Hartree/bohr³.
        stress_tolerance: Maximum admitted stress residual in Hartree/bohr³.
        strain_tolerance: Maximum final principal strain magnitude.
        cell_compliance: Stress-to-strain scale in bohr³/Hartree.
        max_strain: Maximum principal strain magnitude per accepted cell step.
        minimum_volume_ratio: Minimum volume relative to the initial cell.
        line_search_shrink: Backtracking scale factor.
        line_search_min_step: Smallest line-search scale.
        max_line_search_iterations: Maximum cell trials per step.
        armijo_constant: Generalized-enthalpy sufficient-decrease coefficient.
        stress_config: Numerical stress mode and strain controls.
        ionic_config: Bounded fixed-cell ionic controls for coupled relaxation.
    """

    max_steps: int = 12
    relaxation_mode: PeriodicCellRelaxationMode = "cell"
    external_pressure: float = 0.0
    stress_tolerance: float = 1.0e-5
    strain_tolerance: float = 1.0e-4
    cell_compliance: float = 100.0
    max_strain: float = 0.03
    minimum_volume_ratio: float = 0.5
    line_search_shrink: float = 0.5
    line_search_min_step: float = 1.0e-4
    max_line_search_iterations: int = 8
    armijo_constant: float = 1.0e-4
    stress_config: PeriodicStressConfig = field(
        default_factory=lambda: PeriodicStressConfig(mode="isotropic")
    )
    ionic_config: PeriodicGeometryOptimizationConfig = field(
        default_factory=_default_ionic_config
    )

    def __post_init__(self) -> None:
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive non-bool integer")
        if self.relaxation_mode not in {"cell", "ions_and_cell"}:
            raise ValueError("relaxation_mode must be 'cell' or 'ions_and_cell'")
        positive = {
            "stress_tolerance": self.stress_tolerance,
            "strain_tolerance": self.strain_tolerance,
            "cell_compliance": self.cell_compliance,
            "max_strain": self.max_strain,
            "line_search_min_step": self.line_search_min_step,
        }
        if any(
            isinstance(value, (bool, np.bool_))
            or not np.isfinite(value)
            or value <= 0.0
            for value in positive.values()
        ):
            raise ValueError("cell optimization tolerances and scales must be positive")
        if isinstance(self.external_pressure, (bool, np.bool_)) or not np.isfinite(
            self.external_pressure
        ):
            raise ValueError("external_pressure must be finite and non-bool")
        if self.max_strain >= 0.2:
            raise ValueError("max_strain must be smaller than 0.2")
        if not 0.0 < self.minimum_volume_ratio < 1.0:
            raise ValueError("minimum_volume_ratio must lie in (0, 1)")
        if not 0.0 < self.line_search_shrink < 1.0:
            raise ValueError("line_search_shrink must lie in (0, 1)")
        if not 0.0 < self.armijo_constant < 1.0:
            raise ValueError("armijo_constant must lie in (0, 1)")
        if (
            type(self.max_line_search_iterations) is not int
            or self.max_line_search_iterations <= 0
        ):
            raise ValueError(
                "max_line_search_iterations must be a positive non-bool integer"
            )
        if not isinstance(self.stress_config, PeriodicStressConfig):
            raise TypeError("stress_config must be PeriodicStressConfig")
        if not isinstance(self.ionic_config, PeriodicGeometryOptimizationConfig):
            raise TypeError("ionic_config must be PeriodicGeometryOptimizationConfig")

    def to_dict(self) -> dict[str, object]:
        """Return canonical JSON-safe variable-cell controls."""

        return {
            "max_steps": self.max_steps,
            "relaxation_mode": self.relaxation_mode,
            "external_pressure_hartree_per_bohr3": self.external_pressure,
            "stress_tolerance_hartree_per_bohr3": self.stress_tolerance,
            "strain_tolerance": self.strain_tolerance,
            "cell_compliance_bohr3_per_hartree": self.cell_compliance,
            "max_strain": self.max_strain,
            "minimum_volume_ratio": self.minimum_volume_ratio,
            "line_search_shrink": self.line_search_shrink,
            "line_search_min_step": self.line_search_min_step,
            "max_line_search_iterations": self.max_line_search_iterations,
            "armijo_constant": self.armijo_constant,
            "stress_config": self.stress_config.to_dict(),
            "ionic_config": self.ionic_config.to_dict(),
        }


@dataclass(frozen=True)
class PeriodicCellOptimizationStep:
    """One accepted periodic cell step."""

    index: int
    energy: float
    enthalpy: float
    volume: float
    pressure: float
    stress_residual: float
    max_force: float | None
    rms_force: float | None
    strain_norm: float
    accepted_step_size: float
    line_search_iterations: int
    armijo_limit: float
    ionic_steps: int
    cell: np.ndarray
    positions: np.ndarray

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe accepted cell step."""

        return {
            "index": self.index,
            "energy_hartree": self.energy,
            "enthalpy_hartree": self.enthalpy,
            "volume_bohr3": self.volume,
            "pressure_hartree_per_bohr3": self.pressure,
            "stress_residual_hartree_per_bohr3": self.stress_residual,
            "max_force_hartree_per_bohr": self.max_force,
            "rms_force_hartree_per_bohr": self.rms_force,
            "strain_norm": self.strain_norm,
            "accepted_step_size": self.accepted_step_size,
            "line_search_iterations": self.line_search_iterations,
            "armijo_limit_hartree": self.armijo_limit,
            "ionic_steps": self.ionic_steps,
            "cell_matrix_bohr": np.asarray(self.cell, dtype=np.float64).tolist(),
            "positions_bohr": np.asarray(self.positions, dtype=np.float64).tolist(),
        }

    @classmethod
    def _from_dict(cls, payload: Mapping[str, object]) -> PeriodicCellOptimizationStep:
        return cls(
            index=int(payload["index"]),
            energy=float(payload["energy_hartree"]),
            enthalpy=float(payload["enthalpy_hartree"]),
            volume=float(payload["volume_bohr3"]),
            pressure=float(payload["pressure_hartree_per_bohr3"]),
            stress_residual=float(payload["stress_residual_hartree_per_bohr3"]),
            max_force=(
                None
                if payload["max_force_hartree_per_bohr"] is None
                else float(payload["max_force_hartree_per_bohr"])
            ),
            rms_force=(
                None
                if payload["rms_force_hartree_per_bohr"] is None
                else float(payload["rms_force_hartree_per_bohr"])
            ),
            strain_norm=float(payload["strain_norm"]),
            accepted_step_size=float(payload["accepted_step_size"]),
            line_search_iterations=int(payload["line_search_iterations"]),
            armijo_limit=float(payload["armijo_limit_hartree"]),
            ionic_steps=int(payload["ionic_steps"]),
            cell=np.asarray(payload["cell_matrix_bohr"], dtype=np.float64),
            positions=np.asarray(payload["positions_bohr"], dtype=np.float64),
        )


@dataclass(frozen=True)
class PeriodicCellOptimizationResult:
    """Result of periodic cell-only or coupled ion/cell relaxation."""

    status: PeriodicCellStatus
    convergence_reason: str
    initial_system: PeriodicDFTSystem
    final_system: PeriodicDFTSystem
    final_scf: PeriodicSCFResult | None
    final_stress: PeriodicStressResult | None
    final_force: PeriodicForceResult | None
    steps: tuple[PeriodicCellOptimizationStep, ...]
    config: PeriodicCellOptimizationConfig
    elapsed_ms: float
    scf_evaluations: int
    stress_evaluations: int
    line_search_evaluations: int
    ionic_scf_evaluations: int
    lineage: tuple[str, ...] = ()
    checkpoint_manifest: dict[str, object] | None = None

    @property
    def converged(self) -> bool:
        """Whether every configured cell and ion gate passed."""

        return self.status == "converged"

    @property
    def final_enthalpy(self) -> float | None:
        """Return final generalized enthalpy in Hartree."""

        if self.final_scf is None:
            return None
        return float(
            self.final_scf.total_energy
            + self.config.external_pressure * self.final_system.grid.volume
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe result without dense electronic arrays."""

        return {
            "status": self.status,
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
            "elapsed_ms": self.elapsed_ms,
            "accepted_step_count": len(self.steps),
            "final_energy_hartree": (
                None if self.final_scf is None else self.final_scf.total_energy
            ),
            "final_enthalpy_hartree": self.final_enthalpy,
            "final_cell_matrix_bohr": np.asarray(
                self.final_system.grid.cell.matrix,
                dtype=np.float64,
            ).tolist(),
            "final_positions_bohr": np.asarray(
                self.final_system.positions,
                dtype=np.float64,
            ).tolist(),
            "final_stress": (
                None if self.final_stress is None else self.final_stress.to_dict()
            ),
            "final_force": (
                None if self.final_force is None else self.final_force.to_dict()
            ),
            "config": self.config.to_dict(),
            "scf_evaluations": self.scf_evaluations,
            "stress_evaluations": self.stress_evaluations,
            "line_search_evaluations": self.line_search_evaluations,
            "ionic_scf_evaluations": self.ionic_scf_evaluations,
            "steps": [step.to_dict() for step in self.steps],
            "lineage": list(self.lineage),
            "checkpoint_manifest_sha256": (
                None
                if self.checkpoint_manifest is None
                else self.checkpoint_manifest.get("manifest_sha256")
            ),
        }
