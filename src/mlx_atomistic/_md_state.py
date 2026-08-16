"""State, configuration, and result contracts for molecular dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell
from mlx_atomistic.force_evaluation import ForceTerm
from mlx_atomistic.runtime import (
    VIRIAL_SUPPORT_FINITE_DIFFERENCE_ORACLE,
    VIRIAL_SUPPORT_UNSUPPORTED,
    normalize_virial_support,
)
from mlx_atomistic.virtual_sites import VirtualSiteManager


@dataclass(frozen=True)
class StepState:
    """Single MD state."""

    positions: mx.array
    velocities: mx.array
    forces: mx.array
    potential_energy: mx.array
    kinetic_energy: mx.array

    @property
    def total_energy(self) -> mx.array:
        """Potential plus kinetic energy."""

        return self.potential_energy + self.kinetic_energy


@dataclass(frozen=True)
class SimulationResult:
    """Trajectory and diagnostics from an MD run."""

    positions: mx.array
    velocities: mx.array
    potential_energy: mx.array
    kinetic_energy: mx.array
    total_energy: mx.array
    temperature: mx.array


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for molecular dynamics."""

    dt: float = 0.005
    steps: int = 100
    sample_interval: int = 1
    kinetic_energy_scale: float = 1.0
    force_to_acceleration_scale: float = 1.0
    boltzmann_constant: float = 1.0
    evaluation_interval: int = 25
    diagnostic_interval: int = 1
    compile_force_evaluator: bool = False
    pressure_diagnostics: bool = True
    pressure_virial_mode: str = VIRIAL_SUPPORT_FINITE_DIFFERENCE_ORACLE
    initial_step: int = 0
    initial_time: float = 0.0
    virtual_sites: VirtualSiteManager | None = None
    block_size: int = 1
    center_of_mass_motion_interval: int | None = None
    wrap_positions: bool = True
    runtime_profile: bool = False

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            msg = "dt must be positive"
            raise ValueError(msg)
        if self.steps < 0:
            msg = "steps must be non-negative"
            raise ValueError(msg)
        if self.sample_interval <= 0:
            msg = "sample_interval must be positive"
            raise ValueError(msg)
        if self.kinetic_energy_scale <= 0.0:
            msg = "kinetic_energy_scale must be positive"
            raise ValueError(msg)
        if self.force_to_acceleration_scale <= 0.0:
            msg = "force_to_acceleration_scale must be positive"
            raise ValueError(msg)
        if self.boltzmann_constant <= 0.0:
            msg = "boltzmann_constant must be positive"
            raise ValueError(msg)
        pressure_virial_mode = normalize_virial_support(self.pressure_virial_mode)
        if pressure_virial_mode == VIRIAL_SUPPORT_UNSUPPORTED:
            msg = "pressure_virial_mode cannot be unsupported"
            raise ValueError(msg)
        object.__setattr__(self, "pressure_virial_mode", pressure_virial_mode)
        if self.evaluation_interval <= 0:
            msg = "evaluation_interval must be positive"
            raise ValueError(msg)
        if self.diagnostic_interval <= 0:
            msg = "diagnostic_interval must be positive"
            raise ValueError(msg)
        if self.initial_step < 0:
            msg = "initial_step must be non-negative"
            raise ValueError(msg)
        if self.initial_time < 0.0:
            msg = "initial_time must be non-negative"
            raise ValueError(msg)
        if self.block_size < 1:
            msg = "block_size must be a positive integer (1 = per-step execution)"
            raise ValueError(msg)
        if (
            self.center_of_mass_motion_interval is not None
            and self.center_of_mass_motion_interval <= 0
        ):
            msg = "center_of_mass_motion_interval must be positive or None"
            raise ValueError(msg)


@dataclass(frozen=True)
class SimulationState:
    """Current NVE simulation state."""

    positions: mx.array
    velocities: mx.array
    masses: mx.array
    forces: mx.array
    step: int = 0
    time: float = 0.0


@dataclass(frozen=True)
class ReporterEvent:
    """State exposed to runtime reporter callbacks."""

    ensemble: str
    event_type: str
    step: int
    time: float
    state: SimulationState
    potential_energy: mx.array | None = None
    kinetic_energy: mx.array | None = None
    total_energy: mx.array | None = None
    temperature: mx.array | None = None
    energy_by_term: dict[str, mx.array] = field(default_factory=dict)
    virial_tensor: mx.array | None = None
    pressure_tensor: mx.array | None = None
    pressure: mx.array | None = None
    pair_count: int | mx.array | None = None
    rebuild_count: int | mx.array | None = None
    constraint_max_error: mx.array | None = None
    thermostat: dict[str, Any] = field(default_factory=dict)
    barostat: dict[str, Any] = field(default_factory=dict)


class RuntimeReporter(Protocol):
    """Callable observer for sampled frames and diagnostic state."""

    def __call__(self, event: ReporterEvent) -> None:
        """Observe one runtime event."""


@dataclass(frozen=True)
class NVEResult:
    """Sparse trajectory and per-step diagnostics from an NVE simulation."""

    sampled_positions: mx.array
    sampled_velocities: mx.array
    sampled_steps: mx.array
    sampled_time: mx.array
    diagnostic_steps: mx.array
    diagnostic_time: mx.array
    potential_energy: mx.array
    kinetic_energy: mx.array
    total_energy: mx.array
    potential_energy_by_term: dict[str, mx.array]
    temperature: mx.array
    virial_tensor: mx.array
    pressure_tensor: mx.array
    pressure: mx.array
    pair_count: mx.array
    rebuild_count: mx.array
    constraint_max_error: mx.array
    final_state: SimulationState
    nonbonded_report: dict[str, int | float | str | None] = field(default_factory=dict)
    runtime_sync_report: dict[str, int | float] = field(default_factory=dict)

    @property
    def energy_drift(self) -> mx.array:
        """Total energy minus the initial total energy for each diagnostic step."""

        return self.total_energy - self.total_energy[0]

    @property
    def max_energy_drift(self) -> mx.array:
        """Maximum absolute total-energy drift over the run."""

        return mx.max(mx.abs(self.energy_drift))

    @property
    def relative_energy_drift(self) -> mx.array:
        """Energy drift normalized by the absolute initial total energy."""

        denominator = mx.maximum(mx.abs(self.total_energy[0]), 1e-12)
        return self.energy_drift / denominator


@dataclass(frozen=True)
class LangevinThermostat:
    """Langevin thermostat parameters in reduced units."""

    temperature: float = 1.0
    friction: float = 1.0
    seed: int | None = None
    rng_step_offset: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            msg = "temperature must be non-negative"
            raise ValueError(msg)
        if self.friction < 0.0:
            msg = "friction must be non-negative"
            raise ValueError(msg)
        if self.rng_step_offset is not None and self.rng_step_offset < 0:
            msg = "rng_step_offset must be non-negative when provided"
            raise ValueError(msg)


@dataclass(frozen=True)
class NoseHooverThermostat:
    """Deterministic single-variable Nose-Hoover thermostat parameters."""

    temperature: float = 1.0
    relaxation_time: float = 0.1
    thermal_mass: float | None = None
    chain_position: float = 0.0
    chain_velocity: float = 0.0

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            msg = "temperature must be positive for Nose-Hoover"
            raise ValueError(msg)
        if self.relaxation_time <= 0.0:
            msg = "relaxation_time must be positive"
            raise ValueError(msg)
        if self.thermal_mass is not None and self.thermal_mass <= 0.0:
            msg = "thermal_mass must be positive when provided"
            raise ValueError(msg)


Thermostat = LangevinThermostat | NoseHooverThermostat


@dataclass(frozen=True)
class NVTResult:
    """Sparse trajectory and per-step diagnostics from an NVT simulation."""

    sampled_positions: mx.array
    sampled_velocities: mx.array
    sampled_steps: mx.array
    sampled_time: mx.array
    diagnostic_steps: mx.array
    diagnostic_time: mx.array
    potential_energy: mx.array
    kinetic_energy: mx.array
    total_energy: mx.array
    potential_energy_by_term: dict[str, mx.array]
    temperature: mx.array
    virial_tensor: mx.array
    pressure_tensor: mx.array
    pressure: mx.array
    pair_count: mx.array
    rebuild_count: mx.array
    constraint_max_error: mx.array
    final_state: SimulationState
    target_temperature: float
    thermostat_metadata: dict[str, Any] = field(default_factory=dict)
    nonbonded_report: dict[str, int | float | str | None] = field(default_factory=dict)
    runtime_sync_report: dict[str, int | float] = field(default_factory=dict)
    route_profile: dict[str, Any] = field(default_factory=dict)

    @property
    def temperature_error(self) -> mx.array:
        """Instantaneous temperature minus the target thermostat temperature."""

        return self.temperature - self.target_temperature


@dataclass(frozen=True)
class _NVTBoundaryDiagnostics:
    """State and diagnostics already committed at an NPT segment boundary."""

    potential_energy: mx.array
    forces: mx.array
    energy_by_term: dict[str, mx.array]
    virial_tensor: mx.array
    pressure_tensor: mx.array
    pressure: mx.array
    constraint_error: mx.array


@dataclass(frozen=True)
class MonteCarloBarostat:
    """Monte Carlo barostat parameters for isotropic, anisotropic, and membrane NPT."""

    pressure: float = 1.0
    temperature: float = 1.0
    interval: int = 25
    max_log_volume_scale: float = 0.02
    seed: int | None = 11
    mode: str = "isotropic"
    axes: tuple[bool, bool, bool] = (True, True, True)
    membrane_plane: str = "xy"
    normal_axis: str = "z"

    def __post_init__(self) -> None:
        if self.pressure < 0.0:
            msg = "pressure must be non-negative"
            raise ValueError(msg)
        if self.temperature <= 0.0:
            msg = "temperature must be positive"
            raise ValueError(msg)
        if self.interval <= 0:
            msg = "barostat interval must be positive"
            raise ValueError(msg)
        if self.max_log_volume_scale <= 0.0:
            msg = "max_log_volume_scale must be positive"
            raise ValueError(msg)
        if self.max_log_volume_scale >= float(np.log(2.0)):
            msg = "max_log_volume_scale must be smaller than log(2)"
            raise ValueError(msg)
        mode = _normalize_barostat_mode(self.mode)
        object.__setattr__(self, "mode", mode)
        axes = tuple(bool(axis) for axis in self.axes)
        if len(axes) != 3:
            msg = "barostat axes must contain three booleans"
            raise ValueError(msg)
        object.__setattr__(self, "axes", axes)
        if mode == "anisotropic" and not any(axes):
            msg = "anisotropic barostat requires at least one enabled axis"
            raise ValueError(msg)
        plane_axes = _barostat_plane_axes(self.membrane_plane)
        normal_axis = _barostat_axis_index(self.normal_axis)
        if mode == "membrane" and normal_axis in plane_axes:
            msg = "membrane normal_axis must be outside membrane_plane"
            raise ValueError(msg)
        object.__setattr__(self, "membrane_plane", "".join("xyz"[axis] for axis in plane_axes))
        object.__setattr__(self, "normal_axis", "xyz"[normal_axis])


@dataclass(frozen=True)
class BarostatProposal:
    """One explicit Monte Carlo cell proposal and its proposal-density ratio."""

    scale_factors: tuple[float, float, float]
    axis: int | None
    log_reverse_over_forward: float
    kernel: str
    volume_step: float
    source_pme_plan_fingerprints: tuple[str, ...] = ()
    candidate_pme_plan_fingerprints: tuple[str, ...] = ()
    delta_energy: float | None = None
    log_acceptance: float | None = None
    log_uniform_draw: float | None = None


@dataclass(frozen=True)
class NPTResult:
    """NPT production result with delegated NVT trajectory fields."""

    production: NVTResult
    final_state: SimulationState
    final_cell: Cell
    final_force_terms: tuple[ForceTerm, ...]
    cell_lengths: mx.array
    cell_matrix: mx.array
    volume: mx.array
    target_pressure: float
    barostat_attempts: int
    barostat_accepted: int
    barostat_metadata: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str):
        return getattr(self.production, name)

    @property
    def cell_history(self) -> mx.array:
        """Return the sampled cell-matrix history for pressure-coupled runs."""

        return self.cell_matrix


def _normalize_barostat_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "isotropic": "isotropic",
        "iso": "isotropic",
        "anisotropic": "anisotropic",
        "aniso": "anisotropic",
        "membrane": "membrane",
        "semi_isotropic": "membrane",
        "semiisotropic": "membrane",
    }
    if normalized not in aliases:
        msg = "barostat mode must be isotropic, anisotropic, or membrane"
        raise ValueError(msg)
    return aliases[normalized]


def _barostat_axis_index(axis: str | int) -> int:
    if isinstance(axis, int):
        if axis in {0, 1, 2}:
            return axis
        msg = "barostat axis must be x, y, z, 0, 1, or 2"
        raise ValueError(msg)
    normalized = str(axis).strip().lower()
    if normalized in {"x", "0"}:
        return 0
    if normalized in {"y", "1"}:
        return 1
    if normalized in {"z", "2"}:
        return 2
    msg = "barostat axis must be x, y, z, 0, 1, or 2"
    raise ValueError(msg)


def _barostat_plane_axes(plane: str | tuple[str | int, str | int]) -> tuple[int, int]:
    if isinstance(plane, tuple):
        axes = tuple(_barostat_axis_index(axis) for axis in plane)
    else:
        normalized = str(plane).strip().lower().replace("-", "").replace("_", "")
        axes = tuple(_barostat_axis_index(axis) for axis in normalized)
    if len(axes) != 2 or len(set(axes)) != 2:
        msg = "membrane_plane must name two distinct axes"
        raise ValueError(msg)
    return tuple(sorted(axes))


def _validate_barostat_cell_support(cell: Cell, barostat: MonteCarloBarostat) -> None:
    volume = float(np.asarray(cell.volume))
    if not np.isfinite(volume) or volume <= 0.0:
        msg = "NPT barostat requires a positive finite cell volume"
        raise ValueError(msg)
    if barostat.mode == "anisotropic" and not any(barostat.axes):
        msg = "anisotropic barostat requires at least one enabled axis"
        raise ValueError(msg)
