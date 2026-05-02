"""Reusable molecular dynamics protocol runners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from mlx_atomistic.constraints import DistanceConstraints
from mlx_atomistic.core import Cell
from mlx_atomistic.md import (
    ForceTerm,
    LangevinThermostat,
    NVTResult,
    SimulationConfig,
    kinetic_energy,
    simulate_nvt,
)
from mlx_atomistic.minimize import MinimizationResult, minimize_energy
from mlx_atomistic.neighbors import NeighborListManager
from mlx_atomistic.units import MDUnitSystem

NVT_PROOF_MODE = "short_nvt"
SUPPORTED_GPCRMD_PROOF_ENSEMBLE = "nvt"


@dataclass(frozen=True)
class ProtocolCompatibilityReport:
    """Compatibility decision for the GPCRmd MLX proof protocol."""

    accepted: bool
    ensemble: str
    proof_mode: str
    barostat: str
    blockers: tuple[str, ...]
    metadata: dict[str, Any]


class ProtocolCompatibilityError(ValueError):
    """Raised when a requested protocol is outside the GPCRmd MLX proof gate."""

    def __init__(self, report: ProtocolCompatibilityReport) -> None:
        self.report = report
        self.blockers = report.blockers
        blockers = ", ".join(report.blockers)
        super().__init__(f"unsupported protocol blockers: {blockers}")


@dataclass(frozen=True)
class MinimizeThenNVTProtocol:
    """Minimal production-style protocol: minimize, equilibrate, then sample NVT."""

    minimize_steps: int = 200
    minimize_step_size: float = 1e-4
    force_tolerance: float = 1e-3
    equilibration_steps: int = 1000
    production_steps: int = 5000
    dt: float = 0.001
    sample_interval: int = 100
    temperature: float = 300.0
    friction: float = 1.0
    seed: int | None = 7
    diagnostic_interval: int = 1
    compile_force_evaluator: bool = True
    ensemble: str = "NVT"
    proof_mode: str = NVT_PROOF_MODE
    barostat: str | bool | None = None
    npt_barostat: bool = False
    membrane_barostat: str | bool | None = None

    def compatibility_report(
        self,
        *,
        raise_on_blockers: bool = False,
    ) -> ProtocolCompatibilityReport:
        """Return the fail-closed GPCRmd proof compatibility decision."""

        return validate_gpcrmd_protocol_request(
            {
                "ensemble": self.ensemble,
                "proof_mode": self.proof_mode,
                "barostat": self.barostat,
                "npt_barostat": self.npt_barostat,
                "membrane_barostat": self.membrane_barostat,
            },
            raise_on_blockers=raise_on_blockers,
        )

    def protocol_metadata(self) -> dict[str, Any]:
        """Return normalized metadata for accepted NVT proof runs."""

        return self.compatibility_report(raise_on_blockers=True).metadata


@dataclass(frozen=True)
class ProtocolResult:
    """Outputs from a minimize/equilibrate/production workflow."""

    minimization: MinimizationResult
    equilibration: NVTResult | None
    production: NVTResult
    protocol_metadata: dict[str, Any] = field(default_factory=dict)


def validate_gpcrmd_protocol_request(
    protocol_metadata: Mapping[str, Any] | None = None,
    *,
    ensemble: str | None = None,
    proof_mode: str | None = None,
    barostat: str | bool | None = None,
    npt_barostat: bool | None = None,
    membrane_barostat: str | bool | None = None,
    raise_on_blockers: bool = False,
) -> ProtocolCompatibilityReport:
    """Validate the current GPCRmd proof protocol gate.

    The first selected GPCRmd proof is deliberately NVT-only. Pressure-coupled
    requests fail before integration with exact blocker names.
    """

    metadata = dict(protocol_metadata or {})
    requested_ensemble = str(
        _first_present(ensemble, metadata.get("ensemble"), "NVT")
    ).strip()
    requested_proof_mode = str(
        _first_present(proof_mode, metadata.get("proof_mode"), NVT_PROOF_MODE)
    ).strip()
    requested_barostat = _first_present(
        barostat,
        metadata.get("barostat"),
        metadata.get("barostat_type"),
    )
    requested_npt_barostat = _first_present(
        npt_barostat,
        metadata.get("npt_barostat"),
        False,
    )
    requested_membrane_barostat = _first_present(
        membrane_barostat,
        metadata.get("membrane_barostat"),
        metadata.get("membrane-barostat"),
    )

    normalized_ensemble = requested_ensemble.lower()
    normalized_proof_mode = requested_proof_mode.lower()
    npt_barostat_requested = (
        "npt" in normalized_ensemble or _is_requested(requested_npt_barostat)
    )
    barostat_requested = _is_requested(requested_barostat)
    membrane_barostat_requested = _is_requested(requested_membrane_barostat)
    blockers: list[str] = []

    if npt_barostat_requested:
        blockers.append("npt_barostat")
    elif normalized_ensemble != SUPPORTED_GPCRMD_PROOF_ENSEMBLE:
        blockers.append("unsupported_ensemble")
    if normalized_proof_mode != NVT_PROOF_MODE:
        blockers.append("unsupported_proof_mode")
    if barostat_requested:
        blockers.append("barostat")
    if membrane_barostat_requested:
        blockers.append("membrane_barostat")

    blocker_tuple = tuple(dict.fromkeys(blockers))
    barostat_value = "none"
    if barostat_requested:
        barostat_value = str(requested_barostat)
    elif npt_barostat_requested:
        barostat_value = "requested_by_npt_ensemble"
    elif membrane_barostat_requested:
        barostat_value = "membrane"
    barostat_status = (
        "unsupported_requested"
        if npt_barostat_requested or barostat_requested or membrane_barostat_requested
        else "not_required_for_nvt_proof"
    )
    report = ProtocolCompatibilityReport(
        accepted=not blocker_tuple,
        ensemble="NVT" if normalized_ensemble == "nvt" else requested_ensemble,
        proof_mode=NVT_PROOF_MODE,
        barostat=barostat_value,
        blockers=blocker_tuple,
        metadata={
            "ensemble": "NVT" if normalized_ensemble == "nvt" else requested_ensemble,
            "proof_mode": NVT_PROOF_MODE,
            "barostat": barostat_value,
            "barostat_status": barostat_status,
            "npt_barostat": npt_barostat_requested,
            "membrane_barostat": membrane_barostat_requested,
            "unsupported_protocol_blockers": list(blocker_tuple),
        },
    )
    if blocker_tuple and raise_on_blockers:
        raise ProtocolCompatibilityError(report)
    return report


def run_minimize_then_nvt(
    positions,
    velocities,
    masses,
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...],
    *,
    protocol: MinimizeThenNVTProtocol | None = None,
    cell: Cell | None = None,
    constraints: DistanceConstraints | None = None,
    unit_system: MDUnitSystem | None = None,
    neighbor_manager: NeighborListManager | None = None,
) -> ProtocolResult:
    """Run minimization, optional NVT equilibration, and NVT production in MLX."""

    if protocol is None:
        protocol = MinimizeThenNVTProtocol()
    protocol_report = protocol.compatibility_report(raise_on_blockers=True)
    simulation_units = _simulation_unit_kwargs(unit_system)

    minimized = minimize_energy(
        positions,
        force_terms,
        cell=cell,
        max_steps=protocol.minimize_steps,
        step_size=protocol.minimize_step_size,
        force_tolerance=protocol.force_tolerance,
        neighbor_manager=neighbor_manager,
    )

    thermostat = LangevinThermostat(
        temperature=protocol.temperature,
        friction=protocol.friction,
        seed=protocol.seed,
    )
    equilibration = None
    start_positions = minimized.positions
    start_velocities = velocities
    if constraints is not None:
        start_positions, _ = constraints.apply_positions(start_positions, masses, cell)
        start_velocities = constraints.apply_velocities(
            start_positions,
            start_velocities,
            masses,
            cell,
        )
    start_velocities = _rescale_velocities_to_temperature(
        start_velocities,
        masses,
        temperature=protocol.temperature,
        constraints=constraints,
        unit_system=unit_system,
    )
    if protocol.equilibration_steps > 0:
        equilibration = simulate_nvt(
            start_positions,
            start_velocities,
            masses=masses,
            cell=cell,
            force_terms=force_terms,
            config=SimulationConfig(
                dt=protocol.dt,
                steps=protocol.equilibration_steps,
                sample_interval=max(1, protocol.equilibration_steps),
                diagnostic_interval=max(1, protocol.equilibration_steps),
                compile_force_evaluator=protocol.compile_force_evaluator,
                **simulation_units,
            ),
            thermostat=thermostat,
            constraints=constraints,
            neighbor_manager=neighbor_manager,
        )
        start_positions = equilibration.final_state.positions
        start_velocities = equilibration.final_state.velocities
        if constraints is not None:
            start_velocities = constraints.apply_velocities(
                start_positions,
                start_velocities,
                masses,
                cell,
            )
        start_velocities = _rescale_velocities_to_temperature(
            start_velocities,
            masses,
            temperature=protocol.temperature,
            constraints=constraints,
            unit_system=unit_system,
        )

    production = simulate_nvt(
        start_positions,
        start_velocities,
        masses=masses,
        cell=cell,
        force_terms=force_terms,
        config=SimulationConfig(
            dt=protocol.dt,
            steps=protocol.production_steps,
            sample_interval=protocol.sample_interval,
            diagnostic_interval=protocol.diagnostic_interval,
            compile_force_evaluator=protocol.compile_force_evaluator,
            **simulation_units,
        ),
        thermostat=thermostat,
        constraints=constraints,
        neighbor_manager=neighbor_manager,
    )
    return ProtocolResult(
        minimization=minimized,
        equilibration=equilibration,
        production=production,
        protocol_metadata=protocol_report.metadata,
    )


def _simulation_unit_kwargs(unit_system: MDUnitSystem | None) -> dict[str, float]:
    if unit_system is None:
        return {}
    return {
        "kinetic_energy_scale": unit_system.kinetic_energy_scale,
        "force_to_acceleration_scale": unit_system.force_to_acceleration_scale,
        "boltzmann_constant": unit_system.boltzmann_constant,
    }


def _rescale_velocities_to_temperature(
    velocities,
    masses,
    *,
    temperature: float,
    constraints: DistanceConstraints | None,
    unit_system: MDUnitSystem | None,
):
    if temperature <= 0.0:
        return velocities
    units = _simulation_unit_kwargs(unit_system)
    kinetic_energy_scale = units.get("kinetic_energy_scale", 1.0)
    boltzmann_constant = units.get("boltzmann_constant", 1.0)
    constraint_count = 0 if constraints is None else int(constraints.pairs.shape[0])
    dof = max(1, int(masses.shape[0]) * 3 - constraint_count - 3)
    current_ke = kinetic_energy(
        velocities,
        masses,
        kinetic_energy_scale=kinetic_energy_scale,
    )
    target_ke = 0.5 * dof * boltzmann_constant * temperature
    scale = mx.sqrt(mx.maximum(target_ke / mx.maximum(current_ke, 1e-12), 0.0))
    return velocities * scale


def _first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _is_requested(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        return normalized not in {
            "",
            "0",
            "false",
            "no",
            "none",
            "off",
            "disabled",
            "not_requested",
            "not_required",
        }
    return bool(value)


__all__ = [
    "MinimizeThenNVTProtocol",
    "NVT_PROOF_MODE",
    "ProtocolCompatibilityError",
    "ProtocolCompatibilityReport",
    "ProtocolResult",
    "run_minimize_then_nvt",
    "validate_gpcrmd_protocol_request",
]
