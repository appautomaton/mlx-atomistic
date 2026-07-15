"""Run prepared systems with the MLX MD engine."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_atomistic.artifacts import (
    MLXCompatibilityError,
    PreparedMLXArtifact,
    artifact_readiness_report,
    build_mlx_system_from_artifact,
    load_prepared_mlx_artifact,
    validate_mlx_compatibility,
)
from mlx_atomistic.io import (
    TrajectoryRecord,
    load_npz_trajectory,
    load_simulation_checkpoint,
    save_npz_trajectory,
    save_simulation_checkpoint,
    trajectory_record_from_result,
)
from mlx_atomistic.md import (
    LangevinThermostat,
    MonteCarloBarostat,
    RuntimeReporter,
    SimulationConfig,
    simulate_npt,
    simulate_nvt,
)
from mlx_atomistic.minimize import minimize_energy
from mlx_atomistic.neighbors import NeighborListManager
from mlx_atomistic.pme import pme_readiness_report
from mlx_atomistic.prep.gpcrmd import (
    GPCRMD_IMPORT_REPORT_NAME,
    GPCRmdInspectionError,
    GPCRmdTargetError,
    attempt_gpcrmd_prepared_artifact_import,
    write_gpcrmd_import_report,
)
from mlx_atomistic.prep.io import (
    JSON_NAME,
    NPZ_ARRAY_NAMES,
    NPZ_NAME,
    OPTIONAL_NPZ_ARRAY_DEFAULTS,
    VIEW_PDB_NAME,
    load_prepared_system,
    write_view_pdb,
)
from mlx_atomistic.prep.schema import PreparedSystem
from mlx_atomistic.protocols import (
    MinimizeThenNVTProtocol,
    ProtocolCompatibilityError,
    protocol_readiness_report,
    run_minimize_then_nvt,
    validate_gpcrmd_protocol_request,
)
from mlx_atomistic.runtime import get_platform_boundary_report
from mlx_atomistic.steering import SteeredCOMBiasPotential, simulate_steered_nvt
from mlx_atomistic.trajectory_adapters import write_mdtraj_trajectory
from mlx_atomistic.units import ATM_TO_KJ_PER_MOL_ANGSTROM3, MDUnitSystem

TRAJECTORY_NAME = "trajectory.npz"
STEERED_TRAJECTORY_NAME = "steered_trajectory.npz"
GPCRMD_RUN_REPORT_NAME = "gpcrmd_mlx_run_report.json"
PRESSURE_DIAGNOSTIC_ATOM_LIMIT = 50_000
GPCRMD_NEIGHBOR_SKIN = 2.5
GPCRMD_NEIGHBOR_WORKERS = max(1, min(8, os.cpu_count() or 1))


def _artifact_from_prepared_system(
    prepared: PreparedSystem,
    *,
    require_production: bool,
) -> PreparedMLXArtifact:
    metadata = prepared.metadata.to_json_dict()
    arrays: dict[str, np.ndarray] = {
        name: np.asarray(getattr(prepared, name)) for name in NPZ_ARRAY_NAMES
    }
    for name in OPTIONAL_NPZ_ARRAY_DEFAULTS:
        arrays[name] = np.asarray(getattr(prepared, name))
    unit_system = validate_mlx_compatibility(
        metadata,
        require_production=require_production,
        arrays=arrays,
    )
    return PreparedMLXArtifact(
        base_dir=Path("."),
        metadata=metadata,
        arrays=arrays,
        unit_system=unit_system,
    )


def build_mlx_system(
    prepared: PreparedSystem,
    *,
    restraint_k: float = 0.0,
    require_production: bool = False,
    receptor_mass_scale: float | None = None,
    constraint_max_iterations: int = 20,
):
    """Convert a prepared artifact into an MLX system and supported force terms.

    `receptor_mass_scale` is accepted for backward compatibility with older
    notebook demos. Production MD should keep physical masses unchanged.
    """

    if receptor_mass_scale not in {None, 1.0}:
        msg = "receptor_mass_scale is not supported for production MLX artifacts"
        raise ValueError(msg)
    artifact = _artifact_from_prepared_system(
        prepared,
        require_production=require_production,
    )
    system, terms, _ = build_mlx_system_from_artifact(
        artifact,
        restraint_k=restraint_k,
        constraint_max_iterations=constraint_max_iterations,
    )
    return system, tuple(terms)


def initialize_velocities(
    prepared: PreparedSystem,
    masses: np.ndarray,
    *,
    temperature: float,
    seed: int | None,
    kinetic_energy_scale: float = 1.0,
    boltzmann_constant: float = 1.0,
) -> np.ndarray:
    """Create deterministic Maxwell-like velocities if none are stored."""

    velocities = np.asarray(prepared.velocities, dtype=np.float32).copy()
    if not np.allclose(velocities, 0.0) or temperature <= 0.0:
        return velocities

    rng = np.random.default_rng(seed)
    dynamic_mask = np.ones((prepared.atom_count,), dtype=bool)
    variance = (
        boltzmann_constant
        * temperature
        / (kinetic_energy_scale * np.maximum(masses[dynamic_mask], 1e-6))
    )
    random_velocities = rng.normal(
        0.0,
        1.0,
        size=(int(np.count_nonzero(dynamic_mask)), 3),
    )
    velocities[dynamic_mask] = random_velocities * np.sqrt(variance)[:, None]
    total_mass = float(np.sum(masses[dynamic_mask]))
    com_velocity = (
        np.sum(velocities[dynamic_mask] * masses[dynamic_mask, None], axis=0) / total_mass
    )
    velocities[dynamic_mask] -= com_velocity
    return velocities.astype(np.float32)


def _initialize_system_velocities(
    system,
    masses: np.ndarray,
    *,
    temperature: float,
    seed: int | None,
    kinetic_energy_scale: float = 1.0,
    boltzmann_constant: float = 1.0,
) -> np.ndarray:
    velocities = np.asarray(system.velocities, dtype=np.float32).copy()
    if not np.allclose(velocities, 0.0) or temperature <= 0.0:
        return velocities

    rng = np.random.default_rng(seed)
    dynamic_mask = np.ones((len(masses),), dtype=bool)
    variance = (
        boltzmann_constant
        * temperature
        / (kinetic_energy_scale * np.maximum(masses[dynamic_mask], 1e-6))
    )
    random_velocities = rng.normal(
        0.0,
        1.0,
        size=(int(np.count_nonzero(dynamic_mask)), 3),
    )
    velocities[dynamic_mask] = random_velocities * np.sqrt(variance)[:, None]
    total_mass = float(np.sum(masses[dynamic_mask]))
    com_velocity = (
        np.sum(velocities[dynamic_mask] * masses[dynamic_mask, None], axis=0) / total_mass
    )
    velocities[dynamic_mask] -= com_velocity
    return velocities.astype(np.float32)


def _simulation_config_with_virtual_sites(*, virtual_sites=None, **kwargs) -> SimulationConfig:
    return SimulationConfig(virtual_sites=virtual_sites, **kwargs)


class _VirtualSiteForceAdapter:
    name = "virtual_site_force_adapter"
    supports_virial = True

    def __init__(self, force_terms, virtual_sites):
        self.force_terms = tuple(force_terms)
        self.virtual_sites = virtual_sites

    def energy_forces(self, positions, cell=None, pairs=None):
        del pairs
        eval_positions = self.virtual_sites.extend_positions(positions)
        total = mx.array(0.0, dtype=mx.float32)
        total_forces = mx.zeros_like(eval_positions)
        for term in self.force_terms:
            energy, forces = term.energy_forces(eval_positions, cell=cell, pairs=None)
            total = total + energy
            total_forces = total_forces + forces
        return total, self.virtual_sites.redistribute_forces(total_forces, eval_positions)


def _run_virtual_site_minimize_then_nvt(
    positions,
    velocities,
    masses,
    force_terms,
    *,
    protocol: MinimizeThenNVTProtocol,
    virtual_sites,
    cell,
    constraints,
    unit_system,
    pressure_diagnostics=True,
    reporters=None,
):
    simulation_units = {}
    if unit_system is not None:
        simulation_units = {
            "kinetic_energy_scale": unit_system.kinetic_energy_scale,
            "force_to_acceleration_scale": unit_system.force_to_acceleration_scale,
            "boltzmann_constant": unit_system.boltzmann_constant,
        }
    minimized = minimize_energy(
        positions,
        _VirtualSiteForceAdapter(force_terms, virtual_sites),
        cell=cell,
        max_steps=protocol.minimize_steps,
        step_size=protocol.minimize_step_size,
        force_tolerance=protocol.force_tolerance,
    )
    thermostat = LangevinThermostat(
        temperature=protocol.temperature,
        friction=protocol.friction,
        seed=protocol.seed,
    )
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
    start_velocities = _project_and_rescale_velocities(
        np.asarray(start_velocities, dtype=np.float32),
        positions=np.asarray(start_positions, dtype=np.float32),
        masses=masses,
        constraints=None,
        cell=cell,
        temperature=protocol.temperature,
        kinetic_energy_scale=simulation_units.get("kinetic_energy_scale", 1.0),
        boltzmann_constant=simulation_units.get("boltzmann_constant", 1.0),
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
                pressure_diagnostics=pressure_diagnostics,
                virtual_sites=virtual_sites,
                **simulation_units,
            ),
            thermostat=thermostat,
            constraints=constraints,
        )
        start_positions = equilibration.final_state.positions
        start_velocities = equilibration.final_state.velocities
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
            pressure_diagnostics=pressure_diagnostics,
            virtual_sites=virtual_sites,
            **simulation_units,
        ),
        thermostat=thermostat,
        constraints=constraints,
        reporters=reporters,
    )
    return SimpleNamespace(minimization=minimized, equilibration=None, production=production)


def _temperature_degrees_of_freedom(atom_count: int, constraint_count: int) -> int:
    dof = 3 * int(atom_count) - int(constraint_count)
    if atom_count > 1:
        dof -= 3
    return max(1, dof)


def _project_and_rescale_velocities(
    velocities: np.ndarray,
    *,
    positions: np.ndarray,
    masses: np.ndarray,
    constraints,
    cell,
    temperature: float,
    kinetic_energy_scale: float,
    boltzmann_constant: float,
) -> np.ndarray:
    if constraints is not None:
        velocities = np.asarray(
            constraints.apply_velocities(positions, velocities, masses, cell),
            dtype=np.float32,
        )
    if temperature <= 0.0:
        return velocities.astype(np.float32)
    constraint_count = 0 if constraints is None else int(constraints.pairs.shape[0])
    dof = _temperature_degrees_of_freedom(len(masses), constraint_count)
    current_ke = kinetic_energy_scale * 0.5 * float(np.sum(masses[:, None] * velocities**2))
    target_ke = 0.5 * dof * boltzmann_constant * temperature
    if current_ke <= 0.0 or target_ke <= 0.0:
        return velocities.astype(np.float32)
    velocities = velocities * np.sqrt(target_ke / current_ke)
    return velocities.astype(np.float32)


def _production_neighbor_manager(
    system,
    force_terms,
    *,
    require_production: bool,
    neighbor_skin: float = GPCRMD_NEIGHBOR_SKIN,
    neighbor_check_interval: int = 1,
) -> NeighborListManager | None:
    if not require_production:
        return None
    lazy_terms = []
    for term in force_terms:
        topology = getattr(term, "topology", None)
        if topology is None or getattr(topology, "nonbonded_pair_policy", None) != "lazy":
            continue
        lazy_terms.append(term)
    if not lazy_terms:
        return None
    if system.cell is None:
        msg = "large GPCRmd runs require periodic cell-list neighbors"
        raise ValueError(msg)

    cutoffs: list[float] = []
    uses_pme = False
    for term in lazy_terms:
        cutoff = getattr(term, "cutoff", None)
        electrostatics = getattr(term, "electrostatics", "cutoff")
        if cutoff is None:
            msg = "large GPCRmd runs require a finite nonbonded cutoff"
            raise ValueError(msg)
        cutoff_value = float(cutoff)
        if not np.isfinite(cutoff_value) or cutoff_value <= 0.0:
            msg = "large GPCRmd runs require a finite positive nonbonded cutoff"
            raise ValueError(msg)
        cutoffs.append(cutoff_value)
        if electrostatics not in {"cutoff", "pme"}:
            msg = (
                "large GPCRmd runs require cutoff or PME compact neighbors; "
                f"got electrostatics={electrostatics!r}"
            )
            raise ValueError(msg)
        if electrostatics == "pme":
            uses_pme = True
            pme_config = getattr(term, "pme_config", None)
            if pme_config is None or pme_config.real_cutoff is None:
                msg = "production PME neighbors require pme_config.real_cutoff"
                raise ValueError(msg)
            if not np.isclose(
                cutoff_value,
                float(pme_config.real_cutoff),
                rtol=1e-6,
                atol=1e-7,
            ):
                msg = "production PME cutoff must match the nonbonded cutoff"
                raise ValueError(msg)
            if not system.cell.is_orthorhombic:
                msg = "production PME neighbors require an orthorhombic cell"
                raise ValueError(msg)
            cell_lengths = np.asarray(system.cell.lengths, dtype=np.float64)
            if cutoff_value > 0.5 * float(np.min(cell_lengths)) + 1e-7:
                msg = "production PME cutoff must not exceed half the minimum box length"
                raise ValueError(msg)
    reference_cutoff = cutoffs[0]
    if any(
        not np.isclose(value, reference_cutoff, rtol=1e-6, atol=1e-7)
        for value in cutoffs[1:]
    ):
        msg = "production compact-neighbor force terms require one shared cutoff"
        raise ValueError(msg)
    return NeighborListManager(
        system.cell,
        cutoff=reference_cutoff,
        skin=neighbor_skin,
        check_interval=neighbor_check_interval,
        sort_pairs=False,
        max_workers=GPCRMD_NEIGHBOR_WORKERS,
        backend="mlx_cell_blocks" if uses_pme else "auto",
    )


def _bind_fixed_cell_pme_plans(
    force_terms,
    cell,
    *,
    require_production: bool,
    use_npt: bool,
):
    terms = tuple(force_terms)
    pme_terms = tuple(
        term for term in terms if getattr(term, "electrostatics", None) == "pme"
    )
    if not require_production or not pme_terms:
        return terms
    if use_npt:
        msg = "production PME execution plans currently support fixed-cell NVT only"
        raise ValueError(msg)
    if cell is None:
        msg = "production PME execution plan requires a periodic cell"
        raise ValueError(msg)

    bound_terms = []
    for term in terms:
        if getattr(term, "electrostatics", None) != "pme":
            bound_terms.append(term)
            continue
        existing_plan = getattr(term, "pme_plan", None)
        if existing_plan is not None:
            existing_plan.validate(
                cell,
                config=getattr(term, "pme_config", None),
                coulomb_constant=float(getattr(term, "coulomb_constant", 1.0)),
            )
            bound_terms.append(term)
            continue
        binder = getattr(term, "bind_pme_plan", None)
        if not callable(binder):
            msg = "production PME force term does not expose bind_pme_plan"
            raise TypeError(msg)
        bound_terms.append(binder(cell))
    return tuple(bound_terms)


def _pme_execution_plan_diagnostics(force_terms) -> list[dict[str, object]]:
    diagnostics = []
    for term in force_terms:
        plan = getattr(term, "pme_plan", None)
        if plan is None:
            continue
        diagnostics.append(
            {
                "force_term": str(getattr(term, "name", type(term).__name__)),
                **plan.diagnostics,
            }
        )
    return diagnostics


def _compile_force_evaluator_safe(force_terms) -> bool:
    return not any(
        getattr(term, "electrostatics", None) == "pme"
        or getattr(term, "pme_config", None) is not None
        for term in force_terms
    )


def run_mlx(
    prepared: str | Path | PreparedSystem,
    *,
    out: str | Path | None = None,
    steps: int = 2000,
    sample_interval: int = 10,
    dt: float = 0.001,
    temperature: float = 300.0,
    friction: float = 1.0,
    seed: int | None = 7,
    nonbonded_cutoff: float | None = None,
    coulomb_constant: float | None = None,
    restraint_k: float = 5.0,
    receptor_mass_scale: float = 1.0,
    require_production: bool = False,
    minimize_steps: int = 50,
    equilibration_steps: int = 100,
    constraint_max_iterations: int = 4,
    diagnostic_interval: int | None = None,
    neighbor_skin: float = GPCRMD_NEIGHBOR_SKIN,
    neighbor_check_interval: int = 1,
    metadata_overrides: dict[str, Any] | None = None,
    runtime_electrostatics_model: str | None = None,
    reporters: RuntimeReporter | list[RuntimeReporter] | tuple[RuntimeReporter, ...] | None = None,
    checkpoint_out: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    dcd_out: str | Path | None = None,
    xtc_out: str | Path | None = None,
    topology_out: str | Path | None = None,
):
    """Run an MLX NVT trajectory and optionally save `trajectory.npz`."""

    del nonbonded_cutoff, coulomb_constant
    if receptor_mass_scale != 1.0:
        msg = "receptor_mass_scale is no longer supported for MLX production runs"
        raise ValueError(msg)
    if isinstance(prepared, (str, Path)):
        prepared_dir = Path(prepared)
        prepared_system = load_prepared_system(prepared_dir)
        artifact = load_prepared_mlx_artifact(
            prepared_dir,
            require_production=require_production,
        )
    else:
        prepared_dir = None
        prepared_system = prepared
        artifact = _artifact_from_prepared_system(
            prepared_system,
            require_production=require_production,
        )
    protocol_report = validate_gpcrmd_protocol_request(
        prepared_system.metadata.protocol_metadata,
        raise_on_blockers=True,
    )
    if not np.isfinite(neighbor_skin) or neighbor_skin < 0.0:
        msg = "neighbor_skin must be finite and non-negative"
        raise ValueError(msg)
    if neighbor_check_interval <= 0:
        msg = "neighbor_check_interval must be positive"
        raise ValueError(msg)
    if runtime_electrostatics_model is not None:
        artifact = _artifact_with_runtime_electrostatics(
            artifact,
            runtime_electrostatics_model,
        )
    artifact_readiness = artifact_readiness_report(
        artifact.metadata,
        require_production=require_production,
        arrays=artifact.arrays,
    )
    protocol_readiness = protocol_readiness_report(
        prepared_system.metadata.protocol_metadata,
    )
    platform_boundary = _platform_boundary_metadata()
    if diagnostic_interval is None:
        diagnostic_interval = sample_interval if require_production else 1
    system, force_terms, constraints = build_mlx_system_from_artifact(
        artifact,
        restraint_k=restraint_k,
        constraint_max_iterations=constraint_max_iterations,
    )
    use_npt = protocol_report.metadata["ensemble"] == "NPT"
    force_terms = _bind_fixed_cell_pme_plans(
        force_terms,
        system.cell,
        require_production=require_production,
        use_npt=use_npt,
    )
    bound_pme_plan = any(getattr(term, "pme_plan", None) is not None for term in force_terms)
    pressure_diagnostics = (
        artifact.atom_count <= PRESSURE_DIAGNOSTIC_ATOM_LIMIT and not bound_pme_plan
    )
    hmr_state = artifact.hmr_state
    compile_force_evaluator = _compile_force_evaluator_safe(force_terms)
    masses = np.asarray(system.masses, dtype=np.float32)
    unit_system = artifact.unit_system
    kinetic_energy_scale = 1.0 if unit_system is None else unit_system.kinetic_energy_scale
    force_to_acceleration_scale = (
        1.0 if unit_system is None else unit_system.force_to_acceleration_scale
    )
    boltzmann_constant = 1.0 if unit_system is None else unit_system.boltzmann_constant
    checkpoint = None
    run_positions = np.asarray(system.positions, dtype=np.float32)
    initial_step = 0
    initial_time = 0.0
    rng_step_offset = None
    if resume_checkpoint is not None:
        checkpoint = load_simulation_checkpoint(resume_checkpoint)
        run_positions = np.asarray(checkpoint.positions, dtype=np.float32)
        velocities = np.asarray(checkpoint.velocities, dtype=np.float32)
        initial_step = checkpoint.step
        initial_time = checkpoint.time
        rng_step_offset = int(checkpoint.thermostat.get("rng_step_offset", checkpoint.step))
        minimize_steps = 0
        equilibration_steps = 0
    else:
        velocities = _initialize_system_velocities(
            system,
            masses,
            temperature=temperature,
            seed=seed,
            kinetic_energy_scale=kinetic_energy_scale,
            boltzmann_constant=boltzmann_constant,
        )
        velocities = _project_and_rescale_velocities(
            velocities,
            positions=run_positions,
            masses=masses,
            constraints=constraints,
            cell=system.cell,
            temperature=temperature,
            kinetic_energy_scale=kinetic_energy_scale,
            boltzmann_constant=boltzmann_constant,
        )
    neighbor_manager = _production_neighbor_manager(
        system,
        force_terms,
        require_production=require_production,
        neighbor_skin=neighbor_skin,
        neighbor_check_interval=neighbor_check_interval,
    )

    run_started = time.perf_counter()
    pressure_internal = (
        ATM_TO_KJ_PER_MOL_ANGSTROM3
        if unit_system is not None and unit_system.coordinates == "angstrom"
        else 1.0
    )
    if use_npt:
        result = simulate_npt(
            run_positions,
            velocities,
            masses=masses,
            cell=system.cell,
            force_terms=force_terms,
            config=_simulation_config_with_virtual_sites(
                dt=dt,
                steps=steps,
                sample_interval=sample_interval,
                kinetic_energy_scale=kinetic_energy_scale,
                force_to_acceleration_scale=force_to_acceleration_scale,
                boltzmann_constant=boltzmann_constant,
                diagnostic_interval=diagnostic_interval,
                compile_force_evaluator=compile_force_evaluator,
                pressure_diagnostics=pressure_diagnostics,
                initial_step=initial_step,
                initial_time=initial_time,
                virtual_sites=system.virtual_sites,
            ),
            thermostat=LangevinThermostat(
                temperature=temperature,
                friction=friction,
                seed=seed,
                rng_step_offset=rng_step_offset,
            ),
            barostat=MonteCarloBarostat(
                pressure=pressure_internal,
                temperature=temperature,
                seed=seed,
            ),
            constraints=constraints,
            neighbor_manager=neighbor_manager,
            reporters=reporters,
        )
    elif minimize_steps > 0 or equilibration_steps > 0:
        protocol = MinimizeThenNVTProtocol(
            minimize_steps=minimize_steps,
            equilibration_steps=equilibration_steps,
            production_steps=steps,
            dt=dt,
            sample_interval=sample_interval,
            temperature=temperature,
            friction=friction,
            seed=seed,
            diagnostic_interval=diagnostic_interval,
            compile_force_evaluator=compile_force_evaluator,
            ensemble=protocol_report.metadata["ensemble"],
            proof_mode=protocol_report.metadata["proof_mode"],
            barostat=protocol_report.metadata["barostat"],
            npt_barostat=protocol_report.metadata["npt_barostat"],
            membrane_barostat=protocol_report.metadata["membrane_barostat"],
        )
        if system.virtual_sites is None:
            protocol_result = run_minimize_then_nvt(
                run_positions,
                velocities,
                masses=masses,
                force_terms=force_terms,
                protocol=protocol,
                cell=system.cell,
                constraints=constraints,
                unit_system=unit_system,
                neighbor_manager=neighbor_manager,
                pressure_diagnostics=pressure_diagnostics,
                reporters=reporters,
            )
        else:
            protocol_result = _run_virtual_site_minimize_then_nvt(
                run_positions,
                velocities,
                masses=masses,
                force_terms=force_terms,
                protocol=protocol,
                virtual_sites=system.virtual_sites,
                cell=system.cell,
                constraints=constraints,
                unit_system=unit_system,
                pressure_diagnostics=pressure_diagnostics,
                reporters=reporters,
            )
        result = protocol_result.production
    else:
        result = simulate_nvt(
            run_positions,
            velocities,
            masses=masses,
            cell=system.cell,
            force_terms=force_terms,
            config=_simulation_config_with_virtual_sites(
                dt=dt,
                steps=steps,
                sample_interval=sample_interval,
                kinetic_energy_scale=kinetic_energy_scale,
                force_to_acceleration_scale=force_to_acceleration_scale,
                boltzmann_constant=boltzmann_constant,
                diagnostic_interval=diagnostic_interval,
                compile_force_evaluator=compile_force_evaluator,
                pressure_diagnostics=pressure_diagnostics,
                initial_step=initial_step,
                initial_time=initial_time,
                virtual_sites=system.virtual_sites,
            ),
            thermostat=LangevinThermostat(
                temperature=temperature,
                friction=friction,
                seed=seed,
                rng_step_offset=rng_step_offset,
            ),
            constraints=constraints,
            neighbor_manager=neighbor_manager,
            reporters=reporters,
        )
    elapsed_wall_seconds = time.perf_counter() - run_started
    pme_execution_plans = _pme_execution_plan_diagnostics(force_terms)
    if pme_execution_plans:
        result.nonbonded_report["pme_execution_plan_count"] = len(pme_execution_plans)
        result.nonbonded_report["pme_execution_plans"] = pme_execution_plans
    if out is None and prepared_dir is not None:
        out = prepared_dir / TRAJECTORY_NAME
    if checkpoint_out is not None:
        save_simulation_checkpoint(
            checkpoint_out,
            result.final_state,
            cell=result.final_cell if use_npt else system.cell,
            thermostat={
                "temperature": temperature,
                "friction": friction,
                "seed": seed,
                "rng_step_offset": int(result.final_state.step),
            },
            neighbor_policy={
                "skin": neighbor_skin,
                "check_interval": neighbor_check_interval,
                "backend": result.nonbonded_report.get("backend"),
            },
            force_terms=tuple(
                str(getattr(term, "name", type(term).__name__)) for term in force_terms
            ),
            diagnostic_cursor=int(np.asarray(result.diagnostic_steps)[-1]),
            metadata={
                "kind": "mlx_atomistic.checkpoint",
                "source": "prep.run_mlx",
                "resumed_from": None if checkpoint is None else str(resume_checkpoint),
                "platform_readiness": {
                    "artifact": artifact_readiness.to_dict(),
                    "protocol": protocol_readiness.to_dict(),
                },
                "platform_boundary": platform_boundary,
                "hydrogen_mass_repartitioning": hmr_state,
                "pme_execution_plans": pme_execution_plans,
            },
            runtime_sync_report=result.runtime_sync_report,
            runtime_nonbonded_report=result.nonbonded_report,
        )
    if out is not None:
        metadata: dict[str, Any] = {
            "kind": "mlx_atomistic.prep_nvt",
            "engine": "mlx_atomistic",
            "source": "mlx_atomistic",
            "prepared_artifact_version": prepared_system.metadata.artifact_version,
            "parameter_source": prepared_system.metadata.parameter_source,
            "production_force_field": bool(
                prepared_system.metadata.compatibility_report.get(
                    "production_force_field",
                    False,
                )
            ),
            "dt": dt,
            "steps": steps,
            "sample_interval": sample_interval,
            "temperature": temperature,
            "friction": friction,
            "seed": seed,
            "restraint_k": restraint_k,
            "minimize_steps": minimize_steps,
            "equilibration_steps": equilibration_steps,
            "constraint_max_iterations": constraint_max_iterations,
            "diagnostic_interval": diagnostic_interval,
            "pressure_diagnostics": pressure_diagnostics,
            "neighbor_skin": neighbor_skin,
            "neighbor_check_interval": neighbor_check_interval,
            "pressure_diagnostics_reason": (
                None
                if pressure_diagnostics
                else (
                    "disabled_bound_pme_without_analytic_virial"
                    if bound_pme_plan
                    else "disabled_large_system_finite_difference_virial"
                )
            ),
            "elapsed_wall_seconds": elapsed_wall_seconds,
            "integration_steps_per_second": (
                (steps + minimize_steps + equilibration_steps) / elapsed_wall_seconds
                if elapsed_wall_seconds > 0.0
                else None
            ),
            "simulated_time_ps": steps * dt,
            "simulated_ps_per_wall_second": (
                (steps * dt) / elapsed_wall_seconds if elapsed_wall_seconds > 0.0 else None
            ),
            "units": prepared_system.metadata.units,
            "electrostatics_model": prepared_system.metadata.compatibility_report.get(
                "electrostatics_model"
            ),
            "warnings": prepared_system.metadata.warnings,
            "nonbonded_runtime": result.nonbonded_report,
            "pme_execution_plans": pme_execution_plans,
            "resume_checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
            "hydrogen_mass_repartitioning": hmr_state,
        }
        if use_npt:
            metadata["kind"] = "mlx_atomistic.prep_npt"
            metadata["barostat_attempts"] = result.barostat_attempts
            metadata["barostat_accepted"] = result.barostat_accepted
            metadata["pressure_atm"] = 1.0
            metadata["barostat_pressure_internal"] = pressure_internal
        if metadata_overrides:
            metadata.update(metadata_overrides)
        metadata.update(protocol_report.metadata)
        metadata["platform_readiness"] = {
            "artifact": artifact_readiness.to_dict(),
            "protocol": protocol_readiness.to_dict(),
        }
        metadata["platform_boundary"] = platform_boundary
        trajectory_cell = result.final_cell if use_npt else system.cell
        save_npz_trajectory(
            out,
            result,
            symbols=tuple(str(item) for item in prepared_system.symbols.tolist()),
            cell=trajectory_cell,
            metadata=metadata,
        )
    else:
        metadata = {}
        trajectory_cell = result.final_cell if use_npt else system.cell
    auxiliary_outputs = [(dcd_out, "dcd"), (xtc_out, "xtc")]
    if any(path is not None for path, _ in auxiliary_outputs):
        symbols = tuple(str(item) for item in prepared_system.symbols.tolist())
        record = trajectory_record_from_result(
            result,
            symbols=symbols,
            cell=trajectory_cell,
            metadata=metadata,
        )
        topology_path = _trajectory_topology_path(
            prepared_system,
            prepared_dir=prepared_dir,
            out=out,
            topology_out=topology_out,
            auxiliary_outputs=auxiliary_outputs,
        )
        for output_path, output_format in auxiliary_outputs:
            if output_path is not None:
                write_mdtraj_trajectory(
                    topology_path,
                    record,
                    output_path,
                    file_format=output_format,
                )
    return result


def _trajectory_topology_path(
    prepared_system: PreparedSystem,
    *,
    prepared_dir: Path | None,
    out: str | Path | None,
    topology_out: str | Path | None,
    auxiliary_outputs: list[tuple[str | Path | None, str]],
) -> Path:
    if topology_out is not None:
        path = Path(topology_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_view_pdb(path, prepared_system)
        return path
    if prepared_dir is not None:
        path = prepared_dir / VIEW_PDB_NAME
        if not path.exists():
            write_view_pdb(path, prepared_system)
        return path
    if out is not None:
        path = Path(out).parent / VIEW_PDB_NAME
    else:
        first_output = next(Path(item) for item, _ in auxiliary_outputs if item is not None)
        path = first_output.parent / VIEW_PDB_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    write_view_pdb(path, prepared_system)
    return path


def run_gpcrmd_mlx(
    *,
    out: str | Path,
    target_id: str | None = None,
    cache: str | Path | None = None,
    registry_path: str | Path | None = None,
    steps: int = 2000,
    sample_interval: int = 10,
    dt: float = 0.001,
    temperature: float = 300.0,
    friction: float = 1.0,
    seed: int | None = 7,
    restraint_k: float = 5.0,
    minimize_steps: int = 50,
    equilibration_steps: int = 100,
    constraint_max_iterations: int = 4,
    diagnostic_interval: int | None = None,
    neighbor_skin: float = GPCRMD_NEIGHBOR_SKIN,
    neighbor_check_interval: int = 1,
    force: bool = False,
    electrostatics: str = "pme",
) -> dict[str, Any]:
    """Import or load a GPCRmd prepared artifact and run the short MLX NVT path."""

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = out_dir / TRAJECTORY_NAME
    run_report_path = out_dir / GPCRMD_RUN_REPORT_NAME
    import_report_path = out_dir / GPCRMD_IMPORT_REPORT_NAME

    import_payload = None
    compatibility_report = None
    prepared_path = out_dir
    exported = False
    imported = False
    resolved_target_id = target_id
    resolved_dynamics_id = None

    if cache is not None:
        if trajectory_path.exists() and not force:
            payload = _gpcrmd_run_report_payload(
                status="blocked",
                target_id=resolved_target_id,
                dynamics_id=resolved_dynamics_id,
                out_dir=out_dir,
                prepared_path=out_dir if _prepared_artifact_files_exist(out_dir) else None,
                trajectory_path=None,
                planned_trajectory_path=trajectory_path,
                run_report_path=run_report_path,
                import_report_path=import_report_path if import_report_path.exists() else None,
                exported=False,
                imported=False,
                trajectory_written=False,
                blockers=(f"output_exists:{trajectory_path}",),
                compatibility_report=None,
                import_report=None,
                run_metadata=None,
                diagnostic_summary=None,
            )
            _write_gpcrmd_run_report(run_report_path, payload)
            return payload
        try:
            attempt = attempt_gpcrmd_prepared_artifact_import(
                cache,
                out_dir,
                target_id=target_id,
                registry_path=registry_path,
            )
        except (GPCRmdInspectionError, GPCRmdTargetError) as exc:
            _remove_output_file(trajectory_path)
            payload = _gpcrmd_run_report_payload(
                status="blocked",
                target_id=resolved_target_id,
                dynamics_id=resolved_dynamics_id,
                out_dir=out_dir,
                prepared_path=None,
                trajectory_path=None,
                planned_trajectory_path=trajectory_path,
                run_report_path=run_report_path,
                import_report_path=None,
                exported=False,
                imported=False,
                trajectory_written=False,
                blockers=(f"target_selection:{exc}",),
                compatibility_report=None,
                import_report=None,
                run_metadata=None,
                diagnostic_summary=None,
            )
            _write_gpcrmd_run_report(run_report_path, payload)
            return payload
        write_gpcrmd_import_report(import_report_path, attempt)
        import_payload = attempt.to_json_dict()
        compatibility_report = attempt.compatibility_report.to_json_dict()
        resolved_target_id = attempt.target_id
        resolved_dynamics_id = attempt.dynamics_id
        exported = attempt.exported
        imported = attempt.exported
        if not attempt.exported:
            _remove_output_file(trajectory_path)
            payload = _gpcrmd_run_report_payload(
                status="blocked",
                target_id=resolved_target_id,
                dynamics_id=resolved_dynamics_id,
                out_dir=out_dir,
                prepared_path=None,
                trajectory_path=None,
                planned_trajectory_path=trajectory_path,
                run_report_path=run_report_path,
                import_report_path=import_report_path,
                exported=False,
                imported=False,
                trajectory_written=False,
                blockers=attempt.blockers,
                compatibility_report=compatibility_report,
                import_report=import_payload,
                run_metadata=None,
                diagnostic_summary=None,
            )
            _write_gpcrmd_run_report(run_report_path, payload)
            return payload
        prepared_path = Path(str(attempt.prepared_artifact_path))
    else:
        missing = _missing_prepared_artifact_files(out_dir)
        if missing:
            _remove_output_file(trajectory_path)
            payload = _gpcrmd_run_report_payload(
                status="blocked",
                target_id=resolved_target_id,
                dynamics_id=resolved_dynamics_id,
                out_dir=out_dir,
                prepared_path=None,
                trajectory_path=None,
                planned_trajectory_path=trajectory_path,
                run_report_path=run_report_path,
                import_report_path=None,
                exported=False,
                imported=False,
                trajectory_written=False,
                blockers=tuple(f"missing_prepared_artifact:{path}" for path in missing),
                compatibility_report=None,
                import_report=None,
                run_metadata=None,
                diagnostic_summary=None,
            )
            _write_gpcrmd_run_report(run_report_path, payload)
            return payload

    try:
        prepared = load_prepared_system(prepared_path)
    except (FileNotFoundError, ValueError) as exc:
        _remove_output_file(trajectory_path)
        payload = _gpcrmd_run_report_payload(
            status="blocked",
            target_id=resolved_target_id,
            dynamics_id=resolved_dynamics_id,
            out_dir=out_dir,
            prepared_path=prepared_path,
            trajectory_path=None,
            planned_trajectory_path=trajectory_path,
            run_report_path=run_report_path,
            import_report_path=import_report_path if cache is not None else None,
            exported=exported,
            imported=imported,
            trajectory_written=False,
            blockers=(f"prepared_artifact:{exc}",),
            compatibility_report=compatibility_report,
            import_report=import_payload,
            run_metadata=None,
            diagnostic_summary=None,
        )
        _write_gpcrmd_run_report(run_report_path, payload)
        return payload

    identity_blockers, identity = _gpcrmd_artifact_identity_blockers(prepared, target_id=target_id)
    resolved_target_id = str(identity.get("target_id") or resolved_target_id or "")
    resolved_dynamics_id = identity.get("dynamics_id", resolved_dynamics_id)
    artifact_compatibility = prepared.metadata.compatibility_report
    if compatibility_report is None:
        compatibility_report = artifact_compatibility
    if identity_blockers:
        _remove_output_file(trajectory_path)
        payload = _gpcrmd_run_report_payload(
            status="blocked",
            target_id=resolved_target_id or None,
            dynamics_id=resolved_dynamics_id,
            out_dir=out_dir,
            prepared_path=prepared_path,
            trajectory_path=None,
            planned_trajectory_path=trajectory_path,
            run_report_path=run_report_path,
            import_report_path=import_report_path if cache is not None else None,
            exported=exported,
            imported=imported,
            trajectory_written=False,
            blockers=identity_blockers,
            compatibility_report=compatibility_report,
            import_report=import_payload,
            run_metadata=None,
            diagnostic_summary=None,
        )
        _write_gpcrmd_run_report(run_report_path, payload)
        return payload

    if trajectory_path.exists() and not force:
        payload = _gpcrmd_run_report_payload(
            status="blocked",
            target_id=resolved_target_id or None,
            dynamics_id=resolved_dynamics_id,
            out_dir=out_dir,
            prepared_path=prepared_path,
            trajectory_path=None,
            planned_trajectory_path=trajectory_path,
            run_report_path=run_report_path,
            import_report_path=import_report_path if cache is not None else None,
            exported=exported,
            imported=imported,
            trajectory_written=False,
            blockers=(f"output_exists:{trajectory_path}",),
            compatibility_report=compatibility_report,
            import_report=import_payload,
            run_metadata=None,
            diagnostic_summary=None,
        )
        _write_gpcrmd_run_report(run_report_path, payload)
        return payload

    try:
        electrostatics_report = _gpcrmd_electrostatics_report(
            prepared_path,
            requested_electrostatics=electrostatics,
        )
    except (MLXCompatibilityError, FileNotFoundError, ValueError) as exc:
        electrostatics_report = {
            "status": "blocked",
            "route": electrostatics,
            "metadata_model": None,
            "requested_electrostatics": electrostatics,
            "production_ready": False,
            "blockers": (f"readiness_validation:{exc}",),
        }
    electrostatics_blockers = tuple(
        f"electrostatics:{item}" for item in electrostatics_report["blockers"]
    )
    if electrostatics_blockers:
        _remove_output_file(trajectory_path)
        payload = _gpcrmd_run_report_payload(
            status="blocked",
            target_id=resolved_target_id or None,
            dynamics_id=resolved_dynamics_id,
            out_dir=out_dir,
            prepared_path=prepared_path,
            trajectory_path=None,
            planned_trajectory_path=trajectory_path,
            run_report_path=run_report_path,
            import_report_path=import_report_path if cache is not None else None,
            exported=exported,
            imported=imported,
            trajectory_written=False,
            blockers=electrostatics_blockers,
            compatibility_report=compatibility_report,
            import_report=import_payload,
            run_metadata=None,
            diagnostic_summary=None,
            electrostatics_report=electrostatics_report,
        )
        _write_gpcrmd_run_report(run_report_path, payload)
        return payload

    try:
        run_mlx(
            prepared_path,
            out=trajectory_path,
            steps=steps,
            sample_interval=sample_interval,
            dt=dt,
            temperature=temperature,
            friction=friction,
            seed=seed,
            restraint_k=restraint_k,
            require_production=True,
            minimize_steps=minimize_steps,
            equilibration_steps=equilibration_steps,
            constraint_max_iterations=constraint_max_iterations,
            diagnostic_interval=diagnostic_interval,
            neighbor_skin=neighbor_skin,
            neighbor_check_interval=neighbor_check_interval,
            runtime_electrostatics_model=(
                "cutoff"
                if electrostatics_report["route"] == "short-range-prototype"
                else None
            ),
            metadata_overrides={
                "kind": "gpcrmd_mlx_nvt",
                "workflow": "run_gpcrmd_mlx",
                "gpcrmd_target_id": resolved_target_id,
                "gpcrmd_dynamics_id": resolved_dynamics_id,
                "electrostatics_model": electrostatics_report["metadata_model"],
                "electrostatics_route": electrostatics_report["route"],
                "electrostatics_production_ready": electrostatics_report[
                    "production_ready"
                ],
                "electrostatics_readiness_status": electrostatics_report["status"],
            },
        )
    except ProtocolCompatibilityError as exc:
        _remove_output_file(trajectory_path)
        payload = _gpcrmd_run_report_payload(
            status="blocked",
            target_id=resolved_target_id or None,
            dynamics_id=resolved_dynamics_id,
            out_dir=out_dir,
            prepared_path=prepared_path,
            trajectory_path=None,
            planned_trajectory_path=trajectory_path,
            run_report_path=run_report_path,
            import_report_path=import_report_path if cache is not None else None,
            exported=exported,
            imported=imported,
            trajectory_written=False,
            blockers=tuple(f"unsupported_protocol:{item}" for item in exc.blockers),
            compatibility_report=compatibility_report,
            import_report=import_payload,
            run_metadata=None,
            diagnostic_summary=None,
        )
        _write_gpcrmd_run_report(run_report_path, payload)
        return payload
    except (MLXCompatibilityError, FileNotFoundError, ValueError) as exc:
        _remove_output_file(trajectory_path)
        payload = _gpcrmd_run_report_payload(
            status="blocked",
            target_id=resolved_target_id or None,
            dynamics_id=resolved_dynamics_id,
            out_dir=out_dir,
            prepared_path=prepared_path,
            trajectory_path=None,
            planned_trajectory_path=trajectory_path,
            run_report_path=run_report_path,
            import_report_path=import_report_path if cache is not None else None,
            exported=exported,
            imported=imported,
            trajectory_written=False,
            blockers=(f"mlx_run:{exc}",),
            compatibility_report=compatibility_report,
            import_report=import_payload,
            run_metadata=None,
            diagnostic_summary=None,
        )
        _write_gpcrmd_run_report(run_report_path, payload)
        return payload

    record = load_npz_trajectory(trajectory_path)
    payload = _gpcrmd_run_report_payload(
        status="ran",
        target_id=resolved_target_id or None,
        dynamics_id=resolved_dynamics_id,
        out_dir=out_dir,
        prepared_path=prepared_path,
        trajectory_path=trajectory_path,
        planned_trajectory_path=trajectory_path,
        run_report_path=run_report_path,
        import_report_path=import_report_path if cache is not None else None,
        exported=exported,
        imported=imported,
        trajectory_written=True,
        blockers=(),
        compatibility_report=compatibility_report,
        import_report=import_payload,
        run_metadata=record.metadata,
        diagnostic_summary=_trajectory_diagnostic_summary(record),
        electrostatics_report=electrostatics_report,
    )
    _write_gpcrmd_run_report(run_report_path, payload)
    return payload


def _missing_prepared_artifact_files(out_dir: Path) -> tuple[Path, ...]:
    return tuple(path for path in (out_dir / JSON_NAME, out_dir / NPZ_NAME) if not path.exists())


def _artifact_with_runtime_electrostatics(
    artifact: PreparedMLXArtifact,
    model: str,
) -> PreparedMLXArtifact:
    metadata = dict(artifact.metadata)
    report = dict(metadata.get("compatibility_report", {}))
    metadata["electrostatics"] = model
    metadata["electrostatics_model"] = model
    report["electrostatics"] = model
    report["electrostatics_model"] = model
    metadata["compatibility_report"] = report
    return replace(artifact, metadata=metadata)


def _platform_boundary_metadata() -> dict[str, Any]:
    report = get_platform_boundary_report()
    return {
        "product_runtime": report.product_runtime,
        "runtime": report.runtime.to_dict(),
        "sections": [section.name for section in report.sections],
        "reference_engine_policy": dict(report.reference_engine_policy),
    }


def _prepared_artifact_files_exist(out_dir: Path) -> bool:
    return all(path.exists() for path in (out_dir / JSON_NAME, out_dir / NPZ_NAME))


def _gpcrmd_artifact_identity_blockers(
    prepared: PreparedSystem,
    *,
    target_id: str | None,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    source = prepared.metadata.source
    artifact_target_id = source.get("gpcrmd_target_id")
    artifact_dynamics_id = source.get("gpcrmd_dynamics_id")
    blockers: list[str] = []
    if artifact_target_id is None:
        blockers.append("prepared_artifact:not_gpcrmd")
    elif target_id is not None and str(artifact_target_id) != str(target_id):
        blockers.append(f"target_mismatch:requested={target_id}:artifact={artifact_target_id}")
    return tuple(blockers), {
        "target_id": artifact_target_id,
        "dynamics_id": artifact_dynamics_id,
    }


def _gpcrmd_run_report_payload(
    *,
    status: str,
    target_id: str | None,
    dynamics_id: Any,
    out_dir: Path,
    prepared_path: Path | None,
    trajectory_path: Path | None,
    planned_trajectory_path: Path,
    run_report_path: Path,
    import_report_path: Path | None,
    exported: bool,
    imported: bool,
    trajectory_written: bool,
    blockers: tuple[str, ...],
    compatibility_report: dict[str, Any] | None,
    import_report: dict[str, Any] | None,
    run_metadata: dict[str, Any] | None,
    diagnostic_summary: dict[str, Any] | None,
    electrostatics_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "target_id": target_id,
        "dynamics_id": None if dynamics_id is None else int(dynamics_id),
        "out_dir": str(out_dir),
        "prepared_artifact_path": None if prepared_path is None else str(prepared_path),
        "trajectory_path": None if trajectory_path is None else str(trajectory_path),
        "planned_trajectory_path": str(planned_trajectory_path),
        "run_report_path": str(run_report_path),
        "import_report_path": None if import_report_path is None else str(import_report_path),
        "exported": bool(exported),
        "imported": bool(imported),
        "trajectory_written": bool(trajectory_written),
        "blockers": list(blockers),
        "compatibility_report": compatibility_report,
        "import_report": import_report,
        "run_metadata": run_metadata,
        "diagnostic_summary": diagnostic_summary,
        "electrostatics_report": electrostatics_report,
    }


def _gpcrmd_electrostatics_report(
    prepared_path: Path,
    *,
    requested_electrostatics: str,
) -> dict[str, Any]:
    requested = requested_electrostatics.strip().lower().replace("_", "-")
    if requested in {"short-range", "short-range-prototype"}:
        artifact = load_prepared_mlx_artifact(prepared_path, require_production=True)
        artifact_mode = str(
            artifact.metadata.get("electrostatics_model")
            or dict(artifact.metadata.get("compatibility_report", {})).get("electrostatics_model")
            or "cutoff"
        )
        return {
            "status": "prototype_allowed",
            "route": "short-range-prototype",
            "metadata_model": "short_range_electrostatics_prototype",
            "requested_electrostatics": requested_electrostatics,
            "artifact_electrostatics_model": artifact_mode,
            "production_ready": False,
            "blockers": (),
            "warnings": (
                "short-range-prototype is not production GPCRmd PME electrostatics",
            ),
        }
    if requested != "pme":
        return {
            "status": "blocked",
            "route": requested_electrostatics,
            "metadata_model": None,
            "requested_electrostatics": requested_electrostatics,
            "production_ready": False,
            "blockers": ("unknown_requested_mode:expected_pme_or_short-range-prototype",),
        }

    artifact = load_prepared_mlx_artifact(prepared_path, require_production=True)
    from mlx_atomistic.artifacts import _pme_config_from_artifact  # local private gate reuse
    from mlx_atomistic.topology import Topology

    arrays = artifact.arrays
    config = _pme_config_from_artifact(artifact.metadata, arrays, required=False)
    protocol_metadata = dict(artifact.metadata.get("protocol_metadata", {}))
    nonbonded_metadata = dict(protocol_metadata.get("nonbonded", {}))
    nonbonded_cutoff = float(
        artifact.metadata.get("nonbonded_cutoff")
        or nonbonded_metadata.get("cutoff")
        or 10.0
    )
    topology = Topology.from_sequences(
        n_atoms=artifact.atom_count,
        bonds=np.asarray(arrays["bonds"], dtype=np.int32),
        angles=np.asarray(arrays["angles"], dtype=np.int32),
        dihedrals=np.asarray(arrays["dihedrals"], dtype=np.int32),
        impropers=np.asarray(arrays.get("impropers", np.empty((0, 4))), dtype=np.int32),
        partial_charges=np.asarray(arrays["charges"], dtype=np.float32),
        nonbonded_exception_pairs=np.asarray(
            arrays.get("nonbonded_exception_pairs", np.empty((0, 2))),
            dtype=np.int32,
        ),
        exclude_bonds=True,
        nonbonded_cutoff=nonbonded_cutoff,
    )
    report = pme_readiness_report(
        atom_count=artifact.atom_count,
        charges=arrays["charges"],
        cell_lengths=arrays.get("cell_lengths", np.asarray([])),
        config=config,
        nonbonded_cutoff=nonbonded_cutoff,
        exclusion_count=len(topology.exclusion_set),
        one_four_count=len(topology.one_four_set),
        explicit_exception_count=int(
            np.asarray(arrays.get("nonbonded_exception_pairs", np.empty((0, 2)))).shape[0]
        ),
        cell_matrix=(
            None
            if np.asarray(arrays.get("cell_matrix", np.asarray([]))).size == 0
            else arrays["cell_matrix"]
        ),
    )
    return {
        **report,
        "route": "pme",
        "metadata_model": "pme",
        "requested_electrostatics": requested_electrostatics,
        "production_ready": report["status"] == "ready",
    }


def _trajectory_diagnostic_summary(record: TrajectoryRecord) -> dict[str, Any]:
    positions = np.asarray(record.sampled_positions)
    velocities = np.asarray(record.sampled_velocities)
    total_energy = np.asarray(record.total_energy)
    potential_energy = np.asarray(record.potential_energy)
    kinetic_energy = np.asarray(record.kinetic_energy)
    temperature = np.asarray(record.temperature)
    pressure = np.asarray(record.pressure)
    constraint_error = np.asarray(record.constraint_max_error)
    term_finite = {
        name: bool(np.all(np.isfinite(np.asarray(values))))
        for name, values in record.potential_energy_by_term.items()
    }
    return {
        "sampled_frame_count": int(positions.shape[0]),
        "atom_count": int(positions.shape[1]) if positions.ndim == 3 else None,
        "diagnostic_count": int(total_energy.shape[0]),
        "positions_finite": bool(np.all(np.isfinite(positions))),
        "velocities_finite": bool(np.all(np.isfinite(velocities))),
        "potential_energy_finite": bool(np.all(np.isfinite(potential_energy))),
        "kinetic_energy_finite": bool(np.all(np.isfinite(kinetic_energy))),
        "total_energy_finite": bool(np.all(np.isfinite(total_energy))),
        "temperature_finite": bool(np.all(np.isfinite(temperature))),
        "pressure_finite": bool(np.all(np.isfinite(pressure))),
        "energy_terms_finite": term_finite,
        "final_total_energy": _last_float(total_energy),
        "final_temperature": _last_float(temperature),
        "max_constraint_error_A": (
            float(np.max(constraint_error)) if constraint_error.size else None
        ),
    }


def _last_float(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(values.reshape(-1)[-1])


def _write_gpcrmd_run_report(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _remove_output_file(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink()


def run_steered_mlx(
    prepared: str | Path | PreparedSystem,
    *,
    out: str | Path | None = None,
    steps: int = 25_000,
    sample_interval: int = 50,
    dt: float = 0.001,
    temperature: float = 300.0,
    friction: float = 5.0,
    seed: int | None = 7,
    restraint_k: float = 20.0,
    bias_k: float = 200.0,
    target_velocity: float | None = None,
    direction: np.ndarray | None = None,
    minimize_steps: int = 50,
    equilibration_steps: int = 100,
    constraint_max_iterations: int = 4,
    diagnostic_interval: int | None = None,
):
    """Run MLX steered NVT from a prepared ligand-receptor artifact."""

    if isinstance(prepared, (str, Path)):
        prepared_dir = Path(prepared)
        prepared_system = load_prepared_system(prepared_dir)
        artifact = load_prepared_mlx_artifact(prepared_dir, require_production=False)
    else:
        prepared_dir = None
        prepared_system = prepared
        artifact = _artifact_from_prepared_system(prepared_system, require_production=False)
    protocol_report = validate_gpcrmd_protocol_request(
        prepared_system.metadata.protocol_metadata,
        raise_on_blockers=True,
    )
    if diagnostic_interval is None:
        diagnostic_interval = sample_interval

    system, force_terms, constraints = build_mlx_system_from_artifact(
        artifact,
        restraint_k=restraint_k,
        constraint_max_iterations=constraint_max_iterations,
    )
    masses = np.asarray(system.masses, dtype=np.float32)
    unit_system = artifact.unit_system or MDUnitSystem.from_metadata(prepared_system.metadata.units)
    kinetic_energy_scale = unit_system.kinetic_energy_scale
    force_to_acceleration_scale = unit_system.force_to_acceleration_scale
    boltzmann_constant = unit_system.boltzmann_constant
    velocities = _initialize_system_velocities(
        system,
        masses,
        temperature=temperature,
        seed=seed,
        kinetic_energy_scale=kinetic_energy_scale,
        boltzmann_constant=boltzmann_constant,
    )
    velocities = _project_and_rescale_velocities(
        velocities,
        positions=np.asarray(system.positions, dtype=np.float32),
        masses=masses,
        constraints=constraints,
        cell=system.cell,
        temperature=temperature,
        kinetic_energy_scale=kinetic_energy_scale,
        boltzmann_constant=boltzmann_constant,
    )
    ligand_indices = np.flatnonzero(np.asarray(prepared_system.ligand_mask, dtype=bool)).astype(
        np.int32
    )
    if ligand_indices.size == 0:
        msg = "steered MLX requires a non-empty ligand_mask"
        raise ValueError(msg)
    if direction is None:
        direction = np.asarray(
            prepared_system.metadata.selections.get(
                "steering_bias_direction",
                prepared_system.metadata.selections.get("steering_exit_vector", [1.0, 0.0, 0.0]),
            ),
            dtype=np.float32,
        )
    if target_velocity is None:
        target_velocity = float(
            prepared_system.metadata.selections.get("recommended_steering_velocity_A_per_ps", 0.35)
        )
    direction = np.asarray(direction, dtype=np.float32)
    direction = direction / np.linalg.norm(direction)
    start_bias = SteeredCOMBiasPotential(
        ligand_indices,
        direction=direction,
        target=0.0,
        k=bias_k,
        masses=masses,
    )
    start_cv = float(np.asarray(start_bias.collective_variable(system.positions)))

    run_started = time.perf_counter()
    if minimize_steps > 0 or equilibration_steps > 0:
        protocol = MinimizeThenNVTProtocol(
            minimize_steps=minimize_steps,
            equilibration_steps=equilibration_steps,
            production_steps=0,
            dt=dt,
            sample_interval=1,
            temperature=temperature,
            friction=friction,
            seed=seed,
            diagnostic_interval=1,
            compile_force_evaluator=False,
            ensemble=protocol_report.metadata["ensemble"],
            proof_mode=protocol_report.metadata["proof_mode"],
            barostat=protocol_report.metadata["barostat"],
            npt_barostat=protocol_report.metadata["npt_barostat"],
            membrane_barostat=protocol_report.metadata["membrane_barostat"],
        )
        if system.virtual_sites is None:
            protocol_result = run_minimize_then_nvt(
                np.asarray(system.positions, dtype=np.float32),
                velocities,
                masses=masses,
                force_terms=force_terms,
                protocol=protocol,
                cell=system.cell,
                constraints=constraints,
                unit_system=unit_system,
            )
        else:
            protocol_result = _run_virtual_site_minimize_then_nvt(
                np.asarray(system.positions, dtype=np.float32),
                velocities,
                masses=masses,
                force_terms=force_terms,
                protocol=protocol,
                virtual_sites=system.virtual_sites,
                cell=system.cell,
                constraints=constraints,
                unit_system=unit_system,
            )
        start_state = protocol_result.production.final_state
        start_positions = start_state.positions
        start_velocities = start_state.velocities
    else:
        start_positions = system.positions
        start_velocities = velocities

    start_cv = float(
        np.asarray(
            SteeredCOMBiasPotential(
                ligand_indices,
                direction=direction,
                target=0.0,
                k=bias_k,
                masses=masses,
            ).collective_variable(start_positions)
        )
    )
    result = simulate_steered_nvt(
        start_positions,
        start_velocities,
        masses=masses,
        ligand_indices=ligand_indices,
        direction=direction,
        target_start=start_cv,
        target_velocity=target_velocity,
        k=bias_k,
        cell=system.cell,
        force_terms=force_terms,
        config=_simulation_config_with_virtual_sites(
            dt=dt,
            steps=steps,
            sample_interval=sample_interval,
            kinetic_energy_scale=kinetic_energy_scale,
            force_to_acceleration_scale=force_to_acceleration_scale,
            boltzmann_constant=boltzmann_constant,
            diagnostic_interval=diagnostic_interval,
            compile_force_evaluator=False,
            virtual_sites=system.virtual_sites,
        ),
        thermostat=LangevinThermostat(
            temperature=temperature,
            friction=friction,
            seed=seed,
        ),
        constraints=constraints,
    )
    elapsed_wall_seconds = time.perf_counter() - run_started
    if out is None and prepared_dir is not None:
        out = prepared_dir / STEERED_TRAJECTORY_NAME
    if out is not None:
        metadata: dict[str, Any] = {
            "kind": "mlx_steered_md",
            "engine": "mlx_atomistic",
            "prepared_artifact_version": prepared_system.metadata.artifact_version,
            "parameter_source": prepared_system.metadata.parameter_source,
            "production_force_field": False,
            "pdb_id": prepared_system.metadata.source.get("pdb_id"),
            "ligand_resname": prepared_system.metadata.selections.get("ligand_resname"),
            "dt": dt,
            "steps": steps,
            "sample_interval": sample_interval,
            "temperature": temperature,
            "friction": friction,
            "seed": seed,
            "restraint_k": restraint_k,
            "bias_k": bias_k,
            "target_velocity_A_per_ps": target_velocity,
            "steering_direction": direction.astype(float).tolist(),
            "steering_direction_basis": prepared_system.metadata.selections.get(
                "steering_direction_basis",
                "not specified",
            ),
            "start_cv_A": start_cv,
            "minimize_steps": minimize_steps,
            "equilibration_steps": equilibration_steps,
            "constraint_max_iterations": constraint_max_iterations,
            "diagnostic_interval": diagnostic_interval,
            "elapsed_wall_seconds": elapsed_wall_seconds,
            "integration_steps_per_second": (
                (steps + minimize_steps + equilibration_steps) / elapsed_wall_seconds
                if elapsed_wall_seconds > 0.0
                else None
            ),
            "simulated_time_ps": steps * dt,
            "simulated_ps_per_wall_second": (
                (steps * dt) / elapsed_wall_seconds if elapsed_wall_seconds > 0.0 else None
            ),
            "units": prepared_system.metadata.units,
            "warnings": [
                *prepared_system.metadata.warnings,
                "Trajectory uses a moving harmonic COM restraint; interpret it as steered MD.",
            ],
        }
        metadata.update(protocol_report.metadata)
        save_npz_trajectory(
            out,
            result,
            symbols=tuple(str(item) for item in prepared_system.symbols.tolist()),
            cell=system.cell,
            metadata=metadata,
        )
    return result


__all__ = [
    "GPCRMD_RUN_REPORT_NAME",
    "STEERED_TRAJECTORY_NAME",
    "TRAJECTORY_NAME",
    "build_mlx_system",
    "initialize_velocities",
    "run_gpcrmd_mlx",
    "run_mlx",
    "run_steered_mlx",
]
