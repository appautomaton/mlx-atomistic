"""Batched-replica MLX runs for the solvated ligand-receptor example."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_atomistic.constraints import DistanceConstraints
from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.io import save_npz_trajectory
from mlx_atomistic.md import (
    ForceTerm,
    SimulationState,
    _dense_pair_count,
    _energy_forces_by_term,
    _energy_forces_from_terms,
    _named_force_terms,
    _temperature_degrees_of_freedom,
)
from mlx_atomistic.minimize import minimize_energy
from mlx_atomistic.prep.runner import (
    TRAJECTORY_NAME,
    _project_and_rescale_velocities,
    initialize_velocities,
)
from mlx_atomistic.prep.schema import PreparedSystem
from mlx_atomistic.prep.solvated_example import (
    ELECTROSTATICS_MODEL,
    SOLVATED_LIGAND_RECEPTOR_WORKFLOW,
    ensure_solvated_ligand_receptor_prepared,
    validate_complete_solvated_ligand_receptor_system,
)
from mlx_atomistic.units import MDUnitSystem

ALL_REPLICAS_TRAJECTORY_NAME = "replicas_trajectory.npz"
PROFILE_JSON_NAME = "performance_profile.json"
PROFILE_CSV_NAME = "performance_profile.csv"
REPLICA_WORKFLOW = f"{SOLVATED_LIGAND_RECEPTOR_WORKFLOW}_replicas"
DEFAULT_CONSTRAINT_MAX_ITERATIONS = 40


@dataclass(frozen=True)
class ReplicaRunSummary:
    """Paths and metrics from a batched-replica run."""

    prepared_dir: Path
    selected_trajectory_path: Path
    all_replicas_trajectory_path: Path | None
    metadata: dict[str, Any]
    generated_artifact: bool
    generated_trajectory: bool


@dataclass(frozen=True)
class _BatchedNVTResult:
    sampled_positions: mx.array
    sampled_velocities: mx.array
    sampled_steps: mx.array
    sampled_time: mx.array
    diagnostic_steps: mx.array
    diagnostic_time: mx.array
    potential_energy: mx.array
    kinetic_energy: mx.array
    total_energy: mx.array
    temperature: mx.array
    pair_count: mx.array
    rebuild_count: mx.array
    constraint_max_error: mx.array
    selected_potential_energy_by_term: dict[str, mx.array]
    final_state: SimulationState
    target_temperature: float


@dataclass(frozen=True)
class _SelectedReplicaResult:
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
    pair_count: mx.array
    rebuild_count: mx.array
    constraint_max_error: mx.array
    final_state: SimulationState
    target_temperature: float


def run_ligand_receptor_replicas(
    out_dir: str | Path,
    *,
    replicas: int = 4,
    selected_replica: int = 0,
    steps: int = 5000,
    dt: float = 0.001,
    sample_interval: int = 25,
    temperature: float = 300.0,
    friction: float = 1.0,
    seed: int = 7,
    force: bool = False,
    water_count: int = 48,
    minimize_steps: int = 100,
    equilibration_steps: int = 250,
    restraint_k: float = 10.0,
    constraint_max_iterations: int = DEFAULT_CONSTRAINT_MAX_ITERATIONS,
    diagnostic_interval: int | None = None,
    save_all_replicas: bool = False,
) -> ReplicaRunSummary:
    """Run independent copies of the bundled system in one MLX batched loop."""

    if replicas <= 0:
        msg = "replicas must be positive"
        raise ValueError(msg)
    if selected_replica < 0 or selected_replica >= replicas:
        msg = "selected_replica must be in [0, replicas)"
        raise ValueError(msg)
    if steps < 0:
        msg = "steps must be non-negative"
        raise ValueError(msg)
    if sample_interval <= 0:
        msg = "sample_interval must be positive"
        raise ValueError(msg)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    trajectory_path = out_path / TRAJECTORY_NAME
    all_replicas_path = out_path / ALL_REPLICAS_TRAJECTORY_NAME
    if diagnostic_interval is None:
        diagnostic_interval = sample_interval

    prepared_status = ensure_solvated_ligand_receptor_prepared(
        out_path,
        water_count=water_count,
        force=force,
    )
    prepared = prepared_status["prepared"]
    validate_complete_solvated_ligand_receptor_system(prepared)

    stale = force or _selected_replica_trajectory_is_stale(
        trajectory_path,
        replicas=replicas,
        selected_replica=selected_replica,
        steps=steps,
        dt=dt,
        sample_interval=sample_interval,
        water_count=water_count,
        minimize_steps=minimize_steps,
        equilibration_steps=equilibration_steps,
        restraint_k=restraint_k,
        constraint_max_iterations=constraint_max_iterations,
        diagnostic_interval=diagnostic_interval,
    )
    if not stale and (not save_all_replicas or all_replicas_path.exists()):
        metadata = _load_trajectory_metadata(trajectory_path)
        return ReplicaRunSummary(
            prepared_dir=out_path,
            selected_trajectory_path=trajectory_path,
            all_replicas_trajectory_path=all_replicas_path if all_replicas_path.exists() else None,
            metadata=metadata,
            generated_artifact=bool(prepared_status["generated_artifact"]),
            generated_trajectory=False,
        )

    from mlx_atomistic.artifacts import build_mlx_system_from_artifact, load_prepared_mlx_artifact

    artifact = load_prepared_mlx_artifact(out_path, require_production=False)
    system, force_terms, constraints = build_mlx_system_from_artifact(
        artifact,
        restraint_k=restraint_k,
        constraint_max_iterations=constraint_max_iterations,
    )
    unit_system = artifact.unit_system or MDUnitSystem.from_metadata(prepared.metadata.units)

    positions = np.asarray(system.positions, dtype=np.float32)
    if minimize_steps > 0:
        minimized = minimize_energy(
            positions,
            force_terms=force_terms,
            cell=system.cell,
            max_steps=minimize_steps,
        )
        positions = np.asarray(minimized.positions, dtype=np.float32)
    velocities = _initial_replica_velocities(
        prepared,
        masses=np.asarray(system.masses, dtype=np.float32),
        positions=positions,
        replicas=replicas,
        temperature=temperature,
        seed=seed,
        constraints=constraints,
        cell=system.cell,
        unit_system=unit_system,
    )
    positions_batch = np.repeat(positions[None, :, :], replicas, axis=0)

    run_started = time.perf_counter()
    if equilibration_steps > 0:
        equilibration = _simulate_batched_nvt(
            positions_batch,
            velocities,
            masses=np.asarray(system.masses, dtype=np.float32),
            cell=system.cell,
            force_terms=tuple(force_terms),
            constraints=constraints,
            steps=equilibration_steps,
            dt=dt,
            sample_interval=max(1, equilibration_steps),
            diagnostic_interval=max(1, equilibration_steps),
            temperature=temperature,
            friction=friction,
            seed=seed + 10_000,
            unit_system=unit_system,
            selected_replica=selected_replica,
        )
        positions_batch = np.asarray(equilibration.final_state.positions, dtype=np.float32)
        velocities = np.asarray(equilibration.final_state.velocities, dtype=np.float32)

    result = _simulate_batched_nvt(
        positions_batch,
        velocities,
        masses=np.asarray(system.masses, dtype=np.float32),
        cell=system.cell,
        force_terms=tuple(force_terms),
        constraints=constraints,
        steps=steps,
        dt=dt,
        sample_interval=sample_interval,
        diagnostic_interval=diagnostic_interval,
        temperature=temperature,
        friction=friction,
        seed=seed + 20_000,
        unit_system=unit_system,
        selected_replica=selected_replica,
    )
    mx.eval(
        result.sampled_positions,
        result.sampled_velocities,
        result.potential_energy,
        result.kinetic_energy,
        result.total_energy,
        result.temperature,
        result.constraint_max_error,
    )
    elapsed_wall_seconds = time.perf_counter() - run_started

    metadata = _replica_metadata(
        prepared,
        replicas=replicas,
        selected_replica=selected_replica,
        steps=steps,
        dt=dt,
        sample_interval=sample_interval,
        temperature=temperature,
        friction=friction,
        seed=seed,
        water_count=water_count,
        minimize_steps=minimize_steps,
        equilibration_steps=equilibration_steps,
        restraint_k=restraint_k,
        constraint_max_iterations=constraint_max_iterations,
        diagnostic_interval=diagnostic_interval,
        elapsed_wall_seconds=elapsed_wall_seconds,
    )
    selected_result = _select_replica_result(result, selected_replica)
    save_npz_trajectory(
        trajectory_path,
        selected_result,
        symbols=tuple(str(item) for item in prepared.symbols.tolist()),
        cell=system.cell,
        metadata=metadata,
    )
    if save_all_replicas:
        _save_all_replicas_trajectory(
            all_replicas_path,
            result,
            symbols=tuple(str(item) for item in prepared.symbols.tolist()),
            cell=system.cell,
            metadata=metadata,
        )
    return ReplicaRunSummary(
        prepared_dir=out_path,
        selected_trajectory_path=trajectory_path,
        all_replicas_trajectory_path=all_replicas_path if save_all_replicas else None,
        metadata=metadata,
        generated_artifact=bool(prepared_status["generated_artifact"]),
        generated_trajectory=True,
    )


def profile_ligand_receptor_performance(
    out_dir: str | Path,
    *,
    durations_ps: list[float] | tuple[float, ...] = (5.0, 50.0, 200.0),
    replica_counts: list[int] | tuple[int, ...] = (1, 4, 8, 16),
    dt: float = 0.001,
    sample_interval: int = 100,
    temperature: float = 300.0,
    friction: float = 1.0,
    seed: int = 7,
    water_count: int = 48,
    minimize_steps: int = 100,
    equilibration_steps: int = 250,
    restraint_k: float = 10.0,
    constraint_max_iterations: int = DEFAULT_CONSTRAINT_MAX_ITERATIONS,
    diagnostic_interval: int | None = None,
    save_all_replicas: bool = False,
    force: bool = False,
    write_json: bool = True,
    write_csv: bool = True,
) -> list[dict[str, Any]]:
    """Run repeatable single and multi-replica benchmark rows."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for duration_ps in durations_ps:
        steps = int(round(float(duration_ps) / dt))
        for replicas in replica_counts:
            row_dir = out_path / f"{duration_ps:g}ps-r{replicas}"
            component_profile = profile_batched_components(
                row_dir,
                replicas=replicas,
                water_count=water_count,
                restraint_k=restraint_k,
                constraint_max_iterations=constraint_max_iterations,
                force=force,
            )
            summary = run_ligand_receptor_replicas(
                row_dir,
                replicas=replicas,
                selected_replica=0,
                steps=steps,
                dt=dt,
                sample_interval=sample_interval,
                temperature=temperature,
                friction=friction,
                seed=seed,
                water_count=water_count,
                minimize_steps=minimize_steps,
                equilibration_steps=equilibration_steps,
                restraint_k=restraint_k,
                constraint_max_iterations=constraint_max_iterations,
                diagnostic_interval=diagnostic_interval,
                save_all_replicas=save_all_replicas,
                force=force,
            )
            metadata = dict(summary.metadata)
            max_constraint_error = _max_constraint_error(summary.selected_trajectory_path)
            row = {
                "duration_ps": float(duration_ps),
                "steps": steps,
                "replicas": replicas,
                "atoms_per_replica": int(metadata["atoms_per_replica"]),
                "gpu_visible_atoms": int(metadata["gpu_visible_atoms"]),
                "gpu_visible_dense_pair_slots": int(metadata["gpu_visible_dense_pair_slots"]),
                "wall_s": float(metadata["elapsed_wall_seconds"]),
                "per_replica_steps_per_s": float(metadata["per_replica_steps_per_second"]),
                "aggregate_steps_per_s": float(metadata["aggregate_integration_steps_per_second"]),
                "per_replica_ps_per_s": float(metadata["simulated_ps_per_wall_second"]),
                "aggregate_ps_per_s": float(metadata["aggregate_simulated_ps_per_wall_second"]),
                "frames": int(metadata["frames"]),
                "diagnostic_points": int(metadata["diagnostic_points"]),
                "constraint_max_iterations": int(metadata["constraint_max_iterations"]),
                "max_constraint_error_A": max_constraint_error,
                "artifact_size_bytes": _directory_size_bytes(row_dir),
                "selected_trajectory": str(summary.selected_trajectory_path),
                "all_replicas_trajectory": (
                    None
                    if summary.all_replicas_trajectory_path is None
                    else str(summary.all_replicas_trajectory_path)
                ),
                **component_profile,
            }
            rows.append(row)
    if write_json:
        (out_path / PROFILE_JSON_NAME).write_text(json.dumps(rows, indent=2) + "\n")
    if write_csv:
        _write_profile_csv(out_path / PROFILE_CSV_NAME, rows)
    return rows


def profile_batched_components(
    prepared_dir: str | Path,
    *,
    replicas: int = 4,
    water_count: int = 48,
    restraint_k: float = 10.0,
    constraint_max_iterations: int = DEFAULT_CONSTRAINT_MAX_ITERATIONS,
    force: bool = False,
    repeats: int = 3,
) -> dict[str, float | int]:
    """Measure force and constraint component costs on the prepared example."""

    out_path = Path(prepared_dir)
    status = ensure_solvated_ligand_receptor_prepared(
        out_path,
        water_count=water_count,
        force=force,
    )
    prepared = status["prepared"]
    from mlx_atomistic.artifacts import build_mlx_system_from_artifact, load_prepared_mlx_artifact

    artifact = load_prepared_mlx_artifact(out_path, require_production=False)
    system, force_terms, constraints = build_mlx_system_from_artifact(
        artifact,
        restraint_k=restraint_k,
        constraint_max_iterations=constraint_max_iterations,
    )
    positions = mx.stack([as_mx_array(system.positions)] * replicas)
    velocities = mx.zeros_like(positions)
    masses = as_mx_array(system.masses)
    force_evaluator = _make_batched_force_evaluator(
        tuple(force_terms),
        cell=system.cell,
        compile_evaluator=True,
    )
    project_positions, project_velocities = _make_constraint_projectors(
        constraints,
        masses=masses,
        cell=system.cell,
        replicas=replicas,
    )

    energy, forces = force_evaluator(positions)
    mx.eval(energy, forces)
    force_total_ms = _time_repeated(lambda: force_evaluator(positions), repeats=repeats)

    if constraints is not None:
        projected, error = project_positions(positions)
        mx.eval(projected, error)
        constraint_position_ms = _time_repeated(
            lambda: project_positions(positions),
            repeats=repeats,
        )
        projected_velocities = project_velocities(positions, velocities)
        mx.eval(projected_velocities)
        constraint_velocity_ms = _time_repeated(
            lambda: project_velocities(positions, velocities),
            repeats=repeats,
        )
    else:
        constraint_position_ms = 0.0
        constraint_velocity_ms = 0.0

    named_terms = _named_force_terms(tuple(force_terms))
    selected_energy, selected_forces, energy_by_term = _energy_forces_by_term(
        positions[0],
        named_terms,
        cell=system.cell,
        pairs=None,
    )
    mx.eval(selected_energy, selected_forces, *energy_by_term.values())
    diagnostic_by_term_ms = _time_repeated(
        lambda: _energy_forces_by_term(positions[0], named_terms, cell=system.cell, pairs=None),
        repeats=repeats,
    )

    n_atoms = prepared.atom_count
    return {
        "force_total_ms": force_total_ms,
        "constraint_position_ms": constraint_position_ms,
        "constraint_velocity_ms": constraint_velocity_ms,
        "diagnostic_by_term_ms": diagnostic_by_term_ms,
        "component_profile_repeats": repeats,
        "atoms_per_replica": n_atoms,
        "gpu_visible_atoms": n_atoms * replicas,
        "gpu_visible_dense_pair_slots": replicas * n_atoms * n_atoms,
    }


def _simulate_batched_nvt(
    positions,
    velocities,
    *,
    masses: np.ndarray,
    cell: Cell | None,
    force_terms: tuple[ForceTerm, ...],
    constraints: DistanceConstraints | None,
    steps: int,
    dt: float,
    sample_interval: int,
    diagnostic_interval: int,
    temperature: float,
    friction: float,
    seed: int,
    unit_system: MDUnitSystem,
    selected_replica: int,
) -> _BatchedNVTResult:
    positions = as_mx_array(positions)
    velocities = as_mx_array(velocities)
    masses_mx = as_mx_array(masses)
    replicas = positions.shape[0]
    force_evaluator = _make_batched_force_evaluator(
        force_terms,
        cell=cell,
        compile_evaluator=True,
    )
    project_positions, project_velocities = _make_constraint_projectors(
        constraints,
        masses=masses_mx,
        cell=cell,
        replicas=replicas,
    )
    constraint_error = _zero_constraint_error_batch(positions)
    if constraints is not None:
        positions, constraint_error = project_positions(positions)
        velocities = project_velocities(positions, velocities)
    potential_energy, forces = force_evaluator(positions)

    temperature_dof = _temperature_degrees_of_freedom(positions[0], constraints)
    kinetic_energy_scale = unit_system.kinetic_energy_scale
    force_to_acceleration_scale = unit_system.force_to_acceleration_scale
    boltzmann_constant = unit_system.boltzmann_constant

    state = SimulationState(
        positions=positions,
        velocities=velocities,
        masses=masses_mx,
        forces=forces,
    )
    selected_energy_by_term = _selected_energy_by_term(
        state.positions[selected_replica],
        force_terms,
        cell=cell,
    )

    sampled_positions = [state.positions]
    sampled_velocities = [state.velocities]
    sampled_steps = [0]
    sampled_times = [0.0]
    diagnostic_steps = [0]
    diagnostic_times = [0.0]
    potential_energies = [potential_energy]
    kinetic_value = _batched_kinetic_energy(
        state.velocities,
        masses_mx,
        kinetic_energy_scale=kinetic_energy_scale,
    )
    kinetic_energies = [kinetic_value]
    temperatures = [
        _batched_temperature(
            kinetic_value,
            dof=temperature_dof,
            boltzmann_constant=boltzmann_constant,
        )
    ]
    pair_counts = [
        mx.full((replicas,), _dense_pair_count(state.positions[0]), dtype=mx.int32)
    ]
    rebuild_counts = [mx.zeros((replicas,), dtype=mx.int32)]
    constraint_errors = [constraint_error]
    potential_energy_by_term = {
        name: [energy] for name, energy in selected_energy_by_term.items()
    }

    key = mx.random.key(seed)
    velocity_decay = float(np.exp(-friction * dt))
    noise_scale = float(
        np.sqrt(
            (1.0 - velocity_decay * velocity_decay)
            * temperature
            * boltzmann_constant
            / kinetic_energy_scale
        )
    )
    thermal_scale = noise_scale / mx.sqrt(masses_mx)[None, :, None]
    stepper = _make_batched_baoab_stepper(
        force_evaluator=force_evaluator,
        project_positions=project_positions,
        project_velocities=project_velocities,
        cell=cell,
        masses=masses_mx,
        force_to_acceleration_scale=force_to_acceleration_scale,
        dt=dt,
        velocity_decay=velocity_decay,
        thermal_scale=thermal_scale,
        constraints=constraints,
        compile_stepper=True,
    )
    try:
        trial_positions, trial_velocities, trial_forces, trial_energy, trial_error, trial_key = (
            stepper(state.positions, state.velocities, state.forces, key)
        )
        mx.eval(
            trial_positions,
            trial_velocities,
            trial_forces,
            trial_energy,
            trial_error,
            trial_key,
        )
    except Exception:
        stepper = _make_batched_baoab_stepper(
            force_evaluator=force_evaluator,
            project_positions=project_positions,
            project_velocities=project_velocities,
            cell=cell,
            masses=masses_mx,
            force_to_acceleration_scale=force_to_acceleration_scale,
            dt=dt,
            velocity_decay=velocity_decay,
            thermal_scale=thermal_scale,
            constraints=constraints,
            compile_stepper=False,
        )

    for step in range(1, steps + 1):
        next_positions, next_velocities, next_forces, potential_energy, constraint_error, key = (
            stepper(state.positions, state.velocities, state.forces, key)
        )

        state = SimulationState(
            positions=next_positions,
            velocities=next_velocities,
            masses=masses_mx,
            forces=next_forces,
            step=step,
            time=step * dt,
        )

        if step % sample_interval == 0 or step == steps:
            sampled_positions.append(state.positions)
            sampled_velocities.append(state.velocities)
            sampled_steps.append(step)
            sampled_times.append(state.time)
        if step % diagnostic_interval == 0 or step == steps:
            diagnostic_steps.append(step)
            diagnostic_times.append(state.time)
            potential_energies.append(potential_energy)
            selected_energy_by_term = _selected_energy_by_term(
                state.positions[selected_replica],
                force_terms,
                cell=cell,
            )
            for name, energy in selected_energy_by_term.items():
                potential_energy_by_term[name].append(energy)
            kinetic_value = _batched_kinetic_energy(
                state.velocities,
                masses_mx,
                kinetic_energy_scale=kinetic_energy_scale,
            )
            kinetic_energies.append(kinetic_value)
            temperatures.append(
                _batched_temperature(
                    kinetic_value,
                    dof=temperature_dof,
                    boltzmann_constant=boltzmann_constant,
                )
            )
            pair_counts.append(
                mx.full((replicas,), _dense_pair_count(state.positions[0]), dtype=mx.int32)
            )
            rebuild_counts.append(mx.zeros((replicas,), dtype=mx.int32))
            constraint_errors.append(constraint_error)
        if step % 25 == 0 or step == steps:
            mx.eval(state.positions, state.velocities, state.forces, potential_energy)

    potential_energy_series = mx.stack(potential_energies)
    kinetic_energy_series = mx.stack(kinetic_energies)
    return _BatchedNVTResult(
        sampled_positions=mx.stack(sampled_positions),
        sampled_velocities=mx.stack(sampled_velocities),
        sampled_steps=mx.array(sampled_steps, dtype=mx.int32),
        sampled_time=mx.array(sampled_times),
        diagnostic_steps=mx.array(diagnostic_steps, dtype=mx.int32),
        diagnostic_time=mx.array(diagnostic_times),
        potential_energy=potential_energy_series,
        kinetic_energy=kinetic_energy_series,
        total_energy=potential_energy_series + kinetic_energy_series,
        temperature=mx.stack(temperatures),
        pair_count=mx.stack(pair_counts),
        rebuild_count=mx.stack(rebuild_counts),
        constraint_max_error=mx.stack(constraint_errors),
        selected_potential_energy_by_term={
            name: mx.stack(energies) for name, energies in potential_energy_by_term.items()
        },
        final_state=state,
        target_temperature=temperature,
    )


def _make_batched_force_evaluator(
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    compile_evaluator: bool,
):
    def evaluate(batch_positions: mx.array):
        return mx.vmap(
            lambda pos: _energy_forces_from_terms(
                pos,
                force_terms,
                cell=cell,
                pairs=None,
            )
        )(batch_positions)

    return mx.compile(evaluate) if compile_evaluator else evaluate


def _make_batched_baoab_stepper(
    *,
    force_evaluator,
    project_positions,
    project_velocities,
    cell: Cell | None,
    masses: mx.array,
    force_to_acceleration_scale: float,
    dt: float,
    velocity_decay: float,
    thermal_scale: mx.array,
    constraints: DistanceConstraints | None,
    compile_stepper: bool,
):
    def step(
        positions: mx.array,
        velocities: mx.array,
        forces: mx.array,
        key: mx.array,
    ):
        acceleration = force_to_acceleration_scale * forces / masses[None, :, None]
        velocities_half = velocities + 0.5 * dt * acceleration
        next_positions = positions + 0.5 * dt * velocities_half
        if cell is not None:
            next_positions = cell.wrap(next_positions)

        keys = mx.random.split(key, 2)
        next_key = keys[0]
        noise = mx.random.normal(velocities.shape, key=keys[1])
        thermostatted_velocities = velocity_decay * velocities_half + thermal_scale * noise

        next_positions = next_positions + 0.5 * dt * thermostatted_velocities
        if cell is not None:
            next_positions = cell.wrap(next_positions)
        constraint_error = _zero_constraint_error_batch(next_positions)
        if constraints is not None:
            next_positions, constraint_error = project_positions(next_positions)

        potential_energy, next_forces = force_evaluator(next_positions)
        next_acceleration = force_to_acceleration_scale * next_forces / masses[None, :, None]
        next_velocities = thermostatted_velocities + 0.5 * dt * next_acceleration
        if constraints is not None:
            next_velocities = project_velocities(next_positions, next_velocities)
        return (
            next_positions,
            next_velocities,
            next_forces,
            potential_energy,
            constraint_error,
            next_key,
        )

    return mx.compile(step) if compile_stepper else step


def _make_constraint_projectors(
    constraints: DistanceConstraints | None,
    *,
    masses: mx.array,
    cell: Cell | None,
    replicas: int,
):
    if constraints is None:
        def positions_identity(batch_positions):
            return batch_positions, _zero_constraint_error_batch(batch_positions)

        def velocities_identity(_batch_positions, batch_velocities):
            return batch_velocities

        return positions_identity, velocities_identity

    def project_positions_vmap(batch_positions):
        return mx.vmap(lambda pos: constraints.apply_positions(pos, masses, cell))(batch_positions)

    def project_velocities_vmap(batch_positions, batch_velocities):
        return mx.vmap(
            lambda pos, vel: constraints.apply_velocities(pos, vel, masses, cell)
        )(batch_positions, batch_velocities)

    compiled_positions = mx.compile(project_positions_vmap)
    compiled_velocities = mx.compile(project_velocities_vmap)
    test_positions = mx.zeros((replicas, int(constraints._max_pair_index) + 1, 3))
    try:
        projected, error = compiled_positions(test_positions)
        projected_velocities = compiled_velocities(test_positions, test_positions)
        mx.eval(projected, error, projected_velocities)
    except Exception:
        return (
            lambda batch_positions: _project_positions_loop(
                batch_positions,
                constraints=constraints,
                masses=masses,
                cell=cell,
                replicas=replicas,
            ),
            lambda batch_positions, batch_velocities: _project_velocities_loop(
                batch_positions,
                batch_velocities,
                constraints=constraints,
                masses=masses,
                cell=cell,
                replicas=replicas,
            ),
        )
    return compiled_positions, compiled_velocities


def _project_positions_loop(
    batch_positions,
    *,
    constraints: DistanceConstraints,
    masses: mx.array,
    cell: Cell | None,
    replicas: int,
) -> tuple[mx.array, mx.array]:
    positions = []
    errors = []
    for replica in range(replicas):
        projected, error = constraints.apply_positions(batch_positions[replica], masses, cell)
        positions.append(projected)
        errors.append(error)
    return mx.stack(positions), mx.stack(errors)


def _project_velocities_loop(
    batch_positions,
    batch_velocities,
    *,
    constraints: DistanceConstraints,
    masses: mx.array,
    cell: Cell | None,
    replicas: int,
) -> mx.array:
    velocities = []
    for replica in range(replicas):
        velocities.append(
            constraints.apply_velocities(
                batch_positions[replica],
                batch_velocities[replica],
                masses,
                cell,
            )
        )
    return mx.stack(velocities)


def _initial_replica_velocities(
    prepared: PreparedSystem,
    *,
    masses: np.ndarray,
    positions: np.ndarray,
    replicas: int,
    temperature: float,
    seed: int,
    constraints: DistanceConstraints | None,
    cell: Cell | None,
    unit_system: MDUnitSystem,
) -> np.ndarray:
    velocities = []
    for replica in range(replicas):
        replica_velocities = initialize_velocities(
            prepared,
            masses,
            temperature=temperature,
            seed=seed + replica,
            kinetic_energy_scale=unit_system.kinetic_energy_scale,
            boltzmann_constant=unit_system.boltzmann_constant,
        )
        replica_velocities = _project_and_rescale_velocities(
            replica_velocities,
            positions=positions,
            masses=masses,
            constraints=constraints,
            cell=cell,
            temperature=temperature,
            kinetic_energy_scale=unit_system.kinetic_energy_scale,
            boltzmann_constant=unit_system.boltzmann_constant,
        )
        velocities.append(replica_velocities)
    return np.stack(velocities).astype(np.float32)


def _batched_kinetic_energy(
    velocities: mx.array,
    masses: mx.array,
    *,
    kinetic_energy_scale: float,
) -> mx.array:
    return kinetic_energy_scale * 0.5 * mx.sum(
        masses[None, :, None] * velocities * velocities,
        axis=(1, 2),
    )


def _batched_temperature(
    kinetic_energy_value: mx.array,
    *,
    dof: int,
    boltzmann_constant: float,
) -> mx.array:
    return 2.0 * kinetic_energy_value / (dof * boltzmann_constant)


def _selected_energy_by_term(
    positions: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
) -> dict[str, mx.array]:
    _, _, energy_by_term = _energy_forces_by_term(
        positions,
        _named_force_terms(force_terms),
        cell=cell,
        pairs=None,
    )
    return energy_by_term


def _zero_constraint_error_batch(positions: mx.array) -> mx.array:
    return mx.sum(positions[:, :, 0] * 0.0, axis=1)


def _select_replica_result(
    result: _BatchedNVTResult,
    selected_replica: int,
) -> _SelectedReplicaResult:
    return _SelectedReplicaResult(
        sampled_positions=result.sampled_positions[:, selected_replica],
        sampled_velocities=result.sampled_velocities[:, selected_replica],
        sampled_steps=result.sampled_steps,
        sampled_time=result.sampled_time,
        diagnostic_steps=result.diagnostic_steps,
        diagnostic_time=result.diagnostic_time,
        potential_energy=result.potential_energy[:, selected_replica],
        kinetic_energy=result.kinetic_energy[:, selected_replica],
        total_energy=result.total_energy[:, selected_replica],
        potential_energy_by_term=result.selected_potential_energy_by_term,
        temperature=result.temperature[:, selected_replica],
        pair_count=result.pair_count[:, selected_replica],
        rebuild_count=result.rebuild_count[:, selected_replica],
        constraint_max_error=result.constraint_max_error[:, selected_replica],
        final_state=SimulationState(
            positions=result.final_state.positions[selected_replica],
            velocities=result.final_state.velocities[selected_replica],
            masses=result.final_state.masses,
            forces=result.final_state.forces[selected_replica],
            step=result.final_state.step,
            time=result.final_state.time,
        ),
        target_temperature=result.target_temperature,
    )


def _replica_metadata(
    prepared: PreparedSystem,
    *,
    replicas: int,
    selected_replica: int,
    steps: int,
    dt: float,
    sample_interval: int,
    temperature: float,
    friction: float,
    seed: int,
    water_count: int,
    minimize_steps: int,
    equilibration_steps: int,
    restraint_k: float,
    constraint_max_iterations: int,
    diagnostic_interval: int,
    elapsed_wall_seconds: float,
) -> dict[str, Any]:
    atoms_per_replica = prepared.atom_count
    frames = steps // sample_interval + 1
    if steps % sample_interval != 0:
        frames += 1
    diagnostic_points = steps // diagnostic_interval + 1
    if steps % diagnostic_interval != 0:
        diagnostic_points += 1
    return {
        "kind": "mlx_atomistic.prep_nvt_replicas_selected",
        "engine": "mlx_atomistic",
        "source": "mlx_atomistic",
        "workflow": REPLICA_WORKFLOW,
        "base_workflow": SOLVATED_LIGAND_RECEPTOR_WORKFLOW,
        "dataset_id": "t4l-benzene-solvated-short-range-mlx",
        "prepared_artifact_version": prepared.metadata.artifact_version,
        "parameter_source": prepared.metadata.parameter_source,
        "production_force_field": bool(
            prepared.metadata.compatibility_report.get("production_force_field", False)
        ),
        "replicas": replicas,
        "selected_replica": selected_replica,
        "atoms_per_replica": atoms_per_replica,
        "gpu_visible_atoms": atoms_per_replica * replicas,
        "gpu_visible_dense_pair_slots": atoms_per_replica * atoms_per_replica * replicas,
        "frames": frames,
        "diagnostic_points": diagnostic_points,
        "dt": dt,
        "steps": steps,
        "sample_interval": sample_interval,
        "temperature": temperature,
        "friction": friction,
        "seed": seed,
        "water_count": water_count,
        "ion_count": int(np.count_nonzero(prepared.ion_mask)),
        "restraint_k": restraint_k,
        "minimize_steps": minimize_steps,
        "equilibration_steps": equilibration_steps,
        "constraint_max_iterations": constraint_max_iterations,
        "diagnostic_interval": diagnostic_interval,
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "per_replica_steps_per_second": (
            steps / elapsed_wall_seconds if elapsed_wall_seconds > 0.0 else None
        ),
        "aggregate_integration_steps_per_second": (
            (steps * replicas) / elapsed_wall_seconds if elapsed_wall_seconds > 0.0 else None
        ),
        "simulated_time_ps": steps * dt,
        "simulated_ps_per_wall_second": (
            (steps * dt) / elapsed_wall_seconds if elapsed_wall_seconds > 0.0 else None
        ),
        "aggregate_simulated_ps_per_wall_second": (
            (steps * dt * replicas) / elapsed_wall_seconds
            if elapsed_wall_seconds > 0.0
            else None
        ),
        "units": prepared.metadata.units,
        "electrostatics_model": ELECTROSTATICS_MODEL,
        "pme": False,
        "npt_barostat": False,
        "runtime_note": (
            "Batched independent MLX NVT replicas of the same solvated "
            "ligand-receptor system. The selected replica is saved as trajectory.npz "
            "for notebook visualization."
        ),
        "warnings": prepared.metadata.warnings,
    }


def _save_all_replicas_trajectory(
    path: Path,
    result: _BatchedNVTResult,
    *,
    symbols: tuple[str, ...],
    cell: Cell | None,
    metadata: dict[str, Any],
) -> None:
    cell_array = np.asarray([] if cell is None else np.asarray(cell.lengths), dtype=np.float32)
    np.savez_compressed(
        path,
        sampled_positions=np.asarray(result.sampled_positions),
        sampled_velocities=np.asarray(result.sampled_velocities),
        sampled_steps=np.asarray(result.sampled_steps),
        sampled_time=np.asarray(result.sampled_time),
        diagnostic_steps=np.asarray(result.diagnostic_steps),
        diagnostic_time=np.asarray(result.diagnostic_time),
        potential_energy=np.asarray(result.potential_energy),
        kinetic_energy=np.asarray(result.kinetic_energy),
        total_energy=np.asarray(result.total_energy),
        temperature=np.asarray(result.temperature),
        pair_count=np.asarray(result.pair_count),
        rebuild_count=np.asarray(result.rebuild_count),
        constraint_max_error=np.asarray(result.constraint_max_error),
        symbols=np.asarray(list(symbols), dtype=str),
        cell=cell_array,
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def _selected_replica_trajectory_is_stale(
    path: Path,
    *,
    replicas: int,
    selected_replica: int,
    steps: int,
    dt: float,
    sample_interval: int,
    water_count: int,
    minimize_steps: int,
    equilibration_steps: int,
    restraint_k: float,
    constraint_max_iterations: int,
    diagnostic_interval: int | None,
) -> bool:
    if not path.exists():
        return True
    try:
        metadata = _load_trajectory_metadata(path)
    except Exception:
        return True
    return not (
        metadata.get("source") == "mlx_atomistic"
        and metadata.get("workflow") == REPLICA_WORKFLOW
        and int(metadata.get("replicas", -1)) == replicas
        and int(metadata.get("selected_replica", -1)) == selected_replica
        and int(metadata.get("steps", -1)) == steps
        and abs(float(metadata.get("dt", -1.0)) - dt) < 1e-12
        and int(metadata.get("sample_interval", -1)) == sample_interval
        and int(metadata.get("water_count", -1)) == water_count
        and int(metadata.get("minimize_steps", -1)) == minimize_steps
        and int(metadata.get("equilibration_steps", -1)) == equilibration_steps
        and abs(float(metadata.get("restraint_k", -1.0)) - restraint_k) < 1e-12
        and int(metadata.get("constraint_max_iterations", -1)) == constraint_max_iterations
        and int(metadata.get("diagnostic_interval", -1))
        == int(sample_interval if diagnostic_interval is None else diagnostic_interval)
    )


def _load_trajectory_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(np.asarray(data["metadata_json"])))


def _max_constraint_error(path: Path) -> float:
    with np.load(path, allow_pickle=False) as data:
        return float(np.max(np.asarray(data["constraint_max_error"], dtype=np.float32)))


def _directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _time_repeated(callable_obj, *, repeats: int) -> float:
    elapsed_ms = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        values = callable_obj()
        if not isinstance(values, tuple):
            values = (values,)
        mx.eval(*values)
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
    return float(np.median(np.asarray(elapsed_ms, dtype=np.float64)))


def _write_profile_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "ALL_REPLICAS_TRAJECTORY_NAME",
    "PROFILE_CSV_NAME",
    "PROFILE_JSON_NAME",
    "REPLICA_WORKFLOW",
    "ReplicaRunSummary",
    "profile_batched_components",
    "profile_ligand_receptor_performance",
    "run_ligand_receptor_replicas",
    "DEFAULT_CONSTRAINT_MAX_ITERATIONS",
]
