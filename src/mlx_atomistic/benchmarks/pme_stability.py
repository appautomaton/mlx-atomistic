"""Run minimized target-scale PME NVE and NVT stability validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

from mlx_atomistic.artifacts import (
    build_mlx_system_from_artifact,
    load_prepared_mlx_artifact,
)
from mlx_atomistic.benchmarks.pme_fixture import (
    PME_REAL_CUTOFF_ANGSTROM,
    build_pme_fixture,
    fixture_summary,
)
from mlx_atomistic.benchmarks.pme_validation import apply_openmm_pme_manifest
from mlx_atomistic.constraints import DistanceConstraints
from mlx_atomistic.md import (
    LangevinThermostat,
    SimulationConfig,
    instantaneous_temperature,
    simulate_nve,
    simulate_nvt,
)
from mlx_atomistic.minimize import minimize_energy
from mlx_atomistic.neighbors import NeighborListManager
from mlx_atomistic.pme import pme_readiness_report
from mlx_atomistic.prep.io import save_prepared_system

CONSTRAINT_LIMIT_NM = 2.0e-5
NVE_HALF_FS_DRIFT_LIMIT_KJ_MOL_ATOM = 5.0e-2
NVT_TEMPERATURE_RANGE_K = (270.0, 330.0)
NVT_FRICTION_PER_PS = 10.0
RUNTIME_CONSTRAINT_TOLERANCE_ANGSTROM = 1.0e-4
DEFAULT_REFERENCE_DIR = Path(
    "results/dhfr-scale-neutral-pme-validation/openmm-target"
)


def classify_pme_stability(
    *,
    minimization: dict[str, object],
    nve: list[dict[str, object]],
    nvt: dict[str, object],
    pme_readiness: dict[str, object],
) -> dict[str, object]:
    """Classify target PME stability evidence against the approved limits."""

    blockers: list[str] = []
    if not bool(minimization.get("finite")):
        blockers.append("minimization:non_finite")
    if float(minimization.get("final_energy_kj_mol", np.inf)) > float(
        minimization.get("initial_energy_kj_mol", -np.inf)
    ) + 1.0e-4:
        blockers.append("minimization:energy_increased")
    if pme_readiness.get("status") != "ready":
        blockers.append("pme_readiness:not_ready")

    by_dt = {float(row["dt_fs"]): row for row in nve}
    for dt_fs in (1.0, 0.5):
        row = by_dt.get(dt_fs)
        if row is None:
            blockers.append(f"nve:{dt_fs:g}fs:missing")
            continue
        if not bool(row.get("finite")):
            blockers.append(f"nve:{dt_fs:g}fs:non_finite")
        if float(row.get("max_constraint_error_nm", np.inf)) > CONSTRAINT_LIMIT_NM:
            blockers.append(f"nve:{dt_fs:g}fs:constraint_error")
        if row.get("neighbor_backend") != "mlx_cell_pairs":
            blockers.append(f"nve:{dt_fs:g}fs:neighbor_backend")
        if row.get("neighbor_representation") != "pairs":
            blockers.append(f"nve:{dt_fs:g}fs:neighbor_representation")
        if row.get("fallback_reason") is not None:
            blockers.append(f"nve:{dt_fs:g}fs:fallback")

    coarse = by_dt.get(1.0)
    fine = by_dt.get(0.5)
    if fine is not None:
        fine_drift = float(fine.get("max_energy_drift_per_atom_kj_mol", np.inf))
        if fine_drift > NVE_HALF_FS_DRIFT_LIMIT_KJ_MOL_ATOM:
            blockers.append("nve:0.5fs:energy_drift")
        if coarse is not None and fine_drift > float(
            coarse.get("max_energy_drift_per_atom_kj_mol", -np.inf)
        ) + 1.0e-6:
            blockers.append("nve:timestep_convergence")

    if not bool(nvt.get("finite")):
        blockers.append("nvt:non_finite")
    if float(nvt.get("max_constraint_error_nm", np.inf)) > CONSTRAINT_LIMIT_NM:
        blockers.append("nvt:constraint_error")
    mean_temperature = float(nvt.get("mean_temperature_k", np.nan))
    if not NVT_TEMPERATURE_RANGE_K[0] <= mean_temperature <= NVT_TEMPERATURE_RANGE_K[1]:
        blockers.append("nvt:mean_temperature")
    if nvt.get("neighbor_backend") != "mlx_cell_pairs":
        blockers.append("nvt:neighbor_backend")
    if nvt.get("neighbor_representation") != "pairs":
        blockers.append("nvt:neighbor_representation")
    if nvt.get("fallback_reason") is not None:
        blockers.append("nvt:fallback")

    return {
        "status": "passed" if not blockers else "failed",
        "passed": not blockers,
        "blockers": blockers,
        "limits": {
            "constraint_error_nm": CONSTRAINT_LIMIT_NM,
            "nve_0.5fs_drift_per_atom_kj_mol": (
                NVE_HALF_FS_DRIFT_LIMIT_KJ_MOL_ATOM
            ),
            "nvt_mean_temperature_k": list(NVT_TEMPERATURE_RANGE_K),
        },
    }


def run_pme_stability(
    *,
    case: str,
    reference_dir: str | Path,
    nve_ps: float,
    nve_dt_fs: tuple[float, ...],
    nvt_ps: float,
    nvt_dt_fs: float,
    temperature_k: float,
    out_dir: str | Path,
    seed: int = 20260713,
    minimization_steps: int = 100,
) -> dict[str, object]:
    """Run deterministic minimized PME NVE/NVT validation and write raw evidence."""

    if case != "target":
        msg = "PME stability validation currently requires --case target"
        raise ValueError(msg)
    if nve_ps <= 0.0 or nvt_ps <= 0.0:
        msg = "NVE and NVT durations must be positive"
        raise ValueError(msg)
    if any(dt <= 0.0 for dt in (*nve_dt_fs, nvt_dt_fs)):
        msg = "time steps must be positive"
        raise ValueError(msg)
    if temperature_k <= 0.0:
        msg = "temperature must be positive"
        raise ValueError(msg)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reference = Path(reference_dir)
    manifest_path = reference / "reference.json"
    manifest = json.loads(manifest_path.read_text())
    prepared = apply_openmm_pme_manifest(build_pme_fixture(case), manifest)
    prepared_dir = out / "prepared"
    save_prepared_system(prepared, prepared_dir)
    artifact = load_prepared_mlx_artifact(prepared_dir, require_production=True)
    system, force_terms, constraints = build_mlx_system_from_artifact(
        artifact,
        constraint_max_iterations=8,
        eager_nonbonded_pair_limit=0,
    )
    if constraints is not None:
        constraints = DistanceConstraints(
            np.asarray(constraints.pairs, dtype=np.int32),
            distances=np.asarray(constraints.distances, dtype=np.float32),
            tolerance=constraints.tolerance,
            max_iterations=8,
            velocity_max_iterations=4,
        )
    flexible_force_terms = tuple(force_terms)
    force_terms = tuple(
        term
        for term in force_terms
        if constraints is None or getattr(term, "name", None) not in {"bond", "angle"}
    )
    nonbonded = next(
        (term for term in force_terms if getattr(term, "electrostatics", None) == "pme"),
        None,
    )
    if nonbonded is None:
        msg = "prepared stability artifact did not build a PME nonbonded term"
        raise ValueError(msg)
    if float(nonbonded.cutoff) != PME_REAL_CUTOFF_ANGSTROM:
        msg = "prepared stability artifact did not preserve the 9 Angstrom cutoff"
        raise ValueError(msg)
    topology = nonbonded.topology
    if topology is None:
        msg = "prepared stability PME term requires topology"
        raise ValueError(msg)
    pme_readiness = pme_readiness_report(
        atom_count=prepared.atom_count,
        charges=prepared.charges,
        cell_lengths=prepared.cell_lengths,
        config=nonbonded.pme_config,
        nonbonded_cutoff=float(nonbonded.cutoff),
        exclusion_count=int(topology.exclusions.shape[0]),
        one_four_count=int(topology.one_four_pairs.shape[0]),
        explicit_exception_count=int(nonbonded.exception_pairs.shape[0]),
    )
    if pme_readiness["status"] != "ready":
        blockers = ", ".join(str(item) for item in pme_readiness["blockers"])
        raise ValueError(f"PME readiness blocked stability execution: {blockers}")

    units = artifact.unit_system
    if units is None:
        msg = "target stability requires explicit physical units"
        raise ValueError(msg)
    masses = np.asarray(system.masses, dtype=np.float32)
    initial_positions = np.asarray(system.positions, dtype=np.float32)
    minimization_manager = _neighbor_manager(system.cell)
    minimization_start = perf_counter()
    minimized = minimize_energy(
        initial_positions,
        force_terms,
        cell=system.cell,
        max_steps=minimization_steps,
        step_size=1.0e-4,
        force_tolerance=1.0,
        neighbor_manager=minimization_manager,
        constraints=constraints,
        masses=system.masses,
    )
    minimized_positions = minimized.positions
    projected_constraint_error = mx.array(0.0, dtype=mx.float32)
    if constraints is not None:
        minimized_positions, projected_constraint_error = constraints.apply_positions(
            minimized_positions,
            system.masses,
            system.cell,
        )
    final_energy, final_forces, final_neighbors = _evaluate_force_terms(
        minimized_positions,
        force_terms,
        _neighbor_manager(system.cell),
        system.cell,
    )
    mx.eval(final_energy, final_forces, projected_constraint_error)
    projected_final_forces = (
        final_forces
        if constraints is None
        else constraints.apply_velocities(
            minimized_positions,
            final_forces,
            system.masses,
            system.cell,
        )
    )
    mx.eval(projected_final_forces)
    energy_history = np.asarray(minimized.energy_history, dtype=np.float64)
    minimization_payload = {
        "finite": bool(
            np.all(np.isfinite(energy_history))
            and np.isfinite(np.asarray(final_energy)).all()
            and np.isfinite(np.asarray(final_forces)).all()
        ),
        "steps": int(minimized.steps),
        "requested_steps": int(minimization_steps),
        "converged": bool(minimized.converged),
        "convergence_reason": minimized.convergence_reason,
        "initial_energy_kj_mol": float(energy_history[0]),
        "optimizer_final_energy_kj_mol": float(energy_history[-1]),
        "final_energy_kj_mol": float(np.asarray(final_energy)),
        "final_max_force_kj_mol_angstrom": float(
            np.max(np.abs(np.asarray(projected_final_forces, dtype=np.float64)))
        ),
        "projected_constraint_error_nm": float(np.asarray(projected_constraint_error))
        * 0.1,
        "wall_time_s": perf_counter() - minimization_start,
        "neighbor_backend": final_neighbors.backend,
        "neighbor_representation": final_neighbors.representation_kind,
        "pair_count": int(final_neighbors.pair_count),
    }

    initial_velocities = _physical_velocities(
        masses,
        positions=minimized_positions,
        constraints=constraints,
        cell=system.cell,
        temperature_k=temperature_k,
        seed=seed,
        kinetic_energy_scale=units.kinetic_energy_scale,
        boltzmann_constant=units.boltzmann_constant,
    )
    np.savez_compressed(
        out / "minimized_state.npz",
        positions=np.asarray(minimized_positions, dtype=np.float32),
        velocities=np.asarray(initial_velocities, dtype=np.float32),
        masses=masses,
        fixture_hash=np.asarray([prepared.metadata.selections["content_hash"]]),
    )

    nve_rows = []
    for dt_fs in nve_dt_fs:
        run_constraints = _runtime_constraints(constraints, dt_fs=dt_fs)
        row, raw = _run_nve(
            positions=minimized_positions,
            velocities=initial_velocities,
            masses=system.masses,
            cell=system.cell,
            force_terms=force_terms,
            constraints=run_constraints,
            duration_ps=nve_ps,
            dt_fs=dt_fs,
            atom_count=prepared.atom_count,
            kinetic_energy_scale=units.kinetic_energy_scale,
            force_to_acceleration_scale=units.force_to_acceleration_scale,
            boltzmann_constant=units.boltzmann_constant,
        )
        nve_rows.append(row)
        np.savez_compressed(out / f"nve-{dt_fs:g}fs.npz", **raw)

    nvt_constraints = _runtime_constraints(constraints, dt_fs=nvt_dt_fs)
    nvt_row, nvt_raw = _run_nvt(
        positions=minimized_positions,
        velocities=initial_velocities,
        masses=system.masses,
        cell=system.cell,
        force_terms=force_terms,
        constraints=nvt_constraints,
        duration_ps=nvt_ps,
        dt_fs=nvt_dt_fs,
        temperature_k=temperature_k,
        seed=seed,
        kinetic_energy_scale=units.kinetic_energy_scale,
        force_to_acceleration_scale=units.force_to_acceleration_scale,
        boltzmann_constant=units.boltzmann_constant,
    )
    np.savez_compressed(out / "nvt.npz", **nvt_raw)

    classification = classify_pme_stability(
        minimization=minimization_payload,
        nve=nve_rows,
        nvt=nvt_row,
        pme_readiness=pme_readiness,
    )
    payload = {
        **classification,
        "fixture": fixture_summary(prepared),
        "fixture_hash": prepared.metadata.selections["content_hash"],
        "reference_manifest": str(manifest_path),
        "reference_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "pme": {
            "mesh_shape": prepared.pme_mesh_shape.astype(int).tolist(),
            "alpha_per_angstrom": float(prepared.pme_alpha[0]),
            "real_cutoff_angstrom": float(prepared.pme_real_cutoff[0]),
            "assignment_order": int(prepared.pme_assignment_order[0]),
        },
        "pme_readiness": pme_readiness,
        "neighbor_policy": {
            "backend": "mlx_cell_pairs",
            "representation": "pairs",
            "skin_angstrom": 1.0,
            "check_interval": 10,
            "displacement_check_backend": "mlx_scalar",
        },
        "rigid_water_model": {
            "constraints_replace_internal_bond_angle_terms": constraints is not None,
            "artifact_force_terms": [
                str(getattr(term, "name", type(term).__name__))
                for term in flexible_force_terms
            ],
            "dynamics_force_terms": [
                str(getattr(term, "name", type(term).__name__)) for term in force_terms
            ],
            "position_iteration_policy": {
                "dt_fs_at_least_1": 8,
                "dt_fs_below_1": 6,
            },
            "position_constraint_tolerance_angstrom": (
                RUNTIME_CONSTRAINT_TOLERANCE_ANGSTROM
            ),
            "velocity_constraint_iterations": 4,
        },
        "seed": seed,
        "temperature_k": temperature_k,
        "minimization": minimization_payload,
        "nve": nve_rows,
        "nvt": nvt_row,
        "raw_outputs": {
            "minimized_state": str(out / "minimized_state.npz"),
            "nve": [str(out / f"nve-{dt:g}fs.npz") for dt in nve_dt_fs],
            "nvt": str(out / "nvt.npz"),
        },
    }
    (out / "stability.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def _neighbor_manager(cell) -> NeighborListManager:
    return NeighborListManager(
        cell,
        cutoff=PME_REAL_CUTOFF_ANGSTROM,
        skin=1.0,
        check_interval=10,
        sort_pairs=False,
        backend="mlx_cell_pairs",
        displacement_check_backend="mlx_scalar",
    )


def _runtime_constraints(
    constraints: DistanceConstraints | None,
    *,
    dt_fs: float,
) -> DistanceConstraints | None:
    if constraints is None:
        return None
    position_iterations = 8 if dt_fs >= 1.0 else 6
    return DistanceConstraints(
        np.asarray(constraints.pairs, dtype=np.int32),
        distances=np.asarray(constraints.distances, dtype=np.float32),
        tolerance=RUNTIME_CONSTRAINT_TOLERANCE_ANGSTROM,
        max_iterations=position_iterations,
        velocity_max_iterations=4,
    )


def _evaluate_force_terms(positions, force_terms, neighbor_manager, cell):
    neighbors = neighbor_manager.update(positions)
    energy = mx.array(0.0, dtype=mx.float32)
    forces = mx.zeros_like(positions)
    for term in force_terms:
        term_energy, term_forces = term.energy_forces(
            positions,
            cell,
            pairs=neighbors.interactions,
        )
        energy = energy + term_energy
        forces = forces + term_forces
    return energy, forces, neighbors


def _physical_velocities(
    masses: np.ndarray,
    *,
    positions,
    constraints,
    cell,
    temperature_k: float,
    seed: int,
    kinetic_energy_scale: float,
    boltzmann_constant: float,
):
    rng = np.random.default_rng(seed)
    variance = boltzmann_constant * temperature_k / (
        kinetic_energy_scale * np.maximum(masses, 1.0e-6)
    )
    velocities = rng.normal(size=(masses.shape[0], 3)) * np.sqrt(variance)[:, None]
    velocities -= np.sum(velocities * masses[:, None], axis=0) / np.sum(masses)
    velocities_mx = mx.array(velocities, dtype=mx.float32)
    if constraints is not None:
        velocities_mx = constraints.apply_velocities(
            positions,
            velocities_mx,
            mx.array(masses),
            cell,
        )
    constraint_count = 0 if constraints is None else int(constraints.pairs.shape[0])
    dof = max(1, masses.shape[0] * 3 - constraint_count - 3)
    current_temperature = instantaneous_temperature(
        velocities_mx,
        mx.array(masses),
        dof=dof,
        kinetic_energy_scale=kinetic_energy_scale,
        boltzmann_constant=boltzmann_constant,
    )
    velocities_mx = velocities_mx * mx.sqrt(temperature_k / current_temperature)
    return velocities_mx


def _simulation_config(
    *,
    duration_ps: float,
    dt_fs: float,
    kinetic_energy_scale: float,
    force_to_acceleration_scale: float,
    boltzmann_constant: float,
) -> SimulationConfig:
    steps = max(1, int(round(duration_ps * 1000.0 / dt_fs)))
    diagnostic_interval = max(1, steps // 100)
    return SimulationConfig(
        dt=dt_fs * 1.0e-3,
        steps=steps,
        sample_interval=steps,
        kinetic_energy_scale=kinetic_energy_scale,
        force_to_acceleration_scale=force_to_acceleration_scale,
        boltzmann_constant=boltzmann_constant,
        evaluation_interval=max(1, min(25, diagnostic_interval)),
        diagnostic_interval=diagnostic_interval,
        compile_force_evaluator=False,
        pressure_diagnostics=False,
    )


def _run_nve(
    *,
    positions,
    velocities,
    masses,
    cell,
    force_terms,
    constraints,
    duration_ps: float,
    dt_fs: float,
    atom_count: int,
    kinetic_energy_scale: float,
    force_to_acceleration_scale: float,
    boltzmann_constant: float,
):
    config = _simulation_config(
        duration_ps=duration_ps,
        dt_fs=dt_fs,
        kinetic_energy_scale=kinetic_energy_scale,
        force_to_acceleration_scale=force_to_acceleration_scale,
        boltzmann_constant=boltzmann_constant,
    )
    start = perf_counter()
    result = simulate_nve(
        positions,
        velocities,
        masses=masses,
        cell=cell,
        force_terms=force_terms,
        neighbor_manager=_neighbor_manager(cell),
        config=config,
        constraints=constraints,
    )
    mx.eval(
        result.total_energy,
        result.temperature,
        result.constraint_max_error,
        result.final_state.positions,
    )
    total_energy = np.asarray(result.total_energy, dtype=np.float64)
    constraint_nm = np.asarray(result.constraint_max_error, dtype=np.float64) * 0.1
    report = result.nonbonded_report
    finite = bool(
        np.all(np.isfinite(total_energy))
        and np.all(np.isfinite(np.asarray(result.temperature)))
        and np.all(np.isfinite(constraint_nm))
        and np.all(np.isfinite(np.asarray(result.final_state.positions)))
    )
    row = {
        "dt_fs": dt_fs,
        "duration_ps": duration_ps,
        "steps": config.steps,
        "position_constraint_iterations": (
            0 if constraints is None else constraints.max_iterations
        ),
        "velocity_constraint_iterations": (
            0 if constraints is None else constraints._velocity_iterations
        ),
        "constraint_tolerance_angstrom": (
            0.0 if constraints is None else constraints.tolerance
        ),
        "finite": finite,
        "max_energy_drift_per_atom_kj_mol": float(
            np.max(np.abs(total_energy - total_energy[0])) / atom_count
        ),
        "final_energy_drift_per_atom_kj_mol": float(
            abs(total_energy[-1] - total_energy[0]) / atom_count
        ),
        "max_constraint_error_nm": float(np.max(constraint_nm)),
        "mean_temperature_k": float(np.mean(np.asarray(result.temperature))),
        "wall_time_s": perf_counter() - start,
        "neighbor_backend": report["backend"],
        "neighbor_representation": report["representation_kind"],
        "pair_count": int(report["pair_count"]),
        "candidate_count": report["candidate_count"],
        "candidate_waste_count": report["candidate_waste_count"],
        "rebuild_count": int(report["rebuild_count"]),
        "fallback_reason": report["fallback_reason"],
    }
    raw = {
        "diagnostic_time_ps": np.asarray(result.diagnostic_time),
        "total_energy_kj_mol": total_energy,
        "potential_energy_kj_mol": np.asarray(result.potential_energy),
        "kinetic_energy_kj_mol": np.asarray(result.kinetic_energy),
        "temperature_k": np.asarray(result.temperature),
        "constraint_error_nm": constraint_nm,
        "final_positions": np.asarray(result.final_state.positions),
        "final_velocities": np.asarray(result.final_state.velocities),
    }
    return row, raw


def _run_nvt(
    *,
    positions,
    velocities,
    masses,
    cell,
    force_terms,
    constraints,
    duration_ps: float,
    dt_fs: float,
    temperature_k: float,
    seed: int,
    kinetic_energy_scale: float,
    force_to_acceleration_scale: float,
    boltzmann_constant: float,
):
    config = _simulation_config(
        duration_ps=duration_ps,
        dt_fs=dt_fs,
        kinetic_energy_scale=kinetic_energy_scale,
        force_to_acceleration_scale=force_to_acceleration_scale,
        boltzmann_constant=boltzmann_constant,
    )
    start = perf_counter()
    result = simulate_nvt(
        positions,
        velocities,
        masses=masses,
        cell=cell,
        force_terms=force_terms,
        neighbor_manager=_neighbor_manager(cell),
        config=config,
        thermostat=LangevinThermostat(
            temperature=temperature_k,
            friction=NVT_FRICTION_PER_PS,
            seed=seed,
        ),
        constraints=constraints,
    )
    mx.eval(
        result.total_energy,
        result.temperature,
        result.constraint_max_error,
        result.final_state.positions,
    )
    temperatures = np.asarray(result.temperature, dtype=np.float64)
    constraint_nm = np.asarray(result.constraint_max_error, dtype=np.float64) * 0.1
    report = result.nonbonded_report
    finite = bool(
        np.all(np.isfinite(np.asarray(result.total_energy)))
        and np.all(np.isfinite(temperatures))
        and np.all(np.isfinite(constraint_nm))
        and np.all(np.isfinite(np.asarray(result.final_state.positions)))
    )
    row = {
        "dt_fs": dt_fs,
        "duration_ps": duration_ps,
        "steps": config.steps,
        "friction_per_ps": NVT_FRICTION_PER_PS,
        "position_constraint_iterations": (
            0 if constraints is None else constraints.max_iterations
        ),
        "velocity_constraint_iterations": (
            0 if constraints is None else constraints._velocity_iterations
        ),
        "constraint_tolerance_angstrom": (
            0.0 if constraints is None else constraints.tolerance
        ),
        "finite": finite,
        "mean_temperature_k": float(np.mean(temperatures)),
        "final_temperature_k": float(temperatures[-1]),
        "max_constraint_error_nm": float(np.max(constraint_nm)),
        "wall_time_s": perf_counter() - start,
        "neighbor_backend": report["backend"],
        "neighbor_representation": report["representation_kind"],
        "pair_count": int(report["pair_count"]),
        "candidate_count": report["candidate_count"],
        "candidate_waste_count": report["candidate_waste_count"],
        "rebuild_count": int(report["rebuild_count"]),
        "fallback_reason": report["fallback_reason"],
    }
    raw = {
        "diagnostic_time_ps": np.asarray(result.diagnostic_time),
        "total_energy_kj_mol": np.asarray(result.total_energy),
        "potential_energy_kj_mol": np.asarray(result.potential_energy),
        "kinetic_energy_kj_mol": np.asarray(result.kinetic_energy),
        "temperature_k": temperatures,
        "constraint_error_nm": constraint_nm,
        "final_positions": np.asarray(result.final_state.positions),
        "final_velocities": np.asarray(result.final_state.velocities),
    }
    return row, raw


def _parse_dt_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        msg = "at least one NVE time step is required"
        raise argparse.ArgumentTypeError(msg)
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("target",), default="target")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--nve-ps", type=float, default=1.0)
    parser.add_argument("--nve-dt-fs", type=_parse_dt_list, default=(1.0, 0.5))
    parser.add_argument("--nvt-ps", type=float, default=1.0)
    parser.add_argument("--nvt-dt-fs", type=float, default=1.0)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--minimization-steps", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the command-line target PME stability validation."""

    args = _parse_args()
    payload = run_pme_stability(
        case=args.case,
        reference_dir=args.reference,
        nve_ps=args.nve_ps,
        nve_dt_fs=args.nve_dt_fs,
        nvt_ps=args.nvt_ps,
        nvt_dt_fs=args.nvt_dt_fs,
        temperature_k=args.temperature_k,
        out_dir=args.out,
        seed=args.seed,
        minimization_steps=args.minimization_steps,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
