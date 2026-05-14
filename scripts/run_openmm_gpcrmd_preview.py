"""Run an OpenMM reference preview trajectory from a prepared GPCRmd artifact."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from openmm import (
    CustomExternalForce,
    LangevinMiddleIntegrator,
    NonbondedForce,
    Platform,
    System,
    Vec3,
    unit,
)
from openmm.app import Element, Simulation, Topology


def main() -> None:
    args = _parse_args()
    prepared_dir = Path(args.prepared_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared = np.load(prepared_dir / "prepared_system.npz", allow_pickle=False)
    metadata = json.loads((prepared_dir / "prepared_system.json").read_text())
    arrays = _PreparedArrays.from_npz(prepared)
    platform = Platform.getPlatformByName(args.platform)

    topology = _topology_from_arrays(arrays)
    ligand_translation_A = np.asarray(args.ligand_translation_A, dtype=np.float64)
    system, restraint = _system_from_arrays(
        arrays,
        nonbonded_mode=args.nonbonded_mode,
        cutoff_nm=args.cutoff_nm,
        restraint_k=args.restraint_k,
        ligand_restraint_k=args.ligand_restraint_k,
        ligand_translation_A=ligand_translation_A,
    )
    integrator = LangevinMiddleIntegrator(
        args.temperature * unit.kelvin,
        args.friction / unit.picosecond,
        args.dt_ps * unit.picoseconds,
    )
    integrator.setRandomNumberSeed(args.seed)

    sim = Simulation(topology, system, integrator, platform)
    sim.context.setPositions(_positions_to_openmm(arrays.positions_A))
    if args.use_prepared_velocities and np.any(arrays.velocities_A_per_ps):
        sim.context.setVelocities(
            [
                Vec3(*(row * 0.1))
                for row in np.asarray(arrays.velocities_A_per_ps, dtype=np.float64)
            ]
            * unit.nanometer
            / unit.picosecond
        )
    else:
        sim.context.setVelocitiesToTemperature(args.temperature * unit.kelvin, args.seed)
    _update_restraint_targets(
        restraint,
        sim,
        arrays=arrays,
        progress=0.0,
        ligand_translation_A=ligand_translation_A,
    )

    sampled_positions: list[np.ndarray] = []
    sampled_velocities: list[np.ndarray] = []
    sampled_steps: list[int] = []
    sampled_time: list[float] = []
    potential_energy: list[float] = []
    kinetic_energy: list[float] = []
    temperature: list[float] = []

    t0 = time.perf_counter()
    _sample(
        sim,
        arrays=arrays,
        sampled_positions=sampled_positions,
        sampled_velocities=sampled_velocities,
        sampled_steps=sampled_steps,
        sampled_time=sampled_time,
        potential_energy=potential_energy,
        kinetic_energy=kinetic_energy,
        temperature=temperature,
        step=0,
        dt_ps=args.dt_ps,
    )
    for step in range(args.sample_interval, args.steps + 1, args.sample_interval):
        progress = step / args.steps
        _update_restraint_targets(
            restraint,
            sim,
            arrays=arrays,
            progress=progress,
            ligand_translation_A=ligand_translation_A,
        )
        sim.step(args.sample_interval)
        _sample(
            sim,
            arrays=arrays,
            sampled_positions=sampled_positions,
            sampled_velocities=sampled_velocities,
            sampled_steps=sampled_steps,
            sampled_time=sampled_time,
            potential_energy=potential_energy,
            kinetic_energy=kinetic_energy,
            temperature=temperature,
            step=step,
            dt_ps=args.dt_ps,
        )
    elapsed = time.perf_counter() - t0

    positions = np.asarray(sampled_positions, dtype=np.float32)
    trajectory_path = out_dir / "processed_trajectory.npz"
    source = {
        "kind": "gpcrmd_openmm_preview",
        "engine": "openmm",
        "artifact_label": "openmm-reference",
        "workflow": "openmm_short_range_preview",
        "platform": sim.context.getPlatform().getName(),
        "platform_properties": _platform_properties(platform, sim),
        "prepared_artifact": str(prepared_dir),
        "dataset_id": out_dir.name,
        "steps": args.steps,
        "sample_interval": args.sample_interval,
        "dt_ps": args.dt_ps,
        "cutoff_nm": args.cutoff_nm,
        "nonbonded_mode": args.nonbonded_mode,
        "restraint_k_kj_mol_nm2": args.restraint_k,
        "ligand_restraint_k_kj_mol_nm2": args.ligand_restraint_k,
        "ligand_translation_A": ligand_translation_A.tolist(),
        "temperature_K": args.temperature,
        "friction_per_ps": args.friction,
        "elapsed_wall_seconds": elapsed,
        "integration_steps_per_second": args.steps / elapsed,
        "gpu_visible_atoms": int(arrays.positions_A.shape[0]),
        "note": (
            "OpenMM reference preview for notebook visualization; not production "
            "runtime output, full GPCRmd "
            "production physics, PME, constraints, or complete bonded force parity."
        ),
    }
    np.savez_compressed(
        trajectory_path,
        positions=positions,
        time_ps=np.asarray(sampled_time, dtype=np.float32),
        symbols=arrays.symbols.astype(str),
        atom_names=arrays.atom_names.astype(str),
        residue_names=arrays.residue_names.astype(str),
        residue_ids=arrays.residue_ids.astype(np.int32),
        segment_ids=arrays.chain_ids.astype(str),
        ligand_indices=np.flatnonzero(arrays.ligand_mask).astype(np.int32),
        receptor_indices=np.flatnonzero(arrays.receptor_mask).astype(np.int32),
        water_indices=np.flatnonzero(arrays.water_mask).astype(np.int32),
        ion_indices=np.flatnonzero(arrays.ion_mask).astype(np.int32),
        lipid_indices=np.flatnonzero(arrays.lipid_mask).astype(np.int32),
        cell_lengths_A=arrays.cell_lengths_A.astype(np.float32),
        source_json=np.asarray(json.dumps(source)),
    )
    native_path = out_dir / "openmm_preview_trajectory.npz"
    total_energy = np.asarray(potential_energy) + np.asarray(kinetic_energy)
    np.savez_compressed(
        native_path,
        sampled_positions=positions,
        sampled_velocities=np.asarray(sampled_velocities, dtype=np.float32),
        sampled_steps=np.asarray(sampled_steps, dtype=np.int32),
        sampled_time=np.asarray(sampled_time, dtype=np.float32),
        potential_energy=np.asarray(potential_energy, dtype=np.float32),
        kinetic_energy=np.asarray(kinetic_energy, dtype=np.float32),
        total_energy=total_energy.astype(np.float32),
        temperature=np.asarray(temperature, dtype=np.float32),
        symbols=arrays.symbols.astype(str),
        cell=arrays.cell_lengths_A.astype(np.float32),
        metadata_json=np.asarray(json.dumps(source)),
    )
    report = {
        "status": "ran",
        "engine": "openmm",
        "artifact_label": "openmm-reference",
        "workflow": "openmm_short_range_preview",
        "platform": sim.context.getPlatform().getName(),
        "platform_properties": source["platform_properties"],
        "prepared_dir": str(prepared_dir),
        "trajectory_path": str(trajectory_path),
        "native_trajectory_path": str(native_path),
        "steps": args.steps,
        "sample_interval": args.sample_interval,
        "sampled_frame_count": len(sampled_steps),
        "dt_ps": args.dt_ps,
        "simulated_time_ps": args.steps * args.dt_ps,
        "elapsed_wall_seconds": elapsed,
        "integration_steps_per_second": args.steps / elapsed,
        "potential_energy_kj_mol": potential_energy,
        "kinetic_energy_kj_mol": kinetic_energy,
        "temperature_K": temperature,
        "ligand_translation_A": ligand_translation_A.tolist(),
        "prepared_metadata": {
            "parameter_source": metadata.get("parameter_source"),
            "units": metadata.get("units"),
        },
    }
    (out_dir / "openmm_preview_run_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


class _PreparedArrays:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    @classmethod
    def from_npz(cls, data) -> _PreparedArrays:
        return cls(
            symbols=np.asarray(data["symbols"]).astype(str),
            atom_names=np.asarray(data["atom_names"]).astype(str),
            residue_names=np.asarray(data["residue_names"]).astype(str),
            residue_ids=np.asarray(data["residue_ids"], dtype=np.int32),
            chain_ids=np.asarray(data["chain_ids"]).astype(str),
            positions_A=np.asarray(data["positions"], dtype=np.float64),
            velocities_A_per_ps=np.asarray(data["velocities"], dtype=np.float64),
            masses=np.asarray(data["masses"], dtype=np.float64),
            charges=np.asarray(data["charges"], dtype=np.float64),
            sigma_A=np.asarray(data["sigma"], dtype=np.float64),
            epsilon=np.asarray(data["epsilon"], dtype=np.float64),
            exception_pairs=np.asarray(data["nonbonded_exception_pairs"], dtype=np.int32),
            exception_charge_product=np.asarray(
                data["nonbonded_exception_charge_product"], dtype=np.float64
            ),
            exception_sigma_A=np.asarray(data["nonbonded_exception_sigma"], dtype=np.float64),
            exception_epsilon=np.asarray(data["nonbonded_exception_epsilon"], dtype=np.float64),
            ligand_mask=np.asarray(data["ligand_mask"], dtype=bool),
            receptor_mask=np.asarray(data["receptor_mask"], dtype=bool),
            water_mask=np.asarray(data["water_mask"], dtype=bool),
            ion_mask=np.asarray(data["ion_mask"], dtype=bool),
            lipid_mask=np.asarray(data["lipid_mask"], dtype=bool),
            cell_lengths_A=np.asarray(data["cell_lengths"], dtype=np.float64),
        )


def _topology_from_arrays(arrays: _PreparedArrays) -> Topology:
    topology = Topology()
    cell_nm = arrays.cell_lengths_A * 0.1
    topology.setPeriodicBoxVectors(
        (
            Vec3(cell_nm[0], 0, 0),
            Vec3(0, cell_nm[1], 0),
            Vec3(0, 0, cell_nm[2]),
        )
    )
    chains: dict[str, Any] = {}
    residue = None
    previous_residue_key = None
    for index, symbol in enumerate(arrays.symbols):
        chain_id = str(arrays.chain_ids[index] or "A")
        chain = chains.setdefault(chain_id, topology.addChain(chain_id))
        residue_key = (
            chain_id,
            int(arrays.residue_ids[index]),
            str(arrays.residue_names[index]),
        )
        if residue_key != previous_residue_key:
            residue = topology.addResidue(str(arrays.residue_names[index]), chain)
            previous_residue_key = residue_key
        if residue is None:
            msg = "failed to create OpenMM topology residue"
            raise RuntimeError(msg)
        topology.addAtom(
            str(arrays.atom_names[index]),
            _element_for_symbol(symbol),
            residue,
        )
    return topology


def _system_from_arrays(
    arrays: _PreparedArrays,
    *,
    nonbonded_mode: str,
    cutoff_nm: float,
    restraint_k: float,
    ligand_restraint_k: float,
    ligand_translation_A: np.ndarray,
) -> tuple[System, CustomExternalForce]:
    system = System()
    cell_nm = arrays.cell_lengths_A * 0.1
    system.setDefaultPeriodicBoxVectors(
        Vec3(cell_nm[0], 0, 0),
        Vec3(0, cell_nm[1], 0),
        Vec3(0, 0, cell_nm[2]),
    )
    nonbonded = None
    if nonbonded_mode == "short-range":
        nonbonded = NonbondedForce()
        nonbonded.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
        nonbonded.setCutoffDistance(cutoff_nm * unit.nanometer)
        nonbonded.setUseSwitchingFunction(False)
        nonbonded.setUseDispersionCorrection(False)
    restraint = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.setName("moving_preview_restraint")
    restraint.addPerParticleParameter("k")
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")
    translating_ligand = bool(np.linalg.norm(ligand_translation_A) > 0)
    for index, position_A in enumerate(arrays.positions_A):
        system.addParticle(max(float(arrays.masses[index]), 1.0) * unit.dalton)
        if nonbonded is not None:
            nonbonded.addParticle(
                float(arrays.charges[index]) * unit.elementary_charge,
                max(float(arrays.sigma_A[index] * 0.1), 1.0e-6) * unit.nanometer,
                max(float(arrays.epsilon[index]), 0.0) * unit.kilojoule_per_mole,
            )
        particle_k = (
            ligand_restraint_k
            if translating_ligand and arrays.ligand_mask[index]
            else restraint_k
        )
        restraint.addParticle(index, [particle_k, *list(position_A * 0.1)])
    if nonbonded is not None:
        for (i, j), charge_product, sigma_A, epsilon in zip(
            arrays.exception_pairs,
            arrays.exception_charge_product,
            arrays.exception_sigma_A,
            arrays.exception_epsilon,
            strict=True,
        ):
            nonbonded.addException(
                int(i),
                int(j),
                float(charge_product) * unit.elementary_charge**2,
                max(float(sigma_A * 0.1), 1.0e-6) * unit.nanometer,
                max(float(epsilon), 0.0) * unit.kilojoule_per_mole,
            )
        system.addForce(nonbonded)
    elif nonbonded_mode != "none":
        msg = f"unknown nonbonded mode: {nonbonded_mode}"
        raise ValueError(msg)
    system.addForce(restraint)
    return system, restraint


def _update_restraint_targets(
    restraint: CustomExternalForce,
    sim: Simulation,
    *,
    arrays: _PreparedArrays,
    progress: float,
    ligand_translation_A: np.ndarray,
) -> None:
    translation_nm = ligand_translation_A * 0.1 * progress
    if not np.any(translation_nm):
        return
    for index, position_A in enumerate(arrays.positions_A):
        k, x0, y0, z0 = restraint.getParticleParameters(index)[1]
        if arrays.ligand_mask[index]:
            target = position_A * 0.1 + translation_nm
            restraint.setParticleParameters(index, index, [k, *target.tolist()])
        else:
            restraint.setParticleParameters(index, index, [k, x0, y0, z0])
    restraint.updateParametersInContext(sim.context)


def _sample(
    sim: Simulation,
    *,
    arrays: _PreparedArrays,
    sampled_positions: list[np.ndarray],
    sampled_velocities: list[np.ndarray],
    sampled_steps: list[int],
    sampled_time: list[float],
    potential_energy: list[float],
    kinetic_energy: list[float],
    temperature: list[float],
    step: int,
    dt_ps: float,
) -> None:
    state = sim.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    sampled_positions.append(
        state.getPositions(asNumpy=True).value_in_unit(unit.angstrom).astype(np.float32)
    )
    sampled_velocities.append(
        state.getVelocities(asNumpy=True)
        .value_in_unit(unit.angstrom / unit.picosecond)
        .astype(np.float32)
    )
    sampled_steps.append(step)
    sampled_time.append(step * dt_ps)
    pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    potential_energy.append(float(pe))
    kinetic_energy.append(float(ke))
    dof = max(1, 3 * arrays.positions_A.shape[0])
    temperature.append(float(2.0 * ke / (dof * unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
        unit.kilojoule_per_mole / unit.kelvin
    ))))


def _positions_to_openmm(positions_A: np.ndarray):
    return [Vec3(*(row * 0.1)) for row in positions_A] * unit.nanometer


def _platform_properties(platform: Platform, sim: Simulation) -> dict[str, str]:
    properties: dict[str, str] = {}
    for name in platform.getPropertyNames():
        try:
            properties[name] = platform.getPropertyValue(sim.context, name)
        except Exception:
            continue
    return properties


def _element_for_symbol(symbol: str) -> Element:
    normalized = str(symbol).strip().upper()
    atomic_numbers = {
        "H": 1,
        "C": 6,
        "N": 7,
        "O": 8,
        "P": 15,
        "S": 16,
        "CL": 17,
        "K": 19,
        "CA": 20,
        "MG": 12,
        "ZN": 30,
        "NA": 11,
    }
    return Element.getByAtomicNumber(atomic_numbers.get(normalized, 6))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-dir",
        default="notebooks/ligand-receptor-motion/data/gpcrmd-mlx/729-50steps-sample50",
    )
    parser.add_argument(
        "--out",
        default="notebooks/ligand-receptor-motion/data/openmm-preview/729-2000-opencl-preview",
    )
    parser.add_argument("--platform", default="OpenCL")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--sample-interval", type=int, default=20)
    parser.add_argument("--dt-ps", type=float, default=0.0005)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cutoff-nm", type=float, default=1.0)
    parser.add_argument("--restraint-k", type=float, default=1000.0)
    parser.add_argument("--ligand-restraint-k", type=float, default=1000.0)
    parser.add_argument("--ligand-translation-A", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument(
        "--nonbonded-mode",
        choices=("short-range", "none"),
        default="short-range",
    )
    parser.add_argument("--use-prepared-velocities", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
