"""Run a short OpenMM CHARMM/PME GPCRmd reference preview."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from openmm import LangevinMiddleIntegrator, Platform, Vec3, unit
from openmm.app import (
    PME,
    CharmmParameterSet,
    CharmmPsfFile,
    HBonds,
    PDBFile,
    Simulation,
)


def main() -> None:
    args = _parse_args()
    cache_dir = Path(args.cache_dir)
    prepared_dir = Path(args.prepared_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = _PreparedLabels.from_npz(
        np.load(prepared_dir / "prepared_system.npz", allow_pickle=False)
    )
    psf = CharmmPsfFile(str(cache_dir / "15286_dyn_729.psf"))
    pdb = PDBFile(str(cache_dir / "17686_dyn_729.pdb"))
    psf.setBox(
        labels.cell_lengths_A[0] * unit.angstrom,
        labels.cell_lengths_A[1] * unit.angstrom,
        labels.cell_lengths_A[2] * unit.angstrom,
    )
    parameters = CharmmParameterSet(str(cache_dir / "generated_psf_masses.rtf"))
    parameters.readParameterFile(str(cache_dir / "15290_prm_729.prm"), permissive=True)
    system = psf.createSystem(
        parameters,
        nonbondedMethod=PME,
        nonbondedCutoff=args.cutoff_A * unit.angstrom,
        switchDistance=args.switch_A * unit.angstrom,
        constraints=HBonds,
        rigidWater=True,
        ewaldErrorTolerance=args.ewald_tolerance,
    )
    integrator = LangevinMiddleIntegrator(
        args.temperature * unit.kelvin,
        args.friction / unit.picosecond,
        args.dt_ps * unit.picoseconds,
    )
    integrator.setRandomNumberSeed(args.seed)
    platform = Platform.getPlatformByName(args.platform)
    sim = Simulation(psf.topology, system, integrator, platform)
    if args.positions_source == "prepared":
        sim.context.setPositions(
            [Vec3(*(row * 0.1)) for row in labels.positions_A] * unit.nanometer
        )
    else:
        sim.context.setPositions(pdb.positions)

    if args.minimize_steps > 0:
        sim.minimizeEnergy(maxIterations=args.minimize_steps)
    sim.context.setVelocitiesToTemperature(args.temperature * unit.kelvin, args.seed)

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
        atom_count=labels.atom_count,
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
        sim.step(args.sample_interval)
        _sample(
            sim,
            atom_count=labels.atom_count,
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
        "kind": "gpcrmd_openmm_charmm_md",
        "engine": "openmm",
        "artifact_label": "openmm-reference",
        "workflow": "openmm_charmm_pme_md",
        "platform": sim.context.getPlatform().getName(),
        "platform_properties": _platform_properties(platform, sim),
        "cache_dir": str(cache_dir),
        "prepared_artifact": str(prepared_dir),
        "dataset_id": out_dir.name,
        "steps": args.steps,
        "sample_interval": args.sample_interval,
        "dt_ps": args.dt_ps,
        "temperature_K": args.temperature,
        "friction_per_ps": args.friction,
        "cutoff_A": args.cutoff_A,
        "switch_A": args.switch_A,
        "ewald_tolerance": args.ewald_tolerance,
        "constraints": "HBonds",
        "rigid_water": True,
        "minimize_steps": args.minimize_steps,
        "positions_source": args.positions_source,
        "elapsed_wall_seconds": elapsed,
        "integration_steps_per_second": args.steps / elapsed,
        "gpu_visible_atoms": labels.atom_count,
        "note": (
            "OpenMM CHARMM/PME reference preview using GPCRmd PSF/PDB/PRM inputs. "
            "This is not production runtime output and is not validated parity with "
            "the original ACEMD run."
        ),
    }
    np.savez_compressed(
        trajectory_path,
        positions=positions,
        time_ps=np.asarray(sampled_time, dtype=np.float32),
        symbols=labels.symbols.astype(str),
        atom_names=labels.atom_names.astype(str),
        residue_names=labels.residue_names.astype(str),
        residue_ids=labels.residue_ids.astype(np.int32),
        segment_ids=labels.chain_ids.astype(str),
        ligand_indices=np.flatnonzero(labels.ligand_mask).astype(np.int32),
        receptor_indices=np.flatnonzero(labels.receptor_mask).astype(np.int32),
        water_indices=np.flatnonzero(labels.water_mask).astype(np.int32),
        ion_indices=np.flatnonzero(labels.ion_mask).astype(np.int32),
        lipid_indices=np.flatnonzero(labels.lipid_mask).astype(np.int32),
        cell_lengths_A=labels.cell_lengths_A.astype(np.float32),
        source_json=np.asarray(json.dumps(source)),
    )
    native_path = out_dir / "openmm_charmm_md_trajectory.npz"
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
        symbols=labels.symbols.astype(str),
        cell=labels.cell_lengths_A.astype(np.float32),
        metadata_json=np.asarray(json.dumps(source)),
    )
    report = {
        "status": "ran",
        "engine": "openmm",
        "artifact_label": "openmm-reference",
        "workflow": "openmm_charmm_pme_md",
        "platform": sim.context.getPlatform().getName(),
        "platform_properties": source["platform_properties"],
        "cache_dir": str(cache_dir),
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
        "minimize_steps": args.minimize_steps,
        "cutoff_A": args.cutoff_A,
        "switch_A": args.switch_A,
        "ewald_tolerance": args.ewald_tolerance,
    }
    (out_dir / "openmm_charmm_md_run_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


class _PreparedLabels:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    @classmethod
    def from_npz(cls, data) -> _PreparedLabels:
        return cls(
            symbols=np.asarray(data["symbols"]).astype(str),
            atom_names=np.asarray(data["atom_names"]).astype(str),
            residue_names=np.asarray(data["residue_names"]).astype(str),
            residue_ids=np.asarray(data["residue_ids"], dtype=np.int32),
            chain_ids=np.asarray(data["chain_ids"]).astype(str),
            ligand_mask=np.asarray(data["ligand_mask"], dtype=bool),
            receptor_mask=np.asarray(data["receptor_mask"], dtype=bool),
            water_mask=np.asarray(data["water_mask"], dtype=bool),
            ion_mask=np.asarray(data["ion_mask"], dtype=bool),
            lipid_mask=np.asarray(data["lipid_mask"], dtype=bool),
            cell_lengths_A=np.asarray(data["cell_lengths"], dtype=np.float64),
            positions_A=np.asarray(data["positions"], dtype=np.float64),
        )

    @property
    def atom_count(self) -> int:
        return int(self.symbols.shape[0])


def _sample(
    sim: Simulation,
    *,
    atom_count: int,
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
    dof = max(1, 3 * atom_count)
    gas_constant = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
        unit.kilojoule_per_mole / unit.kelvin
    )
    temperature.append(float(2.0 * ke / (dof * gas_constant)))


def _platform_properties(platform: Platform, sim: Simulation) -> dict[str, str]:
    properties: dict[str, str] = {}
    for name in platform.getPropertyNames():
        try:
            properties[name] = platform.getPropertyValue(sim.context, name)
        except Exception:
            continue
    return properties


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default="notebooks/ligand-receptor-motion/data/gpcrmd-cache/729",
    )
    parser.add_argument(
        "--prepared-dir",
        default="notebooks/ligand-receptor-motion/data/gpcrmd-mlx/729-50steps-sample50",
    )
    parser.add_argument(
        "--out",
        default="notebooks/ligand-receptor-motion/data/openmm-md/729-2000-opencl-charmm-pme",
    )
    parser.add_argument("--platform", default="OpenCL")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--sample-interval", type=int, default=20)
    parser.add_argument("--dt-ps", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=310.0)
    parser.add_argument("--friction", type=float, default=0.1)
    parser.add_argument("--cutoff-A", type=float, default=9.0)
    parser.add_argument("--switch-A", type=float, default=7.5)
    parser.add_argument("--ewald-tolerance", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--minimize-steps", type=int, default=100)
    parser.add_argument("--positions-source", choices=("prepared", "pdb"), default="prepared")
    return parser.parse_args()


if __name__ == "__main__":
    main()
