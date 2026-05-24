"""Run a short OpenMM-vs-MLX NPT volume sanity check on the AMBER fixture."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openmm_mlx_parity import (
    DEFAULT_AMBER_FIXTURE,
    PMEParityConfig,
    _pme_config_payload,
    _pme_readiness,
    _with_pme_artifact_settings,
    default_amber_fixture_paths,
)

from mlx_atomistic.artifacts import build_mlx_system_from_artifact, load_prepared_mlx_artifact
from mlx_atomistic.md import (
    LangevinThermostat,
    MonteCarloBarostat,
    SimulationConfig,
    simulate_npt,
)
from mlx_atomistic.prep.io import save_prepared_system
from mlx_atomistic.prep.runner import initialize_velocities
from mlx_atomistic.prep.topology_import import import_amber_prmtop
from mlx_atomistic.units import ATM_TO_KJ_PER_MOL_ANGSTROM3

REPORT_NAME = "openmm_mlx_npt_parity_report.json"


@dataclass(frozen=True)
class NPTParityReport:
    """Machine-readable short NPT comparison."""

    status: str
    fixture: str
    atom_count: int
    prepared_dir: str
    prmtop_path: str
    coords_path: str
    steps: int
    dt_ps: float
    temperature_K: float
    pressure_atm: float
    openmm_platform: str | None
    pme_config: dict[str, Any]
    pme_readiness: dict[str, Any] | None
    openmm_initial_volume_angstrom3: float | None
    openmm_final_volume_angstrom3: float | None
    mlx_initial_volume_angstrom3: float | None
    mlx_final_volume_angstrom3: float | None
    openmm_volume_ratio: float | None
    mlx_volume_ratio: float | None
    volume_ratio_abs_delta: float | None
    mlx_barostat_attempts: int
    mlx_barostat_accepted: int
    volume_ratio_tolerance: float
    passed: bool
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blockers"] = list(self.blockers)
        return payload


def main() -> None:
    args = _parse_args()
    prmtop_path, coords_path = _fixture_paths(args)
    report = run_npt_parity(
        prmtop_path=prmtop_path,
        coords_path=coords_path,
        out_dir=args.out,
        fixture=args.fixture,
        platform_name=args.platform,
        steps=args.steps,
        dt_ps=args.dt,
        temperature_K=args.temperature,
        pressure_atm=args.pressure_atm,
        friction_per_ps=args.friction,
        seed=args.seed,
        volume_ratio_tolerance=args.volume_ratio_tolerance,
        pme_config=_pme_config(args),
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if report.status == "blocked":
        raise SystemExit(2)
    if not report.passed:
        raise SystemExit(1)


def run_npt_parity(
    *,
    prmtop_path: str | Path,
    coords_path: str | Path,
    out_dir: str | Path,
    fixture: str = DEFAULT_AMBER_FIXTURE,
    platform_name: str = "Reference",
    steps: int = 10,
    dt_ps: float = 0.001,
    temperature_K: float = 300.0,
    pressure_atm: float = 1.0,
    friction_per_ps: float = 1.0,
    seed: int = 7,
    volume_ratio_tolerance: float = 0.25,
    pme_config: PMEParityConfig | None = None,
) -> NPTParityReport:
    """Compare short-run volume behavior for the same periodic AMBER fixture."""

    pme_config = PMEParityConfig() if pme_config is None else pme_config
    prmtop = Path(prmtop_path)
    coords = Path(coords_path)
    out = Path(out_dir)
    prepared_dir = out / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    if not prmtop.exists():
        return _blocked_report(
            fixture=fixture,
            prmtop_path=prmtop,
            coords_path=coords,
            prepared_dir=prepared_dir,
            steps=steps,
            dt_ps=dt_ps,
            temperature_K=temperature_K,
            pressure_atm=pressure_atm,
            pme_config=pme_config,
            blocker=f"missing AMBER prmtop: {prmtop}",
            volume_ratio_tolerance=volume_ratio_tolerance,
        )
    if not coords.exists():
        return _blocked_report(
            fixture=fixture,
            prmtop_path=prmtop,
            coords_path=coords,
            prepared_dir=prepared_dir,
            steps=steps,
            dt_ps=dt_ps,
            temperature_K=temperature_K,
            pressure_atm=pressure_atm,
            pme_config=pme_config,
            blocker=f"missing AMBER coordinates: {coords}",
            volume_ratio_tolerance=volume_ratio_tolerance,
        )

    prepared = _with_pme_artifact_settings(
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords),
        pme_config,
    )
    save_prepared_system(prepared, prepared_dir)
    artifact = load_prepared_mlx_artifact(prepared_dir, require_production=True)
    artifact.metadata["nonbonded_cutoff"] = float(pme_config.real_cutoff_angstrom)
    artifact.metadata["electrostatics_model"] = "pme"
    artifact.metadata["pme_config"] = _pme_config_payload(pme_config)
    readiness = _pme_readiness(artifact, pme_config)
    if readiness is not None and readiness["status"] != "ready":
        return _blocked_report(
            fixture=fixture,
            prmtop_path=prmtop,
            coords_path=coords,
            prepared_dir=prepared_dir,
            steps=steps,
            dt_ps=dt_ps,
            temperature_K=temperature_K,
            pressure_atm=pressure_atm,
            pme_config=pme_config,
            pme_readiness=readiness,
            blocker="PME readiness blocked: " + ", ".join(readiness["blockers"]),
            volume_ratio_tolerance=volume_ratio_tolerance,
            atom_count=artifact.atom_count,
        )

    system, force_terms, constraints = build_mlx_system_from_artifact(artifact)
    unit_system = artifact.unit_system
    kinetic_energy_scale = 1.0 if unit_system is None else unit_system.kinetic_energy_scale
    force_to_acceleration_scale = (
        1.0 if unit_system is None else unit_system.force_to_acceleration_scale
    )
    boltzmann_constant = 1.0 if unit_system is None else unit_system.boltzmann_constant
    velocities = initialize_velocities(
        prepared,
        np.asarray(system.masses, dtype=np.float32),
        temperature=temperature_K,
        seed=seed,
        kinetic_energy_scale=kinetic_energy_scale,
        boltzmann_constant=boltzmann_constant,
    )
    mlx_result = simulate_npt(
        system.positions,
        velocities,
        masses=system.masses,
        cell=system.cell,
        force_terms=force_terms,
        constraints=constraints,
        config=SimulationConfig(
            dt=dt_ps,
            steps=steps,
            sample_interval=max(1, steps),
            diagnostic_interval=max(1, steps),
            kinetic_energy_scale=kinetic_energy_scale,
            force_to_acceleration_scale=force_to_acceleration_scale,
            boltzmann_constant=boltzmann_constant,
            pressure_diagnostics=False,
            compile_force_evaluator=False,
        ),
        thermostat=LangevinThermostat(
            temperature=temperature_K,
            friction=friction_per_ps,
            seed=seed,
        ),
        barostat=MonteCarloBarostat(
            pressure=pressure_atm * ATM_TO_KJ_PER_MOL_ANGSTROM3,
            temperature=temperature_K,
            interval=max(1, steps),
            seed=seed,
        ),
    )
    openmm_result = _run_openmm_npt(
        prmtop_path=prmtop,
        coords_path=coords,
        pme_config=pme_config,
        platform_name=platform_name,
        steps=steps,
        dt_ps=dt_ps,
        temperature_K=temperature_K,
        pressure_atm=pressure_atm,
        friction_per_ps=friction_per_ps,
        seed=seed,
    )
    mlx_initial_volume = float(np.asarray(mlx_result.volume[0]))
    mlx_final_volume = float(np.asarray(mlx_result.volume[-1]))
    openmm_initial_volume = float(openmm_result["initial_volume_angstrom3"])
    openmm_final_volume = float(openmm_result["final_volume_angstrom3"])
    mlx_ratio = mlx_final_volume / mlx_initial_volume
    openmm_ratio = openmm_final_volume / openmm_initial_volume
    ratio_delta = abs(mlx_ratio - openmm_ratio)
    passed = bool(
        np.isfinite([mlx_ratio, openmm_ratio, ratio_delta]).all()
        and ratio_delta <= volume_ratio_tolerance
    )
    report = NPTParityReport(
        status="passed" if passed else "failed",
        fixture=fixture,
        atom_count=int(system.atom_count),
        prepared_dir=str(prepared_dir),
        prmtop_path=str(prmtop),
        coords_path=str(coords),
        steps=int(steps),
        dt_ps=float(dt_ps),
        temperature_K=float(temperature_K),
        pressure_atm=float(pressure_atm),
        openmm_platform=str(openmm_result["platform"]),
        pme_config=_pme_config_payload(pme_config),
        pme_readiness=readiness,
        openmm_initial_volume_angstrom3=openmm_initial_volume,
        openmm_final_volume_angstrom3=openmm_final_volume,
        mlx_initial_volume_angstrom3=mlx_initial_volume,
        mlx_final_volume_angstrom3=mlx_final_volume,
        openmm_volume_ratio=float(openmm_ratio),
        mlx_volume_ratio=float(mlx_ratio),
        volume_ratio_abs_delta=float(ratio_delta),
        mlx_barostat_attempts=int(mlx_result.barostat_attempts),
        mlx_barostat_accepted=int(mlx_result.barostat_accepted),
        volume_ratio_tolerance=float(volume_ratio_tolerance),
        passed=passed,
    )
    _write_report(report, out)
    return report


def _run_openmm_npt(
    *,
    prmtop_path: Path,
    coords_path: Path,
    pme_config: PMEParityConfig,
    platform_name: str,
    steps: int,
    dt_ps: float,
    temperature_K: float,
    pressure_atm: float,
    friction_per_ps: float,
    seed: int,
) -> dict[str, Any]:
    import openmm as mm
    from openmm import app, unit

    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    coords = app.AmberInpcrdFile(str(coords_path))
    a, b, c = (float(item) * 0.1 for item in pme_config.cell_lengths_angstrom)
    box_vectors = (
        mm.Vec3(a, 0.0, 0.0),
        mm.Vec3(0.0, b, 0.0),
        mm.Vec3(0.0, 0.0, c),
    ) * unit.nanometer
    prmtop.topology.setPeriodicBoxVectors(box_vectors)
    system = prmtop.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=pme_config.real_cutoff_angstrom * 0.1 * unit.nanometer,
        constraints=None,
        removeCMMotion=False,
    )
    system.setDefaultPeriodicBoxVectors(*box_vectors)
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if isinstance(force, mm.NonbondedForce):
            force.setPMEParameters(
                pme_config.alpha_per_angstrom * 10.0 / unit.nanometer,
                *pme_config.mesh_shape,
            )
    barostat = mm.MonteCarloBarostat(
        pressure_atm * unit.atmospheres,
        temperature_K * unit.kelvin,
        max(1, steps),
    )
    barostat.setRandomNumberSeed(seed)
    system.addForce(barostat)
    integrator = mm.LangevinMiddleIntegrator(
        temperature_K * unit.kelvin,
        friction_per_ps / unit.picosecond,
        dt_ps * unit.picoseconds,
    )
    integrator.setRandomNumberSeed(seed)
    context = mm.Context(
        system,
        integrator,
        mm.Platform.getPlatformByName(platform_name),
    )
    context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(coords.positions)
    context.setVelocitiesToTemperature(temperature_K * unit.kelvin, seed)
    initial = context.getState()
    integrator.step(int(steps))
    final = context.getState()
    return {
        "platform": context.getPlatform().getName(),
        "initial_volume_angstrom3": _state_volume_angstrom3(initial),
        "final_volume_angstrom3": _state_volume_angstrom3(final),
    }


def _state_volume_angstrom3(state: Any) -> float:
    from openmm import unit

    return float(state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3) * 1000.0)


def _blocked_report(
    *,
    fixture: str,
    prmtop_path: Path,
    coords_path: Path,
    prepared_dir: Path,
    steps: int,
    dt_ps: float,
    temperature_K: float,
    pressure_atm: float,
    pme_config: PMEParityConfig,
    blocker: str,
    volume_ratio_tolerance: float,
    atom_count: int = 0,
    pme_readiness: dict[str, Any] | None = None,
) -> NPTParityReport:
    report = NPTParityReport(
        status="blocked",
        fixture=fixture,
        atom_count=int(atom_count),
        prepared_dir=str(prepared_dir),
        prmtop_path=str(prmtop_path),
        coords_path=str(coords_path),
        steps=int(steps),
        dt_ps=float(dt_ps),
        temperature_K=float(temperature_K),
        pressure_atm=float(pressure_atm),
        openmm_platform=None,
        pme_config=_pme_config_payload(pme_config),
        pme_readiness=pme_readiness,
        openmm_initial_volume_angstrom3=None,
        openmm_final_volume_angstrom3=None,
        mlx_initial_volume_angstrom3=None,
        mlx_final_volume_angstrom3=None,
        openmm_volume_ratio=None,
        mlx_volume_ratio=None,
        volume_ratio_abs_delta=None,
        mlx_barostat_attempts=0,
        mlx_barostat_accepted=0,
        volume_ratio_tolerance=float(volume_ratio_tolerance),
        passed=False,
        blockers=(blocker,),
    )
    _write_report(report, prepared_dir.parent)
    return report


def _fixture_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.prmtop is not None or args.coords is not None:
        if args.prmtop is None or args.coords is None:
            msg = "--prmtop and --coords must be provided together"
            raise SystemExit(msg)
        return Path(args.prmtop), Path(args.coords)
    if args.fixture != DEFAULT_AMBER_FIXTURE:
        msg = (
            f"unknown fixture {args.fixture!r}; pass --prmtop and --coords for a "
            "custom AMBER fixture"
        )
        raise SystemExit(msg)
    return default_amber_fixture_paths(Path("."))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        default=DEFAULT_AMBER_FIXTURE,
        help="fixture label; default uses the tracked small AMBER fixture",
    )
    parser.add_argument("--prmtop", type=Path, help="custom AMBER prmtop path")
    parser.add_argument("--coords", type=Path, help="custom AMBER inpcrd/rst7 path")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/md-engine-gap-closure/npt-parity"),
    )
    parser.add_argument("--platform", default="Reference", help="OpenMM platform name")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.001, help="time step in ps")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--pressure-atm", type=float, default=1.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--volume-ratio-tolerance", type=float, default=0.25)
    parser.add_argument(
        "--pme-mesh",
        default="48,48,48",
        help="PME mesh dimensions as nx,ny,nz",
    )
    parser.add_argument("--pme-alpha-per-angstrom", type=float, default=0.35)
    parser.add_argument("--pme-real-cutoff-angstrom", type=float, default=10.0)
    parser.add_argument(
        "--pme-cell-angstrom",
        default="40,40,40",
        help="orthorhombic PME box lengths as a,b,c in Angstrom",
    )
    return parser.parse_args()


def _pme_config(args: argparse.Namespace) -> PMEParityConfig:
    return PMEParityConfig(
        mesh_shape=_parse_triplet(args.pme_mesh, cast=int),
        alpha_per_angstrom=args.pme_alpha_per_angstrom,
        real_cutoff_angstrom=args.pme_real_cutoff_angstrom,
        cell_lengths_angstrom=_parse_triplet(args.pme_cell_angstrom, cast=float),
    )


def _parse_triplet(value: str, *, cast):
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        msg = f"expected three comma-separated values, got {value!r}"
        raise SystemExit(msg)
    return tuple(cast(part) for part in parts)


def _write_report(report: NPTParityReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / REPORT_NAME).write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
