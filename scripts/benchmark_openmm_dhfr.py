"""Run OpenMM reference DHFR benchmark rows.

This is a reference-engine script. OpenMM stays outside the product runtime and
is imported only after CLI validation and input-readiness checks pass.
"""

from __future__ import annotations

import argparse
import json
import math
import platform as platform_module
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_atomistic.benchmarks import get_hardware_info, normalize_benchmark_payload

BENCHMARK_NAME = "openmm_dhfr_reference"
ENGINE = "openmm-reference"
TIMING_METRIC = "ns_per_day"
COMMAND = "uv run python scripts/benchmark_openmm_dhfr.py"

OPENMM_DHFR_MINIMIZED = Path("vendors/openmm/examples/benchmarks/5dfr_minimized.pdb")
OPENMM_DHFR_SOLVATED = Path("vendors/openmm/examples/benchmarks/5dfr_solv-cube_equil.pdb")
AMBER20_JAC_PRMTOP = Path("results/inputs/Amber20_Benchmark_Suite/PME/Topologies/JAC.prmtop")
AMBER20_JAC_INPCRD = Path("results/inputs/Amber20_Benchmark_Suite/PME/Coordinates/JAC.inpcrd")


@dataclass(frozen=True)
class CaseSpec:
    case: str
    fixture: str
    input_paths: tuple[Path, ...]
    timing_metric: str = TIMING_METRIC
    solvent_model: str = ""
    electrostatics_model: str = ""
    force_field_family: str = ""
    openmm_test_name: str = ""
    dt_ps: float = 0.004
    friction_per_ps: float = 1.0
    cutoff_nm: float = 0.9


CASE_SPECS = {
    "dhfr-implicit": CaseSpec(
        case="dhfr-implicit",
        fixture="dhfr_implicit",
        input_paths=(OPENMM_DHFR_MINIMIZED,),
        solvent_model="implicit",
        electrostatics_model="gbsa_obc",
        force_field_family="amber99sb-obc",
        openmm_test_name="gbsa",
        friction_per_ps=91.0,
        cutoff_nm=2.0,
    ),
    "dhfr-explicit-pme": CaseSpec(
        case="dhfr-explicit-pme",
        fixture="dhfr_explicit_pme",
        input_paths=(OPENMM_DHFR_SOLVATED, AMBER20_JAC_PRMTOP, AMBER20_JAC_INPCRD),
        solvent_model="explicit",
        electrostatics_model="pme",
        force_field_family="amber20-jac",
        openmm_test_name="amber20-dhfr",
        friction_per_ps=1.0,
        cutoff_nm=0.9,
    ),
}


@dataclass(frozen=True)
class OpenMMApi:
    app: Any
    openmm: Any
    unit: Any


def main() -> None:
    args = _parse_args()
    payload = build_payload(args)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_format_human_payload(payload))


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Build a normalized runnable or blocked OpenMM DHFR reference payload."""

    _validate_args(args)
    spec = _case_spec(args)
    input_status = _input_status(spec, args.repo_root)
    atom_count = _input_atom_count(spec, args.repo_root)
    if input_status["missing_input_paths"]:
        return _normalize_payload(
            _blocked_payload(
                args,
                spec=spec,
                atom_count=atom_count,
                input_status=input_status,
                blocker="missing OpenMM input path(s): "
                + ", ".join(input_status["missing_input_paths"]),
            ),
            args,
            spec=spec,
        )
    try:
        result = run_reference(args, spec=spec, input_status=input_status, atom_count=atom_count)
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover - platform/package dependent.
        result = _blocked_payload(
            args,
            spec=spec,
            atom_count=atom_count,
            input_status=input_status,
            blocker=f"{type(exc).__name__}: {exc}",
        )
    return _normalize_payload(result, args, spec=spec)


def run_reference(
    args: argparse.Namespace,
    *,
    spec: CaseSpec,
    input_status: dict[str, Any],
    atom_count: int | None,
) -> dict[str, Any]:
    """Run the requested DHFR case through OpenMM."""

    api = _load_openmm()
    platform = api.openmm.Platform.getPlatformByName(args.platform)
    system, positions, setup = _build_system(api, spec, args.repo_root)
    integrator = api.openmm.LangevinMiddleIntegrator(
        args.temperature * api.unit.kelvin,
        spec.friction_per_ps / api.unit.picosecond,
        spec.dt_ps * api.unit.picoseconds,
    )
    integrator.setConstraintTolerance(args.constraint_tolerance)
    integrator.setRandomNumberSeed(args.seed)

    properties: dict[str, str] = {}
    if args.precision is not None and "Precision" in list(platform.getPropertyNames()):
        properties["Precision"] = args.precision
    context = api.openmm.Context(system, integrator, platform, properties)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(args.temperature * api.unit.kelvin, args.seed)

    if args.warmup_steps:
        integrator.step(args.warmup_steps)
        context.getState(getEnergy=True)

    start = time.perf_counter()
    integrator.step(args.steps)
    context.getState(getEnergy=True)
    wall_s = time.perf_counter() - start
    state = context.getState(getEnergy=True)
    simulated_ns = args.steps * spec.dt_ps / 1000.0
    ns_per_day = simulated_ns / wall_s * 86400.0 if wall_s > 0.0 else 0.0
    potential_energy = float(
        state.getPotentialEnergy().value_in_unit(api.unit.kilojoule_per_mole)
    )
    kinetic_energy = float(state.getKineticEnergy().value_in_unit(api.unit.kilojoule_per_mole))
    finite = math.isfinite(ns_per_day) and math.isfinite(potential_energy) and math.isfinite(
        kinetic_energy
    )
    return {
        "status": "ok",
        "case": spec.case,
        "fixture": spec.fixture,
        "system": spec.case,
        "engine": ENGINE,
        "openmm_test_name": spec.openmm_test_name,
        "platform": context.getPlatform().getName(),
        "requested_platform": args.platform,
        "platform_properties": _platform_properties(context.getPlatform(), context),
        "available_platforms": _available_platforms(api),
        "openmm_version": api.openmm.version.version,
        "atom_count": atom_count if atom_count is not None else system.getNumParticles(),
        "particles": system.getNumParticles(),
        "steps": args.steps,
        "step_count": args.steps,
        "warmup_steps": args.warmup_steps,
        "dt_ps": spec.dt_ps,
        "simulated_ns": simulated_ns,
        "wall_s": wall_s,
        "ns_per_day": ns_per_day,
        "timing_value": ns_per_day,
        "timing_unit": "ns/day",
        "temperature_K": args.temperature,
        "friction_per_ps": spec.friction_per_ps,
        "cutoff_nm": spec.cutoff_nm,
        "integrator": "LangevinMiddleIntegrator",
        "finite": finite,
        "potential_energy_kj_mol": potential_energy,
        "kinetic_energy_kj_mol": kinetic_energy,
        "solvent_model": spec.solvent_model,
        "electrostatics_model": spec.electrostatics_model,
        "force_field_family": spec.force_field_family,
        "input_status": input_status,
        "raw_input_paths": [str(path) for path in spec.input_paths],
        "raw_input_metadata": _raw_input_metadata(spec, args.repo_root),
        "setup": setup,
        "blocker": None,
    }


def _build_system(
    api: OpenMMApi,
    spec: CaseSpec,
    repo_root: Path,
) -> tuple[Any, Any, dict[str, Any]]:
    app = api.app
    unit = api.unit
    if spec.case == "dhfr-implicit":
        pdb = app.PDBFile(str(repo_root / OPENMM_DHFR_MINIMIZED))
        force_field = app.ForceField("amber99sb.xml", "amber99_obc.xml")
        system = force_field.createSystem(
            pdb.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=spec.cutoff_nm * unit.nanometer,
            constraints=app.HBonds,
            hydrogenMass=1.5 * unit.amu,
        )
        return system, pdb.positions, {
            "source": "openmm examples benchmark gbsa case",
            "nonbonded_method": "CutoffNonPeriodic",
            "constraints": "HBonds",
            "hydrogen_mass_amu": 1.5,
        }

    prmtop = app.AmberPrmtopFile(str(repo_root / AMBER20_JAC_PRMTOP))
    inpcrd = app.AmberInpcrdFile(str(repo_root / AMBER20_JAC_INPCRD))
    system = prmtop.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=spec.cutoff_nm * unit.nanometer,
        constraints=app.HBonds,
    )
    if inpcrd.boxVectors is not None:
        system.setDefaultPeriodicBoxVectors(*inpcrd.boxVectors)
    return system, inpcrd.positions, {
        "source": "Amber20 Benchmark Suite JAC",
        "nonbonded_method": "PME",
        "constraints": "HBonds",
        "hydrogen_mass_amu": 1.0,
        "box_vectors_present": inpcrd.boxVectors is not None,
    }


def _load_openmm() -> OpenMMApi:
    try:
        import openmm
        import openmm.app as app
        import openmm.unit as unit
    except Exception as exc:  # pragma: no cover - optional reference package.
        msg = f"OpenMM import unavailable: {exc}"
        raise RuntimeError(msg) from exc
    return OpenMMApi(app=app, openmm=openmm, unit=unit)


def _blocked_payload(
    args: argparse.Namespace,
    *,
    spec: CaseSpec,
    atom_count: int | None,
    input_status: dict[str, Any],
    blocker: str,
) -> dict[str, Any]:
    try:
        api = _load_openmm()
        available_platforms = _available_platforms(api)
        openmm_version = api.openmm.version.version
    except Exception as exc:  # pragma: no cover - optional reference package.
        available_platforms = []
        openmm_version = f"<unavailable: {exc}>"
    return {
        "status": "blocked",
        "case": spec.case,
        "fixture": spec.fixture,
        "system": spec.case,
        "engine": ENGINE,
        "openmm_test_name": spec.openmm_test_name,
        "platform": args.platform,
        "requested_platform": args.platform,
        "platform_properties": {},
        "available_platforms": available_platforms,
        "openmm_version": openmm_version,
        "atom_count": atom_count,
        "particles": atom_count,
        "steps": args.steps,
        "step_count": args.steps,
        "warmup_steps": args.warmup_steps,
        "dt_ps": spec.dt_ps,
        "simulated_ns": args.steps * spec.dt_ps / 1000.0,
        "wall_s": None,
        "ns_per_day": None,
        "timing_unit": "ns/day",
        "temperature_K": args.temperature,
        "friction_per_ps": spec.friction_per_ps,
        "cutoff_nm": spec.cutoff_nm,
        "integrator": "LangevinMiddleIntegrator",
        "finite": False,
        "potential_energy_kj_mol": None,
        "kinetic_energy_kj_mol": None,
        "solvent_model": spec.solvent_model,
        "electrostatics_model": spec.electrostatics_model,
        "force_field_family": spec.force_field_family,
        "input_status": input_status,
        "raw_input_paths": [str(path) for path in spec.input_paths],
        "raw_input_metadata": _raw_input_metadata(spec, args.repo_root),
        "blocker": blocker,
    }


def _normalize_payload(
    payload: dict[str, Any],
    args: argparse.Namespace,
    *,
    spec: CaseSpec,
) -> dict[str, Any]:
    payload = dict(payload)
    payload["timing_value"] = payload.get(spec.timing_metric)
    return normalize_benchmark_payload(
        payload,
        benchmark_name=BENCHMARK_NAME,
        fixture=spec.fixture,
        timing_metric=spec.timing_metric,
        hardware=get_hardware_info(),
        runtime={
            "python_version": platform_module.python_version(),
            "reference_engine_version": payload.get("openmm_version"),
            "requested_platform": args.platform,
            "available_platforms": payload.get("available_platforms", []),
            "case": spec.case,
        },
        engine=ENGINE,
        atom_count=payload.get("atom_count"),
        step_count=args.steps,
        evaluation_count=args.steps,
        finite=bool(payload.get("finite")),
        status=payload.get("status"),
        blocker=payload.get("blocker"),
        command=_command_for_args(args),
        raw_output_path=f"results/same-workload-openmm-comparison/openmm-{spec.case}.json",
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        msg = "steps must be positive"
        raise ValueError(msg)
    if args.warmup_steps < 0:
        msg = "warmup_steps must be non-negative"
        raise ValueError(msg)
    if args.temperature <= 0.0:
        msg = "temperature must be positive"
        raise ValueError(msg)
    if args.constraint_tolerance <= 0.0:
        msg = "constraint_tolerance must be positive"
        raise ValueError(msg)


def _case_spec(args: argparse.Namespace) -> CaseSpec:
    return CASE_SPECS[args.case]


def _input_status(spec: CaseSpec, repo_root: Path) -> dict[str, Any]:
    existing: list[str] = []
    missing: list[str] = []
    for path in spec.input_paths:
        if (repo_root / path).exists():
            existing.append(str(path))
        else:
            missing.append(str(path))
    return {
        "existing_input_paths": existing,
        "missing_input_paths": missing,
        "all_inputs_present": not missing,
        "downloads_attempted": False,
    }


def _raw_input_metadata(spec: CaseSpec, repo_root: Path) -> list[dict[str, Any]]:
    metadata = []
    for path in spec.input_paths:
        target = repo_root / path
        metadata.append(
            {
                "path": str(path),
                "exists": target.exists(),
                "size_bytes": target.stat().st_size if target.exists() else None,
            }
        )
    return metadata


def _input_atom_count(spec: CaseSpec, repo_root: Path) -> int | None:
    if spec.case == "dhfr-explicit-pme":
        count = _amber_prmtop_atom_count(repo_root / AMBER20_JAC_PRMTOP)
        if count is not None:
            return count
    return _pdb_atom_count(repo_root / spec.input_paths[0])


def _pdb_atom_count(path: Path) -> int | None:
    if not path.exists():
        return None
    count = 0
    with path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")):
                count += 1
    return count


def _amber_prmtop_atom_count(path: Path) -> int | None:
    if not path.exists():
        return None
    in_pointers = False
    values: list[int] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith("%FLAG "):
                if in_pointers:
                    break
                in_pointers = line.strip() == "%FLAG POINTERS"
                continue
            if not in_pointers or line.startswith("%FORMAT"):
                continue
            values.extend(int(item) for item in line.split())
            if values:
                return values[0]
    return None


def _platform_properties(platform: Any, context: Any) -> dict[str, str]:
    properties = {}
    for name in platform.getPropertyNames():
        try:
            properties[name] = platform.getPropertyValue(context, name)
        except Exception as exc:  # pragma: no cover - platform-specific diagnostics.
            properties[name] = f"<unavailable: {exc}>"
    return properties


def _available_platforms(api: OpenMMApi) -> list[str]:
    return [
        api.openmm.Platform.getPlatform(index).getName()
        for index in range(api.openmm.Platform.getNumPlatforms())
    ]


def _format_human_payload(payload: dict[str, Any]) -> str:
    if payload["status"] == "blocked":
        return (
            f"OpenMM DHFR {payload['case']} on {payload['requested_platform']}: blocked; "
            f"blocker={payload['blocker']}"
        )
    return (
        f"OpenMM DHFR {payload['case']} on {payload['platform']}: "
        f"{payload['ns_per_day']:.3f} ns/day, atom_count={payload['atom_count']}"
    )


def _command_for_args(args: argparse.Namespace) -> str:
    return f"{COMMAND} --case {args.case} --platform {args.platform} --steps {args.steps}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASE_SPECS), required=True)
    parser.add_argument("--platform", default="Reference")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--constraint-tolerance", type=float, default=1e-5)
    parser.add_argument("--precision", choices=["single", "mixed", "double"], default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
