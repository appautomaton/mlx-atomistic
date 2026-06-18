"""Benchmark a small synthetic OpenMM/OpenCL MD run.

This is a reference-engine showcase, not a package runtime path. It keeps
OpenMM outside `mlx_atomistic` while making the OpenCL baseline easy to repeat.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform as platform_module
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.benchmarks import normalize_benchmark_payload

BENCHMARK_NAME = "openmm_opencl_reference"
ENGINE = "openmm-reference"
FIXTURE = "synthetic_lj_periodic"
TIMING_METRIC = "steps_per_s"
COMMAND = "uv run python scripts/benchmark_openmm_opencl.py"
DEFAULT_CASE = "synthetic-lj-periodic"


@dataclass(frozen=True)
class CaseSpec:
    fixture: str
    timing_metric: str
    blocker: str | None = None


CASE_SPECS = {
    DEFAULT_CASE: CaseSpec(fixture=FIXTURE, timing_metric=TIMING_METRIC),
    "gbsa-obc-small": CaseSpec(
        fixture="gbsa_obc_small",
        timing_metric="ms_per_eval",
    ),
    "tip4p-ew-water": CaseSpec(
        fixture="tip4p_ew_water",
        timing_metric="ms_per_eval",
    ),
}

TIP4P_EW_OH_DISTANCE_ANGSTROM = 0.9572
TIP4P_EW_HOH_ANGLE_DEGREES = 104.52
TIP4P_EW_OM_DISTANCE_ANGSTROM = 0.1250


@dataclass(frozen=True)
class OpenMMApi:
    openmm: Any
    GBSAOBCForce: Any
    LangevinMiddleIntegrator: Any
    NonbondedForce: Any
    Platform: Any
    System: Any
    Vec3: Any
    VerletIntegrator: Any
    unit: Any


@dataclass(frozen=True)
class OpenMMBenchmarkResult:
    """One OpenMM synthetic MD benchmark result."""

    engine: str
    platform: str
    platform_properties: dict[str, str]
    available_platforms: list[str]
    openmm_version: str
    particles: int
    steps: int
    warmup_steps: int
    dt_ps: float
    simulated_ns: float
    wall_s: float
    steps_per_s: float
    ns_per_day: float
    integrator: str
    temperature_K: float
    friction_per_ps: float
    cutoff_nm: float
    box_length_nm: float
    finite: bool
    potential_energy_kj_mol: float
    kinetic_energy_kj_mol: float
    status: str = "ok"
    benchmark_name: str = BENCHMARK_NAME
    fixture: str = FIXTURE
    hardware: dict[str, str] | None = None
    atom_count: int = 0
    step_count: int = 0
    openmm_platform: str = ""
    openmm_steps_per_s: float = 0.0
    openmm_ns_per_day: float = 0.0
    blocker: str | None = None


def main() -> None:
    args = _parse_args()
    payload = build_payload(args)
    if args.csv is not None:
        _write_csv(args.csv, payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_format_human_payload(payload))


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Build a normalized OpenMM benchmark payload or blocked payload."""

    try:
        _validate_args(args)
        if args.case == "gbsa-obc-small":
            payload = run_gbsa_obc_reference(
                particles=args.particles,
                evaluations=args.steps,
                warmup_evaluations=args.warmup_steps,
                platform_name=args.platform,
            )
            return _normalize_reference_payload(payload, args)
        if args.case == "tip4p-ew-water":
            payload = run_tip4p_ew_virtual_site_reference(
                particles=args.particles,
                evaluations=args.steps,
                warmup_evaluations=args.warmup_steps,
                platform_name=args.platform,
            )
            return _normalize_reference_payload(payload, args)
        if args.case != DEFAULT_CASE:
            payload = _unsupported_controlled_case_payload(args)
            return _normalize_reference_payload(payload, args)
        result = run_benchmark(
            particles=args.particles,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            dt_ps=args.dt_ps,
            platform_name=args.platform,
            precision=args.precision,
            temperature_K=args.temperature,
            friction_per_ps=args.friction,
            cutoff_nm=args.cutoff_nm,
            spacing_nm=args.spacing_nm,
            seed=args.seed,
        )
        payload = asdict(result)
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover - platform-dependent reference surface.
        payload = _blocked_payload(args, blocker=f"{type(exc).__name__}: {exc}")
    return _normalize_reference_payload(payload, args)


def run_benchmark(
    *,
    particles: int,
    steps: int,
    warmup_steps: int,
    dt_ps: float,
    platform_name: str,
    precision: str | None,
    temperature_K: float,
    friction_per_ps: float,
    cutoff_nm: float,
    spacing_nm: float,
    seed: int,
) -> OpenMMBenchmarkResult:
    """Run one synthetic LJ benchmark through OpenMM."""

    api = _load_openmm()
    platform = api.Platform.getPlatformByName(platform_name)

    positions, box_length_nm = _lattice_positions(particles, spacing_nm)
    if cutoff_nm >= 0.5 * box_length_nm:
        msg = (
            "cutoff_nm must be less than half the box length: "
            f"cutoff_nm={cutoff_nm:g}, box_length_nm={box_length_nm:g}"
        )
        raise ValueError(msg)

    system = _build_lj_system(
        api,
        particles,
        box_length_nm=box_length_nm,
        cutoff_nm=cutoff_nm,
    )
    integrator = api.LangevinMiddleIntegrator(
        temperature_K * api.unit.kelvin,
        friction_per_ps / api.unit.picosecond,
        dt_ps * api.unit.picoseconds,
    )
    integrator.setRandomNumberSeed(seed)
    properties = {}
    if precision is not None and "Precision" in list(platform.getPropertyNames()):
        properties["Precision"] = precision

    context = api.openmm.Context(system, integrator, platform, properties)
    context.setPositions([api.Vec3(*row) for row in positions] * api.unit.nanometer)
    context.setVelocitiesToTemperature(temperature_K * api.unit.kelvin, seed)

    if warmup_steps:
        integrator.step(warmup_steps)

    start = time.perf_counter()
    integrator.step(steps)
    # OpenCL enqueues integration kernels asynchronously, so step() can return
    # before the GPU has done the work. Force the queue to drain inside the timed
    # region (getEnergy requires all forces computed) so wall_s reflects real
    # compute rather than just kernel enqueue.
    context.getState(getEnergy=True)
    wall_s = time.perf_counter() - start

    state = context.getState(getEnergy=True, getPositions=True, getVelocities=True)
    final_positions = np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(api.unit.nanometer)
    )
    final_velocities = np.asarray(
        state.getVelocities(asNumpy=True).value_in_unit(api.unit.nanometer / api.unit.picosecond)
    )
    simulated_ns = steps * dt_ps / 1000.0
    return OpenMMBenchmarkResult(
        engine="openmm-reference",
        platform=context.getPlatform().getName(),
        platform_properties=_platform_properties(platform, context),
        available_platforms=_available_platforms(api),
        openmm_version=api.openmm.version.version,
        particles=particles,
        steps=steps,
        warmup_steps=warmup_steps,
        dt_ps=dt_ps,
        simulated_ns=simulated_ns,
        wall_s=wall_s,
        steps_per_s=steps / wall_s if wall_s > 0.0 else 0.0,
        ns_per_day=simulated_ns / wall_s * 86400.0 if wall_s > 0.0 else 0.0,
        integrator="LangevinMiddleIntegrator",
        temperature_K=temperature_K,
        friction_per_ps=friction_per_ps,
        cutoff_nm=cutoff_nm,
        box_length_nm=box_length_nm,
        finite=bool(np.all(np.isfinite(final_positions)) and np.all(np.isfinite(final_velocities))),
        potential_energy_kj_mol=float(
            state.getPotentialEnergy().value_in_unit(api.unit.kilojoule_per_mole)
        ),
        kinetic_energy_kj_mol=float(
            state.getKineticEnergy().value_in_unit(api.unit.kilojoule_per_mole)
        ),
        atom_count=particles,
        step_count=steps,
        hardware=_hardware_info(),
        openmm_platform=context.getPlatform().getName(),
        openmm_steps_per_s=steps / wall_s if wall_s > 0.0 else 0.0,
        openmm_ns_per_day=simulated_ns / wall_s * 86400.0 if wall_s > 0.0 else 0.0,
    )


def run_gbsa_obc_reference(
    *,
    particles: int,
    evaluations: int,
    warmup_evaluations: int,
    platform_name: str,
) -> dict[str, Any]:
    """Evaluate the synthetic GBSA/OBC force surface through OpenMM."""

    api = _load_openmm()
    platform = api.Platform.getPlatformByName(platform_name)
    positions_a, charges, radii_a, scales = _gbsa_obc_small_arrays(particles)

    system = api.System()
    for _ in range(particles):
        system.addParticle(12.0 * api.unit.dalton)

    surface_area_energy = 2.25936
    gbsa = api.GBSAOBCForce()
    gbsa.setNonbondedMethod(api.GBSAOBCForce.NoCutoff)
    gbsa.setSolventDielectric(78.5)
    gbsa.setSoluteDielectric(1.0)
    gbsa.setSurfaceAreaEnergy(
        surface_area_energy * api.unit.kilojoule_per_mole / api.unit.nanometer**2
    )
    for charge, radius_a, scale in zip(charges, radii_a, scales, strict=True):
        gbsa.addParticle(float(charge), float(radius_a / 10.0), float(scale))
    system.addForce(gbsa)

    integrator = api.VerletIntegrator(0.001 * api.unit.picoseconds)
    context = api.openmm.Context(system, integrator, platform)
    context.setPositions((positions_a / 10.0) * api.unit.nanometer)

    for _ in range(warmup_evaluations):
        context.getState(getEnergy=True, getForces=True)

    start = time.perf_counter()
    state = None
    for _ in range(evaluations):
        state = context.getState(getEnergy=True, getForces=True)
    wall_s = time.perf_counter() - start
    if state is None:  # Defensive guard; argument validation keeps evaluations positive.
        msg = "evaluations must be positive"
        raise ValueError(msg)

    force_unit = api.unit.kilojoule_per_mole / api.unit.nanometer
    forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(force_unit))
    potential_energy = float(
        state.getPotentialEnergy().value_in_unit(api.unit.kilojoule_per_mole)
    )
    finite = bool(np.isfinite(potential_energy) and np.all(np.isfinite(forces)))
    ms_per_eval = wall_s * 1000.0 / evaluations if wall_s > 0.0 else 0.0

    return {
        "status": "ok",
        "benchmark_name": BENCHMARK_NAME,
        "case": "gbsa-obc-small",
        "fixture": "gbsa_obc_small",
        "hardware": _hardware_info(),
        "engine": ENGINE,
        "platform": context.getPlatform().getName(),
        "platform_properties": _platform_properties(platform, context),
        "available_platforms": _available_platforms(api),
        "openmm_version": api.openmm.version.version,
        "particles": particles,
        "atom_count": particles,
        "steps": evaluations,
        "step_count": evaluations,
        "evaluations": evaluations,
        "evaluation_count": evaluations,
        "warmup_steps": warmup_evaluations,
        "wall_s": wall_s,
        "ms_per_eval": ms_per_eval,
        "openmm_platform": context.getPlatform().getName(),
        "integrator": "VerletIntegrator",
        "finite": finite,
        "potential_energy_kj_mol": potential_energy,
        "force_norm_kj_mol_nm": float(np.linalg.norm(forces)),
        "blocker": None,
        "obc_force_setup": {
            "force": "GBSAOBCForce",
            "nonbonded_method": "NoCutoff",
            "solvent_dielectric": 78.5,
            "solute_dielectric": 1.0,
            "surface_area_energy_kj_mol_nm2": surface_area_energy,
            "charge_e": charges.tolist(),
            "radius_angstrom": radii_a.tolist(),
            "radius_nm": (radii_a / 10.0).tolist(),
            "scale": scales.tolist(),
            "positions_angstrom": positions_a.tolist(),
            "positions_nm": (positions_a / 10.0).tolist(),
        },
    }


def run_tip4p_ew_virtual_site_reference(
    *,
    particles: int,
    evaluations: int,
    warmup_evaluations: int,
    platform_name: str,
) -> dict[str, Any]:
    """Time OpenMM TIP4P-Ew M-site reconstruction through computeVirtualSites()."""

    api = _load_openmm()
    platform = api.Platform.getPlatformByName(platform_name)
    water_count = particles // 4
    positions_angstrom = _tip4p_ew_water_positions_angstrom(water_count)
    weights = _tip4p_ew_m_site_weights()

    system = api.System()
    for _ in range(water_count):
        system.addParticle(15.99943 * api.unit.dalton)
        system.addParticle(1.007947 * api.unit.dalton)
        system.addParticle(1.007947 * api.unit.dalton)
        system.addParticle(0.0 * api.unit.dalton)

    for water_index in range(water_count):
        offset = 4 * water_index
        system.setVirtualSite(
            offset + 3,
            api.openmm.ThreeParticleAverageSite(
                offset,
                offset + 1,
                offset + 2,
                weights[0],
                weights[1],
                weights[2],
            ),
        )

    integrator = api.VerletIntegrator(0.001 * api.unit.picoseconds)
    context = api.openmm.Context(system, integrator, platform)
    context.setPositions((positions_angstrom * 0.1) * api.unit.nanometer)

    for _ in range(warmup_evaluations):
        context.computeVirtualSites()

    start = time.perf_counter()
    for _ in range(evaluations):
        context.computeVirtualSites()
    wall_s = time.perf_counter() - start

    positions_nm = np.asarray(
        context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(
            api.unit.nanometer
        )
    )
    virtual_site_indices = list(range(3, particles, 4))
    virtual_site_positions_nm = positions_nm[virtual_site_indices]
    finite = bool(np.all(np.isfinite(positions_nm)))
    ms_per_eval = wall_s * 1000.0 / evaluations if wall_s > 0.0 else 0.0

    return {
        "status": "ok",
        "benchmark_name": BENCHMARK_NAME,
        "case": "tip4p-ew-water",
        "fixture": "tip4p_ew_water",
        "hardware": _hardware_info(),
        "engine": ENGINE,
        "platform": context.getPlatform().getName(),
        "platform_properties": _platform_properties(platform, context),
        "available_platforms": _available_platforms(api),
        "openmm_version": api.openmm.version.version,
        "particles": particles,
        "atom_count": particles,
        "steps": evaluations,
        "step_count": evaluations,
        "evaluations": evaluations,
        "evaluation_count": evaluations,
        "warmup_steps": warmup_evaluations,
        "wall_s": wall_s,
        "ms_per_eval": ms_per_eval,
        "openmm_platform": context.getPlatform().getName(),
        "integrator": "VerletIntegrator",
        "finite": finite,
        "blocker": None,
        "operation_semantics": "virtual_site_reconstruction",
        "openmm_operation": "Context.computeVirtualSites",
        "water_model": "TIP4P-Ew",
        "virtual_site_type": "ThreeParticleAverageSite",
        "virtual_site_count": water_count,
        "virtual_site_indices": virtual_site_indices,
        "virtual_site_positions_nm": virtual_site_positions_nm.tolist(),
        "virtual_site_position_norm_nm": float(np.linalg.norm(virtual_site_positions_nm)),
        "tip4p_ew_geometry": {
            "oh_distance_angstrom": TIP4P_EW_OH_DISTANCE_ANGSTROM,
            "hoh_angle_degrees": TIP4P_EW_HOH_ANGLE_DEGREES,
            "om_distance_angstrom": TIP4P_EW_OM_DISTANCE_ANGSTROM,
            "m_site_weights": {
                "oxygen": weights[0],
                "hydrogen1": weights[1],
                "hydrogen2": weights[2],
            },
            "input_positions_angstrom": positions_angstrom.tolist(),
        },
    }


def _gbsa_obc_small_arrays(particles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.stack(
        [
            np.linspace(0.0, 1.5 * (particles - 1), particles),
            np.sin(np.arange(particles, dtype=np.float32)) * 0.2,
            np.cos(np.arange(particles, dtype=np.float32)) * 0.2,
        ],
        axis=1,
    ).astype(np.float32)
    charges = np.where(np.arange(particles) % 2 == 0, 0.4, -0.35).astype(np.float32)
    radii = np.linspace(1.45, 1.75, particles, dtype=np.float32)
    scales = np.linspace(0.72, 0.85, particles, dtype=np.float32)
    return positions, charges, radii, scales


def _tip4p_ew_m_site_weights() -> tuple[float, float, float]:
    half_angle = np.deg2rad(TIP4P_EW_HOH_ANGLE_DEGREES) / 2.0
    hydrogen_weight = TIP4P_EW_OM_DISTANCE_ANGSTROM / (
        2.0 * TIP4P_EW_OH_DISTANCE_ANGSTROM * np.cos(half_angle)
    )
    oxygen_weight = 1.0 - 2.0 * hydrogen_weight
    return float(oxygen_weight), float(hydrogen_weight), float(hydrogen_weight)


def _tip4p_ew_single_water_positions_angstrom() -> np.ndarray:
    half_angle = np.deg2rad(TIP4P_EW_HOH_ANGLE_DEGREES) / 2.0
    oxygen = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    h1 = np.asarray(
        [
            TIP4P_EW_OH_DISTANCE_ANGSTROM * np.cos(half_angle),
            TIP4P_EW_OH_DISTANCE_ANGSTROM * np.sin(half_angle),
            0.0,
        ],
        dtype=np.float64,
    )
    h2 = np.asarray([h1[0], -h1[1], 0.0], dtype=np.float64)
    weights = _tip4p_ew_m_site_weights()
    m_site = weights[0] * oxygen + weights[1] * h1 + weights[2] * h2
    return np.stack([oxygen, h1, h2, m_site])


def _tip4p_ew_water_positions_angstrom(water_count: int) -> np.ndarray:
    single_water = _tip4p_ew_single_water_positions_angstrom()
    waters = []
    for water_index in range(water_count):
        offset = np.asarray([3.0 * water_index, 0.0, 0.0], dtype=np.float64)
        waters.append(single_water + offset)
    return np.concatenate(waters, axis=0)


def _load_openmm() -> OpenMMApi:
    try:
        import openmm
        from openmm import (
            GBSAOBCForce,
            LangevinMiddleIntegrator,
            NonbondedForce,
            Platform,
            System,
            Vec3,
            VerletIntegrator,
            unit,
        )
    except Exception as exc:  # pragma: no cover - depends on optional reference package.
        msg = f"OpenMM import unavailable: {exc}"
        raise RuntimeError(msg) from exc
    return OpenMMApi(
        openmm=openmm,
        GBSAOBCForce=GBSAOBCForce,
        LangevinMiddleIntegrator=LangevinMiddleIntegrator,
        NonbondedForce=NonbondedForce,
        Platform=Platform,
        System=System,
        Vec3=Vec3,
        VerletIntegrator=VerletIntegrator,
        unit=unit,
    )


def _build_lj_system(
    api: OpenMMApi,
    particles: int,
    *,
    box_length_nm: float,
    cutoff_nm: float,
) -> Any:
    system = api.System()
    nonbonded = api.NonbondedForce()
    nonbonded.setNonbondedMethod(api.NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(cutoff_nm * api.unit.nanometer)
    nonbonded.setUseDispersionCorrection(False)
    for _ in range(particles):
        system.addParticle(39.948 * api.unit.dalton)
        nonbonded.addParticle(
            0.0 * api.unit.elementary_charge,
            0.34 * api.unit.nanometer,
            0.997 * api.unit.kilojoule_per_mole,
        )
    system.addForce(nonbonded)
    system.setDefaultPeriodicBoxVectors(
        api.Vec3(box_length_nm, 0, 0) * api.unit.nanometer,
        api.Vec3(0, box_length_nm, 0) * api.unit.nanometer,
        api.Vec3(0, 0, box_length_nm) * api.unit.nanometer,
    )
    return system


def _lattice_positions(particles: int, spacing_nm: float) -> tuple[np.ndarray, float]:
    if spacing_nm <= 0.0:
        msg = "spacing_nm must be positive"
        raise ValueError(msg)
    side = int(np.ceil(particles ** (1.0 / 3.0)))
    box_length_nm = side * spacing_nm
    coords = []
    for z in range(side):
        for y in range(side):
            for x in range(side):
                coords.append(
                    (
                        (x + 0.5) * spacing_nm,
                        (y + 0.5) * spacing_nm,
                        (z + 0.5) * spacing_nm,
                    )
                )
                if len(coords) == particles:
                    return np.asarray(coords, dtype=np.float64), box_length_nm
    raise RuntimeError("failed to generate lattice positions")


def _platform_properties(platform: Any, context: Any) -> dict[str, str]:
    properties = {}
    for name in platform.getPropertyNames():
        try:
            properties[name] = platform.getPropertyValue(context, name)
        except Exception as exc:  # pragma: no cover - platform-specific diagnostics.
            properties[name] = f"<unavailable: {exc}>"
    return properties


def _available_platforms(api: OpenMMApi | None = None) -> list[str]:
    loaded = api if api is not None else _load_openmm()
    return [
        loaded.Platform.getPlatform(index).getName()
        for index in range(loaded.Platform.getNumPlatforms())
    ]


def _validate_args(args: argparse.Namespace) -> None:
    if args.particles <= 0:
        msg = "particles must be positive"
        raise ValueError(msg)
    if args.steps <= 0:
        msg = "steps must be positive"
        raise ValueError(msg)
    if args.warmup_steps < 0:
        msg = "warmup_steps must be non-negative"
        raise ValueError(msg)
    if args.dt_ps <= 0.0:
        msg = "dt_ps must be positive"
        raise ValueError(msg)
    if args.cutoff_nm <= 0.0:
        msg = "cutoff_nm must be positive"
        raise ValueError(msg)
    if args.spacing_nm <= 0.0:
        msg = "spacing_nm must be positive"
        raise ValueError(msg)
    if args.case == "tip4p-ew-water" and args.particles % 4 != 0:
        msg = "particles must be a multiple of 4 for tip4p-ew-water"
        raise ValueError(msg)


def _hardware_info() -> dict[str, str]:
    return {
        "system": platform_module.system(),
        "release": platform_module.release(),
        "machine": platform_module.machine(),
        "processor": platform_module.processor(),
        "python_version": platform_module.python_version(),
        "platform": platform_module.platform(),
    }


def _case_spec(args: argparse.Namespace) -> CaseSpec:
    return CASE_SPECS[args.case]


def _unsupported_controlled_case_payload(args: argparse.Namespace) -> dict[str, Any]:
    _load_openmm()
    return _blocked_payload(args, blocker=_case_spec(args).blocker or "unsupported OpenMM case")


def _blocked_payload(args: argparse.Namespace, *, blocker: str) -> dict[str, Any]:
    spec = _case_spec(args)
    try:
        api = _load_openmm()
        available_platforms = _available_platforms(api)
        openmm_version = api.openmm.version.version
    except Exception as exc:  # pragma: no cover - import/install diagnostics.
        available_platforms = []
        openmm_version = f"<unavailable: {exc}>"
    return {
        "status": "blocked",
        "benchmark_name": BENCHMARK_NAME,
        "case": args.case,
        "fixture": spec.fixture,
        "hardware": _hardware_info(),
        "engine": ENGINE,
        "platform": args.platform,
        "platform_properties": {},
        "available_platforms": available_platforms,
        "openmm_version": openmm_version,
        "particles": args.particles,
        "atom_count": args.particles,
        "steps": args.steps,
        "step_count": args.steps,
        "warmup_steps": args.warmup_steps,
        "dt_ps": args.dt_ps,
        "simulated_ns": args.steps * args.dt_ps / 1000.0,
        "wall_s": None,
        "steps_per_s": None,
        "ns_per_day": None,
        "openmm_platform": None,
        "openmm_steps_per_s": None,
        "openmm_ns_per_day": None,
        "integrator": "LangevinMiddleIntegrator",
        "temperature_K": args.temperature,
        "friction_per_ps": args.friction,
        "cutoff_nm": args.cutoff_nm,
        "box_length_nm": None,
        "finite": False,
        "potential_energy_kj_mol": None,
        "kinetic_energy_kj_mol": None,
        "blocker": blocker,
    }


def _normalize_reference_payload(
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    spec = _case_spec(args)
    payload = dict(payload)
    payload["timing_value"] = payload.get(spec.timing_metric)
    return normalize_benchmark_payload(
        payload,
        benchmark_name=BENCHMARK_NAME,
        fixture=spec.fixture,
        timing_metric=spec.timing_metric,
        hardware=payload.get("hardware") or _hardware_info(),
        runtime={
            "python_version": platform_module.python_version(),
            "reference_engine_version": payload.get("openmm_version"),
            "requested_platform": args.platform,
            "available_platforms": payload.get("available_platforms", []),
            "precision": args.precision,
            "case": args.case,
        },
        engine=ENGINE,
        atom_count=args.particles,
        step_count=args.steps,
        evaluation_count=args.steps,
        finite=bool(payload.get("finite")),
        status=payload.get("status"),
        blocker=payload.get("blocker"),
        command=_command_for_args(args),
        raw_output_path=None,
    )


def _format_human_payload(payload: dict[str, Any]) -> str:
    if payload.get("status") == "blocked":
        platforms = ",".join(str(item) for item in payload.get("available_platforms", []))
        return (
            f"OpenMM {payload['platform']} {payload['particles']} atoms: blocked; "
            f"blocker={payload.get('blocker')}; available_platforms={platforms}"
        )
    if payload.get("timing_metric") == "ms_per_eval":
        return (
            "OpenMM {platform} {particles} atoms: {ms_per_eval:.6f} ms/eval, "
            "finite={finite}"
        ).format(**payload)
    return (
        "OpenMM {platform} {particles} atoms: {steps_per_s:.1f} steps/s, "
        "{ns_per_day:.3f} ns/day"
    ).format(**payload)


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in payload.items()
    }
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _command_for_args(args: argparse.Namespace) -> str:
    return (
        f"{COMMAND} --case {args.case} --platform {args.platform} "
        f"--particles {args.particles} --steps {args.steps}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASE_SPECS), default=DEFAULT_CASE)
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--dt-ps", type=float, default=0.002)
    parser.add_argument("--platform", default="OpenCL")
    parser.add_argument("--precision", choices=["single", "mixed", "double"], default=None)
    parser.add_argument("--temperature", type=float, default=120.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--cutoff-nm", type=float, default=1.0)
    parser.add_argument("--spacing-nm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--csv", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
