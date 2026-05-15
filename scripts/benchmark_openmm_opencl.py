"""Benchmark a small synthetic OpenMM/OpenCL MD run.

This is a reference-engine showcase, not a package runtime path. It keeps
OpenMM outside `mlx_atomistic` while making the OpenCL baseline easy to repeat.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openmm
from openmm import LangevinMiddleIntegrator, NonbondedForce, Platform, System, Vec3, unit


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


def main() -> None:
    args = _parse_args()
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
    if args.csv is not None:
        _write_csv(args.csv, payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "OpenMM {platform} {particles} atoms: {steps_per_s:.1f} steps/s, "
            "{ns_per_day:.3f} ns/day".format(**payload)
        )


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

    if particles <= 0:
        msg = "particles must be positive"
        raise ValueError(msg)
    if steps <= 0:
        msg = "steps must be positive"
        raise ValueError(msg)
    if warmup_steps < 0:
        msg = "warmup_steps must be non-negative"
        raise ValueError(msg)
    if dt_ps <= 0.0:
        msg = "dt_ps must be positive"
        raise ValueError(msg)
    if cutoff_nm <= 0.0:
        msg = "cutoff_nm must be positive"
        raise ValueError(msg)

    positions, box_length_nm = _lattice_positions(particles, spacing_nm)
    if cutoff_nm >= 0.5 * box_length_nm:
        msg = (
            "cutoff_nm must be less than half the box length: "
            f"cutoff_nm={cutoff_nm:g}, box_length_nm={box_length_nm:g}"
        )
        raise ValueError(msg)

    system = _build_lj_system(
        particles,
        box_length_nm=box_length_nm,
        cutoff_nm=cutoff_nm,
    )
    integrator = LangevinMiddleIntegrator(
        temperature_K * unit.kelvin,
        friction_per_ps / unit.picosecond,
        dt_ps * unit.picoseconds,
    )
    integrator.setRandomNumberSeed(seed)
    platform = Platform.getPlatformByName(platform_name)
    properties = {}
    if precision is not None and "Precision" in list(platform.getPropertyNames()):
        properties["Precision"] = precision

    context = openmm.Context(system, integrator, platform, properties)
    context.setPositions([Vec3(*row) for row in positions] * unit.nanometer)
    context.setVelocitiesToTemperature(temperature_K * unit.kelvin, seed)

    if warmup_steps:
        integrator.step(warmup_steps)

    start = time.perf_counter()
    integrator.step(steps)
    wall_s = time.perf_counter() - start

    state = context.getState(getEnergy=True, getPositions=True, getVelocities=True)
    final_positions = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    final_velocities = np.asarray(
        state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    )
    simulated_ns = steps * dt_ps / 1000.0
    return OpenMMBenchmarkResult(
        engine="openmm-reference",
        platform=context.getPlatform().getName(),
        platform_properties=_platform_properties(platform, context),
        available_platforms=_available_platforms(),
        openmm_version=openmm.version.version,
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
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        ),
        kinetic_energy_kj_mol=float(state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)),
    )


def _build_lj_system(
    particles: int,
    *,
    box_length_nm: float,
    cutoff_nm: float,
) -> System:
    system = System()
    nonbonded = NonbondedForce()
    nonbonded.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(cutoff_nm * unit.nanometer)
    nonbonded.setUseDispersionCorrection(False)
    for _ in range(particles):
        system.addParticle(39.948 * unit.dalton)
        nonbonded.addParticle(
            0.0 * unit.elementary_charge,
            0.34 * unit.nanometer,
            0.997 * unit.kilojoule_per_mole,
        )
    system.addForce(nonbonded)
    system.setDefaultPeriodicBoxVectors(
        Vec3(box_length_nm, 0, 0) * unit.nanometer,
        Vec3(0, box_length_nm, 0) * unit.nanometer,
        Vec3(0, 0, box_length_nm) * unit.nanometer,
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


def _platform_properties(platform: Platform, context: openmm.Context) -> dict[str, str]:
    properties = {}
    for name in platform.getPropertyNames():
        try:
            properties[name] = platform.getPropertyValue(context, name)
        except Exception as exc:  # pragma: no cover - platform-specific diagnostics.
            properties[name] = f"<unavailable: {exc}>"
    return properties


def _available_platforms() -> list[str]:
    return [Platform.getPlatform(index).getName() for index in range(Platform.getNumPlatforms())]


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
