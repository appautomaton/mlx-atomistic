"""Benchmark a synthetic LAMMPS OpenCL/GPU Lennard-Jones MD run.

This is an opt-in reference-engine path. It keeps LAMMPS outside the product
runtime while giving the benchmark harness a real OpenCL throughput command for
the same-workload LJ scaling ladder.

Geometry is generated with the product's own ``fcc_lattice`` so the MLX and
LAMMPS runs share identical particle counts, box, and initial positions at a
matched reduced density. GPU engagement is verified by inspecting the LAMMPS log
for the active ``lj/cut/gpu`` pair build; if it cannot be confirmed the run is
reported as a CPU-only ``diagnostic`` rather than claimed as OpenCL.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform as platform_module
import re
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.benchmarks import normalize_benchmark_payload
from mlx_atomistic.initialize import fcc_lattice

BENCHMARK_NAME = "lammps_opencl_reference"
ENGINE = "lammps-reference"
FIXTURE = "synthetic_lj_periodic"
TIMING_METRIC = "steps_per_s"
COMMAND = "uv run python scripts/benchmark_lammps_opencl.py"
COMPARISON_OUTPUT_ROOT = "results/same-workload-lj-scaling"
PAIR_STYLE = "lj/cut/gpu"
CUTOFF = 2.5
SKIN = 0.3


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
    """Build a normalized LAMMPS benchmark payload or blocked payload."""

    try:
        payload = run_benchmark(
            particles=args.particles,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            dt=args.dt,
            temperature=args.temperature,
            density=args.density,
            fixture=args.fixture,
            opencl_platform=args.opencl_platform,
            opencl_device=args.opencl_device,
            seed=args.seed,
        )
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover - depends on optional reference stack.
        payload = _blocked_payload(args, blocker=f"{type(exc).__name__}: {exc}")
    return _normalize_reference_payload(payload, args)


def run_benchmark(
    *,
    particles: int,
    steps: int,
    warmup_steps: int,
    dt: float,
    temperature: float,
    density: float,
    fixture: str,
    opencl_platform: str,
    opencl_device: str,
    seed: int,
) -> dict[str, Any]:
    """Run one synthetic LJ benchmark through LAMMPS GPU/OpenCL."""

    if fixture != FIXTURE:
        msg = f"unsupported fixture {fixture!r}; only {FIXTURE!r} is available"
        raise RuntimeError(msg)
    if particles <= 0:
        msg = "particles must be positive"
        raise ValueError(msg)
    if steps <= 0:
        msg = "steps must be positive"
        raise ValueError(msg)
    if warmup_steps < 0:
        msg = "warmup_steps must be non-negative"
        raise ValueError(msg)
    if dt <= 0.0:
        msg = "dt must be positive"
        raise ValueError(msg)
    if density <= 0.0:
        msg = "density must be positive"
        raise ValueError(msg)

    positions, box_length = _fcc_positions(particles, density)
    # LAMMPS needs the periodic box to comfortably exceed the ghost-atom cutoff.
    if box_length <= 2.0 * (CUTOFF + SKIN):
        msg = (
            f"system too dense for cutoff: box={box_length:.3g} <= 2*(cutoff+skin)="
            f"{2.0 * (CUTOFF + SKIN):.3g}; increase particles or decrease density"
        )
        raise RuntimeError(msg)

    lammps_class, lammps_version = _load_lammps()
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as handle:
        log_path = handle.name
    try:
        lmp = lammps_class(cmdargs=["-screen", "none", "-log", log_path, "-nocite"])
        try:
            _configure_box(lmp, box_length=box_length, opencl_platform=opencl_platform)
            created = int(
                lmp.create_atoms(
                    int(positions.shape[0]),
                    None,
                    [1] * int(positions.shape[0]),
                    positions.flatten().tolist(),
                )
            )
            _configure_interactions(lmp, dt=dt, temperature=temperature, seed=seed)
            if warmup_steps:
                lmp.command(f"run {warmup_steps} post no")
            start = time.perf_counter()
            lmp.command(f"run {steps} post no")
            wall_s = time.perf_counter() - start
            potential_energy = _extract_lammps_value(lmp, "pe")
            kinetic_energy = _extract_lammps_value(lmp, "ke")
            natoms = int(lmp.get_natoms())
        finally:
            with suppress(Exception):
                lmp.close()
        gpu = _inspect_gpu_log(log_path)
    finally:
        with suppress(OSError):
            Path(log_path).unlink()

    if created != particles or natoms != particles:
        msg = f"atom-count mismatch: requested {particles}, created {created}, present {natoms}"
        raise RuntimeError(msg)

    steps_per_s = steps / wall_s if wall_s > 0.0 else 0.0
    simulated_time = steps * dt
    gpu_active = gpu["gpu_pair_style_active"]
    status = "ok" if gpu_active else "diagnostic"
    blocker = (
        None
        if gpu_active
        else "GPU pair style lj/cut/gpu not confirmed active in LAMMPS log; possible CPU fallback"
    )
    return {
        "status": status,
        "engine": ENGINE,
        "benchmark_name": BENCHMARK_NAME,
        "fixture": FIXTURE,
        "hardware": _hardware_info(),
        "runtime": {
            "python_version": platform_module.python_version(),
            "reference_engine_version": lammps_version,
            "opencl_platform": opencl_platform,
            "opencl_device": None,
            "requested_opencl_device": opencl_device,
            "opencl_device_applied": gpu_active,
            "pair_style": PAIR_STYLE,
        },
        "lammps_version": lammps_version,
        "particles": particles,
        "atom_count": particles,
        "steps": steps,
        "step_count": steps,
        "warmup_steps": warmup_steps,
        "dt": dt,
        "simulated_time_lj": simulated_time,
        "wall_s": wall_s,
        "steps_per_s": steps_per_s,
        "opencl_platform": opencl_platform,
        "opencl_device": None,
        "requested_opencl_device": opencl_device,
        "opencl_device_applied": gpu_active,
        "gpu_pair_style_active": gpu_active,
        "neighbor_build_on_host": gpu["neighbor_build_on_host"],
        "gpu_log_excerpt": gpu["excerpt"],
        "density_lj": density,
        "temperature_lj": temperature,
        "box_length_lj": box_length,
        "potential_energy_lj": potential_energy,
        "kinetic_energy_lj": kinetic_energy,
        "comparison_role": "lammps",
        "comparison_metric_family": "steps/s",
        "comparison_command": COMMAND,
        "comparison_raw_output_path": f"{COMPARISON_OUTPUT_ROOT}/lammps-lj-N{particles}.json",
        "finite": bool(
            math.isfinite(steps_per_s)
            and math.isfinite(potential_energy)
            and math.isfinite(kinetic_energy)
        ),
        "blocker": blocker,
    }


def _load_lammps() -> tuple[Any, str]:
    try:
        from lammps import __version__ as lammps_version
        from lammps import lammps
    except Exception as exc:  # pragma: no cover - depends on optional reference package.
        msg = f"LAMMPS import unavailable: {exc}"
        raise RuntimeError(msg) from exc
    return lammps, str(lammps_version)


def _configure_box(
    lmp: Any,
    *,
    box_length: float,
    opencl_platform: str,
) -> None:
    commands = [
        "clear",
        "units lj",
        "atom_style atomic",
        "boundary p p p",
        f"package gpu 1 platform {opencl_platform}",
        "suffix gpu",
        f"region box block 0 {box_length:.12g} 0 {box_length:.12g} 0 {box_length:.12g}",
        "create_box 1 box",
        "mass 1 1.0",
    ]
    for command in commands:
        lmp.command(command)


def _configure_interactions(
    lmp: Any,
    *,
    dt: float,
    temperature: float,
    seed: int,
) -> None:
    for command in (
        f"pair_style {PAIR_STYLE} {CUTOFF:.12g}",
        "pair_coeff 1 1 1.0 1.0 2.5",
        f"neighbor {SKIN:.12g} bin",
        "neigh_modify every 1 delay 0 check yes",
        f"velocity all create {temperature:.12g} {seed} mom yes rot no dist gaussian",
        "fix integrate all nve",
        f"timestep {dt:.12g}",
        "thermo 0",
    ):
        lmp.command(command)


def _fcc_positions(particles: int, density: float) -> tuple[np.ndarray, float]:
    """Return FCC positions and the cubic box edge, matching MLX ``fcc_lattice``."""

    positions, cell = fcc_lattice(particles, density=density)
    box_length = float(np.asarray(cell.lengths, dtype=np.float64)[0])
    coords = np.asarray(positions, dtype=np.float64)
    return coords, box_length


def _inspect_gpu_log(log_path: str) -> dict[str, Any]:
    """Parse the LAMMPS log to confirm the GPU pair build engaged."""

    try:
        text = Path(log_path).read_text()
    except OSError:
        text = ""
    gpu_pair_style_active = bool(re.search(r"pair\s+lj/cut/gpu,\s*perpetual", text))
    neighbor_build_on_host = "does not support neighbor lists on device" in text
    excerpt = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"GPU|OpenCL|lj/cut/gpu|package gpu|Device\b", line, re.I)
    ]
    return {
        "gpu_pair_style_active": gpu_pair_style_active,
        "neighbor_build_on_host": neighbor_build_on_host,
        "excerpt": excerpt[:12],
    }


def _extract_lammps_value(lmp: Any, name: str) -> float:
    try:
        return float(lmp.get_thermo(name))
    except Exception:
        lmp.command(f"variable benchmark_{name} equal {name}")
        return float(lmp.extract_variable("benchmark_" + name, None, 0))


def _blocked_payload(args: argparse.Namespace, *, blocker: str) -> dict[str, Any]:
    return {
        "status": "blocked",
        "engine": ENGINE,
        "benchmark_name": BENCHMARK_NAME,
        "fixture": args.fixture,
        "hardware": _hardware_info(),
        "runtime": {
            "python_version": platform_module.python_version(),
            "reference_engine_version": None,
            "opencl_platform": args.opencl_platform,
            "opencl_device": None,
            "requested_opencl_device": args.opencl_device,
            "opencl_device_applied": False,
            "pair_style": PAIR_STYLE,
        },
        "lammps_version": None,
        "particles": args.particles,
        "atom_count": args.particles,
        "steps": args.steps,
        "step_count": args.steps,
        "warmup_steps": args.warmup_steps,
        "dt": args.dt,
        "simulated_time_lj": args.steps * args.dt,
        "wall_s": None,
        "steps_per_s": None,
        "opencl_platform": args.opencl_platform,
        "opencl_device": None,
        "requested_opencl_device": args.opencl_device,
        "opencl_device_applied": False,
        "gpu_pair_style_active": False,
        "neighbor_build_on_host": None,
        "gpu_log_excerpt": [],
        "density_lj": args.density,
        "temperature_lj": args.temperature,
        "box_length_lj": None,
        "potential_energy_lj": None,
        "kinetic_energy_lj": None,
        "comparison_role": "lammps",
        "comparison_metric_family": "steps/s",
        "comparison_command": COMMAND,
        "comparison_raw_output_path": f"{COMPARISON_OUTPUT_ROOT}/lammps-lj-N{args.particles}.json",
        "finite": False,
        "blocker": blocker,
    }


def _normalize_reference_payload(
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = dict(payload)
    payload["timing_value"] = payload.get(TIMING_METRIC)
    return normalize_benchmark_payload(
        payload,
        benchmark_name=BENCHMARK_NAME,
        fixture=args.fixture,
        timing_metric=TIMING_METRIC,
        hardware=payload.get("hardware") or _hardware_info(),
        runtime=payload.get("runtime") or {},
        engine=ENGINE,
        atom_count=args.particles,
        step_count=args.steps,
        evaluation_count=args.steps,
        finite=bool(payload.get("finite")),
        status=payload.get("status"),
        blocker=payload.get("blocker"),
        command=COMMAND,
        raw_output_path=None,
    )


def _hardware_info() -> dict[str, str]:
    return {
        "system": platform_module.system(),
        "release": platform_module.release(),
        "machine": platform_module.machine(),
        "processor": platform_module.processor(),
        "python_version": platform_module.python_version(),
        "platform": platform_module.platform(),
    }


def _format_human_payload(payload: dict[str, Any]) -> str:
    if payload.get("status") == "blocked":
        return (
            f"LAMMPS OpenCL {payload['particles']} atoms: blocked; "
            f"blocker={payload.get('blocker')}"
        )
    suffix = "" if payload.get("gpu_pair_style_active") else " (CPU diagnostic)"
    return "LAMMPS OpenCL {particles} atoms: {steps_per_s:.1f} steps/s".format(**payload) + suffix


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
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=0.8)
    parser.add_argument("--fixture", default=FIXTURE)
    parser.add_argument("--opencl-platform", default="0")
    parser.add_argument("--opencl-device", default="0")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--csv", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
