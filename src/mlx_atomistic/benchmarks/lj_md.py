"""Benchmark Lennard-Jones MD force paths."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

from mlx_atomistic.diagnostics import summarize_md_result
from mlx_atomistic.initialize import fcc_lattice, thermal_velocities
from mlx_atomistic.md import (
    LangevinThermostat,
    LennardJonesPotential,
    SimulationConfig,
    simulate,
    simulate_nve,
    simulate_nvt,
)
from mlx_atomistic.neighbors import NeighborListManager, build_neighbor_list
from mlx_atomistic.runtime import get_runtime_info


@dataclass(frozen=True)
class BenchmarkResult:
    mode: str
    particles: int
    steps: int
    pairs: int | None
    rebuilds: int
    ms_per_step: float
    energy_drift: float
    mean_temperature: float
    final_temperature: float
    final_potential_energy_by_term: dict[str, float]


def run_case(
    *,
    particles: int,
    steps: int,
    mode: str,
    density: float,
    temperature: float,
    seed: int,
):
    positions, cell = fcc_lattice(particles, density=density)
    velocities = thermal_velocities(particles, temperature=temperature, seed=seed)
    potential = LennardJonesPotential(cutoff=2.5)

    pairs = None
    pair_count = None
    rebuilds = 0
    if mode == "static-neighbor":
        neighbor_list = build_neighbor_list(positions, cell, cutoff=potential.cutoff or 2.5)
        pairs = neighbor_list.pairs
        pair_count = neighbor_list.pair_count

    start = perf_counter()
    if mode in {"dynamic-neighbor", "nvt-dynamic-neighbor"}:
        neighbor_manager = NeighborListManager(cell, cutoff=potential.cutoff or 2.5)
        config = SimulationConfig(steps=steps, sample_interval=steps)
        if mode == "nvt-dynamic-neighbor":
            result = simulate_nvt(
                positions,
                velocities,
                cell=cell,
                force_terms=potential,
                neighbor_manager=neighbor_manager,
                config=config,
                thermostat=LangevinThermostat(temperature=temperature, friction=1.0, seed=seed),
            )
        else:
            result = simulate_nve(
                positions,
                velocities,
                cell=cell,
                force_terms=potential,
                neighbor_manager=neighbor_manager,
                config=config,
            )
        pair_count = int(np.array(result.pair_count)[-1])
        rebuilds = int(np.array(result.rebuild_count)[-1])
    else:
        result = simulate(
            positions,
            velocities,
            cell=cell,
            potential=potential,
            pairs=pairs,
            steps=steps,
        )
    mx.eval(result.total_energy)
    elapsed = perf_counter() - start

    summary = summarize_md_result(result)
    return BenchmarkResult(
        mode=mode,
        particles=particles,
        steps=steps,
        pairs=pair_count,
        rebuilds=rebuilds,
        ms_per_step=elapsed * 1000.0 / steps,
        energy_drift=float(summary["max_energy_drift"]),
        mean_temperature=float(summary["mean_temperature"]),
        final_temperature=float(summary["final_temperature"]),
        final_potential_energy_by_term=dict(summary.get("final_potential_energy_by_term", {})),
    )


def parse_sizes(value: str | None, fallback: int) -> list[int]:
    """Parse a comma-separated particle-size list."""

    if value is None:
        return [fallback]
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        msg = "--sizes must contain positive integers"
        raise ValueError(msg)
    return sizes


def _write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--density", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sizes", default=None, help="Comma-separated particle counts.")
    parser.add_argument("--csv", default=None, help="Optional path for per-case CSV output.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    sizes = parse_sizes(args.sizes, args.particles)
    results = [
        run_case(
            particles=particles,
            steps=args.steps,
            mode=mode,
            density=args.density,
            temperature=args.temperature,
            seed=args.seed,
        )
        for particles in sizes
        for mode in ("all-pairs", "static-neighbor", "dynamic-neighbor", "nvt-dynamic-neighbor")
    ]
    rows = [asdict(result) for result in results]
    if args.csv is not None:
        _write_csv(args.csv, rows)

    if args.json:
        payload = {
            "runtime": asdict(get_runtime_info()),
            "cases": rows,
        }
        print(json.dumps(payload, indent=2))
        return

    runtime = get_runtime_info()
    print(
        f"runtime mlx={runtime.mlx_version} device={runtime.default_device} "
        f"metal={runtime.metal_available}"
    )
    for result in results:
        pair_text = "-" if result.pairs is None else str(result.pairs)
        print(
            f"{result.mode:9s} particles={result.particles} steps={result.steps} "
            f"pairs={pair_text} rebuilds={result.rebuilds} ms/step={result.ms_per_step:.3f} "
            f"energy_drift={result.energy_drift:.6g} "
            f"mean_T={result.mean_temperature:.4g} final_T={result.final_temperature:.4g}"
        )


if __name__ == "__main__":
    main()
