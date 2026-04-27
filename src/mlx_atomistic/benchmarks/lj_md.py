"""Benchmark Lennard-Jones MD force paths."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--density", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    results = [
        run_case(
            particles=args.particles,
            steps=args.steps,
            mode=mode,
            density=args.density,
            temperature=args.temperature,
            seed=args.seed,
        )
        for mode in ("all-pairs", "static-neighbor", "dynamic-neighbor", "nvt-dynamic-neighbor")
    ]

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
        return

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
