"""Run NVE/NVT stability diagnostics for bonded and Lennard-Jones systems."""

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
from mlx_atomistic.examples import bonded_chain_example, lj_liquid_example
from mlx_atomistic.md import (
    LangevinThermostat,
    LennardJonesPotential,
    SimulationConfig,
    simulate_nve,
    simulate_nvt,
)
from mlx_atomistic.neighbors import NeighborListManager
from mlx_atomistic.runtime import get_runtime_info


@dataclass(frozen=True)
class StabilityBenchmarkResult:
    """One MD stability benchmark row."""

    case: str
    ensemble: str
    particles: int
    steps: int
    dt: float
    ms_per_step: float
    max_energy_drift: float
    relative_energy_drift: float
    initial_temperature: float
    mean_temperature: float
    final_temperature: float
    target_temperature: float | None
    final_pair_count: int | None
    final_rebuild_count: int | None
    finite: bool
    has_nonfinite: bool

    def to_dict(self) -> dict:
        """Return a JSON- and CSV-friendly row."""

        return asdict(self)


def parse_float_list(value: str) -> list[float]:
    """Parse comma-separated floats."""

    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0.0 for item in values):
        msg = "value must contain positive floats"
        raise ValueError(msg)
    return values


def parse_int_list(value: str) -> list[int]:
    """Parse comma-separated positive integers."""

    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        msg = "value must contain positive integers"
        raise ValueError(msg)
    return values


def _finite_result(result) -> bool:
    arrays = [
        np.asarray(result.potential_energy),
        np.asarray(result.kinetic_energy),
        np.asarray(result.total_energy),
        np.asarray(result.temperature),
    ]
    return bool(all(np.all(np.isfinite(array)) for array in arrays))


def _relative_energy_drift(result) -> float:
    total_energy = np.asarray(result.total_energy, dtype=np.float64)
    drift = np.max(np.abs(total_energy - total_energy[0]))
    return float(drift / max(abs(float(total_energy[0])), 1e-12))


def _result_row(
    *,
    case: str,
    ensemble: str,
    particles: int,
    steps: int,
    dt: float,
    elapsed: float,
    result,
) -> StabilityBenchmarkResult:
    summary = summarize_md_result(result, ensemble=ensemble)
    finite = _finite_result(result)
    return StabilityBenchmarkResult(
        case=case,
        ensemble=ensemble,
        particles=particles,
        steps=steps,
        dt=dt,
        ms_per_step=elapsed * 1000.0 / max(steps, 1),
        max_energy_drift=float(summary["max_energy_drift"]),
        relative_energy_drift=_relative_energy_drift(result),
        initial_temperature=float(summary["initial_temperature"]),
        mean_temperature=float(summary["mean_temperature"]),
        final_temperature=float(summary["final_temperature"]),
        target_temperature=summary.get("target_temperature"),
        final_pair_count=summary.get("final_pair_count"),
        final_rebuild_count=summary.get("final_rebuild_count"),
        finite=finite,
        has_nonfinite=not finite,
    )


def _bonded_velocities() -> np.ndarray:
    return np.array(
        [
            [0.03, 0.00, 0.00],
            [-0.01, 0.02, 0.00],
            [0.00, -0.02, 0.01],
            [0.01, 0.00, -0.02],
        ],
        dtype=np.float32,
    )


def run_bonded_nve_case(*, dt: float, steps: int) -> StabilityBenchmarkResult:
    """Run an NVE bonded-chain stability case."""

    positions, _, _, force_terms = bonded_chain_example()
    velocities = _bonded_velocities()
    config = SimulationConfig(dt=dt, steps=steps, sample_interval=max(steps, 1))

    start = perf_counter()
    result = simulate_nve(positions, velocities, force_terms=force_terms, config=config)
    mx.eval(result.total_energy, result.temperature)
    elapsed = perf_counter() - start

    return _result_row(
        case="bonded-chain",
        ensemble="nve",
        particles=positions.shape[0],
        steps=steps,
        dt=dt,
        elapsed=elapsed,
        result=result,
    )


def run_lj_case(
    *,
    ensemble: str,
    particles: int,
    steps: int,
    dt: float,
    temperature: float,
    seed: int,
) -> StabilityBenchmarkResult:
    """Run an LJ liquid NVE or NVT stability case."""

    positions, velocities, cell, _ = lj_liquid_example(
        particles=particles,
        temperature=temperature,
        seed=seed,
    )
    potential = LennardJonesPotential(cutoff=2.5)
    manager = NeighborListManager(cell, cutoff=2.5, skin=0.4)
    config = SimulationConfig(dt=dt, steps=steps, sample_interval=max(steps, 1))

    start = perf_counter()
    if ensemble == "nvt":
        result = simulate_nvt(
            positions,
            velocities,
            cell=cell,
            force_terms=potential,
            neighbor_manager=manager,
            config=config,
            thermostat=LangevinThermostat(temperature=temperature, friction=1.0, seed=seed),
        )
    elif ensemble == "nve":
        result = simulate_nve(
            positions,
            velocities,
            cell=cell,
            force_terms=potential,
            neighbor_manager=manager,
            config=config,
        )
    else:
        msg = "ensemble must be 'nve' or 'nvt'"
        raise ValueError(msg)
    mx.eval(result.total_energy, result.temperature)
    elapsed = perf_counter() - start

    return _result_row(
        case="lj-liquid",
        ensemble=ensemble,
        particles=particles,
        steps=steps,
        dt=dt,
        elapsed=elapsed,
        result=result,
    )


def run_stability_suite(
    *,
    sizes: list[int],
    steps: int,
    bonded_steps: int,
    dt_values: list[float],
    temperature: float,
    seed: int,
) -> list[StabilityBenchmarkResult]:
    """Run the default stability benchmark matrix."""

    results: list[StabilityBenchmarkResult] = []
    for dt in dt_values:
        results.append(run_bonded_nve_case(dt=dt, steps=bonded_steps))
    for particles in sizes:
        results.append(
            run_lj_case(
                ensemble="nve",
                particles=particles,
                steps=steps,
                dt=dt_values[0],
                temperature=temperature,
                seed=seed,
            )
        )
        results.append(
            run_lj_case(
                ensemble="nvt",
                particles=particles,
                steps=steps,
                dt=dt_values[0],
                temperature=temperature,
                seed=seed,
            )
        )
    return results


def build_payload(
    *,
    sizes: list[int],
    steps: int,
    bonded_steps: int,
    dt_values: list[float],
    temperature: float,
    seed: int,
) -> dict:
    """Run stability diagnostics and return a CLI-friendly payload."""

    results = run_stability_suite(
        sizes=sizes,
        steps=steps,
        bonded_steps=bonded_steps,
        dt_values=dt_values,
        temperature=temperature,
        seed=seed,
    )
    cases = [result.to_dict() for result in results]
    return {
        "runtime": asdict(get_runtime_info()),
        "summary": {
            "case_count": len(cases),
            "finite_cases": sum(1 for case in cases if case["finite"]),
            "nonfinite_cases": sum(1 for case in cases if case["has_nonfinite"]),
            "max_relative_energy_drift": max(
                case["relative_energy_drift"] for case in cases
            ),
        },
        "cases": cases,
    }


def _write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="128", help="Comma-separated LJ particle counts.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--bonded-steps", type=int, default=500)
    parser.add_argument("--dt-values", default="0.001,0.002,0.004")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--csv", default=None, help="Optional path for per-case CSV output.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.steps < 0:
        msg = "--steps must be non-negative"
        raise ValueError(msg)
    if args.bonded_steps < 0:
        msg = "--bonded-steps must be non-negative"
        raise ValueError(msg)

    payload = build_payload(
        sizes=parse_int_list(args.sizes),
        steps=args.steps,
        bonded_steps=args.bonded_steps,
        dt_values=parse_float_list(args.dt_values),
        temperature=args.temperature,
        seed=args.seed,
    )
    if args.csv is not None:
        _write_csv(args.csv, payload["cases"])

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    runtime = payload["runtime"]
    print(
        f"runtime mlx={runtime['mlx_version']} device={runtime['default_device']} "
        f"metal={runtime['metal_available']}"
    )
    for case in payload["cases"]:
        finite = "finite" if case["finite"] else "NONFINITE"
        target = "-" if case["target_temperature"] is None else f"{case['target_temperature']:.4g}"
        print(
            f"{case['case']:13s} {case['ensemble']:3s} N={case['particles']:5d} "
            f"steps={case['steps']:5d} dt={case['dt']:.4g} "
            f"ms/step={case['ms_per_step']:.3f} rel_drift={case['relative_energy_drift']:.6g} "
            f"Tmean={case['mean_temperature']:.4g} Ttarget={target} {finite}"
        )


if __name__ == "__main__":
    main()
