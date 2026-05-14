"""Benchmark MLX-first MD nonbonded acceleration paths."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

from mlx_atomistic.forcefields import NonbondedPotential
from mlx_atomistic.initialize import fcc_lattice
from mlx_atomistic.neighbors import build_neighbor_list
from mlx_atomistic.nonbonded import (
    NonbondedBackend,
    estimate_dense_nonbonded_bytes,
    validate_nonbonded_backend,
)
from mlx_atomistic.runtime import get_runtime_info


@dataclass(frozen=True)
class MDAccelerationBenchmarkResult:
    """One MD nonbonded backend benchmark row."""

    backend: str
    particles: int
    evaluations: int
    pairs: int
    rebuild_count: int
    tile_size: int | None
    estimated_pair_bytes: int
    estimated_dense_bytes: int
    neighbor_rebuild_ms_per_eval: float
    force_eval_ms_per_eval: float
    ms_per_eval: float
    ns_per_day_at_dt_0_002: float
    energy: float
    energy_abs_delta: float
    max_force_abs_delta: float

    def to_dict(self) -> dict:
        """Return a JSON- and CSV-safe row."""

        return asdict(self)


def _parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item) for item in value.split(",") if item)
    if not sizes or any(size <= 0 for size in sizes):
        msg = "--sizes must contain positive integers"
        raise ValueError(msg)
    return sizes


def _parse_backends(value: str) -> tuple[NonbondedBackend, ...]:
    backends = tuple(validate_nonbonded_backend(item) for item in value.split(",") if item)
    if not backends:
        msg = "--backends must contain at least one backend"
        raise ValueError(msg)
    return backends


def _mixed_nonbonded_parameters(n_atoms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = np.arange(n_atoms, dtype=np.float32)
    sigma = np.where(index % 2 == 0, 1.0, 1.12).astype(np.float32)
    epsilon = np.where(index % 3 == 0, 1.0, 0.72).astype(np.float32)
    charges = np.where(index % 2 == 0, 0.35, -0.35).astype(np.float32)
    return sigma, epsilon, charges


def _evaluate(
    potential: NonbondedPotential,
    positions: mx.array,
    cell,
    *,
    pairs=None,
) -> tuple[float, np.ndarray]:
    energy, forces = potential.energy_forces(positions, cell=cell, pairs=pairs)
    mx.eval(energy, forces)
    return float(energy), np.asarray(forces)


def _time_backend(
    backend: NonbondedBackend,
    *,
    positions: mx.array,
    cell,
    potential: NonbondedPotential,
    evaluations: int,
    tile_size: int,
    reference_energy: float,
    reference_forces: np.ndarray,
) -> MDAccelerationBenchmarkResult:
    pair_count = positions.shape[0] * (positions.shape[0] - 1) // 2
    timed_potential = NonbondedPotential(
        sigma=potential.sigma,
        epsilon=potential.epsilon,
        charges=potential.charges,
        coulomb_constant=potential.coulomb_constant,
        cutoff=potential.cutoff,
        lj_shift=potential.lj_shift,
        coulomb_shift=potential.coulomb_shift,
        backend=backend,
        tile_size=tile_size,
    )
    energy = reference_energy
    forces = reference_forces
    neighbor_list = None
    rebuild_count = 0
    estimated_pair_bytes = 0
    neighbor_rebuild_elapsed = 0.0
    force_eval_elapsed = 0.0
    if backend == "mlx_pairs":
        rebuild_start = perf_counter()
        neighbor_list = build_neighbor_list(positions, cell, cutoff=2.5, skin=0.4)
        neighbor_rebuild_elapsed += perf_counter() - rebuild_start
        pair_count = neighbor_list.pair_count
        rebuild_count = 1
        estimated_pair_bytes = neighbor_list.estimated_pair_bytes

    start = perf_counter()
    if backend == "python_neighbor":
        for _ in range(evaluations):
            rebuild_start = perf_counter()
            dynamic_neighbors = build_neighbor_list(positions, cell, cutoff=2.5, skin=0.4)
            neighbor_rebuild_elapsed += perf_counter() - rebuild_start
            pair_count = dynamic_neighbors.pair_count
            rebuild_count += 1
            estimated_pair_bytes = dynamic_neighbors.estimated_pair_bytes
            force_start = perf_counter()
            energy, forces = _evaluate(
                timed_potential,
                positions,
                cell,
                pairs=dynamic_neighbors.pairs,
            )
            force_eval_elapsed += perf_counter() - force_start
    elif backend == "mlx_pairs":
        if neighbor_list is None:
            msg = "mlx_pairs benchmark requires a prebuilt neighbor list"
            raise RuntimeError(msg)
        for _ in range(evaluations):
            force_start = perf_counter()
            energy, forces = _evaluate(timed_potential, positions, cell, pairs=neighbor_list.pairs)
            force_eval_elapsed += perf_counter() - force_start
    else:
        for _ in range(evaluations):
            force_start = perf_counter()
            energy, forces = _evaluate(timed_potential, positions, cell)
            force_eval_elapsed += perf_counter() - force_start
    elapsed = perf_counter() - start
    ms_per_eval = elapsed * 1000.0 / evaluations
    ns_per_day = 0.002 * (1000.0 / ms_per_eval) * 86400.0 if ms_per_eval > 0.0 else 0.0
    return MDAccelerationBenchmarkResult(
        backend=backend,
        particles=int(positions.shape[0]),
        evaluations=evaluations,
        pairs=int(pair_count),
        rebuild_count=rebuild_count,
        tile_size=tile_size if backend == "mlx_tiled" else None,
        estimated_pair_bytes=estimated_pair_bytes,
        estimated_dense_bytes=estimate_dense_nonbonded_bytes(
            int(positions.shape[0]),
            components="combined",
        ),
        neighbor_rebuild_ms_per_eval=neighbor_rebuild_elapsed * 1000.0 / evaluations,
        force_eval_ms_per_eval=force_eval_elapsed * 1000.0 / evaluations,
        ms_per_eval=ms_per_eval,
        ns_per_day_at_dt_0_002=ns_per_day,
        energy=energy,
        energy_abs_delta=abs(energy - reference_energy),
        max_force_abs_delta=float(np.max(np.abs(forces - reference_forces))),
    )


def run_benchmark(
    *,
    sizes: tuple[int, ...] = (128, 512, 2048),
    backends: tuple[NonbondedBackend, ...] = (
        "python_neighbor",
        "mlx_pairs",
        "mlx_dense",
        "mlx_tiled",
    ),
    evaluations: int = 3,
    tile_size: int = 512,
    include_large: bool = False,
) -> list[MDAccelerationBenchmarkResult]:
    """Run the MD acceleration benchmark matrix."""

    if include_large and 8192 not in sizes:
        sizes = (*sizes, 8192)
    results: list[MDAccelerationBenchmarkResult] = []
    for particles in sizes:
        positions, cell = fcc_lattice(particles, density=0.8)
        sigma, epsilon, charges = _mixed_nonbonded_parameters(int(positions.shape[0]))
        reference = NonbondedPotential(
            sigma=sigma,
            epsilon=epsilon,
            charges=charges,
            cutoff=2.5,
            backend="mlx_dense",
        )
        reference_energy, reference_forces = _evaluate(reference, positions, cell)
        potential = NonbondedPotential(
            sigma=sigma,
            epsilon=epsilon,
            charges=charges,
            cutoff=2.5,
            backend="auto",
        )
        for backend in backends:
            results.append(
                _time_backend(
                    backend,
                    positions=positions,
                    cell=cell,
                    potential=potential,
                    evaluations=evaluations,
                    tile_size=tile_size,
                    reference_energy=reference_energy,
                    reference_forces=reference_forces,
                )
            )
    return results


def build_payload(
    *,
    sizes: tuple[int, ...] = (128, 512, 2048),
    backends: tuple[NonbondedBackend, ...] = (
        "python_neighbor",
        "mlx_pairs",
        "mlx_dense",
        "mlx_tiled",
    ),
    evaluations: int = 3,
    tile_size: int = 512,
    include_large: bool = False,
) -> dict:
    """Build the JSON payload for the benchmark CLI."""

    results = run_benchmark(
        sizes=sizes,
        backends=backends,
        evaluations=evaluations,
        tile_size=tile_size,
        include_large=include_large,
    )
    rows = [result.to_dict() for result in results]
    fastest = min(rows, key=lambda row: row["ms_per_eval"]) if rows else None
    return {
        "runtime": asdict(get_runtime_info()),
        "sizes": list(sizes),
        "backends": list(backends),
        "evaluations": evaluations,
        "tile_size": tile_size,
        "include_large": include_large,
        "case_count": len(rows),
        "fastest_case": fastest,
        "cases": rows,
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
    parser.add_argument("--sizes", default="128,512,2048")
    parser.add_argument(
        "--backends",
        default="python_neighbor,mlx_pairs,mlx_dense,mlx_tiled",
    )
    parser.add_argument("--evaluations", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--include-large", action="store_true")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.evaluations <= 0:
        msg = "--evaluations must be positive"
        raise ValueError(msg)
    if args.tile_size <= 0:
        msg = "--tile-size must be positive"
        raise ValueError(msg)

    payload = build_payload(
        sizes=_parse_sizes(args.sizes),
        backends=_parse_backends(args.backends),
        evaluations=args.evaluations,
        tile_size=args.tile_size,
        include_large=args.include_large,
    )
    if args.csv is not None:
        _write_csv(args.csv, payload["cases"])
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    fastest = payload["fastest_case"]
    print(
        "md_acceleration "
        f"cases={payload['case_count']} "
        f"fastest={None if fastest is None else fastest['backend']}"
    )


if __name__ == "__main__":
    main()
