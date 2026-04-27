"""Benchmark molecular mechanics force-term evaluation costs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

from mlx_atomistic.examples import bonded_chain_example, charged_dimer_example
from mlx_atomistic.forcefields import CoulombPotential
from mlx_atomistic.initialize import fcc_lattice
from mlx_atomistic.md import LennardJonesPotential
from mlx_atomistic.neighbors import build_neighbor_list
from mlx_atomistic.runtime import get_runtime_info


@dataclass(frozen=True)
class ForceTermBenchmarkResult:
    category: str
    term: str
    evaluations: int
    particles: int
    pairs: int | None
    ms_per_eval: float
    energy: float

    def to_dict(self) -> dict:
        """Return a JSON- and CSV-friendly row."""

        return asdict(self)


def run_term(
    term,
    positions,
    *,
    evaluations: int,
    category: str,
    cell=None,
    pairs=None,
) -> ForceTermBenchmarkResult:
    """Measure repeated energy/force evaluations for one term."""

    energy = None
    forces = None
    start = perf_counter()
    for _ in range(evaluations):
        energy, forces = term.energy_forces(positions, cell=cell, pairs=pairs)
    if energy is not None and forces is not None:
        mx.eval(energy, forces)
    elapsed = perf_counter() - start
    pair_count = None if pairs is None else int(pairs.shape[0])
    return ForceTermBenchmarkResult(
        category=category,
        term=str(getattr(term, "name", type(term).__name__)),
        evaluations=evaluations,
        particles=int(positions.shape[0]),
        pairs=pair_count,
        ms_per_eval=elapsed * 1000.0 / evaluations,
        energy=float(energy),
    )


def run_neighbor_build(*, particles: int, evaluations: int) -> ForceTermBenchmarkResult:
    """Measure periodic neighbor-list construction."""

    positions, cell = fcc_lattice(particles, density=0.8)
    neighbor_list = None
    start = perf_counter()
    for _ in range(evaluations):
        neighbor_list = build_neighbor_list(positions, cell, cutoff=2.5, skin=0.4)
    if neighbor_list is not None:
        mx.eval(neighbor_list.pairs)
    elapsed = perf_counter() - start
    return ForceTermBenchmarkResult(
        category="neighbor-list",
        term="build",
        evaluations=evaluations,
        particles=int(positions.shape[0]),
        pairs=None if neighbor_list is None else int(neighbor_list.pair_count),
        ms_per_eval=elapsed * 1000.0 / evaluations,
        energy=0.0,
    )


def run_benchmark(*, evaluations: int, particles: int = 128) -> list[ForceTermBenchmarkResult]:
    """Run the default molecular mechanics force-term benchmark."""

    positions, _, _, bonded_terms = bonded_chain_example()
    charged_positions, _, _, charged_terms = charged_dimer_example()
    results = [
        run_term(
            term,
            positions,
            evaluations=evaluations,
            category="bonded-autodiff",
        )
        for term in bonded_terms
    ]
    results.extend(
        run_term(
            term,
            charged_positions,
            evaluations=evaluations,
            category="coulomb-direct-small",
        )
        for term in charged_terms
    )
    lj_positions, lj_cell = fcc_lattice(particles, density=0.8)
    neighbor_list = build_neighbor_list(lj_positions, lj_cell, cutoff=2.5, skin=0.4)
    results.append(run_neighbor_build(particles=particles, evaluations=evaluations))
    results.append(
        run_term(
            LennardJonesPotential(cutoff=2.5),
            lj_positions,
            evaluations=evaluations,
            category="lj-pair-eval",
            cell=lj_cell,
            pairs=neighbor_list.pairs,
        )
    )
    charges = np.where(np.arange(lj_positions.shape[0]) % 2 == 0, 1.0, -1.0).astype(np.float32)
    results.append(
        run_term(
            CoulombPotential(charges=charges, cutoff=2.5, shift=True),
            lj_positions,
            evaluations=evaluations,
            category="coulomb-direct",
            cell=lj_cell,
            pairs=neighbor_list.pairs,
        )
    )
    return results


def _write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=int, default=10)
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--csv", default=None, help="Optional path for per-case CSV output.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.evaluations <= 0:
        msg = "--evaluations must be positive"
        raise ValueError(msg)
    if args.particles <= 0:
        msg = "--particles must be positive"
        raise ValueError(msg)

    results = run_benchmark(evaluations=args.evaluations, particles=args.particles)
    rows = [asdict(result) for result in results]
    if args.csv is not None:
        _write_csv(args.csv, rows)
    if args.json:
        print(
            json.dumps(
                {
                    "runtime": asdict(get_runtime_info()),
                    "cases": rows,
                },
                indent=2,
            )
        )
        return

    runtime = get_runtime_info()
    print(
        f"runtime mlx={runtime.mlx_version} device={runtime.default_device} "
        f"metal={runtime.metal_available}"
    )
    for result in results:
        print(
            f"{result.category:20s} {result.term:10s} N={result.particles:5d} "
            f"pairs={result.pairs if result.pairs is not None else '-'} evals={result.evaluations} "
            f"ms/eval={result.ms_per_eval:.3f} energy={result.energy:.6g}"
        )


if __name__ == "__main__":
    main()
