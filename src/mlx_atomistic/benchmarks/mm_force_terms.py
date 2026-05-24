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

from mlx_atomistic.benchmarks import (
    default_benchmark_command,
    get_hardware_info,
    normalize_benchmark_payload,
    normalize_benchmark_row,
)
from mlx_atomistic.examples import (
    bonded_chain_example,
    charged_dimer_example,
    mixed_lj_fluid_example,
    water_like_constrained_example,
)
from mlx_atomistic.forcefields import CoulombPotential
from mlx_atomistic.initialize import fcc_lattice
from mlx_atomistic.md import LennardJonesPotential
from mlx_atomistic.neighbors import build_neighbor_list
from mlx_atomistic.runtime import get_runtime_info
from mlx_atomistic.virtual_sites import VirtualSiteManager, tip4p_ew_virtual_site


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


def run_constraint_projection(*, evaluations: int) -> ForceTermBenchmarkResult:
    """Measure constrained position and velocity projection."""

    system, _, constraints = water_like_constrained_example()
    positions = system.positions
    velocities = system.velocities
    error = None
    start = perf_counter()
    for _ in range(evaluations):
        positions, error = constraints.apply_positions(positions, system.masses, system.cell)
        velocities = constraints.apply_velocities(
            positions,
            velocities,
            system.masses,
            system.cell,
        )
    if error is not None:
        mx.eval(positions, velocities, error)
    elapsed = perf_counter() - start
    return ForceTermBenchmarkResult(
        category="constraints",
        term="distance-project",
        evaluations=evaluations,
        particles=system.atom_count,
        pairs=int(constraints.pairs.shape[0]),
        ms_per_eval=elapsed * 1000.0 / evaluations,
        energy=0.0 if error is None else float(error),
    )


def _tip4p_real_positions(waters: int) -> mx.array:
    base = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.7569503, 0.5858823, 0.0],
            [0.7569503, -0.5858823, 0.0],
        ],
        dtype=np.float32,
    )
    offsets = np.asarray([[3.0 * idx, 0.0, 0.0] for idx in range(waters)], dtype=np.float32)
    return mx.array((base[None, :, :] + offsets[:, None, :]).reshape(waters * 3, 3))


def run_tip4p_virtual_site_reconstruction(
    *,
    evaluations: int,
    waters: int,
) -> ForceTermBenchmarkResult:
    """Measure TIP4P-Ew M-site position reconstruction overhead."""

    positions = _tip4p_real_positions(waters)
    sites = tuple(
        tip4p_ew_virtual_site(3 * idx, 3 * idx + 1, 3 * idx + 2) for idx in range(waters)
    )
    manager = VirtualSiteManager(sites, n_real_atoms=int(positions.shape[0]))
    reconstructed = None
    start = perf_counter()
    for _ in range(evaluations):
        reconstructed = manager.extend_positions(positions)
        mx.eval(reconstructed)
    elapsed = perf_counter() - start
    return ForceTermBenchmarkResult(
        category="virtual-sites",
        term="tip4p-ew-reconstruct",
        evaluations=evaluations,
        particles=manager.n_total_atoms,
        pairs=waters,
        ms_per_eval=elapsed * 1000.0 / evaluations,
        energy=0.0,
    )


def run_tip4p_virtual_site_force_redistribution(
    *,
    evaluations: int,
    waters: int,
) -> ForceTermBenchmarkResult:
    """Measure TIP4P-Ew virtual-site force redistribution overhead."""

    positions = _tip4p_real_positions(waters)
    sites = tuple(
        tip4p_ew_virtual_site(3 * idx, 3 * idx + 1, 3 * idx + 2) for idx in range(waters)
    )
    manager = VirtualSiteManager(sites, n_real_atoms=int(positions.shape[0]))
    full_positions = manager.extend_positions(positions)
    forces = mx.concatenate(
        [
            mx.zeros_like(full_positions[: manager.n_real_atoms]),
            mx.ones_like(full_positions[manager.n_real_atoms :]),
        ],
        axis=0,
    )
    redistributed = None
    start = perf_counter()
    for _ in range(evaluations):
        redistributed = manager.redistribute_forces(forces, full_positions)
        mx.eval(redistributed)
    elapsed = perf_counter() - start
    force_norm = (
        0.0
        if redistributed is None
        else float(mx.sqrt(mx.sum(redistributed * redistributed)))
    )
    return ForceTermBenchmarkResult(
        category="virtual-sites",
        term="tip4p-ew-force-redistribute",
        evaluations=evaluations,
        particles=manager.n_total_atoms,
        pairs=waters,
        ms_per_eval=elapsed * 1000.0 / evaluations,
        energy=force_norm,
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
    mixed_system, mixed_force_field = mixed_lj_fluid_example(particles=particles)
    combined_nonbonded = mixed_force_field.build_force_terms(mixed_system)[-1]
    mixed_neighbor_list = build_neighbor_list(mixed_system.positions, mixed_system.cell, cutoff=2.5)
    results.append(
        run_term(
            combined_nonbonded,
            mixed_system.positions,
            evaluations=evaluations,
            category="combined-nonbonded",
            cell=mixed_system.cell,
            pairs=mixed_neighbor_list.pairs,
        )
    )
    results.append(run_constraint_projection(evaluations=evaluations))
    tip4p_waters = max(1, min(8, particles // 4))
    results.append(
        run_tip4p_virtual_site_reconstruction(
            evaluations=evaluations,
            waters=tip4p_waters,
        )
    )
    results.append(
        run_tip4p_virtual_site_force_redistribution(
            evaluations=evaluations,
            waters=tip4p_waters,
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


def build_payload(*, evaluations: int, particles: int = 128) -> dict:
    """Run force-term benchmarks and return a normalized JSON payload."""

    results = run_benchmark(evaluations=evaluations, particles=particles)
    rows = [
        normalize_benchmark_row(
            result.to_dict(),
            benchmark_name="mm_force_terms",
            fixture=result.category,
            timing_metric="ms_per_eval",
        )
        for result in results
    ]
    hardware = get_hardware_info()
    runtime = asdict(get_runtime_info())
    payload = {
        "benchmark_name": "mm_force_terms",
        "fixture": "synthetic_force_terms",
        "hardware": hardware,
        "runtime": runtime,
        "config": {
            "evaluations": evaluations,
            "particles": particles,
        },
        "case_count": len(rows),
        "cases": rows,
    }
    return normalize_benchmark_payload(
        payload,
        benchmark_name="mm_force_terms",
        fixture="synthetic_force_terms",
        timing_metric="ms_per_eval",
        hardware=hardware,
        runtime=runtime,
        evaluation_count=evaluations,
        finite=all(bool(row["finite"]) for row in rows),
        command=default_benchmark_command("mm_force_terms"),
    )


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

    payload = build_payload(evaluations=args.evaluations, particles=args.particles)
    rows = payload["cases"]
    if args.csv is not None:
        _write_csv(args.csv, rows)
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    runtime = get_runtime_info()
    print(
        f"runtime mlx={runtime.mlx_version} device={runtime.default_device} "
        f"metal={runtime.metal_available}"
    )
    for row in rows:
        print(
            f"{row['category']:20s} {row['term']:10s} N={row['particles']:5d} "
            f"pairs={row['pairs'] if row['pairs'] is not None else '-'} "
            f"evals={row['evaluations']} ms/eval={row['ms_per_eval']:.3f} "
            f"energy={row['energy']:.6g}"
        )


if __name__ == "__main__":
    main()
