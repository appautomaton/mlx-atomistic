"""Benchmark the small-system Ewald reference electrostatics backend."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell
from mlx_atomistic.nonbonded import (
    EwaldReferenceConfig,
    ewald_reference_coulomb_energy_forces,
)
from mlx_atomistic.pme import PMEConfig, pme_coulomb_energy_forces
from mlx_atomistic.runtime import get_runtime_info

SCOPE_NOTE = (
    "Ewald reference is a small-system correctness backend; PME comparison rows "
    "exercise the standalone mesh backend but are not GPCRmd-scale PME."
)


@dataclass(frozen=True)
class EwaldReferenceBenchmarkResult:
    """One Ewald reference benchmark row."""

    case: str
    atoms: int
    evaluations: int
    cell_length: float
    alpha: float
    real_cutoff: float
    reciprocal_cutoff: int
    k_vector_count: int
    real_shift_count: int
    wall_s: float
    ms_per_eval: float
    evals_per_s: float
    energy: float
    coulomb_real: float
    coulomb_reciprocal: float
    coulomb_self: float
    max_force_norm: float
    net_force_norm: float
    finite: bool
    pme_mesh_shape: str
    pme_energy: float
    pme_energy_abs_error: float
    pme_force_max_abs_error: float
    pme_force_rms_error: float
    pme_ms_per_eval: float
    pme_finite: bool

    def to_dict(self) -> dict:
        """Return a JSON- and CSV-safe row."""

        return asdict(self)


def _parse_atom_counts(value: str) -> tuple[int, ...]:
    atom_counts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not atom_counts or any(count <= 1 for count in atom_counts):
        msg = "--atoms must contain integers greater than 1"
        raise ValueError(msg)
    return atom_counts


def _parse_mesh_shape(value: str) -> tuple[int, int, int]:
    mesh_shape = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(mesh_shape) != 3 or any(size < 4 for size in mesh_shape):
        msg = "--pme-mesh must contain three integers >= 4"
        raise ValueError(msg)
    return mesh_shape  # type: ignore[return-value]


def _neutral_charges(atom_count: int) -> np.ndarray:
    charges = np.where(np.arange(atom_count) % 2 == 0, 0.35, -0.35).astype(np.float32)
    charges[-1] -= np.sum(charges, dtype=np.float64).astype(np.float32)
    return charges


def _deterministic_positions(atom_count: int, cell_length: float, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + atom_count)
    low = 0.15 * cell_length
    high = 0.85 * cell_length
    return rng.uniform(low, high, size=(atom_count, 3)).astype(np.float32)


def _real_shift_count(cell_length: float, real_cutoff: float) -> int:
    radius = int(np.ceil(real_cutoff / cell_length)) + 1
    width = 2 * radius + 1
    return width * width * width


def _k_vector_count(reciprocal_cutoff: int) -> int:
    if reciprocal_cutoff == 0:
        return 0
    width = 2 * reciprocal_cutoff + 1
    return width * width * width - 1


def run_case(
    *,
    atom_count: int,
    evaluations: int,
    cell_length: float,
    alpha: float,
    real_cutoff: float,
    reciprocal_cutoff: int,
    pme_mesh_shape: tuple[int, int, int] = (32, 32, 32),
    seed: int = 17,
) -> EwaldReferenceBenchmarkResult:
    """Run one deterministic neutral Ewald reference benchmark case."""

    if evaluations <= 0:
        msg = "evaluations must be positive"
        raise ValueError(msg)
    positions = mx.array(_deterministic_positions(atom_count, cell_length, seed=seed))
    charges = mx.array(_neutral_charges(atom_count))
    cell = Cell.cubic(cell_length)
    config = EwaldReferenceConfig(
        alpha=alpha,
        real_cutoff=real_cutoff,
        reciprocal_cutoff=reciprocal_cutoff,
    )

    energy, forces, components = ewald_reference_coulomb_energy_forces(
        positions,
        charges,
        cell,
        config=config,
    )
    mx.eval(energy, forces, *components.values())

    start = perf_counter()
    for _ in range(evaluations):
        energy, forces, components = ewald_reference_coulomb_energy_forces(
            positions,
            charges,
            cell,
            config=config,
        )
        mx.eval(energy, forces, *components.values())
    wall_s = perf_counter() - start

    pme_config = PMEConfig(
        mesh_shape=pme_mesh_shape,
        alpha=alpha,
        real_cutoff=real_cutoff,
    )
    pme_energy, pme_forces, pme_components = pme_coulomb_energy_forces(
        positions,
        charges,
        cell,
        config=pme_config,
    )
    mx.eval(
        pme_energy,
        pme_forces,
        *(value for value in pme_components.values() if isinstance(value, mx.array)),
    )

    pme_start = perf_counter()
    for _ in range(evaluations):
        pme_energy, pme_forces, pme_components = pme_coulomb_energy_forces(
            positions,
            charges,
            cell,
            config=pme_config,
        )
        mx.eval(
            pme_energy,
            pme_forces,
            *(value for value in pme_components.values() if isinstance(value, mx.array)),
        )
    pme_wall_s = perf_counter() - pme_start

    force_array = np.asarray(forces, dtype=np.float64)
    pme_force_array = np.asarray(pme_forces, dtype=np.float64)
    force_delta = pme_force_array - force_array
    component_values = {
        name: float(np.asarray(value, dtype=np.float64)) for name, value in components.items()
    }
    finite = bool(
        np.isfinite(float(energy))
        and np.all(np.isfinite(force_array))
        and all(np.isfinite(value) for value in component_values.values())
    )
    pme_finite = bool(
        np.isfinite(float(pme_energy))
        and np.all(np.isfinite(pme_force_array))
        and np.all(np.isfinite(force_delta))
    )
    force_norms = np.linalg.norm(force_array, axis=1)
    return EwaldReferenceBenchmarkResult(
        case="ewald_reference_neutral_periodic",
        atoms=atom_count,
        evaluations=evaluations,
        cell_length=cell_length,
        alpha=alpha,
        real_cutoff=real_cutoff,
        reciprocal_cutoff=reciprocal_cutoff,
        k_vector_count=_k_vector_count(reciprocal_cutoff),
        real_shift_count=_real_shift_count(cell_length, real_cutoff),
        wall_s=wall_s,
        ms_per_eval=1000.0 * wall_s / evaluations,
        evals_per_s=evaluations / wall_s if wall_s > 0.0 else 0.0,
        energy=float(energy),
        coulomb_real=component_values["coulomb_real"],
        coulomb_reciprocal=component_values["coulomb_reciprocal"],
        coulomb_self=component_values["coulomb_self"],
        max_force_norm=float(np.max(force_norms)),
        net_force_norm=float(np.linalg.norm(np.sum(force_array, axis=0))),
        finite=finite,
        pme_mesh_shape="x".join(str(size) for size in pme_mesh_shape),
        pme_energy=float(pme_energy),
        pme_energy_abs_error=abs(float(pme_energy) - float(energy)),
        pme_force_max_abs_error=float(np.max(np.abs(force_delta))),
        pme_force_rms_error=float(np.sqrt(np.mean(force_delta * force_delta))),
        pme_ms_per_eval=1000.0 * pme_wall_s / evaluations,
        pme_finite=pme_finite,
    )


def run_benchmark(
    *,
    atom_counts: tuple[int, ...] = (4, 8, 16),
    evaluations: int = 3,
    cell_length: float = 12.0,
    alpha: float = 0.35,
    real_cutoff: float = 5.0,
    reciprocal_cutoff: int = 4,
    pme_mesh_shape: tuple[int, int, int] = (32, 32, 32),
    seed: int = 17,
) -> list[EwaldReferenceBenchmarkResult]:
    """Run the Ewald reference benchmark matrix."""

    return [
        run_case(
            atom_count=atom_count,
            evaluations=evaluations,
            cell_length=cell_length,
            alpha=alpha,
            real_cutoff=real_cutoff,
            reciprocal_cutoff=reciprocal_cutoff,
            pme_mesh_shape=pme_mesh_shape,
            seed=seed,
        )
        for atom_count in atom_counts
    ]


def build_payload(
    *,
    atom_counts: tuple[int, ...] = (4, 8, 16),
    evaluations: int = 3,
    cell_length: float = 12.0,
    alpha: float = 0.35,
    real_cutoff: float = 5.0,
    reciprocal_cutoff: int = 4,
    pme_mesh_shape: tuple[int, int, int] = (32, 32, 32),
    seed: int = 17,
) -> dict:
    """Build the JSON payload for the benchmark CLI."""

    results = run_benchmark(
        atom_counts=atom_counts,
        evaluations=evaluations,
        cell_length=cell_length,
        alpha=alpha,
        real_cutoff=real_cutoff,
        reciprocal_cutoff=reciprocal_cutoff,
        pme_mesh_shape=pme_mesh_shape,
        seed=seed,
    )
    rows = [result.to_dict() for result in results]
    return {
        "runtime": asdict(get_runtime_info()),
        "scope_note": SCOPE_NOTE,
        "atom_counts": list(atom_counts),
        "evaluations": evaluations,
        "cell_length": cell_length,
        "alpha": alpha,
        "real_cutoff": real_cutoff,
        "reciprocal_cutoff": reciprocal_cutoff,
        "pme_mesh_shape": list(pme_mesh_shape),
        "case_count": len(rows),
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
    parser.add_argument("--atoms", default="4,8,16")
    parser.add_argument("--evaluations", type=int, default=3)
    parser.add_argument("--cell-length", type=float, default=12.0)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--real-cutoff", type=float, default=5.0)
    parser.add_argument("--reciprocal-cutoff", type=int, default=4)
    parser.add_argument("--pme-mesh", default="32,32,32")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.evaluations <= 0:
        msg = "--evaluations must be positive"
        raise ValueError(msg)
    if args.cell_length <= 0.0:
        msg = "--cell-length must be positive"
        raise ValueError(msg)
    if args.real_cutoff <= 0.0:
        msg = "--real-cutoff must be positive"
        raise ValueError(msg)

    payload = build_payload(
        atom_counts=_parse_atom_counts(args.atoms),
        evaluations=args.evaluations,
        cell_length=args.cell_length,
        alpha=args.alpha,
        real_cutoff=args.real_cutoff,
        reciprocal_cutoff=args.reciprocal_cutoff,
        pme_mesh_shape=_parse_mesh_shape(args.pme_mesh),
        seed=args.seed,
    )
    if args.csv is not None:
        _write_csv(args.csv, payload["cases"])
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print(
        "ewald_reference "
        f"cases={payload['case_count']} "
        f"note={payload['scope_note']}"
    )


if __name__ == "__main__":
    main()
