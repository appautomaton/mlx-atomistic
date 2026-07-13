"""Validate compact neighbor-listed nonbonded execution against a tiled oracle."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_atomistic.benchmarks import get_hardware_info
from mlx_atomistic.forcefields import NonbondedPotential
from mlx_atomistic.initialize import fcc_lattice
from mlx_atomistic.neighbors import build_neighbor_list
from mlx_atomistic.nonbonded import dense_combined_energy_forces
from mlx_atomistic.runtime import get_runtime_info
from mlx_atomistic.topology import Topology

DEFAULT_SIZES = (1000, 4000, 16000, 50000, 92001)


def _parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size < 4 for size in sizes):
        msg = "sizes must contain integers >= 4"
        raise ValueError(msg)
    return sizes


def _topology(n_atoms: int, *, semantic: bool) -> Topology:
    kwargs: dict[str, Any] = {}
    if semantic:
        kwargs.update(
            bonds=[(0, 1)],
            exclusions=[(1, 2)],
            one_four_pairs=[(0, 3)],
        )
    return Topology.from_sequences(
        n_atoms=n_atoms,
        eager_nonbonded_pair_limit=0,
        **kwargs,
    )


def _parameters(n_atoms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(n_atoms, dtype=np.int32)
    sigma = (0.9 + 0.05 * (indices % 5)).astype(np.float32)
    epsilon = (0.1 + 0.025 * (indices % 7)).astype(np.float32)
    charges = np.where(indices % 2 == 0, 0.1, -0.1).astype(np.float32)
    return sigma, epsilon, charges


def _is_resource_ceiling(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    message = str(exc).lower()
    return any(token in message for token in ("out of memory", "allocation", "resource exhausted"))


def run_case(
    n_atoms: int,
    *,
    density: float,
    cutoff: float,
    tile_size: int,
    semantic_topology: bool,
    energy_atol: float,
    energy_rtol: float,
    force_atol: float,
    force_rtol: float,
) -> dict[str, Any]:
    """Run one compact-pair versus tiled-all-pairs parity case."""

    positions, cell = fcc_lattice(n_atoms, density=density)
    topology = _topology(n_atoms, semantic=semantic_topology)
    sigma, epsilon, charges = _parameters(n_atoms)
    term = NonbondedPotential(
        sigma=sigma,
        epsilon=epsilon,
        charges=charges,
        topology=topology,
        cutoff=cutoff,
        lj_shift=False,
        coulomb_shift=False,
        lj_one_four_scale=0.5,
        coulomb_one_four_scale=0.75,
        backend="auto",
    )

    build_started = perf_counter()
    neighbors = build_neighbor_list(
        positions,
        cell,
        cutoff=cutoff,
        skin=0.0,
        backend="mlx_cell_pairs",
        sort_pairs=False,
    )
    neighbor_build_wall_seconds = perf_counter() - build_started

    pair_started = perf_counter()
    pair_energy, pair_forces = term.energy_forces(
        positions,
        cell,
        pairs=neighbors.pairs,
    )
    mx.eval(pair_energy, pair_forces)
    force_evaluation_wall_seconds = perf_counter() - pair_started

    reference_topology = topology if semantic_topology else None
    reference_started = perf_counter()
    reference_lj, reference_coulomb, reference_forces = dense_combined_energy_forces(
        positions,
        sigma=term.sigma,
        epsilon=term.epsilon,
        charges=term.charges,
        coulomb_constant=term.coulomb_constant,
        cutoff=cutoff,
        lj_shift=term.lj_shift,
        coulomb_shift=term.coulomb_shift,
        cell=cell,
        topology=reference_topology,
        lj_one_four_scale=term.lj_one_four_scale,
        coulomb_one_four_scale=term.coulomb_one_four_scale,
        tile_size=tile_size,
    )
    reference_energy = reference_lj + reference_coulomb
    mx.eval(reference_energy, reference_forces)
    reference_evaluation_wall_seconds = perf_counter() - reference_started

    pair_energy_value = float(np.asarray(pair_energy))
    reference_energy_value = float(np.asarray(reference_energy))
    energy_abs_delta = abs(pair_energy_value - reference_energy_value)
    energy_scale = max(abs(reference_energy_value), 1.0)
    energy_rel_delta = energy_abs_delta / energy_scale
    pair_force_values = np.asarray(pair_forces, dtype=np.float64)
    reference_force_values = np.asarray(reference_forces, dtype=np.float64)
    force_delta = np.abs(pair_force_values - reference_force_values)
    force_max_abs_delta = float(np.max(force_delta)) if force_delta.size else 0.0
    reference_force_max_abs = (
        float(np.max(np.abs(reference_force_values))) if reference_force_values.size else 0.0
    )
    energy_passed = energy_abs_delta <= energy_atol + energy_rtol * energy_scale
    force_passed = force_max_abs_delta <= force_atol + force_rtol * max(
        reference_force_max_abs,
        1.0,
    )

    return {
        "status": "validated" if energy_passed and force_passed else "failed",
        "atom_count": n_atoms,
        "backend": neighbors.backend,
        "fallback_reason": neighbors.fallback_reason,
        "compaction_backend": neighbors.compaction_backend,
        "pair_policy": topology.nonbonded_pair_policy,
        "dense_pair_cache_materialized": topology._nonbonded_pairs is not None,
        "reference_backend": "mlx_tiled_all_pairs",
        "orthorhombic_minimum_image": True,
        "semantic_topology": semantic_topology,
        "exclusion_count": int(topology.exclusions.shape[0]),
        "one_four_count": int(topology.one_four_pairs.shape[0]),
        "pair_count": neighbors.pair_count,
        "compact_pair_count": neighbors.compact_pair_count,
        "candidate_count": neighbors.candidate_count,
        "candidate_waste_count": neighbors.candidate_waste_count,
        "candidate_waste_fraction": neighbors.candidate_waste_fraction,
        "neighbor_build_wall_seconds": neighbor_build_wall_seconds,
        "force_evaluation_wall_seconds": force_evaluation_wall_seconds,
        "reference_evaluation_wall_seconds": reference_evaluation_wall_seconds,
        "pair_energy": pair_energy_value,
        "reference_energy": reference_energy_value,
        "energy_abs_delta": energy_abs_delta,
        "energy_rel_delta": energy_rel_delta,
        "force_max_abs_delta": force_max_abs_delta,
        "reference_force_max_abs": reference_force_max_abs,
        "energy_tolerance": {
            "atol": energy_atol,
            "rtol": energy_rtol,
        },
        "force_tolerance": {
            "atol": force_atol,
            "rtol": force_rtol,
        },
    }


def build_payload(
    *,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    density: float = 0.8,
    cutoff: float = 2.5,
    tile_size: int = 256,
    energy_atol: float = 1e-3,
    energy_rtol: float = 1e-5,
    force_atol: float = 1e-3,
    force_rtol: float = 1e-5,
) -> dict[str, Any]:
    """Build parity rows for the requested atom-count ladder."""

    if not sizes or any(size < 4 for size in sizes):
        msg = "sizes must contain integers >= 4"
        raise ValueError(msg)
    if not np.isfinite(density) or density <= 0.0:
        msg = "density must be finite and positive"
        raise ValueError(msg)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        msg = "cutoff must be finite and positive"
        raise ValueError(msg)
    if tile_size <= 0:
        msg = "tile_size must be positive"
        raise ValueError(msg)

    rows: list[dict[str, Any]] = []
    for index, size in enumerate(sizes):
        try:
            row = run_case(
                size,
                density=density,
                cutoff=cutoff,
                tile_size=tile_size,
                semantic_topology=index == 0,
                energy_atol=energy_atol,
                energy_rtol=energy_rtol,
                force_atol=force_atol,
                force_rtol=force_rtol,
            )
        except Exception as exc:  # pragma: no cover - host resource failures are opt-in.
            row = {
                "status": "resource_ceiling" if _is_resource_ceiling(exc) else "failed",
                "atom_count": size,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        rows.append(row)

    validated_sizes = [
        int(row["atom_count"]) for row in rows if row.get("status") == "validated"
    ]
    required_sizes = {1000, 4000, 16000, 50000}
    return {
        "benchmark_name": "neighbor_nonbonded_parity",
        "fixture": "synthetic_orthorhombic_lj_coulomb",
        "comparison_status": "diagnostic",
        "comparison_reason": (
            "internal MLX parity ladder; no same-physics OpenMM/LAMMPS row is generated here"
        ),
        "hardware": get_hardware_info(),
        "runtime": asdict(get_runtime_info()),
        "config": {
            "sizes": list(sizes),
            "density": density,
            "cutoff": cutoff,
            "tile_size": tile_size,
            "energy_atol": energy_atol,
            "energy_rtol": energy_rtol,
            "force_atol": force_atol,
            "force_rtol": force_rtol,
        },
        "triclinic_status": "deferred_fail_closed",
        "case_count": len(rows),
        "validated_case_count": len(validated_sizes),
        "largest_validated_size": max(validated_sizes) if validated_sizes else None,
        "required_ladder_passed": required_sizes.issubset(validated_sizes),
        "cases": rows,
    }


def write_payload(payload: dict[str, Any], path: str | Path) -> None:
    """Write a parity payload as stable JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> None:
    """Run the parity ladder CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default=",".join(str(size) for size in DEFAULT_SIZES))
    parser.add_argument("--density", type=float, default=0.8)
    parser.add_argument("--cutoff", type=float, default=2.5)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--energy-atol", type=float, default=1e-3)
    parser.add_argument("--energy-rtol", type=float, default=1e-5)
    parser.add_argument("--force-atol", type=float, default=1e-3)
    parser.add_argument("--force-rtol", type=float, default=1e-5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(
        sizes=_parse_sizes(args.sizes),
        density=args.density,
        cutoff=args.cutoff,
        tile_size=args.tile_size,
        energy_atol=args.energy_atol,
        energy_rtol=args.energy_rtol,
        force_atol=args.force_atol,
        force_rtol=args.force_rtol,
    )
    write_payload(payload, args.out)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"validated={payload['validated_case_count']}/{payload['case_count']} "
            f"largest={payload['largest_validated_size']} out={args.out}"
        )


if __name__ == "__main__":
    main()
