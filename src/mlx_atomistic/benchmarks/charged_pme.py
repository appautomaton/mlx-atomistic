"""Prepare and measure deterministic charged-PME benchmark workloads."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_atomistic.artifacts import build_mlx_system_from_artifact, load_prepared_mlx_artifact
from mlx_atomistic.benchmarks import get_hardware_info
from mlx_atomistic.benchmarks.gpcrmd_runtime import max_rss_mb
from mlx_atomistic.md import LangevinThermostat, SimulationConfig, simulate_nvt
from mlx_atomistic.neighbors import NeighborListManager
from mlx_atomistic.prep.io import JSON_NAME, NPZ_NAME, load_prepared_system, save_prepared_system
from mlx_atomistic.prep.supercell import (
    normalize_supercell_replicas,
    prepared_supercell_summary,
    replicate_prepared_system,
)
from mlx_atomistic.runtime import get_runtime_info

SUPERCELL_SUMMARY_NAME = "supercell_summary.json"
RUNTIME_SCHEMA = "mlx_atomistic.charged_pme_runtime.v1"


def prepare_payload(
    *,
    source: str | Path,
    replicas: object,
    out: str | Path,
    assignment_order: int | None = None,
    background_policy: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic prepared-system supercell benchmark artifact.

    Args:
        source: Source prepared-system directory.
        replicas: Three positive integer counts ``(nx, ny, nz)``.
        out: Caller-owned output directory.
        assignment_order: Optional PME assignment-order override.
        background_policy: Optional PME background policy override.

    Returns:
        A JSON-serializable success, blocked, or failed payload. Missing source
        inputs are reported as blocked and do not create the output directory.
    """

    source_path = Path(source)
    out_path = Path(out)
    replica_shape = normalize_supercell_replicas(replicas)
    required_paths = (source_path / JSON_NAME, source_path / NPZ_NAME)
    missing = [str(path) for path in required_paths if not path.is_file()]
    base = {
        "kind": "mlx_atomistic.charged_pme_prepare",
        "source": str(source_path),
        "out": str(out_path),
        "replicas": list(replica_shape),
        "assignment_order_override": assignment_order,
        "background_policy_override": background_policy,
        "written": False,
    }
    if missing:
        return {
            **base,
            "status": "blocked",
            "blockers": ["missing_prepared_source:" + path for path in missing],
            "summary": None,
        }

    try:
        source_prepared = load_prepared_system(source_path)
        replicated = replicate_prepared_system(
            source_prepared,
            replica_shape,
            assignment_order=assignment_order,
            background_policy=background_policy,
        )
        summary = prepared_supercell_summary(
            replicated,
            source_atom_count=source_prepared.atom_count,
            replicas=replica_shape,
        )
        summary.update(
            _supercell_validation_summary(
                source_prepared,
                replicated,
                replica_shape,
            )
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        return {
            **base,
            "status": "failed",
            "blockers": [f"prepared_supercell_failed:{type(exc).__name__}:{exc}"],
            "summary": None,
        }

    save_prepared_system(replicated, out_path)
    summary_path = out_path / SUPERCELL_SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {
        **base,
        "status": "ok",
        "blockers": [],
        "written": True,
        "summary_path": str(summary_path),
        "prepared_json": str(out_path / JSON_NAME),
        "prepared_npz": str(out_path / NPZ_NAME),
        "summary": summary,
    }


def runtime_payload(
    *,
    prepared: str | Path,
    warmups: int,
    steps: int,
    out: str | Path,
    dt_ps: float = 0.004,
    temperature_k: float = 300.0,
    seed: int = 17,
) -> dict[str, Any]:
    """Run bounded fixed-cell charged-PME NVT with one reusable plan.

    Args:
        prepared: Strict production prepared-system directory.
        warmups: Untimed warmup step count; must be positive.
        steps: Measured NVT step count; must be at least two.
        out: JSON output path.
        dt_ps: Timestep in picoseconds. Defaults to ``0.004``.
        temperature_k: Langevin target temperature. Defaults to ``300``.
        seed: Deterministic thermostat seed. Defaults to ``17``.

    Returns:
        JSON-serializable passing, failed, blocked, or resource-ceiling payload.
    """

    prepared_path = Path(prepared)
    out_path = Path(out)
    base = {
        "kind": RUNTIME_SCHEMA,
        "prepared": str(prepared_path),
        "out": str(out_path),
        "warmup_steps": int(warmups),
        "measured_steps": int(steps),
        "dt_ps": float(dt_ps),
        "temperature_target_k": float(temperature_k),
        "seed": int(seed),
        "status": "blocked",
        "passed": False,
        "blockers": [],
        "hardware": get_hardware_info(),
        "runtime": asdict(get_runtime_info()),
    }
    validation_blockers = []
    if warmups <= 0:
        validation_blockers.append("warmups_must_be_positive")
    if steps < 2:
        validation_blockers.append("measured_steps_must_be_at_least_two")
    if not np.isfinite(dt_ps) or dt_ps <= 0.0:
        validation_blockers.append("dt_ps_must_be_finite_positive")
    if not np.isfinite(temperature_k) or temperature_k <= 0.0:
        validation_blockers.append("temperature_k_must_be_finite_positive")
    required = (prepared_path / JSON_NAME, prepared_path / NPZ_NAME)
    validation_blockers.extend(
        f"missing_prepared_input:{path}" for path in required if not path.is_file()
    )
    if validation_blockers:
        return _write_runtime_payload(
            {**base, "blockers": validation_blockers},
            out_path,
        )

    try:
        setup_started = time.perf_counter()
        artifact = load_prepared_mlx_artifact(prepared_path, require_production=True)
        system, force_terms, constraints = build_mlx_system_from_artifact(
            artifact,
            eager_nonbonded_pair_limit=0,
        )
        if system.cell is None:
            raise ValueError("charged PME runtime requires a periodic fixed cell")
        bound_terms = _bind_pme_plans(force_terms, system.cell)
        nonbonded = _find_pme_term(bound_terms)
        topology = nonbonded.topology
        if topology is None:
            raise ValueError("charged PME runtime requires a topology-aware nonbonded term")
        cutoff = float(nonbonded.cutoff)
        neighbor_manager = NeighborListManager(
            system.cell,
            cutoff=cutoff,
            skin=0.3,
            check_interval=1,
            sort_pairs=False,
            backend="mlx_cell_blocks",
        )
        setup_seconds = time.perf_counter() - setup_started
        unit_system = artifact.unit_system
        simulation_units = {
            "kinetic_energy_scale": (
                1.0 if unit_system is None else unit_system.kinetic_energy_scale
            ),
            "force_to_acceleration_scale": (
                1.0 if unit_system is None else unit_system.force_to_acceleration_scale
            ),
            "boltzmann_constant": (
                1.0 if unit_system is None else unit_system.boltzmann_constant
            ),
        }
        plan = nonbonded.pme_plan
        if plan is None:
            raise ValueError("charged PME runtime did not bind an execution plan")

        warmup_started = time.perf_counter()
        warmup_result = simulate_nvt(
            system.positions,
            system.velocities,
            masses=system.masses,
            cell=system.cell,
            force_terms=bound_terms,
            neighbor_manager=neighbor_manager,
            config=_simulation_config(
                steps=warmups,
                dt_ps=dt_ps,
                simulation_units=simulation_units,
            ),
            constraints=constraints,
            thermostat=LangevinThermostat(
                temperature=temperature_k,
                friction=1.0,
                seed=seed,
            ),
        )
        warmup_seconds = time.perf_counter() - warmup_started
        reuse_after_warmup = plan.reuse_count
        measured_neighbor_update_start = neighbor_manager.update_wall_seconds
        measured_neighbor_rebuild_start = neighbor_manager.rebuild_wall_seconds

        measured_started = time.perf_counter()
        measured_result = simulate_nvt(
            warmup_result.final_state.positions,
            warmup_result.final_state.velocities,
            masses=warmup_result.final_state.masses,
            cell=system.cell,
            force_terms=bound_terms,
            neighbor_manager=neighbor_manager,
            config=_simulation_config(
                steps=steps,
                dt_ps=dt_ps,
                simulation_units=simulation_units,
            ),
            constraints=constraints,
            thermostat=LangevinThermostat(
                temperature=temperature_k,
                friction=1.0,
                seed=seed + 1,
            ),
        )
        measured_seconds = time.perf_counter() - measured_started
        mx.eval(
            measured_result.sampled_positions,
            measured_result.sampled_velocities,
            measured_result.potential_energy,
            measured_result.kinetic_energy,
            measured_result.total_energy,
            measured_result.temperature,
            measured_result.constraint_max_error,
            measured_result.final_state.forces,
        )
        arrays = (
            np.asarray(measured_result.sampled_positions),
            np.asarray(measured_result.sampled_velocities),
            np.asarray(measured_result.potential_energy),
            np.asarray(measured_result.kinetic_energy),
            np.asarray(measured_result.total_energy),
            np.asarray(measured_result.temperature),
            np.asarray(measured_result.constraint_max_error),
            np.asarray(measured_result.final_state.forces),
        )
        finite = all(bool(np.all(np.isfinite(value))) for value in arrays)
        simulated_ns = steps * dt_ps / 1000.0
        ns_per_day = (
            simulated_ns / measured_seconds * 86400.0 if measured_seconds > 0.0 else 0.0
        )
        neighbor_list = neighbor_manager.neighbor_list
        topology_report = {
            "pair_policy": topology.nonbonded_pair_policy,
            "pair_cache_materialized": getattr(topology, "_nonbonded_pairs", None)
            is not None,
            "nonbonded_pair_count": topology.nonbonded_pair_count,
        }
        neighbor_report = {
            **measured_result.nonbonded_report,
            "manager_backend": neighbor_manager.backend,
            "representation": (
                None if neighbor_list is None else neighbor_list.representation_kind
            ),
            "fallback_reason": (
                None if neighbor_list is None else neighbor_list.fallback_reason
            ),
            "measured_update_wall_seconds": (
                neighbor_manager.update_wall_seconds - measured_neighbor_update_start
            ),
            "measured_rebuild_wall_seconds": (
                neighbor_manager.rebuild_wall_seconds - measured_neighbor_rebuild_start
            ),
        }
        final_plan = plan.to_dict()
        checks = {
            "warmup_completed": warmups >= 1,
            "measured_steps_completed": steps >= 2,
            "finite_state": finite,
            "fixed_cell": bool(
                np.allclose(
                    np.asarray(system.cell.lengths, dtype=np.float64),
                    np.asarray(plan.cell_lengths, dtype=np.float64),
                    rtol=0.0,
                    atol=1.0e-6,
                )
            ),
            "one_plan_build": final_plan["build_count"] == 1,
            "plan_reused_in_warmup": reuse_after_warmup > 0,
            "plan_reused_in_measurement": final_plan["reuse_count"] > reuse_after_warmup,
            "lazy_topology": topology_report["pair_policy"] == "lazy",
            "pair_cache_unmaterialized": not topology_report["pair_cache_materialized"],
            "neighbor_blocks": (
                neighbor_report["manager_backend"] == "mlx_cell_blocks"
                and neighbor_report["representation"] == "blocks"
            ),
            "no_neighbor_fallback": neighbor_report["fallback_reason"] is None,
            "positive_throughput": math.isfinite(ns_per_day) and ns_per_day > 0.0,
        }
        passed = all(checks.values())
        payload = {
            **base,
            "status": "ok" if passed else "failed",
            "passed": passed,
            "blockers": [] if passed else [name for name, value in checks.items() if not value],
            "atom_count": artifact.atom_count,
            "cell_lengths_angstrom": np.asarray(system.cell.lengths).tolist(),
            "pme": _pme_payload(nonbonded.pme_config),
            "plan": final_plan,
            "topology": topology_report,
            "neighbor": neighbor_report,
            "checks": checks,
            "finite": finite,
            "timings": {
                "setup_seconds": setup_seconds,
                "warmup_seconds": warmup_seconds,
                "measured_seconds": measured_seconds,
                "seconds_per_measured_step": measured_seconds / steps,
                "plan_setup_seconds": final_plan["setup_seconds"],
                "force_evaluation_seconds": measured_result.nonbonded_report.get(
                    "force_evaluation_wall_seconds"
                ),
                "synchronization_seconds": measured_result.runtime_sync_report.get(
                    "runtime_sync_total_wall_seconds"
                ),
                "profile_detail_path": str(out_path.parent / "profile" / "pme-profile.json"),
            },
            "throughput": {
                "simulated_ns": simulated_ns,
                "ns_per_day": ns_per_day,
                "steps_per_second": steps / measured_seconds,
                "openmm_ratio": None,
                "comparison_status": "not_reported_without_matching_runtime_manifest",
            },
            "memory": {
                "max_rss_mb": max_rss_mb(),
                "mlx_active_memory_bytes": _mlx_memory_value("get_active_memory"),
                "mlx_peak_memory_bytes": _mlx_memory_value("get_peak_memory"),
                "mlx_cache_memory_bytes": _mlx_memory_value("get_cache_memory"),
            },
            "state": {
                "potential_energy_kj_mol": _last_float(
                    measured_result.potential_energy
                ),
                "kinetic_energy_kj_mol": _last_float(measured_result.kinetic_energy),
                "total_energy_kj_mol": _last_float(measured_result.total_energy),
                "temperature_k": _last_float(measured_result.temperature),
                "constraint_max_error_angstrom": _last_float(
                    measured_result.constraint_max_error
                ),
                "sampled_step_count": int(
                    np.asarray(measured_result.sampled_steps).shape[0]
                ),
                "diagnostic_step_count": int(
                    np.asarray(measured_result.diagnostic_steps).shape[0]
                ),
            },
            "runtime_sync": measured_result.runtime_sync_report,
        }
        return _write_runtime_payload(payload, out_path)
    except MemoryError as exc:  # pragma: no cover - host resource dependent.
        return _write_runtime_payload(
            {
                **base,
                "status": "resource_ceiling",
                "blockers": [f"MemoryError:{exc}"],
                "memory": {"max_rss_mb": max_rss_mb()},
            },
            out_path,
        )
    except Exception as exc:  # pragma: no cover - heavy host/runtime dependent.
        return _write_runtime_payload(
            {
                **base,
                "status": "failed",
                "blockers": [f"{type(exc).__name__}:{exc}"],
                "memory": {"max_rss_mb": max_rss_mb()},
            },
            out_path,
        )


def _bind_pme_plans(force_terms: list[Any], cell: Any) -> tuple[Any, ...]:
    bound = []
    for term in force_terms:
        if getattr(term, "electrostatics", None) == "pme":
            bound.append(term.bind_pme_plan(cell))
        else:
            bound.append(term)
    return tuple(bound)


def _find_pme_term(force_terms: tuple[Any, ...]) -> Any:
    terms = [term for term in force_terms if getattr(term, "electrostatics", None) == "pme"]
    if len(terms) != 1:
        raise ValueError(f"charged PME runtime expected one PME term, found {len(terms)}")
    return terms[0]


def _simulation_config(
    *,
    steps: int,
    dt_ps: float,
    simulation_units: dict[str, float],
) -> SimulationConfig:
    return SimulationConfig(
        dt=dt_ps,
        steps=steps,
        sample_interval=1,
        diagnostic_interval=1,
        pressure_diagnostics=False,
        compile_force_evaluator=False,
        **simulation_units,
    )


def _pme_payload(config: Any) -> dict[str, Any]:
    return {
        "mesh_shape": list(config.mesh_shape),
        "alpha": float(config.alpha),
        "real_cutoff": float(config.real_cutoff),
        "assignment_order": int(config.assignment_order),
        "charge_tolerance": float(config.charge_tolerance),
        "deconvolve_assignment": bool(config.deconvolve_assignment),
        "background_policy": config.background_policy,
    }


def _mlx_memory_value(name: str) -> int | None:
    accessor = getattr(mx, name, None)
    if not callable(accessor):
        return None
    try:
        return int(accessor())
    except (RuntimeError, TypeError, ValueError):
        return None


def _last_float(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(array.reshape(-1)[-1])


def _write_runtime_payload(payload: dict[str, Any], out: Path) -> dict[str, Any]:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _supercell_validation_summary(source, replicated, replicas) -> dict[str, Any]:
    replica_count = int(np.prod(replicas, dtype=np.int64))
    indexed_names = (
        "bonds",
        "angles",
        "dihedrals",
        "rb_dihedrals",
        "constraints",
        "impropers",
        "nonbonded_pairs",
        "nonbonded_exception_pairs",
        "charmm_cmap_terms",
        "urey_bradley_terms",
        "nbfix_pairs",
        "virtual_site_parent_atoms",
    )
    indexed_count_checks = {
        name: {
            "source": int(np.asarray(getattr(source, name)).shape[0]),
            "actual": int(np.asarray(getattr(replicated, name)).shape[0]),
            "expected": int(np.asarray(getattr(source, name)).shape[0]) * replica_count,
        }
        for name in indexed_names
    }
    source_charge = float(np.sum(source.charges, dtype=np.float64))
    actual_charge = float(np.sum(replicated.charges, dtype=np.float64))
    expected_charge = source_charge * replica_count
    checks = {
        "atom_count": replicated.atom_count == source.atom_count * replica_count,
        "net_charge": bool(np.isclose(actual_charge, expected_charge, rtol=0.0, atol=1e-5)),
        "indexed_term_counts": all(
            item["actual"] == item["expected"] for item in indexed_count_checks.values()
        ),
    }
    return {
        "source_net_charge": source_charge,
        "expected_net_charge": expected_charge,
        "indexed_count_checks": indexed_count_checks,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _parse_replicas(value: str) -> tuple[int, int, int]:
    try:
        return normalize_supercell_replicas(tuple(int(item) for item in value.split(",")))
    except (TypeError, ValueError) as exc:
        msg = "--replicas must be three comma-separated positive integers"
        raise argparse.ArgumentTypeError(msg) from exc


def main(argv: list[str] | None = None) -> None:
    """Run the charged-PME benchmark command-line interface.

    Args:
        argv: Optional argument list; ``None`` reads process arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="replicate a prepared PME system")
    prepare_parser.add_argument("--source", type=Path, required=True)
    prepare_parser.add_argument("--replicas", type=_parse_replicas, required=True)
    prepare_parser.add_argument("--assignment-order", type=int, default=None)
    prepare_parser.add_argument("--background-policy", default=None)
    prepare_parser.add_argument("--out", type=Path, required=True)
    runtime_parser = commands.add_parser("runtime", help="run bounded fixed-cell NVT")
    runtime_parser.add_argument("--prepared", type=Path, required=True)
    runtime_parser.add_argument("--warmups", type=int, default=1)
    runtime_parser.add_argument("--steps", type=int, default=2)
    runtime_parser.add_argument("--dt-ps", type=float, default=0.004)
    runtime_parser.add_argument("--temperature-k", type=float, default=300.0)
    runtime_parser.add_argument("--seed", type=int, default=17)
    runtime_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        payload = prepare_payload(
            source=args.source,
            replicas=args.replicas,
            assignment_order=args.assignment_order,
            background_policy=args.background_policy,
            out=args.out,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        if payload["status"] != "ok":
            raise SystemExit(2)
    elif args.command == "runtime":
        payload = runtime_payload(
            prepared=args.prepared,
            warmups=args.warmups,
            steps=args.steps,
            dt_ps=args.dt_ps,
            temperature_k=args.temperature_k,
            seed=args.seed,
            out=args.out,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not payload["passed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()


__all__ = [
    "RUNTIME_SCHEMA",
    "SUPERCELL_SUMMARY_NAME",
    "main",
    "prepare_payload",
    "runtime_payload",
]
