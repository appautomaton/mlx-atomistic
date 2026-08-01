"""Prepare and measure deterministic charged-PME benchmark workloads."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping
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
RUNTIME_SCHEMA = "mlx_atomistic.charged_pme_runtime.v2"
PROFILE_SCHEMA = "mlx_atomistic.charged_pme_profile.v1"
OPENMM_ADMISSION_SCHEMA = "mlx_atomistic.openmm_runtime_admission.v1"
_PROFILE_STATE_RTOL = 5.0e-5
_PROFILE_STATE_ATOL = 1.0e-5
_PROCESS_MEMORY_LIMIT_BYTES = 40_000_000_000
_SPATIAL_DIRECT_MIN_COMPLETE_SPEEDUP = 0.15
# The isolated direct-force median varies by several percentage points under
# low-power Metal. Keep this as a bounded smell gate; the position-balanced
# complete-wall A/B below is the authoritative retention gate.
_SPATIAL_DIRECT_MIN_KERNEL_SPEEDUP = 0.25
_SPARSE_CORRECTION_MIN_SPEEDUP = 0.20
_RUNTIME_NEIGHBOR_BACKENDS = frozenset({"mlx_cell_pairs", "mlx_cell_tiles"})


def _constraint_route_inventory(
    constraints: object | None,
    *,
    water_atom_count: int,
    molecule_ids_present: bool,
) -> dict[str, Any]:
    """Summarize the runtime constraint routes selected from one artifact."""

    routes = {
        "settle": {"object_count": 0, "cluster_count": 0, "pair_count": 0},
        "shake_clusters": {
            "object_count": 0,
            "cluster_count": 0,
            "pair_count": 0,
            "peripheral_count_histogram": {},
        },
        "generic": {"object_count": 0, "pair_count": 0},
    }

    def visit(constraint: object) -> None:
        children = getattr(constraint, "constraints", None)
        if children is not None:
            for child in children:
                visit(child)
            return
        pair_count = int(constraint.pairs.shape[0])
        family = getattr(constraint, "_profile_family", None)
        if family == "shake_clusters":
            route = routes["shake_clusters"]
            route["object_count"] += 1
            route["cluster_count"] += int(constraint.cluster_count)
            route["pair_count"] += pair_count
            histogram = route["peripheral_count_histogram"]
            for count in np.asarray(constraint.peripheral_counts, dtype=np.int32):
                key = str(int(count))
                histogram[key] = int(histogram.get(key, 0)) + 1
        elif hasattr(constraint, "waters"):
            route = routes["settle"]
            route["object_count"] += 1
            route["cluster_count"] += int(constraint.waters.shape[0])
            route["pair_count"] += pair_count
        else:
            route = routes["generic"]
            route["object_count"] += 1
            route["pair_count"] += pair_count

    if constraints is not None:
        visit(constraints)
    total_pair_count = sum(int(route["pair_count"]) for route in routes.values())
    expected_pair_count = (
        0 if constraints is None else int(constraints.pairs.shape[0])
    )
    water_route_complete = water_atom_count == 0 or (
        molecule_ids_present and int(routes["settle"]["pair_count"]) > 0
    )
    return {
        "routes": routes,
        "total_pair_count": total_pair_count,
        "expected_pair_count": expected_pair_count,
        "pair_count_matches": total_pair_count == expected_pair_count,
        "water_atom_count": int(water_atom_count),
        "molecule_ids_present": bool(molecule_ids_present),
        "performance_eligible": bool(
            total_pair_count == expected_pair_count and water_route_complete
        ),
    }


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
    neighbor_skin: float = 5.5,
    neighbor_check_interval: int = 1,
    sample_interval: int | None = None,
    diagnostic_interval: int | None = None,
    runtime_profile: bool = False,
    neighbor_backend: str = "mlx_cell_tiles",
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
        neighbor_skin: Verlet-list skin in angstrom. Defaults to ``5.5``.
        neighbor_check_interval: Steps between displacement admissions.
            Defaults to ``1``.
        sample_interval: Measured trajectory sampling cadence. ``None`` records
            only the final step. Defaults to ``None``.
        diagnostic_interval: Measured diagnostic cadence. ``None`` evaluates
            only the final step. Defaults to ``None``.
        runtime_profile: Enable synchronized benchmark-only route attribution.
            Defaults to ``False``.
        neighbor_backend: Exact spatial tile candidate or compact-pair control.
            Defaults to ``mlx_cell_tiles``.

    Returns:
        JSON-serializable passing, failed, blocked, or resource-ceiling payload.
    """

    prepared_path = Path(prepared)
    out_path = Path(out)
    sample_interval = steps if sample_interval is None else int(sample_interval)
    diagnostic_interval = (
        steps if diagnostic_interval is None else int(diagnostic_interval)
    )
    base = {
        "kind": RUNTIME_SCHEMA,
        "prepared": str(prepared_path),
        "out": str(out_path),
        "warmup_steps": int(warmups),
        "measured_steps": int(steps),
        "dt_ps": float(dt_ps),
        "temperature_target_k": float(temperature_k),
        "seed": int(seed),
        "neighbor_skin": float(neighbor_skin),
        "neighbor_check_interval": int(neighbor_check_interval),
        "sample_interval": sample_interval,
        "diagnostic_interval": diagnostic_interval,
        "runtime_profile": bool(runtime_profile),
        "neighbor_backend": str(neighbor_backend),
        "process_memory_limit_bytes": _PROCESS_MEMORY_LIMIT_BYTES,
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
    if not np.isfinite(neighbor_skin) or neighbor_skin < 0.0:
        validation_blockers.append("neighbor_skin_must_be_finite_nonnegative")
    if neighbor_check_interval <= 0:
        validation_blockers.append("neighbor_check_interval_must_be_positive")
    if sample_interval <= 0:
        validation_blockers.append("sample_interval_must_be_positive")
    if diagnostic_interval <= 0:
        validation_blockers.append("diagnostic_interval_must_be_positive")
    if neighbor_backend not in _RUNTIME_NEIGHBOR_BACKENDS:
        validation_blockers.append("neighbor_backend_must_be_pairs_or_tiles")
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
            skin=neighbor_skin,
            check_interval=neighbor_check_interval,
            sort_pairs=False,
            backend=neighbor_backend,
            displacement_check_backend="mlx_scalar",
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
                sample_interval=warmups,
                diagnostic_interval=warmups,
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
                sample_interval=sample_interval,
                diagnostic_interval=diagnostic_interval,
                runtime_profile=runtime_profile,
            ),
            constraints=constraints,
            thermostat=LangevinThermostat(
                temperature=temperature_k,
                friction=1.0,
                seed=seed + 1,
            ),
        )
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
        measured_seconds = time.perf_counter() - measured_started
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
        water_mask = np.asarray(
            artifact.arrays.get("water_mask", np.asarray([], dtype=bool)),
            dtype=bool,
        )
        constraint_routes = _constraint_route_inventory(
            constraints,
            water_atom_count=int(np.count_nonzero(water_mask[: artifact.atom_count])),
            molecule_ids_present=artifact.molecule_ids.shape == (artifact.atom_count,),
        )
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
        process_peak_bytes = int(max_rss_mb() * 1024.0 * 1024.0)
        memory = {
            "max_rss_mb": process_peak_bytes / (1024.0 * 1024.0),
            "process_peak_bytes": process_peak_bytes,
            "process_memory_limit_bytes": _PROCESS_MEMORY_LIMIT_BYTES,
            "mlx_active_memory_bytes": _mlx_memory_value("get_active_memory"),
            "mlx_peak_memory_bytes": _mlx_memory_value("get_peak_memory"),
            "mlx_cache_memory_bytes": _mlx_memory_value("get_cache_memory"),
        }
        expected_representation = (
            "tiles" if neighbor_backend == "mlx_cell_tiles" else "pairs"
        )
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
            "constraint_route_pair_count": constraint_routes["pair_count_matches"],
            "performance_eligible_constraint_routes": constraint_routes[
                "performance_eligible"
            ],
            "expected_neighbor_representation": (
                neighbor_report["manager_backend"] == neighbor_backend
                and neighbor_report["representation"] == expected_representation
            ),
            "no_neighbor_fallback": neighbor_report["fallback_reason"] is None,
            "positive_throughput": math.isfinite(ns_per_day) and ns_per_day > 0.0,
            "process_memory_within_limit": (
                process_peak_bytes <= _PROCESS_MEMORY_LIMIT_BYTES
            ),
            "mlx_peak_memory_within_limit": (
                isinstance(memory["mlx_peak_memory_bytes"], int)
                and memory["mlx_peak_memory_bytes"] <= _PROCESS_MEMORY_LIMIT_BYTES
            ),
        }
        if runtime_profile:
            route_profile = measured_result.route_profile
            routes = route_profile.get("routes", {})
            expected_direct_route = (
                "direct_spatial_tiles"
                if neighbor_backend == "mlx_cell_tiles"
                else "direct_lj_screened_coulomb"
            )
            unexpected_direct_route = (
                "direct_lj_screened_coulomb"
                if neighbor_backend == "mlx_cell_tiles"
                else "direct_spatial_tiles"
            )
            checks.update(
                {
                    "route_profile_reconciled": bool(
                        route_profile.get("reconciled", False)
                    ),
                    "route_profile_expected_direct": expected_direct_route in routes,
                    "route_profile_no_direct_fallback": (
                        unexpected_direct_route not in routes
                    ),
                    "route_profile_reciprocal": "reciprocal_pme" in routes,
                    "route_profile_neighbor": "neighbor_update_rebuild" in routes,
                    "route_profile_integration": "integration_thermostat" in routes,
                }
            )
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
            "constraint_routes": constraint_routes,
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
            "memory": memory,
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
            "route_profile": measured_result.route_profile,
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


def audit_openmm_runtime_artifacts(
    *,
    runtime_path: str | Path,
    workload_manifest_path: str | Path,
    manifest_comparison_path: str | Path,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Audit persisted OpenMM evidence without executing a reference engine.

    Args:
        runtime_path: Persisted OpenMM runtime JSON.
        workload_manifest_path: Engine-independent OpenMM workload manifest.
        manifest_comparison_path: Persisted MLX/OpenMM manifest comparison.
        out: Optional admission-report path.

    Returns:
        Admission payload that is fail-closed on missing runtime semantics.
    """

    runtime_path = Path(runtime_path)
    workload_manifest_path = Path(workload_manifest_path)
    manifest_comparison_path = Path(manifest_comparison_path)
    paths = {
        "runtime": runtime_path,
        "workload_manifest": workload_manifest_path,
        "manifest_comparison": manifest_comparison_path,
    }
    payloads: dict[str, Mapping[str, Any]] = {}
    blockers: list[str] = []
    for name, path in paths.items():
        if not path.is_file():
            blockers.append(f"missing_artifact:{name}:{path}")
            continue
        try:
            value = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            blockers.append(f"invalid_artifact:{name}:{type(exc).__name__}:{exc}")
            continue
        if not isinstance(value, Mapping):
            blockers.append(f"invalid_artifact:{name}:root_not_object")
            continue
        payloads[name] = value

    runtime = payloads.get("runtime", {})
    manifest = payloads.get("workload_manifest", {})
    comparison = payloads.get("manifest_comparison", {})
    workload = manifest.get("workload", {})
    workload = workload if isinstance(workload, Mapping) else {}
    manifest_pme = manifest.get("pme", {})
    manifest_pme = manifest_pme if isinstance(manifest_pme, Mapping) else {}
    runtime_pme = runtime.get("pme", {})
    runtime_pme = runtime_pme if isinstance(runtime_pme, Mapping) else {}
    platform = runtime.get("platform_properties", {})
    platform = platform if isinstance(platform, Mapping) else {}

    checks = {
        "manifest_comparison_matched": comparison.get("matched") is True,
        "atom_count_matches": runtime.get("atom_count") == workload.get("atom_count"),
        "replicas_match": runtime.get("replicas") == workload.get("replicas"),
        "measured_steps_present": isinstance(runtime.get("steps"), int),
        "warmups_present": isinstance(runtime.get("warmups"), int),
        "elapsed_seconds_present": (
            isinstance(runtime.get("elapsed_seconds"), int | float)
            and float(runtime.get("elapsed_seconds", 0.0)) > 0.0
        ),
        "single_precision": platform.get("Precision") == "single",
        "hardware_named": bool(platform.get("DeviceName")),
        "pme_mesh_matches": runtime_pme.get("mesh_shape")
        == manifest_pme.get("mesh_shape"),
        "pme_alpha_matches": runtime_pme.get("alpha_per_angstrom")
        == manifest_pme.get("alpha_per_angstrom"),
        "pme_cutoff_matches": runtime_pme.get("cutoff_angstrom")
        == manifest_pme.get("real_cutoff_angstrom"),
        "runtime_operation_declared": runtime.get("operation")
        == "fixed_cell_nvt_step",
        "runtime_manifest_is_nvt": workload.get("operation")
        == "fixed_cell_nvt_step",
        "timestep_declared": isinstance(runtime.get("dt_ps"), int | float),
        "thermostat_declared": isinstance(runtime.get("thermostat"), Mapping),
        "constraints_declared": isinstance(
            runtime.get("constraint_protocol"),
            Mapping,
        ),
        "completion_barrier_declared": runtime.get("completion_barrier")
        == "explicit_device_completion_inside_timer",
        "timing_boundary_declared": isinstance(
            runtime.get("timing_boundary"),
            Mapping,
        ),
        "runtime_binds_workload_manifest": runtime.get("workload_manifest_hash")
        == manifest.get("manifest_hash"),
    }
    blockers.extend(name for name, passed in checks.items() if not passed)
    admission = {
        "kind": OPENMM_ADMISSION_SCHEMA,
        "status": "admitted" if not blockers else "provisional",
        "admitted": not blockers,
        "blockers": blockers,
        "checks": checks,
        "paths": {name: str(path) for name, path in paths.items()},
        "runtime_seconds": runtime.get("elapsed_seconds"),
        "target_seconds_at_ten_times": (
            None
            if not isinstance(runtime.get("elapsed_seconds"), int | float)
            else 10.0 * float(runtime["elapsed_seconds"])
        ),
        "note": (
            "No OpenMM process was executed; this is a static artifact audit."
        ),
    }
    if out is not None:
        _write_runtime_payload(admission, Path(out))
    return admission


def _default_openmm_artifact_paths(
    prepared: Path,
) -> tuple[Path, Path, Path]:
    artifact_root = prepared.parent
    return (
        artifact_root / "openmm_runtime.json",
        artifact_root / "openmm_workload_manifest.json",
        artifact_root / "manifest_comparison.json",
    )


def _profile_tile_inventory(
    prepared: Path,
    *,
    neighbor_skin: float,
    neighbor_check_interval: int,
) -> dict[str, Any]:
    artifact = load_prepared_mlx_artifact(prepared, require_production=True)
    system, force_terms, _ = build_mlx_system_from_artifact(
        artifact,
        eager_nonbonded_pair_limit=0,
    )
    if system.cell is None:
        raise ValueError("tile inventory requires a periodic cell")
    bound_terms = _bind_pme_plans(force_terms, system.cell)
    nonbonded = _find_pme_term(bound_terms)
    tile_manager = NeighborListManager(
        system.cell,
        cutoff=float(nonbonded.cutoff),
        skin=neighbor_skin,
        check_interval=neighbor_check_interval,
        sort_pairs=False,
        backend="mlx_cell_tiles",
        displacement_check_backend="mlx_scalar",
    )
    neighbor_list = tile_manager.update(system.positions)
    tiles = neighbor_list.tiles
    if tiles is None:
        raise RuntimeError("mlx_cell_tiles did not produce NeighborTiles")
    mx.eval(
        tiles.atom_blocks,
        tiles.tile_blocks,
        tiles.member_mask,
        tiles.force_group_starts,
        tiles.force_group_counts,
    )
    pair_manager = NeighborListManager(
        system.cell,
        cutoff=float(nonbonded.cutoff),
        skin=neighbor_skin,
        check_interval=neighbor_check_interval,
        sort_pairs=False,
        backend="mlx_cell_pairs",
        displacement_check_backend="mlx_scalar",
    )
    pair_neighbors = pair_manager.update(system.positions)
    reference_pairs = pair_neighbors.pairs
    if reference_pairs is None:
        raise RuntimeError("mlx_cell_pairs did not produce compact pairs")
    reference_pair_count = int(reference_pairs.shape[0])
    tile_binding = nonbonded._prepare_tile_force_binding(
        system.cell,
        neighbor_list.diagnostic_pairs,
        tiles,
    )
    pair_binding = nonbonded._prepare_force_binding(
        system.cell,
        pair_neighbors.diagnostic_pairs,
    )
    if tile_binding is NotImplemented or pair_binding is NotImplemented:
        raise RuntimeError("JAC direct-force bindings were not admitted")
    if tile_binding.tile_decline_reason is not None:
        raise RuntimeError(
            f"spatial tile force route declined: {tile_binding.tile_decline_reason}"
        )
    tile_forces = nonbonded._direct_forces_from_binding(
        system.positions,
        tile_binding,
    )
    pair_forces = nonbonded._direct_forces_from_binding(
        system.positions,
        pair_binding,
    )
    mx.eval(tile_forces, pair_forces)
    force_delta = np.asarray(tile_forces) - np.asarray(pair_forces)
    force_rms_delta = float(np.sqrt(np.mean(force_delta * force_delta)))
    force_max_delta = float(np.max(np.abs(force_delta)))
    force_max_reference = float(np.max(np.abs(np.asarray(pair_forces))))

    reference_corrections = nonbonded._exception_lj_forces(
        system.positions,
        system.cell,
    ) + nonbonded._periodic_coulomb_correction_forces(
        system.positions,
        system.cell,
    )
    fused_corrections = nonbonded._prepared_sparse_correction_forces(
        system.positions,
        tile_binding,
    )
    mx.eval(reference_corrections, fused_corrections)
    correction_delta = np.asarray(fused_corrections) - np.asarray(reference_corrections)
    correction_rms_delta = float(np.sqrt(np.mean(correction_delta * correction_delta)))
    correction_max_delta = float(np.max(np.abs(correction_delta)))

    def interleaved_medians(reference_call, candidate_call) -> tuple[float, float]:
        calls = {"reference": reference_call, "candidate": candidate_call}
        for _ in range(2):
            for call in calls.values():
                value = call()
                mx.eval(value)
        samples = {"reference": [], "candidate": []}
        for sample in range(6):
            ordered = (
                ("reference", "candidate")
                if sample % 2 == 0
                else ("candidate", "reference")
            )
            for name in ordered:
                started = time.perf_counter()
                value = calls[name]()
                mx.eval(value)
                samples[name].append(time.perf_counter() - started)
        return (
            float(np.median(samples["reference"])),
            float(np.median(samples["candidate"])),
        )

    pair_direct_seconds, tile_direct_seconds = interleaved_medians(
        lambda: nonbonded._direct_forces_from_binding(system.positions, pair_binding),
        lambda: nonbonded._direct_forces_from_binding(system.positions, tile_binding),
    )
    reference_correction_seconds, fused_correction_seconds = interleaved_medians(
        lambda: nonbonded._exception_lj_forces(
            system.positions,
            system.cell,
        )
        + nonbonded._periodic_coulomb_correction_forces(
            system.positions,
            system.cell,
        ),
        lambda: nonbonded._prepared_sparse_correction_forces(
            system.positions,
            tile_binding,
        ),
    )
    diagnostic_pair_bytes = int(neighbor_list.diagnostic_pairs.size) * 4
    tile_topology_bytes = sum(
        int(values.size) * 4
        for values in (
            tile_binding.tile_lj_enabled_mask,
            tile_binding.tile_lj_one_four_mask,
        )
        if values is not None
    )
    pair_scale_bytes = (
        0
        if tile_binding.aligned_lj_scales is None
        else int(tile_binding.aligned_lj_scales.size) * 4
    )
    persistent_breakdown = {
        "tile_geometry_bytes": tiles.estimated_bytes,
        "diagnostic_pair_bytes": diagnostic_pair_bytes,
        "tile_topology_mask_bytes": tile_topology_bytes,
        "pair_scale_bytes": pair_scale_bytes,
    }
    return {
        "backend": tile_manager.backend,
        "block_size": tiles.block_size,
        "block_count": tiles.block_count,
        "tile_count": tiles.tile_count,
        "force_group_count": tiles.force_group_count,
        "exact_pair_count": tiles.exact_pair_count,
        "reference_pair_count": reference_pair_count,
        "pair_inventory_matches": tiles.exact_pair_count == reference_pair_count,
        "raw_candidate_count": tiles.raw_candidate_count,
        "padded_lane_count": tiles.padded_lane_count,
        "padding_waste_count": tiles.padding_waste_count,
        "padding_waste_fraction": tiles.padding_waste_fraction,
        "estimated_persistent_bytes": sum(persistent_breakdown.values()),
        "estimated_persistent_bytes_breakdown": persistent_breakdown,
        "active_lane_fraction": (
            0.0
            if tiles.padded_lane_count == 0
            else tiles.exact_pair_count / tiles.padded_lane_count
        ),
        "force_rms_delta_kj_mol_angstrom": force_rms_delta,
        "force_max_delta_kj_mol_angstrom": force_max_delta,
        "force_max_reference_kj_mol_angstrom": force_max_reference,
        "correction_force_rms_delta_kj_mol_angstrom": correction_rms_delta,
        "correction_force_max_delta_kj_mol_angstrom": correction_max_delta,
        "reference_correction_median_seconds": reference_correction_seconds,
        "fused_correction_median_seconds": fused_correction_seconds,
        "correction_speedup_fraction": (
            0.0
            if reference_correction_seconds <= 0.0
            else 1.0 - fused_correction_seconds / reference_correction_seconds
        ),
        "pair_direct_median_seconds": pair_direct_seconds,
        "tile_direct_median_seconds": tile_direct_seconds,
        "direct_speedup_fraction": (
            0.0
            if pair_direct_seconds <= 0.0
            else 1.0 - tile_direct_seconds / pair_direct_seconds
        ),
        "rebuild_wall_seconds": tile_manager.rebuild_wall_seconds,
        "update_wall_seconds": tile_manager.update_wall_seconds,
        "reference_pair_rebuild_wall_seconds": pair_manager.rebuild_wall_seconds,
    }


def profile_payload(
    *,
    prepared: str | Path,
    warmups: int,
    steps: int,
    seed: int,
    out: str | Path,
    dt_ps: float = 0.004,
    temperature_k: float = 300.0,
    neighbor_skin: float = 5.5,
    neighbor_check_interval: int = 1,
    openmm_runtime: str | Path | None = None,
    openmm_manifest: str | Path | None = None,
    manifest_comparison: str | Path | None = None,
) -> dict[str, Any]:
    """Run a matched pair/tile A/B and synchronized charged-PME route profile.

    Args:
        prepared: Strict production prepared-system directory.
        warmups: Untimed warmup count for each sample.
        steps: Measured step count for each sample.
        seed: Shared deterministic thermostat seed.
        out: Combined profile JSON output path.
        dt_ps: Timestep in picoseconds. Defaults to ``0.004``.
        temperature_k: Langevin target temperature. Defaults to ``300``.
        neighbor_skin: Verlet-list skin in angstrom. Defaults to ``5.5``.
        neighbor_check_interval: Displacement admission cadence. Defaults to ``1``.
        openmm_runtime: Optional persisted OpenMM runtime JSON override.
        openmm_manifest: Optional persisted OpenMM workload manifest override.
        manifest_comparison: Optional persisted manifest comparison override.

    Returns:
        Combined position-balanced pair/tile controls, instrumented tile,
        tile-inventory, and admission evidence.
    """

    prepared_path = Path(prepared)
    out_path = Path(out)
    pair_before_path = out_path.parent / "pair-control-before.json"
    clean_path = out_path.parent / "clean.json"
    clean_repeat_path = out_path.parent / "clean-repeat.json"
    pair_after_path = out_path.parent / "pair-control-after.json"
    instrumented_path = out_path.parent / "instrumented.json"
    admission_path = out_path.parent / "openmm-admission.json"
    default_runtime, default_manifest, default_comparison = (
        _default_openmm_artifact_paths(prepared_path)
    )
    admission = audit_openmm_runtime_artifacts(
        runtime_path=default_runtime if openmm_runtime is None else openmm_runtime,
        workload_manifest_path=(
            default_manifest if openmm_manifest is None else openmm_manifest
        ),
        manifest_comparison_path=(
            default_comparison
            if manifest_comparison is None
            else manifest_comparison
        ),
        out=admission_path,
    )
    pair_control_before = runtime_payload(
        prepared=prepared_path,
        warmups=warmups,
        steps=steps,
        out=pair_before_path,
        dt_ps=dt_ps,
        temperature_k=temperature_k,
        seed=seed,
        neighbor_skin=neighbor_skin,
        neighbor_check_interval=neighbor_check_interval,
        runtime_profile=False,
        neighbor_backend="mlx_cell_pairs",
    )
    clean = runtime_payload(
        prepared=prepared_path,
        warmups=warmups,
        steps=steps,
        out=clean_path,
        dt_ps=dt_ps,
        temperature_k=temperature_k,
        seed=seed,
        neighbor_skin=neighbor_skin,
        neighbor_check_interval=neighbor_check_interval,
        runtime_profile=False,
        neighbor_backend="mlx_cell_tiles",
    )
    clean_repeat = runtime_payload(
        prepared=prepared_path,
        warmups=warmups,
        steps=steps,
        out=clean_repeat_path,
        dt_ps=dt_ps,
        temperature_k=temperature_k,
        seed=seed,
        neighbor_skin=neighbor_skin,
        neighbor_check_interval=neighbor_check_interval,
        runtime_profile=False,
        neighbor_backend="mlx_cell_tiles",
    )
    pair_control_after = runtime_payload(
        prepared=prepared_path,
        warmups=warmups,
        steps=steps,
        out=pair_after_path,
        dt_ps=dt_ps,
        temperature_k=temperature_k,
        seed=seed,
        neighbor_skin=neighbor_skin,
        neighbor_check_interval=neighbor_check_interval,
        runtime_profile=False,
        neighbor_backend="mlx_cell_pairs",
    )
    instrumented = runtime_payload(
        prepared=prepared_path,
        warmups=warmups,
        steps=steps,
        out=instrumented_path,
        dt_ps=dt_ps,
        temperature_k=temperature_k,
        seed=seed,
        neighbor_skin=neighbor_skin,
        neighbor_check_interval=neighbor_check_interval,
        runtime_profile=True,
        neighbor_backend="mlx_cell_tiles",
    )
    tile_inventory: dict[str, Any] | None = None
    inventory_blocker = None
    if (
        pair_control_before.get("passed")
        and clean.get("passed")
        and clean_repeat.get("passed")
        and pair_control_after.get("passed")
        and instrumented.get("passed")
    ):
        try:
            tile_inventory = _profile_tile_inventory(
                prepared_path,
                neighbor_skin=neighbor_skin,
                neighbor_check_interval=neighbor_check_interval,
            )
        except (MemoryError, RuntimeError, TypeError, ValueError) as exc:
            inventory_blocker = f"{type(exc).__name__}:{exc}"

    route_profile = instrumented.get("route_profile", {})
    routes = route_profile.get("routes", {}) if isinstance(route_profile, Mapping) else {}
    state_fields = (
        "potential_energy_kj_mol",
        "kinetic_energy_kj_mol",
        "total_energy_kj_mol",
        "temperature_k",
        "constraint_max_error_angstrom",
    )
    def states_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return all(
            key in left
            and key in right
            and np.isclose(
                float(left[key]),
                float(right[key]),
                rtol=_PROFILE_STATE_RTOL,
                atol=_PROFILE_STATE_ATOL,
            )
            for key in state_fields
        )

    clean_samples = (clean, clean_repeat)
    clean_states = [sample.get("state", {}) for sample in clean_samples]
    instrumented_state = instrumented.get("state", {})
    clean_instrumented_state_consistent = all(
        states_match(state, instrumented_state) for state in clean_states
    )
    control_candidate_state_consistent = all(
        states_match(control.get("state", {}), candidate_state)
        for control in (pair_control_before, pair_control_after)
        for candidate_state in clean_states
    )
    pair_control_seconds = [
        float(control.get("timings", {}).get("measured_seconds", math.inf))
        for control in (pair_control_before, pair_control_after)
    ]
    pair_control_median_seconds = (
        float(np.median(pair_control_seconds))
        if all(math.isfinite(value) and value > 0.0 for value in pair_control_seconds)
        else math.inf
    )
    tile_candidate_seconds = [
        float(sample.get("timings", {}).get("measured_seconds", math.inf))
        for sample in clean_samples
    ]
    tile_candidate_median_seconds = (
        float(np.median(tile_candidate_seconds))
        if all(math.isfinite(value) and value > 0.0 for value in tile_candidate_seconds)
        else math.inf
    )
    complete_wall_speedup_fraction = (
        0.0
        if not math.isfinite(pair_control_median_seconds)
        or pair_control_median_seconds <= 0.0
        or not math.isfinite(tile_candidate_median_seconds)
        else 1.0 - tile_candidate_median_seconds / pair_control_median_seconds
    )
    profile_process_peak_bytes = int(max_rss_mb() * 1024.0 * 1024.0)
    profile_memory = {
        "max_rss_mb": profile_process_peak_bytes / (1024.0 * 1024.0),
        "process_peak_bytes": profile_process_peak_bytes,
        "process_memory_limit_bytes": _PROCESS_MEMORY_LIMIT_BYTES,
        "mlx_active_memory_bytes": _mlx_memory_value("get_active_memory"),
        "mlx_peak_memory_bytes": _mlx_memory_value("get_peak_memory"),
        "mlx_cache_memory_bytes": _mlx_memory_value("get_cache_memory"),
    }
    checks = {
        "pair_control_before_passed": pair_control_before.get("passed") is True,
        "clean_passed": clean.get("passed") is True,
        "clean_repeat_passed": clean_repeat.get("passed") is True,
        "pair_control_after_passed": pair_control_after.get("passed") is True,
        "instrumented_passed": instrumented.get("passed") is True,
        "complete_wall_speedup": (
            complete_wall_speedup_fraction >= _SPATIAL_DIRECT_MIN_COMPLETE_SPEEDUP
        ),
        "route_profile_reconciled": route_profile.get("reconciled") is True,
        "spatial_tile_route_present": "direct_spatial_tiles" in routes,
        "compact_pair_fallback_absent": "direct_lj_screened_coulomb" not in routes,
        "reciprocal_route_present": "reciprocal_pme" in routes,
        "neighbor_route_present": "neighbor_update_rebuild" in routes,
        "integration_route_present": "integration_thermostat" in routes,
        "clean_instrumented_state_consistent": clean_instrumented_state_consistent,
        "control_candidate_state_consistent": control_candidate_state_consistent,
        "process_memory_within_limit": (
            profile_process_peak_bytes <= _PROCESS_MEMORY_LIMIT_BYTES
        ),
        "mlx_peak_memory_within_limit": (
            isinstance(profile_memory["mlx_peak_memory_bytes"], int)
            and profile_memory["mlx_peak_memory_bytes"]
            <= _PROCESS_MEMORY_LIMIT_BYTES
        ),
        "tile_inventory_present": tile_inventory is not None,
        "tile_pair_inventory_matches": (
            tile_inventory is not None
            and tile_inventory.get("pair_inventory_matches") is True
        ),
        "tile_force_rms_within_tolerance": (
            tile_inventory is not None
            and float(tile_inventory.get("force_rms_delta_kj_mol_angstrom", math.inf))
            <= 2.0e-4
        ),
        "tile_force_max_within_tolerance": (
            tile_inventory is not None
            and float(tile_inventory.get("force_max_delta_kj_mol_angstrom", math.inf))
            <= 2.0e-2
        ),
        "tile_active_lane_fraction": (
            tile_inventory is not None
            and float(tile_inventory.get("active_lane_fraction", 0.0)) >= 0.30
        ),
        "tile_scheduled_lane_bound": (
            tile_inventory is not None
            and int(tile_inventory.get("padded_lane_count", 200_000_001))
            <= 200_000_000
        ),
        "tile_direct_latency_speedup": (
            tile_inventory is not None
            and float(tile_inventory.get("direct_speedup_fraction", 0.0))
            >= _SPATIAL_DIRECT_MIN_KERNEL_SPEEDUP
        ),
        "tile_persistent_estimate_within_limit": (
            tile_inventory is not None
            and int(
                tile_inventory.get(
                    "estimated_persistent_bytes",
                    _PROCESS_MEMORY_LIMIT_BYTES + 1,
                )
            )
            <= _PROCESS_MEMORY_LIMIT_BYTES
        ),
        "correction_force_rms_within_tolerance": (
            tile_inventory is not None
            and float(
                tile_inventory.get(
                    "correction_force_rms_delta_kj_mol_angstrom",
                    math.inf,
                )
            )
            <= 2.0e-4
        ),
        "correction_force_max_within_tolerance": (
            tile_inventory is not None
            and float(
                tile_inventory.get(
                    "correction_force_max_delta_kj_mol_angstrom",
                    math.inf,
                )
            )
            <= 2.0e-2
        ),
        "correction_latency_speedup": (
            tile_inventory is not None
            and float(tile_inventory.get("correction_speedup_fraction", 0.0))
            >= _SPARSE_CORRECTION_MIN_SPEEDUP
        ),
    }
    diagnostics = {
        "pair_control_seconds": pair_control_seconds,
        "pair_control_median_seconds": pair_control_median_seconds,
        "tile_candidate_seconds": tile_candidate_seconds,
        "tile_candidate_median_seconds": tile_candidate_median_seconds,
        "complete_wall_speedup_fraction": complete_wall_speedup_fraction,
        "complete_wall_speedup_gate": _SPATIAL_DIRECT_MIN_COMPLETE_SPEEDUP,
        "tile_pair_inventory_matches": (
            tile_inventory is not None
            and tile_inventory.get("pair_inventory_matches") is True
        ),
        "tile_pair_inventory_delta": (
            None
            if tile_inventory is None
            else int(tile_inventory["exact_pair_count"])
            - int(tile_inventory["reference_pair_count"])
        ),
        "tile_force_rms_delta_kj_mol_angstrom": (
            None
            if tile_inventory is None
            else tile_inventory["force_rms_delta_kj_mol_angstrom"]
        ),
        "tile_force_max_delta_kj_mol_angstrom": (
            None
            if tile_inventory is None
            else tile_inventory["force_max_delta_kj_mol_angstrom"]
        ),
        "tile_direct_speedup_fraction": (
            None
            if tile_inventory is None
            else tile_inventory["direct_speedup_fraction"]
        ),
        "tile_direct_speedup_gate": _SPATIAL_DIRECT_MIN_KERNEL_SPEEDUP,
        "correction_force_rms_delta_kj_mol_angstrom": (
            None
            if tile_inventory is None
            else tile_inventory["correction_force_rms_delta_kj_mol_angstrom"]
        ),
        "correction_force_max_delta_kj_mol_angstrom": (
            None
            if tile_inventory is None
            else tile_inventory["correction_force_max_delta_kj_mol_angstrom"]
        ),
        "correction_speedup_fraction": (
            None
            if tile_inventory is None
            else tile_inventory["correction_speedup_fraction"]
        ),
        "correction_speedup_gate": _SPARSE_CORRECTION_MIN_SPEEDUP,
    }
    passed = all(checks.values())
    payload = {
        "kind": PROFILE_SCHEMA,
        "status": "ok" if passed else "failed",
        "passed": passed,
        "blockers": [name for name, value in checks.items() if not value]
        + ([] if inventory_blocker is None else [f"tile_inventory:{inventory_blocker}"]),
        "prepared": str(prepared_path),
        "pair_controls": [pair_control_before, pair_control_after],
        "clean": clean,
        "clean_samples": [clean, clean_repeat],
        "instrumented": instrumented,
        "tile_inventory": tile_inventory,
        "openmm_admission": admission,
        "checks": checks,
        "diagnostics": diagnostics,
        "memory": profile_memory,
    }
    return _write_runtime_payload(payload, out_path)


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
    sample_interval: int,
    diagnostic_interval: int,
    runtime_profile: bool = False,
) -> SimulationConfig:
    return SimulationConfig(
        dt=dt_ps,
        steps=steps,
        sample_interval=sample_interval,
        diagnostic_interval=diagnostic_interval,
        pressure_diagnostics=False,
        compile_force_evaluator=False,
        runtime_profile=runtime_profile,
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
    runtime_parser.add_argument("--neighbor-skin", type=float, default=5.5)
    runtime_parser.add_argument("--neighbor-check-interval", type=int, default=1)
    runtime_parser.add_argument("--sample-interval", type=int, default=None)
    runtime_parser.add_argument("--diagnostic-interval", type=int, default=None)
    runtime_parser.add_argument(
        "--neighbor-backend",
        choices=sorted(_RUNTIME_NEIGHBOR_BACKENDS),
        default="mlx_cell_tiles",
    )
    runtime_parser.add_argument("--out", type=Path, required=True)
    profile_parser = commands.add_parser(
        "profile",
        help="run clean and synchronized fixed-cell NVT profiles",
    )
    profile_parser.add_argument("--prepared", type=Path, required=True)
    profile_parser.add_argument("--warmups", type=int, default=10)
    profile_parser.add_argument("--steps", type=int, default=75)
    profile_parser.add_argument("--dt-ps", type=float, default=0.004)
    profile_parser.add_argument("--temperature-k", type=float, default=300.0)
    profile_parser.add_argument("--seed", type=int, default=17)
    profile_parser.add_argument("--neighbor-skin", type=float, default=5.5)
    profile_parser.add_argument("--neighbor-check-interval", type=int, default=1)
    profile_parser.add_argument("--openmm-runtime", type=Path, default=None)
    profile_parser.add_argument("--openmm-manifest", type=Path, default=None)
    profile_parser.add_argument("--manifest-comparison", type=Path, default=None)
    profile_parser.add_argument("--out", type=Path, required=True)
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
            neighbor_skin=args.neighbor_skin,
            neighbor_check_interval=args.neighbor_check_interval,
            sample_interval=args.sample_interval,
            diagnostic_interval=args.diagnostic_interval,
            neighbor_backend=args.neighbor_backend,
            out=args.out,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not payload["passed"]:
            raise SystemExit(2)
    elif args.command == "profile":
        payload = profile_payload(
            prepared=args.prepared,
            warmups=args.warmups,
            steps=args.steps,
            dt_ps=args.dt_ps,
            temperature_k=args.temperature_k,
            seed=args.seed,
            neighbor_skin=args.neighbor_skin,
            neighbor_check_interval=args.neighbor_check_interval,
            openmm_runtime=args.openmm_runtime,
            openmm_manifest=args.openmm_manifest,
            manifest_comparison=args.manifest_comparison,
            out=args.out,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not payload["passed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()


__all__ = [
    "OPENMM_ADMISSION_SCHEMA",
    "PROFILE_SCHEMA",
    "RUNTIME_SCHEMA",
    "SUPERCELL_SUMMARY_NAME",
    "audit_openmm_runtime_artifacts",
    "main",
    "prepare_payload",
    "profile_payload",
    "runtime_payload",
]
