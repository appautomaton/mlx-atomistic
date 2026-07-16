"""Profile PME path costs for a parity-admitted prepared MLX fixture."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import mlx.core as mx
import numpy as np

from mlx_atomistic.artifacts import build_mlx_system_from_artifact, load_prepared_mlx_artifact
from mlx_atomistic.benchmarks import (
    default_benchmark_command,
    get_hardware_info,
    normalize_benchmark_payload,
    normalize_benchmark_row,
)
from mlx_atomistic.benchmarks.gpcrmd_runtime import max_rss_mb
from mlx_atomistic.benchmarks.pme_validation import manifest_hash
from mlx_atomistic.forcefields import NonbondedPotential
from mlx_atomistic.neighbors import NeighborBlocks, build_neighbor_list
from mlx_atomistic.pme import (
    _assign_charges_bspline_mx,
    _influence_function_mx,
    _interpolate_bspline_mx,
    _mesh_reciprocal_energy_forces_mx,
    _real_space_energy_forces_mx,
    _validate_inputs_mx,
    pme_coulomb_direct_space_energy_forces,
    pme_coulomb_energy_forces,
    pme_direct_space_policy_report,
)
from mlx_atomistic.runtime import get_runtime_info

DEFAULT_OUTPUT_DIR = Path("outputs/benchmarks/pme-profile")
MISSING_FIXTURE_LABEL = Path("user-provided-pme-fixture")
SYNC_TIMING_BLOCKER = (
    "PME exposes stage-level mx.eval barriers in the profiler, but exact "
    "in-function synchronization attribution requires runtime instrumentation "
    "outside pme.py"
)
CHARGED_PARITY_REPORT_NAME = "charged_pme_parity_report.json"
GPCRMD_PARITY_REPORT_NAME = "gpcrmd_pme_parity_report.json"
LEGACY_PARITY_REPORT_NAME = "openmm_mlx_parity_report.json"
GPCRMD_WORKLOAD_MANIFEST_NAME = "mlx-workload-manifest.json"
_PME_CONFIG_FIELDS = (
    "mesh_shape",
    "alpha",
    "real_cutoff",
    "assignment_order",
    "charge_tolerance",
    "deconvolve_assignment",
    "background_policy",
)


@dataclass(frozen=True)
class TimingRow:
    """One PME timing split."""

    name: str
    category: str
    iterations: int
    warmups: int
    mean_s: float
    median_s: float
    min_s: float
    max_s: float

    def to_dict(self) -> dict:
        return asdict(self)


def _time(
    name: str,
    category: str,
    fn: Callable[[], object],
    *,
    eval_outputs: Callable[[object], None],
    warmups: int,
    iterations: int,
) -> TimingRow:
    for _ in range(warmups):
        eval_outputs(fn())

    samples = []
    for _ in range(iterations):
        start = perf_counter()
        eval_outputs(fn())
        samples.append(perf_counter() - start)
    return TimingRow(
        name=name,
        category=category,
        iterations=iterations,
        warmups=warmups,
        mean_s=float(mean(samples)),
        median_s=float(median(samples)),
        min_s=float(min(samples)),
        max_s=float(max(samples)),
    )


def _eval_all(value: object) -> None:
    if isinstance(value, tuple | list):
        mx.eval(*value)
    else:
        mx.eval(value)


def _load_parity_report(report_path: Path) -> dict:
    with report_path.open() as handle:
        report = json.load(handle)
    kind = report.get("kind")
    if kind in {
        "mlx_atomistic.charged_pme_parity",
        "mlx_atomistic.gpcrmd_pme_parity",
    }:
        mlx = dict(report.get("mlx", {}))
        force_metrics = dict(report.get("force_metrics", {}))
        is_gpcrmd = kind == "mlx_atomistic.gpcrmd_pme_parity"
        raw_pme_config = (
            dict(mlx.get("pme_readiness", {}))
            if is_gpcrmd
            else dict(report.get("pme", {}))
        )
        return {
            "report_path": str(report_path),
            "kind": kind,
            "schema": (
                "gpcrmd_pme_parity_v1" if is_gpcrmd else "charged_pme_parity_v1"
            ),
            "status": report.get("status"),
            "passed": bool(report.get("passed", False)),
            "fixture": report.get("fixture"),
            "atom_count": report.get("atom_count"),
            "openmm_nonbonded_method": "PME",
            "total_energy_abs_error_kj_mol": (
                None
                if report.get("atom_count") is None
                else force_metrics.get("energy_error_per_atom_kj_mol", 0.0)
                * int(report["atom_count"])
            ),
            "force_max_abs_error_kj_mol_nm": force_metrics.get(
                "maximum_absolute_kj_mol_nm"
            ),
            "force_rms_abs_error_kj_mol_nm": force_metrics.get(
                "rms_absolute_kj_mol_nm"
            ),
            "pme_readiness": mlx.get("pme_readiness"),
            "pme_config": _canonical_pme_config(raw_pme_config),
            "prepared_dir": (
                report.get("mlx_prepared")
                if is_gpcrmd
                else report.get("normalized_prepared")
            ),
            "manifest_comparison": report.get("manifest_comparison"),
            "mlx_manifest_path": dict(report.get("manifests", {})).get("mlx"),
            "mlx_topology": mlx.get("topology"),
            "mlx_neighbor": mlx.get("neighbor"),
        }
    return {
        "report_path": str(report_path),
        "kind": report.get("kind"),
        "schema": "openmm_mlx_parity_v1",
        "status": report.get("status"),
        "passed": bool(report.get("passed", False)),
        "fixture": report.get("fixture"),
        "atom_count": report.get("atom_count"),
        "openmm_nonbonded_method": report.get("openmm_nonbonded_method"),
        "total_energy_abs_error_kj_mol": report.get("total_energy_abs_error_kj_mol"),
        "force_max_abs_error_kj_mol_nm": report.get("force_max_abs_error_kj_mol_nm"),
        "force_rms_abs_error_kj_mol_nm": report.get("force_rms_abs_error_kj_mol_nm"),
        "pme_readiness": report.get("pme_readiness"),
        "pme_config": report.get("pme_config"),
        "prepared_dir": report.get("prepared_dir"),
        "manifest_comparison": report.get("manifest_comparison"),
    }


def _resolve_fixture_paths(fixture_dir: Path) -> tuple[Path, Path]:
    report_candidates = (
        fixture_dir / GPCRMD_PARITY_REPORT_NAME,
        fixture_dir / CHARGED_PARITY_REPORT_NAME,
        fixture_dir / LEGACY_PARITY_REPORT_NAME,
    )
    prepared_candidates = (
        fixture_dir / "prepared",
        fixture_dir / "mlx-prepared-normalized",
    )
    report_path = next(
        (path for path in report_candidates if path.is_file()),
        report_candidates[-1],
    )
    prepared_dir = next(
        (path for path in prepared_candidates if path.is_dir()),
        prepared_candidates[0],
    )
    return report_path, prepared_dir


def _resolve_profile_paths(
    *,
    fixture_dir: Path | None,
    parity_report: Path | None,
    prepared: Path | None,
) -> tuple[Path, Path, Path]:
    fixture_label = fixture_dir
    discovered_report = None
    discovered_prepared = None
    if fixture_dir is not None:
        discovered_report, discovered_prepared = _resolve_fixture_paths(fixture_dir)

    report_path = parity_report or discovered_report
    prepared_dir = prepared or discovered_prepared
    if fixture_label is None:
        if report_path is not None:
            fixture_label = report_path.parent
        elif prepared_dir is not None:
            fixture_label = prepared_dir
        else:
            fixture_label = MISSING_FIXTURE_LABEL
    if report_path is None:
        report_path = fixture_label / LEGACY_PARITY_REPORT_NAME
    if prepared_dir is None:
        prepared_dir = fixture_label / "prepared"
    return fixture_label, report_path, prepared_dir


def _canonical_pme_config(value: Mapping[str, object]) -> dict[str, object]:
    aliases = {
        "mesh_shape": ("mesh_shape",),
        "alpha": ("alpha", "alpha_per_angstrom"),
        "real_cutoff": ("real_cutoff", "real_cutoff_angstrom"),
        "assignment_order": ("assignment_order",),
        "charge_tolerance": ("charge_tolerance", "charge_tolerance_e"),
        "deconvolve_assignment": ("deconvolve_assignment",),
        "background_policy": ("background_policy",),
    }
    normalized: dict[str, object] = {}
    for field, names in aliases.items():
        for name in names:
            if name in value:
                normalized[field] = value[name]
                break
    return normalized


def _runtime_pme_config(nonbonded: NonbondedPotential) -> dict[str, object]:
    config = nonbonded.pme_config
    if config is None:
        return {}
    return {
        "mesh_shape": list(config.mesh_shape),
        "alpha": float(config.alpha),
        "real_cutoff": (
            None if config.real_cutoff is None else float(config.real_cutoff)
        ),
        "assignment_order": int(config.assignment_order),
        "charge_tolerance": float(config.charge_tolerance),
        "deconvolve_assignment": bool(config.deconvolve_assignment),
        "background_policy": config.background_policy,
    }


def _pme_config_mismatches(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    mismatches: dict[str, dict[str, object]] = {}
    for field, expected_value in expected.items():
        if field not in _PME_CONFIG_FIELDS:
            continue
        actual_value = actual.get(field)
        matches = False
        if field == "mesh_shape":
            try:
                matches = tuple(int(item) for item in expected_value) == tuple(
                    int(item) for item in actual_value
                )
            except (TypeError, ValueError):
                matches = False
        elif field in {"alpha", "real_cutoff", "charge_tolerance"}:
            try:
                matches = bool(
                    np.isclose(
                        float(expected_value),
                        float(actual_value),
                        rtol=1e-6,
                        atol=1e-7,
                    )
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = expected_value == actual_value
        if not matches:
            mismatches[field] = {
                "expected": expected_value,
                "actual": actual_value,
            }
    return mismatches


def _gpcrmd_manifest_admission(
    *,
    prepared_dir: Path,
    parity: Mapping[str, object],
    artifact_atom_count: int,
    runtime_pme_config: Mapping[str, object],
) -> tuple[list[str], dict[str, object]]:
    manifest_path = prepared_dir / GPCRMD_WORKLOAD_MANIFEST_NAME
    diagnostics: dict[str, object] = {
        "path": str(manifest_path),
        "present": manifest_path.is_file(),
        "integrity": False,
        "pme_matches_runtime": False,
        "atom_count_matches": False,
        "fixture_matches": False,
        "runtime_contract_matches": False,
    }
    if not manifest_path.is_file():
        return [f"prepared_workload_manifest:missing:{manifest_path}"], diagnostics
    try:
        payload = json.loads(manifest_path.read_text())
        if not isinstance(payload, dict):
            msg = "root must be an object"
            raise ValueError(msg)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        return [f"prepared_workload_manifest:invalid:{exc}"], diagnostics

    blockers: list[str] = []
    declared_hash = payload.get("manifest_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    computed_hash = manifest_hash(unsigned)
    diagnostics["declared_manifest_sha256"] = declared_hash
    diagnostics["computed_manifest_sha256"] = computed_hash
    diagnostics["integrity"] = bool(declared_hash == computed_hash)
    if declared_hash != computed_hash:
        blockers.append(
            "prepared_workload_manifest:integrity_mismatch:"
            f"declared={declared_hash}:computed={computed_hash}"
        )
    if payload.get("kind") != "gpcrmd_mlx_workload":
        blockers.append(
            "prepared_workload_manifest:kind_mismatch:"
            f"actual={payload.get('kind')}"
        )

    workload = payload.get("workload")
    if not isinstance(workload, Mapping):
        blockers.append("prepared_workload_manifest:workload_missing")
        workload = {}
    manifest_atom_count = workload.get("atom_count")
    diagnostics["manifest_atom_count"] = manifest_atom_count
    diagnostics["atom_count_matches"] = manifest_atom_count == artifact_atom_count
    if manifest_atom_count != artifact_atom_count:
        blockers.append(
            "prepared_workload_manifest:atom_count_mismatch:"
            f"manifest={manifest_atom_count}:artifact={artifact_atom_count}"
        )
    fixture = parity.get("fixture")
    manifest_fixture = workload.get("name")
    diagnostics["manifest_fixture"] = manifest_fixture
    diagnostics["fixture_matches"] = manifest_fixture == fixture
    if manifest_fixture != fixture:
        blockers.append(
            "prepared_workload_manifest:fixture_mismatch:"
            f"manifest={manifest_fixture}:parity={fixture}"
        )

    expected_runtime_contract = {
        "topology_pair_policy": "lazy",
        "eager_nonbonded_pair_limit": 0,
        "neighbor_backend": "mlx_cell_blocks",
        "neighbor_representation": "NeighborBlocks",
        "fixed_cell_pme_plan_reuse": True,
        "dense_or_tiled_fallback_allowed": False,
    }
    runtime_contract = payload.get("runtime_contract")
    if not isinstance(runtime_contract, Mapping):
        runtime_contract = {}
    contract_mismatches = {
        key: {
            "expected": expected,
            "actual": runtime_contract.get(key),
        }
        for key, expected in expected_runtime_contract.items()
        if runtime_contract.get(key) != expected
    }
    diagnostics["runtime_contract_mismatches"] = contract_mismatches
    diagnostics["runtime_contract_matches"] = not contract_mismatches
    if contract_mismatches:
        blockers.append(
            "prepared_workload_manifest:runtime_contract_mismatch:"
            + ",".join(sorted(contract_mismatches))
        )

    raw_pme = payload.get("pme")
    manifest_pme = _canonical_pme_config(
        raw_pme if isinstance(raw_pme, Mapping) else {}
    )
    missing_pme_fields = sorted(set(_PME_CONFIG_FIELDS) - set(manifest_pme))
    pme_mismatches = _pme_config_mismatches(manifest_pme, runtime_pme_config)
    diagnostics["manifest_pme_config"] = manifest_pme
    diagnostics["runtime_pme_config"] = dict(runtime_pme_config)
    diagnostics["missing_pme_fields"] = missing_pme_fields
    diagnostics["pme_mismatches"] = pme_mismatches
    diagnostics["pme_matches_runtime"] = not missing_pme_fields and not pme_mismatches
    if missing_pme_fields:
        blockers.append(
            "prepared_workload_manifest:pme_fields_missing:"
            + ",".join(missing_pme_fields)
        )
    if pme_mismatches:
        blockers.append(
            "prepared_workload_manifest:pme_config_mismatch:"
            + ",".join(sorted(pme_mismatches))
        )
    return blockers, diagnostics


def _profile_admission(
    *,
    parity: Mapping[str, object],
    artifact_atom_count: int,
    nonbonded: NonbondedPotential,
    prepared_dir: Path,
) -> tuple[list[str], dict[str, object]]:
    blockers: list[str] = []
    parity_atom_count = parity.get("atom_count")
    if parity_atom_count != artifact_atom_count:
        blockers.append(
            "atom_count_mismatch:"
            f"parity={parity_atom_count}:artifact={artifact_atom_count}"
        )

    runtime_pme = _runtime_pme_config(nonbonded)
    parity_pme = parity.get("pme_config")
    if not isinstance(parity_pme, Mapping) or not parity_pme:
        blockers.append("parity_pme_config:missing")
        parity_pme = {}
    parity_pme_mismatches = _pme_config_mismatches(parity_pme, runtime_pme)
    if parity_pme_mismatches:
        blockers.append(
            "parity_pme_config:mismatch:"
            + ",".join(sorted(parity_pme_mismatches))
        )

    manifest_comparison = parity.get("manifest_comparison")
    if parity.get("schema") in {
        "charged_pme_parity_v1",
        "gpcrmd_pme_parity_v1",
    } and (
        not isinstance(manifest_comparison, Mapping)
        or manifest_comparison.get("matched") is not True
    ):
        blockers.append("parity_manifest_comparison:not_matched")

    readiness = parity.get("pme_readiness")
    if parity.get("schema") in {
        "charged_pme_parity_v1",
        "gpcrmd_pme_parity_v1",
    }:
        if not isinstance(readiness, Mapping):
            blockers.append("parity_pme_readiness:missing")
        else:
            if readiness.get("status") != "ready":
                blockers.append(
                    "parity_pme_readiness:not_ready:"
                    f"status={readiness.get('status')}"
                )
            if readiness.get("production_executable") is not True:
                blockers.append("parity_pme_readiness:not_production_executable")
            if readiness.get("atom_count") != artifact_atom_count:
                blockers.append(
                    "parity_pme_readiness:atom_count_mismatch:"
                    f"readiness={readiness.get('atom_count')}:"
                    f"artifact={artifact_atom_count}"
                )
            readiness_blockers = readiness.get("blockers")
            if readiness_blockers not in (None, []):
                blockers.append("parity_pme_readiness:has_blockers")

    gpcrmd_diagnostics: dict[str, object] | None = None
    if parity.get("schema") == "gpcrmd_pme_parity_v1":
        gpcrmd_blockers, gpcrmd_diagnostics = _gpcrmd_manifest_admission(
            prepared_dir=prepared_dir,
            parity=parity,
            artifact_atom_count=artifact_atom_count,
            runtime_pme_config=runtime_pme,
        )
        blockers.extend(gpcrmd_blockers)
    diagnostics = {
        "parity_atom_count": parity_atom_count,
        "artifact_atom_count": artifact_atom_count,
        "runtime_pme_config": runtime_pme,
        "parity_pme_config": dict(parity_pme),
        "parity_pme_mismatches": parity_pme_mismatches,
        "manifest_comparison_matched": bool(
            isinstance(manifest_comparison, Mapping)
            and manifest_comparison.get("matched") is True
        ),
        "gpcrmd_workload_manifest": gpcrmd_diagnostics,
    }
    return blockers, diagnostics


def _timing_summary(row: dict | None, *, blocker: str | None = None) -> dict:
    if row is None:
        return {
            "available": False,
            "mean_s": None,
            "median_s": None,
            "min_s": None,
            "max_s": None,
            "blocker": blocker,
        }
    return {
        "available": True,
        "mean_s": row["mean_s"],
        "median_s": row["median_s"],
        "min_s": row["min_s"],
        "max_s": row["max_s"],
        "blocker": None,
    }


def _sum_timing_summaries(rows: list[dict], *, blocker: str | None = None) -> dict:
    if not rows:
        return {
            "available": False,
            "mean_s": None,
            "median_s": None,
            "min_s": None,
            "max_s": None,
            "blocker": blocker,
        }
    return {
        "available": True,
        "mean_s": float(sum(row["mean_s"] for row in rows)),
        "median_s": float(sum(row["median_s"] for row in rows)),
        "min_s": float(sum(row["min_s"] for row in rows)),
        "max_s": float(sum(row["max_s"] for row in rows)),
        "blocker": None,
    }


def _stage_timings(
    rows: list[dict],
    *,
    missing_blocker: str | None = None,
    synchronization_blocker: str | None = SYNC_TIMING_BLOCKER,
) -> dict:
    by_name = {row["name"]: row for row in rows}
    assignment_rows = [
        by_name[name]
        for name in (
            "charge_assignment_bspline",
            "charge_assignment_cic",
            "interpolate_potential",
            "interpolate_field",
        )
        if name in by_name
    ]
    fft_rows = [
        by_name[name]
        for name in ("forward_fft", "influence_function", "inverse_fft_potential_and_fields")
        if name in by_name
    ]
    correction_rows = [
        by_name[name]
        for name in (
            "coulomb_exclusion_correction",
            "coulomb_exception",
            "coulomb_one_four_correction",
        )
        if name in by_name
    ]
    return {
        "pme_total": _timing_summary(by_name.get("pme_coulomb_full"), blocker=missing_blocker),
        "direct_space": _timing_summary(
            by_name.get("real_space_coulomb"),
            blocker=missing_blocker,
        ),
        "reciprocal_space": _timing_summary(
            by_name.get("reciprocal_full"),
            blocker=missing_blocker,
        ),
        "reciprocal_fft_influence": _sum_timing_summaries(fft_rows, blocker=missing_blocker),
        "assignment_interpolation": _sum_timing_summaries(
            assignment_rows,
            blocker=missing_blocker,
        ),
        "corrections": _sum_timing_summaries(correction_rows, blocker=missing_blocker),
        "synchronization": _timing_summary(
            by_name.get("synchronization"),
            blocker=synchronization_blocker,
        ),
        "production_nonbonded_total": _timing_summary(
            by_name.get("production_nonbonded_pme_path"),
            blocker=missing_blocker,
        ),
    }


def _blocked_payload(
    *,
    fixture_dir: Path,
    iterations: int,
    warmups: int,
    blocker: str,
    report_path: Path | None = None,
    prepared_dir: Path | None = None,
    parity: Mapping[str, object] | None = None,
    diagnostics: Mapping[str, object] | None = None,
) -> dict:
    report_path = report_path or fixture_dir / LEGACY_PARITY_REPORT_NAME
    prepared_dir = prepared_dir or fixture_dir / "prepared"
    missing = [{"name": "pme_fixture", "stage": "all", "blocker": blocker}]
    sync_blocker = {
        "name": "synchronization",
        "stage": "synchronization",
        "blocker": SYNC_TIMING_BLOCKER,
    }
    hardware = get_hardware_info()
    runtime = asdict(get_runtime_info())
    parity_payload = (
        dict(parity)
        if parity is not None
        else {
            "report_path": str(report_path),
            "status": "blocked",
            "passed": False,
        }
    )
    fixture = parity_payload.get("fixture") or str(fixture_dir)
    diagnostic_payload = {
        "fixture_dir": str(fixture_dir),
        "prepared_dir": str(prepared_dir),
        "atom_count": parity_payload.get("atom_count"),
    }
    if diagnostics is not None:
        diagnostic_payload.update(diagnostics)
    payload = {
        "benchmark_name": "pme_performance",
        "status": "blocked",
        "hardware": hardware,
        "runtime": runtime,
        "config": {
            "iterations": iterations,
            "warmups": warmups,
            "parity_report": str(report_path),
            "prepared": str(prepared_dir),
        },
        "fixture": fixture,
        "atom_count": parity_payload.get("atom_count"),
        "parity": parity_payload,
        "diagnostics": diagnostic_payload,
        "direct_space_policy": {
            "policy": "not_evaluated",
            "representation": None,
            "uses_shared_neighbor_policy": False,
            "supported": False,
            "real_cutoff": None,
            "minimum_image_safe": None,
            "pair_count": None,
            "compact_pair_count": None,
            "candidate_count": None,
            "candidate_waste_count": None,
            "fallback_reason": None,
            "blocker": blocker,
        },
        "timings": [],
        "stage_timings": _stage_timings([], missing_blocker=blocker),
        "missing_timing_splits": missing + [sync_blocker],
        "unsupported_timing_split_blockers": missing + [sync_blocker],
    }
    return normalize_benchmark_payload(
        payload,
        benchmark_name="pme_performance",
        fixture=fixture,
        timing_metric="median_s",
        hardware=hardware,
        runtime=runtime,
        atom_count=parity_payload.get("atom_count"),
        evaluation_count=iterations,
        finite=False,
        status="blocked",
        blocker=blocker,
        command=default_benchmark_command("pme_performance"),
    )


def _find_pme_nonbonded(force_terms: list[object]) -> NonbondedPotential:
    for term in force_terms:
        if isinstance(term, NonbondedPotential) and term.electrostatics == "pme":
            return term
    msg = "prepared fixture did not build a PME NonbondedPotential"
    raise ValueError(msg)


def _empty_correction_result(positions: mx.array) -> tuple[mx.array, mx.array]:
    return mx.array(0.0, dtype=mx.float32), mx.zeros_like(positions)


def _mlx_memory_value(name: str) -> int | None:
    accessor = getattr(mx, name, None)
    if not callable(accessor):
        return None
    try:
        return int(accessor())
    except (RuntimeError, TypeError, ValueError):
        return None


def build_payload(
    *,
    fixture_dir: Path | None = None,
    parity_report: Path | None = None,
    prepared: Path | None = None,
    iterations: int = 5,
    warmups: int = 1,
) -> dict:
    """Return a PME profile payload for a parity-admitted prepared fixture."""

    fixture_label, report_path, prepared_dir = _resolve_profile_paths(
        fixture_dir=fixture_dir,
        parity_report=parity_report,
        prepared=prepared,
    )
    if fixture_dir is None and parity_report is None and prepared is None:
        return _blocked_payload(
            fixture_dir=fixture_label,
            iterations=iterations,
            warmups=warmups,
            blocker=(
                "PME profiling requires --fixture-dir or both "
                "--parity-report and --prepared"
            ),
            report_path=report_path,
            prepared_dir=prepared_dir,
        )
    if not report_path.is_file():
        return _blocked_payload(
            fixture_dir=fixture_label,
            iterations=iterations,
            warmups=warmups,
            blocker=f"missing PME parity report: {report_path}",
            report_path=report_path,
            prepared_dir=prepared_dir,
        )
    if not prepared_dir.is_dir():
        return _blocked_payload(
            fixture_dir=fixture_label,
            iterations=iterations,
            warmups=warmups,
            blocker=f"missing prepared PME fixture directory: {prepared_dir}",
            report_path=report_path,
            prepared_dir=prepared_dir,
        )

    try:
        parity = _load_parity_report(report_path)
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        return _blocked_payload(
            fixture_dir=fixture_label,
            iterations=iterations,
            warmups=warmups,
            blocker=f"invalid PME parity report: {report_path}: {exc}",
            report_path=report_path,
            prepared_dir=prepared_dir,
        )
    if not parity["passed"]:
        return _blocked_payload(
            fixture_dir=fixture_label,
            iterations=iterations,
            warmups=warmups,
            blocker=f"PME parity report is not passing: {report_path}",
            report_path=report_path,
            prepared_dir=prepared_dir,
            parity=parity,
        )

    try:
        artifact = load_prepared_mlx_artifact(prepared_dir, require_production=True)
        system, force_terms, _ = build_mlx_system_from_artifact(
            artifact,
            eager_nonbonded_pair_limit=0,
        )
        nonbonded = _find_pme_nonbonded(force_terms)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return _blocked_payload(
            fixture_dir=fixture_label,
            iterations=iterations,
            warmups=warmups,
            blocker=f"prepared PME fixture admission failed: {exc}",
            report_path=report_path,
            prepared_dir=prepared_dir,
            parity=parity,
        )
    if nonbonded.pme_config is None:
        return _blocked_payload(
            fixture_dir=fixture_label,
            iterations=iterations,
            warmups=warmups,
            blocker="prepared PME nonbonded term is missing pme_config",
            report_path=report_path,
            prepared_dir=prepared_dir,
            parity=parity,
        )
    if system.cell is None:
        return _blocked_payload(
            fixture_dir=fixture_label,
            iterations=iterations,
            warmups=warmups,
            blocker="prepared PME fixture is missing a periodic cell",
            report_path=report_path,
            prepared_dir=prepared_dir,
            parity=parity,
        )

    admission_blockers, admission_diagnostics = _profile_admission(
        parity=parity,
        artifact_atom_count=artifact.atom_count,
        nonbonded=nonbonded,
        prepared_dir=artifact.base_dir,
    )
    pair_cache_materialized = (
        getattr(nonbonded.topology, "_nonbonded_pairs", None) is not None
    )
    topology_diagnostics = {
        "pair_policy": nonbonded.topology.nonbonded_pair_policy,
        "pair_cache_materialized": pair_cache_materialized,
        "nonbonded_pair_count": nonbonded.topology.nonbonded_pair_count,
    }
    if nonbonded.topology.nonbonded_pair_policy != "lazy":
        admission_blockers.append(
            "prepared_topology:not_lazy:"
            f"policy={nonbonded.topology.nonbonded_pair_policy}"
        )
    if pair_cache_materialized:
        admission_blockers.append("prepared_topology:pair_cache_materialized")
    if admission_blockers:
        return _blocked_payload(
            fixture_dir=fixture_label,
            iterations=iterations,
            warmups=warmups,
            blocker="PME profile admission failed: " + "; ".join(admission_blockers),
            report_path=report_path,
            prepared_dir=prepared_dir,
            parity=parity,
            diagnostics={
                "admission": admission_diagnostics,
                "topology": topology_diagnostics,
            },
        )

    try:
        nonbonded = nonbonded.bind_pme_plan(system.cell)
        positions, charges, cell_lengths, cell_lengths_np = _validate_inputs_mx(
            system.positions,
            nonbonded.charges,
            system.cell,
            charge_tolerance=nonbonded.pme_config.charge_tolerance,
            background_policy=nonbonded.pme_config.background_policy,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        return _blocked_payload(
            fixture_dir=fixture_label,
            iterations=iterations,
            warmups=warmups,
            blocker=f"PME plan/input admission failed: {exc}",
            report_path=report_path,
            prepared_dir=prepared_dir,
            parity=parity,
            diagnostics={
                "admission": admission_diagnostics,
                "topology": topology_diagnostics,
            },
        )

    config = nonbonded.pme_config
    real_cutoff = (
        float(config.real_cutoff)
        if config.real_cutoff is not None
        else 0.5 * float(np.min(cell_lengths_np))
    )
    direct_space_neighbor_report: dict[str, object] = {
        "backend": None,
        "representation_kind": "dense",
        "pair_count": None,
        "compact_pair_count": None,
        "candidate_count": None,
        "candidate_waste_count": None,
        "compaction_backend": None,
        "fallback_reason": None,
        "build_blocker": None,
    }
    try:
        direct_space_neighbors = build_neighbor_list(
            positions,
            system.cell,
            cutoff=real_cutoff,
            skin=0.0,
            backend="mlx_cell_blocks",
        )
        direct_space_interactions = direct_space_neighbors.interactions
        direct_space_neighbor_report = {
            "backend": direct_space_neighbors.backend,
            "representation_kind": direct_space_neighbors.representation_kind,
            "pair_count": int(direct_space_neighbors.pair_count),
            "compact_pair_count": int(direct_space_neighbors.compact_pair_count),
            "candidate_count": direct_space_neighbors.candidate_count,
            "candidate_waste_count": direct_space_neighbors.candidate_waste_count,
            "compaction_backend": direct_space_neighbors.compaction_backend,
            "fallback_reason": direct_space_neighbors.fallback_reason,
            "build_blocker": None,
        }
    except (RuntimeError, TypeError, ValueError) as exc:
        blocker = f"pme_direct_space_shared_neighbor_build_failed:{exc}"
        direct_space_neighbor_report["build_blocker"] = blocker
        return _blocked_payload(
            fixture_dir=fixture_label,
            iterations=iterations,
            warmups=warmups,
            blocker=blocker,
            report_path=report_path,
            prepared_dir=prepared_dir,
            parity=parity,
            diagnostics={
                "admission": admission_diagnostics,
                "topology": topology_diagnostics,
                "direct_space_neighbor": direct_space_neighbor_report,
            },
        )
    neighbor_blockers = []
    if direct_space_neighbors.backend != "mlx_cell_blocks":
        neighbor_blockers.append(
            f"backend={direct_space_neighbors.backend}:expected=mlx_cell_blocks"
        )
    if direct_space_neighbors.representation_kind != "blocks" or not isinstance(
        direct_space_interactions,
        NeighborBlocks,
    ):
        neighbor_blockers.append(
            "representation="
            f"{direct_space_neighbors.representation_kind}:expected=NeighborBlocks"
        )
    if direct_space_neighbors.fallback_reason is not None:
        neighbor_blockers.append(
            f"fallback_reason={direct_space_neighbors.fallback_reason}"
        )
    if neighbor_blockers:
        return _blocked_payload(
            fixture_dir=fixture_label,
            iterations=iterations,
            warmups=warmups,
            blocker=(
                "PME shared-neighbor admission failed: "
                + "; ".join(neighbor_blockers)
            ),
            report_path=report_path,
            prepared_dir=prepared_dir,
            parity=parity,
            diagnostics={
                "admission": admission_diagnostics,
                "topology": topology_diagnostics,
                "direct_space_neighbor": direct_space_neighbor_report,
            },
        )
    direct_space_policy = pme_direct_space_policy_report(
        system.cell,
        config=config,
        pairs=direct_space_interactions,
    )
    if (
        direct_space_policy.get("policy") != "block_candidate"
        or direct_space_policy.get("representation") != "blocks"
        or direct_space_policy.get("uses_shared_neighbor_policy") is not True
        or direct_space_policy.get("fallback_reason") is not None
    ):
        return _blocked_payload(
            fixture_dir=fixture_label,
            iterations=iterations,
            warmups=warmups,
            blocker=(
                "PME direct-space policy attempted dense/tiled fallback: "
                f"{direct_space_policy}"
            ),
            report_path=report_path,
            prepared_dir=prepared_dir,
            parity=parity,
            diagnostics={
                "admission": admission_diagnostics,
                "topology": topology_diagnostics,
                "direct_space_neighbor": direct_space_neighbor_report,
            },
        )

    charge_grid = _assign_charges_bspline_mx(
        positions,
        charges,
        cell_lengths,
        config.mesh_shape,
        assignment_order=config.assignment_order,
    )
    mx.eval(charge_grid)
    rho_hat = mx.fft.fftn(charge_grid)
    influence, k_components, _ = _influence_function_mx(
        cell_lengths_np,
        config.mesh_shape,
        alpha=config.alpha,
        coulomb_constant=nonbonded.coulomb_constant,
        deconvolve_assignment=config.deconvolve_assignment,
        assignment_order=config.assignment_order,
    )
    mx.eval(rho_hat, influence, *k_components)
    phi_hat = influence * rho_hat
    grid_size = int(np.prod(config.mesh_shape))
    potential_grid = mx.real(mx.fft.ifftn(phi_hat)) * float(grid_size)
    field_grid = mx.stack(
        [
            mx.real(mx.fft.ifftn((-1j * k_axis) * phi_hat)) * float(grid_size)
            for k_axis in k_components
        ],
        axis=-1,
    )
    mx.eval(potential_grid, field_grid)

    correction_pairs = nonbonded._ewald_correction_pairs()
    one_four_pairs = nonbonded._ewald_one_four_pairs()
    exception_pairs = nonbonded.exception_pairs

    def correction_components() -> tuple[mx.array, mx.array]:
        if correction_pairs.shape[0] == 0:
            return _empty_correction_result(positions)
        i = correction_pairs[:, 0]
        j = correction_pairs[:, 1]
        return nonbonded._bare_coulomb_components(
            positions,
            system.cell,
            correction_pairs,
            -(nonbonded.charges[i] * nonbonded.charges[j]),
        )

    def exception_components() -> tuple[mx.array, mx.array]:
        if exception_pairs.shape[0] == 0:
            return _empty_correction_result(positions)
        return nonbonded._bare_coulomb_components(
            positions,
            system.cell,
            exception_pairs,
            nonbonded.exception_charge_products,
        )

    def one_four_components() -> tuple[mx.array, mx.array]:
        if one_four_pairs.shape[0] == 0:
            return _empty_correction_result(positions)
        i = one_four_pairs[:, 0]
        j = one_four_pairs[:, 1]
        charge_products = (nonbonded.coulomb_one_four_scale - 1.0) * (
            nonbonded.charges[i] * nonbonded.charges[j]
        )
        return nonbonded._bare_coulomb_components(
            positions,
            system.cell,
            one_four_pairs,
            charge_products,
        )

    dense_reference_supported = artifact.atom_count <= 4096
    rows = [
        _time(
            "real_space_coulomb",
            "pme",
            lambda: pme_coulomb_direct_space_energy_forces(
                positions,
                charges,
                system.cell,
                coulomb_constant=nonbonded.coulomb_constant,
                config=config,
                pairs=direct_space_interactions,
            ),
            eval_outputs=_eval_all,
            warmups=warmups,
            iterations=iterations,
        ),
        *(
            [
                _time(
                    "real_space_coulomb_dense_reference",
                    "pme_reference",
                    lambda: _real_space_energy_forces_mx(
                        positions,
                        charges,
                        cell_lengths,
                        cell_lengths_np,
                        alpha=config.alpha,
                        cutoff=real_cutoff,
                        coulomb_constant=nonbonded.coulomb_constant,
                    ),
                    eval_outputs=_eval_all,
                    warmups=warmups,
                    iterations=iterations,
                )
            ]
            if dense_reference_supported
            else []
        ),
        _time(
            "charge_assignment_bspline",
            "pme",
            lambda: _assign_charges_bspline_mx(
                positions,
                charges,
                cell_lengths,
                config.mesh_shape,
                assignment_order=config.assignment_order,
            ),
            eval_outputs=_eval_all,
            warmups=warmups,
            iterations=iterations,
        ),
        _time(
            "forward_fft",
            "pme",
            lambda: mx.fft.fftn(charge_grid),
            eval_outputs=_eval_all,
            warmups=warmups,
            iterations=iterations,
        ),
        _time(
            "influence_function",
            "pme",
            lambda: _influence_function_mx(
                cell_lengths_np,
                config.mesh_shape,
                alpha=config.alpha,
                coulomb_constant=nonbonded.coulomb_constant,
                deconvolve_assignment=config.deconvolve_assignment,
                assignment_order=config.assignment_order,
            ),
            eval_outputs=lambda value: mx.eval(value[0], *value[1]),
            warmups=warmups,
            iterations=iterations,
        ),
        _time(
            "inverse_fft_potential_and_fields",
            "pme",
            lambda: (
                mx.real(mx.fft.ifftn(influence * rho_hat)) * float(grid_size),
                *[
                    mx.real(mx.fft.ifftn((-1j * k_axis) * influence * rho_hat))
                    * float(grid_size)
                    for k_axis in k_components
                ],
            ),
            eval_outputs=_eval_all,
            warmups=warmups,
            iterations=iterations,
        ),
        _time(
            "interpolate_potential",
            "pme",
            lambda: _interpolate_bspline_mx(
                positions,
                potential_grid,
                cell_lengths,
                assignment_order=config.assignment_order,
            ),
            eval_outputs=_eval_all,
            warmups=warmups,
            iterations=iterations,
        ),
        _time(
            "interpolate_field",
            "pme",
            lambda: _interpolate_bspline_mx(
                positions,
                field_grid,
                cell_lengths,
                assignment_order=config.assignment_order,
            ),
            eval_outputs=_eval_all,
            warmups=warmups,
            iterations=iterations,
        ),
        _time(
            "reciprocal_full",
            "pme",
            lambda: _mesh_reciprocal_energy_forces_mx(
                positions,
                charges,
                cell_lengths,
                cell_lengths_np,
                config=config,
                coulomb_constant=nonbonded.coulomb_constant,
                plan=nonbonded.pme_plan,
            )[:2],
            eval_outputs=_eval_all,
            warmups=warmups,
            iterations=iterations,
        ),
        _time(
            "pme_coulomb_full",
            "pme",
            lambda: pme_coulomb_energy_forces(
                positions,
                charges,
                system.cell,
                coulomb_constant=nonbonded.coulomb_constant,
                config=config,
                direct_space_pairs=direct_space_interactions,
                plan=nonbonded.pme_plan,
            )[:2],
            eval_outputs=_eval_all,
            warmups=warmups,
            iterations=iterations,
        ),
        _time(
            "pme_coulomb_full_reused_plan",
            "pme",
            lambda: pme_coulomb_energy_forces(
                positions,
                charges,
                system.cell,
                coulomb_constant=nonbonded.coulomb_constant,
                config=config,
                direct_space_pairs=direct_space_interactions,
                plan=nonbonded.pme_plan,
            )[:2],
            eval_outputs=_eval_all,
            warmups=warmups,
            iterations=iterations,
        ),
        _time(
            "lj_regular_plus_exception",
            "non_pme_lj",
            lambda: (
                nonbonded._regular_lj_components(
                    positions,
                    system.cell,
                    direct_space_interactions,
                ),
                nonbonded._exception_lj_components(positions, system.cell),
            ),
            eval_outputs=lambda value: mx.eval(value[0][0], value[0][1], value[1][0], value[1][1]),
            warmups=warmups,
            iterations=iterations,
        ),
        _time(
            "coulomb_exclusion_correction",
            "pme_corrections",
            correction_components,
            eval_outputs=_eval_all,
            warmups=warmups,
            iterations=iterations,
        ),
        _time(
            "coulomb_exception",
            "pme_corrections",
            exception_components,
            eval_outputs=_eval_all,
            warmups=warmups,
            iterations=iterations,
        ),
        _time(
            "coulomb_one_four_correction",
            "pme_corrections",
            one_four_components,
            eval_outputs=_eval_all,
            warmups=warmups,
            iterations=iterations,
        ),
        _time(
            "production_nonbonded_pme_path",
            "full_nonbonded",
            lambda: nonbonded._pme_energy_forces_with_components(
                positions,
                system.cell,
                direct_space_interactions,
            )[:2],
            eval_outputs=_eval_all,
            warmups=warmups,
            iterations=iterations,
        ),
        _time(
            "synchronization",
            "runtime",
            lambda: mx.sum(positions[:, 0]),
            eval_outputs=_eval_all,
            warmups=warmups,
            iterations=iterations,
        ),
    ]

    diagnostics = {
        "fixture_dir": str(fixture_label),
        "parity_report": str(report_path),
        "prepared_dir": str(prepared_dir),
        "atom_count": int(artifact.atom_count),
        "mesh_shape": list(config.mesh_shape),
        "assignment_order": config.assignment_order,
        "real_cutoff": real_cutoff,
        "correction_pair_count": int(correction_pairs.shape[0]),
        "exception_pair_count": int(exception_pairs.shape[0]),
        "one_four_pair_count": int(one_four_pairs.shape[0]),
        "net_charge": float(np.sum(np.asarray(nonbonded.charges), dtype=np.float64)),
        "admission": admission_diagnostics,
        "direct_space_neighbor": direct_space_neighbor_report,
        "plan": nonbonded.pme_plan.to_dict(),
        "topology": topology_diagnostics,
        "memory": {
            "max_rss_mb": max_rss_mb(),
            "mlx_active_memory_bytes": _mlx_memory_value("get_active_memory"),
            "mlx_peak_memory_bytes": _mlx_memory_value("get_peak_memory"),
            "mlx_cache_memory_bytes": _mlx_memory_value("get_cache_memory"),
        },
    }
    missing_splits = []
    if not dense_reference_supported:
        missing_splits.append(
            {
                "name": "real_space_coulomb_dense_reference",
                "stage": "reference_only",
                "blocker": (
                    "dense O(N^2) real-space reference is intentionally disabled "
                    "above 4096 atoms"
                ),
            }
        )
    timing_rows = [
        normalize_benchmark_row(
            row.to_dict(),
            benchmark_name="pme_performance",
            fixture=parity.get("fixture"),
            atom_count=diagnostics["atom_count"],
            evaluation_count=iterations,
            timing_metric="median_s",
        )
        for row in rows
    ]
    stage_timings = _stage_timings(timing_rows)
    checks = {
        "parity_passed": bool(parity["passed"]),
        "shared_neighbor_blocks": (
            direct_space_neighbor_report["backend"] == "mlx_cell_blocks"
            and direct_space_neighbor_report["representation_kind"] == "blocks"
            and isinstance(direct_space_interactions, NeighborBlocks)
        ),
        "no_direct_space_fallback": (
            direct_space_policy.get("policy") == "block_candidate"
            and direct_space_policy.get("representation") == "blocks"
            and direct_space_policy.get("fallback_reason") is None
        ),
        "admission_passed": not admission_blockers,
        "lazy_topology": nonbonded.topology.nonbonded_pair_policy == "lazy",
        "pair_cache_unmaterialized": getattr(
            nonbonded.topology,
            "_nonbonded_pairs",
            None,
        )
        is None,
        "one_bound_plan": nonbonded.pme_plan.build_count == 1,
        "plan_reused": nonbonded.pme_plan.reuse_count > 0,
        "direct_timing": bool(stage_timings["direct_space"]["available"]),
        "assignment_timing": bool(
            stage_timings["assignment_interpolation"]["available"]
        ),
        "fft_influence_timing": bool(
            stage_timings["reciprocal_fft_influence"]["available"]
        ),
        "correction_timing": bool(stage_timings["corrections"]["available"]),
        "synchronization_timing": bool(stage_timings["synchronization"]["available"]),
        "full_nonbonded_timing": bool(
            stage_timings["production_nonbonded_total"]["available"]
        ),
    }
    passed = all(checks.values())
    hardware = get_hardware_info()
    runtime = asdict(get_runtime_info())
    payload = {
        "benchmark_name": "pme_performance",
        "status": "ok" if passed else "failed",
        "hardware": hardware,
        "runtime": runtime,
        "config": {
            "iterations": iterations,
            "warmups": warmups,
            "parity_report": str(report_path),
            "prepared": str(prepared_dir),
        },
        "fixture": parity.get("fixture"),
        "atom_count": diagnostics["atom_count"],
        "parity": parity,
        "diagnostics": diagnostics,
        "direct_space_policy": direct_space_policy,
        "timings": timing_rows,
        "stage_timings": stage_timings,
        "checks": checks,
        "missing_timing_splits": missing_splits,
        "unsupported_timing_split_blockers": missing_splits,
    }
    return normalize_benchmark_payload(
        payload,
        benchmark_name="pme_performance",
        fixture=parity.get("fixture"),
        timing_metric="median_s",
        hardware=hardware,
        runtime=runtime,
        atom_count=diagnostics["atom_count"],
        evaluation_count=iterations,
        finite=passed,
        status="ok" if passed else "failed",
        blocker=None if passed else "PME profile acceptance checks failed",
        command=default_benchmark_command("pme_performance"),
    )


def _write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=None)
    parser.add_argument("--parity-report", type=Path, default=None)
    parser.add_argument("--prepared", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.iterations <= 0:
        msg = "--iterations must be positive"
        raise SystemExit(msg)
    if args.warmups < 0:
        msg = "--warmups must be non-negative"
        raise SystemExit(msg)

    raw_output_path = args.out_dir / "pme-profile.json"
    payload = build_payload(
        fixture_dir=args.fixture_dir,
        parity_report=args.parity_report,
        prepared=args.prepared,
        iterations=args.iterations,
        warmups=args.warmups,
    )
    payload["raw_output_path"] = str(raw_output_path)
    _write_payload(raw_output_path, payload)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        if payload["status"] != "ok":
            raise SystemExit(2)
        return

    parity = payload["parity"]
    print(
        "fixture,status,passed,atoms,mesh,split,median_s,category",
    )
    for row in payload["timings"]:
        print(
            f"{parity['fixture']},{parity['status']},{parity['passed']},"
            f"{payload['diagnostics']['atom_count']},"
            f"{'x'.join(str(item) for item in payload['diagnostics']['mesh_shape'])},"
            f"{row['name']},{row['median_s']:.6f},{row['category']}"
        )
    if payload["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
