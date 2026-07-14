"""Profile PME path costs for the existing OpenMM-vs-MLX parity fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
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
from mlx_atomistic.benchmarks.gpcrmd_runtime import max_rss_mb, resident_rss_mb
from mlx_atomistic.benchmarks.same_workload_compare import (
    build_strict_timing_comparison,
)
from mlx_atomistic.forcefields import NonbondedPotential
from mlx_atomistic.neighbors import build_neighbor_list
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
DENSE_REFERENCE_MAX_ATOMS = 4096
DENSE_REFERENCE_TARGET_SKIP = (
    "dense quadratic real-space oracle is restricted to fixtures with at most "
    f"{DENSE_REFERENCE_MAX_ATOMS} atoms and is intentionally skipped at target scale"
)
MEMORY_GROWTH_MIN_MB = 32.0
MEMORY_GROWTH_MIN_FRACTION = 0.05


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


def _metal_memory_mb(name: str) -> float | None:
    getter = getattr(mx, name, None)
    if getter is None:
        getter = getattr(mx.metal, name, None)
    if getter is None:
        return None
    return float(getter()) / (1024.0 * 1024.0)


def _memory_snapshot(evaluation: int) -> dict[str, float | int | None]:
    return {
        "evaluation": evaluation,
        "resident_rss_mb": resident_rss_mb(),
        "max_rss_mb": max_rss_mb(),
        "metal_active_mb": _metal_memory_mb("get_active_memory"),
        "metal_cache_mb": _metal_memory_mb("get_cache_memory"),
        "metal_peak_mb": _metal_memory_mb("get_peak_memory"),
    }


def classify_memory_growth(
    samples: list[dict[str, float | int | None]],
) -> dict[str, object]:
    """Classify repeated-evaluation memory samples for monotonic growth."""

    checks = []
    for field in ("resident_rss_mb", "metal_active_mb"):
        values = [float(row[field]) for row in samples if row.get(field) is not None]
        growth = 0.0 if len(values) < 2 else values[-1] - values[0]
        threshold = (
            MEMORY_GROWTH_MIN_MB
            if not values
            else max(MEMORY_GROWTH_MIN_MB, values[0] * MEMORY_GROWTH_MIN_FRACTION)
        )
        monotonic = len(values) >= 3 and all(
            right > left + 0.5
            for left, right in zip(values[:-1], values[1:], strict=True)
        )
        checks.append(
            {
                "field": field,
                "sample_count": len(values),
                "growth_mb": growth,
                "growth_threshold_mb": threshold,
                "monotonic_increase": monotonic,
                "unbounded_growth_detected": monotonic and growth > threshold,
            }
        )
    failed = [row["field"] for row in checks if row["unbounded_growth_detected"]]
    return {
        "status": "passed" if not failed else "failed",
        "passed": not failed,
        "failed_series": failed,
        "checks": checks,
    }


def _profile_memory(
    fn: Callable[[], object],
    *,
    evaluations: int,
) -> tuple[list[dict[str, float | int | None]], dict[str, object]]:
    reset_peak = getattr(mx, "reset_peak_memory", None)
    if reset_peak is None:
        reset_peak = getattr(mx.metal, "reset_peak_memory", None)
    if reset_peak is not None:
        reset_peak()
    samples = []
    for index in range(max(3, evaluations)):
        _eval_all(fn())
        samples.append(_memory_snapshot(index + 1))
    return samples, classify_memory_growth(samples)


def _load_stability_report(prepared_dir: Path) -> dict[str, object]:
    candidates = (
        prepared_dir.parent.parent / "stability" / "stability.json",
        prepared_dir.parent / "stability.json",
    )
    report_path = next((path for path in candidates if path.exists()), candidates[0])
    if not report_path.exists():
        return {
            "status": "missing",
            "report_path": str(report_path),
            "fixture_hash": None,
            "parameter_manifest_hash": None,
        }
    report = json.loads(report_path.read_text())
    return {
        "status": report.get("status"),
        "report_path": str(report_path),
        "fixture_hash": report.get("fixture_hash"),
        "parameter_manifest_hash": report.get(
            "parameter_manifest_hash",
            report.get("reference_manifest_sha256"),
        ),
        "hardware": report.get("hardware"),
        "runtime": report.get("runtime"),
        "raw_outputs": report.get("raw_outputs"),
    }


def _reference_timing_row(parity: dict, pme_parameters: dict) -> dict[str, object]:
    manifest_path = parity.get("reference_manifest")
    manifest = {}
    if manifest_path is not None and Path(str(manifest_path)).exists():
        manifest = json.loads(Path(str(manifest_path)).read_text())
    fixture = manifest.get("fixture", {})
    manifest_pme = manifest.get("pme", {})
    reference_pme_parameters = {
        "mesh_shape": manifest_pme.get("mesh_shape"),
        "assignment_order": manifest_pme.get("assignment_order"),
        "alpha_per_angstrom": manifest_pme.get("alpha_per_angstrom"),
        "real_cutoff_angstrom": manifest_pme.get("real_cutoff_angstrom"),
    }
    return {
        "operation": manifest.get("operation_semantics"),
        "atom_count": fixture.get("atom_count"),
        "fixture_hash": fixture.get("content_hash"),
        "parameter_manifest_hash": parity.get("parameter_manifest_hash"),
        "pme_parameters": reference_pme_parameters or pme_parameters,
        "step_count": 1,
        "precision": manifest.get("precision"),
        "timing_metric": manifest.get("timing_metric"),
        "timing_value": manifest.get("timing_value"),
    }


def _resolve_fixture_paths(fixture_dir: Path) -> tuple[Path, Path]:
    if (fixture_dir / "prepared_system.json").exists():
        prepared_dir = fixture_dir
        candidates = (
            fixture_dir.parent / "mlx_parity.json",
            fixture_dir / "mlx_parity.json",
            fixture_dir.parent / "openmm_mlx_parity_report.json",
        )
    else:
        prepared_dir = fixture_dir / "prepared"
        candidates = (
            fixture_dir / "openmm_mlx_parity_report.json",
            fixture_dir / "mlx_parity.json",
        )
    report_path = next((path for path in candidates if path.exists()), candidates[0])
    return prepared_dir, report_path


def _load_parity_report(report_path: Path) -> dict:
    with report_path.open() as handle:
        report = json.load(handle)
    fixture_payload = report.get("fixture")
    fixture = (
        fixture_payload.get("fixture")
        if isinstance(fixture_payload, dict)
        else fixture_payload
    )
    atom_count = (
        fixture_payload.get("atom_count")
        if isinstance(fixture_payload, dict)
        else report.get("atom_count")
    )
    fixture_hash = report.get("fixture_hash")
    if fixture_hash is None and isinstance(fixture_payload, dict):
        fixture_hash = fixture_payload.get("content_hash")
    parameter_manifest_hash = report.get("parameter_manifest_hash")
    reference_manifest = report.get("reference_manifest")
    if (
        parameter_manifest_hash is None
        and reference_manifest is not None
        and Path(str(reference_manifest)).exists()
    ):
        parameter_manifest_hash = hashlib.sha256(
            Path(str(reference_manifest)).read_bytes()
        ).hexdigest()
    return {
        "report_path": str(report_path),
        "status": report.get("status"),
        "passed": bool(report.get("passed", False)),
        "fixture": fixture,
        "atom_count": atom_count,
        "fixture_hash": fixture_hash,
        "parameter_manifest_hash": parameter_manifest_hash,
        "reference_manifest": reference_manifest,
        "reference_precision": report.get("reference_precision"),
        "precision": report.get("precision"),
        "hardware": report.get("hardware"),
        "runtime": report.get("runtime"),
        "raw_output_path": report.get("raw_output_path", str(report_path)),
        "openmm_nonbonded_method": report.get("openmm_nonbonded_method"),
        "total_energy_abs_error_kj_mol": report.get("total_energy_abs_error_kj_mol"),
        "force_max_abs_error_kj_mol_nm": report.get("force_max_abs_error_kj_mol_nm"),
        "force_rms_abs_error_kj_mol_nm": report.get("force_rms_abs_error_kj_mol_nm"),
        "pme_readiness": report.get("pme_readiness"),
        "pme_config": report.get("pme_config", report.get("pme")),
        "prepared_dir": report.get("prepared_dir"),
    }


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
        "charge_assignment": _timing_summary(
            by_name.get("charge_assignment_bspline"),
            blocker=missing_blocker,
        ),
        "interpolation": _sum_timing_summaries(
            [
                by_name[name]
                for name in ("interpolate_potential", "interpolate_field")
                if name in by_name
            ],
            blocker=missing_blocker,
        ),
        "forward_fft": _timing_summary(
            by_name.get("forward_fft"),
            blocker=missing_blocker,
        ),
        "inverse_fft_fields": _timing_summary(
            by_name.get("inverse_fft_potential_and_fields"),
            blocker=missing_blocker,
        ),
        "influence_work": _timing_summary(
            by_name.get("influence_function"),
            blocker=missing_blocker,
        ),
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


def _append_missing_split_once(
    entries: list[dict[str, str]],
    *,
    name: str,
    stage: str,
    blocker: str,
) -> None:
    entry = {"name": name, "stage": stage, "blocker": blocker}
    if entry not in entries:
        entries.append(entry)


def _blocked_payload(
    *,
    fixture_dir: Path,
    iterations: int,
    warmups: int,
    blocker: str,
) -> dict:
    missing = [{"name": "pme_fixture", "stage": "all", "blocker": blocker}]
    sync_blocker = {
        "name": "synchronization",
        "stage": "synchronization",
        "blocker": SYNC_TIMING_BLOCKER,
    }
    hardware = get_hardware_info()
    runtime = asdict(get_runtime_info())
    payload = {
        "benchmark_name": "pme_performance",
        "status": "blocked",
        "hardware": hardware,
        "runtime": runtime,
        "config": {
            "iterations": iterations,
            "warmups": warmups,
        },
        "fixture": str(fixture_dir),
        "atom_count": None,
        "parity": {
            "report_path": str(fixture_dir / "openmm_mlx_parity_report.json"),
            "status": "blocked",
            "passed": False,
        },
        "diagnostics": {
            "fixture_dir": str(fixture_dir),
            "prepared_dir": str(fixture_dir / "prepared"),
            "atom_count": None,
        },
        "direct_space_policy": {
            "policy": "fallback",
            "representation": "dense",
            "uses_shared_neighbor_policy": False,
            "supported": False,
            "real_cutoff": None,
            "minimum_image_safe": None,
            "pair_count": None,
            "compact_pair_count": None,
            "candidate_count": None,
            "candidate_waste_count": None,
            "fallback_reason": blocker,
        },
        "timings": [],
        "stage_timings": _stage_timings([], missing_blocker=blocker),
        "missing_timing_splits": missing + [sync_blocker],
        "unsupported_timing_split_blockers": missing + [sync_blocker],
    }
    return normalize_benchmark_payload(
        payload,
        benchmark_name="pme_performance",
        fixture=str(fixture_dir),
        timing_metric="median_s",
        hardware=hardware,
        runtime=runtime,
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


def build_payload(
    *,
    fixture_dir: Path | None = None,
    iterations: int = 5,
    warmups: int = 1,
) -> dict:
    """Return a PME profile payload for the existing parity fixture."""

    if fixture_dir is None:
        return _blocked_payload(
            fixture_dir=MISSING_FIXTURE_LABEL,
            iterations=iterations,
            warmups=warmups,
            blocker="PME profiling requires an explicit --fixture-dir path",
        )
    prepared_dir, report_path = _resolve_fixture_paths(fixture_dir)
    if not report_path.exists():
        return _blocked_payload(
            fixture_dir=fixture_dir,
            iterations=iterations,
            warmups=warmups,
            blocker=f"missing PME parity report: {report_path}",
        )
    if not prepared_dir.exists():
        return _blocked_payload(
            fixture_dir=fixture_dir,
            iterations=iterations,
            warmups=warmups,
            blocker=f"missing prepared PME fixture directory: {prepared_dir}",
        )

    artifact = load_prepared_mlx_artifact(prepared_dir, require_production=True)
    system, force_terms, _ = build_mlx_system_from_artifact(artifact)
    nonbonded = _find_pme_nonbonded(force_terms)
    if nonbonded.pme_config is None:
        msg = "PME nonbonded term is missing pme_config"
        raise ValueError(msg)
    if system.cell is None:
        msg = "PME fixture is missing a periodic cell"
        raise ValueError(msg)

    positions, charges, cell_lengths, cell_lengths_np = _validate_inputs_mx(
        system.positions,
        nonbonded.charges,
        system.cell,
        charge_tolerance=nonbonded.pme_config.charge_tolerance,
    )
    config = nonbonded.pme_config
    real_cutoff = (
        float(config.real_cutoff)
        if config.real_cutoff is not None
        else 0.5 * float(np.min(cell_lengths_np))
    )
    direct_space_interactions = None
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
    shared_neighbor_blocker = None
    try:
        direct_space_neighbors = build_neighbor_list(
            positions,
            system.cell,
            cutoff=real_cutoff,
            skin=0.0,
            backend="mlx_cell_pairs",
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
        shared_neighbor_blocker = f"pme_direct_space_shared_neighbor_build_failed:{exc}"
    if (
        shared_neighbor_blocker is not None
        or direct_space_neighbor_report["backend"] != "mlx_cell_pairs"
        or direct_space_neighbor_report["representation_kind"] != "pairs"
        or direct_space_neighbor_report["fallback_reason"] is not None
    ):
        blocker = shared_neighbor_blocker or (
            "target-safe PME profiling requires mlx_cell_pairs without fallback"
        )
        return _blocked_payload(
            fixture_dir=fixture_dir,
            iterations=iterations,
            warmups=warmups,
            blocker=blocker,
        )
    direct_space_policy = pme_direct_space_policy_report(
        system.cell,
        config=config,
        pairs=direct_space_interactions,
    )
    if shared_neighbor_blocker is not None:
        direct_space_policy = {
            **direct_space_policy,
            "policy": "fallback",
            "representation": "dense",
            "uses_shared_neighbor_policy": False,
            "fallback_reason": shared_neighbor_blocker,
        }

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
            )[:2],
            eval_outputs=_eval_all,
            warmups=warmups,
            iterations=iterations,
        ),
        _time(
            "pme_coulomb_full_shared_direct",
            "pme",
            lambda: pme_coulomb_energy_forces(
                positions,
                charges,
                system.cell,
                coulomb_constant=nonbonded.coulomb_constant,
                config=config,
                direct_space_pairs=direct_space_interactions,
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
    ]

    dense_reference_skipped = artifact.atom_count > DENSE_REFERENCE_MAX_ATOMS
    if not dense_reference_skipped:
        rows.insert(
            1,
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
            ),
        )

    memory_samples, memory_growth = _profile_memory(
        lambda: nonbonded._pme_energy_forces_with_components(
            positions,
            system.cell,
            direct_space_interactions,
        )[:2],
        evaluations=iterations,
    )

    parity = _load_parity_report(report_path)
    stability = _load_stability_report(prepared_dir)
    fixture_hash = parity.get("fixture_hash")
    parameter_manifest_hash = parity.get("parameter_manifest_hash")
    evidence_consistent = bool(
        fixture_hash
        and parameter_manifest_hash
        and stability.get("status") == "passed"
        and stability.get("fixture_hash") == fixture_hash
        and stability.get("parameter_manifest_hash") == parameter_manifest_hash
    )
    provenance_complete = bool(
        parity.get("hardware")
        and parity.get("runtime")
        and stability.get("hardware")
        and stability.get("runtime")
    )
    pme_parameters = {
        "mesh_shape": list(config.mesh_shape),
        "assignment_order": config.assignment_order,
        "alpha_per_angstrom": config.alpha,
        "real_cutoff_angstrom": real_cutoff,
    }
    diagnostics = {
        "fixture_dir": str(fixture_dir),
        "prepared_dir": str(prepared_dir),
        "atom_count": int(artifact.atom_count),
        "fixture_hash": fixture_hash,
        "parameter_manifest_hash": parameter_manifest_hash,
        "mesh_shape": list(config.mesh_shape),
        "assignment_order": config.assignment_order,
        "real_cutoff": real_cutoff,
        "correction_pair_count": int(correction_pairs.shape[0]),
        "exception_pair_count": int(exception_pairs.shape[0]),
        "one_four_pair_count": int(one_four_pairs.shape[0]),
        "net_charge": float(np.sum(np.asarray(nonbonded.charges), dtype=np.float64)),
        "direct_space_neighbor": direct_space_neighbor_report,
        "dense_reference": {
            "executed": not dense_reference_skipped,
            "max_atoms": DENSE_REFERENCE_MAX_ATOMS,
            "skip_reason": DENSE_REFERENCE_TARGET_SKIP
            if dense_reference_skipped
            else None,
        },
    }
    missing_splits = [
        {
            "name": "synchronization",
            "stage": "synchronization",
            "blocker": SYNC_TIMING_BLOCKER,
        }
    ]
    if shared_neighbor_blocker is not None:
        _append_missing_split_once(
            missing_splits,
            name="direct_space_shared_neighbor_policy",
            stage="direct_space",
            blocker=shared_neighbor_blocker,
        )
    if direct_space_policy.get("policy") == "fallback":
        _append_missing_split_once(
            missing_splits,
            name="direct_space_shared_neighbor_policy",
            stage="direct_space",
            blocker=str(direct_space_policy.get("fallback_reason")),
        )
    if dense_reference_skipped:
        _append_missing_split_once(
            missing_splits,
            name="real_space_coulomb_dense_reference",
            stage="direct_space_reference",
            blocker=DENSE_REFERENCE_TARGET_SKIP,
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
    hardware = get_hardware_info()
    runtime = asdict(get_runtime_info())
    stage_timings = _stage_timings(timing_rows)
    production_timing = stage_timings["production_nonbonded_total"]
    mlx_comparison_row = {
        "operation": "production_nonbonded_pme_force_evaluation",
        "atom_count": diagnostics["atom_count"],
        "fixture_hash": fixture_hash,
        "parameter_manifest_hash": parameter_manifest_hash,
        "pme_parameters": pme_parameters,
        "step_count": 1,
        "precision": "float32",
        "timing_metric": "median_s",
        "timing_value": production_timing["median_s"],
    }
    strict_comparison = build_strict_timing_comparison(
        mlx_comparison_row,
        _reference_timing_row(parity, pme_parameters),
    )
    profile_blockers = []
    if not memory_growth["passed"]:
        profile_blockers.append("monotonic resident or active Metal memory growth detected")
    if not evidence_consistent:
        profile_blockers.append("parity and stability evidence hashes or status differ")
    if not provenance_complete:
        profile_blockers.append("parity or stability hardware/runtime provenance is incomplete")
    status = "ok" if not profile_blockers else "failed"
    blocker = None if not profile_blockers else "; ".join(profile_blockers)
    payload = {
        "benchmark_name": "pme_performance",
        "status": status,
        "hardware": hardware,
        "runtime": runtime,
        "operation": "production_nonbonded_pme_force_evaluation",
        "precision": "float32",
        "fixture_hash": fixture_hash,
        "parameter_manifest_hash": parameter_manifest_hash,
        "pme_parameters": pme_parameters,
        "timing_metric": "median_s",
        "timing_value": production_timing["median_s"],
        "step_count": 1,
        "config": {
            "iterations": iterations,
            "warmups": warmups,
        },
        "fixture": parity.get("fixture"),
        "atom_count": diagnostics["atom_count"],
        "parity": parity,
        "stability": stability,
        "evidence_consistent": evidence_consistent,
        "provenance_complete": provenance_complete,
        "diagnostics": diagnostics,
        "direct_space_policy": direct_space_policy,
        "timings": timing_rows,
        "stage_timings": stage_timings,
        "memory": {
            "units": "MB",
            "peak_resident_mb": max(float(row["max_rss_mb"]) for row in memory_samples),
            "peak_metal_mb": max(
                float(row["metal_peak_mb"] or 0.0) for row in memory_samples
            ),
            "samples": memory_samples,
            "growth": memory_growth,
        },
        "same_workload_comparison": {
            **strict_comparison,
            "mlx": mlx_comparison_row,
            "reference": _reference_timing_row(parity, pme_parameters),
        },
        "raw_outputs": {
            "parity": parity.get("raw_output_path"),
            "stability": stability.get("report_path"),
            "prepared": str(prepared_dir),
        },
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
        finite=True,
        status=status,
        blocker=blocker,
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
        iterations=args.iterations,
        warmups=args.warmups,
    )
    payload["raw_output_path"] = str(raw_output_path)
    payload["raw_outputs"] = {
        **dict(payload.get("raw_outputs", {})),
        "profile": str(raw_output_path),
    }
    _write_payload(raw_output_path, payload)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
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


if __name__ == "__main__":
    main()
