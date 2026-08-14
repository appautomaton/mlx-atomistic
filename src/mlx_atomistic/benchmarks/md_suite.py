"""Run and compare the canonical prepared-system MD benchmark suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.benchmarks.charged_pme import runtime_payload
from mlx_atomistic.benchmarks.schema import current_git_commit

CASE_REGISTRY_PATH = Path(__file__).with_name("data") / "md_suite_cases.json"
SUITE_SCHEMA = "mlx_atomistic.md_suite.v2"
COMPARISON_SCHEMA = "mlx_atomistic.md_suite_comparison.v2"
STAGE_PROFILE_SCHEMA = "mlx_atomistic.md_stage_profile.v1"
DEFAULT_LOCAL_SUITE = "local"
DEFAULT_REPEATS = 3
DEFAULT_WARMUP_STEPS = 10
DEFAULT_MEASURED_STEPS = 75
DEFAULT_REHEARSAL_STEPS = 75
DEFAULT_MAXIMUM_RELATIVE_SPREAD = 0.10


@dataclass(frozen=True)
class MDBenchmarkCase:
    """One immutable prepared-system benchmark contract."""

    case_id: str
    description: str
    prepared_path: Path
    expected_atom_count: int
    tier: str
    role: str
    neighbor_backend: str
    features: tuple[str, ...]
    preparation_command: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MDBenchmarkCase:
        """Build and validate one case from the committed registry."""

        case_id = str(payload.get("id", "")).strip()
        if not case_id:
            raise ValueError("MD benchmark case id must be non-empty")
        atom_count = int(payload.get("expected_atom_count", 0))
        if atom_count <= 0:
            raise ValueError(f"{case_id}: expected_atom_count must be positive")
        prepared_path = Path(str(payload.get("prepared_path", "")))
        if prepared_path.is_absolute() or not prepared_path.parts:
            raise ValueError(f"{case_id}: prepared_path must be repository-relative")
        neighbor_backend = str(payload.get("neighbor_backend", "mlx_cell_tiles"))
        if neighbor_backend not in {"mlx_cell_pairs", "mlx_cell_tiles"}:
            raise ValueError(f"{case_id}: unsupported neighbor_backend {neighbor_backend!r}")
        return cls(
            case_id=case_id,
            description=str(payload.get("description", "")).strip(),
            prepared_path=prepared_path,
            expected_atom_count=atom_count,
            tier=str(payload.get("tier", "")).strip(),
            role=str(payload.get("role", "")).strip(),
            neighbor_backend=neighbor_backend,
            features=tuple(str(item) for item in payload.get("features", ())),
            preparation_command=str(payload.get("preparation_command", "")).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation used for fingerprints."""

        return {
            "id": self.case_id,
            "description": self.description,
            "prepared_path": str(self.prepared_path),
            "expected_atom_count": self.expected_atom_count,
            "tier": self.tier,
            "role": self.role,
            "neighbor_backend": self.neighbor_backend,
            "features": list(self.features),
            "preparation_command": self.preparation_command,
        }

    @property
    def fingerprint(self) -> str:
        """Return a deterministic contract fingerprint."""

        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def load_case_registry(path: str | Path = CASE_REGISTRY_PATH) -> dict[str, Any]:
    """Load and validate the committed MD suite registry."""

    registry_path = Path(path)
    payload = json.loads(registry_path.read_text())
    if payload.get("schema") != "mlx_atomistic.md_suite_cases.v1":
        raise ValueError("unsupported MD benchmark case registry schema")
    cases = [MDBenchmarkCase.from_mapping(item) for item in payload.get("cases", ())]
    by_id = {case.case_id: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("MD benchmark case ids must be unique")
    suites = payload.get("suites", {})
    if not isinstance(suites, dict) or not suites:
        raise ValueError("MD benchmark registry must define suites")
    for suite_name, case_ids in suites.items():
        if not case_ids:
            raise ValueError(f"MD benchmark suite {suite_name!r} must not be empty")
        unknown = [str(case_id) for case_id in case_ids if str(case_id) not in by_id]
        if unknown:
            raise ValueError(f"MD benchmark suite {suite_name!r} has unknown cases: {unknown}")
    return {"schema": payload["schema"], "cases": by_id, "suites": suites}


def resolve_cases(
    *,
    registry: Mapping[str, Any],
    suite: str,
    case_ids: Sequence[str] = (),
) -> tuple[MDBenchmarkCase, ...]:
    """Resolve explicit case ids or one named suite in declared order."""

    cases = registry["cases"]
    selected = tuple(case_ids) if case_ids else tuple(registry["suites"].get(suite, ()))
    if not selected:
        choices = ", ".join(sorted(registry["suites"]))
        raise ValueError(f"unknown or empty MD benchmark suite {suite!r}; choose from {choices}")
    unknown = [case_id for case_id in selected if case_id not in cases]
    if unknown:
        raise ValueError(f"unknown MD benchmark cases: {unknown}")
    return tuple(cases[case_id] for case_id in selected)


def case_inventory(
    *,
    repo_root: str | Path,
    registry_path: str | Path = CASE_REGISTRY_PATH,
) -> dict:
    """Return registry entries with local prepared-artifact availability."""

    root = Path(repo_root).resolve()
    registry = load_case_registry(registry_path)
    rows = []
    for case in registry["cases"].values():
        prepared = root / case.prepared_path
        available = all(
            (prepared / name).is_file() for name in ("prepared_system.json", "prepared_system.npz")
        )
        rows.append({**case.to_dict(), "available": available, "resolved_path": str(prepared)})
    return {
        "schema": registry["schema"],
        "suites": registry["suites"],
        "cases": rows,
    }


def run_suite(
    *,
    repo_root: str | Path,
    out: str | Path,
    suite: str = DEFAULT_LOCAL_SUITE,
    case_ids: Sequence[str] = (),
    prepared_overrides: Mapping[str, str | Path] | None = None,
    repeats: int = DEFAULT_REPEATS,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    measured_steps: int = DEFAULT_MEASURED_STEPS,
    rehearsal_steps: int = DEFAULT_REHEARSAL_STEPS,
    maximum_relative_spread: float = DEFAULT_MAXIMUM_RELATIVE_SPREAD,
    neighbor_backend: str | None = None,
    registry_path: str | Path = CASE_REGISTRY_PATH,
    runner: Callable[..., dict[str, Any]] = runtime_payload,
) -> dict[str, Any]:
    """Run selected prepared PME cases and persist a comparison-ready payload."""

    if repeats <= 0 or warmup_steps <= 0 or measured_steps < 2 or rehearsal_steps < 2:
        raise ValueError(
            "repeats and warmup_steps must be positive; "
            "measured_steps and rehearsal_steps must be >= 2"
        )
    if not math.isfinite(maximum_relative_spread) or maximum_relative_spread < 0.0:
        raise ValueError("maximum_relative_spread must be finite and nonnegative")
    root = Path(repo_root).resolve()
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = root / out_path
    registry = load_case_registry(registry_path)
    selected = resolve_cases(registry=registry, suite=suite, case_ids=case_ids)
    overrides = dict(prepared_overrides or {})
    unknown_overrides = sorted(set(overrides) - {case.case_id for case in selected})
    if unknown_overrides:
        raise ValueError(f"prepared overrides do not match selected cases: {unknown_overrides}")

    run_root = out_path.parent / f"{out_path.stem}-raw"
    rows = []
    for case in selected:
        prepared = Path(overrides.get(case.case_id, case.prepared_path))
        if not prepared.is_absolute():
            prepared = root / prepared
        rows.append(
            _run_case(
                case=case,
                prepared=prepared,
                out_dir=run_root / case.case_id,
                repeats=repeats,
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
                rehearsal_steps=rehearsal_steps,
                maximum_relative_spread=maximum_relative_spread,
                neighbor_backend=neighbor_backend or case.neighbor_backend,
                runner=runner,
            )
        )
    passed = all(row["passed"] for row in rows)
    payload = {
        "schema": SUITE_SCHEMA,
        "suite": suite,
        "status": "ok" if passed else "blocked_or_failed",
        "passed": passed,
        "commit": current_git_commit(repo_root=root),
        "repeats": repeats,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "rehearsal_steps": rehearsal_steps,
        "maximum_relative_spread": maximum_relative_spread,
        "neighbor_backend": neighbor_backend or "case_contract",
        "cases": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def profile_suite(
    *,
    repo_root: str | Path,
    out: str | Path,
    suite: str = DEFAULT_LOCAL_SUITE,
    case_ids: Sequence[str] = (),
    prepared_overrides: Mapping[str, str | Path] | None = None,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    measured_steps: int = DEFAULT_MEASURED_STEPS,
    neighbor_backend: str | None = None,
    registry_path: str | Path = CASE_REGISTRY_PATH,
    runner: Callable[..., dict[str, Any]] = runtime_payload,
) -> dict[str, Any]:
    """Persist a clean control and synchronized MD stage map for each case.

    The synchronized route profiler intentionally changes MLX scheduling. Its
    timings attribute exclusive stage ownership and must not be reported as
    clean trajectory throughput. Running the clean control immediately before
    the instrumented sample keeps both views in one auditable artifact.
    """

    if warmup_steps <= 0 or measured_steps < 2:
        raise ValueError("warmup_steps must be positive and measured_steps must be >= 2")
    root = Path(repo_root).resolve()
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = root / out_path
    registry = load_case_registry(registry_path)
    selected = resolve_cases(registry=registry, suite=suite, case_ids=case_ids)
    overrides = dict(prepared_overrides or {})
    unknown_overrides = sorted(set(overrides) - {case.case_id for case in selected})
    if unknown_overrides:
        raise ValueError(f"prepared overrides do not match selected cases: {unknown_overrides}")

    raw_root = out_path.parent / f"{out_path.stem}-raw"
    rows = []
    for case in selected:
        prepared = Path(overrides.get(case.case_id, case.prepared_path))
        if not prepared.is_absolute():
            prepared = root / prepared
        selected_backend = neighbor_backend or case.neighbor_backend
        case_root = raw_root / case.case_id
        clean = runner(
            prepared=prepared,
            warmups=warmup_steps,
            steps=measured_steps,
            out=case_root / "clean.json",
            neighbor_backend=selected_backend,
            runtime_profile=False,
        )
        instrumented = runner(
            prepared=prepared,
            warmups=warmup_steps,
            steps=measured_steps,
            out=case_root / "instrumented.json",
            neighbor_backend=selected_backend,
            runtime_profile=True,
        )
        rows.append(
            _stage_profile_case(
                case=case,
                prepared=prepared,
                neighbor_backend=selected_backend,
                clean=clean,
                instrumented=instrumented,
                clean_output=case_root / "clean.json",
                instrumented_output=case_root / "instrumented.json",
            )
        )

    passed = all(row["passed"] for row in rows)
    payload = {
        "schema": STAGE_PROFILE_SCHEMA,
        "suite": suite,
        "status": "ok" if passed else "blocked_or_failed",
        "passed": passed,
        "commit": current_git_commit(repo_root=root),
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "neighbor_backend": neighbor_backend or "case_contract",
        "profile_semantics": {
            "clean": "end_to_end_throughput_control",
            "instrumented": "synchronized_exclusive_stage_attribution",
            "instrumented_preserves_production_constraint_route": True,
            "instrumented_preserves_lazy_force_schedule": False,
            "final_state_comparison": (
                "diagnostic_only because synchronized floating-point execution can "
                "separate chaotic trajectories over long profiles"
            ),
            "warning": (
                "instrumented timings introduce completion barriers and are not "
                "clean throughput measurements"
            ),
        },
        "cross_case_stage_ranking": _cross_case_stage_ranking(rows),
        "cases": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _stage_profile_case(
    *,
    case: MDBenchmarkCase,
    prepared: Path,
    neighbor_backend: str,
    clean: Mapping[str, Any],
    instrumented: Mapping[str, Any],
    clean_output: Path,
    instrumented_output: Path,
) -> dict[str, Any]:
    blockers = []
    if clean.get("passed") is not True:
        blockers.append("clean_control_failed")
        blockers.extend(f"clean:{item}" for item in clean.get("blockers", ()))
    if instrumented.get("passed") is not True:
        blockers.append("instrumented_profile_failed")
        blockers.extend(f"instrumented:{item}" for item in instrumented.get("blockers", ()))
    atom_counts = {
        int(payload["atom_count"]) for payload in (clean, instrumented) if "atom_count" in payload
    }
    if atom_counts != {case.expected_atom_count}:
        blockers.append(
            f"atom_count_mismatch:expected={case.expected_atom_count}:actual={sorted(atom_counts)}"
        )
    if clean.get("hardware") != instrumented.get("hardware"):
        blockers.append("hardware_mismatch")
    if clean.get("runtime") != instrumented.get("runtime"):
        blockers.append("runtime_mismatch")

    profile = instrumented.get("route_profile", {})
    if not isinstance(profile, Mapping) or profile.get("reconciled") is not True:
        blockers.append("route_profile_not_reconciled")
    routes = profile.get("routes", {}) if isinstance(profile, Mapping) else {}
    if not isinstance(routes, Mapping) or not routes:
        blockers.append("route_profile_missing_routes")
        routes = {}

    state_fields = (
        "potential_energy_kj_mol",
        "kinetic_energy_kj_mol",
        "total_energy_kj_mol",
        "temperature_k",
        "constraint_max_error_angstrom",
    )
    clean_state = clean.get("state", {})
    instrumented_state = instrumented.get("state", {})
    state_consistent = all(
        key in clean_state
        and key in instrumented_state
        and np.isclose(
            float(clean_state[key]),
            float(instrumented_state[key]),
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        for key in state_fields
    )

    instrumented_wall = float(profile.get("instrumented_wall_seconds", 0.0))
    stages: dict[str, dict[str, Any]] = {}
    for route_name, route_payload in routes.items():
        if not isinstance(route_payload, Mapping):
            blockers.append(f"invalid_route:{route_name}")
            continue
        stage_name = _route_stage(str(route_name))
        stage = stages.setdefault(
            stage_name,
            {
                "wall_seconds": 0.0,
                "completion_seconds": 0.0,
                "graph_and_host_seconds": 0.0,
                "route_names": [],
            },
        )
        for field in ("wall_seconds", "completion_seconds", "graph_and_host_seconds"):
            stage[field] += float(route_payload.get(field, 0.0))
        stage["route_names"].append(str(route_name))
    residual_seconds = max(0.0, float(profile.get("residual_seconds", 0.0)))
    if residual_seconds > 0.0:
        stages["unattributed_runtime"] = {
            "wall_seconds": residual_seconds,
            "completion_seconds": 0.0,
            "graph_and_host_seconds": residual_seconds,
            "route_names": [],
        }
    stage_rows = []
    for stage_name, values in stages.items():
        wall_seconds = float(values["wall_seconds"])
        stage_rows.append(
            {
                "stage": stage_name,
                **values,
                "instrumented_wall_fraction": (
                    0.0 if instrumented_wall <= 0.0 else wall_seconds / instrumented_wall
                ),
            }
        )
    stage_rows.sort(key=lambda row: (-row["wall_seconds"], row["stage"]))

    clean_seconds = clean.get("timings", {}).get("seconds_per_measured_step")
    instrumented_seconds = instrumented.get("timings", {}).get("seconds_per_measured_step")
    slowdown = (
        float(instrumented_seconds) / float(clean_seconds)
        if _positive_finite(clean_seconds) and _positive_finite(instrumented_seconds)
        else None
    )
    return {
        "case_id": case.case_id,
        "description": case.description,
        "role": case.role,
        "features": list(case.features),
        "prepared": str(prepared),
        "expected_atom_count": case.expected_atom_count,
        "contract_fingerprint": case.fingerprint,
        "prepared_fingerprint": _prepared_fingerprint(prepared),
        "neighbor_backend": neighbor_backend,
        "status": "ok" if not blockers else "blocked_or_failed",
        "passed": not blockers,
        "blockers": sorted(set(blockers)),
        "hardware": clean.get("hardware"),
        "runtime": clean.get("runtime"),
        "final_state_close": state_consistent,
        "clean_seconds_per_step": clean_seconds,
        "clean_ns_per_day": clean.get("throughput", {}).get("ns_per_day"),
        "instrumented_seconds_per_step": instrumented_seconds,
        "instrumentation_slowdown_ratio": slowdown,
        "instrumented_wall_seconds": instrumented_wall,
        "dominant_stage": None if not stage_rows else stage_rows[0]["stage"],
        "stages": stage_rows,
        "raw_outputs": {
            "clean": str(clean_output),
            "instrumented": str(instrumented_output),
        },
    }


def _route_stage(route_name: str) -> str:
    if route_name in {"neighbor_update_rebuild", "neighbor_force_binding"}:
        return "neighbor_lifecycle"
    if route_name in {"direct_spatial_tiles", "direct_lj_screened_coulomb"}:
        return "direct_nonbonded"
    if route_name == "reciprocal_pme":
        return "reciprocal_pme"
    if route_name == "pme_exceptions_corrections":
        return "pme_sparse_corrections"
    if route_name in {
        "bonded_fused",
        "other_force_terms",
        "urey_bradley",
        "charmm_cmap",
    }:
        return "bonded_and_other_forces"
    if route_name in {
        "force_term_aggregation",
        "pme_force_aggregation",
        "virtual_site_force_redistribution",
    }:
        return "force_aggregation_and_redistribution"
    if route_name == "integration_thermostat":
        return "integration_and_thermostat"
    if route_name == "diagnostics_reporting":
        return "diagnostics_and_reporting"
    if any(token in route_name for token in ("constraint", "settle", "shake", "rattle")):
        return "constraints"
    return "other_runtime"


def _cross_case_stage_ranking(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_stage: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        if row.get("passed") is not True:
            continue
        for stage in row.get("stages", ()):
            by_stage.setdefault(str(stage["stage"]), []).append(
                (str(row["case_id"]), float(stage["instrumented_wall_fraction"]))
            )
    ranking = []
    for stage_name, samples in by_stage.items():
        fractions = [fraction for _, fraction in samples]
        ranking.append(
            {
                "stage": stage_name,
                "profiled_case_count": len(samples),
                "median_instrumented_wall_fraction": statistics.median(fractions),
                "maximum_instrumented_wall_fraction": max(fractions),
                "case_instrumented_wall_fractions": {
                    case_id: fraction for case_id, fraction in sorted(samples)
                },
            }
        )
    ranking.sort(key=lambda row: (-row["median_instrumented_wall_fraction"], row["stage"]))
    return ranking


def compare_suites(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    minimum_jac_speedup: float = 0.03,
    maximum_5dfr_regression: float = 0.03,
) -> dict[str, Any]:
    """Compare matching suite payloads and apply the local two-case gate."""

    blockers: list[str] = []
    if baseline.get("schema") != SUITE_SCHEMA or candidate.get("schema") != SUITE_SCHEMA:
        blockers.append("suite_schema_mismatch")
    if baseline.get("neighbor_backend") != candidate.get("neighbor_backend"):
        blockers.append("neighbor_backend_mismatch")
    for name in (
        "repeats",
        "warmup_steps",
        "measured_steps",
        "rehearsal_steps",
        "maximum_relative_spread",
    ):
        if baseline.get(name) != candidate.get(name):
            blockers.append(f"{name}_mismatch")
    baseline_rows = {row.get("case_id"): row for row in baseline.get("cases", ())}
    candidate_rows = {row.get("case_id"): row for row in candidate.get("cases", ())}
    if set(baseline_rows) != set(candidate_rows):
        blockers.append("case_set_mismatch")

    rows = []
    for case_id in baseline_rows.keys() & candidate_rows.keys():
        before = baseline_rows[case_id]
        after = candidate_rows[case_id]
        row_blockers = []
        if before.get("contract_fingerprint") != after.get("contract_fingerprint"):
            row_blockers.append("contract_fingerprint_mismatch")
        if before.get("prepared_fingerprint") != after.get("prepared_fingerprint"):
            row_blockers.append("prepared_fingerprint_mismatch")
        if before.get("hardware") != after.get("hardware"):
            row_blockers.append("hardware_mismatch")
        if before.get("runtime") != after.get("runtime"):
            row_blockers.append("runtime_mismatch")
        if not before.get("passed") or not after.get("passed"):
            row_blockers.append("case_not_passed")
        baseline_seconds = before.get("median_seconds_per_step")
        candidate_seconds = after.get("median_seconds_per_step")
        speedup = None
        if (
            not row_blockers
            and _positive_finite(baseline_seconds)
            and _positive_finite(candidate_seconds)
        ):
            speedup = float(baseline_seconds) / float(candidate_seconds) - 1.0
        else:
            row_blockers.append("missing_finite_timing")
        rows.append(
            {
                "case_id": case_id,
                "baseline_seconds_per_step": baseline_seconds,
                "candidate_seconds_per_step": candidate_seconds,
                "speedup_fraction": speedup,
                "speedup_percent": None if speedup is None else speedup * 100.0,
                "eligible": not row_blockers,
                "blockers": sorted(set(row_blockers)),
            }
        )
    rows.sort(key=lambda row: row["case_id"])
    by_id = {row["case_id"]: row for row in rows}
    for required in ("dhfr-5dfr-pme", "jac-94k-pme"):
        if required not in by_id:
            blockers.append(f"missing_required_case:{required}")
    if not blockers:
        five = by_id["dhfr-5dfr-pme"]
        jac = by_id["jac-94k-pme"]
        if not five["eligible"]:
            blockers.append("dhfr-5dfr-pme:ineligible")
        elif five["speedup_fraction"] < -maximum_5dfr_regression:
            blockers.append("dhfr-5dfr-pme:regression")
        if not jac["eligible"]:
            blockers.append("jac-94k-pme:ineligible")
        elif jac["speedup_fraction"] < minimum_jac_speedup:
            blockers.append("jac-94k-pme:minimum_speedup_not_met")
    return {
        "schema": COMPARISON_SCHEMA,
        "status": "passed" if not blockers else "failed",
        "passed": not blockers,
        "baseline_commit": baseline.get("commit"),
        "candidate_commit": candidate.get("commit"),
        "minimum_jac_speedup": minimum_jac_speedup,
        "maximum_5dfr_regression": maximum_5dfr_regression,
        "blockers": blockers,
        "cases": rows,
    }


def _run_case(
    *,
    case: MDBenchmarkCase,
    prepared: Path,
    out_dir: Path,
    repeats: int,
    warmup_steps: int,
    measured_steps: int,
    rehearsal_steps: int,
    maximum_relative_spread: float,
    neighbor_backend: str,
    runner: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    prepared_fingerprint = _prepared_fingerprint(prepared)
    samples = []
    blockers = []
    rehearsal_output = out_dir / "rehearsal.json"
    rehearsal = runner(
        prepared=prepared,
        warmups=warmup_steps,
        steps=rehearsal_steps,
        out=rehearsal_output,
        neighbor_backend=neighbor_backend,
    )
    if not rehearsal.get("passed", False):
        blockers.append("rehearsal_failed")
        blockers.extend(str(item) for item in rehearsal.get("blockers", ()))
    for repeat in range(1, repeats + 1):
        sample = runner(
            prepared=prepared,
            warmups=warmup_steps,
            steps=measured_steps,
            out=out_dir / f"repeat-{repeat:02d}.json",
            neighbor_backend=neighbor_backend,
        )
        samples.append(sample)
        if not sample.get("passed", False):
            blockers.extend(str(item) for item in sample.get("blockers", ()))
    atom_counts = {int(sample["atom_count"]) for sample in samples if "atom_count" in sample}
    if atom_counts and atom_counts != {case.expected_atom_count}:
        blockers.append(
            f"atom_count_mismatch:expected={case.expected_atom_count}:actual={sorted(atom_counts)}"
        )
    timings = [
        float(sample["timings"]["seconds_per_measured_step"])
        for sample in samples
        if sample.get("passed")
        and _positive_finite(sample.get("timings", {}).get("seconds_per_measured_step"))
    ]
    throughputs = [
        float(sample["throughput"]["ns_per_day"])
        for sample in samples
        if sample.get("passed") and _positive_finite(sample.get("throughput", {}).get("ns_per_day"))
    ]
    hardware_values = [sample.get("hardware") for sample in samples]
    runtime_values = [sample.get("runtime") for sample in samples]
    if any(value != hardware_values[0] for value in hardware_values[1:]):
        blockers.append("hardware_changed_between_repeats")
    if any(value != runtime_values[0] for value in runtime_values[1:]):
        blockers.append("runtime_changed_between_repeats")
    median_seconds = statistics.median(timings) if timings else None
    relative_spread = (
        (max(timings) - min(timings)) / median_seconds
        if median_seconds is not None and len(timings) > 1
        else 0.0
        if median_seconds is not None
        else None
    )
    if relative_spread is not None and relative_spread > maximum_relative_spread:
        blockers.append(
            "timing_spread_exceeded:"
            f"actual={relative_spread:.6f}:maximum={maximum_relative_spread:.6f}"
        )
    passed = len(timings) == repeats and len(throughputs) == repeats and not blockers
    return {
        "case_id": case.case_id,
        "description": case.description,
        "role": case.role,
        "neighbor_backend": neighbor_backend,
        "features": list(case.features),
        "prepared": str(prepared),
        "expected_atom_count": case.expected_atom_count,
        "contract_fingerprint": case.fingerprint,
        "prepared_fingerprint": prepared_fingerprint,
        "hardware": hardware_values[0] if hardware_values else None,
        "runtime": runtime_values[0] if runtime_values else None,
        "status": "ok" if passed else "blocked_or_failed",
        "passed": passed,
        "blockers": sorted(set(blockers)),
        "rehearsal_passed": rehearsal.get("passed", False),
        "rehearsal_output": str(rehearsal_output),
        "sample_count": len(samples),
        "median_seconds_per_step": median_seconds,
        "median_ns_per_day": statistics.median(throughputs) if throughputs else None,
        "relative_timing_spread": relative_spread,
        "seconds_per_step_samples": timings,
        "ns_per_day_samples": throughputs,
        "raw_outputs": [
            str(out_dir / f"repeat-{repeat:02d}.json") for repeat in range(1, repeats + 1)
        ],
    }


def _positive_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _prepared_fingerprint(path: Path) -> str | None:
    metadata_path = path / "prepared_system.json"
    arrays_path = path / "prepared_system.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("created_at", None)
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    with np.load(arrays_path, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            array = np.ascontiguousarray(archive[name])
            digest.update(name.encode())
            digest.update(array.dtype.str.encode())
            digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _prepared_overrides(values: Sequence[str]) -> dict[str, Path]:
    overrides = {}
    for value in values:
        case_id, separator, path = value.partition("=")
        if not separator or not case_id or not path:
            raise argparse.ArgumentTypeError("--prepared must use CASE_ID=PATH")
        overrides[case_id] = Path(path)
    return overrides


def main(argv: list[str] | None = None) -> int:
    """Run the canonical suite command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    list_parser = commands.add_parser("list", help="list registered cases and availability")
    list_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    run_parser = commands.add_parser("run", help="run one named suite or explicit cases")
    run_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    run_parser.add_argument("--suite", default=DEFAULT_LOCAL_SUITE)
    run_parser.add_argument("--case", action="append", default=[])
    run_parser.add_argument("--prepared", action="append", default=[])
    run_parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    run_parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    run_parser.add_argument("--measured-steps", type=int, default=DEFAULT_MEASURED_STEPS)
    run_parser.add_argument("--rehearsal-steps", type=int, default=DEFAULT_REHEARSAL_STEPS)
    run_parser.add_argument(
        "--maximum-relative-spread",
        type=float,
        default=DEFAULT_MAXIMUM_RELATIVE_SPREAD,
    )
    run_parser.add_argument(
        "--neighbor-backend",
        choices=("mlx_cell_pairs", "mlx_cell_tiles"),
        default=None,
        help="override each case's committed backend contract",
    )
    run_parser.add_argument("--out", type=Path, required=True)
    profile_parser = commands.add_parser(
        "profile",
        help="run clean controls and synchronized whole-step stage attribution",
    )
    profile_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    profile_parser.add_argument("--suite", default=DEFAULT_LOCAL_SUITE)
    profile_parser.add_argument("--case", action="append", default=[])
    profile_parser.add_argument("--prepared", action="append", default=[])
    profile_parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    profile_parser.add_argument("--measured-steps", type=int, default=DEFAULT_MEASURED_STEPS)
    profile_parser.add_argument(
        "--neighbor-backend",
        choices=("mlx_cell_pairs", "mlx_cell_tiles"),
        default=None,
        help="override each case's committed backend contract",
    )
    profile_parser.add_argument("--out", type=Path, required=True)
    compare_parser = commands.add_parser("compare", help="compare baseline and candidate runs")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--minimum-jac-speedup", type=float, default=0.03)
    compare_parser.add_argument("--maximum-5dfr-regression", type=float, default=0.03)
    compare_parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.command == "list":
        payload = case_inventory(repo_root=args.repo_root)
    elif args.command == "run":
        payload = run_suite(
            repo_root=args.repo_root,
            out=args.out,
            suite=args.suite,
            case_ids=args.case,
            prepared_overrides=_prepared_overrides(args.prepared),
            repeats=args.repeats,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
            rehearsal_steps=args.rehearsal_steps,
            maximum_relative_spread=args.maximum_relative_spread,
            neighbor_backend=args.neighbor_backend,
        )
    elif args.command == "profile":
        payload = profile_suite(
            repo_root=args.repo_root,
            out=args.out,
            suite=args.suite,
            case_ids=args.case,
            prepared_overrides=_prepared_overrides(args.prepared),
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
            neighbor_backend=args.neighbor_backend,
        )
    else:
        payload = compare_suites(
            json.loads(args.baseline.read_text()),
            json.loads(args.candidate.read_text()),
            minimum_jac_speedup=args.minimum_jac_speedup,
            maximum_5dfr_regression=args.maximum_5dfr_regression,
        )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CASE_REGISTRY_PATH",
    "COMPARISON_SCHEMA",
    "MDBenchmarkCase",
    "STAGE_PROFILE_SCHEMA",
    "SUITE_SCHEMA",
    "case_inventory",
    "compare_suites",
    "load_case_registry",
    "main",
    "profile_suite",
    "resolve_cases",
    "run_suite",
]
