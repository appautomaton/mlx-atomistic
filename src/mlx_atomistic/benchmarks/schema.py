"""Shared normalized schema helpers for benchmark payloads."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

ENGINE_LABEL = "mlx_atomistic"
SCHEMA_VERSION = "benchmark-normalized-v1"

NORMALIZED_BENCHMARK_FIELDS: tuple[str, ...] = (
    "schema_version",
    "engine",
    "benchmark_name",
    "fixture",
    "system",
    "atom_count",
    "step_count",
    "evaluation_count",
    "timing_metric",
    "timing_value",
    "timing_unit",
    "hardware",
    "runtime",
    "finite",
    "status",
    "blocker",
    "command",
    "commit",
    "raw_output_path",
)

_TIMING_UNITS = {
    "mean_s": "s",
    "median_s": "s",
    "min_s": "s",
    "max_s": "s",
    "wall_s": "s",
    "ms_per_eval": "ms/eval",
    "neighbor_rebuild_ms_per_eval": "ms/eval",
    "neighbor_build_ms_per_eval": "ms/eval",
    "force_eval_ms_per_eval": "ms/eval",
    "force_eval_ms_per_step": "ms/step",
    "steps_per_s": "steps/s",
    "ps_per_s": "ps/s",
    "ns_per_day_at_dt_0_002": "ns/day",
}


def default_benchmark_command(module_name: str) -> str:
    """Return the default uv command for a benchmark module."""

    return f"uv run python -m mlx_atomistic.benchmarks.{module_name}"


def current_git_commit(*, repo_root: Path | None = None) -> str | None:
    """Return the current git commit hash when the checkout is available."""

    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _first_present(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    return None


def _is_finite_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _status_from_finite(*, finite: bool, blocker: str | None) -> str:
    if blocker:
        return "blocked"
    return "ok" if finite else "failed"


def normalize_benchmark_row(
    row: dict[str, Any],
    *,
    benchmark_name: str,
    timing_metric: str,
    timing_unit: str | None = None,
    engine: str = ENGINE_LABEL,
    fixture: str | None = None,
    atom_count: int | None = None,
    step_count: int | None = None,
    evaluation_count: int | None = None,
    status: str | None = None,
    blocker: str | None = None,
) -> dict[str, Any]:
    """Return a benchmark row augmented with normalized schema fields."""

    normalized = dict(row)
    normalized.setdefault("schema_version", SCHEMA_VERSION)
    normalized.setdefault("engine", engine)
    normalized.setdefault("benchmark_name", benchmark_name)
    if fixture is not None:
        normalized.setdefault("fixture", fixture)

    system = _first_present(normalized, ("system", "fixture", "case", "test"))
    normalized.setdefault("system", system)
    normalized["atom_count"] = _first_present(normalized, ("atom_count", "particles", "atoms"))
    if atom_count is not None and normalized["atom_count"] is None:
        normalized["atom_count"] = atom_count
    normalized["step_count"] = _first_present(normalized, ("step_count", "steps"))
    if step_count is not None and normalized["step_count"] is None:
        normalized["step_count"] = step_count
    normalized["evaluation_count"] = _first_present(
        normalized,
        ("evaluation_count", "evaluations", "iterations"),
    )
    if evaluation_count is not None and normalized["evaluation_count"] is None:
        normalized["evaluation_count"] = evaluation_count

    timing_value = normalized.get(timing_metric)
    normalized["timing_metric"] = timing_metric
    normalized["timing_value"] = timing_value
    normalized["timing_unit"] = timing_unit or _TIMING_UNITS.get(timing_metric)
    blocker_value = blocker if blocker is not None else normalized.get("blocker")
    finite = normalized.get("finite")
    if finite is None:
        finite = _is_finite_value(timing_value)
    normalized["finite"] = bool(finite)
    normalized["blocker"] = blocker_value
    normalized["status"] = status or normalized.get("status") or _status_from_finite(
        finite=bool(finite),
        blocker=None if blocker_value is None else str(blocker_value),
    )
    return normalized


def normalize_benchmark_payload(
    payload: dict[str, Any],
    *,
    benchmark_name: str,
    fixture: str | None,
    timing_metric: str,
    hardware: dict[str, Any],
    runtime: dict[str, Any],
    engine: str = ENGINE_LABEL,
    atom_count: int | None = None,
    step_count: int | None = None,
    evaluation_count: int | None = None,
    finite: bool | None = None,
    status: str | None = None,
    blocker: str | None = None,
    command: str | None = None,
    commit: str | None = None,
    raw_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a payload augmented with normalized benchmark provenance."""

    normalized = dict(payload)
    normalized.setdefault("schema_version", SCHEMA_VERSION)
    normalized.setdefault("engine", engine)
    normalized.setdefault("benchmark_name", benchmark_name)
    normalized.setdefault("fixture", fixture)
    normalized.setdefault("system", normalized.get("fixture"))
    normalized.setdefault("atom_count", atom_count)
    normalized.setdefault("step_count", step_count)
    normalized.setdefault("evaluation_count", evaluation_count)
    normalized.setdefault("timing_metric", timing_metric)
    normalized.setdefault("timing_value", None)
    normalized.setdefault("timing_unit", _TIMING_UNITS.get(timing_metric))
    normalized.setdefault("hardware", hardware)
    normalized.setdefault("runtime", runtime)
    blocker_value = blocker if blocker is not None else normalized.get("blocker")
    if finite is None:
        finite = False if blocker_value else all(
            bool(row.get("finite", True)) for row in normalized.get("cases", ())
        )
    normalized["finite"] = bool(finite)
    normalized["status"] = status or normalized.get("status") or _status_from_finite(
        finite=bool(finite),
        blocker=None if blocker_value is None else str(blocker_value),
    )
    normalized["blocker"] = blocker_value
    normalized.setdefault("command", command or default_benchmark_command(benchmark_name))
    normalized.setdefault("commit", commit if commit is not None else current_git_commit())
    normalized.setdefault(
        "raw_output_path",
        None if raw_output_path is None else str(raw_output_path),
    )
    return normalized
