"""Tests for bounded process-tree memory trace summaries."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# The bounded-process tracer reads macOS process memory through libproc, which
# it dlopens at import time. That is macOS-only by design, so this module skips
# on other platforms (the Linux CI runner) rather than failing collection.
if sys.platform != "darwin":
    pytest.skip(
        "bounded-process memory tracing uses the macOS libproc API",
        allow_module_level=True,
    )

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_bounded_process.py"
_SPEC = importlib.util.spec_from_file_location("run_bounded_process", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load bounded-process script")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_memory_trace_summary = _MODULE._memory_trace_summary


def _samples(values: list[int]) -> list[dict[str, float | int]]:
    return [
        {"elapsed_seconds": float(index), "physical_bytes": value, "process_count": 1}
        for index, value in enumerate(values)
    ]


def test_memory_trace_summary_accepts_a_stable_late_plateau() -> None:
    summary = _memory_trace_summary(
        _samples([1, 2, 3, 4, 1_000_000_000, 1_050_000_000, 1_060_000_000, 1_070_000_000])
    )

    assert summary["plateau_evaluated"] is True
    assert summary["plateau_passed"] is True
    assert summary["peak_physical_bytes"] == 1_070_000_000


def test_memory_trace_summary_rejects_large_late_growth() -> None:
    summary = _memory_trace_summary(
        _samples([1, 2, 3, 4, 1_000_000_000, 1_100_000_000, 2_000_000_000, 2_100_000_000])
    )

    assert summary["plateau_evaluated"] is True
    assert summary["plateau_passed"] is False
