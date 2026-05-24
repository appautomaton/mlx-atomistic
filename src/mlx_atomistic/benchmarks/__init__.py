"""Benchmark entrypoints and shared benchmark helpers."""

from __future__ import annotations

import platform

from mlx_atomistic.benchmarks.schema import (
    ENGINE_LABEL,
    NORMALIZED_BENCHMARK_FIELDS,
    SCHEMA_VERSION,
    default_benchmark_command,
    normalize_benchmark_payload,
    normalize_benchmark_row,
)


def get_hardware_info() -> dict[str, str]:
    """Return stable host metadata for benchmark JSON payloads."""

    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


__all__ = [
    "ENGINE_LABEL",
    "NORMALIZED_BENCHMARK_FIELDS",
    "SCHEMA_VERSION",
    "default_benchmark_command",
    "get_hardware_info",
    "normalize_benchmark_payload",
    "normalize_benchmark_row",
]
