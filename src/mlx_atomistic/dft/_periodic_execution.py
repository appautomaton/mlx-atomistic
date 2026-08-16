"""Shared device-boundary and failure-isolation helpers for periodic DFT."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft._runtime_observer import RuntimeObserver, add_observed_work


def _materialize(
    observer: RuntimeObserver | None,
    *values: mx.array,
) -> None:
    """Evaluate one intentional device boundary and account for it."""

    if not values:
        return
    add_observed_work(
        observer,
        {
            "device_materializations": 1,
            "device_materialized_arrays": len(values),
        },
    )
    mx.eval(*values)


def _to_numpy(
    values: mx.array,
    *,
    dtype: np.dtype | type,
    observer: RuntimeObserver | None,
) -> np.ndarray:
    """Cross one already-bounded MLX array to NumPy with telemetry."""

    _materialize(observer, values)
    add_observed_work(
        observer,
        {
            "cpu_bridge_calls": 1,
            "cpu_bridge_elements": int(np.prod(values.shape)),
        },
    )
    return np.asarray(values, dtype=dtype)


def _detached_failure(error: Exception) -> Exception:
    """Copy a failure without retaining traceback frame state."""

    error_type: type[Exception]
    if type(error).__module__ == "builtins":
        error_type = type(error)
        prefix = ""
    else:
        error_type = RuntimeError
        prefix = f"{type(error).__name__}: "
    try:
        message = prefix + str(error)
    except Exception:
        message = prefix + "failure message unavailable"
    try:
        detached = error_type(message)
    except Exception:
        detached = RuntimeError(message)
    detached.__traceback__ = None
    detached.__context__ = None
    detached.__cause__ = None
    return detached
