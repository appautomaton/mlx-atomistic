"""Solve-boundary MLX allocator policy for periodic DFT."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import mlx.core as mx

_DFT_MEMORY_LIMIT_BYTES = 40_000_000_000
_DFT_CACHE_LIMIT_BYTES = 4_000_000_000


@contextmanager
def _bounded_dft_allocator() -> Iterator[None]:
    """Bound free Metal buffers for one SCF solve and restore caller policy."""

    if "gpu" not in str(mx.default_device()).lower():
        yield
        return
    try:
        mx.get_active_memory()
    except RuntimeError:
        yield
        return
    previous_memory = mx.set_memory_limit(_DFT_MEMORY_LIMIT_BYTES)
    try:
        previous_cache = mx.set_cache_limit(_DFT_CACHE_LIMIT_BYTES)
    except Exception:
        mx.set_memory_limit(previous_memory)
        raise
    try:
        yield
    finally:
        mx.set_cache_limit(previous_cache)
        mx.set_memory_limit(previous_memory)
