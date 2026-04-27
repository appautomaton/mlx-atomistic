"""Small FFT wrapper for the DFT prototype."""

from __future__ import annotations

from typing import Literal

import mlx.core as mx
import numpy as np

FFTBackend = Literal["mlx", "numpy"]


def _mlx_fftn(field: mx.array, *, inverse: bool = False) -> mx.array | None:
    fft_module = getattr(mx, "fft", None)
    if fft_module is None:
        return None
    fn_name = "ifftn" if inverse else "fftn"
    fn = getattr(fft_module, fn_name, None)
    if fn is None:
        return None
    try:
        return fn(field, axes=(-3, -2, -1))
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None


def fft_backend() -> FFTBackend:
    """Return the backend that will be used by the FFT helpers."""

    probe = mx.zeros((2, 2, 2), dtype=mx.float32)
    result = _mlx_fftn(probe)
    if result is None:
        return "numpy"
    return "mlx"


def _numpy_fft(field: mx.array, *, inverse: bool = False) -> mx.array:
    field_np = np.array(field)
    fn = np.fft.ifftn if inverse else np.fft.fftn
    transformed = fn(field_np, axes=(-3, -2, -1)).astype(np.complex64)
    return mx.array(transformed)


def fft3(field: mx.array) -> mx.array:
    """Return the 3D FFT of a real or complex field."""

    field_mx = mx.array(field)
    transformed = _mlx_fftn(field_mx)
    if transformed is not None:
        return transformed
    return _numpy_fft(field_mx)


def ifft3(field_g: mx.array) -> mx.array:
    """Return the inverse 3D FFT of a reciprocal-space field."""

    field_mx = mx.array(field_g)
    transformed = _mlx_fftn(field_mx, inverse=True)
    if transformed is not None:
        return transformed
    return _numpy_fft(field_mx, inverse=True)


def real_to_reciprocal(field: mx.array) -> mx.array:
    """Alias for :func:`fft3` used by notebooks and docs."""

    return fft3(field)


def reciprocal_to_real(field_g: mx.array) -> mx.array:
    """Return the real component of an inverse 3D FFT."""

    return mx.real(ifft3(field_g))
