"""Runtime probes for the local MLX environment."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version


@dataclass(frozen=True)
class RuntimeInfo:
    """Small, display-friendly summary of the active MLX runtime."""

    mlx_version: str
    default_device: str
    metal_available: bool


def get_runtime_info() -> RuntimeInfo:
    """Return basic information about MLX and the default device."""

    import mlx.core as mx

    return RuntimeInfo(
        mlx_version=version("mlx"),
        default_device=str(mx.default_device()),
        metal_available=bool(mx.metal.is_available()),
    )
