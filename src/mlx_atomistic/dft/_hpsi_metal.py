"""Private Metal primitives for the compact periodic-DFT Hamiltonian boundary."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import mlx.core as mx

_SCATTER_SOURCE = r"""
    uint item = thread_position_in_grid.x;
    uint lane_count = (uint)shape[0];
    uint vector_count = (uint)shape[1];
    uint bucket_size = (uint)shape[2];
    uint fft_size = (uint)shape[3];
    uint compact_count = lane_count * vector_count * bucket_size;
    if (item >= compact_count) {
        return;
    }

    uint slot = item % bucket_size;
    uint vector_lane = item / bucket_size;
    uint lane = vector_lane / vector_count;
    uint lane_slot = lane * bucket_size + slot;
    if (valid_mask[lane_slot]) {
        uint destination = vector_lane * fft_size + (uint)fft_indices[lane_slot];
        full_grid[destination] = compact_values[item];
    }
"""

_GATHER_COMBINE_SOURCE = r"""
    uint item = thread_position_in_grid.x;
    uint lane_count = (uint)shape[0];
    uint vector_count = (uint)shape[1];
    uint bucket_size = (uint)shape[2];
    uint fft_size = (uint)shape[3];
    uint compact_count = lane_count * vector_count * bucket_size;
    if (item >= compact_count) {
        return;
    }

    uint slot = item % bucket_size;
    uint vector_lane = item / bucket_size;
    uint lane = vector_lane / vector_count;
    uint lane_slot = lane * bucket_size + slot;
    if (valid_mask[lane_slot]) {
        uint source = vector_lane * fft_size + (uint)fft_indices[lane_slot];
        combined[item] = compact_values[item] * complex64_t(kinetic[lane_slot])
            + reciprocal[source]
            + nonlocal_values[item];
    }
"""

_scatter_kernel = None
_gather_combine_kernel = None
_counter_lock = Lock()
_scatter_calls = 0
_gather_combine_calls = 0


@dataclass(frozen=True)
class _HpsiMetalCounters:
    """Count submitted private DFT Metal primitives."""

    scatter_calls: int
    gather_combine_calls: int


def _metal_device_selected() -> bool:
    """Return whether the current default MLX device is a GPU."""

    return "gpu" in str(mx.default_device()).lower()


def _hpsi_metal_counters() -> _HpsiMetalCounters:
    """Return a consistent snapshot of private-kernel submission counts."""

    with _counter_lock:
        return _HpsiMetalCounters(_scatter_calls, _gather_combine_calls)


def _reset_hpsi_metal_counters() -> None:
    """Reset private-kernel submission counts for isolated evidence runs."""

    global _scatter_calls, _gather_combine_calls
    with _counter_lock:
        _scatter_calls = 0
        _gather_combine_calls = 0


def _compact_boundary_supported(
    compact_values: mx.array,
    fft_indices: mx.array,
    valid_mask: mx.array,
) -> bool:
    """Return whether the common compact-boundary inputs are Metal eligible."""

    return (
        _metal_device_selected()
        and compact_values.dtype == mx.complex64
        and compact_values.ndim == 3
        and fft_indices.dtype == mx.int32
        and fft_indices.ndim == 2
        and valid_mask.dtype == mx.bool_
        and valid_mask.shape == fft_indices.shape
        and fft_indices.shape
        == (int(compact_values.shape[0]), int(compact_values.shape[2]))
        and all(int(size) > 0 for size in compact_values.shape)
    )


def _scatter_complex_zeros_metal(
    compact_values: mx.array,
    fft_indices: mx.array,
    valid_mask: mx.array,
    *,
    grid_size: int,
) -> mx.array | None:
    """Scatter compact complex values into an initialized-zero Metal workspace."""

    if type(grid_size) is not int or grid_size <= 0:
        msg = "Metal compact scatter grid_size must be a positive integer"
        raise ValueError(msg)
    if not _compact_boundary_supported(compact_values, fft_indices, valid_mask):
        return None
    lane_count, vector_count, bucket_size = (
        int(size) for size in compact_values.shape
    )
    shape = mx.array(
        [lane_count, vector_count, bucket_size, grid_size],
        dtype=mx.int32,
    )
    compact_count = lane_count * vector_count * bucket_size
    full_grid = _get_scatter_kernel()(
        inputs=[compact_values, fft_indices, valid_mask, shape],
        output_shapes=[(lane_count, vector_count, grid_size)],
        output_dtypes=[mx.complex64],
        grid=(compact_count, 1, 1),
        threadgroup=(min(256, compact_count), 1, 1),
        init_value=0,
    )[0]
    global _scatter_calls
    with _counter_lock:
        _scatter_calls += 1
    return full_grid


def _gather_combine_metal(
    reciprocal: mx.array,
    compact_values: mx.array,
    fft_indices: mx.array,
    valid_mask: mx.array,
    kinetic: mx.array,
    nonlocal_values: mx.array,
) -> mx.array | None:
    """Gather and combine compact Hamiltonian terms in one Metal dispatch."""

    if not _compact_boundary_supported(compact_values, fft_indices, valid_mask):
        return None
    lane_count, vector_count, bucket_size = (
        int(size) for size in compact_values.shape
    )
    if (
        reciprocal.dtype != mx.complex64
        or reciprocal.ndim != 3
        or reciprocal.shape[:2] != (lane_count, vector_count)
        or int(reciprocal.shape[2]) <= 0
        or kinetic.dtype != mx.float32
        or kinetic.shape != (lane_count, bucket_size)
        or nonlocal_values.dtype != mx.complex64
        or nonlocal_values.shape != compact_values.shape
    ):
        return None
    grid_size = int(reciprocal.shape[2])
    shape = mx.array(
        [lane_count, vector_count, bucket_size, grid_size],
        dtype=mx.int32,
    )
    compact_count = lane_count * vector_count * bucket_size
    combined = _get_gather_combine_kernel()(
        inputs=[
            reciprocal,
            compact_values,
            fft_indices,
            valid_mask,
            kinetic,
            nonlocal_values,
            shape,
        ],
        output_shapes=[compact_values.shape],
        output_dtypes=[mx.complex64],
        grid=(compact_count, 1, 1),
        threadgroup=(min(256, compact_count), 1, 1),
        init_value=0,
    )[0]
    global _gather_combine_calls
    with _counter_lock:
        _gather_combine_calls += 1
    return combined


def _get_scatter_kernel():
    """Return the lazily constructed compact-scatter Metal kernel."""

    global _scatter_kernel
    if _scatter_kernel is None:
        _scatter_kernel = mx.fast.metal_kernel(
            name="mlx_atomistic_dft_compact_scatter",
            input_names=["compact_values", "fft_indices", "valid_mask", "shape"],
            output_names=["full_grid"],
            source=_SCATTER_SOURCE,
        )
    return _scatter_kernel


def _get_gather_combine_kernel():
    """Return the lazily constructed gather/combine Metal kernel."""

    global _gather_combine_kernel
    if _gather_combine_kernel is None:
        _gather_combine_kernel = mx.fast.metal_kernel(
            name="mlx_atomistic_dft_gather_combine",
            input_names=[
                "reciprocal",
                "compact_values",
                "fft_indices",
                "valid_mask",
                "kinetic",
                "nonlocal_values",
                "shape",
            ],
            output_names=["combined"],
            source=_GATHER_COMBINE_SOURCE,
        )
    return _gather_combine_kernel
