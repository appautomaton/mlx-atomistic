"""Metal parity tests for the experimental 32-atom interaction engine."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.interaction_engine import (
    _build_interaction_schedule32,
    _build_owner_compute_schedule32,
    _fuse_interaction_halves32,
    _fused_half32_direct_force_only,
    _interaction32_direct_force_only,
    _owner_compute32_direct_force_only,
)
from mlx_atomistic.metal_kernels import _prepared_parameterized_pme_direct_force_only


@pytest.fixture(autouse=True)
def _on_gpu(monkeypatch):
    monkeypatch.setenv("MLX_ATOMISTIC_DEVICE", "gpu")
    previous_device = mx.default_device()
    try:
        gpu = mx.Device(mx.gpu, 0)
        mx.set_default_device(gpu)
        mx.set_default_stream(mx.new_stream(gpu))
        mx.eval(mx.array([1.0], dtype=mx.float32) + 1.0)
    except Exception:  # noqa: BLE001
        mx.set_default_device(previous_device)
        mx.set_default_stream(mx.new_stream(previous_device))
        pytest.skip("Metal GPU unavailable")
    yield
    mx.set_default_device(previous_device)
    mx.set_default_stream(mx.new_stream(previous_device))


@pytest.mark.gpu
@pytest.mark.parametrize("left_slice_size", (4, 8, 16))
def test_interaction32_force_matches_prepared_pair_oracle(left_slice_size):
    grid = np.stack(
        np.meshgrid(np.arange(4), np.arange(4), np.arange(6), indexing="ij"),
        axis=-1,
    )
    positions_np = 2.0 + 1.4 * grid.reshape((-1, 3)).astype(np.float32)
    atom_count = positions_np.shape[0]
    box_lengths = np.asarray([20.0, 20.0, 20.0], dtype=np.float32)
    cutoff = 4.0
    one_four_scale = 0.5
    base_schedule = _build_interaction_schedule32(
        positions_np,
        box_lengths,
        search_radius=4.5,
    )
    excluded = np.sort(
        np.asarray([[base_schedule.atom_order[0], base_schedule.atom_order[32]]]),
        axis=1,
    )
    one_four = np.sort(
        np.asarray([[base_schedule.atom_order[1], base_schedule.atom_order[33]]]),
        axis=1,
    )
    schedule = _build_interaction_schedule32(
        positions_np,
        box_lengths,
        search_radius=4.5,
        lj_exclusion_pairs=excluded,
        lj_one_four_pairs=one_four,
        left_slice_size=left_slice_size,
    )
    assert schedule.ordinary_tile_count > 0
    assert schedule.special_tile_count > schedule.block_count

    pair_i, pair_j = np.triu_indices(atom_count, k=1)
    pairs = np.stack((pair_i, pair_j), axis=1).astype(np.int32)
    lj_scales = np.ones((pairs.shape[0],), dtype=np.float32)
    codes = pairs[:, 0] * atom_count + pairs[:, 1]
    lj_scales[codes == atom_count * excluded[0, 0] + excluded[0, 1]] = 0.0
    lj_scales[codes == atom_count * one_four[0, 0] + one_four[0, 1]] = one_four_scale

    positions = mx.array(positions_np, dtype=mx.float32)
    box = mx.concatenate(
        (
            mx.array(box_lengths, dtype=mx.float32),
            1.0 / mx.array(box_lengths, dtype=mx.float32),
        )
    )
    half_sigma = mx.full((atom_count,), 0.55, dtype=mx.float32)
    sqrt_epsilon = mx.full((atom_count,), np.sqrt(0.2), dtype=mx.float32)
    charges = mx.array(np.linspace(-0.5, 0.5, atom_count), dtype=mx.float32)
    reference = _prepared_parameterized_pme_direct_force_only(
        positions,
        mx.array(pairs, dtype=mx.int32),
        box,
        half_sigma,
        sqrt_epsilon,
        charges,
        mx.array(lj_scales, dtype=mx.float32),
        cutoff=cutoff,
        shift=False,
        switch_distance=None,
        coulomb_constant=1389.35457644382,
        alpha=0.35,
    )
    for canonical_records in (False, True):
        observed = _interaction32_direct_force_only(
            positions,
            schedule,
            box,
            half_sigma,
            sqrt_epsilon,
            charges,
            cutoff=cutoff,
            shift=False,
            switch_distance=None,
            one_four_scale=one_four_scale,
            coulomb_constant=1389.35457644382,
            alpha=0.35,
            _canonical_records=canonical_records,
        )

        mx.eval(reference, observed)
        np.testing.assert_allclose(
            np.asarray(observed),
            np.asarray(reference),
            rtol=2.0e-5,
            atol=2.0e-3,
        )

    if left_slice_size == 16:
        fused_schedule = _fuse_interaction_halves32(schedule)
        fused_observed = _fused_half32_direct_force_only(
            positions,
            fused_schedule,
            box,
            half_sigma,
            sqrt_epsilon,
            charges,
            cutoff=cutoff,
            shift=False,
            switch_distance=None,
            one_four_scale=one_four_scale,
            coulomb_constant=1389.35457644382,
            alpha=0.35,
        )
        mx.eval(fused_observed)
        np.testing.assert_allclose(
            np.asarray(fused_observed),
            np.asarray(reference),
            rtol=2.0e-5,
            atol=2.0e-3,
        )

        owner_schedule = _build_owner_compute_schedule32(
            positions_np,
            box_lengths,
            search_radius=4.5,
            lj_exclusion_pairs=excluded,
            lj_one_four_pairs=one_four,
        )
        owner_observed = _owner_compute32_direct_force_only(
            positions,
            owner_schedule,
            box,
            half_sigma,
            sqrt_epsilon,
            charges,
            cutoff=cutoff,
            shift=False,
            switch_distance=None,
            one_four_scale=one_four_scale,
            coulomb_constant=1389.35457644382,
            alpha=0.35,
        )
        mx.eval(owner_observed)
        np.testing.assert_allclose(
            np.asarray(owner_observed),
            np.asarray(reference),
            rtol=2.0e-5,
            atol=2.0e-3,
        )
