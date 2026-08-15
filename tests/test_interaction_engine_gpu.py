"""Metal parity tests for the experimental 32-atom interaction engine."""

from __future__ import annotations

from math import ceil

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.core import Cell
from mlx_atomistic.forcefields import NonbondedPotential
from mlx_atomistic.interaction_engine import (
    _INTERACTION32_REBUILD_PROFILE_STAGES,
    _assemble_device_fused_half_schedule32,
    _build_device_block_geometry32,
    _build_device_ordinary_schedule32,
    _build_device_special_block_inventory32,
    _build_device_special_schedule32,
    _build_interaction_schedule32,
    _build_owner_compute_schedule32,
    _cell_atom_order,
    _fuse_interaction_halves32,
    _fused_half32_direct_force_only,
    _group_ordinary_tiles,
    _interaction32_direct_force_only,
    _Interaction32ScheduleCapacity,
    _make_atom_blocks,
    _normalize_pairs,
    _owner_compute32_direct_force_only,
    _profile_interaction32_rebuilds,
    _retry_device_fused_half_schedule32,
    _special_block_codes,
    _try_build_device_fused_half_schedule32,
)
from mlx_atomistic.md import LangevinThermostat, SimulationConfig, simulate_nvt
from mlx_atomistic.metal_kernels import _prepared_parameterized_pme_direct_force_only
from mlx_atomistic.neighbors import NeighborListManager, build_neighbor_list
from mlx_atomistic.pme import PMEConfig
from mlx_atomistic.topology import Topology


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


@pytest.mark.gpu
@pytest.mark.parametrize("atom_count", (1, 31, 32, 33, 257))
def test_device_block_geometry_matches_host_oracle(atom_count):
    """Device ordering and periodic block bounds match the host oracle."""

    rng = np.random.default_rng(79 + atom_count)
    box = np.asarray([20.0, 21.0, 22.0], dtype=np.float32)
    positions = rng.uniform(-2.0, 24.0, size=(atom_count, 3)).astype(np.float32)
    search_radius = 4.5
    observed = _build_device_block_geometry32(
        positions,
        box,
        search_radius=search_radius,
    )
    mx.eval(
        observed.atom_order,
        observed.inverse_order,
        observed.center_radius,
        observed.half_extent,
        observed.block_traversal,
    )

    expected_order = _cell_atom_order(positions, box, search_radius)
    (
        _expected_blocks,
        _expected_valid,
        expected_centers,
        expected_extents,
        expected_radii,
        expected_inverse,
    ) = _make_atom_blocks(positions, box, expected_order)
    expected_padded = np.full((observed.padded_atom_count,), -1, dtype=np.int32)
    expected_padded[:atom_count] = expected_order
    np.testing.assert_array_equal(np.asarray(observed.atom_order), expected_padded)
    np.testing.assert_array_equal(np.asarray(observed.inverse_order), expected_inverse)

    observed_center_radius = np.asarray(observed.center_radius)
    center_delta = observed_center_radius[:, :3] - expected_centers
    center_delta -= box * np.rint(center_delta / box)
    np.testing.assert_allclose(center_delta, 0.0, rtol=0.0, atol=2.0e-5)
    np.testing.assert_allclose(
        observed_center_radius[:, 3],
        expected_radii,
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        np.asarray(observed.half_extent),
        expected_extents,
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    np.testing.assert_array_equal(
        np.asarray(observed.block_traversal),
        np.argsort(np.sum(expected_extents, axis=1), kind="stable"),
    )


@pytest.mark.gpu
def test_device_special_block_inventory_matches_host_oracle():
    """Device topology flags identify the same unique special block pairs."""

    rng = np.random.default_rng(97)
    atom_count = 257
    box = np.asarray([20.0, 21.0, 22.0], dtype=np.float32)
    positions = rng.uniform(0.0, box, size=(atom_count, 3)).astype(np.float32)
    exclusions = np.asarray([[0, 1], [0, 40], [100, 200]], dtype=np.int32)
    one_four = np.asarray([[2, 3], [50, 120]], dtype=np.int32)
    geometry = _build_device_block_geometry32(
        positions,
        box,
        search_radius=4.5,
    )
    inventory = _build_device_special_block_inventory32(
        geometry,
        lj_exclusion_pairs=exclusions,
        lj_one_four_pairs=one_four,
    )
    mx.eval(
        geometry.atom_order,
        geometry.inverse_order,
        inventory.block_codes,
        inventory.block_code_unique,
        inventory.special_count,
    )

    expected = _special_block_codes(
        geometry.block_count,
        np.asarray(geometry.inverse_order),
        _normalize_pairs(exclusions, atom_count, "lj_exclusion_pairs"),
        _normalize_pairs(one_four, atom_count, "lj_one_four_pairs"),
    )
    observed = np.unique(np.asarray(inventory.block_codes)).astype(np.int64)
    np.testing.assert_array_equal(observed, expected)
    assert int(np.asarray(inventory.special_count)) == expected.shape[0]


@pytest.mark.gpu
def test_device_ordinary_schedule_matches_periodic_brute_force():
    """Two-pass device scatter preserves exact half-mode memberships."""

    rng = np.random.default_rng(101)
    atom_count = 257
    box = np.asarray([20.0, 21.0, 22.0], dtype=np.float32)
    positions = rng.uniform(0.0, box, size=(atom_count, 3)).astype(np.float32)
    positions[0] = [0.05, 10.0, 10.0]
    positions[1] = [19.95, 10.0, 10.0]
    search_radius = 4.5
    geometry = _build_device_block_geometry32(
        positions,
        box,
        search_radius=search_radius,
    )
    special = _build_device_special_block_inventory32(
        geometry,
        lj_exclusion_pairs=[[0, 1], [10, 100]],
        lj_one_four_pairs=[[20, 200]],
    )
    schedule = _build_device_ordinary_schedule32(
        positions,
        geometry,
        special,
    )
    mx.eval(
        geometry.atom_order,
        special.block_codes,
        schedule.mode_entry_counts,
        schedule.ordinary_left_blocks,
        schedule.ordinary_right_atoms,
        schedule.ordinary_half_modes,
        schedule.ordinary_group_starts,
        schedule.ordinary_group_counts,
    )

    atom_order = np.asarray(geometry.atom_order)
    block_traversal = np.asarray(geometry.block_traversal)
    special_codes = set(np.asarray(special.block_codes).tolist())
    expected: dict[tuple[int, int], int] = {}
    expected_counts = np.zeros((geometry.block_count, 3), dtype=np.int32)
    search_radius2 = search_radius * search_radius
    for traversal_index, left_block in enumerate(block_traversal):
        left_atoms = atom_order[32 * left_block : 32 * (left_block + 1)]
        valid_left = left_atoms >= 0
        left_positions = positions[np.maximum(left_atoms, 0)]
        for right_block in block_traversal[traversal_index + 1 :]:
            low_block = min(left_block, right_block)
            high_block = max(left_block, right_block)
            if low_block * geometry.block_count + high_block in special_codes:
                continue
            for right_slot in range(32):
                right_ordered = 32 * right_block + right_slot
                right_atom = atom_order[right_ordered]
                if right_atom < 0:
                    continue
                delta = left_positions - positions[right_atom]
                delta -= box * np.rint(delta / box)
                member = valid_left & (np.sum(delta * delta, axis=1) < search_radius2)
                mode = int(np.any(member[:16])) | (int(np.any(member[16:])) << 1)
                if mode:
                    expected[(left_block, right_ordered)] = mode
                    expected_counts[traversal_index, mode - 1] += 1

    observed: dict[tuple[int, int], int] = {}
    sentinel = geometry.padded_atom_count
    for left_block, mode, right_row in zip(
        np.asarray(schedule.ordinary_left_blocks),
        np.asarray(schedule.ordinary_half_modes),
        np.asarray(schedule.ordinary_right_atoms),
        strict=True,
    ):
        valid_rights = right_row[right_row < sentinel]
        assert np.all(right_row[valid_rights.shape[0] :] == sentinel)
        for right_ordered in valid_rights:
            key = (int(left_block), int(right_ordered))
            assert key not in observed
            observed[key] = int(mode)

    assert observed == expected
    np.testing.assert_array_equal(
        np.asarray(schedule.mode_entry_counts),
        expected_counts,
    )
    assert schedule.right_entry_count == len(expected)
    expected_starts, expected_group_counts = _group_ordinary_tiles(
        np.asarray(schedule.ordinary_left_blocks),
        np.asarray(schedule.ordinary_half_modes),
        3,
    )
    np.testing.assert_array_equal(
        np.asarray(schedule.ordinary_group_starts),
        expected_starts,
    )
    np.testing.assert_array_equal(
        np.asarray(schedule.ordinary_group_counts),
        expected_group_counts,
    )


@pytest.mark.gpu
def test_device_special_schedule_matches_topology_oracle():
    """Device special work preserves block order and exact topology masks."""

    rng = np.random.default_rng(103)
    atom_count = 257
    box = np.asarray([20.0, 21.0, 22.0], dtype=np.float32)
    positions = rng.uniform(0.0, box, size=(atom_count, 3)).astype(np.float32)
    exclusions = np.asarray([[0, 1], [0, 40], [100, 200]], dtype=np.int32)
    one_four = np.asarray([[2, 3], [50, 120]], dtype=np.int32)
    geometry = _build_device_block_geometry32(
        positions,
        box,
        search_radius=4.5,
    )
    inventory = _build_device_special_block_inventory32(
        geometry,
        lj_exclusion_pairs=exclusions,
        lj_one_four_pairs=one_four,
    )
    schedule = _build_device_special_schedule32(geometry, inventory)
    mx.eval(
        geometry.atom_order,
        schedule.special_blocks,
        schedule.special_work_left_blocks,
        schedule.special_work_left_slices,
        schedule.special_work_right_atoms,
        schedule.special_work_lj_enabled,
        schedule.special_work_lj_one_four,
        schedule.special_work_diagonal,
    )

    atom_order = np.asarray(geometry.atom_order)
    inverse_order = np.asarray(geometry.inverse_order)
    expected_codes = _special_block_codes(
        geometry.block_count,
        inverse_order,
        _normalize_pairs(exclusions, atom_count, "lj_exclusion_pairs"),
        _normalize_pairs(one_four, atom_count, "lj_one_four_pairs"),
    )
    expected_blocks = np.stack(
        (
            expected_codes // geometry.block_count,
            expected_codes % geometry.block_count,
        ),
        axis=1,
    ).astype(np.int32)
    observed_blocks = np.asarray(schedule.special_blocks)
    np.testing.assert_array_equal(observed_blocks, expected_blocks)
    assert schedule.special_tile_count == expected_blocks.shape[0]
    assert schedule.special_work_count == 2 * expected_blocks.shape[0]

    expected_left = np.repeat(expected_blocks[:, 0], 2)
    expected_slices = np.tile(np.asarray([0, 1], dtype=np.int32), expected_blocks.shape[0])
    expected_diagonal = np.repeat(
        (expected_blocks[:, 0] == expected_blocks[:, 1]).astype(np.int32),
        2,
    )
    np.testing.assert_array_equal(
        np.asarray(schedule.special_work_left_blocks),
        expected_left,
    )
    np.testing.assert_array_equal(
        np.asarray(schedule.special_work_left_slices),
        expected_slices,
    )
    np.testing.assert_array_equal(
        np.asarray(schedule.special_work_diagonal),
        expected_diagonal,
    )

    exclusion_set = {tuple(pair) for pair in np.sort(exclusions, axis=1)}
    one_four_set = {tuple(pair) for pair in np.sort(one_four, axis=1)}
    right_rows = np.asarray(schedule.special_work_right_atoms)[::2]
    enabled_rows = np.asarray(schedule.special_work_lj_enabled)[::2]
    one_four_rows = np.asarray(schedule.special_work_lj_one_four)[::2]
    for tile, (left_block, right_block) in enumerate(expected_blocks):
        left_atoms = atom_order[32 * left_block : 32 * (left_block + 1)]
        for right_slot in range(32):
            right_ordered = 32 * right_block + right_slot
            right_atom = atom_order[right_ordered]
            expected_right = (
                right_ordered if right_atom >= 0 else geometry.padded_atom_count
            )
            assert right_rows[tile, right_slot] == expected_right
            enabled_word = 0
            one_four_word = 0
            if right_atom >= 0:
                for left_slot, left_atom in enumerate(left_atoms):
                    if left_atom < 0 or left_atom == right_atom:
                        continue
                    pair = tuple(sorted((int(left_atom), int(right_atom))))
                    if pair not in exclusion_set:
                        enabled_word |= 1 << left_slot
                    if pair in one_four_set:
                        one_four_word |= 1 << left_slot
            assert int(enabled_rows[tile, right_slot]) == enabled_word
            assert int(one_four_rows[tile, right_slot]) == one_four_word

    np.testing.assert_array_equal(
        np.asarray(schedule.special_work_right_atoms)[0::2],
        np.asarray(schedule.special_work_right_atoms)[1::2],
    )
    np.testing.assert_array_equal(
        np.asarray(schedule.special_work_lj_enabled)[0::2],
        np.asarray(schedule.special_work_lj_enabled)[1::2],
    )
    np.testing.assert_array_equal(
        np.asarray(schedule.special_work_lj_one_four)[0::2],
        np.asarray(schedule.special_work_lj_one_four)[1::2],
    )


@pytest.mark.gpu
def test_device_built_fused_half32_force_matches_pair_oracle():
    """Combined device-built ordinary and special work preserves direct forces."""

    grid = np.stack(
        np.meshgrid(np.arange(4), np.arange(4), np.arange(6), indexing="ij"),
        axis=-1,
    )
    positions_np = 2.0 + 1.4 * grid.reshape((-1, 3)).astype(np.float32)
    atom_count = positions_np.shape[0]
    box_lengths = np.asarray([20.0, 20.0, 20.0], dtype=np.float32)
    cutoff = 4.0
    search_radius = 4.5
    one_four_scale = 0.5
    geometry = _build_device_block_geometry32(
        positions_np,
        box_lengths,
        search_radius=search_radius,
    )
    mx.eval(geometry.atom_order)
    atom_order = np.asarray(geometry.atom_order)
    excluded = np.sort(np.asarray([[atom_order[0], atom_order[32]]]), axis=1)
    one_four = np.sort(np.asarray([[atom_order[1], atom_order[33]]]), axis=1)
    inventory = _build_device_special_block_inventory32(
        geometry,
        lj_exclusion_pairs=excluded,
        lj_one_four_pairs=one_four,
    )
    ordinary = _build_device_ordinary_schedule32(
        positions_np,
        geometry,
        inventory,
    )
    special = _build_device_special_schedule32(geometry, inventory)
    schedule = _assemble_device_fused_half_schedule32(
        geometry,
        ordinary,
        special,
    )

    pair_i, pair_j = np.triu_indices(atom_count, k=1)
    pairs = np.stack((pair_i, pair_j), axis=1).astype(np.int32)
    codes = pairs[:, 0] * atom_count + pairs[:, 1]
    lj_scales = np.ones((pairs.shape[0],), dtype=np.float32)
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
    observed = _fused_half32_direct_force_only(
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
    )
    mx.eval(reference, observed)
    np.testing.assert_allclose(
        np.asarray(observed),
        np.asarray(reference),
        rtol=2.0e-5,
        atol=2.0e-3,
    )


@pytest.mark.gpu
def test_device_schedule_capacity_overflow_retry_and_generation_ownership():
    """Overflow stops before schedule publication and retry owns one generation."""

    grid = np.stack(
        np.meshgrid(np.arange(4), np.arange(4), np.arange(6), indexing="ij"),
        axis=-1,
    )
    positions_np = 2.0 + 1.4 * grid.reshape((-1, 3)).astype(np.float32)
    atom_count = positions_np.shape[0]
    box_lengths = np.asarray([20.0, 20.0, 20.0], dtype=np.float32)
    exclusions = np.asarray([[0, 40]], dtype=np.int32)
    one_four = np.asarray([[1, 41]], dtype=np.int32)
    initial = _try_build_device_fused_half_schedule32(
        positions_np,
        box_lengths,
        search_radius=4.5,
        capacity=None,
        generation_value=7,
        lj_exclusion_pairs=exclusions,
        lj_one_four_pairs=one_four,
    )
    assert initial.overflow
    assert initial.schedule is None
    assert initial.overflow_fields == (
        "ordinary_tiles",
        "ordinary_groups",
        "special_tiles",
    )
    logical_counts = (
        initial.inventory.ordinary_tile_count,
        initial.inventory.ordinary_group_count,
        initial.inventory.special_tile_count,
    )
    reserved_counts = (
        initial.recommended_capacity.ordinary_tiles,
        initial.recommended_capacity.ordinary_groups,
        initial.recommended_capacity.special_tiles,
    )
    for logical, reserved in zip(logical_counts, reserved_counts, strict=True):
        assert reserved >= ceil(1.25 * logical)
        assert reserved % 64 == 0

    admitted = _retry_device_fused_half_schedule32(initial)
    assert not admitted.overflow
    assert admitted.inventory is initial.inventory
    assert admitted.schedule is not None
    assert admitted.schedule.generation is not None
    assert admitted.schedule.generation.value == 7
    assert admitted.schedule.generation.capacity == initial.recommended_capacity
    active_schedule = admitted.schedule
    mx.eval(
        active_schedule.atom_order,
        active_schedule.ordinary_right_atoms,
        active_schedule.special_work_lj_enabled,
    )

    too_small = _Interaction32ScheduleCapacity(
        ordinary_tiles=max(0, initial.inventory.ordinary_tile_count - 1),
        ordinary_groups=initial.recommended_capacity.ordinary_groups,
        special_tiles=initial.recommended_capacity.special_tiles,
    )
    rejected = _try_build_device_fused_half_schedule32(
        positions_np,
        box_lengths,
        search_radius=4.5,
        capacity=too_small,
        generation_value=8,
        lj_exclusion_pairs=exclusions,
        lj_one_four_pairs=one_four,
    )
    assert rejected.overflow_fields == ("ordinary_tiles",)
    assert rejected.schedule is None
    assert active_schedule is admitted.schedule
    rejected_retry = _retry_device_fused_half_schedule32(
        rejected,
        capacity=too_small,
    )
    assert rejected_retry.overflow_fields == ("ordinary_tiles",)
    assert rejected_retry.schedule is None

    exact = _Interaction32ScheduleCapacity(
        ordinary_tiles=rejected.inventory.ordinary_tile_count,
        ordinary_groups=rejected.inventory.ordinary_group_count,
        special_tiles=rejected.inventory.special_tile_count,
    )
    exact_attempt = _try_build_device_fused_half_schedule32(
        positions_np,
        box_lengths,
        search_radius=4.5,
        capacity=exact,
        generation_value=8,
        lj_exclusion_pairs=exclusions,
        lj_one_four_pairs=one_four,
    )
    assert not exact_attempt.overflow
    assert exact_attempt.schedule is not None
    assert exact_attempt.schedule.generation is not None
    assert exact_attempt.schedule.generation.capacity == exact

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
    matched = _fused_half32_direct_force_only(
        positions,
        active_schedule,
        box,
        half_sigma,
        sqrt_epsilon,
        charges,
        cutoff=4.0,
        shift=False,
        switch_distance=None,
        one_four_scale=0.5,
        coulomb_constant=1389.35457644382,
        alpha=0.35,
        expected_generation=7,
    )
    mx.eval(matched)
    assert np.all(np.isfinite(np.asarray(matched)))
    with pytest.raises(ValueError, match="generation"):
        _fused_half32_direct_force_only(
            positions,
            active_schedule,
            box,
            half_sigma,
            sqrt_epsilon,
            charges,
            cutoff=4.0,
            shift=False,
            switch_distance=None,
            one_four_scale=0.5,
            coulomb_constant=1389.35457644382,
            alpha=0.35,
            expected_generation=8,
        )


@pytest.mark.gpu
def test_device_schedule_generation_fingerprints_topology_changes():
    """Generation metadata changes when canonical special topology changes."""

    rng = np.random.default_rng(107)
    positions = rng.uniform(0.0, 20.0, size=(257, 3)).astype(np.float32)
    box = np.asarray([20.0, 21.0, 22.0], dtype=np.float32)
    first = _try_build_device_fused_half_schedule32(
        positions,
        box,
        search_radius=4.5,
        capacity=None,
        generation_value=11,
        lj_exclusion_pairs=[[0, 1]],
        lj_one_four_pairs=[[2, 3]],
    )
    first = _retry_device_fused_half_schedule32(first)
    assert first.schedule is not None
    changed = _try_build_device_fused_half_schedule32(
        positions,
        box,
        search_radius=4.5,
        capacity=first.recommended_capacity,
        generation_value=12,
        lj_exclusion_pairs=[[0, 1], [10, 20]],
        lj_one_four_pairs=[[2, 3]],
    )
    if changed.overflow:
        changed = _retry_device_fused_half_schedule32(changed)
    assert changed.schedule is not None
    assert first.schedule.generation is not None
    assert changed.schedule.generation is not None
    assert first.schedule.generation.value == 11
    assert changed.schedule.generation.value == 12
    assert (
        first.schedule.generation.topology_digest
        != changed.schedule.generation.topology_digest
    )


@pytest.mark.gpu
def test_interaction32_rebuild_stage_profile_reconciles_exact_inventory():
    """Opt-in synchronized profiling accounts for complete manager rebuilds."""

    rng = np.random.default_rng(127)
    positions = rng.uniform(0.0, 20.0, size=(257, 3)).astype(np.float32)
    manager = NeighborListManager(
        Cell.orthorhombic([20.0, 20.0, 20.0]),
        cutoff=4.0,
        skin=0.5,
        backend="mlx_interaction32",
        interaction32_exclusion_pairs=[[0, 1], [40, 80]],
        interaction32_one_four_pairs=[[2, 3]],
    )

    with _profile_interaction32_rebuilds() as profiler:
        first = manager.rebuild(positions)
        first_topology = manager._interaction32_topology
        moved = positions.copy()
        moved[0, 0] += 0.01
        second = manager.rebuild(moved)
        assert manager._interaction32_topology is first_topology

    report = profiler.report()
    assert report["schema"] == "mlx_atomistic.interaction32_rebuild_profile.v1"
    assert report["backend"] == "mlx_interaction32"
    assert report["stage_order"] == list(_INTERACTION32_REBUILD_PROFILE_STAGES)
    assert report["rebuild_count"] == 2
    assert report["reconciled"] is True
    assert report["inventories"]["count"] == 2
    assert report["inventories"]["atom_count"]["median"] == 257.0
    assert report["inventories"]["overflow_retry_count"] == {
        "minimum": 0,
        "median": 0.5,
        "maximum": 1,
    }
    assert all(stage["count"] == 2 for stage in report["stages"].values())
    assert first.interaction32 is not None
    assert second.interaction32 is not None
    assert first._pairs is None
    assert second._pairs is None

    manager.interaction32_exclusion_pairs = [[0, 1], [5, 6], [40, 80]]
    changed = manager.rebuild(moved)
    assert manager._interaction32_topology is not first_topology
    assert changed.interaction32 is not None
    assert first.interaction32.generation is not None
    assert changed.interaction32.generation is not None
    assert (
        first.interaction32.generation.topology_digest
        != changed.interaction32.generation.topology_digest
    )

    with (
        _profile_interaction32_rebuilds(),
        pytest.raises(RuntimeError, match="cannot be nested"),
        _profile_interaction32_rebuilds(),
    ):
        pass


@pytest.mark.gpu
def test_fused_half32_nbfix_force_matches_production_tiles():
    """Fused 32-atom work applies the production NBFIX type table."""

    grid = np.stack(
        np.meshgrid(np.arange(4), np.arange(4), np.arange(6), indexing="ij"),
        axis=-1,
    )
    positions_np = 1.5 + 1.35 * grid.reshape((-1, 3)).astype(np.float32)
    atom_count = positions_np.shape[0]
    positions = mx.array(positions_np, dtype=mx.float32)
    box_lengths = np.asarray([20.0, 20.0, 20.0], dtype=np.float32)
    cell = Cell.orthorhombic(box_lengths)
    cutoff = 4.0
    skin = 0.5
    atom_types = np.asarray(["A", "B", "C"] * 32, dtype=str)
    potential = NonbondedPotential(
        sigma=np.linspace(0.9, 1.1, atom_count, dtype=np.float32),
        epsilon=np.linspace(0.12, 0.28, atom_count, dtype=np.float32),
        charges=np.linspace(-0.45, 0.45, atom_count, dtype=np.float32),
        cutoff=cutoff,
        lj_shift=True,
        switch_distance=3.2,
        electrostatics="pme",
        pme_config=PMEConfig(
            mesh_shape=(16, 16, 16),
            alpha=0.35,
            real_cutoff=cutoff,
        ),
        topology=Topology.from_sequences(
            n_atoms=atom_count,
            eager_nonbonded_pair_limit=0,
        ),
        atom_types=atom_types,
        nbfix_type_pairs=[("A", "B"), ("B", "C")],
        nbfix_type_sigma=[1.42, 0.82],
        nbfix_type_epsilon=[0.61, 0.47],
    ).bind_pme_plan(cell)
    neighbors = build_neighbor_list(
        positions,
        cell,
        cutoff=cutoff,
        skin=skin,
        sort_pairs=False,
        backend="mlx_cell_tiles",
    )
    assert neighbors.tiles is not None
    binding = potential._prepare_tile_force_binding(cell, None, neighbors.tiles)
    assert binding is not NotImplemented
    assert binding.tile_decline_reason is None
    assert binding.tile_atom_type_ids is not None
    assert binding.tile_nbfix_type_count == 3

    base_schedule = _build_interaction_schedule32(
        positions_np,
        box_lengths,
        search_radius=cutoff + skin,
        left_slice_size=16,
    )
    schedule = _fuse_interaction_halves32(base_schedule)
    reference = potential._direct_forces_from_binding(positions, binding)
    observed = _fused_half32_direct_force_only(
        positions,
        schedule,
        binding.box_lengths_and_inverses,
        binding.half_sigma,
        binding.sqrt_epsilon,
        potential.charges,
        cutoff=cutoff,
        shift=potential.lj_shift,
        switch_distance=potential.switch_distance,
        one_four_scale=potential.lj_one_four_scale,
        coulomb_constant=potential.coulomb_constant,
        alpha=potential.pme_config.alpha,
        atom_type_ids=binding.tile_atom_type_ids,
        nbfix_type_sigma=binding.tile_nbfix_type_sigma,
        nbfix_type_epsilon=binding.tile_nbfix_type_epsilon,
        nbfix_type_count=binding.tile_nbfix_type_count,
    )
    mx.eval(reference, observed)
    np.testing.assert_allclose(
        np.asarray(observed),
        np.asarray(reference),
        rtol=2.0e-5,
        atol=2.0e-3,
    )


@pytest.mark.gpu
def test_interaction32_neighbor_backend_tracks_moving_nvt_control():
    """The opt-in managed backend preserves a short moving PME trajectory."""

    grid = np.stack(
        np.meshgrid(np.arange(4), np.arange(4), np.arange(6), indexing="ij"),
        axis=-1,
    )
    positions_np = 1.5 + 1.35 * grid.reshape((-1, 3)).astype(np.float32)
    atom_count = positions_np.shape[0]
    velocities_np = np.linspace(
        -0.02,
        0.02,
        atom_count * 3,
        dtype=np.float32,
    ).reshape((atom_count, 3))
    cell = Cell.cubic(20.0)
    cutoff = 4.0
    skin = 0.5
    exclusions = np.asarray([[0, 1], [31, 32]], dtype=np.int32)
    one_four = np.asarray([[2, 5], [33, 37]], dtype=np.int32)
    potential = NonbondedPotential(
        sigma=np.linspace(0.9, 1.1, atom_count, dtype=np.float32),
        epsilon=np.linspace(0.12, 0.28, atom_count, dtype=np.float32),
        charges=np.linspace(-0.45, 0.45, atom_count, dtype=np.float32),
        cutoff=cutoff,
        electrostatics="pme",
        pme_config=PMEConfig(
            mesh_shape=(16, 16, 16),
            alpha=0.35,
            real_cutoff=cutoff,
        ),
        topology=Topology.from_sequences(
            n_atoms=atom_count,
            exclusions=exclusions,
            one_four_pairs=one_four,
            eager_nonbonded_pair_limit=0,
        ),
        lj_one_four_scale=0.5,
        coulomb_one_four_scale=0.8,
    ).bind_pme_plan(cell)
    config = SimulationConfig(
        dt=1.0e-4,
        steps=3,
        sample_interval=3,
        diagnostic_interval=3,
        pressure_diagnostics=False,
    )
    thermostat = LangevinThermostat(temperature=0.0, friction=0.0, seed=7)

    def run(backend):
        manager = NeighborListManager(
            cell,
            cutoff=cutoff,
            skin=skin,
            backend=backend,
            displacement_check_backend="mlx_scalar",
            interaction32_exclusion_pairs=potential._aligned_lj_exclusion_pairs,
            interaction32_one_four_pairs=potential._aligned_lj_one_four_pairs,
        )
        result = simulate_nvt(
            mx.array(positions_np),
            mx.array(velocities_np),
            cell=cell,
            force_terms=potential,
            neighbor_manager=manager,
            config=config,
            thermostat=thermostat,
        )
        mx.eval(result.final_state.positions, result.final_state.forces)
        return result, manager

    control, control_manager = run("mlx_cell_tiles")
    observed, observed_manager = run("mlx_interaction32")

    assert control_manager.rebuild_count == observed_manager.rebuild_count == 1
    assert observed_manager.neighbor_list is not None
    assert observed_manager.neighbor_list.interaction32 is not None
    assert observed_manager.neighbor_list.interaction32.generation is not None
    assert observed_manager.neighbor_list.interaction32.generation.value == 1
    assert observed_manager.neighbor_list.diagnostic_pairs_materialized is False
    assert observed_manager.neighbor_list._diagnostic_tiles is not None
    assert observed_manager.neighbor_list.supports_async_force_submission is True
    first_schedule = observed_manager.neighbor_list.interaction32
    np.testing.assert_allclose(
        np.asarray(observed.final_state.positions),
        np.asarray(control.final_state.positions),
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        np.asarray(observed.final_state.forces),
        np.asarray(control.final_state.forces),
        rtol=2.0e-5,
        atol=2.0e-3,
    )

    displaced = np.asarray(observed.final_state.positions).copy()
    displaced[0, 0] += 0.26
    rebuilt = observed_manager.update(mx.array(displaced))

    assert observed_manager.rebuild_count == 2
    assert rebuilt.interaction32 is not None
    assert rebuilt.interaction32.generation is not None
    assert rebuilt.interaction32.generation.value == 2
    assert first_schedule.generation is not None
    assert rebuilt.interaction32.generation.capacity.ordinary_tiles >= (
        first_schedule.generation.capacity.ordinary_tiles
    )
    assert rebuilt.interaction32.generation.capacity.ordinary_groups >= (
        first_schedule.generation.capacity.ordinary_groups
    )
    assert rebuilt.interaction32.generation.capacity.special_tiles >= (
        first_schedule.generation.capacity.special_tiles
    )

    wrong_topology_manager = NeighborListManager(
        cell,
        cutoff=cutoff,
        skin=skin,
        backend="mlx_interaction32",
        displacement_check_backend="mlx_scalar",
    )
    wrong_topology = wrong_topology_manager.update(mx.array(positions_np))
    assert wrong_topology.interaction32 is not None
    with pytest.raises(ValueError, match="topology does not match"):
        potential._prepare_interaction32_force_binding(
            cell,
            None,
            wrong_topology.interaction32,
        )
