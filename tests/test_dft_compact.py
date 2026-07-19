from __future__ import annotations

import io

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.dft import (
    GTHProjectorChannel,
    PeriodicDavidsonConfig,
    PeriodicDFTSystem,
    PeriodicEigenResult,
    PeriodicGTHNonlocalOperator,
    PeriodicKohnShamOperator,
    PeriodicKPointResult,
    PeriodicSCFConfig,
    PeriodicSCFResult,
    PlaneWaveBasis,
    PseudopotentialData,
    PseudopotentialFormat,
    RealSpaceGrid,
    ReciprocalGrid,
    run_periodic_scf,
    solve_periodic_eigenproblem,
)
from mlx_atomistic.dft._compact import (
    _CompactBatch,
    _CompactLaneState,
    _remap_initial_coefficients,
)
from mlx_atomistic.dft._runtime_observer import RuntimeObserver
from mlx_atomistic.dft.kpoints import KPoint, KPointMesh
from mlx_atomistic.dft.periodic_gth import _GTHProjectorCache
from mlx_atomistic.dft.periodic_scf import _density_from_kpoints
from mlx_atomistic.dft.runtime_state import (
    fixed_density_state_metrics,
    serialize_fixed_density_state,
    serialize_periodic_scf_state,
)


def _basis(
    *,
    cutoff: float = 3.0,
    kpoint: tuple[float, float, float] = (0.25, 0.0, 0.0),
    lane_label: str = "test-lane",
) -> PlaneWaveBasis:
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    return PlaneWaveBasis.from_reduced_kpoint(
        grid,
        cutoff,
        kpoint,
        lane_label=lane_label,
    )


def _random_state(
    basis: PlaneWaveBasis,
    *,
    vectors: int = 3,
    seed: int = 42,
) -> _CompactLaneState:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(vectors, basis.active_count)) + 1j * rng.normal(
        size=(vectors, basis.active_count)
    )
    return basis._state_from_compact(mx.array(values.astype(np.complex64)))


def _hydrogen_gth() -> PseudopotentialData:
    return PseudopotentialData(
        element="H",
        format=PseudopotentialFormat.GTH,
        valence_charge=1.0,
        gth_rloc=0.25,
        gth_coefficients=(-1.0,),
        gth_channels=(GTHProjectorChannel(0, 0.3, ((0.5,),)),),
    )


def _small_eigen_result() -> tuple[PlaneWaveBasis, PeriodicEigenResult]:
    basis = _basis()
    operator = PeriodicKohnShamOperator(
        basis,
        mx.full(basis.grid.shape, 0.7),
    )
    result = solve_periodic_eigenproblem(
        operator,
        n_bands=3,
        config=PeriodicDavidsonConfig(
            max_iterations=12,
            tolerance=2e-5,
            max_subspace_size=12,
        ),
    )
    return basis, result


def test_compact_layout_order_identity_and_shared_reciprocal_metadata():
    first = _basis()
    second = _basis(lane_label="second-lane")
    moved = _basis(kpoint=(-0.25, 0.0, 0.0))
    wider = _basis(cutoff=4.0)

    indices = np.asarray(first.active_flat_indices)
    integer_g = np.asarray(first.active_integer_g)
    shared_integer_g = np.asarray(first.reciprocal_grid.integer_g).reshape((-1, 3))

    assert first.reciprocal_grid is second.reciprocal_grid
    assert np.all(np.diff(indices) > 0)
    assert np.unique(indices).size == first.active_count
    np.testing.assert_array_equal(integer_g, shared_integer_g[indices])
    assert first.basis_fingerprint == second.basis_fingerprint
    assert first.order_fingerprint == second.order_fingerprint
    assert first.lane_id != second.lane_id
    assert moved.basis_fingerprint != first.basis_fingerprint
    assert wider.order_fingerprint != first.order_fingerprint
    legacy_descriptor = ReciprocalGrid(
        first.grid,
        first.reciprocal_grid.vectors,
        first.reciprocal_grid.g2,
        first.reciprocal_grid.zero_mask,
    )
    np.testing.assert_array_equal(
        np.asarray(legacy_descriptor.integer_g),
        np.asarray(first.reciprocal_grid.integer_g),
    )
    assert legacy_descriptor.fingerprint == first.reciprocal_grid.fingerprint


def test_public_active_metadata_mutation_cannot_change_layout_kinetic_or_scatter():
    basis = _basis(cutoff=4.0, lane_label="metadata-mutation")
    state = _random_state(basis, vectors=2, seed=101)
    coordinates = basis.grid.coordinates()
    potential = 0.2 + 0.1 * mx.cos(
        2.0 * np.pi * coordinates[..., 0] / basis.grid.lengths[0]
    )
    operator = PeriodicKohnShamOperator(basis, potential)
    expected_action = operator._apply_compact(state)
    expected_full = state.full_grid_fresh()
    expected_mask = basis.mask
    mx.eval(expected_action.values, expected_full, expected_mask)
    expected_indices = np.asarray(basis.active_flat_indices).copy()
    expected_integer_g = np.asarray(basis.active_integer_g).copy()
    expected_kinetic = np.asarray(basis.active_kinetic_energies).copy()

    leaked_indices = basis.active_flat_indices
    leaked_integer_g = basis.active_integer_g
    leaked_kinetic = basis.active_kinetic_energies
    leaked_indices[:] = mx.zeros_like(leaked_indices)
    leaked_integer_g[:] = mx.zeros_like(leaked_integer_g)
    leaked_kinetic[:] = mx.full(leaked_kinetic.shape, 113.0)

    observed_action = operator._apply_compact(state)
    observed_full = state.full_grid_fresh()
    repacked, _ = basis._state_from_full(expected_full)
    np.testing.assert_allclose(
        np.asarray(observed_action.values),
        np.asarray(expected_action.values),
        atol=3e-6,
    )
    np.testing.assert_array_equal(np.asarray(observed_full), np.asarray(expected_full))
    np.testing.assert_allclose(
        np.asarray(repacked.values),
        np.asarray(state.values),
        atol=0.0,
    )
    np.testing.assert_array_equal(np.asarray(basis.mask), np.asarray(expected_mask))
    np.testing.assert_array_equal(
        np.asarray(basis.active_flat_indices),
        expected_indices,
    )
    np.testing.assert_array_equal(
        np.asarray(basis.active_integer_g),
        expected_integer_g,
    )
    np.testing.assert_array_equal(
        np.asarray(basis.active_kinetic_energies),
        expected_kinetic,
    )


def test_compact_scatter_gather_round_trip_handles_ragged_lanes():
    first = _basis(cutoff=2.5, lane_label="first")
    second = PlaneWaveBasis.from_reduced_kpoint(
        first.grid,
        2.5,
        (-0.25, 0.25, 0.0),
        reciprocal_grid=first.reciprocal_grid,
        lane_label="second",
    )
    states = (_random_state(first), _random_state(second, seed=7))
    batch = _CompactBatch.from_states(states)

    scattered = batch.scatter()
    gathered = batch.gather(scattered)
    restored = batch.unpad(gathered)

    assert batch.values.shape == (
        2,
        3,
        max(first.active_count, second.active_count),
    )
    for expected, observed in zip(states, restored, strict=True):
        np.testing.assert_allclose(np.asarray(observed.values), np.asarray(expected.values))
    for lane_index, basis in enumerate((first, second)):
        dense = np.asarray(scattered[lane_index]).reshape((3, -1))
        inactive = np.ones(basis.grid.size, dtype=bool)
        inactive[np.asarray(basis.active_flat_indices)] = False
        assert np.count_nonzero(dense[:, inactive]) == 0
        assert np.unique(np.asarray(batch.fft_indices[lane_index])).size == batch.bucket_size


def test_compact_batch_memory_rejects_unbounded_padding_and_transient_bytes():
    narrow = _basis(cutoff=0.5, lane_label="narrow")
    wide = PlaneWaveBasis.from_reduced_kpoint(
        narrow.grid,
        5.0,
        (-0.25, 0.25, 0.0),
        reciprocal_grid=narrow.reciprocal_grid,
        lane_label="wide",
    )
    narrow_state = _random_state(narrow, vectors=1)
    wide_state = _random_state(wide, vectors=1, seed=8)

    with pytest.raises(ValueError, match="padding-fraction cap"):
        _CompactBatch.from_states((narrow_state, wide_state))
    with pytest.raises(ValueError, match="transient byte budget"):
        _CompactBatch.from_states(
            (narrow_state,),
            max_transient_bytes=1,
        )


def test_singleton_compact_batch_keeps_private_indices_on_device(monkeypatch):
    basis = _basis(cutoff=4.0, lane_label="device-index-batch")
    state = _random_state(basis, vectors=2, seed=103)

    def reject_numpy_index_work(*args, **kwargs):
        raise AssertionError("singleton compact batch copied active indices through NumPy")

    monkeypatch.setattr(np, "arange", reject_numpy_index_work)
    monkeypatch.setattr(np, "setdiff1d", reject_numpy_index_work)
    monkeypatch.setattr(np, "stack", reject_numpy_index_work)

    batch = _CompactBatch.from_states((state,))
    scattered = batch.scatter()
    restored = batch.unpad(batch.gather(scattered))[0]
    monkeypatch.undo()

    np.testing.assert_allclose(
        np.asarray(restored.values),
        np.asarray(state.values),
        atol=0.0,
    )


def test_compact_multilane_local_action_uses_one_fft_pair(monkeypatch):
    first = _basis(lane_label="first")
    second = PlaneWaveBasis.from_reduced_kpoint(
        first.grid,
        3.0,
        first.kpoint_cartesian,
        reciprocal_grid=first.reciprocal_grid,
        lane_label="second",
    )
    states = (_random_state(first, vectors=4), _random_state(second, vectors=4, seed=9))
    batch = _CompactBatch.from_states(states)
    calls = {"fftn": 0, "ifftn": 0}
    original_fftn = mx.fft.fftn
    original_ifftn = mx.fft.ifftn

    def counted_fftn(*args, **kwargs):
        calls["fftn"] += 1
        assert kwargs["axes"] == (-3, -2, -1)
        return original_fftn(*args, **kwargs)

    def counted_ifftn(*args, **kwargs):
        calls["ifftn"] += 1
        assert kwargs["axes"] == (-3, -2, -1)
        return original_ifftn(*args, **kwargs)

    monkeypatch.setattr(mx.fft, "fftn", counted_fftn)
    monkeypatch.setattr(mx.fft, "ifftn", counted_ifftn)
    acted = batch.apply_local(mx.full(first.grid.shape, 1.25))
    restored = batch.unpad(acted)

    assert calls == {"fftn": 1, "ifftn": 1}
    for expected, observed in zip(states, restored, strict=True):
        np.testing.assert_allclose(
            np.asarray(observed.values),
            1.25 * np.asarray(expected.values),
            atol=3e-6,
        )


def test_explicit_integer_g_remap_rejects_hamiltonian_action():
    source = _basis(cutoff=2.5, lane_label="source")
    target = PlaneWaveBasis.from_reduced_kpoint(
        source.grid,
        4.0,
        (-0.25, 0.0, 0.0),
        reciprocal_grid=source.reciprocal_grid,
        lane_label="target",
    )
    values = np.arange(source.active_count, dtype=np.float32)[None, :].astype(
        np.complex64
    )
    state = source._state_from_compact(mx.array(values))

    remapped = _remap_initial_coefficients(state, target._layout)
    source_by_g = {
        tuple(int(value) for value in integer_g): values[0, index]
        for index, integer_g in enumerate(np.asarray(source.active_integer_g))
    }
    expected = np.asarray(
        [
            source_by_g.get(tuple(int(value) for value in integer_g), 0.0)
            for integer_g in np.asarray(target.active_integer_g)
        ],
        dtype=np.complex64,
    )

    np.testing.assert_array_equal(np.asarray(remapped.values[0]), expected)
    assert remapped.layout.lane_id == target.lane_id
    action = _CompactLaneState(state.values, state.layout, "hamiltonian_action")
    with pytest.raises(ValueError, match="cannot be remapped"):
        _remap_initial_coefficients(action, target._layout)
    with pytest.raises(ValueError, match="basis identity"):
        PeriodicKohnShamOperator(target, mx.zeros(target.grid.shape))._apply_compact(state)


def test_compact_gth_compatibility_matches_dense_formula_boundary():
    basis = _basis(cutoff=4.0)
    state = _random_state(basis, vectors=2)
    dense = state.full_grid_fresh()
    potential = mx.full(basis.grid.shape, 0.2)
    gth = PeriodicGTHNonlocalOperator(_hydrogen_gth(), basis, ((1.0, 2.0, 3.0),))
    operator = PeriodicKohnShamOperator(basis, potential, gth)

    observed_state = operator._apply_compact(state)
    observed = observed_state.full_grid_fresh()
    expected = basis.project(
        basis.apply_kinetic(dense)
        + basis.apply_local(dense, potential)
        + gth.apply(dense)
    )

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=2e-5)
    assert not {
        "mask",
        "shifted_vectors",
        "kinetic_energies",
        "fft_workspace",
    }.intersection(vars(basis))
    assert not {"dense_coefficients", "fft_workspace"}.intersection(vars(operator))


def test_gth_projector_vector_width_matches_independent_compact_oracle():
    basis = _basis(cutoff=4.0)
    state = _random_state(basis, vectors=3, seed=17)
    pseudo = _hydrogen_gth()
    position = np.asarray((1.0, 2.0, 3.0), dtype=np.float64)
    operator = PeriodicGTHNonlocalOperator(pseudo, basis, (position,))

    observed, metrics = operator._apply_compact(state)
    vectors = np.asarray(basis.active_shifted_vectors, dtype=np.float64)
    q = np.linalg.norm(vectors, axis=1)
    channel = pseudo.gth_channels[0]
    prefactor = (
        4.0
        * np.pi
        * np.pi**0.25
        * np.sqrt(2.0 * channel.radius**3 / basis.volume)
    )
    beta = (
        prefactor
        * np.exp(-0.5 * (q * channel.radius) ** 2)
        / np.sqrt(4.0 * np.pi)
        * np.exp(-1j * (vectors @ position))
    ).astype(np.complex64)
    coefficients = np.asarray(state.values)
    overlaps = np.conjugate(beta)[None, :] @ coefficients.T
    expected = (0.5 * overlaps).T @ beta[None, :]

    np.testing.assert_allclose(
        np.asarray(observed.values),
        expected,
        atol=3e-5,
    )
    assert metrics["projector_payload_elements"] == basis.active_count
    assert metrics["projector_elements_generated"] == basis.active_count
    assert metrics["projector_elements_loaded"] == 6 * basis.active_count
    assert all(
        entry.values.shape == (1, basis.active_count)
        for entry in operator._cache._entries.values()
    )


def test_public_shifted_vector_mutation_cannot_change_gth_regeneration():
    basis = _basis(cutoff=4.0, lane_label="gth-vector-mutation")
    state = _random_state(basis, vectors=2, seed=109)
    operator = PeriodicGTHNonlocalOperator(
        _hydrogen_gth(),
        basis,
        ((1.0, 2.0, 3.0),),
    )
    try:
        expected, _ = operator._apply_compact(state)
        mx.eval(expected.values)
        expected_vectors = np.asarray(basis.active_shifted_vectors).copy()
        leaked_vectors = basis.active_shifted_vectors
        leaked_vectors[:] = mx.zeros_like(leaked_vectors)
        operator._cache.clear()

        regenerated, metrics = operator._apply_compact(state)

        np.testing.assert_allclose(
            np.asarray(regenerated.values),
            np.asarray(expected.values),
            atol=3e-6,
        )
        np.testing.assert_array_equal(
            np.asarray(basis.active_shifted_vectors),
            expected_vectors,
        )
        assert metrics["projector_cache_misses"] == 1
    finally:
        operator.close()


def test_public_cell_matrix_mutation_cannot_change_gth_regeneration():
    basis = _basis(cutoff=4.0, lane_label="gth-volume-mutation")
    state = _random_state(basis, vectors=2, seed=113)
    operator = PeriodicGTHNonlocalOperator(
        _hydrogen_gth(),
        basis,
        ((1.0, 2.0, 3.0),),
        cache_budget_bytes=1,
    )
    original_volume = basis.volume
    try:
        expected, first_metrics = operator._apply_compact(state)
        mx.eval(expected.values)

        matrix = basis.grid.cell.matrix
        matrix[0, :] = 2.0 * matrix[0, :]
        mx.eval(matrix)

        regenerated, second_metrics = operator._apply_compact(state)

        assert basis.grid.volume == pytest.approx(2.0 * original_volume)
        assert basis.volume == original_volume
        np.testing.assert_allclose(
            np.asarray(regenerated.values),
            np.asarray(expected.values),
            atol=3e-6,
        )
        assert first_metrics["projector_cache_misses"] == 1
        assert second_metrics["projector_cache_misses"] == 1
        assert first_metrics["projector_elements_generated"] == basis.active_count
        assert second_metrics["projector_elements_generated"] == basis.active_count
        assert operator.cache_info()["entry_count"] == 0
    finally:
        operator.close()


def test_gth_projector_generation_count_cache_hits_and_hpsi_traffic():
    basis = _basis(cutoff=4.0)
    state = _random_state(basis, vectors=3, seed=23)
    gth = PeriodicGTHNonlocalOperator(
        _hydrogen_gth(),
        basis,
        ((1.0, 2.0, 3.0),),
    )
    observer = RuntimeObserver(synchronize=mx.synchronize)
    operator = PeriodicKohnShamOperator(
        basis,
        mx.full(basis.grid.shape, 0.2),
        gth,
        observer,
    )

    first = operator._apply_compact(state)
    second = operator._apply_compact(state)
    mx.eval(first.values, second.values)
    snapshot = observer.snapshot()
    work = snapshot["work_counters"]
    memory = snapshot["memory"]

    assert work["hpsi_calls"] == 2
    assert work["projector_cache_misses"] == 1
    assert work["projector_cache_hits"] == 1
    assert work["projector_elements_generated"] == basis.active_count
    assert work["projector_elements_loaded"] == 12 * basis.active_count
    assert work["projector_traffic_elements"] == 13 * basis.active_count
    assert memory["projector_payload_bytes"] == basis.active_count * 8
    assert memory["persistent_projector_bytes"] == basis.active_count * 8
    assert memory["peak_temporary_bytes"] >= memory["fft_workspace_bytes"]
    assert (
        memory["peak_temporary_bytes"]
        < _CompactBatch._DEFAULT_MAX_TRANSIENT_BYTES
    )


def test_gth_projector_cache_eviction_invalidation_and_context_lifetime():
    basis = _basis(cutoff=4.0)
    entry_bytes = basis.active_count * 8
    cache = _GTHProjectorCache(byte_budget=entry_bytes)
    state = _random_state(basis, vectors=1, seed=31)
    positions = np.asarray(
        ((1.0, 2.0, 3.0), (3.0, 2.0, 1.0)),
        dtype=np.float64,
    )
    first = PeriodicGTHNonlocalOperator(
        _hydrogen_gth(),
        basis,
        positions,
        cache=cache,
    )
    positions[0, 0] = 7.0

    action, metrics = first._apply_compact(state)

    assert first.positions[0, 0] == 1.0
    assert not first.positions.flags.writeable
    assert metrics["projector_cache_evictions"] == 0
    assert cache.entry_count == 1
    assert cache.current_bytes == entry_bytes
    assert cache.current_bytes <= cache.byte_budget
    assert all(not callable(entry) for entry in cache._entries.values())

    second = PeriodicGTHNonlocalOperator(
        _hydrogen_gth(),
        basis,
        ((1.5, 2.0, 3.0),),
        cache=cache,
    )
    assert cache.invalidations == 1
    assert cache.entry_count == 0
    assert cache.current_bytes == 0

    first._apply_compact(state)
    assert cache.invalidations == 2
    assert all(key[0] == first._context_identity for key in cache._entries)
    second._apply_compact(state)
    assert cache.invalidations == 3
    assert all(key[0] == second._context_identity for key in cache._entries)

    cache.close()
    assert cache.current_bytes == 0
    assert np.isfinite(np.asarray(action.values)).all()
    with pytest.raises(RuntimeError, match="closed"):
        cache.get(("closed",))


def test_gth_projector_cache_evicts_lru_only_between_evaluated_actions():
    first_basis = _basis(cutoff=4.0, kpoint=(0.25, 0.0, 0.0))
    second_basis = _basis(cutoff=4.0, kpoint=(-0.25, 0.0, 0.0))
    assert first_basis.active_count == second_basis.active_count
    cache = _GTHProjectorCache(byte_budget=first_basis.active_count * 8)
    first = PeriodicGTHNonlocalOperator(
        _hydrogen_gth(),
        first_basis,
        ((1.0, 2.0, 3.0),),
        cache=cache,
    )
    second = PeriodicGTHNonlocalOperator(
        _hydrogen_gth(),
        second_basis,
        ((1.0, 2.0, 3.0),),
        cache=cache,
    )

    first._apply_compact(_random_state(first_basis, vectors=1, seed=37))
    _, metrics = second._apply_compact(
        _random_state(second_basis, vectors=1, seed=41)
    )

    assert metrics["projector_cache_evictions"] == 1
    assert cache.evictions == 1
    assert cache.entry_count == 1
    assert next(iter(cache._entries))[1] == second_basis.basis_fingerprint


def test_scf_projector_cache_closes_when_runtime_context_raises(monkeypatch):
    import importlib

    periodic_scf = importlib.import_module("mlx_atomistic.dft.periodic_scf")
    captured: dict[str, _GTHProjectorCache] = {}

    def fail_with_cache(*args, projector_cache, **kwargs):
        del args, kwargs
        captured["cache"] = projector_cache
        _, inserted = projector_cache.put(
            ("held",),
            mx.ones((1, 1), dtype=mx.complex64),
        )
        assert inserted
        raise RuntimeError("injected SCF failure")

    monkeypatch.setattr(
        periodic_scf,
        "_run_periodic_scf_with_projector_cache",
        fail_with_cache,
    )
    with pytest.raises(RuntimeError, match="injected SCF failure"):
        periodic_scf.run_periodic_scf(
            object(),
            cutoff_hartree=1.0,
            kpoint_mesh=KPointMesh(
                [KPoint((0.0, 0.0, 0.0), coordinate_system="reduced")]
            ),
        )

    cache = captured["cache"]
    assert cache.current_bytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        cache.get(("held",))


def test_eigen_result_memory_owns_compact_state_and_fresh_public_values():
    basis, result = _small_eigen_result()

    assert result._compact_coefficients.values.shape == (3, basis.active_count)
    first = result.coefficients
    second = result.coefficients
    assert first is not second
    assert first.shape == (3, *basis.grid.shape)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    inactive = ~np.asarray(basis.mask)
    assert np.count_nonzero(np.asarray(first)[:, inactive]) == 0
    metrics = fixed_density_state_metrics(result=result, basis=basis)
    assert metrics == {
        "coefficient_payload_bytes": 3 * basis.active_count * 8,
        "full_grid_coefficient_bytes": 3 * basis.grid.size * 8,
    }


def test_legacy_eigen_result_constructor_compatibility_has_no_dense_cache():
    basis, result = _small_eigen_result()
    dense = result.coefficients
    reconstructed = PeriodicEigenResult(
        result.eigenvalues,
        dense,
        result.residuals,
        result.orthonormality_error,
        result.iterations,
        result.converged,
        result.subspace_size,
        result.restart_count,
    )

    assert reconstructed._basis is None
    assert reconstructed._compact_coefficients.values.ndim == 2
    assert reconstructed._compact_coefficients.values.shape[1] <= basis.active_count
    assert not {"coefficients", "dense_coefficients"}.intersection(vars(reconstructed))
    np.testing.assert_array_equal(
        np.asarray(reconstructed.coefficients),
        np.asarray(dense),
    )


def test_plane_wave_orthonormalize_single_input_shape_compatibility():
    basis = _basis()
    state = _random_state(basis, vectors=1)
    dense_single = state.full_grid_fresh()[0]

    orthonormal = basis.orthonormalize(dense_single)

    assert orthonormal.shape == (1, *basis.grid.shape)


def test_internal_density_continuation_and_serialization_never_read_public_adapter(
    monkeypatch,
):
    basis, result = _small_eigen_result()
    kpoint = PeriodicKPointResult(
        reduced_kpoint=(0.25, 0.0, 0.0),
        weight=1.0,
        basis=basis,
        eigen=result,
    )

    def forbidden(_self):
        raise AssertionError("internal path read public coefficients adapter")

    monkeypatch.setattr(PeriodicEigenResult, "coefficients", property(forbidden))
    density = _density_from_kpoints((kpoint,), occupation=2.0)
    metrics = fixed_density_state_metrics(result=result, basis=basis)
    continued = solve_periodic_eigenproblem(
        PeriodicKohnShamOperator(basis, mx.full(basis.grid.shape, 0.7)),
        n_bands=3,
        config=PeriodicDavidsonConfig(
            max_iterations=12,
            tolerance=2e-5,
            max_subspace_size=12,
        ),
        initial_coefficients=result._compact_coefficients,
    )
    fixed_payloads = serialize_fixed_density_state(
        {
            "result": result,
            "basis": basis,
            "density": density,
            "effective_local_potential": mx.full(basis.grid.shape, 0.7),
        }
    )
    scf_result = PeriodicSCFResult(
        converged=True,
        status="converged",
        iterations=1,
        total_energy=-1.0,
        electron_count=6.0,
        density_residual=0.0,
        energy_delta=0.0,
        density=density,
        kpoints=(kpoint,),
        energy_by_term={"total": -1.0},
        history=(),
        timings={"total": 1.0},
    )
    periodic_payloads = serialize_periodic_scf_state(scf_result)
    system = PeriodicDFTSystem(
        (6.0, 6.0, 6.0),
        (6, 6, 6),
        ((2.0, 3.0, 3.0), (4.0, 3.0, 3.0)),
        _hydrogen_gth(),
    )
    trajectory = run_periodic_scf(
        system,
        cutoff_hartree=2.5,
        kpoint_mesh=KPointMesh(
            [KPoint((0.0, 0.0, 0.0), weight=1.0, coordinate_system="reduced")]
        ),
        n_bands=1,
        config=PeriodicSCFConfig(
            max_iterations=2,
            min_iterations=2,
            density_tolerance=1e-8,
            energy_tolerance=1e-8,
            orbital_tolerance=2e-3,
            mixing_beta=0.5,
            mixer="linear",
            davidson=PeriodicDavidsonConfig(
                max_iterations=20,
                tolerance=2e-3,
                max_subspace_size=12,
            ),
        ),
    )

    assert continued.converged
    assert metrics["coefficient_payload_bytes"] == 3 * basis.active_count * 8
    assert trajectory.iterations == 2
    assert trajectory.timings["eigensolver"] > 0.0
    dense_fixed = np.load(io.BytesIO(fixed_payloads["coefficients.npy"]))
    dense_periodic = np.load(
        io.BytesIO(periodic_payloads["kpoints/0000-coefficients.npy"])
    )
    assert dense_fixed.shape == (3, *basis.grid.shape)
    np.testing.assert_array_equal(dense_fixed, dense_periodic)


def test_selected_56_grid_memory_coefficient_reduction_exceeds_tenfold():
    grid = RealSpaceGrid(
        (56, 56, 56),
        (10.261212861236006, 10.261212861236006, 10.261212861236006),
    )
    basis = PlaneWaveBasis.from_reduced_kpoint(
        grid,
        25.0,
        (-5.0 / 12.0, -5.0 / 12.0, -5.0 / 12.0),
    )
    compact_bytes = 16 * basis.active_count * 8
    dense_bytes = 16 * grid.size * 8

    assert basis.active_count == 6461
    assert compact_bytes == 827_008
    assert dense_bytes / compact_bytes >= 10.0
