from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.dft import (
    GTHProjectorChannel,
    KPoint,
    KPointMesh,
    PeriodicDavidsonConfig,
    PeriodicDFTSystem,
    PeriodicGTHNonlocalOperator,
    PeriodicKohnShamOperator,
    PeriodicSCFConfig,
    PlaneWaveBasis,
    PseudopotentialData,
    PseudopotentialFormat,
    RealSpaceGrid,
    run_periodic_scf,
    solve_periodic_eigenproblem,
)
from mlx_atomistic.dft._compact import _CompactBatch
from mlx_atomistic.dft._runtime_observer import RuntimeObserver


def _restore_runtime(device: mx.Device, stream: mx.Stream) -> None:
    mx.set_default_stream(stream)
    mx.set_default_device(device)


@pytest.mark.gpu
def test_compact_batched_fft_executes_real_metal_op_and_restores_runtime(monkeypatch):
    monkeypatch.setenv("MLX_ATOMISTIC_DEVICE", "gpu")
    previous_device = mx.default_device()
    previous_stream = mx.default_stream(previous_device)
    try:
        metal_device = mx.Device(mx.gpu, 0)
        metal_stream = mx.new_stream(metal_device)
        mx.set_default_device(metal_device)
        mx.set_default_stream(metal_stream)
        probe = mx.array([1.0], dtype=mx.float32) + 1.0
        mx.eval(probe)

        grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
        first = PlaneWaveBasis.from_reduced_kpoint(
            grid,
            3.0,
            (0.25, 0.0, 0.0),
            lane_label="metal:first",
        )
        second = PlaneWaveBasis.from_reduced_kpoint(
            grid,
            3.0,
            (-0.25, 0.0, 0.0),
            reciprocal_grid=first.reciprocal_grid,
            lane_label="metal:second",
        )
        rng = np.random.default_rng(19)
        states = []
        for basis in (first, second):
            values = rng.normal(size=(3, basis.active_count)) + 1j * rng.normal(
                size=(3, basis.active_count)
            )
            states.append(basis._state_from_compact(mx.array(values.astype(np.complex64))))
        batch = _CompactBatch.from_states(states)
        round_trip = batch.from_real(batch.to_real())
        mx.eval(round_trip)
        mx.synchronize()

        for expected, observed in zip(
            states,
            batch.unpad(round_trip),
            strict=True,
        ):
            np.testing.assert_allclose(
                np.asarray(observed.values),
                np.asarray(expected.values),
                atol=3e-6,
            )
        assert mx.default_device() == metal_device
        assert mx.default_stream(metal_device) == metal_stream
    finally:
        _restore_runtime(previous_device, previous_stream)

    assert mx.default_device() == previous_device
    assert mx.default_stream(previous_device) == previous_stream


@pytest.mark.gpu
def test_metal_device_restore_and_stream_restore_after_failure(monkeypatch):
    monkeypatch.setenv("MLX_ATOMISTIC_DEVICE", "gpu")
    previous_device = mx.default_device()
    previous_stream = mx.default_stream(previous_device)

    with pytest.raises(RuntimeError, match="injected Metal test failure"):
        try:
            metal_device = mx.Device(mx.gpu, 0)
            mx.set_default_device(metal_device)
            mx.set_default_stream(mx.new_stream(metal_device))
            probe = mx.array([2.0], dtype=mx.float32) * 3.0
            mx.eval(probe)
            mx.synchronize()
            raise RuntimeError("injected Metal test failure")
        finally:
            _restore_runtime(previous_device, previous_stream)

    assert mx.default_device() == previous_device
    assert mx.default_stream(previous_device) == previous_stream


@pytest.mark.gpu
def test_compact_gth_hpsi_cache_executes_on_real_metal(monkeypatch):
    monkeypatch.setenv("MLX_ATOMISTIC_DEVICE", "gpu")
    previous_device = mx.default_device()
    previous_stream = mx.default_stream(previous_device)
    gth = None
    try:
        metal_device = mx.Device(mx.gpu, 0)
        mx.set_default_device(metal_device)
        mx.set_default_stream(mx.new_stream(metal_device))
        grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
        basis = PlaneWaveBasis.from_reduced_kpoint(
            grid,
            4.0,
            (0.25, 0.0, 0.0),
            lane_label="metal:gth",
        )
        pseudo = PseudopotentialData(
            element="H",
            format=PseudopotentialFormat.GTH,
            valence_charge=1.0,
            gth_rloc=0.25,
            gth_coefficients=(-1.0,),
            gth_channels=(GTHProjectorChannel(0, 0.3, ((0.5,),)),),
        )
        rng = np.random.default_rng(29)
        values = rng.normal(size=(4, basis.active_count)) + 1j * rng.normal(
            size=(4, basis.active_count)
        )
        state = basis._state_from_compact(mx.array(values.astype(np.complex64)))
        gth = PeriodicGTHNonlocalOperator(
            pseudo,
            basis,
            ((1.0, 2.0, 3.0),),
        )
        operator = PeriodicKohnShamOperator(
            basis,
            mx.full(grid.shape, 0.2),
            gth,
        )

        first = operator._apply_compact(state)
        second = operator._apply_compact(state)
        mx.eval(first.values, second.values)
        mx.synchronize()

        assert bool(mx.all(mx.isfinite(first.values)))
        assert gth.cache_info()["entry_count"] == 1
        assert gth.cache_info()["current_bytes"] == basis.active_count * 8
    finally:
        if gth is not None:
            gth.close()
        _restore_runtime(previous_device, previous_stream)

    assert mx.default_device() == previous_device
    assert mx.default_stream(previous_device) == previous_stream


@pytest.mark.gpu
def test_stable_hpsi_capacity_plateaus_across_logical_lane_counts_on_real_metal(
    monkeypatch,
):
    monkeypatch.setenv("MLX_ATOMISTIC_DEVICE", "gpu")
    previous_device = mx.default_device()
    previous_stream = mx.default_stream(previous_device)
    try:
        metal_device = mx.Device(mx.gpu, 0)
        mx.set_default_device(metal_device)
        mx.set_default_stream(mx.new_stream(metal_device))
        mx.synchronize()
        mx.clear_cache()
        grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
        basis = PlaneWaveBasis.from_reduced_kpoint(
            grid,
            4.0,
            (0.25, 0.0, 0.0),
            lane_label="metal:stable-capacity",
        )
        rng = np.random.default_rng(31)
        values = rng.normal(size=(3, basis.active_count)) + 1j * rng.normal(
            size=(3, basis.active_count)
        )
        state = basis._state_from_compact(mx.array(values.astype(np.complex64)))
        operator = PeriodicKohnShamOperator(
            basis,
            mx.full(grid.shape, 0.2),
        )
        cycle_cache = []
        submitted_shapes: set[tuple[int, ...]] = set()
        for _cycle in range(2):
            for logical_lanes in range(1, 4):
                prepared = _CompactBatch.from_states(
                    (state,) * logical_lanes,
                    lane_capacity=3,
                    vector_capacity=3,
                    active_capacity=basis.active_count,
                )
                outcome = PeriodicKohnShamOperator._apply_compact_batch(
                    (operator,) * logical_lanes,
                    (state,) * logical_lanes,
                    prepared_batch=prepared,
                )
                mx.eval(*(action.values for action in outcome.actions.values()))
                mx.synchronize()
                submitted_shapes.add(tuple(int(size) for size in prepared.values.shape))
            cycle_cache.append(int(mx.get_cache_memory()))

        assert submitted_shapes == {(3, 3, basis.active_count)}
        assert cycle_cache[0] > 0
        assert cycle_cache[1] - cycle_cache[0] <= 8 * 1024 * 1024
    finally:
        mx.clear_cache()
        _restore_runtime(previous_device, previous_stream)

    assert mx.default_device() == previous_device
    assert mx.default_stream(previous_device) == previous_stream


@pytest.mark.gpu
def test_incremental_davidson_reuses_paired_hv_on_real_metal(monkeypatch):
    monkeypatch.setenv("MLX_ATOMISTIC_DEVICE", "gpu")
    previous_device = mx.default_device()
    previous_stream = mx.default_stream(previous_device)
    try:
        metal_device = mx.Device(mx.gpu, 0)
        metal_stream = mx.new_stream(metal_device)
        mx.set_default_device(metal_device)
        mx.set_default_stream(metal_stream)
        grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
        basis = PlaneWaveBasis.from_reduced_kpoint(
            grid,
            4.0,
            (0.25, 0.125, 0.0),
            lane_label="metal:davidson",
        )
        coordinates = grid.coordinates()
        potential = 0.3 * mx.cos(2.0 * np.pi * coordinates[..., 0] / 8.0) + 0.1 * mx.sin(
            2.0 * np.pi * coordinates[..., 1] / 8.0
        )
        observer = RuntimeObserver(synchronize=mx.synchronize)
        operator = PeriodicKohnShamOperator(
            basis,
            potential,
            observer=observer,
        )
        submitted_shapes: set[tuple[int, ...]] = set()
        original_apply = PeriodicKohnShamOperator._apply_compact

        def recording_apply(self, coefficients, **kwargs):
            prepared = kwargs.get("prepared_batch")
            assert prepared is not None
            submitted_shapes.add(tuple(int(size) for size in prepared.values.shape))
            return original_apply(self, coefficients, **kwargs)

        monkeypatch.setattr(
            PeriodicKohnShamOperator,
            "_apply_compact",
            recording_apply,
        )

        result = solve_periodic_eigenproblem(
            operator,
            n_bands=2,
            config=PeriodicDavidsonConfig(
                max_iterations=5,
                tolerance=1e-9,
                max_subspace_size=8,
            ),
            observer=observer,
        )
        mx.eval(
            result.eigenvalues,
            result.residuals,
            result._compact_coefficients.values,
        )
        mx.synchronize()

        work = observer.snapshot()["work_counters"]
        assert bool(mx.all(mx.isfinite(result.eigenvalues)))
        assert work["davidson_hv_reused_vectors"] > 0
        assert work["projected_old_old_rebuilds"] == 0
        assert work["hpsi_vector_equivalents"] == (work["davidson_hv_new_vectors"] + 2)
        assert work["hpsi_calls"] == result.iterations + 1
        assert submitted_shapes == {(1, 2, basis.active_count)}
        final_event = [
            event
            for event in observer.snapshot()["events"]
            if event["event"] == "davidson_iteration"
        ][-1]
        assert final_event["residual_source"] == "direct_operator"
        assert mx.default_device() == metal_device
        assert mx.default_stream(metal_device) == metal_stream
    finally:
        _restore_runtime(previous_device, previous_stream)

    assert mx.default_device() == previous_device
    assert mx.default_stream(previous_device) == previous_stream


@pytest.mark.gpu
def test_representative_k_batch_scf_executes_on_real_metal(monkeypatch):
    monkeypatch.setenv("MLX_ATOMISTIC_DEVICE", "gpu")
    previous_device = mx.default_device()
    previous_stream = mx.default_stream(previous_device)
    try:
        metal_device = mx.Device(mx.gpu, 0)
        metal_stream = mx.new_stream(metal_device)
        mx.set_default_device(metal_device)
        mx.set_default_stream(metal_stream)
        pseudo = PseudopotentialData(
            element="H",
            format=PseudopotentialFormat.GTH,
            valence_charge=1.0,
            gth_rloc=0.25,
            gth_coefficients=(-1.0,),
            gth_channels=(GTHProjectorChannel(0, 0.3, ((0.5,),)),),
        )
        system = PeriodicDFTSystem(
            (6.0, 6.0, 6.0),
            (6, 6, 6),
            ((2.0, 3.0, 3.0), (4.0, 3.0, 3.0)),
            pseudo,
        )
        mesh = KPointMesh(
            [
                KPoint(
                    (-0.25, 0.0, 0.0),
                    weight=0.25,
                    coordinate_system="reduced",
                ),
                KPoint(
                    (0.25, 0.0, 0.0),
                    weight=0.25,
                    coordinate_system="reduced",
                ),
                KPoint(
                    (0.0, -0.25, 0.0),
                    weight=0.25,
                    coordinate_system="reduced",
                ),
                KPoint(
                    (0.0, 0.25, 0.0),
                    weight=0.25,
                    coordinate_system="reduced",
                ),
            ]
        )
        observer = RuntimeObserver(synchronize=mx.synchronize)
        result = run_periodic_scf(
            system,
            cutoff_hartree=2.5,
            kpoint_mesh=mesh,
            n_bands=1,
            config=PeriodicSCFConfig(
                max_iterations=3,
                min_iterations=2,
                density_tolerance=0.2,
                energy_tolerance=0.1,
                orbital_tolerance=3e-3,
                mixer="linear",
                davidson=PeriodicDavidsonConfig(
                    max_iterations=12,
                    tolerance=3e-3,
                    max_subspace_size=10,
                ),
                kpoint_batch_size=2,
            ),
            observer=observer,
        )
        mx.eval(result.density)
        mx.synchronize()

        snapshot = observer.snapshot()
        events = [event for event in snapshot["events"] if event["event"] == "kpoint_batch"]
        assert len(result.owned_kpoints) == 2
        assert np.isfinite(result.total_energy)
        assert any(event["status"] == "completed" and event["batch_size"] == 2 for event in events)
        assert snapshot["work_counters"]["hpsi_calls"] > 0
        assert mx.default_device() == metal_device
        assert mx.default_stream(metal_device) == metal_stream
    finally:
        _restore_runtime(previous_device, previous_stream)

    assert mx.default_device() == previous_device
    assert mx.default_stream(previous_device) == previous_stream
