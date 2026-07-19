from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.dft import (
    GTHProjectorChannel,
    PeriodicGTHNonlocalOperator,
    PeriodicKohnShamOperator,
    PlaneWaveBasis,
    PseudopotentialData,
    PseudopotentialFormat,
    RealSpaceGrid,
)
from mlx_atomistic.dft._compact import _CompactBatch


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
            states.append(
                basis._state_from_compact(mx.array(values.astype(np.complex64)))
            )
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
