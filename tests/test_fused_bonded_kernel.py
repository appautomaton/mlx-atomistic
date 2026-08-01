"""Parity tests for the recurring fused bonded-force Metal route."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.core import Cell
from mlx_atomistic.force_runtime import _PreparedForcePipeline
from mlx_atomistic.forcefields import (
    HarmonicAnglePotential,
    HarmonicBondPotential,
    ImproperDihedralPotential,
    PeriodicDihedralPotential,
)


@pytest.fixture(autouse=True)
def _on_gpu(monkeypatch):
    """Run the fused route on Metal and restore the default test device."""

    monkeypatch.setenv("MLX_ATOMISTIC_DEVICE", "gpu")
    previous_device = mx.default_device()
    try:
        gpu = mx.Device(mx.gpu, 0)
        mx.set_default_device(gpu)
        mx.set_default_stream(mx.new_stream(gpu))
        mx.eval(mx.array([1.0], dtype=mx.float32) + 1.0)
    except Exception:  # noqa: BLE001 - any Metal load failure means skip
        mx.set_default_device(previous_device)
        mx.set_default_stream(mx.new_stream(previous_device))
        pytest.skip("Metal GPU unavailable")
    yield
    mx.set_default_device(previous_device)
    mx.set_default_stream(mx.new_stream(previous_device))


@pytest.mark.gpu
def test_fused_bonded_pipeline_matches_standard_force_terms():
    """One fused dispatch preserves all four standard bonded force formulas."""

    positions = mx.array(
        [
            [0.2, 1.0, 1.0],
            [7.9, 1.2, 1.1],
            [7.6, 1.8, 1.4],
            [7.2, 2.1, 2.0],
            [2.0, 3.0, 1.0],
            [2.7, 3.2, 1.3],
            [3.1, 3.8, 1.8],
            [3.8, 4.0, 2.4],
        ],
        dtype=mx.float32,
    )
    cell = Cell.cubic(8.0)
    terms = (
        HarmonicBondPotential(
            [(0, 1), (1, 2), (4, 5)],
            k=[120.0, 80.0, 95.0],
            length=[0.4, 0.75, 0.8],
        ),
        HarmonicAnglePotential(
            [(0, 1, 2), (4, 5, 6)],
            k=[30.0, 45.0],
            angle=[1.8, 2.0],
        ),
        PeriodicDihedralPotential(
            [(0, 1, 2, 3), (4, 5, 6, 7)],
            k=[2.5, 1.7],
            periodicity=[3.0, 2.0],
            phase=[0.3, -0.2],
        ),
        ImproperDihedralPotential(
            [(0, 2, 1, 3), (4, 6, 5, 7)],
            k=[1.2, 0.8],
            periodicity=[0.0, 2.0],
            phase=[0.1, -0.4],
        ),
    )
    reference = mx.zeros_like(positions)
    for term in terms:
        _, term_forces = term.energy_forces(positions, cell)
        reference = reference + term_forces

    pipeline = _PreparedForcePipeline.prepare(
        terms,
        cell=cell,
    )
    binding = pipeline.bind(None)
    assert binding.fused_bonded_binding is not None
    actual = binding.forces(positions)
    mx.eval(reference, actual)

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(reference),
        rtol=3.0e-5,
        atol=3.0e-5,
    )
