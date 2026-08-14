"""Parity tests for the recurring fused bonded-force Metal route."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.charmm_terms import CHARMMCMAPPotential, CHARMMUreyBradleyPotential
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
@pytest.mark.parametrize(
    "included_families",
    [
        frozenset({"angle", "dihedral"}),
        frozenset({"bond", "angle", "dihedral"}),
        frozenset({"bond", "angle", "dihedral", "improper"}),
        frozenset({"bond", "angle", "dihedral", "improper", "urey_bradley"}),
        frozenset({"charmm_cmap"}),
        frozenset(
            {
                "bond",
                "angle",
                "dihedral",
                "improper",
                "urey_bradley",
                "charmm_cmap",
            }
        ),
    ],
    ids=[
        "two-families",
        "three-families",
        "four-families",
        "charmm-urey-bradley",
        "charmm-cmap-only",
        "charmm-cmap",
    ],
)
def test_fused_bonded_pipeline_matches_supported_force_terms(included_families):
    """One fused dispatch preserves every supported bonded-force formula."""

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
    cmap_axis = np.linspace(-np.pi, np.pi, 24, endpoint=False)
    cmap_phi, cmap_psi = np.meshgrid(cmap_axis, cmap_axis, indexing="ij")
    available_terms = {
        "bond": HarmonicBondPotential(
            [(0, 1), (1, 2), (4, 5)],
            k=[120.0, 80.0, 95.0],
            length=[0.4, 0.75, 0.8],
        ),
        "angle": HarmonicAnglePotential(
            [(0, 1, 2), (4, 5, 6)],
            k=[30.0, 45.0],
            angle=[1.8, 2.0],
        ),
        "dihedral": PeriodicDihedralPotential(
            [(0, 1, 2, 3), (4, 5, 6, 7)],
            k=[2.5, 1.7],
            periodicity=[3.0, 2.0],
            phase=[0.3, -0.2],
        ),
        "improper": ImproperDihedralPotential(
            [(0, 2, 1, 3), (4, 6, 5, 7)],
            k=[1.2, 0.8],
            periodicity=[0.0, 2.0],
            phase=[0.1, -0.4],
        ),
        "urey_bradley": CHARMMUreyBradleyPotential(
            [(0, 7, 2), (4, 0, 6)],
            k=[70.0, 55.0],
            distance=[1.1, 1.5],
        ),
        "charmm_cmap": CHARMMCMAPPotential(
            [
                (0, 1, 2, 3, 1, 2, 3, 4),
                (3, 4, 5, 6, 4, 5, 6, 7),
            ],
            cmap_grids=np.stack(
                [
                    np.sin(cmap_phi)
                    + 0.2 * np.cos(cmap_psi)
                    + 0.1 * np.sin(cmap_phi - cmap_psi),
                    0.3 * np.cos(cmap_phi)
                    - np.sin(cmap_psi)
                    + 0.07 * np.cos(2.0 * cmap_phi + cmap_psi),
                ]
            ),
            cmap_indices=[0, 1],
        ),
    }
    terms = tuple(
        term
        for family, term in available_terms.items()
        if family in included_families
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
    if {"urey_bradley", "charmm_cmap"} & included_families:
        assert binding.fused_bonded_binding.term_indices == frozenset(range(len(terms)))
    actual = binding.forces(positions)
    mx.eval(reference, actual)

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(reference),
        rtol=3.0e-5,
        atol=3.0e-5,
    )
