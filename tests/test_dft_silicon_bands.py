from __future__ import annotations

import numpy as np
import pytest

from mlx_atomistic.benchmarks.dft_silicon_bands import (
    HARTREE_TO_EV,
    analyze_silicon_bands,
    load_silicon_band_references,
    silicon_band_path,
    silicon_primitive_cell,
)


def test_silicon_band_reference_bundle_and_full_path_are_pinned():
    references = load_silicon_band_references()
    path = silicon_band_path(points_per_segment=9, references=references)

    assert references["primary_reference"]["energies_ev_relative_to_vbm"][
        "indirect_gap"
    ] == pytest.approx(0.49)
    assert references["secondary_local_qe_reference"]["hardware"].startswith("4 MPI")
    assert len(path.points) == 41
    assert [point.label for point in path.points if point.label is not None] == [
        "Γ",
        "X",
        "W",
        "K",
        "Γ",
        "L",
    ]
    assert all(point.coordinate_system == "reduced" for point in path.points)


def test_silicon_band_path_bounded_profiles_and_primitive_cell():
    gamma = silicon_band_path(profile="gamma")
    short = silicon_band_path(profile="short")
    primitive = silicon_primitive_cell(10.0)

    assert len(gamma.points) == 1
    assert len(short.points) == 3
    assert short.points[1].label == "0.875 X"
    assert np.linalg.det(primitive) == pytest.approx(250.0)
    with pytest.raises(ValueError, match="profile"):
        silicon_band_path(profile="unknown")
    with pytest.raises(ValueError, match="at least two"):
        silicon_band_path(points_per_segment=1)


def test_synthetic_silicon_bands_pass_scientific_analysis():
    path = silicon_band_path(points_per_segment=9)
    point_count = len(path.points)
    energies_ev = np.full((point_count, 24), 8.0)
    weights = np.zeros_like(energies_ev)
    residuals = np.full_like(energies_ev, 4e-7 / HARTREE_TO_EV)
    energies_ev[:, :16] = np.linspace(-20.0, -13.0, 16)
    for index in range(point_count):
        energies_ev[index, 12:16] = (-11.8, -7.0, -2.0, 0.0)
        weights[index, 12:16] = 1.0
        energies_ev[index, 16] = 1.0
        weights[index, 16] = 1.0

    labels = {
        point.label: index
        for index, point in enumerate(path.points)
        if point.label is not None
    }
    gamma = 0
    x_point = labels["X"]
    l_point = labels["L"]
    energies_ev[gamma, 12:16] = (-11.97, -7.0, -2.0, 0.0)
    energies_ev[gamma, 16:18] = (2.48, 3.28)
    weights[gamma, 17] = 1.0
    energies_ev[x_point, 12:16] = (-7.82, -5.0, -2.85, -0.5)
    energies_ev[x_point, 16] = 0.62
    energies_ev[l_point, 12:16] = (-9.63, -6.98, -1.19, -0.4)
    energies_ev[l_point, 16:18] = (1.45, 3.24)
    weights[l_point, 17] = 1.0
    energies_ev[7, 16] = 0.55

    analysis = analyze_silicon_bands(
        energies_ev / HARTREE_TO_EV,
        weights,
        residuals,
        [5e-7] * point_count,
        path,
    )

    assert analysis["status"] == "validated"
    assert analysis["gap"]["indirect_ev"] == pytest.approx(0.55)
    assert analysis["gap"]["cbm_gamma_x_fraction"] == pytest.approx(0.875)
    assert analysis["valence_bandwidth_ev"] == pytest.approx(11.97)
    assert all(analysis["numerical_gates"].values())
    assert all(analysis["scientific_gates"].values())


def test_silicon_band_analysis_rejects_invalid_shapes():
    path = silicon_band_path(profile="gamma")

    with pytest.raises(ValueError, match="inconsistent"):
        analyze_silicon_bands(
            np.zeros((1, 24)),
            np.zeros((1, 23)),
            np.zeros((1, 24)),
            [0.0],
            path,
        )
