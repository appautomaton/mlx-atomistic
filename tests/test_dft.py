import json

import numpy as np
import pytest

from mlx_atomistic.dft import (
    LocalGaussianPseudopotential,
    RealSpaceGrid,
    ReciprocalGrid,
    SCFConfig,
    density_from_orbitals,
    fft3,
    hartree_potential,
    lda_exchange_energy_potential,
    normalize_orbitals,
    reciprocal_to_real,
    run_scf,
)


def test_real_and_reciprocal_grid_geometry():
    grid = RealSpaceGrid((4, 5, 6), [8.0, 10.0, 12.0])
    reciprocal = ReciprocalGrid.from_real_space(grid)

    np.testing.assert_allclose(np.array(grid.spacing), [2.0, 2.0, 2.0], atol=1e-6)
    assert grid.volume == pytest.approx(960.0)
    assert grid.dv == pytest.approx(8.0)
    assert reciprocal.vectors.shape == (4, 5, 6, 3)
    assert reciprocal.g2.shape == grid.shape
    assert bool(np.array(reciprocal.zero_mask)[0, 0, 0])
    assert float(np.array(reciprocal.g2)[0, 0, 0]) == pytest.approx(0.0)


def test_fft_round_trip_preserves_real_field():
    grid = RealSpaceGrid((4, 4, 4), [4.0, 4.0, 4.0])
    coordinates = np.array(grid.coordinates())
    field = np.sin(2.0 * np.pi * coordinates[..., 0] / 4.0).astype(np.float32)

    round_trip = reciprocal_to_real(fft3(field))

    np.testing.assert_allclose(np.array(round_trip), field, atol=1e-5)


def test_orbital_normalization_and_density_integrates_to_electron_count():
    grid = RealSpaceGrid((4, 4, 4), [4.0, 4.0, 4.0])
    coordinates = np.array(grid.coordinates())
    orbital = np.exp(-np.sum((coordinates - 2.0) ** 2, axis=-1)).astype(np.float32)

    normalized = normalize_orbitals(orbital, grid)
    single_density = density_from_orbitals(normalized, grid, occupations=[1.0])
    closed_shell_density = density_from_orbitals(normalized, grid)

    assert float(np.sum(np.abs(np.array(normalized)) ** 2) * grid.dv) == pytest.approx(
        1.0,
        abs=1e-5,
    )
    assert float(np.sum(np.array(single_density)) * grid.dv) == pytest.approx(1.0, abs=1e-5)
    assert float(np.sum(np.array(closed_shell_density)) * grid.dv) == pytest.approx(
        2.0,
        abs=1e-5,
    )


def test_invalid_density_occupations_fail_clearly():
    grid = RealSpaceGrid((4, 4, 4), [4.0, 4.0, 4.0])
    orbital = np.ones(grid.shape, dtype=np.float32)

    with pytest.raises(ValueError, match="occupations length"):
        density_from_orbitals(orbital, grid, occupations=[1.0, 1.0])
    with pytest.raises(ValueError, match="cannot exceed 2"):
        density_from_orbitals(orbital, grid, occupations=[3.0])


def test_local_gaussian_pseudopotential_symmetry_and_finiteness():
    grid = RealSpaceGrid((4, 4, 4), [4.0, 4.0, 4.0])
    local = LocalGaussianPseudopotential(
        centers=[[2.0, 2.0, 2.0]],
        amplitudes=-2.0,
        widths=0.6,
    )

    field = np.array(local.field(grid))

    assert np.isfinite(field).all()
    assert field[1, 1, 1] == pytest.approx(field[2, 2, 2], abs=1e-6)
    assert field[1, 2, 2] == pytest.approx(field[2, 1, 1], abs=1e-6)


def test_hartree_g_zero_removed_and_lda_exchange_finite():
    grid = RealSpaceGrid((4, 4, 4), [4.0, 4.0, 4.0])
    uniform_density = np.ones(grid.shape, dtype=np.float32) * 0.25

    hartree = np.array(hartree_potential(uniform_density, grid))
    exchange_energy, exchange_potential = lda_exchange_energy_potential(uniform_density, grid)

    np.testing.assert_allclose(hartree, np.zeros(grid.shape), atol=1e-6)
    assert np.isfinite(float(exchange_energy))
    assert float(exchange_energy) < 0.0
    assert np.isfinite(np.array(exchange_potential)).all()


def test_scf_toy_one_and_two_electron_runs_are_deterministic_and_json_safe():
    grid = RealSpaceGrid((4, 4, 4), [6.0, 6.0, 6.0])
    local = LocalGaussianPseudopotential(
        centers=[[3.0, 3.0, 3.0]],
        amplitudes=-2.0,
        widths=0.8,
    )
    config = SCFConfig(
        max_iterations=3,
        tolerance=1e-10,
        mixing=0.5,
        solver="dense",
        seed=5,
    )

    one = run_scf(grid, local, electron_count=1.0, config=config)
    two = run_scf(grid, local, electron_count=2.0, config=config)
    two_repeat = run_scf(grid, local, electron_count=2.0, config=config)

    assert one.iterations == 3
    assert two.iterations == 3
    assert one.solver == "dense"
    assert np.isfinite(one.total_energy)
    assert np.isfinite(two.total_energy)
    assert two.electron_count == pytest.approx(2.0, abs=1e-5)
    assert two.total_energy == pytest.approx(two_repeat.total_energy, abs=1e-7)
    assert two.history == two_repeat.history
    json.dumps(two.to_dict())
