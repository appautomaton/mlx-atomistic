import json

import numpy as np
import pytest

from mlx_atomistic.dft import (
    DFTSystem,
    DiracExchange,
    LDACorrelationPZ81,
    LDAExchangeCorrelation,
    LinearMixer,
    LocalGaussianPseudopotential,
    PulayDIISMixer,
    RealSpaceGrid,
    ReciprocalGrid,
    SCFConfig,
    density_from_orbitals,
    fft3,
    hartree_potential,
    lda_exchange_energy_potential,
    local_pseudopotential_forces,
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


def test_xc_known_values_and_combined_components():
    grid = RealSpaceGrid((2, 2, 2), [2.0, 2.0, 2.0])
    density = np.ones(grid.shape, dtype=np.float32) * 0.5
    exchange = DiracExchange().evaluate(density, grid)
    correlation = LDACorrelationPZ81().evaluate(density, grid)
    combined = LDAExchangeCorrelation().evaluate(density, grid)

    coefficient = (3.0 / np.pi) ** (1.0 / 3.0)
    expected_exchange_density = -0.75 * coefficient * 0.5 ** (4.0 / 3.0)
    expected_exchange_potential = -coefficient * 0.5 ** (1.0 / 3.0)

    np.testing.assert_allclose(
        np.array(exchange.energy_density),
        np.full(grid.shape, expected_exchange_density),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(exchange.potential),
        np.full(grid.shape, expected_exchange_potential),
        atol=1e-6,
    )
    assert float(correlation.total_energy) < 0.0
    np.testing.assert_allclose(
        np.array(combined.total_energy),
        np.array(exchange.total_energy + correlation.total_energy),
        atol=1e-6,
    )


def test_xc_potential_matches_finite_difference_derivative():
    grid = RealSpaceGrid((2, 2, 2), [2.0, 2.0, 2.0])
    density = np.ones(grid.shape, dtype=np.float32) * 0.4
    functional = LDAExchangeCorrelation()
    epsilon = 1e-4

    plus = density.copy()
    minus = density.copy()
    plus[0, 0, 0] += epsilon
    minus[0, 0, 0] -= epsilon

    e_plus = float(functional.evaluate(plus, grid).total_energy)
    e_minus = float(functional.evaluate(minus, grid).total_energy)
    derivative = (e_plus - e_minus) / (2.0 * epsilon * grid.dv)
    potential = float(np.array(functional.evaluate(density, grid).potential)[0, 0, 0])

    assert derivative == pytest.approx(potential, abs=2e-3)


def test_dft_system_validation_and_grid_surface():
    system = DFTSystem(
        cell=[6.0, 6.0, 6.0],
        grid_shape=(4, 4, 4),
        electron_count=2.0,
        centers=[[3.0, 3.0, 3.0]],
        amplitudes=-2.0,
        widths=0.8,
    )

    assert system.grid.shape == (4, 4, 4)
    assert system.center_count == 1
    assert system.charges == (2.0,)

    with pytest.raises(ValueError, match="charges length"):
        DFTSystem(
            cell=[6.0, 6.0, 6.0],
            grid_shape=(4, 4, 4),
            electron_count=2.0,
            centers=[[3.0, 3.0, 3.0]],
            amplitudes=-2.0,
            widths=0.8,
            charges=[1.0, 1.0],
        )


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


def test_scf_system_diis_restart_and_structured_failure_status():
    system = DFTSystem(
        cell=[6.0, 6.0, 6.0],
        grid_shape=(4, 4, 4),
        electron_count=2.0,
        centers=[[2.7, 3.0, 3.0], [3.3, 3.0, 3.0]],
        amplitudes=[-1.5, -1.5],
        widths=[0.7, 0.7],
    )
    config = SCFConfig(max_iterations=2, tolerance=1e-30, mixer="diis", solver="dense", seed=9)

    result = run_scf(system, config=config, xc_functional=DiracExchange())
    restarted = run_scf(
        system,
        config=SCFConfig(max_iterations=1, mixer=LinearMixer(beta=0.5), solver="dense", seed=9),
        initial_density=result.density,
        initial_orbitals=result.orbitals,
        xc_functional=DiracExchange(),
    )
    diis = PulayDIISMixer(beta=0.4)
    second = run_scf(
        system,
        config=SCFConfig(max_iterations=2, mixer=diis, solver="dense", seed=9),
        xc_functional=DiracExchange(),
    )

    assert result.status == "max_iterations"
    assert result.failure_reason == "max_iterations_reached"
    assert result.history[0]["density_residual"] is not None
    assert "timings" in result.to_dict()
    assert result.forces is not None
    assert restarted.electron_count == pytest.approx(result.electron_count, abs=1e-5)
    assert second.mixer_metadata["name"] == "pulay-diis"


def test_local_pseudopotential_force_matches_finite_difference_and_sign():
    grid = RealSpaceGrid((8, 8, 8), [6.0, 6.0, 6.0])
    coordinates = np.array(grid.coordinates())
    right_density = np.exp(
        -np.sum((coordinates - np.array([3.6, 3.0, 3.0])) ** 2, axis=-1) / 0.7
    ).astype(np.float32)
    left_density = np.exp(
        -np.sum((coordinates - np.array([2.4, 3.0, 3.0])) ** 2, axis=-1) / 0.7
    ).astype(np.float32)
    right_density *= 2.0 / (np.sum(right_density) * grid.dv)
    left_density *= 2.0 / (np.sum(left_density) * grid.dv)
    pseudopotential = LocalGaussianPseudopotential([[3.0, 3.0, 3.0]], -2.0, 0.8)

    force = np.array(local_pseudopotential_forces(right_density, grid, pseudopotential))[0]
    left_force = np.array(local_pseudopotential_forces(left_density, grid, pseudopotential))[0]

    def local_energy(center_x: float) -> float:
        shifted = LocalGaussianPseudopotential([[center_x, 3.0, 3.0]], -2.0, 0.8)
        return float(np.sum(right_density * np.array(shifted.field(grid))) * grid.dv)

    epsilon = 1e-3
    finite_difference_force = -(
        local_energy(3.0 + epsilon) - local_energy(3.0 - epsilon)
    ) / (2.0 * epsilon)

    assert force[0] > 0.0
    assert left_force[0] < 0.0
    assert force[0] == pytest.approx(finite_difference_force, abs=2e-3)
    np.testing.assert_allclose(force[1:], [0.0, 0.0], atol=1e-5)


def test_local_pseudopotential_force_zero_for_symmetric_uniform_density():
    grid = RealSpaceGrid((8, 8, 8), [6.0, 6.0, 6.0])
    density = np.ones(grid.shape, dtype=np.float32)
    density *= 2.0 / (np.sum(density) * grid.dv)
    pseudopotential = LocalGaussianPseudopotential([[3.0, 3.0, 3.0]], -2.0, 0.8)

    force = np.array(local_pseudopotential_forces(density, grid, pseudopotential))[0]

    np.testing.assert_allclose(force, [0.0, 0.0, 0.0], atol=1e-5)
