from __future__ import annotations

from math import pi

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.dft import (
    DiracExchange,
    LDACorrelationPW92,
    PlaneWaveBasis,
    ProductionPBEExchangeCorrelation,
    RealSpaceGrid,
)


def test_plane_wave_basis_mask_and_metadata_are_deterministic():
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis.from_reduced_kpoint(grid, 2.0, (0.25, 0.0, 0.0))
    shifted = np.asarray(basis.shifted_vectors)
    expected = np.count_nonzero(0.5 * np.sum(shifted * shifted, axis=-1) <= 2.0 + 1e-12)

    assert basis.active_count == expected
    assert basis.to_dict() == {
        "cutoff_hartree": 2.0,
        "kpoint_cartesian_bohr_inverse": [pi / 16.0, 0.0, 0.0],
        "fft_shape": [8, 8, 8],
        "active_count": expected,
        "normalization": "unit-coefficients__real-integral-unit",
    }


def test_plane_wave_round_trip_preserves_masked_coefficients_and_norm():
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis(grid, 3.0)
    rng = np.random.default_rng(42)
    coefficients = rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)
    coefficients = basis.normalize(mx.array(coefficients.astype(np.complex64)))

    orbitals = basis.to_real(coefficients)
    round_trip = basis.to_coefficients(orbitals)

    np.testing.assert_allclose(np.asarray(round_trip), np.asarray(coefficients), atol=2e-6)
    assert float(basis.coefficient_norms(coefficients)[0]) == pytest.approx(1.0, abs=2e-6)
    assert float(basis.real_norms(orbitals)[0]) == pytest.approx(1.0, abs=2e-6)
    inactive = ~np.asarray(basis.mask)
    assert np.count_nonzero(np.asarray(round_trip)[inactive]) == 0


def test_plane_wave_orthonormalization_and_overlap_use_active_basis_only():
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis(grid, 3.0)
    rng = np.random.default_rng(7)
    trial = rng.normal(size=(3, *grid.shape)) + 1j * rng.normal(size=(3, *grid.shape))

    orthonormal = basis.orthonormalize(mx.array(trial.astype(np.complex64)))
    overlap = np.asarray(basis.overlap_matrix(orthonormal))

    np.testing.assert_allclose(overlap, np.eye(3), atol=2e-5)
    assert np.count_nonzero(np.asarray(orthonormal)[:, ~np.asarray(basis.mask)]) == 0


def test_plane_wave_kinetic_and_constant_local_actions_are_exact():
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis(grid, 4.0)
    coefficients = np.zeros(grid.shape, dtype=np.complex64)
    coefficients[1, 0, 0] = 1.0
    kinetic = np.asarray(basis.apply_kinetic(mx.array(coefficients)))
    local = np.asarray(basis.apply_local(mx.array(coefficients), mx.full(grid.shape, 1.25)))

    expected_kinetic = 0.5 * (2.0 * pi / 8.0) ** 2
    assert kinetic[1, 0, 0].real == pytest.approx(expected_kinetic, rel=1e-6)
    np.testing.assert_allclose(local, 1.25 * coefficients, atol=2e-6)


def test_pw92_known_unpolarized_correlation_values():
    functional = LDACorrelationPW92()
    expected = {
        0.5: -0.07661873586910005,
        1.0: -0.05977368580724599,
        2.0: -0.04475949734441541,
        5.0: -0.02821623327462354,
    }

    for rs, expected_energy in expected.items():
        density = 3.0 / (4.0 * pi * rs**3)
        observed = float(functional.correlation_per_particle(mx.array(density)))
        assert observed == pytest.approx(expected_energy, abs=2e-7)


def test_production_pbe_uniform_density_reduces_to_dirac_plus_pw92():
    grid = RealSpaceGrid((4, 4, 4), (4.0, 4.0, 4.0))
    density = mx.full(grid.shape, 0.2)
    production = ProductionPBEExchangeCorrelation().evaluate(density, grid)
    exchange = DiracExchange().evaluate(density, grid)
    correlation = LDACorrelationPW92().evaluate(density, grid)

    assert production.name == "pbe-pw92-gga"
    assert float(production.total_energy) == pytest.approx(
        float(exchange.total_energy + correlation.total_energy),
        abs=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(production.potential),
        np.asarray(exchange.potential + correlation.potential),
        atol=2e-5,
    )


def test_production_pbe_potential_matches_total_energy_finite_difference():
    grid = RealSpaceGrid((3, 3, 3), (3.0, 3.0, 3.0))
    coordinates = np.asarray(grid.coordinates())
    density_np = 0.15 + 0.02 * np.cos(2.0 * pi * coordinates[..., 0] / 3.0)
    density = mx.array(density_np.astype(np.float32))
    functional = ProductionPBEExchangeCorrelation()
    result = functional.evaluate(density, grid)
    index = (1, 1, 1)
    step = 2e-4
    plus = density_np.copy()
    minus = density_np.copy()
    plus[index] += step
    minus[index] -= step
    e_plus = float(functional.evaluate(mx.array(plus.astype(np.float32)), grid).total_energy)
    e_minus = float(functional.evaluate(mx.array(minus.astype(np.float32)), grid).total_energy)
    finite_difference = (e_plus - e_minus) / (2.0 * step * grid.dv)

    assert float(result.potential[index]) == pytest.approx(finite_difference, abs=2e-3)
