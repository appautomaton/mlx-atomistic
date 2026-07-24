from __future__ import annotations

from math import pi

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.dft import (
    DiracExchange,
    GTHProjectorChannel,
    LDACorrelationPW92,
    PeriodicDavidsonConfig,
    PeriodicDFTSystem,
    PeriodicGTHNonlocalOperator,
    PeriodicKohnShamOperator,
    PeriodicSCFConfig,
    PlaneWaveBasis,
    ProductionPBEExchangeCorrelation,
    PseudopotentialData,
    PseudopotentialFormat,
    RealSpaceGrid,
    gth_local_potential_grid,
    gth_local_reciprocal_coefficients,
    periodic_ewald_energy,
    periodic_ewald_forces,
    periodic_gth_local_forces,
    read_gth,
    run_periodic_scf,
    solve_periodic_eigenproblem,
)
from mlx_atomistic.dft.kpoints import KPoint, KPointMesh


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


@pytest.mark.parametrize("low_density", [0.0, 1e-8])
def test_production_pbe_low_density_step_has_finite_energy_and_potential(
    low_density,
):
    grid = RealSpaceGrid((8, 8, 8), (10.0, 10.0, 10.0))
    density = np.full(grid.shape, low_density, dtype=np.float32)
    density[:4] = 0.03

    result = ProductionPBEExchangeCorrelation().evaluate(mx.array(density), grid)

    assert np.isfinite(np.asarray(result.energy_density)).all()
    assert np.isfinite(np.asarray(result.potential)).all()
    assert np.isfinite(float(result.total_energy))


def _silicon_gth() -> PseudopotentialData:
    return PseudopotentialData(
        element="Si",
        format=PseudopotentialFormat.GTH,
        valence_charge=4.0,
        gth_rloc=0.44,
        gth_coefficients=(-6.26928833,),
        gth_channels=(
            GTHProjectorChannel(
                0,
                0.43563383,
                ((8.9517415, -2.70627082), (-2.70627082, 3.4937806)),
            ),
            GTHProjectorChannel(1, 0.49794218, ((2.43127673,),)),
        ),
    )


def test_gth_channel_validation_and_standalone_parser_preserve_full_matrices(tmp_path):
    path = tmp_path / "Si-q4-pbe.gth"
    path.write_text(
        """Goedecker pseudopotential for Si
14 4 260716 zatom,zion,pspdat
10 11 1 2 2001 0 pspcod,pspxc,lmax,lloc,mmax,r2well
0.44 1 -6.26928833
2
0.43563383 2 8.9517415 -2.70627082
3.4937806
0.49794218 1 2.43127673
0
"""
    )

    parsed = read_gth(path, element="Si")

    assert parsed.metadata["functional"] == "PBE"
    assert parsed.gth_channels == _silicon_gth().gth_channels
    assert len(parsed.nonlocal_projectors) == 3
    with pytest.raises(ValueError, match="symmetric"):
        GTHProjectorChannel(0, 0.4, ((1.0, 2.0), (0.0, 1.0)))


def test_gth_local_reciprocal_formula_and_grid_are_real():
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis(grid, 4.0)
    pseudo = _silicon_gth()
    position = ((1.0, 2.0, 3.0),)

    coefficients = np.asarray(gth_local_reciprocal_coefficients(pseudo, basis, position))
    potential = np.asarray(gth_local_potential_grid(pseudo, basis, position))
    rloc = 0.44
    c1 = -6.26928833
    expected_zero = (2.0 * pi * rloc**2 * 4.0 + (2.0 * pi) ** 1.5 * rloc**3 * c1) / grid.volume
    g = 2.0 * pi / 8.0
    rq2 = g * g * rloc * rloc
    expected_g = (
        4.0
        * pi
        * np.exp(-0.5 * rq2)
        * (-4.0 / (g * g) + np.sqrt(pi / 2.0) * rloc**3 * c1)
        / grid.volume
        * np.exp(-1j * g * position[0][0])
    )

    assert coefficients[0, 0, 0].real == pytest.approx(expected_zero, rel=1e-6)
    assert coefficients[1, 0, 0] == pytest.approx(expected_g, rel=2e-6)
    assert np.max(np.abs(np.imag(np.fft.ifftn(coefficients) * grid.size))) < 2e-6
    assert np.isfinite(potential).all()


def test_periodic_gth_local_forces_match_fixed_density_energy_derivative():
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis(grid, 4.0)
    pseudo = _silicon_gth()
    positions = np.array(((1.0, 2.0, 3.0), (5.0, 4.0, 2.0)))
    coordinates = np.asarray(grid.coordinates())
    density = (
        0.02
        + 0.004 * np.cos(2.0 * pi * coordinates[..., 0] / 8.0)
        + 0.003 * np.sin(2.0 * pi * coordinates[..., 1] / 8.0)
        + 0.002 * np.cos(4.0 * pi * coordinates[..., 2] / 8.0)
    ).astype(np.float32)

    observed = np.asarray(
        periodic_gth_local_forces(
            mx.array(density),
            pseudo,
            basis,
            positions,
        )
    )
    reference = np.zeros_like(positions)
    displacement = 1e-2
    for ion_index in range(positions.shape[0]):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[ion_index, axis] += displacement
            minus[ion_index, axis] -= displacement
            energy_plus = float(
                mx.sum(
                    mx.array(density)
                    * gth_local_potential_grid(pseudo, basis, plus)
                )
                * grid.dv
            )
            energy_minus = float(
                mx.sum(
                    mx.array(density)
                    * gth_local_potential_grid(pseudo, basis, minus)
                )
                * grid.dv
            )
            reference[ion_index, axis] = -(
                energy_plus - energy_minus
            ) / (2.0 * displacement)

    np.testing.assert_allclose(observed, reference, atol=5e-5, rtol=2e-4)


def test_periodic_gth_nonlocal_operator_is_hermitian_at_non_gamma_kpoint():
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis.from_reduced_kpoint(grid, 4.0, (0.25, 0.25, -0.25))
    operator = PeriodicGTHNonlocalOperator(_silicon_gth(), basis, ((1.0, 2.0, 3.0),))
    rng = np.random.default_rng(44)
    left = basis.normalize(
        mx.array(
            (rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)).astype(np.complex64)
        )
    )
    right = basis.normalize(
        mx.array(
            (rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)).astype(np.complex64)
        )
    )

    left_right = mx.sum(mx.conjugate(left) * operator.apply(right))
    right_left = mx.sum(mx.conjugate(right) * operator.apply(left))

    left_right_value = np.asarray(left_right).item()
    right_left_value = np.asarray(mx.conjugate(right_left)).item()
    assert left_right_value == pytest.approx(right_left_value, abs=2e-5)
    assert abs(float(operator.energy(mx.stack([left, right]), occupations=[1.0, 0.5]))) > 0.0
    assert operator.to_dict()["angular_projector_count_per_ion"] == 5


def test_periodic_gth_nonlocal_operator_is_cell_translation_invariant():
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis.from_reduced_kpoint(grid, 4.0, (0.25, 0.0, 0.0))
    rng = np.random.default_rng(10)
    orbital = basis.normalize(
        mx.array(
            (rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)).astype(np.complex64)
        )
    )
    first = PeriodicGTHNonlocalOperator(_silicon_gth(), basis, ((1.0, 2.0, 3.0),))
    shifted = PeriodicGTHNonlocalOperator(_silicon_gth(), basis, ((9.0, 2.0, 3.0),))

    first_energy = float(first.energy(orbital, occupations=[1.0]))
    shifted_energy = float(shifted.energy(orbital, occupations=[1.0]))

    assert shifted_energy == pytest.approx(first_energy, abs=2e-5)


def test_periodic_gth_nonlocal_forces_match_fixed_orbital_energy_derivative():
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis.from_reduced_kpoint(
        grid,
        4.0,
        (0.25, 0.125, -0.25),
    )
    positions = np.array(((1.0, 2.0, 3.0), (5.0, 4.0, 2.0)))
    rng = np.random.default_rng(81)
    trial = rng.normal(size=(2, *grid.shape)) + 1j * rng.normal(
        size=(2, *grid.shape)
    )
    orbitals = basis.orthonormalize(
        mx.array(trial.astype(np.complex64))
    )
    occupations = [2.0, 0.75]

    operator = PeriodicGTHNonlocalOperator(
        _silicon_gth(),
        basis,
        positions,
    )
    observed = np.asarray(
        operator.forces(orbitals, occupations=occupations)
    )
    reference = np.zeros_like(positions)
    displacement = 2e-3
    for ion_index in range(positions.shape[0]):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[ion_index, axis] += displacement
            minus[ion_index, axis] -= displacement
            energy_plus = float(
                PeriodicGTHNonlocalOperator(
                    _silicon_gth(),
                    basis,
                    plus,
                ).energy(orbitals, occupations=occupations)
            )
            energy_minus = float(
                PeriodicGTHNonlocalOperator(
                    _silicon_gth(),
                    basis,
                    minus,
                ).energy(orbitals, occupations=occupations)
            )
            reference[ion_index, axis] = -(
                energy_plus - energy_minus
            ) / (2.0 * displacement)

    np.testing.assert_allclose(observed, reference, atol=8e-5, rtol=4e-4)


def test_periodic_ewald_energy_translation_scaling_and_force_consistency():
    charges = [1.0, -1.0]
    positions = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
    lengths = np.array([6.0, 6.0, 6.0])
    energy = periodic_ewald_energy(charges, positions, lengths, tolerance=1e-8)
    translated = periodic_ewald_energy(
        charges,
        positions + np.array([6.0, 0.0, 0.0]),
        lengths,
        tolerance=1e-8,
    )
    scaled = periodic_ewald_energy(
        charges,
        2.0 * positions,
        2.0 * lengths,
        tolerance=1e-8,
    )
    forces = periodic_ewald_forces(
        charges,
        positions,
        lengths,
        tolerance=1e-8,
    )
    finite_difference_forces = periodic_ewald_forces(
        charges,
        positions,
        lengths,
        displacement=2e-4,
        tolerance=1e-8,
        method="finite_difference",
    )

    assert np.isfinite(energy)
    assert translated == pytest.approx(energy, abs=2e-9)
    assert scaled == pytest.approx(0.5 * energy, rel=2e-6)
    np.testing.assert_allclose(
        forces,
        finite_difference_forces,
        atol=2e-8,
        rtol=2e-7,
    )
    np.testing.assert_allclose(np.sum(forces, axis=0), 0.0, atol=2e-8)
    assert forces[0, 0] == pytest.approx(forces[0, 1], rel=2e-6)
    assert forces[0, 0] == pytest.approx(forces[0, 2], rel=2e-6)


def test_periodic_davidson_matches_diagonal_kinetic_local_oracle():
    grid = RealSpaceGrid((6, 6, 6), (6.0, 6.0, 6.0))
    basis = PlaneWaveBasis.from_reduced_kpoint(grid, 3.0, (0.25, 0.0, 0.0))
    constant = 0.7
    operator = PeriodicKohnShamOperator(basis, mx.full(grid.shape, constant))

    result = solve_periodic_eigenproblem(
        operator,
        n_bands=3,
        config=PeriodicDavidsonConfig(
            max_iterations=12,
            tolerance=2e-5,
            max_subspace_size=12,
        ),
    )
    expected = np.sort(np.asarray(basis.kinetic_energies)[np.asarray(basis.mask)])[:3] + constant

    assert result.converged
    assert result.to_dict()["dense_full_hamiltonian"] is False
    np.testing.assert_allclose(np.asarray(result.eigenvalues), expected, atol=3e-5)
    assert result.orthonormality_error <= 2e-5


def test_weighted_periodic_scf_conserves_electrons_without_dense_fallback():
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
            KPoint((-0.25, 0.0, 0.0), weight=0.5, coordinate_system="reduced"),
            KPoint((0.25, 0.0, 0.0), weight=0.5, coordinate_system="reduced"),
        ]
    )
    result = run_periodic_scf(
        system,
        cutoff_hartree=2.5,
        kpoint_mesh=mesh,
        n_bands=1,
        config=PeriodicSCFConfig(
            max_iterations=6,
            min_iterations=2,
            density_tolerance=0.2,
            energy_tolerance=0.1,
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

    assert result.converged
    assert result.electron_count == pytest.approx(2.0, abs=2e-5)
    assert len(result.kpoints) == 2
    assert all(item.eigen.to_dict()["dense_full_hamiltonian"] is False for item in result.kpoints)
    assert all(item.eigen.converged for item in result.kpoints)
    assert np.isfinite(result.total_energy)
