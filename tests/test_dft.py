import json

import numpy as np
import pytest

from mlx_atomistic.dft import (
    DenseHamiltonianReference,
    DFTSystem,
    DiracExchange,
    Ion,
    IonCollection,
    KohnShamOperator,
    LDACorrelationPZ81,
    LDAExchangeCorrelation,
    LinearMixer,
    LocalGaussianPseudopotential,
    LocalPseudopotentialField,
    PseudopotentialFormat,
    PulayDIISMixer,
    RealSpaceGrid,
    ReciprocalGrid,
    SCFConfig,
    SubspaceDiagonalizer,
    apply_kinetic,
    apply_local_potential,
    center_center_energy,
    density_from_orbitals,
    fft3,
    hartree_potential,
    lda_exchange_energy_potential,
    local_pseudopotential_forces,
    normalize_orbitals,
    orthonormality_error,
    read_gth,
    read_upf,
    reciprocal_to_real,
    run_scf,
    scf_total_energy_forces,
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
    assert two.electronic_energy == pytest.approx(two.energy_by_term["electronic"])
    assert two.center_center_energy == pytest.approx(0.0)
    assert two.orbital_eigenvalues is not None
    assert two.orbital_residuals is not None
    assert np.isfinite(np.array(two.orbital_residuals)).all()
    assert two.orthonormality_error < 1e-5
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


def test_upf_and_gth_parsers_capture_local_and_nonlocal_metadata():
    upf = read_upf("vendors/quantum-espresso/pseudo/Si_r.upf")
    single_gth = read_gth("vendors/quantum-espresso/pseudo/C-q4.gth", element="C")
    database_gth = read_gth(
        "vendors/cp2k/data/GTH_POTENTIALS",
        element="H",
        name="GTH-BLYP-q1",
    )

    assert upf.format == PseudopotentialFormat.UPF
    assert upf.element == "Si"
    assert upf.valence_charge == pytest.approx(4.0)
    assert upf.local_grid.size == 1528
    assert upf.nonlocal_available
    assert len(upf.nonlocal_projectors) == 10
    assert np.isfinite(upf.local_potential(np.array([0.0, 1.0, 2.0]))).all()
    assert single_gth.format == PseudopotentialFormat.GTH
    assert single_gth.element == "C"
    assert single_gth.valence_charge == pytest.approx(4.0)
    assert single_gth.nonlocal_available
    assert database_gth.element == "H"
    assert database_gth.valence_charge == pytest.approx(1.0)

    with pytest.raises(ValueError, match="requested GTH entry"):
        read_gth("vendors/cp2k/data/GTH_POTENTIALS", element="Xe", name="missing")
    with pytest.raises(NotImplementedError, match="nonlocal"):
        upf.apply_nonlocal(
            np.ones((2, 2, 2), dtype=np.float32),
            RealSpaceGrid((2, 2, 2), [2, 2, 2]),
        )


def test_ion_collection_local_fields_are_finite_and_periodic():
    grid = RealSpaceGrid((4, 4, 4), [6.0, 6.0, 6.0])
    gth = read_gth("vendors/quantum-espresso/pseudo/H-q1.gth", element="H")
    first = LocalPseudopotentialField(IonCollection([Ion("H", [3.0, 3.0, 3.0], gth)]))
    translated = LocalPseudopotentialField(IonCollection([Ion("H", [9.0, 3.0, 3.0], gth)]))

    first_field = np.array(first.field(grid))
    translated_field = np.array(translated.field(grid))

    assert np.isfinite(first_field).all()
    np.testing.assert_allclose(first_field, translated_field, atol=1e-6)


def test_dft_system_ions_default_electron_count_and_run_scf_for_gth_and_upf():
    gth = read_gth("vendors/quantum-espresso/pseudo/H-q1.gth", element="H")
    upf = read_upf("vendors/quantum-espresso/pseudo/Si_r.upf")
    gth_system = DFTSystem(
        cell=[6.0, 6.0, 6.0],
        grid_shape=(4, 4, 4),
        ions=IonCollection([Ion("H", [3.0, 3.0, 3.0], gth)]),
    )
    upf_system = DFTSystem(
        cell=[8.0, 8.0, 8.0],
        grid_shape=(4, 4, 4),
        ions=IonCollection([Ion("Si", [4.0, 4.0, 4.0], upf)]),
    )

    assert gth_system.electron_count == pytest.approx(1.0)
    assert upf_system.electron_count == pytest.approx(4.0)

    config = SCFConfig(max_iterations=1, solver="dense", seed=17)
    gth_result = run_scf(gth_system, config=config, xc_functional=DiracExchange())
    upf_result = run_scf(upf_system, config=config, xc_functional=DiracExchange())

    assert np.isfinite(gth_result.total_energy)
    assert np.isfinite(upf_result.total_energy)
    assert gth_result.to_dict()["pseudopotential_format"] == "gth"
    assert upf_result.to_dict()["pseudopotential_format"] == "upf"
    assert upf_result.to_dict()["nonlocal_available"]
    assert not upf_result.to_dict()["nonlocal_applied"]
    assert "local_pseudopotential" in upf_result.energy_by_term


def test_ion_local_force_matches_fixed_density_finite_difference():
    grid = RealSpaceGrid((6, 6, 6), [6.0, 6.0, 6.0])
    coordinates = np.array(grid.coordinates())
    density = np.exp(
        -np.sum((coordinates - np.array([3.4, 3.0, 3.0])) ** 2, axis=-1) / 0.8
    ).astype(np.float32)
    density *= 1.0 / (np.sum(density) * grid.dv)
    gth = read_gth("vendors/quantum-espresso/pseudo/H-q1.gth", element="H")
    field = LocalPseudopotentialField(IonCollection([Ion("H", [3.0, 3.0, 3.0], gth)]))
    force = np.array(field.forces(density, grid))[0]

    def local_energy(center_x: float) -> float:
        shifted = LocalPseudopotentialField(
            IonCollection([Ion("H", [center_x, 3.0, 3.0], gth)])
        )
        return float(np.sum(density * np.array(shifted.field(grid))) * grid.dv)

    epsilon = 1e-3
    finite_difference_force = -(
        local_energy(3.0 + epsilon) - local_energy(3.0 - epsilon)
    ) / (2.0 * epsilon)

    assert force[0] == pytest.approx(finite_difference_force, abs=5e-3)
    np.testing.assert_allclose(force[1:], [0.0, 0.0], atol=2e-4)


def test_upf_local_force_matches_fixed_density_finite_difference():
    grid = RealSpaceGrid((6, 6, 6), [8.0, 8.0, 8.0])
    coordinates = np.array(grid.coordinates())
    density = np.exp(
        -np.sum((coordinates - np.array([4.4, 4.0, 4.0])) ** 2, axis=-1) / 1.0
    ).astype(np.float32)
    density *= 4.0 / (np.sum(density) * grid.dv)
    upf = read_upf("vendors/quantum-espresso/pseudo/Si_r.upf")
    field = LocalPseudopotentialField(IonCollection([Ion("Si", [4.0, 4.0, 4.0], upf)]))
    force = np.array(field.forces(density, grid))[0]

    def local_energy(center_x: float) -> float:
        shifted = LocalPseudopotentialField(
            IonCollection([Ion("Si", [center_x, 4.0, 4.0], upf)])
        )
        return float(np.sum(density * np.array(shifted.field(grid))) * grid.dv)

    epsilon = 1e-3
    finite_difference_force = -(
        local_energy(4.0 + epsilon) - local_energy(4.0 - epsilon)
    ) / (2.0 * epsilon)

    assert force[0] == pytest.approx(finite_difference_force, abs=5e-3)
    np.testing.assert_allclose(force[1:], [0.0, 0.0], atol=2e-4)


def test_scf_total_energy_force_check_works_for_gth_ion_system():
    gth = read_gth("vendors/quantum-espresso/pseudo/H-q1.gth", element="H")
    system = DFTSystem(
        cell=[6.0, 6.0, 6.0],
        grid_shape=(4, 4, 4),
        ions=IonCollection([Ion("H", [3.0, 3.0, 3.0], gth)]),
    )

    check = scf_total_energy_forces(
        system,
        config=SCFConfig(max_iterations=2, solver="dense", seed=9),
        xc_functional=DiracExchange(),
        displacement=1e-3,
    )

    assert check.max_abs_error < 0.2
    assert check.result.force_consistency is not None


def test_plane_wave_kinetic_operator_has_expected_eigenvalue():
    grid = RealSpaceGrid((4, 4, 4), [4.0, 4.0, 4.0])
    coordinates = np.array(grid.coordinates())
    wavevector = 2.0 * np.pi / 4.0
    orbital = np.exp(1j * wavevector * coordinates[..., 0]).astype(np.complex64)

    applied = np.array(apply_kinetic(orbital, grid))

    np.testing.assert_allclose(applied, 0.5 * wavevector**2 * orbital, atol=1e-5)


def test_local_operator_multiplication_matches_grid_product():
    grid = RealSpaceGrid((2, 2, 2), [2.0, 2.0, 2.0])
    orbital = (np.ones(grid.shape) + 0.25j).astype(np.complex64)
    potential = np.arange(grid.size, dtype=np.float32).reshape(grid.shape)

    applied = np.array(apply_local_potential(orbital, potential))

    np.testing.assert_allclose(applied, orbital * potential, atol=1e-7)


def test_dense_reference_matvec_matches_operator_application():
    grid = RealSpaceGrid((2, 2, 2), [2.0, 2.0, 2.0])
    density = np.ones(grid.shape, dtype=np.float32) * 0.5
    local = np.linspace(-0.2, 0.3, grid.size, dtype=np.float32).reshape(grid.shape)
    operator = KohnShamOperator.from_density(
        grid,
        local,
        density,
        xc_functional=DiracExchange(),
    )
    reference = DenseHamiltonianReference(operator)
    coordinates = np.array(grid.coordinates())
    trial = (1.0 + 0.1j * coordinates[..., 0]).astype(np.complex64)

    dense_applied = np.array(reference.matvec(trial))
    operator_applied = np.array(operator.apply_hamiltonian(trial))

    np.testing.assert_allclose(dense_applied, operator_applied, atol=1e-5)


def test_dense_reference_and_subspace_solver_agree_on_tiny_grid():
    grid = RealSpaceGrid((2, 2, 2), [2.0, 2.0, 2.0])
    density = np.ones(grid.shape, dtype=np.float32) * 0.5
    local = np.zeros(grid.shape, dtype=np.float32)
    operator = KohnShamOperator.from_density(
        grid,
        local,
        density,
        xc_functional=DiracExchange(),
    )

    dense = DenseHamiltonianReference(operator).diagonalize(n_orbitals=1)
    subspace = SubspaceDiagonalizer(tolerance=1e-5).solve(operator, n_orbitals=1)

    np.testing.assert_allclose(
        np.array(dense.eigenvalues),
        np.array(subspace.eigenvalues),
        atol=1e-6,
    )
    assert orthonormality_error(subspace.orbitals, grid) < 1e-6
    assert np.isfinite(np.array(subspace.residuals)).all()


def test_center_center_energy_symmetric_and_added_to_total_energy():
    system = DFTSystem.two_center(
        cell=[8.0, 8.0, 8.0],
        grid_shape=(4, 4, 4),
        centers=((3.0, 4.0, 4.0), (5.0, 4.0, 4.0)),
        amplitudes=(-1.0, -1.0),
        widths=(0.8, 0.8),
        charges=(1.0, 2.0),
    )
    swapped = DFTSystem.two_center(
        cell=[8.0, 8.0, 8.0],
        grid_shape=(4, 4, 4),
        centers=((5.0, 4.0, 4.0), (3.0, 4.0, 4.0)),
        amplitudes=(-1.0, -1.0),
        widths=(0.8, 0.8),
        charges=(2.0, 1.0),
    )

    assert center_center_energy(system) == pytest.approx(center_center_energy(swapped))
    result = run_scf(
        system,
        config=SCFConfig(max_iterations=1, solver="dense", seed=3),
        xc_functional=DiracExchange(),
    )

    assert result.center_center_energy == pytest.approx(center_center_energy(system))
    assert result.total_energy == pytest.approx(
        result.electronic_energy + result.center_center_energy,
        abs=1e-6,
    )


def test_scf_total_energy_force_check_is_json_safe_for_symmetric_one_center():
    system = DFTSystem.one_center(
        cell=[6.0, 6.0, 6.0],
        grid_shape=(4, 4, 4),
        center=(3.0, 3.0, 3.0),
        electron_count=2.0,
        amplitude=-2.0,
        width=0.8,
    )
    check = scf_total_energy_forces(
        system,
        config=SCFConfig(max_iterations=2, solver="dense", seed=7),
        xc_functional=DiracExchange(),
        displacement=1e-3,
    )

    assert check.max_abs_error < 5e-2
    assert check.result.force_consistency is not None
    json.dumps(check.to_dict())
