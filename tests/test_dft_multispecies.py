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
    PeriodicSCFConfig,
    PlaneWaveBasis,
    PseudopotentialData,
    PseudopotentialFormat,
    ReciprocalGrid,
    gth_local_reciprocal_coefficients,
    periodic_scf_calculation_contract,
    periodic_scf_forces,
    run_periodic_scf,
)
from mlx_atomistic.dft._compact import _CompactBatch


def _magnesium_gth() -> PseudopotentialData:
    return PseudopotentialData(
        element="Mg",
        format=PseudopotentialFormat.GTH,
        valence_charge=2.0,
        gth_rloc=0.35,
        gth_coefficients=(-2.0, 0.25),
        gth_channels=(GTHProjectorChannel(0, 0.38, ((0.7,),)),),
    )


def _oxygen_gth() -> PseudopotentialData:
    return PseudopotentialData(
        element="O",
        format=PseudopotentialFormat.GTH,
        valence_charge=6.0,
        gth_rloc=0.25,
        gth_coefficients=(-4.0, 0.4),
        gth_channels=(
            GTHProjectorChannel(0, 0.28, ((0.8, 0.1), (0.1, 0.5))),
            GTHProjectorChannel(1, 0.31, ((0.6,),)),
        ),
    )


def _mixed_system() -> PeriodicDFTSystem:
    return PeriodicDFTSystem(
        (8.0, 8.0, 8.0),
        (8, 8, 8),
        ((1.0, 2.0, 3.0), (5.0, 4.0, 3.0)),
        pseudopotentials=(_magnesium_gth(), _oxygen_gth()),
    )


def _bounded_binary_system() -> PeriodicDFTSystem:
    first = PseudopotentialData(
        element="A",
        format=PseudopotentialFormat.GTH,
        valence_charge=1.0,
        gth_rloc=0.3,
        gth_coefficients=(-1.0,),
        gth_channels=(GTHProjectorChannel(0, 0.32, ((0.5,),)),),
    )
    second = PseudopotentialData(
        element="B",
        format=PseudopotentialFormat.GTH,
        valence_charge=1.0,
        gth_rloc=0.27,
        gth_coefficients=(-1.2,),
        gth_channels=(GTHProjectorChannel(0, 0.29, ((0.7,),)),),
    )
    return PeriodicDFTSystem(
        (6.0, 6.0, 6.0),
        (6, 6, 6),
        ((2.0, 3.0, 3.0), (4.0, 3.0, 3.0)),
        pseudopotentials=(first, second),
    )


def _asymmetric_binary_system(
    positions=((1.7, 2.8, 3.1), (4.1, 3.5, 2.6)),
) -> PeriodicDFTSystem:
    reference = _bounded_binary_system()
    return PeriodicDFTSystem(
        reference.grid.lengths,
        reference.grid.shape,
        positions,
        pseudopotentials=reference.pseudopotentials,
    )


def _bounded_force_scf_config() -> PeriodicSCFConfig:
    return PeriodicSCFConfig(
        max_iterations=40,
        min_iterations=3,
        density_tolerance=3e-5,
        energy_tolerance=3e-6,
        orbital_tolerance=2e-5,
        mixing_beta=0.5,
        mixer="diis",
        davidson=PeriodicDavidsonConfig(
            max_iterations=32,
            tolerance=2e-5,
            max_subspace_size=20,
        ),
    )


def _random_state(basis: PlaneWaveBasis, *, seed: int):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(2, basis.active_count)) + 1j * rng.normal(
        size=(2, basis.active_count)
    )
    return basis._state_from_compact(mx.array(values.astype(np.complex64)))


def test_periodic_system_canonicalizes_per_ion_species_and_identity():
    system = _mixed_system()

    assert system.symbols == ("Mg", "O")
    assert system.charges == (2.0, 6.0)
    assert system.electron_count == 8.0
    assert not system.is_homogeneous
    assert len(system.fingerprint) == 64
    with pytest.raises(ValueError, match="one shared pseudopotential"):
        _ = system.pseudopotential

    swapped = PeriodicDFTSystem(
        system.grid.lengths,
        system.grid.shape,
        system.positions,
        pseudopotentials=tuple(reversed(system.pseudopotentials)),
    )
    assert swapped.fingerprint != system.fingerprint


def test_periodic_system_preserves_shared_pseudopotential_api_and_validates_inputs():
    magnesium = _magnesium_gth()
    shared = PeriodicDFTSystem(
        (8.0, 8.0, 8.0),
        (8, 8, 8),
        ((1.0, 2.0, 3.0), (5.0, 4.0, 3.0)),
        magnesium,
    )

    assert shared.pseudopotential is magnesium
    assert shared.pseudopotentials == (magnesium, magnesium)
    assert shared.electron_count == 4.0
    with pytest.raises(ValueError, match="exactly one"):
        PeriodicDFTSystem(
            (8.0, 8.0, 8.0),
            (8, 8, 8),
            ((1.0, 2.0, 3.0),),
        )
    with pytest.raises(ValueError, match="exactly one"):
        PeriodicDFTSystem(
            (8.0, 8.0, 8.0),
            (8, 8, 8),
            ((1.0, 2.0, 3.0),),
            magnesium,
            pseudopotentials=(magnesium,),
        )
    with pytest.raises(ValueError, match="ion count"):
        PeriodicDFTSystem(
            (8.0, 8.0, 8.0),
            (8, 8, 8),
            ((1.0, 2.0, 3.0),),
            pseudopotentials=(magnesium, magnesium),
        )


def test_mixed_local_gth_equals_sum_of_species_contributions():
    system = _mixed_system()
    basis = PlaneWaveBasis(system.grid, 4.0)
    mixed = gth_local_reciprocal_coefficients(
        system.pseudopotentials,
        basis,
        system.positions,
    )
    separate = sum(
        (
            gth_local_reciprocal_coefficients(
                pseudopotential,
                basis,
                (position,),
            )
            for pseudopotential, position in zip(
                system.pseudopotentials,
                system.positions,
                strict=True,
            )
        ),
        start=mx.zeros(system.grid.shape, dtype=mx.complex64),
    )

    np.testing.assert_allclose(np.asarray(mixed), np.asarray(separate), atol=2e-6)


def test_mixed_nonlocal_gth_equals_species_sum_and_is_hermitian():
    system = _mixed_system()
    basis = PlaneWaveBasis.from_reduced_kpoint(
        system.grid,
        4.0,
        (0.25, 0.125, -0.25),
    )
    mixed = PeriodicGTHNonlocalOperator(
        system.pseudopotentials,
        basis,
        system.positions,
    )
    separate = [
        PeriodicGTHNonlocalOperator(pseudopotential, basis, (position,))
        for pseudopotential, position in zip(
            system.pseudopotentials,
            system.positions,
            strict=True,
        )
    ]
    rng = np.random.default_rng(47)
    left = basis.normalize(
        mx.array(
            (rng.normal(size=system.grid.shape) + 1j * rng.normal(size=system.grid.shape)).astype(
                np.complex64
            )
        )
    )
    right = basis.normalize(
        mx.array(
            (rng.normal(size=system.grid.shape) + 1j * rng.normal(size=system.grid.shape)).astype(
                np.complex64
            )
        )
    )

    mixed_right = mixed.apply(right)
    summed_right = sum(
        (operator.apply(right) for operator in separate),
        start=mx.zeros(system.grid.shape, dtype=mx.complex64),
    )
    left_right = mx.sum(mx.conjugate(left) * mixed_right)
    right_left = mx.sum(mx.conjugate(right) * mixed.apply(left))

    np.testing.assert_allclose(
        np.asarray(mixed_right),
        np.asarray(summed_right),
        atol=3e-6,
    )
    assert np.asarray(left_right).item() == pytest.approx(
        np.asarray(mx.conjugate(right_left)).item(),
        abs=3e-5,
    )
    metadata = mixed.to_dict()
    assert metadata["species_count"] == 2
    assert metadata["angular_projector_count_per_ion"] == [1, 5]
    assert metadata["angular_projector_count_total"] == 6


def test_mixed_nonlocal_gth_batch_matches_lane_actions():
    system = _mixed_system()
    reciprocal = ReciprocalGrid.from_real_space(system.grid)
    bases = (
        PlaneWaveBasis.from_reduced_kpoint(
            system.grid,
            4.0,
            (0.0, 0.0, 0.0),
            reciprocal_grid=reciprocal,
        ),
        PlaneWaveBasis.from_reduced_kpoint(
            system.grid,
            4.0,
            (0.25, 0.0, 0.0),
            reciprocal_grid=reciprocal,
        ),
    )
    states = tuple(
        _random_state(basis, seed=seed) for basis, seed in zip(bases, (1, 2), strict=True)
    )
    lane_operators = tuple(
        PeriodicGTHNonlocalOperator(
            system.pseudopotentials,
            basis,
            system.positions,
        )
        for basis in bases
    )
    expected = tuple(
        operator._apply_compact(state)[0]
        for operator, state in zip(lane_operators, states, strict=True)
    )
    batch_operators = tuple(
        PeriodicGTHNonlocalOperator(
            system.pseudopotentials,
            basis,
            system.positions,
        )
        for basis in bases
    )
    batch = _CompactBatch.from_states(states)

    observed, _metrics = PeriodicGTHNonlocalOperator._apply_compact_batch(
        batch_operators,
        states,
        batch=batch,
    )

    for expected_state, observed_state in zip(expected, observed, strict=True):
        np.testing.assert_allclose(
            np.asarray(observed_state.values),
            np.asarray(expected_state.values),
            atol=3e-6,
        )


def test_multi_element_checkpoint_contract_records_ordered_species_assignment():
    system = _mixed_system()
    mesh = KPointMesh(
        [KPoint((0.0, 0.0, 0.0), coordinate_system="reduced")]
    )

    contract = periodic_scf_calculation_contract(
        system,
        cutoff_hartree=4.0,
        kpoint_mesh=mesh,
        n_bands=4,
    )

    payload = contract["system"]["pseudopotentials"]
    assert [species["element"] for species in payload["species"]] == ["Mg", "O"]
    assert payload["atom_species"] == [0, 1]

    swapped = PeriodicDFTSystem(
        system.grid.lengths,
        system.grid.shape,
        system.positions,
        pseudopotentials=tuple(reversed(system.pseudopotentials)),
    )
    swapped_contract = periodic_scf_calculation_contract(
        swapped,
        cutoff_hartree=4.0,
        kpoint_mesh=mesh,
        n_bands=4,
    )
    assert swapped_contract != contract


def test_bounded_multi_element_scf_converges_and_binds_system_identity():
    system = _bounded_binary_system()
    mesh = KPointMesh(
        [KPoint((0.0, 0.0, 0.0), coordinate_system="reduced")]
    )
    result = run_periodic_scf(
        system,
        cutoff_hartree=2.5,
        kpoint_mesh=mesh,
        n_bands=1,
        config=PeriodicSCFConfig(
            max_iterations=12,
            min_iterations=2,
            density_tolerance=2e-3,
            energy_tolerance=2e-3,
            orbital_tolerance=5e-4,
            mixing_beta=0.5,
            mixer="linear",
            davidson=PeriodicDavidsonConfig(
                max_iterations=24,
                tolerance=5e-4,
                max_subspace_size=16,
            ),
        ),
    )

    assert result.converged
    # electron_count is a grid quadrature, not an exact integer: it lands within
    # float32 summation error of 2 electrons (mlx-cpu vs the Metal build differ
    # here by ~1e-6), so match with a tolerance rather than bit-exact equality.
    assert result.electron_count == pytest.approx(2.0, abs=1e-5)
    assert result.system_fingerprint == system.fingerprint
    assert float(mx.sum(result.density) * system.grid.dv) == pytest.approx(
        2.0,
        abs=1e-4,
    )
    force_result = periodic_scf_forces(system, result)
    observed = np.asarray(force_result.forces)
    expected = (
        np.asarray(force_result.local)
        + np.asarray(force_result.nonlocal_force)
        + np.asarray(force_result.ion_ewald)
    )

    assert observed.shape == (2, 3)
    assert np.isfinite(observed).all()
    np.testing.assert_allclose(observed, expected, atol=2e-7)
    assert force_result.provenance["pulay"] == (
        "zero_for_fixed_cell_plane_wave_basis"
    )


def test_periodic_scf_force_matches_bounded_total_energy_derivative():
    mesh = KPointMesh(
        [KPoint((0.0, 0.0, 0.0), coordinate_system="reduced")]
    )
    system = _asymmetric_binary_system()
    config = _bounded_force_scf_config()
    reference = run_periodic_scf(
        system,
        cutoff_hartree=2.5,
        kpoint_mesh=mesh,
        n_bands=1,
        config=config,
    )
    assert reference.converged
    analytic = float(periodic_scf_forces(system, reference).forces[0, 0])

    displacement = 0.01
    energies = []
    for offset in (-displacement, displacement):
        positions = np.array(system.positions, copy=True)
        positions[0, 0] += offset
        displaced = _asymmetric_binary_system(positions)
        result = run_periodic_scf(
            displaced,
            cutoff_hartree=2.5,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
        )
        assert result.converged
        energies.append(result.total_energy)
    finite_difference = -(energies[1] - energies[0]) / (
        2.0 * displacement
    )

    assert analytic == pytest.approx(finite_difference, abs=1e-4)
