import numpy as np
import pytest

from mlx_atomistic.charmm_terms import (
    CHARMMCMAPPotential,
    CHARMMForceSwitchNonbondedPotential,
    CHARMMNBFIXPairOverridePotential,
    CHARMMUreyBradleyPotential,
)
from mlx_atomistic.core import Cell
from mlx_atomistic.forcefields import (
    CoulombPotential,
    HarmonicAnglePotential,
    HarmonicBondPotential,
    NonbondedPotential,
    PairRestrictedNonbondedPotential,
    PeriodicDihedralPotential,
    PositionalRestraintPotential,
)
from mlx_atomistic.io import load_npz_trajectory, save_npz_trajectory
from mlx_atomistic.md import (
    LennardJonesPotential,
    SimulationConfig,
    missing_virial_support,
    pressure_tensor,
    simulate_nve,
    simulate_nvt,
    validate_virial_support,
)
from mlx_atomistic.pme import PMEConfig


def _periodic_fixture():
    positions = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [2.2, 1.0, 1.0],
            [1.0, 2.2, 1.0],
            [2.2, 2.2, 1.0],
        ],
        dtype=np.float32,
    )
    velocities = np.asarray(
        [
            [0.02, 0.0, 0.0],
            [-0.01, 0.01, 0.0],
            [0.0, -0.02, 0.0],
            [0.0, 0.01, 0.01],
        ],
        dtype=np.float32,
    )
    terms = [
        HarmonicBondPotential([(0, 1)], k=10.0, length=1.2),
        PositionalRestraintPotential(positions, [True, False, False, False], k=0.2),
        NonbondedPotential(
            sigma=[1.0, 1.0, 1.0, 1.0],
            epsilon=[0.1, 0.1, 0.1, 0.1],
            charges=[1.0, -1.0, 1.0, -1.0],
            cutoff=2.5,
            electrostatics="pme",
            pme_config=PMEConfig(mesh_shape=(8, 8, 8), alpha=0.35, real_cutoff=2.5),
        ),
    ]
    return positions, velocities, Cell.cubic(6.0), terms


def test_nve_reports_finite_periodic_virial_and_pressure_with_pme_terms():
    positions, velocities, cell, terms = _periodic_fixture()

    result = simulate_nve(
        positions,
        velocities,
        cell=cell,
        force_terms=terms,
        config=SimulationConfig(dt=0.001, steps=2, sample_interval=1),
    )

    assert np.asarray(result.virial_tensor).shape == (3, 3, 3)
    assert np.asarray(result.pressure_tensor).shape == (3, 3, 3)
    assert np.asarray(result.pressure).shape == (3,)
    assert np.isfinite(np.asarray(result.virial_tensor)).all()
    assert np.isfinite(np.asarray(result.pressure_tensor)).all()
    assert np.isfinite(np.asarray(result.pressure)).all()
    assert "nonbonded.pme_diagnostics" not in result.potential_energy_by_term


def test_nvt_pressure_diagnostics_follow_sparse_diagnostic_axis():
    positions, velocities, cell, terms = _periodic_fixture()

    result = simulate_nvt(
        positions,
        velocities,
        cell=cell,
        force_terms=terms,
        config=SimulationConfig(
            dt=0.001,
            steps=3,
            sample_interval=3,
            diagnostic_interval=2,
        ),
    )

    assert np.asarray(result.diagnostic_steps).tolist() == [0, 2, 3]
    assert np.asarray(result.pressure).shape == (3,)
    assert np.isfinite(np.asarray(result.pressure_tensor)).all()


def test_trajectory_round_trips_virial_and_pressure_diagnostics(tmp_path):
    positions, velocities, cell, terms = _periodic_fixture()
    result = simulate_nve(
        positions,
        velocities,
        cell=cell,
        force_terms=terms,
        config=SimulationConfig(dt=0.001, steps=1, sample_interval=1),
    )

    path = tmp_path / "trajectory.npz"
    save_npz_trajectory(path, result, cell=cell)
    record = load_npz_trajectory(path)

    np.testing.assert_allclose(record.virial_tensor, np.asarray(result.virial_tensor))
    np.testing.assert_allclose(record.pressure_tensor, np.asarray(result.pressure_tensor))
    np.testing.assert_allclose(record.pressure, np.asarray(result.pressure))


def test_periodic_virial_is_invariant_to_equivalent_wrapped_positions():
    cell = Cell.cubic(6.0)
    velocities = np.zeros((2, 3), dtype=np.float32)
    potential = LennardJonesPotential(cutoff=2.5)
    first = simulate_nve(
        np.asarray([[1.0, 1.0, 1.0], [5.0, 1.0, 1.0]], dtype=np.float32),
        velocities,
        cell=cell,
        force_terms=potential,
        config=SimulationConfig(steps=0),
    )
    equivalent = simulate_nve(
        np.asarray([[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]], dtype=np.float32),
        velocities,
        cell=cell,
        force_terms=potential,
        config=SimulationConfig(steps=0),
    )

    np.testing.assert_allclose(
        np.asarray(first.virial_tensor),
        np.asarray(equivalent.virial_tensor),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(first.pressure_tensor),
        np.asarray(equivalent.pressure_tensor),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(first.pressure),
        np.asarray(equivalent.pressure),
        rtol=1e-5,
        atol=1e-5,
    )


def test_triclinic_pressure_uses_matrix_volume():
    class ZeroVirialTerm:
        supports_virial = True

        def energy_forces(self, positions, cell=None, pairs=None):
            del cell, pairs
            return positions[:, 0].sum() * 0.0, positions * 0.0

    matrix = np.asarray(
        [
            [4.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.5, 0.25, 2.0],
        ],
        dtype=np.float32,
    )
    cell = Cell.triclinic(matrix)
    positions = np.asarray([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float32)
    velocities = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    masses = np.asarray([2.0, 3.0], dtype=np.float32)
    forces = np.zeros_like(positions)

    _, tensor, scalar = pressure_tensor(
        positions,
        velocities,
        masses,
        forces,
        (ZeroVirialTerm(),),
        cell=cell,
        pairs=None,
    )

    kinetic = np.asarray([[2.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 0.0]])
    expected_tensor = kinetic / np.linalg.det(matrix)
    np.testing.assert_allclose(np.asarray(tensor), expected_tensor, atol=1e-5)
    np.testing.assert_allclose(np.asarray(scalar), np.trace(expected_tensor) / 3.0, atol=1e-5)


def test_old_trajectory_without_pressure_fields_loads_zero_defaults(tmp_path):
    path = tmp_path / "old_trajectory.npz"
    steps = np.asarray([0, 1], dtype=np.int32)
    values = np.asarray([0.0, 1.0], dtype=np.float32)
    np.savez_compressed(
        path,
        sampled_positions=np.zeros((2, 2, 3), dtype=np.float32),
        sampled_velocities=np.zeros((2, 2, 3), dtype=np.float32),
        sampled_steps=steps,
        sampled_time=values,
        diagnostic_steps=steps,
        diagnostic_time=values,
        potential_energy=values,
        kinetic_energy=values,
        total_energy=values,
        temperature=values,
        pair_count=steps,
        rebuild_count=steps,
        constraint_max_error=values,
        symbols=np.asarray(["H", "H"], dtype=str),
        cell=np.asarray([6.0, 6.0, 6.0], dtype=np.float32),
        metadata_json=np.asarray("{}"),
        energy_term_names=np.asarray([], dtype=str),
    )

    record = load_npz_trajectory(path)

    np.testing.assert_allclose(record.virial_tensor, np.zeros((2, 3, 3), dtype=np.float32))
    np.testing.assert_allclose(record.pressure_tensor, np.zeros((2, 3, 3), dtype=np.float32))
    np.testing.assert_allclose(record.pressure, np.zeros((2,), dtype=np.float32))


def test_unsupported_terms_report_exact_missing_virial_names():
    class UnsupportedTerm:
        name = "custom_bias"

        def energy_forces(self, positions, cell=None, pairs=None):
            del cell, pairs
            return positions[:, 0].sum() * 0.0, positions * 0.0

    assert missing_virial_support([UnsupportedTerm()]) == ("custom_bias",)
    with pytest.raises(ValueError, match="custom_bias"):
        validate_virial_support([UnsupportedTerm()])


def test_internal_looking_term_without_explicit_virial_support_fails_closed():
    InternalLookingTerm = type(
        "InternalLookingTerm",
        (),
        {
            "__module__": "mlx_atomistic.forcefields",
            "name": "internal_without_virial",
            "energy_forces": lambda self, positions, cell=None, pairs=None: (
                positions[:, 0].sum() * 0.0,
                positions * 0.0,
            ),
        },
    )

    assert missing_virial_support([InternalLookingTerm()]) == ("internal_without_virial",)
    with pytest.raises(ValueError, match="internal_without_virial"):
        validate_virial_support([InternalLookingTerm()])


def test_explicitly_supported_built_in_terms_pass_virial_gate():
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [1.2, 1.0, 0.0],
            [1.4, 1.1, 0.8],
            [1.6, 1.3, 1.2],
        ],
        dtype=np.float32,
    )
    nonbonded = NonbondedPotential(
        sigma=[1.0] * 5,
        epsilon=[0.1] * 5,
        charges=[1.0, -1.0, 1.0, -1.0, 0.0],
        cutoff=2.5,
    )
    terms = [
        LennardJonesPotential(cutoff=2.5),
        HarmonicBondPotential([(0, 1)], k=10.0, length=1.1),
        HarmonicAnglePotential([(0, 1, 2)], k=2.0, angle=1.5),
        PeriodicDihedralPotential([(0, 1, 2, 3)], k=0.2, periodicity=3.0),
        PositionalRestraintPotential(positions, [True, False, False, False, False], k=0.1),
        CoulombPotential(charges=[1.0, -1.0, 1.0, -1.0, 0.0], cutoff=2.5),
        nonbonded,
        PairRestrictedNonbondedPotential(nonbonded, pairs=[(0, 1), (2, 3)]),
        CHARMMUreyBradleyPotential([(0, 1, 2)], k=5.0, distance=1.5),
        CHARMMCMAPPotential(
            [(0, 1, 2, 3, 1, 2, 3, 4)],
            np.zeros((4, 4), dtype=np.float32),
        ),
        CHARMMForceSwitchNonbondedPotential(
            sigma=[1.0] * 5,
            epsilon=[0.1] * 5,
            charges=[1.0, -1.0, 1.0, -1.0, 0.0],
            cutoff=2.5,
            switch_distance=2.0,
        ),
        CHARMMNBFIXPairOverridePotential(
            sigma=[1.0] * 5,
            epsilon=[0.1] * 5,
            charges=[1.0, -1.0, 1.0, -1.0, 0.0],
            nbfix_pairs=[(0, 1)],
            nbfix_sigma=[1.1],
            nbfix_epsilon=[0.2],
            cutoff=2.5,
        ),
    ]

    assert missing_virial_support(terms) == ()
    validate_virial_support(terms)
