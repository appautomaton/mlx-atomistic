import mlx.core as mx
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
    analytic_configurational_virial_tensor,
    finite_difference_configurational_virial_oracle,
    kinetic_pressure_tensor,
    missing_analytic_virial_support,
    missing_virial_support,
    pressure_tensor,
    simulate_nve,
    simulate_nvt,
    validate_analytic_virial_support,
    validate_virial_support,
    virial_readiness_report,
)
from mlx_atomistic.neighbors import NeighborListManager
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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
        virial_mode="finite_difference_oracle",
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


def test_selected_bond_virial_support_is_analytic():
    term = HarmonicBondPotential([(0, 1)], k=10.0, length=1.0)

    report = virial_readiness_report([term])
    production = virial_readiness_report([term], require_analytic=True)

    assert term.analytic_virial_supported is True
    assert report.status == "ready"
    assert report.blockers == ()
    assert report.metadata["term_support"] == {"bond": "analytic"}
    assert production.status == "ready"
    assert production.blockers == ()
    assert missing_analytic_virial_support([term]) == ()
    validate_analytic_virial_support([term])


def test_molecular_kinetic_pressure_uses_center_of_mass_motion():
    velocities = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
        ],
        dtype=np.float32,
    )
    masses = np.ones((4,), dtype=np.float32)

    atom_tensor = kinetic_pressure_tensor(velocities, masses)
    molecular_tensor = kinetic_pressure_tensor(
        velocities,
        masses,
        molecule_ids=np.asarray([0, 0, 1, 1], dtype=np.int32),
    )

    np.testing.assert_allclose(
        np.asarray(atom_tensor),
        np.diag([10.0, 8.0, 0.0]),
    )
    np.testing.assert_allclose(
        np.asarray(molecular_tensor),
        np.diag([8.0, 0.0, 0.0]),
    )


def test_intramolecular_term_reports_explicit_zero_molecular_virial():
    class IntramolecularBond:
        name = "intramolecular_bond"
        supports_virial = True
        analytic_virial_supported = True

        def energy_forces(self, positions, cell=None, pairs=None):
            del pairs
            displacement = positions[0] - positions[1]
            if cell is not None:
                displacement = cell.minimum_image(displacement)
            distance = np.sqrt(np.sum(np.asarray(displacement) ** 2))
            energy = 0.5 * (distance - 1.0) ** 2
            direction = displacement / distance
            force = -(distance - 1.0) * direction
            forces = mx.zeros_like(positions).at[0].add(force).at[1].add(-force)
            return mx.array(energy, dtype=positions.dtype), forces

        def analytic_virial_tensor(
            self,
            positions,
            *,
            cell,
            pairs,
            masses,
            molecule_ids,
        ):
            del cell, pairs, masses, molecule_ids
            return mx.zeros((3, 3), dtype=positions.dtype)

    positions = mx.array([[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]])
    masses = mx.array([1.0, 1.0])
    molecule_ids = np.asarray([0, 0], dtype=np.int32)
    cell = Cell.cubic(6.0)
    term = IntramolecularBond()
    _, forces = term.energy_forces(positions, cell)

    analytic = analytic_configurational_virial_tensor(
        positions,
        forces,
        (term,),
        cell=cell,
        pairs=None,
        masses=masses,
        molecule_ids=molecule_ids,
    )
    oracle = finite_difference_configurational_virial_oracle(
        positions,
        forces,
        (term,),
        cell=cell,
        pairs=None,
        masses=masses,
        molecule_ids=molecule_ids,
    )

    np.testing.assert_array_equal(np.asarray(analytic), np.zeros((3, 3)))
    np.testing.assert_allclose(np.asarray(oracle), np.zeros((3, 3)), atol=2.0e-5)


@pytest.mark.parametrize(
    ("term", "positions"),
    [
        (
            HarmonicBondPotential([(0, 1)], k=4.0, length=1.0),
            [[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]],
        ),
        (
            HarmonicAnglePotential([(0, 1, 2)], k=3.0, angle=1.4),
            [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [2.1, 2.0, 1.0]],
        ),
        (
            PeriodicDihedralPotential(
                [(0, 1, 2, 3)],
                k=0.4,
                periodicity=3.0,
                phase=0.2,
            ),
            [
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [2.2, 2.0, 1.0],
                [3.0, 2.2, 1.8],
            ],
        ),
    ],
)
def test_selected_intramolecular_terms_have_zero_analytic_cell_virial(
    term,
    positions,
):
    positions = mx.array(positions, dtype=mx.float32)
    atom_count = positions.shape[0]
    cell = Cell.cubic(8.0)
    masses = mx.ones((atom_count,), dtype=mx.float32)
    molecule_ids = np.zeros((atom_count,), dtype=np.int32)
    _, forces = term.energy_forces(positions, cell)

    analytic = analytic_configurational_virial_tensor(
        positions,
        forces,
        (term,),
        cell=cell,
        pairs=None,
        masses=masses,
        molecule_ids=molecule_ids,
    )
    oracle = finite_difference_configurational_virial_oracle(
        positions,
        forces,
        (term,),
        cell=cell,
        pairs=None,
        masses=masses,
        molecule_ids=molecule_ids,
    )

    np.testing.assert_allclose(np.asarray(analytic), 0.0, atol=2.0e-6)
    np.testing.assert_allclose(np.asarray(analytic), np.asarray(oracle), atol=2.0e-5)


def test_selected_pme_nonbonded_analytic_virial_matches_oracle():
    positions = mx.array(
        [
            [1.0, 1.0, 1.0],
            [1.9, 1.1, 1.0],
            [4.2, 4.0, 4.0],
            [5.0, 4.2, 4.1],
        ],
        dtype=mx.float32,
    )
    cell = Cell.cubic(8.0)
    config = PMEConfig(
        mesh_shape=(8, 8, 8),
        alpha=0.4,
        real_cutoff=3.0,
        assignment_order=4,
    )
    term = NonbondedPotential(
        sigma=[0.9, 1.0, 1.1, 0.95],
        epsilon=[0.15, 0.2, 0.18, 0.12],
        charges=[0.4, -0.4, 0.25, -0.25],
        cutoff=3.0,
        lj_shift=False,
        electrostatics="pme",
        pme_config=config,
        exception_pairs=[(0, 1)],
        exception_charge_products=[-0.08],
        exception_sigma=[0.95],
        exception_epsilon=[0.1],
    )
    manager = NeighborListManager(
        cell,
        cutoff=3.0,
        skin=0.0,
        backend="mlx_cell_blocks",
    )
    pairs = manager.update(positions).interactions
    molecule_ids = np.asarray([0, 0, 1, 1], dtype=np.int32)
    masses = mx.ones((4,), dtype=mx.float32)
    _, forces = term.energy_forces(positions, cell, pairs)

    analytic = analytic_configurational_virial_tensor(
        positions,
        forces,
        (term,),
        cell=cell,
        pairs=pairs,
        masses=masses,
        molecule_ids=molecule_ids,
    )
    oracle = finite_difference_configurational_virial_oracle(
        positions,
        forces,
        (term,),
        cell=cell,
        pairs=pairs,
        masses=masses,
        molecule_ids=molecule_ids,
        strain_epsilon=2.0e-3,
    )

    np.testing.assert_allclose(
        np.diag(np.asarray(analytic)),
        np.diag(np.asarray(oracle)),
        rtol=7.0e-3,
        atol=3.0e-4,
    )


def test_analytic_pair_virial_matches_molecular_cell_strain_oracle():
    class IntermolecularHarmonicPair:
        name = "intermolecular_pair"
        supports_virial = True
        analytic_virial_supported = True

        def energy_forces(self, positions, cell=None, pairs=None):
            del pairs
            displacement = positions[0] - positions[1]
            if cell is not None:
                displacement = cell.minimum_image(displacement)
            distance = mx.sqrt(mx.sum(displacement * displacement))
            extension = distance - 1.0
            force = -extension * displacement / distance
            forces = mx.zeros_like(positions).at[0].add(force).at[1].add(-force)
            return 0.5 * extension * extension, forces

        def analytic_virial_tensor(
            self,
            positions,
            *,
            cell,
            pairs,
            masses,
            molecule_ids,
        ):
            del pairs, masses, molecule_ids
            displacement = cell.minimum_image(positions[0] - positions[1])
            _, forces = self.energy_forces(positions, cell)
            return mx.outer(displacement, forces[0])

    positions = mx.array([[1.0, 1.0, 1.0], [2.3, 1.4, 1.2]])
    masses = mx.array([1.0, 2.0])
    molecule_ids = np.asarray([0, 1], dtype=np.int32)
    cell = Cell.cubic(6.0)
    term = IntermolecularHarmonicPair()
    _, forces = term.energy_forces(positions, cell)

    analytic = analytic_configurational_virial_tensor(
        positions,
        forces,
        (term,),
        cell=cell,
        pairs=None,
        masses=masses,
        molecule_ids=molecule_ids,
    )
    oracle = finite_difference_configurational_virial_oracle(
        positions,
        forces,
        (term,),
        cell=cell,
        pairs=None,
        masses=masses,
        molecule_ids=molecule_ids,
        strain_epsilon=2.0e-3,
    )

    np.testing.assert_allclose(
        np.diag(np.asarray(analytic)),
        np.diag(np.asarray(oracle)),
        rtol=2.0e-3,
        atol=3.0e-4,
    )
    assert virial_readiness_report([term], require_analytic=True).status == "ready"


def test_analytic_pressure_never_calls_finite_difference_energy_path():
    class AnalyticOnlyTerm:
        name = "analytic_only"
        supports_virial = False
        analytic_virial_supported = True

        def energy_forces(self, positions, cell=None, pairs=None):
            raise AssertionError("analytic pressure must not evaluate energy")

        def analytic_virial_tensor(
            self,
            positions,
            *,
            cell,
            pairs,
            masses,
            molecule_ids,
        ):
            del cell, pairs, masses, molecule_ids
            return mx.diag(mx.array([1.0, 2.0, 3.0], dtype=positions.dtype))

    positions = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    masses = np.ones((1,), dtype=np.float32)
    forces = np.zeros_like(positions)

    virial, tensor, scalar = pressure_tensor(
        positions,
        velocities,
        masses,
        forces,
        (AnalyticOnlyTerm(),),
        cell=Cell.cubic(2.0),
        pairs=None,
        molecule_ids=np.asarray([0], dtype=np.int32),
    )

    np.testing.assert_allclose(np.asarray(virial), np.diag([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(np.asarray(tensor), np.diag([0.125, 0.25, 0.375]))
    assert float(np.asarray(scalar)) == pytest.approx(0.25)
