from math import pi

import numpy as np
import pytest

from mlx_atomistic.constraints import DistanceConstraints
from mlx_atomistic.examples import water_like_constrained_example
from mlx_atomistic.forcefields import CoulombPotential, NonbondedPotential
from mlx_atomistic.io import (
    load_npz_trajectory,
    read_xyz,
    restart_state_from_trajectory,
    save_npz_trajectory,
    write_xyz,
)
from mlx_atomistic.md import (
    LangevinThermostat,
    LennardJonesPotential,
    SimulationConfig,
    simulate_nve,
    simulate_nvt,
)
from mlx_atomistic.mm import (
    AngleParameter,
    AtomType,
    BondParameter,
    DihedralParameter,
    ForceField,
    MMSystem,
    NonbondedParameter,
)
from mlx_atomistic.topology import Topology


def finite_difference_force(term, positions, *, epsilon=1e-3):
    positions = np.array(positions, dtype=np.float32)
    forces = np.zeros_like(positions)
    for atom in range(positions.shape[0]):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom, axis] += epsilon
            minus[atom, axis] -= epsilon
            e_plus = float(np.array(term.energy_forces(plus)[0]))
            e_minus = float(np.array(term.energy_forces(minus)[0]))
            forces[atom, axis] = -(e_plus - e_minus) / (2.0 * epsilon)
    return forces


def _typed_chain():
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1), (1, 2), (2, 3)],
        angles=[(0, 1, 2), (1, 2, 3)],
        dihedrals=[(0, 1, 2, 3)],
        partial_charges=[0.0, 0.0, 0.0, 0.0],
    )
    force_field = ForceField(
        atom_types=[AtomType("A", 1.0), AtomType("B", 2.0)],
        nonbonded=[
            NonbondedParameter("A", sigma=1.0, epsilon=1.0),
            NonbondedParameter("B", sigma=0.8, epsilon=0.5),
        ],
        bonds=[BondParameter(("B", "A"), k=10.0, length=1.0)],
        angles=[
            AngleParameter(("A", "B", "A"), k=2.0, angle=pi / 2.0),
            AngleParameter(("B", "A", "B"), k=3.0, angle=pi / 2.0),
        ],
        dihedrals=[DihedralParameter(("B", "A", "B", "A"), k=0.2, periodicity=3.0)],
        cutoff=None,
        lj_shift=False,
    )
    system = MMSystem.from_sequences(
        symbols=["X", "X", "X", "X"],
        atom_names=["X1", "X2", "X3", "X4"],
        atom_types=["A", "B", "A", "B"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.2, 1.0, 0.0],
            [1.4, 1.1, 0.8],
        ],
        topology=topology,
        atom_type_masses=force_field.atom_type_masses,
    )
    return system, force_field


def test_mm_system_defaults_and_force_field_assignment():
    system, force_field = _typed_chain()

    assert np.array(system.velocities).shape == (4, 3)
    assert np.array(system.charges).tolist() == [0.0, 0.0, 0.0, 0.0]
    assert np.array(system.masses).tolist() == [1.0, 2.0, 1.0, 2.0]

    terms = force_field.build_force_terms(system)
    assert [term.name for term in terms] == ["bond", "angle", "dihedral", "nonbonded"]


def test_force_field_missing_parameter_reports_indices_and_types():
    system, force_field = _typed_chain()
    incomplete = ForceField(
        atom_types=force_field.atom_types,
        nonbonded=force_field.nonbonded,
        bonds=[],
    )

    try:
        incomplete.build_force_terms(system)
    except ValueError as err:
        message = str(err)
    else:
        raise AssertionError("missing parameter did not raise")

    assert "bond (0, 1)" in message
    assert "('A', 'B')" in message


def test_nonbonded_mixing_and_force_matches_finite_difference():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.25, 0.1, 0.0], [0.2, 1.35, 0.0]],
        dtype=np.float32,
    )
    term = NonbondedPotential(
        sigma=[1.0, 0.8, 1.2],
        epsilon=[1.0, 0.25, 0.81],
        charges=[1.0, -0.5, 0.25],
        cutoff=None,
        lj_shift=False,
    )

    sigma, epsilon = term.mixed_pair_parameters(np.array([[0, 1], [1, 2]], dtype=np.int32))
    np.testing.assert_allclose(np.array(sigma), [0.9, 1.0], atol=1e-6)
    np.testing.assert_allclose(np.array(epsilon), [0.5, 0.45], atol=1e-6)

    _, forces = term.energy_forces(positions)
    np.testing.assert_allclose(
        np.array(forces),
        finite_difference_force(term, positions),
        atol=5e-3,
    )


def test_uniform_nonbonded_matches_legacy_lj_and_coulomb():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.4, 0.1, 0.0], [0.2, 1.5, 0.0]],
        dtype=np.float32,
    )
    charges = [1.0, -0.5, 0.25]
    combined = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0],
        epsilon=[1.0, 1.0, 1.0],
        charges=charges,
        cutoff=None,
        lj_shift=False,
    )
    lj = LennardJonesPotential(cutoff=None, shift=False)
    coulomb = CoulombPotential(charges=charges, cutoff=None)

    combined_energy, combined_forces = combined.energy_forces(positions)
    lj_energy, lj_forces = lj.energy_forces(positions)
    coulomb_energy, coulomb_forces = coulomb.energy_forces(positions)

    np.testing.assert_allclose(
        np.array(combined_energy),
        np.array(lj_energy + coulomb_energy),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(combined_forces),
        np.array(lj_forces + coulomb_forces),
        atol=1e-6,
    )


def test_nonbonded_component_energy_diagnostics():
    positions = np.array([[0.0, 0.0, 0.0], [1.4, 0.1, 0.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    term = NonbondedPotential(
        sigma=[1.0, 1.0],
        epsilon=[1.0, 1.0],
        charges=[1.0, -0.5],
        cutoff=None,
        lj_shift=False,
    )

    result = simulate_nve(
        positions,
        velocities,
        force_terms=[term],
        config=SimulationConfig(dt=0.001, steps=1, sample_interval=1),
    )

    assert set(result.potential_energy_by_term) == {"nonbonded.lj", "nonbonded.coulomb"}
    reconstructed = sum(np.array(series) for series in result.potential_energy_by_term.values())
    np.testing.assert_allclose(reconstructed, np.array(result.potential_energy), atol=1e-6)


def test_nonbonded_exclusions_and_independent_one_four_scaling():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0], [3.6, 0.0, 0.0]],
        dtype=np.float32,
    )
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1)],
        dihedrals=[(0, 1, 2, 3)],
        partial_charges=[1.0, -1.0, 1.0, -1.0],
    )
    term = NonbondedPotential(
        sigma=[1.0] * 4,
        epsilon=[1.0] * 4,
        charges=[1.0, -1.0, 1.0, -1.0],
        topology=topology,
        cutoff=None,
        lj_shift=False,
        lj_one_four_scale=0.5,
        coulomb_one_four_scale=0.25,
    )
    components = term.component_energies(positions)
    unscaled = NonbondedPotential(
        sigma=[1.0] * 4,
        epsilon=[1.0] * 4,
        charges=[1.0, -1.0, 1.0, -1.0],
        cutoff=None,
        lj_shift=False,
    )
    pairs = topology.nonbonded_pairs()
    sigma, epsilon = unscaled.mixed_pair_parameters(pairs)

    pair_list = np.array(pairs).tolist()
    assert pair_list == [[0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    np.testing.assert_allclose(np.array(sigma), np.ones(5), atol=1e-6)
    np.testing.assert_allclose(np.array(epsilon), np.ones(5), atol=1e-6)
    expected_lj = 0.0
    expected_coulomb = 0.0
    charges = [1.0, -1.0, 1.0, -1.0]
    for atom_i, atom_j in pair_list:
        distance = float(np.linalg.norm(positions[atom_i] - positions[atom_j]))
        lj_scale = 0.5 if (atom_i, atom_j) == (0, 3) else 1.0
        coulomb_scale = 0.25 if (atom_i, atom_j) == (0, 3) else 1.0
        inv_r6 = (1.0 / distance) ** 6
        expected_lj += lj_scale * 4.0 * (inv_r6 * inv_r6 - inv_r6)
        expected_coulomb += coulomb_scale * charges[atom_i] * charges[atom_j] / distance
    np.testing.assert_allclose(np.array(components["lj"]), expected_lj, atol=1e-6)
    np.testing.assert_allclose(np.array(components["coulomb"]), expected_coulomb, atol=1e-6)


def test_constraints_preserve_distances_and_zero_friction_matches_nve():
    system, force_field, constraints = water_like_constrained_example()
    terms = force_field.build_force_terms(system)
    config = SimulationConfig(dt=0.001, steps=5, sample_interval=5)

    nve = simulate_nve(
        system.positions,
        system.velocities,
        masses=system.masses,
        force_terms=terms,
        config=config,
        constraints=constraints,
    )
    nvt = simulate_nvt(
        system.positions,
        system.velocities,
        masses=system.masses,
        force_terms=terms,
        config=config,
        thermostat=LangevinThermostat(temperature=1.0, friction=0.0, seed=3),
        constraints=constraints,
    )

    assert np.max(np.array(nve.constraint_max_error)) < 1e-5
    np.testing.assert_allclose(
        np.array(nvt.sampled_positions),
        np.array(nve.sampled_positions),
        atol=1e-5,
    )


def test_invalid_constraints_fail_clearly():
    with pytest.raises(ValueError, match="positive"):
        DistanceConstraints([(0, 1)], distances=[0.0])
    constraints = DistanceConstraints([(0, 3)], distances=[1.0])
    with pytest.raises(ValueError, match="outside positions"):
        constraints.apply_positions(
            np.zeros((2, 3), dtype=np.float32),
            np.ones((2,), dtype=np.float32),
        )


def test_xyz_and_npz_trajectory_roundtrip_and_restart(tmp_path):
    system, force_field, constraints = water_like_constrained_example()
    terms = force_field.build_force_terms(system)
    result = simulate_nve(
        system.positions,
        system.velocities,
        masses=system.masses,
        force_terms=terms,
        config=SimulationConfig(dt=0.001, steps=2, sample_interval=1),
        constraints=constraints,
    )

    xyz_path = tmp_path / "water.xyz"
    write_xyz(xyz_path, system.symbols, system.positions, comment="water-like")
    symbols, positions, comment = read_xyz(xyz_path)
    assert symbols == system.symbols
    assert comment == "water-like"
    np.testing.assert_allclose(positions, np.array(system.positions), atol=1e-7)

    npz_path = tmp_path / "traj.npz"
    save_npz_trajectory(
        npz_path,
        result,
        symbols=system.symbols,
        cell=system.cell,
        metadata={"case": "water"},
    )
    record = load_npz_trajectory(npz_path)
    assert record.symbols == system.symbols
    assert record.metadata == {"case": "water"}
    np.testing.assert_allclose(record.sampled_positions, np.array(result.sampled_positions))
    np.testing.assert_array_equal(record.diagnostic_steps, np.array(result.diagnostic_steps))
    np.testing.assert_allclose(record.diagnostic_time, np.array(result.diagnostic_time))
    restart = restart_state_from_trajectory(record, system.masses, terms)
    assert restart.step == 2
    assert np.array(restart.forces).shape == (system.atom_count, 3)
