from math import pi

import numpy as np
import pytest

from mlx_atomistic.charmm_terms import (
    CHARMMCMAPPotential,
    CHARMMForceSwitchNonbondedPotential,
    CHARMMNBFIXPairOverridePotential,
    CHARMMUreyBradleyPotential,
)
from mlx_atomistic.forcefields import (
    CHARMMCMAPPotential as ForcefieldsCHARMMCMAPPotential,
)
from mlx_atomistic.forcefields import (
    CHARMMNBFIXPairOverridePotential as ForcefieldsCHARMMNBFIXPairOverridePotential,
)


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


def positions_for_dihedral(angle, *, offset=0.0):
    return np.array(
        [
            [offset + 0.0, 1.0, 0.0],
            [offset + 0.0, 0.0, 0.0],
            [offset + 1.0, 0.0, 0.0],
            [offset + 1.0, np.cos(angle), -np.sin(angle)],
        ],
        dtype=np.float32,
    )


def positions_for_cmap_angles(phi, psi):
    return np.vstack(
        [
            positions_for_dihedral(phi, offset=0.0),
            positions_for_dihedral(psi, offset=4.0),
        ],
    ).astype(np.float32)


def charmm_force_switch_lj_reference(r, sigma, epsilon, switch_distance, cutoff):
    c12 = 4.0 * epsilon * sigma**12
    c6 = 4.0 * epsilon * sigma**6
    inv_r = 1.0 / r
    inv_r3 = inv_r**3
    inv_r6 = inv_r3**2
    rc3 = cutoff**3
    rc6 = rc3**2
    ri3 = switch_distance**3
    ri6 = ri3**2
    if r <= switch_distance:
        energy = c12 * (inv_r6**2 - 1.0 / (ri6 * rc6)) - c6 * (
            inv_r6 - 1.0 / (ri3 * rc3)
        )
    else:
        energy = c12 * rc6 / (rc6 - ri6) * (inv_r6 - 1.0 / rc6) ** 2 - c6 * rc3 / (
            rc3 - ri3
        ) * (inv_r3 - 1.0 / rc3) ** 2
    return energy


def assert_finite_energy_forces(term, positions):
    energy, forces = term.energy_forces(positions)
    assert np.isfinite(np.array(energy)).all()
    assert np.isfinite(np.array(forces)).all()
    assert np.array(forces).shape == positions.shape


def assert_force_matches_finite_difference(term, positions, *, atol=5e-3):
    _, forces = term.energy_forces(positions)
    np.testing.assert_allclose(
        np.array(forces),
        finite_difference_force(term, positions),
        atol=atol,
    )


def test_charmm_urey_bradley_force_matches_finite_difference():
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.8, 0.4, 0.0], [1.7, 0.1, 0.0]],
        dtype=np.float32,
    )
    term = CHARMMUreyBradleyPotential([(0, 1, 2)], k=3.5, distance=1.45)

    assert term.name == "urey_bradley"
    assert_finite_energy_forces(term, positions)
    assert_force_matches_finite_difference(term, positions)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"k": np.inf, "distance": 1.45}, "k values must be finite"),
        ({"k": -1.0, "distance": 1.45}, "k values must be non-negative"),
        ({"k": 1.0, "distance": np.nan}, "distance values must be finite"),
        ({"k": 1.0, "distance": 0.0}, "distance values must be positive"),
    ],
)
def test_charmm_urey_bradley_rejects_invalid_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CHARMMUreyBradleyPotential([(0, 1, 2)], **kwargs)


def test_charmm_cmap_force_matches_finite_difference():
    axis = np.linspace(-pi, pi, 6, endpoint=False, dtype=np.float32)
    phi_grid, psi_grid = np.meshgrid(axis, axis, indexing="ij")
    grid = (
        0.2 * np.sin(phi_grid)
        + 0.15 * np.cos(2.0 * psi_grid)
        + 0.05 * np.sin(phi_grid - psi_grid)
    )
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [1.4, 1.0, 0.2],
            [1.8, 1.2, 1.1],
            [0.2, 1.6, 0.1],
            [1.1, 1.7, 0.4],
            [1.6, 2.5, 0.9],
            [2.1, 2.4, 1.7],
        ],
        dtype=np.float32,
    )
    term = CHARMMCMAPPotential(
        charmm_cmap_terms=[(0, 1, 2, 3, 4, 5, 6, 7)],
        cmap_grids=grid,
    )

    assert term.name == "charmm_cmap_terms"
    assert_finite_energy_forces(term, positions)
    assert_force_matches_finite_difference(term, positions, atol=2e-2)


def test_charmm_cmap_validation_fails_closed_for_invalid_grid():
    with pytest.raises(ValueError, match="at least a 4x4"):
        CHARMMCMAPPotential(
            charmm_cmap_terms=[(0, 1, 2, 3, 4, 5, 6, 7)],
            cmap_grids=np.ones((3, 3), dtype=np.float32),
        )


def test_charmm_cmap_reference_energy_at_grid_node():
    grid = np.arange(36, dtype=np.float32).reshape(6, 6)
    phi_index = 1
    psi_index = 4
    phi = -pi + 2.0 * pi * phi_index / 6
    psi = -pi + 2.0 * pi * psi_index / 6
    term = CHARMMCMAPPotential(
        charmm_cmap_terms=[(0, 1, 2, 3, 4, 5, 6, 7)],
        cmap_grids=grid,
    )

    energy = term.potential_energy(positions_for_cmap_angles(phi, psi))

    assert float(np.array(energy)) == pytest.approx(float(grid[phi_index, psi_index]), abs=2e-5)


def test_charmm_cmap_reference_energy_is_periodic_across_seam():
    axis = np.linspace(-pi, pi, 8, endpoint=False, dtype=np.float32)
    phi_grid, psi_grid = np.meshgrid(axis, axis, indexing="ij")
    grid = np.sin(phi_grid) + 0.25 * np.cos(psi_grid)
    term = CHARMMCMAPPotential(
        charmm_cmap_terms=[(0, 1, 2, 3, 4, 5, 6, 7)],
        cmap_grids=grid,
    )

    left = term.potential_energy(positions_for_cmap_angles(-pi + 0.02, pi / 4))
    right = term.potential_energy(positions_for_cmap_angles(pi - 0.02, pi / 4))

    assert float(np.array(left)) == pytest.approx(float(np.array(right)), abs=6e-2)


def test_charmm_cmap_uses_per_term_map_indices():
    grid0 = np.arange(36, dtype=np.float32).reshape(6, 6)
    grid1 = (100.0 + np.arange(36, dtype=np.float32)).reshape(6, 6)
    phi0_index, psi0_index = 2, 1
    phi1_index, psi1_index = 4, 3
    phi0 = -pi + 2.0 * pi * phi0_index / 6
    psi0 = -pi + 2.0 * pi * psi0_index / 6
    phi1 = -pi + 2.0 * pi * phi1_index / 6
    psi1 = -pi + 2.0 * pi * psi1_index / 6
    positions = np.vstack(
        [
            positions_for_cmap_angles(phi0, psi0),
            positions_for_cmap_angles(phi1, psi1) + np.array([0.0, 4.0, 0.0], dtype=np.float32),
        ],
    ).astype(np.float32)
    term = CHARMMCMAPPotential(
        charmm_cmap_terms=[
            (0, 1, 2, 3, 4, 5, 6, 7),
            (8, 9, 10, 11, 12, 13, 14, 15),
        ],
        cmap_grids=np.stack([grid0, grid1]),
        cmap_indices=[0, 1],
    )

    energy = term.potential_energy(positions)

    expected = grid0[phi0_index, psi0_index] + grid1[phi1_index, psi1_index]
    assert float(np.array(energy)) == pytest.approx(float(expected), abs=2e-3)


def test_charmm_force_switch_nonbonded_matches_reference_values_in_switching_interval():
    r = 1.72
    sigma = 1.1
    epsilon = 0.35
    positions = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=np.float32)
    term = CHARMMForceSwitchNonbondedPotential(
        sigma=[sigma, sigma],
        epsilon=[epsilon, epsilon],
        charges=[0.0, 0.0],
        cutoff=2.0,
        switch_distance=1.5,
    )

    energy, forces = term.energy_forces(positions)
    expected_energy = charmm_force_switch_lj_reference(
        r,
        sigma,
        epsilon,
        switch_distance=1.5,
        cutoff=2.0,
    )
    h = 1e-5
    expected_force_x = (
        charmm_force_switch_lj_reference(r + h, sigma, epsilon, 1.5, 2.0)
        - charmm_force_switch_lj_reference(r - h, sigma, epsilon, 1.5, 2.0)
    ) / (2.0 * h)

    assert float(np.array(energy)) == pytest.approx(expected_energy, rel=2e-6)
    assert np.array(forces)[0, 0] == pytest.approx(expected_force_x, rel=2e-5)
    assert np.array(forces)[1, 0] == pytest.approx(-expected_force_x, rel=2e-5)


def test_charmm_force_switch_nonbonded_has_finite_force_switch():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.35, 0.1, 0.0], [0.1, 1.65, 0.0]],
        dtype=np.float32,
    )
    term = CHARMMForceSwitchNonbondedPotential(
        sigma=[1.0, 1.0, 0.9],
        epsilon=[0.2, 0.25, 0.15],
        charges=[0.0, 0.0, 0.0],
        cutoff=2.0,
        switch_distance=1.5,
    )

    assert term.name == "charmm_force_switch_nonbonded"
    assert_finite_energy_forces(term, positions)
    assert_force_matches_finite_difference(term, positions, atol=8e-3)


def test_charmm_force_switch_nonbonded_rejects_unsupported_exception_wiring():
    with pytest.raises(ValueError, match="does not yet support topology or exception overrides"):
        CHARMMForceSwitchNonbondedPotential(
            sigma=[1.0, 1.0],
            epsilon=[0.2, 0.2],
            charges=[0.0, 0.0],
            cutoff=2.0,
            switch_distance=1.5,
            exception_pairs=[(0, 1)],
        )


def test_charmm_force_switch_nonbonded_rejects_nonfinite_coulomb_constant():
    with pytest.raises(ValueError, match="coulomb_constant must be finite"):
        CHARMMForceSwitchNonbondedPotential(
            sigma=[1.0, 1.0],
            epsilon=[0.2, 0.2],
            charges=[0.0, 0.0],
            cutoff=2.0,
            switch_distance=1.5,
            coulomb_constant=np.inf,
        )


def test_charmm_nbfix_pair_override_changes_selected_lj_pair_and_matches_finite_difference():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.35, 0.0, 0.0], [0.0, 1.7, 0.0]],
        dtype=np.float32,
    )
    base = CHARMMNBFIXPairOverridePotential(
        sigma=[1.0, 1.0, 1.0],
        epsilon=[0.2, 0.2, 0.2],
        charges=[0.1, -0.2, 0.0],
        nbfix_pairs=[],
        nbfix_sigma=[],
        nbfix_epsilon=[],
        cutoff=2.2,
        switch_distance=1.8,
    )
    overridden = CHARMMNBFIXPairOverridePotential(
        sigma=[1.0, 1.0, 1.0],
        epsilon=[0.2, 0.2, 0.2],
        charges=[0.1, -0.2, 0.0],
        nbfix_pairs=[(0, 1)],
        nbfix_sigma=[1.15],
        nbfix_epsilon=[0.5],
        cutoff=2.2,
        switch_distance=1.8,
    )

    base_energy, _ = base.energy_forces(positions)
    override_energy, _ = overridden.energy_forces(positions)
    components = overridden.component_energies(positions)

    assert overridden.name == "nbfix_pair_overrides"
    assert np.isfinite(np.array(components["nbfix_lj_correction"])).all()
    assert float(np.array(base_energy)) != pytest.approx(float(np.array(override_energy)))
    assert_finite_energy_forces(overridden, positions)
    assert_force_matches_finite_difference(overridden, positions, atol=1e-2)


@pytest.mark.parametrize("pairs", [[(0, 0)], [(0, 1), (1, 0)]])
def test_charmm_nbfix_pair_override_rejects_invalid_pairs(pairs):
    with pytest.raises(ValueError, match="self pairs|duplicate pairs"):
        CHARMMNBFIXPairOverridePotential(
            sigma=[1.0, 1.0],
            epsilon=[0.2, 0.2],
            charges=[0.0, 0.0],
            nbfix_pairs=pairs,
            nbfix_sigma=[1.1] * len(pairs),
            nbfix_epsilon=[0.3] * len(pairs),
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sigma": [1.0, np.nan]}, "sigma values must be finite and positive"),
        ({"sigma": [1.0, 0.0]}, "sigma values must be finite and positive"),
        ({"epsilon": [0.2, np.inf]}, "epsilon values must be finite and non-negative"),
        ({"epsilon": [0.2, -0.1]}, "epsilon values must be finite and non-negative"),
        ({"charges": [0.0, np.nan]}, "charges values must be finite"),
        ({"coulomb_constant": np.inf}, "coulomb_constant must be finite"),
    ],
)
def test_charmm_nbfix_pair_override_rejects_invalid_base_parameters(kwargs, match):
    params = {
        "sigma": [1.0, 1.0],
        "epsilon": [0.2, 0.2],
        "charges": [0.0, 0.0],
        "nbfix_pairs": [(0, 1)],
        "nbfix_sigma": [1.1],
        "nbfix_epsilon": [0.3],
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        CHARMMNBFIXPairOverridePotential(**params)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"cutoff": np.inf}, "cutoff must be positive"),
        ({"cutoff": -1.0}, "cutoff must be positive"),
        ({"cutoff": None, "switch_distance": 1.0}, "switch_distance requires a cutoff"),
        (
            {"cutoff": 2.0, "switch_distance": np.nan},
            "switch_distance must be non-negative and smaller than cutoff",
        ),
        (
            {"cutoff": 2.0, "switch_distance": -0.1},
            "switch_distance must be non-negative and smaller than cutoff",
        ),
        (
            {"cutoff": 2.0, "switch_distance": 2.0},
            "switch_distance must be non-negative and smaller than cutoff",
        ),
    ],
)
def test_charmm_nbfix_pair_override_rejects_invalid_cutoff_and_switch(kwargs, match):
    params = {
        "sigma": [1.0, 1.0],
        "epsilon": [0.2, 0.2],
        "charges": [0.0, 0.0],
        "nbfix_pairs": [(0, 1)],
        "nbfix_sigma": [1.1],
        "nbfix_epsilon": [0.3],
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        CHARMMNBFIXPairOverridePotential(**params)


def test_charmm_nbfix_component_energies_reject_restricted_pairs():
    term = CHARMMNBFIXPairOverridePotential(
        sigma=[1.0, 1.0, 1.0],
        epsilon=[0.2, 0.2, 0.2],
        charges=[0.0, 0.0, 0.0],
        nbfix_pairs=[(0, 1)],
        nbfix_sigma=[1.1],
        nbfix_epsilon=[0.3],
    )
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.35, 0.0, 0.0], [0.0, 1.7, 0.0]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="require full-system nonbonded evaluation"):
        term.component_energies(positions, pairs=np.array([[0, 2]], dtype=np.int32))


def test_charmm_public_names_are_reexported_from_forcefields():
    assert ForcefieldsCHARMMCMAPPotential is CHARMMCMAPPotential
    assert ForcefieldsCHARMMNBFIXPairOverridePotential is CHARMMNBFIXPairOverridePotential
