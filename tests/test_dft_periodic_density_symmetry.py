from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.dft import (
    GammaCenteredGrid,
    GTHProjectorChannel,
    PeriodicDavidsonConfig,
    PeriodicDFTSystem,
    PeriodicSCFConfig,
    PseudopotentialData,
    PseudopotentialFormat,
    build_time_reversal_ownership,
    cubic_reciprocal_symmetry_operations,
    periodic_analytic_stress,
    periodic_finite_difference_stress,
    periodic_scf_calculation_contract,
    periodic_scf_forces,
    reduce_kpoint_mesh_by_symmetry,
    run_periodic_scf,
)
from mlx_atomistic.dft._periodic_density_symmetry import (
    _build_density_symmetry_plan,
    _grid_source_permutation,
)


def test_grid_source_permutation_applies_reciprocal_axis_operation():
    shape = (3, 3, 2)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    swap_xy = ((0, 1, 0), (1, 0, 0), (0, 0, 1))
    inversion = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))

    swapped = values.reshape(-1)[_grid_source_permutation(shape, swap_xy)].reshape(shape)
    inverted = values.reshape(-1)[_grid_source_permutation(shape, inversion)].reshape(shape)
    expected_inverted = values[
        np.ix_(
            np.remainder(-np.arange(shape[0]), shape[0]),
            np.remainder(-np.arange(shape[1]), shape[1]),
            np.remainder(-np.arange(shape[2]), shape[2]),
        )
    ]

    np.testing.assert_array_equal(swapped, np.swapaxes(values, 0, 1))
    np.testing.assert_array_equal(inverted, expected_inverted)
    with pytest.raises(ValueError, match="incompatible"):
        _grid_source_permutation((3, 4, 2), swap_xy)


def test_density_symmetry_plan_groups_point_group_orbits_by_owner():
    full = GammaCenteredGrid((4, 4, 4))
    reduced = reduce_kpoint_mesh_by_symmetry(
        full,
        cubic_reciprocal_symmetry_operations(),
    )
    ownership = build_time_reversal_ownership(reduced)
    shape = (4, 4, 4)
    plan = _build_density_symmetry_plan(reduced, ownership, shape)

    assert plan is not None
    assert plan.full_point_count == len(full.points)
    assert plan.persistent_bytes == len(plan.permutations) * np.prod(shape) * 4
    assert len({np.asarray(permutation).tobytes() for permutation in plan.permutations}) == len(
        plan.permutations
    )

    owner_densities = {
        index: mx.array(
            np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 100.0 * index
        )
        for index in ownership.owned_indices
    }
    observed = sum(
        (
            plan.expand(index, owner_densities[index])
            for index in ownership.owned_indices
        ),
        mx.zeros(shape, dtype=mx.float32),
    )
    expected = np.zeros(shape, dtype=np.float32)
    symmetry = reduced.to_dict()["point_group_symmetry"]
    for explicit_index, orbit in enumerate(symmetry["orbits"]):
        owner_index = ownership.entry_for(explicit_index).owner_index
        source = np.asarray(owner_densities[owner_index]).reshape(-1)
        for member in orbit["members"]:
            permutation = _grid_source_permutation(
                shape,
                member["reciprocal_operation"],
            )
            expected += member["original_weight"] * source[permutation].reshape(shape)

    np.testing.assert_allclose(np.asarray(observed), expected, atol=2.0e-5)


def test_density_symmetry_plan_combines_time_reversal_partner_orbits():
    identity = (((1, 0, 0), (0, 1, 0), (0, 0, 1)),)
    reduced = reduce_kpoint_mesh_by_symmetry(
        GammaCenteredGrid((4, 1, 1)),
        identity,
    )
    ownership = build_time_reversal_ownership(reduced)
    plan = _build_density_symmetry_plan(reduced, ownership, (4, 2, 2))

    assert plan is not None
    assert ownership.owned_indices == (0, 1, 2)
    assert ownership.entry_for(1).aggregated_weight == pytest.approx(0.5)
    assert sum(term.weight for term in plan.terms_by_explicit_index[1]) == pytest.approx(0.5)
    assert not plan.terms_by_explicit_index[3]
    constant = mx.ones((4, 2, 2), dtype=mx.float32)
    expanded = plan.expand(1, constant)
    np.testing.assert_allclose(np.asarray(expanded), 0.5, atol=1.0e-7)


def test_tiny_cubic_scf_point_group_reduction_matches_full_mesh():
    pseudo = PseudopotentialData(
        element="He",
        format=PseudopotentialFormat.GTH,
        valence_charge=2.0,
        gth_rloc=0.3,
        gth_coefficients=(-1.0,),
        gth_channels=(GTHProjectorChannel(0, 0.3, ((0.5,),)),),
    )
    system = PeriodicDFTSystem(
        (6.0, 6.0, 6.0),
        (4, 4, 4),
        ((0.0, 0.0, 0.0),),
        pseudo,
    )
    full_mesh = GammaCenteredGrid((2, 2, 2))
    reduced_mesh = reduce_kpoint_mesh_by_symmetry(
        full_mesh,
        cubic_reciprocal_symmetry_operations(),
    )
    config = PeriodicSCFConfig(
        max_iterations=16,
        min_iterations=2,
        density_tolerance=1.0e-3,
        energy_tolerance=1.0e-4,
        orbital_tolerance=1.0e-3,
        mixing_beta=0.5,
        mixer="linear",
        davidson=PeriodicDavidsonConfig(
            max_iterations=16,
            tolerance=5.0e-4,
            max_subspace_size=8,
        ),
    )

    full = run_periodic_scf(
        system,
        cutoff_hartree=1.5,
        kpoint_mesh=full_mesh,
        n_bands=1,
        config=config,
    )
    reduced = run_periodic_scf(
        system,
        cutoff_hartree=1.5,
        kpoint_mesh=reduced_mesh,
        n_bands=1,
        config=config,
    )

    assert full.converged
    assert reduced.converged
    assert full.point_group_symmetry_reduced is False
    assert reduced.point_group_symmetry_reduced is True
    assert reduced.to_dict()["point_group_symmetry_reduced"] is True
    full_contract = periodic_scf_calculation_contract(
        system,
        cutoff_hartree=1.5,
        kpoint_mesh=full_mesh,
        n_bands=1,
        config=config,
    )
    reduced_contract = periodic_scf_calculation_contract(
        system,
        cutoff_hartree=1.5,
        kpoint_mesh=reduced_mesh,
        n_bands=1,
        config=config,
    )
    assert full_contract["point_group_symmetry"] is None
    assert reduced_contract["point_group_symmetry"]["full_point_count"] == 8
    assert reduced_contract != full_contract
    assert reduced.total_energy == pytest.approx(full.total_energy, abs=3.0e-5)
    np.testing.assert_allclose(
        np.asarray(reduced.density),
        np.asarray(full.density),
        atol=3.0e-4,
    )
    with pytest.raises(ValueError, match="forces.*point-group"):
        periodic_scf_forces(system, reduced)
    with pytest.raises(ValueError, match="analytic stress.*point-group"):
        periodic_analytic_stress(
            system,
            cutoff_hartree=1.5,
            kpoint_mesh=reduced_mesh,
        )
    with pytest.raises(ValueError, match="finite-difference stress.*point-group"):
        periodic_finite_difference_stress(
            system,
            cutoff_hartree=1.5,
            kpoint_mesh=reduced_mesh,
        )
