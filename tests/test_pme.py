import numpy as np
import pytest

import mlx_atomistic.pme as pme_module
from mlx_atomistic.benchmarks import ewald_reference
from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.neighbors import build_neighbor_list
from mlx_atomistic.nonbonded import EwaldReferenceConfig, ewald_reference_coulomb_energy_forces
from mlx_atomistic.pme import (
    PMEConfig,
    PMEExecutionPlan,
    PMEPlanMismatchError,
    _assign_charges_bspline_mx,
    _assign_charges_cic_mx,
    _influence_function_mx,
    _interpolate_bspline_mx,
    _interpolate_cic_mx,
    assign_charges_bspline,
    assign_charges_cic,
    pme_coulomb_direct_space_energy_forces,
    pme_coulomb_energy_forces,
    pme_coulomb_reciprocal_space_energy_forces,
    pme_coulomb_total_energy_forces,
    pme_direct_space_policy_report,
    pme_force_scope_report,
    pme_platform_readiness_report,
    pme_readiness_report,
)


def _positions():
    return as_mx_array(
        [
            [1.0, 1.0, 1.0],
            [4.0, 1.2, 1.1],
            [2.0, 3.0, 5.0],
            [6.0, 7.0, 8.0],
        ]
    )


def _charges():
    return as_mx_array([0.7, -0.2, -0.3, -0.2])


@pytest.mark.parametrize("assignment_order", [2, 4, 5])
def test_pme_config_accepts_supported_assignment_orders(assignment_order):
    config = PMEConfig(assignment_order=assignment_order)

    assert config.assignment_order == assignment_order


@pytest.mark.parametrize("assignment_order", [1, 3, 6])
def test_pme_config_rejects_unsupported_assignment_orders(assignment_order):
    with pytest.raises(ValueError, match="assignment_order must be one of 2, 4, or 5"):
        PMEConfig(assignment_order=assignment_order)


def test_pme_config_rejects_unknown_background_policy():
    with pytest.raises(ValueError, match="background_policy"):
        PMEConfig(background_policy="implicit_magic")


def test_pme_execution_plan_materializes_once_and_reuses_all_force_scopes(monkeypatch):
    cell = Cell.cubic(12.0)
    config = PMEConfig(mesh_shape=(16, 16, 16), alpha=0.35, real_cutoff=5.0)
    setup_calls = 0
    original = pme_module._influence_function_mx

    def counted_influence(*args, **kwargs):
        nonlocal setup_calls
        setup_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pme_module, "_influence_function_mx", counted_influence)
    plan = PMEExecutionPlan(cell, config=config)
    matching_plan = PMEExecutionPlan.build(cell, config=config)

    total_energy, total_forces = pme_coulomb_total_energy_forces(
        _positions(),
        _charges(),
        cell,
        config=config,
        plan=plan,
    )
    component_energy, component_forces, components = pme_coulomb_energy_forces(
        _positions(),
        _charges(),
        cell,
        plan=plan,
    )
    direct_energy, direct_forces = pme_coulomb_direct_space_energy_forces(
        _positions(),
        _charges(),
        cell,
        config=config,
        plan=plan,
    )
    reciprocal_energy, reciprocal_forces = pme_coulomb_reciprocal_space_energy_forces(
        _positions(),
        _charges(),
        cell,
        config=config,
        plan=plan,
    )

    assert setup_calls == 2
    assert matching_plan.fingerprint == plan.fingerprint
    assert plan.build_count == 1
    assert plan.reuse_count == 4
    assert plan.setup_seconds >= 0.0
    assert plan.estimated_resident_bytes == 16 * 16 * 16 * 4 * 4
    assert plan.diagnostics["backend"] == "mlx_fft_cic"
    assert plan.diagnostics["dtype"] == "float32"
    assert plan.diagnostics["reuse_count"] == 4
    assert components["diagnostics"].plan_fingerprint == plan.fingerprint
    assert components["diagnostics"].plan_build_count == 1
    assert components["diagnostics"].plan_reuse_count == 2
    np.testing.assert_allclose(np.asarray(component_energy), np.asarray(total_energy), atol=1e-6)
    np.testing.assert_allclose(np.asarray(component_forces), np.asarray(total_forces), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(direct_energy + reciprocal_energy),
        np.asarray(total_energy),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(direct_forces + reciprocal_forces),
        np.asarray(total_forces),
        atol=1e-6,
    )


def test_pme_execution_plan_rejects_mismatches_and_rebuilds_explicitly():
    cell = Cell.cubic(12.0)
    config = PMEConfig(mesh_shape=(16, 16, 16), alpha=0.35, real_cutoff=5.0)
    plan = PMEExecutionPlan(cell, config=config, coulomb_constant=2.0)

    with pytest.raises(PMEPlanMismatchError, match="pme_execution_plan_mismatch") as error:
        plan.validate(Cell.cubic(13.0), config=config, coulomb_constant=2.0)
    assert "cell_matrix" in error.value.mismatches

    with pytest.raises(PMEPlanMismatchError) as error:
        plan.validate(
            cell,
            config=PMEConfig(mesh_shape=(20, 16, 16), alpha=0.35, real_cutoff=5.0),
            coulomb_constant=2.0,
        )
    assert "mesh_shape" in error.value.mismatches

    with pytest.raises(PMEPlanMismatchError) as error:
        plan.validate(cell, config=config, coulomb_constant=3.0)
    assert "coulomb_constant" in error.value.mismatches

    with pytest.raises(PMEPlanMismatchError) as error:
        plan.validate(cell, config=config, coulomb_constant=2.0, dtype="float16")
    assert "dtype" in error.value.mismatches

    with pytest.raises(PMEPlanMismatchError) as error:
        plan.validate(cell, config=config, coulomb_constant=2.0, backend="other_backend")
    assert "backend" in error.value.mismatches

    rebuilt = plan.rebuild(cell=Cell.cubic(13.0))
    assert rebuilt is not plan
    assert rebuilt.fingerprint != plan.fingerprint
    assert rebuilt.build_count == 1
    assert rebuilt.reuse_count == 0


def test_pme_execution_plan_mismatch_is_enforced_by_public_evaluator():
    cell = Cell.cubic(12.0)
    config = PMEConfig(mesh_shape=(16, 16, 16), alpha=0.35, real_cutoff=5.0)
    plan = PMEExecutionPlan(cell, config=config)

    with pytest.raises(PMEPlanMismatchError, match="fields=alpha"):
        pme_coulomb_total_energy_forces(
            _positions(),
            _charges(),
            cell,
            config=PMEConfig(mesh_shape=(16, 16, 16), alpha=0.4, real_cutoff=5.0),
            plan=plan,
        )

    assert plan.reuse_count == 0


def test_cic_charge_assignment_conserves_charge():
    charge_grid = assign_charges_cic(_positions(), _charges(), Cell.cubic(12.0), (12, 14, 16))

    np.testing.assert_allclose(np.asarray(charge_grid).sum(), 0.0, atol=1e-6)
    assert np.count_nonzero(np.asarray(charge_grid)) > len(_charges())


@pytest.mark.parametrize("assignment_order", [2, 4, 5])
def test_bspline_charge_assignment_conserves_charge_with_periodic_wrapping(assignment_order):
    positions = as_mx_array(
        [
            [-0.25, 0.1, 12.1],
            [12.0, 4.2, -0.3],
            [2.4, 11.8, 5.1],
            [6.7, 7.3, 8.9],
        ]
    )
    charges = as_mx_array([0.7, -0.2, 0.4, -0.1])

    charge_grid = assign_charges_bspline(
        positions,
        charges,
        Cell.cubic(12.0),
        (12, 14, 16),
        assignment_order=assignment_order,
    )

    np.testing.assert_allclose(
        np.asarray(charge_grid).sum(),
        np.asarray(charges).sum(),
        atol=2e-6,
    )
    assert np.count_nonzero(np.asarray(charge_grid)) > len(charges)


def test_order_two_bspline_assignment_and_interpolation_match_cic_wrappers():
    cell_lengths = as_mx_array([12.0, 12.0, 12.0])
    mesh_shape = (8, 10, 12)
    charge_grid = assign_charges_bspline(
        _positions(),
        _charges(),
        Cell.cubic(12.0),
        mesh_shape,
        assignment_order=2,
    )
    cic_grid = assign_charges_cic(_positions(), _charges(), Cell.cubic(12.0), mesh_shape)
    private_grid = _assign_charges_bspline_mx(
        _positions(),
        _charges(),
        cell_lengths,
        mesh_shape,
        assignment_order=2,
    )
    private_cic_grid = _assign_charges_cic_mx(_positions(), _charges(), cell_lengths, mesh_shape)
    scalar_grid = as_mx_array(np.arange(np.prod(mesh_shape), dtype=np.float32).reshape(mesh_shape))

    np.testing.assert_allclose(np.asarray(charge_grid), np.asarray(cic_grid), atol=0.0)
    np.testing.assert_allclose(np.asarray(private_grid), np.asarray(private_cic_grid), atol=0.0)
    np.testing.assert_allclose(
        np.asarray(
            _interpolate_bspline_mx(
                _positions(),
                scalar_grid,
                cell_lengths,
                assignment_order=2,
            )
        ),
        np.asarray(_interpolate_cic_mx(_positions(), scalar_grid, cell_lengths)),
        atol=0.0,
    )


def test_cic_charge_assignment_refuses_fractional_mesh_dimensions():
    with pytest.raises(ValueError, match="mesh_shape dimensions"):
        assign_charges_cic(_positions(), _charges(), Cell.cubic(12.0), (12, 14.5, 16))


def test_pme_matches_ewald_reference_on_neutral_periodic_fixture():
    cell = Cell.cubic(12.0)
    ewald_energy, ewald_forces, _ = ewald_reference_coulomb_energy_forces(
        _positions(),
        _charges(),
        cell,
        config=EwaldReferenceConfig(alpha=0.35, real_cutoff=5.0, reciprocal_cutoff=8),
    )

    pme_energy, pme_forces, components = pme_coulomb_energy_forces(
        _positions(),
        _charges(),
        cell,
        config=PMEConfig(mesh_shape=(32, 32, 32), alpha=0.35, real_cutoff=5.0),
    )

    np.testing.assert_allclose(np.asarray(pme_energy), np.asarray(ewald_energy), atol=2e-3)
    np.testing.assert_allclose(np.asarray(pme_forces), np.asarray(ewald_forces), atol=2e-4)
    assert set(components) == {
        "coulomb_real",
        "coulomb_reciprocal",
        "coulomb_self",
        "coulomb_background",
        "diagnostics",
    }
    diagnostics = components["diagnostics"]
    assert diagnostics.mesh_shape == (32, 32, 32)
    assert diagnostics.reciprocal_modes == 32 * 32 * 32 - 1
    assert diagnostics.direct_space_policy == "dense"
    np.testing.assert_allclose(diagnostics.charge_grid_sum, 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(pme_forces).sum(axis=0), np.zeros(3), atol=1e-6)


@pytest.mark.parametrize("assignment_order", [4, 5])
def test_pme_energy_forces_are_finite_for_higher_assignment_orders(assignment_order):
    energy, forces, components = pme_coulomb_energy_forces(
        _positions(),
        _charges(),
        Cell.cubic(12.0),
        config=PMEConfig(
            mesh_shape=(24, 24, 24),
            alpha=0.35,
            real_cutoff=5.0,
            assignment_order=assignment_order,
        ),
    )

    diagnostics = components["diagnostics"]
    assert diagnostics.assignment_order == assignment_order
    assert np.isfinite(np.asarray(energy))
    assert np.all(np.isfinite(np.asarray(forces)))
    np.testing.assert_allclose(diagnostics.charge_grid_sum, 0.0, atol=2e-6)
    np.testing.assert_allclose(np.asarray(forces).sum(axis=0), np.zeros(3), atol=2e-5)


def test_pme_influence_deconvolution_uses_configured_assignment_order():
    cell_lengths = np.asarray([12.0, 12.0, 12.0], dtype=np.float64)
    influence_order_2, _, _ = _influence_function_mx(
        cell_lengths,
        (8, 8, 8),
        alpha=0.35,
        coulomb_constant=1.0,
        deconvolve_assignment=True,
        assignment_order=2,
    )
    influence_order_4, _, _ = _influence_function_mx(
        cell_lengths,
        (8, 8, 8),
        alpha=0.35,
        coulomb_constant=1.0,
        deconvolve_assignment=True,
        assignment_order=4,
    )
    no_deconv_order_2, _, _ = _influence_function_mx(
        cell_lengths,
        (8, 8, 8),
        alpha=0.35,
        coulomb_constant=1.0,
        deconvolve_assignment=False,
        assignment_order=2,
    )
    no_deconv_order_4, _, _ = _influence_function_mx(
        cell_lengths,
        (8, 8, 8),
        alpha=0.35,
        coulomb_constant=1.0,
        deconvolve_assignment=False,
        assignment_order=4,
    )

    assert not np.allclose(np.asarray(influence_order_2), np.asarray(influence_order_4))
    np.testing.assert_allclose(
        np.asarray(no_deconv_order_2),
        np.asarray(no_deconv_order_4),
        atol=0.0,
    )


def test_pme_energy_and_forces_are_wrapping_invariant():
    cell = Cell.cubic(12.0)
    config = PMEConfig(mesh_shape=(24, 24, 24), alpha=0.35, real_cutoff=5.0)
    energy, forces, _ = pme_coulomb_energy_forces(_positions(), _charges(), cell, config=config)
    shifted_positions = cell.wrap(_positions() + as_mx_array([12.0, -24.0, 36.0]))
    shifted_energy, shifted_forces, _ = pme_coulomb_energy_forces(
        shifted_positions,
        _charges(),
        cell,
        config=config,
    )

    np.testing.assert_allclose(np.asarray(shifted_energy), np.asarray(energy), atol=1e-6)
    np.testing.assert_allclose(np.asarray(shifted_forces), np.asarray(forces), atol=1e-6)


def test_pme_refuses_non_neutral_system_without_background_policy():
    with pytest.raises(ValueError, match="uniform_neutralizing_plasma"):
        pme_coulomb_energy_forces(
            _positions(),
            as_mx_array([0.7, -0.2, -0.3, -0.1]),
            Cell.cubic(12.0),
        )


def test_pme_uniform_background_matches_openmm_plasma_formula():
    charges = as_mx_array([0.7, -0.2, -0.3, -0.1])
    cell = Cell.cubic(12.0)
    config = PMEConfig(
        mesh_shape=(24, 24, 24),
        alpha=0.35,
        real_cutoff=5.0,
        background_policy="uniform_neutralizing_plasma",
    )

    energy, forces, components = pme_coulomb_energy_forces(
        _positions(),
        charges,
        cell,
        config=config,
    )
    direct_energy, direct_forces = pme_coulomb_direct_space_energy_forces(
        _positions(),
        charges,
        cell,
        config=config,
    )
    reciprocal_energy, reciprocal_forces = pme_coulomb_reciprocal_space_energy_forces(
        _positions(),
        charges,
        cell,
        config=config,
    )

    net_charge = float(np.asarray(charges).sum())
    expected = -np.pi * net_charge**2 / (2.0 * float(cell.volume) * config.alpha**2)
    np.testing.assert_allclose(
        np.asarray(components["coulomb_background"]),
        expected,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(energy),
        np.asarray(direct_energy + reciprocal_energy),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(forces),
        np.asarray(direct_forces + reciprocal_forces),
        atol=1e-6,
    )
    diagnostics = components["diagnostics"]
    assert diagnostics.background_policy == "uniform_neutralizing_plasma"
    np.testing.assert_allclose(diagnostics.background_energy, expected, rtol=1e-5)


def test_pme_uniform_background_is_zero_for_neutral_system():
    cell = Cell.cubic(12.0)
    reject = PMEConfig(mesh_shape=(16, 16, 16), alpha=0.35, real_cutoff=5.0)
    plasma = PMEConfig(
        mesh_shape=(16, 16, 16),
        alpha=0.35,
        real_cutoff=5.0,
        background_policy="uniform_neutralizing_plasma",
    )

    reject_energy, reject_forces, _ = pme_coulomb_energy_forces(
        _positions(), _charges(), cell, config=reject
    )
    plasma_energy, plasma_forces, components = pme_coulomb_energy_forces(
        _positions(), _charges(), cell, config=plasma
    )

    np.testing.assert_allclose(np.asarray(plasma_energy), np.asarray(reject_energy), atol=1e-6)
    np.testing.assert_allclose(np.asarray(plasma_forces), np.asarray(reject_forces), atol=1e-6)
    np.testing.assert_allclose(np.asarray(components["coulomb_background"]), 0.0, atol=1e-7)


def test_pme_direct_space_compact_pairs_match_dense_policy():
    cell = Cell.cubic(12.0)
    config = PMEConfig(mesh_shape=(24, 24, 24), alpha=0.35, real_cutoff=5.0)
    neighbors = build_neighbor_list(
        _positions(),
        cell,
        cutoff=5.0,
        skin=0.0,
        backend="mlx_cell_pairs",
    )

    dense_energy, dense_forces = pme_coulomb_direct_space_energy_forces(
        _positions(),
        _charges(),
        cell,
        config=config,
    )
    pair_energy, pair_forces = pme_coulomb_direct_space_energy_forces(
        _positions(),
        _charges(),
        cell,
        config=config,
        pairs=neighbors.pairs,
    )
    dense_total, dense_total_forces = pme_coulomb_total_energy_forces(
        _positions(),
        _charges(),
        cell,
        config=config,
    )
    pair_total, pair_total_forces = pme_coulomb_total_energy_forces(
        _positions(),
        _charges(),
        cell,
        config=config,
        direct_space_pairs=neighbors.pairs,
    )
    report = pme_direct_space_policy_report(cell, config=config, pairs=neighbors.pairs)

    assert report["policy"] == "compact_pair"
    assert report["representation"] == "pairs"
    assert report["uses_shared_neighbor_policy"] is True
    assert report["compact_pair_count"] == neighbors.compact_pair_count
    np.testing.assert_allclose(np.asarray(pair_energy), np.asarray(dense_energy), atol=1e-6)
    np.testing.assert_allclose(np.asarray(pair_forces), np.asarray(dense_forces), atol=1e-6)
    np.testing.assert_allclose(np.asarray(pair_total), np.asarray(dense_total), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(pair_total_forces),
        np.asarray(dense_total_forces),
        atol=1e-6,
    )


def test_pme_direct_space_block_candidates_match_dense_policy():
    cell = Cell.cubic(12.0)
    config = PMEConfig(mesh_shape=(24, 24, 24), alpha=0.35, real_cutoff=5.0)
    neighbors = build_neighbor_list(
        _positions(),
        cell,
        cutoff=5.0,
        skin=0.0,
        backend="mlx_cell_blocks",
        block_size=2,
    )

    dense_energy, dense_forces = pme_coulomb_direct_space_energy_forces(
        _positions(),
        _charges(),
        cell,
        config=config,
    )
    block_energy, block_forces = pme_coulomb_direct_space_energy_forces(
        _positions(),
        _charges(),
        cell,
        config=config,
        pairs=neighbors.interactions,
    )
    report = pme_direct_space_policy_report(cell, config=config, pairs=neighbors.interactions)

    assert report["policy"] == "block_candidate"
    assert report["representation"] == "blocks"
    assert report["candidate_count"] == neighbors.candidate_count
    np.testing.assert_allclose(np.asarray(block_energy), np.asarray(dense_energy), atol=1e-6)
    np.testing.assert_allclose(np.asarray(block_forces), np.asarray(dense_forces), atol=1e-6)


def test_pme_direct_space_pair_policy_falls_back_outside_minimum_image_contract():
    cell = Cell.cubic(12.0)
    config = PMEConfig(mesh_shape=(24, 24, 24), alpha=0.35, real_cutoff=7.0)
    neighbors = build_neighbor_list(_positions(), cell, cutoff=5.0, skin=0.0)

    dense_energy, dense_forces = pme_coulomb_direct_space_energy_forces(
        _positions(),
        _charges(),
        cell,
        config=config,
    )
    fallback_energy, fallback_forces = pme_coulomb_direct_space_energy_forces(
        _positions(),
        _charges(),
        cell,
        config=config,
        pairs=neighbors.pairs,
    )
    report = pme_direct_space_policy_report(cell, config=config, pairs=neighbors.pairs)

    assert report["policy"] == "fallback"
    assert report["representation"] == "dense"
    assert report["uses_shared_neighbor_policy"] is False
    assert "half_min_box" in str(report["fallback_reason"])
    np.testing.assert_allclose(np.asarray(fallback_energy), np.asarray(dense_energy), atol=1e-6)
    np.testing.assert_allclose(np.asarray(fallback_forces), np.asarray(dense_forces), atol=1e-6)


@pytest.mark.parametrize(
    "positions",
    [
        [[np.nan, 1.0, 1.0], [4.0, 1.2, 1.1], [2.0, 3.0, 5.0], [6.0, 7.0, 8.0]],
        [[np.inf, 1.0, 1.0], [4.0, 1.2, 1.1], [2.0, 3.0, 5.0], [6.0, 7.0, 8.0]],
    ],
)
def test_pme_refuses_non_finite_positions(positions):
    with pytest.raises(ValueError, match="positions must be finite"):
        pme_coulomb_energy_forces(as_mx_array(positions), _charges(), Cell.cubic(12.0))


@pytest.mark.parametrize(
    "charges",
    [
        [np.nan, -0.2, -0.3, -0.2],
        [np.inf, -0.2, -0.3, -0.2],
    ],
)
def test_pme_refuses_non_finite_charges(charges):
    with pytest.raises(ValueError, match="charges must be finite"):
        pme_coulomb_energy_forces(_positions(), as_mx_array(charges), Cell.cubic(12.0))


def test_pme_refuses_invalid_mesh_settings_and_cells():
    with pytest.raises(ValueError, match="mesh_shape dimensions"):
        PMEConfig(mesh_shape=(2, 8, 8))
    with pytest.raises(ValueError, match="assignment_order must be one of 2, 4, or 5"):
        PMEConfig(assignment_order=3)
    with pytest.raises(ValueError, match="positive orthorhombic"):
        pme_coulomb_energy_forces(
            _positions(),
            _charges(),
            Cell.orthorhombic([12.0, 0.0, 12.0]),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": np.nan}, "alpha must be finite and positive"),
        ({"alpha": np.inf}, "alpha must be finite and positive"),
        ({"real_cutoff": np.nan}, "real_cutoff must be finite and positive"),
        ({"real_cutoff": np.inf}, "real_cutoff must be finite and positive"),
        ({"charge_tolerance": np.nan}, "charge_tolerance must be finite and non-negative"),
        ({"charge_tolerance": np.inf}, "charge_tolerance must be finite and non-negative"),
    ],
)
def test_pme_config_rejects_non_finite_public_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PMEConfig(**kwargs)


def test_pme_fails_closed_for_triclinic_cells():
    cell = Cell.triclinic(
        [
            [12.0, 0.0, 0.0],
            [1.0, 12.0, 0.0],
            [0.0, 0.0, 12.0],
        ]
    )

    with pytest.raises(ValueError, match="triclinic"):
        pme_coulomb_energy_forces(_positions(), _charges(), cell)
    with pytest.raises(ValueError, match="triclinic"):
        pme_direct_space_policy_report(cell)


def test_pme_readiness_report_accepts_mlx_fft_backend_for_production():
    report = pme_readiness_report(
        atom_count=4,
        charges=np.asarray(_charges()),
        cell_lengths=np.asarray([12.0, 12.0, 12.0], dtype=np.float32),
        config=PMEConfig(
            mesh_shape=(16, 16, 16),
            alpha=0.35,
            real_cutoff=5.0,
            assignment_order=4,
        ),
        nonbonded_cutoff=5.0,
        exclusion_count=3,
        one_four_count=1,
        explicit_exception_count=1,
    )

    assert report["status"] == "ready"
    assert report["backend"] == "mlx_fft_cic"
    assert report["production_executable"] is True
    assert report["assignment_order"] == 4
    assert report["runtime_envelope"]["assignment"] == "cardinal_b_spline_order_4"
    assert report["runtime_envelope"]["supported_assignment_orders"] == (2, 4, 5)
    assert report["checks"]["neutrality"] is True
    assert report["checks"]["box"] is True
    assert report["checks"]["mesh_shape"] is True
    assert report["checks"]["alpha"] is True
    assert report["checks"]["cutoff"] is True
    assert report["checks"]["atom_count"] is True
    assert report["checks"]["exclusions"] is True
    assert report["checks"]["one_four_corrections"] is True
    assert report["checks"]["explicit_exceptions"] is True
    assert report["blockers"] == ()
    assert report["virial"]["status"] == "finite_difference_cell_strain"
    assert report["force_scopes"]["total"]["production_total_only"] is True
    assert report["force_scopes"]["components"]["diagnostic_components"] is True
    assert report["force_scopes"]["direct_space"]["direct_space"] is True
    assert report["force_scopes"]["reciprocal_space"]["reciprocal_space"] is True
    assert pme_force_scope_report("total_only")["scope"] == "total"


def test_pme_readiness_accepts_charged_uniform_background_policy():
    charges = np.asarray([0.7, -0.2, -0.3, -0.1], dtype=np.float32)
    report = pme_readiness_report(
        atom_count=4,
        charges=charges,
        cell_lengths=np.asarray([12.0, 12.0, 12.0], dtype=np.float32),
        config=PMEConfig(
            mesh_shape=(16, 16, 16),
            alpha=0.35,
            real_cutoff=5.0,
            background_policy="uniform_neutralizing_plasma",
        ),
        nonbonded_cutoff=5.0,
        exclusion_count=0,
        one_four_count=0,
        explicit_exception_count=0,
    )

    assert report["status"] == "ready"
    assert report["checks"]["neutrality"] is False
    assert report["checks"]["charge_policy"] is True
    assert report["background_policy"] == "uniform_neutralizing_plasma"
    assert report["blockers"] == ()


def test_pme_readiness_admits_94k_atoms_and_maximum_validated_mesh():
    atom_count = 94_232
    charges = np.zeros((atom_count,), dtype=np.float32)
    charges[0] = -44.0
    report = pme_readiness_report(
        atom_count=atom_count,
        charges=charges,
        cell_lengths=np.asarray([120.0, 120.0, 90.0], dtype=np.float32),
        cell_matrix=np.diag([120.0, 120.0, 90.0]).astype(np.float32),
        config=PMEConfig(
            mesh_shape=(128, 128, 64),
            alpha=0.35,
            real_cutoff=10.0,
            assignment_order=5,
            background_policy="uniform_neutralizing_plasma",
        ),
        nonbonded_cutoff=10.0,
        exclusion_count=0,
        one_four_count=0,
        explicit_exception_count=0,
    )

    assert report["status"] == "ready"
    assert report["mesh_points"] == 1_048_576
    assert report["runtime_envelope"]["max_atoms"] == 100_000
    assert report["runtime_envelope"]["max_mesh_points"] == 1_048_576
    assert report["checks"]["orthorhombic_cell"] is True
    assert report["checks"]["minimum_image_cutoff"] is True


@pytest.mark.parametrize(
    ("atom_count", "mesh_shape", "cell_matrix", "cutoff", "expected"),
    [
        (100_001, (16, 16, 16), None, 5.0, "atom_count:outside_pme_runtime_envelope"),
        (
            4,
            (256, 128, 64),
            None,
            5.0,
            "mesh_points:outside_pme_runtime_envelope",
        ),
        (
            4,
            (16, 16, 16),
            np.asarray([[12.0, 0.0, 0.0], [1.0, 12.0, 0.0], [0.0, 0.0, 12.0]]),
            5.0,
            "cell_shape:orthorhombic_required",
        ),
        (4, (16, 16, 16), None, 7.0, "pme_cutoff:exceeds_half_minimum_box_length"),
    ],
)
def test_pme_readiness_rejects_independent_plan_envelope_limits(
    atom_count,
    mesh_shape,
    cell_matrix,
    cutoff,
    expected,
):
    report = pme_readiness_report(
        atom_count=atom_count,
        charges=np.zeros((atom_count,), dtype=np.float32),
        cell_lengths=np.asarray([12.0, 12.0, 12.0], dtype=np.float32),
        cell_matrix=cell_matrix,
        config=PMEConfig(mesh_shape=mesh_shape, alpha=0.35, real_cutoff=cutoff),
        nonbonded_cutoff=cutoff,
        exclusion_count=0,
        one_four_count=0,
        explicit_exception_count=0,
    )

    assert report["status"] == "blocked"
    assert any(str(blocker).startswith(expected) for blocker in report["blockers"])


def test_pme_readiness_rejects_nonbonded_cutoff_mismatch():
    report = pme_readiness_report(
        atom_count=4,
        charges=np.zeros((4,), dtype=np.float32),
        cell_lengths=np.asarray([12.0, 12.0, 12.0], dtype=np.float32),
        config=PMEConfig(mesh_shape=(16, 16, 16), alpha=0.35, real_cutoff=5.0),
        nonbonded_cutoff=4.0,
        exclusion_count=0,
        one_four_count=0,
        explicit_exception_count=0,
    )

    assert report["checks"]["cutoff_match"] is False
    assert "pme_cutoff:mismatch_with_nonbonded_cutoff" in report["blockers"]


def test_pme_platform_readiness_report_uses_shared_schema():
    report = pme_platform_readiness_report(
        atom_count=4,
        charges=np.asarray(_charges()),
        cell_lengths=np.asarray([12.0, 12.0, 12.0], dtype=np.float32),
        config=PMEConfig(
            mesh_shape=(16, 16, 16),
            alpha=0.35,
            real_cutoff=5.0,
            assignment_order=5,
        ),
        nonbonded_cutoff=5.0,
        exclusion_count=3,
        one_four_count=1,
        explicit_exception_count=1,
    )
    payload = report.to_dict()

    assert payload["name"] == "pme"
    assert payload["status"] == "ready"
    assert payload["blockers"] == []
    assert payload["metadata"]["backend"] == "mlx_fft_cic"
    assert payload["metadata"]["assignment_order"] == 5


def test_ewald_benchmark_payload_includes_pme_comparison():
    payload = ewald_reference.build_payload(
        atom_counts=(4,),
        evaluations=1,
        reciprocal_cutoff=2,
        pme_mesh_shape=(16, 16, 16),
    )

    row = payload["cases"][0]
    assert row["pme_mesh_shape"] == "16x16x16"
    assert row["pme_finite"]
    assert row["pme_energy_abs_error"] < 5e-3
    assert row["pme_force_max_abs_error"] < 5e-4
