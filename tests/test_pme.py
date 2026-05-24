import numpy as np
import pytest

from mlx_atomistic.benchmarks import ewald_reference
from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.neighbors import build_neighbor_list
from mlx_atomistic.nonbonded import EwaldReferenceConfig, ewald_reference_coulomb_energy_forces
from mlx_atomistic.pme import (
    PMEConfig,
    _assign_charges_bspline_mx,
    _assign_charges_cic_mx,
    _influence_function_mx,
    _interpolate_bspline_mx,
    _interpolate_cic_mx,
    assign_charges_bspline,
    assign_charges_cic,
    pme_coulomb_direct_space_energy_forces,
    pme_coulomb_energy_forces,
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
    with pytest.raises(ValueError, match="non-neutral background policy is not implemented"):
        pme_coulomb_energy_forces(
            _positions(),
            as_mx_array([0.7, -0.2, -0.3, -0.1]),
            Cell.cubic(12.0),
        )


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
