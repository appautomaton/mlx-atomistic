import numpy as np
import pytest

from mlx_atomistic.benchmarks import ewald_reference
from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.nonbonded import EwaldReferenceConfig, ewald_reference_coulomb_energy_forces
from mlx_atomistic.pme import (
    PMEConfig,
    assign_charges_cic,
    pme_coulomb_energy_forces,
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


def test_cic_charge_assignment_conserves_charge():
    charge_grid = assign_charges_cic(_positions(), _charges(), Cell.cubic(12.0), (12, 14, 16))

    np.testing.assert_allclose(np.asarray(charge_grid).sum(), 0.0, atol=1e-6)
    assert np.count_nonzero(np.asarray(charge_grid)) > len(_charges())


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
    np.testing.assert_allclose(diagnostics.charge_grid_sum, 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(pme_forces).sum(axis=0), np.zeros(3), atol=1e-6)


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
    with pytest.raises(ValueError, match="assignment_order=2"):
        PMEConfig(assignment_order=4)
    with pytest.raises(ValueError, match="positive orthorhombic"):
        pme_coulomb_energy_forces(
            _positions(),
            _charges(),
            Cell.orthorhombic([12.0, 0.0, 12.0]),
        )


def test_pme_readiness_report_accepts_mlx_fft_backend_for_production():
    report = pme_readiness_report(
        atom_count=4,
        charges=np.asarray(_charges()),
        cell_lengths=np.asarray([12.0, 12.0, 12.0], dtype=np.float32),
        config=PMEConfig(mesh_shape=(16, 16, 16), alpha=0.35, real_cutoff=5.0),
        nonbonded_cutoff=5.0,
        exclusion_count=3,
        one_four_count=1,
        explicit_exception_count=1,
    )

    assert report["status"] == "ready"
    assert report["backend"] == "mlx_fft_cic"
    assert report["production_executable"] is True
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
