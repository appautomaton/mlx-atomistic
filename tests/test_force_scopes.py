import numpy as np
import pytest

import mlx_atomistic.forcefields as forcefields
from mlx_atomistic.core import Cell
from mlx_atomistic.forcefields import NonbondedPotential, PairRestrictedNonbondedPotential
from mlx_atomistic.pme import (
    PMEConfig,
    pme_coulomb_direct_space_energy_forces,
    pme_coulomb_reciprocal_space_energy_forces,
    pme_coulomb_total_energy_forces,
    pme_force_scope_report,
)


def test_cutoff_total_and_component_scopes_match():
    positions = np.array(
        [[0.1, 0.0, 0.0], [1.4, 0.2, 0.0], [0.3, 1.5, 0.1]],
        dtype=np.float32,
    )
    term = NonbondedPotential(
        sigma=[1.0, 1.1, 1.2],
        epsilon=[0.2, 0.3, 0.4],
        charges=[0.3, -0.4, 0.1],
        cutoff=None,
        lj_shift=False,
    )

    total_report = term.force_scope_report("total")
    component_report = term.force_scope_report("diagnostic")
    total_energy, total_forces = term.energy_forces_for_scope(positions, scope="total")
    component_energy, component_forces = term.energy_forces_for_scope(
        positions,
        scope="components",
    )
    _, _, components = term.energy_forces_with_components(positions)

    assert total_report["production_total_only"] is True
    assert total_report["component_work"] is False
    assert component_report["diagnostic_components"] is True
    assert component_report["component_work"] is True
    np.testing.assert_allclose(np.asarray(component_energy), np.asarray(total_energy), atol=1e-6)
    np.testing.assert_allclose(np.asarray(component_forces), np.asarray(total_forces), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(components["lj"] + components["coulomb"]),
        np.asarray(total_energy),
        atol=1e-6,
    )


def test_pme_total_scope_avoids_component_helper(monkeypatch):
    positions = np.array(
        [[1.0, 1.0, 1.0], [4.0, 1.2, 1.1], [2.0, 3.0, 5.0], [6.0, 7.0, 8.0]],
        dtype=np.float32,
    )
    charges = np.array([0.7, -0.2, -0.3, -0.2], dtype=np.float32)
    cell = Cell.cubic(12.0)
    config = PMEConfig(mesh_shape=(16, 16, 16), alpha=0.35, real_cutoff=5.0)
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0, 1.0],
        epsilon=[0.0, 0.0, 0.0, 0.0],
        charges=charges,
        cutoff=5.0,
        electrostatics="pme",
        pme_config=config,
    )
    calls = {"total": 0, "components": 0}
    original_total = forcefields.pme_coulomb_total_energy_forces
    original_components = forcefields.pme_coulomb_energy_forces

    def total_wrapper(*args, **kwargs):
        calls["total"] += 1
        return original_total(*args, **kwargs)

    def components_wrapper(*args, **kwargs):
        calls["components"] += 1
        return original_components(*args, **kwargs)

    monkeypatch.setattr(forcefields, "pme_coulomb_total_energy_forces", total_wrapper)
    monkeypatch.setattr(forcefields, "pme_coulomb_energy_forces", components_wrapper)

    total_energy, total_forces = term.energy_forces_for_scope(positions, cell, scope="total")

    assert calls == {"total": 1, "components": 0}
    component_energy, component_forces = term.energy_forces_for_scope(
        positions,
        cell,
        scope="components",
    )
    assert calls == {"total": 1, "components": 1}
    np.testing.assert_allclose(np.asarray(component_energy), np.asarray(total_energy), atol=1e-6)
    np.testing.assert_allclose(np.asarray(component_forces), np.asarray(total_forces), atol=1e-6)


def test_pme_direct_and_reciprocal_scopes_sum_to_total():
    positions = np.array(
        [[1.0, 1.0, 1.0], [4.0, 1.2, 1.1], [2.0, 3.0, 5.0], [6.0, 7.0, 8.0]],
        dtype=np.float32,
    )
    charges = np.array([0.7, -0.2, -0.3, -0.2], dtype=np.float32)
    cell = Cell.cubic(12.0)
    config = PMEConfig(mesh_shape=(16, 16, 16), alpha=0.35, real_cutoff=5.0)
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0, 1.0],
        epsilon=[0.0, 0.0, 0.0, 0.0],
        charges=charges,
        cutoff=5.0,
        electrostatics="pme",
        pme_config=config,
    )

    total_energy, total_forces = term.energy_forces_for_scope(positions, cell, scope="total")
    direct_energy, direct_forces = term.energy_forces_for_scope(
        positions,
        cell,
        scope="direct_space",
    )
    reciprocal_energy, reciprocal_forces = term.energy_forces_for_scope(
        positions,
        cell,
        scope="reciprocal_space",
    )
    standalone_total_energy, standalone_total_forces = pme_coulomb_total_energy_forces(
        positions,
        charges,
        cell,
        config=config,
    )
    standalone_direct_energy, standalone_direct_forces = pme_coulomb_direct_space_energy_forces(
        positions,
        charges,
        cell,
        config=config,
    )
    standalone_reciprocal_energy, standalone_reciprocal_forces = (
        pme_coulomb_reciprocal_space_energy_forces(positions, charges, cell, config=config)
    )

    assert term.force_scope_report("direct")["direct_space"] is True
    assert term.force_scope_report("reciprocal")["reciprocal_space"] is True
    assert pme_force_scope_report("reciprocal")["execution_path"] == "pme_reciprocal_space"
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
    np.testing.assert_allclose(
        np.asarray(standalone_direct_energy + standalone_reciprocal_energy),
        np.asarray(standalone_total_energy),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(standalone_direct_forces + standalone_reciprocal_forces),
        np.asarray(standalone_total_forces),
        atol=1e-6,
    )


def test_unsupported_reciprocal_scope_reports_metadata_and_fails_closed():
    positions = np.array(
        [[0.1, 0.0, 0.0], [1.4, 0.2, 0.0], [0.3, 1.5, 0.1]],
        dtype=np.float32,
    )
    term = NonbondedPotential(
        sigma=[1.0, 1.1, 1.2],
        epsilon=[0.2, 0.3, 0.4],
        charges=[0.3, -0.4, 0.1],
        cutoff=None,
        lj_shift=False,
    )

    report = term.force_scope_report("reciprocal_space")

    assert report["supported"] is False
    assert "PME" in str(report["unsupported_reason"])
    with pytest.raises(ValueError, match="unsupported"):
        term.energy_forces_for_scope(positions, scope="reciprocal_space")


@pytest.mark.parametrize("electrostatics", ["pme", "ewald_reference"])
def test_pair_restricted_full_system_scope_reports_unsupported(electrostatics):
    positions = np.array(
        [[1.0, 1.0, 1.0], [4.0, 1.2, 1.1], [2.0, 3.0, 5.0], [6.0, 7.0, 8.0]],
        dtype=np.float32,
    )
    kwargs = {}
    if electrostatics == "pme":
        kwargs["pme_config"] = PMEConfig(mesh_shape=(16, 16, 16), alpha=0.35, real_cutoff=5.0)
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0, 1.0],
        epsilon=[0.0, 0.0, 0.0, 0.0],
        charges=[0.7, -0.2, -0.3, -0.2],
        cutoff=5.0,
        electrostatics=electrostatics,
        **kwargs,
    )
    restricted = PairRestrictedNonbondedPotential(term, pairs=np.array([[0, 1]], dtype=np.int32))
    scopes = ["total", "components"]
    if electrostatics == "pme":
        scopes.extend(["direct_space", "reciprocal_space"])

    for scope in scopes:
        report = restricted.force_scope_report(scope)

        assert report["supported"] is False
        assert report["requires_full_system"] is True
        assert report["execution_path"] == "pair_restricted_unsupported"
        assert "full-system evaluation" in str(report["unsupported_reason"])
        with pytest.raises(ValueError, match="unsupported"):
            restricted.energy_forces_for_scope(positions, Cell.cubic(12.0), scope=scope)

    with pytest.raises(ValueError, match="full-system evaluation"):
        restricted.energy_forces(positions, Cell.cubic(12.0))
