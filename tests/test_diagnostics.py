import mlx.core as mx
import numpy as np

import mlx_atomistic.forcefields as forcefields
from mlx_atomistic.core import Cell
from mlx_atomistic.diagnostics import summarize_md_result
from mlx_atomistic.forcefields import NonbondedPotential
from mlx_atomistic.md import (
    LangevinThermostat,
    LennardJonesPotential,
    SimulationConfig,
    simulate_nve,
    simulate_nvt,
)
from mlx_atomistic.nonbonded import EwaldReferenceConfig
from mlx_atomistic.pme import PMEConfig


class CountingComponentTerm:
    name = "counting"
    supports_virial = True

    def __init__(self):
        self.total_calls = 0
        self.component_calls = 0

    def energy_forces(self, positions, cell=None, pairs=None):
        del cell, pairs
        self.total_calls += 1
        return mx.sum(positions * 0.0), mx.zeros_like(positions)

    def energy_forces_with_components(self, positions, cell=None, pairs=None):
        del cell, pairs
        self.component_calls += 1
        energy = mx.sum(positions * 0.0)
        return energy, mx.zeros_like(positions), {"constant": energy}


def test_summarize_nve_result_reports_scalar_diagnostics():
    positions = np.array([[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    result = simulate_nve(
        positions,
        velocities,
        cell=Cell.cubic(6.0),
        force_terms=LennardJonesPotential(cutoff=2.5),
        config=SimulationConfig(dt=0.001, steps=2, sample_interval=2),
    )

    summary = summarize_md_result(result)

    assert summary["ensemble"] == "nve"
    assert summary["steps"] == 2
    assert isinstance(summary["max_energy_drift"], float)
    assert isinstance(summary["final_pair_count"], int)
    assert isinstance(summary["final_rebuild_count"], int)


def test_summarize_nvt_result_reports_temperature_diagnostics():
    positions = np.array([[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    result = simulate_nvt(
        positions,
        velocities,
        cell=Cell.cubic(6.0),
        force_terms=LennardJonesPotential(cutoff=2.5),
        config=SimulationConfig(dt=0.001, steps=2, sample_interval=2),
        thermostat=LangevinThermostat(temperature=1.5, friction=1.0, seed=2),
    )

    summary = summarize_md_result(result)

    assert summary["ensemble"] == "nvt"
    assert summary["target_temperature"] == 1.5
    assert isinstance(summary["final_temperature_error"], float)
    assert isinstance(summary["mean_temperature_error"], float)


def test_non_diagnostic_nve_steps_use_total_force_path_but_keep_components():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    term = CountingComponentTerm()

    result = simulate_nve(
        positions,
        velocities,
        force_terms=term,
        config=SimulationConfig(
            dt=0.001,
            steps=3,
            sample_interval=3,
            diagnostic_interval=2,
            pressure_diagnostics=False,
        ),
    )

    assert np.asarray(result.diagnostic_steps).tolist() == [0, 2, 3]
    assert set(result.potential_energy_by_term) == {"counting.constant"}
    assert term.component_calls == 3
    assert term.total_calls == 1


def test_non_diagnostic_nvt_steps_use_total_force_path_but_keep_components():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    term = CountingComponentTerm()

    result = simulate_nvt(
        positions,
        velocities,
        force_terms=term,
        config=SimulationConfig(
            dt=0.001,
            steps=3,
            sample_interval=3,
            diagnostic_interval=2,
            pressure_diagnostics=False,
        ),
        thermostat=LangevinThermostat(temperature=0.0, friction=0.0, seed=7),
    )

    assert np.asarray(result.diagnostic_steps).tolist() == [0, 2, 3]
    assert set(result.potential_energy_by_term) == {"counting.constant"}
    assert term.component_calls == 3
    assert term.total_calls == 1


def test_ewald_total_force_path_skips_component_builder(monkeypatch):
    positions = np.array(
        [[1.0, 1.0, 1.0], [3.0, 2.0, 1.5], [2.0, 4.0, 3.0]],
        dtype=np.float32,
    )
    cell = Cell.cubic(8.0)
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0],
        epsilon=[0.0, 0.0, 0.0],
        charges=[1.0, -0.5, -0.5],
        electrostatics="ewald_reference",
        ewald_config=EwaldReferenceConfig(alpha=0.35, real_cutoff=4.0, reciprocal_cutoff=3),
    )
    expected_energy, expected_forces, expected_components = term.energy_forces_with_components(
        positions,
        cell,
    )

    def component_builder_was_called(*args, **kwargs):
        raise AssertionError("component Ewald path should not run")

    monkeypatch.setattr(
        forcefields,
        "ewald_reference_coulomb_energy_forces",
        component_builder_was_called,
    )

    energy, forces = term.energy_forces(positions, cell)

    np.testing.assert_allclose(np.array(energy), np.array(expected_energy), atol=1e-6)
    np.testing.assert_allclose(np.array(forces), np.array(expected_forces), atol=1e-6)
    assert set(expected_components) >= {"coulomb_real", "coulomb_reciprocal", "coulomb_self"}


def test_pme_total_force_path_skips_component_builder(monkeypatch):
    positions = np.array(
        [[1.0, 1.0, 1.0], [4.0, 1.2, 1.1], [2.0, 3.0, 5.0]],
        dtype=np.float32,
    )
    cell = Cell.cubic(12.0)
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0],
        epsilon=[0.0, 0.0, 0.0],
        charges=[1.0, -0.5, -0.5],
        cutoff=5.0,
        electrostatics="pme",
        pme_config=PMEConfig(mesh_shape=(12, 12, 12), alpha=0.35, real_cutoff=5.0),
    )
    expected_energy, expected_forces, expected_components = term.energy_forces_with_components(
        positions,
        cell,
    )

    def component_builder_was_called(*args, **kwargs):
        raise AssertionError("component PME path should not run")

    monkeypatch.setattr(forcefields, "pme_coulomb_energy_forces", component_builder_was_called)

    energy, forces = term.energy_forces(positions, cell)

    np.testing.assert_allclose(np.array(energy), np.array(expected_energy), atol=1e-6)
    np.testing.assert_allclose(np.array(forces), np.array(expected_forces), atol=1e-6)
    assert expected_components["pme_diagnostics"].mesh_shape == (12, 12, 12)
