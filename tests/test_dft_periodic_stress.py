from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.core import Cell
from mlx_atomistic.dft import (
    GTHProjectorChannel,
    KPoint,
    KPointMesh,
    PeriodicDFTSystem,
    PeriodicSCFConfig,
    PeriodicSCFResult,
    PeriodicStressConfig,
    PseudopotentialData,
    PseudopotentialFormat,
    periodic_analytic_stress,
    periodic_finite_difference_stress,
)


def _system() -> PeriodicDFTSystem:
    pseudo = PseudopotentialData(
        element="He",
        format=PseudopotentialFormat.GTH,
        valence_charge=2.0,
        gth_rloc=0.3,
        gth_coefficients=(-1.0,),
    )
    cell = Cell.triclinic(
        (
            (6.0, 0.0, 0.0),
            (0.8, 5.5, 0.0),
            (0.3, 0.6, 6.2),
        )
    )
    fractional = np.asarray(((0.2, 0.3, 0.4),))
    return PeriodicDFTSystem(
        cell,
        (4, 4, 4),
        fractional @ np.asarray(cell.matrix),
        pseudo,
    )


def _small_silicon_system() -> PeriodicDFTSystem:
    pseudo = PseudopotentialData(
        element="Si",
        format=PseudopotentialFormat.GTH,
        valence_charge=4.0,
        gth_rloc=0.44,
        gth_coefficients=(-6.26928833,),
        gth_channels=(
            GTHProjectorChannel(
                0,
                0.43563383,
                ((8.9517415, -2.70627082), (-2.70627082, 3.4937806)),
            ),
            GTHProjectorChannel(1, 0.49794218, ((2.43127673,),)),
        ),
    )
    return PeriodicDFTSystem(
        (6.0, 6.0, 6.0),
        (4, 4, 4),
        ((1.2, 1.8, 2.4),),
        pseudo,
    )
def _mesh() -> KPointMesh:
    return KPointMesh(
        (KPoint((0.0, 0.0, 0.0), coordinate_system="reduced"),)
    )


def _mock_elastic_scf(
    monkeypatch,
    source,
    target_stress,
    *,
    drift_topology=False,
    topology_safe_below=None,
):
    import mlx_atomistic.dft.periodic_stress as module

    reference = np.asarray(source.grid.cell.matrix, dtype=np.float64)
    reference_volume = float(np.linalg.det(reference))
    calls = []

    def run_scf(system, **kwargs):
        matrix = np.asarray(system.grid.cell.matrix, dtype=np.float64)
        deformation = np.linalg.inv(reference) @ matrix
        strain = 0.5 * (deformation + deformation.T) - np.eye(3)
        energy = -reference_volume * float(np.sum(target_stress * strain))
        integer_g = np.asarray(((0, 0, 0), (1, 0, 0)), dtype=np.int32)
        relative_strain = matrix[0, 0] / reference[0, 0] - 1.0
        if (
            drift_topology
            and relative_strain > 0.0
            and (topology_safe_below is None or relative_strain > topology_safe_below)
        ):
            integer_g[1, 0] = 2
        basis = SimpleNamespace(
            active_integer_g=integer_g,
            active_count=integer_g.shape[0],
        )
        calls.append(kwargs.get("initial_density"))
        return PeriodicSCFResult(
            converged=True,
            status="converged",
            iterations=2,
            total_energy=energy,
            electron_count=system.electron_count,
            density_residual=0.0,
            energy_delta=0.0,
            density=mx.full(system.grid.shape, system.electron_count / system.grid.volume),
            kpoints=(
                SimpleNamespace(
                    basis=basis,
                    eigen=SimpleNamespace(_compact_coefficients=object()),
                ),
            ),
            energy_by_term={"total": energy},
            history=(),
            timings={"total": 0.1},
            system_fingerprint=system.fingerprint,
        )

    monkeypatch.setattr(module, "run_periodic_scf", run_scf)
    monkeypatch.setattr(module, "_run_periodic_scf_fixed_topology", run_scf)
    return calls


def _mock_volume_only_scf(monkeypatch, pressure):
    import mlx_atomistic.dft.periodic_stress as module

    def run_scf(system, **_kwargs):
        volume = float(system.grid.volume)
        energy = -pressure * volume
        basis = SimpleNamespace(
            active_integer_g=np.asarray(((0, 0, 0),), dtype=np.int32),
            active_count=1,
        )
        return PeriodicSCFResult(
            converged=True,
            status="converged",
            iterations=2,
            total_energy=energy,
            electron_count=system.electron_count,
            density_residual=0.0,
            energy_delta=0.0,
            density=mx.full(system.grid.shape, system.electron_count / volume),
            kpoints=(
                SimpleNamespace(
                    basis=basis,
                    eigen=SimpleNamespace(_compact_coefficients=object()),
                ),
            ),
            energy_by_term={"total": energy},
            history=(),
            timings={"total": 0.1},
            system_fingerprint=system.fingerprint,
        )

    monkeypatch.setattr(module, "run_periodic_scf", run_scf)
    monkeypatch.setattr(module, "_run_periodic_scf_fixed_topology", run_scf)


def test_periodic_stress_modes_recover_compression_positive_elastic_oracle(monkeypatch):
    system = _system()
    target = np.asarray(
        (
            (0.01, 0.002, -0.003),
            (0.002, 0.02, 0.004),
            (-0.003, 0.004, 0.03),
        )
    )
    calls = _mock_elastic_scf(monkeypatch, system, target)

    isotropic = periodic_finite_difference_stress(
        system,
        cutoff_hartree=2.0,
        kpoint_mesh=_mesh(),
        n_bands=1,
        config=PeriodicStressConfig(
            mode="isotropic",
            electronic_response="reconverged",
        ),
    )
    diagonal = periodic_finite_difference_stress(
        system,
        cutoff_hartree=2.0,
        kpoint_mesh=_mesh(),
        n_bands=1,
        config=PeriodicStressConfig(
            mode="diagonal",
            electronic_response="reconverged",
        ),
    )
    symmetric = periodic_finite_difference_stress(
        system,
        cutoff_hartree=2.0,
        kpoint_mesh=_mesh(),
        n_bands=1,
        config=PeriodicStressConfig(
            mode="symmetric",
            electronic_response="reconverged",
        ),
    )

    expected_pressure = float(np.trace(target) / 3.0)
    np.testing.assert_allclose(
        isotropic.stress,
        np.eye(3) * expected_pressure,
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.diag(diagonal.stress),
        np.diag(target),
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(symmetric.stress, target, atol=1e-6, rtol=0.0)
    assert symmetric.pressure == pytest.approx(expected_pressure, abs=1e-6)
    assert isotropic.scf_evaluations == 3
    assert diagonal.scf_evaluations == 7
    assert symmetric.scf_evaluations == 13
    assert isotropic.continuation_density_uses == 2
    assert all(value is None for value in (calls[0], calls[3], calls[10]))
    assert symmetric.to_dict()["sign_convention"] == "compression-positive"


def test_periodic_stress_rejects_active_plane_wave_topology_crossing(monkeypatch):
    system = _system()
    _mock_elastic_scf(monkeypatch, system, np.eye(3), drift_topology=True)

    with pytest.raises(ValueError, match="fixed integer-G topology"):
        periodic_finite_difference_stress(
            system,
            cutoff_hartree=2.0,
            kpoint_mesh=_mesh(),
            n_bands=1,
            config=PeriodicStressConfig(
                mode="isotropic",
                electronic_response="reconverged",
            ),
        )


def test_periodic_stress_is_invariant_to_translation_and_equivalent_cell_basis(
    monkeypatch,
):
    system = _system()
    pressure = 0.015
    _mock_volume_only_scf(monkeypatch, pressure)
    matrix = np.asarray(system.grid.cell.matrix, dtype=np.float64)
    translated = system.with_positions(np.asarray(system.positions) + matrix[0])
    unimodular = np.asarray(((1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    equivalent = system.with_cell(unimodular @ matrix, scale_positions=False)
    config = PeriodicStressConfig(
        mode="symmetric",
        electronic_response="reconverged",
    )

    reports = [
        periodic_finite_difference_stress(
            candidate,
            cutoff_hartree=2.0,
            kpoint_mesh=_mesh(),
            n_bands=1,
            config=config,
        )
        for candidate in (system, translated, equivalent)
    ]

    for report in reports:
        np.testing.assert_allclose(
            report.stress,
            pressure * np.eye(3),
            atol=2.0e-6,
            rtol=0.0,
        )


def test_periodic_system_cell_update_preserves_fractional_or_cartesian_positions():
    system = _system()
    original_cell = np.asarray(system.grid.cell.matrix, dtype=np.float64)
    fractional = np.asarray(system.positions) @ np.linalg.inv(original_cell)
    replacement = original_cell @ np.diag((1.1, 0.9, 1.05))

    scaled = system.with_cell(replacement)
    fixed = system.with_cell(replacement, scale_positions=False)

    np.testing.assert_allclose(
        np.asarray(scaled.positions) @ np.linalg.inv(replacement),
        fractional,
        atol=1e-12,
    )
    np.testing.assert_array_equal(fixed.positions, system.positions)
    assert scaled.grid.shape == system.grid.shape
    assert scaled.pseudopotentials == system.pseudopotentials
    with pytest.raises(ValueError, match="scale_positions"):
        system.with_cell(replacement, scale_positions=1)  # type: ignore[arg-type]


def test_periodic_stress_config_fails_closed():
    with pytest.raises(ValueError, match="stress mode"):
        PeriodicStressConfig(mode="cell")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strain_step"):
        PeriodicStressConfig(strain_step=0.0)
    with pytest.raises(ValueError, match="reuse_scf_state"):
        PeriodicStressConfig(reuse_scf_state=1)  # type: ignore[arg-type]


@pytest.mark.slow
def test_frozen_variational_stress_uses_one_real_base_scf():
    from mlx_atomistic.dft._periodic_analytic_stress import (
        _PeriodicAnalyticStressGraph,
    )
    from mlx_atomistic.dft.gga import ProductionPBEExchangeCorrelation

    system = _small_silicon_system()
    mesh = _mesh()
    scf_config = PeriodicSCFConfig(
        max_iterations=30,
        density_tolerance=2.0e-4,
        energy_tolerance=2.0e-5,
        orbital_tolerance=2.0e-4,
    )
    report = periodic_finite_difference_stress(
        system,
        cutoff_hartree=2.0,
        kpoint_mesh=mesh,
        n_bands=2,
        config=PeriodicStressConfig(mode="isotropic"),
        scf_config=scf_config,
    )
    analytic = periodic_analytic_stress(
        system,
        cutoff_hartree=2.0,
        kpoint_mesh=mesh,
        n_bands=2,
        config=PeriodicStressConfig(mode="isotropic"),
        scf_config=scf_config,
        base_result=report.base_scf,
    )

    assert np.isfinite(report.pressure)
    assert report.scf_evaluations == 1
    assert report.base_variational_energy_error is not None
    assert report.base_variational_energy_error <= report.config.variational_energy_tolerance
    assert all(sample.scf_iterations is None for sample in report.samples)
    assert analytic.method == "analytic"
    assert analytic.scf_evaluations == 0
    assert analytic.base_variational_energy_error is not None
    assert analytic.base_variational_energy_error <= analytic.config.variational_energy_tolerance
    assert analytic.pressure == pytest.approx(report.pressure, abs=2.0e-6)
    assert set(analytic.base_energy_by_term) == {
        "kinetic",
        "local_gth",
        "nonlocal_gth",
        "hartree",
        "xc",
        "ion_ewald",
        "entropy_correction",
        "stationary_reference_correction",
        "total",
    }

    graph = _PeriodicAnalyticStressGraph(
        system,
        report.base_scf,
        ProductionPBEExchangeCorrelation(),
    )
    zero = mx.zeros((3, 3), dtype=mx.float32)
    step = 2.0e-3
    shear = mx.zeros((3, 3), dtype=mx.float32)
    shear = shear.at[0, 1].add(0.5)
    shear = shear.at[1, 0].add(0.5)
    derivatives = [
        mx.grad(
            lambda strain, component=index: graph.energy_terms(strain)[component]
        )(zero)
        for index in range(len(graph.energy_terms(zero)))
    ]
    mx.eval(*derivatives)
    for direction in (mx.eye(3, dtype=mx.float32), shear):
        plus = graph.energy_terms(step * direction)
        minus = graph.energy_terms(-step * direction)
        mx.eval(*plus, *minus)
        for derivative, plus_term, minus_term in zip(
            derivatives,
            plus,
            minus,
            strict=True,
        ):
            analytic_directional = float(mx.sum(derivative * direction))
            central_directional = (
                float(plus_term) - float(minus_term)
            ) / (2.0 * step)
            assert analytic_directional == pytest.approx(
                central_directional,
                abs=2.0e-4,
            )
