from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic._artifact_identity import ArtifactIntegrityError
from mlx_atomistic.dft import (
    KPoint,
    KPointMesh,
    PeriodicCellOptimizationConfig,
    PeriodicDFTSystem,
    PeriodicForceResult,
    PeriodicSCFConfig,
    PeriodicSCFResult,
    PeriodicStressConfig,
    PeriodicStressResult,
    PseudopotentialData,
    PseudopotentialFormat,
    optimize_periodic_cell,
)


def _system(length: float = 5.4) -> PeriodicDFTSystem:
    pseudo = PseudopotentialData(
        element="He",
        format=PseudopotentialFormat.GTH,
        valence_charge=2.0,
        gth_rloc=0.3,
        gth_coefficients=(-1.0,),
    )
    return PeriodicDFTSystem(
        (length, length, length),
        (4, 4, 4),
        ((0.2 * length, 0.3 * length, 0.4 * length),),
        pseudo,
    )


def _mesh() -> KPointMesh:
    return KPointMesh((KPoint((0.0, 0.0, 0.0)),))


def _mock_elastic_runtime(monkeypatch, *, target_length: float = 6.0):
    import mlx_atomistic.dft.periodic_cell_optimization as module

    target_volume = target_length**3
    calls: list[dict[str, object]] = []
    results: list[PeriodicSCFResult] = []

    def make_scf(system: PeriodicDFTSystem) -> PeriodicSCFResult:
        volume = float(system.grid.volume)
        logarithmic_volume = float(np.log(volume / target_volume))
        energy = 0.5 * logarithmic_volume**2
        basis = SimpleNamespace(
            active_integer_g=np.asarray(((0, 0, 0),), dtype=np.int32),
            active_count=1,
        )
        eigen = SimpleNamespace(_compact_coefficients=object())
        return PeriodicSCFResult(
            converged=True,
            status="converged",
            iterations=2,
            total_energy=energy,
            electron_count=system.electron_count,
            density_residual=0.0,
            energy_delta=0.0,
            density=mx.full(system.grid.shape, system.electron_count / volume),
            kpoints=(SimpleNamespace(basis=basis, eigen=eigen),),
            energy_by_term={"total": energy},
            history=(),
            timings={"total": 0.1},
            system_fingerprint=system.fingerprint,
        )

    def run_scf(system, **kwargs):
        result = make_scf(system)
        calls.append(
            {
                "system": system,
                "initial_density": kwargs.get("initial_density"),
                "initial_coefficients": kwargs.get("initial_coefficients"),
                "basis_integer_g": kwargs.get("basis_integer_g"),
            }
        )
        results.append(result)
        return result

    def stress(system, *, config, base_result, **_kwargs):
        volume = float(system.grid.volume)
        pressure = -float(np.log(volume / target_volume)) / volume
        return PeriodicStressResult(
            stress=np.eye(3) * pressure,
            pressure=pressure,
            base_scf=base_result,
            samples=(),
            config=config,
            elapsed_ms=0.1,
            scf_evaluations=0,
            continuation_density_uses=0,
        )

    monkeypatch.setattr(module, "run_periodic_scf", run_scf)
    monkeypatch.setattr(module, "_run_periodic_scf_fixed_topology", run_scf)
    monkeypatch.setattr(module, "periodic_finite_difference_stress", stress)
    return calls, results, make_scf


def _config(**overrides) -> PeriodicCellOptimizationConfig:
    values = {
        "max_steps": 12,
        "stress_tolerance": 5.0e-7,
        "strain_tolerance": 5.0e-5,
        "cell_compliance": 50.0,
        "max_strain": 0.08,
        "stress_config": PeriodicStressConfig(mode="isotropic"),
    }
    values.update(overrides)
    return PeriodicCellOptimizationConfig(**values)


def test_cell_only_relaxation_converges_elastic_oracle(monkeypatch):
    calls, _results, _make_scf = _mock_elastic_runtime(monkeypatch)
    initial = _system()
    initial_fractional = np.asarray(initial.positions) @ np.linalg.inv(
        np.asarray(initial.grid.cell.matrix)
    )

    result = optimize_periodic_cell(
        initial,
        cutoff_hartree=2.0,
        kpoint_mesh=_mesh(),
        n_bands=1,
        config=_config(),
        scf_config=PeriodicSCFConfig(max_iterations=4),
    )

    assert result.status == "converged"
    assert result.scf_evaluations == len(calls)
    assert calls[0]["basis_integer_g"] is None
    assert all(call["basis_integer_g"] is not None for call in calls[1:])
    assert result.stress_evaluations == 1 + len(result.steps)
    assert all(step.enthalpy <= step.armijo_limit for step in result.steps)
    np.testing.assert_allclose(
        np.diag(result.final_system.grid.cell.matrix),
        6.0,
        atol=2.0e-3,
    )
    np.testing.assert_allclose(
        np.asarray(result.final_system.positions)
        @ np.linalg.inv(np.asarray(result.final_system.grid.cell.matrix)),
        initial_fractional,
        atol=1.0e-7,
    )


def test_rejected_cell_trials_reuse_only_accepted_density(monkeypatch):
    calls, results, _make_scf = _mock_elastic_runtime(monkeypatch)

    result = optimize_periodic_cell(
        _system(5.9),
        cutoff_hartree=2.0,
        kpoint_mesh=_mesh(),
        n_bands=1,
        config=_config(max_steps=1, cell_compliance=1000.0, max_strain=0.19),
        scf_config=PeriodicSCFConfig(max_iterations=4),
    )

    assert result.steps[0].line_search_iterations > 1
    first_trials = calls[1 : 1 + result.steps[0].line_search_iterations]
    assert first_trials
    assert all(call["initial_density"] is results[0].density for call in first_trials)
    assert all(
        call["initial_coefficients"][0]
        is results[0].continuation_coefficients[0]
        for call in first_trials
    )


def test_coupled_relaxation_composes_fixed_cell_ionic_optimizer(monkeypatch):
    calls, _results, make_scf = _mock_elastic_runtime(monkeypatch)
    import mlx_atomistic.dft.periodic_cell_optimization as module

    ionic_calls: list[object] = []

    def optimize_ions(system, **kwargs):
        ionic_calls.append(kwargs.get("initial_density"))
        scf = make_scf(system)
        zeros = mx.zeros((len(system.positions), 3))
        force = PeriodicForceResult(
            forces=zeros,
            local=zeros,
            nonlocal_force=zeros,
            ion_ewald=zeros,
            timings={"total": 0.1},
            provenance={"test": "zero-force"},
        )
        return SimpleNamespace(
            status="converged",
            converged=True,
            final_system=system,
            final_scf=scf,
            final_force=force,
            scf_evaluations=1,
            steps=(),
        )

    monkeypatch.setattr(
        module,
        "_optimize_periodic_geometry_fixed_topology",
        optimize_ions,
    )
    result = optimize_periodic_cell(
        _system(),
        cutoff_hartree=2.0,
        kpoint_mesh=_mesh(),
        n_bands=1,
        config=_config(relaxation_mode="ions_and_cell"),
        scf_config=PeriodicSCFConfig(max_iterations=4),
    )

    assert result.status == "converged"
    assert result.ionic_scf_evaluations == len(ionic_calls)
    assert result.scf_evaluations == len(calls) + len(ionic_calls)
    assert ionic_calls[0] is None
    assert all(seed is not None for seed in ionic_calls[1:])


def test_cell_checkpoint_resume_matches_uninterrupted(tmp_path, monkeypatch):
    _mock_elastic_runtime(monkeypatch)
    kwargs = {
        "cutoff_hartree": 2.0,
        "kpoint_mesh": _mesh(),
        "n_bands": 1,
        "config": _config(),
        "scf_config": PeriodicSCFConfig(max_iterations=4),
    }
    uninterrupted = optimize_periodic_cell(_system(), **kwargs)
    checkpoint = tmp_path / "cell-checkpoint"
    partial = optimize_periodic_cell(
        _system(),
        **kwargs,
        checkpoint_to=checkpoint,
        checkpoint_step=2,
    )
    resumed = optimize_periodic_cell(
        _system(),
        **kwargs,
        resume_from=checkpoint,
    )

    assert partial.status == "checkpointed"
    assert partial.checkpoint_manifest is not None
    assert resumed.status == uninterrupted.status == "converged"
    assert len(resumed.steps) == len(uninterrupted.steps)
    assert resumed.lineage == (partial.checkpoint_manifest["manifest_sha256"],)
    assert resumed.final_enthalpy == pytest.approx(
        uninterrupted.final_enthalpy,
        abs=1.0e-12,
    )
    np.testing.assert_allclose(
        resumed.final_system.grid.cell.matrix,
        uninterrupted.final_system.grid.cell.matrix,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        resumed.final_system.positions,
        uninterrupted.final_system.positions,
        atol=1.0e-12,
    )

    with pytest.raises(ArtifactIntegrityError, match="settings do not match"):
        optimize_periodic_cell(
            _system(),
            **{**kwargs, "cutoff_hartree": 3.0},
            resume_from=checkpoint,
        )


def test_initial_stress_failure_is_not_mislabeled_as_scf_failure(monkeypatch):
    calls, _results, _make_scf = _mock_elastic_runtime(monkeypatch)
    import mlx_atomistic.dft.periodic_cell_optimization as module

    def fail_stress(*_args, **_kwargs):
        raise ValueError("non-smooth frozen derivative")

    monkeypatch.setattr(module, "periodic_finite_difference_stress", fail_stress)
    result = optimize_periodic_cell(
        _system(),
        cutoff_hartree=2.0,
        kpoint_mesh=_mesh(),
        n_bands=1,
        config=_config(),
        scf_config=PeriodicSCFConfig(max_iterations=4),
    )

    assert result.status == "stress_failed"
    assert result.steps == ()
    assert result.scf_evaluations == len(calls) == 1


def test_cell_optimization_config_fails_closed():
    with pytest.raises(ValueError, match="relaxation_mode"):
        PeriodicCellOptimizationConfig(relaxation_mode="both")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_strain"):
        PeriodicCellOptimizationConfig(max_strain=0.2)
    with pytest.raises(ValueError, match="external_pressure"):
        PeriodicCellOptimizationConfig(external_pressure=np.inf)
