from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic._artifact_identity import ArtifactIntegrityError
from mlx_atomistic.dft import (
    KPoint,
    KPointMesh,
    PeriodicDFTSystem,
    PeriodicForceResult,
    PeriodicGeometryOptimizationConfig,
    PeriodicSCFConfig,
    PseudopotentialData,
    PseudopotentialFormat,
    optimize_periodic_geometry,
)


def _system(position=(3.1, 3.0, 3.0)):
    pseudo = PseudopotentialData(
        element="He",
        format=PseudopotentialFormat.GTH,
        valence_charge=2.0,
        gth_rloc=0.3,
        gth_coefficients=(-1.0,),
    )
    return PeriodicDFTSystem(
        (6.0, 6.0, 6.0),
        (4, 4, 4),
        (position,),
        pseudo,
    )


def _mesh():
    return KPointMesh((KPoint((0.0, 0.0, 0.0)),))


def _mock_harmonic(monkeypatch, *, force_constant=1.0, converged=True):
    import mlx_atomistic.dft.periodic_optimization as module

    target = np.asarray(((3.0, 3.0, 3.0),), dtype=np.float64)
    calls = []
    results = []

    def run_scf(system, **kwargs):
        delta = np.asarray(system.positions, dtype=np.float64) - target
        result = SimpleNamespace(
            converged=converged,
            status="converged" if converged else "max_iterations",
            total_energy=0.5 * force_constant * float(np.sum(delta * delta)),
            electron_count=system.electron_count,
            iterations=3,
            density=mx.full(system.grid.shape, 0.01 * (len(results) + 1)),
            continuation_coefficients=(
                mx.full((1, *system.grid.shape), complex(len(results) + 1)),
            ),
        )
        calls.append(
            {
                "system": system,
                "initial_density": kwargs.get("initial_density"),
                "initial_coefficients": kwargs.get("initial_coefficients"),
            }
        )
        results.append(result)
        return result

    def forces(system, _result):
        delta = np.asarray(system.positions, dtype=np.float64) - target
        values = mx.array((-force_constant * delta).astype(np.float32))
        zeros = mx.zeros_like(values)
        return PeriodicForceResult(
            forces=values,
            local=values,
            nonlocal_force=zeros,
            ion_ewald=zeros,
            timings={"total": 0.1},
            provenance={"test": "harmonic"},
        )

    monkeypatch.setattr(module, "run_periodic_scf", run_scf)
    monkeypatch.setattr(module, "periodic_scf_forces", forces)
    return calls, results


def _config(**overrides):
    values = {
        "max_steps": 12,
        "force_tolerance": 2.0e-3,
        "rms_force_tolerance": 2.0e-3,
        "displacement_tolerance": 3.0e-3,
        "initial_step_size": 0.5,
        "max_step": 0.2,
    }
    values.update(overrides)
    return PeriodicGeometryOptimizationConfig(**values)


def test_periodic_geometry_config_and_system_position_update_fail_closed():
    source = _system()
    moved = source.with_positions(((3.0, 3.0, 3.0),))

    assert moved.fingerprint != source.fingerprint
    assert moved.grid.shape == source.grid.shape
    np.testing.assert_array_equal(moved.grid.cell.matrix, source.grid.cell.matrix)
    assert moved.pseudopotentials == source.pseudopotentials
    assert moved.electron_count == source.electron_count
    with pytest.raises(ValueError, match="fixed-cell ions"):
        PeriodicGeometryOptimizationConfig(
            relaxation_mode="cell"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="armijo_constant"):
        PeriodicGeometryOptimizationConfig(armijo_constant=1.0)
    with pytest.raises(ValueError, match="reuse_scf_state"):
        PeriodicGeometryOptimizationConfig(
            reuse_scf_state=1  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="finite"):
        source.with_positions(((np.nan, 0.0, 0.0),))


def test_periodic_relaxation_converges_without_duplicate_scf_refresh(monkeypatch):
    calls, _results = _mock_harmonic(monkeypatch)

    result = optimize_periodic_geometry(
        _system(),
        cutoff_hartree=1.0,
        kpoint_mesh=_mesh(),
        n_bands=1,
        config=_config(),
        scf_config=PeriodicSCFConfig(max_iterations=4),
    )

    assert result.status == "converged"
    assert result.convergence_reason == "force_and_displacement_tolerances"
    assert result.scf_evaluations == 1 + result.line_search_evaluations
    assert len(calls) == result.scf_evaluations
    assert result.continuation_density_uses == result.line_search_evaluations
    assert result.continuation_coefficient_uses == result.line_search_evaluations
    assert all(step.energy <= step.armijo_limit for step in result.steps)
    assert all(step.energy_delta < 0.0 for step in result.steps)
    assert result.steps[-1].step_norm <= result.config.displacement_tolerance
    assert result.steps[-1].max_force <= result.config.force_tolerance
    np.testing.assert_allclose(result.final_positions, ((3.0, 3.0, 3.0),), atol=2e-3)


def test_rejected_trials_reuse_only_the_last_accepted_electronic_state(monkeypatch):
    calls, results = _mock_harmonic(monkeypatch, force_constant=4.0)

    result = optimize_periodic_geometry(
        _system(),
        cutoff_hartree=1.0,
        kpoint_mesh=_mesh(),
        n_bands=1,
        config=_config(
            max_steps=1,
            initial_step_size=1.0,
            force_tolerance=1.0e-6,
            rms_force_tolerance=1.0e-6,
        ),
        scf_config=PeriodicSCFConfig(max_iterations=4),
    )

    assert result.line_search_evaluations == 3
    assert result.scf_evaluations == 4
    assert calls[0]["initial_density"] is None
    for call in calls[1:]:
        assert call["initial_density"] is results[0].density
        assert call["initial_coefficients"] is results[0].continuation_coefficients


def test_periodic_relaxation_rejects_an_unconverged_initial_scf(monkeypatch):
    calls, _results = _mock_harmonic(monkeypatch, converged=False)

    result = optimize_periodic_geometry(
        _system(),
        cutoff_hartree=1.0,
        kpoint_mesh=_mesh(),
        n_bands=1,
        config=_config(),
        scf_config=PeriodicSCFConfig(max_iterations=4),
    )

    assert result.status == "scf_failed"
    assert result.steps == ()
    assert result.scf_evaluations == 1
    assert len(calls) == 1


def test_periodic_relaxation_checkpoint_resume_matches_uninterrupted(
    tmp_path,
    monkeypatch,
):
    _mock_harmonic(monkeypatch)
    kwargs = {
        "cutoff_hartree": 1.0,
        "kpoint_mesh": _mesh(),
        "n_bands": 1,
        "config": _config(),
        "scf_config": PeriodicSCFConfig(max_iterations=4),
    }
    uninterrupted = optimize_periodic_geometry(_system(), **kwargs)
    checkpoint = tmp_path / "relaxation-checkpoint"
    partial = optimize_periodic_geometry(
        _system(),
        **kwargs,
        checkpoint_to=checkpoint,
        checkpoint_step=2,
    )
    resumed = optimize_periodic_geometry(
        _system(),
        **kwargs,
        resume_from=checkpoint,
    )

    assert partial.status == "checkpointed"
    assert partial.checkpoint_manifest is not None
    assert resumed.status == uninterrupted.status == "converged"
    assert len(resumed.steps) == len(uninterrupted.steps)
    assert resumed.lineage == (partial.checkpoint_manifest["manifest_sha256"],)
    assert resumed.final_energy == pytest.approx(uninterrupted.final_energy, abs=1e-12)
    np.testing.assert_allclose(
        resumed.final_positions,
        uninterrupted.final_positions,
        atol=1e-12,
    )

    with pytest.raises(ArtifactIntegrityError, match="settings do not match"):
        optimize_periodic_geometry(
            _system(),
            **{**kwargs, "cutoff_hartree": 2.0},
            resume_from=checkpoint,
        )
