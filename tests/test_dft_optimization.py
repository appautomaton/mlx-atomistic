import json

import numpy as np
import pytest

from mlx_atomistic.dft import (
    GeometryOptimizationConfig,
    SCFConfig,
    geometry_demo_system,
    load_geometry_optimization,
    optimize_geometry,
    save_geometry_optimization,
)
from mlx_atomistic.dft.optimize import main as optimize_main


def _scf_config() -> SCFConfig:
    return SCFConfig(
        max_iterations=20,
        solver="dense",
        seed=29,
        convergence_mode="either",
    )


def test_geometry_optimization_config_validation():
    with pytest.raises(ValueError, match="max_steps"):
        GeometryOptimizationConfig(max_steps=0)
    with pytest.raises(ValueError, match="line_search_shrink"):
        GeometryOptimizationConfig(line_search_shrink=1.0)
    with pytest.raises(ValueError, match="optimizer"):
        GeometryOptimizationConfig(optimizer="bad")  # type: ignore[arg-type]


def test_gaussian_dimer_one_step_lowers_or_preserves_energy():
    result = optimize_geometry(
        geometry_demo_system("gaussian-dimer", grid_shape=(4, 4, 4)),
        config=GeometryOptimizationConfig(
            max_steps=1,
            optimizer="steepest_descent",
            scf_config=_scf_config(),
        ),
    )

    assert len(result.steps) == 1
    assert result.steps[0].energy_delta <= 1e-10
    assert np.isfinite(result.final_energy)
    json.dumps(result.to_dict())


def test_gth_relaxation_reduces_force_and_preserves_electron_count():
    system = geometry_demo_system("gth-h2", grid_shape=(4, 4, 4))

    result = optimize_geometry(
        system,
        config=GeometryOptimizationConfig(max_steps=2, scf_config=_scf_config()),
    )

    assert len(result.steps) == 2
    assert result.steps[-1].max_force <= result.steps[0].max_force
    for step in result.steps:
        assert step.electron_count == pytest.approx(system.electron_count, abs=1e-5)


def test_line_search_failure_is_structured_for_too_large_step():
    result = optimize_geometry(
        geometry_demo_system("gth-h2", grid_shape=(4, 4, 4)),
        config=GeometryOptimizationConfig(
            max_steps=1,
            initial_step_size=10.0,
            max_step=10.0,
            max_line_search_iterations=1,
            scf_config=_scf_config(),
        ),
    )

    assert result.status == "line_search_failed"
    assert result.convergence_reason == "line_search_exhausted"
    assert result.steps == ()


def test_geometry_positions_are_wrapped_into_periodic_cell():
    system = geometry_demo_system("gth-h2", grid_shape=(4, 4, 4)).with_centers(
        ((11.15, 4.0, 4.0), (-3.15, 4.0, 4.0))
    )

    result = optimize_geometry(
        system,
        config=GeometryOptimizationConfig(max_steps=1, scf_config=_scf_config()),
    )

    positions = result.final_positions
    lengths = np.array(result.final_system.cell.lengths, dtype=np.float64)
    assert np.all(positions >= 0.0)
    assert np.all(positions < lengths)


def test_geometry_optimization_npz_round_trip(tmp_path):
    path = tmp_path / "relaxation.npz"
    result = optimize_geometry(
        geometry_demo_system("gth-h2", grid_shape=(4, 4, 4)),
        config=GeometryOptimizationConfig(max_steps=1, scf_config=_scf_config()),
    )

    save_geometry_optimization(path, result, metadata={"system": "gth-h2"})
    record = load_geometry_optimization(path)

    assert record.positions.shape == (1, 2, 3)
    assert record.forces.shape == (1, 2, 3)
    assert record.energies.shape == (1,)
    assert record.max_forces.shape == (1,)
    assert record.statuses == ("accepted",)
    assert record.metadata["user"]["system"] == "gth-h2"
    json.dumps(record.to_dict())


def test_geometry_optimization_cli_json_smoke(capsys):
    optimize_main(
        [
            "--system",
            "gth-h2",
            "--steps",
            "1",
            "--grid",
            "4,4,4",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["system"] == "gth-h2"
    assert payload["step_count"] == 1
    assert payload["status"] in {"converged", "max_steps"}
    assert payload["result"]["final_energy"] == payload["final_energy"]
