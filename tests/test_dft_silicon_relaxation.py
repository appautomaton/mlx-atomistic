from __future__ import annotations

import numpy as np

from mlx_atomistic.benchmarks.dft_silicon import ANGSTROM_TO_BOHR
from mlx_atomistic.benchmarks.dft_silicon_relaxation import (
    _geometry_comparison,
    _optimization_config,
    compare_continuation_reports,
    compare_restart_reports,
)


def _report(*, status="converged", steps=2, iterations=(8, 6), offset=0.0):
    positions = np.asarray(((1.0 + offset, 2.0, 3.0),)).tolist()
    return {
        "result": {
            "status": status,
            "accepted_step_count": steps,
            "final_energy_hartree": -10.0 + offset,
            "final_positions_bohr": positions,
            "steps": [{"scf_iterations": value} for value in iterations],
        },
        "runtime": {"complete_wall_seconds": 4.0},
    }


def test_geometry_comparison_removes_only_uniform_periodic_translation():
    ideal = np.asarray(((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)))
    translated = ideal + np.asarray((0.2, -0.1, 0.3))

    comparison = _geometry_comparison(translated, ideal)

    assert comparison["passed"] is True
    assert comparison["maximum_translation_aligned_error_angstrom"] < 1.0e-12

    translated[1, 0] += 0.03 * ANGSTROM_TO_BOHR
    comparison = _geometry_comparison(translated, ideal)
    assert np.isclose(
        comparison["maximum_translation_aligned_error_angstrom"],
        0.015,
    )
    assert comparison["passed"] is False


def test_restart_and_continuation_comparisons_lock_scientific_and_work_gates():
    continued = _report(iterations=(8, 6))
    cold = _report(iterations=(12, 10))

    restart = compare_restart_reports(continued, _report(iterations=(8, 6)))
    efficiency = compare_continuation_reports(continued, cold)

    assert restart["passed"] is True
    assert efficiency["passed"] is True
    assert efficiency["iteration_reduction"] == 8
    assert efficiency["wall_time_is_diagnostic"] is True

    drifted = _report(iterations=(8, 6), offset=1.0e-3)
    assert compare_restart_reports(continued, drifted)["passed"] is False


def test_silicon_relaxation_protocol_keeps_locked_thresholds():
    config = _optimization_config(reuse_scf_state=True)

    assert config.max_steps == 12
    assert config.force_tolerance == 5.0e-4
    assert config.rms_force_tolerance == 3.0e-4
    assert config.displacement_tolerance == 3.0e-3
    assert config.reuse_scf_state is True
