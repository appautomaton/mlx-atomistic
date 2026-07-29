import json
from pathlib import Path

import numpy as np
import pytest

from mlx_atomistic.benchmarks.dhfr_npt import (
    DHFRNPTValidationError,
    load_validation_contract,
    payload_fingerprint,
)
from scripts import calibrate_openmm_dhfr_npt as calibration
from scripts import run_openmm_mlx_dhfr_npt_v2 as runner


def _calibration_run(seed, axis, *, accepted_prefixes, elapsed=40.0):
    cells = np.broadcast_to(
        np.eye(3, dtype=np.float64),
        (calibration.ATTEMPTS_PER_AXIS + 1, 3, 3),
    ).copy()
    unsigned = {
        "schema": calibration.CALIBRATION_RUN_SCHEMA,
        "seed": seed,
        "axis": axis,
        "scheduled_attempts": calibration.ATTEMPTS_PER_AXIS,
        "prefixes": [
            {"attempts": prefix, "accepted_moves": accepted}
            for prefix, accepted in zip(
                calibration.CALIBRATION_PREFIXES,
                accepted_prefixes,
                strict=True,
            )
        ],
        "accepted_moves": accepted_prefixes[-1],
        "cell_history_angstrom": cells.tolist(),
        "finite": True,
        "disabled_axes_unchanged": True,
        "cell_change_tolerance_angstrom": (
            calibration.CELL_CHANGE_TOLERANCE_ANGSTROM
        ),
        "source_manifest_fingerprint": "a" * 64,
        "protocol_fingerprint": "b" * 64,
        "platform": "Reference",
        "openmm_version": "8.5.1.dev-f7fa0c2",
        "elapsed_seconds": elapsed,
    }
    return {**unsigned, "report_fingerprint": payload_fingerprint(unsigned)}


def _write_calibration_runs(tmp_path, accepted):
    for seed in calibration.CALIBRATION_SEEDS:
        for axis in calibration.CALIBRATION_AXES:
            values = accepted.get((seed, axis), (0, 0, 0, 0))
            path = calibration.calibration_run_path(
                tmp_path,
                seed=seed,
                axis=axis,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    _calibration_run(seed, axis, accepted_prefixes=values)
                )
            )


def test_calibration_selects_smallest_shared_budget_and_preserves_prefixes(
    tmp_path,
):
    accepted = {
        (101, "x"): (0, 1, 1, 1),
        (211, "z"): (0, 1, 1, 2),
    }
    _write_calibration_runs(tmp_path, accepted)

    report = calibration.select_calibration_budget(tmp_path)

    assert report["status"] == "selected"
    assert report["selected_formal_attempts"] == 60
    assert [candidate["formal_total_attempts"] for candidate in report["candidates"]] == [
        30,
        60,
        90,
        120,
    ]
    assert [candidate["qualifies"] for candidate in report["candidates"]] == [
        False,
        True,
        True,
        True,
    ]
    assert len(report["runs"]) == 6
    assert len(report["run_fingerprints"]) == 6


def test_calibration_reports_no_qualifier_without_weakening_rule(tmp_path):
    _write_calibration_runs(tmp_path, {})

    report = calibration.select_calibration_budget(tmp_path)

    assert report["status"] == "no_qualifier"
    assert report["selected_formal_attempts"] is None
    assert not any(candidate["qualifies"] for candidate in report["candidates"])


def test_calibration_rejects_tampered_or_incomplete_evidence(tmp_path):
    _write_calibration_runs(tmp_path, {(101, "x"): (1, 1, 1, 1)})
    path = calibration.calibration_run_path(tmp_path, seed=101, axis="x")
    report = json.loads(path.read_text())
    report["accepted_moves"] = 4
    path.write_text(json.dumps(report))

    with pytest.raises(DHFRNPTValidationError, match="fingerprint"):
        calibration.select_calibration_budget(tmp_path)

    path.unlink()
    with pytest.raises(DHFRNPTValidationError, match="missing or unreadable"):
        calibration.select_calibration_budget(tmp_path)


def test_calibration_rejects_target_seed_and_protocol_drift():
    with pytest.raises(DHFRNPTValidationError, match="target or diagnostic"):
        calibration._validate_requested_protocol(
            seeds=(7, 211),
            axes=calibration.CALIBRATION_AXES,
            attempts_per_axis=calibration.ATTEMPTS_PER_AXIS,
            prefixes=calibration.CALIBRATION_PREFIXES,
            platform_name="Reference",
        )


def test_selected_calibration_report_reconciles_runs_and_budget(tmp_path):
    accepted = {
        (101, "x"): (1, 1, 1, 1),
        (211, "z"): (1, 1, 1, 1),
    }
    _write_calibration_runs(tmp_path, accepted)
    report = calibration.select_calibration_budget(tmp_path)

    validated = calibration.validate_calibration_report(report)

    assert validated["selected_formal_attempts"] == 30
    tampered = json.loads(json.dumps(report))
    tampered["selected_formal_attempts"] = 120
    unsigned = dict(tampered)
    unsigned.pop("report_fingerprint")
    tampered["report_fingerprint"] = payload_fingerprint(unsigned)
    with pytest.raises(DHFRNPTValidationError, match="selected budget"):
        calibration.validate_calibration_report(tampered)


def test_diagnostic_workload_uses_selected_budget_without_target_seed_input(
    tmp_path,
):
    accepted = {
        (101, "x"): (1, 1, 1, 1),
        (211, "z"): (1, 1, 1, 1),
    }
    _write_calibration_runs(tmp_path, accepted)
    report = calibration.select_calibration_budget(tmp_path)

    workload = runner.build_draft_workload(load_validation_contract(), report)

    assert workload["steps"] == 30 * 25
    assert workload["barostat"]["expected_attempts"] == 30
    assert np.expm1(workload["barostat"]["max_log_volume_scale"]) == (
        pytest.approx(runner.OPENMM_INITIAL_VOLUME_FRACTION)
    )
    assert workload["constraint_max_iterations"] == 40
    assert workload["seeds"] == [7, 19]
    with pytest.raises(DHFRNPTValidationError, match="formal target seeds"):
        runner._run_diagnostic(
            prepared_dir=Path("unused"),
            calibration_path=Path("unused"),
            out_dir=tmp_path,
            seed=7,
        )


def _artifact_record(path):
    import hashlib

    return {
        "path": path.name,
        "byte_size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _diagnostic_report(directory):
    trajectory = directory / "mlx_trajectory.npz"
    checkpoint = directory / "mlx_checkpoint.npz"
    trajectory.write_bytes(b"trajectory")
    checkpoint.write_bytes(b"checkpoint")
    checks = {
        "finite": True,
        "attempt_schedule": True,
        "cell_evolution": True,
    }
    unsigned = {
        "schema": runner.DIAGNOSTIC_REPORT_SCHEMA,
        "scope": runner.DIAGNOSTIC_SCOPE,
        "status": "passed",
        "blockers": [],
        "checks": checks,
        "seed": runner.DIAGNOSTIC_SEED,
        "selected_formal_attempts": 30,
        "mlx_elapsed_seconds": 120.0,
        "source_manifest_fingerprint": "a" * 64,
        "calibration_report_fingerprint": "b" * 64,
        "calibration_protocol_fingerprint": "c" * 64,
        "openmm_version": "8.5.1.dev-f7fa0c2",
        "implementation_fingerprint": "d" * 64,
        "implementation_inventory": [],
        "evidence": {
            "mlx": {"barostat_attempts": 30},
            "artifacts": {
                trajectory.name: _artifact_record(trajectory),
                checkpoint.name: _artifact_record(checkpoint),
            },
        },
    }
    return {**unsigned, "report_fingerprint": payload_fingerprint(unsigned)}


def test_diagnostic_verification_requires_passing_memory_plateau(tmp_path):
    report = _diagnostic_report(tmp_path)
    (tmp_path / "report.json").write_text(json.dumps(report))
    memory = {
        "bounded_process_exceeded": False,
        "bounded_process_limit_bytes": runner.PROCESS_TREE_MAX_BYTES,
        "bounded_process_peak_physical_bytes": 12_000_000_000,
        "bounded_process_returncode": 0,
        "bounded_process_timed_out": False,
        "memory_trace_summary": {
            "plateau_evaluated": True,
            "plateau_passed": True,
        },
    }
    (tmp_path / "memory.json").write_text(json.dumps(memory))

    verified = runner.verify_diagnostic_directory(tmp_path)

    assert verified["status"] == "passed"
    assert verified["selected_formal_attempts"] == 30
    memory["memory_trace_summary"]["plateau_passed"] = False
    (tmp_path / "memory.json").write_text(json.dumps(memory))
    with pytest.raises(DHFRNPTValidationError, match="plateau_passed"):
        runner.verify_diagnostic_directory(tmp_path)

    with pytest.raises(DHFRNPTValidationError, match="40 attempts"):
        calibration._validate_requested_protocol(
            seeds=calibration.CALIBRATION_SEEDS,
            axes=calibration.CALIBRATION_AXES,
            attempts_per_axis=39,
            prefixes=calibration.CALIBRATION_PREFIXES,
            platform_name="Reference",
        )
