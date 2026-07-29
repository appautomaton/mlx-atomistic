import json

import numpy as np
import pytest

from mlx_atomistic.benchmarks.dhfr_npt import (
    DHFRNPTValidationError,
    payload_fingerprint,
)
from scripts import calibrate_openmm_dhfr_npt as calibration


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

    with pytest.raises(DHFRNPTValidationError, match="40 attempts"):
        calibration._validate_requested_protocol(
            seeds=calibration.CALIBRATION_SEEDS,
            axes=calibration.CALIBRATION_AXES,
            attempts_per_axis=39,
            prefixes=calibration.CALIBRATION_PREFIXES,
            platform_name="Reference",
        )
