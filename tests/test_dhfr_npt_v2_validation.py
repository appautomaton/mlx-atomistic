import json
from pathlib import Path

import numpy as np
import pytest

from mlx_atomistic.benchmarks import dhfr_npt_v2
from mlx_atomistic.benchmarks.dhfr_npt import (
    DEFAULT_CONTRACT_PATH as V1_CONTRACT_PATH,
)
from mlx_atomistic.benchmarks.dhfr_npt import (
    DHFRNPTValidationError,
    load_validation_contract,
    payload_fingerprint,
)
from scripts import calibrate_openmm_dhfr_npt as calibration
from scripts import run_bounded_dhfr_npt_v2 as bounded_runner
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


def _formal_summary(engine, *, accepted=0, attempts=30):
    summary = {
        "finite": True,
        "minimum_volume_ratio": 0.99,
        "maximum_volume_ratio": 1.01,
        "mean_volume_ratio": 1.0,
        "maximum_cell_off_diagonal_angstrom": 0.0,
        "maximum_constraint_error_angstrom": 1.0e-5,
        "maximum_temperature_K": 310.0,
        "mean_pressure_bar": 100.0,
        "maximum_abs_pressure_bar": 200.0,
        "maximum_energy_excursion_per_atom_kj_mol": 1.0,
    }
    if engine == "mlx":
        summary.update(
            {
                "barostat_attempts": attempts,
                "barostat_accepted": accepted,
                "final_pme_plan_fingerprints": ["a" * 64],
            }
        )
    else:
        summary.update(
            {
                "configured_barostat_attempts": attempts,
                "observed_cell_changes": accepted,
                "platform": "Reference",
                "openmm_version": "8.5.1.dev-f7fa0c2",
            }
        )
    return summary


def _restart_evidence():
    return {
        "passed": True,
        "split_step": 375,
        "tolerance": 1.0e-6,
        "maximum_abs_error": 0.0,
        "final_state_max_abs_errors": {},
        "sampled_max_abs_errors": {},
        "energy_term_max_abs_errors": {},
        "index_fields_match": True,
        "barostat_and_rng_state_match": True,
    }


def _artifact_names(seed, engine):
    if engine == "openmm":
        return {"openmm_samples.npz"}
    names = {"mlx_trajectory.npz", "mlx_checkpoint.npz"}
    if seed == 7:
        names.update(
            {
                "mlx_resume_first_trajectory.npz",
                "mlx_resume_split_checkpoint.npz",
                "mlx_resume_second_trajectory.npz",
                "mlx_resume_final_checkpoint.npz",
            }
        )
    return names


def _engine_report(contract, *, seed, engine, accepted=0, attempts=30):
    summary = _formal_summary(engine, accepted=accepted, attempts=attempts)
    restart = _restart_evidence() if engine == "mlx" and seed == 7 else None
    checks = dhfr_npt_v2.build_engine_checks(
        contract=contract,
        seed=seed,
        engine=engine,
        summary=summary,
        checkpoint_resume=restart,
    )
    artifacts = {
        name: {"path": name, "byte_size": 1, "sha256": "b" * 64}
        for name in _artifact_names(seed, engine)
    }
    return dhfr_npt_v2.build_engine_report(
        contract=contract,
        source_manifest_fingerprint=contract["provenance"][
            "source_manifest_fingerprint"
        ],
        seed=seed,
        engine=engine,
        summary=summary,
        checks=checks,
        artifacts=artifacts,
        checkpoint_resume=restart,
        openmm_version=(
            contract["engines"]["openmm_version"]
            if engine == "openmm"
            else None
        ),
    )


def _memory_record(*, plateau):
    return {
        "passed": True,
        "checks": {
            "limit": True,
            "below_limit": True,
            "not_exceeded": True,
            "not_timed_out": True,
            "worker_passed": True,
            "plateau": True,
        },
        "peak_physical_bytes": 1_000_000,
        "timeout_seconds": 100.0,
        "plateau_summary": {
            "plateau_evaluated": plateau,
            "plateau_passed": plateau,
        },
    }


def _seed_report(contract, *, seed, accepted=0):
    engines = {
        engine: _engine_report(
            contract,
            seed=seed,
            engine=engine,
            accepted=accepted,
        )
        for engine in ("mlx", "openmm")
    }
    memory = {
        "mlx": _memory_record(plateau=True),
        "openmm": _memory_record(plateau=False),
    }
    delta = {"mean_volume_ratio": 0.0, "mean_pressure_bar": 0.0}
    checks = {
        "mlx_phase": True,
        "openmm_phase": True,
        "mlx_memory": True,
        "openmm_memory": True,
        "aggregate_volume_compatibility": True,
        "aggregate_pressure_compatibility": True,
        "restart_policy": True,
    }
    unsigned = {
        "schema": dhfr_npt_v2.SEED_REPORT_SCHEMA,
        "seed": seed,
        "contract_fingerprint": dhfr_npt_v2.contract_fingerprint(contract),
        "source_manifest_fingerprint": contract["provenance"][
            "source_manifest_fingerprint"
        ],
        "status": "passed",
        "blockers": [],
        "checks": checks,
        "engines": engines,
        "memory": memory,
        "engine_delta": delta,
    }
    return {**unsigned, "report_fingerprint": payload_fingerprint(unsigned)}


def _resign(payload):
    result = json.loads(json.dumps(payload))
    result.pop("report_fingerprint", None)
    return {**result, "report_fingerprint": payload_fingerprint(result)}


def test_v2_timeout_derivation_rounds_up_and_rejects_caps():
    timing = dhfr_npt_v2.derive_formal_timeouts(
        attempts=30,
        mlx_seconds=962.654772791022,
        reference_seconds_per_attempt=5.500452113525535,
    )

    assert timing["unrounded_seconds"]["7"] == pytest.approx(3721.936197133544)
    assert timing["unrounded_seconds"]["19"] == pytest.approx(2422.352253865664)
    assert timing["rounded_seconds"] == {"7": 3900, "19": 2700}
    with pytest.raises(DHFRNPTValidationError, match="hard cap"):
        dhfr_npt_v2.derive_formal_timeouts(
            attempts=30,
            mlx_seconds=30_000.0,
            reference_seconds_per_attempt=1.0,
        )
    with pytest.raises(DHFRNPTValidationError, match="finite and positive"):
        dhfr_npt_v2.derive_formal_timeouts(
            attempts=30,
            mlx_seconds=float("nan"),
            reference_seconds_per_attempt=1.0,
        )


def test_v2_contract_rejects_v1_mixing_and_gate_drift(tmp_path):
    with pytest.raises(DHFRNPTValidationError, match="schema"):
        dhfr_npt_v2.load_contract(V1_CONTRACT_PATH)

    contract = dhfr_npt_v2.load_contract()
    contract["npt_gates"]["maximum_constraint_error_angstrom"] = 1.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(DHFRNPTValidationError, match="numerical gates"):
        dhfr_npt_v2.load_contract(path)


def test_formal_engine_reports_preserve_zero_acceptance_and_fail_attempt_drift():
    contract = dhfr_npt_v2.load_contract()

    report = _engine_report(contract, seed=19, engine="mlx")

    assert report["status"] == "passed"
    assert report["summary"]["barostat_accepted"] == 0
    incomplete = _engine_report(
        contract,
        seed=19,
        engine="mlx",
        attempts=29,
    )
    assert incomplete["status"] == "failed"
    assert incomplete["blockers"] == ["attempt_schedule"]


def test_formal_engine_report_rejects_restart_and_openmm_version_drift():
    contract = dhfr_npt_v2.load_contract()
    report = _engine_report(contract, seed=7, engine="mlx")
    del report["artifacts"]["mlx_resume_split_checkpoint.npz"]
    report = _resign(report)
    with pytest.raises(DHFRNPTValidationError, match="artifacts"):
        dhfr_npt_v2.validate_engine_report(report, contract=contract)

    with pytest.raises(DHFRNPTValidationError, match="OpenMM version"):
        dhfr_npt_v2.build_engine_report(
            contract=contract,
            source_manifest_fingerprint=contract["provenance"][
                "source_manifest_fingerprint"
            ],
            seed=19,
            engine="openmm",
            summary=_formal_summary("openmm"),
            checks=dhfr_npt_v2.build_engine_checks(
                contract=contract,
                seed=19,
                engine="openmm",
                summary=_formal_summary("openmm"),
                checkpoint_resume=None,
            ),
            artifacts={
                "openmm_samples.npz": {
                    "path": "openmm_samples.npz",
                    "byte_size": 1,
                    "sha256": "b" * 64,
                }
            },
            openmm_version="wrong",
        )


def test_formal_loader_rejects_diagnostic_schema_and_artifact_tampering(
    tmp_path,
):
    contract = dhfr_npt_v2.load_contract()
    diagnostic = _diagnostic_report(tmp_path)
    with pytest.raises(DHFRNPTValidationError, match="schema"):
        dhfr_npt_v2.validate_engine_report(diagnostic, contract=contract)

    artifact = tmp_path / "mlx_trajectory.npz"
    artifact.write_bytes(b"x")
    checkpoint = tmp_path / "mlx_checkpoint.npz"
    checkpoint.write_bytes(b"x")
    report = _engine_report(contract, seed=19, engine="mlx")
    report["artifacts"] = {
        path.name: _artifact_record(path)
        for path in (artifact, checkpoint)
    }
    report = _resign(report)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    artifact.write_bytes(b"tampered")
    with pytest.raises(DHFRNPTValidationError, match="tampered"):
        dhfr_npt_v2.load_engine_report(
            report_path,
            contract=contract,
            seed=19,
            engine="mlx",
        )


def test_seed_report_rejects_missing_engine_and_seed_substitution():
    contract = dhfr_npt_v2.load_contract()
    report = _seed_report(contract, seed=19)
    del report["engines"]["openmm"]
    report = _resign(report)
    with pytest.raises(DHFRNPTValidationError, match="engine inventory"):
        dhfr_npt_v2._validate_seed_report(
            report,
            contract=contract,
            seed=19,
        )

    substituted = _seed_report(contract, seed=19)
    substituted["seed"] = 7
    substituted = _resign(substituted)
    with pytest.raises(DHFRNPTValidationError, match="invalid or incomplete"):
        dhfr_npt_v2._validate_seed_report(
            substituted,
            contract=contract,
            seed=19,
        )


def test_pooled_acceptance_is_applied_only_after_both_seeds(
    tmp_path,
    monkeypatch,
):
    contract = dhfr_npt_v2.load_contract()
    for seed in dhfr_npt_v2.FORMAL_SEEDS:
        path = tmp_path / f"seed-{seed}" / "report.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_seed_report(contract, seed=seed)))
    monkeypatch.setattr(
        dhfr_npt_v2,
        "validate_prepared_boundary",
        lambda *_: {
            "manifest_fingerprint": contract["provenance"][
                "source_manifest_fingerprint"
            ]
        },
    )

    report = dhfr_npt_v2.finalize(
        contract_path=dhfr_npt_v2.DEFAULT_CONTRACT_PATH,
        prepared_dir=tmp_path / "unused",
        input_root=tmp_path,
        output_path=tmp_path / "report.json",
    )

    assert all(seed["status"] == "passed" for seed in report["seeds"])
    assert report["status"] == "failed"
    assert report["pooled_accepted_moves"] == {"mlx": 0, "openmm": 0}


def test_freeze_preflight_rejects_source_drift_and_existing_formal_output(
    tmp_path,
    monkeypatch,
):
    contract = dhfr_npt_v2.load_contract()
    monkeypatch.setattr(
        dhfr_npt_v2,
        "validate_prepared_boundary",
        lambda *_: {
            "manifest_fingerprint": contract["provenance"][
                "source_manifest_fingerprint"
            ]
        },
    )
    monkeypatch.setattr(
        dhfr_npt_v2,
        "implementation_inventory",
        lambda *_: contract["implementation_freeze"]["inventory"],
    )
    formal = tmp_path / "formal"

    result = dhfr_npt_v2.freeze_check(
        contract_path=dhfr_npt_v2.DEFAULT_CONTRACT_PATH,
        prepared_dir=tmp_path / "unused",
        formal_root=formal,
    )

    assert result["status"] == "passed"
    monkeypatch.setattr(dhfr_npt_v2, "implementation_inventory", lambda *_: [])
    with pytest.raises(DHFRNPTValidationError, match="source drifted"):
        dhfr_npt_v2.freeze_check(
            contract_path=dhfr_npt_v2.DEFAULT_CONTRACT_PATH,
            prepared_dir=tmp_path / "unused",
            formal_root=formal,
        )

    monkeypatch.setattr(
        dhfr_npt_v2,
        "implementation_inventory",
        lambda *_: contract["implementation_freeze"]["inventory"],
    )
    second = tmp_path / "second-formal"
    (second / "seed-7").mkdir(parents=True)
    with pytest.raises(DHFRNPTValidationError, match="before source-freeze"):
        dhfr_npt_v2.freeze_check(
            contract_path=dhfr_npt_v2.DEFAULT_CONTRACT_PATH,
            prepared_dir=tmp_path / "unused",
            formal_root=second,
        )


def test_v2_supervisor_uses_contract_limits_without_importing_libproc(tmp_path):
    contract = dhfr_npt_v2.load_contract()
    contract_path = tmp_path / "contract.json"
    command = bounded_runner.build_phase_command(
        contract=contract,
        contract_path=contract_path,
        prepared=tmp_path / "prepared",
        formal_root=tmp_path / "formal",
        seed=7,
        engine="mlx",
        timeout_seconds=3900.0,
        split_resume=True,
        repo_root=Path.cwd(),
    )

    assert command[command.index("--max-bytes") + 1] == "40000000000"
    assert command[command.index("--timeout-seconds") + 1] == "3900.0"
    assert command[command.index("--contract") + 1] == str(contract_path)
    assert "--split-resume" in command
    with pytest.raises(SystemExit):
        bounded_runner._parse_args(
            [
                "--seed",
                "7",
                "--prepared",
                "prepared",
                "--out",
                "formal",
                "--timeout-seconds",
                "1",
            ]
        )
    formal = runner._parse_args(
        [
            "--stage",
            "formal",
            "--engine",
            "mlx",
            "--seed",
            "7",
            "--prepared",
            "prepared",
            "--contract",
            "contract.json",
            "--out",
            "formal",
            "--split-resume",
        ]
    )
    assert formal.calibration is None


def test_v2_supervisor_cannot_exchange_seed_limits(monkeypatch, tmp_path):
    contract = dhfr_npt_v2.load_contract()
    observed = []
    monotonic = iter([100.0, 100.0, 100.0, 100.0])
    monkeypatch.setattr(bounded_runner.dhfr_npt_v2, "load_contract", lambda _: contract)
    monkeypatch.setattr(bounded_runner.dhfr_npt_v2, "freeze_check", lambda **_: {})
    monkeypatch.setattr(
        bounded_runner.dhfr_npt_v2,
        "reconcile_seed_directory",
        lambda **_: {"status": "passed"},
    )
    monkeypatch.setattr(bounded_runner.time, "monotonic", lambda: next(monotonic))

    def completed(command, check):
        observed.append(command)
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(bounded_runner.subprocess, "run", completed)

    assert (
        bounded_runner.run_seed(
            contract_path=tmp_path / "contract.json",
            prepared=tmp_path / "prepared",
            formal_root=tmp_path / "formal",
            seed=19,
            split_resume=False,
        )
        == 0
    )
    timeouts = [
        float(command[command.index("--timeout-seconds") + 1])
        for command in observed
    ]
    assert timeouts == [2700.0, 2700.0]
