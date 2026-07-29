import copy
import json
from pathlib import Path

import pytest

from mlx_atomistic.benchmarks import dhfr_npt_runtime as runtime
from mlx_atomistic.benchmarks.dhfr_npt_v2 import load_contract
from scripts import run_dhfr_npt_runtime as runner


def _workload(*, steps=runtime.BASELINE_STEPS):
    return runtime.build_runtime_workload(
        load_contract(),
        source_identity={"manifest_fingerprint": "a" * 64},
        steps=steps,
        seed=runtime.RUNTIME_SEED,
    )


def _worker(tmp_path, *, engine="openmm", steps=runtime.BASELINE_STEPS):
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifact = tmp_path / "samples.npz"
    artifact.write_bytes(b"sample")
    engine_identity = (
        {
            "name": "openmm",
            "platform": "OpenCL",
            "precision": "single",
            "version": runtime.EXPECTED_OPENMM_VERSION,
            "device": {"DeviceIndex": "0"},
            "role": "reference-only",
        }
        if engine == "openmm"
        else {
            "name": "mlx",
            "backend": "Metal",
            "device": "Device(gpu, 0)",
            "version": "0.test",
            "role": "product-runtime",
        }
    )
    return {
        "schema": runtime.WORKER_SCHEMA,
        "engine": engine_identity,
        "host": {
            "machine": "arm64",
            "chip": "Apple M5 Max",
            "power": {"source": "AC Power", "settings": "lowpowermode 0"},
        },
        "workload": _workload(steps=steps),
        "process": {"pid": 123, "run_id": "run-1"},
        "completion_barrier": {
            "performed": True,
            "boundary": "before_timer_stop",
            "kind": "test barrier",
        },
        "timing": {
            "worker_wall_seconds": 10.0,
            "setup_wall_seconds": 1.0,
            "steady_state_integration_wall_seconds": 6.0,
            "synchronization_diagnostics_wall_seconds": 2.0,
            "persistence_wall_seconds": 0.5,
            "unaccounted_wall_seconds": 0.5,
        },
        "numerical": {"finite": True},
        "checks": {"finite": True, "attempt_schedule": True},
        "artifacts": [
            runtime.artifact_record(artifact, relative_to=tmp_path),
        ],
    }


def _memory_trace():
    return {
        "bounded_process_exceeded": False,
        "bounded_process_limit_bytes": runtime.PROCESS_TREE_MAX_BYTES,
        "bounded_process_orphans_terminated": 0,
        "bounded_process_peak_physical_bytes": 4_000_000_000,
        "bounded_process_returncode": 0,
        "bounded_process_timed_out": False,
        "bounded_process_timeout_seconds": runtime.SAMPLE_TIMEOUT_SECONDS,
        "memory_trace_summary": {
            "peak_physical_bytes": 4_000_000_000,
            "plateau_evaluated": True,
            "plateau_passed": True,
        },
        "samples": [],
    }


def _instrumented_profile():
    return {
        "profile_steps": runtime.PROFILE_STEPS,
        "projection_steps": runtime.BASELINE_STEPS,
        "root_wall_seconds": 8.0,
        "routes": {
            "barostat.attempt": {
                "calls": 3,
                "root_calls": 3,
                "first_inclusive_wall_seconds": 4.0,
                "first_exclusive_wall_seconds": 3.0,
                "recurring_inclusive_mean_wall_seconds": 2.0,
                "recurring_exclusive_mean_wall_seconds": 1.0,
                "total_inclusive_wall_seconds": 8.0,
                "total_exclusive_wall_seconds": 5.0,
                "projected_750_calls": 30,
                "projected_750_exclusive_wall_seconds": 32.0,
            },
            "constraints.position_projection": {
                "calls": 75,
                "root_calls": 75,
                "first_inclusive_wall_seconds": 0.05,
                "first_exclusive_wall_seconds": 0.05,
                "recurring_inclusive_mean_wall_seconds": 0.04,
                "recurring_exclusive_mean_wall_seconds": 0.04,
                "total_inclusive_wall_seconds": 3.01,
                "total_exclusive_wall_seconds": 3.01,
                "projected_750_calls": 750,
                "projected_750_exclusive_wall_seconds": 30.01,
            },
        },
    }


def _sample(
    tmp_path,
    *,
    engine="openmm",
    wall=12.0,
    run_id="run-1",
    steps=runtime.BASELINE_STEPS,
    mode=None,
    profile=None,
):
    worker = _worker(tmp_path, engine=engine, steps=steps)
    worker["process"]["run_id"] = run_id
    if mode is not None:
        worker["mode"] = mode
    if profile is not None:
        worker["profile"] = profile
    memory = tmp_path / "memory.json"
    memory.write_text(json.dumps(_memory_trace()))
    return runtime.finalize_sample_report(
        worker,
        complete_wall_seconds=wall,
        memory_trace=_memory_trace(),
        memory_record=runtime.artifact_record(memory, relative_to=tmp_path),
    )


def test_runtime_workload_preserves_exact_protocol_and_bounds_prefix():
    baseline = _workload()
    profile = _workload(steps=runtime.PROFILE_STEPS)

    assert baseline["target"]["atom_count"] == 23_558
    assert baseline["protocol"]["pme"]["mesh_shape"] == [56, 56, 56]
    assert baseline["protocol"]["constraint_max_iterations"] == 40
    assert baseline["steps"] == 750
    assert baseline["expected_attempts"] == 30
    assert baseline["seed"] == 313
    assert profile["steps"] == 75
    assert profile["expected_attempts"] == 3
    assert profile["role"] == "profile-prefix"


@pytest.mark.parametrize(
    ("steps", "seed"),
    [(50, runtime.RUNTIME_SEED), (runtime.BASELINE_STEPS, 7)],
)
def test_runtime_workload_rejects_unapproved_steps_or_seed(steps, seed):
    with pytest.raises(runtime.DHFRNPTRuntimeError):
        runtime.build_runtime_workload(
            load_contract(),
            source_identity={"manifest_fingerprint": "a" * 64},
            steps=steps,
            seed=seed,
        )


def test_worker_report_requires_real_completion_barrier(tmp_path):
    payload = _worker(tmp_path)
    payload["completion_barrier"]["performed"] = False
    path = tmp_path / "worker.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(
        runtime.DHFRNPTRuntimeError,
        match="completion barrier",
    ):
        runtime.load_worker_report(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("platform", "Reference"),
        ("precision", "mixed"),
        ("version", "8.5.1"),
    ],
)
def test_worker_report_rejects_openmm_identity_substitution(
    tmp_path,
    field,
    value,
):
    payload = _worker(tmp_path)
    payload["engine"][field] = value
    path = tmp_path / "worker.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(runtime.DHFRNPTRuntimeError, match="identity mismatch"):
        runtime.load_worker_report(path, expected_engine="openmm")


def test_worker_report_rejects_nonreconciling_timing(tmp_path):
    payload = _worker(tmp_path)
    payload["timing"]["unaccounted_wall_seconds"] = 2.0
    path = tmp_path / "worker.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(runtime.DHFRNPTRuntimeError, match="do not reconcile"):
        runtime.load_worker_report(path)


def test_finalize_sample_binds_memory_and_complete_wall(tmp_path):
    sample = _sample(tmp_path, wall=12.0)
    report_path = tmp_path / "report.json"
    runtime.atomic_write_json(report_path, sample)

    loaded = runtime.load_sample_report(
        report_path,
        expected_engine="openmm",
        artifact_root=tmp_path,
    )

    assert loaded["timing"]["complete_wall_seconds"] == 12.0
    assert loaded["timing"]["unaccounted_wall_seconds"] == 2.5
    assert loaded["memory"]["peak_physical_bytes"] == 4_000_000_000


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bounded_process_exceeded", True),
        ("bounded_process_timed_out", True),
        ("bounded_process_returncode", 1),
    ],
)
def test_finalize_sample_rejects_failed_supervisor(tmp_path, field, value):
    trace = _memory_trace()
    trace[field] = value
    worker = _worker(tmp_path)
    memory = tmp_path / "memory.json"
    memory.write_text(json.dumps(trace))

    with pytest.raises(
        runtime.DHFRNPTRuntimeError,
        match="supervisor did not pass",
    ):
        runtime.finalize_sample_report(
            worker,
            complete_wall_seconds=12.0,
            memory_trace=trace,
            memory_record=runtime.artifact_record(memory, relative_to=tmp_path),
        )


def test_sample_artifact_tampering_fails_closed(tmp_path):
    sample = _sample(tmp_path)
    report_path = tmp_path / "report.json"
    runtime.atomic_write_json(report_path, sample)
    (tmp_path / "samples.npz").write_bytes(b"tampered")

    with pytest.raises(runtime.DHFRNPTRuntimeError, match="size mismatch"):
        runtime.load_sample_report(
            report_path,
            artifact_root=tmp_path,
        )


def test_batch_reports_raw_samples_and_median(tmp_path):
    first = _sample(tmp_path / "one", wall=14.0, run_id="run-1")
    second = _sample(tmp_path / "two", wall=10.0, run_id="run-2")
    third = _sample(tmp_path / "three", wall=12.0, run_id="run-3")

    report = runtime.build_batch_report(
        [first, second, third],
        engine="openmm",
    )

    assert report["complete_wall_seconds_samples"] == [14.0, 10.0, 12.0]
    assert report["median_complete_wall_seconds"] == 12.0
    assert report["sample_count"] == 3


def test_batch_rejects_reused_worker_identity(tmp_path):
    first = _sample(tmp_path / "one", run_id="same")
    second = _sample(tmp_path / "two", run_id="same")

    with pytest.raises(runtime.DHFRNPTRuntimeError, match="reused"):
        runtime.build_batch_report([first, second], engine="openmm")


def test_bounded_command_starts_fresh_worker_under_supervisor(tmp_path):
    command = runner.build_bounded_worker_command(
        engine="openmm",
        prepared=tmp_path / "prepared",
        sample_dir=tmp_path / "sample-001",
        steps=runtime.BASELINE_STEPS,
        seed=runtime.RUNTIME_SEED,
        platform_name="OpenCL",
        precision="single",
        contract=Path("contract.json"),
        max_bytes=runtime.PROCESS_TREE_MAX_BYTES,
        timeout_seconds=runtime.SAMPLE_TIMEOUT_SECONDS,
    )

    separator = command.index("--")
    supervisor = command[:separator]
    worker = command[separator + 1 :]
    assert str(runner.BOUNDED_PROCESS_SCRIPT) in supervisor
    assert "40000000000" in supervisor
    assert "600.0" in supervisor
    assert worker[1] == str(Path(runner.__file__).resolve())
    assert worker[2] == "_worker"
    assert worker[worker.index("--platform") + 1] == "OpenCL"
    assert worker[worker.index("--precision") + 1] == "single"


def test_bounded_command_marks_only_instrumented_worker(tmp_path):
    command = runner.build_bounded_worker_command(
        engine="mlx",
        prepared=tmp_path / "prepared",
        sample_dir=tmp_path / "sample-001",
        steps=runtime.PROFILE_STEPS,
        seed=runtime.RUNTIME_SEED,
        platform_name="Metal",
        precision="float32",
        contract=Path("contract.json"),
        max_bytes=runtime.PROCESS_TREE_MAX_BYTES,
        timeout_seconds=runtime.SAMPLE_TIMEOUT_SECONDS,
        instrumented=True,
    )

    worker = command[command.index("--") + 1 :]
    assert "--instrumented" in worker


def test_mlx_version_comes_from_installed_distribution():
    assert runner._package_version("mlx")


def test_batch_rerun_skips_completed_valid_sample(tmp_path, monkeypatch):
    expected_workload = _workload()
    sample = _sample(tmp_path / "sample", run_id="finished")
    sample_path = tmp_path / "batch" / "sample-001" / "report.json"
    sample_path.parent.mkdir(parents=True)
    sample_path.write_text("{}")
    batch = runtime.build_batch_report([sample], engine="openmm")

    monkeypatch.setattr(
        runtime,
        "load_runtime_workload",
        lambda *args, **kwargs: expected_workload,
    )
    monkeypatch.setattr(
        runtime,
        "load_sample_report",
        lambda *args, **kwargs: copy.deepcopy(sample),
    )
    monkeypatch.setattr(runner, "verify_batch", lambda *args, **kwargs: batch)

    def fail_if_run(*args, **kwargs):
        raise AssertionError("completed sample must not be rerun")

    monkeypatch.setattr(runner.subprocess, "run", fail_if_run)

    result = runner.run_batch(
        engine="openmm",
        prepared=tmp_path / "prepared",
        out=tmp_path / "batch",
        steps=runtime.BASELINE_STEPS,
        repetitions=1,
        seed=runtime.RUNTIME_SEED,
        platform_name="OpenCL",
        precision="single",
        contract_path=Path("contract.json"),
        max_bytes=runtime.PROCESS_TREE_MAX_BYTES,
        sample_timeout_seconds=runtime.SAMPLE_TIMEOUT_SECONDS,
    )

    assert result == batch
    assert runner.read_status(tmp_path / "batch")["state"] == "passed"


def test_status_is_read_only_and_reports_not_started(tmp_path):
    assert runner.read_status(tmp_path) == {
        "schema": runtime.STATUS_SCHEMA,
        "state": "not-started",
        "completed": 0,
        "requested": None,
        "active_sample": None,
    }
    assert not list(tmp_path.iterdir())


def test_route_projection_keeps_one_cold_cost_and_scales_recurrence():
    events = [
        {
            "inclusive_wall_seconds": 4.0,
            "exclusive_wall_seconds": 3.0,
            "root": True,
        },
        {
            "inclusive_wall_seconds": 2.0,
            "exclusive_wall_seconds": 1.0,
            "root": True,
        },
        {
            "inclusive_wall_seconds": 2.0,
            "exclusive_wall_seconds": 1.0,
            "root": True,
        },
    ]

    summary = runner._summarize_route_events(
        events,
        steps=runtime.PROFILE_STEPS,
    )

    assert summary["projected_750_calls"] == 30
    assert summary["projected_750_exclusive_wall_seconds"] == 32.0
    assert summary["first_exclusive_wall_seconds"] == 3.0
    assert summary["recurring_exclusive_mean_wall_seconds"] == 1.0


def test_profile_and_hotspot_reports_preserve_diagnostic_boundary(tmp_path):
    clean = _sample(
        tmp_path / "clean",
        engine="mlx",
        wall=10.0,
        steps=runtime.PROFILE_STEPS,
        mode="clean",
    )
    instrumented = _sample(
        tmp_path / "instrumented",
        engine="mlx",
        wall=12.0,
        steps=runtime.PROFILE_STEPS,
        mode="instrumented",
        profile=_instrumented_profile(),
    )

    profile = runtime.build_profile_report(clean, instrumented)
    hotspots = runtime.build_hotspot_report(
        profile,
        route_audit={
            "barostat.attempt": {
                "path": "src/mlx_atomistic/md.py",
                "line": 3955,
                "symbol": "_attempt_barostat_move",
                "hypothesis": "Reuse proposal state.",
            },
            "constraints.position_projection": {
                "path": "src/mlx_atomistic/constraints.py",
                "line": 74,
                "symbol": "DistanceConstraints.apply_positions",
                "hypothesis": "Compile projection.",
            },
        },
    )

    assert profile["role"] == "diagnostic-prefix-only"
    assert profile["profiling_overhead_wall_seconds"] == 2.0
    assert profile["reconciliation"]["relative_error"] <= 0.10
    assert hotspots["selected_route"] == "barostat.attempt"
    assert hotspots["candidates"][0]["projected_750_calls"] == 30
    assert hotspots["ranking_resolution"] == "ambiguous-close-low-recurrence"
    profile_path = tmp_path / "profile.json"
    hotspots_path = tmp_path / "hotspots.json"
    runtime.atomic_write_json(profile_path, profile)
    runtime.atomic_write_json(hotspots_path, hotspots)
    assert runtime.load_profile_report(profile_path) == profile
    assert (
        runtime.load_hotspot_report(
            hotspots_path,
            expected_profile_fingerprint=profile["report_fingerprint"],
        )
        == hotspots
    )
    tampered = copy.deepcopy(hotspots)
    tampered["selected_route"] = "constraints.position_projection"
    hotspots_path.write_text(json.dumps(tampered))
    with pytest.raises(runtime.DHFRNPTRuntimeError, match="fingerprint"):
        runtime.load_hotspot_report(hotspots_path)


def test_profile_report_rejects_unlabeled_instrumented_sample(tmp_path):
    clean = _sample(
        tmp_path / "clean",
        engine="mlx",
        steps=runtime.PROFILE_STEPS,
    )
    instrumented = _sample(
        tmp_path / "instrumented",
        engine="mlx",
        steps=runtime.PROFILE_STEPS,
        profile=_instrumented_profile(),
    )

    with pytest.raises(runtime.DHFRNPTRuntimeError, match="not labeled"):
        runtime.build_profile_report(clean, instrumented)


def test_profile_route_source_audit_has_concrete_anchors():
    audit = runner._route_source_audit()

    assert audit["barostat.attempt"]["path"] == "src/mlx_atomistic/md.py"
    assert audit["barostat.attempt"]["line"] > 0
    assert audit["barostat.attempt"]["extracts_scalar"] is True
    assert audit["pme.reciprocal"]["symbol"] == (
        "_mesh_reciprocal_energy_forces_mx"
    )
