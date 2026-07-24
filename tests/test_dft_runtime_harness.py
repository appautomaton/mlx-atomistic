from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import weakref
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import mlx_atomistic.dft.periodic_scf as periodic_scf_module
from mlx_atomistic import _artifact_identity as artifact_identity
from mlx_atomistic._artifact_identity import (
    ArtifactIntegrityError,
    AtomicGeneration,
    canonical_json_bytes,
    confined_path,
    inspect_generation,
    inventory_fingerprint,
    sha256_bytes,
    source_inventory,
)
from mlx_atomistic.benchmarks import dft_runtime as runtime_cli
from mlx_atomistic.benchmarks import dft_runtime_contract as contract
from mlx_atomistic.benchmarks import dft_runtime_core as runtime_core
from mlx_atomistic.benchmarks.dft_runtime import main
from mlx_atomistic.benchmarks.dft_runtime_contract import (
    TARGET_CHIP,
    build_source_fingerprints,
    collect_host_provenance,
    host_admission,
    load_workload,
    parse_current_power_source,
    parse_power_profiles,
    prepare_workload,
    results_output_path,
)
from mlx_atomistic.benchmarks.dft_runtime_core import (
    PRE_ARCHITECTURE_REV,
    _finalize_report,
    _formal_admission,
    _publish_report,
    inspect_artifact,
    run_compare,
    run_fixed_density,
    run_full_scf,
    supervise_full_scf_worker,
)
from mlx_atomistic.dft import (
    PeriodicDavidsonConfig,
    PeriodicKohnShamOperator,
    PlaneWaveBasis,
    RealSpaceGrid,
    solve_periodic_eigenproblem,
)
from mlx_atomistic.dft._runtime_observer import RuntimeObserver


def _oracle_module():
    path = Path(__file__).parents[1] / "scripts/run_dft_runtime_oracle.py"
    spec = importlib.util.spec_from_file_location("dft_runtime_oracle_test_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load DFT runtime oracle module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gth_database(path: Path, *, selected_local: str = "-6.26928833") -> Path:
    path.write_text(
        f"""# unrelated leading entry
H GTH-PBE-q1 GTH-PBE
  1
  0.2 1 -1.0
  0
# selected entry
Si GTH-PBE-q4 GTH-PBE
  2 2
  0.44000000 1 {selected_local}
  2
  0.43563383 2 8.95174150 -2.70627082
                 3.49378060
  0.49794218 1 2.43127673
# unrelated trailing entry
P GTH-PBE-q5 GTH-PBE
  2 3
  0.43 1 -5.8
  0
"""
    )
    return path


def _host_outputs(
    *,
    source: str = "AC Power",
    ac_mode: int = 1,
    battery_mode: int = 1,
    mode_key: str = "lowpowermode",
):
    return {
        ("system_profiler", "SPHardwareDataType"): {
            "status": "ok",
            "stdout": """Hardware:
    Model Name: MacBook Pro
    Model Identifier: Mac16,9
    Chip: Apple M5 Max
    Memory: 128 GB
    Serial Number (system): SECRET
""",
        },
        ("sw_vers",): {
            "status": "ok",
            "stdout": "ProductName:\tmacOS\nProductVersion:\t26.0\nBuildVersion:\t25A1\n",
        },
        ("pmset", "-g", "batt"): {
            "status": "ok",
            "stdout": f"Now drawing from '{source}'\n",
        },
        ("pmset", "-g", "custom"): {
            "status": "ok",
            "stdout": (
                f"Battery Power:\n {mode_key} {battery_mode}\n sleep 1\n"
                f"AC Power:\n {mode_key} {ac_mode}\n sleep 0\n"
            ),
        },
        ("sysctl", "-n", "kern.thermal_pressure"): {
            "status": "blocked",
            "error": "unavailable",
        },
    }


def _runner(outputs, calls):
    def run(command):
        calls.append(tuple(command))
        return outputs[tuple(command)]

    return run


def _fake_fixed_report(
    path: Path,
    *,
    runtime: str,
    protocol: str,
    revision: str,
    power_source: str,
    elapsed: float,
    baseline_seal_fingerprint: str | None,
    numerical_status: str = "passed",
    timing_status: str = "admitted",
    diagnostic: bool = False,
    dirty: bool = False,
) -> Path:
    identity = {
        "workload_fingerprint": "w" * 64,
        "protocol_fingerprint": protocol,
        "runtime_fingerprint": runtime,
        "execution_contract_fingerprint": runtime,
    }
    host = {
        "chip": TARGET_CHIP,
        "power_source": power_source,
        "active_power_profile": {"lowpowermode": 1},
        "power_mode_key": "lowpowermode",
        "low_power_mode": 1,
    }
    comparison_protocol = {
        "workload_fingerprint": "w" * 64,
        "selected_gth_resource": {
            "role": "si_gth_pbe_q4",
            "byte_size": 1,
            "sha256": "g" * 64,
        },
        "protocol_fingerprint": protocol,
        **runtime_core._host_protocol(host),
    }
    run_protocol = {
        "warmups": 1,
        "samples": 5,
        "fresh": True,
        "diagnostic": diagnostic,
        "resumed": False,
    }
    statuses = {
        "numerical_status": numerical_status,
        "resume_integrity_status": "fresh-no-resume",
        "timing_admission_status": timing_status,
    }
    admission = {"passed": True, "blockers": []}
    report = _finalize_report(
        {
            "schema_version": "mlx-atomistic.dft-runtime-report.v1",
            "kind": "fixed-density",
            "identity": identity,
            "context": {"git": {"revision": revision, "dirty": dirty}},
            "host": host,
            "comparison_protocol": comparison_protocol,
            "run_protocol": run_protocol,
            "samples": [{} for _ in range(5)],
            "summary": {"median_elapsed_seconds": elapsed},
            "baseline_seal_fingerprint": baseline_seal_fingerprint,
            "statuses": statuses,
            "admission": admission,
            "formal_admission": _formal_admission(
                statuses=statuses,
                command_admission=admission,
                producer_git={"revision": revision, "dirty": dirty},
                run_protocol=run_protocol,
                report_kind="fixed-density",
                host_protocol=host,
            ),
        }
    )
    _publish_report(
        out=path,
        artifact_kind="dft-runtime-fixed-density",
        artifact_schema="mlx-atomistic.dft-runtime-report.v1",
        report=report,
    )
    return path


def _fake_baseline_seal(
    path: Path,
    *,
    protocol: str,
    runtime: str,
    revision: str,
    base_revision: str = PRE_ARCHITECTURE_REV,
    parent_revision: str | None = None,
    extra: dict[str, object] | None = None,
) -> tuple[Path, str]:
    values = dict(extra or {})
    parent = (
        runtime_core.BASELINE_EXPECTED_PARENT_REV
        if parent_revision is None
        else parent_revision
    )
    workload_fingerprint = str(values.get("workload_fingerprint", "w" * 64))
    identity = {
        "workload_fingerprint": workload_fingerprint,
        "protocol_fingerprint": protocol,
        "runtime_fingerprint": runtime,
        "execution_contract_fingerprint": runtime,
    }
    selected_resource = values.get(
        "selected_gth_resource",
        {
            "role": "si_gth_pbe_q4",
            "byte_size": 1,
            "sha256": "g" * 64,
        },
    )
    host = values.get(
        "host",
        {
            "chip": TARGET_CHIP,
            "power_source": "AC Power",
            "active_power_profile": {"lowpowermode": 1},
            "power_mode_key": "lowpowermode",
            "low_power_mode": 1,
        },
    )
    sealed_host_protocol = values.get(
        "host_protocol",
        runtime_core._host_protocol(host),
    )
    comparison_protocol = values.get(
        "comparison_protocol",
        {
            "workload_fingerprint": workload_fingerprint,
            "selected_gth_resource": selected_resource,
            "protocol_fingerprint": protocol,
            **runtime_core._host_protocol(host),
        },
    )
    run_protocol = {
        "warmups": 1,
        "samples": 5,
        "fresh": True,
        "diagnostic": False,
        "resumed": False,
    }
    statuses = {
        "numerical_status": "passed",
        "resume_integrity_status": "fresh-no-resume",
        "timing_admission_status": "admitted",
    }
    admission = {"passed": True, "blockers": []}
    formal_admission = _formal_admission(
        statuses=statuses,
        command_admission=admission,
        producer_git={"revision": revision, "dirty": False},
        run_protocol=run_protocol,
        report_kind="fixed-density",
        host_protocol=host,
    )
    median_elapsed = values.get("median_elapsed_seconds", 8.0)
    raw_elapsed = values.get("raw_elapsed_seconds", [median_elapsed] * 5)
    representative_observation = values.get(
        "representative_observation",
        {"work_counters": {}, "memory": {}},
    )
    fixed_density_eigenvalues = values.get(
        "fixed_density_eigenvalues_hartree",
        [float(index) for index in range(16)],
    )
    eigenvalue_tolerance = values.get(
        "fixed_density_eigenvalue_tolerance_hartree",
        1e-5,
    )
    baseline_structure_audit = values.get(
        "baseline_structure_audit",
        {"passed": True},
    )
    baseline_diff_audit = values.get(
        "baseline_diff_audit",
        {
            "base_revision": base_revision,
            "baseline_revision": revision,
            "baseline_parent_revision": parent,
            "checks": {
                "git_commands_succeeded": True,
                "pre_architecture_revision_is_ancestor": True,
                "baseline_history_has_no_merge_commits": True,
                "baseline_parent_is_expected_prebaseline_revision": True,
                "baseline_revision_is_distinct": True,
                "diff_is_nonempty": True,
                "diff_paths_are_allowed": True,
                "diff_records_are_parseable_regular_files": True,
            },
            "allowed_paths": sorted(runtime_core.BASELINE_ALLOWED_DIFF_PATHS),
            "changed_files": [
                {
                    "status": "M",
                    "path": sorted(runtime_core.BASELINE_ALLOWED_DIFF_PATHS)[0],
                    "byte_size": 1,
                    "sha256": "d" * 64,
                }
            ],
            "patch_sha256": "c" * 64,
            "passed": True,
        },
    )
    report = _finalize_report(
        {
            "schema_version": runtime_core.REPORT_SCHEMA,
            "kind": "fixed-density",
            "identity": identity,
            "context": {
                "protocol_inventory": [],
                "runtime_inventory": [],
                "git": {
                    "revision": revision,
                    "parent": parent,
                    "dirty": False,
                }
            },
            "host": host,
            "comparison_protocol": comparison_protocol,
            "run_protocol": run_protocol,
            "warmup_results": [{}],
            "samples": [{} for _ in range(5)],
            "summary": {
                "median_elapsed_seconds": median_elapsed,
                "raw_elapsed_seconds": raw_elapsed,
                "representative_observation": representative_observation,
                "fixed_density_eigenvalues": {
                    "representative_eigenvalues_hartree": fixed_density_eigenvalues,
                    "tolerance_hartree": eigenvalue_tolerance,
                },
                "baseline_structure_audit": baseline_structure_audit,
                "baseline_diff_audit": baseline_diff_audit,
            },
            "baseline_seal_fingerprint": None,
            "statuses": statuses,
            "admission": admission,
            "formal_admission": formal_admission,
        }
    )
    unsigned = {
        "schema_version": runtime_core.SEAL_SCHEMA,
        "baseline_rev": revision,
        "base_rev": base_revision,
        "parent_rev": parent,
        "dirty": False,
        "workload_fingerprint": workload_fingerprint,
        "selected_gth_resource": selected_resource,
        "protocol_fingerprint": protocol,
        "protocol_inventory": [],
        "baseline_runtime_fingerprint": runtime,
        "baseline_runtime_inventory": [],
        "comparison_protocol": comparison_protocol,
        "host_protocol": sealed_host_protocol,
        "median_elapsed_seconds": median_elapsed,
        "raw_elapsed_seconds": raw_elapsed,
        "representative_observation": representative_observation,
        "fixed_density_eigenvalues_hartree": fixed_density_eigenvalues,
        "fixed_density_eigenvalue_tolerance_hartree": eigenvalue_tolerance,
        "baseline_structure_audit": baseline_structure_audit,
        "baseline_diff_audit": baseline_diff_audit,
        "target_optimizations_absent": list(
            runtime_core.BASELINE_ABSENT_OPTIMIZATIONS
        ),
        "report_fingerprint": report["report_fingerprint"],
        **values,
    }
    seal = {**unsigned, "seal_fingerprint": sha256_bytes(canonical_json_bytes(unsigned))}
    with AtomicGeneration(
        path,
        "dft-runtime-fixed-density",
        runtime_core.REPORT_SCHEMA,
        identity=identity,
        metadata={
            "admission": admission,
            "formal_admission": formal_admission,
        },
    ) as generation:
        generation.write_json("report.json", report)
        generation.write_json("seal.json", seal)
        generation.publish()
    return path, str(seal["seal_fingerprint"])


def _sleeping_process_group_worker(queue, manifest_path, gth_source, state_root):
    del manifest_path, gth_source
    isolated = False
    if hasattr(os, "setsid"):
        os.setsid()
        isolated = True
    heartbeat = Path(state_root, "heartbeat")
    code = (
        "from pathlib import Path\n"
        "import signal,sys,time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "path=Path(sys.argv[1])\n"
        "i=0\n"
        "while True:\n"
        " path.write_text(str(i))\n"
        " i+=1\n"
        " time.sleep(0.02)\n"
    )
    child = subprocess.Popen([sys.executable, "-c", code, str(heartbeat)])
    Path(state_root, "child.pid").write_text(str(child.pid))
    queue.put({"type": "ready", "process_group_isolated": isolated})
    queue.put({"type": "progress", "event": {"event": "fake-worker-started"}})
    time.sleep(60)


def _delayed_ready_process_group_worker(queue, manifest_path, gth_source, state_root):
    del manifest_path, gth_source
    if hasattr(os, "setsid"):
        os.setsid()
    heartbeat = Path(state_root, "heartbeat")
    code = (
        "from pathlib import Path\n"
        "import signal,sys,time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "path=Path(sys.argv[1])\n"
        "i=0\n"
        "while True:\n"
        " path.write_text(str(i))\n"
        " i+=1\n"
        " time.sleep(0.02)\n"
    )
    child = subprocess.Popen([sys.executable, "-c", code, str(heartbeat)])
    Path(state_root, "child.pid").write_text(str(child.pid))
    deadline = time.monotonic() + 1.0
    while not heartbeat.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    queue.put({"type": "progress", "event": {"event": "child-spawned-before-ready"}})
    time.sleep(60)


def _result_then_crash_worker(queue, manifest_path, gth_source, state_root):
    del manifest_path, gth_source, state_root
    isolated = False
    if hasattr(os, "setsid"):
        os.setsid()
        isolated = True
    queue.put({"type": "ready", "process_group_isolated": isolated})
    queue.put({"type": "result", "result": {"numerical_passed": True}})
    queue.close()
    queue.join_thread()
    os._exit(7)


def test_canonical_json_is_stable_and_rejects_nonfinite():
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("nan")})


def test_atomic_generation_round_trip_is_confined_and_no_overwrite(tmp_path):
    destination = tmp_path / "generation"
    with AtomicGeneration(
        destination,
        "unit-test",
        "unit-test.v1",
        identity={"fingerprint": "a" * 64},
    ) as generation:
        generation.write_json("nested/report.json", {"ok": True})
        generation.publish()

    before = sorted(path.relative_to(destination) for path in destination.rglob("*"))
    manifest = inspect_generation(destination / "nested/report.json")
    after = sorted(path.relative_to(destination) for path in destination.rglob("*"))
    assert manifest["complete"] is True
    assert before == after
    with (
        pytest.raises(FileExistsError),
        AtomicGeneration(destination, "unit-test", "unit-test.v1"),
    ):
        pass
    for invalid in ("../escape", "/absolute", "a/../escape", "a\\b"):
        with pytest.raises(ValueError):
            confined_path(tmp_path, invalid)


def test_atomic_generation_detects_tamper_extra_file_and_symlink_escape(tmp_path):
    destination = tmp_path / "generation"
    with AtomicGeneration(destination, "unit-test", "unit-test.v1") as generation:
        generation.write_bytes("payload.bin", b"trusted")
        generation.publish()
    (destination / "payload.bin").write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError, match="inventory|checksum"):
        inspect_generation(destination)

    extra = tmp_path / "extra"
    with AtomicGeneration(extra, "unit-test", "unit-test.v1") as generation:
        generation.write_bytes("payload.bin", b"trusted")
        generation.publish()
    (extra / "unexpected.bin").write_bytes(b"extra")
    with pytest.raises(ArtifactIntegrityError, match="inventory"):
        inspect_generation(extra)

    escaped = tmp_path / "escaped"
    with AtomicGeneration(escaped, "unit-test", "unit-test.v1") as generation:
        outside = tmp_path / "outside"
        outside.write_bytes(b"outside")
        generation.path("unsafe").symlink_to(outside)
        with pytest.raises(ArtifactIntegrityError, match="regular file"):
            generation.publish()


def test_atomic_generation_race_and_faults_never_replace_or_leak(tmp_path):
    destination = tmp_path / "generation"

    def competing_publisher(stage):
        if stage == "before_rename":
            destination.mkdir()

    with AtomicGeneration(
        destination,
        "unit-test",
        "unit-test.v1",
        fault_hook=competing_publisher,
    ) as generation:
        generation.write_bytes("payload.bin", b"candidate")
        with pytest.raises(FileExistsError):
            generation.publish()
    assert destination.is_dir()
    assert not list(destination.iterdir())
    assert not list(tmp_path.glob(".generation.tmp-*"))

    failed = tmp_path / "failed"

    def injected_failure(stage):
        if stage == "after_manifest":
            raise RuntimeError("injected publication failure")

    with (
        pytest.raises(RuntimeError, match="injected publication failure"),
        AtomicGeneration(
            failed,
            "unit-test",
            "unit-test.v1",
            fault_hook=injected_failure,
        ) as generation,
    ):
        generation.write_bytes("payload.bin", b"candidate")
        generation.publish()
    assert not failed.exists()
    assert not list(tmp_path.glob(".failed.tmp-*"))


def test_unpublished_crash_temporary_generation_is_not_inspectable(tmp_path):
    def crash_before_rename(stage):
        if stage == "before_rename":
            raise RuntimeError("crash before rename")

    crashed = AtomicGeneration(
        tmp_path / "crashed",
        "unit-test",
        "unit-test.v1",
        fault_hook=crash_before_rename,
    )
    crashed.__enter__()
    crashed.write_bytes("payload.bin", b"candidate")
    with pytest.raises(RuntimeError, match="crash before rename"):
        crashed.publish()
    temporary = crashed.root
    assert (temporary / artifact_identity.GENERATION_MANIFEST).is_file()
    with pytest.raises(ArtifactIntegrityError, match="unpublished temporary"):
        inspect_generation(temporary)


def test_atomic_generation_rejects_hardlinks_and_traversal_errors(
    tmp_path, monkeypatch
):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"mutable")
    with AtomicGeneration(
        tmp_path / "hardlink",
        "unit-test",
        "unit-test.v1",
    ) as generation:
        os.link(outside, generation.path("payload.bin"))
        with pytest.raises(ArtifactIntegrityError, match="hard-linked"):
            generation.publish()

    manifest_generation = tmp_path / "manifest-hardlink"
    with AtomicGeneration(
        manifest_generation,
        "unit-test",
        "unit-test.v1",
    ) as generation:
        generation.write_bytes("payload.bin", b"payload")
        generation.publish()
    os.link(
        manifest_generation / artifact_identity.GENERATION_MANIFEST,
        tmp_path / "linked-manifest.json",
    )
    with pytest.raises(ArtifactIntegrityError, match="manifest.*hard-linked"):
        inspect_generation(manifest_generation)

    def failed_walk(root, *, followlinks, onerror):
        del root, followlinks
        onerror(PermissionError("denied"))
        return iter(())

    monkeypatch.setattr(artifact_identity.os, "walk", failed_walk)
    with AtomicGeneration(
        tmp_path / "unreadable",
        "unit-test",
        "unit-test.v1",
    ) as generation:
        generation.write_bytes("payload.bin", b"payload")
        with pytest.raises(ArtifactIntegrityError, match="traversal failed"):
            generation.publish()


def test_report_extra_tree_paths_integrity_and_symlink_rejection(tmp_path):
    report = _finalize_report(
        {
            "schema_version": runtime_core.FULL_SCF_SCHEMA,
            "kind": "unit-report",
            "identity": {"workload_fingerprint": "w" * 64},
            "admission": {"passed": True, "blockers": []},
        }
    )
    tree = tmp_path / "state"
    (tree / "final-state").mkdir(parents=True)
    (tree / "final-state/data.bin").write_bytes(b"state")
    published = _publish_report(
        out=tmp_path / "published",
        artifact_kind="unit-report",
        artifact_schema="unit-report.v1",
        report=report,
        extra_tree=tree,
    )
    assert Path(published["artifact"]) == tmp_path / "published"
    assert Path(published["report"]) == tmp_path / "published/report.json"
    assert Path(published["artifact"]).is_dir()
    assert Path(published["report"]).is_file()
    assert inspect_generation(published["artifact"])["complete"] is True

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    unsafe = tmp_path / "unsafe-state"
    unsafe.mkdir()
    (unsafe / "leak.bin").symlink_to(outside)
    with pytest.raises(ArtifactIntegrityError, match="regular file"):
        _publish_report(
            out=tmp_path / "unsafe-published",
            artifact_kind="unit-report",
            artifact_schema="unit-report.v1",
            report=report,
            extra_tree=unsafe,
        )
    assert not (tmp_path / "unsafe-published").exists()


@pytest.mark.parametrize(
    "envelope_override",
    (
        {"artifact_kind": "dft-runtime-comparison"},
        {"artifact_schema": runtime_core.COMPARISON_SCHEMA},
        {"identity": {"workload_fingerprint": "forged"}},
        {"metadata": {"admission": {"passed": False, "blockers": ["forged"]}}},
        {
            "metadata": {
                "admission": {"passed": True, "blockers": []},
                "formal_admission": {"passed": False, "blockers": ["forged"]},
            }
        },
    ),
)
def test_report_loader_rejects_mismatched_generation_envelope(
    tmp_path, envelope_override
):
    identity = {
        "workload_fingerprint": "w" * 64,
        "protocol_fingerprint": "p" * 64,
        "runtime_fingerprint": "r" * 64,
        "execution_contract_fingerprint": "e" * 64,
    }
    admission = {"passed": True, "blockers": []}
    formal_admission = {"passed": False, "blockers": ["formal-test-only"]}
    report = _finalize_report(
        {
            "schema_version": runtime_core.REPORT_SCHEMA,
            "kind": "fixed-density",
            "identity": identity,
            "statuses": {
                "numerical_status": "blocked",
                "resume_integrity_status": "blocked",
                "timing_admission_status": "blocked",
            },
            "admission": admission,
            "formal_admission": formal_admission,
        }
    )
    envelope = {
        "artifact_kind": "dft-runtime-fixed-density",
        "artifact_schema": runtime_core.REPORT_SCHEMA,
        "identity": identity,
        "metadata": {
            "admission": admission,
            "formal_admission": formal_admission,
        },
        **envelope_override,
    }
    destination = tmp_path / f"mismatch-{len(list(tmp_path.iterdir()))}"
    with AtomicGeneration(
        destination,
        envelope["artifact_kind"],
        envelope["artifact_schema"],
        identity=envelope["identity"],
        metadata=envelope["metadata"],
    ) as generation:
        generation.write_json("report.json", report)
        generation.publish()
    with pytest.raises(ValueError, match="generation envelope"):
        runtime_core._load_report(destination)


@pytest.mark.parametrize(
    ("field", "forged"),
    (
        ("low_power_mode", None),
        ("low_power_mode", 0),
        ("power_mode_key", "lowpowermode"),
    ),
)
def test_report_loader_rejects_laundered_raw_host_normalization(
    tmp_path, field, forged
):
    identity = {
        "workload_fingerprint": "w" * 64,
        "protocol_fingerprint": "p" * 64,
        "runtime_fingerprint": "r" * 64,
        "execution_contract_fingerprint": "e" * 64,
    }
    host = {
        "chip": TARGET_CHIP,
        "power_source": "AC Power",
        "active_power_profile": {"powermode": 1},
        "power_mode_key": "powermode",
        "low_power_mode": 1,
    }
    host[field] = forged
    report = _finalize_report(
        {
            "schema_version": runtime_core.REPORT_SCHEMA,
            "kind": "fixed-density",
            "identity": identity,
            "context": {"git": {"dirty": False}},
            "host": host,
            "run_protocol": {
                "warmups": 1,
                "samples": 5,
                "fresh": True,
                "resumed": False,
                "diagnostic": False,
            },
            "statuses": {
                "numerical_status": "passed",
                "resume_integrity_status": "fresh-no-resume",
                "timing_admission_status": "admitted",
            },
            "admission": {"passed": True, "blockers": []},
            "formal_admission": {"passed": True, "blockers": []},
        }
    )
    destination = tmp_path / f"laundered-host-{field}-{forged!s}"
    _publish_report(
        out=destination,
        artifact_kind="dft-runtime-fixed-density",
        artifact_schema=runtime_core.REPORT_SCHEMA,
        report=report,
    )
    with pytest.raises(ValueError, match="formal admission is inconsistent"):
        runtime_core._load_report(destination)


def test_baseline_seal_requires_bound_report_and_generation_envelope(tmp_path):
    valid, _fingerprint = _fake_baseline_seal(
        tmp_path / "valid-seal",
        protocol="p" * 64,
        runtime="r" * 64,
        revision="b" * 40,
    )
    loaded = runtime_core._load_seal(valid)
    report = json.loads((valid / "report.json").read_text())
    generation_manifest = inspect_generation(valid)
    assert loaded["report_fingerprint"] == report["report_fingerprint"]

    detached = tmp_path / "detached-seal"
    with AtomicGeneration(
        detached,
        "dft-runtime-fixed-density",
        runtime_core.REPORT_SCHEMA,
        identity=generation_manifest["identity"],
        metadata=generation_manifest["metadata"],
    ) as generation:
        generation.write_json("seal.json", loaded)
        generation.publish()
    with pytest.raises(FileNotFoundError):
        runtime_core._load_seal(detached)

    wrong_envelope = tmp_path / "wrong-seal-envelope"
    with AtomicGeneration(
        wrong_envelope,
        "dft-runtime-comparison",
        runtime_core.REPORT_SCHEMA,
        identity=generation_manifest["identity"],
        metadata=generation_manifest["metadata"],
    ) as generation:
        generation.write_json("report.json", report)
        generation.write_json("seal.json", loaded)
        generation.publish()
    with pytest.raises(ValueError, match="generation envelope"):
        runtime_core._load_seal(wrong_envelope)

    forged = dict(loaded)
    forged["report_fingerprint"] = "f" * 64
    unsigned = {
        key: value for key, value in forged.items() if key != "seal_fingerprint"
    }
    forged["seal_fingerprint"] = sha256_bytes(canonical_json_bytes(unsigned))
    wrong_binding = tmp_path / "wrong-seal-binding"
    with AtomicGeneration(
        wrong_binding,
        "dft-runtime-fixed-density",
        runtime_core.REPORT_SCHEMA,
        identity=generation_manifest["identity"],
        metadata=generation_manifest["metadata"],
    ) as generation:
        generation.write_json("report.json", report)
        generation.write_json("seal.json", forged)
        generation.publish()
    with pytest.raises(ValueError, match="co-published report"):
        runtime_core._load_seal(wrong_binding)
    with pytest.raises(ValueError, match="co-published report"):
        inspect_artifact(artifact=wrong_binding, require_admitted=True)


@pytest.mark.parametrize("audit", ({"passed": False}, None, "invalid"))
def test_baseline_seal_rejects_failed_or_malformed_diff_audit(tmp_path, audit):
    artifact, _fingerprint = _fake_baseline_seal(
        tmp_path / f"invalid-audit-{len(list(tmp_path.iterdir()))}",
        protocol="p" * 64,
        runtime="r" * 64,
        revision="b" * 40,
        extra={"baseline_diff_audit": audit},
    )
    with pytest.raises(ValueError, match="inconsistent with its admitted report"):
        runtime_core._load_seal(artifact)


def test_baseline_seal_creation_and_comparison_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError, match="cannot also consume"):
        run_fixed_density(
            manifest_path="unused",
            gth_source="unused",
            out=tmp_path / "unused",
            warmups=1,
            samples=5,
            fresh=True,
            diagnostic=False,
            seal=True,
            compare_seal="unused",
        )


def test_baseline_diff_audit_fails_closed_for_git_and_structure_drift(
    tmp_path, monkeypatch
):
    def git(root, *arguments):
        return subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def repository(name):
        root = tmp_path / name
        (root / "src/mlx_atomistic/dft").mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname='audit-fixture'\n")
        (root / "allowed.py").write_text("VALUE = 1\n")
        git(root, "init", "-q")
        git(root, "config", "user.email", "audit@example.invalid")
        git(root, "config", "user.name", "Audit Fixture")
        git(root, "add", ".")
        git(root, "commit", "-qm", "base")
        return root, git(root, "rev-parse", "HEAD")

    root, base = repository("passing")
    (root / "allowed.py").write_text("VALUE = 2\n")
    git(root, "add", "allowed.py")
    git(root, "commit", "-qm", "allowed change")
    monkeypatch.setattr(runtime_core, "PRE_ARCHITECTURE_REV", base)
    monkeypatch.setattr(runtime_core, "BASELINE_EXPECTED_PARENT_REV", base)
    monkeypatch.setattr(
        runtime_core,
        "BASELINE_ALLOWED_DIFF_PATHS",
        frozenset({"allowed.py"}),
    )
    admitted = runtime_core._baseline_diff_audit(root)
    assert admitted["passed"] is True
    admitted_git = {
        "revision": admitted["baseline_revision"],
        "parent": admitted["baseline_parent_revision"],
    }
    assert runtime_core._baseline_diff_audit_matches(admitted, admitted_git) is True
    assert runtime_core._baseline_diff_audit_matches(
        {"passed": True, "checks": {"made_up": True}},
        admitted_git,
    ) is False
    assert admitted["patch_sha256"] == runtime_core._baseline_diff_audit(root)[
        "patch_sha256"
    ]

    monkeypatch.setattr(runtime_core, "BASELINE_ALLOWED_DIFF_PATHS", frozenset())
    disallowed = runtime_core._baseline_diff_audit(root)
    assert disallowed["passed"] is False
    assert disallowed["checks"]["diff_paths_are_allowed"] is False

    renamed_root, renamed_base = repository("renamed")
    git(renamed_root, "mv", "allowed.py", "renamed.py")
    git(renamed_root, "commit", "-qm", "rename")
    monkeypatch.setattr(runtime_core, "PRE_ARCHITECTURE_REV", renamed_base)
    monkeypatch.setattr(runtime_core, "BASELINE_EXPECTED_PARENT_REV", renamed_base)
    monkeypatch.setattr(
        runtime_core,
        "BASELINE_ALLOWED_DIFF_PATHS",
        frozenset({"allowed.py", "renamed.py"}),
    )
    renamed = runtime_core._baseline_diff_audit(renamed_root)
    assert renamed["passed"] is False
    assert renamed["checks"]["diff_records_are_parseable_regular_files"] is False

    parent_root, parent_base = repository("additive-gap-fix")
    (parent_root / "allowed.py").write_text("VALUE = 2\n")
    git(parent_root, "add", "allowed.py")
    git(parent_root, "commit", "-qm", "middle")
    middle_revision = git(parent_root, "rev-parse", "HEAD")
    (parent_root / "allowed.py").write_text("VALUE = 3\n")
    git(parent_root, "add", "allowed.py")
    git(parent_root, "commit", "-qm", "final")
    monkeypatch.setattr(runtime_core, "PRE_ARCHITECTURE_REV", parent_base)
    monkeypatch.setattr(
        runtime_core,
        "BASELINE_EXPECTED_PARENT_REV",
        middle_revision,
    )
    additive = runtime_core._baseline_diff_audit(parent_root)
    assert additive["passed"] is True
    assert additive["baseline_parent_revision"] == middle_revision
    assert additive["checks"]["pre_architecture_revision_is_ancestor"] is True

    sibling_root, sibling_base = repository("sibling-root")
    main_branch = git(sibling_root, "branch", "--show-current")
    git(sibling_root, "checkout", "-qb", "sibling", sibling_base)
    (sibling_root / "allowed.py").write_text("VALUE = 20\n")
    git(sibling_root, "add", "allowed.py")
    git(sibling_root, "commit", "-qm", "sibling")
    sibling_revision = git(sibling_root, "rev-parse", "HEAD")
    git(sibling_root, "checkout", "-q", main_branch)
    (sibling_root / "allowed.py").write_text("VALUE = 2\n")
    git(sibling_root, "add", "allowed.py")
    git(sibling_root, "commit", "-qm", "main")
    monkeypatch.setattr(runtime_core, "PRE_ARCHITECTURE_REV", sibling_revision)
    monkeypatch.setattr(
        runtime_core,
        "BASELINE_EXPECTED_PARENT_REV",
        git(sibling_root, "rev-parse", "HEAD^"),
    )
    sibling = runtime_core._baseline_diff_audit(sibling_root)
    assert sibling["passed"] is False
    assert sibling["checks"]["pre_architecture_revision_is_ancestor"] is False

    empty_root, empty_base = repository("empty")
    monkeypatch.setattr(runtime_core, "PRE_ARCHITECTURE_REV", empty_base)
    monkeypatch.setattr(runtime_core, "BASELINE_EXPECTED_PARENT_REV", None)
    empty = runtime_core._baseline_diff_audit(empty_root)
    assert empty["passed"] is False
    assert empty["checks"]["diff_is_nonempty"] is False

    def unavailable_git(*args, **kwargs):
        del args, kwargs
        raise subprocess.TimeoutExpired(("git", "diff"), 10.0)

    monkeypatch.setattr(runtime_core.subprocess, "run", unavailable_git)
    unavailable = runtime_core._baseline_diff_audit(empty_root)
    assert unavailable["passed"] is False
    assert unavailable["checks"]["git_commands_succeeded"] is False


def test_source_inventories_are_relocation_stable_and_scope_changes(tmp_path, monkeypatch):
    def checkout(name: str) -> Path:
        root = tmp_path / name
        (root / "src/mlx_atomistic/dft").mkdir(parents=True)
        (root / "docs").mkdir()
        (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
        (root / "protocol.py").write_text("PROTOCOL = 1\n")
        (root / "runtime.py").write_text("RUNTIME = 1\n")
        (root / "src/mlx_atomistic/dft/hot.py").write_text("HOT = 1\n")
        (root / "docs/readme.md").write_text("docs v1\n")
        return root

    first = checkout("one")
    second = checkout("two")
    monkeypatch.setattr(contract, "PROTOCOL_SOURCE_PATHS", ("protocol.py",))
    monkeypatch.setattr(contract, "RUNTIME_SOURCE_PATHS", ("runtime.py",))
    monkeypatch.setattr(contract, "RUNTIME_SOURCE_ROOTS", ("src/mlx_atomistic/dft",))
    original = build_source_fingerprints(first)
    relocated = build_source_fingerprints(second)
    assert original == relocated

    (second / "docs/readme.md").write_text("docs v2\n")
    assert build_source_fingerprints(second) == original
    (second / "src/mlx_atomistic/dft/hot.py").write_text("HOT = 2\n")
    hot = build_source_fingerprints(second)
    assert hot["protocol_fingerprint"] == original["protocol_fingerprint"]
    assert hot["runtime_fingerprint"] != original["runtime_fingerprint"]
    (second / "protocol.py").write_text("PROTOCOL = 2\n")
    assert build_source_fingerprints(second)["protocol_fingerprint"] != original[
        "protocol_fingerprint"
    ]


def test_source_inventory_rejects_escape_and_hashes_repo_relative_files(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "source.py").write_text("VALUE = 1\n")
    files = source_inventory(root, logical_paths=("source.py",))
    assert files[0]["path"] == "source.py"
    assert inventory_fingerprint("scope", files) == inventory_fingerprint("scope", files)
    with pytest.raises(ValueError):
        source_inventory(root, logical_paths=("../escape.py",))


def test_cli_output_paths_are_confined_to_a_checkout_results_tree(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "src/mlx_atomistic/dft").mkdir(parents=True)
    (checkout / "results").mkdir()
    (checkout / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    expected = checkout / "results/run"
    assert results_output_path("results/run", repo_root=checkout) == expected
    assert results_output_path(expected, repo_root=checkout) == expected
    with pytest.raises(ValueError, match="below"):
        results_output_path(tmp_path / "outside", repo_root=checkout)
    with pytest.raises(ValueError, match="below"):
        results_output_path("results", repo_root=checkout)


def test_real_source_scopes_include_frozen_protocol_and_complete_dft_tree():
    sources = build_source_fingerprints()
    protocol_paths = {record["path"] for record in sources["protocol_inventory"]}
    runtime_paths = {record["path"] for record in sources["runtime_inventory"]}
    assert protocol_paths == set(contract.PROTOCOL_SOURCE_PATHS)
    expected_dft = {
        path.as_posix() for path in Path("src/mlx_atomistic/dft").rglob("*.py")
    }
    assert expected_dft <= runtime_paths
    assert set(contract.RUNTIME_SOURCE_PATHS) <= runtime_paths


def test_prepare_workload_is_path_independent_and_pins_complete_mesh(tmp_path):
    first_directory = tmp_path / "a"
    second_directory = tmp_path / "elsewhere"
    first_directory.mkdir()
    second_directory.mkdir()
    first_source = _gth_database(first_directory / "GTH_POTENTIALS")
    second_source = _gth_database(second_directory / "parameters.dat")
    second_source.write_text(second_source.read_text() + "\nC GTH-PBE-q4\n 2 2\n")
    first = prepare_workload(gth_source=first_source, out=tmp_path / "prepared-a")
    second = prepare_workload(gth_source=second_source, out=tmp_path / "prepared-b")
    first_manifest_path = Path(first["manifest"])
    second_manifest_path = Path(second["manifest"])
    assert first_manifest_path.read_bytes() == second_manifest_path.read_bytes()
    assert inspect_artifact(artifact=first["artifact"])["passed"] is True
    manifest, selected = load_workload(first_manifest_path, gth_source=second_source)
    assert manifest["system"]["atom_count"] == 8
    assert manifest["system"]["electron_count"] == 32
    assert manifest["system"]["occupied_band_count"] == 16
    assert manifest["physics"]["fft_shape"] == [56, 56, 56]
    assert manifest["solver"]["scf"]["orbital_tolerance"] == 1e-6
    assert manifest["solver"]["scf"]["adaptive_eigensolver_tolerance"] is True
    assert manifest["solver"]["scf"]["initial_eigensolver_tolerance"] == 1e-2
    assert manifest["solver"]["scf"]["eigensolver_tolerance_scale"] == 0.1
    assert manifest["solver"]["davidson"]["tolerance"] == 1e-6
    assert manifest["solver"]["davidson"]["max_iterations"] == 48
    assert len(manifest["physics"]["kpoints"]) == 216
    assert manifest["physics"]["representative_count"] == 108
    assert len(selected) == manifest["resources"][0]["byte_size"]
    serialized = json.dumps(manifest)
    assert str(first_source) not in serialized
    assert str(second_source) not in serialized
    assert set(manifest["resources"][0]) == {"role", "byte_size", "sha256"}
    assert [row["fft_shape"][0] for row in manifest["engineering_ladder"]] == [
        8,
        32,
        48,
        56,
    ]
    assert all(row["band_count"] == 16 for row in manifest["engineering_ladder"])
    assert all("oracle" in row for row in manifest["engineering_ladder"])
    for point in manifest["physics"]["kpoints"]:
        partner = manifest["physics"]["kpoints"][point["partner_index"]]
        assert partner["partner_index"] == point["index"]
        assert partner["owner_index"] == point["owner_index"]

    changed = _gth_database(tmp_path / "changed", selected_local="-6.0")
    with pytest.raises(ValueError, match="resource"):
        load_workload(first_manifest_path, gth_source=changed)
    duplicate = _gth_database(tmp_path / "duplicate")
    duplicate.write_text(duplicate.read_text() + "\n" + duplicate.read_text())
    with pytest.raises(ValueError, match="ambiguous"):
        prepare_workload(gth_source=duplicate, out=tmp_path / "ambiguous")


@pytest.mark.parametrize(
    "envelope_override",
    (
        {"artifact_kind": "dft-runtime-fixed-density"},
        {"artifact_schema": "workload.invalid"},
        {"identity": {"workload_fingerprint": "forged"}},
    ),
)
def test_workload_loader_rejects_mismatched_generation_envelope(
    tmp_path, envelope_override
):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    selected = contract.extract_selected_gth(source)
    manifest = contract.build_workload_manifest(selected)
    envelope = {
        "artifact_kind": "dft-runtime-workload",
        "artifact_schema": contract.WORKLOAD_SCHEMA,
        "identity": {"workload_fingerprint": manifest["workload_fingerprint"]},
        **envelope_override,
    }
    destination = tmp_path / f"workload-mismatch-{len(list(tmp_path.iterdir()))}"
    with AtomicGeneration(
        destination,
        envelope["artifact_kind"],
        envelope["artifact_schema"],
        identity=envelope["identity"],
    ) as generation:
        generation.write_json("manifest.json", manifest)
        generation.write_bytes("resources/Si-GTH-PBE-q4.gth", selected)
        generation.publish()
    with pytest.raises(ValueError, match="generation envelope"):
        load_workload(destination / "manifest.json", gth_source=source)
    with pytest.raises(ValueError, match="generation envelope"):
        inspect_artifact(artifact=destination)


def test_workload_loader_rejects_mismatched_published_resource(tmp_path):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    selected = contract.extract_selected_gth(source)
    manifest = contract.build_workload_manifest(selected)
    destination = tmp_path / "workload-wrong-resource"
    with AtomicGeneration(
        destination,
        "dft-runtime-workload",
        contract.WORKLOAD_SCHEMA,
        identity={"workload_fingerprint": manifest["workload_fingerprint"]},
    ) as generation:
        generation.write_json("manifest.json", manifest)
        generation.write_bytes("resources/Si-GTH-PBE-q4.gth", b"wrong\n")
        generation.publish()
    with pytest.raises(ValueError, match="published GTH resource"):
        load_workload(destination / "manifest.json", gth_source=source)
    with pytest.raises(ValueError, match="published GTH resource"):
        inspect_artifact(artifact=destination)


def test_workload_loader_rejects_standalone_manifest_without_completion(tmp_path):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    selected = contract.extract_selected_gth(source)
    manifest_path = tmp_path / "standalone.json"
    manifest_path.write_bytes(
        canonical_json_bytes(contract.build_workload_manifest(selected)) + b"\n"
    )
    with pytest.raises(ArtifactIntegrityError, match="completed generation"):
        load_workload(manifest_path, gth_source=source)


@pytest.mark.parametrize(
    ("field", "value"),
    (("schema_version", "workload.unknown"), ("target_id", "wrong-target")),
)
def test_workload_inspection_rejects_payload_schema_or_target_drift(
    tmp_path, field, value
):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    selected = contract.extract_selected_gth(source)
    payload = contract.build_workload_manifest(selected)
    payload[field] = value
    unsigned = {
        key: item for key, item in payload.items() if key != "workload_fingerprint"
    }
    payload["workload_fingerprint"] = contract.workload_fingerprint(unsigned)
    destination = tmp_path / f"drift-{field}"
    with AtomicGeneration(
        destination,
        "dft-runtime-workload",
        contract.WORKLOAD_SCHEMA,
        identity={"workload_fingerprint": payload["workload_fingerprint"]},
    ) as generation:
        generation.write_json("manifest.json", payload)
        generation.write_bytes("resources/Si-GTH-PBE-q4.gth", selected)
        generation.publish()
    with pytest.raises(ValueError, match="schema|target"):
        inspect_artifact(artifact=destination)
    with pytest.raises(ValueError, match="schema|target"):
        load_workload(destination / "manifest.json", gth_source=source)


def test_oracle_dense_and_compact_state_decoding_is_confined(tmp_path):
    oracle = _oracle_module()
    state = tmp_path / "state"
    state.mkdir()
    shape = (2, 2, 2)
    dense = np.zeros((2, *shape), dtype=np.complex64)
    indices = np.array([0, 3, 7], dtype=np.int32)
    compact = np.array(
        [[1.0 + 2.0j, 3.0 - 1.0j, 2.0], [0.5j, -2.0, 4.0 + 1.0j]],
        dtype=np.complex64,
    )
    dense.reshape(2, -1)[:, indices] = compact
    np.save(state / "dense.npy", dense, allow_pickle=False)
    np.save(state / "compact.npy", compact, allow_pickle=False)
    np.save(state / "indices.npy", indices, allow_pickle=False)
    np.save(state / "eigenvalues.npy", np.array([-1.0, 0.5]), allow_pickle=False)
    loaded_dense = oracle._load_coefficients(
        state,
        {"coefficient_file": "dense.npy"},
        shape,
    )
    loaded_compact = oracle._load_coefficients(
        state,
        {
            "compact_coefficient_file": "compact.npy",
            "compact_index_file": "indices.npy",
        },
        shape,
    )
    np.testing.assert_array_equal(loaded_dense, dense)
    np.testing.assert_array_equal(loaded_compact, dense)
    lane_coefficients, lane_eigenvalues = oracle._load_lane_state(
        state,
        {
            "compact_coefficient_file": "compact.npy",
            "compact_index_file": "indices.npy",
            "eigenvalue_file": "eigenvalues.npy",
        },
        shape,
    )
    np.testing.assert_array_equal(lane_coefficients, dense)
    np.testing.assert_array_equal(lane_eigenvalues, np.array([-1.0, 0.5]))

    with pytest.raises(ValueError, match="mixes full and compact"):
        oracle._load_coefficients(
            state,
            {
                "coefficient_file": "dense.npy",
                "compact_coefficient_file": "compact.npy",
                "compact_index_file": "indices.npy",
            },
            shape,
        )
    for name, invalid in (
        ("duplicate.npy", np.array([0, 0, 1], dtype=np.int32)),
        ("outside.npy", np.array([0, 1, 8], dtype=np.int32)),
        ("fractional.npy", np.array([0.0, 1.0, 2.5], dtype=np.float64)),
    ):
        np.save(state / name, invalid, allow_pickle=False)
        with pytest.raises(ValueError, match="integer dtype|bounds|duplicated"):
            oracle._load_coefficients(
                state,
                {
                    "compact_coefficient_file": "compact.npy",
                    "compact_index_file": name,
                },
                shape,
            )

    outside = tmp_path / "outside-state.npy"
    np.save(outside, dense, allow_pickle=False)
    for unsafe_name in ("../outside-state.npy", str(outside)):
        with pytest.raises(ValueError, match="confined|canonical"):
            oracle._load_coefficients(
                state,
                {"coefficient_file": unsafe_name},
                shape,
            )
    (state / "linked.npy").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes|symlink"):
        oracle._load_coefficients(
            state,
            {"coefficient_file": "linked.npy"},
            shape,
        )
    np.save(state / "wrong-band-count.npy", np.array([-1.0]), allow_pickle=False)
    with pytest.raises(ValueError, match="band count"):
        oracle._load_lane_state(
            state,
            {
                "coefficient_file": "dense.npy",
                "eigenvalue_file": "wrong-band-count.npy",
            },
            shape,
        )


def test_oracle_selected_state_contract_rejects_stale_or_weakened_inputs(tmp_path):
    oracle = _oracle_module()
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    prepared = prepare_workload(gth_source=source, out=tmp_path / "workload")
    manifest, _selected = load_workload(prepared["manifest"], gth_source=source)
    sources = build_source_fingerprints()
    generation = {
        "artifact_kind": "dft-runtime-full-scf",
        "artifact_schema_version": runtime_core.FULL_SCF_SCHEMA,
        "manifest_sha256": "a" * 64,
        "identity": {
            "workload_fingerprint": manifest["workload_fingerprint"],
            "protocol_fingerprint": sources["protocol_fingerprint"],
            "runtime_fingerprint": sources["runtime_fingerprint"],
        },
    }
    shape = list(manifest["physics"]["fft_shape"])
    lanes = [
        {
            "index": point["index"],
            "reduced_kpoint": point["reduced_coordinates"],
            "weight": float(point["weight"]["numerator"])
            / float(point["weight"]["denominator"]),
            "grid_shape": shape,
        }
        for point in manifest["physics"]["kpoints"]
    ]
    metadata = {
        "schema_version": "mlx-atomistic.periodic-scf-state.v1",
        "grid_shape": shape,
        "status": "converged",
        "converged": True,
        "total_energy_hartree": -10.0,
        "kpoint_count": len(lanes),
    }
    validated = oracle._validate_state_contract(
        generation=generation,
        metadata=metadata,
        lanes=lanes,
        manifest=manifest,
        current_sources=sources,
    )
    assert validated["converged_energy_state"] is True
    assert validated["generation_manifest_sha256"] == "a" * 64

    same_identity_new_bytes = {**generation, "manifest_sha256": "b" * 64}
    rebound = oracle._validate_state_contract(
        generation=same_identity_new_bytes,
        metadata=metadata,
        lanes=lanes,
        manifest=manifest,
        current_sources=sources,
    )
    assert rebound["generation_identity"] == validated["generation_identity"]
    assert rebound["generation_manifest_sha256"] != validated[
        "generation_manifest_sha256"
    ]

    stale = json.loads(json.dumps(generation))
    stale["identity"]["runtime_fingerprint"] = "s" * 64
    with pytest.raises(ValueError, match="runtime_fingerprint"):
        oracle._validate_state_contract(
            generation=stale,
            metadata=metadata,
            lanes=lanes,
            manifest=manifest,
            current_sources=sources,
        )
    wrong_schema = {**generation, "artifact_schema_version": "full-scf.invalid"}
    with pytest.raises(ValueError, match="full-SCF state generation"):
        oracle._validate_state_contract(
            generation=wrong_schema,
            metadata=metadata,
            lanes=lanes,
            manifest=manifest,
            current_sources=sources,
        )
    missing_schema = dict(generation)
    missing_schema.pop("artifact_schema_version")
    with pytest.raises(ValueError, match="full-SCF state generation"):
        oracle._validate_state_contract(
            generation=missing_schema,
            metadata=metadata,
            lanes=lanes,
            manifest=manifest,
            current_sources=sources,
        )
    wrong_workload = json.loads(json.dumps(generation))
    wrong_workload["identity"]["workload_fingerprint"] = "w" * 64
    with pytest.raises(ValueError, match="workload_fingerprint"):
        oracle._validate_state_contract(
            generation=wrong_workload,
            metadata=metadata,
            lanes=lanes,
            manifest=manifest,
            current_sources=sources,
        )
    wrong_grid = {**metadata, "grid_shape": [8, 8, 8]}
    with pytest.raises(ValueError, match="grid shape"):
        oracle._validate_state_contract(
            generation=generation,
            metadata=wrong_grid,
            lanes=lanes,
            manifest=manifest,
            current_sources=sources,
        )
    wrong_lanes = json.loads(json.dumps(lanes))
    wrong_lanes[0]["weight"] = 0.5
    with pytest.raises(ValueError, match="lane 0"):
        oracle._validate_state_contract(
            generation=generation,
            metadata=metadata,
            lanes=wrong_lanes,
            manifest=manifest,
            current_sources=sources,
        )
    missing_energy = {
        key: value
        for key, value in metadata.items()
        if key != "total_energy_hartree"
    }
    incomplete = oracle._validate_state_contract(
        generation=generation,
        metadata=missing_energy,
        lanes=lanes,
        manifest=manifest,
        current_sources=sources,
    )
    assert incomplete["converged_energy_state"] is False

    oracle.results_output_path = lambda output: Path(output)
    failed_output = tmp_path / "oracle-failure"
    with pytest.raises(SystemExit) as exited:
        oracle.main(
            [
                "--manifest",
                str(prepared["manifest"]),
                "--gth-source",
                str(source),
                "--state",
                str(tmp_path / "missing-state"),
                "--out",
                str(failed_output),
            ]
        )
    assert exited.value.code == 2
    assert inspect_generation(failed_output)["complete"] is True
    failure = json.loads((failed_output / "report.json").read_text())
    assert failure["admission"]["passed"] is False
    assert failure["admission"]["blockers"] == ["oracle_execution_failed"]
    assert inspect_artifact(
        artifact=failed_output,
        require_admitted=True,
        require_numerical=True,
    )["passed"] is False


def test_oracle_success_and_failed_gate_artifacts_are_inspectable(
    tmp_path, monkeypatch, capsys
):
    oracle = _oracle_module()
    sources = {
        "protocol_fingerprint": "p" * 64,
        "runtime_fingerprint": "r" * 64,
    }
    clean_git = {
        "revision": "c" * 40,
        "parent": "b" * 40,
        "dirty": False,
    }
    base_report = {
        "schema_version": runtime_core.ORACLE_SCHEMA,
        "workload_fingerprint": "w" * 64,
        "state_schema": "mlx-atomistic.periodic-scf-state.v1",
        "state_artifact": {
            "artifact_kind": "dft-runtime-full-scf",
            "artifact_schema_version": runtime_core.FULL_SCF_SCHEMA,
            "generation_manifest_sha256": "a" * 64,
            "generation_identity": {},
            "full_kpoint_contract": True,
            "converged_energy_state": True,
        },
        "electron_count": 32.0,
        "maximum_orthonormality_error": 0.0,
        "energy_by_term_hartree": {"total": -10.0},
        "gates": {"state_contract": True},
        "passed": True,
    }
    monkeypatch.setattr(oracle, "results_output_path", lambda output: Path(output))
    monkeypatch.setattr(oracle, "build_source_fingerprints", lambda: sources)
    monkeypatch.setattr(oracle, "collect_git_provenance", lambda: clean_git)
    monkeypatch.setattr(oracle, "evaluate_state", lambda **kwargs: dict(base_report))

    successful = tmp_path / "oracle-success"
    oracle.main(
        [
            "--manifest",
            "unused",
            "--gth-source",
            "unused",
            "--state",
            "unused",
            "--out",
            str(successful),
            "--require-gates",
        ]
    )
    capsys.readouterr()
    assert inspect_artifact(
        artifact=successful,
        require_admitted=True,
        require_numerical=True,
    )["passed"] is True

    failed_report = {**base_report, "gates": {"state_contract": False}, "passed": False}
    monkeypatch.setattr(oracle, "evaluate_state", lambda **kwargs: failed_report)
    failed = tmp_path / "oracle-failed-gate"
    with pytest.raises(SystemExit) as exited:
        oracle.main(
            [
                "--manifest",
                "unused",
                "--gth-source",
                "unused",
                "--state",
                "unused",
                "--out",
                str(failed),
                "--require-gates",
            ]
        )
    assert exited.value.code == 2
    capsys.readouterr()
    inspection = inspect_artifact(
        artifact=failed,
        require_admitted=True,
        require_numerical=True,
    )
    assert inspection["passed"] is False
    failed_payload = json.loads((failed / "report.json").read_text())
    assert failed_payload["admission"] == {
        "passed": False,
        "blockers": ["oracle_gate_failed"],
    }


def test_prepare_refuses_existing_destination_and_cli_inspect_is_read_only(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(runtime_cli, "results_output_path", lambda output: Path(output))
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    destination = tmp_path / "workload"
    main(["prepare", "--gth-source", str(source), "--out", str(destination), "--json"])
    prepared = json.loads(capsys.readouterr().out)
    before = {
        path.relative_to(destination): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in destination.rglob("*")
        if path.is_file()
    }
    inspected = inspect_artifact(artifact=destination)
    after = {
        path.relative_to(destination): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert prepared["status"] == "prepared"
    assert inspected["passed"] is True
    assert before == after
    with pytest.raises(FileExistsError):
        prepare_workload(gth_source=source, out=destination)


def test_cli_publishes_structured_failure_for_operational_setup_error(
    tmp_path, monkeypatch, capsys
):
    destination = tmp_path / "failed-full-scf"
    monkeypatch.setattr(runtime_cli, "results_output_path", lambda output: Path(output))

    def fail_setup(**kwargs):
        del kwargs
        raise OSError("host inspection unavailable")

    monkeypatch.setattr(runtime_cli, "run_full_scf", fail_setup)
    with pytest.raises(SystemExit) as exited:
        main(
            [
                "full-scf",
                "--manifest",
                "unused-manifest",
                "--gth-source",
                "unused-gth",
                "--out",
                str(destination),
                "--fresh",
                "--timeout-seconds",
                "5",
                "--json",
            ]
        )
    assert exited.value.code == 2
    assert inspect_generation(destination)["complete"] is True
    report = json.loads((destination / "report.json").read_text())
    assert report["kind"] == "command-failure"
    assert report["failure"]["error_type"] == "OSError"
    assert report["admission"]["passed"] is False
    assert "host inspection unavailable" in capsys.readouterr().err


@pytest.mark.parametrize("source", ["AC Power", "Battery Power"])
@pytest.mark.parametrize("mode_key", ["lowpowermode", "powermode"])
def test_host_admission_accepts_either_source_at_active_low_power(source, mode_key):
    calls = []
    provenance = collect_host_provenance(
        _runner(
            _host_outputs(
                source=source,
                ac_mode=1,
                battery_mode=1,
                mode_key=mode_key,
            ),
            calls,
        )
    )
    admission = host_admission(
        provenance,
        required_chip=TARGET_CHIP,
        require_low_power=True,
    )
    assert admission == {"admitted": True, "blockers": []}
    assert provenance["power_source"] == source
    assert provenance["power_mode_key"] == mode_key
    assert provenance["low_power_mode"] == 1
    assert provenance["inspection_policy"] == "read-only-getters-only"
    assert set(calls) == set(contract.READ_ONLY_HOST_COMMANDS)
    assert "SECRET" not in json.dumps(provenance)


def test_host_admission_selects_only_current_profile_and_fails_closed():
    calls = []
    battery = collect_host_provenance(
        _runner(_host_outputs(source="Battery Power", ac_mode=1, battery_mode=0), calls)
    )
    assert "active_power_mode_not_one" in host_admission(
        battery,
        required_chip=TARGET_CHIP,
        require_low_power=True,
    )["blockers"]
    wrong_chip = dict(battery)
    wrong_chip["chip"] = "Apple M4 Max"
    assert "chip_mismatch" in host_admission(
        wrong_chip,
        required_chip=TARGET_CHIP,
        require_low_power=False,
    )["blockers"]
    assert parse_current_power_source("Now drawing from 'AC Power'") == "AC Power"
    assert parse_power_profiles("AC Power:\n lowpowermode 1\n")["AC Power"] == {
        "lowpowermode": 1
    }
    assert parse_power_profiles("AC Power:\n powermode 1\n")["AC Power"] == {
        "powermode": 1
    }
    with pytest.raises(ValueError):
        parse_current_power_source("unknown")
    with pytest.raises(ValueError):
        parse_power_profiles("AC Power:\n lowpowermode nope\n")
    with pytest.raises(ValueError):
        parse_power_profiles("AC Power:\n powermode nope\n")


def test_host_admission_rejects_missing_or_conflicting_power_mode_keys():
    missing = {
        "chip": TARGET_CHIP,
        "power_source": "AC Power",
        "active_power_profile": {"sleep": 0},
        "blockers": [],
    }
    assert "active_power_mode_missing" in host_admission(
        missing,
        required_chip=TARGET_CHIP,
        require_low_power=True,
    )["blockers"]
    conflicting = {
        **missing,
        "active_power_profile": {"lowpowermode": 1, "powermode": 0},
    }
    assert "active_power_mode_conflict" in host_admission(
        conflicting,
        required_chip=TARGET_CHIP,
        require_low_power=True,
    )["blockers"]


@pytest.mark.parametrize(
    ("field", "forged"),
    (
        ("low_power_mode", None),
        ("low_power_mode", 0),
        ("low_power_mode", True),
        ("power_mode_key", None),
        ("power_mode_key", "lowpowermode"),
    ),
)
def test_host_power_mode_declared_normalization_mismatch_blocks_formal_admission(
    field, forged
):
    provenance = collect_host_provenance(
        _runner(_host_outputs(mode_key="powermode"), [])
    )
    provenance[field] = forged
    admission = host_admission(
        provenance,
        required_chip=TARGET_CHIP,
        require_low_power=True,
    )
    assert "active_power_mode_normalization_mismatch" in admission["blockers"]

    formal = _formal_admission(
        statuses={
            "numerical_status": "passed",
            "resume_integrity_status": "fresh-no-resume",
            "timing_admission_status": "admitted",
        },
        command_admission={"passed": True, "blockers": []},
        producer_git={"dirty": False},
        run_protocol={
            "warmups": 1,
            "samples": 5,
            "fresh": True,
            "resumed": False,
            "diagnostic": False,
        },
        report_kind="fixed-density",
        host_protocol=provenance,
    )
    assert formal["passed"] is False
    assert "formal_target_host_low_power_mismatch" in formal["blockers"]


def test_host_power_mode_uses_only_active_source_and_alias_is_not_identity():
    legacy_calls = []
    current_calls = []
    legacy_outputs = _host_outputs(mode_key="lowpowermode")
    current_outputs = _host_outputs(mode_key="powermode")
    current_outputs[("pmset", "-g", "custom")]["stdout"] = (
        "Battery Power:\n lowpowermode 0\n sleep 1\n"
        "AC Power:\n powermode 1\n sleep 0\n"
    )
    legacy = collect_host_provenance(_runner(legacy_outputs, legacy_calls))
    current = collect_host_provenance(_runner(current_outputs, current_calls))
    assert current["active_power_profile"] == {"powermode": 1, "sleep": 0}
    assert current["power_mode_key"] == "powermode"
    assert host_admission(
        current,
        required_chip=TARGET_CHIP,
        require_low_power=True,
    )["admitted"] is True
    assert runtime_core._host_protocol(legacy) == runtime_core._host_protocol(current)
    manifest = {
        "workload_fingerprint": "w" * 64,
        "resources": [{"role": "gth"}],
        "solver": {},
        "initialization": {},
        "measurement": {"synchronization": "synchronized"},
    }
    context = {
        "protocol_fingerprint": "p" * 64,
        "execution_contract": {
            "lock": {"sha256": "l" * 64},
            "environment": {
                "python_version": "3.13",
                "mlx_version": "test",
                "precision": "complex64/float32",
                **periodic_scf_module._eigensolve_provenance(),
                "selected_device": "Device(gpu, 0)",
            },
        },
    }
    legacy_protocol = runtime_core._comparison_protocol(manifest, context, legacy)
    current_protocol = runtime_core._comparison_protocol(manifest, context, current)
    assert legacy_protocol == current_protocol
    for field, value in periodic_scf_module._eigensolve_provenance().items():
        assert legacy_protocol[field] == value


@pytest.mark.parametrize(
    "custom",
    (
        "AC Power:\n lowpowermode 1\n powermode 0\n",
        "AC Power:\n powermode 0\n lowpowermode 1\n",
    ),
)
def test_host_power_mode_conflict_from_pmset_fails_closed(custom):
    outputs = _host_outputs()
    outputs[("pmset", "-g", "custom")]["stdout"] = custom
    provenance = collect_host_provenance(_runner(outputs, []))
    assert provenance["low_power_mode"] is None
    assert provenance["power_mode_key"] is None
    assert "active_power_mode_conflict" in provenance["blockers"]
    assert host_admission(
        provenance,
        required_chip=TARGET_CHIP,
        require_low_power=True,
    )["admitted"] is False


@pytest.mark.parametrize(
    ("custom", "expected_blocker"),
    (
        ("AC Power:\n sleep 0\n", "active_power_mode_missing"),
        ("AC Power:\n lowpowermode 0\n", "active_power_mode_not_one"),
        ("AC Power:\n powermode 0\n", "active_power_mode_not_one"),
        ("AC Power:\n powermode 2\n", "active_power_mode_not_one"),
    ),
)
def test_host_power_mode_missing_or_non_low_fails_closed(custom, expected_blocker):
    outputs = _host_outputs()
    outputs[("pmset", "-g", "custom")]["stdout"] = custom
    provenance = collect_host_provenance(_runner(outputs, []))
    assert expected_blocker in host_admission(
        provenance,
        required_chip=TARGET_CHIP,
        require_low_power=True,
    )["blockers"]


def test_host_power_mode_equal_dual_aliases_are_unambiguous():
    outputs = _host_outputs()
    outputs[("pmset", "-g", "custom")]["stdout"] = (
        "AC Power:\n lowpowermode 1\n powermode 1\n"
    )
    provenance = collect_host_provenance(_runner(outputs, []))
    assert provenance["power_mode_key"] == "lowpowermode+powermode"
    assert provenance["low_power_mode"] == 1
    assert host_admission(
        provenance,
        required_chip=TARGET_CHIP,
        require_low_power=True,
    )["admitted"] is True


def test_host_power_mode_malformed_current_key_is_unparsed():
    outputs = _host_outputs()
    outputs[("pmset", "-g", "custom")]["stdout"] = "AC Power:\n powermode nope\n"
    provenance = collect_host_provenance(_runner(outputs, []))
    assert provenance["low_power_mode"] is None
    assert "power_profiles_unparsed" in provenance["blockers"]
    assert host_admission(
        provenance,
        required_chip=TARGET_CHIP,
        require_low_power=True,
    )["admitted"] is False


def test_compare_allows_runtime_drift_but_rejects_power_source_mismatch(
    tmp_path, monkeypatch
):
    sources = build_source_fingerprints()
    protocol = str(sources["protocol_fingerprint"])
    optimized_runtime = str(sources["runtime_fingerprint"])
    baseline_revision = "b" * 40
    baseline_seal, seal_fingerprint = _fake_baseline_seal(
        tmp_path / "seal",
        protocol=protocol,
        runtime="b" * 64,
        revision=baseline_revision,
        parent_revision=runtime_core.BASELINE_EXPECTED_PARENT_REV,
    )
    monkeypatch.setattr(
        runtime_core,
        "collect_git_provenance",
        lambda repo_root=None: {"revision": "c" * 40, "dirty": False},
    )
    baseline = baseline_seal
    optimized = _fake_fixed_report(
        tmp_path / "optimized",
        runtime=optimized_runtime,
        protocol=protocol,
        revision="o" * 40,
        power_source="AC Power",
        elapsed=2.0,
        baseline_seal_fingerprint=seal_fingerprint,
    )
    compared = run_compare(
        baseline=baseline,
        optimized=optimized,
        baseline_seal=baseline_seal,
        out=tmp_path / "comparison",
        fresh=True,
        require_chip=TARGET_CHIP,
        require_low_power=True,
        require_matched_power_source=True,
        require_admitted=True,
        require_speedup=4.0,
    )
    assert compared["admission"]["passed"] is True
    assert compared["report_payload"]["summary"]["speedup"] == pytest.approx(4.0)
    assert compared["report_payload"]["baseline_identity"]["runtime_fingerprint"] != (
        compared["report_payload"]["optimized_identity"]["runtime_fingerprint"]
    )

    monkeypatch.setattr(
        runtime_core,
        "collect_git_provenance",
        lambda repo_root=None: {"revision": "c" * 40, "dirty": True},
    )
    dirty_producer = run_compare(
        baseline=baseline,
        optimized=optimized,
        baseline_seal=baseline_seal,
        out=tmp_path / "dirty-producer-comparison",
        fresh=True,
    )
    assert dirty_producer["admission"]["passed"] is False
    assert "comparison_producer_checkout_dirty" in dirty_producer["admission"][
        "blockers"
    ]
    monkeypatch.setattr(
        runtime_core,
        "collect_git_provenance",
        lambda repo_root=None: {"revision": "c" * 40, "dirty": False},
    )

    stale_runtime = _fake_fixed_report(
        tmp_path / "stale-runtime",
        runtime="s" * 64,
        protocol=protocol,
        revision="o" * 40,
        power_source="AC Power",
        elapsed=2.0,
        baseline_seal_fingerprint=seal_fingerprint,
    )
    stale_comparison = run_compare(
        baseline=baseline,
        optimized=stale_runtime,
        baseline_seal=baseline_seal,
        out=tmp_path / "stale-runtime-comparison",
        fresh=True,
    )
    assert stale_comparison["admission"]["passed"] is False
    assert "comparison_optimized_runtime_mismatch" in stale_comparison[
        "admission"
    ]["blockers"]

    diagnostic = _fake_fixed_report(
        tmp_path / "diagnostic",
        runtime=optimized_runtime,
        protocol=protocol,
        revision="o" * 40,
        power_source="AC Power",
        elapsed=2.0,
        baseline_seal_fingerprint=seal_fingerprint,
        timing_status="diagnostic",
        diagnostic=True,
    )
    inspected = inspect_artifact(artifact=diagnostic, require_admitted=True)
    assert inspected["passed"] is False
    assert "artifact_not_admitted" in inspected["blockers"]
    with pytest.raises(ValueError, match="integrity_only"):
        inspect_artifact(
            artifact=diagnostic,
            integrity_only=True,
            require_admitted=True,
        )
    diagnostic_comparison = run_compare(
        baseline=baseline,
        optimized=diagnostic,
        baseline_seal=baseline_seal,
        out=tmp_path / "diagnostic-comparison",
        fresh=True,
        require_admitted=True,
    )
    assert diagnostic_comparison["admission"]["passed"] is False
    assert "optimized_not_admitted" in diagnostic_comparison["admission"]["blockers"]

    nonnumerical = _fake_fixed_report(
        tmp_path / "nonnumerical",
        runtime=optimized_runtime,
        protocol=protocol,
        revision="o" * 40,
        power_source="AC Power",
        elapsed=2.0,
        baseline_seal_fingerprint=seal_fingerprint,
        numerical_status="blocked",
    )
    inspected = inspect_artifact(artifact=nonnumerical, require_admitted=True)
    assert inspected["passed"] is False
    assert "artifact_not_admitted" in inspected["blockers"]
    nonnumerical_comparison = run_compare(
        baseline=baseline,
        optimized=nonnumerical,
        baseline_seal=baseline_seal,
        out=tmp_path / "nonnumerical-comparison",
        fresh=True,
        require_admitted=True,
    )
    assert nonnumerical_comparison["admission"]["passed"] is False
    assert "optimized_not_admitted" in nonnumerical_comparison["admission"]["blockers"]

    dirty = _fake_fixed_report(
        tmp_path / "dirty",
        runtime=optimized_runtime,
        protocol=protocol,
        revision="o" * 40,
        power_source="AC Power",
        elapsed=2.0,
        baseline_seal_fingerprint=seal_fingerprint,
        dirty=True,
    )
    assert inspect_artifact(artifact=dirty, require_admitted=True)["passed"] is False
    dirty_comparison = run_compare(
        baseline=baseline,
        optimized=dirty,
        baseline_seal=baseline_seal,
        out=tmp_path / "dirty-comparison",
        fresh=True,
        require_admitted=True,
    )
    assert dirty_comparison["admission"]["passed"] is False
    assert "optimized_not_admitted" in dirty_comparison["admission"]["blockers"]

    battery = _fake_fixed_report(
        tmp_path / "battery",
        runtime=optimized_runtime,
        protocol=protocol,
        revision="o" * 40,
        power_source="Battery Power",
        elapsed=2.0,
        baseline_seal_fingerprint=seal_fingerprint,
    )
    mismatch = run_compare(
        baseline=baseline,
        optimized=battery,
        baseline_seal=baseline_seal,
        out=tmp_path / "mismatch",
        fresh=True,
        require_chip=TARGET_CHIP,
        require_low_power=True,
        require_matched_power_source=True,
        require_admitted=True,
    )
    assert mismatch["admission"]["passed"] is False
    assert "power_source_mismatch" in mismatch["admission"]["blockers"]


def test_fixed_density_seal_records_dual_sources_and_frozen_work_profile(
    tmp_path, monkeypatch
):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    prepared = prepare_workload(gth_source=source, out=tmp_path / "workload")
    host = {
        "model": "MacBook Pro",
        "model_identifier": "Mac17,7",
        "chip": TARGET_CHIP,
        "machine": "arm64",
        "macos": {"ProductVersion": "26.5.2"},
        "power_source": "Battery Power",
        "active_power_profile": {"lowpowermode": 1},
        "power_mode_key": "lowpowermode",
        "low_power_mode": 1,
        "blockers": [],
    }
    context = {
        "execution_contract": {
            "lock": {"sha256": "l" * 64},
            "environment": {
                "python_version": "3.13",
                "mlx_version": "test",
                "precision": "complex64/float32",
                **periodic_scf_module._eigensolve_provenance(),
                "selected_device": "Device(gpu, 0)",
            },
        },
        "execution_contract_fingerprint": "e" * 64,
        "protocol_inventory": [{"path": "protocol.py", "sha256": "p" * 64}],
        "protocol_fingerprint": "p" * 64,
        "runtime_inventory": [{"path": "runtime.py", "sha256": "r" * 64}],
        "runtime_fingerprint": "r" * 64,
        "git": {
            "revision": "b" * 40,
            "parent": runtime_core.BASELINE_EXPECTED_PARENT_REV,
            "dirty": False,
        },
    }
    dense_bytes = 16 * 56**3 * 8
    maximum_hpsi_width = 64
    hpsi_vector_equivalents = 100
    projector_payload_bytes = 56**3 * 8 * 5 * 8
    projector_elements_generated = (
        hpsi_vector_equivalents * projector_payload_bytes // 8
    )
    fft_workspace_bytes = 2 * maximum_hpsi_width * 56**3 * 8
    work = {
        "fft_submissions": 100,
        "fft_vector_equivalents": 100,
        "hpsi_vector_equivalents": hpsi_vector_equivalents,
        "davidson_hv_reused_vectors": 0,
        "representative_lane_solves": 0,
        "partner_reconstructions": 0,
        "projector_elements_generated": projector_elements_generated,
        "projector_elements_loaded": 2 * projector_elements_generated,
        "projector_traffic_elements": 3 * projector_elements_generated,
    }
    memory = {
        "coefficient_payload_bytes": dense_bytes,
        "projector_payload_bytes": projector_payload_bytes,
        "projector_traffic_bytes": 3 * projector_elements_generated * 8,
        "peak_temporary_bytes": (
            fft_workspace_bytes + maximum_hpsi_width * projector_payload_bytes
        ),
        "fft_workspace_bytes": fft_workspace_bytes,
        "process_high_water_bytes": 4_500_000_000,
        "unified_memory_high_water_bytes": 4_000_000_000,
    }
    calls = []

    def fake_sample(**kwargs):
        calls.append(kwargs)
        return (
            {
                "status": "ok",
                "numerical_passed": True,
                **periodic_scf_module._eigensolve_provenance(),
                "wall_elapsed_seconds": 2.0,
                "eigenvalues_hartree": [float(index) for index in range(16)],
                "observation": {
                    "work_counters": dict(work),
                    "memory": dict(memory),
                    "events": [
                        {
                            "event": "davidson_iteration",
                            "subspace_size": maximum_hpsi_width,
                        }
                    ],
                },
            },
            object(),
        )

    monkeypatch.setattr(runtime_core, "collect_host_provenance", lambda: host)
    monkeypatch.setattr(runtime_core, "build_execution_context", lambda **kwargs: context)
    monkeypatch.setattr(runtime_core, "_fixed_density_sample", fake_sample)

    def passing_diff_audit(repo_root=None):
        del repo_root
        return {
            "base_revision": PRE_ARCHITECTURE_REV,
            "baseline_revision": context["git"]["revision"],
            "baseline_parent_revision": context["git"]["parent"],
            "checks": {
                "git_commands_succeeded": True,
                "pre_architecture_revision_is_ancestor": True,
                "baseline_history_has_no_merge_commits": True,
                "baseline_parent_is_expected_prebaseline_revision": True,
                "baseline_revision_is_distinct": True,
                "diff_is_nonempty": True,
                "diff_paths_are_allowed": True,
                "diff_records_are_parseable_regular_files": True,
            },
            "allowed_paths": sorted(runtime_core.BASELINE_ALLOWED_DIFF_PATHS),
            "changed_files": [
                {
                    "status": "M",
                    "path": sorted(runtime_core.BASELINE_ALLOWED_DIFF_PATHS)[0],
                    "byte_size": 1,
                    "sha256": "d" * 64,
                }
            ],
            "patch_sha256": "c" * 64,
            "passed": True,
        }

    monkeypatch.setattr(runtime_core, "_baseline_diff_audit", passing_diff_audit)
    result = run_fixed_density(
        manifest_path=prepared["manifest"],
        gth_source=source,
        out=tmp_path / "baseline",
        warmups=1,
        samples=5,
        fresh=True,
        diagnostic=False,
        require_clean=True,
        require_chip=TARGET_CHIP,
        require_low_power=True,
        require_numerical=True,
        seal=True,
    )
    assert len(calls) == 6
    assert result["admission"]["passed"] is True
    for sample in [
        *result["report_payload"]["warmup_results"],
        *result["report_payload"]["samples"],
    ]:
        for field, value in periodic_scf_module._eigensolve_provenance().items():
            assert sample[field] == value
    seal = json.loads((tmp_path / "baseline/seal.json").read_text())
    assert seal["protocol_fingerprint"] == "p" * 64
    assert seal["baseline_runtime_fingerprint"] == "r" * 64
    assert seal["base_rev"] == PRE_ARCHITECTURE_REV
    assert seal["parent_rev"] == runtime_core.BASELINE_EXPECTED_PARENT_REV
    assert seal["host_protocol"]["power_source"] == "Battery Power"
    assert seal["baseline_structure_audit"]["passed"] is True
    assert seal["baseline_diff_audit"]["passed"] is True
    assert seal["selected_gth_resource"]["role"] == "si_gth_pbe_q4"
    assert seal["fixed_density_eigenvalues_hartree"] == [
        float(index) for index in range(16)
    ]
    assert inspect_generation(tmp_path / "baseline")["complete"] is True

    calls.clear()

    def missing_late_memory_sample(**kwargs):
        sample, state = fake_sample(**kwargs)
        if len(calls) == 6:
            sample["observation"]["memory"]["unified_memory_high_water_bytes"] = None
        return sample, state

    monkeypatch.setattr(
        runtime_core,
        "_fixed_density_sample",
        missing_late_memory_sample,
    )
    incomplete_memory = run_fixed_density(
        manifest_path=prepared["manifest"],
        gth_source=source,
        out=tmp_path / "incomplete-memory",
        warmups=1,
        samples=5,
        fresh=True,
        diagnostic=False,
        require_clean=True,
        require_chip=TARGET_CHIP,
        require_low_power=True,
        require_numerical=True,
        seal=True,
    )
    assert len(calls) == 6
    assert incomplete_memory["admission"]["passed"] is False
    assert "baseline_structure_audit_failed" in incomplete_memory["admission"][
        "blockers"
    ]
    assert not (tmp_path / "incomplete-memory/seal.json").exists()
    monkeypatch.setattr(runtime_core, "_fixed_density_sample", fake_sample)
    calls.clear()

    def mismatched_provenance_sample(**kwargs):
        sample, state = fake_sample(**kwargs)
        if len(calls) == 6:
            sample["projected_eigensolve_backend"] = "forged"
        return sample, state

    monkeypatch.setattr(
        runtime_core,
        "_fixed_density_sample",
        mismatched_provenance_sample,
    )
    mismatched_provenance = run_fixed_density(
        manifest_path=prepared["manifest"],
        gth_source=source,
        out=tmp_path / "mismatched-provenance",
        warmups=1,
        samples=5,
        fresh=True,
        diagnostic=False,
        require_clean=True,
        require_chip=TARGET_CHIP,
        require_low_power=True,
        require_numerical=True,
        seal=True,
    )
    assert len(calls) == 6
    assert mismatched_provenance["admission"]["passed"] is False
    assert "eigensolve_provenance_mismatch" in mismatched_provenance["admission"][
        "blockers"
    ]
    assert not (tmp_path / "mismatched-provenance/seal.json").exists()
    monkeypatch.setattr(runtime_core, "_fixed_density_sample", fake_sample)
    calls.clear()

    context["git"]["parent"] = "a" * 40
    wrong_parent = run_fixed_density(
        manifest_path=prepared["manifest"],
        gth_source=source,
        out=tmp_path / "wrong-parent",
        warmups=1,
        samples=5,
        fresh=True,
        diagnostic=False,
        require_clean=True,
        require_chip=TARGET_CHIP,
        require_low_power=True,
        require_numerical=True,
        seal=True,
    )
    assert wrong_parent["admission"]["passed"] is False
    assert "baseline_diff_audit_failed" in wrong_parent["admission"]["blockers"]
    assert not (tmp_path / "wrong-parent/seal.json").exists()
    context["git"]["parent"] = runtime_core.BASELINE_EXPECTED_PARENT_REV

    def wrong_base_audit(repo_root=None):
        audit = passing_diff_audit(repo_root)
        audit["base_revision"] = "f" * 40
        return audit

    monkeypatch.setattr(runtime_core, "_baseline_diff_audit", wrong_base_audit)
    rejected = run_fixed_density(
        manifest_path=prepared["manifest"],
        gth_source=source,
        out=tmp_path / "wrong-base",
        warmups=1,
        samples=5,
        fresh=True,
        diagnostic=False,
        require_clean=True,
        require_chip=TARGET_CHIP,
        require_low_power=True,
        require_numerical=True,
        seal=True,
    )
    assert rejected["admission"]["passed"] is False
    assert "baseline_diff_audit_failed" in rejected["admission"]["blockers"]
    assert not (tmp_path / "wrong-base/seal.json").exists()

    context["git"]["parent"] = runtime_core.BASELINE_EXPECTED_PARENT_REV

    def failed_sample(**kwargs):
        del kwargs
        return (
            {
                "status": "blocked",
                "numerical_passed": False,
                **periodic_scf_module._eigensolve_provenance(),
                "wall_elapsed_seconds": 2.0,
                "eigenvalues_hartree": [float(index) for index in range(16)],
                "observation": {
                    "work_counters": dict(work),
                    "memory": dict(memory),
                    "events": [
                        {
                            "event": "davidson_iteration",
                            "subspace_size": maximum_hpsi_width,
                        }
                    ],
                },
            },
            object(),
        )

    monkeypatch.setattr(runtime_core, "_fixed_density_sample", failed_sample)
    nonconverged = run_fixed_density(
        manifest_path=prepared["manifest"],
        gth_source=source,
        out=tmp_path / "nonconverged",
        warmups=0,
        samples=1,
        fresh=True,
        diagnostic=False,
        require_numerical=False,
    )
    assert nonconverged["admission"]["passed"] is False
    assert "numerical_result_failed" in nonconverged["admission"]["blockers"]


@pytest.mark.parametrize(("failed_call", "expected_calls"), ((0, 1), (1, 2)))
def test_fixed_density_stops_after_first_numerical_failure(
    tmp_path, monkeypatch, failed_call, expected_calls
):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    prepared = prepare_workload(gth_source=source, out=tmp_path / "workload")
    host = {
        "chip": TARGET_CHIP,
        "power_source": "AC Power",
        "active_power_profile": {"powermode": 1},
        "power_mode_key": "powermode",
        "low_power_mode": 1,
    }
    context = {
        "execution_contract": {
            "lock": {"sha256": "l" * 64},
            "environment": {
                "python_version": "3.13",
                "mlx_version": "test",
                "precision": "complex64/float32",
                **periodic_scf_module._eigensolve_provenance(),
                "selected_device": "Device(gpu, 0)",
            },
        },
        "execution_contract_fingerprint": "e" * 64,
        "protocol_inventory": [],
        "protocol_fingerprint": "p" * 64,
        "runtime_inventory": [],
        "runtime_fingerprint": "r" * 64,
        "git": {"revision": "r" * 40, "parent": "p" * 40, "dirty": False},
    }
    calls = []
    state_refs = []

    class SampleState:
        pass

    def sampled(**kwargs):
        del kwargs
        if state_refs:
            assert state_refs[-1]() is None
        call_index = len(calls)
        calls.append(call_index)
        passed = call_index != failed_call
        state = SampleState()
        state_refs.append(weakref.ref(state))
        return (
            {
                "status": "ok" if passed else "blocked",
                "numerical_passed": passed,
                **periodic_scf_module._eigensolve_provenance(),
                "wall_elapsed_seconds": 1.0,
                "eigenvalues_hartree": [float(index) for index in range(16)],
                "observation": {"work_counters": {}, "memory": {}},
            },
            state,
        )

    monkeypatch.setattr(runtime_core, "collect_host_provenance", lambda: host)
    monkeypatch.setattr(runtime_core, "build_execution_context", lambda **kwargs: context)
    monkeypatch.setattr(runtime_core, "_fixed_density_sample", sampled)
    result = run_fixed_density(
        manifest_path=prepared["manifest"],
        gth_source=source,
        out=tmp_path / f"failed-call-{failed_call}",
        warmups=1,
        samples=5,
        fresh=True,
        diagnostic=True,
        require_numerical=True,
    )

    assert calls == list(range(expected_calls))
    assert all(reference() is None for reference in state_refs)
    assert result["admission"]["passed"] is False
    assert "numerical_result_failed" in result["admission"]["blockers"]


def test_fixed_density_compare_seal_blocks_wrong_eigenvalues(tmp_path, monkeypatch):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    prepared = prepare_workload(gth_source=source, out=tmp_path / "workload")
    manifest, _selected = load_workload(prepared["manifest"], gth_source=source)
    host = {
        "model": "MacBook Pro",
        "model_identifier": "Mac17,7",
        "chip": TARGET_CHIP,
        "machine": "arm64",
        "macos": {"ProductVersion": "26.5.2"},
        "power_source": "AC Power",
        "active_power_profile": {"lowpowermode": 1},
        "power_mode_key": "lowpowermode",
        "low_power_mode": 1,
    }
    context = {
        "execution_contract": {
            "lock": {"sha256": "l" * 64},
            "environment": {
                "python_version": "3.13",
                "mlx_version": "test",
                "precision": "complex64/float32",
                **periodic_scf_module._eigensolve_provenance(),
                "selected_device": "Device(gpu, 0)",
            },
        },
        "execution_contract_fingerprint": "e" * 64,
        "protocol_inventory": [],
        "protocol_fingerprint": "p" * 64,
        "runtime_inventory": [],
        "runtime_fingerprint": "o" * 64,
        "git": {"revision": "o" * 40, "parent": "b" * 40, "dirty": True},
    }
    baseline_eigenvalues = [float(index) for index in range(16)]
    baseline_observation = {
        "work_counters": {"projector_traffic_elements": 1000},
        "memory": {
            "coefficient_payload_bytes": 16 * 56**3 * 8,
            "projector_payload_bytes": 4096,
        },
    }
    seal_path, _seal_fingerprint = _fake_baseline_seal(
        tmp_path / "seal",
        protocol="p" * 64,
        runtime="b" * 64,
        revision="b" * 40,
        extra={
            "workload_fingerprint": manifest["workload_fingerprint"],
            "selected_gth_resource": manifest["resources"][0],
            "host": host,
            "comparison_protocol": runtime_core._comparison_protocol(
                manifest,
                context,
                host,
            ),
            "fixed_density_eigenvalues_hartree": baseline_eigenvalues,
            "representative_observation": baseline_observation,
            "median_elapsed_seconds": 8.0,
        },
    )
    current_eigenvalues = list(baseline_eigenvalues)
    current_eigenvalues[-1] += 1e-3

    def wrong_sample(**kwargs):
        del kwargs
        return (
            {
                "status": "ok",
                "numerical_passed": True,
                **periodic_scf_module._eigensolve_provenance(),
                "wall_elapsed_seconds": 2.0,
                "eigenvalues_hartree": current_eigenvalues,
                "observation": baseline_observation,
            },
            object(),
        )

    monkeypatch.setattr(runtime_core, "collect_host_provenance", lambda: host)
    monkeypatch.setattr(runtime_core, "build_execution_context", lambda **kwargs: context)
    monkeypatch.setattr(runtime_core, "_fixed_density_sample", wrong_sample)
    result = run_fixed_density(
        manifest_path=prepared["manifest"],
        gth_source=source,
        out=tmp_path / "wrong-eigenvalues",
        warmups=0,
        samples=1,
        fresh=True,
        diagnostic=True,
        require_numerical=True,
        compare_seal=seal_path,
    )
    assert result["admission"]["passed"] is False
    assert "fixed_density_eigenvalue_parity_failed" in result["admission"]["blockers"]
    assert result["report_payload"]["summary"]["metrics_against_seal"][
        "eigenvalue_max_abs_error_hartree"
    ] == pytest.approx(1e-3)
    assert inspect_generation(tmp_path / "wrong-eigenvalues")["complete"] is True


def test_runtime_observer_reconciles_exclusive_phases_and_counters():
    current = [0.0]
    delivered = []
    synchronizations = []

    def clock():
        return current[0]

    observer = RuntimeObserver(
        callback=delivered.append,
        synchronize=lambda: synchronizations.append(current[0]),
        clock=clock,
    )
    observer.emit("setup", status="started")
    with observer.phase("setup"):
        current[0] += 1.0
        with observer.phase("hpsi"):
            current[0] += 2.0
        current[0] += 1.0
    observer.add_work("hpsi_calls", 1)
    observer.record_memory("persistent_coefficient_bytes", 128)
    observer.record_peak_memory("fft_workspace_bytes", 64)
    observer.record_peak_memory("fft_workspace_bytes", 32)
    observer.record_peak_memory("fft_workspace_bytes", 256)
    current[0] += 1.0
    snapshot = observer.snapshot()
    phases = snapshot["phase_seconds"]
    assert phases["setup"] == pytest.approx(2.0)
    assert phases["hpsi"] == pytest.approx(2.0)
    assert sum(phases.values()) == pytest.approx(snapshot["total_elapsed_seconds"])
    assert snapshot["work_counters"]["hpsi_calls"] == 1
    assert snapshot["memory"]["fft_workspace_bytes"] == 256
    assert delivered[0]["sequence"] == 1
    assert len(synchronizations) == 4


def test_runtime_observer_can_measure_materialized_phase_without_device_barriers():
    current = [0.0]
    synchronizations = []
    observer = RuntimeObserver(
        synchronize=lambda: synchronizations.append(current[0]),
        clock=lambda: current[0],
    )

    with observer.phase("orthogonalization", synchronize=False):
        current[0] += 2.0

    assert synchronizations == []
    assert observer.snapshot()["phase_seconds"]["orthogonalization"] == pytest.approx(2.0)


def test_logical_hpsi_memory_scales_with_observed_vector_width():
    grid_count = 56**3
    vector_count = 64
    projector_payload_bytes = grid_count * 8 * 5 * 8
    projector_elements = vector_count * projector_payload_bytes // 8
    fft_workspace, peak_temporary = periodic_scf_module._logical_hpsi_memory(
        vector_count=vector_count,
        grid_count=grid_count,
        projector_elements=projector_elements,
    )
    assert fft_workspace == 179_830_784
    assert peak_temporary == 3_776_446_464


def test_projected_eigh_uses_complex128_lapack_and_returns_runtime_precision(
    monkeypatch,
):
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
    unitary, _ = np.linalg.qr(raw)
    clustered = np.concatenate(
        [np.linspace(-2.0, -1.0, 16), 0.25 + np.arange(48) * 1e-8]
    )
    matrix = ((unitary * clustered[None, :]) @ unitary.conj().T).astype(
        np.complex64
    )
    observed = {}
    lapack_eigh = np.linalg.eigh

    def capture_dtype(projected):
        observed["dtype"] = projected.dtype
        observed["shape"] = projected.shape
        return lapack_eigh(projected)

    converted = []

    def capture_mlx_array(value):
        array = np.asarray(value)
        converted.append(array.dtype)
        return array

    monkeypatch.setattr(periodic_scf_module.np.linalg, "eigh", capture_dtype)
    monkeypatch.setattr(periodic_scf_module.mx, "array", capture_mlx_array)
    values_mx, vectors_mx = periodic_scf_module._projected_eigh(matrix)
    values = np.asarray(values_mx)
    vectors = np.asarray(vectors_mx)
    residual = matrix @ vectors - vectors * values[None, :]
    overlap = vectors.conj().T @ vectors

    assert observed == {"dtype": np.dtype(np.complex128), "shape": (64, 64)}
    assert converted == [np.dtype(np.float32), np.dtype(np.complex64)]
    assert values_mx.dtype == np.float32
    assert vectors_mx.dtype == np.complex64
    assert np.max(np.abs(residual)) < 2e-6
    assert np.max(np.abs(overlap - np.eye(64))) < 2e-6


def test_projected_eigh_batch_uses_one_complex128_lapack_bridge(monkeypatch):
    matrices = (
        np.diag(np.array([-2.0, -1.0, 0.5], dtype=np.float32)).astype(np.complex64),
        np.array(
            [[-1.5, 0.2j, 0.0], [-0.2j, -0.25, 0.1], [0.0, 0.1, 0.75]],
            dtype=np.complex64,
        ),
    )
    observed_shapes = []
    lapack_eigh = np.linalg.eigh

    def capture_batch(projected):
        observed_shapes.append((projected.dtype, projected.shape))
        return lapack_eigh(projected)

    monkeypatch.setattr(periodic_scf_module.np.linalg, "eigh", capture_batch)

    solved = periodic_scf_module._projected_eigh_batch(matrices)

    assert observed_shapes == [(np.dtype(np.complex128), (2, 3, 3))]
    for matrix, (values, vectors) in zip(matrices, solved, strict=True):
        values_np = np.asarray(values)
        vectors_np = np.asarray(vectors)
        residual = matrix @ vectors_np - vectors_np * values_np[None, :]
        assert values.dtype == mx.float32
        assert vectors.dtype == mx.complex64
        assert np.max(np.abs(residual)) < 1e-6


@pytest.mark.parametrize(
    ("matrix", "message"),
    (
        (np.ones((2, 3), dtype=np.complex64), "non-empty and square"),
        (
            np.array([[1.0, np.nan], [np.nan, 2.0]], dtype=np.complex64),
            "must be finite",
        ),
    ),
)
def test_projected_eigh_rejects_malformed_or_nonfinite_matrix(matrix, message):
    with pytest.raises(ValueError, match=message):
        periodic_scf_module._projected_eigh(matrix)


@pytest.mark.gpu
def test_periodic_davidson_observer_counts_single_hpsi_hook_without_numerical_drift():
    grid = RealSpaceGrid((6, 6, 6), (6.0, 6.0, 6.0))
    basis = PlaneWaveBasis.from_reduced_kpoint(grid, 3.0, (0.25, 0.0, 0.0))
    config = PeriodicDavidsonConfig(
        max_iterations=12,
        tolerance=2e-5,
        max_subspace_size=12,
    )
    plain = solve_periodic_eigenproblem(
        PeriodicKohnShamOperator(basis, mx.full(grid.shape, 0.7)),
        n_bands=3,
        config=config,
    )
    events = []
    observer = RuntimeObserver(callback=events.append, synchronize=mx.synchronize)
    observed = solve_periodic_eigenproblem(
        PeriodicKohnShamOperator(basis, mx.full(grid.shape, 0.7), observer=observer),
        n_bands=3,
        config=config,
        observer=observer,
    )
    snapshot = observer.snapshot()
    work = snapshot["work_counters"]
    assert observed.iterations == 1
    assert work["hpsi_calls"] == 1
    assert work["hpsi_vector_equivalents"] == 3
    assert work["davidson_hv_new_vectors"] == 3
    assert work["davidson_hv_reused_vectors"] == 0
    assert work["projected_old_old_rebuilds"] == 0
    assert work["fft_vector_equivalents"] == 2 * work["hpsi_vector_equivalents"]
    assert len([event for event in events if event["event"] == "davidson_iteration"]) == (
        observed.iterations
    )
    observed_metadata = observed.to_dict()
    for field, value in periodic_scf_module._eigensolve_provenance().items():
        assert observed_metadata[field] == value
    np.testing.assert_allclose(np.asarray(observed.eigenvalues), np.asarray(plain.eigenvalues))


@pytest.mark.slow
def test_supervised_timeout_kills_worker_group_and_preserves_progress(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    result = supervise_full_scf_worker(
        manifest_path="unused",
        gth_source="unused",
        state_root=state,
        timeout_seconds=0.5,
        worker=_sleeping_process_group_worker,
    )
    assert result["status"] == "timeout"
    assert result["worker_alive_after_cleanup"] is False
    assert result["progress_prefix"] == [
        {"event": "fake-worker-started"},
        {
            "event": "full_scf_timeout",
            "status": "failed",
            "timeout_seconds": 0.5,
        },
    ]
    assert (state / "child.pid").is_file()
    heartbeat = state / "heartbeat"
    assert heartbeat.is_file()
    stopped_value = heartbeat.read_text()
    time.sleep(0.2)
    assert heartbeat.read_text() == stopped_value


@pytest.mark.slow
def test_supervised_timeout_before_ready_still_kills_process_group(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    result = supervise_full_scf_worker(
        manifest_path="unused",
        gth_source="unused",
        state_root=state,
        timeout_seconds=1.5,
        worker=_delayed_ready_process_group_worker,
    )
    assert result["status"] == "timeout"
    assert result["worker_alive_after_cleanup"] is False
    assert result["process_group_isolated"] is True
    assert result["progress_prefix"][0] == {"event": "child-spawned-before-ready"}
    heartbeat = state / "heartbeat"
    assert heartbeat.is_file()
    stopped_value = heartbeat.read_text()
    time.sleep(0.2)
    assert heartbeat.read_text() == stopped_value


@pytest.mark.slow
def test_supervised_result_requires_clean_worker_exit(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    result = supervise_full_scf_worker(
        manifest_path="unused",
        gth_source="unused",
        state_root=state,
        timeout_seconds=2.0,
        worker=_result_then_crash_worker,
    )
    assert result["result"] == {"numerical_passed": True}
    assert result["worker_exitcode"] == 7
    assert result["status"] == "failed"


@pytest.mark.slow
def test_supervised_progress_callback_failure_still_kills_process_group(tmp_path):
    state = tmp_path / "state"
    state.mkdir()

    def reject_progress(event):
        raise RuntimeError(f"callback rejected {event['event']}")

    with pytest.raises(RuntimeError, match="callback rejected fake-worker-started"):
        supervise_full_scf_worker(
            manifest_path="unused",
            gth_source="unused",
            state_root=state,
            timeout_seconds=5.0,
            progress=reject_progress,
            worker=_sleeping_process_group_worker,
        )
    child_pid = int((state / "child.pid").read_text())
    deadline = time.monotonic() + 1.0
    child_alive = True
    while child_alive and time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            child_alive = False
        else:
            time.sleep(0.01)
    assert child_alive is False
    heartbeat = state / "heartbeat"
    if heartbeat.is_file():
        stopped_value = heartbeat.read_text()
        time.sleep(0.2)
        assert heartbeat.read_text() == stopped_value


def test_full_scf_publication_deadline_is_part_of_formal_admission(
    tmp_path, monkeypatch
):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    prepared = prepare_workload(gth_source=source, out=tmp_path / "workload")
    host = {
        "chip": TARGET_CHIP,
        "power_source": "AC Power",
        "active_power_profile": {"lowpowermode": 1},
        "power_mode_key": "lowpowermode",
        "low_power_mode": 1,
        "macos": {"ProductVersion": "26.5.2"},
    }
    context = {
        "execution_contract_fingerprint": "e" * 64,
        "protocol_inventory": [],
        "protocol_fingerprint": "p" * 64,
        "runtime_inventory": [],
        "runtime_fingerprint": "r" * 64,
        "git": {"revision": "r" * 40, "parent": "p" * 40, "dirty": False},
    }

    def fake_supervisor(**kwargs):
        state_root = Path(kwargs["state_root"])
        (state_root / "final-state").mkdir()
        (state_root / "final-state/state.bin").write_bytes(b"state")
        return {
            "status": "ok",
            "timed_out": False,
            "worker_exitcode": 0,
            "progress_prefix": [],
            "result": {"numerical_passed": True},
            "error": None,
            "worker_alive_after_cleanup": False,
        }

    monkeypatch.setattr(runtime_core, "collect_host_provenance", lambda: host)
    monkeypatch.setattr(runtime_core, "build_execution_context", lambda **kwargs: context)
    monkeypatch.setattr(runtime_core, "supervise_full_scf_worker", fake_supervisor)
    original_rename = artifact_identity._rename_noreplace

    def slow_rename(*args, **kwargs):
        time.sleep(0.08)
        original_rename(*args, **kwargs)

    monkeypatch.setattr(artifact_identity, "_rename_noreplace", slow_rename)
    blocked = run_full_scf(
        manifest_path=prepared["manifest"],
        gth_source=source,
        out=tmp_path / "slow-publication",
        fresh=True,
        timeout_seconds=0.05,
        require_numerical=True,
        require_success=True,
    )
    assert blocked["admission"]["passed"] is False
    assert "elapsed_time_gate_failed" in blocked["admission"]["blockers"]
    assert blocked["report_payload"]["summary"]["elapsed_seconds"] > 0.05
    assert inspect_generation(tmp_path / "slow-publication")["complete"] is True
    assert inspect_generation(
        tmp_path / "slow-publication.publication"
    )["complete"] is True
    assert inspect_artifact(
        artifact=tmp_path / "slow-publication",
        require_admitted=True,
    )["passed"] is False

    monkeypatch.setattr(artifact_identity, "_rename_noreplace", original_rename)
    admitted = run_full_scf(
        manifest_path=prepared["manifest"],
        gth_source=source,
        out=tmp_path / "fast-publication",
        fresh=True,
        timeout_seconds=5.0,
        require_numerical=True,
        require_success=True,
    )
    assert admitted["formal_admission"]["passed"] is True
    science_report = json.loads((tmp_path / "fast-publication/report.json").read_text())
    assert science_report["formal_admission"]["passed"] is False
    assert admitted["report_payload"]["published_artifact"][
        "artifact_manifest_sha256"
    ] == inspect_generation(tmp_path / "fast-publication")["manifest_sha256"]
    inconsistent = json.loads(json.dumps(admitted["report_payload"]))
    inconsistent["published_artifact"]["elapsed_seconds"] = 50.0
    inconsistent["published_artifact"]["maximum_elapsed_seconds"] = 5.0
    inconsistent["summary"]["elapsed_seconds"] = 50.0
    with pytest.raises(ValueError, match="does not bind"):
        runtime_core._validate_full_scf_publication_binding(
            artifact_root=tmp_path / "fast-publication",
            artifact_generation=inspect_generation(tmp_path / "fast-publication"),
            artifact_report=science_report,
            attestation_root=tmp_path / "fast-publication.publication",
            attestation_generation=inspect_generation(
                tmp_path / "fast-publication.publication"
            ),
            attestation_report=inconsistent,
        )
    failed_subject = json.loads(json.dumps(science_report))
    failed_subject["summary"]["numerical_passed"] = False
    failed_subject["summary"]["success"] = False
    failed_subject["statuses"]["numerical_status"] = "blocked"
    failed_subject["admission"] = {
        "passed": False,
        "blockers": ["full_scf_worker_failed", "numerical_result_failed"],
    }
    laundered = json.loads(json.dumps(admitted["report_payload"]))
    laundered["summary"]["numerical_passed"] = False
    laundered["summary"]["success"] = False
    with pytest.raises(ValueError, match="does not bind"):
        runtime_core._validate_full_scf_publication_binding(
            artifact_root=tmp_path / "fast-publication",
            artifact_generation=inspect_generation(tmp_path / "fast-publication"),
            artifact_report=failed_subject,
            attestation_root=tmp_path / "fast-publication.publication",
            attestation_generation=inspect_generation(
                tmp_path / "fast-publication.publication"
            ),
            attestation_report=laundered,
        )
    dirty_subject = json.loads(json.dumps(science_report))
    dirty_subject["context"]["git"]["dirty"] = True
    with pytest.raises(ValueError, match="does not bind"):
        runtime_core._validate_full_scf_publication_binding(
            artifact_root=tmp_path / "fast-publication",
            artifact_generation=inspect_generation(tmp_path / "fast-publication"),
            artifact_report=dirty_subject,
            attestation_root=tmp_path / "fast-publication.publication",
            attestation_generation=inspect_generation(
                tmp_path / "fast-publication.publication"
            ),
            attestation_report=admitted["report_payload"],
        )
    assert inspect_artifact(
        artifact=tmp_path / "fast-publication",
        require_admitted=True,
        require_numerical=True,
        max_elapsed_seconds=5.0,
    )["passed"] is True
    assert inspect_artifact(
        artifact=tmp_path / "fast-publication.publication",
        require_admitted=True,
        require_numerical=True,
        max_elapsed_seconds=5.0,
    )["passed"] is True
    moved = tmp_path / "fast-publication-moved"
    (tmp_path / "fast-publication").rename(moved)
    with pytest.raises(ArtifactIntegrityError, match="not found"):
        inspect_artifact(
            artifact=tmp_path / "fast-publication.publication",
            require_admitted=True,
        )
    moved.rename(tmp_path / "fast-publication")
    (tmp_path / "fast-publication/report.json").write_text("{}")
    with pytest.raises(ArtifactIntegrityError, match="inventory|checksum"):
        inspect_artifact(
            artifact=tmp_path / "fast-publication.publication",
            require_admitted=True,
        )
