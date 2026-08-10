"""Frozen measurement and admission core for the MLX periodic DFT runtime."""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import platform
import queue as queue_module
import shutil
import signal
import statistics
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic._artifact_identity import (
    ArtifactIntegrityError,
    AtomicGeneration,
    canonical_json_bytes,
    generation_root,
    inspect_generation,
    read_generation_json,
    sha256_bytes,
    sha256_file,
)
from mlx_atomistic.benchmarks.dft_runtime_contract import (
    GTH_ELEMENT,
    GTH_NAME,
    TARGET_CHIP,
    build_source_fingerprints,
    collect_host_provenance,
    find_repo_root,
    host_admission,
    load_workload,
)

REPORT_SCHEMA = "mlx-atomistic.dft-runtime-report.v1"
SEAL_SCHEMA = "mlx-atomistic.dft-runtime-baseline-seal.v1"
COMPARISON_SCHEMA = "mlx-atomistic.dft-runtime-comparison.v1"
LADDER_SCHEMA = "mlx-atomistic.dft-runtime-ladder.v1"
FULL_SCF_SCHEMA = "mlx-atomistic.dft-runtime-full-scf.v1"
FULL_SCF_PUBLICATION_SCHEMA = "mlx-atomistic.dft-runtime-full-scf-publication.v1"
ORACLE_SCHEMA = "mlx-atomistic.dft-runtime-oracle.v1"
COMMAND_FAILURE_SCHEMA = "mlx-atomistic.dft-runtime-command-failure.v1"

_REPORT_ENVELOPE_CONTRACTS = {
    "fixed-density": ("dft-runtime-fixed-density", REPORT_SCHEMA),
    "fixed-density-comparison": ("dft-runtime-comparison", COMPARISON_SCHEMA),
    "engineering-ladder": ("dft-runtime-ladder", LADDER_SCHEMA),
    "full-scf": ("dft-runtime-full-scf", FULL_SCF_SCHEMA),
    "full-scf-publication-attestation": (
        "dft-runtime-full-scf-publication",
        FULL_SCF_PUBLICATION_SCHEMA,
    ),
    "oracle": ("dft-runtime-oracle", ORACLE_SCHEMA),
    "command-failure": ("dft-runtime-command-failure", COMMAND_FAILURE_SCHEMA),
}
PRE_ARCHITECTURE_REV = "038263effcd017b5ad47426fe5c2ff68077004f6"
BASELINE_EXPECTED_PARENT_REV = "a0d85fdfd370595388355b8ec4bd8b857c671c3b"
BASELINE_ALLOWED_DIFF_PATHS = frozenset(
    {
        "scripts/run_dft_runtime_oracle.py",
        "src/mlx_atomistic/_artifact_identity.py",
        "src/mlx_atomistic/benchmarks/dft_runtime.py",
        "src/mlx_atomistic/benchmarks/dft_runtime_contract.py",
        "src/mlx_atomistic/benchmarks/dft_runtime_core.py",
        "src/mlx_atomistic/dft/_runtime_observer.py",
        "src/mlx_atomistic/dft/periodic_scf.py",
        "src/mlx_atomistic/dft/runtime_state.py",
        "tests/test_dft_runtime_harness.py",
        "tests/test_runtime_boundaries.py",
    }
)
BASELINE_ABSENT_OPTIMIZATIONS = (
    "compact-coefficient-storage",
    "batched-fft-hpsi",
    "incremental-hv",
    "representative-only-execution",
)
_EIGENSOLVE_PROVENANCE_FIELDS = (
    "full_grid_precision",
    "projected_eigensolve_device",
    "projected_eigensolve_backend",
    "projected_eigensolve_precision",
    "projected_eigensolve_output_precision",
)
BASELINE_DIFF_CHECK_NAMES = frozenset(
    {
        "git_commands_succeeded",
        "pre_architecture_revision_is_ancestor",
        "baseline_history_has_no_merge_commits",
        "baseline_parent_is_expected_prebaseline_revision",
        "baseline_revision_is_distinct",
        "diff_is_nonempty",
        "diff_paths_are_allowed",
        "diff_records_are_parseable_regular_files",
    }
)

ProgressCallback = Callable[[dict[str, object]], None]


def _git_command(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def collect_git_provenance(repo_root: str | Path | None = None) -> dict[str, object]:
    """Collect read-only Git revision and whole-tree cleanliness provenance."""

    root = find_repo_root(repo_root)
    revision = _git_command(root, "rev-parse", "HEAD")
    parent = _git_command(root, "rev-parse", "HEAD^")
    porcelain = _git_command(root, "status", "--porcelain", "--untracked-files=all")
    return {
        "revision": revision,
        "parent": parent,
        "dirty": porcelain is None or bool(porcelain),
        "status_available": porcelain is not None,
    }


def _baseline_diff_audit(repo_root: str | Path | None = None) -> dict[str, object]:
    root = find_repo_root(repo_root)
    revision = _git_command(root, "rev-parse", "HEAD")
    parent = _git_command(root, "rev-parse", "HEAD^")
    merge_base = _git_command(root, "merge-base", PRE_ARCHITECTURE_REV, "HEAD")
    merge_commits = _git_command(
        root,
        "rev-list",
        "--merges",
        f"{PRE_ARCHITECTURE_REV}..HEAD",
    )
    try:
        completed = subprocess.run(
            (
                "git",
                "diff",
                "--name-status",
                "--no-renames",
                f"{PRE_ARCHITECTURE_REV}..HEAD",
            ),
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
            timeout=10.0,
        )
        patch = subprocess.run(
            (
                "git",
                "diff",
                "--binary",
                "--full-index",
                "--no-renames",
                f"{PRE_ARCHITECTURE_REV}..HEAD",
            ),
            cwd=root,
            capture_output=True,
            check=False,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        completed = None
        patch = None
    records: list[dict[str, object]] = []
    parse_error = False
    if completed is not None and completed.returncode == 0:
        for line in completed.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) != 2:
                parse_error = True
                continue
            status, logical_path = fields
            path = root / logical_path
            if status not in {"A", "M"} or path.is_symlink() or not path.is_file():
                parse_error = True
                continue
            records.append(
                {
                    "status": status,
                    "path": logical_path,
                    "byte_size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    checks = {
        "git_commands_succeeded": (
            completed is not None
            and completed.returncode == 0
            and patch is not None
            and patch.returncode == 0
            and merge_base is not None
            and merge_commits is not None
        ),
        "pre_architecture_revision_is_ancestor": merge_base == PRE_ARCHITECTURE_REV,
        "baseline_history_has_no_merge_commits": merge_commits == "",
        "baseline_parent_is_expected_prebaseline_revision": (
            parent == BASELINE_EXPECTED_PARENT_REV
        ),
        "baseline_revision_is_distinct": revision not in {None, PRE_ARCHITECTURE_REV},
        "diff_is_nonempty": bool(records),
        "diff_paths_are_allowed": all(
            str(record["path"]) in BASELINE_ALLOWED_DIFF_PATHS for record in records
        ),
        "diff_records_are_parseable_regular_files": not parse_error,
    }
    return {
        "base_revision": PRE_ARCHITECTURE_REV,
        "baseline_revision": revision,
        "baseline_parent_revision": parent,
        "allowed_paths": sorted(BASELINE_ALLOWED_DIFF_PATHS),
        "changed_files": records,
        "patch_sha256": (
            sha256_bytes(patch.stdout)
            if patch is not None and patch.returncode == 0
            else None
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _baseline_diff_audit_matches(
    audit: object,
    git: Mapping[str, object],
) -> bool:
    if not isinstance(audit, dict) or audit.get("passed") is not True:
        return False
    checks = audit.get("checks")
    if (
        not isinstance(checks, dict)
        or set(checks) != BASELINE_DIFF_CHECK_NAMES
        or any(value is not True for value in checks.values())
        or audit.get("base_revision") != PRE_ARCHITECTURE_REV
        or audit.get("baseline_revision") != git.get("revision")
        or audit.get("baseline_parent_revision") != git.get("parent")
        or audit.get("baseline_parent_revision") != BASELINE_EXPECTED_PARENT_REV
        or audit.get("allowed_paths") != sorted(BASELINE_ALLOWED_DIFF_PATHS)
        or not _is_sha256(audit.get("patch_sha256"))
    ):
        return False
    records = audit.get("changed_files")
    if not isinstance(records, list) or not records:
        return False
    paths: list[str] = []
    for record in records:
        if (
            not isinstance(record, dict)
            or record.get("status") not in {"A", "M"}
            or not isinstance(record.get("path"), str)
            or record.get("path") not in BASELINE_ALLOWED_DIFF_PATHS
            or type(record.get("byte_size")) is not int
            or record["byte_size"] < 0
            or not _is_sha256(record.get("sha256"))
        ):
            return False
        paths.append(str(record["path"]))
    return len(paths) == len(set(paths))


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _runtime_environment() -> dict[str, object]:
    import mlx.core as mx

    from mlx_atomistic.dft.periodic_scf import _eigensolve_provenance
    from mlx_atomistic.runtime import get_runtime_info

    runtime = get_runtime_info()
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "mlx_version": runtime.mlx_version,
        "default_device": runtime.default_device,
        "metal_available": runtime.metal_available,
        "selected_device": str(mx.default_device()),
        "precision": "complex64/float32",
        **_eigensolve_provenance(),
    }


def _host_protocol(host: Mapping[str, object]) -> dict[str, object]:
    return {
        "model": host.get("model"),
        "model_identifier": host.get("model_identifier"),
        "chip": host.get("chip"),
        "machine": host.get("machine"),
        "macos": host.get("macos"),
        "power_source": host.get("power_source"),
        "low_power_mode": host.get("low_power_mode"),
    }


def build_execution_context(
    *,
    manifest: Mapping[str, object],
    command_kind: str,
    host: Mapping[str, object],
    repo_root: str | Path | None = None,
    settings_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the pre-run execution contract and adjacent Git provenance."""

    root = find_repo_root(repo_root)
    sources = build_source_fingerprints(root)
    environment = _runtime_environment()
    lock_path = root / "uv.lock"
    lock = {
        "path": "uv.lock",
        "byte_size": lock_path.stat().st_size,
        "sha256": sha256_file(lock_path),
    }
    contract: dict[str, object] = {
        "schema_version": "mlx-atomistic.dft-runtime-execution-contract.v1",
        "command_kind": command_kind,
        "workload_fingerprint": manifest["workload_fingerprint"],
        "protocol_fingerprint": sources["protocol_fingerprint"],
        "runtime_fingerprint": sources["runtime_fingerprint"],
        "solver": manifest["solver"],
        "initialization": manifest["initialization"],
        "settings_override": dict(settings_override or {}),
        "lock": lock,
        "environment": environment,
        "host_protocol": _host_protocol(host),
        "synchronization": manifest["measurement"]["synchronization"],
    }
    fingerprint = sha256_bytes(canonical_json_bytes(contract))
    return {
        "execution_contract": contract,
        "execution_contract_fingerprint": fingerprint,
        **sources,
        "git": collect_git_provenance(root),
    }


def _report_fingerprint(report: Mapping[str, object]) -> str:
    unsigned = {key: value for key, value in report.items() if key != "report_fingerprint"}
    return sha256_bytes(canonical_json_bytes(unsigned))


def _finalize_report(report: dict[str, object]) -> dict[str, object]:
    report["report_fingerprint"] = _report_fingerprint(report)
    return report


def _formal_admission(
    *,
    statuses: Mapping[str, object],
    command_admission: Mapping[str, object],
    producer_git: Mapping[str, object] | None,
    run_protocol: Mapping[str, object] | None,
    report_kind: str,
    host_protocol: Mapping[str, object] | None,
) -> dict[str, object]:
    blockers: list[str] = []
    if command_admission.get("passed") is not True:
        blockers.append("command_admission_blocked")
    if statuses.get("numerical_status") != "passed":
        blockers.append("numerical_status_not_passed")
    if statuses.get("resume_integrity_status") != "fresh-no-resume":
        blockers.append("resume_integrity_status_not_fresh")
    if statuses.get("timing_admission_status") != "admitted":
        blockers.append("timing_status_not_admitted")
    if not isinstance(producer_git, Mapping) or producer_git.get("dirty") is not False:
        blockers.append("producer_checkout_not_clean")
    if not isinstance(run_protocol, Mapping):
        blockers.append("run_protocol_missing")
    else:
        if run_protocol.get("fresh") is not True:
            blockers.append("run_not_fresh")
        if run_protocol.get("resumed") is not False:
            blockers.append("run_was_resumed")
        if run_protocol.get("diagnostic", False) is not False:
            blockers.append("run_is_diagnostic")
        if report_kind == "fixed-density" and (
            run_protocol.get("warmups") != 1 or run_protocol.get("samples") != 5
        ):
            blockers.append("formal_fixed_density_cadence_mismatch")
        if report_kind in {"full-scf", "full-scf-publication-attestation"} and (
            run_protocol.get("warmups") != 0
            or run_protocol.get("new_process") is not True
        ):
            blockers.append("formal_full_scf_cadence_mismatch")
    raw_host_report = report_kind in {
        "fixed-density",
        "full-scf",
        "full-scf-publication-attestation",
    }
    comparison_report = report_kind == "fixed-density-comparison"
    raw_host_matches = not raw_host_report or (
        isinstance(host_protocol, Mapping)
        and host_admission(
            host_protocol,
            required_chip=TARGET_CHIP,
            require_low_power=True,
        )["admitted"]
        is True
    )
    comparison_host_matches = not comparison_report or (
        isinstance(host_protocol, Mapping)
        and host_protocol.get("chip") == TARGET_CHIP
        and host_protocol.get("low_power_mode") == 1
        and host_protocol.get("power_source") in {"AC Power", "Battery Power"}
    )
    if not raw_host_matches or not comparison_host_matches:
        blockers.append("formal_target_host_low_power_mismatch")
    return {"passed": not blockers, "blockers": blockers}


def _derived_formal_admission(report: Mapping[str, object]) -> dict[str, object] | None:
    statuses = report.get("statuses")
    command_admission = report.get("admission")
    if not isinstance(statuses, dict) or not isinstance(command_admission, dict):
        return None
    return _formal_admission(
        statuses=statuses,
        command_admission=command_admission,
        producer_git=(
            report.get("context", {}).get("git")
            if isinstance(report.get("context"), dict)
            else report.get("producer_context", {}).get("git")
            if isinstance(report.get("producer_context"), dict)
            else None
        ),
        run_protocol=(
            report.get("run_protocol")
            if isinstance(report.get("run_protocol"), dict)
            else None
        ),
        report_kind=str(report.get("kind", "")),
        host_protocol=(
            report.get("host")
            if isinstance(report.get("host"), dict)
            else report.get("matched_protocol")
            if isinstance(report.get("matched_protocol"), dict)
            else None
        ),
    )


def _report_is_formally_admitted(report: Mapping[str, object]) -> bool:
    declared = report.get("formal_admission")
    observed = _derived_formal_admission(report)
    return isinstance(declared, dict) and declared == observed and observed["passed"] is True


def _copy_extra_tree(generation: AtomicGeneration, extra_tree: str | Path) -> None:
    tree = Path(extra_tree)
    if tree.is_symlink() or not tree.is_dir():
        msg = f"extra artifact tree must be a regular non-symlink directory: {tree}"
        raise ArtifactIntegrityError(msg)
    root = tree.resolve()

    def walk_error(error: OSError) -> None:
        msg = f"extra artifact tree traversal failed: {error}"
        raise ArtifactIntegrityError(msg) from error

    for directory, names, files in os.walk(
        root,
        followlinks=False,
        onerror=walk_error,
    ):
        directory_path = Path(directory)
        for name in names:
            path = directory_path / name
            if path.is_symlink():
                msg = f"extra artifact tree may not contain symlink directories: {path}"
                raise ArtifactIntegrityError(msg)
        for name in files:
            source = directory_path / name
            if source.is_symlink() or not source.is_file():
                msg = f"extra artifact tree payload must be a regular file: {source}"
                raise ArtifactIntegrityError(msg)
            relative = source.relative_to(root).as_posix()
            payload_destination = generation.path(relative)
            with source.open("rb") as reader, payload_destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())


def _publish_report(
    *,
    out: str | Path,
    artifact_kind: str,
    artifact_schema: str,
    report: Mapping[str, object],
    extra_json: Mapping[str, object] | None = None,
    extra_files: Mapping[str, bytes] | None = None,
    extra_tree: str | Path | None = None,
    before_report_write: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    identity = report.get("identity")
    if not isinstance(identity, dict):
        msg = "DFT runtime report identity is missing"
        raise ValueError(msg)
    artifact_destination = Path(out)
    mutable_report = report if isinstance(report, dict) else dict(report)
    with AtomicGeneration(
        destination=artifact_destination,
        artifact_kind=artifact_kind,
        artifact_schema_version=artifact_schema,
        identity=identity,
    ) as generation:
        for relative_path, value in (extra_json or {}).items():
            generation.write_json(relative_path, value)
        for relative_path, value in (extra_files or {}).items():
            generation.write_bytes(relative_path, value)
        if extra_tree is not None:
            _copy_extra_tree(generation, extra_tree)
        if before_report_write is not None:
            before_report_write(mutable_report)
        generation.metadata = {
            "admission": mutable_report.get("admission"),
            "formal_admission": mutable_report.get("formal_admission"),
        }
        generation.write_json("report.json", mutable_report)
        generation.publish()
    timing_status = mutable_report.get("statuses", {}).get("timing_admission_status")
    command_passed = mutable_report.get("admission", {}).get("passed") is True
    return {
        "artifact": str(artifact_destination),
        "report": str(artifact_destination / "report.json"),
        "status": (
            "ok"
            if command_passed and timing_status == "admitted"
            else "diagnostic"
            if command_passed
            else "blocked"
        ),
        "admission": mutable_report.get("admission"),
        "formal_admission": mutable_report.get("formal_admission"),
    }


def _require_absent_output(out: str | Path) -> None:
    destination = Path(out).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)


def _progress_wrapper(
    callback: ProgressCallback | None,
    *,
    sample_kind: str,
    sample_index: int,
) -> ProgressCallback | None:
    if callback is None:
        return None

    def wrapped(event: dict[str, object]) -> None:
        callback({**event, "sample_kind": sample_kind, "sample_index": sample_index})

    return wrapped


def _selected_point(manifest: Mapping[str, object]) -> tuple[float, float, float]:
    physics = manifest["physics"]
    lane_index = int(physics["fixed_density_lane_index"])
    point = physics["kpoints"][lane_index]
    return tuple(float(value) for value in point["reduced_coordinates"])


def _davidson_config(manifest: Mapping[str, object]):
    from mlx_atomistic.dft import PeriodicDavidsonConfig

    settings = manifest["solver"]["davidson"]
    return PeriodicDavidsonConfig(
        max_iterations=int(settings["max_iterations"]),
        tolerance=float(settings["tolerance"]),
        max_subspace_size=int(settings["max_subspace_size"]),
        preconditioner_floor=float(settings["preconditioner_floor"]),
    )


def _fixed_density_sample(
    *,
    manifest: Mapping[str, object],
    gth_source: str | Path,
    progress: ProgressCallback | None,
    grid_shape: Sequence[int] | None = None,
    cutoff_hartree: float | None = None,
    reduced_kpoint: Sequence[float] | None = None,
    band_count: int | None = None,
) -> tuple[dict[str, object], object]:
    import mlx.core as mx

    from mlx_atomistic.dft import (
        PeriodicDFTSystem,
        PeriodicGTHNonlocalOperator,
        PeriodicKohnShamOperator,
        PlaneWaveBasis,
        ProductionPBEExchangeCorrelation,
        gth_local_potential_grid,
        hartree_potential,
        read_gth,
        solve_periodic_eigenproblem,
    )
    from mlx_atomistic.dft._runtime_observer import RuntimeObserver
    from mlx_atomistic.dft.periodic_scf import _eigensolve_provenance
    from mlx_atomistic.dft.runtime_state import fixed_density_state_metrics

    track_metal_memory = bool(mx.metal.is_available())
    if track_metal_memory:
        mx.synchronize()
        mx.reset_peak_memory()
    observer = RuntimeObserver(callback=progress)
    wall_start = time.perf_counter()
    try:
        observer.emit("setup", status="started", stage="fixed_density")
        with observer.phase("setup"):
            system_values = manifest["system"]
            physics = manifest["physics"]
            shape = tuple(int(value) for value in (grid_shape or physics["fft_shape"]))
            cutoff = float(
                physics["kinetic_cutoff_hartree"]
                if cutoff_hartree is None
                else cutoff_hartree
            )
            point = (
                _selected_point(manifest)
                if reduced_kpoint is None
                else tuple(float(value) for value in reduced_kpoint)
            )
            bands = int(
                system_values["occupied_band_count"]
                if band_count is None
                else band_count
            )
            if len(point) != 3 or bands <= 0:
                msg = "fixed-density lane requires one 3-vector and positive band count"
                raise ValueError(msg)
            lattice = float(system_values["lattice_constant_bohr"])
            fractional = np.asarray(system_values["fractional_positions"], dtype=np.float64)
            positions = fractional * lattice
            pseudo = read_gth(gth_source, element=GTH_ELEMENT, name=GTH_NAME)
            system = PeriodicDFTSystem(
                (lattice, lattice, lattice),
                shape,
                positions,
                pseudo,
                electron_count=float(system_values["electron_count"]),
            )
            density = mx.full(system.grid.shape, system.electron_count / system.grid.volume)
            gamma_basis = PlaneWaveBasis(system.grid, cutoff)
            local = gth_local_potential_grid(pseudo, gamma_basis, positions)
            hartree = hartree_potential(density, system.grid)
            xc = ProductionPBEExchangeCorrelation().evaluate(density, system.grid)
            effective = local + hartree + xc.potential
            basis = PlaneWaveBasis.from_reduced_kpoint(
                system.grid,
                cutoff,
                point,
            )
            nonlocal_operator = PeriodicGTHNonlocalOperator(pseudo, basis, positions)
            operator = PeriodicKohnShamOperator(
                basis,
                effective,
                nonlocal_operator,
                observer,
            )
        observer.record_memory("shared_full_grid_bytes", system.grid.size * 4 * 4)
        observer.record_memory("persistent_projector_bytes", 0)
        observer.emit(
            "setup",
            status="completed",
            stage="fixed_density",
            active_count=basis.active_count,
        )
        result = solve_periodic_eigenproblem(
            operator,
            n_bands=bands,
            config=_davidson_config(manifest),
            observer=observer,
        )
        mx.synchronize()
        state_metrics = fixed_density_state_metrics(result=result, basis=basis)
        coefficient_bytes = int(state_metrics["coefficient_payload_bytes"])
        observer.record_memory("persistent_coefficient_bytes", coefficient_bytes)
        observer.record_memory("coefficient_payload_bytes", coefficient_bytes)
        interim = observer.snapshot()
        projector_traffic = int(interim["work_counters"]["projector_traffic_elements"])
        observer.record_memory("projector_traffic_bytes", projector_traffic * 8)
        max_residual = float(mx.max(result.residuals))
        eigenvalues = np.asarray(result.eigenvalues, dtype=np.float64).tolist()
        mx.synchronize()
        observer.record_memory("process_high_water_bytes", _process_high_water_bytes())
        observer.record_memory(
            "unified_memory_high_water_bytes",
            int(mx.get_peak_memory()) if track_metal_memory else None,
        )
        numerical_passed = bool(
            result.converged
            and max_residual <= float(manifest["solver"]["davidson"]["tolerance"])
            and result.orthonormality_error
            <= float(manifest["numerical_gates"]["orthonormality_max"])
            and np.all(np.isfinite(eigenvalues))
        )
        observer.emit(
            "completion",
            stage="fixed_density",
            status="converged" if numerical_passed else "numerical_failure",
            iterations=result.iterations,
        )
        observation = observer.snapshot()
        sample = {
            "status": "ok" if numerical_passed else "blocked",
            "numerical_passed": numerical_passed,
            "wall_elapsed_seconds": time.perf_counter() - wall_start,
            "grid_shape": list(system.grid.shape),
            "cutoff_hartree": cutoff,
            "reduced_kpoint": list(point),
            "band_count": bands,
            "active_count": basis.active_count,
            "full_grid_count": system.grid.size,
            "eigenvalues_hartree": eigenvalues,
            "max_residual": max_residual,
            "orthonormality_error": result.orthonormality_error,
            "iterations": result.iterations,
            "subspace_size": result.subspace_size,
            "restart_count": result.restart_count,
            **_eigensolve_provenance(),
            "observation": observation,
        }
        return sample, {
            "result": result,
            "basis": basis,
            "density": density,
            "effective_local_potential": effective,
        }
    except Exception as error:
        observer.emit(
            "failure",
            stage="fixed_density",
            error_type=type(error).__name__,
            message=str(error),
        )
        raise


def _process_high_water_bytes() -> int | None:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    return value if platform.system() == "Darwin" else value * 1024


def _load_seal(path: str | Path) -> dict[str, object]:
    generation = inspect_generation(path)
    if (
        generation.get("artifact_kind") != "dft-runtime-fixed-density"
        or generation.get("artifact_schema_version") != REPORT_SCHEMA
    ):
        msg = "DFT runtime baseline seal has an invalid generation envelope"
        raise ValueError(msg)
    report = _load_report(path, generation=generation)
    if report.get("kind") != "fixed-density":
        msg = "DFT runtime baseline seal requires a fixed-density report"
        raise ValueError(msg)
    seal = read_generation_json(path, "seal.json")
    if not isinstance(seal, dict) or seal.get("schema_version") != SEAL_SCHEMA:
        msg = "unsupported DFT runtime baseline seal"
        raise ValueError(msg)
    unsigned = {key: value for key, value in seal.items() if key != "seal_fingerprint"}
    if seal.get("seal_fingerprint") != sha256_bytes(canonical_json_bytes(unsigned)):
        msg = "DFT runtime baseline seal fingerprint mismatch"
        raise ValueError(msg)
    if seal.get("report_fingerprint") != report.get("report_fingerprint"):
        msg = "DFT runtime baseline seal does not bind its co-published report"
        raise ValueError(msg)
    identity = report.get("identity")
    context = report.get("context")
    host = report.get("host")
    comparison = report.get("comparison_protocol")
    summary = report.get("summary")
    git = context.get("git") if isinstance(context, dict) else None
    eigenvalues = (
        summary.get("fixed_density_eigenvalues")
        if isinstance(summary, dict)
        else None
    )
    if (
        not _report_is_formally_admitted(report)
        or report.get("baseline_seal_fingerprint") is not None
        or not isinstance(identity, dict)
        or not isinstance(context, dict)
        or not isinstance(git, dict)
        or not isinstance(host, dict)
        or not isinstance(comparison, dict)
        or not isinstance(summary, dict)
        or not isinstance(eigenvalues, dict)
        or seal.get("baseline_rev") != git.get("revision")
        or seal.get("base_rev") != PRE_ARCHITECTURE_REV
        or seal.get("parent_rev") != git.get("parent")
        or seal.get("dirty") is not False
        or git.get("dirty") is not False
        or seal.get("workload_fingerprint") != identity.get("workload_fingerprint")
        or seal.get("protocol_fingerprint") != identity.get("protocol_fingerprint")
        or seal.get("baseline_runtime_fingerprint")
        != identity.get("runtime_fingerprint")
        or seal.get("protocol_inventory") != context.get("protocol_inventory")
        or seal.get("baseline_runtime_inventory") != context.get("runtime_inventory")
        or seal.get("selected_gth_resource")
        != comparison.get("selected_gth_resource")
        or seal.get("comparison_protocol") != comparison
        or seal.get("host_protocol") != _host_protocol(host)
        or seal.get("median_elapsed_seconds")
        != summary.get("median_elapsed_seconds")
        or seal.get("raw_elapsed_seconds") != summary.get("raw_elapsed_seconds")
        or seal.get("representative_observation")
        != summary.get("representative_observation")
        or seal.get("fixed_density_eigenvalues_hartree")
        != eigenvalues.get("representative_eigenvalues_hartree")
        or seal.get("fixed_density_eigenvalue_tolerance_hartree")
        != eigenvalues.get("tolerance_hartree")
        or seal.get("baseline_structure_audit")
        != summary.get("baseline_structure_audit")
        or seal.get("baseline_diff_audit") != summary.get("baseline_diff_audit")
        or not isinstance(seal.get("baseline_structure_audit"), dict)
        or seal.get("baseline_structure_audit", {}).get("passed") is not True
        or not _baseline_diff_audit_matches(seal.get("baseline_diff_audit"), git)
        or seal.get("target_optimizations_absent")
        != list(BASELINE_ABSENT_OPTIMIZATIONS)
    ):
        msg = "DFT runtime baseline seal is inconsistent with its admitted report"
        raise ValueError(msg)
    return seal


def _seal_compatibility(
    seal: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    context: Mapping[str, object],
    host: Mapping[str, object],
) -> list[str]:
    blockers: list[str] = []
    if seal.get("workload_fingerprint") != manifest.get("workload_fingerprint"):
        blockers.append("seal_workload_mismatch")
    resources = manifest.get("resources")
    selected_resource = resources[0] if isinstance(resources, list) and resources else None
    if seal.get("selected_gth_resource") != selected_resource:
        blockers.append("seal_selected_gth_mismatch")
    if seal.get("protocol_fingerprint") != context.get("protocol_fingerprint"):
        blockers.append("seal_protocol_mismatch")
    if seal.get("dirty") is not False:
        blockers.append("seal_baseline_dirty")
    if seal.get("base_rev") != PRE_ARCHITECTURE_REV:
        blockers.append("seal_base_revision_mismatch")
    baseline_audit = seal.get("baseline_diff_audit")
    if not isinstance(baseline_audit, dict) or baseline_audit.get("passed") is not True:
        blockers.append("seal_baseline_diff_audit_failed")
    baseline_revision = seal.get("baseline_rev")
    current_revision = context.get("git", {}).get("revision")
    if current_revision == baseline_revision and (
        seal.get("baseline_runtime_fingerprint") != context.get("runtime_fingerprint")
    ):
        blockers.append("seal_baseline_runtime_mismatch")
    sealed_host = seal.get("host_protocol")
    if isinstance(sealed_host, dict):
        for field in ("chip", "power_source", "low_power_mode"):
            if sealed_host.get(field) != _host_protocol(host).get(field):
                blockers.append(f"seal_host_{field}_mismatch")
    sealed_contract = seal.get("comparison_protocol")
    current_contract = _comparison_protocol(manifest, context, host)
    if sealed_contract != current_contract:
        blockers.append("seal_comparison_protocol_mismatch")
    return blockers


def _comparison_protocol(
    manifest: Mapping[str, object],
    context: Mapping[str, object],
    host: Mapping[str, object],
) -> dict[str, object]:
    execution = context["execution_contract"]
    return {
        "workload_fingerprint": manifest["workload_fingerprint"],
        "selected_gth_resource": manifest["resources"][0],
        "protocol_fingerprint": context["protocol_fingerprint"],
        "solver": manifest["solver"],
        "initialization": manifest["initialization"],
        "lock": execution["lock"],
        "python_version": execution["environment"]["python_version"],
        "mlx_version": execution["environment"]["mlx_version"],
        "macos": host.get("macos"),
        "precision": execution["environment"]["precision"],
        **{
            field: execution["environment"][field]
            for field in _EIGENSOLVE_PROVENANCE_FIELDS
        },
        "selected_device": execution["environment"]["selected_device"],
        "chip": host.get("chip"),
        "power_source": host.get("power_source"),
        "low_power_mode": host.get("low_power_mode"),
        "synchronization": manifest["measurement"]["synchronization"],
    }


def _representative_observation(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not samples:
        return {"work_counters": {}, "memory": {}}
    observations = [sample["observation"] for sample in samples]
    work = observations[0]["work_counters"]
    memory = observations[0]["memory"]
    return {"work_counters": dict(work), "memory": dict(memory)}


def _fixed_density_eigenvalue_evidence(
    samples: Sequence[Mapping[str, object]],
    *,
    tolerance: float,
) -> dict[str, object]:
    values: list[np.ndarray] = []
    for sample in samples:
        candidate = np.asarray(sample.get("eigenvalues_hartree", ()), dtype=np.float64)
        if candidate.ndim != 1 or candidate.size == 0 or not np.all(np.isfinite(candidate)):
            return {
                "representative_eigenvalues_hartree": None,
                "maximum_sample_delta_hartree": None,
                "tolerance_hartree": tolerance,
                "passed": False,
            }
        values.append(candidate)
    if not values or any(candidate.shape != values[0].shape for candidate in values):
        return {
            "representative_eigenvalues_hartree": None,
            "maximum_sample_delta_hartree": None,
            "tolerance_hartree": tolerance,
            "passed": False,
        }
    stacked = np.stack(values, axis=0)
    representative = np.median(stacked, axis=0)
    maximum_delta = float(np.max(np.abs(stacked - representative[None, :])))
    return {
        "representative_eigenvalues_hartree": representative.tolist(),
        "maximum_sample_delta_hartree": maximum_delta,
        "tolerance_hartree": tolerance,
        "passed": maximum_delta <= tolerance,
    }


def _baseline_structure_audit(
    manifest: Mapping[str, object],
    representative: Mapping[str, object],
    samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    memory = representative.get("memory", {})
    work = representative.get("work_counters", {})
    grid_count = int(np.prod(manifest["physics"]["fft_shape"]))
    band_count = int(manifest["system"]["occupied_band_count"])
    expected_dense_bytes = grid_count * band_count * 8
    sample_memory_audits: list[dict[str, object]] = []
    for index, sample in enumerate(samples):
        observation = sample.get("observation", {})
        sample_memory = (
            observation.get("memory", {}) if isinstance(observation, Mapping) else {}
        )
        sample_work = (
            observation.get("work_counters", {})
            if isinstance(observation, Mapping)
            else {}
        )
        events = (
            observation.get("events", []) if isinstance(observation, Mapping) else []
        )
        widths = [
            int(
                event.get(
                    "subspace_size",
                    event.get("maximum_subspace_size", 0),
                )
            )
            for event in events
            if isinstance(event, Mapping)
            and event.get("event") in {"davidson_iteration", "davidson_round"}
            and type(
                event.get(
                    "subspace_size",
                    event.get("maximum_subspace_size"),
                )
            )
            is int
            and int(
                event.get(
                    "subspace_size",
                    event.get("maximum_subspace_size", 0),
                )
            )
            > 0
        ]
        maximum_width = max([band_count, *widths])
        hpsi_vectors = sample_work.get("hpsi_vector_equivalents")
        projector_generated = sample_work.get("projector_elements_generated")
        projector_loaded = sample_work.get("projector_elements_loaded")
        projector_traffic = sample_work.get("projector_traffic_elements")
        projector_work_is_integral = (
            type(hpsi_vectors) is int
            and hpsi_vectors > 0
            and type(projector_generated) is int
            and projector_generated > 0
            and projector_generated % hpsi_vectors == 0
        )
        expected_projector_payload = (
            projector_generated // hpsi_vectors * 8
            if projector_work_is_integral
            else None
        )
        expected_fft_workspace = 2 * maximum_width * grid_count * 8
        expected_peak_temporary = (
            expected_fft_workspace + maximum_width * expected_projector_payload
            if expected_projector_payload is not None
            else None
        )
        process_high_water = sample_memory.get("process_high_water_bytes")
        unified_high_water = sample_memory.get("unified_memory_high_water_bytes")
        memory_checks = {
            "davidson_width_observed": bool(widths),
            "dense_coefficient_payload": sample_memory.get(
                "coefficient_payload_bytes"
            )
            == expected_dense_bytes,
            "projector_work_relationships": (
                projector_work_is_integral
                and projector_loaded == 2 * projector_generated
                and projector_traffic == 3 * projector_generated
                and sample_memory.get("projector_traffic_bytes")
                == projector_traffic * 8
            ),
            "projector_payload_matches_work": (
                expected_projector_payload is not None
                and sample_memory.get("projector_payload_bytes")
                == expected_projector_payload
            ),
            "fft_workspace_matches_observed_width": sample_memory.get(
                "fft_workspace_bytes"
            )
            == expected_fft_workspace,
            "peak_temporary_matches_retained_lazy_graph": (
                expected_peak_temporary is not None
                and sample_memory.get("peak_temporary_bytes")
                == expected_peak_temporary
            ),
            "process_high_water_recorded": (
                type(process_high_water) is int and process_high_water > 0
            ),
            "unified_high_water_recorded": (
                type(unified_high_water) is int and unified_high_water > 0
            ),
        }
        sample_memory_audits.append(
            {
                "sample_index": index,
                "maximum_hpsi_width": maximum_width,
                "expected_projector_payload_bytes": expected_projector_payload,
                "expected_fft_workspace_bytes": expected_fft_workspace,
                "expected_peak_temporary_bytes": expected_peak_temporary,
                "checks": memory_checks,
                "passed": all(memory_checks.values()),
            }
        )
    checks = {
        "full_grid_coefficient_storage": memory.get("coefficient_payload_bytes")
        == expected_dense_bytes,
        "per_vector_fft_submission": work.get("fft_submissions")
        == work.get("fft_vector_equivalents"),
        "no_incremental_hv_reuse": work.get("davidson_hv_reused_vectors") == 0,
        "no_representative_execution": work.get("representative_lane_solves") == 0,
        "no_partner_reconstruction": work.get("partner_reconstructions") == 0,
        "full_grid_projector_payload": isinstance(
            memory.get("projector_payload_bytes"), int | float
        )
        and int(memory["projector_payload_bytes"]) > 0,
        "work_counters_stable": all(
            sample.get("observation", {}).get("work_counters") == work for sample in samples
        ),
        "all_sample_memory_evidence_complete": (
            len(sample_memory_audits) == len(samples)
            and bool(sample_memory_audits)
            and all(audit["passed"] is True for audit in sample_memory_audits)
        ),
    }
    return {
        "expected_dense_coefficient_bytes": expected_dense_bytes,
        "sample_memory_audits": sample_memory_audits,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_fixed_density(
    *,
    manifest_path: str | Path,
    gth_source: str | Path,
    out: str | Path,
    warmups: int,
    samples: int,
    fresh: bool,
    diagnostic: bool,
    require_clean: bool = False,
    require_chip: str | None = None,
    require_low_power: bool = False,
    require_numerical: bool = False,
    seal: bool = False,
    compare_seal: str | Path | None = None,
    require_speedup: float | None = None,
    require_coefficient_reduction: float | None = None,
    require_projector_payload_reduction: float | None = None,
    require_projector_traffic_reduction: float | None = None,
    progress: ProgressCallback | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Run and atomically publish the synchronized one-k fixed-density protocol."""

    if warmups < 0 or samples <= 0:
        msg = "warmups must be non-negative and samples must be positive"
        raise ValueError(msg)
    if seal and compare_seal is not None:
        msg = "baseline seal creation cannot also consume a comparison seal"
        raise ValueError(msg)
    _require_absent_output(out)
    manifest, _selected = load_workload(manifest_path, gth_source=gth_source)
    host = collect_host_provenance()
    context = build_execution_context(
        manifest=manifest,
        command_kind="fixed-density",
        host=host,
        repo_root=repo_root,
        settings_override={"warmups": warmups, "samples": samples},
    )
    blockers: list[str] = []
    requested_host = require_chip is not None or require_low_power
    if requested_host:
        admission = host_admission(
            host,
            required_chip=require_chip,
            require_low_power=require_low_power,
        )
        blockers.extend(str(item) for item in admission["blockers"])
    if require_clean and context["git"]["dirty"]:
        blockers.append("dirty_checkout")
    if not fresh:
        blockers.append("fresh_namespace_not_requested")
    sealed: dict[str, object] | None = None
    if compare_seal is not None:
        sealed = _load_seal(compare_seal)
        blockers.extend(
            _seal_compatibility(sealed, manifest=manifest, context=context, host=host)
        )

    warmup_results: list[dict[str, object]] = []
    measured: list[dict[str, object]] = []
    numerical_errors: list[dict[str, str]] = []
    if not blockers:
        for index in range(warmups):
            try:
                sample, sample_state = _fixed_density_sample(
                    manifest=manifest,
                    gth_source=gth_source,
                    progress=_progress_wrapper(
                        progress,
                        sample_kind="warmup",
                        sample_index=index,
                    ),
                )
                warmup_results.append(sample)
                del sample_state
                if sample.get("numerical_passed") is not True:
                    break
            except Exception as error:
                numerical_errors.append(
                    {"stage": "warmup", "type": type(error).__name__, "message": str(error)}
                )
                break
        warmups_succeeded = len(warmup_results) == warmups and all(
            sample.get("numerical_passed") is True for sample in warmup_results
        )
        if not numerical_errors and warmups_succeeded:
            for index in range(samples):
                try:
                    sample, sample_state = _fixed_density_sample(
                        manifest=manifest,
                        gth_source=gth_source,
                        progress=_progress_wrapper(
                            progress,
                            sample_kind="measured",
                            sample_index=index,
                        ),
                    )
                    measured.append(sample)
                    del sample_state
                    if sample.get("numerical_passed") is not True:
                        break
                except Exception as error:
                    numerical_errors.append(
                        {
                            "stage": "measured",
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    break
    if numerical_errors:
        blockers.append("numerical_execution_failed")
    expected_eigensolve_provenance = {
        field: context["execution_contract"]["environment"][field]
        for field in _EIGENSOLVE_PROVENANCE_FIELDS
    }
    produced_samples = [*warmup_results, *measured]
    if produced_samples and any(
        {field: sample.get(field) for field in _EIGENSOLVE_PROVENANCE_FIELDS}
        != expected_eigensolve_provenance
        for sample in produced_samples
    ):
        blockers.append("eigensolve_provenance_mismatch")
    eigenvalue_tolerance = float(
        manifest["numerical_gates"]["fixed_density_eigenvalue_abs_hartree"]
    )
    eigenvalue_evidence = _fixed_density_eigenvalue_evidence(
        measured,
        tolerance=eigenvalue_tolerance,
    )
    warmups_passed = len(warmup_results) == warmups and all(
        bool(sample.get("numerical_passed")) for sample in warmup_results
    )
    numerical_passed = (
        warmups_passed
        and len(measured) == samples
        and all(bool(sample["numerical_passed"]) for sample in measured)
        and eigenvalue_evidence["passed"] is True
    )
    if not numerical_passed:
        blockers.append("numerical_result_failed")
    if not warmups_passed:
        blockers.append("warmup_numerical_failed")
    if require_numerical and not numerical_passed:
        blockers.append("numerical_gate_failed")
    elapsed_values = [float(sample["wall_elapsed_seconds"]) for sample in measured]
    median_elapsed = statistics.median(elapsed_values) if elapsed_values else None
    representative = _representative_observation(measured)
    baseline_audit = _baseline_structure_audit(manifest, representative, measured)
    metrics = _comparison_metrics(
        sealed,
        median_elapsed,
        representative,
        eigenvalue_evidence,
    )
    if sealed is not None:
        eigenvalue_error = metrics.get("eigenvalue_max_abs_error_hartree")
        if (
            not isinstance(eigenvalue_error, int | float)
            or float(eigenvalue_error) > eigenvalue_tolerance
        ):
            blockers.append("fixed_density_eigenvalue_parity_failed")
    _apply_metric_gate(blockers, "speedup", metrics.get("speedup"), require_speedup)
    _apply_metric_gate(
        blockers,
        "coefficient_reduction",
        metrics.get("coefficient_reduction"),
        require_coefficient_reduction,
    )
    _apply_metric_gate(
        blockers,
        "projector_payload_reduction",
        metrics.get("projector_payload_reduction"),
        require_projector_payload_reduction,
    )
    _apply_metric_gate(
        blockers,
        "projector_traffic_reduction",
        metrics.get("projector_traffic_reduction"),
        require_projector_traffic_reduction,
    )
    if seal and (warmups != 1 or samples != 5):
        blockers.append("seal_requires_one_warmup_and_five_samples")
    baseline_diff_audit = _baseline_diff_audit(repo_root) if seal else None
    if seal and (diagnostic or not require_numerical or not numerical_passed):
        blockers.append("seal_requires_formal_numerical_success")
    if seal and (not require_clean or require_chip != TARGET_CHIP or not require_low_power):
        blockers.append("seal_requires_clean_target_host_low_power_gate")
    if seal and not baseline_audit["passed"]:
        blockers.append("baseline_structure_audit_failed")
    if seal and (
        not _baseline_diff_audit_matches(baseline_diff_audit, context["git"])
    ):
        blockers.append("baseline_diff_audit_failed")
    unique_blockers = sorted(set(blockers))
    identity = {
        "workload_fingerprint": manifest["workload_fingerprint"],
        "protocol_fingerprint": context["protocol_fingerprint"],
        "runtime_fingerprint": context["runtime_fingerprint"],
        "execution_contract_fingerprint": context["execution_contract_fingerprint"],
    }
    statuses = {
        "numerical_status": "passed" if numerical_passed else "blocked",
        "resume_integrity_status": "fresh-no-resume",
        "timing_admission_status": (
            "diagnostic" if diagnostic else "admitted" if not unique_blockers else "blocked"
        ),
    }
    command_admission = {
        "passed": not unique_blockers,
        "blockers": unique_blockers,
    }
    report = _finalize_report(
        {
            "schema_version": REPORT_SCHEMA,
            "kind": "fixed-density",
            "identity": identity,
            "context": context,
            "host": host,
            "comparison_protocol": _comparison_protocol(manifest, context, host),
            "run_protocol": {
                "warmups": warmups,
                "samples": samples,
                "fresh": fresh,
                "diagnostic": diagnostic,
                "resumed": False,
            },
            "warmup_results": warmup_results,
            "samples": measured,
            "numerical_errors": numerical_errors,
            "baseline_seal_fingerprint": (
                sealed.get("seal_fingerprint") if sealed is not None else None
            ),
            "summary": {
                "median_elapsed_seconds": median_elapsed,
                "raw_elapsed_seconds": elapsed_values,
                "representative_observation": representative,
                "fixed_density_eigenvalues": eigenvalue_evidence,
                "metrics_against_seal": metrics,
                "baseline_structure_audit": baseline_audit,
                "baseline_diff_audit": baseline_diff_audit,
            },
            "statuses": statuses,
            "admission": command_admission,
            "formal_admission": _formal_admission(
                statuses=statuses,
                command_admission=command_admission,
                producer_git=context["git"],
                run_protocol={
                    "warmups": warmups,
                    "samples": samples,
                    "fresh": fresh,
                    "diagnostic": diagnostic,
                    "resumed": False,
                },
                report_kind="fixed-density",
                host_protocol=host,
            ),
        }
    )
    extras: dict[str, object] = {}
    if seal and not unique_blockers:
        baseline_seal = _build_baseline_seal(
            manifest=manifest,
            report=report,
            context=context,
            host=host,
        )
        extras["seal.json"] = baseline_seal
    published = _publish_report(
        out=out,
        artifact_kind="dft-runtime-fixed-density",
        artifact_schema=REPORT_SCHEMA,
        report=report,
        extra_json=extras,
    )
    return {**published, "report_payload": report}


def _comparison_metrics(
    seal: Mapping[str, object] | None,
    median_elapsed: float | None,
    representative: Mapping[str, object],
    eigenvalue_evidence: Mapping[str, object],
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "speedup": None,
        "coefficient_reduction": None,
        "projector_payload_reduction": None,
        "projector_traffic_reduction": None,
        "eigenvalue_max_abs_error_hartree": None,
    }
    if seal is None:
        return metrics
    baseline_elapsed = seal.get("median_elapsed_seconds")
    if isinstance(baseline_elapsed, int | float) and median_elapsed and median_elapsed > 0.0:
        metrics["speedup"] = float(baseline_elapsed) / median_elapsed
    baseline_observation = seal.get("representative_observation")
    if not isinstance(baseline_observation, dict):
        return metrics
    baseline_memory = baseline_observation.get("memory", {})
    current_memory = representative.get("memory", {})
    baseline_work = baseline_observation.get("work_counters", {})
    current_work = representative.get("work_counters", {})
    metrics["coefficient_reduction"] = _ratio(
        baseline_memory.get("coefficient_payload_bytes"),
        current_memory.get("coefficient_payload_bytes"),
    )
    metrics["projector_payload_reduction"] = _ratio(
        baseline_memory.get("projector_payload_bytes"),
        current_memory.get("projector_payload_bytes"),
    )
    metrics["projector_traffic_reduction"] = _ratio(
        baseline_work.get("projector_traffic_elements"),
        current_work.get("projector_traffic_elements"),
    )
    baseline_eigenvalues = seal.get("fixed_density_eigenvalues_hartree")
    current_eigenvalues = eigenvalue_evidence.get(
        "representative_eigenvalues_hartree"
    )
    if isinstance(baseline_eigenvalues, list) and isinstance(current_eigenvalues, list):
        baseline_array = np.asarray(baseline_eigenvalues, dtype=np.float64)
        current_array = np.asarray(current_eigenvalues, dtype=np.float64)
        if (
            baseline_array.ndim == 1
            and baseline_array.shape == current_array.shape
            and baseline_array.size > 0
            and np.all(np.isfinite(baseline_array))
            and np.all(np.isfinite(current_array))
        ):
            metrics["eigenvalue_max_abs_error_hartree"] = float(
                np.max(np.abs(baseline_array - current_array))
            )
    return metrics


def _ratio(numerator: object, denominator: object) -> float | None:
    if not isinstance(numerator, int | float) or not isinstance(denominator, int | float):
        return None
    if denominator <= 0:
        return math.inf if numerator > 0 else None
    return float(numerator) / float(denominator)


def _apply_metric_gate(
    blockers: list[str],
    name: str,
    observed: object,
    required: float | None,
) -> None:
    if required is None:
        return
    if not math.isfinite(required) or required < 0.0:
        msg = f"{name} gate must be finite and non-negative"
        raise ValueError(msg)
    if not isinstance(observed, int | float) or float(observed) < required:
        blockers.append(f"{name}_gate_failed")


def _build_baseline_seal(
    *,
    manifest: Mapping[str, object],
    report: Mapping[str, object],
    context: Mapping[str, object],
    host: Mapping[str, object],
) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": SEAL_SCHEMA,
        "baseline_rev": context["git"]["revision"],
        "base_rev": PRE_ARCHITECTURE_REV,
        "parent_rev": context["git"]["parent"],
        "dirty": context["git"]["dirty"],
        "workload_fingerprint": manifest["workload_fingerprint"],
        "selected_gth_resource": manifest["resources"][0],
        "protocol_fingerprint": context["protocol_fingerprint"],
        "protocol_inventory": context["protocol_inventory"],
        "baseline_runtime_fingerprint": context["runtime_fingerprint"],
        "baseline_runtime_inventory": context["runtime_inventory"],
        "comparison_protocol": report["comparison_protocol"],
        "host_protocol": _host_protocol(host),
        "median_elapsed_seconds": report["summary"]["median_elapsed_seconds"],
        "raw_elapsed_seconds": report["summary"]["raw_elapsed_seconds"],
        "representative_observation": report["summary"]["representative_observation"],
        "fixed_density_eigenvalues_hartree": report["summary"][
            "fixed_density_eigenvalues"
        ]["representative_eigenvalues_hartree"],
        "fixed_density_eigenvalue_tolerance_hartree": manifest["numerical_gates"][
            "fixed_density_eigenvalue_abs_hartree"
        ],
        "baseline_structure_audit": report["summary"]["baseline_structure_audit"],
        "baseline_diff_audit": report["summary"]["baseline_diff_audit"],
        "report_fingerprint": report["report_fingerprint"],
        "target_optimizations_absent": list(BASELINE_ABSENT_OPTIMIZATIONS),
    }
    seal = dict(unsigned)
    seal["seal_fingerprint"] = sha256_bytes(canonical_json_bytes(unsigned))
    return seal


def _load_report(
    path: str | Path,
    *,
    generation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if generation is None:
        generation = inspect_generation(path)
    report = read_generation_json(path, "report.json")
    if not isinstance(report, dict):
        msg = "DFT runtime report must be a JSON object"
        raise ValueError(msg)
    expected_envelope = _REPORT_ENVELOPE_CONTRACTS.get(str(report.get("kind")))
    if expected_envelope is None:
        msg = "unsupported DFT runtime report kind or schema"
        raise ValueError(msg)
    expected_kind, expected_schema = expected_envelope
    if report.get("schema_version") != expected_schema:
        msg = "unsupported DFT runtime report kind or schema"
        raise ValueError(msg)
    metadata = generation.get("metadata")
    if (
        generation.get("artifact_kind") != expected_kind
        or generation.get("artifact_schema_version") != expected_schema
        or generation.get("identity") != report.get("identity")
        or not isinstance(metadata, dict)
        or "admission" not in metadata
        or metadata.get("admission") != report.get("admission")
        or "formal_admission" not in metadata
        or metadata.get("formal_admission") != report.get("formal_admission")
    ):
        msg = "DFT runtime report does not match its generation envelope"
        raise ValueError(msg)
    unsigned = {key: value for key, value in report.items() if key != "report_fingerprint"}
    if report.get("report_fingerprint") != sha256_bytes(canonical_json_bytes(unsigned)):
        msg = "DFT runtime report fingerprint mismatch"
        raise ValueError(msg)
    if "formal_admission" in report:
        statuses = report.get("statuses")
        admission = report.get("admission")
        if not isinstance(statuses, dict) or not isinstance(admission, dict):
            msg = "DFT runtime report formal admission inputs are missing"
            raise ValueError(msg)
        expected = _derived_formal_admission(report)
        if expected is None:
            msg = "DFT runtime report formal admission inputs are missing"
            raise ValueError(msg)
        if report.get("formal_admission") != expected:
            msg = "DFT runtime report formal admission is inconsistent"
            raise ValueError(msg)
    return report


def _baseline_lineage_blockers(
    seal_artifact: str | Path,
    report: Mapping[str, object],
) -> tuple[dict[str, object] | None, list[str]]:
    try:
        seal = _load_seal(seal_artifact)
    except (ArtifactIntegrityError, FileNotFoundError, ValueError):
        return None, ["baseline_seal_missing_or_invalid"]
    blockers: list[str] = []
    identity = report.get("identity", {})
    context = report.get("context", {})
    if seal.get("report_fingerprint") != report.get("report_fingerprint"):
        blockers.append("baseline_report_seal_lineage_mismatch")
    if seal.get("workload_fingerprint") != identity.get("workload_fingerprint"):
        blockers.append("baseline_seal_workload_mismatch")
    if seal.get("protocol_fingerprint") != identity.get("protocol_fingerprint"):
        blockers.append("baseline_seal_protocol_mismatch")
    if seal.get("baseline_runtime_fingerprint") != identity.get("runtime_fingerprint"):
        blockers.append("baseline_seal_runtime_mismatch")
    if seal.get("baseline_rev") != context.get("git", {}).get("revision"):
        blockers.append("baseline_seal_revision_mismatch")
    if seal.get("base_rev") != PRE_ARCHITECTURE_REV:
        blockers.append("baseline_seal_base_mismatch")
    if seal.get("dirty") is not False:
        blockers.append("baseline_seal_dirty")
    return seal, blockers


def run_compare(
    *,
    baseline: str | Path,
    optimized: str | Path,
    baseline_seal: str | Path | None,
    out: str | Path,
    fresh: bool,
    require_chip: str | None = None,
    require_low_power: bool = False,
    require_matched_power_source: bool = False,
    require_admitted: bool = False,
    require_speedup: float | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Compare two admitted fixed-density reports and publish the result."""

    _require_absent_output(out)
    baseline_report = _load_report(baseline)
    optimized_report = _load_report(optimized)
    blockers: list[str] = []
    producer_sources = build_source_fingerprints(repo_root)
    producer_git = collect_git_provenance(repo_root)
    if producer_git.get("dirty") is not False:
        blockers.append("comparison_producer_checkout_dirty")
    baseline_identity = baseline_report.get("identity", {})
    optimized_identity = optimized_report.get("identity", {})
    for field in ("workload_fingerprint", "protocol_fingerprint"):
        if baseline_identity.get(field) != optimized_identity.get(field):
            blockers.append(f"comparison_{field}_mismatch")
    if producer_sources["protocol_fingerprint"] != optimized_identity.get(
        "protocol_fingerprint"
    ):
        blockers.append("comparison_producer_protocol_mismatch")
    if producer_sources["runtime_fingerprint"] != optimized_identity.get(
        "runtime_fingerprint"
    ):
        blockers.append("comparison_optimized_runtime_mismatch")
    if baseline_report.get("comparison_protocol") != optimized_report.get(
        "comparison_protocol"
    ):
        blockers.append("comparison_protocol_mismatch")
    for label, report in (("baseline", baseline_report), ("optimized", optimized_report)):
        if not _report_is_formally_admitted(report):
            blockers.append(f"{label}_not_admitted")
        if len(report.get("samples", [])) != 5:
            blockers.append(f"{label}_sample_count_mismatch")
        protocol = report.get("run_protocol", {})
        if (
            protocol.get("warmups") != 1
            or protocol.get("samples") != 5
            or protocol.get("fresh") is not True
            or protocol.get("resumed") is not False
        ):
            blockers.append(f"{label}_cadence_mismatch")
    loaded_baseline_seal: dict[str, object] | None = None
    if baseline_seal is None:
        blockers.append("baseline_seal_not_supplied")
    else:
        loaded_baseline_seal, lineage_blockers = _baseline_lineage_blockers(
            baseline_seal,
            baseline_report,
        )
        blockers.extend(lineage_blockers)
    if loaded_baseline_seal is not None and optimized_report.get(
        "baseline_seal_fingerprint"
    ) != loaded_baseline_seal.get("seal_fingerprint"):
        blockers.append("optimized_baseline_seal_lineage_mismatch")
    for label, report in (("baseline", baseline_report), ("optimized", optimized_report)):
        host = report.get("host", {})
        if require_chip is not None and host.get("chip") != require_chip:
            blockers.append(f"{label}_chip_mismatch")
        if require_low_power and host.get("low_power_mode") != 1:
            blockers.append(f"{label}_low_power_mode_mismatch")
    baseline_source = baseline_report.get("host", {}).get("power_source")
    optimized_source = optimized_report.get("host", {}).get("power_source")
    if require_matched_power_source and baseline_source != optimized_source:
        blockers.append("power_source_mismatch")
    baseline_median = baseline_report.get("summary", {}).get("median_elapsed_seconds")
    optimized_median = optimized_report.get("summary", {}).get("median_elapsed_seconds")
    speedup = _ratio(baseline_median, optimized_median)
    _apply_metric_gate(blockers, "speedup", speedup, require_speedup)
    if not fresh:
        blockers.append("fresh_namespace_not_requested")
    unique = sorted(set(blockers))
    comparison_contract = {
        "schema_version": "mlx-atomistic.dft-runtime-comparison-contract.v1",
        "workload_fingerprint": optimized_identity.get("workload_fingerprint"),
        "protocol_fingerprint": producer_sources["protocol_fingerprint"],
        "runtime_fingerprint": producer_sources["runtime_fingerprint"],
        "baseline_report_fingerprint": baseline_report.get("report_fingerprint"),
        "optimized_report_fingerprint": optimized_report.get("report_fingerprint"),
        "require_admitted_requested": require_admitted,
    }
    identity = {
        "workload_fingerprint": comparison_contract["workload_fingerprint"],
        "protocol_fingerprint": comparison_contract["protocol_fingerprint"],
        "runtime_fingerprint": comparison_contract["runtime_fingerprint"],
        "execution_contract_fingerprint": sha256_bytes(
            canonical_json_bytes(comparison_contract)
        ),
    }
    statuses = {
        "numerical_status": (
            "passed"
            if baseline_report.get("statuses", {}).get("numerical_status") == "passed"
            and optimized_report.get("statuses", {}).get("numerical_status") == "passed"
            else "blocked"
        ),
        "resume_integrity_status": "fresh-no-resume",
        "timing_admission_status": "admitted" if not unique else "blocked",
    }
    command_admission = {"passed": not unique, "blockers": unique}
    report = _finalize_report(
        {
            "schema_version": COMPARISON_SCHEMA,
            "kind": "fixed-density-comparison",
            "identity": identity,
            "producer_context": {
                "comparison_contract": comparison_contract,
                "protocol_inventory": producer_sources["protocol_inventory"],
                "runtime_inventory": producer_sources["runtime_inventory"],
                "git": producer_git,
            },
            "run_protocol": {
                "fresh": fresh,
                "resumed": False,
                "diagnostic": False,
            },
            "baseline_identity": baseline_identity,
            "optimized_identity": optimized_identity,
            "matched_protocol": baseline_report.get("comparison_protocol"),
            "summary": {
                "baseline_median_elapsed_seconds": baseline_median,
                "optimized_median_elapsed_seconds": optimized_median,
                "speedup": speedup,
                "baseline_power_source": baseline_source,
                "optimized_power_source": optimized_source,
            },
            "statuses": statuses,
            "admission": command_admission,
            "formal_admission": _formal_admission(
                statuses=statuses,
                command_admission=command_admission,
                producer_git=producer_git,
                run_protocol={
                    "fresh": fresh,
                    "resumed": False,
                    "diagnostic": False,
                },
                report_kind="fixed-density-comparison",
                host_protocol=baseline_report.get("comparison_protocol"),
            ),
        }
    )
    published = _publish_report(
        out=out,
        artifact_kind="dft-runtime-comparison",
        artifact_schema=COMPARISON_SCHEMA,
        report=report,
    )
    return {**published, "report_payload": report}


def _validate_full_scf_publication_binding(
    *,
    artifact_root: Path,
    artifact_generation: Mapping[str, object],
    artifact_report: Mapping[str, object],
    attestation_root: Path,
    attestation_generation: Mapping[str, object],
    attestation_report: Mapping[str, object],
) -> None:
    published = attestation_report.get("published_artifact")
    elapsed = published.get("elapsed_seconds") if isinstance(published, dict) else None
    maximum = (
        published.get("maximum_elapsed_seconds") if isinstance(published, dict) else None
    )
    run_protocol = attestation_report.get("run_protocol")
    summary = attestation_report.get("summary")
    statuses = attestation_report.get("statuses")
    admission = attestation_report.get("admission")
    subject_identity = artifact_report.get("identity")
    subject_admission = artifact_report.get("admission")
    attestation_identity = attestation_report.get("identity")
    expected_execution_fingerprint = (
        sha256_bytes(
            canonical_json_bytes(
                {
                    "source_execution_contract_fingerprint": subject_identity.get(
                        "execution_contract_fingerprint"
                    ),
                    "publication_contract": published,
                }
            )
        )
        if isinstance(subject_identity, dict) and isinstance(published, dict)
        else None
    )
    subject_blockers = (
        subject_admission.get("blockers")
        if isinstance(subject_admission, dict)
        and isinstance(subject_admission.get("blockers"), list)
        else None
    )
    expected_blockers = (
        sorted(
            set(
                [str(blocker) for blocker in subject_blockers]
                + (
                    ["elapsed_time_gate_failed"]
                    if isinstance(elapsed, int | float)
                    and isinstance(maximum, int | float)
                    and float(elapsed) > float(maximum)
                    else []
                )
            )
        )
        if subject_blockers is not None
        else None
    )
    expected_timing_status = (
        "diagnostic"
        if isinstance(run_protocol, dict) and run_protocol.get("diagnostic") is True
        else "admitted"
        if expected_blockers == []
        else "blocked"
    )
    if (
        artifact_report.get("kind") != "full-scf"
        or artifact_report.get("schema_version") != FULL_SCF_SCHEMA
        or artifact_generation.get("artifact_kind") != "dft-runtime-full-scf"
        or artifact_generation.get("artifact_schema_version") != FULL_SCF_SCHEMA
        or artifact_generation.get("identity") != artifact_report.get("identity")
        or not isinstance(artifact_generation.get("metadata"), dict)
        or artifact_generation.get("metadata", {}).get("admission")
        != artifact_report.get("admission")
        or artifact_generation.get("metadata", {}).get("formal_admission")
        != artifact_report.get("formal_admission")
        or artifact_report.get("publication_attestation_name") != attestation_root.name
        or attestation_report.get("kind") != "full-scf-publication-attestation"
        or attestation_report.get("schema_version") != FULL_SCF_PUBLICATION_SCHEMA
        or attestation_generation.get("artifact_kind")
        != "dft-runtime-full-scf-publication"
        or attestation_generation.get("artifact_schema_version")
        != FULL_SCF_PUBLICATION_SCHEMA
        or attestation_generation.get("identity") != attestation_report.get("identity")
        or not isinstance(attestation_generation.get("metadata"), dict)
        or attestation_generation.get("metadata", {}).get("admission")
        != attestation_report.get("admission")
        or attestation_generation.get("metadata", {}).get("formal_admission")
        != attestation_report.get("formal_admission")
        or attestation_report.get("context") != artifact_report.get("context")
        or attestation_report.get("host") != artifact_report.get("host")
        or run_protocol != artifact_report.get("run_protocol")
        or not isinstance(published, dict)
        or published.get("artifact_name") != artifact_root.name
        or published.get("artifact_manifest_sha256")
        != artifact_generation.get("manifest_sha256")
        or published.get("artifact_report_fingerprint")
        != artifact_report.get("report_fingerprint")
        or published.get("schema_version") != FULL_SCF_PUBLICATION_SCHEMA
        or not isinstance(elapsed, int | float)
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
        or not isinstance(maximum, int | float)
        or not math.isfinite(float(maximum))
        or float(maximum) <= 0.0
        or not isinstance(run_protocol, dict)
        or run_protocol.get("timeout_seconds") != maximum
        or artifact_report.get("run_protocol", {}).get("timeout_seconds") != maximum
        or not isinstance(summary, dict)
        or summary.get("elapsed_seconds") != elapsed
        or not isinstance(statuses, dict)
        or not isinstance(admission, dict)
        or not isinstance(admission.get("blockers"), list)
        or not isinstance(subject_admission, dict)
        or not isinstance(subject_admission.get("blockers"), list)
        or admission
        != {
            "passed": expected_blockers == [],
            "blockers": expected_blockers,
        }
        or statuses.get("timing_admission_status") != expected_timing_status
        or (
            float(elapsed) > float(maximum)
            and (
                admission.get("passed") is not False
                or "elapsed_time_gate_failed" not in admission.get("blockers", ())
                or statuses.get("timing_admission_status") == "admitted"
            )
        )
        or (
            statuses.get("timing_admission_status") == "admitted"
            and float(elapsed) > float(maximum)
        )
        or not isinstance(subject_identity, dict)
        or not isinstance(attestation_identity, dict)
        or attestation_identity.get("execution_contract_fingerprint")
        != expected_execution_fingerprint
        or summary.get("numerical_passed")
        != artifact_report.get("summary", {}).get("numerical_passed")
        or summary.get("success") != artifact_report.get("summary", {}).get("success")
        or statuses.get("numerical_status")
        != (
            "passed"
            if artifact_report.get("summary", {}).get("numerical_passed") is True
            else "blocked"
        )
        or statuses.get("resume_integrity_status")
        != artifact_report.get("statuses", {}).get("resume_integrity_status")
        or any(
            attestation_identity.get(field) != subject_identity.get(field)
            for field in (
                "workload_fingerprint",
                "protocol_fingerprint",
                "runtime_fingerprint",
            )
        )
    ):
        msg = "full-SCF publication attestation does not bind its artifact"
        raise ValueError(msg)


def inspect_artifact(
    *,
    artifact: str | Path,
    integrity_only: bool = False,
    require_current_protocol_match: bool = False,
    require_current_runtime_match: bool = False,
    require_admitted: bool = False,
    require_numerical: bool = False,
    require_speedup: float | None = None,
    max_elapsed_seconds: float | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Strictly read and validate an artifact without repairing or publishing it."""

    if integrity_only and any(
        (
            require_current_protocol_match,
            require_current_runtime_match,
            require_admitted,
            require_numerical,
            require_speedup is not None,
            max_elapsed_seconds is not None,
        )
    ):
        msg = "integrity_only cannot be combined with admission or metric gates"
        raise ValueError(msg)
    if max_elapsed_seconds is not None and (
        not math.isfinite(max_elapsed_seconds) or max_elapsed_seconds <= 0.0
    ):
        msg = "max_elapsed_seconds must be finite and positive"
        raise ValueError(msg)
    generation = inspect_generation(artifact)
    root = generation_root(artifact)
    if integrity_only:
        return {
            "status": "ok",
            "passed": True,
            "integrity": "passed",
            "artifact_kind": generation["artifact_kind"],
            "identity": generation.get("identity"),
            "blockers": [],
        }
    report_path = root / "report.json"
    seal_path = root / "seal.json"
    workload_path = root / "manifest.json"
    if report_path.is_file():
        payload = _load_report(root)
        if seal_path.is_file():
            _load_seal(root)
        if payload.get("kind") == "full-scf":
            attestation_name = payload.get("publication_attestation_name")
            if (
                not isinstance(attestation_name, str)
                or not attestation_name
                or Path(attestation_name).name != attestation_name
            ):
                msg = "full-SCF publication attestation name is invalid"
                raise ValueError(msg)
            attestation_root = root.parent / attestation_name
            if attestation_root.exists() or attestation_root.is_symlink():
                attestation_generation = inspect_generation(attestation_root)
                attestation = _load_report(attestation_root)
                _validate_full_scf_publication_binding(
                    artifact_root=root,
                    artifact_generation=generation,
                    artifact_report=payload,
                    attestation_root=attestation_root,
                    attestation_generation=attestation_generation,
                    attestation_report=attestation,
                )
                generation = attestation_generation
                payload = attestation
        elif payload.get("kind") == "full-scf-publication-attestation":
            published = payload.get("published_artifact")
            artifact_name = (
                published.get("artifact_name") if isinstance(published, dict) else None
            )
            if (
                not isinstance(artifact_name, str)
                or not artifact_name
                or Path(artifact_name).name != artifact_name
            ):
                msg = "full-SCF published artifact name is invalid"
                raise ValueError(msg)
            artifact_root = root.parent / artifact_name
            artifact_generation = inspect_generation(artifact_root)
            artifact_report = _load_report(artifact_root)
            _validate_full_scf_publication_binding(
                artifact_root=artifact_root,
                artifact_generation=artifact_generation,
                artifact_report=artifact_report,
                attestation_root=root,
                attestation_generation=generation,
                attestation_report=payload,
            )
    elif seal_path.is_file():
        payload = _load_seal(root)
    elif workload_path.is_file():
        from mlx_atomistic.benchmarks.dft_runtime_contract import (
            TARGET_ID,
            WORKLOAD_SCHEMA,
            _validate_workload_invariants,
            workload_fingerprint,
        )

        payload = json.loads(workload_path.read_bytes())
        if not isinstance(payload, dict):
            msg = "DFT runtime workload must be a JSON object"
            raise ValueError(msg)
        if (
            payload.get("schema_version") != WORKLOAD_SCHEMA
            or payload.get("target_id") != TARGET_ID
        ):
            msg = "unsupported DFT runtime workload schema or target"
            raise ValueError(msg)
        unsigned = {
            key: value for key, value in payload.items() if key != "workload_fingerprint"
        }
        if payload.get("workload_fingerprint") != workload_fingerprint(unsigned):
            msg = "DFT runtime workload fingerprint mismatch"
            raise ValueError(msg)
        identity = generation.get("identity")
        if (
            generation.get("artifact_kind") != "dft-runtime-workload"
            or generation.get("artifact_schema_version") != WORKLOAD_SCHEMA
            or not isinstance(identity, dict)
            or identity.get("workload_fingerprint")
            != payload.get("workload_fingerprint")
        ):
            msg = "DFT runtime workload does not match its generation envelope"
            raise ValueError(msg)
        resources = payload.get("resources")
        resource_path = root / "resources/Si-GTH-PBE-q4.gth"
        if (
            not isinstance(resources, list)
            or len(resources) != 1
            or not isinstance(resources[0], dict)
            or resources[0].get("role") != "si_gth_pbe_q4"
            or resource_path.is_symlink()
            or not resource_path.is_file()
            or resources[0].get("byte_size") != resource_path.stat().st_size
            or resources[0].get("sha256") != sha256_file(resource_path)
        ):
            msg = "published GTH resource does not match the workload"
            raise ValueError(msg)
        _validate_workload_invariants(payload)
    else:
        msg = "artifact has no recognized inspectable payload"
        raise ValueError(msg)
    blockers: list[str] = []
    identity = payload.get("identity", generation.get("identity", {}))
    if require_current_protocol_match or require_current_runtime_match:
        current = build_source_fingerprints(repo_root)
        if require_current_protocol_match and identity.get("protocol_fingerprint") != current.get(
            "protocol_fingerprint"
        ):
            blockers.append("current_protocol_mismatch")
        if require_current_runtime_match and identity.get("runtime_fingerprint") != current.get(
            "runtime_fingerprint"
        ):
            blockers.append("current_runtime_mismatch")
    if require_admitted and not _report_is_formally_admitted(payload):
        blockers.append("artifact_not_admitted")
    if require_numerical:
        statuses = payload.get("statuses", {})
        if statuses.get("numerical_status") != "passed":
            blockers.append("artifact_numerical_status_blocked")
    speedup = payload.get("summary", {}).get("speedup")
    if speedup is None:
        speedup = payload.get("summary", {}).get("metrics_against_seal", {}).get("speedup")
    _apply_metric_gate(blockers, "speedup", speedup, require_speedup)
    if max_elapsed_seconds is not None:
        publication_gate = payload.get("publication_gate")
        if isinstance(publication_gate, dict):
            enforced_maximum = publication_gate.get("maximum_elapsed_seconds")
            if (
                publication_gate.get("passed") is not True
                or not isinstance(enforced_maximum, int | float)
                or float(enforced_maximum) > max_elapsed_seconds
            ):
                blockers.append("elapsed_time_gate_failed")
        else:
            elapsed = payload.get("summary", {}).get("elapsed_seconds")
            if not isinstance(elapsed, int | float) or elapsed > max_elapsed_seconds:
                blockers.append("elapsed_time_gate_failed")
    unique = sorted(set(blockers))
    return {
        "status": "ok" if not unique else "blocked",
        "passed": not unique,
        "integrity": "passed",
        "artifact_kind": generation["artifact_kind"],
        "identity": identity,
        "blockers": unique,
    }


def run_ladder(
    *,
    manifest_path: str | Path,
    gth_source: str | Path,
    out: str | Path,
    rungs: Sequence[int],
    fresh: bool,
    require_chip: str | None = None,
    require_low_power: bool = False,
    require_success: bool = False,
    allow_failed_rung: bool = False,
    progress: ProgressCallback | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Run the measured fixed-density engineering ladder in dependency order."""

    _require_absent_output(out)
    manifest, _selected = load_workload(manifest_path, gth_source=gth_source)
    host = collect_host_provenance()
    context = build_execution_context(
        manifest=manifest,
        command_kind="ladder",
        host=host,
        repo_root=repo_root,
        settings_override={"rungs": list(rungs)},
    )
    blockers: list[str] = []
    if require_chip is not None or require_low_power:
        blockers.extend(
            host_admission(
                host,
                required_chip=require_chip,
                require_low_power=require_low_power,
            )["blockers"]
        )
    if not fresh:
        blockers.append("fresh_namespace_not_requested")
    ladder_by_size = {
        int(entry["fft_shape"][0]): entry for entry in manifest["engineering_ladder"]
    }
    if list(rungs) != sorted(set(int(value) for value in rungs)):
        blockers.append("ladder_rungs_not_strictly_increasing")
    if any(int(value) not in ladder_by_size for value in rungs):
        blockers.append("unknown_ladder_rung")
    rows: list[dict[str, object]] = []
    final_state: Mapping[str, object] | None = None
    if not blockers:
        for rung_index, size in enumerate(rungs):
            config = ladder_by_size[int(size)]
            try:
                sample, state = _fixed_density_sample(
                    manifest=manifest,
                    gth_source=gth_source,
                    progress=_progress_wrapper(
                        progress,
                        sample_kind="ladder",
                        sample_index=rung_index,
                    ),
                    grid_shape=config["fft_shape"],
                    cutoff_hartree=float(config["cutoff_hartree"]),
                    reduced_kpoint=config["reduced_kpoint"],
                    band_count=int(config["band_count"]),
                )
            except Exception as error:
                sample = {
                    "status": "blocked",
                    "numerical_passed": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                state = None
            process_bytes = sample.get("observation", {}).get("memory", {}).get(
                "process_high_water_bytes"
            )
            elapsed = sample.get("wall_elapsed_seconds")
            oracle = config["oracle"]
            oracle_passed = bool(sample.get("numerical_passed")) and (
                isinstance(sample.get("max_residual"), int | float)
                and float(sample["max_residual"]) <= float(oracle["maximum_residual"])
                and isinstance(sample.get("orthonormality_error"), int | float)
                and float(sample["orthonormality_error"])
                <= float(oracle["maximum_orthonormality_error"])
            )
            measured_gates = {
                "numerical": oracle_passed,
                "elapsed": isinstance(elapsed, int | float)
                and float(elapsed) <= float(config["max_elapsed_seconds"]),
                "memory": process_bytes is None
                or int(process_bytes) <= int(config["max_process_bytes"]),
            }
            row = {
                "rung": int(size),
                "config": config,
                "sample": sample,
                "measured_gates": measured_gates,
                "passed": all(measured_gates.values()),
                "override_used": bool(allow_failed_rung and not all(measured_gates.values())),
                "measurements": {
                    "active_full_ratio": (
                        float(sample["active_count"]) / float(sample["full_grid_count"])
                        if isinstance(sample.get("active_count"), int | float)
                        and isinstance(sample.get("full_grid_count"), int | float)
                        and float(sample["full_grid_count"]) > 0.0
                        else None
                    ),
                    "work_counters": sample.get("observation", {}).get(
                        "work_counters", {}
                    ),
                    "memory": sample.get("observation", {}).get("memory", {}),
                },
                "projected_next_rung": None,
            }
            rows.append(row)
            if row["passed"]:
                final_state = state
            else:
                blockers.append(f"ladder_rung_{size}_failed")
                if not allow_failed_rung:
                    break
            if rung_index + 1 < len(rungs):
                next_size = int(rungs[rung_index + 1])
                next_config = ladder_by_size[next_size]
                current_grid = int(np.prod(config["fft_shape"]))
                next_grid = int(np.prod(next_config["fft_shape"]))
                scale = next_grid / current_grid
                projected_elapsed = (
                    float(elapsed) * scale if isinstance(elapsed, int | float) else None
                )
                projected_process = (
                    int(math.ceil(int(process_bytes) * scale))
                    if isinstance(process_bytes, int | float)
                    else None
                )
                launch_admitted = bool(row["passed"]) and (
                    projected_elapsed is not None
                    and projected_elapsed <= float(next_config["max_elapsed_seconds"])
                    and (
                        projected_process is None
                        or projected_process <= int(next_config["max_process_bytes"])
                    )
                )
                row["projected_next_rung"] = {
                    "rung": next_size,
                    "grid_work_scale": scale,
                    "elapsed_seconds": projected_elapsed,
                    "process_bytes": projected_process,
                    "launch_admitted": launch_admitted,
                    "override_used": bool(allow_failed_rung and not launch_admitted),
                }
                if not launch_admitted and not allow_failed_rung:
                    blockers.append(f"ladder_rung_{next_size}_projection_blocked")
                    break
    if require_success and (
        len(rows) != len(rungs) or any(not bool(row["passed"]) for row in rows)
    ):
        blockers.append("ladder_success_gate_failed")
    unique = sorted(set(str(item) for item in blockers))
    numerical_passed = bool(rows) and len(rows) == len(rungs) and all(
        row.get("sample", {}).get("numerical_passed") is True for row in rows
    )
    identity = {
        "workload_fingerprint": manifest["workload_fingerprint"],
        "protocol_fingerprint": context["protocol_fingerprint"],
        "runtime_fingerprint": context["runtime_fingerprint"],
        "execution_contract_fingerprint": context["execution_contract_fingerprint"],
    }
    statuses = {
        "numerical_status": "passed" if numerical_passed else "blocked",
        "resume_integrity_status": "fresh-no-resume",
        "timing_admission_status": "diagnostic",
    }
    command_admission = {"passed": not unique, "blockers": unique}
    report = _finalize_report(
        {
            "schema_version": LADDER_SCHEMA,
            "kind": "engineering-ladder",
            "identity": identity,
            "context": context,
            "host": host,
            "run_protocol": {
                "fresh": fresh,
                "resumed": False,
                "diagnostic": True,
                "rungs": list(rungs),
            },
            "rows": rows,
            "summary": {
                "requested_rungs": list(rungs),
                "completed_rungs": [row["rung"] for row in rows],
                "final_rung": rows[-1]["rung"] if rows else None,
            },
            "statuses": statuses,
            "admission": command_admission,
            "formal_admission": _formal_admission(
                statuses=statuses,
                command_admission=command_admission,
                producer_git=context["git"],
                run_protocol={
                    "fresh": fresh,
                    "resumed": False,
                    "diagnostic": True,
                    "rungs": list(rungs),
                },
                report_kind="engineering-ladder",
                host_protocol=host,
            ),
        }
    )
    extra_files: dict[str, bytes] = {}
    if (
        final_state is not None
        and rows
        and rows[-1]["passed"]
        and rows[-1]["rung"] == int(rungs[-1])
    ):
        from mlx_atomistic.dft.runtime_state import serialize_fixed_density_state

        extra_files = {
            f"{int(rungs[-1])}/final-state/{path}": payload
            for path, payload in serialize_fixed_density_state(final_state).items()
        }
    published = _publish_report(
        out=out,
        artifact_kind="dft-runtime-ladder",
        artifact_schema=LADDER_SCHEMA,
        report=report,
        extra_files=extra_files,
    )
    return {**published, "report_payload": report}


def _full_scf_science(
    *,
    manifest_path: str,
    gth_source: str,
    state_root: str,
    progress: ProgressCallback | None,
) -> dict[str, object]:
    import mlx.core as mx

    from mlx_atomistic.dft import (
        KPoint,
        KPointMesh,
        PeriodicDFTSystem,
        PeriodicSCFConfig,
        read_gth,
        run_periodic_scf,
    )
    from mlx_atomistic.dft._runtime_observer import RuntimeObserver
    from mlx_atomistic.dft.runtime_state import serialize_periodic_scf_state

    observer = RuntimeObserver(callback=progress, detail_events=False)
    start = time.perf_counter()
    try:
        observer.emit("setup", status="started", stage="full_scf_worker")
        with observer.phase("setup"):
            manifest, _selected = load_workload(
                manifest_path,
                gth_source=gth_source,
            )
            system_values = manifest["system"]
            physics = manifest["physics"]
            lattice = float(system_values["lattice_constant_bohr"])
            fractional = np.asarray(
                system_values["fractional_positions"],
                dtype=np.float64,
            )
            positions = fractional * lattice
            pseudo = read_gth(gth_source, element=GTH_ELEMENT, name=GTH_NAME)
            system = PeriodicDFTSystem(
                (lattice, lattice, lattice),
                physics["fft_shape"],
                positions,
                pseudo,
                electron_count=float(system_values["electron_count"]),
            )
            mesh = KPointMesh(
                [
                    KPoint(
                        point["reduced_coordinates"],
                        weight=float(point["weight"]["numerator"])
                        / float(point["weight"]["denominator"]),
                        coordinate_system="reduced",
                    )
                    for point in physics["kpoints"]
                ]
            )
            scf = manifest["solver"]["scf"]
            config = PeriodicSCFConfig(
                max_iterations=int(scf["max_iterations"]),
                min_iterations=int(scf["min_iterations"]),
                density_tolerance=float(scf["density_tolerance"]),
                energy_tolerance=float(scf["energy_tolerance_hartree"]),
                orbital_tolerance=float(scf["orbital_tolerance"]),
                mixing_beta=float(scf["mixing_beta"]),
                mixer=str(scf["mixer"]),
                adaptive_eigensolver_tolerance=bool(
                    scf["adaptive_eigensolver_tolerance"]
                ),
                initial_eigensolver_tolerance=float(
                    scf["initial_eigensolver_tolerance"]
                ),
                eigensolver_tolerance_scale=float(
                    scf["eigensolver_tolerance_scale"]
                ),
                davidson=_davidson_config(manifest),
            )
        observer.emit("setup", status="completed", stage="full_scf_worker")
        result = run_periodic_scf(
            system,
            cutoff_hartree=float(physics["kinetic_cutoff_hartree"]),
            kpoint_mesh=mesh,
            n_bands=int(system_values["occupied_band_count"]),
            config=config,
            observer=observer,
        )
        mx.synchronize()
        electron_error = abs(result.electron_count - float(system_values["electron_count"]))
        maximum_orthonormality = max(
            item.eigen.orthonormality_error for item in result.kpoints
        )
        numerical_passed = bool(
            result.converged
            and electron_error
            <= float(manifest["numerical_gates"]["electron_count_abs_per_cell"])
            and maximum_orthonormality
            <= float(manifest["numerical_gates"]["orthonormality_max"])
        )
        observer.emit("persistence", status="started", stage="full_scf_state")
        with observer.phase("persistence"):
            state_directory = Path(state_root) / "final-state"
            for relative, payload in serialize_periodic_scf_state(result).items():
                destination = state_directory / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
        observer.emit("persistence", status="completed", stage="full_scf_state")
        return {
            "status": "ok" if numerical_passed else "blocked",
            "numerical_passed": numerical_passed,
            "elapsed_seconds": time.perf_counter() - start,
            "result": result.to_dict(),
            "electron_count_error": electron_error,
            "maximum_orthonormality_error": maximum_orthonormality,
            "observation": observer.snapshot(),
        }
    except Exception as error:
        observer.emit(
            "failure",
            stage="full_scf",
            error_type=type(error).__name__,
            message=str(error),
        )
        raise


def _full_scf_worker(
    queue: Any,
    manifest_path: str,
    gth_source: str,
    state_root: str,
) -> None:
    isolated = False
    try:
        if hasattr(os, "setsid"):
            os.setsid()
            isolated = True
        queue.put({"type": "ready", "process_group_isolated": isolated})

        def progress(event: dict[str, object]) -> None:
            queue.put({"type": "progress", "event": event})

        result = _full_scf_science(
            manifest_path=manifest_path,
            gth_source=gth_source,
            state_root=state_root,
            progress=progress,
        )
        queue.put({"type": "result", "result": result})
    except BaseException as error:
        queue.put(
            {
                "type": "error",
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )


def _terminate_process(process: mp.Process, *, isolated_group: bool) -> bool:
    process_group: int | None = None
    if process.pid is not None and hasattr(os, "killpg"):
        if process.is_alive():
            with suppress(PermissionError, ProcessLookupError):
                os.kill(process.pid, signal.SIGSTOP)
            with suppress(PermissionError, ProcessLookupError):
                isolated_group = os.getpgid(process.pid) == process.pid
        if isolated_group or not process.is_alive():
            process_group = process.pid
    if process_group is not None:
        with suppress(PermissionError, ProcessLookupError):
            os.killpg(process_group, signal.SIGTERM)
    elif process.is_alive():
        process.kill()
    process.join(timeout=2.0)
    if process_group is not None:
        with suppress(PermissionError, ProcessLookupError):
            os.killpg(process_group, signal.SIGKILL)
    elif process.is_alive():
        process.kill()
    process.join(timeout=2.0)
    return process_group is not None


def supervise_full_scf_worker(
    *,
    manifest_path: str | Path,
    gth_source: str | Path,
    state_root: str | Path,
    timeout_seconds: float,
    deadline_monotonic: float | None = None,
    progress: ProgressCallback | None = None,
    worker: Callable[..., None] = _full_scf_worker,
) -> dict[str, object]:
    """Run a full-SCF worker in an isolated process group with hard cleanup."""

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        msg = "timeout_seconds must be positive"
        raise ValueError(msg)
    if deadline_monotonic is not None and not math.isfinite(deadline_monotonic):
        msg = "deadline_monotonic must be finite"
        raise ValueError(msg)
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=worker,
        args=(queue, str(manifest_path), str(gth_source), str(state_root)),
        daemon=False,
    )
    deadline = (
        time.monotonic() + timeout_seconds
        if deadline_monotonic is None
        else deadline_monotonic
    )
    process.start()
    events: list[dict[str, object]] = []
    result: dict[str, object] | None = None
    error: dict[str, object] | None = None
    isolated_group = False
    timed_out = False
    cleanup_complete = False
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                timed_out = True
                break
            try:
                message = queue.get(timeout=min(0.1, remaining))
            except queue_module.Empty:
                if not process.is_alive():
                    break
                continue
            message_type = message.get("type")
            if message_type == "ready":
                isolated_group = bool(message.get("process_group_isolated"))
            elif message_type == "progress":
                event = dict(message["event"])
                events.append(event)
                if progress is not None:
                    progress(event)
            elif message_type == "result":
                result = dict(message["result"])
                break
            elif message_type == "error":
                error = dict(message)
                break
        if timed_out:
            terminal_event = {
                "event": "full_scf_timeout",
                "status": "failed",
                "timeout_seconds": timeout_seconds,
            }
            events.append(terminal_event)
            if progress is not None:
                progress(dict(terminal_event))
        if result is not None and not timed_out:
            process.join(timeout=2.0)
        isolated_group = _terminate_process(
            process,
            isolated_group=isolated_group,
        ) or isolated_group
        cleanup_complete = True
    finally:
        if not cleanup_complete and process.pid is not None:
            isolated_group = _terminate_process(
                process,
                isolated_group=isolated_group,
            ) or isolated_group
        queue.close()
        queue.join_thread()
    return {
        "status": (
            "timeout"
            if timed_out
            else "ok"
            if result is not None and error is None and process.exitcode == 0
            else "failed"
        ),
        "timed_out": timed_out,
        "worker_exitcode": process.exitcode,
        "progress_prefix": events,
        "result": result,
        "error": error,
        "worker_alive_after_cleanup": process.is_alive(),
        "process_group_isolated": isolated_group,
    }


def run_full_scf(
    *,
    manifest_path: str | Path,
    gth_source: str | Path,
    out: str | Path,
    fresh: bool,
    timeout_seconds: float,
    diagnostic: bool = False,
    require_clean: bool = False,
    require_chip: str | None = None,
    require_low_power: bool = False,
    require_numerical: bool = False,
    require_success: bool = False,
    progress: ProgressCallback | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Run fresh full SCF in a supervised process and atomically publish evidence."""

    command_started = time.monotonic()
    publication_deadline = command_started + timeout_seconds
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        msg = "timeout_seconds must be positive"
        raise ValueError(msg)
    artifact_destination = Path(out).expanduser().resolve(strict=False)
    publication_destination = artifact_destination.with_name(
        f"{artifact_destination.name}.publication"
    )
    _require_absent_output(artifact_destination)
    _require_absent_output(publication_destination)
    manifest, _selected = load_workload(manifest_path, gth_source=gth_source)
    host = collect_host_provenance()
    context = build_execution_context(
        manifest=manifest,
        command_kind="full-scf",
        host=host,
        repo_root=repo_root,
        settings_override={"timeout_seconds": timeout_seconds},
    )
    blockers: list[str] = []
    if require_chip is not None or require_low_power:
        blockers.extend(
            host_admission(
                host,
                required_chip=require_chip,
                require_low_power=require_low_power,
            )["blockers"]
        )
    if require_clean and context["git"]["dirty"]:
        blockers.append("dirty_checkout")
    if not fresh:
        blockers.append("fresh_namespace_not_requested")
    temporary_state = Path(tempfile.mkdtemp(prefix="mlx-atomistic-full-scf-"))
    supervision: dict[str, object] = {
        "status": "not-started",
        "timed_out": False,
        "progress_prefix": [],
        "result": None,
        "worker_alive_after_cleanup": False,
    }
    try:
        if not blockers:
            remaining = publication_deadline - time.monotonic()
            if remaining <= 0.0:
                supervision = {
                    **supervision,
                    "status": "timeout",
                    "timed_out": True,
                    "progress_prefix": [
                        {
                            "event": "full_scf_timeout",
                            "status": "failed",
                            "timeout_seconds": timeout_seconds,
                            "stage": "pre_worker_setup",
                        }
                    ],
                }
            else:
                supervision = supervise_full_scf_worker(
                    manifest_path=manifest_path,
                    gth_source=gth_source,
                    state_root=temporary_state,
                    timeout_seconds=remaining,
                    deadline_monotonic=publication_deadline,
                    progress=progress,
                )
        result = supervision.get("result")
        numerical_passed = bool(
            isinstance(result, dict) and result.get("numerical_passed") is True
        )
        success = supervision.get("status") == "ok" and numerical_passed
        if supervision.get("timed_out"):
            blockers.append("full_scf_timeout")
        if supervision.get("status") != "ok":
            blockers.append("full_scf_worker_failed")
        if supervision.get("worker_alive_after_cleanup"):
            blockers.append("orphaned_full_scf_worker")
        if not numerical_passed:
            blockers.append("numerical_result_failed")
        if require_numerical and not numerical_passed:
            blockers.append("numerical_gate_failed")
        if require_success and not success:
            blockers.append("full_scf_success_gate_failed")
        identity = {
            "workload_fingerprint": manifest["workload_fingerprint"],
            "protocol_fingerprint": context["protocol_fingerprint"],
            "runtime_fingerprint": context["runtime_fingerprint"],
            "execution_contract_fingerprint": context[
                "execution_contract_fingerprint"
            ],
        }
        run_protocol = {
            "fresh": fresh,
            "warmups": 0,
            "resumed": False,
            "new_process": True,
            "timeout_seconds": timeout_seconds,
            "diagnostic": diagnostic,
            "timing_boundary": (
                "command-entry-through-atomic-report-generation-publication-"
                "and-parent-sync"
            ),
        }
        candidate_statuses = {
            "numerical_status": "passed" if numerical_passed else "blocked",
            "resume_integrity_status": "fresh-no-resume",
            "timing_admission_status": "awaiting-publication-attestation",
        }
        candidate_admission = {
            "passed": not blockers,
            "blockers": sorted(set(str(item) for item in blockers)),
        }
        report: dict[str, object] = {
            "schema_version": FULL_SCF_SCHEMA,
            "kind": "full-scf",
            "identity": identity,
            "context": context,
            "host": host,
            "run_protocol": run_protocol,
            "supervision": supervision,
            "summary": {
                "elapsed_seconds_to_report_write": None,
                "numerical_passed": numerical_passed,
                "success": success,
            },
            "publication_attestation_name": publication_destination.name,
            "statuses": candidate_statuses,
            "admission": candidate_admission,
            "formal_admission": _formal_admission(
                statuses=candidate_statuses,
                command_admission=candidate_admission,
                producer_git=context["git"],
                run_protocol=run_protocol,
                report_kind="full-scf",
                host_protocol=host,
            ),
        }

        def finalize_full_scf_report(payload: dict[str, object]) -> None:
            payload["summary"]["elapsed_seconds_to_report_write"] = (
                time.monotonic() - command_started
            )
            _finalize_report(payload)

        published = _publish_report(
            out=artifact_destination,
            artifact_kind="dft-runtime-full-scf",
            artifact_schema=FULL_SCF_SCHEMA,
            report=report,
            extra_tree=(
                temporary_state if (temporary_state / "final-state").is_dir() else None
            ),
            before_report_write=finalize_full_scf_report,
        )
        elapsed = time.monotonic() - command_started
        final_blockers = list(blockers)
        if elapsed > timeout_seconds:
            final_blockers.append("elapsed_time_gate_failed")
        unique = sorted(set(str(item) for item in final_blockers))
        statuses = {
            "numerical_status": "passed" if numerical_passed else "blocked",
            "resume_integrity_status": "fresh-no-resume",
            "timing_admission_status": (
                "diagnostic"
                if diagnostic
                else "admitted"
                if not unique
                else "blocked"
            ),
        }
        command_admission = {"passed": not unique, "blockers": unique}
        artifact_manifest = inspect_generation(artifact_destination)
        publication_contract = {
            "schema_version": FULL_SCF_PUBLICATION_SCHEMA,
            "artifact_name": artifact_destination.name,
            "artifact_manifest_sha256": artifact_manifest["manifest_sha256"],
            "artifact_report_fingerprint": report["report_fingerprint"],
            "elapsed_seconds": elapsed,
            "maximum_elapsed_seconds": timeout_seconds,
        }
        publication_identity = {
            **identity,
            "execution_contract_fingerprint": sha256_bytes(
                canonical_json_bytes(
                    {
                        "source_execution_contract_fingerprint": identity[
                            "execution_contract_fingerprint"
                        ],
                        "publication_contract": publication_contract,
                    }
                )
            ),
        }
        publication_report = _finalize_report(
            {
                "schema_version": FULL_SCF_PUBLICATION_SCHEMA,
                "kind": "full-scf-publication-attestation",
                "identity": publication_identity,
                "context": context,
                "host": host,
                "run_protocol": run_protocol,
                "published_artifact": publication_contract,
                "summary": {
                    "elapsed_seconds": elapsed,
                    "numerical_passed": numerical_passed,
                    "success": success,
                },
                "statuses": statuses,
                "admission": command_admission,
                "formal_admission": _formal_admission(
                    statuses=statuses,
                    command_admission=command_admission,
                    producer_git=context["git"],
                    run_protocol=run_protocol,
                    report_kind="full-scf-publication-attestation",
                    host_protocol=host,
                ),
            }
        )
        publication = _publish_report(
            out=publication_destination,
            artifact_kind="dft-runtime-full-scf-publication",
            artifact_schema=FULL_SCF_PUBLICATION_SCHEMA,
            report=publication_report,
        )
        return {
            "artifact": published["artifact"],
            "report": published["report"],
            "publication_attestation": publication["artifact"],
            "publication_report": publication["report"],
            "status": publication["status"],
            "admission": publication["admission"],
            "formal_admission": publication["formal_admission"],
            "report_payload": publication_report,
        }
    finally:
        shutil.rmtree(temporary_state, ignore_errors=True)
