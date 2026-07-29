"""Frozen contract and staged evidence helpers for bounded DHFR NPT validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA = "mlx-atomistic.dhfr-npt-validation-contract.v1"
STAGE_REPORT_SCHEMA = "mlx-atomistic.dhfr-npt-stage-report.v1"
FINAL_REPORT_SCHEMA = "mlx-atomistic.dhfr-npt-final-report.v1"
DEFAULT_CONTRACT_PATH = Path(__file__).with_name("data") / (
    "dhfr_npt_validation_contract.json"
)
SOURCE_MANIFEST_NAME = "source_manifest.json"
VALID_STAGES = ("fixed", "npt")
FINAL_TARGET_SCOPE = "openmm-5dfr-final-target"


class DHFRNPTValidationError(RuntimeError):
    """Raised when DHFR NPT evidence violates the frozen validation contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes for fingerprinting.

    Args:
        value: JSON-compatible value to encode.

    Returns:
        Canonical JSON bytes with sorted keys and compact separators.
    """

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def payload_fingerprint(value: Any) -> str:
    """Return the SHA-256 fingerprint of a JSON-compatible payload.

    Args:
        value: JSON-compatible value to fingerprint.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_validation_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    """Load and validate the frozen DHFR NPT contract.

    Args:
        path: Contract JSON path. Defaults to the packaged contract.

    Returns:
        Validated contract payload.

    Raises:
        DHFRNPTValidationError: If the contract is malformed or incomplete.
    """

    contract_path = Path(path)
    try:
        payload = json.loads(contract_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        msg = f"cannot load DHFR NPT contract: {contract_path}"
        raise DHFRNPTValidationError(msg) from error
    if not isinstance(payload, dict) or payload.get("schema") != CONTRACT_SCHEMA:
        msg = "DHFR NPT contract schema is missing or unsupported"
        raise DHFRNPTValidationError(msg)
    _require_contract_shape(payload)
    _require_finite_json(payload, context="contract")
    return payload


def contract_fingerprint(contract: Mapping[str, Any]) -> str:
    """Return the immutable fingerprint of a validated contract.

    Args:
        contract: Loaded validation contract.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """

    return payload_fingerprint(dict(contract))


def validate_source_manifest(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, bool]:
    """Validate exact 5dfr source identity against the frozen contract.

    Args:
        manifest: Prepared artifact source manifest.
        contract: Frozen validation contract.

    Returns:
        Named source-identity checks, all true.

    Raises:
        DHFRNPTValidationError: If any exact identity check fails.
    """

    target = _mapping(contract.get("target"), name="contract target")
    source = _mapping(manifest.get("source"), name="source manifest source")
    identity = _mapping(manifest.get("identity"), name="source manifest identity")
    forces = _mapping(manifest.get("forces"), name="source manifest forces")
    construction = _mapping(
        manifest.get("construction"),
        name="source manifest construction",
    )
    pme = _mapping(manifest.get("pme"), name="source manifest PME")
    pdb = _mapping(source.get("pdb"), name="source PDB record")
    benchmark = _mapping(
        source.get("vendor_benchmark"),
        name="vendor benchmark record",
    )
    resources = {
        str(record.get("resource", "")).rsplit("/", 1)[-1]: record.get("sha256")
        for record in _sequence(
            source.get("forcefield_resources"),
            name="force-field resources",
        )
        if isinstance(record, Mapping)
    }
    expected_resources = _mapping(
        target.get("forcefield_resources"),
        name="contract force-field resources",
    )
    workload_pme = _mapping(
        _mapping(contract.get("workload"), name="contract workload").get("pme"),
        name="contract PME",
    )
    checks = {
        "manifest_schema": (
            manifest.get("schema") == "mlx-atomistic.openmm-5dfr-preparation.v1"
        ),
        "case_id": manifest.get("case_id") == target.get("case_id"),
        "pdb_path": pdb.get("path") == target.get("pdb_path"),
        "pdb_sha256": pdb.get("sha256") == target.get("pdb_sha256"),
        "vendor_benchmark_path": (
            benchmark.get("path") == target.get("vendor_benchmark_path")
        ),
        "vendor_benchmark_sha256": (
            benchmark.get("sha256") == target.get("vendor_benchmark_sha256")
        ),
        "forcefield_resources": resources == dict(expected_resources),
        "atom_count": identity.get("atom_count") == target.get("atom_count"),
        "molecule_count": (
            identity.get("molecule_count") == target.get("molecule_count")
        ),
        "molecule_identity": (
            identity.get("molecule_identity_sha256")
            == target.get("molecule_identity_sha256")
        ),
        "atom_order": (
            identity.get("atom_order_sha256") == target.get("atom_order_sha256")
        ),
        "force_classes": (
            forces.get("classes_sha256") == target.get("force_classes_sha256")
        ),
        "force_parameters": (
            forces.get("parameters_sha256") == target.get("parameters_sha256")
        ),
        "constraints": (
            forces.get("constraints_sha256") == target.get("constraints_sha256")
        ),
        "forcefield_names": (
            construction.get("forcefield_files") == ["amber99sb.xml", "tip3p.xml"]
        ),
        "construction": (
            construction.get("nonbonded_method") == "PME"
            and construction.get("constraints") == "HBonds"
            and construction.get("rigid_water") is True
            and construction.get("hydrogen_mass_amu")
            == _mapping(contract.get("workload"), name="contract workload").get(
                "hydrogen_mass_amu"
            )
        ),
        "pme": all(
            pme.get(name) == expected
            for name, expected in workload_pme.items()
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        msg = "5dfr source identity mismatch: " + ", ".join(blockers)
        raise DHFRNPTValidationError(msg)
    return checks


def validate_prepared_boundary(
    prepared_dir: str | Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a prepared artifact and its exact source manifest.

    Args:
        prepared_dir: Directory containing prepared-system files and manifest.
        contract: Frozen validation contract.

    Returns:
        Source identity payload for embedding in stage reports.

    Raises:
        DHFRNPTValidationError: If manifest or artifact files fail closed.
    """

    base = Path(prepared_dir)
    manifest_path = base / SOURCE_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        msg = f"cannot load exact source manifest: {manifest_path}"
        raise DHFRNPTValidationError(msg) from error
    if not isinstance(manifest, dict):
        msg = "exact source manifest must be a JSON object"
        raise DHFRNPTValidationError(msg)
    claimed_fingerprint = manifest.get("manifest_fingerprint")
    unsigned = dict(manifest)
    unsigned.pop("manifest_fingerprint", None)
    actual_fingerprint = payload_fingerprint(unsigned)
    if claimed_fingerprint != actual_fingerprint:
        msg = "exact source manifest fingerprint does not match its contents"
        raise DHFRNPTValidationError(msg)
    source_checks = validate_source_manifest(manifest, contract)
    artifact_records = _mapping(
        manifest.get("artifact_files"),
        name="source manifest artifact files",
    )
    for name in ("prepared_system.json", "prepared_system.npz", "view.pdb"):
        record = _mapping(
            artifact_records.get(name),
            name=f"artifact file record {name}",
        )
        path = base / name
        if not path.is_file() or path.is_symlink():
            msg = f"prepared artifact file is missing or unsafe: {name}"
            raise DHFRNPTValidationError(msg)
        if path.stat().st_size != int(record.get("byte_size", -1)):
            msg = f"prepared artifact byte size mismatch: {name}"
            raise DHFRNPTValidationError(msg)
        if _file_sha256(path) != record.get("sha256"):
            msg = f"prepared artifact digest mismatch: {name}"
            raise DHFRNPTValidationError(msg)
    identity = _mapping(manifest.get("identity"), name="source manifest identity")
    return {
        "case_id": manifest["case_id"],
        "manifest_fingerprint": actual_fingerprint,
        "contract_fingerprint": contract_fingerprint(contract),
        "atom_count": int(identity["atom_count"]),
        "molecule_count": int(identity["molecule_count"]),
        "molecule_identity_sha256": str(identity["molecule_identity_sha256"]),
        "source_checks": source_checks,
    }


def stage_report_path(
    out_dir: str | Path,
    *,
    stage: str,
    seed: int | None = None,
) -> Path:
    """Return the canonical path for one stage report.

    Args:
        out_dir: Root output directory.
        stage: ``"fixed"`` or ``"npt"``.
        seed: Required for NPT and forbidden for fixed state.

    Returns:
        Canonical report path below ``out_dir``.
    """

    normalized_stage = _validated_stage_seed(stage, seed)
    root = Path(out_dir)
    if normalized_stage == "fixed":
        return root / "fixed" / "report.json"
    return root / "npt" / f"seed-{seed}" / "report.json"


def build_stage_report(
    *,
    contract: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    stage: str,
    seed: int | None,
    scope: str,
    evidence: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    """Build a complete stage report and fail on unsuccessful checks.

    Args:
        contract: Frozen validation contract.
        source_identity: Validated prepared-source identity.
        stage: ``"fixed"`` or ``"npt"``.
        seed: Required for NPT and forbidden for fixed state.
        scope: Evidence scope; final target uses `FINAL_TARGET_SCOPE`.
        evidence: JSON-compatible measurements and provenance.
        checks: Named boolean scientific and integrity checks.

    Returns:
        Complete stage report with immutable identities.

    Raises:
        DHFRNPTValidationError: If checks are empty, non-boolean, or failed.
    """

    normalized_stage = _validated_stage_seed(stage, seed)
    check_payload = dict(checks)
    if not check_payload or any(type(value) is not bool for value in check_payload.values()):
        msg = "stage checks must be a non-empty mapping of booleans"
        raise DHFRNPTValidationError(msg)
    blockers = sorted(name for name, passed in check_payload.items() if not passed)
    report = {
        "schema": STAGE_REPORT_SCHEMA,
        "stage": normalized_stage,
        "seed": seed,
        "scope": str(scope),
        "final_target": scope == FINAL_TARGET_SCOPE,
        "contract_fingerprint": contract_fingerprint(contract),
        "source_manifest_fingerprint": source_identity.get(
            "manifest_fingerprint"
        ),
        "source_identity": dict(source_identity),
        "status": "passed" if not blockers else "failed",
        "checks": check_payload,
        "blockers": blockers,
        "evidence": dict(evidence),
    }
    _require_finite_json(report, context="stage report")
    return report


def write_stage_report_atomic(path: str | Path, report: Mapping[str, Any]) -> None:
    """Atomically publish one validated stage report.

    Args:
        path: Destination report path.
        report: Complete report payload.
    """

    destination = Path(path)
    validate_stage_report(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_completed_stage(
    path: str | Path,
    *,
    contract: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    stage: str,
    seed: int | None,
) -> dict[str, Any] | None:
    """Load an already completed stage only when all identities still match.

    Args:
        path: Canonical stage report path.
        contract: Current frozen contract.
        source_identity: Current validated source identity.
        stage: Expected stage.
        seed: Expected seed.

    Returns:
        Valid passing report, or ``None`` when the path does not exist or contains
        a structurally valid failed attempt for the same workload.

    Raises:
        DHFRNPTValidationError: If an existing report is unsafe, corrupt, or
            belongs to another workload.
    """

    report_path = Path(path)
    if not report_path.exists():
        return None
    if (
        not report_path.is_file()
        or report_path.is_symlink()
        or report_path.stat().st_nlink != 1
    ):
        msg = f"stage report path is not a safe regular file: {report_path}"
        raise DHFRNPTValidationError(msg)
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        msg = f"existing stage report is incomplete or corrupt: {report_path}"
        raise DHFRNPTValidationError(msg) from error
    validate_stage_report(report)
    expected_stage = _validated_stage_seed(stage, seed)
    expected = {
        "stage": expected_stage,
        "seed": seed,
        "contract_fingerprint": contract_fingerprint(contract),
        "source_manifest_fingerprint": source_identity.get(
            "manifest_fingerprint"
        ),
    }
    mismatches = [
        name for name, value in expected.items() if report.get(name) != value
    ]
    if mismatches:
        msg = "existing stage report identity mismatch: " + ", ".join(mismatches)
        raise DHFRNPTValidationError(msg)
    _validate_stage_artifacts(report_path, report)
    if report.get("status") != "passed" or not all(report["checks"].values()):
        return None
    return report


def validate_stage_report(report: Mapping[str, Any]) -> None:
    """Validate the structural and finite-data contract of one stage report.

    Args:
        report: Stage report payload.

    Raises:
        DHFRNPTValidationError: If the report is structurally invalid.
    """

    if not isinstance(report, Mapping) or report.get("schema") != STAGE_REPORT_SCHEMA:
        msg = "stage report schema is missing or unsupported"
        raise DHFRNPTValidationError(msg)
    _validated_stage_seed(str(report.get("stage")), report.get("seed"))
    for name in ("contract_fingerprint", "source_manifest_fingerprint", "scope"):
        if not isinstance(report.get(name), str) or not report[name]:
            msg = f"stage report {name} is missing"
            raise DHFRNPTValidationError(msg)
    source_identity = _mapping(
        report.get("source_identity"),
        name="stage report source identity",
    )
    if (
        source_identity.get("contract_fingerprint")
        != report["contract_fingerprint"]
        or source_identity.get("manifest_fingerprint")
        != report["source_manifest_fingerprint"]
    ):
        msg = "stage report source identity does not reconcile with top-level identity"
        raise DHFRNPTValidationError(msg)
    if report.get("final_target") is not (
        report.get("scope") == FINAL_TARGET_SCOPE
    ):
        msg = "stage report final-target label does not reconcile with scope"
        raise DHFRNPTValidationError(msg)
    checks = report.get("checks")
    if not isinstance(checks, Mapping) or not checks:
        msg = "stage report checks are missing"
        raise DHFRNPTValidationError(msg)
    if any(type(value) is not bool for value in checks.values()):
        msg = "stage report checks must be booleans"
        raise DHFRNPTValidationError(msg)
    blockers = report.get("blockers")
    if not isinstance(blockers, list):
        msg = "stage report blockers must be a list"
        raise DHFRNPTValidationError(msg)
    expected_blockers = sorted(name for name, passed in checks.items() if not passed)
    if sorted(blockers) != expected_blockers:
        msg = "stage report blockers do not reconcile with checks"
        raise DHFRNPTValidationError(msg)
    expected_status = "passed" if not blockers else "failed"
    if report.get("status") != expected_status:
        msg = "stage report status does not reconcile with checks"
        raise DHFRNPTValidationError(msg)
    _require_finite_json(report, context="stage report")


def build_final_report(
    *,
    contract: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    fixed_report: Mapping[str, Any],
    npt_reports: Sequence[Mapping[str, Any]],
    checks: Mapping[str, bool],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine validated target stages without mixing identities.

    Args:
        contract: Frozen validation contract.
        source_identity: Validated exact target identity.
        fixed_report: Passing fixed-state report.
        npt_reports: Passing NPT reports, one for each declared seed.
        checks: Final aggregate checks.
        evidence: Aggregate measurements and limitations.

    Returns:
        Final machine-readable validation report.

    Raises:
        DHFRNPTValidationError: If a stage or identity is missing or mismatched.
    """

    expected_contract = contract_fingerprint(contract)
    expected_source = source_identity.get("manifest_fingerprint")
    expected_seeds = list(
        _sequence(
            _mapping(contract.get("workload"), name="contract workload").get(
                "seeds"
            ),
            name="contract seeds",
        )
    )
    reports = [fixed_report, *npt_reports]
    for report in reports:
        validate_stage_report(report)
        if report.get("status") != "passed" or report.get("scope") != FINAL_TARGET_SCOPE:
            msg = "final report requires passing final-target stages"
            raise DHFRNPTValidationError(msg)
        if (
            report.get("contract_fingerprint") != expected_contract
            or report.get("source_manifest_fingerprint") != expected_source
        ):
            msg = "final report refuses mixed contract or source identities"
            raise DHFRNPTValidationError(msg)
    if fixed_report.get("stage") != "fixed" or fixed_report.get("seed") is not None:
        msg = "final report requires one unseeded fixed stage"
        raise DHFRNPTValidationError(msg)
    actual_seeds = sorted(
        int(report["seed"])
        for report in npt_reports
        if report.get("stage") == "npt"
    )
    if actual_seeds != sorted(int(seed) for seed in expected_seeds):
        msg = "final report does not contain every declared NPT seed"
        raise DHFRNPTValidationError(msg)
    check_payload = dict(checks)
    if not check_payload or any(type(value) is not bool for value in check_payload.values()):
        msg = "final checks must be a non-empty mapping of booleans"
        raise DHFRNPTValidationError(msg)
    blockers = [name for name, passed in check_payload.items() if not passed]
    payload = {
        "schema": FINAL_REPORT_SCHEMA,
        "contract_fingerprint": expected_contract,
        "source_manifest_fingerprint": expected_source,
        "status": "passed" if not blockers else "failed",
        "checks": check_payload,
        "blockers": blockers,
        "stages": reports,
        "evidence": dict(evidence),
        "claim": dict(_mapping(contract.get("claim"), name="contract claim")),
    }
    _require_finite_json(payload, context="final report")
    return payload


def _require_contract_shape(contract: Mapping[str, Any]) -> None:
    required = (
        "claim",
        "target",
        "workload",
        "resource_limits",
        "fixed_state_gates",
        "npt_gates",
        "gate_basis",
    )
    missing = [name for name in required if not isinstance(contract.get(name), Mapping)]
    if missing:
        msg = "DHFR NPT contract sections are missing: " + ", ".join(missing)
        raise DHFRNPTValidationError(msg)
    workload = _mapping(contract["workload"], name="contract workload")
    seeds = _sequence(workload.get("seeds"), name="contract seeds")
    if list(seeds) != [7, 19]:
        msg = "DHFR NPT contract must preserve predeclared seeds 7 and 19"
        raise DHFRNPTValidationError(msg)
    barostat = _mapping(workload.get("barostat"), name="contract barostat")
    if (
        workload.get("steps") != 250
        or workload.get("dt_ps") != 0.001
        or workload.get("temperature_K") != 300.0
        or workload.get("pressure_bar") != 1.0
        or barostat.get("mode") != "anisotropic"
        or barostat.get("interval") != 25
        or barostat.get("axes") != [True, True, True]
    ):
        msg = "DHFR NPT workload drifted from the selected 300 K, 1 bar contract"
        raise DHFRNPTValidationError(msg)
    resources = _mapping(
        contract["resource_limits"],
        name="contract resource limits",
    )
    if resources.get("process_tree_max_bytes") != 40_000_000_000:
        msg = "DHFR NPT process-tree bound must remain 40 GB"
        raise DHFRNPTValidationError(msg)


def _validated_stage_seed(stage: str, seed: int | None) -> str:
    normalized = str(stage).strip().lower()
    if normalized not in VALID_STAGES:
        msg = f"stage must be one of {', '.join(VALID_STAGES)}"
        raise DHFRNPTValidationError(msg)
    if normalized == "fixed" and seed is not None:
        msg = "fixed stage must not declare a seed"
        raise DHFRNPTValidationError(msg)
    if normalized == "npt" and (not isinstance(seed, int) or isinstance(seed, bool)):
        msg = "NPT stage requires an integer seed"
        raise DHFRNPTValidationError(msg)
    return normalized


def _require_finite_json(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite_json(item, context=context)
        return
    if isinstance(value, list | tuple):
        for item in value:
            _require_finite_json(item, context=context)
        return
    if isinstance(value, float) and not math.isfinite(value):
        msg = f"{context} contains a non-finite number"
        raise DHFRNPTValidationError(msg)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        msg = f"{name} must be an object"
        raise DHFRNPTValidationError(msg)
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, list | tuple):
        msg = f"{name} must be an array"
        raise DHFRNPTValidationError(msg)
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_stage_artifacts(
    report_path: Path,
    report: Mapping[str, Any],
) -> None:
    evidence = _mapping(report.get("evidence"), name="stage report evidence")
    artifacts = _mapping(evidence.get("artifacts"), name="stage report artifacts")
    if not artifacts:
        msg = "completed stage report does not declare its artifacts"
        raise DHFRNPTValidationError(msg)
    root = report_path.parent.resolve()
    for name, raw_record in artifacts.items():
        record = _mapping(raw_record, name=f"stage artifact record {name}")
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute() or not relative.parts:
            msg = f"stage artifact path is unsafe: {name}"
            raise DHFRNPTValidationError(msg)
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            msg = f"stage artifact escapes its report directory: {name}"
            raise DHFRNPTValidationError(msg) from error
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            msg = f"stage artifact is missing or unsafe: {name}"
            raise DHFRNPTValidationError(msg)
        if path.stat().st_size != int(record.get("byte_size", -1)):
            msg = f"stage artifact byte size mismatch: {name}"
            raise DHFRNPTValidationError(msg)
        if _file_sha256(path) != record.get("sha256"):
            msg = f"stage artifact digest mismatch: {name}"
            raise DHFRNPTValidationError(msg)


def _verify_command(args: argparse.Namespace) -> int:
    contract = load_validation_contract(args.contract)
    source_identity = validate_prepared_boundary(args.prepared, contract)
    input_path = Path(args.input)
    direct_report = input_path / "report.json"
    stage_directory_name = (
        "fixed" if args.stage == "fixed" else f"seed-{args.seed}"
    )
    path = (
        input_path
        if input_path.is_file()
        else direct_report
        if direct_report.is_file() or input_path.name == stage_directory_name
        else stage_report_path(input_path, stage=args.stage, seed=args.seed)
    )
    report = load_completed_stage(
        path,
        contract=contract,
        source_identity=source_identity,
        stage=args.stage,
        seed=args.seed,
    )
    if report is None:
        msg = f"stage report does not exist: {path}"
        raise DHFRNPTValidationError(msg)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the DHFR NPT evidence-inspection command.

    Args:
        argv: Optional command-line arguments. Defaults to process arguments.

    Returns:
        Process exit status.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    verify.add_argument("--prepared", type=Path, required=True)
    verify.add_argument("--stage", choices=VALID_STAGES, required=True)
    verify.add_argument("--seed", type=int)
    verify.add_argument("--input", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "verify":
        return _verify_command(arguments)
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTRACT_SCHEMA",
    "DEFAULT_CONTRACT_PATH",
    "DHFRNPTValidationError",
    "FINAL_REPORT_SCHEMA",
    "FINAL_TARGET_SCOPE",
    "STAGE_REPORT_SCHEMA",
    "build_final_report",
    "build_stage_report",
    "canonical_json_bytes",
    "contract_fingerprint",
    "load_completed_stage",
    "load_validation_contract",
    "main",
    "payload_fingerprint",
    "stage_report_path",
    "validate_prepared_boundary",
    "validate_source_manifest",
    "validate_stage_report",
    "write_stage_report_atomic",
]
