"""Fail-closed transferability matrix for the periodic GTH runtime."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic._artifact_identity import sha256_bytes

CONTRACT_SCHEMA = "mlx-atomistic.dft-gth-transferability.v1"
CONTRACT_SHA256 = "906685a1ce847009c0a731fd733908684920dc643c442d6bc5b84bca67a7f319"


def _contract_path() -> Path:
    return Path(__file__).with_name("data") / "dft_gth_transferability_contract.json"


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def load_gth_transferability_contract() -> dict[str, Any]:
    """Load the hash-pinned periodic GTH transferability contract."""

    raw = _contract_path().read_bytes()
    if sha256_bytes(raw) != CONTRACT_SHA256:
        raise ValueError("GTH transferability contract hash mismatch")
    payload = json.loads(raw)
    if payload.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("unsupported GTH transferability contract schema")
    return payload


def _metric_result(metric: Mapping[str, Any]) -> dict[str, Any]:
    name = metric.get("name")
    value = metric.get("value")
    minimum = metric.get("minimum")
    maximum = metric.get("maximum")
    if not isinstance(name, str) or not name:
        raise ValueError("transferability metric name must be non-empty")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
    ):
        raise ValueError(f"transferability metric {name} must be finite")
    if minimum is None and maximum is None:
        raise ValueError(f"transferability metric {name} has no bound")
    bounds: dict[str, float] = {}
    for bound_name, bound in (("minimum", minimum), ("maximum", maximum)):
        if bound is None:
            continue
        if (
            isinstance(bound, bool)
            or not isinstance(bound, (int, float))
            or not np.isfinite(float(bound))
        ):
            raise ValueError(f"transferability metric {name} bound must be finite")
        bounds[bound_name] = float(bound)
    observed = float(value)
    passed = (
        ("minimum" not in bounds or observed >= bounds["minimum"])
        and ("maximum" not in bounds or observed <= bounds["maximum"])
    )
    return {"name": name, "value": observed, **bounds, "passed": passed}


def _identity_result(identity: Mapping[str, Any]) -> dict[str, Any]:
    resources = identity.get("pseudopotential_sha256")
    if not isinstance(resources, Mapping) or not resources:
        raise ValueError("transferability identity requires pseudopotential resources")
    if any(
        not isinstance(name, str) or not name or not _is_sha256(digest)
        for name, digest in resources.items()
    ):
        raise ValueError("transferability pseudopotential identity is invalid")
    reference = identity.get("reference_protocol_sha256")
    if not _is_sha256(reference):
        raise ValueError("transferability reference protocol identity is invalid")
    calculation = identity.get("calculation_fingerprint")
    runtime = identity.get("runtime_fingerprint")
    for name, value in (
        ("calculation_fingerprint", calculation),
        ("runtime_fingerprint", runtime),
    ):
        if value is not None and not _is_sha256(value):
            raise ValueError(f"transferability {name} is invalid")
    return {
        "pseudopotential_sha256": dict(resources),
        "reference_protocol_sha256": reference,
        "calculation_fingerprint": calculation,
        "runtime_fingerprint": runtime,
        "complete": calculation is not None and runtime is not None,
    }


def _method_validation_result(validation: Mapping[str, Any]) -> dict[str, Any]:
    scope = validation.get("scope")
    passed = validation.get("passed")
    if not isinstance(scope, str) or not scope:
        raise ValueError("transferability method validation scope is invalid")
    if not isinstance(passed, bool):
        raise ValueError("transferability method validation verdict is invalid")
    raw_metrics = validation.get("metrics", ())
    if (
        not isinstance(raw_metrics, Sequence)
        or isinstance(raw_metrics, (str, bytes))
        or any(not isinstance(metric, Mapping) for metric in raw_metrics)
    ):
        raise ValueError("transferability method validation metrics are invalid")
    metrics = tuple(_metric_result(metric) for metric in raw_metrics)
    if metrics and passed != all(metric["passed"] for metric in metrics):
        raise ValueError("transferability method validation verdict disagrees with metrics")
    return {"scope": scope, "passed": passed, "metrics": list(metrics)}


def _coverage_result(
    requirements: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category, required_values in requirements.items():
        if (
            not isinstance(category, str)
            or not isinstance(required_values, Sequence)
            or isinstance(required_values, (str, bytes))
            or not required_values
        ):
            raise ValueError("transferability coverage requirements are invalid")
        required = {str(value) for value in required_values}
        observed = {
            str(value)
            for case in cases
            for value in case.get("coverage", {}).get(category, ())
        }
        missing = sorted(required - observed)
        result[category] = {
            "required": sorted(required),
            "observed": sorted(observed),
            "missing": missing,
            "passed": not missing,
        }
    return result


def build_gth_transferability_report(
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate material coverage, identities, metrics, and blocked candidates."""

    payload = load_gth_transferability_contract() if contract is None else dict(contract)
    if payload.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("unsupported GTH transferability contract schema")
    raw_cases = payload.get("cases")
    if (
        not isinstance(raw_cases, Sequence)
        or isinstance(raw_cases, (str, bytes))
        or not raw_cases
        or any(not isinstance(case, Mapping) for case in raw_cases)
    ):
        raise ValueError("GTH transferability cases are invalid")
    cases = []
    case_ids: set[str] = set()
    blockers = []
    for case in raw_cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError("GTH transferability case IDs must be unique")
        case_ids.add(case_id)
        evidence_label = case.get("evidence_label")
        if evidence_label not in {
            "source",
            "current-verified",
            "project-derived",
            "historical-frozen",
        }:
            raise ValueError(f"unsupported evidence label for {case_id}")
        identity = _identity_result(case.get("identity", {}))
        method_validation = _method_validation_result(
            case.get("method_validation", {})
        )
        raw_metrics = case.get("metrics")
        if (
            not isinstance(raw_metrics, Sequence)
            or isinstance(raw_metrics, (str, bytes))
            or not raw_metrics
        ):
            raise ValueError(f"transferability case {case_id} has no metrics")
        metrics = tuple(_metric_result(metric) for metric in raw_metrics)
        science_passed = method_validation["passed"] and all(
            metric["passed"] for metric in metrics
        )
        if not method_validation["passed"]:
            blockers.append(f"case:{case_id}:method_validation")
        if not science_passed:
            blockers.extend(
                f"case:{case_id}:metric:{metric['name']}"
                for metric in metrics
                if not metric["passed"]
            )
        if not identity["complete"]:
            blockers.append(f"case:{case_id}:identity_incomplete")
        cases.append(
            {
                "case_id": case_id,
                "evidence_label": evidence_label,
                "coverage": dict(case.get("coverage", {})),
                "identity": identity,
                "method_validation": method_validation,
                "metrics": list(metrics),
                "science_passed": science_passed,
            }
        )
    requirements = payload.get("coverage_requirements")
    if not isinstance(requirements, Mapping):
        raise ValueError("GTH transferability coverage requirements are missing")
    coverage = _coverage_result(requirements, raw_cases)
    coverage_complete = all(item["passed"] for item in coverage.values())
    blockers.extend(
        f"coverage:{category}:{value}"
        for category, result in coverage.items()
        for value in result["missing"]
    )
    candidates = []
    for candidate in payload.get("candidates", ()):
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("GTH transferability candidate ID is invalid")
        for identity_name in (
            "workload_fingerprint",
            "calculation_fingerprint",
            "implementation_fingerprint",
        ):
            if not _is_sha256(candidate.get(identity_name)):
                raise ValueError(f"candidate {candidate_id} {identity_name} is invalid")
        method_validation = _method_validation_result(
            candidate.get("method_validation", {})
        )
        metrics = tuple(_metric_result(metric) for metric in candidate.get("metrics", ()))
        passed = (
            method_validation["passed"]
            and bool(metrics)
            and all(metric["passed"] for metric in metrics)
        )
        if not passed:
            blockers.append(f"candidate:{candidate_id}:not_admitted")
        if not method_validation["passed"]:
            blockers.append(f"candidate:{candidate_id}:method_validation")
        candidates.append(
            {
                "candidate_id": candidate_id,
                "method_validation": method_validation,
                "metrics": list(metrics),
                "passed": passed,
                "elapsed_wall_seconds": float(candidate["elapsed_wall_seconds"]),
                "peak_temporary_bytes": int(candidate["peak_temporary_bytes"]),
            }
        )
    strict_science_passed = all(case["science_passed"] for case in cases)
    identity_complete = all(case["identity"]["complete"] for case in cases)
    verified = coverage_complete and strict_science_passed and identity_complete
    return {
        "schema_version": CONTRACT_SCHEMA,
        "coverage": coverage,
        "coverage_complete": coverage_complete,
        "strict_science_passed": strict_science_passed,
        "identity_complete": identity_complete,
        "production_envelope_verified": verified,
        "cases": cases,
        "candidates": candidates,
        "blockers": sorted(set(blockers)),
    }


def main(argv: list[str] | None = None) -> None:
    """Print the committed periodic GTH transferability report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = build_gth_transferability_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            " ".join(
                f"{name}={report[name]}"
                for name in (
                    "coverage_complete",
                    "strict_science_passed",
                    "identity_complete",
                    "production_envelope_verified",
                )
            )
        )


if __name__ == "__main__":
    main()
