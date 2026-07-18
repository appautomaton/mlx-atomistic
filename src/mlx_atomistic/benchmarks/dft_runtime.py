"""Stable command-line harness for MLX-only periodic DFT runtime evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from mlx_atomistic._artifact_identity import canonical_json_bytes, sha256_bytes
from mlx_atomistic.benchmarks.dft_runtime_contract import (
    build_source_fingerprints,
    prepare_workload,
    results_output_path,
)
from mlx_atomistic.benchmarks.dft_runtime_core import (
    COMMAND_FAILURE_SCHEMA,
    _finalize_report,
    _publish_report,
    inspect_artifact,
    run_compare,
    run_fixed_density,
    run_full_scf,
    run_ladder,
)


def _progress(event: dict[str, object]) -> None:
    sys.stderr.buffer.write(canonical_json_bytes(event) + b"\n")
    sys.stderr.flush()


def _rungs(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        msg = "rungs must be comma-separated integers"
        raise argparse.ArgumentTypeError(msg) from error
    if not parsed:
        msg = "at least one ladder rung is required"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m mlx_atomistic.benchmarks.dft_runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="publish the immutable workload")
    prepare.add_argument("--gth-source", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--json", action="store_true")

    inspect = subparsers.add_parser("inspect", help="read-only artifact validation")
    inspect.add_argument("--artifact", required=True)
    inspect.add_argument("--integrity-only", action="store_true")
    inspect.add_argument("--require-current-protocol-match", action="store_true")
    inspect.add_argument("--require-current-runtime-match", action="store_true")
    inspect.add_argument("--require-current-source-match", action="store_true")
    inspect.add_argument("--require-admitted", action="store_true")
    inspect.add_argument("--require-numerical", action="store_true")
    inspect.add_argument("--require-speedup", type=float)
    inspect.add_argument("--max-elapsed-seconds", type=float)
    inspect.add_argument("--json", action="store_true")

    fixed = subparsers.add_parser("fixed-density", help="run fixed-density evidence")
    _add_numerical_paths(fixed)
    fixed.add_argument("--warmups", type=int, required=True)
    fixed.add_argument("--samples", type=int, required=True)
    fixed.add_argument("--fresh", action="store_true")
    fixed.add_argument("--diagnostic", action="store_true")
    fixed.add_argument("--require-clean", action="store_true")
    fixed.add_argument("--require-chip")
    fixed.add_argument("--require-low-power", action="store_true")
    fixed.add_argument("--require-numerical", action="store_true")
    fixed.add_argument("--seal", action="store_true")
    fixed.add_argument("--compare-seal")
    fixed.add_argument("--require-speedup", type=float)
    fixed.add_argument("--require-coefficient-reduction", type=float)
    fixed.add_argument("--require-projector-payload-reduction", type=float)
    fixed.add_argument("--require-projector-traffic-reduction", type=float)

    ladder = subparsers.add_parser("ladder", help="run progressive scale evidence")
    _add_numerical_paths(ladder)
    ladder.add_argument("--rungs", required=True, type=_rungs)
    ladder.add_argument("--fresh", action="store_true")
    ladder.add_argument("--require-chip")
    ladder.add_argument("--require-low-power", action="store_true")
    ladder.add_argument("--require-success", action="store_true")
    ladder.add_argument("--allow-failed-rung", action="store_true")

    compare = subparsers.add_parser("compare", help="compare matched raw reports")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--optimized", required=True)
    compare.add_argument("--baseline-seal")
    compare.add_argument("--out", required=True)
    compare.add_argument("--fresh", action="store_true")
    compare.add_argument("--require-chip")
    compare.add_argument("--require-low-power", action="store_true")
    compare.add_argument("--require-matched-power-source", action="store_true")
    compare.add_argument("--require-admitted", action="store_true")
    compare.add_argument("--require-speedup", type=float)
    compare.add_argument("--json", action="store_true")

    full = subparsers.add_parser("full-scf", help="run supervised fresh full SCF")
    _add_numerical_paths(full)
    full.add_argument("--fresh", action="store_true")
    full.add_argument("--diagnostic", action="store_true")
    full.add_argument("--timeout-seconds", type=float, required=True)
    full.add_argument("--require-clean", action="store_true")
    full.add_argument("--require-chip")
    full.add_argument("--require-low-power", action="store_true")
    full.add_argument("--require-numerical", action="store_true")
    full.add_argument("--require-success", action="store_true")
    return parser


def _add_numerical_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gth-source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--json", action="store_true")


def _dispatch(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "prepare":
        result = prepare_workload(gth_source=args.gth_source, out=args.out)
    elif args.command == "inspect":
        source_match = args.require_current_source_match
        result = inspect_artifact(
            artifact=args.artifact,
            integrity_only=args.integrity_only,
            require_current_protocol_match=args.require_current_protocol_match
            or source_match,
            require_current_runtime_match=args.require_current_runtime_match or source_match,
            require_admitted=args.require_admitted,
            require_numerical=args.require_numerical,
            require_speedup=args.require_speedup,
            max_elapsed_seconds=args.max_elapsed_seconds,
        )
    elif args.command == "fixed-density":
        result = run_fixed_density(
            manifest_path=args.manifest,
            gth_source=args.gth_source,
            out=args.out,
            warmups=args.warmups,
            samples=args.samples,
            fresh=args.fresh,
            diagnostic=args.diagnostic,
            require_clean=args.require_clean,
            require_chip=args.require_chip,
            require_low_power=args.require_low_power,
            require_numerical=args.require_numerical,
            seal=args.seal,
            compare_seal=args.compare_seal,
            require_speedup=args.require_speedup,
            require_coefficient_reduction=args.require_coefficient_reduction,
            require_projector_payload_reduction=args.require_projector_payload_reduction,
            require_projector_traffic_reduction=args.require_projector_traffic_reduction,
            progress=_progress,
        )
    elif args.command == "ladder":
        result = run_ladder(
            manifest_path=args.manifest,
            gth_source=args.gth_source,
            out=args.out,
            rungs=args.rungs,
            fresh=args.fresh,
            require_chip=args.require_chip,
            require_low_power=args.require_low_power,
            require_success=args.require_success,
            allow_failed_rung=args.allow_failed_rung,
            progress=_progress,
        )
    elif args.command == "compare":
        result = run_compare(
            baseline=args.baseline,
            optimized=args.optimized,
            baseline_seal=args.baseline_seal,
            out=args.out,
            fresh=args.fresh,
            require_chip=args.require_chip,
            require_low_power=args.require_low_power,
            require_matched_power_source=args.require_matched_power_source,
            require_admitted=args.require_admitted,
            require_speedup=args.require_speedup,
        )
    else:
        result = run_full_scf(
            manifest_path=args.manifest,
            gth_source=args.gth_source,
            out=args.out,
            fresh=args.fresh,
            diagnostic=args.diagnostic,
            timeout_seconds=args.timeout_seconds,
            require_clean=args.require_clean,
            require_chip=args.require_chip,
            require_low_power=args.require_low_power,
            require_numerical=args.require_numerical,
            require_success=args.require_success,
            progress=_progress,
        )
    return result


def _publish_command_failure(
    args: argparse.Namespace,
    error: Exception,
) -> dict[str, object]:
    try:
        sources = build_source_fingerprints()
        source_error = None
    except Exception as inventory_error:
        sources = {
            "protocol_fingerprint": None,
            "runtime_fingerprint": None,
        }
        source_error = {
            "error_type": type(inventory_error).__name__,
            "message": str(inventory_error),
        }
    failure_contract = {
        "schema_version": COMMAND_FAILURE_SCHEMA,
        "command": args.command,
        "protocol_fingerprint": sources["protocol_fingerprint"],
        "runtime_fingerprint": sources["runtime_fingerprint"],
    }
    identity = {
        "workload_fingerprint": None,
        "protocol_fingerprint": sources["protocol_fingerprint"],
        "runtime_fingerprint": sources["runtime_fingerprint"],
        "execution_contract_fingerprint": sha256_bytes(
            canonical_json_bytes(failure_contract)
        ),
    }
    report = _finalize_report(
        {
            "schema_version": COMMAND_FAILURE_SCHEMA,
            "kind": "command-failure",
            "identity": identity,
            "command": args.command,
            "failure": {
                "stage": "preflight-or-execution",
                "error_type": type(error).__name__,
                "message": str(error),
                "source_inventory_error": source_error,
            },
            "statuses": {
                "numerical_status": "blocked",
                "resume_integrity_status": "blocked",
                "timing_admission_status": "blocked",
            },
            "admission": {
                "passed": False,
                "blockers": ["command_execution_failed"],
            },
        }
    )
    return _publish_report(
        out=args.out,
        artifact_kind="dft-runtime-command-failure",
        artifact_schema=COMMAND_FAILURE_SCHEMA,
        report=report,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run one stable DFT runtime harness command."""

    args = _parser().parse_args(argv)
    if hasattr(args, "out"):
        args.out = str(results_output_path(args.out))
    try:
        result = _dispatch(args)
    except FileExistsError:
        raise
    except Exception as error:
        _progress(
            {
                "event": "failure",
                "stage": "preflight-or-execution",
                "command": args.command,
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )
        destination = Path(args.out) if hasattr(args, "out") else None
        if destination is None or destination.exists() or destination.is_symlink():
            raise SystemExit(2) from error
        try:
            result = _publish_command_failure(args, error)
        except Exception:
            raise SystemExit(2) from error
    summary = {key: value for key, value in result.items() if key != "report_payload"}
    if getattr(args, "json", False):
        sys.stdout.buffer.write(canonical_json_bytes(summary) + b"\n")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    sys.stdout.flush()
    admission = result.get("admission")
    if isinstance(admission, dict) and admission.get("passed") is False:
        raise SystemExit(2)
    if result.get("passed") is False or result.get("status") == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
