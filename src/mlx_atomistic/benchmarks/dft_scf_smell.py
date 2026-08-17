"""Run the bounded representative-k-point periodic SCF development gate."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import mlx.core as mx

from mlx_atomistic._artifact_identity import canonical_json_bytes, sha256_bytes
from mlx_atomistic.benchmarks._dft_scf_gate_cases import (
    CASE_NAMES,
    load_scf_gate_case,
    scf_gate_config,
)
from mlx_atomistic.benchmarks.dft_runtime_contract import (
    build_source_fingerprints,
    collect_host_provenance,
)
from mlx_atomistic.dft import run_periodic_scf
from mlx_atomistic.dft._runtime_observer import RuntimeObserver

SCHEMA = "mlx-atomistic.dft-scf-smell.v2"
type _HpsiSubmissionSignature = tuple[tuple[int, ...], tuple[str, ...], int, int]


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        msg = "value must be positive"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mlx_atomistic.benchmarks.dft_scf_smell",
        description=(
            "Run a partial-Brillouin-zone SCF gate. This is not a complete "
            "216-explicit/108-representative-k-point production result."
        ),
    )
    parser.add_argument("--case", choices=CASE_NAMES, default="silicon")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--gth-source",
        type=Path,
        help="Required only for the silicon runtime workload.",
    )
    parser.add_argument("--mode", choices=("fixed", "adaptive"), required=True)
    parser.add_argument(
        "--hpsi-shape-policy",
        choices=("stable", "finite-buckets"),
        default="stable",
    )
    parser.add_argument("--representatives", type=_positive_integer, default=8)
    parser.add_argument(
        "--shape-profile",
        action="store_true",
        help="Collect completed Hpsi batch shapes; profiled timings are diagnostic only.",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def _hpsi_purpose_totals(
    submissions: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, int | float]]:
    """Aggregate logical work, physical capacity, and time by Hpsi purpose."""

    purpose_totals: dict[str, dict[str, int | float]] = {}
    for row in submissions:
        purposes = [str(value) for value in row["purposes"]]
        logical = [int(value) for value in row["logical_vector_counts"]]
        count = int(row["count"])
        seconds = float(row["seconds"])
        lane_count = len(purposes)
        for purpose in sorted(set(purposes)):
            lane_indices = [
                index for index, value in enumerate(purposes) if value == purpose
            ]
            totals = purpose_totals.setdefault(
                purpose,
                {
                    "submissions": 0,
                    "lane_applications": 0,
                    "logical_vector_equivalents": 0,
                    "physical_lane_vector_equivalents": 0,
                    "inclusive_seconds": 0.0,
                    "lane_weighted_seconds": 0.0,
                },
            )
            totals["submissions"] += count
            totals["lane_applications"] += count * len(lane_indices)
            totals["logical_vector_equivalents"] += count * sum(
                logical[index] for index in lane_indices
            )
            totals["physical_lane_vector_equivalents"] += (
                count * len(lane_indices) * int(row["physical_vector_capacity"])
            )
            totals["inclusive_seconds"] += seconds
            totals["lane_weighted_seconds"] += (
                0.0 if lane_count == 0 else seconds * len(lane_indices) / lane_count
            )
    return purpose_totals


def _hpsi_shape_profile(events: Sequence[dict[str, object]]) -> dict[str, object]:
    """Summarize Hpsi purpose, timing, and shape for bounded diagnostics."""

    started_at: dict[tuple[int, int], float] = {}
    signatures: Counter[_HpsiSubmissionSignature] = Counter()
    signature_seconds: dict[_HpsiSubmissionSignature, float] = {}
    iteration_submissions: dict[int, list[dict[str, object]]] = {}
    for event in events:
        if event.get("event") != "kpoint_batch":
            continue
        submission_key = (int(event["scf_iteration"]), int(event["batch_index"]))
        if event.get("status") == "started":
            started_at[submission_key] = float(event["elapsed_seconds"])
            continue
        if event.get("status") != "completed":
            continue
        logical = tuple(int(value) for value in event["logical_vector_counts"])
        purposes = tuple(str(value) for value in event["purposes"])
        if len(purposes) != len(logical):
            msg = "Hpsi submission purposes must match its logical lanes"
            raise ValueError(msg)
        signature = (
            logical,
            purposes,
            int(event["lane_capacity"]),
            int(event["vector_count"]),
        )
        signatures[signature] += 1
        started = started_at.pop(submission_key, None)
        seconds = 0.0
        if started is not None:
            seconds = max(float(event["elapsed_seconds"]) - started, 0.0)
            signature_seconds[signature] = signature_seconds.get(
                signature,
                0.0,
            ) + seconds
        iteration_submissions.setdefault(submission_key[0], []).append(
            {
                "logical_vector_counts": logical,
                "purposes": purposes,
                "physical_vector_capacity": int(event["vector_count"]),
                "count": 1,
                "seconds": seconds,
            }
        )
    submissions = [
        {
            "logical_vector_counts": list(logical),
            "purposes": list(purposes),
            "logical_lane_count": len(logical),
            "physical_lane_capacity": lanes,
            "physical_vector_capacity": vectors,
            "count": count,
            "seconds": signature_seconds.get(
                (logical, purposes, lanes, vectors),
                0.0,
            ),
        }
        for (logical, purposes, lanes, vectors), count in sorted(signatures.items())
    ]
    baseline_calls = sum(int(row["count"]) for row in submissions)
    baseline_submitted = sum(
        int(row["physical_lane_capacity"])
        * int(row["physical_vector_capacity"])
        * int(row["count"])
        for row in submissions
    )
    logical_vectors = sum(
        sum(int(value) for value in row["logical_vector_counts"])
        * int(row["count"])
        for row in submissions
    )
    purpose_totals = _hpsi_purpose_totals(submissions)
    candidates: list[dict[str, int | float | bool]] = []
    for tail_lanes in (1, 2, 4):
        for tail_vectors in (4, 8, 16):
            calls = 0
            submitted = 0
            for row in submissions:
                count = int(row["count"])
                main_lanes = int(row["physical_lane_capacity"])
                main_vectors = int(row["physical_vector_capacity"])
                logical = [int(value) for value in row["logical_vector_counts"]]
                tail_capacity_lanes = min(tail_lanes, main_lanes)
                tail_capacity_vectors = min(tail_vectors, main_vectors)
                if logical and max(logical) <= tail_capacity_vectors:
                    tail_calls = math.ceil(len(logical) / tail_capacity_lanes)
                    calls += count * tail_calls
                    submitted += (
                        count * tail_calls * tail_capacity_lanes * tail_capacity_vectors
                    )
                else:
                    main_calls = math.ceil(len(logical) / main_lanes)
                    calls += count * main_calls
                    submitted += count * main_calls * main_lanes * main_vectors
            reduction = (
                0.0
                if baseline_submitted == 0
                else 1.0 - submitted / baseline_submitted
            )
            call_ratio = 0.0 if baseline_calls == 0 else calls / baseline_calls
            candidates.append(
                {
                    "lanes": tail_lanes,
                    "vectors": tail_vectors,
                    "predicted_calls": calls,
                    "predicted_submitted_vector_equivalents": submitted,
                    "predicted_submitted_reduction": reduction,
                    "predicted_call_ratio": call_ratio,
                    "qualifies": reduction >= 0.25 and call_ratio <= 1.35,
                }
            )
    qualified = [candidate for candidate in candidates if bool(candidate["qualifies"])]
    selected = min(
        qualified,
        key=lambda candidate: (
            int(candidate["predicted_submitted_vector_equivalents"]),
            int(candidate["predicted_calls"]),
            int(candidate["lanes"]) * int(candidate["vectors"]),
        ),
        default=None,
    )
    return {
        "submissions": submissions,
        "baseline_calls": baseline_calls,
        "baseline_logical_vector_equivalents": logical_vectors,
        "baseline_submitted_vector_equivalents": baseline_submitted,
        "purpose_totals": purpose_totals,
        "scf_iteration_purpose_totals": [
            {
                "scf_iteration": iteration,
                "purpose_totals": _hpsi_purpose_totals(rows),
            }
            for iteration, rows in sorted(iteration_submissions.items())
        ],
        "tail_candidates": candidates,
        "selected_tail_capacity": (
            None
            if selected is None
            else {"lanes": int(selected["lanes"]), "vectors": int(selected["vectors"])}
        ),
    }


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    sources = build_source_fingerprints()
    host = collect_host_provenance()
    case = load_scf_gate_case(
        arguments.case,
        arguments.manifest,
        gth_source=arguments.gth_source,
        representatives=arguments.representatives,
    )
    observer = RuntimeObserver(
        synchronize=mx.synchronize,
        detail_events=arguments.shape_profile,
    )
    config = scf_gate_config(
        case,
        mode=arguments.mode,
        hpsi_shape_policy=arguments.hpsi_shape_policy,
    )
    started = perf_counter()
    result = run_periodic_scf(
        case.system,
        cutoff_hartree=case.cutoff_hartree,
        kpoint_mesh=case.kpoint_mesh,
        n_bands=case.occupied_band_count,
        config=config,
        observer=observer,
    )
    mx.synchronize()
    elapsed = perf_counter() - started
    observation = observer.snapshot()
    maximum_residual = max(
        float(mx.max(point.eigen.residuals)) for point in result.kpoints
    )
    maximum_overlap = max(point.eigen.orthonormality_error for point in result.kpoints)
    electron_error = abs(result.electron_count - case.system.electron_count)
    gates = case.manifest["numerical_gates"]
    numerical_passed = bool(
        result.converged
        and electron_error <= float(gates["electron_count_abs_per_cell"])
        and maximum_overlap <= float(gates["orthonormality_max"])
        and maximum_residual <= float(case.manifest["solver"]["davidson"]["tolerance"])
    )
    report = {
        "schema": SCHEMA,
        "scope": "partial-brillouin-zone-development-gate",
        "production_full_scf_result": False,
        "includes_scf_density_loop": True,
        "includes_persistence": False,
        "case": case.name,
        "profile": case.profile,
        "target_id": case.target_id,
        "manifest": str(arguments.manifest),
        "manifest_sha256": sha256_bytes(case.manifest_bytes),
        "workload_fingerprint": case.workload_fingerprint,
        "runtime_fingerprint": sources["runtime_fingerprint"],
        "host": host,
        "resources": case.resource_records(),
        "protocol": {
            "cutoff_hartree": case.cutoff_hartree,
            "fft_shape": list(case.system.grid.shape),
            "occupied_band_count": case.occupied_band_count,
            "max_batch_transient_bytes": case.max_batch_transient_bytes,
        },
        "mode": arguments.mode,
        "hpsi_shape_policy": arguments.hpsi_shape_policy,
        "elapsed_seconds": elapsed,
        "converged": result.converged,
        "numerical_passed": numerical_passed,
        "iterations": result.iterations,
        "representative_kpoints": len(result.kpoints),
        "selected_owner_indices": list(case.selected_owner_indices),
        "total_energy_hartree": result.total_energy,
        "energy_hartree_per_atom": result.total_energy / case.atom_count,
        "electron_error": electron_error,
        "density_residual": result.density_residual,
        "maximum_orbital_residual": maximum_residual,
        "maximum_overlap_error": maximum_overlap,
        "eigensolver_tolerances": [
            row["eigensolver_tolerance"] for row in result.history
        ],
        "eigensolver_methods": [row["eigensolver_method"] for row in result.history],
        "scf_history": list(result.history),
        "work_counters": observation["work_counters"],
        "phase_seconds": observation["phase_seconds"],
        "memory": observation["memory"],
        "hpsi_shapes": observation["hpsi_shapes"],
    }
    if arguments.shape_profile:
        report["hpsi_shape_profile"] = _hpsi_shape_profile(observation["events"])
    return report


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = _run(arguments)
    except (KeyError, OSError, TypeError, ValueError) as error:
        _parser().error(str(error))
    payload = canonical_json_bytes(report)
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_bytes(payload + b"\n")
    if arguments.json:
        print(payload.decode(), flush=True)
    else:
        print(
            f"{report['mode']}: {report['elapsed_seconds']:.3f} s, "
            f"{report['iterations']} SCF iterations, "
            f"{report['representative_kpoints']} representative k-points",
            flush=True,
        )
    return 0 if report["numerical_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
