"""Compare normalized MLX and OpenMM same-workload benchmark JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mlx_atomistic.benchmarks import (
    get_hardware_info,
    normalize_benchmark_payload,
    normalize_benchmark_row,
)
from mlx_atomistic.runtime import get_runtime_info

BENCHMARK_NAME = "same_workload_openmm_comparison"
ENGINE = "mlx-openmm-comparison"
OUTPUT_ROOT = "outputs/benchmarks/same-workload-openmm-comparison"

PAIR_SPECS = {
    "lj-synthetic-loop": {
        "metric": "steps_per_s",
        "metric_family": "steps/s",
        "mlx_raw_output_path": f"{OUTPUT_ROOT}/mlx-lj-synthetic-loop.json",
        "openmm_raw_output_path": f"{OUTPUT_ROOT}/openmm-lj-synthetic-loop.json",
    },
    "gbsa-obc-small": {
        "metric": "ms_per_eval",
        "metric_family": "ms/eval",
        "mlx_raw_output_path": f"{OUTPUT_ROOT}/mlx-gbsa-obc-small.json",
        "openmm_raw_output_path": f"{OUTPUT_ROOT}/openmm-gbsa-obc-small.json",
    },
    "tip4p-ew-water": {
        "metric": "ms_per_eval",
        "metric_family": "ms/eval",
        "mlx_raw_output_path": f"{OUTPUT_ROOT}/mlx-tip4p-ew-water.json",
        "openmm_raw_output_path": f"{OUTPUT_ROOT}/openmm-tip4p-ew-water.json",
    },
    "dhfr-implicit": {
        "metric": "ns_per_day",
        "metric_family": "ns/day",
        "mlx_raw_output_path": f"{OUTPUT_ROOT}/mlx-dhfr-implicit.json",
        "openmm_raw_output_path": f"{OUTPUT_ROOT}/openmm-dhfr-implicit.json",
    },
    "dhfr-explicit-pme": {
        "metric": "ns_per_day",
        "metric_family": "ns/day",
        "mlx_raw_output_path": f"{OUTPUT_ROOT}/mlx-dhfr-explicit-pme.json",
        "openmm_raw_output_path": f"{OUTPUT_ROOT}/openmm-dhfr-explicit-pme.json",
    },
}

OPENMM_PAIR_BY_KEY = {
    "synthetic-lj-periodic": "lj-synthetic-loop",
    "synthetic_lj_periodic": "lj-synthetic-loop",
    "gbsa-obc-small": "gbsa-obc-small",
    "gbsa_obc_small": "gbsa-obc-small",
    "tip4p-ew-water": "tip4p-ew-water",
    "tip4p_ew_water": "tip4p-ew-water",
    "dhfr-implicit": "dhfr-implicit",
    "dhfr_implicit": "dhfr-implicit",
    "dhfr-explicit-pme": "dhfr-explicit-pme",
    "dhfr_explicit_pme": "dhfr-explicit-pme",
}


def load_json_payloads(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in paths]


def build_summary(
    *,
    mlx_payloads: list[dict[str, Any]],
    openmm_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a normalized same-workload comparison summary."""

    mlx_rows = _index_mlx_rows(mlx_payloads)
    openmm_rows = _index_openmm_rows(openmm_payloads)
    rows = [
        _comparison_row(pair_id, mlx_rows.get(pair_id), openmm_rows.get(pair_id))
        for pair_id in PAIR_SPECS
    ]
    hardware = get_hardware_info()
    runtime = asdict(get_runtime_info())
    payload = {
        "benchmark_name": BENCHMARK_NAME,
        "fixture": "same_workload_openmm_controlled",
        "hardware": hardware,
        "runtime": runtime,
        "pair_count": len(rows),
        "comparable_pair_count": sum(row["comparison_status"] == "comparable" for row in rows),
        "blocked_pair_count": sum(row["comparison_status"] == "blocked" for row in rows),
        "diagnostic_pair_count": sum(row["comparison_status"] == "diagnostic" for row in rows),
        "cases": rows,
    }
    return normalize_benchmark_payload(
        payload,
        benchmark_name=BENCHMARK_NAME,
        fixture="same_workload_openmm_controlled",
        timing_metric="openmm_to_mlx_ratio",
        hardware=hardware,
        runtime=runtime,
        engine=ENGINE,
        finite=all(row["comparison_status"] != "failed" for row in rows),
        command="uv run python -m mlx_atomistic.benchmarks.same_workload_compare",
    )


def _index_mlx_rows(payloads: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _payload_rows(payloads):
        pair_id = row.get("comparison_pair_id")
        if pair_id not in PAIR_SPECS:
            continue
        current = rows.get(str(pair_id))
        if current is None or _mlx_selection_rank(row) < _mlx_selection_rank(current):
            rows[str(pair_id)] = row
    return rows


def _index_openmm_rows(payloads: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _payload_rows(payloads):
        key = row.get("case") or row.get("fixture") or row.get("system")
        pair_id = OPENMM_PAIR_BY_KEY.get(str(key))
        if pair_id is not None:
            rows[pair_id] = row
    return rows


def _payload_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        cases = payload.get("cases")
        if isinstance(cases, list):
            rows.extend(case for case in cases if isinstance(case, dict))
        else:
            rows.append(payload)
    return rows


def _mlx_selection_rank(row: dict[str, Any]) -> int:
    if row.get("feature") == "tip4p_ew":
        return 0
    if row.get("operation") == "obc_pair_accumulation_and_force":
        return 0
    return 1


def _comparison_row(
    pair_id: str,
    mlx_row: dict[str, Any] | None,
    openmm_row: dict[str, Any] | None,
) -> dict[str, Any]:
    spec = PAIR_SPECS[pair_id]
    row: dict[str, Any] = {
        "pair_id": pair_id,
        "fixture": pair_id,
        "system": pair_id,
        "metric_family": spec["metric_family"],
        "mlx_status": None if mlx_row is None else mlx_row.get("status"),
        "openmm_status": None if openmm_row is None else openmm_row.get("status"),
        "mlx_timing_metric": None if mlx_row is None else mlx_row.get("timing_metric"),
        "openmm_timing_metric": None if openmm_row is None else openmm_row.get("timing_metric"),
        "mlx_timing_value": None if mlx_row is None else mlx_row.get("timing_value"),
        "openmm_timing_value": None if openmm_row is None else openmm_row.get("timing_value"),
        "mlx_command": None if mlx_row is None else _command(mlx_row),
        "openmm_command": None if openmm_row is None else _command(openmm_row),
        "mlx_raw_output_path": spec["mlx_raw_output_path"],
        "openmm_raw_output_path": spec["openmm_raw_output_path"],
        "mlx_hardware": None if mlx_row is None else mlx_row.get("hardware"),
        "openmm_hardware": None if openmm_row is None else openmm_row.get("hardware"),
        "mlx_runtime": None if mlx_row is None else mlx_row.get("runtime"),
        "openmm_runtime": None if openmm_row is None else openmm_row.get("runtime"),
        "mlx_operation": None if mlx_row is None else mlx_row.get("operation"),
        "openmm_operation_semantics": None
        if openmm_row is None
        else openmm_row.get("operation_semantics"),
        "openmm_operation": None if openmm_row is None else openmm_row.get("openmm_operation"),
        "openmm_to_mlx_ratio": None,
        "ratio_direction": None,
        "comparison_status": "blocked",
        "blocker": None,
    }
    status, blocker = _classify(pair_id, mlx_row, openmm_row)
    row["comparison_status"] = status
    row["blocker"] = blocker
    if status == "comparable":
        ratio = float(openmm_row["timing_value"]) / float(mlx_row["timing_value"])  # type: ignore[index]
        row["openmm_to_mlx_ratio"] = ratio
        row["ratio_direction"] = (
            "higher favors OpenMM for throughput metrics; lower favors OpenMM for latency metrics"
        )
    return normalize_benchmark_row(
        row,
        benchmark_name=BENCHMARK_NAME,
        timing_metric="openmm_to_mlx_ratio",
        engine=ENGINE,
        fixture=pair_id,
        status="ok" if status == "comparable" else status,
        blocker=blocker,
    )


def _classify(
    pair_id: str,
    mlx_row: dict[str, Any] | None,
    openmm_row: dict[str, Any] | None,
) -> tuple[str, str | None]:
    if mlx_row is None:
        return "blocked", "missing MLX normalized row for pair"
    if openmm_row is None:
        return "blocked", "missing OpenMM normalized row for pair"
    side_rows = (("MLX", mlx_row), ("OpenMM", openmm_row))
    for side, row in side_rows:
        status = row.get("status")
        if status == "diagnostic":
            return "diagnostic", f"{side} status is diagnostic: {_row_reason(row)}"
    for side, row in side_rows:
        status = row.get("status")
        if status != "ok":
            return "blocked", f"{side} status is {status}: {_row_reason(row)}"
    metric = PAIR_SPECS[pair_id]["metric"]
    if mlx_row.get("timing_metric") != metric or openmm_row.get("timing_metric") != metric:
        return "diagnostic", "timing metrics differ; ratio suppressed"
    if not _positive_number(mlx_row.get("timing_value")) or not _positive_number(
        openmm_row.get("timing_value")
    ):
        return "diagnostic", "timing values are missing or non-positive; ratio suppressed"
    if mlx_row.get("atom_count") != openmm_row.get("atom_count"):
        return "diagnostic", "atom counts differ; ratio suppressed"
    if (
        mlx_row.get("step_count") is not None
        or openmm_row.get("step_count") is not None
    ) and mlx_row.get("step_count") != openmm_row.get("step_count"):
        return "diagnostic", "step counts differ; ratio suppressed"
    if (
        mlx_row.get("evaluation_count") is not None
        or openmm_row.get("evaluation_count") is not None
    ) and mlx_row.get("evaluation_count") != openmm_row.get("evaluation_count"):
        return "diagnostic", "evaluation counts differ; ratio suppressed"
    for field, label in (
        ("fixture_hash", "fixture hashes"),
        ("parameter_manifest_hash", "parameter manifest hashes"),
        ("precision", "precision modes"),
        ("pme_parameters", "PME parameters"),
    ):
        mlx_value = mlx_row.get(field)
        openmm_value = openmm_row.get(field)
        if (mlx_value is not None or openmm_value is not None) and mlx_value != openmm_value:
            return "diagnostic", f"{label} differ; ratio suppressed"
    parity_status = _classify_controlled_parity(pair_id, mlx_row, openmm_row)
    if parity_status is not None:
        return parity_status
    return "comparable", None


def _classify_controlled_parity(
    pair_id: str,
    mlx_row: dict[str, Any],
    openmm_row: dict[str, Any],
) -> tuple[str, str] | None:
    if pair_id == "tip4p-ew-water":
        mlx_operation = mlx_row.get("operation")
        openmm_semantics = openmm_row.get("operation_semantics")
        openmm_operation = openmm_row.get("openmm_operation")
        if (
            mlx_operation == "m_site_reconstruction"
            and openmm_semantics == "virtual_site_reconstruction"
            and openmm_operation == "Context.computeVirtualSites"
        ):
            return None
        return (
            "diagnostic",
            "TIP4P operations differ; ratio suppressed "
            f"(MLX operation={mlx_operation}, OpenMM semantics={openmm_semantics}, "
            f"OpenMM operation={openmm_operation})",
        )
    if pair_id == "gbsa-obc-small":
        mlx_operation = mlx_row.get("operation")
        openmm_setup = openmm_row.get("obc_force_setup")
        openmm_force = openmm_setup.get("force") if isinstance(openmm_setup, dict) else None
        if mlx_operation == "obc_pair_accumulation_and_force" and openmm_force == "GBSAOBCForce":
            return None
        return (
            "diagnostic",
            "GBSA operations differ; ratio suppressed "
            f"(MLX operation={mlx_operation}, OpenMM OBC force={openmm_force})",
        )
    if pair_id in {"dhfr-implicit", "dhfr-explicit-pme"}:
        return _classify_dhfr_parity(pair_id, mlx_row, openmm_row)
    return None


def _classify_dhfr_parity(
    pair_id: str,
    mlx_row: dict[str, Any],
    openmm_row: dict[str, Any],
) -> tuple[str, str] | None:
    expected = (
        ("implicit", "gbsa_obc")
        if pair_id == "dhfr-implicit"
        else ("explicit", "pme")
    )
    mlx_semantics = (mlx_row.get("solvent_model"), mlx_row.get("electrostatics_model"))
    openmm_semantics = (
        openmm_row.get("solvent_model"),
        openmm_row.get("electrostatics_model"),
    )
    if mlx_semantics != expected or openmm_semantics != expected:
        return (
            "diagnostic",
            "DHFR solvent/electrostatics semantics differ; ratio suppressed "
            f"(expected={expected}, MLX={mlx_semantics}, OpenMM={openmm_semantics})",
        )
    mlx_dt = mlx_row.get("dt_ps")
    openmm_dt = openmm_row.get("dt_ps")
    if mlx_dt is not None and openmm_dt is not None and float(mlx_dt) != float(openmm_dt):
        return "diagnostic", "DHFR timestep differs; ratio suppressed"
    return None


def _row_reason(row: dict[str, Any]) -> str:
    return str(
        row.get("blocker")
        or row.get("diagnostic_reason")
        or row.get("reason")
        or "no concrete reason provided"
    )


def _positive_number(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def build_strict_timing_comparison(
    mlx_row: dict[str, Any],
    reference_row: dict[str, Any],
) -> dict[str, Any]:
    """Compare timing rows only when their full workload contracts match.

    Args:
        mlx_row: MLX timing row with workload-manifest fields.
        reference_row: Reference timing row using the same schema.

    Returns:
        Comparable status and ratio, or a diagnostic blocker with no ratio.
    """

    required_fields = (
        "operation",
        "atom_count",
        "fixture_hash",
        "parameter_manifest_hash",
        "pme_parameters",
        "step_count",
        "precision",
        "timing_metric",
    )
    for field in required_fields:
        mlx_value = mlx_row.get(field)
        reference_value = reference_row.get(field)
        if mlx_value is None or reference_value is None:
            return {
                "status": "diagnostic",
                "ratio": None,
                "blocker": f"missing {field}; ratio suppressed",
            }
        if mlx_value != reference_value:
            return {
                "status": "diagnostic",
                "ratio": None,
                "blocker": f"{field} differs; ratio suppressed",
            }
    mlx_timing = mlx_row.get("timing_value")
    reference_timing = reference_row.get("timing_value")
    if not _positive_number(mlx_timing) or not _positive_number(reference_timing):
        return {
            "status": "diagnostic",
            "ratio": None,
            "blocker": "timing values are missing or non-positive; ratio suppressed",
        }
    return {
        "status": "comparable",
        "ratio": float(reference_timing) / float(mlx_timing),
        "blocker": None,
    }


def _command(row: dict[str, Any]) -> str | None:
    return row.get("comparison_command") or row.get("command")


# --- LJ scaling-ladder comparison (production-size, multi-engine) ---
#
# The semantic pairs above (gbsa/tip4p/dhfr) match one MLX row to one OpenMM row
# by a fixed pair id with workload-specific parity gates. The LJ scaling ladder
# is a different shape: it pairs a *sweep* of synthetic-LJ sizes across up to
# three engines (MLX, OpenMM, LAMMPS) by matching (atom_count, step_count). It
# lives in its own builder so the delicate semantic-parity machinery above stays
# untouched, and so LAMMPS -- which has no gbsa/tip4p analog -- only ever enters
# through this LJ path.

SCALING_BENCHMARK_NAME = "same_workload_lj_scaling"
SCALING_ENGINE = "mlx-reference-scaling-comparison"
SCALING_OUTPUT_ROOT = "outputs/benchmarks/same-workload-lj-scaling"
SCALING_METRIC = "steps_per_s"

_MLX_LJ_CASES = {"synthetic_lj"}
_OPENMM_LJ_KEYS = {"synthetic-lj-periodic", "synthetic_lj_periodic"}
_LAMMPS_LJ_KEYS = {"synthetic_lj_periodic", "synthetic-lj-periodic"}


def _row_workload_key(row: dict[str, Any]) -> str:
    return str(row.get("case") or row.get("fixture") or row.get("system") or "")


def _row_is_lj_synthetic(row: dict[str, Any], *, role: str) -> bool:
    if role == "mlx":
        return row.get("case") in _MLX_LJ_CASES
    if role == "openmm":
        return _row_workload_key(row) in _OPENMM_LJ_KEYS
    if role == "lammps":
        return _row_workload_key(row) in _LAMMPS_LJ_KEYS
    return False


def _scaling_size_key(row: dict[str, Any]) -> tuple[int, int] | None:
    try:
        return (int(row.get("atom_count")), int(row.get("step_count")))
    except (TypeError, ValueError):
        return None


def _scaling_timing(row: dict[str, Any] | None) -> Any:
    if row is None:
        return None
    value = row.get("timing_value")
    return value if value is not None else row.get(SCALING_METRIC)


def _scaling_selection_rank(row: dict[str, Any]) -> tuple[int, int]:
    ok = 0 if row.get("status") == "ok" else 1
    finite = 0 if _positive_number(_scaling_timing(row)) else 1
    return (ok, finite)


def _index_scaling_rows(
    payloads: list[dict[str, Any]],
    *,
    role: str,
) -> dict[tuple[int, int], dict[str, Any]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for row in _payload_rows(payloads):
        if not isinstance(row, dict) or not _row_is_lj_synthetic(row, role=role):
            continue
        key = _scaling_size_key(row)
        if key is None:
            continue
        current = rows.get(key)
        if current is None or _scaling_selection_rank(row) < _scaling_selection_rank(current):
            rows[key] = row
    return rows


def _scaling_reference(
    role: str,
    mlx_row: dict[str, Any] | None,
    ref_row: dict[str, Any] | None,
) -> tuple[str, float | None, str | None]:
    """Classify one reference engine against MLX for a single ladder size."""

    if ref_row is None:
        return "blocked", None, f"missing {role} row for this size"
    ref_status = ref_row.get("status")
    if ref_status == "diagnostic":
        return "diagnostic", None, f"{role} status is diagnostic: {_row_reason(ref_row)}"
    if ref_status != "ok":
        return "blocked", None, f"{role} status is {ref_status}: {_row_reason(ref_row)}"
    if mlx_row is None or mlx_row.get("status") != "ok":
        return "blocked", None, "MLX row missing or not ok for this size"
    mlx_value = _scaling_timing(mlx_row)
    ref_value = _scaling_timing(ref_row)
    if not _positive_number(mlx_value) or not _positive_number(ref_value):
        return "diagnostic", None, "timing values missing or non-positive; ratio suppressed"
    return "comparable", float(ref_value) / float(mlx_value), None


def _scaling_row(
    key: tuple[int, int],
    mlx_row: dict[str, Any] | None,
    openmm_row: dict[str, Any] | None,
    lammps_row: dict[str, Any] | None,
) -> dict[str, Any]:
    atom_count, step_count = key
    openmm_status, openmm_ratio, openmm_blocker = _scaling_reference("openmm", mlx_row, openmm_row)
    lammps_status, lammps_ratio, lammps_blocker = _scaling_reference("lammps", mlx_row, lammps_row)
    sub_statuses = (openmm_status, lammps_status)
    if mlx_row is None or mlx_row.get("status") != "ok":
        comparison_status = "blocked"
    elif "comparable" in sub_statuses:
        comparison_status = "comparable"
    elif "diagnostic" in sub_statuses:
        comparison_status = "diagnostic"
    else:
        comparison_status = "blocked"
    pair_id = f"lj-synthetic@N={atom_count}"
    row = {
        "pair_id": pair_id,
        "workload": "lj-synthetic-scaling",
        "fixture": pair_id,
        "system": pair_id,
        "atom_count": atom_count,
        "step_count": step_count,
        "metric_family": "steps/s",
        "mlx_status": None if mlx_row is None else mlx_row.get("status"),
        "mlx_steps_per_s": _scaling_timing(mlx_row),
        "mlx_command": None if mlx_row is None else _command(mlx_row),
        "openmm_status": openmm_status,
        "openmm_steps_per_s": _scaling_timing(openmm_row),
        "openmm_to_mlx_ratio": openmm_ratio,
        "openmm_command": None if openmm_row is None else _command(openmm_row),
        "lammps_status": lammps_status,
        "lammps_steps_per_s": _scaling_timing(lammps_row),
        "lammps_to_mlx_ratio": lammps_ratio,
        "lammps_command": None if lammps_row is None else _command(lammps_row),
        "comparison_status": comparison_status,
        "blocker": openmm_blocker or lammps_blocker,
        "ratio_direction": (
            "ratio = reference_steps_per_s / mlx_steps_per_s; "
            ">1 means the reference engine runs more steps/s (the MLX gap)"
        ),
    }
    return normalize_benchmark_row(
        row,
        benchmark_name=SCALING_BENCHMARK_NAME,
        timing_metric="openmm_to_mlx_ratio",
        engine=SCALING_ENGINE,
        fixture=pair_id,
        status="ok" if comparison_status == "comparable" else comparison_status,
        blocker=row["blocker"],
    )


def build_scaling_summary(
    *,
    mlx_payloads: list[dict[str, Any]],
    openmm_payloads: list[dict[str, Any]] | None = None,
    lammps_payloads: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the LJ size-ladder MLX-vs-reference scaling comparison.

    Synthetic-LJ rows from each engine are matched by ``(atom_count,
    step_count)``, so a sweep of sizes pairs cleanly without colliding on a
    single ``comparison_pair_id``. Ratios are only computed for sizes where MLX
    and the reference engine both ran ``ok`` at the same particle/step count.
    """

    mlx_rows = _index_scaling_rows(mlx_payloads, role="mlx")
    openmm_rows = _index_scaling_rows(openmm_payloads or [], role="openmm")
    lammps_rows = _index_scaling_rows(lammps_payloads or [], role="lammps")
    keys = sorted(set(mlx_rows) | set(openmm_rows) | set(lammps_rows))
    cases = [
        _scaling_row(key, mlx_rows.get(key), openmm_rows.get(key), lammps_rows.get(key))
        for key in keys
    ]
    hardware = get_hardware_info()
    runtime = asdict(get_runtime_info())
    payload = {
        "benchmark_name": SCALING_BENCHMARK_NAME,
        "fixture": "lj_synthetic_scaling",
        "hardware": hardware,
        "runtime": runtime,
        "size_count": len(cases),
        "comparable_count": sum(row["comparison_status"] == "comparable" for row in cases),
        "diagnostic_count": sum(row["comparison_status"] == "diagnostic" for row in cases),
        "blocked_count": sum(row["comparison_status"] == "blocked" for row in cases),
        "atom_counts": [row["atom_count"] for row in cases],
        "cases": cases,
    }
    return normalize_benchmark_payload(
        payload,
        benchmark_name=SCALING_BENCHMARK_NAME,
        fixture="lj_synthetic_scaling",
        timing_metric="reference_to_mlx_ratio",
        hardware=hardware,
        runtime=runtime,
        engine=SCALING_ENGINE,
        finite=all(row["comparison_status"] != "failed" for row in cases),
        command="uv run python -m mlx_atomistic.benchmarks.same_workload_compare --scaling",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-json", type=Path, action="append", default=[])
    parser.add_argument("--openmm-json", type=Path, action="append", default=[])
    parser.add_argument("--lammps-json", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="LJ size-ladder comparison across MLX, OpenMM, and LAMMPS",
    )
    args = parser.parse_args(argv)

    if args.scaling:
        payload = build_scaling_summary(
            mlx_payloads=load_json_payloads(args.mlx_json),
            openmm_payloads=load_json_payloads(args.openmm_json),
            lammps_payloads=load_json_payloads(args.lammps_json),
        )
    else:
        payload = build_summary(
            mlx_payloads=load_json_payloads(args.mlx_json),
            openmm_payloads=load_json_payloads(args.openmm_json),
        )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    if args.json or args.out is None:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
