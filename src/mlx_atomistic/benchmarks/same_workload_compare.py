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
OUTPUT_ROOT = "results/same-workload-openmm-comparison"

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
    if metric == "steps_per_s" and mlx_row.get("step_count") != openmm_row.get("step_count"):
        return "diagnostic", "step counts differ; ratio suppressed"
    if metric == "ns_per_day" and mlx_row.get("step_count") != openmm_row.get("step_count"):
        return "diagnostic", "step counts differ; ratio suppressed"
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


def _command(row: dict[str, Any]) -> str | None:
    return row.get("comparison_command") or row.get("command")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-json", type=Path, action="append", default=[])
    parser.add_argument("--openmm-json", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

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
