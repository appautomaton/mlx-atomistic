"""Calibrate bounded DHFR anisotropic proposals with OpenMM Reference only."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.benchmarks.dhfr_npt import (
    DEFAULT_CONTRACT_PATH,
    DHFRNPTValidationError,
    contract_fingerprint,
    load_validation_contract,
    payload_fingerprint,
    validate_prepared_boundary,
)
from mlx_atomistic.prep.io import load_prepared_system

CALIBRATION_PROTOCOL_SCHEMA = "mlx-atomistic.dhfr-npt-v2-calibration-protocol.v1"
CALIBRATION_RUN_SCHEMA = "mlx-atomistic.dhfr-npt-v2-calibration-run.v1"
CALIBRATION_REPORT_SCHEMA = "mlx-atomistic.dhfr-npt-v2-calibration-report.v1"
CALIBRATION_SEEDS = (101, 211)
CALIBRATION_AXES = ("x", "y", "z")
CALIBRATION_PREFIXES = (10, 20, 30, 40)
ATTEMPTS_PER_AXIS = 40
FORMAL_BUDGETS = (30, 60, 90, 120)
TARGET_AND_DIAGNOSTIC_SEEDS = frozenset((7, 19, 313))
CELL_CHANGE_TOLERANCE_ANGSTROM = 1.0e-7


def calibration_protocol(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable OpenMM-only calibration protocol."""

    workload = _mapping(contract.get("workload"), name="v1 workload")
    return {
        "schema": CALIBRATION_PROTOCOL_SCHEMA,
        "case_id": _mapping(contract.get("target"), name="v1 target")["case_id"],
        "v1_contract_fingerprint": contract_fingerprint(contract),
        "temperature_K": float(workload["temperature_K"]),
        "pressure_bar": float(workload["pressure_bar"]),
        "dt_ps": float(workload["dt_ps"]),
        "friction_per_ps": float(workload["friction_per_ps"]),
        "barostat_interval": int(workload["barostat"]["interval"]),
        "seeds": list(CALIBRATION_SEEDS),
        "axes": list(CALIBRATION_AXES),
        "attempts_per_axis": ATTEMPTS_PER_AXIS,
        "prefixes": list(CALIBRATION_PREFIXES),
        "formal_budgets": list(FORMAL_BUDGETS),
        "platform": "Reference",
    }


def calibration_run_path(root: str | Path, *, seed: int, axis: str) -> Path:
    """Return the canonical report path for one calibration run."""

    _validate_seed_axis(seed, axis)
    return Path(root) / f"seed-{seed}" / f"axis-{axis}" / "report.json"


def validate_calibration_run(
    report: Mapping[str, Any],
    *,
    expected_seed: int | None = None,
    expected_axis: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize one axis-isolated calibration report."""

    payload = dict(report)
    if payload.get("schema") != CALIBRATION_RUN_SCHEMA:
        raise DHFRNPTValidationError("calibration run schema is unsupported")
    fingerprint = payload.pop("report_fingerprint", None)
    if fingerprint != payload_fingerprint(payload):
        raise DHFRNPTValidationError("calibration run fingerprint mismatch")
    seed = int(payload.get("seed", -1))
    axis = str(payload.get("axis", ""))
    _validate_seed_axis(seed, axis)
    if expected_seed is not None and seed != expected_seed:
        raise DHFRNPTValidationError("calibration run seed/path mismatch")
    if expected_axis is not None and axis != expected_axis:
        raise DHFRNPTValidationError("calibration run axis/path mismatch")
    if int(payload.get("scheduled_attempts", -1)) != ATTEMPTS_PER_AXIS:
        raise DHFRNPTValidationError("calibration run attempt count mismatch")
    if payload.get("platform") != "Reference":
        raise DHFRNPTValidationError("calibration requires OpenMM Reference")
    if not isinstance(payload.get("openmm_version"), str) or not payload[
        "openmm_version"
    ]:
        raise DHFRNPTValidationError("calibration run OpenMM version is missing")
    if not _is_sha256(payload.get("source_manifest_fingerprint")):
        raise DHFRNPTValidationError("calibration source fingerprint is invalid")
    if not _is_sha256(payload.get("protocol_fingerprint")):
        raise DHFRNPTValidationError("calibration protocol fingerprint is invalid")
    elapsed = float(payload.get("elapsed_seconds", math.nan))
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise DHFRNPTValidationError("calibration elapsed time is invalid")
    cells = np.asarray(payload.get("cell_history_angstrom"), dtype=np.float64)
    if cells.shape != (ATTEMPTS_PER_AXIS + 1, 3, 3) or not np.all(
        np.isfinite(cells)
    ):
        raise DHFRNPTValidationError("calibration cell history is invalid")
    prefix_records = payload.get("prefixes")
    if not isinstance(prefix_records, list) or len(prefix_records) != len(
        CALIBRATION_PREFIXES
    ):
        raise DHFRNPTValidationError("calibration prefix evidence is incomplete")
    previous_accepted = -1
    for expected, record in zip(CALIBRATION_PREFIXES, prefix_records, strict=True):
        if not isinstance(record, Mapping):
            raise DHFRNPTValidationError("calibration prefix record is invalid")
        attempts = int(record.get("attempts", -1))
        accepted = int(record.get("accepted_moves", -1))
        if attempts != expected:
            raise DHFRNPTValidationError("calibration prefix schedule mismatch")
        if not previous_accepted <= accepted <= attempts:
            raise DHFRNPTValidationError("calibration acceptance count is invalid")
        previous_accepted = accepted
    if int(payload.get("accepted_moves", -1)) != previous_accepted:
        raise DHFRNPTValidationError("calibration final acceptance count mismatch")
    if payload.get("finite") is not True or payload.get("disabled_axes_unchanged") is not True:
        raise DHFRNPTValidationError("calibration run invariants failed")
    return {**payload, "report_fingerprint": str(fingerprint)}


def select_calibration_budget(input_dir: str | Path) -> dict[str, Any]:
    """Load all six calibration runs and select the first qualifying budget."""

    reports = []
    identities: dict[str, set[str]] = {
        "source_manifest_fingerprint": set(),
        "protocol_fingerprint": set(),
        "openmm_version": set(),
    }
    observed_pairs: set[tuple[int, str]] = set()
    for seed in CALIBRATION_SEEDS:
        for axis in CALIBRATION_AXES:
            path = calibration_run_path(input_dir, seed=seed, axis=axis)
            try:
                raw = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise DHFRNPTValidationError(
                    f"calibration run is missing or unreadable: {path}"
                ) from error
            if not isinstance(raw, Mapping):
                raise DHFRNPTValidationError("calibration run must be a JSON object")
            report = validate_calibration_run(
                raw,
                expected_seed=seed,
                expected_axis=axis,
            )
            pair = (int(report["seed"]), str(report["axis"]))
            if pair in observed_pairs:
                raise DHFRNPTValidationError("duplicate calibration run evidence")
            observed_pairs.add(pair)
            reports.append(report)
            for name in identities:
                identities[name].add(str(report[name]))
    mismatched = [name for name, values in identities.items() if len(values) != 1]
    if mismatched:
        raise DHFRNPTValidationError(
            "calibration run identity mismatch: " + ", ".join(mismatched)
        )

    candidates = _build_candidates(reports)
    selected = next(
        (
            int(candidate["formal_total_attempts"])
            for candidate in candidates
            if candidate["qualifies"]
        ),
        None,
    )
    unsigned = {
        "schema": CALIBRATION_REPORT_SCHEMA,
        "status": "selected" if selected is not None else "no_qualifier",
        "selected_formal_attempts": selected,
        "source_manifest_fingerprint": next(
            iter(identities["source_manifest_fingerprint"])
        ),
        "protocol_fingerprint": next(iter(identities["protocol_fingerprint"])),
        "openmm_version": next(iter(identities["openmm_version"])),
        "run_fingerprints": sorted(
            str(report["report_fingerprint"]) for report in reports
        ),
        "candidates": candidates,
        "runs": reports,
    }
    return {**unsigned, "report_fingerprint": payload_fingerprint(unsigned)}


def validate_calibration_report(
    report: Mapping[str, Any],
    *,
    require_selected: bool = True,
) -> dict[str, Any]:
    """Validate a complete six-run calibration report and its selection."""

    payload = dict(report)
    fingerprint = payload.pop("report_fingerprint", None)
    if payload.get("schema") != CALIBRATION_REPORT_SCHEMA:
        raise DHFRNPTValidationError("calibration report schema is unsupported")
    if fingerprint != payload_fingerprint(payload):
        raise DHFRNPTValidationError("calibration report fingerprint mismatch")
    status = payload.get("status")
    if status not in {"selected", "no_qualifier"}:
        raise DHFRNPTValidationError("calibration report status is invalid")
    if require_selected and status != "selected":
        raise DHFRNPTValidationError("calibration did not select a formal budget")
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != (
        len(CALIBRATION_SEEDS) * len(CALIBRATION_AXES)
    ):
        raise DHFRNPTValidationError("calibration report runs are incomplete")
    validated = [validate_calibration_run(run) for run in runs]
    actual_pairs = {(run["seed"], run["axis"]) for run in validated}
    expected_pairs = {
        (seed, axis)
        for seed in CALIBRATION_SEEDS
        for axis in CALIBRATION_AXES
    }
    if actual_pairs != expected_pairs or len(actual_pairs) != len(validated):
        raise DHFRNPTValidationError("calibration report run inventory mismatch")
    for name in (
        "source_manifest_fingerprint",
        "protocol_fingerprint",
        "openmm_version",
    ):
        values = {str(run[name]) for run in validated}
        if values != {str(payload.get(name))}:
            raise DHFRNPTValidationError(
                f"calibration report {name} does not reconcile"
            )
    expected_run_fingerprints = sorted(
        str(run["report_fingerprint"]) for run in validated
    )
    if payload.get("run_fingerprints") != expected_run_fingerprints:
        raise DHFRNPTValidationError(
            "calibration report run fingerprints do not reconcile"
        )
    expected_candidates = _build_candidates(validated)
    if payload.get("candidates") != expected_candidates:
        raise DHFRNPTValidationError(
            "calibration report candidates do not reconcile"
        )
    expected_selected = next(
        (
            int(candidate["formal_total_attempts"])
            for candidate in expected_candidates
            if candidate["qualifies"]
        ),
        None,
    )
    if payload.get("selected_formal_attempts") != expected_selected:
        raise DHFRNPTValidationError(
            "calibration report selected budget does not reconcile"
        )
    expected_status = "selected" if expected_selected is not None else "no_qualifier"
    if status != expected_status:
        raise DHFRNPTValidationError(
            "calibration report status does not reconcile"
        )
    return {**payload, "report_fingerprint": str(fingerprint)}


def _build_candidates(reports: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for prefix, formal_budget in zip(
        CALIBRATION_PREFIXES,
        FORMAL_BUDGETS,
        strict=True,
    ):
        seed_evidence = []
        qualifies = True
        for seed in CALIBRATION_SEEDS:
            axis_evidence = []
            accepted_total = 0
            for axis in CALIBRATION_AXES:
                run = next(
                    report
                    for report in reports
                    if report["seed"] == seed and report["axis"] == axis
                )
                prefix_record = next(
                    record for record in run["prefixes"] if record["attempts"] == prefix
                )
                accepted = int(prefix_record["accepted_moves"])
                accepted_total += accepted
                axis_evidence.append(
                    {
                        "axis": axis,
                        "attempts": prefix,
                        "accepted_moves": accepted,
                    }
                )
            seed_qualifies = (
                all(record["attempts"] >= 10 for record in axis_evidence)
                and accepted_total >= 1
            )
            qualifies = qualifies and seed_qualifies
            seed_evidence.append(
                {
                    "seed": seed,
                    "accepted_moves": accepted_total,
                    "qualifies": seed_qualifies,
                    "axes": axis_evidence,
                }
            )
        candidate = {
            "axis_prefix_attempts": prefix,
            "formal_total_attempts": formal_budget,
            "qualifies": qualifies,
            "seeds": seed_evidence,
        }
        candidates.append(candidate)
    return candidates


def _run_calibration(
    *,
    prepared_dir: Path,
    out_dir: Path,
    seeds: Sequence[int],
    axes: Sequence[str],
    attempts_per_axis: int,
    prefixes: Sequence[int],
    platform_name: str,
) -> None:
    _validate_requested_protocol(
        seeds=seeds,
        axes=axes,
        attempts_per_axis=attempts_per_axis,
        prefixes=prefixes,
        platform_name=platform_name,
    )
    contract = load_validation_contract(DEFAULT_CONTRACT_PATH)
    source_identity = validate_prepared_boundary(prepared_dir, contract)
    protocol = calibration_protocol(contract)
    protocol_fingerprint = payload_fingerprint(protocol)
    manifest = json.loads((prepared_dir / "source_manifest.json").read_text())
    expected_openmm_version = str(
        _mapping(manifest.get("source"), name="source manifest source").get(
            "openmm_version"
        )
    )
    prepared = load_prepared_system(prepared_dir)
    for seed in seeds:
        for axis in axes:
            report_path = calibration_run_path(out_dir, seed=seed, axis=axis)
            if report_path.is_file():
                existing = validate_calibration_run(
                    json.loads(report_path.read_text()),
                    expected_seed=seed,
                    expected_axis=axis,
                )
                if (
                    existing["source_manifest_fingerprint"]
                    != source_identity["manifest_fingerprint"]
                    or existing["protocol_fingerprint"] != protocol_fingerprint
                    or existing["openmm_version"] != expected_openmm_version
                ):
                    raise DHFRNPTValidationError(
                        "existing calibration run identity mismatch"
                    )
                print(f"reused seed={seed} axis={axis}")
                continue
            report = _execute_openmm_axis_run(
                prepared,
                contract=contract,
                source_manifest_fingerprint=str(
                    source_identity["manifest_fingerprint"]
                ),
                protocol_fingerprint=protocol_fingerprint,
                expected_openmm_version=expected_openmm_version,
                seed=seed,
                axis=axis,
                platform_name=platform_name,
            )
            _write_json_atomic(report_path, report)
            print(
                f"completed seed={seed} axis={axis} "
                f"accepted={report['accepted_moves']}"
            )


def _execute_openmm_axis_run(
    prepared,
    *,
    contract: Mapping[str, Any],
    source_manifest_fingerprint: str,
    protocol_fingerprint: str,
    expected_openmm_version: str,
    seed: int,
    axis: str,
    platform_name: str,
) -> dict[str, Any]:
    _validate_seed_axis(seed, axis)
    try:
        import openmm as mm
        from openmm import app, unit
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise DHFRNPTValidationError("OpenMM is required for calibration") from error

    if mm.version.version != expected_openmm_version:
        raise DHFRNPTValidationError("OpenMM calibration version drifted")
    workload = _mapping(contract.get("workload"), name="v1 workload")
    target = _mapping(contract.get("target"), name="v1 target")
    pdb = app.PDBFile(str(target["pdb_path"]))
    force_field = app.ForceField("amber99sb.xml", "tip3p.xml")
    system = force_field.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=9.0 * unit.angstrom,
        constraints=app.HBonds,
        rigidWater=True,
        removeCMMotion=True,
        hydrogenMass=1.5 * unit.amu,
    )
    nonbonded = _single_force(mm.NonbondedForce, system)
    pme = _mapping(workload.get("pme"), name="v1 PME")
    nonbonded.setEwaldErrorTolerance(float(pme["ewald_error_tolerance"]))
    nonbonded.setUseDispersionCorrection(bool(pme["dispersion_correction"]))
    nonbonded.setPMEParameters(
        float(pme["alpha_per_angstrom"]) * 10.0 / unit.nanometer,
        *(int(value) for value in pme["mesh_shape"]),
    )
    cmm = _single_force(mm.CMMotionRemover, system)
    cmm.setFrequency(int(workload["center_of_mass_motion_interval"]))
    scale_axes = tuple(name == axis for name in CALIBRATION_AXES)
    interval = int(workload["barostat"]["interval"])
    barostat = mm.MonteCarloAnisotropicBarostat(
        mm.Vec3(
            float(workload["pressure_bar"]),
            float(workload["pressure_bar"]),
            float(workload["pressure_bar"]),
        )
        * unit.bar,
        float(workload["temperature_K"]) * unit.kelvin,
        *scale_axes,
        interval,
    )
    barostat.setRandomNumberSeed(seed)
    system.addForce(barostat)
    integrator = mm.LangevinMiddleIntegrator(
        float(workload["temperature_K"]) * unit.kelvin,
        float(workload["friction_per_ps"]) / unit.picosecond,
        float(workload["dt_ps"]) * unit.picoseconds,
    )
    integrator.setRandomNumberSeed(seed)
    started = time.perf_counter()
    context = mm.Context(
        system,
        integrator,
        mm.Platform.getPlatformByName(platform_name),
    )
    initial_cell = np.asarray(prepared.cell_matrix, dtype=np.float64)
    context.setPeriodicBoxVectors(*_openmm_box(mm, unit, np.diag(initial_cell)))
    context.setPositions(
        np.asarray(prepared.positions, dtype=np.float64) * unit.angstrom
    )
    context.applyConstraints(integrator.getConstraintTolerance())
    context.setVelocitiesToTemperature(
        float(workload["temperature_K"]) * unit.kelvin,
        seed,
    )
    context.applyVelocityConstraints(integrator.getConstraintTolerance())
    cells = [_context_cell(context, unit)]
    for _ in range(ATTEMPTS_PER_AXIS):
        integrator.step(interval)
        cells.append(_context_cell(context, unit))
    elapsed = time.perf_counter() - started
    cell_history = np.asarray(cells, dtype=np.float64)
    changes = (
        np.max(np.abs(np.diff(cell_history, axis=0)), axis=(1, 2))
        > CELL_CHANGE_TOLERANCE_ANGSTROM
    )
    accepted_prefix = np.cumsum(changes, dtype=np.int64)
    disabled = [index for index, enabled in enumerate(scale_axes) if not enabled]
    disabled_unchanged = bool(
        np.max(
            np.abs(
                cell_history[:, disabled, disabled]
                - cell_history[0, disabled, disabled]
            )
        )
        <= CELL_CHANGE_TOLERANCE_ANGSTROM
    )
    prefixes = [
        {
            "attempts": prefix,
            "accepted_moves": int(accepted_prefix[prefix - 1]),
        }
        for prefix in CALIBRATION_PREFIXES
    ]
    unsigned = {
        "schema": CALIBRATION_RUN_SCHEMA,
        "seed": seed,
        "axis": axis,
        "scheduled_attempts": ATTEMPTS_PER_AXIS,
        "prefixes": prefixes,
        "accepted_moves": int(accepted_prefix[-1]),
        "cell_history_angstrom": cell_history.tolist(),
        "finite": bool(np.all(np.isfinite(cell_history))),
        "disabled_axes_unchanged": disabled_unchanged,
        "cell_change_tolerance_angstrom": CELL_CHANGE_TOLERANCE_ANGSTROM,
        "source_manifest_fingerprint": source_manifest_fingerprint,
        "protocol_fingerprint": protocol_fingerprint,
        "platform": context.getPlatform().getName(),
        "openmm_version": mm.version.version,
        "elapsed_seconds": elapsed,
    }
    del context
    del integrator
    report = {**unsigned, "report_fingerprint": payload_fingerprint(unsigned)}
    validate_calibration_run(report, expected_seed=seed, expected_axis=axis)
    return report


def _context_cell(context, unit) -> np.ndarray:
    state = context.getState()
    vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
    return np.asarray(vectors, dtype=np.float64)


def _openmm_box(mm, unit, lengths_angstrom: np.ndarray):
    a, b, c = np.asarray(lengths_angstrom, dtype=np.float64) * 0.1
    return (
        mm.Vec3(float(a), 0.0, 0.0),
        mm.Vec3(0.0, float(b), 0.0),
        mm.Vec3(0.0, 0.0, float(c)),
    ) * unit.nanometer


def _single_force(force_type, system):
    selected = [
        system.getForce(index)
        for index in range(system.getNumForces())
        if isinstance(system.getForce(index), force_type)
    ]
    if len(selected) != 1:
        raise DHFRNPTValidationError(
            f"expected exactly one {force_type.__name__}"
        )
    return selected[0]


def _validate_requested_protocol(
    *,
    seeds: Sequence[int],
    axes: Sequence[str],
    attempts_per_axis: int,
    prefixes: Sequence[int],
    platform_name: str,
) -> None:
    if tuple(seeds) != CALIBRATION_SEEDS:
        if any(int(seed) in TARGET_AND_DIAGNOSTIC_SEEDS for seed in seeds):
            raise DHFRNPTValidationError(
                "target or diagnostic seeds cannot calibrate the budget"
            )
        raise DHFRNPTValidationError("calibration seeds must be 101 and 211")
    if tuple(axes) != CALIBRATION_AXES:
        raise DHFRNPTValidationError("calibration axes must be x, y, and z")
    if attempts_per_axis != ATTEMPTS_PER_AXIS:
        raise DHFRNPTValidationError("calibration requires 40 attempts per axis")
    if tuple(prefixes) != CALIBRATION_PREFIXES:
        raise DHFRNPTValidationError(
            "calibration prefixes must be 10, 20, 30, and 40"
        )
    if platform_name != "Reference":
        raise DHFRNPTValidationError("calibration platform must be Reference")


def _validate_seed_axis(seed: int, axis: str) -> None:
    if int(seed) not in CALIBRATION_SEEDS:
        raise DHFRNPTValidationError("calibration run seed is not declared")
    if axis not in CALIBRATION_AXES:
        raise DHFRNPTValidationError("calibration run axis is not declared")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DHFRNPTValidationError(f"{name} must be an object")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--axes", nargs="+")
    parser.add_argument("--attempts-per-axis", type=int)
    parser.add_argument("--prefixes", nargs="+", type=int)
    parser.add_argument("--platform", default="Reference")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.select_only:
        if args.input is None:
            parser.error("--input is required with --select-only")
        return args
    required = {
        "--prepared": args.prepared,
        "--seeds": args.seeds,
        "--axes": args.axes,
        "--attempts-per-axis": args.attempts_per_axis,
        "--prefixes": args.prefixes,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("missing calibration arguments: " + ", ".join(missing))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run calibration trajectories or select their shared proposal budget."""

    args = _parse_args(argv)
    if args.select_only:
        report = select_calibration_budget(args.input)
        _write_json_atomic(args.out, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "selected_formal_attempts": report[
                        "selected_formal_attempts"
                    ],
                    "report_fingerprint": report["report_fingerprint"],
                },
                sort_keys=True,
            )
        )
        return 0 if report["status"] == "selected" else 1
    _run_calibration(
        prepared_dir=args.prepared,
        out_dir=args.out,
        seeds=args.seeds,
        axes=args.axes,
        attempts_per_axis=args.attempts_per_axis,
        prefixes=args.prefixes,
        platform_name=args.platform,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
