"""Benchmark-only stage profiler for the periodic DFT Hamiltonian path."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import sys
import tarfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_atomistic._artifact_identity import (
    AtomicGeneration,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from mlx_atomistic.benchmarks.dft_runtime_contract import (
    GTH_ELEMENT,
    GTH_NAME,
    build_source_fingerprints,
    collect_host_provenance,
    find_repo_root,
    load_workload,
    results_output_path,
)
from mlx_atomistic.benchmarks.dft_runtime_core import collect_git_provenance
from mlx_atomistic.dft._compact import _CompactBatch, _CompactLaneState
from mlx_atomistic.dft.periodic_gth import PeriodicGTHNonlocalOperator
from mlx_atomistic.dft.periodic_scf import PeriodicKohnShamOperator
from mlx_atomistic.runtime import get_runtime_info

PROFILE_SCHEMA = "mlx-atomistic.dft-hpsi-stage-profile.v3"
PROFILE_ARTIFACT_KIND = "dft-hpsi-stage-profile"
LOCAL_FFT_PROFILE_STAGES = (
    "inverse-fft",
    "potential-multiply",
    "forward-fft",
    "gather",
)
PROFILE_STAGES = (
    "scatter",
    *LOCAL_FFT_PROFILE_STAGES,
    "local-fft",
    "gth",
    "hpsi",
)
DEFAULT_VECTOR_COUNTS = (1, 2, 4, 8, 16, 32, 64)

StageAction = Callable[[], tuple[mx.array, ...]]


@dataclass(frozen=True)
class _ProfileContext:
    basis: object
    nonlocal_operator: PeriodicGTHNonlocalOperator
    operator: PeriodicKohnShamOperator
    grid_shape: tuple[int, int, int]


@dataclass(frozen=True)
class _ProfileCase:
    vector_count: int
    state: _CompactLaneState
    batch: _CompactBatch
    scattered: mx.array
    batched_potential: mx.array
    real_orbitals: mx.array
    weighted_orbitals: mx.array
    reciprocal_grid: mx.array
    nonlocal_operator: PeriodicGTHNonlocalOperator
    operator: PeriodicKohnShamOperator


def _parse_vector_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        msg = "vector counts must be comma-separated positive integers"
        raise argparse.ArgumentTypeError(msg) from error
    if not counts or any(count <= 0 for count in counts) or len(set(counts)) != len(counts):
        msg = "vector counts must be unique positive integers"
        raise argparse.ArgumentTypeError(msg)
    return counts


def _validate_protocol(
    *,
    vector_counts: Sequence[int],
    warmups: int,
    samples: int,
    capture_stage: str | None,
    capture_vector_count: int | None,
    capture_repetitions: int,
    selected_device: str,
    metal_available: bool,
) -> None:
    if not vector_counts or any(type(value) is not int or value <= 0 for value in vector_counts):
        msg = "vector_counts must contain positive non-bool integers"
        raise ValueError(msg)
    if len(set(vector_counts)) != len(vector_counts):
        msg = "vector_counts must not contain duplicates"
        raise ValueError(msg)
    if type(warmups) is not int or warmups < 0:
        msg = "warmups must be a non-negative integer"
        raise ValueError(msg)
    if type(samples) is not int or samples <= 0:
        msg = "samples must be a positive integer"
        raise ValueError(msg)
    if type(capture_repetitions) is not int or capture_repetitions <= 0:
        msg = "capture_repetitions must be a positive integer"
        raise ValueError(msg)
    if capture_stage is None:
        if capture_vector_count is not None:
            msg = "capture_vector_count requires capture_stage"
            raise ValueError(msg)
        return
    if capture_stage not in PROFILE_STAGES:
        msg = f"capture_stage must be one of: {', '.join(PROFILE_STAGES)}"
        raise ValueError(msg)
    if capture_vector_count is None:
        msg = "capture_stage requires capture_vector_count"
        raise ValueError(msg)
    if capture_vector_count not in vector_counts:
        msg = "capture_vector_count must be present in vector_counts"
        raise ValueError(msg)
    if not metal_available or "gpu" not in selected_device.lower():
        msg = "Metal capture requires an available Metal GPU selected as the default device"
        raise RuntimeError(msg)


def _selected_point(manifest: Mapping[str, object]) -> tuple[float, float, float]:
    physics = manifest["physics"]
    lane_index = int(physics["fixed_density_lane_index"])
    point = physics["kpoints"][lane_index]
    return tuple(float(value) for value in point["reduced_coordinates"])


def _build_profile_context(
    manifest: Mapping[str, object],
    *,
    gth_source: str | Path,
) -> _ProfileContext:
    from mlx_atomistic.dft import (
        PeriodicDFTSystem,
        PlaneWaveBasis,
        ProductionPBEExchangeCorrelation,
        gth_local_potential_grid,
        hartree_potential,
        read_gth,
    )

    system_values = manifest["system"]
    physics = manifest["physics"]
    shape = tuple(int(value) for value in physics["fft_shape"])
    cutoff = float(physics["kinetic_cutoff_hartree"])
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
    effective = (
        gth_local_potential_grid(pseudo, gamma_basis, positions)
        + hartree_potential(density, system.grid)
        + ProductionPBEExchangeCorrelation().evaluate(density, system.grid).potential
    )
    basis = PlaneWaveBasis.from_reduced_kpoint(
        system.grid,
        cutoff,
        _selected_point(manifest),
        lane_label="dft-hpsi-stage-profile",
    )
    nonlocal_operator = PeriodicGTHNonlocalOperator(pseudo, basis, positions)
    operator = PeriodicKohnShamOperator(basis, effective, nonlocal_operator)
    return _ProfileContext(
        basis=basis,
        nonlocal_operator=nonlocal_operator,
        operator=operator,
        grid_shape=shape,
    )


def _build_profile_case(
    context: _ProfileContext,
    vector_count: int,
) -> _ProfileCase:
    rng = np.random.default_rng(10_000 + vector_count)
    active_count = int(context.basis.active_count)
    values = rng.normal(size=(vector_count, active_count)) + 1j * rng.normal(
        size=(vector_count, active_count)
    )
    row_norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = (values / row_norms).astype(np.complex64)
    state = context.basis._state_from_compact(mx.array(values))
    batch = _CompactBatch.from_states((state,))
    scattered = batch.scatter()
    batched_potential = mx.array(context.operator._effective_local_potential)[None, None, ...]
    real_orbitals = batch.to_real(scattered=scattered)
    weighted_orbitals = real_orbitals * batched_potential
    reciprocal_grid = (
        mx.fft.fftn(
            weighted_orbitals,
            s=batch.grid_shape,
            axes=(-3, -2, -1),
        )
        * sqrt(batch.volume)
        / batch.grid_size
    )
    mx.eval(
        state.values,
        batch.values,
        batch.fft_indices,
        batch.valid_mask,
        scattered,
        batched_potential,
        real_orbitals,
        weighted_orbitals,
        reciprocal_grid,
    )
    mx.synchronize()
    return _ProfileCase(
        vector_count=vector_count,
        state=state,
        batch=batch,
        scattered=scattered,
        batched_potential=batched_potential,
        real_orbitals=real_orbitals,
        weighted_orbitals=weighted_orbitals,
        reciprocal_grid=reciprocal_grid,
        nonlocal_operator=context.nonlocal_operator,
        operator=context.operator,
    )


def _stage_action(case: _ProfileCase, stage: str) -> StageAction:
    if stage == "scatter":

        def scatter() -> tuple[mx.array, ...]:
            return (case.batch.scatter(),)

        return scatter
    if stage == "inverse-fft":

        def inverse_fft() -> tuple[mx.array, ...]:
            return (case.batch.to_real(scattered=case.scattered),)

        return inverse_fft
    if stage == "potential-multiply":

        def potential_multiply() -> tuple[mx.array, ...]:
            return (case.real_orbitals * case.batched_potential,)

        return potential_multiply
    if stage == "forward-fft":

        def forward_fft() -> tuple[mx.array, ...]:
            return (
                mx.fft.fftn(
                    case.weighted_orbitals,
                    s=case.batch.grid_shape,
                    axes=(-3, -2, -1),
                )
                * sqrt(case.batch.volume)
                / case.batch.grid_size,
            )

        return forward_fft
    if stage == "gather":

        def gather() -> tuple[mx.array, ...]:
            return (case.batch.gather(case.reciprocal_grid),)

        return gather
    if stage == "local-fft":

        def local_fft() -> tuple[mx.array, ...]:
            return (
                case.batch.apply_local(
                    case.operator._effective_local_potential,
                    scattered=case.scattered,
                ),
            )

        return local_fft
    if stage == "gth":

        def gth() -> tuple[mx.array, ...]:
            actions, _metrics = PeriodicGTHNonlocalOperator._apply_compact_batch(
                (case.nonlocal_operator,),
                (case.state,),
                batch=case.batch,
                evaluate=False,
            )
            return (actions[0].values,)

        return gth
    if stage == "hpsi":

        def hpsi() -> tuple[mx.array, ...]:
            action = case.operator._apply_compact(
                case.state,
                prepared_batch=case.batch,
            )
            return (action.values,)

        return hpsi
    msg = f"unknown profile stage: {stage}"
    raise ValueError(msg)


def _execute(action: StageAction) -> tuple[mx.array, ...]:
    outputs = action()
    mx.eval(*outputs)
    mx.synchronize()
    return outputs


def _time_action(action: StageAction) -> tuple[float, tuple[mx.array, ...]]:
    mx.synchronize()
    started = time.perf_counter()
    outputs = _execute(action)
    return time.perf_counter() - started, outputs


def _output_summary(outputs: Sequence[mx.array]) -> dict[str, object]:
    finite = True
    real_sum = 0.0
    imaginary_sum = 0.0
    squared_norm = 0.0
    maximum_absolute = 0.0
    shapes = []
    for output in outputs:
        value = mx.array(output).astype(mx.complex64)
        shapes.append([int(size) for size in value.shape])
        metrics = mx.stack(
            (
                mx.sum(mx.real(value)),
                mx.sum(mx.imag(value)),
                mx.sum(mx.abs(value) ** 2),
                mx.max(mx.abs(value)),
            )
        )
        is_finite = mx.all(mx.isfinite(value))
        mx.eval(metrics, is_finite)
        values = np.asarray(metrics, dtype=np.float64)
        finite = finite and bool(is_finite)
        real_sum += float(values[0])
        imaginary_sum += float(values[1])
        squared_norm += float(values[2])
        maximum_absolute = max(maximum_absolute, float(values[3]))
    return {
        "finite": finite,
        "shapes": shapes,
        "real_sum": real_sum,
        "imaginary_sum": imaginary_sum,
        "squared_norm": squared_norm,
        "maximum_absolute": maximum_absolute,
    }


def _measure_case(
    case: _ProfileCase,
    *,
    warmups: int,
    samples: int,
) -> dict[str, object]:
    stage_reports: dict[str, object] = {}
    for stage in PROFILE_STAGES:
        action = _stage_action(case, stage)
        for _ in range(warmups):
            _execute(action)
        raw = []
        outputs: tuple[mx.array, ...] = ()
        for _ in range(samples):
            elapsed, outputs = _time_action(action)
            raw.append(elapsed)
        stage_reports[stage] = {
            "median_seconds": statistics.median(raw),
            "minimum_seconds": min(raw),
            "maximum_seconds": max(raw),
            "raw_seconds": raw,
            "output": _output_summary(outputs),
        }
    gth_seconds = float(stage_reports["gth"]["median_seconds"])
    local_fft_seconds = float(stage_reports["local-fft"]["median_seconds"])
    hpsi_seconds = float(stage_reports["hpsi"]["median_seconds"])
    local_substage_seconds = {
        stage: float(stage_reports[stage]["median_seconds"]) for stage in LOCAL_FFT_PROFILE_STAGES
    }
    return {
        "vector_count": case.vector_count,
        "active_count": case.state.layout.active_count,
        "projector_count": case.nonlocal_operator._projector_count,
        "projector_payload_bytes": (
            case.nonlocal_operator._projector_count * case.state.layout.active_count * 8
        ),
        "projector_logical_bytes_per_gth_apply": (
            2
            * case.vector_count
            * case.nonlocal_operator._projector_count
            * case.state.layout.active_count
            * 8
        ),
        "compact_payload_bytes": int(case.batch.values.size) * 8,
        "fft_workspace_bytes": (
            2 * case.batch.lane_capacity * case.batch.vector_count * case.batch.grid_size * 8
        ),
        "stages": stage_reports,
        "attribution": {
            "gth_over_hpsi": gth_seconds / hpsi_seconds,
            "local_fft_over_hpsi": local_fft_seconds / hpsi_seconds,
            "local_fft_substages_over_hpsi": {
                stage: seconds / hpsi_seconds for stage, seconds in local_substage_seconds.items()
            },
            "local_fft_substages_over_local_fft": {
                stage: seconds / local_fft_seconds
                for stage, seconds in local_substage_seconds.items()
            },
            "local_fft_substage_median_sum_over_local_fft": (
                sum(local_substage_seconds.values()) / local_fft_seconds
            ),
            "independent_stage_medians_are_additive": False,
        },
    }


def _profile_contract(
    *,
    workload_fingerprint: str,
    vector_counts: Sequence[int],
    warmups: int,
    samples: int,
    capture_stage: str | None,
    capture_vector_count: int | None,
    capture_repetitions: int,
    selected_device: str,
    mlx_version: str,
    profiler_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": PROFILE_SCHEMA,
        "workload_fingerprint": workload_fingerprint,
        "vector_counts": list(vector_counts),
        "warmups_per_stage": warmups,
        "samples_per_stage": samples,
        "stage_order": list(PROFILE_STAGES),
        "synchronization": "before-and-after-each-stage-sample",
        "cache_policy": (
            "materialize-local-fft-substage-inputs-and-warm-gth-projector-cache-before-stage-timing"
        ),
        "timing_scope": "python-graph-construction-through-device-completion",
        "capture": {
            "stage": capture_stage,
            "vector_count": capture_vector_count,
            "repetitions": capture_repetitions if capture_stage is not None else 0,
        },
        "selected_device": selected_device,
        "mlx_version": mlx_version,
        "profiler_sha256": profiler_sha256,
    }


def _finalize_report(report: dict[str, object]) -> dict[str, object]:
    unsigned = {key: value for key, value in report.items() if key != "report_fingerprint"}
    report["report_fingerprint"] = sha256_bytes(canonical_json_bytes(unsigned))
    return report


def _archive_capture(capture_path: Path) -> tuple[str, str]:
    if capture_path.is_file() and not capture_path.is_symlink():
        return capture_path.name, "gputrace"
    if capture_path.is_symlink() or not capture_path.is_dir():
        msg = "Metal capture did not produce a regular file or bundle directory"
        raise RuntimeError(msg)
    archive_path = capture_path.with_name(f"{capture_path.name}.tar.gz")
    with tarfile.open(archive_path, mode="x:gz", dereference=True) as archive:
        archive.add(capture_path, arcname=capture_path.name, recursive=True)
    shutil.rmtree(capture_path)
    return archive_path.name, "tar-gzip-wrapped-gputrace"


def run_profile(
    *,
    manifest_path: str | Path,
    gth_source: str | Path,
    out: str | Path,
    vector_counts: Sequence[int] = DEFAULT_VECTOR_COUNTS,
    warmups: int = 2,
    samples: int = 7,
    capture_stage: str | None = None,
    capture_vector_count: int | None = None,
    capture_repetitions: int = 3,
) -> dict[str, object]:
    """Run synchronized Hpsi stage measurements and publish one atomic artifact."""

    manifest, _selected_gth = load_workload(manifest_path, gth_source=gth_source)
    runtime = get_runtime_info()
    selected_device = str(mx.default_device())
    counts = tuple(vector_counts)
    _validate_protocol(
        vector_counts=counts,
        warmups=warmups,
        samples=samples,
        capture_stage=capture_stage,
        capture_vector_count=capture_vector_count,
        capture_repetitions=capture_repetitions,
        selected_device=selected_device,
        metal_available=runtime.metal_available,
    )
    root = find_repo_root()
    profiler_path = Path(__file__).resolve()
    contract = _profile_contract(
        workload_fingerprint=str(manifest["workload_fingerprint"]),
        vector_counts=counts,
        warmups=warmups,
        samples=samples,
        capture_stage=capture_stage,
        capture_vector_count=capture_vector_count,
        capture_repetitions=capture_repetitions,
        selected_device=selected_device,
        mlx_version=runtime.mlx_version,
        profiler_sha256=sha256_file(profiler_path),
    )
    sources = build_source_fingerprints(root)
    identity = {
        "workload_fingerprint": manifest["workload_fingerprint"],
        "runtime_fingerprint": sources["runtime_fingerprint"],
        "profile_contract_fingerprint": sha256_bytes(canonical_json_bytes(contract)),
    }
    host = collect_host_provenance()
    context = _build_profile_context(manifest, gth_source=gth_source)
    try:
        cases = {count: _build_profile_case(context, count) for count in counts}
        smallest = cases[counts[0]]
        _execute(_stage_action(smallest, "gth"))
        case_reports = [
            _measure_case(cases[count], warmups=warmups, samples=samples) for count in counts
        ]
        if not all(
            stage["output"]["finite"] for case in case_reports for stage in case["stages"].values()
        ):
            msg = "profiled Hpsi stage produced a non-finite output"
            raise RuntimeError(msg)

        destination = Path(out)
        with AtomicGeneration(
            destination=destination,
            artifact_kind=PROFILE_ARTIFACT_KIND,
            artifact_schema_version=PROFILE_SCHEMA,
            identity=identity,
            metadata={"status": "diagnostic", "capture_stage": capture_stage},
        ) as generation:
            capture = None
            if capture_stage is not None:
                capture_path = generation.path("metal.gputrace")
                capture_action = _stage_action(
                    cases[int(capture_vector_count)],
                    capture_stage,
                )
                _execute(capture_action)
                mx.metal.start_capture(str(capture_path))
                try:
                    for _ in range(capture_repetitions):
                        _execute(capture_action)
                finally:
                    mx.metal.stop_capture()
                capture_name, capture_format = _archive_capture(capture_path)
                capture = {
                    "path": capture_name,
                    "format": capture_format,
                    "stage": capture_stage,
                    "vector_count": capture_vector_count,
                    "repetitions": capture_repetitions,
                }
            report = _finalize_report(
                {
                    "schema_version": PROFILE_SCHEMA,
                    "kind": "dft-hpsi-stage-profile",
                    "identity": identity,
                    "contract": contract,
                    "git": collect_git_provenance(root),
                    "host": host,
                    "runtime": {
                        **runtime.to_dict(),
                        "selected_device": selected_device,
                        "python_version": platform.python_version(),
                    },
                    "workload": {
                        "target_id": manifest["target_id"],
                        "grid_shape": list(context.grid_shape),
                        "cutoff_hartree": manifest["physics"]["kinetic_cutoff_hartree"],
                        "reduced_kpoint": list(_selected_point(manifest)),
                    },
                    "cases": case_reports,
                    "capture": capture,
                    "interpretation": {
                        "stage_medians_are_independently_synchronized": True,
                        "stage_medians_are_additive": False,
                        "local_fft_substage_inputs_are_prematerialized": True,
                        "projector_bytes_are_logical_algorithmic_traffic": True,
                        "product_runtime_auto_route_modified": False,
                    },
                    "status": "diagnostic",
                }
            )
            generation.write_json("report.json", report)
            generation.publish()
    finally:
        context.nonlocal_operator.close()
    return {
        "status": "diagnostic",
        "artifact": str(Path(out)),
        "report": str(Path(out) / "report.json"),
        "capture": (str(Path(out) / str(capture["path"])) if capture_stage is not None else None),
        "profile_contract_fingerprint": identity["profile_contract_fingerprint"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m mlx_atomistic.benchmarks.dft_hpsi_profile")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gth-source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--vector-counts",
        type=_parse_vector_counts,
        default=DEFAULT_VECTOR_COUNTS,
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--capture-stage", choices=PROFILE_STAGES)
    parser.add_argument("--capture-vector-count", type=int)
    parser.add_argument("--capture-repetitions", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the benchmark-only Hpsi stage profiler."""

    args = _parser().parse_args(argv)
    result = run_profile(
        manifest_path=args.manifest,
        gth_source=args.gth_source,
        out=results_output_path(args.out),
        vector_counts=args.vector_counts,
        warmups=args.warmups,
        samples=args.samples,
        capture_stage=args.capture_stage,
        capture_vector_count=args.capture_vector_count,
        capture_repetitions=args.capture_repetitions,
    )
    if args.json:
        sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
