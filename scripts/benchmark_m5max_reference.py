"""M5 Max reference benchmark harness for OpenMM and LAMMPS.

This script is the repo-controlled command surface for reference-engine
benchmark provenance, execution, and validation. It intentionally keeps OpenMM
and LAMMPS outside the `mlx_atomistic` product runtime.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from mlx_atomistic.benchmarks import get_hardware_info, normalize_benchmark_payload

BENCHMARK_NAME = "m5max_reference_environment"
COMMAND = "uv run python scripts/benchmark_m5max_reference.py"
ENGINE = "reference-environment"
FIXTURE = "m5max_reference"
MANIFEST_VERSION = "m5max-reference-manifest-v1"
RESULT_ROOT = Path("results/m5max-reference")
TIMING_METRIC = "environment_probe"
LAMMPS_BENCHMARK_NAME = "m5max_lammps_official"
LAMMPS_TIMING_METRIC = "loop_time_s"
OPENMM_BENCHMARK_NAME = "m5max_openmm_official"
OPENMM_TIMING_METRIC = "ns_per_day"
OPENMM_CASE_TIMEOUT_S = 20 * 60

KNOWN_LAMMPS_GPU_STYLES = {
    "fix": {
        "npt/gpu",
        "nve/gpu",
    },
    "kspace": {
        "pppm/gpu",
    },
    "pair": {
        "eam/gpu",
        "lj/charmm/coul/long/gpu",
        "lj/cut/gpu",
    },
}

LAMMPS_CASES: dict[str, dict[str, Any]] = {
    "lj": {
        "input_script": "in.lj",
        "dependencies": (),
        "description": "atomic Lennard-Jones fluid",
        "styles": (
            {"kind": "pair", "style": "lj/cut", "gpu_style": "lj/cut/gpu"},
            {"kind": "fix", "style": "nve", "gpu_style": "nve/gpu"},
        ),
    },
    "chain": {
        "input_script": "in.chain",
        "dependencies": ("data.chain",),
        "description": "FENE bead-spring polymer melt",
        "styles": (
            {"kind": "bond", "style": "fene", "gpu_style": None},
            {"kind": "pair", "style": "lj/cut", "gpu_style": "lj/cut/gpu"},
            {"kind": "fix", "style": "nve", "gpu_style": "nve/gpu"},
            {"kind": "fix", "style": "langevin", "gpu_style": None},
        ),
    },
    "eam": {
        "input_script": "in.eam",
        "dependencies": ("Cu_u3.eam",),
        "description": "bulk Cu EAM solid",
        "styles": (
            {"kind": "pair", "style": "eam", "gpu_style": "eam/gpu"},
            {"kind": "fix", "style": "nve", "gpu_style": "nve/gpu"},
        ),
    },
    "chute": {
        "input_script": "in.chute",
        "dependencies": ("data.chute",),
        "description": "granular chute flow",
        "styles": (
            {"kind": "pair", "style": "gran/hooke/history", "gpu_style": None},
            {"kind": "fix", "style": "gravity", "gpu_style": None},
            {"kind": "fix", "style": "freeze", "gpu_style": None},
            {"kind": "fix", "style": "nve/sphere", "gpu_style": None},
        ),
    },
    "rhodo": {
        "input_script": "in.rhodo",
        "dependencies": ("data.rhodo",),
        "description": "rhodopsin protein in solvated lipid bilayer",
        "styles": (
            {"kind": "bond", "style": "harmonic", "gpu_style": None},
            {"kind": "angle", "style": "charmm", "gpu_style": None},
            {"kind": "dihedral", "style": "charmm", "gpu_style": None},
            {"kind": "improper", "style": "harmonic", "gpu_style": None},
            {
                "kind": "pair",
                "style": "lj/charmm/coul/long",
                "gpu_style": "lj/charmm/coul/long/gpu",
            },
            {"kind": "kspace", "style": "pppm", "gpu_style": "pppm/gpu"},
            {"kind": "fix", "style": "shake", "gpu_style": None},
            {"kind": "fix", "style": "npt", "gpu_style": "npt/gpu"},
        ),
    },
}

OPENMM_CASES: dict[str, dict[str, Any]] = {
    "dhfr": {
        "description": "DHFR GBSA/RF/PME official benchmark group",
        "tests": "gbsa,rf,pme",
        "working_directory": "vendors/openmm/examples/benchmarks",
        "script": "benchmark.py",
        "output": "results/m5max-reference/openmm/dhfr.json",
        "inputs": (
            "vendors/openmm/examples/benchmarks/5dfr_minimized.pdb",
            "vendors/openmm/examples/benchmarks/5dfr_solv-cube_equil.pdb",
        ),
        "external_inputs": (),
    },
    "apoa1": {
        "description": "ApoA1 RF/PME/LJPME official benchmark group",
        "tests": "apoa1rf,apoa1pme,apoa1ljpme",
        "working_directory": "vendors/openmm/examples/benchmarks",
        "script": "benchmark.py",
        "output": "results/m5max-reference/openmm/apoa1.json",
        "inputs": ("vendors/openmm/examples/benchmarks/apoa1.pdb",),
        "external_inputs": (),
    },
    "amber20": {
        "description": "Amber20 Cellulose and STMV official benchmark group",
        "tests": "amber20-cellulose,amber20-stmv",
        "working_directory": "results/inputs",
        "script": "../../vendors/openmm/examples/benchmarks/benchmark.py",
        "output": "results/m5max-reference/openmm/amber20.json",
        "inputs": (),
        "external_inputs": ("https://ambermd.org/Amber20_Benchmark_Suite.tar.gz",),
    },
}


def main() -> None:
    args = _parse_args()
    if args.command == "environment":
        payload = build_environment_payload(output_path=args.output)
    elif args.command == "lammps":
        payload = build_lammps_payload(
            cases=args.case or ["all"],
            classify_only=args.classify_only,
            output_root=args.output_root,
        )
    elif args.command == "openmm":
        payload = build_openmm_payload(
            cases=args.case or ["all"],
            dry_run=args.dry_run,
            output_root=args.output_root,
            seconds=args.seconds,
        )
    elif args.command == "run":
        payload = run_suite(output_root=args.output_root, seconds=args.seconds)
    elif args.command == "validate":
        payload = validate_manifest(args.manifest)
    else:  # pragma: no cover - argparse choices keep this unreachable.
        raise ValueError(f"unsupported command {args.command!r}")

    if getattr(args, "output", None) is not None:
        _write_json(args.output, payload)
    if args.command == "run":
        _write_json(args.output_root / "manifest.json", payload)
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "validate":
        print(_format_validation(payload))
    else:
        print(_format_payload(payload))


def run_suite(*, output_root: Path = RESULT_ROOT, seconds: float = 30.0) -> dict[str, Any]:
    """Run the full reference benchmark suite and write manifest artifacts."""

    environment = build_environment_payload(output_path=output_root / "environment.json")
    _write_json(output_root / "environment.json", environment)
    lammps_payload = build_lammps_payload(
        cases=["all"],
        classify_only=False,
        output_root=output_root,
    )
    _write_json(output_root / "lammps" / "manifest.json", lammps_payload)
    openmm_payload = build_openmm_payload(
        cases=["all"],
        dry_run=False,
        output_root=output_root,
        seconds=seconds,
    )
    _write_json(output_root / "openmm" / "manifest.json", openmm_payload)

    statuses = {
        "environment": environment["status"],
        "lammps": lammps_payload["status"],
        "openmm": openmm_payload["status"],
    }
    status = "blocked" if "blocked" in statuses.values() else (
        "diagnostic" if "diagnostic" in statuses.values() else "ok"
    )
    blockers = [
        f"{name}: {payload.get('blocker')}"
        for name, payload in (
            ("environment", environment),
            ("lammps", lammps_payload),
            ("openmm", openmm_payload),
        )
        if payload.get("blocker")
    ]
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "status": status,
        "blocker": "; ".join(blockers) if blockers else None,
        "created_at_unix": time.time(),
        "result_root": str(output_root),
        "environment": environment,
        "lammps": lammps_payload,
        "openmm": openmm_payload,
        "required_cases": {
            "openmm": sorted(OPENMM_CASES),
            "lammps": sorted(LAMMPS_CASES),
        },
    }
    return manifest


def build_environment_payload(*, output_path: Path | None = None) -> dict[str, Any]:
    """Return normalized environment provenance for reference benchmark runs."""

    openmm_probe = _probe_openmm()
    lammps_probe = _probe_lammps()
    case_statuses = {openmm_probe["status"], lammps_probe["status"]}
    status = "ok" if case_statuses == {"ok"} else "diagnostic"
    blocker = None
    if status != "ok":
        blockers = [
            f"{case['engine']}: {case['blocker']}"
            for case in (openmm_probe, lammps_probe)
            if case.get("blocker")
        ]
        blocker = "; ".join(blockers) or "reference environment probe incomplete"

    raw_output_path = output_path or RESULT_ROOT / "environment.json"
    payload = {
        "status": status,
        "benchmark_name": BENCHMARK_NAME,
        "fixture": FIXTURE,
        "engine": ENGINE,
        "cases": [openmm_probe, lammps_probe],
        "case_count": 2,
        "command_surface": {
            "harness": f"{COMMAND} environment --json",
            "lammps_console_script": ".venv/bin/lmp",
            "lammps_console_script_policy": (
                "do not rely on the console script for final runs; "
                "use the packaged executable recorded by the harness"
            ),
        },
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "blocker": blocker,
        "finite": status == "ok",
    }
    return normalize_benchmark_payload(
        payload,
        benchmark_name=BENCHMARK_NAME,
        fixture=FIXTURE,
        timing_metric=TIMING_METRIC,
        hardware=get_hardware_info(),
        runtime={
            "python_version": platform.python_version(),
            "openmm_probe_status": openmm_probe["status"],
            "lammps_probe_status": lammps_probe["status"],
        },
        engine=ENGINE,
        finite=status == "ok",
        status=status,
        blocker=blocker,
        command=f"{COMMAND} environment --json",
        raw_output_path=raw_output_path,
    )


def build_openmm_payload(
    *,
    cases: list[str],
    dry_run: bool,
    output_root: Path = RESULT_ROOT,
    seconds: float = 30.0,
) -> dict[str, Any]:
    """Return OpenMM official benchmark dry-run records."""

    selected = list(OPENMM_CASES) if cases == ["all"] else cases
    unknown = sorted(set(selected) - set(OPENMM_CASES))
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"unknown OpenMM benchmark case(s): {names}")

    rows = [
        _build_openmm_case_record(
            case,
            dry_run=dry_run,
            output_root=output_root,
            seconds=seconds,
        )
        for case in selected
    ]
    statuses = {row["status"] for row in rows}
    if "blocked" in statuses:
        status = "blocked"
    elif "diagnostic" in statuses:
        status = "diagnostic"
    else:
        status = "ok"
    blocker = None
    if status != "ok":
        blockers = [f"{row['case']}: {row['blocker']}" for row in rows if row.get("blocker")]
        blocker = "; ".join(blockers) or "OpenMM benchmark suite requires diagnostics"

    payload = {
        "status": status,
        "benchmark_name": OPENMM_BENCHMARK_NAME,
        "fixture": "openmm_official_bench",
        "engine": "openmm-reference",
        "cases": rows,
        "case_count": len(rows),
        "required_cases": sorted(OPENMM_CASES),
        "execution_mode": "dry_run" if dry_run else "run",
        "blocker": blocker,
        "finite": status == "ok",
    }
    return normalize_benchmark_payload(
        payload,
        benchmark_name=OPENMM_BENCHMARK_NAME,
        fixture="openmm_official_bench",
        timing_metric=OPENMM_TIMING_METRIC,
        hardware=get_hardware_info(),
        runtime={
            "python_version": platform.python_version(),
            "platform": "OpenCL",
            "precision": "single",
            "seconds": seconds,
        },
        engine="openmm-reference",
        finite=status == "ok",
        status=status,
        blocker=blocker,
        command=_openmm_harness_command(cases=selected, dry_run=dry_run),
        raw_output_path=output_root / "openmm" / "manifest.json",
    )


def _build_openmm_case_record(
    case: str,
    *,
    dry_run: bool,
    output_root: Path,
    seconds: float,
) -> dict[str, Any]:
    repo_root = _repo_root()
    spec = OPENMM_CASES[case]
    raw_output_path = output_root / "openmm" / f"{case}.json"
    upstream_raw_output_path = output_root / "openmm" / f"{case}-upstream.json"
    missing = [
        path
        for path in spec["inputs"]
        if not (repo_root / path).exists()
    ]
    if missing:
        status = "blocked"
        blocker = "missing OpenMM benchmark input(s): " + ", ".join(missing)
    elif dry_run:
        status = "diagnostic"
        blocker = "dry run; benchmark not executed"
    else:
        status = "ok"
        blocker = None
    record = {
        "engine": "openmm-reference",
        "benchmark_name": OPENMM_BENCHMARK_NAME,
        "case": case,
        "fixture": f"openmm_{case}",
        "system": case,
        "description": spec["description"],
        "tests": spec["tests"],
        "platform": "OpenCL",
        "precision": "single",
        "seconds": seconds,
        "timeout_s": OPENMM_CASE_TIMEOUT_S,
        "working_directory": spec["working_directory"],
        "script": spec["script"],
        "inputs": list(spec["inputs"]),
        "external_inputs": list(spec["external_inputs"]),
        "input_policy": "external inputs stay under results/inputs",
        "raw_output_path": str(raw_output_path),
        "upstream_raw_output_path": str(upstream_raw_output_path),
        "command": _openmm_case_command(
            case=case,
            output_path=upstream_raw_output_path,
            seconds=seconds,
        ),
        "status": status,
        "blocker": blocker,
        "ns_per_day": None,
        "timing_metric": OPENMM_TIMING_METRIC,
        "timing_value": None,
        "timing_unit": "ns/day",
        "finite": False,
    }
    if not dry_run and not missing:
        record = _execute_openmm_case(record)
    return record


def build_lammps_payload(
    *,
    cases: list[str],
    classify_only: bool,
    output_root: Path = RESULT_ROOT,
) -> dict[str, Any]:
    """Return LAMMPS official benchmark classification records."""

    selected = sorted(LAMMPS_CASES) if cases == ["all"] else cases
    unknown = sorted(set(selected) - set(LAMMPS_CASES))
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"unknown LAMMPS benchmark case(s): {names}")

    rows = [
        _build_lammps_case_record(case, classify_only=classify_only, output_root=output_root)
        for case in selected
    ]
    statuses = {row["status"] for row in rows}
    if "blocked" in statuses:
        status = "blocked"
    elif "diagnostic" in statuses:
        status = "diagnostic"
    else:
        status = "ok"
    blocker = None
    if status != "ok":
        blockers = [f"{row['case']}: {row['blocker']}" for row in rows if row.get("blocker")]
        blocker = "; ".join(blockers) or "LAMMPS benchmark suite requires diagnostics"

    payload = {
        "status": status,
        "benchmark_name": LAMMPS_BENCHMARK_NAME,
        "fixture": "lammps_official_bench",
        "engine": "lammps-reference",
        "cases": rows,
        "case_count": len(rows),
        "required_cases": sorted(LAMMPS_CASES),
        "execution_mode": "classify_only" if classify_only else "run",
        "blocker": blocker,
        "finite": status == "ok",
    }
    return normalize_benchmark_payload(
        payload,
        benchmark_name=LAMMPS_BENCHMARK_NAME,
        fixture="lammps_official_bench",
        timing_metric=LAMMPS_TIMING_METRIC,
        hardware=get_hardware_info(),
        runtime={
            "python_version": platform.python_version(),
            "classification_source": "repo_static_lammps_official_inputs",
            "known_gpu_style_source": "local_lammps_help_snapshot",
        },
        engine="lammps-reference",
        finite=status == "ok",
        status=status,
        blocker=blocker,
        command=_lammps_harness_command(cases=selected, classify_only=classify_only),
        raw_output_path=output_root / "lammps" / "manifest.json",
    )


def _build_lammps_case_record(
    case: str,
    *,
    classify_only: bool,
    output_root: Path,
) -> dict[str, Any]:
    repo_root = _repo_root()
    bench_root = repo_root / "vendors" / "lammps" / "bench"
    spec = LAMMPS_CASES[case]
    input_script = bench_root / spec["input_script"]
    dependency_paths = [bench_root / path for path in spec["dependencies"]]
    missing = [
        str(path.relative_to(repo_root))
        for path in (input_script, *dependency_paths)
        if not path.exists()
    ]
    styles = [_style_mapping(style) for style in spec["styles"]]
    acceleration_class = _acceleration_class(styles)
    work_dir = output_root / "lammps" / case / "work"
    raw_output_path = output_root / "lammps" / f"{case}.json"
    command = _lammps_case_command(case=case, classify_only=classify_only)

    if missing:
        status = "blocked"
        blocker = "missing LAMMPS benchmark input(s): " + ", ".join(missing)
    elif classify_only:
        status = "diagnostic"
        blocker = "classification only; benchmark not executed"
    elif acceleration_class == "cpu_only_diagnostic":
        status = "diagnostic"
        blocker = "official input has no mapped GPU/OpenCL styles in this build"
    elif acceleration_class == "partial_gpu_opencl":
        status = "diagnostic"
        blocker = "official input has partial GPU/OpenCL style coverage"
    else:
        status = "ok"
        blocker = None

    record = {
        "engine": "lammps-reference",
        "benchmark_name": LAMMPS_BENCHMARK_NAME,
        "case": case,
        "fixture": f"lammps_{case}",
        "system": case,
        "description": spec["description"],
        "input_script": str(input_script.relative_to(repo_root)),
        "dependencies": [str(path.relative_to(repo_root)) for path in dependency_paths],
        "work_dir": str(work_dir),
        "raw_output_path": str(raw_output_path),
        "styles": styles,
        "acceleration_classification": acceleration_class,
        "status": status,
        "blocker": blocker,
        "command": command,
        "loop_time_s": None,
        "steps": 100,
        "atom_count": 32000,
        "timing_metric": LAMMPS_TIMING_METRIC,
        "timing_value": None,
        "timing_unit": "s",
        "finite": False,
    }
    if not classify_only and not missing:
        record = _execute_lammps_case(record)
    return record


def _style_mapping(style: dict[str, str | None]) -> dict[str, Any]:
    kind = str(style["kind"])
    gpu_style = style.get("gpu_style")
    gpu_available = bool(gpu_style and gpu_style in KNOWN_LAMMPS_GPU_STYLES.get(kind, set()))
    return {
        "kind": kind,
        "style": style["style"],
        "gpu_style": gpu_style,
        "gpu_available": gpu_available,
    }


def _acceleration_class(styles: list[dict[str, Any]]) -> str:
    available = [bool(style["gpu_available"]) for style in styles]
    if all(available):
        return "full_gpu_opencl"
    if any(available):
        return "partial_gpu_opencl"
    return "cpu_only_diagnostic"


def _probe_openmm() -> dict[str, Any]:
    try:
        import openmm
        from openmm import Platform
    except Exception as exc:  # pragma: no cover - optional reference package.
        return {
            "engine": "openmm-reference",
            "status": "diagnostic",
            "version": None,
            "available_platforms": [],
            "opencl_available": False,
            "opencl_platform_properties": [],
            "blocker": f"OpenMM import unavailable: {type(exc).__name__}: {exc}",
        }

    platforms = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
    opencl_properties: list[str] = []
    if "OpenCL" in platforms:
        try:
            opencl = Platform.getPlatformByName("OpenCL")
            opencl_properties = list(opencl.getPropertyNames())
        except Exception:
            opencl_properties = []
    return {
        "engine": "openmm-reference",
        "status": "ok",
        "version": openmm.version.version,
        "available_platforms": platforms,
        "opencl_available": "OpenCL" in platforms,
        "opencl_platform_properties": opencl_properties,
        "blocker": None,
    }


def _probe_lammps() -> dict[str, Any]:
    try:
        dist = metadata.distribution("lammps")
        import lammps
    except Exception as exc:  # pragma: no cover - optional reference package.
        return {
            "engine": "lammps-reference",
            "status": "diagnostic",
            "version": None,
            "package_path": None,
            "packaged_executable": None,
            "packaged_executable_exists": False,
            "build_config": {},
            "gpu_api": None,
            "gpu_precision": None,
            "pkg_gpu": None,
            "blocker": f"LAMMPS import unavailable: {type(exc).__name__}: {exc}",
        }

    package_path = Path(lammps.__file__).resolve().parent
    build_config = _read_lammps_build_config(dist)
    config_settings = build_config.get("config_settings", {})
    executable = package_path / "lmp"
    return {
        "engine": "lammps-reference",
        "status": "ok",
        "version": str(getattr(lammps, "__version__", "")),
        "package_path": str(package_path),
        "packaged_executable": str(executable),
        "packaged_executable_exists": executable.exists(),
        "build_config": build_config,
        "gpu_api": config_settings.get("cmake.define.GPU_API"),
        "gpu_precision": config_settings.get("cmake.define.GPU_PREC"),
        "pkg_gpu": config_settings.get("cmake.define.PKG_GPU"),
        "blocker": None,
    }


def _read_lammps_build_config(dist: metadata.Distribution) -> dict[str, Any]:
    for file in dist.files or ():
        if str(file).endswith("uv_build.json"):
            path = Path(dist.locate_file(file))
            try:
                return json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                return {}
    return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def validate_manifest(path: Path) -> dict[str, Any]:
    """Validate the suite manifest required by the plan."""

    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "blocked",
            "manifest": str(path),
            "blocker": f"manifest unreadable: {type(exc).__name__}: {exc}",
        }

    blockers: list[str] = []
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        blockers.append("manifest_version mismatch")
    openmm_cases = _case_names(manifest.get("openmm", {}))
    lammps_cases = _case_names(manifest.get("lammps", {}))
    if openmm_cases != set(OPENMM_CASES):
        blockers.append(f"OpenMM cases mismatch: {sorted(openmm_cases)}")
    if lammps_cases != set(LAMMPS_CASES):
        blockers.append(f"LAMMPS cases mismatch: {sorted(lammps_cases)}")
    for section in ("environment", "openmm", "lammps"):
        payload = manifest.get(section)
        if not isinstance(payload, dict):
            blockers.append(f"missing manifest section: {section}")
            continue
        if payload.get("status") not in {"ok", "blocked", "diagnostic"}:
            blockers.append(f"{section} has invalid status {payload.get('status')!r}")
    for section in ("openmm", "lammps"):
        for case in manifest.get(section, {}).get("cases", []):
            raw_path = case.get("raw_output_path")
            if raw_path and not (_repo_root() / raw_path).exists():
                blockers.append(f"missing raw output: {raw_path}")
    status = "blocked" if blockers else "ok"
    return {
        "status": status,
        "manifest": str(path),
        "blocker": "; ".join(blockers) if blockers else None,
        "openmm_cases": sorted(openmm_cases),
        "lammps_cases": sorted(lammps_cases),
    }


def _case_names(payload: dict[str, Any]) -> set[str]:
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        return set()
    return {str(case.get("case")) for case in cases if isinstance(case, dict)}


def _format_payload(payload: dict[str, Any]) -> str:
    if payload.get("benchmark_name") == BENCHMARK_NAME:
        return _format_environment(payload)
    return f"{payload.get('benchmark_name', 'm5max_reference')}: {payload['status']}"


def _format_environment(payload: dict[str, Any]) -> str:
    cases = ", ".join(f"{case['engine']}={case['status']}" for case in payload["cases"])
    return f"M5 Max reference environment: {payload['status']} ({cases})"


def _format_validation(payload: dict[str, Any]) -> str:
    if payload["status"] == "ok":
        return f"Manifest valid: {payload['manifest']}"
    return f"Manifest blocked: {payload['blocker']}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _lammps_harness_command(*, cases: list[str], classify_only: bool) -> str:
    parts = [COMMAND, "lammps"]
    if classify_only:
        parts.append("--classify-only")
    if cases != sorted(LAMMPS_CASES):
        for case in cases:
            parts.extend(("--case", case))
    parts.append("--json")
    return " ".join(parts)


def _lammps_case_command(*, case: str, classify_only: bool) -> str:
    command = _lammps_harness_command(cases=[case], classify_only=classify_only)
    return re.sub(r"\s+", " ", command).strip()


def _openmm_harness_command(*, cases: list[str], dry_run: bool) -> str:
    parts = [COMMAND, "openmm"]
    if dry_run:
        parts.append("--dry-run")
    if cases != sorted(OPENMM_CASES):
        for case in cases:
            parts.extend(("--case", case))
    parts.append("--json")
    return " ".join(parts)


def _openmm_case_command(*, case: str, output_path: Path, seconds: float) -> str:
    spec = OPENMM_CASES[case]
    working_directory = Path(spec["working_directory"])
    script = spec["script"]
    outfile = _absolute_output_path(output_path)
    return (
        f"cd {working_directory} && uv run --project {_uv_project_for_workdir(working_directory)} "
        f"python {script} --platform OpenCL --test {spec['tests']} --seconds {seconds:g} "
        f"--precision single --outfile {outfile}"
    )


def _absolute_output_path(output_path: Path) -> str:
    return str((_repo_root() / output_path).resolve())


def _uv_project_for_workdir(working_directory: Path) -> str:
    depth = len(working_directory.parts)
    return "/".join([".."] * depth) or "."


def _execute_openmm_case(record: dict[str, Any]) -> dict[str, Any]:
    repo_root = _repo_root()
    working_directory = repo_root / record["working_directory"]
    working_directory.mkdir(parents=True, exist_ok=True)
    upstream = repo_root / record["upstream_raw_output_path"]
    upstream.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "uv",
        "run",
        "--project",
        _uv_project_for_workdir(Path(record["working_directory"])),
        "python",
        record["script"],
        "--platform",
        "OpenCL",
        "--test",
        record["tests"],
        "--seconds",
        str(record["seconds"]),
        "--precision",
        "single",
        "--outfile",
        _absolute_output_path(Path(record["upstream_raw_output_path"])),
    ]
    try:
        result = subprocess.run(
            args,
            cwd=working_directory,
            text=True,
            capture_output=True,
            check=False,
            timeout=float(record["timeout_s"]),
        )
    except subprocess.TimeoutExpired as exc:
        record["command_args"] = args
        record["stdout"] = _subprocess_output_text(exc.stdout)
        record["stderr"] = _subprocess_output_text(exc.stderr)
        record["returncode"] = None
        record.update(
            {
                "status": "blocked",
                "blocker": f"OpenMM command exceeded timeout_s={record['timeout_s']}",
                "finite": False,
            }
        )
        _write_json(repo_root / record["raw_output_path"], record)
        return record

    record["command_args"] = args
    record["stdout"] = result.stdout
    record["stderr"] = result.stderr
    record["returncode"] = result.returncode
    if result.returncode != 0:
        failure_excerpt = _last_nonempty_line(result.stderr) or _last_nonempty_line(result.stdout)
        record.update(
            {
                "status": "blocked",
                "blocker": (
                    f"OpenMM command failed with return code {result.returncode}: "
                    f"{failure_excerpt}"
                    if failure_excerpt
                    else f"OpenMM command failed with return code {result.returncode}"
                ),
                "failure_excerpt": failure_excerpt,
                "finite": False,
            }
        )
    else:
        benchmarks = _load_openmm_benchmarks(upstream)
        if benchmarks:
            values = [
                float(row["ns_per_day"])
                for row in benchmarks
                if row.get("ns_per_day") is not None
            ]
            record.update(
                {
                    "status": "ok",
                    "blocker": None,
                    "benchmarks": benchmarks,
                    "benchmark_count": len(benchmarks),
                    "ns_per_day": min(values) if values else None,
                    "timing_value": min(values) if values else None,
                    "finite": bool(values),
                }
            )
        else:
            record.update(
                {
                    "status": "diagnostic",
                    "blocker": "OpenMM command produced no benchmark rows",
                    "benchmarks": [],
                    "benchmark_count": 0,
                    "finite": False,
                }
            )
    _write_json(repo_root / record["raw_output_path"], record)
    return record


def _subprocess_output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _last_nonempty_line(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _load_openmm_benchmarks(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload.get("benchmarks", [])
    if not isinstance(rows, list):
        return []
    return rows


def _execute_lammps_case(record: dict[str, Any]) -> dict[str, Any]:
    repo_root = _repo_root()
    probe = _probe_lammps()
    executable = probe.get("packaged_executable")
    if not executable:
        record.update(
            {
                "status": "blocked",
                "blocker": "LAMMPS packaged executable unavailable",
                "finite": False,
            }
        )
        _write_json(repo_root / record["raw_output_path"], record)
        return record

    work_dir = repo_root / record["work_dir"]
    work_dir.mkdir(parents=True, exist_ok=True)
    for relative in (record["input_script"], *record["dependencies"]):
        source = repo_root / relative
        shutil.copy2(source, work_dir / source.name)
    screen_path = work_dir / "screen.txt"
    log_path = work_dir / "log.lammps"
    args = [
        str(executable),
        "-screen",
        str(screen_path),
        "-log",
        str(log_path),
        "-nocite",
        "-sf",
        "gpu",
        "-pk",
        "gpu",
        "1",
        "platform",
        "0",
        "-in",
        Path(record["input_script"]).name,
    ]
    result = subprocess.run(args, cwd=work_dir, text=True, capture_output=True, check=False)
    record["command_args"] = args
    record["stdout"] = result.stdout
    record["stderr"] = result.stderr
    record["returncode"] = result.returncode
    record["screen_output_path"] = str(screen_path.relative_to(repo_root))
    record["log_output_path"] = str(log_path.relative_to(repo_root))
    text = (
        _read_text(log_path) + "\n" + _read_text(screen_path) + "\n"
        + result.stdout + result.stderr
    )
    loop = _parse_lammps_loop_time(text)
    if result.returncode != 0:
        failure_excerpt = _first_lammps_error_line(text)
        record.update(
            {
                "status": "blocked",
                "blocker": (
                    f"LAMMPS command failed with return code {result.returncode}: "
                    f"{failure_excerpt}"
                    if failure_excerpt
                    else f"LAMMPS command failed with return code {result.returncode}"
                ),
                "failure_excerpt": failure_excerpt,
                "finite": False,
            }
        )
    elif loop is None:
        record.update(
            {
                "status": "diagnostic",
                "blocker": "LAMMPS command produced no parseable Loop time",
                "finite": False,
            }
        )
    else:
        loop_time_s, procs, steps, atoms = loop
        status = (
            "ok"
            if record["acceleration_classification"] == "full_gpu_opencl"
            else "diagnostic"
        )
        blocker = None
        if status == "diagnostic":
            blocker = f"{record['acceleration_classification']} case executed with caveats"
        record.update(
            {
                "status": status,
                "blocker": blocker,
                "loop_time_s": loop_time_s,
                "timing_value": loop_time_s,
                "timing_unit": "s",
                "procs": procs,
                "steps": steps,
                "atom_count": atoms,
                "finite": loop_time_s > 0.0,
            }
        )
    _write_json(repo_root / record["raw_output_path"], record)
    return record


def _first_lammps_error_line(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("ERROR:"):
            return stripped
    return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def _parse_lammps_loop_time(text: str) -> tuple[float, int, int, int] | None:
    match = re.search(
        r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs\s+for\s+"
        r"(\d+)\s+steps\s+with\s+(\d+)\s+atoms",
        text,
    )
    if match is None:
        return None
    return (
        float(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    environment = subparsers.add_parser("environment")
    environment.add_argument("--json", action="store_true")
    environment.add_argument("--output", type=Path)

    lammps = subparsers.add_parser("lammps")
    lammps.add_argument(
        "--case",
        action="append",
        choices=("all", *sorted(LAMMPS_CASES)),
        default=None,
    )
    lammps.add_argument("--classify-only", action="store_true")
    lammps.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    lammps.add_argument("--json", action="store_true")
    lammps.add_argument("--output", type=Path)

    openmm = subparsers.add_parser("openmm")
    openmm.add_argument(
        "--case",
        action="append",
        choices=("all", *sorted(OPENMM_CASES)),
        default=None,
    )
    openmm.add_argument("--dry-run", action="store_true")
    openmm.add_argument("--seconds", type=float, default=30.0)
    openmm.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    openmm.add_argument("--json", action="store_true")
    openmm.add_argument("--output", type=Path)

    run = subparsers.add_parser("run")
    run.add_argument("--seconds", type=float, default=30.0)
    run.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    run.add_argument("--json", action="store_true")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--json", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main()
