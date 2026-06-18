"""Run the synthetic-LJ same-workload scaling ladder across MLX, OpenMM, LAMMPS.

This orchestrator drives all three engines at identical ``(particles, steps)``
on the local host, captures raw normalized JSON per engine under a gitignored
results directory, and aggregates them into a single MLX-vs-reference scaling
summary via :mod:`mlx_atomistic.benchmarks.same_workload_compare`.

Throughput is compared as ``steps_per_s``. Reduced-unit ``ns/day`` is not
cross-engine comparable for Lennard-Jones, so it is intentionally not reported.
MLX and LAMMPS share reduced LJ units and the same ``fcc_lattice`` geometry at
density 0.8 (genuine same physics); OpenMM runs the same particle/step counts in
physical units and is a throughput reference, not a bit-identical workload.

OpenMM and LAMMPS commands live here under ``scripts/`` per the reference-engine
boundary; the MLX command stays in the ``mlx_atomistic.benchmarks`` module.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from mlx_atomistic.benchmarks import same_workload_compare as swc

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIZES = "1000,4000,16000,50000"
# Per-size step counts: many steps where a step is cheap (small N), few where it
# is expensive (large N), so each point reaches steady state without an
# unbounded 50k run. Steps are matched across engines within each size.
DEFAULT_STEPS = "3000,2000,800,300"
DEFAULT_OUT_DIR = "results/same-workload-lj-scaling"
DEFAULT_TIMEOUT_S = 60 * 60


def _parse_int_list(arg: str, count: int) -> list[int]:
    """Parse a comma-separated int list, broadcasting a single value to ``count``."""

    values = [int(token) for token in arg.split(",") if token.strip()]
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise SystemExit(
            f"expected 1 or {count} comma-separated values, got {len(values)}: {arg!r}"
        )
    return values


def _capture_json(cmd: list[str], out_path: Path, *, timeout: int) -> dict[str, Any] | None:
    """Run a benchmark command, persist its stdout, and return parsed JSON."""

    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"  ! timed out after {timeout}s: {' '.join(cmd)}", file=sys.stderr)
        return None
    if result.stdout:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.stdout)
    if result.returncode != 0:
        print(
            f"  ! exit {result.returncode}: {' '.join(cmd)}\n{result.stderr.strip()[:400]}",
            file=sys.stderr,
        )
    try:
        return json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        print(f"  ! could not parse JSON from: {' '.join(cmd)}", file=sys.stderr)
        return None


def run_ladder(
    *,
    sizes: list[int],
    steps_list: list[int],
    warmup_steps: int,
    density: float,
    dt: float,
    out_dir: Path,
    openmm_platform: str,
    openmm_spacing_nm: float,
    opencl_platform: str,
    opencl_device: str,
    run_openmm: bool,
    run_lammps: bool,
    block_size: int = 1,
    neighbor_skin: float = 0.4,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Run the ladder on all selected engines and return the scaling summary.

    Each size uses its own step count (``steps_list``) so cheap small systems
    get enough steps to reach steady state while expensive large systems stay
    bounded in wall time. Steps are matched across engines per size, which is
    what the parity gate requires.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    mlx_payloads: list[dict[str, Any]] = []
    openmm_payloads: list[dict[str, Any]] = []
    lammps_payloads: list[dict[str, Any]] = []

    for n, steps in zip(sizes, steps_list, strict=True):
        print(f"MLX md_performance N={n} steps={steps}")
        mlx = _capture_json(
            [
                sys.executable,
                "-m",
                "mlx_atomistic.benchmarks.md_performance",
                "--sizes",
                str(n),
                "--steps",
                str(steps),
                "--dt",
                str(dt),
                "--block-size",
                str(block_size),
                "--neighbor-skin",
                str(neighbor_skin),
                "--json",
            ],
            out_dir / f"mlx-lj-N{n}.json",
            timeout=timeout,
        )
        if mlx is not None:
            mlx_payloads.append(mlx)
        if run_openmm:
            print(f"OpenMM OpenCL N={n} steps={steps}")
            payload = _capture_json(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "benchmark_openmm_opencl.py"),
                    "--platform",
                    openmm_platform,
                    "--particles",
                    str(n),
                    "--steps",
                    str(steps),
                    "--warmup-steps",
                    str(warmup_steps),
                    "--spacing-nm",
                    str(openmm_spacing_nm),
                    "--json",
                ],
                out_dir / f"openmm-lj-N{n}.json",
                timeout=timeout,
            )
            if payload is not None:
                openmm_payloads.append(payload)
        if run_lammps:
            print(f"LAMMPS OpenCL N={n} steps={steps}")
            payload = _capture_json(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "benchmark_lammps_opencl.py"),
                    "--particles",
                    str(n),
                    "--steps",
                    str(steps),
                    "--warmup-steps",
                    str(warmup_steps),
                    "--density",
                    str(density),
                    "--dt",
                    str(dt),
                    "--opencl-platform",
                    opencl_platform,
                    "--opencl-device",
                    opencl_device,
                    "--json",
                ],
                out_dir / f"lammps-lj-N{n}.json",
                timeout=timeout,
            )
            if payload is not None:
                lammps_payloads.append(payload)

    summary = swc.build_scaling_summary(
        mlx_payloads=mlx_payloads,
        openmm_payloads=openmm_payloads,
        lammps_payloads=lammps_payloads,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _fmt(value: Any, spec: str = "10.1f") -> str:
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return f"{'-':>{int(spec.split('.')[0])}}"


def format_table(summary: dict[str, Any]) -> str:
    """Render a TUI-friendly scaling table from the summary payload."""

    header = (
        f"{'atoms':>8} {'mlx s/s':>10} {'openmm s/s':>11} {'omm/mlx':>8} "
        f"{'lammps s/s':>11} {'lmp/mlx':>8}  status"
    )
    lines = [header, "-" * len(header)]
    for row in summary.get("cases", []):
        lines.append(
            f"{row.get('atom_count', '-'):>8} "
            f"{_fmt(row.get('mlx_steps_per_s')):>10} "
            f"{_fmt(row.get('openmm_steps_per_s')):>11} "
            f"{_fmt(row.get('openmm_to_mlx_ratio'), '8.2f'):>8} "
            f"{_fmt(row.get('lammps_steps_per_s')):>11} "
            f"{_fmt(row.get('lammps_to_mlx_ratio'), '8.2f'):>8}  "
            f"{row.get('comparison_status')}"
        )
    lines.append("")
    lines.append(
        "ratio = reference_steps_per_s / mlx_steps_per_s (>1 means the reference is faster)"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default=DEFAULT_SIZES, help="comma-separated particle counts")
    parser.add_argument(
        "--steps",
        default=DEFAULT_STEPS,
        help="per-size steps (one value broadcasts to all sizes, or one per size)",
    )
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument(
        "--density", type=float, default=0.8, help="reduced LJ density (MLX fixed 0.8)"
    )
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--openmm-platform", default="OpenCL")
    parser.add_argument("--openmm-spacing-nm", type=float, default=0.5)
    parser.add_argument("--opencl-platform", default="0")
    parser.add_argument("--opencl-device", default="0")
    parser.add_argument("--skip-openmm", action="store_true")
    parser.add_argument("--skip-lammps", action="store_true")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--block-size",
        type=int,
        default=1,
        help="MLX batched-block size (>1 enables the compiled NVT fast path)",
    )
    parser.add_argument(
        "--neighbor-skin",
        type=float,
        default=0.4,
        help="MLX neighbor-list skin (larger skin => fewer rebuilds, pairs with batched blocks)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the summary JSON instead of a table"
    )
    args = parser.parse_args(argv)

    sizes = [int(token) for token in args.sizes.split(",") if token.strip()]
    steps_list = _parse_int_list(args.steps, len(sizes))
    summary = run_ladder(
        sizes=sizes,
        steps_list=steps_list,
        warmup_steps=args.warmup_steps,
        density=args.density,
        dt=args.dt,
        out_dir=args.out_dir,
        openmm_platform=args.openmm_platform,
        openmm_spacing_nm=args.openmm_spacing_nm,
        opencl_platform=args.opencl_platform,
        opencl_device=args.opencl_device,
        run_openmm=not args.skip_openmm,
        run_lammps=not args.skip_lammps,
        block_size=args.block_size,
        neighbor_skin=args.neighbor_skin,
        timeout=args.timeout,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print()
        print(format_table(summary))
        print(f"\nraw outputs + summary.json under {args.out_dir}")


if __name__ == "__main__":
    main()
