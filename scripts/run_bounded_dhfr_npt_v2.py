"""Run one frozen DHFR NPT v2 seed through separately bounded engine phases."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from mlx_atomistic.benchmarks import dhfr_npt_v2
from mlx_atomistic.benchmarks.dhfr_npt import DHFRNPTValidationError


def build_phase_command(
    *,
    contract: dict,
    contract_path: Path,
    prepared: Path,
    formal_root: Path,
    seed: int,
    engine: str,
    timeout_seconds: float,
    split_resume: bool,
    repo_root: Path | None = None,
) -> list[str]:
    """Build one child-only macOS supervisor command."""

    if timeout_seconds <= 0.0:
        raise DHFRNPTValidationError("formal seed exhausted its frozen timeout")
    root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else Path(repo_root).resolve()
    )
    trace = formal_root / f"seed-{seed}" / f"{engine}-memory.json"
    worker = [
        sys.executable,
        str(root / "scripts" / "run_openmm_mlx_dhfr_npt_v2.py"),
        "--stage",
        "formal",
        "--engine",
        engine,
        "--seed",
        str(seed),
        "--prepared",
        str(prepared),
        "--contract",
        str(contract_path),
        "--out",
        str(formal_root),
    ]
    if split_resume:
        worker.append("--split-resume")
    return [
        sys.executable,
        str(root / "scripts" / "run_bounded_process.py"),
        "--max-bytes",
        str(contract["resource_limits"]["process_tree_max_bytes"]),
        "--timeout-seconds",
        str(timeout_seconds),
        "--trace-out",
        str(trace),
        "--",
        *worker,
    ]


def run_seed(
    *,
    contract_path: Path,
    prepared: Path,
    formal_root: Path,
    seed: int,
    split_resume: bool,
) -> int:
    """Run MLX and OpenMM in separate bounded process trees."""

    contract = dhfr_npt_v2.load_contract(contract_path)
    if seed not in dhfr_npt_v2.FORMAL_SEEDS:
        raise DHFRNPTValidationError("formal seed must be 7 or 19")
    required_split = seed == int(contract["restart_gate"]["seed"])
    if split_resume is not required_split:
        raise DHFRNPTValidationError(
            "seed 7 requires --split-resume and seed 19 forbids it"
        )
    dhfr_npt_v2.freeze_check(
        contract_path=contract_path,
        prepared_dir=prepared,
        formal_root=formal_root,
    )
    total_timeout = float(
        contract["resource_limits"]["seed_timeout_seconds"][str(seed)]
    )
    deadline = time.monotonic() + total_timeout
    for engine in ("mlx", "openmm"):
        remaining = deadline - time.monotonic()
        command = build_phase_command(
            contract=contract,
            contract_path=contract_path,
            prepared=prepared,
            formal_root=formal_root,
            seed=seed,
            engine=engine,
            timeout_seconds=remaining,
            split_resume=split_resume and engine == "mlx",
        )
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
        dhfr_npt_v2.freeze_check(
            contract_path=contract_path,
            prepared_dir=prepared,
            formal_root=formal_root,
        )
    report = dhfr_npt_v2.reconcile_seed_directory(
        contract_path=contract_path,
        prepared_dir=prepared,
        formal_root=formal_root,
        seed=seed,
    )
    return 0 if report["status"] == "passed" else 1


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-resume", action="store_true")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=dhfr_npt_v2.DEFAULT_CONTRACT_PATH,
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one contract-bounded formal seed."""

    args = _parse_args(argv)
    return run_seed(
        contract_path=args.contract,
        prepared=args.prepared,
        formal_root=args.out,
        seed=args.seed,
        split_resume=args.split_resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
