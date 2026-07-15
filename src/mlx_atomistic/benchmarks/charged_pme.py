"""Prepare and measure deterministic charged-PME benchmark workloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.prep.io import JSON_NAME, NPZ_NAME, load_prepared_system, save_prepared_system
from mlx_atomistic.prep.supercell import (
    normalize_supercell_replicas,
    prepared_supercell_summary,
    replicate_prepared_system,
)

SUPERCELL_SUMMARY_NAME = "supercell_summary.json"


def prepare_payload(
    *,
    source: str | Path,
    replicas: object,
    out: str | Path,
    assignment_order: int | None = None,
    background_policy: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic prepared-system supercell benchmark artifact.

    Args:
        source: Source prepared-system directory.
        replicas: Three positive integer counts ``(nx, ny, nz)``.
        out: Caller-owned output directory.
        assignment_order: Optional PME assignment-order override.
        background_policy: Optional PME background policy override.

    Returns:
        A JSON-serializable success, blocked, or failed payload. Missing source
        inputs are reported as blocked and do not create the output directory.
    """

    source_path = Path(source)
    out_path = Path(out)
    replica_shape = normalize_supercell_replicas(replicas)
    required_paths = (source_path / JSON_NAME, source_path / NPZ_NAME)
    missing = [str(path) for path in required_paths if not path.is_file()]
    base = {
        "kind": "mlx_atomistic.charged_pme_prepare",
        "source": str(source_path),
        "out": str(out_path),
        "replicas": list(replica_shape),
        "assignment_order_override": assignment_order,
        "background_policy_override": background_policy,
        "written": False,
    }
    if missing:
        return {
            **base,
            "status": "blocked",
            "blockers": ["missing_prepared_source:" + path for path in missing],
            "summary": None,
        }

    try:
        source_prepared = load_prepared_system(source_path)
        replicated = replicate_prepared_system(
            source_prepared,
            replica_shape,
            assignment_order=assignment_order,
            background_policy=background_policy,
        )
        summary = prepared_supercell_summary(
            replicated,
            source_atom_count=source_prepared.atom_count,
            replicas=replica_shape,
        )
        summary.update(
            _supercell_validation_summary(
                source_prepared,
                replicated,
                replica_shape,
            )
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        return {
            **base,
            "status": "failed",
            "blockers": [f"prepared_supercell_failed:{type(exc).__name__}:{exc}"],
            "summary": None,
        }

    save_prepared_system(replicated, out_path)
    summary_path = out_path / SUPERCELL_SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {
        **base,
        "status": "ok",
        "blockers": [],
        "written": True,
        "summary_path": str(summary_path),
        "prepared_json": str(out_path / JSON_NAME),
        "prepared_npz": str(out_path / NPZ_NAME),
        "summary": summary,
    }


def _supercell_validation_summary(source, replicated, replicas) -> dict[str, Any]:
    replica_count = int(np.prod(replicas, dtype=np.int64))
    indexed_names = (
        "bonds",
        "angles",
        "dihedrals",
        "rb_dihedrals",
        "constraints",
        "impropers",
        "nonbonded_pairs",
        "nonbonded_exception_pairs",
        "charmm_cmap_terms",
        "urey_bradley_terms",
        "nbfix_pairs",
        "virtual_site_parent_atoms",
    )
    indexed_count_checks = {
        name: {
            "source": int(np.asarray(getattr(source, name)).shape[0]),
            "actual": int(np.asarray(getattr(replicated, name)).shape[0]),
            "expected": int(np.asarray(getattr(source, name)).shape[0]) * replica_count,
        }
        for name in indexed_names
    }
    source_charge = float(np.sum(source.charges, dtype=np.float64))
    actual_charge = float(np.sum(replicated.charges, dtype=np.float64))
    expected_charge = source_charge * replica_count
    checks = {
        "atom_count": replicated.atom_count == source.atom_count * replica_count,
        "net_charge": bool(np.isclose(actual_charge, expected_charge, rtol=0.0, atol=1e-5)),
        "indexed_term_counts": all(
            item["actual"] == item["expected"] for item in indexed_count_checks.values()
        ),
    }
    return {
        "source_net_charge": source_charge,
        "expected_net_charge": expected_charge,
        "indexed_count_checks": indexed_count_checks,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _parse_replicas(value: str) -> tuple[int, int, int]:
    try:
        return normalize_supercell_replicas(tuple(int(item) for item in value.split(",")))
    except (TypeError, ValueError) as exc:
        msg = "--replicas must be three comma-separated positive integers"
        raise argparse.ArgumentTypeError(msg) from exc


def main(argv: list[str] | None = None) -> None:
    """Run the charged-PME benchmark command-line interface.

    Args:
        argv: Optional argument list; ``None`` reads process arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="replicate a prepared PME system")
    prepare_parser.add_argument("--source", type=Path, required=True)
    prepare_parser.add_argument("--replicas", type=_parse_replicas, required=True)
    prepare_parser.add_argument("--assignment-order", type=int, default=None)
    prepare_parser.add_argument("--background-policy", default=None)
    prepare_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        payload = prepare_payload(
            source=args.source,
            replicas=args.replicas,
            assignment_order=args.assignment_order,
            background_policy=args.background_policy,
            out=args.out,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        if payload["status"] != "ok":
            raise SystemExit(2)


if __name__ == "__main__":
    main()


__all__ = ["SUPERCELL_SUMMARY_NAME", "main", "prepare_payload"]
