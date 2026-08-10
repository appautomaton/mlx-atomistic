"""Prepare and run the bounded diamond-carbon equation-of-state validation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mlx_atomistic._artifact_identity import canonical_json_bytes, sha256_bytes
from mlx_atomistic.benchmarks.dft_silicon import parse_gth_entry

WORKLOAD_SCHEMA = "mlx-atomistic.dft-carbon-workload.v1"
TARGET_ID = "diamond-carbon-conventional-pbe-gth-q4"
GTH_ELEMENT = "C"
GTH_NAME = "GTH-PBE-q4"

CARBON_FRACTIONAL_POSITIONS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
    (0.25, 0.25, 0.25),
    (0.25, 0.75, 0.75),
    (0.75, 0.25, 0.75),
    (0.75, 0.75, 0.25),
)


def _write_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"refusing to replace mismatched existing file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _unsigned_workload(*, resource_sha256: str, source_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": WORKLOAD_SCHEMA,
        "target_id": TARGET_ID,
        "resource": {
            "path": "resources/C-GTH-PBE-q4.gth",
            "sha256": resource_sha256,
            "source_database_sha256": source_sha256,
            "element": GTH_ELEMENT,
            "name": GTH_NAME,
            "functional": "PBE",
            "valence_charge": 4,
        },
        "system": {
            "name": "diamond-carbon-conventional-cubic",
            "atom_count": 8,
            "symbols": [GTH_ELEMENT] * 8,
            "fractional_positions": [list(row) for row in CARBON_FRACTIONAL_POSITIONS],
            "electron_count": 32,
            "spin_mode": "unpolarized",
            "occupancy_per_band": 2,
            "occupied_band_count": 16,
        },
        "physics": {
            "exchange_correlation": "PBE-PW92",
            "pseudopotential": "C GTH-PBE-q4",
            "kpoint_centering": "monkhorst-pack-even-half-shift",
            "occupation": "zero-temperature fixed occupations",
        },
        "solver": {
            "scf": {
                "max_iterations": 80,
                "min_iterations": 2,
                "density_tolerance": 1.0e-6,
                "energy_tolerance_hartree": 8.0e-6,
                "orbital_tolerance": 1.0e-6,
                "mixing_beta": 0.35,
                "mixer": "diis",
                "adaptive_eigensolver_tolerance": True,
                "initial_eigensolver_tolerance": 1.0e-2,
                "eigensolver_tolerance_scale": 0.1,
            },
            "davidson": {
                "max_iterations": 48,
                "tolerance": 1.0e-6,
                "max_subspace_size": 64,
                "preconditioner_floor": 0.25,
            },
        },
        "numerical_gates": {
            "electron_count_abs_per_cell": 1.0e-4,
            "orthonormality_max": 1.0e-4,
        },
        "validation": {
            "volume_factors": [0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06],
            "cutoff_candidates_hartree": [30.0, 40.0, 50.0],
            "kpoint_mesh": [6, 6, 6],
            "kpoint_spot_check_mesh": [8, 8, 8],
            "memory_limit_bytes": 40_000_000_000,
            "point_timeout_seconds": 180.0,
        },
    }


def prepare_carbon_workload(
    *,
    gth_source: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    """Extract carbon GTH data and prepare a portable validation workload."""

    source = Path(gth_source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"GTH source does not exist: {source}")
    entry = parse_gth_entry(source, element=GTH_ELEMENT, name=GTH_NAME)
    if entry.valence_charge != 4.0:
        raise ValueError("C GTH-PBE-q4 must represent four valence electrons")
    resource = ("\n".join(entry.source_lines) + "\n").encode()
    output = Path(out)
    resource_path = output / "resources" / "C-GTH-PBE-q4.gth"
    _write_exact(resource_path, resource)
    unsigned = _unsigned_workload(
        resource_sha256=sha256_bytes(resource),
        source_sha256=sha256_bytes(source.read_bytes()),
    )
    manifest = {
        **unsigned,
        "workload_fingerprint": sha256_bytes(canonical_json_bytes(unsigned)),
    }
    manifest_path = output / "manifest.json"
    _write_exact(manifest_path, canonical_json_bytes(manifest))
    return {
        "status": "prepared",
        "target_id": TARGET_ID,
        "manifest": str(manifest_path),
        "gth_path": str(resource_path),
        "gth_sha256": sha256_bytes(resource),
        "workload_fingerprint": manifest["workload_fingerprint"],
    }


def load_carbon_workload(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load and strictly validate a prepared carbon workload."""

    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != WORKLOAD_SCHEMA or payload.get("target_id") != TARGET_ID:
        raise ValueError("unsupported carbon workload schema or target")
    expected = payload.get("workload_fingerprint")
    unsigned = {key: value for key, value in payload.items() if key != "workload_fingerprint"}
    if expected != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError("carbon workload fingerprint mismatch")
    relative = payload.get("resource", {}).get("path")
    if relative != "resources/C-GTH-PBE-q4.gth":
        raise ValueError("carbon workload resource path mismatch")
    resource = (manifest_path.parent / relative).resolve()
    if not resource.is_relative_to(manifest_path.parent) or not resource.is_file():
        raise ValueError("carbon workload resource is missing or unconfined")
    if (
        resource.is_symlink()
        or sha256_bytes(resource.read_bytes()) != payload["resource"]["sha256"]
    ):
        raise ValueError("carbon workload resource hash mismatch")
    if payload.get("system", {}).get("fractional_positions") != [
        list(row) for row in CARBON_FRACTIONAL_POSITIONS
    ]:
        raise ValueError("carbon workload structure mismatch")
    return payload, resource


def _print_payload(payload: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(" ".join(f"{key}={value}" for key, value in payload.items()))


def main(argv: list[str] | None = None) -> None:
    """Run the diamond-carbon EOS preparation and validation CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--gth-source", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--json", action="store_true")

    point = subparsers.add_parser("eos-point")
    point.add_argument("--manifest", type=Path, required=True)
    point.add_argument("--profile", required=True)
    point.add_argument("--volume-index", type=int, required=True)
    point.add_argument("--out", type=Path, required=True)
    point.add_argument("--initial-density", type=Path)
    point.add_argument("--json", action="store_true")

    validate = subparsers.add_parser("validate-eos")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--out", type=Path, required=True)
    validate.add_argument("--dry-run", action="store_true")
    validate.add_argument("--summarize-only", action="store_true")
    validate.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare_carbon_workload(gth_source=args.gth_source, out=args.out)
    elif args.command == "eos-point":
        from mlx_atomistic.benchmarks.dft_carbon_eos_runner import run_carbon_eos_point

        payload = run_carbon_eos_point(
            manifest_path=args.manifest,
            profile=args.profile,
            volume_index=args.volume_index,
            out=args.out,
            initial_density_path=args.initial_density,
        )
    else:
        from mlx_atomistic.benchmarks.dft_carbon_eos_runner import (
            run_carbon_eos_validation,
        )

        payload = run_carbon_eos_validation(
            manifest_path=args.manifest,
            out=args.out,
            dry_run=args.dry_run,
            summarize_only=args.summarize_only,
        )
    _print_payload(payload, as_json=args.json)


if __name__ == "__main__":
    main()
