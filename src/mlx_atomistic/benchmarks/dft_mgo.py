"""Prepare and run bounded rock-salt MgO equation-of-state validation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mlx_atomistic._artifact_identity import canonical_json_bytes, sha256_bytes
from mlx_atomistic.benchmarks.dft_silicon import parse_gth_entry

WORKLOAD_SCHEMA = "mlx-atomistic.dft-mgo-workload.v1"
TARGET_ID = "rocksalt-mgo-conventional-pbe-gth"

MGO_SYMBOLS = ("Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O")
MGO_FRACTIONAL_POSITIONS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.0),
    (0.5, 0.5, 0.5),
    (0.0, 0.0, 0.5),
    (0.0, 0.5, 0.0),
)
RESOURCE_SPECS = {
    "mg_q2": {
        "element": "Mg",
        "name": "GTH-PBE-q2",
        "valence_charge": 2,
        "path": "resources/Mg-GTH-PBE-q2.gth",
    },
    "mg_q10": {
        "element": "Mg",
        "name": "GTH-PBE-q10",
        "valence_charge": 10,
        "path": "resources/Mg-GTH-PBE-q10.gth",
    },
    "o_q6": {
        "element": "O",
        "name": "GTH-PBE-q6",
        "valence_charge": 6,
        "path": "resources/O-GTH-PBE-q6.gth",
    },
}


def _write_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"refusing to replace mismatched existing file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _unsigned_workload(
    *,
    resources: Mapping[str, Mapping[str, Any]],
    source_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": WORKLOAD_SCHEMA,
        "target_id": TARGET_ID,
        "resources": dict(resources),
        "source_database_sha256": source_sha256,
        "system": {
            "name": "rocksalt-mgo-conventional-cubic",
            "atom_count": 8,
            "symbols": list(MGO_SYMBOLS),
            "fractional_positions": [list(row) for row in MGO_FRACTIONAL_POSITIONS],
            "q2_electron_count": 32,
            "q2_occupied_band_count": 16,
            "q10_electron_count": 64,
            "q10_occupied_band_count": 32,
            "spin_mode": "unpolarized",
            "occupancy_per_band": 2,
        },
        "physics": {
            "exchange_correlation": "PBE-PW92",
            "accepted_pseudopotentials": [
                "Mg GTH-PBE-q2",
                "O GTH-PBE-q6",
            ],
            "high_accuracy_feasibility_pseudopotentials": [
                "Mg GTH-PBE-q10",
                "O GTH-PBE-q6",
            ],
            "kpoint_centering": "monkhorst-pack-even-half-shift",
            "occupation": "zero-temperature fixed occupations",
        },
        "solver": {
            "scf": {
                "max_iterations": 80,
                "min_iterations": 2,
                "density_tolerance": 1.0e-6,
                "energy_tolerance_hartree": 8.0e-6,
                "orbital_tolerance": 2.0e-6,
                "mixing_beta": 0.3,
                "mixer": "diis",
                "adaptive_eigensolver_tolerance": True,
                "initial_eigensolver_tolerance": 1.0e-2,
                "eigensolver_tolerance_scale": 0.1,
            },
            "davidson": {
                "max_iterations": 48,
                "tolerance": 2.0e-6,
                "max_subspace_size": 96,
                "preconditioner_floor": 0.25,
                "tolerance_rationale": (
                    "Bounded MgO A/B evidence found a stable 1.8e-6 "
                    "complex64 residual floor with less than 0.001 meV/atom "
                    "energy change when the iteration cap increased."
                ),
            },
        },
        "numerical_gates": {
            "electron_count_abs_per_cell": 1.0e-4,
            "orthonormality_max": 1.0e-4,
        },
        "validation": {
            "volume_factors": [0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06],
            "cutoff_candidates_hartree": [
                25.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
            ],
            "kpoint_screen_meshes": [[4, 4, 4], [6, 6, 6]],
            "accepted_kpoint_mesh": [6, 6, 6],
            "memory_limit_bytes": 40_000_000_000,
            "point_timeout_seconds": 1800.0,
        },
    }


def prepare_mgo_workload(
    *,
    gth_source: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    """Extract Mg/O GTH data and prepare a portable MgO validation workload."""

    source = Path(gth_source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"GTH source does not exist: {source}")
    output = Path(out)
    resources: dict[str, dict[str, Any]] = {}
    for resource_id, spec in RESOURCE_SPECS.items():
        entry = parse_gth_entry(
            source,
            element=str(spec["element"]),
            name=str(spec["name"]),
        )
        expected_charge = float(spec["valence_charge"])
        if entry.valence_charge != expected_charge:
            raise ValueError(
                f"{spec['element']} {spec['name']} must represent "
                f"{expected_charge:g} valence electrons"
            )
        payload = ("\n".join(entry.source_lines) + "\n").encode()
        resource_path = output / str(spec["path"])
        _write_exact(resource_path, payload)
        resources[resource_id] = {
            **spec,
            "sha256": sha256_bytes(payload),
        }
    unsigned = _unsigned_workload(
        resources=resources,
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
        "resource_sha256": {
            key: value["sha256"] for key, value in resources.items()
        },
        "workload_fingerprint": manifest["workload_fingerprint"],
    }


def load_mgo_workload(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Load and strictly validate a prepared rock-salt MgO workload."""

    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != WORKLOAD_SCHEMA or payload.get(
        "target_id"
    ) != TARGET_ID:
        raise ValueError("unsupported MgO workload schema or target")
    expected = payload.get("workload_fingerprint")
    unsigned = {
        key: value for key, value in payload.items() if key != "workload_fingerprint"
    }
    if expected != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError("MgO workload fingerprint mismatch")
    if payload.get("system", {}).get("symbols") != list(MGO_SYMBOLS) or payload.get(
        "system", {}
    ).get("fractional_positions") != [
        list(row) for row in MGO_FRACTIONAL_POSITIONS
    ]:
        raise ValueError("MgO workload structure mismatch")
    resources = payload.get("resources")
    if not isinstance(resources, dict) or set(resources) != set(RESOURCE_SPECS):
        raise ValueError("MgO workload resources are incomplete")
    resolved: dict[str, Path] = {}
    for resource_id, expected_spec in RESOURCE_SPECS.items():
        resource = resources.get(resource_id)
        if not isinstance(resource, dict) or any(
            resource.get(key) != value for key, value in expected_spec.items()
        ):
            raise ValueError(f"MgO workload resource identity mismatch: {resource_id}")
        resource_path = (manifest_path.parent / str(resource["path"])).resolve()
        if (
            not resource_path.is_relative_to(manifest_path.parent)
            or not resource_path.is_file()
            or resource_path.is_symlink()
            or sha256_bytes(resource_path.read_bytes()) != resource.get("sha256")
        ):
            raise ValueError(f"MgO workload resource hash mismatch: {resource_id}")
        resolved[resource_id] = resource_path
    return payload, resolved


def _print_payload(payload: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(" ".join(f"{key}={value}" for key, value in payload.items()))


def main(argv: list[str] | None = None) -> None:
    """Run the rock-salt MgO EOS preparation and validation CLI."""

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
        result = prepare_mgo_workload(gth_source=args.gth_source, out=args.out)
    elif args.command == "eos-point":
        from mlx_atomistic.benchmarks.dft_mgo_eos_runner import run_mgo_eos_point

        result = run_mgo_eos_point(
            manifest_path=args.manifest,
            profile=args.profile,
            volume_index=args.volume_index,
            out=args.out,
            initial_density_path=args.initial_density,
        )
    else:
        from mlx_atomistic.benchmarks.dft_mgo_eos_runner import (
            run_mgo_eos_validation,
        )

        result = run_mgo_eos_validation(
            manifest_path=args.manifest,
            out=args.out,
            dry_run=args.dry_run,
            summarize_only=args.summarize_only,
        )
    _print_payload(result, as_json=args.json)


if __name__ == "__main__":
    main()
