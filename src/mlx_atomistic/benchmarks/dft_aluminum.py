"""Prepare and run the bounded fcc Aluminum metallic validation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mlx_atomistic._artifact_identity import canonical_json_bytes, sha256_bytes
from mlx_atomistic.benchmarks.dft_aluminum_eos import load_aluminum_eos_references
from mlx_atomistic.benchmarks.dft_silicon import parse_gth_entry
from mlx_atomistic.dft import (
    GammaCenteredGrid,
    KPointMesh,
    cubic_reciprocal_symmetry_operations,
    reduce_kpoint_mesh_by_symmetry,
)

WORKLOAD_SCHEMA = "mlx-atomistic.dft-aluminum-workload.v2"
TARGET_ID = "fcc-aluminum-conventional-pbe-gth-q3"
GTH_ELEMENT = "Al"
GTH_NAME = "GTH-PBE-q3"
KPOINT_MESH_SIZES = (7, 11, 15, 19, 23, 27)

ALUMINUM_FRACTIONAL_POSITIONS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
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


def _reduced_mesh_payload(size: int) -> dict[str, Any]:
    full = GammaCenteredGrid((size, size, size))
    reduced = reduce_kpoint_mesh_by_symmetry(
        full,
        cubic_reciprocal_symmetry_operations(),
    )
    mesh_payload = reduced.to_dict()
    return {
        "size": [size, size, size],
        "centering": "gamma",
        "full_point_count": len(full.points),
        "representative_point_count": len(reduced.points),
        "points": mesh_payload["points"],
        "point_group_symmetry": mesh_payload["point_group_symmetry"],
    }


def _unsigned_workload(
    *,
    resource_sha256: str,
    source_sha256: str,
    reduced_meshes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    references = load_aluminum_eos_references()
    return {
        "schema_version": WORKLOAD_SCHEMA,
        "target_id": TARGET_ID,
        "resource": {
            "path": "resources/Al-GTH-PBE-q3.gth",
            "sha256": resource_sha256,
            "source_database_sha256": source_sha256,
            "element": GTH_ELEMENT,
            "name": GTH_NAME,
            "functional": "PBE",
            "valence_charge": 3,
        },
        "system": {
            "name": "fcc-aluminum-conventional-cubic",
            "atom_count": 4,
            "symbols": [GTH_ELEMENT] * 4,
            "fractional_positions": [list(row) for row in ALUMINUM_FRACTIONAL_POSITIONS],
            "electron_count": 12,
            "spin_mode": "unpolarized",
            "occupancy_per_band": 2,
        },
        "physics": {
            "exchange_correlation": "PBE-PW92",
            "pseudopotential": "Al GTH-PBE-q3",
            "occupation": "Fermi-Dirac",
            "smearing_width_hartree": 0.00225,
            "energy_observable": "Helmholtz free energy F=E-TS",
            "kpoint_centering": "gamma",
            "kpoint_symmetry": "caller-verified full cubic point group",
        },
        "solver": {
            "scf": {
                "max_iterations": 100,
                "min_iterations": 2,
                "density_tolerance": 1.0e-6,
                "energy_tolerance_hartree": 8.0e-6,
                "orbital_tolerance": 1.0e-6,
                "mixing_beta": 0.2,
                "mixer": "diis",
                "adaptive_eigensolver_tolerance": True,
                "initial_eigensolver_tolerance": 1.0e-2,
                "eigensolver_tolerance_scale": 0.1,
            },
            "davidson": {
                "max_iterations": 56,
                "tolerance": 1.0e-6,
                "max_subspace_size": 64,
                "preconditioner_floor": 0.25,
            },
        },
        "numerical_gates": {
            "electron_count_abs_per_cell": 1.0e-4,
            "orbital_residual_max": 1.0e-6,
            "orthonormality_max": 1.0e-4,
            "highest_band_occupation_max": 1.0e-6,
            "free_energy_identity_abs_hartree": 5.0e-6,
        },
        "validation": {
            "central_conventional_lattice_angstrom": references["protocol"][
                "central_conventional_lattice_angstrom"
            ],
            "volume_factors": references["protocol"]["volume_factors"],
            "band_capacity_candidates": [8, 9, 10, 11, 12, 16, 20, 26],
            "cutoff_candidates_hartree": [15.0, 20.0, 25.0, 30.0],
            "fft_shape_by_cutoff": {"15": 36, "20": 40, "25": 44, "30": 48},
            "kpoint_mesh_sizes": list(KPOINT_MESH_SIZES),
            "maximum_kpoint_spacing_inverse_angstrom": 0.06,
            "screen_volume_indices": [2, 3, 4],
            "full_volume_indices": list(range(7)),
            "full_grid_oracle_size": 4,
            "memory_limit_bytes": 40_000_000_000,
            "point_timeout_seconds": 900.0,
        },
        "reduced_kpoint_meshes": dict(reduced_meshes),
    }


def prepare_aluminum_workload(
    *,
    gth_source: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    """Extract Aluminum GTH data and prepare a portable metallic workload."""

    source = Path(gth_source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"GTH source does not exist: {source}")
    entry = parse_gth_entry(source, element=GTH_ELEMENT, name=GTH_NAME)
    if entry.valence_charge != 3.0:
        raise ValueError("Al GTH-PBE-q3 must represent three valence electrons")
    resource = ("\n".join(entry.source_lines) + "\n").encode()
    references = load_aluminum_eos_references()
    expected_resource_sha256 = references["local_pseudopotential"]["extracted_entry_sha256"]
    if sha256_bytes(resource) != expected_resource_sha256:
        raise ValueError("Al GTH-PBE-q3 source entry does not match the locked reference")

    output = Path(out)
    resource_path = output / "resources" / "Al-GTH-PBE-q3.gth"
    _write_exact(resource_path, resource)
    reduced_meshes = {str(size): _reduced_mesh_payload(size) for size in KPOINT_MESH_SIZES}
    unsigned = _unsigned_workload(
        resource_sha256=sha256_bytes(resource),
        source_sha256=sha256_bytes(source.read_bytes()),
        reduced_meshes=reduced_meshes,
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
        "kpoint_representatives": {
            size: reduced_meshes[str(size)]["representative_point_count"]
            for size in KPOINT_MESH_SIZES
        },
    }


def _validate_reduced_meshes(payload: Mapping[str, Any]) -> None:
    meshes = payload.get("reduced_kpoint_meshes", {})
    if set(meshes) != {str(size) for size in KPOINT_MESH_SIZES}:
        raise ValueError("Aluminum workload k-point ladder mismatch")
    for size in KPOINT_MESH_SIZES:
        mesh = meshes[str(size)]
        if mesh.get("size") != [size, size, size]:
            raise ValueError("Aluminum workload k-point mesh size mismatch")
        if mesh.get("full_point_count") != size**3:
            raise ValueError("Aluminum workload full k-point count mismatch")
        points = mesh.get("points", [])
        if mesh.get("representative_point_count") != len(points):
            raise ValueError("Aluminum workload representative count mismatch")
        try:
            restored = KPointMesh.from_dict(
                {
                    "points": points,
                    "point_group_symmetry": mesh.get("point_group_symmetry"),
                }
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Aluminum workload k-point symmetry is invalid") from error
        if len(restored.points) != len(points):
            raise ValueError("Aluminum workload representative count mismatch")


def load_aluminum_workload(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load and strictly validate a prepared Aluminum workload."""

    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != WORKLOAD_SCHEMA or payload.get("target_id") != TARGET_ID:
        raise ValueError("unsupported Aluminum workload schema or target")
    expected = payload.get("workload_fingerprint")
    unsigned = {key: value for key, value in payload.items() if key != "workload_fingerprint"}
    if expected != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError("Aluminum workload fingerprint mismatch")
    relative = payload.get("resource", {}).get("path")
    if relative != "resources/Al-GTH-PBE-q3.gth":
        raise ValueError("Aluminum workload resource path mismatch")
    resource = (manifest_path.parent / relative).resolve()
    if not resource.is_relative_to(manifest_path.parent) or not resource.is_file():
        raise ValueError("Aluminum workload resource is missing or unconfined")
    if (
        resource.is_symlink()
        or sha256_bytes(resource.read_bytes()) != payload["resource"]["sha256"]
    ):
        raise ValueError("Aluminum workload resource hash mismatch")
    if payload.get("system", {}).get("fractional_positions") != [
        list(row) for row in ALUMINUM_FRACTIONAL_POSITIONS
    ]:
        raise ValueError("Aluminum workload structure mismatch")
    _validate_reduced_meshes(payload)
    return payload, resource


def kpoint_mesh_from_workload(workload: Mapping[str, Any], size: int) -> KPointMesh:
    """Rebuild one fingerprinted reduced k-point mesh from a workload."""

    try:
        payload = workload["reduced_kpoint_meshes"][str(int(size))]
    except KeyError as error:
        raise ValueError(f"Aluminum workload does not contain k-point mesh {size}") from error
    return KPointMesh.from_dict(
        {
            "points": payload["points"],
            "point_group_symmetry": payload["point_group_symmetry"],
        }
    )


def _print_payload(payload: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(" ".join(f"{key}={value}" for key, value in payload.items()))


def main(argv: list[str] | None = None) -> None:
    """Run the Aluminum workload preparation and validation CLI."""

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
        payload = prepare_aluminum_workload(gth_source=args.gth_source, out=args.out)
    elif args.command == "eos-point":
        from mlx_atomistic.benchmarks.dft_aluminum_eos_runner import run_aluminum_eos_point

        payload = run_aluminum_eos_point(
            manifest_path=args.manifest,
            profile=args.profile,
            volume_index=args.volume_index,
            out=args.out,
            initial_density_path=args.initial_density,
        )
    else:
        from mlx_atomistic.benchmarks.dft_aluminum_eos_runner import (
            run_aluminum_eos_validation,
        )

        payload = run_aluminum_eos_validation(
            manifest_path=args.manifest,
            out=args.out,
            dry_run=args.dry_run,
            summarize_only=args.summarize_only,
        )
    _print_payload(payload, as_json=args.json)


if __name__ == "__main__":
    main()
