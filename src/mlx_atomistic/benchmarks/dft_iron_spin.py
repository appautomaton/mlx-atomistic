"""Source-bound collinear-spin validation for body-centered-cubic Iron."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_atomistic._artifact_identity import canonical_json_bytes, sha256_bytes
from mlx_atomistic.benchmarks.dft_runtime_contract import (
    build_source_fingerprints,
    collect_host_provenance,
)
from mlx_atomistic.benchmarks.dft_silicon import parse_gth_entry
from mlx_atomistic.dft import (
    GammaCenteredGrid,
    KPoint,
    KPointMesh,
    PeriodicCollinearSpinConfig,
    PeriodicDavidsonConfig,
    PeriodicDFTSystem,
    PeriodicFermiDiracSmearing,
    PeriodicSCFConfig,
    RuntimeObserver,
    cubic_reciprocal_symmetry_operations,
    read_gth,
    reciprocal_symmetry_operations_for_cell,
    reduce_kpoint_mesh_by_symmetry,
    run_periodic_scf,
)

WORKLOAD_SCHEMA = "mlx-atomistic.dft-iron-spin-workload.v2"
POINT_SCHEMA = "mlx-atomistic.dft-iron-spin-point.v1"
REPORT_SCHEMA = "mlx-atomistic.dft-iron-spin-validation.v1"
TARGET_ID = "bcc-iron-primitive-pbe-gth-q8-collinear"
GTH_ELEMENT = "Fe"
GTH_NAME = "GTH-PBE-q8"
PSEUDOPOTENTIAL_SPECS = {
    "GTH-PBE-q8": {
        "resource_sha256": "0b2f0ef0da75173979344192e3029d43530e9622ae4f2b6b459b6b241727f7e3",
        "valence_charge": 8,
        "d_projector_radius_bohr": 0.30621323095055,
        "initial_magnetization": 2.0,
    },
    "GTH-PBE-q16": {
        "resource_sha256": "b01c203a4837b7becf1b2f76b188bf6365ec5aec216dccf1ed4fda6b9d65fdcb",
        "valence_charge": 16,
        "d_projector_radius_bohr": 0.22321053924521,
        "initial_magnetization": 2.2,
    },
}
GTH_RESOURCE_SHA256 = str(PSEUDOPOTENTIAL_SPECS[GTH_NAME]["resource_sha256"])
LATTICE_BOHR = 5.42
ATOM_COUNT = 1
ELECTRON_COUNT = 8.0
PRIMITIVE_CELL_MATRIX = 0.5 * LATTICE_BOHR * np.asarray(
    ((-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0)),
    dtype=np.float64,
)
CONVENTIONAL_SUPERCELL_MATRIX = np.asarray(
    ((0, 1, 1), (1, 0, 1), (1, 1, 0)),
    dtype=np.int64,
)
FRACTIONAL_POSITIONS = ((0.0, 0.0, 0.0),)
REFERENCE_MOMENT_PER_ATOM = 2.33
REFERENCE_MOMENT_TOLERANCE = 0.4

PROFILE_SPECS: dict[str, dict[str, int | float]] = {
    "smoke": {
        "cutoff_hartree": 6.0,
        "grid_size": 12,
        "kpoint_size": 2,
        "n_bands": 6,
        "max_iterations": 70,
        "density_tolerance": 1.0e-3,
        "energy_tolerance": 2.0e-6,
        "orbital_tolerance": 2.0e-5,
        "davidson_tolerance": 2.0e-5,
        "magnetization_mixing_beta": 0.35,
    },
    "selected": {
        "cutoff_hartree": 75.0,
        "grid_size": 28,
        "kpoint_size": 4,
        "n_bands": 6,
        "max_iterations": 90,
        "density_tolerance": 1.0e-3,
        "energy_tolerance": 2.0e-6,
        "orbital_tolerance": 2.0e-5,
        "davidson_tolerance": 2.0e-5,
        "magnetization_mixing_beta": 0.35,
    },
    "cutoff-check": {
        "cutoff_hartree": 100.0,
        "grid_size": 32,
        "kpoint_size": 4,
        "n_bands": 6,
        "max_iterations": 90,
        "density_tolerance": 1.0e-3,
        "energy_tolerance": 2.0e-6,
        "orbital_tolerance": 2.0e-5,
        "davidson_tolerance": 2.0e-5,
        "magnetization_mixing_beta": 0.35,
    },
    "kpoint-check": {
        "cutoff_hartree": 75.0,
        "grid_size": 28,
        "kpoint_size": 6,
        "n_bands": 6,
        "max_iterations": 90,
        "density_tolerance": 1.0e-3,
        "energy_tolerance": 2.0e-6,
        "orbital_tolerance": 2.0e-5,
        "davidson_tolerance": 2.0e-5,
        "magnetization_mixing_beta": 0.35,
    },
}

Q16_PROFILE_SPECS: dict[str, dict[str, int | float]] = {
    "smoke": {
        "cutoff_hartree": 10.0,
        "grid_size": 16,
        "kpoint_size": 2,
        "n_bands": 10,
        "max_iterations": 70,
        "density_tolerance": 1.0e-3,
        "energy_tolerance": 2.0e-6,
        "orbital_tolerance": 2.0e-5,
        "davidson_tolerance": 2.0e-5,
        "magnetization_mixing_beta": 0.35,
    },
    "selected": {
        "cutoff_hartree": 150.0,
        "grid_size": 36,
        "kpoint_size": 4,
        "n_bands": 10,
        "max_iterations": 90,
        "density_tolerance": 1.0e-3,
        "energy_tolerance": 2.0e-6,
        "orbital_tolerance": 2.0e-5,
        "davidson_tolerance": 2.0e-5,
        "magnetization_mixing_beta": 0.35,
    },
    "cutoff-check": {
        "cutoff_hartree": 200.0,
        "grid_size": 40,
        "kpoint_size": 4,
        "n_bands": 10,
        "max_iterations": 90,
        "density_tolerance": 1.0e-3,
        "energy_tolerance": 2.0e-6,
        "orbital_tolerance": 2.0e-5,
        "davidson_tolerance": 2.0e-5,
        "magnetization_mixing_beta": 0.35,
    },
    "kpoint-check": {
        "cutoff_hartree": 150.0,
        "grid_size": 36,
        "kpoint_size": 6,
        "n_bands": 10,
        "max_iterations": 90,
        "density_tolerance": 1.0e-3,
        "energy_tolerance": 2.0e-6,
        "orbital_tolerance": 2.0e-5,
        "davidson_tolerance": 2.0e-5,
        "magnetization_mixing_beta": 0.35,
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


def _d_projector_tail_ratio(cutoff_hartree: float, radius: float) -> float:
    """Return the Fe q8 first d-projector radial amplitude relative to its peak."""

    q = np.sqrt(2.0 * cutoff_hartree)
    amplitude = q * q * np.exp(-0.5 * (q * radius) ** 2)
    peak = (2.0 / (radius * radius)) * np.exp(-1.0)
    return float(amplitude / peak)


def _primitive_reciprocal_operations() -> tuple[tuple[tuple[int, ...], ...], ...]:
    return reciprocal_symmetry_operations_for_cell(
        PRIMITIVE_CELL_MATRIX,
        cubic_reciprocal_symmetry_operations(),
    )


def _primitive_unfolded_mesh(size: int) -> KPointMesh:
    conventional = GammaCenteredGrid((size, size, size))
    transform = np.linalg.inv(CONVENTIONAL_SUPERCELL_MATRIX).T
    cosets = (np.zeros(3, dtype=np.float64), np.asarray((1.0, 0.0, 0.0)))
    points = []
    for point in conventional.points:
        for coset in cosets:
            reduced = (np.asarray(point.vector) + coset) @ transform
            reduced -= np.floor(reduced + 0.5)
            points.append(
                KPoint(
                    reduced,
                    weight=0.5 * point.weight,
                    coordinate_system="reduced",
                )
            )
    return KPointMesh(tuple(points))


def _mesh_payload(size: int) -> dict[str, object]:
    full = _primitive_unfolded_mesh(size)
    reduced = reduce_kpoint_mesh_by_symmetry(
        full,
        _primitive_reciprocal_operations(),
    )
    mesh_payload = reduced.to_dict()
    return {
        "size": [size, size, size],
        "sampling_identity": "exact index-2 conventional-cell unfolding",
        "full_point_count": len(full.points),
        "points": mesh_payload["points"],
        "point_group_symmetry": mesh_payload["point_group_symmetry"],
    }


def prepare_iron_spin_workload(
    *,
    gth_source: str | Path,
    out: str | Path,
    gth_name: str = GTH_NAME,
) -> dict[str, object]:
    """Extract one Fe GTH variant and prepare the fingerprinted validation input."""

    source = Path(gth_source).expanduser().resolve()
    if gth_name not in PSEUDOPOTENTIAL_SPECS:
        raise ValueError(f"unsupported Fe GTH variant: {gth_name}")
    pseudo_spec = PSEUDOPOTENTIAL_SPECS[gth_name]
    profiles = Q16_PROFILE_SPECS if gth_name == "GTH-PBE-q16" else PROFILE_SPECS
    entry = parse_gth_entry(source, element=GTH_ELEMENT, name=gth_name)
    resource = ("\n".join(entry.source_lines) + "\n").encode()
    if (
        entry.valence_charge != float(pseudo_spec["valence_charge"])
        or sha256_bytes(resource) != pseudo_spec["resource_sha256"]
    ):
        raise ValueError(f"Fe {gth_name} does not match the locked UZH resource")
    mesh_sizes = sorted({int(spec["kpoint_size"]) for spec in profiles.values()})
    target_id = f"bcc-iron-primitive-pbe-{gth_name.lower()}-collinear"
    resource_name = f"Fe-{gth_name}.gth"
    electron_count = int(pseudo_spec["valence_charge"])
    initial_magnetization = float(pseudo_spec["initial_magnetization"])
    d_radius = float(pseudo_spec["d_projector_radius_bohr"])
    unsigned = {
        "schema_version": WORKLOAD_SCHEMA,
        "target_id": target_id,
        "resource": {
            "path": f"resources/{resource_name}",
            "sha256": pseudo_spec["resource_sha256"],
            "source_database_sha256": sha256_bytes(source.read_bytes()),
            "source_library": "CP2K POTENTIAL_UZH",
            "element": GTH_ELEMENT,
            "name": gth_name,
            "valence_charge": electron_count,
        },
        "system": {
            "cell": "bcc one-atom primitive",
            "conventional_lattice_bohr": LATTICE_BOHR,
            "cell_matrix_bohr": PRIMITIVE_CELL_MATRIX.tolist(),
            "atom_count": ATOM_COUNT,
            "fractional_positions": [list(row) for row in FRACTIONAL_POSITIONS],
            "electron_count": electron_count,
            "kpoint_sampling": "exact unfolding from the conventional cubic mesh",
        },
        "physics": {
            "exchange_correlation": "spin-PBE-PW92",
            "occupation": "Fermi-Dirac",
            "smearing_width_hartree": 0.01,
            "spin_mode": "unconstrained-collinear",
            "initial_magnetization_per_cell": initial_magnetization,
            "energy_observable": "Helmholtz free energy F=E-TS",
        },
        "profiles": profiles,
        "representation": {
            "hardest_channel": (
                f"Fe {gth_name} d projector, l=2, radius={d_radius:.14f} bohr"
            ),
            "radial_tail_ratio_at_cutoff": {
                name: _d_projector_tail_ratio(float(spec["cutoff_hartree"]), d_radius)
                for name, spec in profiles.items()
            },
            "selection_rule": (
                "selected reaches the percent-scale radial tail; "
                "cutoff-check checks it"
            ),
        },
        "kpoint_meshes": {str(size): _mesh_payload(size) for size in mesh_sizes},
        "reference": {
            "kind": "published-pbe-context",
            "lattice_angstrom": 2.87,
            "moment_bohr_magneton_per_atom": REFERENCE_MOMENT_PER_ATOM,
            "moment_abs_tolerance": REFERENCE_MOMENT_TOLERANCE,
            "source": "https://doi.org/10.1107/S2053273321008792",
            "energy_gate": "spin-polarized free energy below matched unpolarized free energy",
        },
        "validation": {
            "moment_convergence_abs_per_atom": 0.12,
            "free_energy_convergence_hartree_per_atom": 0.01,
            "symmetry_moment_abs_per_atom": 0.02,
            "symmetry_free_energy_abs_hartree_per_atom": 5.0e-5,
            "electron_count_abs_per_cell": 1.0e-4,
            "required_points": [
                "smoke-spin",
                "selected-spin",
                "selected-spin-full",
                "selected-unpolarized",
                "cutoff-check-spin",
                "kpoint-check-spin",
            ],
        },
    }
    manifest = {
        **unsigned,
        "workload_fingerprint": sha256_bytes(canonical_json_bytes(unsigned)),
    }
    output = Path(out)
    resource_path = output / "resources" / resource_name
    manifest_path = output / "manifest.json"
    _write_exact(resource_path, resource)
    _write_exact(manifest_path, canonical_json_bytes(manifest))
    return {
        "status": "prepared",
        "target_id": target_id,
        "manifest": str(manifest_path),
        "resource": str(resource_path),
        "workload_fingerprint": manifest["workload_fingerprint"],
    }


def load_iron_spin_workload(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load and validate a prepared Iron spin workload."""

    manifest_path = Path(path).resolve()
    workload = json.loads(manifest_path.read_text())
    unsigned = {key: value for key, value in workload.items() if key != "workload_fingerprint"}
    resource_metadata = workload.get("resource", {})
    gth_name = resource_metadata.get("name")
    pseudo_spec = PSEUDOPOTENTIAL_SPECS.get(gth_name)
    expected_target = f"bcc-iron-primitive-pbe-{str(gth_name).lower()}-collinear"
    if (
        workload.get("schema_version") != WORKLOAD_SCHEMA
        or pseudo_spec is None
        or workload.get("target_id") != expected_target
        or workload.get("workload_fingerprint")
        != sha256_bytes(canonical_json_bytes(unsigned))
        or resource_metadata.get("sha256") != pseudo_spec["resource_sha256"]
        or workload.get("system", {}).get("electron_count")
        != pseudo_spec["valence_charge"]
    ):
        raise ValueError("Iron spin workload identity is invalid")
    resource = (manifest_path.parent / workload["resource"]["path"]).resolve()
    if (
        not resource.is_relative_to(manifest_path.parent)
        or resource.is_symlink()
        or not resource.is_file()
        or sha256_bytes(resource.read_bytes()) != workload["resource"]["sha256"]
    ):
        raise ValueError("Iron spin workload resource is missing or mismatched")
    meshes = workload.get("kpoint_meshes")
    if not isinstance(meshes, Mapping) or not meshes:
        raise ValueError("Iron spin workload k-point meshes are missing")
    try:
        for size, payload in meshes.items():
            if not isinstance(payload, Mapping) or int(size) <= 0:
                raise ValueError
            restored = KPointMesh.from_dict(
                {
                    "points": payload["points"],
                    "point_group_symmetry": payload["point_group_symmetry"],
                }
            )
            symmetry = restored.to_dict()["point_group_symmetry"]
            if symmetry["full_point_count"] != payload["full_point_count"]:
                raise ValueError
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Iron spin workload k-point symmetry is invalid") from error
    return workload, resource


def _kpoint_mesh(
    workload: Mapping[str, Any],
    size: int,
    *,
    mode: str = "reduced",
) -> KPointMesh:
    if mode == "full":
        return _primitive_unfolded_mesh(size)
    if mode != "reduced":
        raise ValueError(f"unsupported Iron k-point mode: {mode}")
    payload = workload["kpoint_meshes"][str(size)]
    return KPointMesh.from_dict(
        {
            "points": payload["points"],
            "point_group_symmetry": payload["point_group_symmetry"],
        }
    )


def _scf_config(
    spec: Mapping[str, int | float],
    *,
    polarized: bool,
    initial_magnetization: float,
) -> PeriodicSCFConfig:
    return PeriodicSCFConfig(
        max_iterations=int(spec["max_iterations"]),
        min_iterations=2,
        density_tolerance=float(spec["density_tolerance"]),
        energy_tolerance=float(spec["energy_tolerance"]),
        orbital_tolerance=float(spec["orbital_tolerance"]),
        mixing_beta=0.2,
        mixer="diis",
        adaptive_eigensolver_tolerance=True,
        initial_eigensolver_tolerance=1.0e-2,
        eigensolver_tolerance_scale=0.1,
        smearing=PeriodicFermiDiracSmearing(width_hartree=0.01),
        spin=(
            PeriodicCollinearSpinConfig(
                mode="unconstrained",
                magnetization=None,
                initial_magnetization=initial_magnetization,
                magnetization_mixing_beta=float(spec["magnetization_mixing_beta"]),
            )
            if polarized
            else None
        ),
        davidson=PeriodicDavidsonConfig(
            max_iterations=48,
            tolerance=float(spec["davidson_tolerance"]),
            max_subspace_size=48,
            preconditioner_floor=0.25,
        ),
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = canonical_json_bytes(dict(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _material_protocol_record() -> dict[str, object]:
    source = Path(__file__).resolve()
    root = source.parents[3]
    return {
        "path": str(source.relative_to(root)),
        "byte_size": source.stat().st_size,
        "sha256": sha256_bytes(source.read_bytes()),
    }


def run_iron_spin_point(
    *,
    manifest_path: str | Path,
    profile: str,
    polarized: bool,
    kpoint_mode: str = "reduced",
    out: str | Path,
) -> dict[str, object]:
    """Run one exact Iron spin or matched unpolarized validation point."""

    workload, resource = load_iron_spin_workload(manifest_path)
    profiles = workload.get("profiles", {})
    if profile not in profiles:
        raise ValueError(f"unknown Iron spin profile: {profile}")
    spec = profiles[profile]
    grid_size = int(spec["grid_size"])
    gth_name = str(workload["resource"]["name"])
    pseudo = read_gth(resource, element=GTH_ELEMENT, name=gth_name)
    positions = np.asarray(FRACTIONAL_POSITIONS, dtype=np.float64) @ PRIMITIVE_CELL_MATRIX
    system = PeriodicDFTSystem(
        PRIMITIVE_CELL_MATRIX,
        (grid_size,) * 3,
        positions,
        pseudo,
    )
    observer = RuntimeObserver(synchronize=mx.synchronize, detail_events=False)
    mx.synchronize()
    started = perf_counter()
    result = run_periodic_scf(
        system,
        cutoff_hartree=float(spec["cutoff_hartree"]),
        kpoint_mesh=_kpoint_mesh(
            workload,
            int(spec["kpoint_size"]),
            mode=kpoint_mode,
        ),
        n_bands=int(spec["n_bands"]),
        config=_scf_config(
            spec,
            polarized=polarized,
            initial_magnetization=float(
                workload["physics"]["initial_magnetization_per_cell"]
            ),
        ),
        observer=observer,
    )
    mx.synchronize()
    elapsed = perf_counter() - started
    moment = None if result.integrated_magnetization is None else result.integrated_magnetization
    source_fingerprints = build_source_fingerprints()
    source_fingerprints["material_protocol"] = _material_protocol_record()
    payload: dict[str, object] = {
        "schema_version": POINT_SCHEMA,
        "workload_fingerprint": workload["workload_fingerprint"],
        "profile": profile,
        "polarized": polarized,
        "kpoint_mode": kpoint_mode,
        "settings": dict(spec),
        "result": {
            "converged": result.converged,
            "status": result.status,
            "iterations": result.iterations,
            "free_energy_hartree": result.total_energy,
            "internal_energy_hartree": result.internal_energy,
            "electron_count": result.electron_count,
            "integrated_magnetization": moment,
            "moment_per_atom": None if moment is None else moment / ATOM_COUNT,
            "spin_channels": [
                {
                    "label": channel.label,
                    "electron_count": channel.electron_count,
                    "chemical_potential_hartree": channel.chemical_potential,
                    "eigenvalues_hartree": [
                        [float(value) for value in np.asarray(point.eigen.eigenvalues)]
                        for point in channel.kpoints
                    ],
                    "occupations": [list(point.occupations or ()) for point in channel.kpoints],
                }
                for channel in result.spin_channels
            ],
            "density_residual": result.density_residual,
            "energy_delta_hartree": result.energy_delta,
            "elapsed_wall_seconds": elapsed,
            "history": [dict(row) for row in result.history],
            "observation": observer.snapshot(),
        },
        "source_fingerprints": source_fingerprints,
        "host": collect_host_provenance(),
    }
    _write_json(Path(out), payload)
    return payload


def _point_name(
    profile: str,
    polarized: bool,
    kpoint_mode: str = "reduced",
) -> str:
    base = f"{profile}-{'spin' if polarized else 'unpolarized'}"
    return base if kpoint_mode == "reduced" else f"{base}-{kpoint_mode}"


def _load_matching_point(
    path: Path,
    *,
    workload: Mapping[str, Any],
    profile: str,
    polarized: bool,
    kpoint_mode: str,
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    point = json.loads(path.read_text())
    material_protocol = point.get("source_fingerprints", {}).get("material_protocol")
    if (
        point.get("schema_version") != POINT_SCHEMA
        or point.get("workload_fingerprint") != workload["workload_fingerprint"]
        or point.get("profile") != profile
        or point.get("polarized") is not polarized
        or point.get("kpoint_mode") != kpoint_mode
        or point.get("settings") != workload["profiles"][profile]
        or material_protocol != _material_protocol_record()
    ):
        raise ValueError(f"existing Iron spin point does not match its manifest: {path}")
    return point


def run_iron_spin_validation(
    *,
    manifest_path: str | Path,
    out: str | Path,
    dry_run: bool = False,
) -> dict[str, object]:
    """Run the locked five-point Iron validation or return its exact plan."""

    workload, _ = load_iron_spin_workload(manifest_path)
    plan = (
        ("smoke", True, "reduced"),
        ("selected", True, "reduced"),
        ("selected", True, "full"),
        ("selected", False, "reduced"),
        ("cutoff-check", True, "reduced"),
        ("kpoint-check", True, "reduced"),
    )
    if dry_run:
        return {
            "status": "planned",
            "points": [_point_name(*point) for point in plan],
            "workload_fingerprint": workload["workload_fingerprint"],
        }
    root = Path(out)
    points: dict[str, dict[str, object]] = {}
    for profile, polarized, kpoint_mode in plan:
        name = _point_name(profile, polarized, kpoint_mode)
        point_path = root / "points" / f"{name}.json"
        point = _load_matching_point(
            point_path,
            workload=workload,
            profile=profile,
            polarized=polarized,
            kpoint_mode=kpoint_mode,
        )
        if point is None:
            point = run_iron_spin_point(
                manifest_path=manifest_path,
                profile=profile,
                polarized=polarized,
                kpoint_mode=kpoint_mode,
                out=point_path,
            )
        points[name] = point
        if not bool(point["result"]["converged"]):
            break
    required = set(workload["validation"]["required_points"])
    complete = set(points) == required
    selected = points.get("selected-spin", {}).get("result", {})
    selected_full = points.get("selected-spin-full", {}).get("result", {})
    unpolarized = points.get("selected-unpolarized", {}).get("result", {})
    cutoff = points.get("cutoff-check-spin", {}).get("result", {})
    kpoint = points.get("kpoint-check-spin", {}).get("result", {})
    moment = selected.get("moment_per_atom")
    gates = {
        "complete": complete,
        "all_converged": complete
        and all(point["result"]["converged"] for point in points.values()),
        "electron_count": complete
        and all(
            abs(
                float(point["result"]["electron_count"])
                - float(workload["system"]["electron_count"])
            )
            <= float(workload["validation"]["electron_count_abs_per_cell"])
            for point in points.values()
        ),
        "moment_reference": (
            moment is not None
            and abs(float(moment) - REFERENCE_MOMENT_PER_ATOM) <= REFERENCE_MOMENT_TOLERANCE
        ),
        "magnetic_energy_ordering": (
            complete
            and float(selected["free_energy_hartree"])
            < float(unpolarized["free_energy_hartree"])
        ),
        "symmetry_moment": (
            complete
            and abs(
                float(selected["moment_per_atom"])
                - float(selected_full["moment_per_atom"])
            )
            <= float(workload["validation"]["symmetry_moment_abs_per_atom"])
        ),
        "symmetry_free_energy": (
            complete
            and abs(
                float(selected["free_energy_hartree"])
                - float(selected_full["free_energy_hartree"])
            )
            / ATOM_COUNT
            <= float(
                workload["validation"][
                    "symmetry_free_energy_abs_hartree_per_atom"
                ]
            )
        ),
        "cutoff_moment_convergence": (
            complete
            and abs(float(moment) - float(cutoff["moment_per_atom"]))
            <= float(workload["validation"]["moment_convergence_abs_per_atom"])
        ),
        "kpoint_moment_convergence": (
            complete
            and abs(float(moment) - float(kpoint["moment_per_atom"]))
            <= float(workload["validation"]["moment_convergence_abs_per_atom"])
        ),
        "cutoff_free_energy_convergence": (
            complete
            and abs(
                float(selected["free_energy_hartree"])
                - float(cutoff["free_energy_hartree"])
            )
            / ATOM_COUNT
            <= float(workload["validation"]["free_energy_convergence_hartree_per_atom"])
        ),
        "kpoint_free_energy_convergence": (
            complete
            and abs(
                float(selected["free_energy_hartree"])
                - float(kpoint["free_energy_hartree"])
            )
            / ATOM_COUNT
            <= float(workload["validation"]["free_energy_convergence_hartree_per_atom"])
        ),
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "verified" if all(gates.values()) else "failed",
        "verified": all(gates.values()),
        "workload_fingerprint": workload["workload_fingerprint"],
        "gates": gates,
        "symmetry_oracle": {
            "reduced_free_energy_hartree": selected.get("free_energy_hartree"),
            "full_free_energy_hartree": selected_full.get("free_energy_hartree"),
            "free_energy_abs_hartree_per_atom": (
                None
                if not complete
                else abs(
                    float(selected["free_energy_hartree"])
                    - float(selected_full["free_energy_hartree"])
                )
                / ATOM_COUNT
            ),
            "moment_abs_per_atom": (
                None
                if not complete
                else abs(
                    float(selected["moment_per_atom"])
                    - float(selected_full["moment_per_atom"])
                )
            ),
        },
        "points": points,
    }
    _write_json(root / "report.json", report)
    return report


def main() -> None:
    """Run the Iron spin workload command-line interface."""

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--gth-source", type=Path, required=True)
    prepare.add_argument(
        "--gth-name",
        choices=tuple(PSEUDOPOTENTIAL_SPECS),
        default=GTH_NAME,
    )
    prepare.add_argument("--out", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--out", type=Path, required=True)
    validate.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload = (
        prepare_iron_spin_workload(
            gth_source=args.gth_source,
            out=args.out,
            gth_name=args.gth_name,
        )
        if args.command == "prepare"
        else run_iron_spin_validation(
            manifest_path=args.manifest,
            out=args.out,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
