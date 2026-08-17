"""Material cases for the bounded periodic SCF development gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic._artifact_identity import sha256_bytes
from mlx_atomistic.benchmarks.dft_carbon import load_carbon_workload
from mlx_atomistic.benchmarks.dft_carbon_eos import (
    validation_lattice_constants as carbon_lattice_constants,
)
from mlx_atomistic.benchmarks.dft_carbon_eos_runner import (
    PROFILE_SPECS as CARBON_PROFILE_SPECS,
)
from mlx_atomistic.benchmarks.dft_mgo import load_mgo_workload
from mlx_atomistic.benchmarks.dft_mgo_eos import (
    validation_lattice_constants as mgo_lattice_constants,
)
from mlx_atomistic.benchmarks.dft_mgo_eos_runner import (
    PROFILE_SPECS as MGO_PROFILE_SPECS,
)
from mlx_atomistic.benchmarks.dft_runtime_contract import load_workload
from mlx_atomistic.benchmarks.dft_silicon import ANGSTROM_TO_BOHR
from mlx_atomistic.dft import (
    KPoint,
    KPointMesh,
    MonkhorstPackGrid,
    PeriodicDavidsonConfig,
    PeriodicDFTSystem,
    PeriodicSCFConfig,
    build_time_reversal_ownership,
    read_gth,
)

CASE_NAMES = ("silicon", "carbon", "mgo")
_SILICON_MAX_BATCH_TRANSIENT_BYTES = 512 * 1024**2


@dataclass(frozen=True)
class SCFGateCase:
    """Fully resolved input for one bounded periodic SCF gate."""

    name: str
    profile: str
    target_id: str
    manifest: dict[str, Any]
    manifest_bytes: bytes
    system: PeriodicDFTSystem
    kpoint_mesh: KPointMesh
    selected_owner_indices: tuple[int, ...]
    cutoff_hartree: float
    occupied_band_count: int
    max_batch_transient_bytes: int
    resource_paths: tuple[Path, ...]

    @property
    def atom_count(self) -> int:
        """Return the number of atoms in the periodic cell."""

        return int(self.manifest["system"]["atom_count"])

    @property
    def workload_fingerprint(self) -> str:
        """Return the validated workload identity."""

        return str(self.manifest["workload_fingerprint"])

    def resource_records(self) -> list[dict[str, str]]:
        """Return stable resource paths and content hashes."""

        return [
            {"path": str(path), "sha256": sha256_bytes(path.read_bytes())}
            for path in self.resource_paths
        ]


def _selected_indices(indices: tuple[int, ...], representatives: int) -> tuple[int, ...]:
    if representatives > len(indices):
        msg = (
            f"requested {representatives} representative k-points, but the "
            f"workload contains only {len(indices)} owners"
        )
        raise ValueError(msg)
    return indices[:representatives]


def _owner_points(
    workload: dict[str, Any], representatives: int
) -> list[dict[str, Any]]:
    owners = [
        point
        for point in workload["physics"]["kpoints"]
        if point["role"] == "owner"
    ]
    selected = _selected_indices(tuple(range(len(owners))), representatives)
    return [owners[index] for index in selected]


def _manifest_owner_mesh(
    workload: dict[str, Any], representatives: int
) -> tuple[KPointMesh, tuple[int, ...]]:
    selected = _owner_points(workload, representatives)
    return (
        KPointMesh(
            [
                KPoint(
                    point["reduced_coordinates"],
                    weight=float(point["weight"]["numerator"])
                    / float(point["weight"]["denominator"]),
                    coordinate_system="reduced",
                )
                for point in selected
            ]
        ),
        tuple(int(point["index"]) for point in selected),
    )


def _generated_owner_mesh(
    size: tuple[int, int, int], representatives: int
) -> tuple[KPointMesh, tuple[int, ...]]:
    full_mesh = MonkhorstPackGrid(size)
    ownership = build_time_reversal_ownership(full_mesh)
    selected = _selected_indices(ownership.representative_indices, representatives)
    return (
        KPointMesh(
            [
                KPoint(
                    full_mesh.points[index].vector,
                    weight=full_mesh.points[index].weight,
                    coordinate_system="reduced",
                )
                for index in selected
            ]
        ),
        selected,
    )


def _silicon_case(
    manifest_path: Path,
    *,
    gth_source: Path | None,
    representatives: int,
) -> SCFGateCase:
    if gth_source is None:
        raise ValueError("the silicon case requires --gth-source")
    manifest, _resource = load_workload(manifest_path, gth_source=gth_source)
    workload = dict(manifest)
    system_values = workload["system"]
    physics = workload["physics"]
    lattice = float(system_values["lattice_constant_bohr"])
    mesh, indices = _manifest_owner_mesh(workload, representatives)
    resource_path = (manifest_path.parent / "resources/Si-GTH-PBE-q4.gth").resolve()
    system = PeriodicDFTSystem(
        (lattice, lattice, lattice),
        physics["fft_shape"],
        np.asarray(system_values["fractional_positions"], dtype=np.float64) * lattice,
        read_gth(resource_path, element="Si", name="GTH-PBE-q4"),
        electron_count=float(system_values["electron_count"]),
    )
    return SCFGateCase(
        name="silicon",
        profile="production",
        target_id=str(workload["target_id"]),
        manifest=workload,
        manifest_bytes=manifest_path.read_bytes(),
        system=system,
        kpoint_mesh=mesh,
        selected_owner_indices=indices,
        cutoff_hartree=float(physics["kinetic_cutoff_hartree"]),
        occupied_band_count=int(system_values["occupied_band_count"]),
        max_batch_transient_bytes=_SILICON_MAX_BATCH_TRANSIENT_BYTES,
        resource_paths=(resource_path,),
    )


def _carbon_case(manifest_path: Path, *, representatives: int) -> SCFGateCase:
    workload, resource = load_carbon_workload(manifest_path)
    settings = CARBON_PROFILE_SPECS["cutoff40"]
    system_values = workload["system"]
    lattice = carbon_lattice_constants()[3] * ANGSTROM_TO_BOHR
    mesh, indices = _generated_owner_mesh(
        tuple(settings["kpoint_mesh"]),
        representatives,
    )
    system = PeriodicDFTSystem(
        (lattice, lattice, lattice),
        settings["fft_shape"],
        np.asarray(system_values["fractional_positions"], dtype=np.float64) * lattice,
        read_gth(resource, element="C", name="GTH-PBE-q4"),
        electron_count=float(system_values["electron_count"]),
    )
    return SCFGateCase(
        name="carbon",
        profile="cutoff40",
        target_id=str(workload["target_id"]),
        manifest=workload,
        manifest_bytes=manifest_path.read_bytes(),
        system=system,
        kpoint_mesh=mesh,
        selected_owner_indices=indices,
        cutoff_hartree=float(settings["cutoff_hartree"]),
        occupied_band_count=int(system_values["occupied_band_count"]),
        max_batch_transient_bytes=int(settings["max_batch_transient_bytes"]),
        resource_paths=(resource,),
    )


def _mgo_case(manifest_path: Path, *, representatives: int) -> SCFGateCase:
    workload, resources = load_mgo_workload(manifest_path)
    settings = MGO_PROFILE_SPECS["q2-c70-k6"]
    system_values = workload["system"]
    lattice = mgo_lattice_constants()[3] * ANGSTROM_TO_BOHR
    mesh, indices = _generated_owner_mesh(
        tuple(settings["kpoint_mesh"]),
        representatives,
    )
    magnesium = read_gth(resources["mg_q2"], element="Mg", name="GTH-PBE-q2")
    oxygen = read_gth(resources["o_q6"], element="O", name="GTH-PBE-q6")
    system = PeriodicDFTSystem(
        (lattice, lattice, lattice),
        settings["fft_shape"],
        np.asarray(system_values["fractional_positions"], dtype=np.float64) * lattice,
        electron_count=float(system_values["q2_electron_count"]),
        pseudopotentials=(magnesium,) * 4 + (oxygen,) * 4,
    )
    return SCFGateCase(
        name="mgo",
        profile="q2-c70-k6",
        target_id=str(workload["target_id"]),
        manifest=workload,
        manifest_bytes=manifest_path.read_bytes(),
        system=system,
        kpoint_mesh=mesh,
        selected_owner_indices=indices,
        cutoff_hartree=float(settings["cutoff_hartree"]),
        occupied_band_count=int(system_values["q2_occupied_band_count"]),
        max_batch_transient_bytes=int(settings["max_batch_transient_bytes"]),
        resource_paths=(resources["mg_q2"], resources["o_q6"]),
    )


def load_scf_gate_case(
    name: str,
    manifest_path: Path,
    *,
    gth_source: Path | None,
    representatives: int,
) -> SCFGateCase:
    """Load one validated material case for the bounded SCF gate."""

    if name == "silicon":
        return _silicon_case(
            manifest_path,
            gth_source=gth_source,
            representatives=representatives,
        )
    if name == "carbon":
        return _carbon_case(manifest_path, representatives=representatives)
    if name == "mgo":
        return _mgo_case(manifest_path, representatives=representatives)
    raise ValueError(f"unknown DFT SCF gate case: {name}")


def scf_gate_config(
    case: SCFGateCase,
    *,
    mode: str,
    hpsi_shape_policy: str,
) -> PeriodicSCFConfig:
    """Build the material's production solver controls for a bounded gate."""

    scf = case.manifest["solver"]["scf"]
    davidson = case.manifest["solver"]["davidson"]
    return PeriodicSCFConfig(
        max_iterations=int(scf["max_iterations"]),
        min_iterations=int(scf["min_iterations"]),
        density_tolerance=float(scf["density_tolerance"]),
        energy_tolerance=float(scf["energy_tolerance_hartree"]),
        orbital_tolerance=float(scf["orbital_tolerance"]),
        mixing_beta=float(scf["mixing_beta"]),
        mixer=str(scf["mixer"]),
        adaptive_eigensolver_tolerance=mode == "adaptive",
        initial_eigensolver_tolerance=float(scf["initial_eigensolver_tolerance"]),
        eigensolver_tolerance_scale=float(scf["eigensolver_tolerance_scale"]),
        davidson=PeriodicDavidsonConfig(
            max_iterations=int(davidson["max_iterations"]),
            tolerance=float(davidson["tolerance"]),
            max_subspace_size=int(davidson["max_subspace_size"]),
            preconditioner_floor=float(davidson["preconditioner_floor"]),
        ),
        kpoint_batch_size=8,
        max_batch_transient_bytes=case.max_batch_transient_bytes,
        hpsi_shape_policy=hpsi_shape_policy,
    )
