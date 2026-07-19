"""Prepare and inspect the bounded bulk-silicon DFT parity workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

WORKLOAD_SCHEMA = "mlx-atomistic.dft-silicon-workload.v1"
SOURCE_SCHEMA = "mlx-atomistic.dft-silicon-gth-source.v1"
TARGET_ID = "bulk-silicon-diamond-conventional-pbe-gth-q4"
GTH_ELEMENT = "Si"
GTH_NAME = "GTH-PBE-q4"
ANGSTROM_TO_BOHR = 1.8897261254578281
HARTREE_PER_BOHR3_TO_GPA = 29421.02648438959
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = 51.422067476325886

_SILICON_FRACTIONAL_POSITIONS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
    (0.25, 0.25, 0.25),
    (0.25, 0.75, 0.75),
    (0.75, 0.25, 0.75),
    (0.75, 0.75, 0.25),
)


@dataclass(frozen=True)
class GTHChannel:
    """One angular-momentum channel from a CP2K-style GTH entry."""

    angular_momentum: int
    radius: float
    coupling_matrix: tuple[tuple[float, ...], ...]

    @property
    def projector_count(self) -> int:
        """Number of radial projectors in the channel."""

        return len(self.coupling_matrix)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe channel representation."""

        return {
            "angular_momentum": self.angular_momentum,
            "radius_bohr": self.radius,
            "projector_count": self.projector_count,
            "coupling_matrix_hartree": [list(row) for row in self.coupling_matrix],
        }


@dataclass(frozen=True)
class GTHEntry:
    """Parsed CP2K-style GTH pseudopotential entry."""

    element: str
    names: tuple[str, ...]
    charge_shells: tuple[int, ...]
    local_radius: float
    local_coefficients: tuple[float, ...]
    channels: tuple[GTHChannel, ...]
    source_lines: tuple[str, ...]

    @property
    def valence_charge(self) -> float:
        """Valence charge represented by the entry."""

        return float(sum(self.charge_shells))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe entry representation."""

        return {
            "element": self.element,
            "names": list(self.names),
            "charge_shells": list(self.charge_shells),
            "valence_charge": self.valence_charge,
            "local_radius_bohr": self.local_radius,
            "local_coefficients_hartree": list(self.local_coefficients),
            "channel_count": len(self.channels),
            "channels": [channel.to_dict() for channel in self.channels],
        }


def _numbers(line: str) -> list[float]:
    return [float(value) for value in line.split()]


def _payload_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", maxsplit=1)[0].strip()
        if line:
            lines.append(line)
    return lines


def _symmetric_upper(rows: list[list[float]], count: int) -> tuple[tuple[float, ...], ...]:
    matrix = np.zeros((count, count), dtype=np.float64)
    for row_index, values in enumerate(rows):
        expected = count - row_index
        if len(values) != expected:
            msg = (
                "GTH coupling row has "
                f"{len(values)} values; expected {expected} for row {row_index}"
            )
            raise ValueError(msg)
        matrix[row_index, row_index:] = values
        matrix[row_index:, row_index] = values
    return tuple(tuple(float(value) for value in row) for row in matrix)


def parse_gth_entry(
    path: str | Path,
    *,
    element: str = GTH_ELEMENT,
    name: str = GTH_NAME,
) -> GTHEntry:
    """Parse one entry from a CP2K-style GTH database.

    Args:
        path: CP2K-style GTH database path.
        element: Element symbol to select. Defaults to ``"Si"``.
        name: Entry alias to select. Defaults to ``"GTH-PBE-q4"``.

    Returns:
        The complete selected GTH entry.

    Raises:
        ValueError: If the entry is missing or structurally incomplete.
    """

    source = Path(path)
    payload = _payload_lines(source)
    header_index = None
    for index, line in enumerate(payload):
        parts = line.split()
        if parts and parts[0] == element and name in parts[1:]:
            header_index = index
            break
    if header_index is None:
        msg = f"GTH entry {element} {name} was not found in {source}"
        raise ValueError(msg)

    cursor = header_index
    header = payload[cursor].split()
    cursor += 1
    if cursor + 2 >= len(payload):
        msg = "GTH entry is incomplete"
        raise ValueError(msg)
    charge_shells = tuple(int(value) for value in payload[cursor].split())
    cursor += 1
    local_parts = payload[cursor].split()
    cursor += 1
    if len(local_parts) < 3:
        msg = "GTH local-potential line is incomplete"
        raise ValueError(msg)
    local_radius = float(local_parts[0])
    local_count = int(local_parts[1])
    local_coefficients = tuple(float(value) for value in local_parts[2:])
    if len(local_coefficients) != local_count:
        msg = "GTH local coefficient count does not match its declaration"
        raise ValueError(msg)
    channel_count = int(payload[cursor].split()[0])
    cursor += 1
    channels: list[GTHChannel] = []
    for angular_momentum in range(channel_count):
        first = payload[cursor].split()
        cursor += 1
        if len(first) < 3:
            msg = f"GTH channel l={angular_momentum} is incomplete"
            raise ValueError(msg)
        radius = float(first[0])
        projector_count = int(first[1])
        if projector_count <= 0:
            msg = "GTH projector count must be positive"
            raise ValueError(msg)
        upper_rows = [[float(value) for value in first[2:]]]
        for _ in range(1, projector_count):
            if cursor >= len(payload):
                msg = f"GTH channel l={angular_momentum} coupling matrix is incomplete"
                raise ValueError(msg)
            upper_rows.append(_numbers(payload[cursor]))
            cursor += 1
        channels.append(
            GTHChannel(
                angular_momentum=angular_momentum,
                radius=radius,
                coupling_matrix=_symmetric_upper(upper_rows, projector_count),
            )
        )

    source_lines = tuple(payload[header_index:cursor])
    return GTHEntry(
        element=header[0],
        names=tuple(header[1:]),
        charge_shells=charge_shells,
        local_radius=local_radius,
        local_coefficients=local_coefficients,
        channels=tuple(channels),
        source_lines=source_lines,
    )


def _upper_rows(matrix: tuple[tuple[float, ...], ...]) -> list[list[float]]:
    return [list(row[index:]) for index, row in enumerate(matrix)]


def render_qe_gth(entry: GTHEntry) -> str:
    """Render a deterministic standalone GTH file accepted by Quantum ESPRESSO.

    Args:
        entry: Parsed GTH entry.

    Returns:
        Standalone GTH file text.

    Raises:
        ValueError: If the selected entry is outside the bounded silicon/PBE scope.
    """

    if entry.element != GTH_ELEMENT or GTH_NAME not in entry.names:
        msg = "only the selected Si GTH-PBE-q4 entry is admitted"
        raise ValueError(msg)
    if len(entry.channels) != 2:
        msg = "the selected silicon GTH entry must contain s and p channels"
        raise ValueError(msg)
    lines = [
        "Goedecker pseudopotential for Si",
        f"14 {entry.valence_charge:g} 260716 zatom,zion,pspdat",
        "10 11 1 2 2001 0 pspcod,pspxc,lmax,lloc,mmax,r2well",
        " ".join(
            [
                f"{entry.local_radius:.12g}",
                str(len(entry.local_coefficients)),
                *(f"{value:.12g}" for value in entry.local_coefficients),
            ]
        ),
        str(len(entry.channels)),
    ]
    for channel in entry.channels:
        rows = _upper_rows(channel.coupling_matrix)
        lines.append(
            " ".join(
                [
                    f"{channel.radius:.12g}",
                    str(channel.projector_count),
                    *(f"{value:.12g}" for value in rows[0]),
                ]
            )
        )
        lines.extend(" ".join(f"{value:.12g}" for value in row) for row in rows[1:])
        if channel.angular_momentum > 0:
            for row_index in range(channel.projector_count):
                lines.append(" ".join("0" for _ in range(channel.projector_count - row_index)))
    return "\n".join(lines) + "\n"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_exact(path: Path, payload: bytes) -> str:
    digest = _sha256_bytes(payload)
    if path.exists():
        current = path.read_bytes()
        if current != payload:
            msg = f"refusing to replace mismatched existing file: {path}"
            raise ValueError(msg)
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return digest


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _command_text(command: list[str] | None) -> str:
    values = sys.argv if command is None else command
    return shlex.join(str(value) for value in values)


def _workload_manifest(
    *,
    entry: GTHEntry,
    source_manifest_path: Path,
    gth_path: Path,
    gth_sha256: str,
    command: str,
) -> dict[str, Any]:
    lattice_angstrom = 5.43
    volume_lattice = [5.25, 5.31, 5.37, 5.43, 5.49, 5.55, 5.61]
    payload: dict[str, Any] = {
        "schema_version": WORKLOAD_SCHEMA,
        "target_id": TARGET_ID,
        "engine": "mlx_atomistic",
        "reference_engine": "quantum_espresso",
        "command": command,
        "pseudopotential": {
            "element": entry.element,
            "name": GTH_NAME,
            "format": "gth",
            "functional": "PBE",
            "valence_charge": entry.valence_charge,
            "source_manifest": str(source_manifest_path),
            "path": str(gth_path),
            "sha256": gth_sha256,
            "channel_count": len(entry.channels),
            "channels": [channel.to_dict() for channel in entry.channels],
        },
        "system": {
            "name": "diamond-silicon-conventional-cubic",
            "cell_family": "cubic-orthorhombic",
            "lattice_constant_angstrom": lattice_angstrom,
            "lattice_constant_bohr": lattice_angstrom * ANGSTROM_TO_BOHR,
            "atom_count": len(_SILICON_FRACTIONAL_POSITIONS),
            "symbols": [GTH_ELEMENT] * len(_SILICON_FRACTIONAL_POSITIONS),
            "fractional_positions": [list(row) for row in _SILICON_FRACTIONAL_POSITIONS],
            "electron_count": entry.valence_charge * len(_SILICON_FRACTIONAL_POSITIONS),
            "spin_mode": "unpolarized",
            "occupancy_per_band": 2.0,
            "occupied_band_count": 16,
        },
        "numerics": {
            "functional": "production-pbe-pw92",
            "kinetic_cutoff_candidates_hartree": [10.0, 15.0, 20.0, 25.0, 30.0],
            "fft_shape_candidates": [
                [32, 32, 32],
                [40, 40, 40],
                [48, 48, 48],
                [56, 56, 56],
                [64, 64, 64],
            ],
            "kpoint_mesh_candidates": [[2, 2, 2], [3, 3, 3], [4, 4, 4]],
            "kpoint_centering": "monkhorst-pack-qe-compatible",
            "scf": {
                "max_iterations": 80,
                "density_tolerance": 1e-6,
                "energy_tolerance_hartree_per_atom": 1e-6,
                "orbital_tolerance": 1e-6,
                "orthonormality_tolerance": 1e-4,
                "electron_count_tolerance": 1e-4,
                "mixer": "diis",
                "mixing_beta": 0.35,
            },
            "convergence_thresholds": {
                "energy_mev_per_atom": 0.5,
                "force_ev_per_angstrom": 0.01,
                "stress_gpa": 0.05,
            },
        },
        "cases": {
            "equilibrium": {
                "kind": "scf",
                "lattice_constant_angstrom": lattice_angstrom,
            },
            "displaced_atom": {
                "kind": "force",
                "lattice_constant_angstrom": lattice_angstrom,
                "atom_index": 0,
                "axis": 0,
                "offset_angstrom": 0.02,
                "finite_difference_step_angstrom": 0.005,
                "step_check_angstrom": 0.0025,
            },
            "strain_minus": {
                "kind": "stress",
                "isotropic_strain": -0.01,
                "finite_difference_strain": 0.0025,
                "step_check_strain": 0.00125,
            },
            "strain_plus": {
                "kind": "stress",
                "isotropic_strain": 0.01,
                "finite_difference_strain": 0.0025,
                "step_check_strain": 0.00125,
            },
            "volume_scan": {
                "kind": "lattice_scan",
                "lattice_constants_angstrom": volume_lattice,
            },
        },
        "comparison_tolerances": {
            "energy_mev_per_atom": 5.0,
            "force_rms_ev_per_angstrom": 0.02,
            "force_max_component_ev_per_angstrom": 0.05,
            "stress_max_component_gpa": 0.2,
            "lattice_constant_relative": 0.005,
            "basis_lattice_constant_relative": 0.001,
            "rerun_energy_hartree_per_atom": 1e-5,
        },
        "units": {
            "internal_length": "bohr",
            "internal_energy": "hartree",
            "positions": "fractional",
            "force": "hartree/bohr",
            "stress": "gigapascal",
            "reference_energy": "rydberg",
            "reference_force": "rydberg/bohr",
            "reference_stress": "kilobar",
        },
        "target_host": {
            "model": "MacBook Pro",
            "chip": "Apple M5 Max",
            "low_power_mode_required": True,
            "pmset_expected_lowpowermode": 1,
            "required_provenance": [
                "system_profiler",
                "sw_vers",
                "pmset",
                "power_source",
                "thermal_pressure",
                "mlx_version",
                "warmups",
                "repetitions",
                "synchronization",
            ],
        },
    }
    fingerprint_payload = dict(payload)
    payload["fingerprint_sha256"] = _sha256_bytes(_canonical_json(fingerprint_payload))
    return payload


def prepare_workload(
    *,
    gth_source: str | Path,
    out: str | Path,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Prepare the canonical silicon pseudopotential and workload manifests.

    Args:
        gth_source: Caller-provided CP2K-style GTH database.
        out: Caller-provided workload output directory.
        command: Optional command tokens for deterministic provenance. Defaults
            to the current process arguments.

    Returns:
        JSON-safe preparation summary.
    """

    source_path = Path(gth_source).expanduser().resolve()
    if not source_path.is_file():
        msg = f"GTH source does not exist: {source_path}"
        raise FileNotFoundError(msg)
    output_root = Path(out).expanduser()
    entry = parse_gth_entry(source_path)
    gth_bytes = render_qe_gth(entry).encode()
    gth_path = output_root / "source" / "Si-q4-pbe.gth"
    gth_sha256 = _write_exact(gth_path, gth_bytes)
    source_manifest_path = output_root / "source" / "gth-manifest.json"
    command_text = _command_text(command)
    source_manifest = {
        "schema_version": SOURCE_SCHEMA,
        "target_id": TARGET_ID,
        "source_path": str(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "source_sha256": _sha256_bytes(source_path.read_bytes()),
        "selected_entry": entry.to_dict(),
        "selected_entry_text": "\n".join(entry.source_lines) + "\n",
        "extracted_path": str(gth_path),
        "extracted_size_bytes": len(gth_bytes),
        "extracted_sha256": gth_sha256,
        "license": "CP2K repository GPL-2.0-or-later; parameter citation required",
        "citations": [
            "Goedecker, Teter, and Hutter, Phys. Rev. B 54, 1703 (1996)",
            "Krack, Theor. Chem. Acc. 114, 145 (2005)",
        ],
        "command": command_text,
    }
    _write_exact(source_manifest_path, _canonical_json(source_manifest))
    workload_path = output_root / "manifest.json"
    workload = _workload_manifest(
        entry=entry,
        source_manifest_path=source_manifest_path,
        gth_path=gth_path,
        gth_sha256=gth_sha256,
        command=command_text,
    )
    _write_exact(workload_path, _canonical_json(workload))
    return {
        "status": "prepared",
        "target_id": TARGET_ID,
        "source_manifest": str(source_manifest_path),
        "workload_manifest": str(workload_path),
        "gth_path": str(gth_path),
        "gth_sha256": gth_sha256,
        "case_count": len(workload["cases"]),
    }


def inspect_workload(path: str | Path) -> dict[str, Any]:
    """Load and summarize a prepared silicon workload manifest.

    Args:
        path: Workload manifest path.

    Returns:
        JSON-safe manifest summary.

    Raises:
        ValueError: If the schema, fingerprint, source, or bounded target is invalid.
    """

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != WORKLOAD_SCHEMA:
        msg = "unsupported silicon workload schema"
        raise ValueError(msg)
    if payload.get("target_id") != TARGET_ID:
        msg = "unexpected silicon workload target"
        raise ValueError(msg)
    expected = payload.get("fingerprint_sha256")
    fingerprint_payload = {
        key: value for key, value in payload.items() if key != "fingerprint_sha256"
    }
    observed = _sha256_bytes(_canonical_json(fingerprint_payload))
    if expected != observed:
        msg = "silicon workload fingerprint mismatch"
        raise ValueError(msg)
    gth = Path(payload["pseudopotential"]["path"])
    if not gth.is_file():
        msg = "prepared silicon GTH file is missing"
        raise ValueError(msg)
    if _sha256_bytes(gth.read_bytes()) != payload["pseudopotential"]["sha256"]:
        msg = "prepared silicon GTH hash mismatch"
        raise ValueError(msg)
    return {
        "status": "ready",
        "target_id": payload["target_id"],
        "atom_count": payload["system"]["atom_count"],
        "electron_count": payload["system"]["electron_count"],
        "pseudopotential_sha256": payload["pseudopotential"]["sha256"],
        "case_ids": sorted(payload["cases"]),
        "target_chip": payload["target_host"]["chip"],
        "low_power_mode_required": payload["target_host"]["low_power_mode_required"],
        "fingerprint_sha256": expected,
    }


def run_mlx_smoke(
    *,
    manifest_path: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    """Run a compact full-GTH silicon periodic-SCF smoke case.

    Args:
        manifest_path: Prepared silicon workload manifest.
        out: Caller-provided output directory.

    Returns:
        JSON-safe periodic SCF smoke summary.
    """

    from mlx_atomistic.dft import (
        KPoint,
        KPointMesh,
        PeriodicDavidsonConfig,
        PeriodicDFTSystem,
        PeriodicSCFConfig,
        read_gth,
        run_periodic_scf,
    )

    inspect_workload(manifest_path)
    manifest = json.loads(Path(manifest_path).read_text())
    system_data = manifest["system"]
    lattice = float(system_data["lattice_constant_bohr"])
    positions = lattice * np.asarray(system_data["fractional_positions"], dtype=np.float64)
    pseudopotential = read_gth(manifest["pseudopotential"]["path"], element=GTH_ELEMENT)
    system = PeriodicDFTSystem(
        (lattice, lattice, lattice),
        (8, 8, 8),
        positions,
        pseudopotential,
        electron_count=float(system_data["electron_count"]),
    )
    mesh = KPointMesh([KPoint((0.0, 0.0, 0.0), coordinate_system="reduced")])
    result = run_periodic_scf(
        system,
        cutoff_hartree=2.0,
        kpoint_mesh=mesh,
        n_bands=int(system_data["occupied_band_count"]),
        config=PeriodicSCFConfig(
            max_iterations=8,
            min_iterations=2,
            density_tolerance=0.3,
            energy_tolerance=0.5,
            orbital_tolerance=5e-3,
            mixing_beta=0.5,
            mixer="linear",
            davidson=PeriodicDavidsonConfig(
                max_iterations=24,
                tolerance=5e-3,
                max_subspace_size=48,
            ),
        ),
    )
    output_root = Path(out)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "mlx-atomistic.dft-silicon-mlx-smoke.v1",
        "target_id": manifest["target_id"],
        "manifest_fingerprint": manifest["fingerprint_sha256"],
        "case": "equilibrium",
        "smoke": True,
        "grid_shape": list(system.grid.shape),
        "cutoff_hartree": 2.0,
        "kpoint_mesh": [1, 1, 1],
        "pseudopotential_sha256": manifest["pseudopotential"]["sha256"],
        "result": result.to_dict(),
    }
    report_path = output_root / "report.json"
    report_path.write_bytes(_canonical_json(payload))
    return {
        "status": result.status,
        "converged": result.converged,
        "report": str(report_path),
        "total_energy_hartree": result.total_energy,
        "electron_count": result.electron_count,
        "kpoint_count": len(result.kpoints),
        "dense_full_hamiltonian": False,
    }


def finite_difference_force_array(
    energy_function: Any,
    positions_bohr: np.ndarray,
    *,
    displacement_bohr: float,
) -> np.ndarray:
    """Return complete central-difference forces for a geometry energy function.

    Args:
        energy_function: Callable accepting Cartesian positions in bohr and
            returning energy in Hartree.
        positions_bohr: Cartesian positions with shape ``(n_atoms, 3)``.
        displacement_bohr: Positive central-difference displacement in bohr.

    Returns:
        Complete force array in Hartree/bohr.
    """

    positions = np.asarray(positions_bohr, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions_bohr must have shape (n_atoms, 3)"
        raise ValueError(msg)
    if displacement_bohr <= 0.0:
        msg = "displacement_bohr must be positive"
        raise ValueError(msg)
    forces = np.zeros_like(positions)
    for atom_index in range(positions.shape[0]):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom_index, axis] += displacement_bohr
            minus[atom_index, axis] -= displacement_bohr
            forces[atom_index, axis] = -(
                float(energy_function(plus)) - float(energy_function(minus))
            ) / (2.0 * displacement_bohr)
    return forces


def isotropic_stress_tensor(
    energy_function: Any,
    lattice_bohr: float,
    *,
    strain_step: float,
) -> np.ndarray:
    """Return a cubic stress tensor from an isotropic energy derivative.

    Args:
        energy_function: Callable accepting a cubic lattice length in bohr and
            returning energy in Hartree with fractional coordinates fixed.
        lattice_bohr: Base cubic lattice constant in bohr.
        strain_step: Positive dimensionless central-difference strain.

    Returns:
        Symmetry-complete ``3x3`` stress tensor in GPa using
        ``sigma_ii = (dE/dstrain)/(3V)`` and zero shear components.
    """

    if lattice_bohr <= 0.0 or strain_step <= 0.0:
        msg = "lattice_bohr and strain_step must be positive"
        raise ValueError(msg)
    e_plus = float(energy_function(lattice_bohr * (1.0 + strain_step)))
    e_minus = float(energy_function(lattice_bohr * (1.0 - strain_step)))
    derivative = (e_plus - e_minus) / (2.0 * strain_step)
    diagonal = derivative / (3.0 * lattice_bohr**3) * HARTREE_PER_BOHR3_TO_GPA
    return np.diag([diagonal, diagonal, diagonal])


def fit_lattice_curve(
    lattice_constants_angstrom: Sequence[float],
    energies_hartree: Sequence[float],
) -> dict[str, Any]:
    """Fit a guarded quadratic equilibrium lattice constant.

    Args:
        lattice_constants_angstrom: Ordered cubic lattice samples in Angstrom.
        energies_hartree: One total energy per lattice sample in Hartree.

    Returns:
        JSON-safe fit status, coefficients, minimum, and conditioning metadata.
    """

    lattice = np.asarray(lattice_constants_angstrom, dtype=np.float64)
    energies = np.asarray(energies_hartree, dtype=np.float64)
    if lattice.shape != energies.shape or lattice.ndim != 1 or lattice.size != 7:
        msg = "lattice fitting requires seven matching one-dimensional samples"
        raise ValueError(msg)
    if not np.isfinite(lattice).all() or not np.isfinite(energies).all():
        msg = "lattice fitting inputs must be finite"
        raise ValueError(msg)
    if np.any(np.diff(lattice) <= 0.0):
        msg = "lattice constants must be strictly increasing"
        raise ValueError(msg)
    centered = lattice - float(np.mean(lattice))
    design = np.stack([centered * centered, centered, np.ones_like(centered)], axis=1)
    condition = float(np.linalg.cond(design))
    coefficients = np.linalg.lstsq(design, energies, rcond=None)[0]
    curvature, slope, offset = (float(value) for value in coefficients)
    if curvature <= 0.0 or not np.isfinite(condition):
        return {
            "status": "blocked",
            "blocker": "nonconvex_or_ill_conditioned_lattice_fit",
            "condition_number": condition,
        }
    minimum = float(np.mean(lattice) - slope / (2.0 * curvature))
    interior = float(lattice[0]) < minimum < float(lattice[-1])
    observed_minimum_index = int(np.argmin(energies))
    observed_interior = 0 < observed_minimum_index < lattice.size - 1
    status = "ok" if interior and observed_interior else "blocked"
    return {
        "status": status,
        "blocker": None if status == "ok" else "lattice_minimum_not_interior",
        "equilibrium_lattice_constant_angstrom": minimum,
        "observed_minimum_index": observed_minimum_index,
        "observed_minimum_angstrom": float(lattice[observed_minimum_index]),
        "condition_number": condition,
        "quadratic_coefficients_centered": [curvature, slope, offset],
    }


def _parse_pmset_power_mode(output: str) -> tuple[str | None, int | None]:
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in {"lowpowermode", "powermode"}:
            try:
                return parts[0], int(parts[-1])
            except ValueError:
                return parts[0], None
    return None, None


def _energy_accounting_residual(result: Any) -> float:
    terms = result.energy_by_term
    required = {"band", "hartree", "xc", "density_xc_potential", "ion_ewald", "total"}
    missing = sorted(required.difference(terms))
    if missing:
        msg = f"periodic energy accounting is missing terms: {', '.join(missing)}"
        raise ValueError(msg)
    reconstructed = (
        terms["band"]
        - terms["hartree"]
        + terms["xc"]
        - terms["density_xc_potential"]
        + terms["ion_ewald"]
    )
    return float(reconstructed - terms["total"])


def collect_host_provenance() -> dict[str, Any]:
    """Collect host, power, thermal, Python, and MLX benchmark provenance.

    Returns:
        JSON-safe host metadata. Unavailable queries are retained as blockers
        rather than guessed.
    """

    from mlx_atomistic.runtime import get_runtime_info

    def run(*command: str) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as error:
            return {"status": "blocked", "error": str(error), "command": list(command)}
        return {
            "status": "ok" if completed.returncode == 0 else "blocked",
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "command": list(command),
        }

    hardware = run("system_profiler", "SPHardwareDataType")
    operating_system = run("sw_vers")
    power = run("pmset", "-g")
    power_profiles = run("pmset", "-g", "custom")
    battery = run("pmset", "-g", "batt")
    thermal = run("sysctl", "-n", "kern.thermal_pressure")
    power_mode_key, low_power_mode = _parse_pmset_power_mode(power.get("stdout", ""))
    hardware_text = hardware.get("stdout", "")
    chip = next(
        (
            line.split(":", maxsplit=1)[1].strip()
            for line in hardware_text.splitlines()
            if line.strip().startswith("Chip:")
        ),
        None,
    )
    model = next(
        (
            line.split(":", maxsplit=1)[1].strip()
            for line in hardware_text.splitlines()
            if line.strip().startswith("Model Name:")
        ),
        None,
    )
    memory = next(
        (
            line.split(":", maxsplit=1)[1].strip()
            for line in hardware_text.splitlines()
            if line.strip().startswith("Memory:")
        ),
        None,
    )
    runtime = get_runtime_info()
    return {
        "model": model,
        "chip": chip,
        "memory": memory,
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "mlx_version": runtime.mlx_version,
        "default_mlx_device": runtime.default_device,
        "metal_available": runtime.metal_available,
        "power_mode_key": power_mode_key,
        "low_power_mode": low_power_mode,
        "low_power_mode_confirmed": low_power_mode == 1,
        "power_source": battery,
        "thermal_pressure": thermal,
        "hardware_query": hardware,
        "os_query": operating_system,
        "power_query": power,
        "power_profiles_query": power_profiles,
        "environment_device": os.environ.get("MLX_ATOMISTIC_DEVICE"),
    }


def _periodic_settings(profile: str) -> dict[str, Any]:
    if profile == "diagnostic":
        return {
            "profile": profile,
            "fft_shape": (8, 8, 8),
            "cutoff_hartree": 2.0,
            "kpoint_mesh": (1, 1, 1),
            "scf": {
                "max_iterations": 8,
                "min_iterations": 2,
                "density_tolerance": 0.3,
                "energy_tolerance": 0.5,
                "orbital_tolerance": 5e-3,
                "mixing_beta": 0.5,
                "mixer": "linear",
                "davidson_max_iterations": 24,
                "davidson_tolerance": 5e-3,
                "davidson_max_subspace": 48,
            },
        }
    msg = f"unsupported execution profile: {profile}"
    raise ValueError(msg)


def _kpoint_mesh(size: Sequence[int]) -> Any:
    from mlx_atomistic.dft import KPoint, KPointMesh, MonkhorstPackGrid

    parsed = tuple(int(value) for value in size)
    if parsed == (1, 1, 1):
        return KPointMesh([KPoint((0.0, 0.0, 0.0), coordinate_system="reduced")])
    return MonkhorstPackGrid(parsed)


def run_mlx_workload(
    *,
    manifest_path: str | Path,
    out: str | Path,
    profile: str = "diagnostic",
    case: str = "all",
    repetitions: int = 2,
) -> dict[str, Any]:
    """Run persisted silicon energy, force, stress, lattice, and profile evidence.

    Args:
        manifest_path: Prepared silicon workload manifest.
        out: Caller-provided output directory.
        profile: Execution profile. Slice 5 admits ``"diagnostic"``.
        case: ``"all"`` or one manifest case identifier.
        repetitions: Equilibrium reproducibility repetitions. Defaults to ``2``.

    Returns:
        JSON-safe workload summary and raw artifact paths.
    """

    import mlx.core as mx

    from mlx_atomistic.dft import (
        PeriodicDavidsonConfig,
        PeriodicDFTSystem,
        PeriodicSCFConfig,
        read_gth,
        run_periodic_scf,
    )

    inspect_workload(manifest_path)
    manifest = json.loads(Path(manifest_path).read_text())
    if repetitions < 2:
        msg = "repetitions must be at least two for deterministic evidence"
        raise ValueError(msg)
    if case != "all" and case not in manifest["cases"]:
        msg = f"unknown silicon case: {case}"
        raise ValueError(msg)
    settings = _periodic_settings(profile)
    output_root = Path(out)
    geometry_root = output_root / "geometries"
    geometry_root.mkdir(parents=True, exist_ok=True)
    pseudo = read_gth(manifest["pseudopotential"]["path"], element=GTH_ELEMENT)
    fractional = np.asarray(manifest["system"]["fractional_positions"], dtype=np.float64)
    electron_count = float(manifest["system"]["electron_count"])
    occupied_bands = int(manifest["system"]["occupied_band_count"])
    mesh = _kpoint_mesh(settings["kpoint_mesh"])
    scf_values = settings["scf"]
    scf_config = PeriodicSCFConfig(
        max_iterations=scf_values["max_iterations"],
        min_iterations=scf_values["min_iterations"],
        density_tolerance=scf_values["density_tolerance"],
        energy_tolerance=scf_values["energy_tolerance"],
        orbital_tolerance=scf_values["orbital_tolerance"],
        mixing_beta=scf_values["mixing_beta"],
        mixer=scf_values["mixer"],
        davidson=PeriodicDavidsonConfig(
            max_iterations=scf_values["davidson_max_iterations"],
            tolerance=scf_values["davidson_tolerance"],
            max_subspace_size=scf_values["davidson_max_subspace"],
        ),
    )
    cache: dict[str, Any] = {}
    all_records: list[tuple[Any, dict[str, Any]]] = []

    def run_geometry(
        lattice_bohr: float,
        positions_bohr: np.ndarray,
        *,
        label: str,
        continuation: Any | None = None,
        force_new: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        geometry_payload = {
            "lattice_bohr": round(float(lattice_bohr), 12),
            "positions_bohr": np.round(np.asarray(positions_bohr), 12).tolist(),
            "settings": settings,
        }
        key = _sha256_bytes(_canonical_json(geometry_payload))[:16]
        if not force_new and key in cache:
            return cache[key]
        system = PeriodicDFTSystem(
            (lattice_bohr, lattice_bohr, lattice_bohr),
            settings["fft_shape"],
            positions_bohr,
            pseudo,
            electron_count=electron_count,
        )
        initial_density = None if continuation is None else continuation.density
        initial_coefficients = (
            None
            if continuation is None
            else [
                item.eigen._compact_coefficients for item in continuation.kpoints
            ]
        )
        start = perf_counter()
        result = run_periodic_scf(
            system,
            cutoff_hartree=settings["cutoff_hartree"],
            kpoint_mesh=mesh,
            n_bands=occupied_bands,
            config=scf_config,
            initial_density=initial_density,
            initial_coefficients=initial_coefficients,
        )
        mx.eval(
            result.density,
            *[
                item.eigen._compact_coefficients.values
                for item in result.kpoints
            ],
        )
        elapsed = perf_counter() - start
        case_root = geometry_root / f"{key}-{label}"
        case_root.mkdir(parents=True, exist_ok=True)
        arrays = {"density": np.asarray(result.density)}
        for index, item in enumerate(result.kpoints):
            compact = item.eigen._compact_coefficients
            arrays[f"coefficients_{index}"] = np.asarray(
                compact.layout.unpack_fresh(compact.values)
            )
            arrays[f"eigenvalues_{index}"] = np.asarray(item.eigen.eigenvalues)
            arrays[f"residuals_{index}"] = np.asarray(item.eigen.residuals)
        array_path = case_root / "arrays.npz"
        np.savez_compressed(array_path, **arrays)
        summary = {
            "label": label,
            "geometry_key": key,
            "lattice_bohr": lattice_bohr,
            "positions_bohr": np.asarray(positions_bohr).tolist(),
            "elapsed_wall_seconds": elapsed,
            "energy_accounting_residual_hartree": _energy_accounting_residual(result),
            "result": result.to_dict(),
            "arrays": str(array_path),
        }
        summary_path = case_root / "result.json"
        summary_path.write_bytes(_canonical_json(summary))
        record = (result, {**summary, "result_path": str(summary_path)})
        all_records.append(record)
        if not force_new:
            cache[key] = record
        return record

    def branch_record(
        result: Any,
        summary: dict[str, Any],
        **metadata: Any,
    ) -> dict[str, Any]:
        return {
            **metadata,
            "geometry_key": summary["geometry_key"],
            "result_path": summary["result_path"],
            "arrays": summary["arrays"],
            "total_energy_hartree": result.total_energy,
            "converged": bool(result.converged),
            "electron_count": result.electron_count,
            "density_residual": result.density_residual,
            "energy_delta_hartree": result.energy_delta,
            "max_orbital_residual": max(
                float(np.max(np.asarray(item.eigen.residuals))) for item in result.kpoints
            ),
            "max_orthonormality_error": max(
                item.eigen.orthonormality_error for item in result.kpoints
            ),
        }

    equilibrium_angstrom = float(manifest["system"]["lattice_constant_angstrom"])
    equilibrium_bohr = equilibrium_angstrom * ANGSTROM_TO_BOHR
    equilibrium_positions = equilibrium_bohr * fractional
    selected_cases = set(manifest["cases"]) if case == "all" else {case}
    cases: dict[str, Any] = {}
    equilibrium_results = []
    if "equilibrium" in selected_cases or case == "all":
        for repetition in range(repetitions):
            result, summary = run_geometry(
                equilibrium_bohr,
                equilibrium_positions,
                label=f"equilibrium-r{repetition}",
                force_new=True,
            )
            equilibrium_results.append(result)
            cases.setdefault("equilibrium", {"repetitions": []})["repetitions"].append(summary)
        energies = [result.total_energy for result in equilibrium_results]
        rerun_delta = (max(energies) - min(energies)) / len(fractional)
        rerun_threshold = float(manifest["comparison_tolerances"]["rerun_energy_hartree_per_atom"])
        equilibrium_complete = len(equilibrium_results) == repetitions and all(
            result.converged and np.isfinite(result.total_energy) for result in equilibrium_results
        )
        cases["equilibrium"].update(
            {
                "rerun_energy_delta_hartree_per_atom": rerun_delta,
                "rerun_energy_threshold_hartree_per_atom": rerun_threshold,
                "deterministic": rerun_delta <= rerun_threshold,
                "complete": equilibrium_complete,
                "comparable": equilibrium_complete and rerun_delta <= rerun_threshold,
            }
        )
    parent = equilibrium_results[0] if equilibrium_results else None

    if "displaced_atom" in selected_cases:
        case_data = manifest["cases"]["displaced_atom"]
        displaced = equilibrium_positions.copy()
        displaced[int(case_data["atom_index"]), int(case_data["axis"])] += (
            float(case_data["offset_angstrom"]) * ANGSTROM_TO_BOHR
        )
        displaced_result, displaced_summary = run_geometry(
            equilibrium_bohr,
            displaced,
            label="displaced-base",
            continuation=parent,
        )

        main_step = float(case_data["finite_difference_step_angstrom"]) * ANGSTROM_TO_BOHR
        check_step = float(case_data["step_check_angstrom"]) * ANGSTROM_TO_BOHR
        force_branches = []
        forces = np.zeros_like(displaced)
        for atom_index in range(displaced.shape[0]):
            for axis in range(3):
                branch_results: dict[str, Any] = {}
                for sign, direction in (("plus", 1.0), ("minus", -1.0)):
                    positions = displaced.copy()
                    positions[atom_index, axis] += direction * main_step
                    branch_result, branch_summary = run_geometry(
                        equilibrium_bohr,
                        positions,
                        label=f"force-a{atom_index}-{axis}-{sign}-main",
                        continuation=displaced_result,
                    )
                    branch_results[sign] = branch_result
                    force_branches.append(
                        branch_record(
                            branch_result,
                            branch_summary,
                            atom_index=atom_index,
                            axis=axis,
                            sign=sign,
                            step_kind="main",
                            displacement_bohr=main_step,
                        )
                    )
                forces[atom_index, axis] = -(
                    branch_results["plus"].total_energy - branch_results["minus"].total_energy
                ) / (2.0 * main_step)
        atom_index = int(case_data["atom_index"])
        axis = int(case_data["axis"])
        check_results: dict[str, Any] = {}
        for sign, direction in (("plus", 1.0), ("minus", -1.0)):
            positions = displaced.copy()
            positions[atom_index, axis] += direction * check_step
            branch_result, branch_summary = run_geometry(
                equilibrium_bohr,
                positions,
                label=f"force-a{atom_index}-{axis}-{sign}-check",
                continuation=displaced_result,
            )
            check_results[sign] = branch_result
            force_branches.append(
                branch_record(
                    branch_result,
                    branch_summary,
                    atom_index=atom_index,
                    axis=axis,
                    sign=sign,
                    step_kind="check",
                    displacement_bohr=check_step,
                )
            )
        check_force = -(
            check_results["plus"].total_energy - check_results["minus"].total_energy
        ) / (2.0 * check_step)
        step_delta = float(abs(check_force - forces[atom_index, axis]))
        step_delta_ev_per_angstrom = step_delta * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
        step_threshold = float(
            manifest["numerics"]["convergence_thresholds"]["force_ev_per_angstrom"]
        )
        branches_converged = all(row["converged"] for row in force_branches)
        complete_force = bool(forces.shape == (len(fractional), 3) and np.isfinite(forces).all())
        cases["displaced_atom"] = {
            "base": displaced_summary,
            "forces_hartree_per_bohr": forces.tolist(),
            "force_units": "hartree/bohr",
            "finite_difference_step_bohr": main_step,
            "step_check_bohr": check_step,
            "selected_component_step_delta_hartree_per_bohr": step_delta,
            "selected_component_step_delta_ev_per_angstrom": step_delta_ev_per_angstrom,
            "step_stability_threshold_ev_per_angstrom": step_threshold,
            "step_stable": step_delta_ev_per_angstrom <= step_threshold,
            "branches_converged": branches_converged,
            "branches": force_branches,
            "complete": complete_force,
            "comparable": complete_force
            and branches_converged
            and step_delta_ev_per_angstrom <= step_threshold,
        }

    for stress_case in ("strain_minus", "strain_plus"):
        if stress_case not in selected_cases:
            continue
        case_data = manifest["cases"][stress_case]
        base_lattice = equilibrium_bohr * (1.0 + float(case_data["isotropic_strain"]))
        base_positions = base_lattice * fractional
        base_result, base_summary = run_geometry(
            base_lattice,
            base_positions,
            label=f"{stress_case}-base",
            continuation=parent,
        )

        stress_branches = []

        def stress_at_step(
            strain_step: float,
            step_kind: str,
            *,
            _stress_case: str = stress_case,
            _base_result: Any = base_result,
            _base_lattice: float = base_lattice,
            _stress_branches: list[dict[str, Any]] = stress_branches,
        ) -> np.ndarray:
            branch_results: dict[str, Any] = {}
            for sign, direction in (("plus", 1.0), ("minus", -1.0)):
                lattice_bohr = _base_lattice * (1.0 + direction * strain_step)
                branch_result, branch_summary = run_geometry(
                    lattice_bohr,
                    lattice_bohr * fractional,
                    label=f"{_stress_case}-{sign}-{step_kind}",
                    continuation=_base_result,
                )
                branch_results[sign] = branch_result
                _stress_branches.append(
                    branch_record(
                        branch_result,
                        branch_summary,
                        sign=sign,
                        step_kind=step_kind,
                        strain_step=strain_step,
                    )
                )
            derivative = (
                branch_results["plus"].total_energy - branch_results["minus"].total_energy
            ) / (2.0 * strain_step)
            diagonal = derivative / (3.0 * _base_lattice**3) * HARTREE_PER_BOHR3_TO_GPA
            return np.diag([diagonal, diagonal, diagonal])

        stress = stress_at_step(
            float(case_data["finite_difference_strain"]),
            "main",
        )
        stress_check = stress_at_step(
            float(case_data["step_check_strain"]),
            "check",
        )
        stress_step_delta = float(np.max(np.abs(stress - stress_check)))
        stress_threshold = float(manifest["numerics"]["convergence_thresholds"]["stress_gpa"])
        branches_converged = all(row["converged"] for row in stress_branches)
        complete_stress = bool(stress.shape == (3, 3) and np.isfinite(stress).all())
        cases[stress_case] = {
            "base": base_summary,
            "stress_gpa": stress.tolist(),
            "stress_step_check_gpa": stress_check.tolist(),
            "stress_units": "gigapascal",
            "step_delta_gpa": stress_step_delta,
            "step_stability_threshold_gpa": stress_threshold,
            "step_stable": stress_step_delta <= stress_threshold,
            "branches_converged": branches_converged,
            "branches": stress_branches,
            "off_diagonal_provenance": "zero by cubic isotropic symmetry",
            "complete": complete_stress,
            "comparable": complete_stress
            and branches_converged
            and stress_step_delta <= stress_threshold,
        }

    if "volume_scan" in selected_cases:
        lattice_values = manifest["cases"]["volume_scan"]["lattice_constants_angstrom"]
        volume_rows = []
        energies = []
        continuation = parent
        for lattice_angstrom in lattice_values:
            lattice_bohr = float(lattice_angstrom) * ANGSTROM_TO_BOHR
            result, summary = run_geometry(
                lattice_bohr,
                lattice_bohr * fractional,
                label=f"volume-{lattice_angstrom}",
                continuation=continuation,
            )
            continuation = result
            energies.append(result.total_energy)
            volume_rows.append(summary)
        fit = fit_lattice_curve(lattice_values, energies)
        volume_complete = (
            len(volume_rows) == 7
            and all(row["result"]["converged"] for row in volume_rows)
            and fit["status"] == "ok"
        )
        cases["volume_scan"] = {
            "rows": volume_rows,
            "lattice_constants_angstrom": lattice_values,
            "energies_hartree": energies,
            "energy_units": "hartree",
            "fit": fit,
            "basis_stability": {
                "status": "blocked",
                "blocker": "next_denser_basis_not_run_in_diagnostic_profile",
                "current_fft_shape": list(settings["fft_shape"]),
                "current_cutoff_hartree": settings["cutoff_hartree"],
            },
            "complete": volume_complete,
            "comparable": False,
        }

    all_scf = [record[0] for record in all_records]
    finite = all(np.isfinite(result.total_energy) for result in all_scf)
    converged = all(result.converged for result in all_scf)
    complete = len(cases) == len(selected_cases) and all(
        value.get("complete", False) for value in cases.values()
    )
    energy_accounting_residuals = [abs(_energy_accounting_residual(result)) for result in all_scf]
    max_energy_accounting_residual = max(energy_accounting_residuals, default=float("inf"))
    max_electron_count_error = max(
        (abs(result.electron_count - electron_count) for result in all_scf),
        default=float("inf"),
    )
    max_orthonormality_error = max(
        (item.eigen.orthonormality_error for result in all_scf for item in result.kpoints),
        default=float("inf"),
    )
    energy_accounting_consistent = max_energy_accounting_residual <= 1e-10
    all_cases_comparable = len(cases) == len(manifest["cases"]) and all(
        value.get("comparable", False) for value in cases.values()
    )
    comparison_blockers = []
    if profile == "diagnostic":
        comparison_blockers.append("diagnostic_profile_not_admitted_for_qe_parity")
    if len(cases) != len(manifest["cases"]):
        comparison_blockers.append("incomplete_manifest_case_set")
    for case_id, value in cases.items():
        if not value.get("comparable", False):
            comparison_blockers.append(f"case_not_comparable:{case_id}")
    host = collect_host_provenance()
    profile_rows = [
        {
            "geometry_key": summary[1]["geometry_key"],
            "label": summary[1]["label"],
            "result_path": summary[1]["result_path"],
            "elapsed_wall_seconds": summary[1]["elapsed_wall_seconds"],
            "scf_timings_ms": summary[0].timings,
        }
        for summary in all_records
    ]
    run_protocol = {
        "warmup_count": 0,
        "repetitions": repetitions,
        "synchronization": ("mx.eval(density, coefficient stacks) before stopping each wall timer"),
        "cadence": "serial geometry execution",
        "timing_sample_count": len(profile_rows),
    }
    timing_blockers = []
    if host["chip"] != manifest["target_host"]["chip"]:
        timing_blockers.append("target_chip_not_confirmed")
    if not host["low_power_mode_confirmed"]:
        timing_blockers.append("low_power_mode_not_confirmed")
    report = {
        "schema_version": "mlx-atomistic.dft-silicon-mlx-report.v1",
        "target_id": manifest["target_id"],
        "manifest_fingerprint": manifest["fingerprint_sha256"],
        "profile": profile,
        "admission_status": "diagnostic" if profile == "diagnostic" else "candidate",
        "status": (
            "ok" if finite and converged and complete and energy_accounting_consistent else "failed"
        ),
        "comparison_status": "comparable" if all_cases_comparable else "diagnostic",
        "comparison_blockers": sorted(set(comparison_blockers)),
        "finite": finite,
        "converged": converged,
        "complete": complete,
        "internal_gates": {
            "max_energy_accounting_residual_hartree": max_energy_accounting_residual,
            "energy_accounting_consistent": energy_accounting_consistent,
            "max_electron_count_error": max_electron_count_error,
            "max_orthonormality_error": max_orthonormality_error,
        },
        "settings": settings,
        "host": host,
        "run_protocol": run_protocol,
        "timing_admission": {
            "status": "admitted" if not timing_blockers else "blocked",
            "blockers": timing_blockers,
        },
        "cases": cases,
        "profile_rows": profile_rows,
        "raw_geometry_count": len(all_records),
    }
    report_path = output_root / "report.json"
    report_path.write_bytes(_canonical_json(report))
    profile_path = output_root / "profile.json"
    profile_path.write_bytes(
        _canonical_json(
            {
                "schema_version": "mlx-atomistic.dft-silicon-profile.v1",
                "target_id": manifest["target_id"],
                "host": host,
                "settings": settings,
                "run_protocol": run_protocol,
                "timing_admission": report["timing_admission"],
                "rows": profile_rows,
            }
        )
    )
    return {
        "status": report["status"],
        "admission_status": report["admission_status"],
        "report": str(report_path),
        "profile": str(profile_path),
        "geometry_count": report["raw_geometry_count"],
        "host_low_power_mode_confirmed": host["low_power_mode_confirmed"],
    }


def main(argv: list[str] | None = None) -> None:
    """Run the silicon workload preparation and inspection CLI.

    Args:
        argv: Optional argument vector. Defaults to process arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare the bounded silicon workload.")
    prepare.add_argument("--gth-database", "--gth-source", dest="gth_source", required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--json", action="store_true")

    inspect = subparsers.add_parser("inspect", help="Inspect a prepared workload manifest.")
    inspect.add_argument("--manifest", type=Path, required=True)
    inspect.add_argument("--json", action="store_true")

    mlx = subparsers.add_parser("mlx", help="Run the MLX silicon workload.")
    mlx.add_argument("--manifest", type=Path, required=True)
    mlx.add_argument("--case", default="all")
    mlx.add_argument("--smoke", action="store_true")
    mlx.add_argument("--profile", default="diagnostic")
    mlx.add_argument("--repetitions", type=int, default=2)
    mlx.add_argument("--out", type=Path, required=True)
    mlx.add_argument("--json", action="store_true")

    compare = subparsers.add_parser("compare", help="Compare normalized MLX and QE reports.")
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--mlx", type=Path, required=True)
    compare.add_argument("--qe", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare_workload(
            gth_source=args.gth_source,
            out=args.out,
            command=[
                sys.executable,
                "-m",
                "mlx_atomistic.benchmarks.dft_silicon",
                *(argv or sys.argv[1:]),
            ],
        )
    elif args.command == "inspect":
        payload = inspect_workload(args.manifest)
    elif args.command == "mlx":
        if args.smoke:
            if args.case != "equilibrium":
                msg = "--smoke admits only --case equilibrium"
                raise ValueError(msg)
            payload = run_mlx_smoke(manifest_path=args.manifest, out=args.out)
        else:
            payload = run_mlx_workload(
                manifest_path=args.manifest,
                out=args.out,
                profile=args.profile,
                case=args.case,
                repetitions=args.repetitions,
            )
    else:
        from mlx_atomistic.benchmarks.dft_silicon_parity import (
            compare_silicon_reports,
        )

        payload = compare_silicon_reports(
            manifest_path=args.manifest,
            mlx_report_path=args.mlx,
            qe_report_path=args.qe,
            out=args.out,
        )
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(" ".join(f"{key}={value}" for key, value in payload.items()))


if __name__ == "__main__":
    main()
