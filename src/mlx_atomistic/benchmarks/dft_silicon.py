"""Prepare and inspect the bounded bulk-silicon DFT parity workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

WORKLOAD_SCHEMA = "mlx-atomistic.dft-silicon-workload.v1"
SOURCE_SCHEMA = "mlx-atomistic.dft-silicon-gth-source.v1"
TARGET_ID = "bulk-silicon-diamond-conventional-pbe-gth-q4"
GTH_ELEMENT = "Si"
GTH_NAME = "GTH-PBE-q4"
ANGSTROM_TO_BOHR = 1.8897261254578281

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
                lines.append(
                    " ".join("0" for _ in range(channel.projector_count - row_index))
                )
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
    else:
        payload = inspect_workload(args.manifest)
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(" ".join(f"{key}={value}" for key, value in payload.items()))


if __name__ == "__main__":
    main()
