"""Frozen workload, source-identity, and read-only host contract for DFT timing."""

from __future__ import annotations

import json
import platform
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from mlx_atomistic._artifact_identity import (
    GENERATION_MANIFEST,
    ArtifactIntegrityError,
    AtomicGeneration,
    canonical_json_bytes,
    confined_path,
    inspect_generation,
    inventory_fingerprint,
    resource_record,
    sha256_bytes,
    source_inventory,
)

WORKLOAD_SCHEMA = "mlx-atomistic.dft-runtime-workload.v1"
WORKLOAD_FINGERPRINT_SCHEMA = "mlx-atomistic.dft-runtime-workload-fingerprint.v1"
TARGET_ID = "silicon-8atom-pbe-gth-q4-25ha-56x56x56-6x6x6"
GTH_ELEMENT = "Si"
GTH_NAME = "GTH-PBE-q4"
ANGSTROM_TO_BOHR = 1.8897261254578281
TARGET_CHIP = "Apple M5 Max"

PROTOCOL_SOURCE_PATHS = (
    "src/mlx_atomistic/_artifact_identity.py",
    "src/mlx_atomistic/benchmarks/dft_runtime.py",
    "src/mlx_atomistic/benchmarks/dft_runtime_contract.py",
    "src/mlx_atomistic/benchmarks/dft_runtime_core.py",
    "src/mlx_atomistic/dft/_runtime_observer.py",
    "scripts/run_dft_runtime_oracle.py",
)
RUNTIME_SOURCE_PATHS = (
    "src/mlx_atomistic/__init__.py",
    "src/mlx_atomistic/_artifact_identity.py",
    "src/mlx_atomistic/core.py",
    "src/mlx_atomistic/runtime.py",
)
RUNTIME_SOURCE_ROOTS = ("src/mlx_atomistic/dft",)

READ_ONLY_HOST_COMMANDS = (
    ("system_profiler", "SPHardwareDataType"),
    ("sw_vers",),
    ("pmset", "-g", "batt"),
    ("pmset", "-g", "custom"),
    ("sysctl", "-n", "kern.thermal_pressure"),
)
POWER_MODE_KEYS = ("lowpowermode", "powermode")

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


def _normalized_gth_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", maxsplit=1)[0].strip()
        if line:
            lines.append(" ".join(line.split()))
    return lines


def extract_selected_gth(path: str | Path) -> bytes:
    """Extract canonical bytes for the selected silicon GTH entry."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    lines = _normalized_gth_lines(source)
    matches = [
        index
        for index, line in enumerate(lines)
        if (parts := line.split()) and parts[0] == GTH_ELEMENT and GTH_NAME in parts[1:]
    ]
    if not matches:
        msg = f"GTH entry {GTH_ELEMENT} {GTH_NAME} was not found"
        raise ValueError(msg)
    if len(matches) != 1:
        msg = f"GTH entry {GTH_ELEMENT} {GTH_NAME} is ambiguous"
        raise ValueError(msg)
    header_index = matches[0]
    cursor = header_index + 1
    try:
        charge_shells = tuple(int(value) for value in lines[cursor].split())
        cursor += 1
        local = lines[cursor].split()
        cursor += 1
        local_count = int(local[1])
        if len(local[2:]) != local_count:
            msg = "selected GTH local coefficient count is inconsistent"
            raise ValueError(msg)
        channel_count = int(lines[cursor].split()[0])
        cursor += 1
        for _channel in range(channel_count):
            first = lines[cursor].split()
            cursor += 1
            projector_count = int(first[1])
            if projector_count <= 0 or len(first[2:]) != projector_count:
                msg = "selected GTH projector declaration is inconsistent"
                raise ValueError(msg)
            for row_index in range(1, projector_count):
                row = lines[cursor].split()
                cursor += 1
                if len(row) != projector_count - row_index:
                    msg = "selected GTH coupling matrix is incomplete"
                    raise ValueError(msg)
    except (IndexError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("selected GTH"):
            raise
        msg = "selected GTH entry is structurally incomplete"
        raise ValueError(msg) from error
    if sum(charge_shells) != 4 or channel_count != 2:
        msg = "selected silicon GTH entry must be q4 with s and p channels"
        raise ValueError(msg)
    return ("\n".join(lines[header_index:cursor]) + "\n").encode("utf-8")


def _kpoint_manifest(size: int = 6) -> list[dict[str, object]]:
    total = size**3
    denominator = 2 * size
    points: list[dict[str, object]] = []
    for flat_index in range(total):
        i = flat_index // (size * size)
        j = (flat_index // size) % size
        k = flat_index % size
        indices = (i, j, k)
        numerators = tuple(2 * index - (size - 1) for index in indices)
        partner_indices = tuple(size - 1 - index for index in indices)
        partner = (
            partner_indices[0] * size * size + partner_indices[1] * size + partner_indices[2]
        )
        owner = min(flat_index, partner)
        points.append(
            {
                "index": flat_index,
                "mesh_indices": list(indices),
                "reduced_numerators": list(numerators),
                "reduced_denominator": denominator,
                "reduced_coordinates": [value / denominator for value in numerators],
                "weight": {"numerator": 1, "denominator": total},
                "partner_index": partner,
                "owner_index": owner,
                "role": "owner" if flat_index == owner else "partner",
            }
        )
    return points


def _unsigned_workload(gth_bytes: bytes) -> dict[str, object]:
    lattice_angstrom = 5.43
    kpoints = _kpoint_manifest(6)
    return {
        "schema_version": WORKLOAD_SCHEMA,
        "target_id": TARGET_ID,
        "resources": [resource_record("si_gth_pbe_q4", gth_bytes)],
        "system": {
            "name": "diamond-silicon-conventional-cubic",
            "lattice_constant_angstrom": lattice_angstrom,
            "lattice_constant_bohr": lattice_angstrom * ANGSTROM_TO_BOHR,
            "atom_count": 8,
            "symbols": [GTH_ELEMENT] * 8,
            "fractional_positions": [list(row) for row in _SILICON_FRACTIONAL_POSITIONS],
            "electron_count": 32,
            "spin_mode": "unpolarized",
            "occupancy_per_band": 2,
            "occupied_band_count": 16,
        },
        "physics": {
            "exchange_correlation": "PBE-PW92",
            "pseudopotential": GTH_NAME,
            "kinetic_cutoff_hartree": 25.0,
            "fft_shape": [56, 56, 56],
            "kpoint_mesh": [6, 6, 6],
            "kpoint_centering": "monkhorst-pack-even-half-shift",
            "kpoints": kpoints,
            "representative_count": sum(point["role"] == "owner" for point in kpoints),
            "fixed_density_lane_index": 0,
        },
        "solver": {
            "scf": {
                "max_iterations": 80,
                "min_iterations": 2,
                "density_tolerance": 1e-6,
                "energy_tolerance_hartree": 8e-6,
                "orbital_tolerance": 1e-6,
                "mixing_beta": 0.35,
                "mixer": "diis",
            },
            "davidson": {
                "max_iterations": 48,
                "tolerance": 1e-6,
                "max_subspace_size": 64,
                "preconditioner_floor": 0.5,
            },
        },
        "initialization": {
            "density": "uniform-electron-count-over-cell-volume",
            "orbitals": "lowest-kinetic-active-plane-waves",
            "random_seed": None,
        },
        "measurement": {
            "warmups": 1,
            "samples": 5,
            "synchronization": "mx.synchronize-before-after-named-phases",
            "target_chip": TARGET_CHIP,
            "required_lowpowermode": 1,
            "power_source_policy": "record-and-match-ac-or-battery",
        },
        "engineering_ladder": [
            {
                "fft_shape": [8, 8, 8],
                "cutoff_hartree": 2.0,
                "kpoint_mesh": [1, 1, 1],
                "reduced_kpoint": [0.0, 0.0, 0.0],
                "band_count": 16,
                "oracle": {
                    "kind": "fixed-density-residual-and-orthonormality",
                    "maximum_residual": 1e-6,
                    "maximum_orthonormality_error": 1e-4,
                },
                "max_elapsed_seconds": 15.0,
                "max_process_bytes": 2_000_000_000,
            },
            {
                "fft_shape": [32, 32, 32],
                "cutoff_hartree": 10.0,
                "kpoint_mesh": [2, 2, 2],
                "reduced_kpoint": [-0.25, -0.25, -0.25],
                "band_count": 16,
                "oracle": {
                    "kind": "fixed-density-residual-and-orthonormality",
                    "maximum_residual": 1e-6,
                    "maximum_orthonormality_error": 1e-4,
                },
                "max_elapsed_seconds": 90.0,
                "max_process_bytes": 8_000_000_000,
            },
            {
                "fft_shape": [48, 48, 48],
                "cutoff_hartree": 20.0,
                "kpoint_mesh": [4, 4, 4],
                "reduced_kpoint": [-0.375, -0.375, -0.375],
                "band_count": 16,
                "oracle": {
                    "kind": "fixed-density-residual-and-orthonormality",
                    "maximum_residual": 1e-6,
                    "maximum_orthonormality_error": 1e-4,
                },
                "max_elapsed_seconds": 240.0,
                "max_process_bytes": 16_000_000_000,
            },
            {
                "fft_shape": [56, 56, 56],
                "cutoff_hartree": 25.0,
                "kpoint_mesh": [6, 6, 6],
                "reduced_kpoint": [-5.0 / 12.0, -5.0 / 12.0, -5.0 / 12.0],
                "band_count": 16,
                "oracle": {
                    "kind": "fixed-density-residual-and-orthonormality",
                    "maximum_residual": 1e-6,
                    "maximum_orthonormality_error": 1e-4,
                },
                "max_elapsed_seconds": 300.0,
                "max_process_bytes": 32_000_000_000,
            },
        ],
        "numerical_gates": {
            "fixed_density_eigenvalue_abs_hartree": 1e-5,
            "energy_abs_hartree_per_atom": 1e-5,
            "electron_count_abs_per_cell": 1e-4,
            "orthonormality_max": 1e-4,
        },
    }


def workload_fingerprint(unsigned_manifest: Mapping[str, object]) -> str:
    """Return the canonical workload fingerprint."""

    envelope = {
        "schema_version": WORKLOAD_FINGERPRINT_SCHEMA,
        "manifest": dict(unsigned_manifest),
    }
    return sha256_bytes(canonical_json_bytes(envelope))


def build_workload_manifest(gth_bytes: bytes) -> dict[str, object]:
    """Build the path-independent selected silicon workload manifest."""

    unsigned = _unsigned_workload(gth_bytes)
    manifest = dict(unsigned)
    manifest["workload_fingerprint"] = workload_fingerprint(unsigned)
    return manifest


def prepare_workload(
    *,
    gth_source: str | Path,
    out: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Publish the immutable workload and selected GTH resource."""

    selected = extract_selected_gth(gth_source)
    manifest = build_workload_manifest(selected)
    sources = build_source_fingerprints(repo_root)
    identity = {
        "workload_fingerprint": manifest["workload_fingerprint"],
        "protocol_fingerprint": sources["protocol_fingerprint"],
        "runtime_fingerprint": sources["runtime_fingerprint"],
    }
    destination = Path(out)
    with AtomicGeneration(
        destination=destination,
        artifact_kind="dft-runtime-workload",
        artifact_schema_version=WORKLOAD_SCHEMA,
        identity=identity,
    ) as generation:
        generation.write_json("manifest.json", manifest)
        generation.write_bytes("resources/Si-GTH-PBE-q4.gth", selected)
        generation.publish()
    return {
        "status": "prepared",
        "artifact": str(destination),
        "manifest": str(destination / "manifest.json"),
        "workload_fingerprint": manifest["workload_fingerprint"],
        "protocol_fingerprint": sources["protocol_fingerprint"],
        "runtime_fingerprint": sources["runtime_fingerprint"],
    }


def load_workload(
    manifest_path: str | Path,
    *,
    gth_source: str | Path,
) -> tuple[dict[str, object], bytes]:
    """Validate a workload manifest and caller-resolved GTH source."""

    path = Path(manifest_path)
    generation_manifest = path.parent / GENERATION_MANIFEST
    if not generation_manifest.exists() and not generation_manifest.is_symlink():
        msg = "DFT runtime workload requires a completed generation"
        raise ArtifactIntegrityError(msg)
    generation = inspect_generation(path)
    try:
        manifest = json.loads(path.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        msg = "DFT runtime workload is not valid JSON"
        raise ValueError(msg) from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != WORKLOAD_SCHEMA:
        msg = "unsupported DFT runtime workload schema"
        raise ValueError(msg)
    if manifest.get("target_id") != TARGET_ID:
        msg = "unexpected DFT runtime workload target"
        raise ValueError(msg)
    declared = manifest.get("workload_fingerprint")
    unsigned = {key: value for key, value in manifest.items() if key != "workload_fingerprint"}
    if declared != workload_fingerprint(unsigned):
        msg = "DFT runtime workload fingerprint mismatch"
        raise ValueError(msg)
    identity = generation.get("identity")
    if (
        generation.get("artifact_kind") != "dft-runtime-workload"
        or generation.get("artifact_schema_version") != WORKLOAD_SCHEMA
        or not isinstance(identity, dict)
        or identity.get("workload_fingerprint") != declared
    ):
        msg = "DFT runtime workload does not match its generation envelope"
        raise ValueError(msg)
    selected = extract_selected_gth(gth_source)
    expected_resource = resource_record("si_gth_pbe_q4", selected)
    if manifest.get("resources") != [expected_resource]:
        msg = "selected GTH resource does not match the workload"
        raise ValueError(msg)
    embedded_resource = confined_path(
        path.parent,
        "resources/Si-GTH-PBE-q4.gth",
        must_exist=True,
    )
    if (
        embedded_resource.is_symlink()
        or not embedded_resource.is_file()
        or embedded_resource.read_bytes() != selected
    ):
        msg = "published GTH resource does not match the workload"
        raise ValueError(msg)
    _validate_workload_invariants(manifest)
    return manifest, selected


def _validate_workload_invariants(manifest: Mapping[str, object]) -> None:
    if (
        manifest.get("schema_version") != WORKLOAD_SCHEMA
        or manifest.get("target_id") != TARGET_ID
    ):
        msg = "unsupported DFT runtime workload schema or target"
        raise ValueError(msg)
    system = manifest.get("system")
    physics = manifest.get("physics")
    solver = manifest.get("solver")
    if (
        not isinstance(system, dict)
        or not isinstance(physics, dict)
        or not isinstance(solver, dict)
    ):
        msg = "DFT runtime workload sections are missing"
        raise ValueError(msg)
    if (
        system.get("atom_count") != 8
        or system.get("electron_count") != 32
        or system.get("occupied_band_count") != 16
        or physics.get("kinetic_cutoff_hartree") != 25.0
        or physics.get("fft_shape") != [56, 56, 56]
        or physics.get("kpoint_mesh") != [6, 6, 6]
    ):
        msg = "DFT runtime workload physics drifted from the frozen target"
        raise ValueError(msg)
    scf = solver.get("scf")
    davidson = solver.get("davidson")
    if (
        not isinstance(scf, dict)
        or not isinstance(davidson, dict)
        or scf.get("density_tolerance") != 1e-6
        or scf.get("energy_tolerance_hartree") != 8e-6
        or scf.get("orbital_tolerance") != 1e-6
        or davidson.get("max_iterations") != 48
        or davidson.get("tolerance") != 1e-6
    ):
        msg = "DFT runtime solver contract drifted from the frozen target"
        raise ValueError(msg)
    points = physics.get("kpoints")
    if not isinstance(points, list) or points != _kpoint_manifest(6):
        msg = "DFT runtime workload k-point ownership map is invalid"
        raise ValueError(msg)


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the checkout root needed for path-independent source inventories."""

    current = Path.cwd() if start is None else Path(start)
    current = current.expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src/mlx_atomistic/dft"
        ).is_dir():
            return candidate
    msg = f"mlx-atomistic repository root not found from {current}"
    raise FileNotFoundError(msg)


def results_output_path(
    output: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """Resolve a caller output below the repository results directory.

    Args:
        output: Relative or absolute requested generation path.
        repo_root: Optional repository root override for focused tests.

    Returns:
        Absolute absent-or-existing path below the repository results directory.

    Raises:
        ValueError: If the output is the results root or escapes it.
    """

    root = find_repo_root(repo_root)
    results_root = (root / "results").resolve(strict=False)
    requested = Path(output).expanduser()
    candidate = (
        requested.resolve(strict=False)
        if requested.is_absolute()
        else (root / requested).resolve(strict=False)
    )
    if candidate != results_root and candidate.is_relative_to(results_root):
        return candidate
    if requested.is_absolute():
        for ancestor in candidate.parents:
            checkout = ancestor.parent
            if (
                ancestor.name == "results"
                and candidate != ancestor
                and (checkout / "pyproject.toml").is_file()
                and (checkout / "src/mlx_atomistic/dft").is_dir()
            ):
                return candidate
    msg = f"DFT runtime outputs must be generation paths below {results_root}"
    raise ValueError(msg)


def build_source_fingerprints(
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Build frozen protocol and complete periodic-runtime source identities."""

    root = find_repo_root(repo_root)
    protocol_files = source_inventory(root, logical_paths=PROTOCOL_SOURCE_PATHS)
    runtime_files = source_inventory(
        root,
        logical_paths=RUNTIME_SOURCE_PATHS,
        recursive_roots=RUNTIME_SOURCE_ROOTS,
    )
    return {
        "protocol_inventory": protocol_files,
        "protocol_fingerprint": inventory_fingerprint("dft-runtime-protocol", protocol_files),
        "runtime_inventory": runtime_files,
        "runtime_fingerprint": inventory_fingerprint("periodic-dft-runtime", runtime_files),
    }


def parse_current_power_source(output: str) -> str:
    """Parse the normalized current source from ``pmset -g batt`` output."""

    match = re.search(r"Now drawing from '([^']+)'", output)
    if match is None or match.group(1) not in {"AC Power", "Battery Power"}:
        msg = "current power source could not be parsed"
        raise ValueError(msg)
    return match.group(1)


def parse_power_profiles(output: str) -> dict[str, dict[str, int]]:
    """Parse named integer settings from ``pmset -g custom`` output."""

    profiles: dict[str, dict[str, int]] = {}
    current: str | None = None
    for raw in output.splitlines():
        stripped = raw.strip()
        if stripped.endswith(":") and stripped[:-1] in {"AC Power", "Battery Power"}:
            current = stripped[:-1]
            profiles.setdefault(current, {})
            continue
        if not stripped or current is None:
            continue
        parts = stripped.split()
        if len(parts) != 2:
            continue
        key, raw_value = parts
        try:
            value = int(raw_value)
        except ValueError:
            if key in POWER_MODE_KEYS:
                msg = f"active power profile has non-integer {key}"
                raise ValueError(msg) from None
            continue
        if key in profiles[current]:
            msg = f"power profile contains duplicate key {key}"
            raise ValueError(msg)
        profiles[current][key] = value
    return profiles


def _active_power_mode(profile: Mapping[str, object]) -> tuple[str | None, int | None]:
    observed = [
        (key, profile[key])
        for key in POWER_MODE_KEYS
        if key in profile
    ]
    if not observed:
        return None, None
    if any(type(value) is not int for _key, value in observed):
        msg = "active power profile has a non-integer power-mode value"
        raise ValueError(msg)
    values = {value for _key, value in observed}
    if len(values) != 1:
        msg = "active power profile has conflicting power-mode keys"
        raise ValueError(msg)
    return "+".join(key for key, _value in observed), int(observed[0][1])


HostRunner = Callable[[Sequence[str]], Mapping[str, object]]


def _run_host_command(command: Sequence[str]) -> dict[str, object]:
    requested = tuple(command)
    if requested not in READ_ONLY_HOST_COMMANDS:
        msg = f"host command is not in the read-only allowlist: {requested!r}"
        raise ValueError(msg)
    try:
        completed = subprocess.run(
            requested,
            capture_output=True,
            check=False,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "blocked", "error": str(error), "command": list(requested)}
    return {
        "status": "ok" if completed.returncode == 0 else "blocked",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "command": list(requested),
    }


def _query(runner: HostRunner, command: Sequence[str]) -> Mapping[str, object]:
    requested = tuple(command)
    if requested not in READ_ONLY_HOST_COMMANDS:
        msg = f"host command is not in the read-only allowlist: {requested!r}"
        raise ValueError(msg)
    return runner(requested)


def collect_host_provenance(runner: HostRunner | None = None) -> dict[str, object]:
    """Collect normalized read-only host and active-power provenance."""

    execute = _run_host_command if runner is None else runner
    results = {command: _query(execute, command) for command in READ_ONLY_HOST_COMMANDS}
    hardware = results[("system_profiler", "SPHardwareDataType")]
    battery = results[("pmset", "-g", "batt")]
    custom = results[("pmset", "-g", "custom")]
    operating_system = results[("sw_vers",)]
    thermal = results[("sysctl", "-n", "kern.thermal_pressure")]
    blockers: list[str] = []

    def query_stdout(result: Mapping[str, object], blocker: str) -> str:
        if result.get("status") != "ok":
            blockers.append(blocker)
            return ""
        return str(result.get("stdout", ""))

    hardware_text = query_stdout(hardware, "hardware_query_failed")
    battery_text = query_stdout(battery, "power_source_query_failed")
    custom_text = query_stdout(custom, "power_profiles_query_failed")
    os_text = query_stdout(operating_system, "os_query_failed")
    chip = _hardware_value(hardware_text, "Chip")
    model = _hardware_value(hardware_text, "Model Name")
    model_identifier = _hardware_value(hardware_text, "Model Identifier")
    memory = _hardware_value(hardware_text, "Memory")
    try:
        power_source = parse_current_power_source(battery_text)
    except ValueError:
        power_source = None
        if battery_text:
            blockers.append("power_source_unparsed")
    try:
        profiles = parse_power_profiles(custom_text)
    except ValueError:
        profiles = {}
        if custom_text:
            blockers.append("power_profiles_unparsed")
    active_profile = profiles.get(power_source, {}) if power_source is not None else {}
    try:
        power_mode_key, low_power_mode = _active_power_mode(active_profile)
    except ValueError:
        power_mode_key = None
        low_power_mode = None
        blockers.append("active_power_mode_conflict")
    return {
        "model": model,
        "model_identifier": model_identifier,
        "chip": chip,
        "memory": memory,
        "machine": platform.machine(),
        "macos": _parse_sw_vers(os_text),
        "power_source": power_source,
        "active_power_profile": dict(active_profile),
        "power_mode_key": power_mode_key,
        "low_power_mode": low_power_mode,
        "thermal_pressure": (
            str(thermal.get("stdout", "")).strip() if thermal.get("status") == "ok" else None
        ),
        "query_status": {
            "hardware": hardware.get("status"),
            "operating_system": operating_system.get("status"),
            "power_source": battery.get("status"),
            "power_profiles": custom.get("status"),
            "thermal_pressure": thermal.get("status"),
        },
        "blockers": sorted(set(blockers)),
        "inspection_policy": "read-only-getters-only",
    }


def _hardware_value(output: str, label: str) -> str | None:
    prefix = f"{label}:"
    return next(
        (
            line.split(":", maxsplit=1)[1].strip()
            for line in output.splitlines()
            if line.strip().startswith(prefix)
        ),
        None,
    )


def _parse_sw_vers(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        if ":" in line:
            key, value = line.split(":", maxsplit=1)
            parsed[key.strip()] = value.strip()
    return parsed


def host_admission(
    provenance: Mapping[str, object],
    *,
    required_chip: str | None = None,
    require_low_power: bool = False,
) -> dict[str, object]:
    """Evaluate fail-closed chip and active Low Power requirements."""

    blockers = list(provenance.get("blockers", []))
    if required_chip is not None and provenance.get("chip") != required_chip:
        blockers.append("chip_mismatch")
    if provenance.get("power_source") not in {"AC Power", "Battery Power"}:
        blockers.append("power_source_missing")
    if require_low_power:
        active = provenance.get("active_power_profile")
        if not isinstance(active, dict):
            blockers.append("active_power_mode_missing")
        else:
            try:
                _key, value = _active_power_mode(active)
            except ValueError:
                blockers.append("active_power_mode_conflict")
            else:
                if value is None:
                    blockers.append("active_power_mode_missing")
                elif value != 1:
                    blockers.append("active_power_mode_not_one")
                elif (
                    type(provenance.get("low_power_mode")) is not int
                    or provenance.get("low_power_mode") != value
                    or type(provenance.get("power_mode_key")) is not str
                    or provenance.get("power_mode_key") != _key
                ):
                    blockers.append("active_power_mode_normalization_mismatch")
    unique = sorted(set(str(blocker) for blocker in blockers))
    return {"admitted": not unique, "blockers": unique}
