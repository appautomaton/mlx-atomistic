#!/usr/bin/env python3
"""Independent NumPy full-grid evaluator for persisted MLX DFT runtime states."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic._artifact_identity import (
    canonical_json_bytes,
    confined_path,
    generation_root,
    inspect_generation,
)
from mlx_atomistic.benchmarks.dft_runtime_contract import (
    build_source_fingerprints,
    load_workload,
    results_output_path,
)
from mlx_atomistic.benchmarks.dft_runtime_core import (
    FULL_SCF_SCHEMA,
    ORACLE_SCHEMA,
    _finalize_report,
    _formal_admission,
    _publish_report,
    collect_git_provenance,
)


def _load_array(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _state_payload(state: Path, logical_path: object) -> Path:
    if not isinstance(logical_path, str):
        msg = "state payload names must be strings"
        raise ValueError(msg)
    path = confined_path(state, logical_path, must_exist=True)
    if path.is_symlink() or not path.is_file():
        msg = f"state payload must be a regular non-symlink file: {logical_path}"
        raise ValueError(msg)
    return path


def _load_coefficients(state: Path, entry: dict[str, Any], grid_shape: tuple[int, ...]):
    full_name = entry.get("coefficient_file")
    compact_name = entry.get("compact_coefficient_file")
    index_name = entry.get("compact_index_file")
    if full_name is not None:
        if compact_name is not None or index_name is not None:
            msg = "state lane mixes full and compact coefficient encodings"
            raise ValueError(msg)
        full = _load_array(_state_payload(state, full_name))
        expected = (full.shape[0], *grid_shape) if full.ndim >= 1 else ()
        if (
            full.ndim != 4
            or full.shape != expected
            or full.shape[1:] != grid_shape
            or full.dtype != np.complex64
        ):
            msg = "dense coefficients must be complex64 with shape (bands, *grid_shape)"
            raise ValueError(msg)
        return full
    if compact_name is None or index_name is None:
        msg = "state lane has no complete coefficient encoding"
        raise ValueError(msg)
    compact = _load_array(_state_payload(state, compact_name))
    raw_indices = _load_array(_state_payload(state, index_name))
    if not np.issubdtype(raw_indices.dtype, np.integer):
        msg = "compact coefficient indices must use an integer dtype"
        raise ValueError(msg)
    indices = raw_indices.astype(np.int64, copy=False)
    grid_size = int(np.prod(grid_shape))
    if (
        compact.ndim != 2
        or compact.dtype != np.complex64
        or indices.ndim != 1
        or compact.shape[1] != indices.size
    ):
        msg = "compact coefficient and index shapes are inconsistent"
        raise ValueError(msg)
    if (
        np.any(indices < 0)
        or np.any(indices >= grid_size)
        or np.unique(indices).size != indices.size
    ):
        msg = "compact coefficient indices are out of bounds or duplicated"
        raise ValueError(msg)
    full = np.zeros((compact.shape[0], grid_size), dtype=np.complex64)
    full[:, indices] = compact
    return full.reshape((compact.shape[0], *grid_shape))


def _state_lanes(state: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads(_state_payload(state, "metadata.json").read_text())
    if not isinstance(metadata, dict):
        msg = "DFT runtime state metadata must be an object"
        raise ValueError(msg)
    schema = metadata.get("schema_version")
    if schema == "mlx-atomistic.dft-fixed-density-state.v1":
        lane = {
            "weight": 1.0,
            "coefficient_file": "coefficients.npy",
            "eigenvalue_file": "eigenvalues.npy",
        }
        return metadata, [lane]
    if schema == "mlx-atomistic.periodic-scf-state.v1":
        entries = metadata.get("kpoints")
        if not isinstance(entries, list) or any(
            not isinstance(entry, dict) or not isinstance(entry.get("index"), int)
            for entry in entries
        ):
            msg = "periodic SCF state k-point metadata is invalid"
            raise ValueError(msg)
        lanes = [
            {
                **entry,
                "coefficient_file": f"kpoints/{entry['index']:04d}-coefficients.npy",
                "eigenvalue_file": f"kpoints/{entry['index']:04d}-eigenvalues.npy",
            }
            for entry in entries
        ]
        return metadata, lanes
    if schema == "mlx-atomistic.periodic-scf-compact-state.v2":
        entries = metadata.get("owned_lanes")
        if not isinstance(entries, list) or any(
            not isinstance(entry, dict) for entry in entries
        ):
            msg = "compact periodic SCF owned-lane metadata is invalid"
            raise ValueError(msg)
        return metadata, list(entries)
    msg = f"unsupported DFT runtime state schema: {schema!r}"
    raise ValueError(msg)


def _load_lane_state(
    state: Path,
    lane: dict[str, Any],
    grid_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = _load_coefficients(state, lane, grid_shape)
    eigenvalues = _load_array(_state_payload(state, lane["eigenvalue_file"]))
    if (
        eigenvalues.ndim != 1
        or not np.issubdtype(eigenvalues.dtype, np.floating)
        or eigenvalues.shape[0] != coefficients.shape[0]
        or not np.all(np.isfinite(eigenvalues))
    ):
        msg = "state eigenvalues must be a finite real vector matching band count"
        raise ValueError(msg)
    return coefficients, eigenvalues


def _validate_state_contract(
    *,
    generation: dict[str, Any],
    metadata: dict[str, Any],
    lanes: list[dict[str, Any]],
    manifest: dict[str, Any],
    current_sources: dict[str, Any],
) -> dict[str, object]:
    identity = generation.get("identity")
    if not isinstance(identity, dict):
        msg = "state generation identity is missing"
        raise ValueError(msg)
    expected_identity = {
        "workload_fingerprint": manifest["workload_fingerprint"],
        "protocol_fingerprint": current_sources["protocol_fingerprint"],
        "runtime_fingerprint": current_sources["runtime_fingerprint"],
    }
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            msg = f"state generation {field} does not match the oracle contract"
            raise ValueError(msg)
    if (
        generation.get("artifact_kind") != "dft-runtime-full-scf"
        or generation.get("artifact_schema_version") != FULL_SCF_SCHEMA
    ):
        msg = "selected-workload oracle requires a full-SCF state generation"
        raise ValueError(msg)
    manifest_sha256 = generation.get("manifest_sha256")
    if (
        not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in manifest_sha256)
    ):
        msg = "state generation manifest digest is missing or invalid"
        raise ValueError(msg)
    expected_shape = tuple(int(value) for value in manifest["physics"]["fft_shape"])
    observed_shape = tuple(int(value) for value in metadata.get("grid_shape", ()))
    if observed_shape != expected_shape:
        msg = "state grid shape does not match the selected workload"
        raise ValueError(msg)
    schema = metadata.get("schema_version")
    points = manifest["physics"]["kpoints"]
    if schema == "mlx-atomistic.periodic-scf-state.v1":
        if metadata.get("kpoint_count") != len(points) or len(lanes) != len(points):
            msg = "state k-point count does not match the selected workload"
            raise ValueError(msg)
        for expected_index, (lane, point) in enumerate(zip(lanes, points, strict=True)):
            expected_weight = float(point["weight"]["numerator"]) / float(
                point["weight"]["denominator"]
            )
            if (
                lane.get("index") != expected_index
                or tuple(float(value) for value in lane.get("reduced_kpoint", ()))
                != tuple(float(value) for value in point["reduced_coordinates"])
                or not math.isclose(
                    float(lane.get("weight", math.nan)),
                    expected_weight,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or tuple(int(value) for value in lane.get("grid_shape", ()))
                != expected_shape
            ):
                msg = f"state k-point lane {expected_index} does not match the manifest"
                raise ValueError(msg)
    elif schema == "mlx-atomistic.periodic-scf-compact-state.v2":
        owners = [point for point in points if point["role"] == "owner"]
        if len(lanes) != len(owners):
            msg = "compact state representative count does not match the manifest"
            raise ValueError(msg)
        for lane, point in zip(lanes, owners, strict=True):
            owner_index = int(point["owner_index"])
            partner_index = int(point["partner_index"])
            expected_indices = sorted({owner_index, partner_index})
            expected_weight = sum(
                float(points[index]["weight"]["numerator"])
                / float(points[index]["weight"]["denominator"])
                for index in expected_indices
            )
            if (
                lane.get("owner_index") != owner_index
                or sorted(lane.get("explicit_indices", ())) != expected_indices
                or tuple(float(value) for value in lane.get("reduced_kpoint", ()))
                != tuple(float(value) for value in point["reduced_coordinates"])
                or not math.isclose(
                    float(lane.get("aggregate_weight", math.nan)),
                    expected_weight,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                msg = f"compact state owner lane {owner_index} does not match the manifest"
                raise ValueError(msg)
    else:
        msg = "selected-workload oracle requires a periodic SCF state"
        raise ValueError(msg)
    energy = metadata.get("total_energy_hartree")
    state_complete = (
        metadata.get("converged") is True
        and metadata.get("status") == "converged"
        and isinstance(energy, int | float)
        and math.isfinite(float(energy))
    )
    return {
        "artifact_kind": generation["artifact_kind"],
        "artifact_schema_version": generation["artifact_schema_version"],
        "generation_manifest_sha256": manifest_sha256,
        "generation_identity": identity,
        "full_kpoint_contract": True,
        "converged_energy_state": state_complete,
    }


def _reciprocal_components(
    shape: tuple[int, int, int], lengths: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        2.0 * np.pi * np.fft.fftfreq(count, d=float(length) / count)
        for count, length in zip(shape, lengths, strict=True)
    )


def _density_gradient(density: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    components = _reciprocal_components(density.shape, lengths)
    gx, gy, gz = np.meshgrid(*components, indexing="ij")
    density_g = np.fft.fftn(density)
    return np.stack(
        [
            np.fft.ifftn(1j * vector * density_g).real
            for vector in (gx, gy, gz)
        ]
    )


def _pw92_per_particle(rho: np.ndarray) -> np.ndarray:
    a = 0.0310907
    alpha1 = 0.21370
    beta1, beta2, beta3, beta4 = 7.5957, 3.5876, 1.6382, 0.49294
    rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    denominator = 2.0 * a * (
        beta1 * np.sqrt(rs) + beta2 * rs + beta3 * rs**1.5 + beta4 * rs * rs
    )
    return -2.0 * a * (1.0 + alpha1 * rs) * np.log1p(1.0 / denominator)


def _pbe_energy_density(rho: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    rho = np.maximum(rho, 1e-12)
    kappa = 0.804
    mu = 0.2195149727645171
    beta = 0.06672455060314922
    gamma = (1.0 - math.log(2.0)) / (np.pi * np.pi)
    cx = (3.0 / np.pi) ** (1.0 / 3.0)
    dirac = -0.75 * cx * rho ** (4.0 / 3.0)
    kf = (3.0 * np.pi * np.pi * rho) ** (1.0 / 3.0)
    s2 = sigma / (4.0 * kf * kf * rho * rho)
    enhancement = 1.0 + kappa - kappa / (1.0 + mu * s2 / kappa)
    exchange = dirac * enhancement
    ks = np.sqrt(4.0 * kf / np.pi)
    t2 = sigma / (4.0 * ks * ks * rho * rho)
    eps_c = _pw92_per_particle(rho)
    a = (beta / gamma) / np.expm1(-eps_c / gamma)
    at2 = a * t2
    h = gamma * np.log(
        1.0 + (beta / gamma) * t2 * (1.0 + at2) / (1.0 + at2 + at2 * at2)
    )
    return exchange + rho * (eps_c + h)


def _pbe_energy_potential(
    density: np.ndarray, lengths: np.ndarray
) -> tuple[float, np.ndarray]:
    rho = np.maximum(np.asarray(density, dtype=np.float64), 1e-12)
    gradient = _density_gradient(rho, lengths)
    sigma = np.sum(gradient * gradient, axis=0)
    energy_density = _pbe_energy_density(rho, sigma)
    rho_step = np.maximum(np.abs(rho) * 2e-6, 1e-9)
    sigma_step = np.maximum(np.abs(sigma) * 2e-6, 1e-12)
    f_rho = (
        _pbe_energy_density(rho + rho_step, sigma)
        - _pbe_energy_density(np.maximum(rho - rho_step, 1e-12), sigma)
    ) / (rho + rho_step - np.maximum(rho - rho_step, 1e-12))
    sigma_low = np.maximum(sigma - sigma_step, 0.0)
    f_sigma = (
        _pbe_energy_density(rho, sigma + sigma_step)
        - _pbe_energy_density(rho, sigma_low)
    ) / (sigma + sigma_step - sigma_low)
    flux = 2.0 * f_sigma[None, ...] * gradient
    components = _reciprocal_components(rho.shape, lengths)
    gx, gy, gz = np.meshgrid(*components, indexing="ij")
    divergence = np.zeros_like(rho)
    for vector, field in zip((gx, gy, gz), flux, strict=True):
        divergence += np.fft.ifftn(1j * vector * np.fft.fftn(field)).real
    dv = float(np.prod(lengths) / np.prod(rho.shape))
    return float(np.sum(energy_density) * dv), f_rho - divergence


def _hartree(density: np.ndarray, lengths: np.ndarray) -> tuple[float, np.ndarray]:
    components = _reciprocal_components(density.shape, lengths)
    gx, gy, gz = np.meshgrid(*components, indexing="ij")
    g2 = gx * gx + gy * gy + gz * gz
    density_g = np.fft.fftn(density)
    potential_g = np.zeros_like(density_g)
    nonzero = g2 > 0.0
    potential_g[nonzero] = 4.0 * np.pi * density_g[nonzero] / g2[nonzero]
    potential = np.fft.ifftn(potential_g).real
    dv = float(np.prod(lengths) / np.prod(density.shape))
    return 0.5 * float(np.sum(density * potential) * dv), potential


def _ewald(charges: np.ndarray, positions: np.ndarray, lengths: np.ndarray) -> float:
    from math import erfc

    eta = 5.0 / float(np.min(lengths))
    tolerance = 1e-10
    cutoff_factor = math.sqrt(-math.log(tolerance))
    real_cutoff = cutoff_factor / eta
    ranges = [
        range(-int(np.ceil(real_cutoff / length)) - 1, int(np.ceil(real_cutoff / length)) + 2)
        for length in lengths
    ]
    real_energy = 0.0
    for first_index, first in enumerate(positions):
        for second_index, second in enumerate(positions):
            for image in np.ndindex(*(len(values) for values in ranges)):
                translation = np.array(
                    [ranges[axis][image[axis]] * lengths[axis] for axis in range(3)]
                )
                displacement = first - second + translation
                distance = float(np.linalg.norm(displacement))
                if 1e-14 < distance <= real_cutoff:
                    real_energy += (
                        charges[first_index]
                        * charges[second_index]
                        * erfc(eta * distance)
                        / distance
                    )
    real_energy *= 0.5
    reciprocal_cutoff = 2.0 * eta * cutoff_factor
    maximum = np.ceil(reciprocal_cutoff * lengths / (2.0 * np.pi)).astype(int)
    reciprocal = 0.0
    for h in range(-int(maximum[0]), int(maximum[0]) + 1):
        for k in range(-int(maximum[1]), int(maximum[1]) + 1):
            for ell in range(-int(maximum[2]), int(maximum[2]) + 1):
                if h == k == ell == 0:
                    continue
                vector = 2.0 * np.pi * np.array([h, k, ell]) / lengths
                g2 = float(np.dot(vector, vector))
                if math.sqrt(g2) > reciprocal_cutoff:
                    continue
                structure = np.sum(charges * np.exp(-1j * (positions @ vector)))
                reciprocal += np.exp(-g2 / (4.0 * eta * eta)) * abs(structure) ** 2 / g2
    volume = float(np.prod(lengths))
    reciprocal *= 2.0 * np.pi / volume
    self_energy = -eta / math.sqrt(np.pi) * float(np.sum(charges * charges))
    total_charge = float(np.sum(charges))
    background = -np.pi * total_charge**2 / (2.0 * eta * eta * volume)
    return float(real_energy + reciprocal + self_energy + background)


def evaluate_state(
    *,
    manifest_path: str | Path,
    gth_source: str | Path,
    state_path: str | Path,
    current_sources: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Recompute density, electron count, orthonormality, and DFT energy terms."""

    manifest, _selected = load_workload(manifest_path, gth_source=gth_source)
    requested_state = Path(state_path).expanduser()
    if requested_state.is_symlink():
        msg = "oracle state directory may not be a symlink"
        raise ValueError(msg)
    state = requested_state.resolve()
    root = generation_root(state)
    generation = inspect_generation(root)
    if not state.is_relative_to(root) or not state.is_dir():
        msg = "oracle state directory must be inside its completed generation"
        raise ValueError(msg)
    metadata, lanes = _state_lanes(state)
    sources = build_source_fingerprints() if current_sources is None else current_sources
    state_contract = _validate_state_contract(
        generation=generation,
        metadata=metadata,
        lanes=lanes,
        manifest=manifest,
        current_sources=sources,
    )
    shape = tuple(int(value) for value in metadata["grid_shape"])
    if len(shape) != 3 or any(value <= 0 for value in shape):
        msg = "oracle state grid shape must contain three positive dimensions"
        raise ValueError(msg)
    lattice = float(manifest["system"]["lattice_constant_bohr"])
    lengths = np.array([lattice, lattice, lattice], dtype=np.float64)
    volume = float(np.prod(lengths))
    density = np.zeros(shape, dtype=np.float64)
    band_energy = 0.0
    maximum_orthonormality = 0.0
    total_weight = 0.0
    expected_band_count = int(manifest["system"]["occupied_band_count"])
    for lane in lanes:
        coefficients, eigenvalues = _load_lane_state(state, lane, shape)
        if coefficients.shape[0] != expected_band_count:
            msg = "state lane band count does not match the selected workload"
            raise ValueError(msg)
        weight = float(lane.get("aggregate_weight", lane.get("weight", 1.0)))
        if not math.isfinite(weight) or weight <= 0.0:
            msg = "state lane weights must be finite and positive"
            raise ValueError(msg)
        total_weight += weight
        orbitals = np.fft.ifftn(coefficients, axes=(-3, -2, -1)) * np.prod(shape) / math.sqrt(
            volume
        )
        density += weight * 2.0 * np.sum(np.abs(orbitals) ** 2, axis=0)
        band_energy += weight * 2.0 * float(np.sum(eigenvalues))
        flat = coefficients.reshape((coefficients.shape[0], -1))
        overlap = flat @ flat.conj().T
        maximum_orthonormality = max(
            maximum_orthonormality,
            float(np.max(np.abs(overlap - np.eye(overlap.shape[0])))),
        )
    dv = volume / float(np.prod(shape))
    electron_count = float(np.sum(density) * dv)
    hartree_energy, hartree_potential = _hartree(density, lengths)
    xc_energy, xc_potential = _pbe_energy_potential(density, lengths)
    fractional = np.asarray(manifest["system"]["fractional_positions"], dtype=np.float64)
    positions = fractional * lattice
    charges = np.full(len(positions), 4.0, dtype=np.float64)
    ewald_energy = _ewald(charges, positions, lengths)
    density_xc = float(np.sum(density * xc_potential) * dv)
    total_energy = band_energy - hartree_energy + xc_energy - density_xc + ewald_energy
    expected_electrons = float(manifest["system"]["electron_count"])
    stored_total_energy = metadata.get("total_energy_hartree")
    gates = {
        "state_contract": state_contract["converged_energy_state"] is True,
        "kpoint_weight_sum": math.isclose(
            total_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "electron_count": abs(electron_count - expected_electrons)
        <= float(manifest["numerical_gates"]["electron_count_abs_per_cell"]),
        "orthonormality": maximum_orthonormality
        <= float(manifest["numerical_gates"]["orthonormality_max"]),
        "finite": bool(
            np.isfinite(total_energy)
            and np.all(np.isfinite(density))
            and np.all(np.isfinite(hartree_potential))
        ),
        "total_energy": isinstance(stored_total_energy, int | float)
        and math.isfinite(float(stored_total_energy))
        and abs(total_energy - float(stored_total_energy))
        <= (
            float(manifest["numerical_gates"]["energy_abs_hartree_per_atom"])
            * int(manifest["system"]["atom_count"])
        ),
    }
    return {
        "schema_version": ORACLE_SCHEMA,
        "workload_fingerprint": manifest["workload_fingerprint"],
        "state_schema": metadata["schema_version"],
        "state_artifact": state_contract,
        "electron_count": electron_count,
        "maximum_orthonormality_error": maximum_orthonormality,
        "energy_by_term_hartree": {
            "band": band_energy,
            "hartree": hartree_energy,
            "xc": xc_energy,
            "density_xc_potential": density_xc,
            "ion_ewald": ewald_energy,
            "total": total_energy,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def main(argv: list[str] | None = None) -> None:
    """Run the independent full-grid state evaluator."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gth-source", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-gates", action="store_true")
    args = parser.parse_args(argv)
    destination = results_output_path(args.out)
    sources: dict[str, Any] = {
        "protocol_fingerprint": None,
        "runtime_fingerprint": None,
    }
    producer_git: dict[str, object]
    try:
        producer_git = collect_git_provenance()
    except Exception:
        producer_git = {"revision": None, "parent": None, "dirty": True}
    execution_failed = False
    try:
        sources = build_source_fingerprints()
        report = evaluate_state(
            manifest_path=args.manifest,
            gth_source=args.gth_source,
            state_path=args.state,
            current_sources=sources,
        )
    except Exception as error:
        execution_failed = True
        report = {
            "schema_version": ORACLE_SCHEMA,
            "workload_fingerprint": None,
            "failure": {
                "error_type": type(error).__name__,
                "message": str(error),
            },
            "passed": False,
            "oracle_blockers": ["oracle_execution_failed"],
        }
    identity = {
        "workload_fingerprint": report.get("workload_fingerprint"),
        "protocol_fingerprint": sources["protocol_fingerprint"],
        "runtime_fingerprint": sources["runtime_fingerprint"],
    }
    passed = report.get("passed") is True
    blockers = list(report.pop("oracle_blockers", ()))
    if not passed and not blockers:
        blockers.append("oracle_gate_failed")
    statuses = {
        "numerical_status": "passed" if passed else "blocked",
        "resume_integrity_status": "fresh-no-resume",
        "timing_admission_status": "admitted" if passed else "blocked",
    }
    admission = {"passed": passed, "blockers": blockers}
    run_protocol = {
        "fresh": True,
        "resumed": False,
        "diagnostic": False,
        "state_generation_manifest_sha256": (
            report.get("state_artifact", {}).get("generation_manifest_sha256")
            if isinstance(report.get("state_artifact"), dict)
            else None
        ),
    }
    report = _finalize_report(
        {
            **report,
            "schema_version": ORACLE_SCHEMA,
            "kind": "oracle",
            "identity": identity,
            "context": {"git": producer_git},
            "run_protocol": run_protocol,
            "statuses": statuses,
            "admission": admission,
            "formal_admission": _formal_admission(
                statuses=statuses,
                command_admission=admission,
                producer_git=producer_git,
                run_protocol=run_protocol,
                report_kind="oracle",
                host_protocol=None,
            ),
        }
    )
    published = _publish_report(
        out=destination,
        artifact_kind="dft-runtime-oracle",
        artifact_schema=ORACLE_SCHEMA,
        report=report,
    )
    summary = {"artifact": published["artifact"], "passed": passed}
    sys.stdout.buffer.write(canonical_json_bytes(summary) + b"\n")
    sys.stdout.flush()
    if execution_failed or (args.require_gates and not passed):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
