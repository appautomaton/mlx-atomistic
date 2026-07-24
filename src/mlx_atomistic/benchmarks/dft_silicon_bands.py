"""Frozen-density silicon band-structure benchmark and scientific validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.dft import (
    BandPath,
    KPoint,
    PeriodicDavidsonConfig,
    PeriodicDFTSystem,
    PeriodicFrozenDensity,
    fold_band_path_to_supercell,
    read_gth,
    run_periodic_band_structure,
    unfold_periodic_band_structure,
)

REFERENCE_SCHEMA = "mlx-atomistic.silicon-band-references.v1"
REFERENCE_SHA256 = "3cd9dd8a11b695ef6f05866f451a6c8320025b1b5f4b5638b58c1a0615033ffe"
REPORT_SCHEMA = "mlx-atomistic.silicon-band-report.v1"
HARTREE_TO_EV = 27.211386245988

_FULL_LABELS = ("GAMMA", "X", "W", "K", "GAMMA", "L")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _reference_path() -> Path:
    return Path(__file__).with_name("data") / "silicon_band_references.json"


def load_silicon_band_references() -> dict[str, Any]:
    """Load the pinned, source-attributed silicon band reference bundle."""

    path = _reference_path()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        msg = "silicon band reference bundle hash mismatch"
        raise ValueError(msg)
    payload = json.loads(raw)
    if payload.get("schema_version") != REFERENCE_SCHEMA:
        msg = "unsupported silicon band reference schema"
        raise ValueError(msg)
    material = payload.get("material", {})
    if (
        material.get("name") != "diamond silicon"
        or material.get("functional") != "PBE"
        or material.get("primitive_occupied_band_count") != 4
    ):
        msg = "silicon band reference material identity mismatch"
        raise ValueError(msg)
    return payload


def silicon_primitive_cell(lattice_bohr: float) -> np.ndarray:
    """Return conventional FCC primitive direct-lattice rows for silicon."""

    if not np.isfinite(lattice_bohr) or lattice_bohr <= 0.0:
        msg = "lattice_bohr must be finite and positive"
        raise ValueError(msg)
    return 0.5 * float(lattice_bohr) * np.asarray(
        ((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
        dtype=np.float64,
    )


def silicon_band_path(
    *,
    points_per_segment: int = 9,
    profile: str = "full",
    references: dict[str, Any] | None = None,
) -> BandPath:
    """Build the primitive silicon path used by the validation benchmark.

    Args:
        points_per_segment: Inclusive samples per full path segment.
        profile: ``"gamma"``, ``"short"``, or ``"full"``.
        references: Optional already-loaded reference bundle.

    Returns:
        Reduced-coordinate primitive-cell path. Adjacent segment endpoints are
        represented only once.
    """

    payload = load_silicon_band_references() if references is None else references
    raw_points = payload["path_convention"]["points"]
    points = {
        label: np.asarray(vector, dtype=np.float64)
        for label, vector in raw_points.items()
    }
    if profile == "gamma":
        return BandPath(
            [KPoint(points["GAMMA"], label="Γ", coordinate_system="reduced")]
        )
    if profile == "short":
        return BandPath(
            [
                KPoint(points["GAMMA"], label="Γ", coordinate_system="reduced"),
                KPoint(
                    0.875 * points["X"],
                    label="0.875 X",
                    coordinate_system="reduced",
                ),
                KPoint(points["X"], label="X", coordinate_system="reduced"),
            ]
        )
    if profile != "full":
        msg = "profile must be 'gamma', 'short', or 'full'"
        raise ValueError(msg)
    if type(points_per_segment) is not int or points_per_segment < 2:
        msg = "points_per_segment must be an integer of at least two"
        raise ValueError(msg)

    sampled: list[KPoint] = []
    for segment_index, (start_label, end_label) in enumerate(
        zip(_FULL_LABELS[:-1], _FULL_LABELS[1:], strict=True)
    ):
        start = points[start_label]
        end = points[end_label]
        for local_index in range(points_per_segment):
            if segment_index > 0 and local_index == 0:
                continue
            fraction = local_index / (points_per_segment - 1)
            label = None
            if local_index == 0:
                label = "Γ" if start_label == "GAMMA" else start_label
            elif local_index == points_per_segment - 1:
                label = "Γ" if end_label == "GAMMA" else end_label
            sampled.append(
                KPoint(
                    (1.0 - fraction) * start + fraction * end,
                    label=label,
                    coordinate_system="reduced",
                )
            )
    return BandPath(sampled)


def _weighted_levels(
    energies_ev: np.ndarray,
    weights: np.ndarray,
    indices: slice,
    *,
    minimum_weight: float = 0.02,
) -> np.ndarray:
    selected = energies_ev[indices][weights[indices] >= minimum_weight]
    if selected.size == 0:
        msg = "unfolded path point contains no admitted primitive spectral levels"
        raise ValueError(msg)
    return np.sort(selected)


def _cluster_degenerate_levels(
    levels_ev: np.ndarray,
    *,
    tolerance_ev: float = 0.02,
) -> np.ndarray:
    """Collapse numerically split members of one degenerate energy level."""

    levels = np.sort(np.asarray(levels_ev, dtype=np.float64))
    if levels.size == 0:
        return levels
    clusters: list[list[float]] = [[float(levels[0])]]
    for value in levels[1:]:
        if float(value) - clusters[-1][-1] <= tolerance_ev:
            clusters[-1].append(float(value))
        else:
            clusters.append([float(value)])
    return np.asarray(
        [float(np.mean(cluster)) for cluster in clusters],
        dtype=np.float64,
    )


def analyze_silicon_bands(
    eigenvalues_hartree: np.ndarray,
    spectral_weights: np.ndarray,
    residuals: np.ndarray,
    overlap_errors: Sequence[float],
    primitive_path: BandPath,
    *,
    supercell_occupied_bands: int = 16,
    references: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare unfolded silicon bands with source-attributed PBE targets."""

    payload = load_silicon_band_references() if references is None else references
    energies = np.asarray(eigenvalues_hartree, dtype=np.float64) * HARTREE_TO_EV
    weights = np.asarray(spectral_weights, dtype=np.float64)
    residual_values = np.asarray(residuals, dtype=np.float64)
    if (
        energies.ndim != 2
        or weights.shape != energies.shape
        or residual_values.shape != energies.shape
        or energies.shape[0] != len(primitive_path.points)
        or supercell_occupied_bands <= 0
        or supercell_occupied_bands >= energies.shape[1]
    ):
        msg = "silicon band analysis arrays or occupied-band metadata are inconsistent"
        raise ValueError(msg)
    if not (
        np.isfinite(energies).all()
        and np.isfinite(weights).all()
        and np.isfinite(residual_values).all()
    ):
        msg = "silicon band analysis requires finite arrays"
        raise ValueError(msg)

    occupied = slice(0, supercell_occupied_bands)
    unoccupied = slice(supercell_occupied_bands, energies.shape[1])
    occupied_levels = [
        _weighted_levels(row, weight, occupied)
        for row, weight in zip(energies, weights, strict=True)
    ]
    unoccupied_levels = [
        _weighted_levels(row, weight, unoccupied)
        for row, weight in zip(energies, weights, strict=True)
    ]
    gamma_index = 0
    gamma_vbm = float(np.max(occupied_levels[gamma_index]))
    gamma_valence_bottom = float(np.min(occupied_levels[gamma_index]))
    valence_bandwidth = gamma_vbm - gamma_valence_bottom
    global_vbm = max(float(np.max(levels)) for levels in occupied_levels)
    global_cbm = min(float(np.min(levels)) for levels in unoccupied_levels)
    gap = global_cbm - global_vbm
    cbm_index = min(
        range(len(unoccupied_levels)),
        key=lambda index: float(np.min(unoccupied_levels[index])),
    )

    labels: dict[str, int] = {}
    for index, point in enumerate(primitive_path.points):
        if point.label is not None and point.label not in labels:
            labels[point.label] = index
    gamma_x_end = labels.get("X")
    gamma_x_fraction = None
    if gamma_x_end is not None and gamma_x_end > gamma_index and cbm_index <= gamma_x_end:
        gamma_vector = np.asarray(
            primitive_path.points[gamma_index].vector,
            dtype=np.float64,
        )
        x_vector = np.asarray(
            primitive_path.points[gamma_x_end].vector,
            dtype=np.float64,
        )
        cbm_vector = np.asarray(
            primitive_path.points[cbm_index].vector,
            dtype=np.float64,
        )
        segment = x_vector - gamma_vector
        gamma_x_fraction = float(
            np.dot(cbm_vector - gamma_vector, segment) / np.dot(segment, segment)
        )

    aligned = energies - gamma_vbm
    primary = payload["primary_reference"]["energies_ev_relative_to_vbm"]
    comparisons: dict[str, Any] = {}
    target_specs = {
        "gamma_valence_bottom": ("Γ", [primary["gamma_valence_bottom"]], occupied),
        "gamma_conduction": ("Γ", primary["gamma_conduction"], unoccupied),
        "x_valence": ("X", primary["x_valence"], occupied),
        "x_conduction": ("X", [primary["x_conduction"]], unoccupied),
        "l_valence": ("L", primary["l_valence"], occupied),
        "l_conduction": ("L", primary["l_conduction"], unoccupied),
    }
    for name, (label, targets, index_slice) in target_specs.items():
        if label not in labels:
            continue
        point_index = labels[label]
        levels = _weighted_levels(
            aligned[point_index],
            weights[point_index],
            index_slice,
        )
        clusters = _cluster_degenerate_levels(levels)
        remaining = list(float(value) for value in clusters)
        evaluated_targets = [float(value) for value in targets[: len(remaining)]]
        matched: list[float] = []
        for target in evaluated_targets:
            selected_index = int(np.argmin(np.abs(np.asarray(remaining) - target)))
            matched.append(remaining.pop(selected_index))
        errors = [
            abs(value - target)
            for value, target in zip(matched, evaluated_targets, strict=True)
        ]
        comparisons[name] = {
            "reference_ev": evaluated_targets,
            "calculated_ev": matched,
            "absolute_errors_ev": errors,
            "not_evaluated_reference_ev": [
                float(value) for value in targets[len(evaluated_targets) :]
            ],
            "degeneracy_cluster_tolerance_ev": 0.02,
        }

    thresholds = payload["validation_thresholds"]
    occupied_weight_sum = np.sum(weights[:, occupied], axis=1)
    numerical_gates = {
        "eigensolver_residual": float(np.max(residual_values))
        <= float(thresholds["max_eigensolver_residual"]),
        "overlap_error": float(np.max(overlap_errors))
        <= float(thresholds["max_overlap_error"]),
        "primitive_occupied_weight": float(
            np.max(np.abs(occupied_weight_sum - 4.0))
        )
        <= float(thresholds["primitive_occupied_weight_tolerance"]),
    }
    scientific_gates: dict[str, bool] = {}
    if gamma_x_end is not None:
        scientific_gates["indirect_gap"] = (
            float(thresholds["indirect_gap_ev"][0])
            <= gap
            <= float(thresholds["indirect_gap_ev"][1])
        )
        scientific_gates["cbm_location_gamma_x"] = (
            gamma_x_fraction is not None
            and float(thresholds["gamma_x_cbm_fraction"][0])
            <= gamma_x_fraction
            <= float(thresholds["gamma_x_cbm_fraction"][1])
        )
        scientific_gates["valence_bandwidth"] = (
            float(thresholds["valence_bandwidth_ev"][0])
            <= valence_bandwidth
            <= float(thresholds["valence_bandwidth_ev"][1])
        )
    if "L" in labels:
        tolerance = float(thresholds["high_symmetry_energy_tolerance_ev"])
        scientific_gates["high_symmetry_order_and_energy"] = all(
            max(comparison["absolute_errors_ev"]) <= tolerance
            for comparison in comparisons.values()
        )

    return {
        "status": (
            "validated"
            if all(numerical_gates.values())
            and scientific_gates
            and all(scientific_gates.values())
            else "numerically_valid"
            if all(numerical_gates.values())
            else "failed"
        ),
        "gap": {
            "indirect_ev": gap,
            "vbm_ev_raw": global_vbm,
            "cbm_ev_raw": global_cbm,
            "cbm_path_index": cbm_index,
            "cbm_gamma_x_fraction": gamma_x_fraction,
        },
        "valence_bandwidth_ev": valence_bandwidth,
        "gamma_vbm_ev_raw": gamma_vbm,
        "max_eigensolver_residual": float(np.max(residual_values)),
        "max_overlap_error": float(np.max(overlap_errors)),
        "primitive_occupied_weight_sum": occupied_weight_sum.tolist(),
        "primitive_occupied_weight_max_error": float(
            np.max(np.abs(occupied_weight_sum - 4.0))
        ),
        "high_symmetry_comparisons": comparisons,
        "numerical_gates": numerical_gates,
        "scientific_gates": scientific_gates,
        "references": {
            "primary": payload["primary_reference"],
            "path_convention": payload["path_convention"],
            "secondary_local_qe_reference": payload["secondary_local_qe_reference"],
        },
    }


def _translation_error(density: np.ndarray) -> float:
    if any(count % 2 != 0 for count in density.shape):
        msg = "conventional silicon density grid must have even dimensions"
        raise ValueError(msg)
    shifts = (
        (0, density.shape[1] // 2, density.shape[2] // 2),
        (density.shape[0] // 2, 0, density.shape[2] // 2),
        (density.shape[0] // 2, density.shape[1] // 2, 0),
    )
    return max(
        float(np.max(np.abs(density - np.roll(density, shift, axis=(0, 1, 2)))))
        for shift in shifts
    )


def _load_source(
    result_path: Path,
    manifest_path: Path,
    gth_path: Path,
) -> tuple[PeriodicDFTSystem, PeriodicFrozenDensity, dict[str, Any]]:
    report = json.loads(result_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    result = report.get("result", {})
    settings = report.get("settings", {})
    if not result.get("converged") or result.get("status") != "converged":
        msg = "source periodic SCF artifact must be converged"
        raise ValueError(msg)
    if (
        settings.get("cutoff_hartree") != 25.0
        or settings.get("fft_shape") != [56, 56, 56]
        or settings.get("kpoint_mesh") != [6, 6, 6]
    ):
        msg = "source SCF artifact must use the validated 25 Ha, 56^3, 6^3 protocol"
        raise ValueError(msg)
    pseudo_manifest = manifest.get("pseudopotential", {})
    if pseudo_manifest.get("sha256") != _sha256(gth_path):
        msg = "GTH pseudopotential does not match the workload manifest"
        raise ValueError(msg)
    arrays_path = Path(report["arrays"])
    if not arrays_path.is_absolute():
        repository_candidate = Path.cwd() / arrays_path
        result_candidate = result_path.parent / arrays_path
        arrays_path = (
            repository_candidate
            if repository_candidate.exists()
            else result_candidate
        )
    with np.load(arrays_path) as arrays:
        if "density" not in arrays.files:
            msg = "source SCF arrays do not contain density"
            raise ValueError(msg)
        density = np.array(arrays["density"], dtype=np.float32, copy=True)
    lattice = float(report["lattice_bohr"])
    positions = np.asarray(report["positions_bohr"], dtype=np.float64)
    electron_count = float(result["electron_count"])
    if (
        positions.shape != (8, 3)
        or electron_count != 32.0
        or result.get("iterations", 0) <= 0
        or float(result.get("density_residual", np.inf)) > 1e-6
    ):
        msg = "source SCF artifact is not the validated eight-atom silicon state"
        raise ValueError(msg)
    pseudo = read_gth(gth_path, element="Si")
    system = PeriodicDFTSystem(
        (lattice, lattice, lattice),
        settings["fft_shape"],
        positions,
        pseudo,
        electron_count=32.0,
    )
    frozen = PeriodicFrozenDensity(density, 25.0, 32.0)
    provenance = {
        "scf_result_path": str(result_path),
        "scf_result_sha256": _sha256(result_path),
        "scf_arrays_path": str(arrays_path),
        "scf_arrays_sha256": _sha256(arrays_path),
        "workload_manifest_path": str(manifest_path),
        "workload_manifest_sha256": _sha256(manifest_path),
        "gth_path": str(gth_path),
        "gth_sha256": _sha256(gth_path),
        "source_scf_iterations": int(result["iterations"]),
        "source_density_residual": float(result["density_residual"]),
        "source_energy_delta_hartree": float(result["energy_delta_hartree"]),
        "density_primitive_translation_max_error": _translation_error(density),
    }
    if provenance["density_primitive_translation_max_error"] > 1e-6:
        msg = "source density does not preserve conventional-to-primitive translations"
        raise ValueError(msg)
    return system, frozen, provenance


def run_silicon_band_benchmark(
    *,
    scf_result_path: str | Path,
    manifest_path: str | Path,
    gth_path: str | Path,
    output_directory: str | Path,
    profile: str = "full",
    points_per_segment: int = 9,
) -> dict[str, Any]:
    """Run and persist the bounded frozen-density silicon band benchmark."""

    references = load_silicon_band_references()
    system, frozen, provenance = _load_source(
        Path(scf_result_path),
        Path(manifest_path),
        Path(gth_path),
    )
    lattice = float(np.asarray(system.grid.lengths)[0])
    primitive_path = silicon_band_path(
        points_per_segment=points_per_segment,
        profile=profile,
        references=references,
    )
    folded = fold_band_path_to_supercell(
        silicon_primitive_cell(lattice),
        lattice * np.eye(3),
        primitive_path,
    )
    bands = run_periodic_band_structure(
        system,
        frozen,
        folded.supercell_path,
        n_bands=24,
        guard_bands=2,
        config=PeriodicDavidsonConfig(
            max_iterations=80,
            tolerance=1e-6,
            max_subspace_size=64,
            preconditioner_floor=0.25,
        ),
    )
    unfolded = unfold_periodic_band_structure(bands, folded)
    eigenvalues = np.asarray(bands.eigenvalues)
    residuals = np.asarray(bands.residuals)
    weights = np.asarray(unfolded.spectral_weights)
    overlap_errors = [point.eigen.orthonormality_error for point in bands.points]
    analysis = analyze_silicon_bands(
        eigenvalues,
        weights,
        residuals,
        overlap_errors,
        primitive_path,
        references=references,
    )

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    arrays_path = output / f"bands-{profile}.npz"
    np.savez_compressed(
        arrays_path,
        primitive_kpoints=np.asarray([point.vector for point in primitive_path.points]),
        supercell_kpoints=np.asarray([point.vector for point in folded.supercell_path.points]),
        path_distances_inverse_bohr=folded.path_distances,
        eigenvalues_hartree=eigenvalues,
        eigenvalues_ev=eigenvalues * HARTREE_TO_EV,
        residuals=residuals,
        spectral_weights=weights,
        overlap_errors=np.asarray(overlap_errors),
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "profile": profile,
        "scientific_scope": {
            "material": "diamond silicon",
            "cell": "eight-atom conventional cubic supercell",
            "interpretation": "plane-wave unfolded to the two-atom FCC primitive cell",
            "functional": "PBE",
            "pseudopotential": "GTH-PBE-q4",
            "cutoff_hartree": 25.0,
            "source_scf_mesh": [6, 6, 6],
            "source_scf_explicit_kpoints": 216,
            "source_scf_time_reversal_representatives": 108,
            "path": "Γ-X-W-K-Γ-L",
            "path_point_count": len(primitive_path.points),
            "requested_supercell_bands": 24,
            "davidson_guard_bands": 2,
            "occupied_supercell_bands": 16,
            "occupied_primitive_bands": 4,
            "self_consistency_iterations": 0,
        },
        "provenance": provenance,
        "solver": {
            "name": "block-davidson-rayleigh-ritz",
            "max_iterations": 80,
            "residual_tolerance": 1e-6,
            "max_subspace_size": 64,
            "preconditioner_floor": 0.25,
            "dense_full_hamiltonian": False,
        },
        "timings_ms": bands.timings,
        "analysis": analysis,
        "folding": folded.to_dict(),
        "arrays": str(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
    }
    report_path = output / f"report-{profile}.json"
    report_path.write_bytes(_canonical_json(report))
    return {**report, "report": str(report_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scf-result", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gth", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--profile",
        choices=("gamma", "short", "full"),
        default="full",
    )
    parser.add_argument("--points-per-segment", type=int, default=9)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line silicon band benchmark."""

    args = _parser().parse_args(argv)
    report = run_silicon_band_benchmark(
        scf_result_path=args.scf_result,
        manifest_path=args.manifest,
        gth_path=args.gth,
        output_directory=args.out,
        profile=args.profile,
        points_per_segment=args.points_per_segment,
    )
    print(
        json.dumps(
            {
                "status": report["analysis"]["status"],
                "profile": report["profile"],
                "gap": report["analysis"]["gap"],
                "valence_bandwidth_ev": report["analysis"]["valence_bandwidth_ev"],
                "timings_ms": report["timings_ms"],
                "report": report["report"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["analysis"]["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
