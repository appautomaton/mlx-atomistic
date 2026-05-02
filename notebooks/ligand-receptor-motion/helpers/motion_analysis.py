"""Ligand-receptor trajectory analysis helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProcessedTrajectory:
    """Lightweight processed ligand-receptor trajectory."""

    positions: np.ndarray
    time_ps: np.ndarray
    symbols: np.ndarray
    atom_names: np.ndarray
    residue_names: np.ndarray
    residue_ids: np.ndarray
    segment_ids: np.ndarray
    ligand_indices: np.ndarray
    receptor_indices: np.ndarray
    source: dict[str, Any]
    water_indices: np.ndarray | None = None
    ion_indices: np.ndarray | None = None
    lipid_indices: np.ndarray | None = None
    cell_lengths_A: np.ndarray | None = None

    @classmethod
    def load(cls, path: str | Path) -> ProcessedTrajectory:
        with np.load(path, allow_pickle=False) as data:
            source = json.loads(str(np.asarray(data["source_json"])))
            water_indices = (
                np.asarray(data["water_indices"], dtype=np.int32)
                if "water_indices" in data.files
                else np.asarray([], dtype=np.int32)
            )
            ion_indices = (
                np.asarray(data["ion_indices"], dtype=np.int32)
                if "ion_indices" in data.files
                else np.asarray([], dtype=np.int32)
            )
            lipid_indices = (
                np.asarray(data["lipid_indices"], dtype=np.int32)
                if "lipid_indices" in data.files
                else np.asarray([], dtype=np.int32)
            )
            return cls(
                positions=np.asarray(data["positions"], dtype=np.float32),
                time_ps=np.asarray(data["time_ps"], dtype=np.float32),
                symbols=np.asarray(data["symbols"]).astype(str),
                atom_names=np.asarray(data["atom_names"]).astype(str),
                residue_names=np.asarray(data["residue_names"]).astype(str),
                residue_ids=np.asarray(data["residue_ids"], dtype=np.int32),
                segment_ids=np.asarray(data["segment_ids"]).astype(str),
                ligand_indices=np.asarray(data["ligand_indices"], dtype=np.int32),
                receptor_indices=np.asarray(data["receptor_indices"], dtype=np.int32),
                source=source,
                water_indices=water_indices,
                ion_indices=ion_indices,
                lipid_indices=lipid_indices,
                cell_lengths_A=(
                    np.asarray(data["cell_lengths_A"], dtype=np.float32)
                    if "cell_lengths_A" in data.files and np.asarray(data["cell_lengths_A"]).size
                    else None
                ),
            )

    @property
    def frame_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def atom_count(self) -> int:
        return int(self.positions.shape[1])


def save_processed_trajectory(path: str | Path, trajectory: ProcessedTrajectory) -> None:
    """Save a processed ligand-receptor trajectory."""

    positions = np.asarray(trajectory.positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[2] != 3:
        msg = "positions must have shape (n_frames, n_atoms, 3)"
        raise ValueError(msg)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        positions=positions,
        time_ps=np.asarray(trajectory.time_ps, dtype=np.float32),
        symbols=np.asarray(trajectory.symbols, dtype=str),
        atom_names=np.asarray(trajectory.atom_names, dtype=str),
        residue_names=np.asarray(trajectory.residue_names, dtype=str),
        residue_ids=np.asarray(trajectory.residue_ids, dtype=np.int32),
        segment_ids=np.asarray(trajectory.segment_ids, dtype=str),
        ligand_indices=np.asarray(trajectory.ligand_indices, dtype=np.int32),
        receptor_indices=np.asarray(trajectory.receptor_indices, dtype=np.int32),
        water_indices=np.asarray(_optional_indices(trajectory.water_indices), dtype=np.int32),
        ion_indices=np.asarray(_optional_indices(trajectory.ion_indices), dtype=np.int32),
        lipid_indices=np.asarray(_optional_indices(trajectory.lipid_indices), dtype=np.int32),
        cell_lengths_A=_cell_lengths_or_empty(trajectory),
        source_json=np.asarray(json.dumps(trajectory.source)),
    )


def align_trajectory_to_reference(
    positions: np.ndarray,
    *,
    align_indices: np.ndarray,
    reference_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Rigidly align all frames to the reference using selected atoms."""

    positions = np.asarray(positions, dtype=np.float32)
    align_indices = np.asarray(align_indices, dtype=np.int32)
    if positions.ndim != 3 or positions.shape[2] != 3:
        msg = "positions must have shape (n_frames, n_atoms, 3)"
        raise ValueError(msg)
    if align_indices.size < 3:
        msg = "alignment requires at least three atoms"
        raise ValueError(msg)
    reference = (
        positions[0, align_indices].copy()
        if reference_positions is None
        else np.asarray(reference_positions, dtype=np.float32)
    )
    reference_center = reference.mean(axis=0)
    reference_centered = reference - reference_center
    aligned = np.empty_like(positions)
    for frame_index, frame in enumerate(positions):
        mobile = frame[align_indices]
        mobile_center = mobile.mean(axis=0)
        mobile_centered = mobile - mobile_center
        rotation = _kabsch_rotation(mobile_centered, reference_centered)
        aligned[frame_index] = (frame - mobile_center) @ rotation + reference_center
    return aligned


def raw_ligand_com(trajectory: ProcessedTrajectory) -> np.ndarray:
    """Return raw wrapped ligand center of geometry per frame."""

    return trajectory.positions[:, trajectory.ligand_indices].mean(axis=1)


def ligand_com(trajectory: ProcessedTrajectory) -> np.ndarray:
    """Return whole-ligand center of geometry with frame-to-frame PBC unwrapping."""

    centers = _whole_ligand_com(trajectory)
    cell_lengths = _cell_lengths(trajectory)
    if cell_lengths is None or centers.shape[0] <= 1:
        return centers
    return unwrap_points_by_minimum_image(centers, cell_lengths=cell_lengths)


def _whole_ligand_com(trajectory: ProcessedTrajectory) -> np.ndarray:
    ligand_positions = np.asarray(
        trajectory.positions[:, trajectory.ligand_indices],
        dtype=np.float32,
    )
    cell_lengths = _cell_lengths(trajectory)
    if cell_lengths is None or ligand_positions.shape[1] == 0:
        return ligand_positions.mean(axis=1)
    centers = []
    for frame in ligand_positions:
        anchor = frame[0]
        whole = anchor[None, :] + minimum_image_delta(frame - anchor[None, :], cell_lengths)
        centers.append(whole.mean(axis=0))
    return np.asarray(centers, dtype=np.float32)


def ligand_com_displacement(trajectory: ProcessedTrajectory) -> np.ndarray:
    """Return PBC-aware ligand COM displacement from the first frame."""

    centers = ligand_com(trajectory)
    return np.linalg.norm(centers - centers[0], axis=1)


def pbc_corrected_ligand_positions(trajectory: ProcessedTrajectory) -> np.ndarray:
    """Return ligand coordinates as a whole molecule following unwrapped COM."""

    ligand_positions = np.asarray(
        trajectory.positions[:, trajectory.ligand_indices],
        dtype=np.float32,
    )
    cell_lengths = _cell_lengths(trajectory)
    if cell_lengths is None or ligand_positions.shape[1] == 0:
        return ligand_positions
    corrected_centers = ligand_com(trajectory)
    whole_frames = []
    for frame_index, frame in enumerate(ligand_positions):
        anchor = frame[0]
        whole = anchor[None, :] + minimum_image_delta(frame - anchor[None, :], cell_lengths)
        whole_center = whole.mean(axis=0)
        whole_frames.append(whole + corrected_centers[frame_index] - whole_center)
    return np.asarray(whole_frames, dtype=np.float32)


def minimum_image_delta(delta: np.ndarray, cell_lengths: np.ndarray) -> np.ndarray:
    """Return minimum-image displacement vectors for orthorhombic cell lengths."""

    delta = np.asarray(delta, dtype=np.float32)
    cell = np.asarray(cell_lengths, dtype=np.float32)
    return delta - cell * np.round(delta / cell)


def nearest_periodic_image(
    positions: np.ndarray,
    *,
    reference: np.ndarray,
    cell_lengths: np.ndarray | None,
) -> np.ndarray:
    """Shift positions into the nearest periodic image around a reference point."""

    positions = np.asarray(positions, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if cell_lengths is None:
        return positions
    return reference[None, :] + minimum_image_delta(positions - reference[None, :], cell_lengths)


def unwrap_points_by_minimum_image(
    points: np.ndarray,
    *,
    cell_lengths: np.ndarray,
) -> np.ndarray:
    """Unwrap a point trajectory by accumulating minimum-image frame deltas."""

    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] <= 1:
        return points.copy()
    unwrapped = np.empty_like(points)
    unwrapped[0] = points[0]
    for frame_index in range(1, points.shape[0]):
        delta = minimum_image_delta(points[frame_index] - points[frame_index - 1], cell_lengths)
        unwrapped[frame_index] = unwrapped[frame_index - 1] + delta
    return unwrapped


def contact_counts(
    trajectory: ProcessedTrajectory,
    *,
    cutoff_A: float = 4.5,
) -> np.ndarray:
    """Return protein-ligand atom contact counts per frame."""

    ligand = trajectory.positions[:, trajectory.ligand_indices]
    receptor = trajectory.positions[:, trajectory.receptor_indices]
    counts = []
    cutoff2 = cutoff_A * cutoff_A
    for frame_ligand, frame_receptor in zip(ligand, receptor, strict=True):
        delta = frame_ligand[:, None, :] - frame_receptor[None, :, :]
        delta = _minimum_image_for_trajectory(trajectory, delta)
        distances2 = np.sum(delta * delta, axis=-1)
        counts.append(int(np.count_nonzero(distances2 <= cutoff2)))
    return np.asarray(counts, dtype=np.int32)


def water_counts_around_ligand(
    trajectory: ProcessedTrajectory,
    *,
    cutoff_A: float = 5.0,
) -> np.ndarray:
    """Return water oxygen counts near any ligand atom per frame."""

    water_indices = _optional_indices(trajectory.water_indices)
    if water_indices.size == 0:
        return np.zeros((trajectory.frame_count,), dtype=np.int32)
    water_oxygens = water_indices[
        np.char.upper(trajectory.symbols[water_indices].astype(str)) == "O"
    ]
    if water_oxygens.size == 0:
        return np.zeros((trajectory.frame_count,), dtype=np.int32)
    ligand = trajectory.positions[:, trajectory.ligand_indices]
    waters = trajectory.positions[:, water_oxygens]
    cutoff2 = cutoff_A * cutoff_A
    counts = []
    for frame_ligand, frame_waters in zip(ligand, waters, strict=True):
        delta = frame_waters[:, None, :] - frame_ligand[None, :, :]
        delta = _minimum_image_for_trajectory(trajectory, delta)
        distances2 = np.sum(delta * delta, axis=-1)
        counts.append(int(np.count_nonzero(np.any(distances2 <= cutoff2, axis=1))))
    return np.asarray(counts, dtype=np.int32)


def ion_counts_around_ligand(
    trajectory: ProcessedTrajectory,
    *,
    cutoff_A: float = 6.0,
) -> np.ndarray:
    """Return ion counts near any ligand atom per frame."""

    ion_indices = _optional_indices(trajectory.ion_indices)
    if ion_indices.size == 0:
        return np.zeros((trajectory.frame_count,), dtype=np.int32)
    ligand = trajectory.positions[:, trajectory.ligand_indices]
    ions = trajectory.positions[:, ion_indices]
    cutoff2 = cutoff_A * cutoff_A
    counts = []
    for frame_ligand, frame_ions in zip(ligand, ions, strict=True):
        delta = frame_ions[:, None, :] - frame_ligand[None, :, :]
        delta = _minimum_image_for_trajectory(trajectory, delta)
        distances2 = np.sum(delta * delta, axis=-1)
        counts.append(int(np.count_nonzero(np.any(distances2 <= cutoff2, axis=1))))
    return np.asarray(counts, dtype=np.int32)


def hydrogen_bond_counts(
    trajectory: ProcessedTrajectory,
    *,
    donor_acceptor_cutoff_A: float = 3.5,
    hydrogen_acceptor_cutoff_A: float = 2.6,
) -> pd.DataFrame:
    """Return simple geometric ligand-receptor/water hydrogen-bond counts."""

    symbols = np.char.upper(trajectory.symbols.astype(str))
    hydrogen_indices = np.flatnonzero(symbols == "H")
    hetero_indices = np.flatnonzero(np.isin(symbols, ["N", "O", "S"]))
    if hydrogen_indices.size == 0 or hetero_indices.size == 0:
        return _not_available_frame_table(trajectory, "missing hydrogens or hetero atoms")

    ligand_set = set(trajectory.ligand_indices.tolist())
    receptor_set = set(trajectory.receptor_indices.tolist())
    water_set = set(_optional_indices(trajectory.water_indices).tolist())
    rows = []
    for frame_index, time_ps in enumerate(trajectory.time_ps):
        positions = trajectory.positions[frame_index]
        ligand_receptor = 0
        water_bridges = 0
        for hydrogen in hydrogen_indices:
            hydrogen_distances = _norms(
                trajectory,
                positions[hetero_indices] - positions[hydrogen],
            )
            donor_candidates = hetero_indices[
                hydrogen_distances <= 1.25
            ]
            if donor_candidates.size == 0:
                continue
            donor = int(donor_candidates[0])
            acceptor_distances = _norms(
                trajectory,
                positions[hetero_indices] - positions[hydrogen],
            )
            acceptor_candidates = hetero_indices[
                acceptor_distances <= hydrogen_acceptor_cutoff_A
            ]
            for acceptor in acceptor_candidates.tolist():
                acceptor = int(acceptor)
                if acceptor == donor:
                    continue
                donor_acceptor_distance = float(
                    _norms(trajectory, positions[acceptor] - positions[donor])
                )
                if donor_acceptor_distance > donor_acceptor_cutoff_A:
                    continue
                if _crosses_sets(donor, acceptor, ligand_set, receptor_set):
                    ligand_receptor += 1
                if water_set and (
                    _crosses_sets(donor, acceptor, ligand_set, water_set)
                    or _crosses_sets(donor, acceptor, receptor_set, water_set)
                ):
                    water_bridges += 1
        rows.append(
            {
                "frame": frame_index,
                "time_ps": float(time_ps),
                "ligand_receptor_hbond_count": ligand_receptor,
                "water_involved_hbond_count": water_bridges,
                "available": True,
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def residue_contact_occupancy(
    trajectory: ProcessedTrajectory,
    *,
    cutoff_A: float = 4.5,
) -> pd.DataFrame:
    """Return receptor residue contact occupancy against any ligand atom."""

    ligand_positions = trajectory.positions[:, trajectory.ligand_indices]
    receptor_positions = trajectory.positions[:, trajectory.receptor_indices]
    receptor_names = trajectory.residue_names[trajectory.receptor_indices]
    receptor_ids = trajectory.residue_ids[trajectory.receptor_indices]
    segment_ids = trajectory.segment_ids[trajectory.receptor_indices]
    residue_keys = np.asarray(
        [
            f"{seg}:{resname}{resid}"
            for seg, resname, resid in zip(segment_ids, receptor_names, receptor_ids, strict=True)
        ]
    )
    rows = []
    for residue in sorted(set(residue_keys.tolist())):
        atom_mask = residue_keys == residue
        per_frame_min = []
        for frame_index in range(trajectory.frame_count):
            residue_positions = receptor_positions[frame_index][atom_mask]
            delta = (
                ligand_positions[frame_index, :, None, :]
                - residue_positions[None, :, :]
            )
            delta = _minimum_image_for_trajectory(trajectory, delta)
            distances = np.sqrt(np.sum(delta * delta, axis=-1))
            per_frame_min.append(float(np.min(distances)))
        per_frame_min_array = np.asarray(per_frame_min, dtype=np.float32)
        contact_mask = per_frame_min_array <= cutoff_A
        if np.any(contact_mask):
            contact_frames = np.flatnonzero(contact_mask)
            first_contact_ps = float(trajectory.time_ps[contact_frames[0]])
            last_contact_ps = float(trajectory.time_ps[contact_frames[-1]])
        else:
            first_contact_ps = np.nan
            last_contact_ps = np.nan
        rows.append(
            {
                "residue": residue,
                "contact_occupancy": float(np.mean(contact_mask)),
                "min_distance_A": float(np.min(per_frame_min_array)),
                "first_contact_ps": first_contact_ps,
                "last_contact_ps": last_contact_ps,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["contact_occupancy", "min_distance_A"], ascending=[False, True])
        .reset_index(drop=True)
    )


def closest_residues(
    trajectory: ProcessedTrajectory,
    *,
    top_n: int = 12,
) -> pd.DataFrame:
    """Return closest receptor residues to the ligand across frames."""

    rows = []
    ligand_positions = trajectory.positions[:, trajectory.ligand_indices]
    receptor_positions = trajectory.positions[:, trajectory.receptor_indices]
    receptor_names = trajectory.residue_names[trajectory.receptor_indices]
    receptor_ids = trajectory.residue_ids[trajectory.receptor_indices]
    segment_ids = trajectory.segment_ids[trajectory.receptor_indices]
    residue_keys = np.asarray(
        [
            f"{seg}:{resname}{resid}"
            for seg, resname, resid in zip(segment_ids, receptor_names, receptor_ids, strict=True)
        ]
    )
    for frame_index, time_ps in enumerate(trajectory.time_ps):
        delta = ligand_positions[frame_index, :, None, :] - receptor_positions[frame_index, None]
        delta = _minimum_image_for_trajectory(trajectory, delta)
        distances = np.sqrt(np.sum(delta * delta, axis=-1))
        per_atom_min = np.min(distances, axis=0)
        residue_min: dict[str, float] = {}
        for key, distance in zip(residue_keys, per_atom_min, strict=True):
            residue_min[key] = min(float(distance), residue_min.get(key, float("inf")))
        for key, distance in sorted(residue_min.items(), key=lambda item: item[1])[:top_n]:
            rows.append(
                {
                    "frame": frame_index,
                    "time_ps": float(time_ps),
                    "residue": key,
                    "min_distance_A": distance,
                }
            )
    return pd.DataFrame(rows)


def motion_gate_report(
    trajectory: ProcessedTrajectory,
    *,
    min_ligand_com_displacement_A: float = 8.0,
    min_contact_count_delta: int = 10,
    contact_cutoff_A: float = 4.5,
) -> dict[str, Any]:
    """Return visible-motion gate metrics."""

    displacement = ligand_com_displacement(trajectory)
    contacts = contact_counts(trajectory, cutoff_A=contact_cutoff_A)
    contact_delta = int(np.max(contacts) - np.min(contacts))
    return {
        "frames": trajectory.frame_count,
        "atoms": trajectory.atom_count,
        "max_ligand_com_displacement_A": float(np.max(displacement)),
        "final_ligand_com_displacement_A": float(displacement[-1]),
        "min_contacts": int(np.min(contacts)),
        "max_contacts": int(np.max(contacts)),
        "contact_count_delta": contact_delta,
        "passes_motion_gate": bool(
            np.max(displacement) >= min_ligand_com_displacement_A
            and contact_delta >= min_contact_count_delta
        ),
        "min_ligand_com_displacement_A": min_ligand_com_displacement_A,
        "min_contact_count_delta": min_contact_count_delta,
        "contact_cutoff_A": contact_cutoff_A,
    }


def trajectory_quality_report(
    trajectory: ProcessedTrajectory,
    *,
    max_constraint_error_A: float | None = None,
    raw_jump_threshold_A: float | None = None,
    corrected_jump_threshold_A: float = 5.0,
    displacement_warning_A: float = 8.0,
) -> dict[str, Any]:
    """Return visualization-quality metrics for raw and PBC-corrected motion."""

    raw_centers = raw_ligand_com(trajectory)
    corrected_centers = ligand_com(trajectory)
    raw_steps = _frame_step_lengths(raw_centers)
    corrected_steps = _frame_step_lengths(corrected_centers)
    cell_lengths = _cell_lengths(trajectory)
    if raw_jump_threshold_A is None:
        raw_jump_threshold_A = (
            min(10.0, 0.25 * float(np.min(cell_lengths))) if cell_lengths is not None else 10.0
        )
    raw_jump_mask = raw_steps > raw_jump_threshold_A
    corrected_jump_mask = corrected_steps > corrected_jump_threshold_A
    raw_large_step_count = int(np.count_nonzero(raw_jump_mask))
    raw_pbc_jump_count = int(
        np.count_nonzero(raw_jump_mask & (corrected_steps < 0.5 * raw_steps))
    )
    contacts = contact_counts(trajectory)
    contact_delta = int(np.max(contacts) - np.min(contacts)) if contacts.size else 0
    displacement = np.linalg.norm(corrected_centers - corrected_centers[0], axis=1)
    warnings = []
    if cell_lengths is None:
        warnings.append("Missing cell lengths; PBC correction is disabled.")
    elif raw_pbc_jump_count:
        warnings.append(
            f"Corrected {raw_pbc_jump_count} raw ligand COM jump(s) caused by PBC wrapping."
        )
    if int(np.count_nonzero(corrected_jump_mask)):
        warnings.append(
            "PBC-corrected ligand COM still has large frame-to-frame jumps; inspect timestep, "
            "sampling stride, and trajectory stability."
        )
    if float(np.max(displacement)) >= displacement_warning_A and contact_delta >= 10:
        warnings.append(
            "Large ligand displacement/contact change in short NVT should be interpreted as "
            "diffusion-like motion, not binding or unbinding."
        )
    if max_constraint_error_A is not None and max_constraint_error_A > 5e-4:
        warnings.append(
            "Constraint error is high; inspect MD stability before interpreting motion."
        )
    return {
        "pbc_corrected_display": cell_lengths is not None,
        "cell_lengths_A": [] if cell_lengths is None else cell_lengths.astype(float).tolist(),
        "raw_ligand_com_max_step_A": float(np.max(raw_steps)) if raw_steps.size else 0.0,
        "unwrapped_ligand_com_max_step_A": (
            float(np.max(corrected_steps)) if corrected_steps.size else 0.0
        ),
        "raw_large_step_count": raw_large_step_count,
        "raw_pbc_jump_count": raw_pbc_jump_count,
        "pbc_corrected_jump_count": raw_pbc_jump_count,
        "final_unwrapped_ligand_com_displacement_A": float(displacement[-1]),
        "max_unwrapped_ligand_com_displacement_A": float(np.max(displacement)),
        "contact_count_delta": contact_delta,
        "max_constraint_error_A": max_constraint_error_A,
        "warnings": warnings,
    }


def analysis_tables(trajectory: ProcessedTrajectory) -> dict[str, pd.DataFrame]:
    """Build compact notebook analysis tables."""

    displacement = ligand_com_displacement(trajectory)
    contacts = contact_counts(trajectory)
    frame_df = pd.DataFrame(
        {
            "frame": np.arange(trajectory.frame_count, dtype=np.int32),
            "time_ps": trajectory.time_ps,
            "ligand_com_displacement_A": displacement,
            "contact_count_4p5A": contacts,
            "water_count_5A": water_counts_around_ligand(trajectory),
            "ion_count_6A": ion_counts_around_ligand(trajectory),
        }
    )
    summary_df = pd.DataFrame([motion_gate_report(trajectory)])
    quality = trajectory_quality_report(trajectory)
    return {
        "summary": summary_df,
        "quality": pd.DataFrame([{k: v for k, v in quality.items() if k != "warnings"}]),
        "quality_warnings": pd.DataFrame({"warning": quality["warnings"]}),
        "frames": frame_df,
        "closest_residues": closest_residues(trajectory),
        "contact_occupancy": residue_contact_occupancy(trajectory),
        "hydrogen_bonds": hydrogen_bond_counts(trajectory),
    }


def build_synthetic_motion_fixture(path: str | Path) -> ProcessedTrajectory:
    """Create a tiny deterministic fixture with obvious ligand translation."""

    receptor = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 2.0, 2.0],
            [2.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    ligand_start = np.asarray([[1.0, 1.0, 1.0], [1.7, 1.0, 1.0]], dtype=np.float32)
    frames = []
    for step in np.linspace(0.0, 12.0, 9, dtype=np.float32):
        ligand = ligand_start + np.asarray([step, 0.0, 0.0], dtype=np.float32)
        frames.append(np.vstack([receptor, ligand]))
    positions = np.asarray(frames, dtype=np.float32)
    trajectory = ProcessedTrajectory(
        positions=positions,
        time_ps=np.linspace(0.0, 8000.0, positions.shape[0], dtype=np.float32),
        symbols=np.asarray(["C", "N", "O", "C", "S", "C", "O"], dtype=str),
        atom_names=np.asarray(["CA", "N", "O", "CB", "SG", "C1", "O1"], dtype=str),
        residue_names=np.asarray(["REC"] * 5 + ["LIG"] * 2, dtype=str),
        residue_ids=np.asarray([1, 1, 1, 2, 2, 10, 10], dtype=np.int32),
        segment_ids=np.asarray(["A"] * 5 + ["L"] * 2, dtype=str),
        ligand_indices=np.asarray([5, 6], dtype=np.int32),
        receptor_indices=np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
        water_indices=np.asarray([], dtype=np.int32),
        ion_indices=np.asarray([], dtype=np.int32),
        cell_lengths_A=None,
        source={
            "kind": "test_fixture",
            "dataset_id": "synthetic-visible-ligand-translation",
            "title": "Synthetic visible ligand translation fixture",
            "note": "For tests only; not used as the public-data notebook path.",
        },
    )
    save_processed_trajectory(path, trajectory)
    return trajectory


def build_synthetic_pbc_wrap_fixture(path: str | Path) -> ProcessedTrajectory:
    """Create a fixture where raw ligand COM wraps across a periodic boundary."""

    receptor = np.asarray(
        [
            [9.5, 5.0, 5.0],
            [9.5, 6.0, 5.0],
            [9.5, 5.0, 6.0],
            [8.8, 5.5, 5.0],
        ],
        dtype=np.float32,
    )
    ligand_frames = [
        np.asarray([[9.6, 5.0, 5.0], [9.9, 5.0, 5.0]], dtype=np.float32),
        np.asarray([[0.1, 5.0, 5.0], [0.4, 5.0, 5.0]], dtype=np.float32),
        np.asarray([[0.6, 5.0, 5.0], [0.9, 5.0, 5.0]], dtype=np.float32),
    ]
    positions = np.asarray([np.vstack([receptor, ligand]) for ligand in ligand_frames])
    trajectory = ProcessedTrajectory(
        positions=positions,
        time_ps=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        symbols=np.asarray(["C", "N", "O", "S", "C", "C"], dtype=str),
        atom_names=np.asarray(["CA", "N", "O", "SG", "C1", "C2"], dtype=str),
        residue_names=np.asarray(["REC"] * 4 + ["LIG"] * 2, dtype=str),
        residue_ids=np.asarray([1, 1, 1, 1, 10, 10], dtype=np.int32),
        segment_ids=np.asarray(["A"] * 4 + ["L"] * 2, dtype=str),
        ligand_indices=np.asarray([4, 5], dtype=np.int32),
        receptor_indices=np.asarray([0, 1, 2, 3], dtype=np.int32),
        water_indices=np.asarray([], dtype=np.int32),
        ion_indices=np.asarray([], dtype=np.int32),
        cell_lengths_A=np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
        source={"kind": "test_fixture", "dataset_id": "synthetic-pbc-wrap"},
    )
    save_processed_trajectory(path, trajectory)
    return trajectory


def _kabsch_rotation(mobile_centered: np.ndarray, reference_centered: np.ndarray) -> np.ndarray:
    covariance = mobile_centered.T @ reference_centered
    v, _, wt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float32)
    correction[2, 2] = np.sign(np.linalg.det(v @ wt))
    return (v @ correction @ wt).astype(np.float32)


def _optional_indices(indices: np.ndarray | None) -> np.ndarray:
    if indices is None:
        return np.asarray([], dtype=np.int32)
    return np.asarray(indices, dtype=np.int32)


def _cell_lengths(trajectory: ProcessedTrajectory) -> np.ndarray | None:
    if trajectory.cell_lengths_A is None:
        return None
    cell = np.asarray(trajectory.cell_lengths_A, dtype=np.float32)
    if cell.size != 3:
        return None
    return cell.reshape((3,))


def _cell_lengths_or_empty(trajectory: ProcessedTrajectory) -> np.ndarray:
    cell = _cell_lengths(trajectory)
    if cell is None:
        return np.asarray([], dtype=np.float32)
    return cell


def _minimum_image_for_trajectory(
    trajectory: ProcessedTrajectory,
    delta: np.ndarray,
) -> np.ndarray:
    cell = _cell_lengths(trajectory)
    if cell is None:
        return np.asarray(delta, dtype=np.float32)
    return minimum_image_delta(delta, cell)


def _norms(trajectory: ProcessedTrajectory, delta: np.ndarray) -> np.ndarray:
    corrected = _minimum_image_for_trajectory(trajectory, delta)
    return np.linalg.norm(corrected, axis=-1)


def _frame_step_lengths(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 1:
        return np.asarray([], dtype=np.float32)
    return np.linalg.norm(np.diff(points, axis=0), axis=1)


def _crosses_sets(
    left: int,
    right: int,
    first: set[int],
    second: set[int],
) -> bool:
    return (left in first and right in second) or (left in second and right in first)


def _not_available_frame_table(trajectory: ProcessedTrajectory, note: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": np.arange(trajectory.frame_count, dtype=np.int32),
            "time_ps": trajectory.time_ps,
            "ligand_receptor_hbond_count": np.zeros((trajectory.frame_count,), dtype=np.int32),
            "water_involved_hbond_count": np.zeros((trajectory.frame_count,), dtype=np.int32),
            "available": False,
            "note": note,
        }
    )
