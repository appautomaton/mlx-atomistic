"""Notebook visualization helpers for ligand-receptor motion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .motion_analysis import (
    ProcessedTrajectory,
    ligand_com,
    minimum_image_delta,
    nearest_periodic_image,
    pbc_corrected_ligand_positions,
)

ELEMENT_COLORS = {
    "H": "#d8d8d8",
    "C": "#4f5965",
    "N": "#1f58d6",
    "O": "#dc2626",
    "P": "#f59e0b",
    "S": "#eab308",
    "CL": "#16a34a",
    "MG": "#10b981",
    "ZN": "#7c3aed",
}

COVALENT_RADII_A = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "P": 1.07,
    "S": 1.05,
    "CL": 1.02,
    "MG": 1.30,
    "ZN": 1.22,
}


def playback_table(trajectory: ProcessedTrajectory) -> pd.DataFrame:
    """Return frame/time metadata for the active trajectory."""

    if trajectory.frame_count > 1:
        ps_per_frame = float(np.median(np.diff(trajectory.time_ps)))
    else:
        ps_per_frame = 0.0
    source = trajectory.source
    return pd.DataFrame(
        [
            {
                "source": source.get("kind", "unknown"),
                "dataset_id": source.get("dataset_id", ""),
                "frames": trajectory.frame_count,
                "atoms": trajectory.atom_count,
                "ligand_atoms": int(trajectory.ligand_indices.size),
                "receptor_atoms": int(trajectory.receptor_indices.size),
                "replicas": source.get("replicas", 1),
                "selected_replica": source.get("selected_replica", 0),
                "gpu_visible_atoms": source.get("gpu_visible_atoms", trajectory.atom_count),
                "display": "PBC-corrected" if trajectory.cell_lengths_A is not None else "raw",
                "first_time_ps": float(trajectory.time_ps[0]),
                "last_time_ps": float(trajectory.time_ps[-1]),
                "ps_per_frame": ps_per_frame,
                "aggregate_steps_per_s": source.get("aggregate_steps_per_s"),
            }
        ]
    )


def make_ligand_motion_figure(
    trajectory: ProcessedTrajectory,
    *,
    title: str = "Ligand translation in aligned receptor coordinates",
    frame_interval_ms: int = 180,
) -> go.Figure:
    """Create a Plotly 3D animation with one active ligand pose and a COM trail."""

    positions = np.asarray(trajectory.positions, dtype=np.float32)
    ligand_indices = np.asarray(trajectory.ligand_indices, dtype=np.int32)
    receptor_indices = _pocket_receptor_indices(trajectory, cutoff_A=8.0, max_atoms=2500)
    ligand_positions = pbc_corrected_ligand_positions(trajectory)
    centers = ligand_com(trajectory)
    receptor_positions = nearest_periodic_image(
        positions[0, receptor_indices, :],
        reference=centers[0],
        cell_lengths=trajectory.cell_lengths_A,
    )
    ligand_symbols = _upper_symbols(trajectory.symbols[ligand_indices])
    receptor_symbols = _upper_symbols(trajectory.symbols[receptor_indices])
    ligand_bonds = infer_bonds(ligand_positions[0], ligand_symbols)
    receptor_bonds = infer_bonds(receptor_positions, receptor_symbols)
    receptor_colors = [_color_for_element(symbol) for symbol in receptor_symbols]
    ligand_colors = [_color_for_element(symbol) for symbol in ligand_symbols]
    solvent_indices = _nearby_water_indices(trajectory, cutoff_A=8.0, max_atoms=500)
    ion_indices = _nearby_ion_indices(trajectory, cutoff_A=10.0, max_atoms=200)
    lipid_indices = _nearby_lipid_indices(trajectory, cutoff_A=14.0, max_atoms=900)

    data = [
        go.Scatter3d(
            x=receptor_positions[:, 0],
            y=receptor_positions[:, 1],
            z=receptor_positions[:, 2],
            mode="markers",
            marker={
                "size": 3,
                "opacity": 0.34,
                "color": receptor_colors,
                "line": {"width": 0},
            },
            text=_hover_labels(trajectory, receptor_indices),
            hovertemplate="%{text}<extra>receptor pocket</extra>",
            name="receptor pocket atoms",
        ),
        _bond_trace(
            receptor_positions,
            receptor_bonds,
            color="#64748b",
            name="receptor pocket sticks",
            opacity=0.46,
            width=3,
        ),
        _residue_label_trace(trajectory, receptor_indices, receptor_positions, ligand_positions[0]),
        _ligand_marker_trace(ligand_positions[0], ligand_colors, trajectory, ligand_indices),
        _bond_trace(ligand_positions[0], ligand_bonds, color="#111827", name="active ligand bonds"),
        go.Scatter3d(
            x=ligand_positions[0, :, 0],
            y=ligand_positions[0, :, 1],
            z=ligand_positions[0, :, 2],
            mode="markers",
            marker={"size": 9, "opacity": 0.22, "color": "#2563eb"},
            text=_hover_labels(trajectory, ligand_indices),
            hovertemplate="%{text}<extra>initial ligand pose</extra>",
            name="initial ligand pose",
        ),
        _bond_trace(
            ligand_positions[0],
            ligand_bonds,
            color="#2563eb",
            name="initial ligand bonds",
            opacity=0.22,
        ),
        go.Scatter3d(
            x=centers[:, 0],
            y=centers[:, 1],
            z=centers[:, 2],
            mode="lines",
            line={"color": "#7c3aed", "width": 5},
            name="ligand COM path",
            hoverinfo="skip",
            visible="legendonly",
        ),
        _com_progress_trace(centers, 0),
        go.Scatter3d(
            x=[centers[0, 0], centers[-1, 0]],
            y=[centers[0, 1], centers[-1, 1]],
            z=[centers[0, 2], centers[-1, 2]],
            mode="lines",
            line={"color": "#ef4444", "width": 4, "dash": "dash"},
            name="net displacement axis",
            hoverinfo="skip",
            visible="legendonly",
        ),
        _solvent_trace(
            trajectory,
            solvent_indices=solvent_indices,
            frame_index=0,
            reference=centers[0],
        ),
        _ion_trace(trajectory, ion_indices=ion_indices, frame_index=0, reference=centers[0]),
        _lipid_trace(
            trajectory,
            lipid_indices=lipid_indices,
            frame_index=0,
            reference=centers[0],
        ),
    ]
    frames = [
        go.Frame(
            name=str(frame_index),
            data=[
                _ligand_marker_trace(
                    ligand_positions[frame_index],
                    ligand_colors,
                    trajectory,
                    ligand_indices,
                    frame_index=frame_index,
                ),
                _bond_trace(
                    ligand_positions[frame_index],
                    ligand_bonds,
                    color="#111827",
                    name="active ligand bonds",
                ),
                _com_progress_trace(centers, frame_index),
                _solvent_trace(
                    trajectory,
                    solvent_indices=solvent_indices,
                    frame_index=frame_index,
                    reference=centers[frame_index],
                ),
                _ion_trace(
                    trajectory,
                    ion_indices=ion_indices,
                    frame_index=frame_index,
                    reference=centers[frame_index],
                ),
                _lipid_trace(
                    trajectory,
                    lipid_indices=lipid_indices,
                    frame_index=frame_index,
                    reference=centers[frame_index],
                ),
            ],
            traces=[3, 4, 8, 10, 11, 12],
        )
        for frame_index in range(trajectory.frame_count)
    ]
    figure = go.Figure(data=data, frames=frames)
    figure.update_layout(
        title=title,
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        height=720,
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "x (A)", "showbackground": False},
            "yaxis": {"title": "y (A)", "showbackground": False},
            "zaxis": {"title": "z (A)", "showbackground": False},
        },
        legend={"orientation": "h", "y": 0.98, "x": 0.02},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.02,
                "y": 0,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_interval_ms, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.18,
                "y": 0,
                "currentvalue": {"prefix": "frame "},
                "steps": [
                    {
                        "label": str(frame_index),
                        "method": "animate",
                        "args": [
                            [str(frame_index)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    }
                    for frame_index in range(trajectory.frame_count)
                ],
            }
        ],
    )
    return figure


def infer_bonds(positions: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    """Infer display-only bonds from first-frame distances."""

    positions = np.asarray(positions, dtype=np.float32)
    symbols = _upper_symbols(symbols)
    bonds: list[tuple[int, int]] = []
    for i in range(positions.shape[0]):
        radius_i = COVALENT_RADII_A.get(symbols[i], 0.76)
        for j in range(i + 1, positions.shape[0]):
            radius_j = COVALENT_RADII_A.get(symbols[j], 0.76)
            max_distance = 1.25 * (radius_i + radius_j) + 0.25
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            if 0.35 <= distance <= max_distance:
                bonds.append((i, j))
    return np.asarray(bonds, dtype=np.int32).reshape((-1, 2))


def _ligand_marker_trace(
    positions: np.ndarray,
    colors: list[str],
    trajectory: ProcessedTrajectory,
    ligand_indices: np.ndarray,
    *,
    frame_index: int = 0,
) -> go.Scatter3d:
    time_ps = float(trajectory.time_ps[frame_index])
    labels = _hover_labels(trajectory, ligand_indices)
    return go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        marker={"size": 8, "opacity": 0.95, "color": colors, "line": {"width": 1}},
        text=labels,
        hovertemplate=f"%{{text}}<br>time={time_ps:.3f} ps<extra>active ligand</extra>",
        name="active ligand pose",
    )


def _bond_trace(
    positions: np.ndarray,
    bonds: np.ndarray,
    *,
    color: str,
    name: str,
    opacity: float = 0.8,
    width: int = 5,
) -> go.Scatter3d:
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for i, j in bonds:
        x.extend([float(positions[i, 0]), float(positions[j, 0]), None])
        y.extend([float(positions[i, 1]), float(positions[j, 1]), None])
        z.extend([float(positions[i, 2]), float(positions[j, 2]), None])
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={"color": color, "width": width},
        opacity=opacity,
        name=name,
        hoverinfo="skip",
    )


def _residue_label_trace(
    trajectory: ProcessedTrajectory,
    receptor_indices: np.ndarray,
    receptor_positions: np.ndarray,
    ligand_positions: np.ndarray,
    *,
    max_labels: int = 5,
) -> go.Scatter3d:
    labels, centers = _closest_residue_labels(
        trajectory,
        receptor_indices,
        receptor_positions,
        ligand_positions,
        max_labels=max_labels,
    )
    return go.Scatter3d(
        x=centers[:, 0],
        y=centers[:, 1],
        z=centers[:, 2],
        mode="text",
        text=labels,
        textfont={"size": 11, "color": "#334155"},
        name="pocket residue labels",
        hoverinfo="skip",
        visible="legendonly",
    )


def _closest_residue_labels(
    trajectory: ProcessedTrajectory,
    receptor_indices: np.ndarray,
    receptor_positions: np.ndarray,
    ligand_positions: np.ndarray,
    *,
    max_labels: int,
) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for local_index, atom_index in enumerate(receptor_indices.tolist()):
        key = (
            f"{trajectory.segment_ids[atom_index]}:"
            f"{trajectory.residue_names[atom_index]}{trajectory.residue_ids[atom_index]}"
        )
        grouped.setdefault(key, []).append(local_index)

    rows = []
    for label, local_indices in grouped.items():
        residue_positions = receptor_positions[local_indices]
        delta = ligand_positions[:, None, :] - residue_positions[None, :, :]
        min_distance = float(np.sqrt(np.sum(delta * delta, axis=-1)).min())
        rows.append((min_distance, label, residue_positions.mean(axis=0)))
    rows.sort(key=lambda item: item[0])
    selected = rows[:max_labels]
    if not selected:
        return [], np.empty((0, 3), dtype=np.float32)
    labels = [label for _, label, _ in selected]
    centers = np.asarray([center for _, _, center in selected], dtype=np.float32)
    return labels, centers


def _com_progress_trace(centers: np.ndarray, frame_index: int) -> go.Scatter3d:
    current = centers[: frame_index + 1]
    return go.Scatter3d(
        x=current[:, 0],
        y=current[:, 1],
        z=current[:, 2],
        mode="lines+markers",
        line={"color": "#f97316", "width": 7},
        marker={"size": 4, "color": "#f97316"},
        name="current progress",
        hoverinfo="skip",
    )


def _hover_labels(trajectory: ProcessedTrajectory, indices: np.ndarray) -> list[str]:
    labels = []
    for index in indices:
        labels.append(
            f"{trajectory.segment_ids[index]}:{trajectory.residue_names[index]}"
            f"{trajectory.residue_ids[index]} {trajectory.atom_names[index]}"
        )
    return labels


def _color_for_element(symbol: str) -> str:
    return ELEMENT_COLORS.get(symbol.upper(), "#6b7280")


def _upper_symbols(symbols: np.ndarray) -> np.ndarray:
    return np.char.upper(np.asarray(symbols).astype(str))


def _nearby_water_indices(
    trajectory: ProcessedTrajectory,
    *,
    cutoff_A: float,
    max_atoms: int,
) -> np.ndarray:
    water_indices = (
        np.asarray([], dtype=np.int32)
        if trajectory.water_indices is None
        else np.asarray(trajectory.water_indices, dtype=np.int32)
    )
    if water_indices.size == 0:
        return water_indices
    ligand_center = ligand_com(trajectory)[0]
    water_positions = nearest_periodic_image(
        trajectory.positions[0, water_indices],
        reference=ligand_center,
        cell_lengths=trajectory.cell_lengths_A,
    )
    return _nearest_indices_by_distance(
        water_indices,
        water_positions,
        ligand_center,
        cutoff_A,
        max_atoms,
    )


def _pocket_receptor_indices(
    trajectory: ProcessedTrajectory,
    *,
    cutoff_A: float,
    max_atoms: int,
) -> np.ndarray:
    receptor_indices = np.asarray(trajectory.receptor_indices, dtype=np.int32)
    if receptor_indices.size == 0:
        return receptor_indices
    receptor_positions = np.asarray(trajectory.positions[0, receptor_indices], dtype=np.float32)
    ligand_positions = pbc_corrected_ligand_positions(trajectory)[0]
    delta = receptor_positions[:, None, :] - ligand_positions[None, :, :]
    if trajectory.cell_lengths_A is not None:
        delta = minimum_image_delta(delta, trajectory.cell_lengths_A)
    distances = np.linalg.norm(delta, axis=-1).min(axis=1)
    selected_order = np.flatnonzero(distances <= cutoff_A)
    selected_order = selected_order[np.argsort(distances[selected_order])]
    selected = receptor_indices[selected_order[:max_atoms]]
    return selected if selected.size else receptor_indices[: min(max_atoms, receptor_indices.size)]

def _nearby_ion_indices(
    trajectory: ProcessedTrajectory,
    *,
    cutoff_A: float,
    max_atoms: int,
) -> np.ndarray:
    ion_indices = _optional_display_indices(trajectory.ion_indices)
    if ion_indices.size == 0:
        return ion_indices
    ligand_center = ligand_com(trajectory)[0]
    ion_positions = nearest_periodic_image(
        trajectory.positions[0, ion_indices],
        reference=ligand_center,
        cell_lengths=trajectory.cell_lengths_A,
    )
    return _nearest_indices_by_distance(
        ion_indices,
        ion_positions,
        ligand_center,
        cutoff_A,
        max_atoms,
    )


def _nearby_lipid_indices(
    trajectory: ProcessedTrajectory,
    *,
    cutoff_A: float,
    max_atoms: int,
) -> np.ndarray:
    lipid_indices = _optional_display_indices(trajectory.lipid_indices)
    if lipid_indices.size == 0:
        return lipid_indices
    ligand_center = ligand_com(trajectory)[0]
    lipid_positions = nearest_periodic_image(
        trajectory.positions[0, lipid_indices],
        reference=ligand_center,
        cell_lengths=trajectory.cell_lengths_A,
    )
    return _nearest_indices_by_distance(
        lipid_indices,
        lipid_positions,
        ligand_center,
        cutoff_A,
        max_atoms,
    )


def _nearest_indices_by_distance(
    indices: np.ndarray,
    positions: np.ndarray,
    reference: np.ndarray,
    cutoff_A: float,
    max_atoms: int,
) -> np.ndarray:
    distances = np.linalg.norm(positions - reference[None, :], axis=1)
    within_cutoff = np.flatnonzero(distances <= cutoff_A)
    selected_order = within_cutoff[np.argsort(distances[within_cutoff])]
    return indices[selected_order[:max_atoms]]


def _solvent_trace(
    trajectory: ProcessedTrajectory,
    *,
    solvent_indices: np.ndarray,
    frame_index: int,
    reference: np.ndarray | None = None,
) -> go.Scatter3d:
    if solvent_indices.size == 0:
        return go.Scatter3d(x=[], y=[], z=[], mode="markers", name="nearby waters")
    if reference is None:
        reference = ligand_com(trajectory)[frame_index]
    positions = nearest_periodic_image(
        trajectory.positions[frame_index, solvent_indices],
        reference=reference,
        cell_lengths=trajectory.cell_lengths_A,
    )
    symbols = _upper_symbols(trajectory.symbols[solvent_indices])
    colors = ["#38bdf8" if symbol == "O" else "#e5e7eb" for symbol in symbols]
    sizes = [4 if symbol == "O" else 2 for symbol in symbols]
    return go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        marker={"size": sizes, "opacity": 0.46, "color": colors},
        text=_hover_labels(trajectory, solvent_indices),
        hovertemplate="%{text}<extra>nearby water</extra>",
        name="nearby waters",
    )


def _ion_trace(
    trajectory: ProcessedTrajectory,
    *,
    ion_indices: np.ndarray,
    frame_index: int,
    reference: np.ndarray | None = None,
) -> go.Scatter3d:
    if ion_indices.size == 0:
        return go.Scatter3d(x=[], y=[], z=[], mode="markers", name="ions")
    if reference is None:
        reference = ligand_com(trajectory)[frame_index]
    positions = nearest_periodic_image(
        trajectory.positions[frame_index, ion_indices],
        reference=reference,
        cell_lengths=trajectory.cell_lengths_A,
    )
    return go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        marker={"size": 8, "opacity": 0.72, "color": "#22c55e", "symbol": "diamond"},
        text=_hover_labels(trajectory, ion_indices),
        hovertemplate="%{text}<extra>ion</extra>",
        name="ions",
    )


def _lipid_trace(
    trajectory: ProcessedTrajectory,
    *,
    lipid_indices: np.ndarray,
    frame_index: int,
    reference: np.ndarray | None = None,
) -> go.Scatter3d:
    if lipid_indices.size == 0:
        return go.Scatter3d(x=[], y=[], z=[], mode="markers", name="selected membrane context")
    if reference is None:
        reference = ligand_com(trajectory)[frame_index]
    positions = nearest_periodic_image(
        trajectory.positions[frame_index, lipid_indices],
        reference=reference,
        cell_lengths=trajectory.cell_lengths_A,
    )
    return go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        marker={"size": 3, "opacity": 0.18, "color": "#a3a3a3"},
        text=_hover_labels(trajectory, lipid_indices),
        hovertemplate="%{text}<extra>selected membrane context</extra>",
        name="selected membrane context",
    )


def _optional_display_indices(indices: np.ndarray | None) -> np.ndarray:
    if indices is None:
        return np.asarray([], dtype=np.int32)
    return np.asarray(indices, dtype=np.int32)
