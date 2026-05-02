"""Preloaded Plotly trajectory preview helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from MDAnalysis.lib.distances import distance_array

ELEMENT_COLORS = {
    "H": "#f2f2f2",
    "C": "#8c8c8c",
    "N": "#3050f8",
    "O": "#ff0d0d",
    "P": "#ff8c00",
    "S": "#ffff30",
}


@dataclass(frozen=True)
class TrajectoryPreview:
    """Tables and Plotly figures for the trajectory preview cell."""

    playback_df: pd.DataFrame
    controls_df: pd.DataFrame
    element_df: pd.DataFrame
    ligand_motion_summary_df: pd.DataFrame
    ligand_motion_df: pd.DataFrame
    ligand_motion_figure: go.Figure
    trajectory_figure: go.Figure


def build_trajectory_preview(
    prepared_artifact,
    trajectory_record,
    *,
    viewer_height: int,
    inspection_cutoff_angstrom: float,
    play_interval_ms: int,
    original_ligand_ghost_color: str,
) -> TrajectoryPreview:
    """Build tables and a preloaded Plotly animation from stored trajectory frames."""

    positions = np.asarray(trajectory_record.sampled_positions, dtype=np.float32)
    sampled_steps = np.asarray(trajectory_record.sampled_steps)
    sampled_time = np.asarray(trajectory_record.sampled_time, dtype=np.float32)
    displacement = positions - positions[0]
    atom_displacement = np.sqrt(np.sum(displacement * displacement, axis=2))
    rms_displacement = np.sqrt(np.mean(atom_displacement * atom_displacement, axis=1))
    max_atom_displacement = np.max(atom_displacement, axis=1)
    symbols = np.asarray(prepared_artifact.symbols).astype(str)
    hydrogen_count = int(np.count_nonzero(np.char.upper(symbols) == "H"))
    metadata = dict(trajectory_record.metadata)
    frame_dt_ps = float(sampled_time[1] - sampled_time[0]) if len(sampled_time) > 1 else 0.0
    frame_step = int(sampled_steps[1] - sampled_steps[0]) if len(sampled_steps) > 1 else 0

    playback_df = pd.DataFrame(
        [
            {
                "frames": int(positions.shape[0]),
                "frame_index_range": f"0..{positions.shape[0] - 1}",
                "first_step": int(sampled_steps[0]),
                "last_step": int(sampled_steps[-1]),
                "md_dt_ps": float(metadata.get("dt", np.nan)),
                "md_steps_per_frame": int(metadata.get("sample_interval", frame_step)),
                "frame_step": frame_step,
                "ps_per_frame": frame_dt_ps,
                "diagnostic_interval": metadata.get("diagnostic_interval"),
                "constraint_max_iterations": metadata.get("constraint_max_iterations"),
                "first_time_ps": float(sampled_time[0]),
                "last_time_ps": float(sampled_time[-1]),
                "elapsed_wall_seconds": metadata.get("elapsed_wall_seconds"),
                "simulated_ps_per_wall_second": metadata.get("simulated_ps_per_wall_second"),
                "hydrogen_atoms": hydrogen_count,
                "final_rms_displacement_A": float(rms_displacement[-1]),
                "max_atom_displacement_A": float(max_atom_displacement.max()),
            }
        ]
    )
    controls_df = pd.DataFrame(
        [
            {
                "control": "Plotly play",
                "meaning": "advance preloaded trajectory frames sequentially",
            },
            {"control": "Plotly pause", "meaning": "stop at the current frame"},
            {
                "control": "Plotly slider",
                "meaning": "manual frame selector; each frame is a stored MLX sample",
            },
            {
                "control": "cyan ATP ghost",
                "meaning": "static ATP pose at frame 0 for displacement reference",
            },
        ]
    )
    element_df = (
        pd.Series(symbols)
        .value_counts()
        .sort_index()
        .rename_axis("element")
        .reset_index(name="atom_count")
    )

    ligand_indices = np.flatnonzero(np.asarray(prepared_artifact.ligand_mask, dtype=bool))
    receptor_indices = np.flatnonzero(np.asarray(prepared_artifact.receptor_mask, dtype=bool))
    if ligand_indices.size and receptor_indices.size:
        first_frame_distances = distance_array(
            positions[0, ligand_indices], positions[0, receptor_indices]
        )
        nearby_receptor_indices = receptor_indices[
            np.min(first_frame_distances, axis=0) <= inspection_cutoff_angstrom
        ]
    else:
        nearby_receptor_indices = receptor_indices
    inspection_indices = np.asarray(
        sorted(set(nearby_receptor_indices.tolist()) | set(ligand_indices.tolist())),
        dtype=np.int32,
    )
    masses = np.asarray(prepared_artifact.masses, dtype=np.float64)

    ligand_com = _mass_weighted_center(positions, ligand_indices, masses)
    receptor_com = _mass_weighted_center(positions, receptor_indices, masses)
    pocket_com = _mass_weighted_center(positions, nearby_receptor_indices, masses)
    ligand_relative_to_receptor = ligand_com - receptor_com
    ligand_relative_to_pocket = ligand_com - pocket_com
    ligand_motion_df = pd.DataFrame(
        {
            "frame": np.arange(positions.shape[0], dtype=np.int32),
            "time_ps": sampled_time,
            "atp_com_displacement_A": np.linalg.norm(ligand_com - ligand_com[0], axis=1),
            "atp_com_vs_receptor_A": np.linalg.norm(
                ligand_relative_to_receptor - ligand_relative_to_receptor[0],
                axis=1,
            ),
            "atp_com_vs_pocket_A": np.linalg.norm(
                ligand_relative_to_pocket - ligand_relative_to_pocket[0],
                axis=1,
            ),
        }
    )
    ligand_motion_summary_df = pd.DataFrame(
        [
            {
                "final_atp_com_vs_pocket_A": float(
                    ligand_motion_df["atp_com_vs_pocket_A"].iloc[-1]
                ),
                "max_atp_com_vs_pocket_A": float(ligand_motion_df["atp_com_vs_pocket_A"].max()),
                "final_atp_com_vs_receptor_A": float(
                    ligand_motion_df["atp_com_vs_receptor_A"].iloc[-1]
                ),
                "max_atp_com_vs_receptor_A": float(ligand_motion_df["atp_com_vs_receptor_A"].max()),
                "interpretation": (
                    "translation is small when ATP stays bound; "
                    "rocking/stretching can still be visible"
                ),
            }
        ]
    )
    ligand_motion_figure = px.line(
        ligand_motion_df,
        x="time_ps",
        y=["atp_com_vs_pocket_A", "atp_com_vs_receptor_A"],
        title="ATP center-of-mass motion relative to receptor/pocket",
    )
    trajectory_figure = _build_plotly_trajectory_figure(
        prepared_artifact,
        positions,
        symbols=symbols,
        ligand_indices=ligand_indices,
        receptor_indices=receptor_indices,
        inspection_indices=inspection_indices,
        viewer_height=viewer_height,
        play_interval_ms=play_interval_ms,
        original_ligand_ghost_color=original_ligand_ghost_color,
    )
    return TrajectoryPreview(
        playback_df=playback_df,
        controls_df=controls_df,
        element_df=element_df,
        ligand_motion_summary_df=ligand_motion_summary_df,
        ligand_motion_df=ligand_motion_df,
        ligand_motion_figure=ligand_motion_figure,
        trajectory_figure=trajectory_figure,
    )


def _mass_weighted_center(frame_positions: np.ndarray, indices: np.ndarray, masses: np.ndarray):
    indices = np.asarray(indices, dtype=np.int32)
    weights = masses[indices]
    return np.sum(frame_positions[:, indices, :] * weights[None, :, None], axis=1) / weights.sum()


def _atom_colors(symbols: np.ndarray, indices: np.ndarray, fallback="#9aa0a6"):
    return [ELEMENT_COLORS.get(str(symbols[index]).upper(), fallback) for index in indices]


def _atom_sizes(symbols: np.ndarray, indices: np.ndarray, heavy_size=7, hydrogen_size=3):
    return [
        hydrogen_size if str(symbols[index]).upper() == "H" else heavy_size for index in indices
    ]


def _bond_segments(frame_positions, bonds, allowed_indices):
    allowed = set(int(index) for index in np.asarray(allowed_indices, dtype=np.int32).tolist())
    xs, ys, zs = [], [], []
    for i, j in np.asarray(bonds, dtype=np.int32):
        i = int(i)
        j = int(j)
        if i not in allowed or j not in allowed:
            continue
        xs.extend([float(frame_positions[i, 0]), float(frame_positions[j, 0]), None])
        ys.extend([float(frame_positions[i, 1]), float(frame_positions[j, 1]), None])
        zs.extend([float(frame_positions[i, 2]), float(frame_positions[j, 2]), None])
    return xs, ys, zs


def _atom_marker_trace(
    prepared_artifact,
    symbols: np.ndarray,
    name,
    frame_positions,
    indices,
    opacity=1.0,
    heavy_size=7,
    hydrogen_size=3,
):
    indices = np.asarray(indices, dtype=np.int32)
    return go.Scatter3d(
        name=name,
        mode="markers",
        x=frame_positions[indices, 0],
        y=frame_positions[indices, 1],
        z=frame_positions[indices, 2],
        marker={
            "size": _atom_sizes(
                symbols, indices, heavy_size=heavy_size, hydrogen_size=hydrogen_size
            ),
            "color": _atom_colors(symbols, indices),
            "opacity": opacity,
            "line": {"color": "#555", "width": 0.5},
        },
        text=[f"{prepared_artifact.atom_names[index]} {symbols[index]}" for index in indices],
        hoverinfo="text",
    )


def _build_plotly_trajectory_figure(
    prepared_artifact,
    positions: np.ndarray,
    *,
    symbols: np.ndarray,
    ligand_indices: np.ndarray,
    receptor_indices: np.ndarray,
    inspection_indices: np.ndarray,
    viewer_height: int,
    play_interval_ms: int,
    original_ligand_ghost_color: str,
):
    ligand_set = set(ligand_indices.tolist())
    ligand_bonds = np.asarray(
        [
            [int(i), int(j)]
            for i, j in np.asarray(prepared_artifact.bonds, dtype=np.int32)
            if int(i) in ligand_set and int(j) in ligand_set
        ],
        dtype=np.int32,
    )
    ligand_elements = [element for element in sorted(set(symbols[ligand_indices]))]
    ligand_element_indices = {
        element: ligand_indices[symbols[ligand_indices] == element] for element in ligand_elements
    }

    pocket_bond_x, pocket_bond_y, pocket_bond_z = _bond_segments(
        positions[0],
        prepared_artifact.bonds,
        np.intersect1d(inspection_indices, receptor_indices),
    )
    ligand_bond_x, ligand_bond_y, ligand_bond_z = _bond_segments(
        positions[0],
        ligand_bonds,
        ligand_indices,
    )

    figure_data = [
        go.Scatter3d(
            name="frame-0 pocket bonds",
            mode="lines",
            x=pocket_bond_x,
            y=pocket_bond_y,
            z=pocket_bond_z,
            line={"color": "rgba(120,120,120,0.35)", "width": 4},
            hoverinfo="skip",
        ),
        _atom_marker_trace(
            prepared_artifact,
            symbols,
            "frame-0 pocket atoms",
            positions[0],
            inspection_indices,
            opacity=0.30,
            heavy_size=4,
            hydrogen_size=2,
        ),
        go.Scatter3d(
            name="frame-0 ATP reference",
            mode="markers",
            x=positions[0, ligand_indices, 0],
            y=positions[0, ligand_indices, 1],
            z=positions[0, ligand_indices, 2],
            marker={
                "size": _atom_sizes(symbols, ligand_indices, 5, 2),
                "color": original_ligand_ghost_color,
                "opacity": 0.18,
            },
            hoverinfo="skip",
        ),
        go.Scatter3d(
            name="ATP bonds",
            mode="lines",
            x=ligand_bond_x,
            y=ligand_bond_y,
            z=ligand_bond_z,
            line={"color": "#444", "width": 6},
            hoverinfo="skip",
        ),
    ]
    dynamic_trace_indices = [len(figure_data) - 1]
    for element, indices in ligand_element_indices.items():
        figure_data.append(
            _atom_marker_trace(
                prepared_artifact,
                symbols,
                f"ATP {element}",
                positions[0],
                indices,
                opacity=1.0,
                heavy_size=9,
                hydrogen_size=4,
            )
        )
        dynamic_trace_indices.append(len(figure_data) - 1)

    frames = []
    for frame_index in range(positions.shape[0]):
        frame_positions = positions[frame_index]
        x, y, z = _bond_segments(frame_positions, ligand_bonds, ligand_indices)
        frame_traces = [
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line={"color": "#444", "width": 6},
                hoverinfo="skip",
            )
        ]
        for element, indices in ligand_element_indices.items():
            frame_traces.append(
                _atom_marker_trace(
                    prepared_artifact,
                    symbols,
                    f"ATP {element}",
                    frame_positions,
                    indices,
                    opacity=1.0,
                    heavy_size=9,
                    hydrogen_size=4,
                )
            )
        frames.append(
            go.Frame(name=str(frame_index), data=frame_traces, traces=dynamic_trace_indices)
        )

    slider_steps = [
        {
            "label": str(frame_index),
            "method": "animate",
            "args": [
                [str(frame_index)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for frame_index in range(positions.shape[0])
    ]
    fig = go.Figure(data=figure_data, frames=frames)
    fig.update_layout(
        title="ATP trajectory preview against frame-0 receptor pocket",
        width=None,
        height=viewer_height,
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "x (A)", "showbackground": False},
            "yaxis": {"title": "y (A)", "showbackground": False},
            "zaxis": {"title": "z (A)", "showbackground": False},
        },
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "y": 0,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": play_interval_ms, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "frame "},
                "pad": {"t": 38},
                "steps": slider_steps,
            }
        ],
    )
    return fig
