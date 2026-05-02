"""Notebook helpers for prepared MLX trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from atomistic_prep.io import VIEW_PDB_NAME, load_prepared_system
from atomistic_prep.schema import PreparedSystem

TRAJECTORY_NAME = "trajectory.npz"


@dataclass(frozen=True)
class PreparedTrajectoryRecord:
    """Minimal trajectory record for notebook visualization."""

    sampled_positions: np.ndarray
    sampled_steps: np.ndarray
    sampled_time: np.ndarray
    symbols: tuple[str, ...]
    metadata: dict
    diagnostic_steps: np.ndarray | None = None
    diagnostic_time: np.ndarray | None = None
    potential_energy: np.ndarray | None = None
    kinetic_energy: np.ndarray | None = None
    total_energy: np.ndarray | None = None
    potential_energy_by_term: dict[str, np.ndarray] | None = None
    temperature: np.ndarray | None = None
    pair_count: np.ndarray | None = None
    rebuild_count: np.ndarray | None = None
    constraint_max_error: np.ndarray | None = None


@dataclass(frozen=True)
class PreparedTrajectoryBundle:
    """Loaded prepared-system arrays plus a notebook-ready MDAnalysis universe."""

    universe: object
    prepared: PreparedSystem
    trajectory: PreparedTrajectoryRecord
    prepared_dir: Path
    view_path: Path
    trajectory_path: Path


def load_prepared_npz_trajectory(path: str | Path) -> PreparedTrajectoryRecord:
    """Load the trajectory arrays needed by notebook viewers without importing MLX."""

    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(np.asarray(data["metadata_json"])))
        symbols = tuple(str(item) for item in np.asarray(data["symbols"]).tolist())
        term_names_data = data["energy_term_names"] if "energy_term_names" in data.files else []
        term_names = tuple(str(item) for item in np.asarray(term_names_data).tolist())
        potential_energy_by_term = {
            name: np.asarray(data[f"energy_term::{name}"]) for name in term_names
        }
        diagnostic_steps, diagnostic_time = _load_diagnostic_axis(data, metadata)
        return PreparedTrajectoryRecord(
            sampled_positions=np.asarray(data["sampled_positions"], dtype=np.float32),
            sampled_steps=np.asarray(data["sampled_steps"]),
            sampled_time=np.asarray(data["sampled_time"]),
            diagnostic_steps=diagnostic_steps,
            diagnostic_time=diagnostic_time,
            symbols=symbols,
            metadata=metadata,
            potential_energy=np.asarray(data["potential_energy"]),
            kinetic_energy=np.asarray(data["kinetic_energy"]),
            total_energy=np.asarray(data["total_energy"]),
            potential_energy_by_term=potential_energy_by_term,
            temperature=np.asarray(data["temperature"]),
            pair_count=np.asarray(data["pair_count"]),
            rebuild_count=np.asarray(data["rebuild_count"]),
            constraint_max_error=np.asarray(data["constraint_max_error"]),
        )


def _load_diagnostic_axis(data, metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "diagnostic_steps" in data.files:
        diagnostic_steps = np.asarray(data["diagnostic_steps"])
    else:
        diagnostic_steps = np.arange(len(np.asarray(data["total_energy"])), dtype=np.int32)
    if "diagnostic_time" in data.files:
        diagnostic_time = np.asarray(data["diagnostic_time"])
    else:
        dt = float(metadata.get("dt", 1.0))
        diagnostic_time = diagnostic_steps.astype(np.float32) * dt
    return diagnostic_steps, diagnostic_time


def load_prepared_trajectory_bundle(prepared_dir: str | Path) -> PreparedTrajectoryBundle:
    """Load and cross-check all prepared artifacts for notebook visualization."""

    base_dir = Path(prepared_dir)
    view_path = base_dir / VIEW_PDB_NAME
    trajectory_path = base_dir / TRAJECTORY_NAME
    if not view_path.exists():
        msg = f"missing prepared visualization structure: {view_path}"
        raise FileNotFoundError(msg)
    if not trajectory_path.exists():
        msg = f"missing prepared MLX trajectory: {trajectory_path}"
        raise FileNotFoundError(msg)

    prepared = load_prepared_system(base_dir)
    record = load_prepared_npz_trajectory(trajectory_path)
    if record.sampled_positions.ndim != 3 or record.sampled_positions.shape[2] != 3:
        msg = "trajectory sampled_positions must have shape (n_frames, n_atoms, 3)"
        raise ValueError(msg)
    if record.sampled_positions.shape[1] != prepared.atom_count:
        msg = (
            "prepared_system atom count does not match trajectory atom count: "
            f"{prepared.atom_count} != {record.sampled_positions.shape[1]}"
        )
        raise ValueError(msg)
    prepared_symbols = tuple(str(item) for item in prepared.symbols.tolist())
    if record.symbols and record.symbols != prepared_symbols:
        msg = "prepared_system symbols do not match trajectory symbols"
        raise ValueError(msg)

    metadata = dict(record.metadata)
    dt = metadata.get("dt")
    sample_interval = metadata.get("sample_interval")
    frame_dt = 1.0
    if dt is not None and sample_interval is not None:
        frame_dt = float(dt) * float(sample_interval)
    universe = make_mdanalysis_universe(view_path, record.sampled_positions, dt=frame_dt)
    if len(universe.atoms) != prepared.atom_count:
        msg = (
            "view.pdb atom count does not match prepared_system atom count: "
            f"{len(universe.atoms)} != {prepared.atom_count}"
        )
        raise ValueError(msg)
    return PreparedTrajectoryBundle(
        universe=universe,
        prepared=prepared,
        trajectory=record,
        prepared_dir=base_dir,
        view_path=view_path,
        trajectory_path=trajectory_path,
    )


def make_mdanalysis_universe(
    view_path: str | Path,
    positions: np.ndarray,
    *,
    dt: float = 1.0,
):
    """Create an independent MDAnalysis universe for one notebook viewer.

    NGLView advances frames by mutating the MDAnalysis trajectory cursor. Reusing
    one Universe across multiple live widgets can make independent players fight
    over that cursor, which appears as frame jumping in the notebook.
    """

    try:
        import MDAnalysis as mda
    except ImportError as exc:
        msg = "make_mdanalysis_universe requires MDAnalysis; install the viz extra."
        raise RuntimeError(msg) from exc

    coordinates = np.asarray(positions, dtype=np.float32).copy()
    if coordinates.ndim != 3 or coordinates.shape[2] != 3:
        msg = "positions must have shape (n_frames, n_atoms, 3)"
        raise ValueError(msg)
    universe = mda.Universe(str(view_path))
    universe.load_new(coordinates, order="fac", dt=float(dt))
    if len(universe.atoms) != coordinates.shape[1]:
        msg = (
            "view structure atom count does not match trajectory atom count: "
            f"{len(universe.atoms)} != {coordinates.shape[1]}"
        )
        raise ValueError(msg)
    return universe


def trajectory_to_multimodel_pdb(
    prepared: PreparedSystem,
    trajectory: PreparedTrajectoryRecord,
    *,
    max_frames: int | None = 250,
) -> str:
    """Serialize prepared trajectory frames as a browser-side multi-model PDB."""

    prepared.validate()
    positions = np.asarray(trajectory.sampled_positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[1:] != (prepared.atom_count, 3):
        msg = "trajectory sampled_positions must have shape (n_frames, n_atoms, 3)"
        raise ValueError(msg)
    if max_frames is not None and max_frames <= 0:
        msg = "max_frames must be positive or None"
        raise ValueError(msg)

    frame_count = positions.shape[0]
    stride = 1
    if max_frames is not None and frame_count > max_frames:
        stride = int(np.ceil(frame_count / max_frames))

    lines: list[str] = []
    for model_number, frame_index in enumerate(range(0, frame_count, stride), start=1):
        lines.append(f"MODEL     {model_number:4d}")
        frame = positions[frame_index]
        for atom_index in range(prepared.atom_count):
            record = "HETATM" if bool(prepared.ligand_mask[atom_index]) else "ATOM  "
            serial = atom_index + 1
            atom_name = str(prepared.atom_names[atom_index])[:4]
            resname = str(prepared.residue_names[atom_index])[:3]
            chain = (str(prepared.chain_ids[atom_index]) or "A")[:1]
            resid = int(prepared.residue_ids[atom_index])
            x, y, z = frame[atom_index]
            element = str(prepared.symbols[atom_index]).strip().upper()[:2].rjust(2)
            lines.append(
                f"{record}{serial:5d} {atom_name:^4s} {resname:>3s} {chain:1s}"
                f"{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {element:>2s}"
            )
        lines.append("ENDMDL")
    for bond in np.asarray(prepared.bonds, dtype=np.int32):
        i, j = int(bond[0]) + 1, int(bond[1]) + 1
        lines.append(f"CONECT{i:5d}{j:5d}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def py3dmol_frame_player_html(
    view: Any,
    *,
    sampled_steps: np.ndarray,
    sampled_time: np.ndarray,
    interval_ms: int = 100,
) -> str:
    """Return py3Dmol viewer HTML with visible frame controls."""

    if interval_ms <= 0:
        msg = "interval_ms must be positive"
        raise ValueError(msg)
    steps = [int(item) for item in np.asarray(sampled_steps).tolist()]
    times = [float(item) for item in np.asarray(sampled_time).tolist()]
    if not steps:
        msg = "sampled_steps must contain at least one frame"
        raise ValueError(msg)
    if len(times) != len(steps):
        msg = "sampled_time must match sampled_steps length"
        raise ValueError(msg)

    viewer_html = view.write_html()
    viewer_id = str(view.uniqueid)
    control_id = f"mlx_frame_player_{viewer_id}"
    steps_json = json.dumps(steps)
    times_json = json.dumps(times)
    return f"""
<div id="{control_id}"
     style="font: 13px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
            margin: 0 0 8px 0;">
  <button type="button" data-role="play" style="width: 74px;">Play</button>
  <input type="range" data-role="slider" min="0" max="{len(steps) - 1}"
         value="0" step="1" style="width: min(620px, 70%); vertical-align: middle;">
  <span data-role="label"
        style="display: inline-block; min-width: 220px; margin-left: 8px;">frame 0</span>
</div>
{viewer_html}
<script>
$3Dmolpromise.then(function() {{
  const root = document.getElementById("{control_id}");
  const playButton = root.querySelector('[data-role="play"]');
  const slider = root.querySelector('[data-role="slider"]');
  const label = root.querySelector('[data-role="label"]');
  const steps = {steps_json};
  const times = {times_json};
  const viewer = viewer_{viewer_id};
  let frame = 0;
  let timer = null;

  function setFrame(index) {{
    frame = Math.max(0, Math.min(steps.length - 1, Number(index)));
    slider.value = frame;
    label.textContent = "frame " + frame + "/" + (steps.length - 1)
      + " | step " + steps[frame]
      + " | time " + Number(times[frame]).toFixed(4);
    const update = viewer.setFrame(frame);
    if (update && typeof update.then === "function") {{
      update.then(function() {{ viewer.render(); }});
    }} else {{
      viewer.render();
    }}
  }}

  slider.addEventListener("input", function(event) {{
    setFrame(event.target.value);
  }});
  playButton.addEventListener("click", function() {{
    if (timer !== null) {{
      window.clearInterval(timer);
      timer = null;
      playButton.textContent = "Play";
      return;
    }}
    playButton.textContent = "Pause";
    timer = window.setInterval(function() {{
      setFrame((frame + 1) % steps.length);
    }}, {int(interval_ms)});
  }});
  setFrame(0);
}});
</script>
"""


def load_prepared_trajectory_universe(prepared_dir: str | Path):
    """Load all prepared artifacts and return the historical `(universe, record)` pair."""

    bundle = load_prepared_trajectory_bundle(prepared_dir)
    return bundle.universe, bundle.trajectory


__all__ = [
    "TRAJECTORY_NAME",
    "PreparedTrajectoryBundle",
    "PreparedTrajectoryRecord",
    "load_prepared_npz_trajectory",
    "load_prepared_trajectory_bundle",
    "load_prepared_trajectory_universe",
    "make_mdanalysis_universe",
    "py3dmol_frame_player_html",
    "trajectory_to_multimodel_pdb",
]
