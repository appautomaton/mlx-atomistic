"""Configuration for the macromolecule visualization notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LIGAND_RESNAMES = ["ATP", "ADP", "ANP", "ACP", "MG"]
LIPID_RESNAMES = ["POPC", "POPE", "POPS", "POPG", "DPPC", "DOPC", "CHOL", "CARD", "CDL"]
RECEPTOR_SELECTION = "protein"
LIGAND_SELECTION = "resname ATP or resname ADP or resname ANP"
VIEWER_WIDTH = "100%"
VIEWER_HEIGHT = 560
STRUCTURE_FORMAT_BY_SUFFIX = {
    ".cif": "cif",
    ".mmcif": "cif",
    ".pdb": "pdb",
    ".ent": "pdb",
    ".sdf": "sdf",
    ".mol": "sdf",
}


@dataclass(frozen=True)
class NotebookPaths:
    """Paths used by the bundled 4DW1 ATP-pocket notebook workflow."""

    data_dir: Path
    atp_receptor_pdb: Path
    atp_ligand_sdf: Path
    prepared_dir: Path
    prepared_trajectory: Path


@dataclass(frozen=True)
class MDProtocol:
    """Short notebook-native production trajectory protocol."""

    steps: int = 10000
    sample_interval: int = 25
    dt_ps: float = 0.002
    temperature_k: float = 300.0
    friction_per_ps: float = 10.0
    seed: int = 7
    restraint_k: float = 5.0
    minimize_steps: int = 50
    equilibration_steps: int = 100
    constraint_max_iterations: int = 4
    diagnostic_interval: int = 25


@dataclass(frozen=True)
class PreviewSettings:
    """Visualization settings for static and trajectory previews."""

    viewer_width: str = VIEWER_WIDTH
    viewer_height: int = VIEWER_HEIGHT
    prep_cutoff_angstrom: float = 8.0
    inspection_cutoff_angstrom: float = 5.0
    trajectory_play_interval_ms: int = 250
    original_ligand_ghost_color: str = "cyan"


def find_data_dir() -> Path:
    """Find the notebook data directory from repo-root or notebook-local cwd."""

    for candidate in [Path("notebooks/macromolecule-viz/data"), Path("data")]:
        if candidate.exists():
            return candidate
    msg = "Could not find macromolecule-viz data directory from this notebook working directory."
    raise FileNotFoundError(msg)


def notebook_paths(data_dir: Path | None = None) -> NotebookPaths:
    """Return the standard bundled 4DW1 data and generated-artifact paths."""

    data_dir = find_data_dir() if data_dir is None else Path(data_dir)
    prepared_dir = data_dir / "prepared/4dw1-atp"
    return NotebookPaths(
        data_dir=data_dir,
        atp_receptor_pdb=data_dir / "4dw1_atp_bound_p2x4.pdb",
        atp_ligand_sdf=data_dir / "atp_pubchem_5957_3d.sdf",
        prepared_dir=prepared_dir,
        prepared_trajectory=prepared_dir / "trajectory.npz",
    )
