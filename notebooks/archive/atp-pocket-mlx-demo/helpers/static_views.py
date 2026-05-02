"""Static py3Dmol structure previews for the notebook."""

from __future__ import annotations

from pathlib import Path

import py3Dmol

from helpers.config import (
    LIGAND_RESNAMES,
    LIPID_RESNAMES,
    STRUCTURE_FORMAT_BY_SUFFIX,
    VIEWER_HEIGHT,
    VIEWER_WIDTH,
)


def style_macromolecule(view):
    """Style a protein/receptor structure for quick static inspection."""

    view.setBackgroundColor("white")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.addStyle({"hetflag": False}, {"stick": {"radius": 0.16, "color": "lightgray"}})
    view.addSurface(py3Dmol.VDW, {"opacity": 0.18, "color": "white"}, {"hetflag": False})

    for resname in LIGAND_RESNAMES:
        view.addStyle({"resn": resname}, {"stick": {"radius": 0.26, "colorscheme": "greenCarbon"}})
        view.addStyle({"resn": resname}, {"sphere": {"scale": 0.28, "colorscheme": "greenCarbon"}})
    for resname in LIPID_RESNAMES:
        view.addStyle({"resn": resname}, {"line": {"color": "gray"}})
    return view


def style_ligand(view):
    """Style a ligand-only SDF/MOL view."""

    view.setBackgroundColor("white")
    view.setStyle({}, {"stick": {"radius": 0.18}})
    view.addStyle({"elem": "P"}, {"sphere": {"scale": 0.30, "color": "orange"}})
    view.addStyle({"elem": "O"}, {"sphere": {"scale": 0.22, "color": "red"}})
    view.addStyle({"elem": "N"}, {"sphere": {"scale": 0.22, "color": "blue"}})
    return view


def infer_structure_format(path: str | Path) -> str:
    """Infer py3Dmol format from the file suffix."""

    return STRUCTURE_FORMAT_BY_SUFFIX.get(Path(path).suffix.lower(), "pdb")


def load_structure_view(
    path: str | Path | None = None,
    *,
    pdb_id: str = "6gq6",
    file_format: str | None = None,
    width: str = VIEWER_WIDTH,
    height: int = VIEWER_HEIGHT,
):
    """Return a py3Dmol static structure view from a local file or PDB id."""

    if path is None:
        view = py3Dmol.view(query=f"pdb:{pdb_id}", width=width, height=height)
        style_macromolecule(view)
    else:
        path = Path(path)
        if file_format is None:
            file_format = infer_structure_format(path)
        view = py3Dmol.view(width=width, height=height)
        view.addModel(path.read_text(), file_format)
        if file_format == "sdf":
            style_ligand(view)
        else:
            style_macromolecule(view)

    view.zoomTo()
    return view
