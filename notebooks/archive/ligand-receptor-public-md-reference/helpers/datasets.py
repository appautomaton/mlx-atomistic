"""Dataset manifest, cache, and processing helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .motion_analysis import (
    ProcessedTrajectory,
    align_trajectory_to_reference,
    motion_gate_report,
    save_processed_trajectory,
)


@dataclass(frozen=True)
class DatasetResource:
    """One downloadable topology or trajectory resource."""

    filename: str
    url: str
    size_bytes: int
    checksum: str

    @property
    def checksum_algorithm(self) -> str:
        return self.checksum.split(":", 1)[0]

    @property
    def checksum_value(self) -> str:
        return self.checksum.split(":", 1)[1]


@dataclass(frozen=True)
class MotionGate:
    """Minimum visible-motion criteria for a processed trajectory."""

    min_ligand_com_displacement_A: float = 8.0
    min_contact_count_delta: int = 10
    contact_cutoff_A: float = 4.5


@dataclass(frozen=True)
class PublicTrajectoryDataset:
    """Manifest entry for a public ligand-receptor trajectory dataset."""

    id: str
    title: str
    source_url: str
    doi: str
    license: str
    description: str
    topology: DatasetResource
    trajectory: DatasetResource
    receptor_selection: str
    ligand_selection: str
    align_selection: str
    motion_gate: MotionGate


@dataclass(frozen=True)
class DatasetStatus:
    """Cache/processed status for a dataset."""

    dataset: PublicTrajectoryDataset
    topology_path: Path
    trajectory_path: Path
    processed_path: Path
    topology_cached: bool
    trajectory_cached: bool
    processed_cached: bool

    @property
    def ready_to_process(self) -> bool:
        return self.topology_cached and self.trajectory_cached

    @property
    def ready_to_visualize(self) -> bool:
        return self.processed_cached


def load_manifest(path: str | Path) -> dict[str, PublicTrajectoryDataset]:
    """Load and validate a dataset manifest."""

    payload = json.loads(Path(path).read_text())
    datasets = {}
    for row in payload.get("datasets", []):
        dataset = _dataset_from_json(row)
        datasets[dataset.id] = dataset
    if not datasets:
        msg = "dataset manifest must contain at least one dataset"
        raise ValueError(msg)
    default_id = payload.get("default_dataset")
    if default_id is not None and default_id not in datasets:
        msg = f"default_dataset {default_id!r} is not present in datasets"
        raise ValueError(msg)
    return datasets


def default_dataset_id(path: str | Path) -> str:
    """Return the manifest default dataset id."""

    payload = json.loads(Path(path).read_text())
    default_id = payload.get("default_dataset")
    if not default_id:
        msg = "dataset manifest is missing default_dataset"
        raise ValueError(msg)
    return str(default_id)


def dataset_status(
    dataset: PublicTrajectoryDataset,
    *,
    cache_dir: str | Path,
    processed_dir: str | Path,
) -> DatasetStatus:
    """Return cache and processed-output status for a dataset."""

    cache = Path(cache_dir) / dataset.id
    processed = Path(processed_dir) / dataset.id
    topology_path = cache / dataset.topology.filename
    trajectory_path = cache / dataset.trajectory.filename
    processed_path = processed / "processed_trajectory.npz"
    return DatasetStatus(
        dataset=dataset,
        topology_path=topology_path,
        trajectory_path=trajectory_path,
        processed_path=processed_path,
        topology_cached=topology_path.exists(),
        trajectory_cached=trajectory_path.exists(),
        processed_cached=processed_path.exists(),
    )


def download_dataset(
    dataset: PublicTrajectoryDataset,
    *,
    cache_dir: str | Path,
    force: bool = False,
) -> tuple[Path, Path]:
    """Download topology and trajectory resources into the local cache."""

    cache = Path(cache_dir) / dataset.id
    cache.mkdir(parents=True, exist_ok=True)
    topology_path = _download_resource(dataset.topology, cache, force=force)
    trajectory_path = _download_resource(dataset.trajectory, cache, force=force)
    return topology_path, trajectory_path


def process_cached_dataset(
    dataset: PublicTrajectoryDataset,
    *,
    cache_dir: str | Path,
    processed_dir: str | Path,
    stride: int = 10,
    max_frames: int = 500,
    pocket_cutoff_A: float = 6.0,
    force: bool = False,
) -> ProcessedTrajectory:
    """Build a lightweight receptor-pocket plus ligand processed trajectory."""

    if stride <= 0:
        msg = "stride must be positive"
        raise ValueError(msg)
    if max_frames <= 0:
        msg = "max_frames must be positive"
        raise ValueError(msg)

    status = dataset_status(dataset, cache_dir=cache_dir, processed_dir=processed_dir)
    if not status.ready_to_process:
        msg = (
            "dataset is not cached; run download first. Missing: "
            f"{status.topology_path if not status.topology_cached else ''} "
            f"{status.trajectory_path if not status.trajectory_cached else ''}"
        )
        raise FileNotFoundError(msg)
    if status.processed_path.exists() and not force:
        return ProcessedTrajectory.load(status.processed_path)

    try:
        import MDAnalysis as mda
        from MDAnalysis.lib.distances import distance_array
    except ImportError as exc:
        msg = "processing public trajectories requires the optional viz extra: uv sync --extra viz"
        raise RuntimeError(msg) from exc

    universe = mda.Universe(str(status.topology_path), str(status.trajectory_path))
    _add_pbc_cleanup_if_available(universe)
    receptor = universe.select_atoms(dataset.receptor_selection)
    ligand = universe.select_atoms(dataset.ligand_selection)
    align_atoms = universe.select_atoms(dataset.align_selection)
    if not len(receptor):
        msg = f"receptor selection matched zero atoms: {dataset.receptor_selection}"
        raise ValueError(msg)
    if not len(ligand):
        msg = f"ligand selection matched zero atoms: {dataset.ligand_selection}"
        raise ValueError(msg)
    if len(align_atoms) < 3:
        msg = f"alignment selection must match at least 3 atoms: {dataset.align_selection}"
        raise ValueError(msg)

    frame_indices = list(range(0, len(universe.trajectory), stride))[:max_frames]
    raw_positions = []
    sampled_time = []
    reference_align = None
    align_indices = align_atoms.indices
    receptor_indices = receptor.indices
    ligand_indices = ligand.indices
    for frame_index in frame_indices:
        ts = universe.trajectory[frame_index]
        if reference_align is None:
            reference_align = universe.atoms.positions[align_indices].copy()
        raw_positions.append(universe.atoms.positions.copy())
        sampled_time.append(float(getattr(ts, "time", frame_index)))

    aligned_positions = align_trajectory_to_reference(
        np.asarray(raw_positions, dtype=np.float32),
        align_indices=align_indices,
        reference_positions=np.asarray(reference_align, dtype=np.float32),
    )

    first_distances = distance_array(
        aligned_positions[0, ligand_indices],
        aligned_positions[0, receptor_indices],
    )
    pocket_receptor_indices = receptor_indices[
        np.min(first_distances, axis=0) <= pocket_cutoff_A
    ]
    if pocket_receptor_indices.size == 0:
        msg = (
            "pocket selection produced zero receptor atoms; check ligand/receptor "
            f"selections or increase pocket_cutoff_A={pocket_cutoff_A:g}"
        )
        raise ValueError(msg)
    keep_indices = np.asarray(
        sorted(set(pocket_receptor_indices.tolist()) | set(ligand_indices.tolist())),
        dtype=np.int32,
    )
    index_map = {int(index): local for local, index in enumerate(keep_indices.tolist())}
    local_ligand = np.asarray([index_map[int(index)] for index in ligand_indices], dtype=np.int32)
    local_receptor = np.asarray(
        [index_map[int(index)] for index in pocket_receptor_indices],
        dtype=np.int32,
    )
    processed_positions = aligned_positions[:, keep_indices, :]
    atomgroup = universe.atoms[keep_indices]
    sampled_time_array = _monotonic_time_axis(np.asarray(sampled_time, dtype=np.float32))
    processed = ProcessedTrajectory(
        positions=processed_positions,
        time_ps=sampled_time_array,
        symbols=_atomgroup_elements(atomgroup),
        atom_names=np.asarray(atomgroup.names, dtype=str),
        residue_names=np.asarray(atomgroup.resnames, dtype=str),
        residue_ids=np.asarray(atomgroup.resids, dtype=np.int32),
        segment_ids=_atomgroup_segment_ids(atomgroup),
        ligand_indices=local_ligand,
        receptor_indices=local_receptor,
        source={
            "kind": "public_md",
            "dataset_id": dataset.id,
            "title": dataset.title,
            "source_url": dataset.source_url,
            "doi": dataset.doi,
            "license": dataset.license,
            "topology": str(status.topology_path),
            "trajectory": str(status.trajectory_path),
            "stride": stride,
            "pocket_cutoff_A": pocket_cutoff_A,
            "receptor_selection": dataset.receptor_selection,
            "ligand_selection": dataset.ligand_selection,
            "align_selection": dataset.align_selection,
            "raw_time_ps_first": float(sampled_time[0]),
            "raw_time_ps_last": float(sampled_time[-1]),
            "time_axis_note": (
                "raw trajectory timestamps were converted to a monotonic sampled axis"
            ),
        },
    )
    report = motion_gate_report(
        processed,
        min_ligand_com_displacement_A=dataset.motion_gate.min_ligand_com_displacement_A,
        min_contact_count_delta=dataset.motion_gate.min_contact_count_delta,
        contact_cutoff_A=dataset.motion_gate.contact_cutoff_A,
    )
    if not report["passes_motion_gate"]:
        msg = (
            "processed trajectory failed visible-motion gate: "
            f"max_ligand_com_displacement_A={report['max_ligand_com_displacement_A']:.3f}, "
            f"contact_count_delta={report['contact_count_delta']}"
        )
        raise ValueError(msg)

    status.processed_path.parent.mkdir(parents=True, exist_ok=True)
    save_processed_trajectory(status.processed_path, processed)
    return processed


def cache_instructions(status: DatasetStatus, *, manifest_path: str | Path) -> str:
    """Return markdown instructions for downloading and processing a dataset."""

    return f"""
No processed ligand-receptor trajectory is loaded.

Dataset: `{status.dataset.id}`  
Source: {status.dataset.source_url}  
License: `{status.dataset.license}`

Run:

```bash
uv sync --extra viz
uv run python notebooks/ligand-receptor-motion/scripts/prepare_public_dataset.py \\
  --manifest {manifest_path} \\
  --dataset {status.dataset.id} \\
  --download \\
  --process
```

Raw cache files are written under `notebooks/ligand-receptor-motion/data/cache/`.
Processed subsets are written under `notebooks/ligand-receptor-motion/data/processed/`.
Both are ignored by git.
"""


def _dataset_from_json(row: dict[str, Any]) -> PublicTrajectoryDataset:
    return PublicTrajectoryDataset(
        id=str(row["id"]),
        title=str(row["title"]),
        source_url=str(row["source_url"]),
        doi=str(row.get("doi", "")),
        license=str(row.get("license", "")),
        description=str(row.get("description", "")),
        topology=_resource_from_json(row["topology"]),
        trajectory=_resource_from_json(row["trajectory"]),
        receptor_selection=str(row["receptor_selection"]),
        ligand_selection=str(row["ligand_selection"]),
        align_selection=str(row["align_selection"]),
        motion_gate=MotionGate(**row.get("motion_gate", {})),
    )


def _resource_from_json(row: dict[str, Any]) -> DatasetResource:
    resource = DatasetResource(
        filename=str(row["filename"]),
        url=str(row["url"]),
        size_bytes=int(row["size_bytes"]),
        checksum=str(row["checksum"]),
    )
    if resource.checksum_algorithm not in {"md5", "sha256"}:
        msg = f"unsupported checksum algorithm: {resource.checksum_algorithm}"
        raise ValueError(msg)
    if not resource.url.startswith(("https://", "http://")):
        msg = f"resource URL must be http(s): {resource.url}"
        raise ValueError(msg)
    return resource


def _download_resource(resource: DatasetResource, target_dir: Path, *, force: bool) -> Path:
    path = target_dir / resource.filename
    if path.exists() and not force:
        _verify_checksum(path, resource)
        return path
    temp_path = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(resource.url, timeout=60) as response, temp_path.open(
        "wb"
    ) as handle:
        shutil.copyfileobj(response, handle)
    temp_path.replace(path)
    _verify_checksum(path, resource)
    return path


def _verify_checksum(path: Path, resource: DatasetResource) -> None:
    if path.stat().st_size != resource.size_bytes:
        msg = (
            f"size mismatch for {path}: expected {resource.size_bytes} bytes, "
            f"got {path.stat().st_size}"
        )
        raise ValueError(msg)
    digest = hashlib.new(resource.checksum_algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != resource.checksum_value:
        msg = f"checksum mismatch for {path}: expected {resource.checksum}"
        raise ValueError(msg)


def _add_pbc_cleanup_if_available(universe: Any) -> None:
    """Use MDAnalysis unwrap when topology/box data are present.

    Public protein-ligand packages are often already stripped/postprocessed.
    When bond/box information is unavailable, receptor-frame alignment still
    removes global translation/rotation without inventing coordinates.
    """

    try:
        from MDAnalysis.transformations import unwrap

        _ = universe.atoms.bonds
        if len(universe.trajectory) and universe.trajectory[0].dimensions is not None:
            universe.trajectory.add_transformations(unwrap(universe.atoms))
    except Exception:
        return


def _atomgroup_elements(atomgroup: Any) -> np.ndarray:
    try:
        elements = np.asarray(atomgroup.elements).astype(str)
        if np.all(elements != ""):
            return elements
    except Exception:
        pass
    inferred = [_infer_element_from_name(name) for name in np.asarray(atomgroup.names).astype(str)]
    return np.asarray(inferred, dtype=str)


def _atomgroup_segment_ids(atomgroup: Any) -> np.ndarray:
    try:
        return np.asarray(atomgroup.segids, dtype=str)
    except Exception:
        return np.asarray([""] * len(atomgroup), dtype=str)


def _infer_element_from_name(atom_name: str) -> str:
    stripped = atom_name.strip().upper()
    if not stripped:
        return "C"
    stripped = stripped.lstrip("0123456789")
    if not stripped:
        return "C"
    if len(stripped) >= 2 and stripped[:2] in {"CL", "BR", "NA", "MG", "ZN", "CA", "FE"}:
        return stripped[:2].title()
    return stripped[0]


def _monotonic_time_axis(raw_time_ps: np.ndarray) -> np.ndarray:
    """Return a strictly nondecreasing sampled time axis in ps.

    Some public merged XTC files preserve segment-local timestamps that reset at
    concatenation boundaries. The processed notebook subset needs a playback and
    plot axis that advances with frame order, so use the positive median spacing
    when a reset is detected.
    """

    raw_time_ps = np.asarray(raw_time_ps, dtype=np.float32)
    if raw_time_ps.size <= 1:
        return raw_time_ps
    differences = np.diff(raw_time_ps)
    if np.all(differences >= 0):
        return raw_time_ps - raw_time_ps[0]
    positive = differences[differences > 0]
    spacing = float(np.median(positive)) if positive.size else 1.0
    return np.arange(raw_time_ps.size, dtype=np.float32) * np.float32(spacing)
