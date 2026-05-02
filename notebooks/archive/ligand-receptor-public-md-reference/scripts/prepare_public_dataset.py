"""Download and process public ligand-receptor trajectory datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from helpers.datasets import (  # noqa: E402
    dataset_status,
    default_dataset_id,
    download_dataset,
    load_manifest,
    process_cached_dataset,
)
from helpers.motion_analysis import motion_gate_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download/cache and process a public ligand-receptor MD trajectory.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=NOTEBOOK_DIR / "datasets.json",
        help="Dataset manifest path.",
    )
    parser.add_argument("--dataset", default=None, help="Dataset id. Defaults to manifest default.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=NOTEBOOK_DIR / "data/cache",
        help="Raw downloaded topology/trajectory cache directory.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=NOTEBOOK_DIR / "data/processed",
        help="Processed subset output directory.",
    )
    parser.add_argument("--download", action="store_true", help="Download/cache raw files.")
    parser.add_argument("--process", action="store_true", help="Build processed subset.")
    parser.add_argument("--force", action="store_true", help="Overwrite cached/processed files.")
    parser.add_argument("--stride", type=int, default=10, help="Frame stride for the subset.")
    parser.add_argument("--max-frames", type=int, default=500, help="Maximum processed frames.")
    parser.add_argument(
        "--pocket-cutoff",
        type=float,
        default=6.0,
        help="First-frame receptor pocket cutoff in Angstrom.",
    )
    parser.add_argument("--list", action="store_true", help="List manifest datasets and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = load_manifest(args.manifest)
    dataset_id = args.dataset or default_dataset_id(args.manifest)
    if args.list:
        for dataset in datasets.values():
            total_size_mb = (
                dataset.topology.size_bytes + dataset.trajectory.size_bytes
            ) / 1_000_000
            print(f"{dataset.id}\t{total_size_mb:.1f} MB\t{dataset.title}")
            print(f"  source: {dataset.source_url}")
        return 0
    if dataset_id not in datasets:
        raise SystemExit(f"dataset {dataset_id!r} is not in {args.manifest}")
    dataset = datasets[dataset_id]
    if not args.download and not args.process:
        args.download = True
        args.process = True

    status = dataset_status(
        dataset,
        cache_dir=args.cache_dir,
        processed_dir=args.processed_dir,
    )
    print(f"Dataset: {dataset.id}")
    print(f"Source: {dataset.source_url}")
    print(f"Topology cached: {status.topology_cached} ({status.topology_path})")
    print(f"Trajectory cached: {status.trajectory_cached} ({status.trajectory_path})")
    print(f"Processed cached: {status.processed_cached} ({status.processed_path})")

    if args.download:
        topology_path, trajectory_path = download_dataset(
            dataset,
            cache_dir=args.cache_dir,
            force=args.force,
        )
        print(f"Downloaded topology: {topology_path}")
        print(f"Downloaded trajectory: {trajectory_path}")

    if args.process:
        processed = process_cached_dataset(
            dataset,
            cache_dir=args.cache_dir,
            processed_dir=args.processed_dir,
            stride=args.stride,
            max_frames=args.max_frames,
            pocket_cutoff_A=args.pocket_cutoff,
            force=args.force,
        )
        report = motion_gate_report(
            processed,
            min_ligand_com_displacement_A=dataset.motion_gate.min_ligand_com_displacement_A,
            min_contact_count_delta=dataset.motion_gate.min_contact_count_delta,
            contact_cutoff_A=dataset.motion_gate.contact_cutoff_A,
        )
        output_path = dataset_status(
            dataset,
            cache_dir=args.cache_dir,
            processed_dir=args.processed_dir,
        ).processed_path
        print(f"Processed subset: {output_path}")
        print(f"Frames: {report['frames']}")
        print(f"Atoms: {report['atoms']}")
        print(
            "Max ligand COM displacement A: "
            f"{report['max_ligand_com_displacement_A']:.3f}"
        )
        print(f"Contact count delta: {report['contact_count_delta']}")
        print(f"Passes visible-motion gate: {report['passes_motion_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
