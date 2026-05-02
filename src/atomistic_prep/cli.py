"""Command line interface for atomistic preparation workflows."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from atomistic_prep.gpcrmd import (
    GPCRMD_IMPORT_REPORT_NAME,
    GPCRmdInspectionError,
    attempt_gpcrmd_prepared_artifact_import,
    gpcrmd_mlx_compatibility_report,
    gpcrmd_mlx_readiness_inventory,
    inspect_gpcrmd_cache,
    write_gpcrmd_import_report,
)
from atomistic_prep.io import JSON_NAME, save_prepared_system
from atomistic_prep.prepare import (
    MissingPrepDependencyError,
    ProductionPrepNotImplementedError,
    optional_prep_dependency_status,
    prepare_p2x4_atp,
)
from atomistic_prep.solvated_example import (
    DEFAULT_CONSTRAINT_MAX_ITERATIONS,
    ensure_solvated_ligand_receptor_example,
)
from atomistic_prep.t4l_benzene import prepare_t4l_benzene
from atomistic_prep.topology_import import (
    TopologyImportError,
    import_amber_prmtop,
    import_charmm_with_parmed,
)

TRAJECTORY_NAME = "trajectory.npz"
STEERED_TRAJECTORY_NAME = "steered_trajectory.npz"


def default_4dw1_path() -> Path | None:
    """Return the local notebook copy of 4DW1 if it exists."""

    candidates = [
        Path.cwd() / "notebooks/archive/atp-pocket-mlx-demo/data/4dw1_atp_bound_p2x4.pdb",
        Path.cwd() / "notebooks/macromolecule-viz/data/4dw1_atp_bound_p2x4.pdb",
        Path.cwd() / "data/4dw1_atp_bound_p2x4.pdb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def add_prepare_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "prepare-p2x4-atp",
        help="Prepare a 4DW1 ATP-bound receptor pocket artifact.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for prepared artifacts.",
    )
    parser.add_argument("--pdb", type=Path, default=None, help="Path to a local 4DW1 PDB file.")
    parser.add_argument("--cutoff", type=float, default=8.0, help="Pocket cutoff in Angstrom.")
    parser.add_argument(
        "--backend",
        default="production_mlx",
        choices=["production_mlx", "production", "generic_mlx"],
        help=(
            "Preparation backend. production_mlx builds the bundled internal-template "
            "4DW1 ATP pocket and fails closed for unsupported selections."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing prepared artifacts.",
    )
    parser.set_defaults(func=command_prepare_p2x4_atp)


def add_prepare_t4l_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "prepare-t4l-benzene",
        help="Write the bundled T4 lysozyme L99A / benzene SMD fixture artifact.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output prepared-artifact dir.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts.")
    parser.set_defaults(func=command_prepare_t4l_benzene)


def add_import_amber_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "import-amber",
        help="Import AMBER prmtop plus inpcrd/rst7 into a production MLX artifact.",
    )
    parser.add_argument("--prmtop", required=True, type=Path, help="AMBER topology file.")
    parser.add_argument("--coords", required=True, type=Path, help="AMBER inpcrd/rst7 coordinates.")
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output prepared-artifact directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing prepared artifacts.",
    )
    parser.set_defaults(func=command_import_amber)


def add_import_charmm_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "import-charmm",
        help="Import CHARMM PSF/parameter/coordinate files into a production MLX artifact.",
    )
    parser.add_argument("--psf", required=True, type=Path, help="CHARMM PSF topology file.")
    parser.add_argument(
        "--params",
        required=True,
        nargs="+",
        type=Path,
        help="CHARMM parameter/toppar files parsed by ParmEd.",
    )
    parser.add_argument("--coords", required=True, type=Path, help="Coordinate file for the PSF.")
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output prepared-artifact directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing prepared artifacts.",
    )
    parser.set_defaults(func=command_import_charmm)


def add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "run-mlx",
        help="Run short MLX NVT from a prepared artifact.",
    )
    parser.add_argument("--prepared", required=True, type=Path, help="Prepared artifact directory.")
    parser.add_argument("--out", type=Path, default=None, help="Trajectory output path.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nonbonded-cutoff", type=float, default=None)
    parser.add_argument("--coulomb-constant", type=float, default=None)
    parser.add_argument("--restraint-k", type=float, default=5.0)
    parser.add_argument("--receptor-mass-scale", type=float, default=1.0)
    parser.add_argument("--minimize-steps", type=int, default=50)
    parser.add_argument("--equilibration-steps", type=int, default=100)
    parser.add_argument("--constraint-max-iterations", type=int, default=4)
    parser.add_argument("--diagnostic-interval", type=int, default=None)
    parser.add_argument(
        "--require-production",
        action="store_true",
        help="Refuse to run artifacts that are not production force-field exports.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing trajectory.")
    parser.set_defaults(func=command_run_mlx)


def add_run_ligand_receptor_example_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "run-ligand-receptor-example",
        help="Build and run the bundled solvated ligand-receptor MLX NVT example.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for prepared artifact and trajectory.",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sample-interval", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--water-count", type=int, default=48)
    parser.add_argument("--minimize-steps", type=int, default=100)
    parser.add_argument("--equilibration-steps", type=int, default=250)
    parser.add_argument("--restraint-k", type=float, default=10.0)
    parser.add_argument(
        "--constraint-max-iterations",
        type=int,
        default=DEFAULT_CONSTRAINT_MAX_ITERATIONS,
    )
    parser.add_argument("--diagnostic-interval", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite generated outputs.")
    parser.set_defaults(func=command_run_ligand_receptor_example)


def add_run_ligand_receptor_replicas_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "run-ligand-receptor-replicas",
        help="Run batched independent MLX replicas of the solvated ligand-receptor example.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for prepared artifact and selected-replica trajectory.",
    )
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--selected-replica", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sample-interval", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--water-count", type=int, default=48)
    parser.add_argument("--minimize-steps", type=int, default=100)
    parser.add_argument("--equilibration-steps", type=int, default=250)
    parser.add_argument("--restraint-k", type=float, default=10.0)
    parser.add_argument(
        "--constraint-max-iterations",
        type=int,
        default=DEFAULT_CONSTRAINT_MAX_ITERATIONS,
    )
    parser.add_argument("--diagnostic-interval", type=int, default=None)
    parser.add_argument(
        "--save-all-replicas",
        action="store_true",
        help="Also save replicas_trajectory.npz with all sampled replica coordinates.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite generated outputs.")
    parser.set_defaults(func=command_run_ligand_receptor_replicas)


def add_profile_ligand_receptor_performance_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "profile-ligand-receptor-performance",
        help="Benchmark single and batched MLX replicas for the solvated example.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Profile output directory.")
    parser.add_argument("--durations-ps", nargs="+", type=float, default=[5.0, 50.0, 200.0])
    parser.add_argument("--replicas", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sample-interval", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--water-count", type=int, default=48)
    parser.add_argument("--minimize-steps", type=int, default=100)
    parser.add_argument("--equilibration-steps", type=int, default=250)
    parser.add_argument("--restraint-k", type=float, default=10.0)
    parser.add_argument(
        "--constraint-max-iterations",
        type=int,
        default=DEFAULT_CONSTRAINT_MAX_ITERATIONS,
    )
    parser.add_argument("--diagnostic-interval", type=int, default=None)
    parser.add_argument("--save-all-replicas", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not write performance_profile.json.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write performance_profile.csv.",
    )
    parser.set_defaults(func=command_profile_ligand_receptor_performance)


def add_gpcrmd_inspect_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "gpcrmd-inspect",
        help="Inspect a local GPCRmd cache or manifest without running simulation.",
    )
    parser.add_argument("--target", default=None, help="GPCRmd target ID.")
    parser.add_argument(
        "--cache",
        required=True,
        type=Path,
        help="Local GPCRmd package directory, downloaded file, or JSON manifest.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Optional target registry JSON.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--compatibility",
        action="store_true",
        help="Include the fail-closed MLX compatibility report.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero when any expected GPCRmd file is missing.",
    )
    parser.set_defaults(func=command_gpcrmd_inspect)


def add_gpcrmd_import_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "gpcrmd-import",
        help="Attempt GPCRmd cache import into MLX prepared-artifact format.",
    )
    parser.add_argument("--target", default=None, help="GPCRmd target ID.")
    parser.add_argument(
        "--cache",
        required=True,
        type=Path,
        help="Local GPCRmd package directory, downloaded file, or JSON manifest.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output directory for the report.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Optional target registry JSON.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--require-export",
        action="store_true",
        help="Exit nonzero when no prepared MLX artifact is exported.",
    )
    parser.set_defaults(func=command_gpcrmd_import)


def add_run_gpcrmd_mlx_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "run-gpcrmd-mlx",
        help="Import or load a GPCRmd artifact and run the short MLX NVT proof path.",
    )
    parser.add_argument("--target", default=None, help="GPCRmd target ID.")
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help=(
            "Local GPCRmd package directory, downloaded file, or JSON manifest. "
            "If omitted, --out must already contain a GPCRmd prepared artifact."
        ),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Optional target registry JSON.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output directory.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--restraint-k", type=float, default=5.0)
    parser.add_argument("--minimize-steps", type=int, default=50)
    parser.add_argument("--equilibration-steps", type=int, default=100)
    parser.add_argument("--constraint-max-iterations", type=int, default=4)
    parser.add_argument("--diagnostic-interval", type=int, default=None)
    parser.add_argument(
        "--electrostatics",
        choices=("pme", "short-range-prototype"),
        default="pme",
        help=(
            "GPCRmd electrostatics route. PME is required for production; "
            "short-range-prototype must be explicitly requested."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing trajectory.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.set_defaults(func=command_run_gpcrmd_mlx)


def add_benchmark_gpcrmd_mlx_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "benchmark-gpcrmd-mlx",
        help="Benchmark the GPCRmd MLX run path and write JSON/CSV timing rows.",
    )
    parser.add_argument("--target", default=None, help="GPCRmd target ID.")
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Local GPCRmd package directory, downloaded file, or JSON manifest.",
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=None,
        help="Existing GPCRmd prepared artifact to copy into each benchmark case.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Optional target registry JSON.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Benchmark output directory.")
    parser.add_argument("--durations-ps", nargs="+", type=float, default=[0.01])
    parser.add_argument(
        "--electrostatics-modes",
        default="artifact",
        help="Comma-separated artifact,cutoff,ewald_reference,pme comparison requests.",
    )
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--restraint-k", type=float, default=5.0)
    parser.add_argument("--minimize-steps", type=int, default=0)
    parser.add_argument("--equilibration-steps", type=int, default=0)
    parser.add_argument("--constraint-max-iterations", type=int, default=4)
    parser.add_argument("--diagnostic-interval", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite benchmark case outputs.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable payload.")
    parser.add_argument("--no-json-file", action="store_true", help="Do not write JSON report.")
    parser.add_argument("--no-csv", action="store_true", help="Do not write CSV report.")
    parser.set_defaults(func=command_benchmark_gpcrmd_mlx)


def add_run_steered_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "run-steered-mlx",
        help="Run MLX steered NVT from a prepared ligand-receptor artifact.",
    )
    parser.add_argument("--prepared", required=True, type=Path, help="Prepared artifact directory.")
    parser.add_argument("--out", type=Path, default=None, help="Trajectory output path.")
    parser.add_argument("--steps", type=int, default=25_000)
    parser.add_argument("--sample-interval", type=int, default=50)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--friction", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--restraint-k", type=float, default=20.0)
    parser.add_argument("--bias-k", type=float, default=200.0)
    parser.add_argument("--target-velocity", type=float, default=None)
    parser.add_argument("--minimize-steps", type=int, default=50)
    parser.add_argument("--equilibration-steps", type=int, default=100)
    parser.add_argument("--constraint-max-iterations", type=int, default=4)
    parser.add_argument("--diagnostic-interval", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing trajectory.")
    parser.set_defaults(func=command_run_steered_mlx)


def add_benchmark_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "benchmark-p2x4-atp",
        help="Benchmark bundled ATP-pocket MLX production runs.",
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=None,
        help="Prepared artifact directory. Defaults to the bundled notebook artifact if present.",
    )
    parser.add_argument("--pdb", type=Path, default=None, help="Path to a local 4DW1 PDB file.")
    parser.add_argument("--durations-ps", nargs="+", type=float, default=[1.0, 10.0, 20.0])
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--sample-interval", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--friction", type=float, default=10.0)
    parser.add_argument("--restraint-k", type=float, default=5.0)
    parser.add_argument("--minimize-steps", type=int, default=50)
    parser.add_argument("--equilibration-steps", type=int, default=100)
    parser.add_argument("--constraint-max-iterations", type=int, default=4)
    parser.add_argument("--diagnostic-interval", type=int, default=None)
    parser.set_defaults(func=command_benchmark_p2x4_atp)


def add_validate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "validate",
        help="Validate an MLX prepared artifact without running dynamics.",
    )
    parser.add_argument("--prepared", required=True, type=Path, help="Prepared artifact directory.")
    parser.add_argument(
        "--require-production",
        action="store_true",
        help="Require physical units and production force-field metadata.",
    )
    parser.set_defaults(func=command_validate)


def add_status_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "prep-deps",
        help="Show optional production preparation dependency availability.",
    )
    parser.set_defaults(func=command_prep_deps)


def command_prepare_p2x4_atp(args: argparse.Namespace) -> int:
    pdb_path = args.pdb or default_4dw1_path()
    if pdb_path is None:
        raise SystemExit(
            "No local 4DW1 PDB found. Pass --pdb "
            "notebooks/archive/atp-pocket-mlx-demo/data/4dw1_atp_bound_p2x4.pdb"
        )
    out = args.out
    if out.exists() and (out / JSON_NAME).exists() and not args.force:
        raise SystemExit(f"{out} already contains {JSON_NAME}; pass --force to overwrite.")
    try:
        prepared = prepare_p2x4_atp(
            pdb_path=pdb_path,
            cutoff_angstrom=args.cutoff,
            backend=args.backend,
        )
    except (MissingPrepDependencyError, ProductionPrepNotImplementedError) as exc:
        raise SystemExit(str(exc)) from exc
    save_prepared_system(prepared, out)
    report = prepared.metadata.compatibility_report
    print(f"Wrote prepared system: {out}")
    print(f"Atoms: {prepared.atom_count}")
    print(f"Production force field: {bool(report.get('production_force_field', False))}")
    print(f"Hydrogens: {int(report.get('hydrogen_count', 0))}")
    print(f"MLX-supported bonds: {prepared.bonds.shape[0]}")
    print(f"MLX-supported angles: {prepared.angles.shape[0]}")
    print(f"MLX-supported dihedrals: {prepared.dihedrals.shape[0]}")
    print(f"MLX-supported impropers: {prepared.impropers.shape[0]}")
    print(f"Nonbonded exceptions: {prepared.nonbonded_exception_pairs.shape[0]}")
    for warning in prepared.metadata.warnings:
        print(f"Warning: {warning}")
    return 0


def command_prepare_t4l_benzene(args: argparse.Namespace) -> int:
    _guard_prepared_output(args.out, force=args.force)
    prepared = prepare_t4l_benzene()
    save_prepared_system(prepared, args.out)
    _print_prepared_summary(prepared, args.out)
    return 0


def _guard_prepared_output(out: Path, *, force: bool) -> None:
    if out.exists() and (out / JSON_NAME).exists() and not force:
        raise SystemExit(f"{out} already contains {JSON_NAME}; pass --force to overwrite.")


def command_import_amber(args: argparse.Namespace) -> int:
    _guard_prepared_output(args.out, force=args.force)
    try:
        prepared = import_amber_prmtop(prmtop_path=args.prmtop, coords_path=args.coords)
    except TopologyImportError as exc:
        raise SystemExit(str(exc)) from exc
    save_prepared_system(prepared, args.out)
    _print_prepared_summary(prepared, args.out)
    return 0


def command_import_charmm(args: argparse.Namespace) -> int:
    _guard_prepared_output(args.out, force=args.force)
    try:
        prepared = import_charmm_with_parmed(
            psf_path=args.psf,
            params=args.params,
            coords_path=args.coords,
        )
    except TopologyImportError as exc:
        raise SystemExit(str(exc)) from exc
    save_prepared_system(prepared, args.out)
    _print_prepared_summary(prepared, args.out)
    return 0


def _print_prepared_summary(prepared, out: Path) -> None:
    report = prepared.metadata.compatibility_report
    print(f"Wrote prepared system: {out}")
    print(f"Atoms: {prepared.atom_count}")
    print(f"Hydrogens: {int(report.get('hydrogen_count', 0))}")
    print(f"Production force field: {bool(report.get('production_force_field', False))}")
    print(f"Supported terms: {', '.join(report.get('supported_terms', []))}")
    for warning in prepared.metadata.warnings:
        print(f"Warning: {warning}")


def command_run_mlx(args: argparse.Namespace) -> int:
    from atomistic_prep.runner import run_mlx
    from mlx_atomistic.io import load_npz_trajectory

    out = args.out or args.prepared / TRAJECTORY_NAME
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists; pass --force to overwrite.")
    result = run_mlx(
        args.prepared,
        out=out,
        steps=args.steps,
        sample_interval=args.sample_interval,
        dt=args.dt,
        temperature=args.temperature,
        friction=args.friction,
        seed=args.seed,
        nonbonded_cutoff=args.nonbonded_cutoff,
        coulomb_constant=args.coulomb_constant,
        restraint_k=args.restraint_k,
        receptor_mass_scale=args.receptor_mass_scale,
        require_production=args.require_production,
        minimize_steps=args.minimize_steps,
        equilibration_steps=args.equilibration_steps,
        constraint_max_iterations=args.constraint_max_iterations,
        diagnostic_interval=args.diagnostic_interval,
    )
    frame_count = int(np.asarray(result.sampled_positions).shape[0])
    record = load_npz_trajectory(out)
    metadata = record.metadata
    print(f"Wrote MLX trajectory: {out}")
    print(f"Frames: {frame_count}")
    print(f"Steps: {args.steps}")
    print(f"Sample interval: {args.sample_interval}")
    print(f"Diagnostic interval: {metadata.get('diagnostic_interval')}")
    print(f"Constraint max iterations: {metadata.get('constraint_max_iterations')}")
    if metadata.get("elapsed_wall_seconds") is not None:
        print(f"Elapsed wall seconds: {metadata['elapsed_wall_seconds']:.3f}")
    if metadata.get("integration_steps_per_second") is not None:
        print(f"Integration steps/s: {metadata['integration_steps_per_second']:.3f}")
    if metadata.get("simulated_ps_per_wall_second") is not None:
        print(f"Simulated ps/s: {metadata['simulated_ps_per_wall_second']:.3f}")
    print(
        "Max constraint error A: "
        f"{float(np.max(np.asarray(record.constraint_max_error))):.6g}"
    )
    return 0


def command_run_ligand_receptor_example(args: argparse.Namespace) -> int:
    from mlx_atomistic.io import load_npz_trajectory

    bundle = ensure_solvated_ligand_receptor_example(
        args.out,
        steps=args.steps,
        dt=args.dt,
        sample_interval=args.sample_interval,
        temperature=args.temperature,
        friction=args.friction,
        water_count=args.water_count,
        minimize_steps=args.minimize_steps,
        equilibration_steps=args.equilibration_steps,
        restraint_k=args.restraint_k,
        constraint_max_iterations=args.constraint_max_iterations,
        diagnostic_interval=args.diagnostic_interval,
        force=args.force,
        electrostatics=args.electrostatics,
    )
    prepared = bundle["prepared"]
    trajectory_path = bundle["trajectory_path"]
    record = load_npz_trajectory(trajectory_path)
    metadata = record.metadata
    frame_count = int(np.asarray(record.sampled_positions).shape[0])
    print(f"Prepared artifact: {bundle['prepared_dir']}")
    print(f"MLX trajectory: {trajectory_path}")
    print(f"Generated artifact: {bundle['generated_artifact']}")
    print(f"Generated trajectory: {bundle['generated_trajectory']}")
    print(f"Atoms: {prepared.atom_count}")
    print(f"Ligand atoms: {int(np.count_nonzero(prepared.ligand_mask))}")
    print(f"Receptor atoms: {int(np.count_nonzero(prepared.receptor_mask))}")
    print(f"Water atoms: {int(np.count_nonzero(prepared.water_mask))}")
    print(f"Ion atoms: {int(np.count_nonzero(prepared.ion_mask))}")
    print(f"Frames: {frame_count}")
    print(f"Simulated ps: {metadata.get('simulated_time_ps')}")
    print(f"Electrostatics: {metadata.get('electrostatics_model')}")
    print("PME: not implemented")
    if metadata.get("elapsed_wall_seconds") is not None:
        print(f"Elapsed wall seconds: {metadata['elapsed_wall_seconds']:.3f}")
    if metadata.get("integration_steps_per_second") is not None:
        print(f"Integration steps/s: {metadata['integration_steps_per_second']:.3f}")
    print(
        "Max constraint error A: "
        f"{float(np.max(np.asarray(record.constraint_max_error))):.6g}"
    )
    return 0


def command_run_ligand_receptor_replicas(args: argparse.Namespace) -> int:
    from atomistic_prep.replicas import run_ligand_receptor_replicas
    from mlx_atomistic.io import load_npz_trajectory

    summary = run_ligand_receptor_replicas(
        args.out,
        replicas=args.replicas,
        selected_replica=args.selected_replica,
        steps=args.steps,
        dt=args.dt,
        sample_interval=args.sample_interval,
        temperature=args.temperature,
        friction=args.friction,
        seed=args.seed,
        water_count=args.water_count,
        minimize_steps=args.minimize_steps,
        equilibration_steps=args.equilibration_steps,
        restraint_k=args.restraint_k,
        constraint_max_iterations=args.constraint_max_iterations,
        diagnostic_interval=args.diagnostic_interval,
        save_all_replicas=args.save_all_replicas,
        force=args.force,
    )
    record = load_npz_trajectory(summary.selected_trajectory_path)
    metadata = record.metadata
    print(f"Prepared artifact: {summary.prepared_dir}")
    print(f"Selected-replica MLX trajectory: {summary.selected_trajectory_path}")
    if summary.all_replicas_trajectory_path is not None:
        print(f"All-replica trajectory: {summary.all_replicas_trajectory_path}")
    print(f"Replicas: {metadata.get('replicas')}")
    print(f"Selected replica: {metadata.get('selected_replica')}")
    print(f"Atoms per replica: {metadata.get('atoms_per_replica')}")
    print(f"GPU-visible atoms: {metadata.get('gpu_visible_atoms')}")
    print(f"Frames: {int(np.asarray(record.sampled_positions).shape[0])}")
    print(f"Simulated ps per replica: {metadata.get('simulated_time_ps')}")
    print(f"Electrostatics: {metadata.get('electrostatics_model')}")
    print(f"Constraint max iterations: {metadata.get('constraint_max_iterations')}")
    if metadata.get("elapsed_wall_seconds") is not None:
        print(f"Elapsed wall seconds: {metadata['elapsed_wall_seconds']:.3f}")
    if metadata.get("per_replica_steps_per_second") is not None:
        print(f"Per-replica steps/s: {metadata['per_replica_steps_per_second']:.3f}")
    if metadata.get("aggregate_integration_steps_per_second") is not None:
        print(
            "Aggregate integration steps/s: "
            f"{metadata['aggregate_integration_steps_per_second']:.3f}"
        )
    if metadata.get("aggregate_simulated_ps_per_wall_second") is not None:
        print(
            "Aggregate simulated ps/s: "
            f"{metadata['aggregate_simulated_ps_per_wall_second']:.3f}"
        )
    print(
        "Max constraint error A: "
        f"{float(np.max(np.asarray(record.constraint_max_error))):.6g}"
    )
    return 0


def command_profile_ligand_receptor_performance(args: argparse.Namespace) -> int:
    from atomistic_prep.replicas import profile_ligand_receptor_performance

    rows = profile_ligand_receptor_performance(
        args.out,
        durations_ps=args.durations_ps,
        replica_counts=args.replicas,
        dt=args.dt,
        sample_interval=args.sample_interval,
        temperature=args.temperature,
        friction=args.friction,
        seed=args.seed,
        water_count=args.water_count,
        minimize_steps=args.minimize_steps,
        equilibration_steps=args.equilibration_steps,
        restraint_k=args.restraint_k,
        constraint_max_iterations=args.constraint_max_iterations,
        diagnostic_interval=args.diagnostic_interval,
        save_all_replicas=args.save_all_replicas,
        force=args.force,
        write_json=not args.no_json,
        write_csv=not args.no_csv,
    )
    print(
        "duration_ps,replicas,wall_s,per_replica_steps_per_s,"
        "aggregate_steps_per_s,aggregate_ps_per_s,max_constraint_error_A,artifact_size_bytes"
    )
    for row in rows:
        print(
            f"{row['duration_ps']:g},"
            f"{row['replicas']},"
            f"{row['wall_s']:.3f},"
            f"{row['per_replica_steps_per_s']:.3f},"
            f"{row['aggregate_steps_per_s']:.3f},"
            f"{row['aggregate_ps_per_s']:.3f},"
            f"{row['max_constraint_error_A']:.6g},"
            f"{row['artifact_size_bytes']}"
        )
    if not args.no_json:
        print(json.dumps({"json": str(args.out / "performance_profile.json")}))
    if not args.no_csv:
        print(json.dumps({"csv": str(args.out / "performance_profile.csv")}))
    return 0


def command_gpcrmd_inspect(args: argparse.Namespace) -> int:
    try:
        report = inspect_gpcrmd_cache(
            args.cache,
            target_id=args.target,
            registry_path=args.registry,
        )
    except GPCRmdInspectionError as exc:
        raise SystemExit(str(exc)) from exc

    payload = report.to_json_dict()
    if args.compatibility:
        payload["mlx_compatibility"] = gpcrmd_mlx_compatibility_report(report).to_json_dict()
        payload["mlx_readiness_inventory"] = gpcrmd_mlx_readiness_inventory(
            report
        ).to_json_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_gpcrmd_inspection(report)
        if args.compatibility:
            _print_gpcrmd_compatibility(gpcrmd_mlx_compatibility_report(report))
            _print_gpcrmd_readiness_inventory(gpcrmd_mlx_readiness_inventory(report))

    if args.require_complete and not report.complete:
        missing = ", ".join(str(file_id) for file_id in report.missing_file_ids)
        raise SystemExit(f"GPCRmd cache is missing expected file IDs: {missing}")
    return 0


def _print_gpcrmd_inspection(report) -> None:
    target = report.target
    print(f"Target: {target.target_id}")
    print(f"Dynamics ID: {target.dynamics_id}")
    print(f"PDB ID: {target.pdb_id}")
    print(f"Receptor: {target.receptor}")
    print(f"Cache: {report.cache_path}")
    print(f"Cache kind: {report.cache_kind}")
    print(f"Cache exists: {report.cache_exists}")
    print(f"Expected files: {len(report.file_statuses)}")
    print(f"Present files: {report.present_file_count}")
    print(f"Complete: {report.complete}")
    if report.missing_file_ids:
        print("Missing file IDs: " + ", ".join(str(item) for item in report.missing_file_ids))
    print("Files:")
    for status in report.file_statuses:
        marker = "present" if status.present else "missing"
        path = f" path={status.path}" if status.path else ""
        print(f"  {status.file_id} {status.role} {marker}{path}")


def command_gpcrmd_import(args: argparse.Namespace) -> int:
    try:
        attempt = attempt_gpcrmd_prepared_artifact_import(
            args.cache,
            args.out,
            target_id=args.target,
            registry_path=args.registry,
        )
    except GPCRmdInspectionError as exc:
        raise SystemExit(str(exc)) from exc

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / GPCRMD_IMPORT_REPORT_NAME
    write_gpcrmd_import_report(report_path, attempt)
    payload = attempt.to_json_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"GPCRmd import report: {report_path}")
        print(f"Target: {attempt.target_id}")
        print(f"Exported prepared artifact: {attempt.exported}")
        print(f"Prepared artifact path: {attempt.prepared_artifact_path}")
        print(f"Blockers: {', '.join(attempt.blockers) or 'none'}")
        print(f"Next engine slice: {attempt.compatibility_report.next_engine_slice}")
    if args.require_export and not attempt.exported:
        raise SystemExit("GPCRmd cache was not exported to an MLX prepared artifact.")
    return 0


def command_run_gpcrmd_mlx(args: argparse.Namespace) -> int:
    from atomistic_prep.runner import run_gpcrmd_mlx

    payload = run_gpcrmd_mlx(
        target_id=args.target,
        cache=args.cache,
        registry_path=args.registry,
        out=args.out,
        steps=args.steps,
        sample_interval=args.sample_interval,
        dt=args.dt,
        temperature=args.temperature,
        friction=args.friction,
        seed=args.seed,
        restraint_k=args.restraint_k,
        minimize_steps=args.minimize_steps,
        equilibration_steps=args.equilibration_steps,
        constraint_max_iterations=args.constraint_max_iterations,
        diagnostic_interval=args.diagnostic_interval,
        electrostatics=args.electrostatics,
        force=args.force,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"GPCRmd MLX run report: {payload['run_report_path']}")
        print(f"Target: {payload['target_id']}")
        print(f"Status: {payload['status']}")
        print(f"Prepared artifact path: {payload['prepared_artifact_path']}")
        print(f"Trajectory path: {payload['trajectory_path']}")
        print(f"Blockers: {', '.join(payload['blockers']) or 'none'}")
        diagnostics = payload.get("diagnostic_summary") or {}
        if diagnostics:
            print(f"Frames: {diagnostics.get('sampled_frame_count')}")
            print(f"Diagnostics: {diagnostics.get('diagnostic_count')}")
            print(f"Total energy finite: {diagnostics.get('total_energy_finite')}")
            print(f"Temperature finite: {diagnostics.get('temperature_finite')}")
        print(f"Max constraint error A: {diagnostics.get('max_constraint_error_A')}")
    return 0 if payload["status"] == "ran" else 1


def command_benchmark_gpcrmd_mlx(args: argparse.Namespace) -> int:
    from atomistic_prep.gpcrmd_benchmark import (
        GPCRMD_BENCHMARK_CSV_NAME,
        GPCRMD_BENCHMARK_JSON_NAME,
        benchmark_gpcrmd_mlx,
    )

    if args.cache is None and args.prepared is None:
        raise SystemExit("benchmark-gpcrmd-mlx requires --cache or --prepared.")
    payload = benchmark_gpcrmd_mlx(
        out=args.out,
        target_id=args.target,
        cache=args.cache,
        registry_path=args.registry,
        prepared=args.prepared,
        durations_ps=tuple(args.durations_ps),
        electrostatics_modes=tuple(
            item.strip() for item in args.electrostatics_modes.split(",") if item.strip()
        ),
        dt=args.dt,
        sample_interval=args.sample_interval,
        temperature=args.temperature,
        friction=args.friction,
        seed=args.seed,
        restraint_k=args.restraint_k,
        minimize_steps=args.minimize_steps,
        equilibration_steps=args.equilibration_steps,
        constraint_max_iterations=args.constraint_max_iterations,
        diagnostic_interval=args.diagnostic_interval,
        force=args.force,
        write_json=not args.no_json_file,
        write_csv=not args.no_csv,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("case,status,duration_ps,steps,wall_s,steps_per_s,ps_per_s,blockers")
        for row in payload["cases"]:
            print(
                f"{row['case']},"
                f"{row['status']},"
                f"{row['duration_ps']:g},"
                f"{row['steps']},"
                f"{_format_optional_float(row['total_wall_s'])},"
                f"{_format_optional_float(row['integration_steps_per_s'])},"
                f"{_format_optional_float(row['ps_per_s'])},"
                f"{row['blockers']}"
            )
        if not args.no_json_file:
            print(json.dumps({"json": str(args.out / GPCRMD_BENCHMARK_JSON_NAME)}))
        if not args.no_csv:
            print(json.dumps({"csv": str(args.out / GPCRMD_BENCHMARK_CSV_NAME)}))
    return 0 if payload["blocked_case_count"] == 0 else 1


def _format_optional_float(value: object) -> str:
    return "" if value is None else f"{float(value):.6g}"


def _print_gpcrmd_compatibility(report) -> None:
    print("MLX compatibility:")
    print(f"  Runnable now: {report.runnable_now}")
    print(f"  Supported now: {', '.join(report.supported_now)}")
    print(f"  Missing input: {', '.join(report.missing_input) or 'none'}")
    print(f"  Unsupported physics: {', '.join(report.unsupported_physics) or 'none'}")
    print(f"  Next engine slice: {report.next_engine_slice}")


def _print_gpcrmd_readiness_inventory(inventory) -> None:
    print("MLX readiness inventory:")
    print(f"  Target decision: {inventory.target_decision['status']}")
    print(
        "  Required import files: "
        + ", ".join(item["role"] for item in inventory.required_files)
    )
    print(
        "  Optional analysis features: "
        + ", ".join(item["feature"] for item in inventory.optional_analysis_features)
    )
    print(
        "  First engine blockers: "
        + ", ".join(item["name"] for item in inventory.first_engine_blockers)
    )


def command_run_steered_mlx(args: argparse.Namespace) -> int:
    from atomistic_prep.runner import run_steered_mlx
    from mlx_atomistic.io import load_npz_trajectory

    out = args.out or args.prepared / STEERED_TRAJECTORY_NAME
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists; pass --force to overwrite.")
    result = run_steered_mlx(
        args.prepared,
        out=out,
        steps=args.steps,
        sample_interval=args.sample_interval,
        dt=args.dt,
        temperature=args.temperature,
        friction=args.friction,
        seed=args.seed,
        restraint_k=args.restraint_k,
        bias_k=args.bias_k,
        target_velocity=args.target_velocity,
        minimize_steps=args.minimize_steps,
        equilibration_steps=args.equilibration_steps,
        constraint_max_iterations=args.constraint_max_iterations,
        diagnostic_interval=args.diagnostic_interval,
    )
    frame_count = int(np.asarray(result.sampled_positions).shape[0])
    record = load_npz_trajectory(out)
    metadata = record.metadata
    print(f"Wrote MLX steered trajectory: {out}")
    print(f"Frames: {frame_count}")
    print(f"Steps: {args.steps}")
    print(f"Sample interval: {args.sample_interval}")
    print(f"Simulated ps: {metadata.get('simulated_time_ps')}")
    print(f"Target velocity A/ps: {metadata.get('target_velocity_A_per_ps')}")
    print(f"Bias k: {metadata.get('bias_k')}")
    print(f"Final CV A: {float(np.asarray(result.sampled_cv)[-1]):.3f}")
    print(f"Final target A: {float(np.asarray(result.sampled_target)[-1]):.3f}")
    if metadata.get("elapsed_wall_seconds") is not None:
        print(f"Elapsed wall seconds: {metadata['elapsed_wall_seconds']:.3f}")
    return 0


def command_benchmark_p2x4_atp(args: argparse.Namespace) -> int:
    from atomistic_prep.runner import run_mlx
    from mlx_atomistic.io import load_npz_trajectory

    prepared = args.prepared
    temp_dir_context = None
    if prepared is None:
        default_prepared = Path("notebooks/archive/atp-pocket-mlx-demo/data/prepared/4dw1-atp")
        if default_prepared.exists():
            prepared = default_prepared
        else:
            pdb_path = args.pdb or default_4dw1_path()
            if pdb_path is None:
                raise SystemExit("No prepared artifact or bundled 4DW1 PDB found.")
            temp_dir_context = tempfile.TemporaryDirectory(prefix="mlx-p2x4-atp-bench-")
            prepared = Path(temp_dir_context.name) / "prepared"
            prepared_system = prepare_p2x4_atp(
                pdb_path=pdb_path,
                cutoff_angstrom=8.0,
                backend="production_mlx",
            )
            save_prepared_system(prepared_system, prepared)

    try:
        print("duration_ps,steps,frames,wall_s,steps_per_s,ps_per_s,max_constraint_error_A")
        for duration_ps in args.durations_ps:
            steps = int(round(float(duration_ps) / args.dt))
            with tempfile.TemporaryDirectory(prefix="mlx-p2x4-atp-run-") as run_dir:
                out = Path(run_dir) / "trajectory.npz"
                result = run_mlx(
                    prepared,
                    out=out,
                    steps=steps,
                    sample_interval=args.sample_interval,
                    dt=args.dt,
                    temperature=args.temperature,
                    friction=args.friction,
                    restraint_k=args.restraint_k,
                    minimize_steps=args.minimize_steps,
                    equilibration_steps=args.equilibration_steps,
                    constraint_max_iterations=args.constraint_max_iterations,
                    diagnostic_interval=args.diagnostic_interval,
                    require_production=True,
                )
                record = load_npz_trajectory(out)
                metadata = record.metadata
                print(
                    f"{duration_ps:g},"
                    f"{steps},"
                    f"{int(np.asarray(result.sampled_positions).shape[0])},"
                    f"{float(metadata.get('elapsed_wall_seconds', np.nan)):.3f},"
                    f"{float(metadata.get('integration_steps_per_second', np.nan)):.3f},"
                    f"{float(metadata.get('simulated_ps_per_wall_second', np.nan)):.3f},"
                    f"{float(np.max(np.asarray(record.constraint_max_error))):.6g}"
                )
    finally:
        if temp_dir_context is not None:
            temp_dir_context.cleanup()
    return 0


def command_validate(args: argparse.Namespace) -> int:
    from mlx_atomistic.artifacts import load_prepared_mlx_artifact

    artifact = load_prepared_mlx_artifact(
        args.prepared,
        require_production=args.require_production,
    )
    report = artifact.metadata.get("compatibility_report", {})
    print(f"Prepared artifact: {artifact.base_dir}")
    print(f"Atoms: {artifact.atom_count}")
    print(f"Production force field: {bool(report.get('production_force_field', False))}")
    print("MLX compatibility: ok")
    return 0


def command_prep_deps(args: argparse.Namespace) -> int:
    del args
    for module_name, present in optional_prep_dependency_status().items():
        print(f"{module_name}: {'available' if present else 'missing'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atomistic-prep")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_prepare_parser(subparsers)
    add_prepare_t4l_parser(subparsers)
    add_import_amber_parser(subparsers)
    add_import_charmm_parser(subparsers)
    add_gpcrmd_inspect_parser(subparsers)
    add_gpcrmd_import_parser(subparsers)
    add_run_parser(subparsers)
    add_run_gpcrmd_mlx_parser(subparsers)
    add_benchmark_gpcrmd_mlx_parser(subparsers)
    add_run_ligand_receptor_example_parser(subparsers)
    add_run_ligand_receptor_replicas_parser(subparsers)
    add_profile_ligand_receptor_performance_parser(subparsers)
    add_run_steered_parser(subparsers)
    add_benchmark_parser(subparsers)
    add_validate_parser(subparsers)
    add_status_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
