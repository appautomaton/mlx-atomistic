"""Run a fixed-coordinate OpenMM-vs-MLX parity check for a production artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from openmm_mlx_parity import (
    DEFAULT_AMBER_FIXTURE,
    DEFAULT_CHARMM_FIXTURE,
    DEFAULT_GROMACS_FIXTURE,
    ParityTolerances,
    PMEParityConfig,
    default_amber_fixture_paths,
    default_charmm_fixture_paths,
    default_gromacs_fixture_paths,
    run_amber_openmm_mlx_parity,
    run_charmm_openmm_mlx_parity,
    run_gromacs_openmm_mlx_parity,
)


def main() -> None:
    args = _parse_args()
    report = _run_report(args)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if report.status == "blocked":
        raise SystemExit(2)
    if not report.passed:
        raise SystemExit(1)


def _run_report(args: argparse.Namespace):
    if args.pme and args.source_kind != "amber":
        msg = "--pme is only supported for AMBER parity"
        raise SystemExit(msg)
    if args.source_kind == "amber":
        prmtop_path, coords_path, fixture = _amber_fixture_paths(args)
        return run_amber_openmm_mlx_parity(
            prmtop_path=prmtop_path,
            coords_path=coords_path,
            out_dir=args.out,
            fixture=fixture,
            platform_name=args.platform,
            mlx_nonbonded_cutoff_angstrom=args.mlx_nonbonded_cutoff_angstrom,
            tolerances=_tolerances(args),
            pme_config=_pme_config(args),
        )
    if args.source_kind == "charmm":
        psf_path, params, openmm_params, coords_path, fixture = _charmm_fixture_paths(args)
        return run_charmm_openmm_mlx_parity(
            psf_path=psf_path,
            params=params,
            openmm_params=openmm_params,
            coords_path=coords_path,
            out_dir=args.out,
            fixture=fixture,
            platform_name=args.platform,
            mlx_nonbonded_cutoff_angstrom=args.mlx_nonbonded_cutoff_angstrom,
            tolerances=_tolerances(args),
        )
    if args.source_kind == "gromacs":
        top_path, gro_path, fixture = _gromacs_fixture_paths(args)
        return run_gromacs_openmm_mlx_parity(
            top_path=top_path,
            gro_path=gro_path,
            out_dir=args.out,
            fixture=fixture,
            platform_name=args.platform,
            mlx_nonbonded_cutoff_angstrom=args.mlx_nonbonded_cutoff_angstrom,
            tolerances=_tolerances(args),
        )
    msg = f"unknown source kind {args.source_kind!r}"
    raise SystemExit(msg)


def _amber_fixture_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    fixture = args.fixture or DEFAULT_AMBER_FIXTURE
    if args.prmtop is not None or args.coords is not None:
        if args.prmtop is None or args.coords is None:
            msg = "--prmtop and --coords must be provided together for AMBER parity"
            raise SystemExit(msg)
        return Path(args.prmtop), Path(args.coords), fixture
    if fixture != DEFAULT_AMBER_FIXTURE:
        msg = (
            f"unknown AMBER fixture {fixture!r}; pass --prmtop and --coords for a "
            "custom AMBER fixture"
        )
        raise SystemExit(msg)
    prmtop_path, coords_path = default_amber_fixture_paths(Path("."))
    return prmtop_path, coords_path, fixture


def _charmm_fixture_paths(
    args: argparse.Namespace,
) -> tuple[Path, tuple[Path, ...], tuple[Path, ...], Path, str]:
    fixture = args.fixture or DEFAULT_CHARMM_FIXTURE
    native_params = tuple(Path(path) for path in (args.params or ()))
    openmm_params = tuple(Path(path) for path in (args.openmm_params or ()))
    if args.psf is not None or args.coords is not None or native_params or openmm_params:
        if args.psf is None or args.coords is None or not native_params:
            msg = "--psf, --param, and --coords must be provided for custom CHARMM parity"
            raise SystemExit(msg)
        if not openmm_params:
            openmm_params = native_params
        return Path(args.psf), native_params, openmm_params, Path(args.coords), fixture
    if fixture != DEFAULT_CHARMM_FIXTURE:
        msg = (
            f"unknown CHARMM fixture {fixture!r}; pass --psf, --param, and --coords "
            "for a custom CHARMM fixture"
        )
        raise SystemExit(msg)
    psf_path, prm_path, rtf_path, coords_path = default_charmm_fixture_paths(Path("."))
    return psf_path, (prm_path,), (rtf_path, prm_path), coords_path, fixture


def _gromacs_fixture_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    fixture = args.fixture or DEFAULT_GROMACS_FIXTURE
    if args.top is not None or args.gro is not None:
        if args.top is None or args.gro is None:
            msg = "--top and --gro must be provided together for GROMACS parity"
            raise SystemExit(msg)
        return Path(args.top), Path(args.gro), fixture
    if fixture != DEFAULT_GROMACS_FIXTURE:
        msg = (
            f"unknown GROMACS fixture {fixture!r}; pass --top and --gro for a custom "
            "GROMACS fixture"
        )
        raise SystemExit(msg)
    top_path, gro_path = default_gromacs_fixture_paths(Path("."))
    return top_path, gro_path, fixture


def _fixture_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    prmtop_path, coords_path, _ = _amber_fixture_paths(args)
    return prmtop_path, coords_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-kind",
        choices=("amber", "charmm", "gromacs"),
        default="amber",
        help="input format to compare; defaults to AMBER",
    )
    parser.add_argument(
        "--fixture",
        default=None,
        help="fixture label; default uses the tracked fixture for --source-kind",
    )
    parser.add_argument("--prmtop", type=Path, help="custom AMBER prmtop path")
    parser.add_argument(
        "--coords",
        type=Path,
        help="custom AMBER inpcrd/rst7 or CHARMM PDB coordinate path",
    )
    parser.add_argument("--psf", type=Path, help="custom CHARMM PSF path")
    parser.add_argument(
        "--param",
        dest="params",
        action="append",
        type=Path,
        help="custom CHARMM parameter path; repeat for multiple native parameters",
    )
    parser.add_argument(
        "--openmm-param",
        dest="openmm_params",
        action="append",
        type=Path,
        help="OpenMM-only CHARMM topology/parameter path; repeat for RTF and PRM inputs",
    )
    parser.add_argument("--top", type=Path, help="custom GROMACS topology path")
    parser.add_argument("--gro", type=Path, help="custom GROMACS coordinate path")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/md-engine-gap-closure/parity-fixture"),
        help="output directory for prepared artifact and parity report",
    )
    parser.add_argument("--platform", default="Reference", help="OpenMM platform name")
    parser.add_argument(
        "--mlx-nonbonded-cutoff-angstrom",
        type=float,
        default=1.0e6,
        help="large cutoff used to align the MLX cutoff path with OpenMM NoCutoff",
    )
    parser.add_argument(
        "--pme",
        action="store_true",
        help="compare an AMBER fixture with periodic PME electrostatics",
    )
    parser.add_argument(
        "--pme-mesh",
        default="48,48,48",
        help="PME mesh dimensions as nx,ny,nz",
    )
    parser.add_argument("--pme-alpha-per-angstrom", type=float, default=0.35)
    parser.add_argument("--pme-real-cutoff-angstrom", type=float, default=10.0)
    parser.add_argument(
        "--pme-cell-angstrom",
        default="40,40,40",
        help="orthorhombic PME box lengths as a,b,c in Angstrom",
    )
    parser.add_argument("--total-energy-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--component-energy-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--force-max-tolerance", type=float, default=12.0)
    parser.add_argument("--force-rms-tolerance", type=float, default=3.0)
    return parser.parse_args(argv)


def _tolerances(args: argparse.Namespace) -> ParityTolerances:
    if args.pme:
        default_total = 2.0e-3
        default_component = 2.0e-3
        return ParityTolerances(
            total_energy_abs_kj_mol=(
                5.0e-2
                if args.total_energy_tolerance == default_total
                else args.total_energy_tolerance
            ),
            component_energy_abs_kj_mol=(
                5.0e-2
                if args.component_energy_tolerance == default_component
                else args.component_energy_tolerance
            ),
            force_max_abs_kj_mol_nm=args.force_max_tolerance,
            force_rms_abs_kj_mol_nm=args.force_rms_tolerance,
        )
    return ParityTolerances(
        total_energy_abs_kj_mol=args.total_energy_tolerance,
        component_energy_abs_kj_mol=args.component_energy_tolerance,
        force_max_abs_kj_mol_nm=args.force_max_tolerance,
        force_rms_abs_kj_mol_nm=args.force_rms_tolerance,
    )


def _pme_config(args: argparse.Namespace) -> PMEParityConfig | None:
    if not args.pme:
        return None
    return PMEParityConfig(
        mesh_shape=_parse_triplet(args.pme_mesh, cast=int),
        alpha_per_angstrom=args.pme_alpha_per_angstrom,
        real_cutoff_angstrom=args.pme_real_cutoff_angstrom,
        cell_lengths_angstrom=_parse_triplet(args.pme_cell_angstrom, cast=float),
    )


def _parse_triplet(value: str, *, cast):
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        msg = f"expected three comma-separated values, got {value!r}"
        raise SystemExit(msg)
    return tuple(cast(part) for part in parts)


if __name__ == "__main__":
    main()
