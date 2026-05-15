"""Run a fixed-coordinate OpenMM-vs-MLX parity check for a production artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openmm_mlx_parity import (
    DEFAULT_AMBER_FIXTURE,
    ParityTolerances,
    PMEParityConfig,
    default_amber_fixture_paths,
    run_amber_openmm_mlx_parity,
)


def main() -> None:
    args = _parse_args()
    prmtop_path, coords_path = _fixture_paths(args)
    report = run_amber_openmm_mlx_parity(
        prmtop_path=prmtop_path,
        coords_path=coords_path,
        out_dir=args.out,
        fixture=args.fixture,
        platform_name=args.platform,
        mlx_nonbonded_cutoff_angstrom=args.mlx_nonbonded_cutoff_angstrom,
        tolerances=_tolerances(args),
        pme_config=_pme_config(args),
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if report.status == "blocked":
        raise SystemExit(2)
    if not report.passed:
        raise SystemExit(1)


def _fixture_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.prmtop is not None or args.coords is not None:
        if args.prmtop is None or args.coords is None:
            msg = "--prmtop and --coords must be provided together"
            raise SystemExit(msg)
        return Path(args.prmtop), Path(args.coords)
    if args.fixture != DEFAULT_AMBER_FIXTURE:
        msg = (
            f"unknown fixture {args.fixture!r}; pass --prmtop and --coords for a "
            "custom AMBER fixture"
        )
        raise SystemExit(msg)
    return default_amber_fixture_paths(Path("."))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        default=DEFAULT_AMBER_FIXTURE,
        help="fixture label; default uses the small vendored OpenMM AMBER fixture",
    )
    parser.add_argument("--prmtop", type=Path, help="custom AMBER prmtop path")
    parser.add_argument("--coords", type=Path, help="custom AMBER inpcrd/rst7 path")
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
        help="compare the fixture with periodic PME electrostatics",
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
    return parser.parse_args()


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
