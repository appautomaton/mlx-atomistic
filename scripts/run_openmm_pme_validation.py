"""Generate matched OpenMM PME electrostatic references for MLX fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.unit as unit

from mlx_atomistic.benchmarks.pme_fixture import (
    PME_ASSIGNMENT_ORDER,
    PME_ERROR_TOLERANCE,
    PME_REAL_CUTOFF_ANGSTROM,
    build_pme_fixture,
    fixture_summary,
)
from mlx_atomistic.benchmarks.pme_validation import array_hash, deterministic_configurations
from mlx_atomistic.units import COULOMB_CONSTANT_KJ_MOL_ANGSTROM


def build_openmm_reference(
    *,
    case: str,
    configurations: int,
    out_dir: str | Path,
    platform_name: str = "Reference",
    precision: str = "double",
) -> dict[str, object]:
    """Evaluate identical fixture configurations with OpenMM PME."""

    prepared = build_pme_fixture(case)
    system = mm.System()
    force = mm.NonbondedForce()
    force.setNonbondedMethod(mm.NonbondedForce.PME)
    force.setCutoffDistance(PME_REAL_CUTOFF_ANGSTROM * 0.1 * unit.nanometer)
    force.setEwaldErrorTolerance(PME_ERROR_TOLERANCE)
    force.setUseDispersionCorrection(False)
    for mass, charge, sigma in zip(
        prepared.masses,
        prepared.charges,
        prepared.sigma,
        strict=True,
    ):
        system.addParticle(float(mass) * unit.dalton)
        force.addParticle(
            float(charge) * unit.elementary_charge,
            float(sigma) * 0.1 * unit.nanometer,
            0.0 * unit.kilojoule_per_mole,
        )
    for left, right in prepared.nonbonded_exception_pairs.tolist():
        force.addException(
            int(left),
            int(right),
            0.0 * unit.elementary_charge**2,
            1.0 * unit.nanometer,
            0.0 * unit.kilojoule_per_mole,
        )
    system.addForce(force)
    box_nm = prepared.cell_lengths.astype(np.float64) * 0.1
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box_nm[0], 0.0, 0.0),
        mm.Vec3(0.0, box_nm[1], 0.0),
        mm.Vec3(0.0, 0.0, box_nm[2]),
    )
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    platform = mm.Platform.getPlatformByName(platform_name)
    properties = {}
    if platform_name in {"CUDA", "HIP", "OpenCL"}:
        properties["Precision"] = precision
    context = mm.Context(system, integrator, platform, properties)
    configs = deterministic_configurations(prepared, count=configurations)
    rows = []
    force_arrays = {}
    for index, positions in enumerate(configs):
        context.setPositions(positions * 0.1 * unit.nanometer)
        state = context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        force_key = f"forces_{index}"
        force_arrays[force_key] = np.asarray(forces, dtype=np.float64) * 0.1
        rows.append(
            {
                "configuration": index,
                "position_hash": array_hash(positions),
                "force_key": force_key,
                "energy_kj_mol": float(energy),
                "finite": bool(np.isfinite(energy) and np.all(np.isfinite(forces))),
            }
        )
    alpha, nx, ny, nz = force.getPMEParametersInContext(context)
    # OpenMM reports the resolved Ewald alpha as a plain float in nm^-1.
    alpha_per_angstrom = float(alpha) * 0.1
    del context, integrator

    payload = {
        "status": "reference_ready" if all(row["finite"] for row in rows) else "failed",
        "operation_semantics": "pme_electrostatics_with_topology_exclusions",
        "fixture": fixture_summary(prepared),
        "configuration_count": configurations,
        "exception_count": int(prepared.nonbonded_exception_pairs.shape[0]),
        "topology": {
            "charge_hash": array_hash(prepared.charges),
            "exception_pairs_hash": array_hash(prepared.nonbonded_exception_pairs),
            "exception_charge_product_hash": array_hash(
                prepared.nonbonded_exception_charge_product
            ),
        },
        "coulomb_constant_kj_mol_angstrom": COULOMB_CONSTANT_KJ_MOL_ANGSTROM,
        "openmm_version": mm.version.full_version,
        "platform": platform_name,
        "precision": precision,
        "pme": {
            "error_tolerance": PME_ERROR_TOLERANCE,
            "real_cutoff_angstrom": PME_REAL_CUTOFF_ANGSTROM,
            "alpha_per_angstrom": float(alpha_per_angstrom),
            "mesh_shape": [int(nx), int(ny), int(nz)],
            "assignment_order": PME_ASSIGNMENT_ORDER,
        },
        "rows": rows,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reference.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(out / "reference_forces.npz", **force_arrays)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("water-small", "salt-small", "target"), required=True)
    parser.add_argument("--configurations", type=int, default=5)
    parser.add_argument("--platform", default="Reference")
    parser.add_argument("--precision", default="double")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_openmm_reference(
        case=args.case,
        configurations=args.configurations,
        out_dir=args.out,
        platform_name=args.platform,
        precision=args.precision,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "reference_ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
