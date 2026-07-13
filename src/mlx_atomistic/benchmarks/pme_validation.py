"""Scientific PME validation against converged direct Ewald references."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_atomistic.benchmarks.pme_fixture import build_pme_fixture, fixture_summary
from mlx_atomistic.core import Cell
from mlx_atomistic.nonbonded import (
    EwaldReferenceConfig,
    ewald_reference_coulomb_energy_forces,
)
from mlx_atomistic.pme import PMEConfig, pme_coulomb_energy_forces
from mlx_atomistic.prep.schema import PreparedSystem
from mlx_atomistic.units import COULOMB_CONSTANT_KJ_MOL_ANGSTROM


@dataclass(frozen=True)
class ForceErrorMetrics:
    """Scale-aware force and energy differences."""

    normalized_rms: float
    normalized_maximum: float
    energy_error_per_atom_kj_mol: float


def force_error_metrics(
    candidate_forces: np.ndarray,
    reference_forces: np.ndarray,
    *,
    candidate_energy: float,
    reference_energy: float,
) -> ForceErrorMetrics:
    """Return the normalized force metrics defined by the scientific SPEC."""

    candidate = np.asarray(candidate_forces, dtype=np.float64)
    reference = np.asarray(reference_forces, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.ndim != 2 or candidate.shape[1] != 3:
        msg = "candidate and reference forces must have matching shape (n_atoms, 3)"
        raise ValueError(msg)
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(reference)):
        msg = "force metrics require finite candidate and reference forces"
        raise ValueError(msg)
    reference_rms_denominator = float(np.sum(reference * reference))
    reference_maximum = float(np.max(np.linalg.norm(reference, axis=1)))
    if reference_rms_denominator <= 0.0 or reference_maximum <= 0.0:
        msg = "force metrics require a non-zero reference force field"
        raise ValueError(msg)
    delta = candidate - reference
    normalized_rms = sqrt_ratio(float(np.sum(delta * delta)), reference_rms_denominator)
    normalized_maximum = float(np.max(np.linalg.norm(delta, axis=1))) / reference_maximum
    atom_count = candidate.shape[0]
    return ForceErrorMetrics(
        normalized_rms=normalized_rms,
        normalized_maximum=normalized_maximum,
        energy_error_per_atom_kj_mol=abs(float(candidate_energy) - float(reference_energy))
        / atom_count,
    )


def sqrt_ratio(numerator: float, denominator: float) -> float:
    """Return a finite square-root ratio or fail closed."""

    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        msg = "normalized metric denominator must be finite and positive"
        raise ValueError(msg)
    return float(np.sqrt(max(0.0, numerator) / denominator))


def deterministic_configurations(
    prepared: PreparedSystem,
    *,
    count: int = 5,
    displacement_angstrom: float = 0.025,
    seed: int | None = None,
) -> tuple[np.ndarray, ...]:
    """Return deterministic rigid-residue displacements for a fixture."""

    if count <= 0:
        msg = "configuration count must be positive"
        raise ValueError(msg)
    if displacement_angstrom < 0.0:
        msg = "displacement_angstrom must be non-negative"
        raise ValueError(msg)
    box = np.asarray(prepared.cell_lengths, dtype=np.float64)
    if box.shape != (3,) or np.any(box <= 0.0):
        msg = "PME validation configurations require a positive orthorhombic box"
        raise ValueError(msg)
    residue_ids = np.asarray(prepared.residue_ids, dtype=np.int64)
    unique_residues, inverse = np.unique(residue_ids, return_inverse=True)
    if seed is None:
        seed = int(prepared.metadata.source.get("seed", 0))
    rng = np.random.default_rng(seed)
    positions = np.asarray(prepared.positions, dtype=np.float64)
    configurations = []
    for index in range(count):
        if index == 0:
            displaced = positions.copy()
        else:
            translations = rng.uniform(
                -displacement_angstrom,
                displacement_angstrom,
                size=(unique_residues.size, 3),
            )
            displaced = positions + translations[inverse]
        configurations.append(np.mod(displaced, box).astype(np.float32))
    return tuple(configurations)


def run_ewald_convergence(
    prepared: PreparedSystem,
    *,
    reciprocal_cutoffs: tuple[int, ...] = (4, 6, 8, 10),
    configurations: int = 5,
    convergence_tolerance: float = 2.0e-4,
) -> dict[str, object]:
    """Converge direct Ewald and compare fifth-order PME on each configuration."""

    cutoffs = tuple(int(value) for value in reciprocal_cutoffs)
    if len(cutoffs) < 2 or any(value <= 0 for value in cutoffs):
        msg = "reciprocal_cutoffs must contain at least two positive integers"
        raise ValueError(msg)
    if tuple(sorted(set(cutoffs))) != cutoffs:
        msg = "reciprocal_cutoffs must be strictly increasing"
        raise ValueError(msg)
    if convergence_tolerance <= 0.0:
        msg = "convergence_tolerance must be positive"
        raise ValueError(msg)

    pme_config = PMEConfig(
        mesh_shape=tuple(int(value) for value in prepared.pme_mesh_shape.tolist()),
        alpha=float(prepared.pme_alpha[0]),
        real_cutoff=float(prepared.pme_real_cutoff[0]),
        assignment_order=int(prepared.pme_assignment_order[0]),
        charge_tolerance=float(prepared.pme_charge_tolerance[0]),
        deconvolve_assignment=bool(prepared.pme_deconvolve_assignment[0]),
    )
    cell = Cell.orthorhombic(prepared.cell_lengths.tolist())
    charges = mx.array(prepared.charges)
    rows: list[dict[str, object]] = []
    all_converged = True
    for config_index, positions_np in enumerate(
        deterministic_configurations(prepared, count=configurations)
    ):
        positions = mx.array(positions_np)
        ewald_results: list[tuple[float, np.ndarray]] = []
        for reciprocal_cutoff in cutoffs:
            energy, forces, _ = ewald_reference_coulomb_energy_forces(
                positions,
                charges,
                cell,
                coulomb_constant=COULOMB_CONSTANT_KJ_MOL_ANGSTROM,
                config=EwaldReferenceConfig(
                    alpha=pme_config.alpha,
                    real_cutoff=pme_config.real_cutoff,
                    reciprocal_cutoff=reciprocal_cutoff,
                    charge_tolerance=pme_config.charge_tolerance,
                ),
            )
            mx.eval(energy, forces)
            ewald_results.append((float(np.asarray(energy)), np.asarray(forces)))
        previous_energy, previous_forces = ewald_results[-2]
        reference_energy, reference_forces = ewald_results[-1]
        convergence = force_error_metrics(
            previous_forces,
            reference_forces,
            candidate_energy=previous_energy,
            reference_energy=reference_energy,
        )
        converged = convergence.normalized_rms <= convergence_tolerance
        all_converged = all_converged and converged

        pme_energy, pme_forces, pme_components = pme_coulomb_energy_forces(
            positions,
            charges,
            cell,
            coulomb_constant=COULOMB_CONSTANT_KJ_MOL_ANGSTROM,
            config=pme_config,
        )
        mx.eval(pme_energy, pme_forces)
        pme_metrics = force_error_metrics(
            np.asarray(pme_forces),
            reference_forces,
            candidate_energy=float(np.asarray(pme_energy)),
            reference_energy=reference_energy,
        )
        diagnostics = pme_components["diagnostics"]
        net_force = np.sum(np.asarray(pme_forces, dtype=np.float64), axis=0)
        force_l1 = float(np.sum(np.linalg.norm(np.asarray(pme_forces), axis=1)))
        rows.append(
            {
                "configuration": config_index,
                "converged": converged,
                "convergence": asdict(convergence),
                "pme_vs_ewald": asdict(pme_metrics),
                "ewald_energy_kj_mol": reference_energy,
                "pme_energy_kj_mol": float(np.asarray(pme_energy)),
                "grid_charge_error_e": abs(
                    float(diagnostics.charge_grid_sum)
                    - float(np.sum(prepared.charges, dtype=np.float64))
                ),
                "normalized_net_force": float(np.linalg.norm(net_force)) / force_l1
                if force_l1 > 0.0
                else None,
                "finite": bool(
                    np.isfinite(reference_energy)
                    and np.all(np.isfinite(reference_forces))
                    and np.isfinite(float(np.asarray(pme_energy)))
                    and np.all(np.isfinite(np.asarray(pme_forces)))
                ),
            }
        )
    return {
        "status": "passed" if all_converged else "failed",
        "fixture": fixture_summary(prepared),
        "reciprocal_cutoffs": list(cutoffs),
        "convergence_tolerance": convergence_tolerance,
        "configuration_count": configurations,
        "rows": rows,
    }


def _parse_cutoffs(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("water-small", "salt-small"), default="water-small")
    parser.add_argument("--configurations", type=int, default=5)
    parser.add_argument("--reciprocal-cutoffs", default="4,6,8,10")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = run_ewald_convergence(
        build_pme_fixture(args.case),
        reciprocal_cutoffs=_parse_cutoffs(args.reciprocal_cutoffs),
        configurations=args.configurations,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.json or args.out is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

