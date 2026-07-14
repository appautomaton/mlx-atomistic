"""Scientific PME validation against converged direct Ewald references."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_atomistic.benchmarks.pme_fixture import (
    PME_ASSIGNMENT_ORDER as FIXTURE_PME_ORDER,
)
from mlx_atomistic.benchmarks.pme_fixture import (
    PME_REAL_CUTOFF_ANGSTROM,
    build_pme_fixture,
    fixture_summary,
)
from mlx_atomistic.core import Cell
from mlx_atomistic.forcefields import NonbondedPotential
from mlx_atomistic.neighbors import build_neighbor_list
from mlx_atomistic.nonbonded import (
    EwaldReferenceConfig,
    ewald_reference_coulomb_energy_forces,
)
from mlx_atomistic.pme import PMEConfig, pme_coulomb_energy_forces
from mlx_atomistic.prep.schema import PreparedSystem
from mlx_atomistic.topology import Topology
from mlx_atomistic.units import COULOMB_CONSTANT_KJ_MOL_ANGSTROM


@dataclass(frozen=True)
class ForceErrorMetrics:
    """Scale-aware force and energy differences."""

    normalized_rms: float
    normalized_maximum: float
    energy_error_per_atom_kj_mol: float


FORCE_RMS_LIMIT = 1.0e-3
FORCE_MAXIMUM_LIMIT = 5.0e-3
ENERGY_PER_ATOM_LIMIT_KJ_MOL = 2.0e-2


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


def array_hash(values: np.ndarray) -> str:
    """Return a stable hash for one numerical array."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def apply_openmm_pme_manifest(
    prepared: PreparedSystem,
    manifest: dict[str, object],
) -> PreparedSystem:
    """Apply resolved OpenMM PME parameters after validating fixture identity."""

    fixture = dict(manifest.get("fixture", {}))
    expected_fixture = fixture_summary(prepared)
    expected_hash = str(prepared.metadata.selections["content_hash"])
    checks = {
        "fixture_hash": str(fixture.get("content_hash", "")) == expected_hash,
        "atom_count": int(fixture.get("atom_count", -1)) == prepared.atom_count,
        "fixture_name": fixture.get("fixture") == expected_fixture["fixture"],
        "site_count": fixture.get("site_count") == expected_fixture["site_count"],
        "water_count": fixture.get("water_count") == expected_fixture["water_count"],
        "sodium_count": fixture.get("sodium_count") == expected_fixture["sodium_count"],
        "chloride_count": fixture.get("chloride_count")
        == expected_fixture["chloride_count"],
        "net_charge": np.isclose(
            float(fixture.get("net_charge_e", np.nan)),
            float(expected_fixture["net_charge_e"]),
            rtol=0.0,
            atol=1.0e-7,
        ),
        "box": np.allclose(
            np.asarray(fixture.get("box_lengths_A", []), dtype=np.float64),
            prepared.cell_lengths,
            rtol=0.0,
            atol=1.0e-6,
        ),
    }
    topology = dict(manifest.get("topology", {}))
    checks.update(
        {
            "charge_hash": topology.get("charge_hash") == array_hash(prepared.charges),
            "exception_pairs_hash": topology.get("exception_pairs_hash")
            == array_hash(prepared.nonbonded_exception_pairs),
            "exception_charge_product_hash": topology.get(
                "exception_charge_product_hash"
            )
            == array_hash(prepared.nonbonded_exception_charge_product),
        }
    )
    pme = dict(manifest.get("pme", {}))
    checks.update(
        {
            "cutoff": float(pme.get("real_cutoff_angstrom", np.nan))
            == PME_REAL_CUTOFF_ANGSTROM,
            "assignment_order": int(pme.get("assignment_order", -1))
            == FIXTURE_PME_ORDER,
            "alpha": np.isfinite(float(pme.get("alpha_per_angstrom", np.nan))),
            "mesh": len(pme.get("mesh_shape", [])) == 3
            and all(int(value) >= 4 for value in pme.get("mesh_shape", [])),
            "coulomb_constant": np.isclose(
                float(manifest.get("coulomb_constant_kj_mol_angstrom", np.nan)),
                COULOMB_CONSTANT_KJ_MOL_ANGSTROM,
                rtol=0.0,
                atol=1.0e-9,
            ),
            "exception_count": int(manifest.get("exception_count", -1))
            == prepared.nonbonded_exception_pairs.shape[0],
        }
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        msg = "OpenMM PME manifest does not match the MLX fixture: " + ", ".join(failed)
        raise ValueError(msg)
    mesh_shape = np.asarray(pme["mesh_shape"], dtype=np.int32)
    alpha = float(pme["alpha_per_angstrom"])
    pme_config = {
        **prepared.metadata.pme_config,
        "mesh_shape": mesh_shape.astype(int).tolist(),
        "alpha": alpha,
        "real_cutoff": PME_REAL_CUTOFF_ANGSTROM,
        "assignment_order": FIXTURE_PME_ORDER,
        "parameter_authority": "openmm_context",
        "openmm_platform": manifest.get("platform"),
        "openmm_precision": manifest.get("precision"),
    }
    return replace(
        prepared,
        metadata=replace(prepared.metadata, pme_config=pme_config),
        pme_mesh_shape=mesh_shape,
        pme_alpha=np.asarray([alpha], dtype=np.float32),
        pme_real_cutoff=np.asarray([PME_REAL_CUTOFF_ANGSTROM], dtype=np.float32),
        pme_assignment_order=np.asarray([FIXTURE_PME_ORDER], dtype=np.int32),
    )


def run_openmm_parity(
    prepared: PreparedSystem,
    reference_dir: str | Path,
) -> dict[str, object]:
    """Compare the production MLX PME electrostatic path with OpenMM output."""

    reference_path = Path(reference_dir)
    manifest = json.loads((reference_path / "reference.json").read_text())
    prepared = apply_openmm_pme_manifest(prepared, manifest)
    with np.load(reference_path / "reference_forces.npz", allow_pickle=False) as data:
        reference_forces = {name: np.asarray(data[name]) for name in data.files}
    configurations = deterministic_configurations(
        prepared,
        count=int(manifest["configuration_count"]),
    )
    rows = []
    all_passed = True
    for row, positions in zip(manifest["rows"], configurations, strict=True):
        if row["position_hash"] != array_hash(positions):
            configuration = row["configuration"]
            msg = f"OpenMM reference position hash mismatch for configuration {configuration}"
            raise ValueError(msg)
        key = str(row["force_key"])
        if key not in reference_forces:
            msg = f"OpenMM reference force array is missing {key}"
            raise ValueError(msg)
        energy, forces, diagnostics = _mlx_production_coulomb(prepared, positions)
        metrics = force_error_metrics(
            forces,
            reference_forces[key],
            candidate_energy=energy,
            reference_energy=float(row["energy_kj_mol"]),
        )
        passed = bool(
            metrics.normalized_rms <= FORCE_RMS_LIMIT
            and metrics.normalized_maximum <= FORCE_MAXIMUM_LIMIT
            and metrics.energy_error_per_atom_kj_mol <= ENERGY_PER_ATOM_LIMIT_KJ_MOL
            and diagnostics["grid_charge_error_e"] <= 1.0e-3
            and diagnostics["normalized_net_force"] <= 1.0e-5
        )
        all_passed = all_passed and passed
        rows.append(
            {
                "configuration": int(row["configuration"]),
                "passed": passed,
                "metrics": asdict(metrics),
                **diagnostics,
            }
        )
    return {
        "status": "passed" if all_passed else "failed",
        "passed": all_passed,
        "fixture": fixture_summary(prepared),
        "reference_manifest": str(reference_path / "reference.json"),
        "reference_platform": manifest["platform"],
        "reference_precision": manifest["precision"],
        "pme": manifest["pme"],
        "configuration_count": len(rows),
        "thresholds": {
            "normalized_rms": FORCE_RMS_LIMIT,
            "normalized_maximum": FORCE_MAXIMUM_LIMIT,
            "energy_error_per_atom_kj_mol": ENERGY_PER_ATOM_LIMIT_KJ_MOL,
        },
        "rows": rows,
    }


def _mlx_production_coulomb(
    prepared: PreparedSystem,
    positions: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, float | str | int | None]]:
    pme_config = PMEConfig(
        mesh_shape=tuple(int(value) for value in prepared.pme_mesh_shape.tolist()),
        alpha=float(prepared.pme_alpha[0]),
        real_cutoff=float(prepared.pme_real_cutoff[0]),
        assignment_order=int(prepared.pme_assignment_order[0]),
        charge_tolerance=float(prepared.pme_charge_tolerance[0]),
        deconvolve_assignment=bool(prepared.pme_deconvolve_assignment[0]),
    )
    topology = Topology.from_sequences(
        n_atoms=prepared.atom_count,
        bonds=prepared.bonds,
        angles=prepared.angles,
        dihedrals=prepared.dihedrals,
        partial_charges=prepared.charges,
        nonbonded_exception_pairs=prepared.nonbonded_exception_pairs,
        exclude_bonds=True,
        nonbonded_cutoff=PME_REAL_CUTOFF_ANGSTROM,
        eager_nonbonded_pair_limit=0,
    )
    nonbonded = NonbondedPotential(
        sigma=prepared.sigma,
        epsilon=np.zeros_like(prepared.epsilon),
        charges=prepared.charges,
        coulomb_constant=COULOMB_CONSTANT_KJ_MOL_ANGSTROM,
        cutoff=PME_REAL_CUTOFF_ANGSTROM,
        electrostatics="pme",
        topology=topology,
        exception_pairs=prepared.nonbonded_exception_pairs,
        exception_charge_products=prepared.nonbonded_exception_charge_product,
        exception_sigma=prepared.nonbonded_exception_sigma,
        exception_epsilon=np.zeros_like(prepared.nonbonded_exception_epsilon),
        pme_config=pme_config,
    )
    cell = Cell.orthorhombic(prepared.cell_lengths.tolist())
    neighbors = build_neighbor_list(
        positions,
        cell,
        cutoff=PME_REAL_CUTOFF_ANGSTROM,
        skin=0.0,
        backend="mlx_cell_blocks",
    )
    energy, forces, components = nonbonded.energy_forces_with_components(
        mx.array(positions),
        cell,
        pairs=neighbors.interactions,
    )
    mx.eval(energy, forces)
    force_array = np.asarray(forces, dtype=np.float64)
    force_l1 = float(np.sum(np.linalg.norm(force_array, axis=1)))
    pme_diagnostics = components["pme_diagnostics"]
    return (
        float(np.asarray(components["coulomb"])),
        force_array,
        {
            "grid_charge_error_e": abs(
                float(pme_diagnostics.charge_grid_sum)
                - float(np.sum(prepared.charges, dtype=np.float64))
            ),
            "normalized_net_force": float(np.linalg.norm(np.sum(force_array, axis=0)))
            / force_l1
            if force_l1 > 0.0
            else 0.0,
            "neighbor_backend": neighbors.backend,
            "neighbor_representation": neighbors.representation_kind,
            "pair_count": int(neighbors.pair_count),
            "compact_pair_count": int(neighbors.compact_pair_count),
            "candidate_count": neighbors.candidate_count,
            "candidate_waste_count": neighbors.candidate_waste_count,
        },
    )


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
    parser.add_argument(
        "--case",
        choices=("water-small", "salt-small", "target"),
        default="water-small",
    )
    parser.add_argument("--configurations", type=int, default=5)
    parser.add_argument("--reciprocal-cutoffs", default="4,6,8,10")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prepared = build_pme_fixture(args.case)
    if args.reference is None:
        if args.case == "target":
            msg = "target validation requires --reference; direct Ewald is small-case only"
            raise SystemExit(msg)
        payload = run_ewald_convergence(
            prepared,
            reciprocal_cutoffs=_parse_cutoffs(args.reciprocal_cutoffs),
            configurations=args.configurations,
        )
    else:
        payload = run_openmm_parity(prepared, args.reference)
    if args.out is not None:
        output_path = args.out
        if args.reference is not None or output_path.suffix == "":
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / "mlx_parity.json"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.json or args.out is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
