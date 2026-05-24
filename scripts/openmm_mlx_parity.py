"""OpenMM-vs-MLX force and energy parity helpers."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_atomistic.artifacts import (
    MLXCompatibilityError,
    artifact_readiness_report,
    build_mlx_system_from_artifact,
    load_prepared_mlx_artifact,
)
from mlx_atomistic.pme import PMEConfig, pme_readiness_report
from mlx_atomistic.prep.io import save_prepared_system
from mlx_atomistic.prep.schema import (
    ARTIFACT_VERSION,
    PreparedSystem,
    PreparedSystemMetadata,
    empty_indices,
)
from mlx_atomistic.prep.topology_import import (
    TopologyImportError,
    import_amber_prmtop,
    import_charmm_psf,
    import_gromacs_top_gro,
)
from mlx_atomistic.topology import Topology
from mlx_atomistic.validation import build_platform_validation_evidence
from mlx_atomistic.virtual_sites import tip4p_ew_reference_positions, tip4p_ew_virtual_site

DEFAULT_AMBER_FIXTURE = "amber-alanine-dipeptide-implicit"
DEFAULT_CHARMM_FIXTURE = "charmm-native-mini"
DEFAULT_GROMACS_FIXTURE = "gromacs-native-mini"
REPORT_NAME = "openmm_mlx_parity_report.json"
OPENMM_REFERENCE_ROLE = "reference-only validation; not a product runtime dependency"
ACCEPTANCE_CRITERIA_BY_SOURCE = {
    "amber": ("AC-03", "AC-07"),
    "charmm": ("AC-04", "AC-07"),
    "gromacs": ("AC-05", "AC-07"),
}
GAP_IDS_BY_SOURCE = {
    "amber": ("P2-PARSE-01", "P2-PARITY-01"),
    "charmm": ("P2-PARSE-02", "P2-PARITY-01"),
    "gromacs": ("P2-PARSE-03", "P2-PARITY-01"),
}
PARITY_COMPONENT_KEYS = (
    "bond",
    "angle",
    "torsion",
    "rb_dihedral",
    "urey_bradley",
    "charmm_cmap",
    "nonbonded",
)
TIP4P_EW_CHARGE_H = 0.52422
TIP4P_EW_CHARGE_M = -1.04844
TIP4P_EW_SIGMA_O_ANGSTROM = 3.16435
TIP4P_EW_EPSILON_O_KJ_MOL = 0.680946
MLX_COMPONENT_KEYS = (
    *PARITY_COMPONENT_KEYS,
    "dihedral",
    "improper",
    "charmm_cmap_terms",
    "nonbonded.coulomb",
    "nonbonded.lj",
    "nonbonded.real",
    "nonbonded.reciprocal",
)


@dataclass(frozen=True)
class ParityTolerances:
    """Numerical tolerances for fixed-coordinate parity checks."""

    total_energy_abs_kj_mol: float = 2.0e-3
    component_energy_abs_kj_mol: float = 2.0e-3
    force_max_abs_kj_mol_nm: float = 12.0
    force_rms_abs_kj_mol_nm: float = 3.0


@dataclass(frozen=True)
class PMEParityConfig:
    """Periodic PME settings used by the OpenMM-vs-MLX parity harness."""

    mesh_shape: tuple[int, int, int] = (48, 48, 48)
    alpha_per_angstrom: float = 0.35
    real_cutoff_angstrom: float = 10.0
    cell_lengths_angstrom: tuple[float, float, float] = (40.0, 40.0, 40.0)
    assignment_order: int = 2
    charge_tolerance: float = 1.0e-5
    deconvolve_assignment: bool = True

    def to_pme_config(self) -> PMEConfig:
        return PMEConfig(
            mesh_shape=self.mesh_shape,
            alpha=self.alpha_per_angstrom,
            real_cutoff=self.real_cutoff_angstrom,
            assignment_order=self.assignment_order,
            charge_tolerance=self.charge_tolerance,
            deconvolve_assignment=self.deconvolve_assignment,
        )


@dataclass(frozen=True)
class OpenMMMLXParityReport:
    """Machine-readable result for one OpenMM-vs-MLX parity run."""

    status: str
    fixture: str
    source_kind: str
    atom_count: int
    prepared_dir: str
    prmtop_path: str
    coords_path: str
    openmm_platform: str
    openmm_nonbonded_method: str
    reference_engine: str
    reference_engine_role: str
    mlx_nonbonded_cutoff_angstrom: float
    artifact_readiness: dict[str, Any] | None
    pme_config: dict[str, Any] | None
    pme_readiness: dict[str, Any] | None
    readiness: dict[str, Any]
    platform_evidence: dict[str, Any]
    source_paths: dict[str, Any]
    total_energy_openmm_kj_mol: float | None
    total_energy_mlx_kj_mol: float | None
    total_energy_abs_error_kj_mol: float | None
    component_energy_openmm_kj_mol: dict[str, float]
    component_energy_mlx_kj_mol: dict[str, float]
    component_energy_abs_error_kj_mol: dict[str, float]
    force_shape: tuple[int, int] | None
    force_max_abs_error_kj_mol_nm: float | None
    force_rms_abs_error_kj_mol_nm: float | None
    unsupported_terms: tuple[str, ...]
    unmapped_mlx_components: tuple[str, ...]
    unmapped_openmm_components: tuple[str, ...]
    tolerances: ParityTolerances
    passed: bool
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["force_shape"] = list(self.force_shape) if self.force_shape is not None else None
        payload["unsupported_terms"] = list(self.unsupported_terms)
        payload["unmapped_mlx_components"] = list(self.unmapped_mlx_components)
        payload["unmapped_openmm_components"] = list(self.unmapped_openmm_components)
        payload["blockers"] = list(self.blockers)
        return payload


def default_amber_fixture_paths(root: str | Path = ".") -> tuple[Path, Path]:
    """Return the tracked small AMBER fixture paths used for reproducible parity tests."""

    base = Path(root) / "tests" / "fixtures" / "amber"
    return base / "alanine-dipeptide-implicit.prmtop", base / "alanine-dipeptide-implicit.inpcrd"


def default_charmm_fixture_paths(root: str | Path = ".") -> tuple[Path, Path, Path, Path]:
    """Return the tracked CHARMM parity fixture paths."""

    base = Path(root) / "tests" / "fixtures" / "charmm"
    return (
        base / "parity-mini.psf",
        base / "parity-mini.prm",
        base / "native-mini.rtf",
        base / "native-mini.pdb",
    )


def default_gromacs_fixture_paths(root: str | Path = ".") -> tuple[Path, Path]:
    """Return the tracked GROMACS parity fixture paths."""

    base = Path(root) / "tests" / "fixtures" / "gromacs"
    return base / "native-mini.top", base / "native-mini.gro"


def run_amber_openmm_mlx_parity(
    *,
    prmtop_path: str | Path,
    coords_path: str | Path,
    out_dir: str | Path,
    fixture: str = DEFAULT_AMBER_FIXTURE,
    platform_name: str = "Reference",
    tolerances: ParityTolerances | None = None,
    mlx_nonbonded_cutoff_angstrom: float = 1.0e6,
    pme_config: PMEParityConfig | None = None,
) -> OpenMMMLXParityReport:
    """Build an AMBER artifact and compare fixed-coordinate OpenMM and MLX energies."""

    tolerances = ParityTolerances() if tolerances is None else tolerances
    prmtop = Path(prmtop_path)
    coords = Path(coords_path)
    out = Path(out_dir)
    prepared_dir = out / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    if not prmtop.exists():
        return _blocked_report(
            fixture=fixture,
            prmtop_path=prmtop,
            coords_path=coords,
            prepared_dir=prepared_dir,
            platform_name=platform_name,
            cutoff=mlx_nonbonded_cutoff_angstrom,
            tolerances=tolerances,
            blocker=f"missing AMBER prmtop: {prmtop}",
        )
    if not coords.exists():
        return _blocked_report(
            fixture=fixture,
            prmtop_path=prmtop,
            coords_path=coords,
            prepared_dir=prepared_dir,
            platform_name=platform_name,
            cutoff=mlx_nonbonded_cutoff_angstrom,
            tolerances=tolerances,
            blocker=f"missing AMBER coordinates: {coords}",
        )

    try:
        prepared = import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)
    except TopologyImportError as exc:
        blocker = str(exc)
        return _blocked_report(
            fixture=fixture,
            source_kind="amber",
            prmtop_path=prmtop,
            coords_path=coords,
            prepared_dir=prepared_dir,
            platform_name=platform_name,
            cutoff=(
                float(pme_config.real_cutoff_angstrom)
                if pme_config is not None
                else float(mlx_nonbonded_cutoff_angstrom)
            ),
            tolerances=tolerances,
            blocker=blocker,
            unsupported_terms=_unsupported_terms_from_blocker(blocker),
            pme_config=pme_config,
        )
    if pme_config is not None:
        prepared = _with_pme_artifact_settings(prepared, pme_config)
    effective_cutoff = (
        float(pme_config.real_cutoff_angstrom)
        if pme_config is not None
        else float(mlx_nonbonded_cutoff_angstrom)
    )
    return _run_prepared_openmm_mlx_parity(
        prepared=prepared,
        out_dir=out,
        prepared_dir=prepared_dir,
        fixture=fixture,
        source_kind="amber",
        topology_path=prmtop,
        coords_path=coords,
        source_paths={
            "prmtop": str(prmtop),
            "inpcrd": str(coords),
            "topology": str(prmtop),
            "coordinates": str(coords),
        },
        platform_name=platform_name,
        tolerances=tolerances,
        effective_cutoff=effective_cutoff,
        openmm_nonbonded_method="PME" if pme_config is not None else "NoCutoff",
        pme_config=pme_config,
        openmm_evaluator=lambda expected_counts: _evaluate_openmm_amber(
            prmtop_path=prmtop,
            coords_path=coords,
            platform_name=platform_name,
            pme_config=pme_config,
            expected_component_counts=expected_counts,
        ),
    )


def evaluate_tip4p_ew_openmm_mlx_parity(*, platform_name: str = "Reference") -> dict[str, Any]:
    """Compare a persisted periodic TIP4P-Ew artifact against OpenMM."""

    import openmm as mm
    from openmm import unit

    pme_config = PMEParityConfig(
        mesh_shape=(16, 16, 16),
        alpha_per_angstrom=0.35,
        real_cutoff_angstrom=8.0,
        cell_lengths_angstrom=(20.0, 20.0, 20.0),
    )
    prepared = _tip4p_ew_prepared_artifact(pme_config)
    with tempfile.TemporaryDirectory() as tmp:
        save_prepared_system(prepared, tmp)
        artifact = load_prepared_mlx_artifact(tmp, require_production=True)
        mlx_system, force_terms, _ = build_mlx_system_from_artifact(artifact)

    assert mlx_system.virtual_sites is not None
    eval_positions = mlx_system.virtual_sites.extend_positions(mlx_system.positions)
    mlx_total, _, _ = _evaluate_mlx(force_terms, eval_positions, mlx_system.cell)

    positions_angstrom = np.asarray(prepared.positions, dtype=np.float64)
    charges = np.asarray(
        [0.0, TIP4P_EW_CHARGE_H, TIP4P_EW_CHARGE_H, TIP4P_EW_CHARGE_M] * 2,
        dtype=np.float32,
    )
    sigma = np.asarray([TIP4P_EW_SIGMA_O_ANGSTROM, 1.0, 1.0, 1.0] * 2, dtype=np.float32)
    epsilon = np.asarray([TIP4P_EW_EPSILON_O_KJ_MOL, 0.0, 0.0, 0.0] * 2, dtype=np.float32)

    system = mm.System()
    for mass in (15.99943, 1.007947, 1.007947, 0.0) * 2:
        system.addParticle(mass * unit.dalton)
    site = tip4p_ew_virtual_site(0, 1, 2)
    system.setVirtualSite(
        3,
        mm.ThreeParticleAverageSite(
            site.particle1,
            site.particle2,
            site.particle3,
            site.weight1,
            site.weight2,
            site.weight3,
        ),
    )
    system.setVirtualSite(
        7,
        mm.ThreeParticleAverageSite(
            4,
            5,
            6,
            site.weight1,
            site.weight2,
            site.weight3,
        ),
    )
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(pme_config.real_cutoff_angstrom * 0.1 * unit.nanometer)
    nonbonded.setPMEParameters(
        pme_config.alpha_per_angstrom * 10.0 / unit.nanometer,
        *pme_config.mesh_shape,
    )
    for charge, sigma_value, epsilon_value in zip(charges, sigma, epsilon, strict=True):
        nonbonded.addParticle(
            float(charge) * unit.elementary_charge,
            float(sigma_value) * 0.1 * unit.nanometer,
            float(epsilon_value) * unit.kilojoule_per_mole,
        )
    system.addForce(nonbonded)
    box_vectors = (
        mm.Vec3(2.0, 0.0, 0.0),
        mm.Vec3(0.0, 2.0, 0.0),
        mm.Vec3(0.0, 0.0, 2.0),
    ) * unit.nanometer
    system.setDefaultPeriodicBoxVectors(*box_vectors)
    context = mm.Context(
        system,
        mm.VerletIntegrator(0.001 * unit.picoseconds),
        mm.Platform.getPlatformByName(platform_name),
    )
    context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(positions_angstrom * 0.1 * unit.nanometer)
    context.computeVirtualSites()
    openmm_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    mlx_energy_value = float(mlx_total)
    return {
        "reference_engine": "openmm",
        "reference_engine_role": OPENMM_REFERENCE_ROLE,
        "total_energy_openmm_kj_mol": float(openmm_energy),
        "total_energy_mlx_kj_mol": mlx_energy_value,
        "total_energy_abs_error_kj_mol": abs(mlx_energy_value - float(openmm_energy)),
        "positions_angstrom": positions_angstrom,
        "artifact_atom_count": artifact.atom_count,
        "runtime_atom_count": mlx_system.atom_count,
        "runtime_virtual_site_count": mlx_system.virtual_sites.n_virtual_sites,
    }


def _tip4p_ew_prepared_artifact(pme_config: PMEParityConfig) -> PreparedSystem:
    water = tip4p_ew_reference_positions().astype(np.float32)
    positions = np.vstack([water, water + np.asarray([4.2, 0.3, 0.1], dtype=np.float32)])
    metadata = PreparedSystemMetadata(
        artifact_version=ARTIFACT_VERSION,
        source={"kind": "test_tip4p_ew"},
        selections={"atom_count": 8, "hydrogen_count": 4, "water_atom_count": 8},
        units={
            "coordinates": "angstrom",
            "mass": "dalton",
            "charge": "elementary_charge",
            "energy": "kilojoule_per_mole",
            "time": "picosecond",
            "temperature": "kelvin",
        },
        parameter_source="tip4p_ew_openmm_reference_test",
        compatibility_report={
            "engine": "mlx_atomistic",
            "production_force_field": True,
            "hydrogens_present": True,
            "hydrogen_count": 4,
            "water_present": True,
            "periodic_box_present": True,
            "supported_terms": ["virtual_site", "nonbonded_lj_coulomb", "pme", "water"],
            "required_terms": ["virtual_site", "nonbonded_lj_coulomb", "pme"],
            "unsupported_terms": [],
            "rejected_terms": [],
            "virtual_sites_present": True,
            "water_model": "tip4p_ew",
            "term_counts": {"virtual_site": 2},
        },
        pme_config=_pme_config_payload(pme_config),
    )
    return PreparedSystem(
        metadata=metadata,
        symbols=np.asarray(["O", "H", "H", "M", "O", "H", "H", "M"], dtype=str),
        atom_names=np.asarray(["O", "H1", "H2", "M", "O", "H1", "H2", "M"], dtype=str),
        atom_types=np.asarray(["OW", "HW", "HW", "MW", "OW", "HW", "HW", "MW"], dtype=str),
        residue_names=np.asarray(["WAT"] * 8, dtype=str),
        residue_ids=np.asarray([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32),
        chain_ids=np.asarray(["A"] * 8, dtype=str),
        positions=positions,
        velocities=np.zeros_like(positions, dtype=np.float32),
        masses=np.asarray([15.99943, 1.007947, 1.007947, 0.0] * 2, dtype=np.float32),
        charges=np.asarray(
            [0.0, TIP4P_EW_CHARGE_H, TIP4P_EW_CHARGE_H, TIP4P_EW_CHARGE_M] * 2,
            dtype=np.float32,
        ),
        sigma=np.asarray([TIP4P_EW_SIGMA_O_ANGSTROM, 1.0, 1.0, 1.0] * 2, dtype=np.float32),
        epsilon=np.asarray([TIP4P_EW_EPSILON_O_KJ_MOL, 0.0, 0.0, 0.0] * 2, dtype=np.float32),
        bonds=empty_indices(2),
        bond_k=np.asarray([], dtype=np.float32),
        bond_length=np.asarray([], dtype=np.float32),
        angles=empty_indices(3),
        angle_k=np.asarray([], dtype=np.float32),
        angle_theta=np.asarray([], dtype=np.float32),
        dihedrals=empty_indices(4),
        dihedral_k=np.asarray([], dtype=np.float32),
        dihedral_periodicity=np.asarray([], dtype=np.float32),
        dihedral_phase=np.asarray([], dtype=np.float32),
        nonbonded_pairs=empty_indices(2),
        ligand_mask=np.zeros((8,), dtype=bool),
        receptor_mask=np.zeros((8,), dtype=bool),
        restraint_mask=np.zeros((8,), dtype=bool),
        reference_positions=positions.copy(),
        cell_lengths=np.asarray(pme_config.cell_lengths_angstrom, dtype=np.float32),
        water_mask=np.ones((8,), dtype=bool),
        pme_mesh_shape=np.asarray(pme_config.mesh_shape, dtype=np.int32),
        pme_alpha=np.asarray([pme_config.alpha_per_angstrom], dtype=np.float32),
        pme_real_cutoff=np.asarray([pme_config.real_cutoff_angstrom], dtype=np.float32),
        pme_assignment_order=np.asarray([pme_config.assignment_order], dtype=np.int32),
        pme_charge_tolerance=np.asarray([pme_config.charge_tolerance], dtype=np.float32),
        pme_deconvolve_assignment=np.asarray([pme_config.deconvolve_assignment], dtype=bool),
        virtual_site_parent_atoms=np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32),
        virtual_site_weights=np.asarray(
            [[0.78664654, 0.10667673, 0.10667673, 0.0]] * 2,
            dtype=np.float32,
        ),
        virtual_site_types=np.asarray(["tip4p_ew", "tip4p_ew"], dtype=str),
    )


def run_charmm_openmm_mlx_parity(
    *,
    psf_path: str | Path,
    params: Sequence[str | Path],
    coords_path: str | Path,
    out_dir: str | Path,
    fixture: str = DEFAULT_CHARMM_FIXTURE,
    platform_name: str = "Reference",
    tolerances: ParityTolerances | None = None,
    mlx_nonbonded_cutoff_angstrom: float = 1.0e6,
    openmm_params: Sequence[str | Path] | None = None,
) -> OpenMMMLXParityReport:
    """Build a CHARMM artifact and compare fixed-coordinate OpenMM and MLX energies."""

    tolerances = ParityTolerances() if tolerances is None else tolerances
    psf = Path(psf_path)
    coords = Path(coords_path)
    native_params = tuple(Path(path) for path in params)
    reference_params = (
        tuple(Path(path) for path in openmm_params)
        if openmm_params is not None
        else native_params
    )
    out = Path(out_dir)
    prepared_dir = out / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "psf": str(psf),
        "params": [str(path) for path in native_params],
        "openmm_params": [str(path) for path in reference_params],
        "coordinates": str(coords),
        "topology": str(psf),
    }
    missing = _first_missing_path((psf, coords, *native_params, *reference_params))
    if missing is not None:
        return _blocked_report(
            fixture=fixture,
            source_kind="charmm",
            prmtop_path=psf,
            coords_path=coords,
            prepared_dir=prepared_dir,
            platform_name=platform_name,
            cutoff=mlx_nonbonded_cutoff_angstrom,
            tolerances=tolerances,
            blocker=f"missing CHARMM fixture input: {missing}",
            openmm_nonbonded_method="NoCutoff",
            source_paths=source_paths,
        )

    try:
        prepared = import_charmm_psf(psf_path=psf, params=native_params, coords_path=coords)
    except TopologyImportError as exc:
        blocker = str(exc)
        return _blocked_report(
            fixture=fixture,
            source_kind="charmm",
            prmtop_path=psf,
            coords_path=coords,
            prepared_dir=prepared_dir,
            platform_name=platform_name,
            cutoff=mlx_nonbonded_cutoff_angstrom,
            tolerances=tolerances,
            blocker=blocker,
            unsupported_terms=_unsupported_terms_from_blocker(blocker),
            openmm_nonbonded_method="NoCutoff",
            source_paths=source_paths,
        )

    return _run_prepared_openmm_mlx_parity(
        prepared=prepared,
        out_dir=out,
        prepared_dir=prepared_dir,
        fixture=fixture,
        source_kind="charmm",
        topology_path=psf,
        coords_path=coords,
        source_paths=source_paths,
        platform_name=platform_name,
        tolerances=tolerances,
        effective_cutoff=float(mlx_nonbonded_cutoff_angstrom),
        openmm_nonbonded_method="NoCutoff",
        pme_config=None,
        openmm_evaluator=lambda expected_counts: _evaluate_openmm_charmm(
            psf_path=psf,
            params=reference_params,
            coords_path=coords,
            platform_name=platform_name,
            expected_component_counts=expected_counts,
        ),
    )


def run_gromacs_openmm_mlx_parity(
    *,
    top_path: str | Path,
    gro_path: str | Path,
    out_dir: str | Path,
    fixture: str = DEFAULT_GROMACS_FIXTURE,
    platform_name: str = "Reference",
    tolerances: ParityTolerances | None = None,
    mlx_nonbonded_cutoff_angstrom: float = 1.0e6,
) -> OpenMMMLXParityReport:
    """Build a GROMACS artifact and compare fixed-coordinate OpenMM and MLX energies."""

    tolerances = ParityTolerances() if tolerances is None else tolerances
    top = Path(top_path)
    gro = Path(gro_path)
    out = Path(out_dir)
    prepared_dir = out / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "top": str(top),
        "gro": str(gro),
        "topology": str(top),
        "coordinates": str(gro),
    }
    missing = _first_missing_path((top, gro))
    if missing is not None:
        return _blocked_report(
            fixture=fixture,
            source_kind="gromacs",
            prmtop_path=top,
            coords_path=gro,
            prepared_dir=prepared_dir,
            platform_name=platform_name,
            cutoff=mlx_nonbonded_cutoff_angstrom,
            tolerances=tolerances,
            blocker=f"missing GROMACS fixture input: {missing}",
            openmm_nonbonded_method="NoCutoff",
            source_paths=source_paths,
        )

    try:
        prepared = import_gromacs_top_gro(top_path=top, gro_path=gro)
    except TopologyImportError as exc:
        blocker = str(exc)
        return _blocked_report(
            fixture=fixture,
            source_kind="gromacs",
            prmtop_path=top,
            coords_path=gro,
            prepared_dir=prepared_dir,
            platform_name=platform_name,
            cutoff=mlx_nonbonded_cutoff_angstrom,
            tolerances=tolerances,
            blocker=blocker,
            unsupported_terms=_unsupported_terms_from_blocker(blocker),
            openmm_nonbonded_method="NoCutoff",
            source_paths=source_paths,
        )

    return _run_prepared_openmm_mlx_parity(
        prepared=prepared,
        out_dir=out,
        prepared_dir=prepared_dir,
        fixture=fixture,
        source_kind="gromacs",
        topology_path=top,
        coords_path=gro,
        source_paths=source_paths,
        platform_name=platform_name,
        tolerances=tolerances,
        effective_cutoff=float(mlx_nonbonded_cutoff_angstrom),
        openmm_nonbonded_method="NoCutoff",
        pme_config=None,
        openmm_evaluator=lambda expected_counts: _evaluate_openmm_gromacs(
            top_path=top,
            gro_path=gro,
            platform_name=platform_name,
            expected_component_counts=expected_counts,
        ),
    )


def _blocked_report(
    *,
    fixture: str,
    source_kind: str = "amber",
    prmtop_path: Path,
    coords_path: Path,
    prepared_dir: Path,
    platform_name: str,
    cutoff: float,
    tolerances: ParityTolerances,
    blocker: str,
    atom_count: int = 0,
    unsupported_terms: tuple[str, ...] = (),
    artifact_readiness: dict[str, Any] | None = None,
    pme_config: PMEParityConfig | None = None,
    pme_readiness: dict[str, Any] | None = None,
    openmm_nonbonded_method: str | None = None,
    source_paths: dict[str, Any] | None = None,
) -> OpenMMMLXParityReport:
    readiness = _readiness_payload(
        artifact_readiness=artifact_readiness,
        pme_readiness=pme_readiness,
        unsupported_terms=unsupported_terms,
        blockers=(blocker,),
    )
    nonbonded_method = openmm_nonbonded_method or (
        "PME" if pme_config is not None else "NoCutoff"
    )
    report = OpenMMMLXParityReport(
        status="blocked",
        fixture=fixture,
        source_kind=source_kind,
        atom_count=atom_count,
        prepared_dir=str(prepared_dir),
        prmtop_path=str(prmtop_path),
        coords_path=str(coords_path),
        openmm_platform=platform_name,
        openmm_nonbonded_method=nonbonded_method,
        reference_engine="openmm",
        reference_engine_role=OPENMM_REFERENCE_ROLE,
        mlx_nonbonded_cutoff_angstrom=float(cutoff),
        artifact_readiness=artifact_readiness,
        pme_config=None if pme_config is None else _pme_config_payload(pme_config),
        pme_readiness=pme_readiness,
        readiness=readiness,
        platform_evidence=_parity_platform_evidence(
            status="blocked",
            fixture=fixture,
            finite_outputs=False,
            acceptance_criteria=_acceptance_criteria(source_kind),
            gap_ids=_gap_ids(source_kind),
            artifact_readiness=artifact_readiness,
            pme_readiness=pme_readiness,
            metrics={
                "blocker": blocker,
                "atom_count": atom_count,
                "source_kind": source_kind,
                "openmm_nonbonded_method": nonbonded_method,
                "unsupported_terms": list(unsupported_terms),
            },
        ),
        source_paths=source_paths
        or {
            "topology": str(prmtop_path),
            "coordinates": str(coords_path),
        },
        total_energy_openmm_kj_mol=None,
        total_energy_mlx_kj_mol=None,
        total_energy_abs_error_kj_mol=None,
        component_energy_openmm_kj_mol={},
        component_energy_mlx_kj_mol={},
        component_energy_abs_error_kj_mol={},
        force_shape=None,
        force_max_abs_error_kj_mol_nm=None,
        force_rms_abs_error_kj_mol_nm=None,
        unsupported_terms=unsupported_terms,
        unmapped_mlx_components=(),
        unmapped_openmm_components=(),
        tolerances=tolerances,
        passed=False,
        blockers=(blocker,),
    )
    _write_report(report, prepared_dir.parent)
    return report


def _run_prepared_openmm_mlx_parity(
    *,
    prepared: Any,
    out_dir: Path,
    prepared_dir: Path,
    fixture: str,
    source_kind: str,
    topology_path: Path,
    coords_path: Path,
    source_paths: dict[str, Any],
    platform_name: str,
    tolerances: ParityTolerances,
    effective_cutoff: float,
    openmm_nonbonded_method: str,
    pme_config: PMEParityConfig | None,
    openmm_evaluator: Any,
) -> OpenMMMLXParityReport:
    save_prepared_system(prepared, prepared_dir)
    artifact = load_prepared_mlx_artifact(prepared_dir, require_production=True)
    artifact.metadata["nonbonded_cutoff"] = effective_cutoff
    if pme_config is not None:
        artifact.metadata["electrostatics_model"] = "pme"
        artifact.metadata["pme_config"] = _pme_config_payload(pme_config)
    artifact_readiness = artifact_readiness_report(
        artifact.metadata,
        require_production=True,
        arrays=artifact.arrays,
    ).to_dict()
    expected_component_counts = _expected_component_counts(artifact)
    pme_readiness = _pme_readiness(artifact, pme_config)
    if pme_readiness is not None and pme_readiness["status"] != "ready":
        return _blocked_report(
            fixture=fixture,
            source_kind=source_kind,
            prmtop_path=topology_path,
            coords_path=coords_path,
            prepared_dir=prepared_dir,
            platform_name=platform_name,
            cutoff=effective_cutoff,
            tolerances=tolerances,
            blocker="PME readiness blocked: " + ", ".join(pme_readiness["blockers"]),
            atom_count=artifact.atom_count,
            artifact_readiness=artifact_readiness,
            pme_config=pme_config,
            pme_readiness=pme_readiness,
            openmm_nonbonded_method=openmm_nonbonded_method,
            source_paths=source_paths,
        )

    unsupported_terms = _unsupported_terms(artifact.metadata)
    if unsupported_terms:
        return _blocked_report(
            fixture=fixture,
            source_kind=source_kind,
            prmtop_path=topology_path,
            coords_path=coords_path,
            prepared_dir=prepared_dir,
            platform_name=platform_name,
            cutoff=effective_cutoff,
            tolerances=tolerances,
            blocker="unsupported force-field terms: " + ", ".join(unsupported_terms),
            atom_count=artifact.atom_count,
            unsupported_terms=unsupported_terms,
            artifact_readiness=artifact_readiness,
            pme_config=pme_config,
            pme_readiness=pme_readiness,
            openmm_nonbonded_method=openmm_nonbonded_method,
            source_paths=source_paths,
        )

    try:
        system, force_terms, _ = build_mlx_system_from_artifact(artifact)
    except MLXCompatibilityError as exc:
        return _blocked_report(
            fixture=fixture,
            source_kind=source_kind,
            prmtop_path=topology_path,
            coords_path=coords_path,
            prepared_dir=prepared_dir,
            platform_name=platform_name,
            cutoff=effective_cutoff,
            tolerances=tolerances,
            blocker=str(exc),
            atom_count=artifact.atom_count,
            unsupported_terms=unsupported_terms,
            artifact_readiness=artifact_readiness,
            pme_config=pme_config,
            pme_readiness=pme_readiness,
            openmm_nonbonded_method=openmm_nonbonded_method,
            source_paths=source_paths,
        )

    try:
        openmm_result = openmm_evaluator(expected_component_counts)
    except Exception as exc:  # pragma: no cover - covered through blocked report shape.
        blocker = (
            f"OpenMM {source_kind} reference load/evaluation failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return _blocked_report(
            fixture=fixture,
            source_kind=source_kind,
            prmtop_path=topology_path,
            coords_path=coords_path,
            prepared_dir=prepared_dir,
            platform_name=platform_name,
            cutoff=effective_cutoff,
            tolerances=tolerances,
            blocker=blocker,
            atom_count=artifact.atom_count,
            unsupported_terms=unsupported_terms,
            artifact_readiness=artifact_readiness,
            pme_config=pme_config,
            pme_readiness=pme_readiness,
            openmm_nonbonded_method=openmm_nonbonded_method,
            source_paths=source_paths,
        )
    openmm_unsupported = tuple(openmm_result.get("unsupported_terms", ()))
    if openmm_unsupported:
        combined_unsupported = tuple(dict.fromkeys((*unsupported_terms, *openmm_unsupported)))
        return _blocked_report(
            fixture=fixture,
            source_kind=source_kind,
            prmtop_path=topology_path,
            coords_path=coords_path,
            prepared_dir=prepared_dir,
            platform_name=str(openmm_result["platform"]),
            cutoff=effective_cutoff,
            tolerances=tolerances,
            blocker="unsupported OpenMM force classes: " + ", ".join(openmm_unsupported),
            atom_count=artifact.atom_count,
            unsupported_terms=combined_unsupported,
            artifact_readiness=artifact_readiness,
            pme_config=pme_config,
            pme_readiness=pme_readiness,
            openmm_nonbonded_method=openmm_nonbonded_method,
            source_paths=source_paths,
        )

    mlx_total, mlx_forces, mlx_components = _evaluate_mlx(
        force_terms,
        system.positions,
        system.cell,
    )
    component_errors = _component_errors(
        mlx_components,
        openmm_result["component_energy_kj_mol"],
    )
    missing_components = tuple(
        sorted(set(openmm_result["component_energy_kj_mol"]) - set(mlx_components))
    )
    mlx_forces_kj_mol_nm = mlx_forces * 10.0
    openmm_forces = openmm_result["forces_kj_mol_nm"]
    force_delta = mlx_forces_kj_mol_nm - openmm_forces
    force_max = float(np.max(np.abs(force_delta)))
    force_rms = float(np.sqrt(np.mean(force_delta * force_delta)))
    total_error = abs(float(mlx_total) - float(openmm_result["total_energy_kj_mol"]))
    passed = bool(
        not missing_components
        and total_error <= tolerances.total_energy_abs_kj_mol
        and all(
            error <= tolerances.component_energy_abs_kj_mol
            for error in component_errors.values()
        )
        and force_max <= tolerances.force_max_abs_kj_mol_nm
        and force_rms <= tolerances.force_rms_abs_kj_mol_nm
    )
    finite_outputs = bool(
        np.isfinite(
            [
                mlx_total,
                openmm_result["total_energy_kj_mol"],
                total_error,
                force_max,
                force_rms,
            ]
        ).all()
        and np.all(np.isfinite(mlx_forces))
        and np.all(np.isfinite(openmm_forces))
    )
    readiness = _readiness_payload(
        artifact_readiness=artifact_readiness,
        pme_readiness=pme_readiness,
        unsupported_terms=unsupported_terms,
        blockers=(),
    )
    status = "passed" if passed else "failed"
    platform_evidence = _parity_platform_evidence(
        status=status,
        fixture=fixture,
        finite_outputs=finite_outputs,
        acceptance_criteria=_acceptance_criteria(source_kind),
        gap_ids=_gap_ids(source_kind),
        artifact_readiness=artifact_readiness,
        pme_readiness=pme_readiness,
        metrics={
            "atom_count": artifact.atom_count,
            "source_kind": source_kind,
            "total_energy_abs_error_kj_mol": float(total_error),
            "component_energy_abs_error_kj_mol": component_errors,
            "force_max_abs_error_kj_mol_nm": force_max,
            "force_rms_abs_error_kj_mol_nm": force_rms,
            "openmm_nonbonded_method": openmm_nonbonded_method,
            "unmapped_openmm_components": list(missing_components),
        },
    )
    report = OpenMMMLXParityReport(
        status=status,
        fixture=fixture,
        source_kind=source_kind,
        atom_count=artifact.atom_count,
        prepared_dir=str(prepared_dir),
        prmtop_path=str(topology_path),
        coords_path=str(coords_path),
        openmm_platform=str(openmm_result["platform"]),
        openmm_nonbonded_method=openmm_nonbonded_method,
        reference_engine="openmm",
        reference_engine_role=OPENMM_REFERENCE_ROLE,
        mlx_nonbonded_cutoff_angstrom=effective_cutoff,
        artifact_readiness=artifact_readiness,
        pme_config=None if pme_config is None else _pme_config_payload(pme_config),
        pme_readiness=pme_readiness,
        readiness=readiness,
        platform_evidence=platform_evidence,
        source_paths=source_paths,
        total_energy_openmm_kj_mol=float(openmm_result["total_energy_kj_mol"]),
        total_energy_mlx_kj_mol=float(mlx_total),
        total_energy_abs_error_kj_mol=float(total_error),
        component_energy_openmm_kj_mol=openmm_result["component_energy_kj_mol"],
        component_energy_mlx_kj_mol=mlx_components,
        component_energy_abs_error_kj_mol=component_errors,
        force_shape=tuple(int(item) for item in openmm_forces.shape),
        force_max_abs_error_kj_mol_nm=force_max,
        force_rms_abs_error_kj_mol_nm=force_rms,
        unsupported_terms=unsupported_terms,
        unmapped_mlx_components=tuple(sorted(set(mlx_components) - set(MLX_COMPONENT_KEYS))),
        unmapped_openmm_components=missing_components,
        tolerances=tolerances,
        passed=passed,
    )
    _write_report(report, out_dir)
    return report


def _acceptance_criteria(source_kind: str) -> tuple[str, ...]:
    return ACCEPTANCE_CRITERIA_BY_SOURCE.get(source_kind, ("AC-07",))


def _gap_ids(source_kind: str) -> tuple[str, ...]:
    return GAP_IDS_BY_SOURCE.get(source_kind, ("P2-PARITY-01",))


def _readiness_payload(
    *,
    artifact_readiness: dict[str, Any] | None,
    pme_readiness: dict[str, Any] | None,
    unsupported_terms: tuple[str, ...],
    blockers: tuple[str, ...],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ready": not blockers
        and not unsupported_terms
        and (artifact_readiness is None or artifact_readiness.get("status") == "ready")
        and (pme_readiness is None or pme_readiness.get("status") == "ready"),
        "unsupported_terms": list(unsupported_terms),
        "blockers": list(blockers),
    }
    if artifact_readiness is not None:
        payload["artifact"] = artifact_readiness
    if pme_readiness is not None:
        payload["pme"] = pme_readiness
    return payload


def _parity_platform_evidence(
    *,
    status: str,
    fixture: str,
    finite_outputs: bool,
    acceptance_criteria: tuple[str, ...],
    gap_ids: tuple[str, ...],
    artifact_readiness: dict[str, Any] | None,
    pme_readiness: dict[str, Any] | None,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    readiness: dict[str, Any] = {}
    if artifact_readiness is not None:
        readiness["artifact"] = artifact_readiness
    if pme_readiness is not None:
        readiness["pme"] = pme_readiness
    return build_platform_validation_evidence(
        name="openmm_mlx_parity",
        status=status,
        fixture=fixture,
        acceptance_criteria=acceptance_criteria,
        gap_ids=gap_ids,
        finite_outputs=finite_outputs,
        reference_engine="openmm",
        reference_role=OPENMM_REFERENCE_ROLE,
        readiness=readiness,
        metrics=metrics,
    ).to_dict()


def _with_pme_artifact_settings(prepared: Any, config: PMEParityConfig) -> Any:
    pme_config = config.to_pme_config()
    report = dict(prepared.metadata.compatibility_report)
    required_terms = list(report.get("required_terms", ()))
    supported_terms = list(report.get("supported_terms", ()))
    for terms in (required_terms, supported_terms):
        if "pme" not in terms:
            terms.append("pme")
    report.update(
        {
            "periodic_box_present": True,
            "electrostatics_model": "pme",
            "required_terms": required_terms,
            "supported_terms": supported_terms,
        }
    )
    metadata = replace(
        prepared.metadata,
        compatibility_report=report,
        pme_config=_pme_config_payload(config),
    )
    return replace(
        prepared,
        metadata=metadata,
        cell_lengths=np.asarray(config.cell_lengths_angstrom, dtype=np.float32),
        cell_matrix=np.asarray([], dtype=np.float32),
        pme_mesh_shape=np.asarray(pme_config.mesh_shape, dtype=np.int32),
        pme_alpha=np.asarray([pme_config.alpha], dtype=np.float32),
        pme_real_cutoff=np.asarray([pme_config.real_cutoff], dtype=np.float32),
        pme_assignment_order=np.asarray([pme_config.assignment_order], dtype=np.int32),
        pme_charge_tolerance=np.asarray([pme_config.charge_tolerance], dtype=np.float32),
        pme_deconvolve_assignment=np.asarray([pme_config.deconvolve_assignment], dtype=bool),
    )


def _pme_config_payload(config: PMEParityConfig) -> dict[str, Any]:
    pme_config = config.to_pme_config()
    return {
        "mesh_shape": list(pme_config.mesh_shape),
        "alpha": float(pme_config.alpha),
        "real_cutoff": (
            None if pme_config.real_cutoff is None else float(pme_config.real_cutoff)
        ),
        "assignment_order": int(pme_config.assignment_order),
        "charge_tolerance": float(pme_config.charge_tolerance),
        "deconvolve_assignment": bool(pme_config.deconvolve_assignment),
    }


def _pme_readiness(artifact: Any, config: PMEParityConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None
    arrays = artifact.arrays
    topology = Topology.from_sequences(
        n_atoms=artifact.atom_count,
        bonds=np.asarray(arrays["bonds"], dtype=np.int32),
        angles=np.asarray(arrays["angles"], dtype=np.int32),
        dihedrals=np.asarray(arrays["dihedrals"], dtype=np.int32),
        impropers=np.asarray(arrays.get("impropers", np.empty((0, 4))), dtype=np.int32),
        partial_charges=np.asarray(arrays["charges"], dtype=np.float32),
        nonbonded_exception_pairs=np.asarray(
            arrays.get("nonbonded_exception_pairs", np.empty((0, 2))),
            dtype=np.int32,
        ),
        exclude_bonds=True,
        nonbonded_cutoff=float(config.real_cutoff_angstrom),
    )
    return pme_readiness_report(
        atom_count=artifact.atom_count,
        charges=arrays["charges"],
        cell_lengths=arrays.get("cell_lengths", np.asarray([])),
        config=config.to_pme_config(),
        nonbonded_cutoff=float(config.real_cutoff_angstrom),
        exclusion_count=len(topology.exclusion_set),
        one_four_count=len(topology.one_four_set),
        explicit_exception_count=int(
            np.asarray(arrays.get("nonbonded_exception_pairs", np.empty((0, 2)))).shape[0]
        ),
    )


def _unsupported_terms(metadata: dict[str, Any]) -> tuple[str, ...]:
    report = dict(metadata.get("compatibility_report", {}))
    terms = [*report.get("unsupported_terms", ()), *report.get("rejected_terms", ())]
    return tuple(str(term) for term in terms)


def _first_missing_path(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if not path.exists():
            return path
    return None


def _unsupported_terms_from_blocker(blocker: str) -> tuple[str, ...]:
    prefix = "unsupported_terms:"
    if not blocker.startswith(prefix):
        if blocker.startswith("unsupported AMBER 1-4 scaling"):
            return ("amber_invalid_14_scaling",)
        if blocker.startswith("unsupported AMBER exclusions"):
            return ("amber_malformed_exclusions",)
        if blocker.startswith("unsupported AMBER periodic box"):
            return ("amber_invalid_periodic_box",)
        return ()
    return tuple(term for term in blocker.removeprefix(prefix).split(",") if term)


def _expected_component_counts(artifact: Any) -> dict[str, int]:
    arrays = artifact.arrays
    return {
        "bond": _array_row_count(arrays.get("bonds")),
        "angle": _array_row_count(arrays.get("angles")),
        "torsion": (
            _array_row_count(arrays.get("dihedrals"))
            + _array_row_count(arrays.get("impropers"))
        ),
        "rb_dihedral": _array_row_count(arrays.get("rb_dihedrals")),
        "urey_bradley": _array_row_count(arrays.get("urey_bradley_terms")),
        "charmm_cmap": _array_row_count(arrays.get("charmm_cmap_terms")),
        "nonbonded": 1 if int(getattr(artifact, "atom_count", 0)) > 1 else 0,
    }


def _array_row_count(value: Any) -> int:
    if value is None:
        return 0
    array = np.asarray(value)
    if array.size == 0:
        return 0
    if array.ndim == 0:
        return int(array.size)
    return int(array.shape[0])


def _evaluate_mlx(
    force_terms: list[Any],
    positions: Any,
    cell: Any,
) -> tuple[float, np.ndarray, dict[str, float]]:
    total = mx.array(0.0, dtype=mx.float32)
    forces = mx.zeros_like(positions)
    components: dict[str, float] = {}
    for term in force_terms:
        term_name = str(getattr(term, "name", type(term).__name__))
        if hasattr(term, "energy_forces_with_components"):
            energy, term_forces, term_components = term.energy_forces_with_components(
                positions,
                cell,
            )
            for name, value in term_components.items():
                if isinstance(value, dict):
                    continue
                try:
                    array_value = np.asarray(value)
                except TypeError:
                    continue
                if array_value.shape != ():
                    continue
                try:
                    components[f"{term_name}.{name}"] = float(array_value)
                except (TypeError, ValueError):
                    continue
        else:
            energy, term_forces = term.energy_forces(positions, cell)
        components[term_name] = float(np.asarray(energy))
        total = total + energy
        forces = forces + term_forces
    mx.eval(total, forces)
    components["torsion"] = components.get("dihedral", 0.0) + components.get("improper", 0.0)
    if "charmm_cmap_terms" in components:
        components["charmm_cmap"] = components["charmm_cmap_terms"]
    return float(np.asarray(total)), np.asarray(forces, dtype=np.float64), components


def _evaluate_openmm_amber(
    *,
    prmtop_path: Path,
    coords_path: Path,
    platform_name: str,
    pme_config: PMEParityConfig | None = None,
    expected_component_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    import openmm as mm
    from openmm import app, unit

    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    coords = app.AmberInpcrdFile(str(coords_path))
    box_vectors = None
    if pme_config is not None:
        a, b, c = (float(item) * 0.1 for item in pme_config.cell_lengths_angstrom)
        box_vectors = (
            mm.Vec3(a, 0.0, 0.0),
            mm.Vec3(0.0, b, 0.0),
            mm.Vec3(0.0, 0.0, c),
        ) * unit.nanometer
        prmtop.topology.setPeriodicBoxVectors(box_vectors)
    system_kwargs: dict[str, Any] = {
        "nonbondedMethod": app.PME if pme_config is not None else app.NoCutoff,
        "constraints": None,
        "removeCMMotion": False,
    }
    if pme_config is not None:
        system_kwargs["nonbondedCutoff"] = (
            pme_config.real_cutoff_angstrom * 0.1 * unit.nanometer
        )
    openmm_system = prmtop.createSystem(**system_kwargs)
    if box_vectors is not None:
        openmm_system.setDefaultPeriodicBoxVectors(*box_vectors)
        for force_index in range(openmm_system.getNumForces()):
            force = openmm_system.getForce(force_index)
            if isinstance(force, mm.NonbondedForce):
                force.setPMEParameters(
                    pme_config.alpha_per_angstrom * 10.0 / unit.nanometer,
                    *pme_config.mesh_shape,
                )
    return _evaluate_openmm_system(
        openmm_system=openmm_system,
        positions=coords.positions,
        platform_name=platform_name,
        box_vectors=box_vectors,
        source_kind="amber",
        expected_component_counts=expected_component_counts,
    )


def _evaluate_openmm_charmm(
    *,
    psf_path: Path,
    params: Sequence[Path],
    coords_path: Path,
    platform_name: str,
    expected_component_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    from openmm import app

    psf = app.CharmmPsfFile(str(psf_path))
    pdb = app.PDBFile(str(coords_path))
    parameter_set = app.CharmmParameterSet(*(str(path) for path in params))
    openmm_system = psf.createSystem(
        parameter_set,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        removeCMMotion=False,
    )
    return _evaluate_openmm_system(
        openmm_system=openmm_system,
        positions=pdb.positions,
        platform_name=platform_name,
        source_kind="charmm",
        expected_component_counts=expected_component_counts,
    )


def _evaluate_openmm_gromacs(
    *,
    top_path: Path,
    gro_path: Path,
    platform_name: str,
    expected_component_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    from openmm import app

    gro = app.GromacsGroFile(str(gro_path))
    top = app.GromacsTopFile(
        str(top_path),
        periodicBoxVectors=gro.getPeriodicBoxVectors(),
        includeDir=str(top_path.parent),
    )
    openmm_system = top.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        removeCMMotion=False,
    )
    return _evaluate_openmm_system(
        openmm_system=openmm_system,
        positions=gro.positions,
        platform_name=platform_name,
        box_vectors=gro.getPeriodicBoxVectors(),
        source_kind="gromacs",
        expected_component_counts=expected_component_counts,
    )


def _evaluate_openmm_system(
    *,
    openmm_system: Any,
    positions: Any,
    platform_name: str,
    box_vectors: Any = None,
    source_kind: str = "",
    expected_component_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    import openmm as mm
    from openmm import unit

    for index in range(openmm_system.getNumForces()):
        openmm_system.getForce(index).setForceGroup(index)
    platform = mm.Platform.getPlatformByName(platform_name)
    context = mm.Context(
        openmm_system,
        mm.VerletIntegrator(0.001 * unit.picoseconds),
        platform,
    )
    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    components: dict[str, float] = {}
    unsupported: list[str] = []
    force_class_counts: dict[str, int] = {}
    for index in range(openmm_system.getNumForces()):
        force = openmm_system.getForce(index)
        group_state = context.getState(getEnergy=True, groups={index})
        force_class = type(force).__name__
        occurrence = force_class_counts.get(force_class, 0)
        force_class_counts[force_class] = occurrence + 1
        key = _openmm_component_name(
            force,
            source_kind=source_kind,
            occurrence=occurrence,
            expected_component_counts=expected_component_counts,
        )
        if key is None:
            if force_class != "CMMotionRemover":
                unsupported.append(force_class)
            continue
        components[key] = components.get(key, 0.0) + float(
            group_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
    forces = np.asarray(
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        ),
        dtype=np.float64,
    )
    return {
        "platform": context.getPlatform().getName(),
        "total_energy_kj_mol": float(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        ),
        "component_energy_kj_mol": components,
        "forces_kj_mol_nm": forces,
        "unsupported_terms": tuple(sorted(set(unsupported))),
    }


def _openmm_component_name(
    force: Any,
    *,
    source_kind: str = "",
    occurrence: int = 0,
    expected_component_counts: dict[str, int] | None = None,
) -> str | None:
    force_class_name = force if isinstance(force, str) else type(force).__name__
    if force_class_name == "HarmonicBondForce" and occurrence > 0:
        if (
            source_kind == "charmm"
            and occurrence == 1
            and _expects_component(expected_component_counts, "urey_bradley")
        ):
            return "urey_bradley"
        return None
    static_mapping = {
        "HarmonicBondForce": "bond",
        "HarmonicAngleForce": "angle",
        "PeriodicTorsionForce": "torsion",
        "CustomTorsionForce": "torsion",
        "RBTorsionForce": "rb_dihedral",
        "CMAPTorsionForce": "charmm_cmap",
        "NonbondedForce": "nonbonded",
        "CustomNonbondedForce": "nonbonded",
    }
    if force_class_name in static_mapping:
        return static_mapping[force_class_name]
    if force_class_name == "CustomBondForce" and not isinstance(force, str):
        if source_kind != "charmm" or not _expects_component(
            expected_component_counts,
            "urey_bradley",
        ):
            return None
        expression = str(force.getEnergyFunction()).lower()
        if "theta" not in expression and ("r0" in expression or "ub" in expression):
            return "urey_bradley"
    return None


def _expects_component(
    expected_component_counts: dict[str, int] | None,
    component: str,
) -> bool:
    if expected_component_counts is None:
        return False
    return int(expected_component_counts.get(component, 0)) > 0


def _component_errors(
    mlx_components: dict[str, float],
    openmm_components: dict[str, float],
) -> dict[str, float]:
    keys = sorted(set(openmm_components) & set(PARITY_COMPONENT_KEYS))
    return {
        key: abs(float(mlx_components[key]) - float(openmm_components[key]))
        for key in keys
        if key in mlx_components
    }


def _write_report(report: OpenMMMLXParityReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / REPORT_NAME).write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    )


__all__ = [
    "DEFAULT_AMBER_FIXTURE",
    "DEFAULT_CHARMM_FIXTURE",
    "DEFAULT_GROMACS_FIXTURE",
    "REPORT_NAME",
    "OpenMMMLXParityReport",
    "PMEParityConfig",
    "ParityTolerances",
    "default_amber_fixture_paths",
    "default_charmm_fixture_paths",
    "default_gromacs_fixture_paths",
    "evaluate_tip4p_ew_openmm_mlx_parity",
    "run_amber_openmm_mlx_parity",
    "run_charmm_openmm_mlx_parity",
    "run_gromacs_openmm_mlx_parity",
]
