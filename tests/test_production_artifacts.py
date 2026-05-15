from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mlx_atomistic.artifacts import (
    MLXCompatibilityError,
    PreparedMLXArtifact,
    build_mlx_system_from_artifact,
    load_prepared_mlx_artifact,
    validate_mlx_compatibility,
)
from mlx_atomistic.core import Cell
from mlx_atomistic.minimize import minimize_energy
from mlx_atomistic.prep.io import (
    load_prepared_system,
    save_prepared_system,
    synthetic_prepared_system,
)
from mlx_atomistic.topology import Topology


def _production_fixture():
    prepared = synthetic_prepared_system()
    metadata = replace(
        prepared.metadata,
        units={
            "coordinates": "angstrom",
            "mass": "dalton",
            "charge": "elementary_charge",
            "energy": "kilojoule_per_mole",
            "time": "picosecond",
            "temperature": "kelvin",
        },
        compatibility_report={
            "production_force_field": True,
            "hydrogens_present": True,
            "hydrogen_count": 1,
            "supported_terms": [
                "harmonic_bond",
                "nonbonded_lj_coulomb",
                "nonbonded_exception",
                "distance_constraint",
            ],
            "required_terms": [
                "harmonic_bond",
                "nonbonded_lj_coulomb",
                "nonbonded_exception",
                "distance_constraint",
            ],
            "unsupported_terms": [],
        },
        parameter_source="production_fixture",
    )
    return replace(
        prepared,
        metadata=metadata,
        symbols=np.asarray(["H", "O"], dtype=str),
        atom_names=np.asarray(["H1", "O1"], dtype=str),
        atom_types=np.asarray(["H", "O"], dtype=str),
        masses=np.asarray([1.008, 15.999], dtype=np.float32),
        constraints=np.asarray([[0, 1]], dtype=np.int32),
        constraint_distance=np.asarray([1.25], dtype=np.float32),
        nonbonded_exception_pairs=np.asarray([[0, 1]], dtype=np.int32),
        nonbonded_exception_charge_product=np.asarray([0.0], dtype=np.float32),
        nonbonded_exception_sigma=np.asarray([0.0], dtype=np.float32),
        nonbonded_exception_epsilon=np.asarray([0.0], dtype=np.float32),
    )


def test_core_artifact_loader_rejects_non_production_when_required(tmp_path):
    save_prepared_system(synthetic_prepared_system(), tmp_path)

    with pytest.raises(MLXCompatibilityError, match="not marked as a production"):
        load_prepared_mlx_artifact(tmp_path, require_production=True)


def test_pair_restricted_demo_term_validates_without_production_requirement(tmp_path):
    prepared = synthetic_prepared_system()
    metadata = replace(
        prepared.metadata,
        compatibility_report={
            "production_force_field": False,
            "supported_terms": ["harmonic_bond", "pair_restricted_lj_coulomb"],
            "unsupported_terms": [],
        },
    )
    prepared = replace(
        prepared,
        metadata=metadata,
        nonbonded_pairs=np.asarray([[0, 1]], dtype=np.int32),
    )
    save_prepared_system(prepared, tmp_path)

    artifact = load_prepared_mlx_artifact(tmp_path, require_production=False)
    system, terms, constraints = build_mlx_system_from_artifact(artifact)

    assert system.atom_count == prepared.atom_count
    assert [term.name for term in terms] == ["bond", "pair_restricted_nonbonded"]
    assert constraints is None
    with pytest.raises(MLXCompatibilityError, match="not marked as a production"):
        load_prepared_mlx_artifact(tmp_path, require_production=True)


def test_core_artifact_loader_builds_production_system_terms_and_constraints(tmp_path):
    prepared = _production_fixture()
    save_prepared_system(prepared, tmp_path)

    artifact = load_prepared_mlx_artifact(tmp_path, require_production=True)
    system, terms, constraints = build_mlx_system_from_artifact(artifact)

    assert system.atom_count == prepared.atom_count
    assert [term.name for term in terms] == ["bond", "nonbonded"]
    assert constraints is not None
    assert constraints.pairs.shape[0] == 1

    _, _, tuned_constraints = build_mlx_system_from_artifact(
        artifact,
        constraint_max_iterations=4,
    )
    assert tuned_constraints is not None
    assert tuned_constraints.max_iterations == 4


def test_large_artifact_build_defers_dense_topology_pairs(tmp_path):
    n_atoms = 2100
    arrays = {
        "symbols": np.asarray(["H"] * n_atoms, dtype=str),
        "atom_names": np.asarray([f"H{index}" for index in range(n_atoms)], dtype=str),
        "atom_types": np.asarray(["H"] * n_atoms, dtype=str),
        "positions": np.zeros((n_atoms, 3), dtype=np.float32),
        "velocities": np.zeros((n_atoms, 3), dtype=np.float32),
        "masses": np.ones(n_atoms, dtype=np.float32),
        "charges": np.zeros(n_atoms, dtype=np.float32),
        "sigma": np.ones(n_atoms, dtype=np.float32),
        "epsilon": np.zeros(n_atoms, dtype=np.float32),
        "bonds": np.asarray([[0, 1]], dtype=np.int32),
        "bond_k": np.asarray([1.0], dtype=np.float32),
        "bond_length": np.asarray([1.0], dtype=np.float32),
        "angles": np.empty((0, 3), dtype=np.int32),
        "angle_k": np.asarray([], dtype=np.float32),
        "angle_theta": np.asarray([], dtype=np.float32),
        "dihedrals": np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        "dihedral_k": np.asarray([0.0], dtype=np.float32),
        "dihedral_periodicity": np.asarray([1.0], dtype=np.float32),
        "dihedral_phase": np.asarray([0.0], dtype=np.float32),
        "nonbonded_exception_pairs": np.asarray([[2, 3]], dtype=np.int32),
        "nonbonded_exception_charge_product": np.asarray([0.0], dtype=np.float32),
        "nonbonded_exception_sigma": np.asarray([0.0], dtype=np.float32),
        "nonbonded_exception_epsilon": np.asarray([0.0], dtype=np.float32),
    }
    metadata = {
        "nonbonded_cutoff": 10.0,
        "compatibility_report": {
            "production_force_field": True,
            "supported_terms": [
                "harmonic_bond",
                "periodic_dihedral",
                "nonbonded_lj_coulomb",
                "nonbonded_exception",
            ],
            "required_terms": [
                "harmonic_bond",
                "periodic_dihedral",
                "nonbonded_lj_coulomb",
                "nonbonded_exception",
            ],
            "unsupported_terms": [],
        },
    }
    artifact = PreparedMLXArtifact(tmp_path, metadata, arrays, unit_system=None)

    system, terms, constraints = build_mlx_system_from_artifact(artifact)

    assert constraints is None
    assert system.topology.nonbonded_pair_policy == "lazy"
    assert system.topology._nonbonded_pairs is None
    assert system.topology.nonbonded_build_report == {
        "pair_policy": "lazy",
        "atom_count": n_atoms,
        "cutoff": 10.0,
        "exclusions": 2,
        "exceptions": 1,
        "one_four_pairs": 1,
        "nonbonded_pairs": n_atoms * (n_atoms - 1) // 2 - 2,
    }
    nonbonded = terms[-1]
    pairs_backend = replace(nonbonded, backend="mlx_pairs")
    with pytest.raises(
        ValueError,
        match="lazy topology requires a runtime nonbonded pair provider",
    ):
        pairs_backend.energy_forces(system.positions, cell=system.cell)
    assert system.topology._nonbonded_pairs is None


def test_compatibility_report_fails_closed_for_unsupported_terms():
    metadata = _production_fixture().metadata.to_json_dict()
    metadata["compatibility_report"]["unsupported_terms"] = ["PME"]

    with pytest.raises(MLXCompatibilityError, match="unsupported force-field terms"):
        validate_mlx_compatibility(metadata, require_production=True)


def test_artifact_electrostatics_modes_are_validated():
    metadata = _production_fixture().metadata.to_json_dict()
    metadata["compatibility_report"]["electrostatics_model"] = "ewald_reference"

    unit_system = validate_mlx_compatibility(metadata, require_production=True)

    assert unit_system is not None

    metadata["compatibility_report"]["electrostatics_model"] = "pme"
    with pytest.raises(MLXCompatibilityError, match="pme_config"):
        validate_mlx_compatibility(metadata, require_production=True)

    metadata["pme_config"] = {
        "mesh_shape": [8, 8, 8],
        "alpha": 0.35,
        "real_cutoff": 5.0,
        "assignment_order": 2,
        "charge_tolerance": 1e-5,
    }
    assert validate_mlx_compatibility(metadata, require_production=True) is not None

    metadata["compatibility_report"]["electrostatics_model"] = "reaction_field"
    with pytest.raises(MLXCompatibilityError, match="unknown electrostatics mode"):
        validate_mlx_compatibility(metadata, require_production=True)


def test_artifact_required_pme_term_requires_config():
    metadata = _production_fixture().metadata.to_json_dict()
    metadata["compatibility_report"]["required_terms"] = [
        "nonbonded_lj_coulomb",
        "pme_ewald_periodic_electrostatics",
    ]

    with pytest.raises(MLXCompatibilityError, match="pme_config"):
        validate_mlx_compatibility(metadata, require_production=True)


def test_virtual_site_water_model_fails_closed(tmp_path):
    prepared = _production_fixture()
    metadata = replace(
        prepared.metadata,
        compatibility_report={
            **prepared.metadata.compatibility_report,
            "water_model": "TIP4P-Ew",
        },
    )
    save_prepared_system(replace(prepared, metadata=metadata), tmp_path)

    with pytest.raises(MLXCompatibilityError, match="virtual_site water model"):
        load_prepared_mlx_artifact(tmp_path, require_production=True)


def test_hidden_hmr_masses_fail_closed_without_policy(tmp_path):
    prepared = replace(
        _production_fixture(),
        masses=np.asarray([3.024, 13.983], dtype=np.float32),
    )
    save_prepared_system(prepared, tmp_path)

    with pytest.raises(MLXCompatibilityError, match="hydrogen_mass_repartitioning detected"):
        load_prepared_mlx_artifact(tmp_path, require_production=True)


def test_declared_hmr_masses_are_represented_by_artifact_masses(tmp_path):
    prepared = _production_fixture()
    metadata = replace(
        prepared.metadata,
        compatibility_report={
            **prepared.metadata.compatibility_report,
            "hydrogen_mass_repartitioning": "represented_by_masses",
        },
    )
    prepared = replace(
        prepared,
        metadata=metadata,
        masses=np.asarray([3.024, 13.983], dtype=np.float32),
    )
    save_prepared_system(prepared, tmp_path)

    artifact = load_prepared_mlx_artifact(tmp_path, require_production=True)
    system, _, constraints = build_mlx_system_from_artifact(artifact)

    assert constraints is not None
    np.testing.assert_allclose(np.asarray(system.masses), [3.024, 13.983])


def test_hmr_force_term_request_fails_closed(tmp_path):
    prepared = _production_fixture()
    report = {
        **prepared.metadata.compatibility_report,
        "required_terms": [
            *prepared.metadata.compatibility_report["required_terms"],
            "hydrogen_mass_repartitioning",
        ],
    }
    save_prepared_system(
        replace(prepared, metadata=replace(prepared.metadata, compatibility_report=report)),
        tmp_path,
    )

    with pytest.raises(MLXCompatibilityError, match="hydrogen_mass_repartitioning"):
        load_prepared_mlx_artifact(tmp_path, require_production=True)


def test_gpcrmd_electrostatics_gate_accepts_ready_pme_artifact(tmp_path):
    from mlx_atomistic.prep.runner import _gpcrmd_electrostatics_report

    prepared = replace(
        _pme_fixture_with_config_arrays(),
        metadata=replace(
            _pme_fixture_with_config_arrays().metadata,
            source={"kind": "gpcrmd", "gpcrmd_target_id": "fixture"},
        ),
    )
    save_prepared_system(prepared, tmp_path)

    report = _gpcrmd_electrostatics_report(tmp_path, requested_electrostatics="pme")

    assert report["status"] == "ready"
    assert report["route"] == "pme"
    assert report["production_ready"] is True
    assert report["backend"] == "mlx_fft_cic"
    assert report["blockers"] == ()


def test_gpcrmd_short_range_prototype_report_is_explicitly_non_production(tmp_path):
    from mlx_atomistic.prep.runner import _gpcrmd_electrostatics_report

    prepared = replace(
        _production_fixture(),
        metadata=replace(
            _production_fixture().metadata,
            source={"kind": "gpcrmd", "gpcrmd_target_id": "fixture"},
        ),
    )
    save_prepared_system(prepared, tmp_path)

    report = _gpcrmd_electrostatics_report(
        tmp_path,
        requested_electrostatics="short-range-prototype",
    )

    assert report["status"] == "prototype_allowed"
    assert report["route"] == "short-range-prototype"
    assert report["metadata_model"] == "short_range_electrostatics_prototype"
    assert report["production_ready"] is False
    assert report["blockers"] == ()
    assert "not production GPCRmd PME" in report["warnings"][0]


def test_gpcrmd_production_neighbor_manager_requires_optimized_cutoff_route():
    from mlx_atomistic.prep.runner import GPCRMD_NEIGHBOR_SKIN, _production_neighbor_manager

    system = SimpleNamespace(cell=Cell.cubic(4.0))
    topology = Topology.from_sequences(n_atoms=4, eager_nonbonded_pair_limit=0)
    cutoff_term = SimpleNamespace(topology=topology, cutoff=1.0, electrostatics="cutoff")
    manager = _production_neighbor_manager(
        system,
        (cutoff_term,),
        require_production=True,
    )

    assert manager is not None
    assert manager.skin == GPCRMD_NEIGHBOR_SKIN

    pme_term = SimpleNamespace(topology=topology, cutoff=1.0, electrostatics="pme")
    with pytest.raises(ValueError, match="optimized cutoff neighbor route"):
        _production_neighbor_manager(
            system,
            (pme_term,),
            require_production=True,
        )


def test_gpcrmd_short_range_prototype_pme_artifact_runs_cutoff_not_pme(tmp_path):
    from mlx_atomistic.io import load_npz_trajectory
    from mlx_atomistic.prep.runner import TRAJECTORY_NAME, run_gpcrmd_mlx

    prepared = _pme_fixture_with_config_arrays()
    prepared = replace(
        prepared,
        metadata=replace(
            prepared.metadata,
            source={
                "kind": "gpcrmd",
                "gpcrmd_target_id": "fixture",
                "gpcrmd_dynamics_id": 1,
            },
        ),
    )
    save_prepared_system(prepared, tmp_path)

    payload = run_gpcrmd_mlx(
        out=tmp_path,
        steps=1,
        sample_interval=1,
        temperature=0.0,
        restraint_k=0.0,
        minimize_steps=0,
        equilibration_steps=0,
        diagnostic_interval=1,
        electrostatics="short-range-prototype",
    )
    record = load_npz_trajectory(tmp_path / TRAJECTORY_NAME)

    assert payload["status"] == "ran"
    assert payload["electrostatics_report"]["artifact_electrostatics_model"] == "pme"
    assert payload["run_metadata"]["electrostatics_model"] == (
        "short_range_electrostatics_prototype"
    )
    assert payload["run_metadata"]["electrostatics_production_ready"] is False
    assert "nonbonded.lj" in record.potential_energy_by_term
    assert "nonbonded.coulomb" in record.potential_energy_by_term
    assert not any("pme" in name for name in record.potential_energy_by_term)


def test_pme_artifact_builds_nonbonded_pme_with_config_arrays(tmp_path):
    prepared = _production_fixture()
    metadata = replace(
        prepared.metadata,
        pme_config={
            "mesh_shape": [8, 8, 8],
            "alpha": 0.35,
            "real_cutoff": 5.0,
            "assignment_order": 2,
            "charge_tolerance": 1e-5,
        },
        compatibility_report={
            **prepared.metadata.compatibility_report,
            "electrostatics_model": "pme",
            "supported_terms": [
                *prepared.metadata.compatibility_report["supported_terms"],
                "pme_mesh_periodic_electrostatics",
            ],
            "required_terms": [
                *prepared.metadata.compatibility_report["required_terms"],
                "pme_mesh_periodic_electrostatics",
            ],
        },
    )
    prepared = replace(
        prepared,
        metadata=metadata,
        cell_lengths=np.asarray([12.0, 12.0, 12.0], dtype=np.float32),
        pme_mesh_shape=np.asarray([8, 8, 8], dtype=np.int32),
        pme_alpha=np.asarray([0.35], dtype=np.float32),
        pme_real_cutoff=np.asarray([5.0], dtype=np.float32),
        pme_assignment_order=np.asarray([2], dtype=np.int32),
        pme_charge_tolerance=np.asarray([1e-5], dtype=np.float32),
        pme_deconvolve_assignment=np.asarray([True], dtype=bool),
    )
    save_prepared_system(prepared, tmp_path)

    artifact = load_prepared_mlx_artifact(tmp_path, require_production=True)
    system, terms, _ = build_mlx_system_from_artifact(artifact)
    nonbonded = terms[-1]
    energy, forces, components = nonbonded.energy_forces_with_components(
        system.positions,
        system.cell,
    )

    assert nonbonded.electrostatics == "pme"
    assert nonbonded.pme_config.mesh_shape == (8, 8, 8)
    assert np.isfinite(float(np.asarray(energy)))
    assert np.all(np.isfinite(np.asarray(forces)))
    assert "pme_diagnostics" in components


def _pme_fixture_with_config_arrays():
    prepared = _production_fixture()
    metadata = replace(
        prepared.metadata,
        pme_config={
            "mesh_shape": [8, 8, 8],
            "alpha": 0.35,
            "real_cutoff": 5.0,
            "assignment_order": 2,
            "charge_tolerance": 1e-5,
        },
        compatibility_report={
            **prepared.metadata.compatibility_report,
            "electrostatics_model": "pme",
            "supported_terms": [
                *prepared.metadata.compatibility_report["supported_terms"],
                "pme_mesh_periodic_electrostatics",
            ],
            "required_terms": [
                *prepared.metadata.compatibility_report["required_terms"],
                "pme_mesh_periodic_electrostatics",
            ],
        },
    )
    return replace(
        prepared,
        metadata=metadata,
        cell_lengths=np.asarray([12.0, 12.0, 12.0], dtype=np.float32),
        pme_mesh_shape=np.asarray([8, 8, 8], dtype=np.int32),
        pme_alpha=np.asarray([0.35], dtype=np.float32),
        pme_real_cutoff=np.asarray([5.0], dtype=np.float32),
        pme_assignment_order=np.asarray([2], dtype=np.int32),
        pme_charge_tolerance=np.asarray([1e-5], dtype=np.float32),
        pme_deconvolve_assignment=np.asarray([True], dtype=bool),
    )


def _replace_npz_array(artifact_dir, name, value):
    npz_path = artifact_dir / "prepared_system.npz"
    with np.load(npz_path, allow_pickle=False) as data:
        payload = {array_name: np.asarray(data[array_name]) for array_name in data.files}
    payload[name] = np.asarray(value)
    np.savez_compressed(npz_path, **payload)


@pytest.mark.parametrize(
    ("array_name", "value", "match"),
    [
        ("pme_alpha", np.asarray([np.inf], dtype=np.float32), "pme_alpha"),
        ("pme_alpha", np.asarray([0.0], dtype=np.float32), "pme_alpha"),
        ("pme_real_cutoff", np.asarray([np.inf], dtype=np.float32), "pme_real_cutoff"),
        ("pme_real_cutoff", np.asarray([-1.0], dtype=np.float32), "pme_real_cutoff"),
        ("pme_assignment_order", np.asarray([3], dtype=np.int32), "pme_assignment_order"),
        ("pme_assignment_order", np.asarray([2.5], dtype=np.float32), "pme_assignment_order"),
        ("pme_charge_tolerance", np.asarray([np.nan], dtype=np.float32), "pme_charge_tolerance"),
        ("pme_charge_tolerance", np.asarray([-1.0], dtype=np.float32), "pme_charge_tolerance"),
        ("pme_mesh_shape", np.asarray([8, 8, 3], dtype=np.int32), "pme_mesh_shape"),
        ("pme_mesh_shape", np.asarray([8.5, 8, 8], dtype=np.float32), "pme_mesh_shape"),
        ("pme_mesh_shape", np.asarray([np.inf, 8, 8], dtype=np.float32), "pme_mesh_shape"),
        ("pme_mesh_shape", np.asarray([8, 8], dtype=np.int32), "pme_mesh_shape"),
    ],
)
def test_pme_artifact_load_rejects_invalid_config_arrays(
    tmp_path,
    array_name,
    value,
    match,
):
    save_prepared_system(_pme_fixture_with_config_arrays(), tmp_path)
    _replace_npz_array(tmp_path, array_name, value)

    with pytest.raises(MLXCompatibilityError, match=match):
        load_prepared_mlx_artifact(tmp_path, require_production=True)


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("pme_alpha", np.asarray([np.inf], dtype=np.float32), "pme_alpha"),
        ("pme_real_cutoff", np.asarray([np.inf], dtype=np.float32), "pme_real_cutoff"),
        ("pme_assignment_order", np.asarray([3], dtype=np.int32), "pme_assignment_order"),
        ("pme_assignment_order", np.asarray([2.5], dtype=np.float32), "pme_assignment_order"),
        ("pme_charge_tolerance", np.asarray([-1.0], dtype=np.float32), "pme_charge_tolerance"),
        ("pme_mesh_shape", np.asarray([8, 8, 3], dtype=np.int32), "pme_mesh_shape"),
        ("pme_mesh_shape", np.asarray([8.5, 8, 8], dtype=np.float32), "pme_mesh_shape"),
        ("pme_mesh_shape", np.asarray([np.inf, 8, 8], dtype=np.float32), "pme_mesh_shape"),
        ("pme_mesh_shape", np.asarray([8, 8], dtype=np.int32), "pme_mesh_shape"),
    ],
)
def test_prepared_system_validate_rejects_invalid_pme_arrays(field_name, value, match):
    prepared = replace(_pme_fixture_with_config_arrays(), **{field_name: value})

    with pytest.raises(ValueError, match=match):
        prepared.validate()


def test_ewald_reference_artifact_builds_force_term_with_valid_cell(tmp_path):
    prepared = _production_fixture()
    metadata = replace(
        prepared.metadata,
        compatibility_report={
            **prepared.metadata.compatibility_report,
            "electrostatics_model": "ewald_reference",
            "supported_terms": [
                *prepared.metadata.compatibility_report["supported_terms"],
                "ewald_reference_electrostatics",
            ],
            "required_terms": [
                *prepared.metadata.compatibility_report["required_terms"],
                "ewald_reference_electrostatics",
            ],
        },
    )
    prepared = replace(
        prepared,
        metadata=metadata,
        cell_lengths=np.asarray([12.0, 12.0, 12.0], dtype=np.float32),
    )
    save_prepared_system(prepared, tmp_path)

    artifact = load_prepared_mlx_artifact(tmp_path, require_production=True)
    system, terms, _ = build_mlx_system_from_artifact(artifact)
    nonbonded = terms[-1]
    energy, forces, components = nonbonded.energy_forces_with_components(
        system.positions,
        system.cell,
    )

    assert nonbonded.electrostatics == "ewald_reference"
    assert system.cell is not None
    assert np.isfinite(float(np.asarray(energy)))
    assert np.all(np.isfinite(np.asarray(forces)))
    assert "coulomb_real" in components
    assert "coulomb_self" in components


def test_ewald_reference_artifact_requires_cell_and_neutral_charges(tmp_path):
    prepared = _production_fixture()
    metadata = replace(
        prepared.metadata,
        compatibility_report={
            **prepared.metadata.compatibility_report,
            "electrostatics_model": "ewald_reference",
        },
    )
    save_prepared_system(replace(prepared, metadata=metadata), tmp_path / "missing-cell")

    with pytest.raises(MLXCompatibilityError, match="cell_lengths"):
        load_prepared_mlx_artifact(tmp_path / "missing-cell", require_production=True)

    charged = replace(
        prepared,
        metadata=metadata,
        charges=np.asarray([0.2, -0.1], dtype=np.float32),
        cell_lengths=np.asarray([12.0, 12.0, 12.0], dtype=np.float32),
    )
    save_prepared_system(charged, tmp_path / "charged")

    with pytest.raises(MLXCompatibilityError, match="must be neutral"):
        load_prepared_mlx_artifact(tmp_path / "charged", require_production=True)


def _charmm_artifact_fixture():
    prepared = synthetic_prepared_system()
    n_atoms = 8
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [1.4, 1.0, 0.2],
            [1.8, 1.2, 1.1],
            [0.2, 1.6, 0.1],
            [1.1, 1.7, 0.4],
            [1.6, 2.5, 0.9],
            [2.1, 2.4, 1.7],
        ],
        dtype=np.float32,
    )
    metadata = replace(
        prepared.metadata,
        units={
            "coordinates": "angstrom",
            "mass": "dalton",
            "charge": "elementary_charge",
            "energy": "kilojoule_per_mole",
            "time": "picosecond",
            "temperature": "kelvin",
        },
        parameter_source="charmm36_fixture",
        protocol_metadata={
            "ensemble": "nvt",
            "temperature_kelvin": 310.0,
            "timestep_femtoseconds": 2.0,
        },
        compatibility_report={
            "production_force_field": True,
            "hydrogens_present": True,
            "hydrogen_count": n_atoms,
            "supported_terms": [
                "nonbonded_lj_coulomb",
                "charmm_cmap_terms",
                "urey_bradley",
                "nbfix_pair_overrides",
                "lipid",
                "water",
                "ion",
                "receptor",
                "ligand",
            ],
            "required_terms": [
                "nonbonded_lj_coulomb",
                "charmm_cmap_terms",
                "urey_bradley",
                "nbfix_pair_overrides",
                "lipid",
                "water",
                "ion",
                "receptor",
                "ligand",
            ],
            "unsupported_terms": [],
        },
    )
    return replace(
        prepared,
        metadata=metadata,
        symbols=np.asarray(["H"] * n_atoms, dtype=str),
        atom_names=np.asarray([f"H{index}" for index in range(n_atoms)], dtype=str),
        atom_types=np.asarray(["H"] * n_atoms, dtype=str),
        residue_names=np.asarray(
            ["LIG", "REC", "REC", "LIP", "LIP", "HOH", "NA", "REC"],
            dtype=str,
        ),
        residue_ids=np.arange(1, n_atoms + 1, dtype=np.int32),
        chain_ids=np.asarray(["A"] * n_atoms, dtype=str),
        positions=positions,
        velocities=np.zeros((n_atoms, 3), dtype=np.float32),
        masses=np.ones((n_atoms,), dtype=np.float32),
        charges=np.zeros((n_atoms,), dtype=np.float32),
        sigma=np.ones((n_atoms,), dtype=np.float32),
        epsilon=np.full((n_atoms,), 0.1, dtype=np.float32),
        bonds=np.empty((0, 2), dtype=np.int32),
        bond_k=np.asarray([], dtype=np.float32),
        bond_length=np.asarray([], dtype=np.float32),
        angles=np.empty((0, 3), dtype=np.int32),
        angle_k=np.asarray([], dtype=np.float32),
        angle_theta=np.asarray([], dtype=np.float32),
        dihedrals=np.empty((0, 4), dtype=np.int32),
        dihedral_k=np.asarray([], dtype=np.float32),
        dihedral_periodicity=np.asarray([], dtype=np.float32),
        dihedral_phase=np.asarray([], dtype=np.float32),
        nonbonded_pairs=np.empty((0, 2), dtype=np.int32),
        ligand_mask=np.asarray([True, False, False, False, False, False, False, False]),
        receptor_mask=np.asarray([False, True, True, False, False, False, False, True]),
        restraint_mask=np.zeros((n_atoms,), dtype=bool),
        reference_positions=positions.copy(),
        water_mask=np.asarray([False, False, False, False, False, True, False, False]),
        ion_mask=np.asarray([False, False, False, False, False, False, True, False]),
        lipid_mask=np.asarray([False, False, False, True, True, False, False, False]),
        charmm_cmap_terms=np.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32),
        charmm_cmap_grid_indices=np.asarray([0], dtype=np.int32),
        charmm_cmap_grids=np.zeros((1, 6, 6), dtype=np.float32),
        urey_bradley_terms=np.asarray([[0, 1, 2]], dtype=np.int32),
        urey_bradley_k=np.asarray([3.0], dtype=np.float32),
        urey_bradley_distance=np.asarray([1.4], dtype=np.float32),
        nbfix_pairs=np.asarray([[0, 1]], dtype=np.int32),
        nbfix_sigma=np.asarray([1.1], dtype=np.float32),
        nbfix_epsilon=np.asarray([0.2], dtype=np.float32),
    )


def test_charmm_lipid_protocol_arrays_round_trip_and_build(tmp_path):
    prepared = _charmm_artifact_fixture()
    save_prepared_system(prepared, tmp_path)

    reloaded = load_prepared_system(tmp_path)
    artifact = load_prepared_mlx_artifact(tmp_path, require_production=True)
    system, terms, constraints = build_mlx_system_from_artifact(artifact)

    assert constraints is None
    assert reloaded.metadata.artifact_version == 2
    assert reloaded.metadata.protocol_metadata["ensemble"] == "nvt"
    assert reloaded.lipid_mask.shape == (prepared.atom_count,)
    assert reloaded.charmm_cmap_terms.shape == (1, 8)
    assert reloaded.urey_bradley_terms.shape == (1, 3)
    assert reloaded.nbfix_pairs.shape == (1, 2)
    assert [term.name for term in terms] == [
        "urey_bradley",
        "charmm_cmap_terms",
        "nonbonded",
    ]
    for term in terms:
        energy, forces = term.energy_forces(system.positions, system.cell)
        assert np.isfinite(np.asarray(energy)).all()
        assert np.isfinite(np.asarray(forces)).all()


def test_requested_charmm_term_requires_shape_valid_arrays(tmp_path):
    prepared = _production_fixture()
    metadata = replace(
        prepared.metadata,
        compatibility_report={
            **prepared.metadata.compatibility_report,
            "supported_terms": [
                *prepared.metadata.compatibility_report["supported_terms"],
                "charmm_cmap_terms",
            ],
            "required_terms": [
                *prepared.metadata.compatibility_report["required_terms"],
                "charmm_cmap_terms",
            ],
        },
    )
    save_prepared_system(replace(prepared, metadata=metadata), tmp_path)

    with pytest.raises(MLXCompatibilityError, match="charmm_cmap_terms"):
        load_prepared_mlx_artifact(tmp_path, require_production=True)


def test_charmm_arrays_cannot_be_hidden_by_metadata_only(tmp_path):
    prepared = _charmm_artifact_fixture()
    report = {
        **prepared.metadata.compatibility_report,
        "supported_terms": ["nonbonded_lj_coulomb"],
        "required_terms": ["nonbonded_lj_coulomb"],
        "rejected_terms": [],
    }
    save_prepared_system(
        replace(prepared, metadata=replace(prepared.metadata, compatibility_report=report)),
        tmp_path,
    )

    with pytest.raises(MLXCompatibilityError, match="undeclared force-field arrays"):
        load_prepared_mlx_artifact(tmp_path, require_production=True)


def test_in_memory_runner_rejects_hidden_charmm_arrays():
    from mlx_atomistic.prep.runner import build_mlx_system

    prepared = _charmm_artifact_fixture()
    report = {
        **prepared.metadata.compatibility_report,
        "supported_terms": ["nonbonded_lj_coulomb"],
        "required_terms": ["nonbonded_lj_coulomb"],
        "rejected_terms": [],
    }
    prepared = replace(
        prepared,
        metadata=replace(prepared.metadata, compatibility_report=report),
    )

    with pytest.raises(MLXCompatibilityError, match="undeclared force-field arrays"):
        build_mlx_system(prepared, require_production=True)


def test_compact_nbfix_arrays_cannot_be_hidden_by_metadata_only(tmp_path):
    prepared = replace(
        _production_fixture(),
        nbfix_type_pairs=np.asarray([["O", "H"]], dtype=str),
        nbfix_type_sigma=np.asarray([2.672696], dtype=np.float32),
        nbfix_type_epsilon=np.asarray([0.4184], dtype=np.float32),
    )
    save_prepared_system(prepared, tmp_path)

    with pytest.raises(
        MLXCompatibilityError,
        match="undeclared force-field arrays: nbfix_pair_overrides",
    ):
        load_prepared_mlx_artifact(tmp_path, require_production=True)


def test_declared_compact_nbfix_artifact_builds_nonbonded_runtime(tmp_path):
    prepared = _production_fixture()
    report = {
        **prepared.metadata.compatibility_report,
        "supported_terms": [
            *prepared.metadata.compatibility_report["supported_terms"],
            "nbfix_pair_overrides",
        ],
        "required_terms": [
            *prepared.metadata.compatibility_report["required_terms"],
            "nbfix_pair_overrides",
        ],
    }
    prepared = replace(
        prepared,
        metadata=replace(prepared.metadata, compatibility_report=report),
        nbfix_type_pairs=np.asarray([["O", "H"], ["CLGR1", "H"]], dtype=str),
        nbfix_type_sigma=np.asarray([2.672696, 1.1], dtype=np.float32),
        nbfix_type_epsilon=np.asarray([0.4184, 0.2], dtype=np.float32),
    )
    save_prepared_system(prepared, tmp_path)

    artifact = load_prepared_mlx_artifact(tmp_path, require_production=True)
    system, terms, _ = build_mlx_system_from_artifact(artifact)
    nonbonded = next(term for term in terms if term.name == "nonbonded")

    assert artifact.arrays["nbfix_type_pairs"].tolist() == [["O", "H"], ["CLGR1", "H"]]
    assert nonbonded.nbfix_type_pairs.tolist() == [["O", "H"]]
    assert nonbonded.has_nbfix
    assert all(term.name != "nbfix_pair_overrides" for term in terms)
    energy, forces = nonbonded.energy_forces(system.positions, system.cell)
    assert np.isfinite(np.asarray(energy)).all()
    assert np.isfinite(np.asarray(forces)).all()


def test_nbfix_respects_explicit_exception_pairs_in_artifacts(tmp_path):
    prepared = _production_fixture()
    metadata = replace(
        prepared.metadata,
        compatibility_report={
            **prepared.metadata.compatibility_report,
            "supported_terms": [
                *prepared.metadata.compatibility_report["supported_terms"],
                "nbfix_pair_overrides",
            ],
            "required_terms": [
                *prepared.metadata.compatibility_report["required_terms"],
                "nbfix_pair_overrides",
            ],
        },
    )
    prepared = replace(
        prepared,
        metadata=metadata,
        nbfix_pairs=np.asarray([[0, 1]], dtype=np.int32),
        nbfix_sigma=np.asarray([1.0], dtype=np.float32),
        nbfix_epsilon=np.asarray([0.2], dtype=np.float32),
    )
    save_prepared_system(prepared, tmp_path)
    artifact = load_prepared_mlx_artifact(tmp_path, require_production=True)
    system, terms, _ = build_mlx_system_from_artifact(artifact)
    nonbonded = next(term for term in terms if term.name == "nonbonded")

    components = nonbonded.component_energies(system.positions, system.cell)

    assert nonbonded.has_nbfix
    np.testing.assert_allclose(np.asarray(components["lj"]), 0.0, atol=1e-7)


def test_minimization_lowers_energy_for_harmonic_fixture(tmp_path):
    prepared = _production_fixture()
    displaced = replace(
        prepared,
        positions=np.asarray([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32),
    )
    save_prepared_system(displaced, tmp_path)
    artifact = load_prepared_mlx_artifact(tmp_path, require_production=True)
    system, terms, _ = build_mlx_system_from_artifact(artifact)

    result = minimize_energy(
        system.positions,
        terms,
        max_steps=20,
        step_size=0.01,
        force_tolerance=1e-5,
    )

    final_energy = float(np.asarray(result.energy_history[-1]))
    initial_energy = float(np.asarray(result.energy_history[0]))
    assert final_energy < initial_energy
