from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from mlx_atomistic.benchmarks.charged_pme import prepare_payload
from mlx_atomistic.prep.io import (
    load_prepared_system,
    save_prepared_system,
    synthetic_prepared_system,
)
from mlx_atomistic.prep.schema import ARTIFACT_VERSION, PreparedSystem
from mlx_atomistic.prep.supercell import (
    PreparedSupercellError,
    prepared_supercell_summary,
    replicate_prepared_system,
)


def _complete_prepared_fixture() -> PreparedSystem:
    base = synthetic_prepared_system()
    metadata = replace(
        base.metadata,
        artifact_version=ARTIFACT_VERSION,
        source={"kind": "fixture", "path": "source"},
        selections={
            "atom_count": 4,
            "hydrogen_count": 2,
            "water_atom_count": 2,
            "system_charge": 0.0,
        },
        compatibility_report={
            "production_force_field": True,
            "supported_terms": [
                "harmonic_bond",
                "harmonic_angle",
                "periodic_dihedral",
                "periodic_improper",
                "distance_constraint",
                "nonbonded_exception",
                "charmm_cmap",
                "urey_bradley",
                "nbfix_pair_overrides",
                "virtual_site",
                "pme",
            ],
            "required_terms": ["pme"],
            "unsupported_terms": [],
            "hydrogen_count": 2,
            "array_term_counts": {
                "harmonic_bond": 1,
                "harmonic_angle": 1,
                "periodic_dihedral": 1,
                "periodic_improper": 1,
                "distance_constraint": 1,
                "nonbonded_exception": 1,
                "pme": 1,
            },
            "term_counts": {"bonds": 1, "virtual_site": 1},
            "term_counts_normalized": {"harmonic_bond": 1, "virtual_site": 1},
        },
        pme_config={
            "mesh_shape": [8, 10, 12],
            "alpha": 0.35,
            "real_cutoff": 5.0,
            "assignment_order": 4,
            "charge_tolerance": 1e-5,
            "deconvolve_assignment": True,
            "background_policy": "uniform_neutralizing_plasma",
        },
        protocol_metadata={"nonbonded": {"cutoff": 5.0}},
    )
    positions = np.asarray(
        [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5], [1.0, 1.0, 0.5]],
        dtype=np.float32,
    )
    return replace(
        base,
        metadata=metadata,
        symbols=np.asarray(["H", "O", "H", "X"], dtype=str),
        atom_names=np.asarray(["H1", "O1", "H2", "VS"], dtype=str),
        atom_types=np.asarray(["A", "B", "C", "V"], dtype=str),
        residue_names=np.asarray(["WAT", "WAT", "LIG", "LIG"], dtype=str),
        residue_ids=np.asarray([1, 1, 2, 2], dtype=np.int32),
        chain_ids=np.asarray(["A", "A", "B", "B"], dtype=str),
        positions=positions,
        velocities=np.arange(12, dtype=np.float32).reshape(4, 3) * 0.01,
        masses=np.asarray([1.0, 16.0, 1.0, 0.1], dtype=np.float32),
        charges=np.asarray([0.4, -0.6, 0.3, 0.1], dtype=np.float32),
        sigma=np.asarray([1.0, 1.1, 1.2, 1.3], dtype=np.float32),
        epsilon=np.asarray([0.1, 0.2, 0.3, 0.0], dtype=np.float32),
        bonds=np.asarray([[0, 1]], dtype=np.int32),
        bond_k=np.asarray([100.0], dtype=np.float32),
        bond_length=np.asarray([1.0], dtype=np.float32),
        angles=np.asarray([[0, 1, 2]], dtype=np.int32),
        angle_k=np.asarray([20.0], dtype=np.float32),
        angle_theta=np.asarray([1.5], dtype=np.float32),
        dihedrals=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        dihedral_k=np.asarray([2.0], dtype=np.float32),
        dihedral_periodicity=np.asarray([3.0], dtype=np.float32),
        dihedral_phase=np.asarray([0.2], dtype=np.float32),
        nonbonded_pairs=np.asarray([[0, 2]], dtype=np.int32),
        ligand_mask=np.asarray([False, False, True, True]),
        receptor_mask=np.asarray([True, True, False, False]),
        restraint_mask=np.asarray([True, False, False, False]),
        reference_positions=positions.copy(),
        cell_lengths=np.asarray([10.0, 12.0, 14.0], dtype=np.float32),
        cell_matrix=np.diag([10.0, 12.0, 14.0]).astype(np.float32),
        rb_dihedrals=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        rb_c0=np.asarray([0.1], dtype=np.float32),
        rb_c1=np.asarray([0.2], dtype=np.float32),
        rb_c2=np.asarray([0.3], dtype=np.float32),
        rb_c3=np.asarray([0.4], dtype=np.float32),
        rb_c4=np.asarray([0.5], dtype=np.float32),
        rb_c5=np.asarray([0.6], dtype=np.float32),
        constraints=np.asarray([[0, 1]], dtype=np.int32),
        constraint_distance=np.asarray([1.0], dtype=np.float32),
        impropers=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        improper_k=np.asarray([1.0], dtype=np.float32),
        improper_periodicity=np.asarray([2.0], dtype=np.float32),
        improper_phase=np.asarray([0.0], dtype=np.float32),
        nonbonded_exception_pairs=np.asarray([[0, 3]], dtype=np.int32),
        nonbonded_exception_charge_product=np.asarray([0.04], dtype=np.float32),
        nonbonded_exception_sigma=np.asarray([1.1], dtype=np.float32),
        nonbonded_exception_epsilon=np.asarray([0.05], dtype=np.float32),
        water_mask=np.asarray([True, True, False, False]),
        ion_mask=np.asarray([False, False, True, False]),
        lipid_mask=np.asarray([False, False, False, True]),
        gbsa_radius=np.asarray([1.0, 1.1, 1.2, 1.3], dtype=np.float32),
        gbsa_scale=np.asarray([0.8, 0.8, 0.7, 0.7], dtype=np.float32),
        pme_mesh_shape=np.asarray([8, 10, 12], dtype=np.int32),
        pme_alpha=np.asarray([0.35], dtype=np.float32),
        pme_real_cutoff=np.asarray([5.0], dtype=np.float32),
        pme_assignment_order=np.asarray([4], dtype=np.int32),
        pme_charge_tolerance=np.asarray([1e-5], dtype=np.float32),
        pme_deconvolve_assignment=np.asarray([True], dtype=bool),
        pme_background_policy=np.asarray(["uniform_neutralizing_plasma"], dtype=str),
        charmm_cmap_terms=np.asarray([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=np.int32),
        charmm_cmap_grid_indices=np.asarray([0], dtype=np.int32),
        charmm_cmap_grids=np.arange(16, dtype=np.float32).reshape(1, 4, 4),
        urey_bradley_terms=np.asarray([[0, 1, 2]], dtype=np.int32),
        urey_bradley_k=np.asarray([3.0], dtype=np.float32),
        urey_bradley_distance=np.asarray([1.4], dtype=np.float32),
        nbfix_pairs=np.asarray([[1, 2]], dtype=np.int32),
        nbfix_sigma=np.asarray([1.5], dtype=np.float32),
        nbfix_epsilon=np.asarray([0.4], dtype=np.float32),
        nbfix_type_pairs=np.asarray([["A", "B"]], dtype=str),
        nbfix_type_sigma=np.asarray([1.6], dtype=np.float32),
        nbfix_type_epsilon=np.asarray([0.5], dtype=np.float32),
        virtual_site_parent_atoms=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        virtual_site_weights=np.asarray([[0.2, 0.3, 0.5, 0.0]], dtype=np.float32),
        virtual_site_types=np.asarray(["three_particle_average"], dtype=str),
    )


def test_replicate_prepared_system_offsets_every_supported_index_and_parameter():
    source = _complete_prepared_fixture()
    replicated = replicate_prepared_system(
        source,
        (2, 2, 1),
        assignment_order=5,
        background_policy="uniform_neutralizing_plasma",
    )

    assert replicated.atom_count == 16
    np.testing.assert_allclose(replicated.cell_lengths, [20.0, 24.0, 14.0])
    np.testing.assert_allclose(replicated.positions[:4], source.positions)
    np.testing.assert_allclose(
        replicated.positions[4:8],
        source.positions + np.asarray([0.0, 12.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        replicated.positions[8:12],
        source.positions + np.asarray([10.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(replicated.bonds, [[0, 1], [4, 5], [8, 9], [12, 13]])
    np.testing.assert_array_equal(
        replicated.virtual_site_parent_atoms,
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
    )
    for index_name in (
        "bonds",
        "angles",
        "dihedrals",
        "nonbonded_pairs",
        "rb_dihedrals",
        "constraints",
        "impropers",
        "nonbonded_exception_pairs",
        "charmm_cmap_terms",
        "urey_bradley_terms",
        "nbfix_pairs",
        "virtual_site_parent_atoms",
    ):
        assert np.asarray(getattr(replicated, index_name)).shape[0] == 4
    np.testing.assert_array_equal(replicated.bond_k, np.repeat(source.bond_k, 4))
    np.testing.assert_array_equal(replicated.charmm_cmap_grids, source.charmm_cmap_grids)
    np.testing.assert_array_equal(replicated.nbfix_type_pairs, source.nbfix_type_pairs)
    np.testing.assert_array_equal(replicated.pme_mesh_shape, [16, 20, 12])
    np.testing.assert_array_equal(replicated.pme_assignment_order, [5])
    assert replicated.pme_background_policy.tolist() == ["uniform_neutralizing_plasma"]
    assert len(set(replicated.chain_ids.tolist())) == 8
    assert replicated.metadata.artifact_version == ARTIFACT_VERSION
    assert replicated.metadata.selections["atom_count"] == 16
    assert replicated.metadata.compatibility_report["array_term_counts"]["harmonic_bond"] == 4
    assert replicated.metadata.compatibility_report["array_term_counts"]["pme"] == 1
    assert replicated.metadata.protocol_metadata["nonbonded"]["cutoff"] == 5.0


def test_replicate_prepared_system_is_deterministic_and_does_not_mutate_source():
    source = _complete_prepared_fixture()
    source_arrays = {
        field.name: np.asarray(getattr(source, field.name)).copy()
        for field in fields(source)
        if field.name != "metadata"
    }
    source_metadata = source.metadata.to_json_dict()

    first = replicate_prepared_system(source, (2, 1, 1))
    second = replicate_prepared_system(source, (2, 1, 1))

    for field in fields(source):
        if field.name == "metadata":
            continue
        np.testing.assert_array_equal(getattr(first, field.name), getattr(second, field.name))
        np.testing.assert_array_equal(getattr(source, field.name), source_arrays[field.name])
    assert first.metadata.to_json_dict() == second.metadata.to_json_dict()
    assert source.metadata.to_json_dict() == source_metadata


def test_replicated_prepared_system_round_trips_with_global_arrays_retained(tmp_path):
    replicated = replicate_prepared_system(_complete_prepared_fixture(), (2, 1, 1))
    save_prepared_system(replicated, tmp_path)

    loaded = load_prepared_system(tmp_path)
    summary = prepared_supercell_summary(
        loaded,
        source_atom_count=4,
        replicas=(2, 1, 1),
    )

    assert loaded.atom_count == 8
    np.testing.assert_array_equal(loaded.charmm_cmap_grids, replicated.charmm_cmap_grids)
    np.testing.assert_array_equal(loaded.nbfix_type_pairs, replicated.nbfix_type_pairs)
    assert summary["atom_count"] == 8
    assert summary["indexed_term_counts"]["bonds"] == 2


def test_replicate_prepared_system_rejects_triclinic_source():
    source = _complete_prepared_fixture()
    matrix = np.asarray([[10.0, 0.0, 0.0], [1.0, 12.0, 0.0], [0.0, 0.0, 14.0]])
    triclinic = replace(
        source,
        cell_matrix=matrix.astype(np.float32),
        cell_lengths=np.linalg.norm(matrix, axis=1).astype(np.float32),
    )
    triclinic.validate()

    with pytest.raises(PreparedSupercellError, match="orthorhombic"):
        replicate_prepared_system(triclinic, (2, 1, 1))


def test_charged_pme_prepare_blocks_missing_source_without_partial_output(tmp_path):
    out = tmp_path / "out"
    payload = prepare_payload(
        source=tmp_path / "missing",
        replicas=(2, 2, 1),
        assignment_order=5,
        background_policy="uniform_neutralizing_plasma",
        out=out,
    )

    assert payload["status"] == "blocked"
    assert payload["written"] is False
    assert payload["blockers"]
    assert not out.exists()


def test_charged_pme_prepare_writes_only_below_output_and_validates_counts(tmp_path):
    source_dir = tmp_path / "source"
    out = tmp_path / "result" / "prepared"
    save_prepared_system(_complete_prepared_fixture(), source_dir)

    payload = prepare_payload(
        source=source_dir,
        replicas=(2, 1, 1),
        assignment_order=5,
        background_policy="uniform_neutralizing_plasma",
        out=out,
    )

    assert payload["status"] == "ok"
    assert payload["summary"]["passed"] is True
    assert payload["summary"]["atom_count"] == 8
    assert Path(payload["summary_path"]).is_relative_to(out)
    assert load_prepared_system(out).atom_count == 8
