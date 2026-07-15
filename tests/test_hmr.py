from dataclasses import replace

import numpy as np
import pytest

from mlx_atomistic.artifacts import (
    MLXCompatibilityError,
    build_mlx_system_from_artifact,
    load_prepared_mlx_artifact,
)
from mlx_atomistic.prep import apply_hydrogen_mass_repartitioning
from mlx_atomistic.prep.io import save_prepared_system, synthetic_prepared_system


def _hmr_fixture():
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
            "physical_units": True,
            "hydrogens_present": True,
            "hydrogen_count": 2,
            "supported_terms": ["harmonic_bond", "distance_constraint"],
            "required_terms": ["harmonic_bond", "distance_constraint"],
            "unsupported_terms": [],
        },
        parameter_source="hmr_fixture",
    )
    return replace(
        prepared,
        metadata=metadata,
        symbols=np.asarray(["C", "H", "H"], dtype=str),
        atom_names=np.asarray(["C1", "H1", "H2"], dtype=str),
        atom_types=np.asarray(["C", "H", "H"], dtype=str),
        residue_names=np.asarray(["LIG", "LIG", "LIG"], dtype=str),
        residue_ids=np.asarray([1, 1, 1], dtype=np.int32),
        chain_ids=np.asarray(["A", "A", "A"], dtype=str),
        positions=np.asarray(
            [[0.0, 0.0, 0.0], [1.09, 0.0, 0.0], [-1.09, 0.0, 0.0]],
            dtype=np.float32,
        ),
        velocities=np.zeros((3, 3), dtype=np.float32),
        masses=np.asarray([12.011, 1.008, 1.008], dtype=np.float32),
        charges=np.asarray([-0.2, 0.1, 0.1], dtype=np.float32),
        sigma=np.asarray([3.4, 2.5, 2.5], dtype=np.float32),
        epsilon=np.asarray([0.1, 0.0, 0.0], dtype=np.float32),
        bonds=np.asarray([[0, 2], [1, 0]], dtype=np.int32),
        bond_k=np.asarray([300.0, 300.0], dtype=np.float32),
        bond_length=np.asarray([1.09, 1.09], dtype=np.float32),
        constraints=np.asarray([[0, 1], [0, 2]], dtype=np.int32),
        constraint_distance=np.asarray([1.09, 1.09], dtype=np.float32),
        ligand_mask=np.asarray([True, True, True]),
        receptor_mask=np.asarray([False, False, False]),
        restraint_mask=np.asarray([False, False, False]),
        reference_positions=np.asarray(
            [[0.0, 0.0, 0.0], [1.09, 0.0, 0.0], [-1.09, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )


def test_hmr_repartitions_selected_hydrogen_mass_deterministically():
    prepared = _hmr_fixture()

    hmr = apply_hydrogen_mass_repartitioning(prepared, hydrogen_indices=[2, 1])

    np.testing.assert_allclose(hmr.masses, [7.979, 3.024, 3.024], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.sum(hmr.masses), np.sum(prepared.masses), atol=1e-6)
    provenance = hmr.metadata.protocol_metadata["hydrogen_mass_repartitioning"]
    assert provenance["status"] == "represented_by_masses"
    assert provenance["policy"] == {
        "kind": "hydrogen_mass_repartitioning",
        "selection": "all_bonded_hydrogens",
        "target_hydrogen_mass": 3.024,
        "min_heavy_atom_mass": 0.0,
        "heavy_mass_policy": "subtract_hydrogen_mass_delta_from_bonded_heavy_atom",
        "require_constraints": True,
        "virtual_sites_supported": False,
        "hydrogen_indices": [1, 2],
    }
    assert [record["hydrogen_index"] for record in provenance["selected_hydrogens"]] == [1, 2]
    assert provenance["heavy_atoms"] == [
        {
            "heavy_atom_index": 0,
            "original_mass": pytest.approx(12.01099967956543),
            "transformed_mass": pytest.approx(7.97899967956543),
            "mass_delta": pytest.approx(-4.032),
        }
    ]


def test_hmr_promotes_large_float32_mass_arrays_to_preserve_total_mass():
    base = _hmr_fixture()
    pair_count = 10
    atom_count = pair_count * 2
    bonds = np.asarray(
        [[2 * pair, 2 * pair + 1] for pair in range(pair_count)],
        dtype=np.int32,
    )
    positions = np.asarray(
        [[float(index), 0.0, 0.0] for index in range(atom_count)],
        dtype=np.float32,
    )
    prepared = replace(
        base,
        metadata=replace(
            base.metadata,
            compatibility_report={
                **base.metadata.compatibility_report,
                "hydrogen_count": pair_count,
            },
        ),
        symbols=np.asarray(["C", "H"] * pair_count, dtype=str),
        atom_names=np.asarray([f"A{index}" for index in range(atom_count)], dtype=str),
        atom_types=np.asarray(["C", "H"] * pair_count, dtype=str),
        residue_names=np.asarray(["LIG"] * atom_count, dtype=str),
        residue_ids=np.arange(atom_count, dtype=np.int32),
        chain_ids=np.asarray(["A"] * atom_count, dtype=str),
        positions=positions,
        velocities=np.zeros((atom_count, 3), dtype=np.float32),
        masses=np.tile(np.asarray([12.011, 1.008], dtype=np.float32), pair_count),
        charges=np.zeros(atom_count, dtype=np.float32),
        sigma=np.ones(atom_count, dtype=np.float32),
        epsilon=np.zeros(atom_count, dtype=np.float32),
        bonds=bonds,
        bond_k=np.full(pair_count, 300.0, dtype=np.float32),
        bond_length=np.full(pair_count, 1.09, dtype=np.float32),
        constraints=bonds,
        constraint_distance=np.full(pair_count, 1.09, dtype=np.float32),
        ligand_mask=np.ones(atom_count, dtype=bool),
        receptor_mask=np.zeros(atom_count, dtype=bool),
        restraint_mask=np.zeros(atom_count, dtype=bool),
        reference_positions=positions.copy(),
    )

    hmr = apply_hydrogen_mass_repartitioning(prepared, target_hydrogen_mass=4.032)
    provenance = hmr.metadata.protocol_metadata["hydrogen_mass_repartitioning"]

    assert hmr.masses.dtype == np.float64
    assert provenance["input_mass_dtype"] == "float32"
    assert provenance["stored_mass_dtype"] == "float64"
    assert float(np.sum(hmr.masses, dtype=np.float64)) == pytest.approx(
        float(np.sum(prepared.masses, dtype=np.float64)),
        abs=1e-10,
    )


def test_hmr_explicit_subset_records_explicit_selection_policy():
    prepared = _hmr_fixture()

    hmr = apply_hydrogen_mass_repartitioning(prepared, hydrogen_indices=[2])

    provenance = hmr.metadata.protocol_metadata["hydrogen_mass_repartitioning"]
    assert provenance["policy"]["selection"] == "explicit_hydrogen_indices"
    assert provenance["policy"]["hydrogen_indices"] == [2]
    assert [record["hydrogen_index"] for record in provenance["selected_hydrogens"]] == [2]
    np.testing.assert_allclose(hmr.masses, [9.995, 1.008, 3.024], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.sum(hmr.masses), np.sum(prepared.masses), atol=1e-6)


def test_hmr_rejects_non_floating_mass_dtype():
    prepared = replace(_hmr_fixture(), masses=np.asarray([12, 1, 1], dtype=np.int32))

    with pytest.raises(TypeError, match="floating dtype"):
        apply_hydrogen_mass_repartitioning(prepared)


def test_hmr_rejects_low_precision_dtype_that_loses_total_mass():
    prepared = replace(
        _hmr_fixture(),
        masses=np.asarray([12.011, 1.008, 1.008], dtype=np.float16),
    )

    with pytest.raises(ValueError, match="dtype conversion"):
        apply_hydrogen_mass_repartitioning(prepared, target_hydrogen_mass=5.9396463370412)


def test_hmr_artifact_round_trip_reports_provenance_without_virtual_sites(tmp_path):
    hmr = apply_hydrogen_mass_repartitioning(_hmr_fixture())
    save_prepared_system(hmr, tmp_path)

    artifact = load_prepared_mlx_artifact(tmp_path, require_production=True)
    system, _, constraints = build_mlx_system_from_artifact(artifact)

    assert constraints is not None
    np.testing.assert_allclose(np.asarray(system.masses), hmr.masses)
    assert artifact.hmr_state["provenance_available"] is True
    assert artifact.hmr_state["policy"]["virtual_sites_supported"] is False
    assert artifact.hmr_state["original_masses"] == pytest.approx(
        np.asarray(_hmr_fixture().masses, dtype=float).tolist()
    )
    assert artifact.metadata["compatibility_report"]["virtual_sites_present"] is False


def test_hmr_fails_when_required_constraint_is_missing():
    prepared = replace(
        _hmr_fixture(),
        constraints=np.asarray([[0, 1]], dtype=np.int32),
        constraint_distance=np.asarray([1.09], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="missing constraint"):
        apply_hydrogen_mass_repartitioning(prepared)


def test_hmr_rejects_virtual_site_artifacts():
    prepared = _hmr_fixture()
    metadata = replace(
        prepared.metadata,
        compatibility_report={
            **prepared.metadata.compatibility_report,
            "virtual_sites_present": True,
        },
    )

    with pytest.raises(ValueError, match="virtual-site"):
        apply_hydrogen_mass_repartitioning(replace(prepared, metadata=metadata))


def test_hmr_provenance_must_match_artifact_masses(tmp_path):
    hmr = apply_hydrogen_mass_repartitioning(_hmr_fixture())
    metadata = replace(
        hmr.metadata,
        protocol_metadata={
            "hydrogen_mass_repartitioning": {
                **hmr.metadata.protocol_metadata["hydrogen_mass_repartitioning"],
                "transformed_masses": [7.0, 3.024, 3.024],
            }
        },
    )
    save_prepared_system(replace(hmr, metadata=metadata), tmp_path)

    with pytest.raises(MLXCompatibilityError, match="transformed masses"):
        load_prepared_mlx_artifact(tmp_path, require_production=True)
