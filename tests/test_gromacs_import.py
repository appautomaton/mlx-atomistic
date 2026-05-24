from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_import_gromacs_top_gro_fixture_maps_supported_terms(tmp_path: Path):
    from mlx_atomistic.artifacts import load_prepared_mlx_artifact
    from mlx_atomistic.prep import import_gromacs_top_gro
    from mlx_atomistic.prep.io import load_prepared_system, save_prepared_system

    fixture_root = Path("tests/fixtures/gromacs")
    prepared = import_gromacs_top_gro(
        top_path=fixture_root / "native-mini.top",
        gro_path=fixture_root / "native-mini.gro",
    )
    report = prepared.metadata.compatibility_report

    assert prepared.metadata.source["parser"] == "native_gromacs_top_gro"
    assert prepared.metadata.parameter_source == "gromacs_top_gro_native"
    assert prepared.atom_count == 8
    np.testing.assert_allclose(prepared.positions[1], [1.09, 0.0, 0.0])
    np.testing.assert_allclose(prepared.cell_lengths, [25.0, 26.0, 27.0])
    np.testing.assert_allclose(prepared.sigma[:4], [3.4, 2.5, 3.0, 2.5])
    np.testing.assert_allclose(prepared.bond_length[:3], [1.09, 1.43, 0.96])
    np.testing.assert_allclose(prepared.bond_k[0], 2845.12, rtol=1e-6)
    assert prepared.bonds.shape == (6, 2)
    assert prepared.angles.shape == (4, 3)
    assert prepared.dihedrals.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert prepared.rb_dihedrals.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    np.testing.assert_allclose(prepared.rb_c0, [0.1, 0.1])
    np.testing.assert_allclose(prepared.rb_c1, [-0.2, -0.2])
    assert [0, 3] in prepared.nonbonded_exception_pairs.tolist()
    pair_index = prepared.nonbonded_exception_pairs.tolist().index([0, 3])
    np.testing.assert_allclose(
        prepared.nonbonded_exception_charge_product[pair_index],
        0.10 * 0.05 * 0.8333333333,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        prepared.nonbonded_exception_sigma[pair_index],
        0.5 * (3.4 + 2.5),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        prepared.nonbonded_exception_epsilon[pair_index],
        (0.276144 * 0.125520) ** 0.5 * 0.5,
        rtol=1e-6,
    )
    assert "rb_dihedral" in report["supported_terms"]
    assert "nonbonded_exception" in report["supported_terms"]
    assert report["term_counts"]["rb_dihedrals"] == 2
    assert report["term_counts"]["gromacs_molecule_instances"] == 2
    assert report["unsupported_terms"] == []

    save_prepared_system(prepared, tmp_path)
    reloaded = load_prepared_system(tmp_path)
    assert reloaded.metadata.source["parser"] == "native_gromacs_top_gro"
    np.testing.assert_array_equal(reloaded.rb_dihedrals, prepared.rb_dihedrals)

    artifact = load_prepared_mlx_artifact(tmp_path, require_production=True)
    assert "rb_dihedral" in artifact.metadata["compatibility_report"]["required_terms"]
    np.testing.assert_array_equal(artifact.arrays["rb_dihedrals"], prepared.rb_dihedrals)


def test_import_gromacs_top_gro_blocks_preprocessor_directive(tmp_path: Path):
    from mlx_atomistic.prep import TopologyImportError, import_gromacs_top_gro

    fixture_root = Path("tests/fixtures/gromacs")
    top = tmp_path / "preprocessed.top"
    top.write_text('#include "forcefield.itp"\n' + (fixture_root / "native-mini.top").read_text())

    with pytest.raises(
        TopologyImportError,
        match="unsupported_terms:gromacs_preprocessor_directive:include",
    ):
        import_gromacs_top_gro(top_path=top, gro_path=fixture_root / "native-mini.gro")


def test_import_gromacs_top_gro_blocks_unsupported_directive(tmp_path: Path):
    from mlx_atomistic.prep import TopologyImportError, import_gromacs_top_gro

    fixture_root = Path("tests/fixtures/gromacs")
    top = tmp_path / "virtual-site.top"
    top.write_text(
        (fixture_root / "native-mini.top").read_text()
        + "\n[ virtual_sites2 ]\n1 2 3 1 0.5\n"
    )

    with pytest.raises(
        TopologyImportError,
        match="unsupported_terms:gromacs_directive_virtual_sites2",
    ):
        import_gromacs_top_gro(top_path=top, gro_path=fixture_root / "native-mini.gro")


def test_import_gromacs_top_gro_blocks_unsupported_combination_rule(tmp_path: Path):
    from mlx_atomistic.prep import TopologyImportError, import_gromacs_top_gro

    fixture_root = Path("tests/fixtures/gromacs")
    top = tmp_path / "combination-rule-1.top"
    top.write_text(
        (fixture_root / "native-mini.top").read_text().replace(
            "1 2 yes 0.5 0.8333333333",
            "1 1 yes 0.5 0.8333333333",
        )
    )

    with pytest.raises(
        TopologyImportError,
        match="unsupported_terms:gromacs_combination_rule_1",
    ):
        import_gromacs_top_gro(top_path=top, gro_path=fixture_root / "native-mini.gro")


def test_import_gromacs_top_gro_blocks_pairs_when_gen_pairs_is_no(tmp_path: Path):
    from mlx_atomistic.prep import TopologyImportError, import_gromacs_top_gro

    fixture_root = Path("tests/fixtures/gromacs")
    top = tmp_path / "pairs-without-generated-parameters.top"
    top.write_text(
        (fixture_root / "native-mini.top").read_text().replace(
            "1 2 yes 0.5 0.8333333333",
            "1 2 no 0.5 0.8333333333",
        )
    )

    with pytest.raises(
        TopologyImportError,
        match="unsupported_terms:gromacs_pairs_without_generated_parameters",
    ):
        import_gromacs_top_gro(top_path=top, gro_path=fixture_root / "native-mini.gro")


def test_import_gromacs_top_gro_blocks_generated_pairs_without_pair_records(
    tmp_path: Path,
):
    from mlx_atomistic.prep import TopologyImportError, import_gromacs_top_gro

    fixture_root = Path("tests/fixtures/gromacs")
    text = (fixture_root / "native-mini.top").read_text()
    pairless = text.replace("\n[ pairs ]\n1 4 1\n", "\n")
    top = tmp_path / "generated-pairs.top"
    top.write_text(pairless)

    with pytest.raises(TopologyImportError, match="unsupported_terms:gromacs_generated_pairs"):
        import_gromacs_top_gro(top_path=top, gro_path=fixture_root / "native-mini.gro")
