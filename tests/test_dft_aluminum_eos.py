from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import mlx_atomistic.benchmarks.dft_aluminum as aluminum_workload
from mlx_atomistic.benchmarks.dft_aluminum import (
    ALUMINUM_FRACTIONAL_POSITIONS,
    kpoint_mesh_from_workload,
    load_aluminum_workload,
    prepare_aluminum_workload,
)
from mlx_atomistic.benchmarks.dft_aluminum_eos import (
    birch_murnaghan_energy,
    compare_fit_to_reference,
    fit_birch_murnaghan,
    fit_cubic_aluminum_eos,
    load_aluminum_eos_references,
    reference_fit,
    validation_lattice_constants,
)
from mlx_atomistic.benchmarks.dft_aluminum_eos_runner import (
    _profile_settings,
    _selected_runtime_summary,
    run_aluminum_eos_point,
    run_aluminum_eos_validation,
)
from mlx_atomistic.benchmarks.dft_eos import HARTREE_TO_EV


def _gth_source(path):
    path.write_text(
        """Al GTH-PBE-q3 GTH-PBE
2    1
0.45000000    1    -7.55476126
2
0.48743529    2     6.95993832    -1.88883584
2.43847659
0.56218949    1     1.86529857
"""
    )
    return path


def test_aluminum_reference_bundle_is_pinned_and_cp2k_context_is_excellent():
    references = load_aluminum_eos_references()
    lattice = validation_lattice_constants(references)
    primary = reference_fit(references["references"]["all_electron_average"])
    cp2k = references["references"]["cp2k_gth"]
    rows = cp2k["eos_volume_energy_ev"]
    fit = fit_birch_murnaghan(
        [volume for volume, _energy in rows],
        [energy for _volume, energy in rows],
    )

    assert lattice[3] == pytest.approx(4.040422065345)
    assert primary["equilibrium_lattice_constant_angstrom"] == pytest.approx(4.040861093109186)
    assert fit["equilibrium_volume_angstrom3_per_atom"] == pytest.approx(
        cp2k["fit"]["equilibrium_volume_angstrom3"], rel=2.0e-8
    )
    comparison = compare_fit_to_reference(
        {**reference_fit(cp2k), "status": "ok"},
        primary,
    )
    assert comparison["excellent"] is True
    assert comparison["metrics"]["delta_mev_per_atom"] == pytest.approx(0.9954359681)


def test_aluminum_workload_is_hash_guarded_and_persists_reduced_mesh(tmp_path, monkeypatch):
    monkeypatch.setattr(aluminum_workload, "KPOINT_MESH_SIZES", (3,))
    source = _gth_source(tmp_path / "GTH_POTENTIALS")
    prepared = prepare_aluminum_workload(gth_source=source, out=tmp_path / "workload")
    manifest, resource = load_aluminum_workload(prepared["manifest"])
    mesh = kpoint_mesh_from_workload(manifest, 3)

    assert manifest["system"]["fractional_positions"] == [
        list(row) for row in ALUMINUM_FRACTIONAL_POSITIONS
    ]
    assert manifest["physics"]["smearing_width_hartree"] == 0.00225
    assert manifest["reduced_kpoint_meshes"]["3"]["full_point_count"] == 27
    assert len(mesh.points) == 4
    assert sum(point.weight for point in mesh.points) == pytest.approx(1.0)

    resource.write_text(resource.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_aluminum_workload(prepared["manifest"])


def test_cubic_aluminum_fit_uses_four_atom_cell_free_energies():
    references = load_aluminum_eos_references()
    primary = reference_fit(references["references"]["all_electron_average"])
    lattice = np.asarray(validation_lattice_constants(references))
    energies = birch_murnaghan_energy(
        lattice**3 / 4.0,
        -50.0,
        primary["equilibrium_volume_angstrom3_per_atom"],
        primary["bulk_modulus_ev_angstrom3"],
        primary["bulk_derivative"],
    )

    fit = fit_cubic_aluminum_eos(lattice, energies * 4.0 / HARTREE_TO_EV)

    assert fit["status"] == "ok"
    assert fit["atom_count"] == 4
    assert fit["equilibrium_lattice_constant_angstrom"] == pytest.approx(
        primary["equilibrium_lattice_constant_angstrom"], rel=1.0e-8
    )


def test_aluminum_validation_dry_run_exposes_locked_decision_ladder(tmp_path, monkeypatch):
    monkeypatch.setattr(aluminum_workload, "KPOINT_MESH_SIZES", (3,))
    prepared = prepare_aluminum_workload(
        gth_source=_gth_source(tmp_path / "GTH_POTENTIALS"),
        out=tmp_path / "workload",
    )

    plan = run_aluminum_eos_validation(
        manifest_path=prepared["manifest"],
        out=tmp_path / "validation",
        dry_run=True,
    )

    assert plan["status"] == "planned"
    assert plan["kpoint_mesh_sizes"] == [3]
    assert plan["cutoff_candidates_hartree"] == [15, 20, 25, 30]
    assert plan["band_capacity_candidates"] == [8, 9, 10, 11, 12, 16, 20, 26]
    assert plan["maximum_point_count"] == 57


def test_single_aluminum_point_persists_metallic_free_energy_evidence(tmp_path, monkeypatch):
    import mlx.core as mx

    import mlx_atomistic.benchmarks.dft_runtime_contract as runtime_contract
    import mlx_atomistic.dft as dft

    monkeypatch.setattr(aluminum_workload, "KPOINT_MESH_SIZES", (3,))
    prepared = prepare_aluminum_workload(
        gth_source=_gth_source(tmp_path / "GTH_POTENTIALS"),
        out=tmp_path / "workload",
    )
    manifest, _resource = load_aluminum_workload(prepared["manifest"])
    assert _profile_settings(manifest, "c15-k3-b12")["kpoint_mode"] == "reduced"

    eigen = SimpleNamespace(
        orthonormality_error=2.0e-7,
        residuals=np.full(12, 8.0e-7),
    )
    occupations = (2.0, 2.0, 2.0, 2.0, 1.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0)
    result = SimpleNamespace(
        converged=True,
        total_energy=-50.00225,
        internal_energy=-50.0,
        electronic_entropy=1.0,
        chemical_potential=-0.15,
        electron_count=12.0,
        owned_kpoints=[SimpleNamespace(eigen=eigen)],
        kpoints=[SimpleNamespace(eigen=eigen, occupations=occupations)],
        iterations=11,
        density_residual=2.0e-7,
        energy_delta=3.0e-7,
        timings={"total": 1.0},
        density=np.ones((4, 4, 4), dtype=np.float32),
    )
    monkeypatch.setattr(dft, "PeriodicDFTSystem", lambda *args, **kwargs: object())
    monkeypatch.setattr(dft, "read_gth", lambda *args, **kwargs: object())
    monkeypatch.setattr(dft, "run_periodic_scf", lambda *args, **kwargs: result)
    monkeypatch.setattr(
        runtime_contract,
        "collect_host_provenance",
        lambda: {
            "power_source": "Battery Power",
            "low_power_mode": 1,
            "thermal_pressure": "Nominal",
        },
    )
    monkeypatch.setattr(mx, "synchronize", lambda: None)

    payload = run_aluminum_eos_point(
        manifest_path=prepared["manifest"],
        profile="c15-k3-b12",
        volume_index=3,
        out=tmp_path / "point.json",
    )

    assert payload["numerical_passed"] is True
    assert payload["result"]["fractional_occupation_count"] == 3
    assert payload["result"]["highest_band_occupation"] == 0.0
    assert payload["result"]["free_energy_identity_error_hartree"] < 1.0e-12
    assert payload["host"]["low_power_mode"] == 1
    assert (tmp_path / "density.npy").is_file()


def test_runtime_summary_separates_process_memory_from_cumulative_traffic(tmp_path):
    point_dir = tmp_path / "points" / "selected" / "v0"
    point_dir.mkdir(parents=True)
    (point_dir / "memory.json").write_text('{"bounded_process_peak_physical_bytes": 3000000000}')
    rows = [
        {
            "point": {"profile": "selected", "volume_index": 0},
            "host": {
                "power_source": "AC Power",
                "low_power_mode": 0,
                "thermal_pressure": None,
            },
            "result": {
                "elapsed_wall_seconds": 7.0,
                "observation": {
                    "memory": {
                        "peak_temporary_bytes": 80000000,
                        "persistent_coefficient_bytes": 20000000,
                        "persistent_projector_bytes": 15000000,
                        "shared_full_grid_bytes": 5000000,
                        "projector_streamed_bytes": 19000000000,
                    }
                },
            },
        }
    ]

    summary = _selected_runtime_summary(rows, "selected", tmp_path)

    assert summary["maximum_process_physical_bytes"] == 3000000000
    assert summary["maximum_runtime_peak_temporary_bytes"] == 80000000
    assert summary["maximum_runtime_persistent_payload_bytes"] == 40000000
    assert "projector_streamed_bytes" not in summary
