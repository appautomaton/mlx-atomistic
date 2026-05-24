import json

import numpy as np
import pytest

from mlx_atomistic.dft import (
    BandPath,
    DavidsonDiagonalizer,
    DenseHamiltonianReference,
    DFTSystem,
    FermiDiracOccupations,
    FixedOccupations,
    Ion,
    IonCollection,
    KohnShamOperator,
    KPoint,
    KPointMesh,
    MonkhorstPackGrid,
    NonlocalPseudopotentialOperator,
    ReferenceDFTCase,
    SCFConfig,
    compare_reference_case,
    dft_qm_scope_readiness_report,
    finite_difference_stress,
    get_dft_qm_scope_report,
    load_dense_scf_restart,
    magnetization_density,
    read_upf,
    run_band_structure,
    run_scf,
    save_dense_scf_restart,
    spin_density_from_orbitals,
)


def test_nonlocal_projector_operator_is_hermitian_and_matches_dense_reference():
    upf = read_upf("vendors/quantum-espresso/pseudo/Si_r.upf")
    system = DFTSystem(
        cell=(8.0, 8.0, 8.0),
        grid_shape=(4, 4, 4),
        ions=IonCollection([Ion("Si", (4.0, 4.0, 4.0), upf)]),
    )
    result = run_scf(
        system,
        config=SCFConfig(max_iterations=1, solver="dense", seed=3),
    )
    nonlocal_operator = NonlocalPseudopotentialOperator.from_ions(system.ions, system.grid)
    operator = KohnShamOperator.from_density(
        system.grid,
        system.pseudopotential.field(system.grid),
        result.density,
        nonlocal_operator=nonlocal_operator,
    )
    reference = DenseHamiltonianReference(operator)
    trial = result.orbitals[0]

    dense = np.array(reference.matvec(trial))
    applied = np.array(operator.apply_hamiltonian(trial))

    assert nonlocal_operator.projectors.count > 0
    np.testing.assert_allclose(dense, applied, atol=1e-5)
    assert np.isfinite(float(nonlocal_operator.energy(result.orbitals, occupations=[2.0, 2.0])))


def test_davidson_agrees_with_dense_on_tiny_grid():
    system = DFTSystem.one_center(grid_shape=(4, 4, 4))
    result = run_scf(system, config=SCFConfig(max_iterations=1, solver="dense", seed=5))
    operator = KohnShamOperator.from_density(
        system.grid,
        system.pseudopotential.field(system.grid),
        result.density,
    )

    dense = DenseHamiltonianReference(operator).diagonalize(1)
    davidson = DavidsonDiagonalizer().solve(operator, n_orbitals=1)

    np.testing.assert_allclose(
        np.array(dense.eigenvalues),
        np.array(davidson.eigenvalues),
        atol=1e-6,
    )
    assert davidson.metadata["solver"] == "davidson-dense-reference"


def test_spin_occupations_and_magnetization_are_json_safe():
    fixed = FixedOccupations([1.0, 0.5], spin_mode="polarized").resolve()
    fermi = FermiDiracOccupations(2.0, temperature=0.05).resolve([-0.5, -0.1, 0.2])
    system = DFTSystem.one_center(grid_shape=(4, 4, 4))
    result = run_scf(system, config=SCFConfig(max_iterations=1, solver="dense", seed=11))

    up, down = spin_density_from_orbitals(
        result.orbitals,
        result.orbitals,
        system.grid,
        up_occupations=[1.0],
        down_occupations=[1.0],
    )
    magnetization = magnetization_density(up, down)

    assert fixed.electron_count == pytest.approx(1.5)
    assert fermi.electron_count == pytest.approx(2.0, abs=1e-8)
    assert float(np.sum(np.array(magnetization)) * system.grid.dv) == pytest.approx(0.0, abs=1e-6)
    json.dumps(fermi.to_dict())


def test_kpoint_mesh_and_band_structure_reuse_density():
    system = DFTSystem.one_center(grid_shape=(4, 4, 4))
    result = run_scf(system, config=SCFConfig(max_iterations=1, solver="dense", seed=5))
    gamma = KPointMesh.gamma()
    mesh = MonkhorstPackGrid((1, 1, 2))
    path = BandPath([KPoint.gamma(), KPoint((0.25, 0.0, 0.0), label="X")])

    bands = run_band_structure(system, result, path, n_bands=1)

    assert len(gamma.points) == 1
    assert len(mesh.points) == 2
    assert bands.eigenvalues.shape == (2, 1)
    assert bands.reused_density
    json.dumps(bands.to_dict())


def test_stress_restart_and_reference_comparison(tmp_path):
    system = DFTSystem.one_center(grid_shape=(4, 4, 4))
    config = SCFConfig(max_iterations=1, solver="dense", seed=13)
    result = run_scf(system, config=config)
    stress = finite_difference_stress(system, config=config, displacement=1e-3)
    restart_path = tmp_path / "restart.npz"

    save_dense_scf_restart(
        restart_path,
        result,
        positions=np.array(system.centers),
        cell_lengths=np.array(system.cell.lengths),
        metadata={"case": "one-center"},
    )
    restart = load_dense_scf_restart(restart_path)
    reference = ReferenceDFTCase(
        name="toy",
        source="static-fixture",
        expected_energy=result.total_energy,
        energy_tolerance=1e-8,
    )
    comparison = compare_reference_case(reference, observed_energy=result.total_energy)

    assert stress.stress.shape == (3,)
    assert restart.density.shape == system.grid.shape
    assert restart.metadata["user"]["case"] == "one-center"
    assert comparison.passed
    json.dumps(stress.to_dict())
    json.dumps(restart.to_dict())
    json.dumps(comparison.to_dict())


def test_dft_qm_scope_report_classifies_cp2k_qe_boundaries():
    report = get_dft_qm_scope_report()
    payload = report.to_dict()
    entries = {entry["feature"]: entry for entry in payload["entries"]}

    assert payload["product_runtime"] == "mlx_atomistic"
    assert "cp2k" in payload["reference_policy"]
    assert "quantum_espresso" in payload["reference_policy"]
    assert entries["plane_wave_scf"]["status"] == "proof-level"
    assert entries["static_reference_comparison"]["status"] == "supported"
    assert entries["qmmm_orchestration"]["status"] == "deferred"
    assert entries["external_runtime_execution"]["status"] == "anti-goal"
    assert "CP2K Quickstep" in entries["plane_wave_scf"]["reference_families"]
    assert "Quantum ESPRESSO PWscf" in entries["plane_wave_scf"]["reference_families"]
    json.dumps(payload)

    ready = dft_qm_scope_readiness_report("pwscf").to_dict()
    assert ready["name"] == "dft_qm_scope"
    assert ready["status"] == "proof-level"
    assert ready["blockers"] == []

    deferred = dft_qm_scope_readiness_report("qmmm").to_dict()
    assert deferred["status"] == "deferred"
    assert deferred["blockers"] == ["qmmm_orchestration:deferred"]

    unknown = dft_qm_scope_readiness_report("cp2k-runtime-wrapper").to_dict()
    assert unknown["status"] == "fail-closed"
    assert unknown["blockers"] == ["unknown_dft_qm_feature:cp2k_runtime_wrapper"]
