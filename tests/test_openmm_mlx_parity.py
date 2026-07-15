import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.reference

pytest.importorskip("openmm")
_HELPER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "openmm_mlx_parity.py"
_SPEC = importlib.util.spec_from_file_location("openmm_mlx_parity", _HELPER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HELPER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HELPER
_SPEC.loader.exec_module(_HELPER)
_CLI_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_openmm_mlx_parity.py"
_CLI_SPEC = importlib.util.spec_from_file_location("run_openmm_mlx_parity", _CLI_PATH)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
_CLI = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(_CLI)
_CHARGED_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_charged_pme_parity.py"
_CHARGED_SPEC = importlib.util.spec_from_file_location(
    "run_charged_pme_parity",
    _CHARGED_PATH,
)
assert _CHARGED_SPEC is not None and _CHARGED_SPEC.loader is not None
_CHARGED = importlib.util.module_from_spec(_CHARGED_SPEC)
sys.modules[_CHARGED_SPEC.name] = _CHARGED
_CHARGED_SPEC.loader.exec_module(_CHARGED)

DEFAULT_AMBER_FIXTURE = _HELPER.DEFAULT_AMBER_FIXTURE
DEFAULT_CHARMM_FIXTURE = _HELPER.DEFAULT_CHARMM_FIXTURE
DEFAULT_GROMACS_FIXTURE = _HELPER.DEFAULT_GROMACS_FIXTURE
PMEParityConfig = _HELPER.PMEParityConfig
ParityTolerances = _HELPER.ParityTolerances
REPORT_NAME = _HELPER.REPORT_NAME
default_amber_fixture_paths = _HELPER.default_amber_fixture_paths
default_charmm_fixture_paths = _HELPER.default_charmm_fixture_paths
default_gromacs_fixture_paths = _HELPER.default_gromacs_fixture_paths
evaluate_tip4p_ew_openmm_mlx_parity = _HELPER.evaluate_tip4p_ew_openmm_mlx_parity
run_amber_openmm_mlx_parity = _HELPER.run_amber_openmm_mlx_parity
run_charmm_openmm_mlx_parity = _HELPER.run_charmm_openmm_mlx_parity
run_gromacs_openmm_mlx_parity = _HELPER.run_gromacs_openmm_mlx_parity


def _read_report_json(out_dir: Path) -> dict:
    return json.loads((out_dir / REPORT_NAME).read_text())


def _openmm_temperature_kelvin(state, *, dof: int) -> float:
    unit = pytest.importorskip("openmm.unit")

    kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    return 2.0 * kinetic / (dof * unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
        unit.kilojoule_per_mole / unit.kelvin
    ))


def _require_default_fixture() -> tuple[Path, Path]:
    prmtop, coords = default_amber_fixture_paths()
    assert prmtop.exists(), f"tracked AMBER parity prmtop is missing: {prmtop}"
    assert coords.exists(), f"tracked AMBER parity coordinates are missing: {coords}"
    return prmtop, coords


def _require_charmm_fixture() -> tuple[Path, Path, Path, Path]:
    psf, prm, rtf, coords = default_charmm_fixture_paths()
    assert psf.exists(), f"tracked CHARMM parity psf is missing: {psf}"
    assert prm.exists(), f"tracked CHARMM parity prm is missing: {prm}"
    assert rtf.exists(), f"tracked CHARMM parity rtf is missing: {rtf}"
    assert coords.exists(), f"tracked CHARMM parity coordinates are missing: {coords}"
    return psf, prm, rtf, coords


def _require_gromacs_fixture() -> tuple[Path, Path]:
    top, gro = default_gromacs_fixture_paths()
    assert top.exists(), f"tracked GROMACS parity topology is missing: {top}"
    assert gro.exists(), f"tracked GROMACS parity coordinates are missing: {gro}"
    return top, gro


class _HarmonicWell:
    def __init__(self, target):
        self.target = np.asarray(target, dtype=np.float32)

    def energy_forces(self, positions, cell=None, pairs=None):
        import mlx.core as mx

        target = mx.array(self.target, dtype=positions.dtype)
        displacement = positions - target
        return 0.5 * mx.sum(displacement * displacement), -displacement


class _ZeroVirialForce:
    name = "zero"
    supports_virial = True

    def energy_forces(self, positions, cell=None, pairs=None):
        import mlx.core as mx

        return mx.sum(positions[:, 0] * 0.0), mx.zeros_like(positions)


def test_openmm_comparable_l_bfgs_minimization_fixture():
    openmm = pytest.importorskip("openmm")
    unit = pytest.importorskip("openmm.unit")

    from mlx_atomistic.minimize import minimize_energy

    target = np.asarray([[0.25, -0.1, 0.4]], dtype=np.float32)
    initial = np.asarray([[2.0, -1.0, 0.5]], dtype=np.float32)
    result = minimize_energy(
        initial,
        _HarmonicWell(target),
        method="l-bfgs",
        max_steps=50,
        force_tolerance=1e-5,
    )

    system = openmm.System()
    system.addParticle(1.0)
    force = openmm.CustomExternalForce("0.5*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    force.addParticle(0, target[0].astype(float).tolist())
    system.addForce(force)
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator)
    context.setPositions(initial * unit.nanometer)
    openmm.LocalEnergyMinimizer.minimize(context, tolerance=1e-5, maxIterations=50)
    state = context.getState(getEnergy=True)
    openmm_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    assert float(np.asarray(result.energy)) <= openmm_energy + 1e-5
    np.testing.assert_allclose(np.asarray(result.positions), target, atol=1e-4)


def test_default_amber_fixture_paths_are_present():
    prmtop, coords = _require_default_fixture()

    assert prmtop.exists()
    assert coords.exists()


def test_charmm_and_gromacs_cli_fixture_routing_uses_format_defaults():
    charmm_args = _CLI._parse_args(["--source-kind", "charmm"])
    psf, params, openmm_params, coords, fixture = _CLI._charmm_fixture_paths(charmm_args)
    expected_psf, expected_prm, expected_rtf, expected_coords = default_charmm_fixture_paths()

    assert fixture == DEFAULT_CHARMM_FIXTURE
    assert psf == expected_psf
    assert params == (expected_prm,)
    assert openmm_params == (expected_rtf, expected_prm)
    assert coords == expected_coords

    gromacs_args = _CLI._parse_args(["--source-kind", "gromacs"])
    top, gro, fixture = _CLI._gromacs_fixture_paths(gromacs_args)
    expected_top, expected_gro = default_gromacs_fixture_paths()

    assert fixture == DEFAULT_GROMACS_FIXTURE
    assert top == expected_top
    assert gro == expected_gro


def test_cli_run_report_routes_charmm_and_gromacs(monkeypatch, tmp_path: Path):
    routed: list[tuple[str, dict]] = []
    charmm_sentinel = object()
    gromacs_sentinel = object()

    def fake_charmm(**kwargs):
        routed.append(("charmm", kwargs))
        return charmm_sentinel

    def fake_gromacs(**kwargs):
        routed.append(("gromacs", kwargs))
        return gromacs_sentinel

    monkeypatch.setattr(_CLI, "run_charmm_openmm_mlx_parity", fake_charmm)
    monkeypatch.setattr(_CLI, "run_gromacs_openmm_mlx_parity", fake_gromacs)

    charmm_args = _CLI._parse_args(
        ["--source-kind", "charmm", "--out", str(tmp_path / "charmm")]
    )
    assert _CLI._run_report(charmm_args) is charmm_sentinel
    assert routed[-1][0] == "charmm"
    assert routed[-1][1]["fixture"] == DEFAULT_CHARMM_FIXTURE
    assert routed[-1][1]["params"]
    assert routed[-1][1]["openmm_params"]

    gromacs_args = _CLI._parse_args(
        ["--source-kind", "gromacs", "--out", str(tmp_path / "gromacs")]
    )
    assert _CLI._run_report(gromacs_args) is gromacs_sentinel
    assert routed[-1][0] == "gromacs"
    assert routed[-1][1]["fixture"] == DEFAULT_GROMACS_FIXTURE
    assert routed[-1][1]["top_path"].name.endswith(".top")
    assert routed[-1][1]["gro_path"].name.endswith(".gro")


def test_amber_openmm_mlx_parity_fixture_passes(tmp_path: Path):
    prmtop, coords = _require_default_fixture()

    report = run_amber_openmm_mlx_parity(
        prmtop_path=prmtop,
        coords_path=coords,
        out_dir=tmp_path,
        fixture=DEFAULT_AMBER_FIXTURE,
    )

    assert report.status == "passed"
    assert report.passed
    assert report.atom_count == 22
    assert report.reference_engine == "openmm"
    assert "reference-only" in report.reference_engine_role
    assert report.artifact_readiness is not None
    assert report.artifact_readiness["status"] == "ready"
    assert report.platform_evidence["product_runtime"] == "mlx_atomistic"
    assert report.platform_evidence["reference_engine"] == "openmm"
    assert report.platform_evidence["acceptance_criteria"] == ["AC-03", "AC-07"]
    assert report.platform_evidence["gap_ids"] == ["P2-PARSE-01", "P2-PARITY-01"]
    assert report.platform_evidence["finite_outputs"] is True
    assert report.unsupported_terms == ()
    assert report.blockers == ()
    assert report.total_energy_abs_error_kj_mol is not None
    assert report.total_energy_abs_error_kj_mol <= report.tolerances.total_energy_abs_kj_mol
    assert report.force_max_abs_error_kj_mol_nm is not None
    assert report.force_rms_abs_error_kj_mol_nm is not None
    assert report.force_max_abs_error_kj_mol_nm <= report.tolerances.force_max_abs_kj_mol_nm
    assert report.force_rms_abs_error_kj_mol_nm <= report.tolerances.force_rms_abs_kj_mol_nm
    assert {"bond", "angle", "torsion", "nonbonded"}.issubset(
        report.component_energy_abs_error_kj_mol
    )
    assert (tmp_path / REPORT_NAME).exists()
    assert (tmp_path / "prepared" / "prepared_system.json").exists()


def test_charmm_openmm_mlx_parity_fixture_passes(tmp_path: Path):
    psf, prm, rtf, coords = _require_charmm_fixture()

    report = run_charmm_openmm_mlx_parity(
        psf_path=psf,
        params=[prm],
        openmm_params=[rtf, prm],
        coords_path=coords,
        out_dir=tmp_path,
        fixture=DEFAULT_CHARMM_FIXTURE,
        tolerances=ParityTolerances(
            total_energy_abs_kj_mol=5.0e-3,
            component_energy_abs_kj_mol=5.0e-3,
            force_max_abs_kj_mol_nm=15.0,
            force_rms_abs_kj_mol_nm=4.0,
        ),
    )

    assert report.status == "passed"
    assert report.passed
    assert report.source_kind == "charmm"
    assert report.atom_count == 8
    assert report.reference_engine == "openmm"
    assert report.readiness["ready"] is True
    assert report.artifact_readiness is not None
    assert report.artifact_readiness["status"] == "ready"
    assert report.platform_evidence["acceptance_criteria"] == ["AC-04", "AC-07"]
    assert report.platform_evidence["gap_ids"] == ["P2-PARSE-02", "P2-PARITY-01"]
    assert report.unsupported_terms == ()
    assert report.blockers == ()
    assert report.unmapped_openmm_components == ()
    charmm_components = {
        "bond",
        "angle",
        "torsion",
        "urey_bradley",
        "charmm_cmap",
        "nonbonded",
    }
    assert charmm_components.issubset(report.component_energy_abs_error_kj_mol)
    assert charmm_components.issubset(report.component_energy_openmm_kj_mol)
    for component in charmm_components:
        assert abs(report.component_energy_openmm_kj_mol[component]) > 1.0e-12
    assert report.total_energy_abs_error_kj_mol is not None
    assert report.total_energy_abs_error_kj_mol <= report.tolerances.total_energy_abs_kj_mol
    assert report.force_max_abs_error_kj_mol_nm is not None
    assert report.force_rms_abs_error_kj_mol_nm is not None
    assert report.force_max_abs_error_kj_mol_nm <= report.tolerances.force_max_abs_kj_mol_nm
    assert report.force_rms_abs_error_kj_mol_nm <= report.tolerances.force_rms_abs_kj_mol_nm
    assert (tmp_path / REPORT_NAME).exists()


def test_gromacs_openmm_mlx_parity_fixture_passes_or_blocks_explicitly(tmp_path: Path):
    top, gro = _require_gromacs_fixture()

    report = run_gromacs_openmm_mlx_parity(
        top_path=top,
        gro_path=gro,
        out_dir=tmp_path,
        fixture=DEFAULT_GROMACS_FIXTURE,
        tolerances=ParityTolerances(
            total_energy_abs_kj_mol=5.0e-3,
            component_energy_abs_kj_mol=5.0e-3,
            force_max_abs_kj_mol_nm=15.0,
            force_rms_abs_kj_mol_nm=4.0,
        ),
    )

    assert report.source_kind == "gromacs"
    assert report.reference_engine == "openmm"
    assert report.readiness["unsupported_terms"] == list(report.unsupported_terms)
    assert report.platform_evidence["acceptance_criteria"] == ["AC-05", "AC-07"]
    assert report.platform_evidence["gap_ids"] == ["P2-PARSE-03", "P2-PARITY-01"]
    assert (tmp_path / REPORT_NAME).exists()
    if report.status == "blocked":
        assert report.blockers
        assert report.blockers[0].startswith("OpenMM gromacs reference")
        return

    assert report.status == "passed"
    assert report.passed
    assert report.atom_count == 8
    assert report.readiness["ready"] is True
    assert report.unsupported_terms == ()
    assert report.blockers == ()
    assert report.unmapped_openmm_components == ()
    assert {"bond", "angle", "torsion", "rb_dihedral", "nonbonded"}.issubset(
        report.component_energy_abs_error_kj_mol
    )
    assert report.total_energy_abs_error_kj_mol is not None
    assert report.total_energy_abs_error_kj_mol <= report.tolerances.total_energy_abs_kj_mol
    assert report.force_max_abs_error_kj_mol_nm is not None
    assert report.force_rms_abs_error_kj_mol_nm is not None
    assert report.force_max_abs_error_kj_mol_nm <= report.tolerances.force_max_abs_kj_mol_nm
    assert report.force_rms_abs_error_kj_mol_nm <= report.tolerances.force_rms_abs_kj_mol_nm


def test_openmm_component_mapping_covers_charmm_and_gromacs_force_classes():
    assert _HELPER._openmm_component_name("CMAPTorsionForce") == "charmm_cmap"
    assert _HELPER._openmm_component_name("RBTorsionForce") == "rb_dihedral"
    assert _HELPER._openmm_component_name("CustomTorsionForce") == "torsion"
    assert _HELPER._openmm_component_name("CustomNonbondedForce") == "nonbonded"
    assert (
        _HELPER._openmm_component_name(
            "HarmonicBondForce",
            source_kind="charmm",
            occurrence=1,
            expected_component_counts={"urey_bradley": 1},
        )
        == "urey_bradley"
    )
    assert (
        _HELPER._openmm_component_name(
            "HarmonicBondForce",
            source_kind="amber",
            occurrence=1,
            expected_component_counts={"urey_bradley": 1},
        )
        is None
    )
    assert (
        _HELPER._openmm_component_name(
            "HarmonicBondForce",
            source_kind="charmm",
            occurrence=1,
            expected_component_counts={"urey_bradley": 0},
        )
        is None
    )
    assert (
        _HELPER._openmm_component_name(
            "HarmonicBondForce",
            source_kind="charmm",
            occurrence=2,
            expected_component_counts={"urey_bradley": 1},
        )
        is None
    )


def test_amber_openmm_ambiguous_second_harmonic_bond_force_is_unsupported():
    openmm = pytest.importorskip("openmm")
    unit = pytest.importorskip("openmm.unit")

    system = openmm.System()
    for _ in range(2):
        system.addParticle(1.0)
    for length in (0.10, 0.12):
        force = openmm.HarmonicBondForce()
        force.addBond(
            0,
            1,
            length * unit.nanometer,
            100.0 * unit.kilojoule_per_mole / unit.nanometer**2,
        )
        system.addForce(force)

    result = _HELPER._evaluate_openmm_system(
        openmm_system=system,
        positions=np.asarray([[0.0, 0.0, 0.0], [0.11, 0.0, 0.0]]) * unit.nanometer,
        platform_name="Reference",
        source_kind="amber",
        expected_component_counts={"bond": 2, "urey_bradley": 0},
    )

    assert result["unsupported_terms"] == ("HarmonicBondForce",)
    assert set(result["component_energy_kj_mol"]) == {"bond"}


def test_openmm_constrained_water_geometry_matches_settle_projection():
    openmm = pytest.importorskip("openmm")
    unit = pytest.importorskip("openmm.unit")

    from mlx_atomistic.constraints import SettleWaterConstraints

    oh_nm = 0.1
    hh_nm = 0.15
    initial = np.asarray(
        [[0.0, 0.0, 0.0], [0.112, 0.008, 0.0], [-0.012, 0.089, 0.0]],
        dtype=np.float64,
    )

    mlx_constraints = SettleWaterConstraints([(0, 1, 2)], oh_distance=oh_nm, hh_distance=hh_nm)
    mlx_projected, mlx_error = mlx_constraints.apply_positions(
        initial,
        masses=np.asarray([16.0, 1.0, 1.0], dtype=np.float32),
    )
    mlx_projected = np.asarray(mlx_projected, dtype=np.float64)

    system = openmm.System()
    for mass in (16.0, 1.0, 1.0):
        system.addParticle(mass)
    system.addConstraint(0, 1, oh_nm * unit.nanometer)
    system.addConstraint(0, 2, oh_nm * unit.nanometer)
    system.addConstraint(1, 2, hh_nm * unit.nanometer)
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.001 * unit.picoseconds),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(initial * unit.nanometer)
    context.applyConstraints(1.0e-8)
    openmm_projected = np.asarray(
        context.getState(getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(unit.nanometer),
        dtype=np.float64,
    )

    def distances(positions: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                np.linalg.norm(positions[1] - positions[0]),
                np.linalg.norm(positions[2] - positions[0]),
                np.linalg.norm(positions[2] - positions[1]),
            ],
            dtype=np.float64,
        )

    tolerance_nm = 2.0e-5
    np.testing.assert_allclose(distances(mlx_projected), [oh_nm, oh_nm, hh_nm], atol=1e-6)
    np.testing.assert_allclose(distances(openmm_projected), [oh_nm, oh_nm, hh_nm], atol=1e-8)
    np.testing.assert_allclose(
        distances(mlx_projected),
        distances(openmm_projected),
        atol=tolerance_nm,
    )
    assert float(np.asarray(mlx_error)) <= tolerance_nm


def test_tip4p_ew_openmm_mlx_fixed_pair_energy_parity():
    result = evaluate_tip4p_ew_openmm_mlx_parity()

    assert result["reference_engine"] == "openmm"
    assert "reference-only" in result["reference_engine_role"]
    assert result["artifact_atom_count"] == 8
    assert result["runtime_atom_count"] == 6
    assert result["runtime_virtual_site_count"] == 2
    assert result["total_energy_abs_error_kj_mol"] <= 5.0e-1


def test_openmm_triclinic_periodic_distance_matches_mlx_minimum_image():
    openmm = pytest.importorskip("openmm")
    unit = pytest.importorskip("openmm.unit")

    from mlx_atomistic.core import Cell

    matrix_nm = np.asarray(
        [
            [4.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.5, 0.25, 2.0],
        ],
        dtype=np.float64,
    )
    fractional_displacement = np.asarray([[0.6, -0.55, 0.49]], dtype=np.float64)
    positions = np.vstack(
        [
            np.zeros((1, 3), dtype=np.float64),
            fractional_displacement @ matrix_nm,
        ]
    )
    cell = Cell.triclinic(matrix_nm)
    mlx_distance_nm = float(
        np.linalg.norm(np.asarray(cell.minimum_image(positions[1:] - positions[:1]))[0])
    )

    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    force = openmm.CustomBondForce("r")
    force.setUsesPeriodicBoundaryConditions(True)
    force.addBond(0, 1, [])
    system.addForce(force)
    vectors = tuple(openmm.Vec3(*row) for row in matrix_nm)
    system.setDefaultPeriodicBoxVectors(*vectors)
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.001 * unit.picoseconds),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPeriodicBoxVectors(*vectors)
    context.setPositions(positions * unit.nanometer)
    openmm_distance_nm = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )

    tolerance_nm = 1.0e-6
    np.testing.assert_allclose(mlx_distance_nm, openmm_distance_nm, atol=tolerance_nm)


def test_openmm_nose_hoover_temperature_statistics_are_bounded_like_mlx():
    openmm = pytest.importorskip("openmm")
    unit = pytest.importorskip("openmm.unit")

    from mlx_atomistic.core import Cell
    from mlx_atomistic.md import NoseHooverThermostat, SimulationConfig, simulate_nvt

    target_temperature_kelvin = 100.0
    system = openmm.System()
    for _ in range(4):
        system.addParticle(39.9)
    zero_force = openmm.CustomExternalForce("0")
    for atom in range(4):
        zero_force.addParticle(atom, [])
    system.addForce(zero_force)
    integrator = openmm.NoseHooverIntegrator(
        target_temperature_kelvin * unit.kelvin,
        10.0 / unit.picosecond,
        0.001 * unit.picoseconds,
    )
    context = openmm.Context(
        system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(np.zeros((4, 3)) * unit.nanometer)
    context.setVelocitiesToTemperature(target_temperature_kelvin * unit.kelvin, 7)
    openmm_temperatures = []
    for _ in range(6):
        integrator.step(2)
        openmm_temperatures.append(
            _openmm_temperature_kelvin(context.getState(getEnergy=True), dof=12)
        )

    positions = np.asarray(
        [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [2.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    velocities = np.asarray(
        [[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, -0.5, 0.0]],
        dtype=np.float32,
    )
    result = simulate_nvt(
        positions,
        velocities,
        masses=np.ones((4,), dtype=np.float32),
        cell=Cell.cubic(5.0),
        force_terms=_ZeroVirialForce(),
        config=SimulationConfig(dt=0.001, steps=12, sample_interval=2, diagnostic_interval=2),
        thermostat=NoseHooverThermostat(temperature=1.0, relaxation_time=0.1),
    )

    openmm_ratio = np.asarray(openmm_temperatures, dtype=np.float64) / target_temperature_kelvin
    mlx_ratio = np.asarray(result.temperature, dtype=np.float64) / result.target_temperature
    tolerance_ratio = (0.01, 3.0)
    assert np.isfinite(openmm_ratio).all()
    assert np.isfinite(mlx_ratio).all()
    assert tolerance_ratio[0] <= float(np.mean(openmm_ratio)) <= tolerance_ratio[1]
    assert tolerance_ratio[0] <= float(np.mean(mlx_ratio)) <= tolerance_ratio[1]
    assert result.thermostat_metadata["family"] == "nose_hoover"


def test_openmm_anisotropic_barostat_cell_trend_is_bounded_like_mlx():
    openmm = pytest.importorskip("openmm")
    unit = pytest.importorskip("openmm.unit")

    from mlx_atomistic.core import Cell
    from mlx_atomistic.md import (
        LangevinThermostat,
        MonteCarloBarostat,
        SimulationConfig,
        simulate_npt,
    )

    system = openmm.System()
    for _ in range(2):
        system.addParticle(39.9)
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(4.0, 0.0, 0.0),
        openmm.Vec3(0.0, 4.0, 0.0),
        openmm.Vec3(0.0, 0.0, 4.0),
    )
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(0.5 * unit.nanometer)
    for _ in range(2):
        nonbonded.addParticle(0.0, 1.0, 0.0)
    system.addForce(nonbonded)
    barostat = openmm.MonteCarloAnisotropicBarostat(
        openmm.Vec3(0.0, 0.0, 0.0) * unit.bar,
        100.0 * unit.kelvin,
        True,
        False,
        True,
        1,
    )
    barostat.setRandomNumberSeed(4)
    system.addForce(barostat)
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(
        system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(np.asarray([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) * unit.nanometer)
    context.setVelocitiesToTemperature(100.0 * unit.kelvin, 3)
    initial_openmm_volume = context.getState().getPeriodicBoxVolume().value_in_unit(
        unit.nanometer**3
    )
    integrator.step(5)
    openmm_state = context.getState()
    final_openmm_volume = openmm_state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)
    openmm_lengths = np.asarray(
        [
            vector[i].value_in_unit(unit.nanometer)
            for i, vector in enumerate(openmm_state.getPeriodicBoxVectors())
        ],
        dtype=np.float64,
    )

    cell = Cell.orthorhombic([4.0, 4.0, 4.0])
    result = simulate_npt(
        np.asarray([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        masses=np.ones((2,), dtype=np.float32),
        cell=cell,
        force_terms=_ZeroVirialForce(),
        config=SimulationConfig(dt=0.001, steps=1, sample_interval=1, diagnostic_interval=1),
        thermostat=LangevinThermostat(temperature=0.0, friction=0.0, seed=11),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0,
            seed=4,
            max_log_volume_scale=0.02,
            mode="anisotropic",
            axes=(True, False, True),
        ),
    )

    tolerance = {
        "min_volume_ratio": 0.95,
        "max_volume_ratio": 1.10,
        "fixed_y_atol": 1.0e-6,
    }
    openmm_volume_ratio = final_openmm_volume / initial_openmm_volume
    mlx_volume_ratio = float(np.asarray(result.volume)[-1] / np.asarray(result.volume)[0])
    assert tolerance["min_volume_ratio"] <= openmm_volume_ratio <= tolerance["max_volume_ratio"]
    assert tolerance["min_volume_ratio"] <= mlx_volume_ratio <= tolerance["max_volume_ratio"]
    assert openmm_lengths[1] == pytest.approx(4.0, abs=tolerance["fixed_y_atol"])
    np.testing.assert_allclose(np.asarray(result.final_cell.lengths)[1], 4.0, atol=1.0e-6)
    assert result.barostat_metadata["mode"] == "anisotropic"
    assert result.barostat_attempts == 1


def test_amber_openmm_mlx_pme_parity_fixture_passes(tmp_path: Path):
    prmtop, coords = _require_default_fixture()

    report = run_amber_openmm_mlx_parity(
        prmtop_path=prmtop,
        coords_path=coords,
        out_dir=tmp_path,
        fixture=DEFAULT_AMBER_FIXTURE,
        pme_config=PMEParityConfig(),
        tolerances=ParityTolerances(
            total_energy_abs_kj_mol=5.0e-2,
            component_energy_abs_kj_mol=5.0e-2,
            force_max_abs_kj_mol_nm=12.0,
            force_rms_abs_kj_mol_nm=3.0,
        ),
    )

    assert report.status == "passed"
    assert report.passed
    assert report.openmm_nonbonded_method == "PME"
    assert report.pme_readiness is not None
    assert report.pme_readiness["status"] == "ready"
    assert report.pme_readiness["backend"] == "mlx_fft_cic"
    assert report.pme_readiness["blockers"] == ()
    assert report.platform_evidence["readiness"]["pme"]["status"] == "ready"
    assert report.total_energy_abs_error_kj_mol is not None
    assert report.total_energy_abs_error_kj_mol <= report.tolerances.total_energy_abs_kj_mol
    assert (
        report.component_energy_abs_error_kj_mol["nonbonded"]
        <= report.tolerances.component_energy_abs_kj_mol
    )
    assert report.force_max_abs_error_kj_mol_nm is not None
    assert report.force_rms_abs_error_kj_mol_nm is not None
    assert report.force_max_abs_error_kj_mol_nm <= report.tolerances.force_max_abs_kj_mol_nm
    assert report.force_rms_abs_error_kj_mol_nm <= report.tolerances.force_rms_abs_kj_mol_nm


@pytest.mark.parametrize("assignment_order", [4, 5])
def test_pme_parity_helpers_preserve_assignment_order(tmp_path: Path, assignment_order: int):
    from mlx_atomistic.artifacts import load_prepared_mlx_artifact
    from mlx_atomistic.prep.io import save_prepared_system, synthetic_prepared_system

    config = PMEParityConfig(
        mesh_shape=(8, 8, 8),
        real_cutoff_angstrom=5.0,
        cell_lengths_angstrom=(12.0, 12.0, 12.0),
        assignment_order=assignment_order,
    )
    prepared = _HELPER._with_pme_artifact_settings(synthetic_prepared_system(), config)
    save_prepared_system(prepared, tmp_path)
    artifact = load_prepared_mlx_artifact(tmp_path)
    readiness = _HELPER._pme_readiness(artifact, config)

    assert prepared.metadata.pme_config["assignment_order"] == assignment_order
    assert int(prepared.pme_assignment_order[0]) == assignment_order
    assert artifact.metadata["pme_config"]["assignment_order"] == assignment_order
    assert int(artifact.arrays["pme_assignment_order"][0]) == assignment_order
    assert readiness["assignment_order"] == assignment_order
    assert readiness["runtime_envelope"]["assignment"] == (
        f"cardinal_b_spline_order_{assignment_order}"
    )


def test_amber_parity_fixture_can_opt_into_pme_assignment_order_metadata(tmp_path: Path):
    from mlx_atomistic.artifacts import load_prepared_mlx_artifact
    from mlx_atomistic.prep.io import save_prepared_system
    from mlx_atomistic.prep.topology_import import import_amber_prmtop

    prmtop, coords = _require_default_fixture()
    config = PMEParityConfig(
        mesh_shape=(16, 16, 16),
        real_cutoff_angstrom=5.0,
        cell_lengths_angstrom=(18.0, 19.0, 20.0),
        assignment_order=4,
    )

    prepared = import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)
    prepared = _HELPER._with_pme_artifact_settings(prepared, config)
    save_prepared_system(prepared, tmp_path)
    artifact = load_prepared_mlx_artifact(tmp_path, require_production=True)
    readiness = _HELPER._pme_readiness(artifact, config)

    assert artifact.metadata["source"]["kind"] == "amber"
    assert artifact.metadata["pme_config"]["assignment_order"] == 4
    assert int(artifact.arrays["pme_assignment_order"][0]) == 4
    assert artifact.metadata["compatibility_report"]["periodic_box_present"] is True
    assert readiness["status"] == "ready"
    assert readiness["assignment_order"] == 4


def test_amber_pme_override_clears_stale_cell_matrix_metadata(tmp_path: Path):
    from dataclasses import replace

    from mlx_atomistic.artifacts import load_prepared_mlx_artifact
    from mlx_atomistic.prep.io import save_prepared_system
    from mlx_atomistic.prep.topology_import import import_amber_prmtop

    prmtop, coords = _require_default_fixture()
    stale_matrix = np.asarray(
        [
            [31.0, 0.0, 0.0],
            [2.0, 32.0, 0.0],
            [0.5, 1.0, 33.0],
        ],
        dtype=np.float32,
    )
    config = PMEParityConfig(
        mesh_shape=(16, 16, 16),
        real_cutoff_angstrom=5.0,
        cell_lengths_angstrom=(18.0, 19.0, 20.0),
        assignment_order=4,
    )

    prepared = import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)
    prepared = replace(
        prepared,
        cell_lengths=np.linalg.norm(stale_matrix, axis=1).astype(np.float32),
        cell_matrix=stale_matrix,
    )
    prepared = _HELPER._with_pme_artifact_settings(prepared, config)
    save_prepared_system(prepared, tmp_path)
    artifact = load_prepared_mlx_artifact(tmp_path, require_production=True)

    assert prepared.cell_matrix.size == 0
    assert artifact.arrays["cell_matrix"].size == 0
    np.testing.assert_allclose(artifact.arrays["cell_lengths"], [18.0, 19.0, 20.0])
    assert artifact.metadata["pme_config"]["assignment_order"] == 4


def test_amber_parity_returns_blocked_report_for_unsupported_import(tmp_path: Path):
    prmtop = tmp_path / "unsupported.prmtop"
    coords = tmp_path / "unsupported.inpcrd"
    prmtop.write_text(
        """%VERSION  VERSION_STAMP = V0001.000
%FLAG POINTERS
%FORMAT(10I8)
       2       1
%FLAG ATOM_NAME
%FORMAT(20a4)
H1  H2
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00
%FLAG MASS
%FORMAT(5E16.8)
  1.00800000E+00  1.00800000E+00
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       1
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  4.00000000E+00
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  4.00000000E+00
%FLAG LENNARD_JONES_CCOEF
%FORMAT(5E16.8)
  1.00000000E+00
"""
    )
    coords.write_text(
        """unsupported
    2
  0.0000000  0.0000000  0.0000000  1.0000000  0.0000000  0.0000000
"""
    )

    report = run_amber_openmm_mlx_parity(
        prmtop_path=prmtop,
        coords_path=coords,
        out_dir=tmp_path / "out",
    )

    assert report.status == "blocked"
    assert not report.passed
    assert report.blockers == ("unsupported_terms:amber_12_6_4_lj",)
    assert report.unsupported_terms == ("amber_12_6_4_lj",)
    assert report.platform_evidence["status"] == "blocked"
    assert report.platform_evidence["metrics"]["blocker"] == "unsupported_terms:amber_12_6_4_lj"
    payload = _read_report_json(tmp_path / "out")
    assert payload["status"] == "blocked"
    assert payload["blockers"] == list(report.blockers)


def test_amber_negative_water_oh_pair_with_zero_lj_is_allowed(tmp_path: Path):
    from mlx_atomistic.prep.topology_import import (
        _amber_allowed_negative_lj_pair_policy,
        _amber_lj_type_pair_parameters,
        _read_amber_prmtop,
    )

    prmtop = tmp_path / "water-negative-lj.prmtop"
    prmtop.write_text(
        """%VERSION  VERSION_STAMP = V0001.000
%FLAG POINTERS
%FORMAT(10I8)
       2       2
%FLAG ATOM_NAME
%FORMAT(20a4)
O   H1
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
OW  HW
%FLAG RESIDUE_LABEL
%FORMAT(20a4)
WAT
%FLAG RESIDUE_POINTER
%FORMAT(10I8)
       1
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       2
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1      -1      -1       3
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  5.81935564E+05  0.00000000E+00  0.00000000E+00
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  5.94825035E+02  0.00000000E+00  0.00000000E+00
%FLAG HBOND_ACOEF
%FORMAT(5E16.8)
  0.00000000E+00
%FLAG HBOND_BCOEF
%FORMAT(5E16.8)
  0.00000000E+00
"""
    )

    topology = _read_amber_prmtop(prmtop)

    assert _amber_lj_type_pair_parameters(topology, 1, 2) == (1.0, 0.0)
    assert _amber_lj_type_pair_parameters(topology, 2, 1) == (1.0, 0.0)
    policy = _amber_allowed_negative_lj_pair_policy(topology)
    assert policy["status"] == "allowed_zero_lj_water_pairs"
    assert len(policy["affected_type_pairs"]) == 2


def test_amber_negative_non_water_pair_still_blocks(tmp_path: Path):
    from mlx_atomistic.prep.topology_import import (
        TopologyImportError,
        _amber_lj_type_pair_parameters,
        _read_amber_prmtop,
    )

    prmtop = tmp_path / "non-water-negative-lj.prmtop"
    prmtop.write_text(
        """%VERSION  VERSION_STAMP = V0001.000
%FLAG POINTERS
%FORMAT(10I8)
       2       2
%FLAG ATOM_NAME
%FORMAT(20a4)
O   H1
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
OW  HW
%FLAG RESIDUE_LABEL
%FORMAT(20a4)
ALA
%FLAG RESIDUE_POINTER
%FORMAT(10I8)
       1
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       2
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1      -1      -1       3
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  5.81935564E+05  0.00000000E+00  0.00000000E+00
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  5.94825035E+02  0.00000000E+00  0.00000000E+00
"""
    )

    topology = _read_amber_prmtop(prmtop)

    with pytest.raises(TopologyImportError, match="amber_10_12_nonbonded"):
        _amber_lj_type_pair_parameters(topology, 1, 2)


def test_amber_parity_reports_unsupported_terms_for_malformed_exclusions(tmp_path: Path):
    prmtop = tmp_path / "bad-exclusions.prmtop"
    coords = tmp_path / "bad-exclusions.inpcrd"
    prmtop.write_text(
        """%VERSION  VERSION_STAMP = V0001.000
%FLAG POINTERS
%FORMAT(10I8)
       2       1
%FLAG ATOM_NAME
%FORMAT(20a4)
H1  H2
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00
%FLAG MASS
%FORMAT(5E16.8)
  1.00800000E+00  1.00800000E+00
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       1
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  4.00000000E+00
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  4.00000000E+00
%FLAG NUMBER_EXCLUDED_ATOMS
%FORMAT(10I8)
       1
%FLAG EXCLUDED_ATOMS_LIST
%FORMAT(10I8)
       2
"""
    )
    coords.write_text(
        """bad exclusions
    2
  0.0000000  0.0000000  0.0000000  1.0000000  0.0000000  0.0000000
"""
    )

    report = run_amber_openmm_mlx_parity(
        prmtop_path=prmtop,
        coords_path=coords,
        out_dir=tmp_path / "out",
    )

    assert report.status == "blocked"
    assert report.blockers == ("unsupported_terms:amber_malformed_exclusions",)
    assert report.unsupported_terms == ("amber_malformed_exclusions",)
    assert report.platform_evidence["metrics"]["blocker"] == (
        "unsupported_terms:amber_malformed_exclusions"
    )


def test_amber_parity_reports_unsupported_terms_for_invalid_atom_index(tmp_path: Path):
    prmtop = tmp_path / "bad-atom-index.prmtop"
    coords = tmp_path / "bad-atom-index.inpcrd"
    prmtop.write_text(
        """%VERSION  VERSION_STAMP = V0001.000
%FLAG POINTERS
%FORMAT(10I8)
       2       1
%FLAG ATOM_NAME
%FORMAT(20a4)
H1  H2
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00
%FLAG MASS
%FORMAT(5E16.8)
  1.00800000E+00  1.00800000E+00
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       1
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  4.00000000E+00
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  4.00000000E+00
%FLAG BONDS_INC_HYDROGEN
%FORMAT(10I8)
       0       6       1
%FLAG BOND_FORCE_CONSTANT
%FORMAT(5E16.8)
  1.00000000E+02
%FLAG BOND_EQUIL_VALUE
%FORMAT(5E16.8)
  1.00000000E+00
"""
    )
    coords.write_text(
        """bad atom index
    2
  0.0000000  0.0000000  0.0000000  1.0000000  0.0000000  0.0000000
"""
    )

    report = run_amber_openmm_mlx_parity(
        prmtop_path=prmtop,
        coords_path=coords,
        out_dir=tmp_path / "out",
    )

    assert report.status == "blocked"
    assert report.blockers == ("unsupported_terms:amber_malformed_bond_parameters",)
    assert report.unsupported_terms == ("amber_malformed_bond_parameters",)
    assert report.platform_evidence["metrics"]["blocker"] == (
        "unsupported_terms:amber_malformed_bond_parameters"
    )


def test_amber_parity_reports_unsupported_terms_for_atom_array_mismatch(tmp_path: Path):
    prmtop = tmp_path / "bad-atom-arrays.prmtop"
    coords = tmp_path / "bad-atom-arrays.inpcrd"
    prmtop.write_text(
        """%VERSION  VERSION_STAMP = V0001.000
%FLAG POINTERS
%FORMAT(10I8)
       2       1
%FLAG ATOM_NAME
%FORMAT(20a4)
H1  H2
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00
%FLAG MASS
%FORMAT(5E16.8)
  1.00800000E+00  1.00800000E+00
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       1
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
H
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  4.00000000E+00
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  4.00000000E+00
"""
    )
    coords.write_text(
        """bad atom arrays
    2
  0.0000000  0.0000000  0.0000000  1.0000000  0.0000000  0.0000000
"""
    )

    report = run_amber_openmm_mlx_parity(
        prmtop_path=prmtop,
        coords_path=coords,
        out_dir=tmp_path / "out",
    )

    assert report.status == "blocked"
    assert report.blockers == ("unsupported_terms:amber_malformed_atom_arrays",)
    assert report.unsupported_terms == ("amber_malformed_atom_arrays",)
    assert report.platform_evidence["metrics"]["blocker"] == (
        "unsupported_terms:amber_malformed_atom_arrays"
    )


def test_amber_parity_reports_unsupported_terms_for_malformed_restart_number(
    tmp_path: Path,
):
    prmtop = tmp_path / "bad-restart-number.prmtop"
    coords = tmp_path / "bad-restart-number.inpcrd"
    prmtop.write_text(
        """%VERSION  VERSION_STAMP = V0001.000
%FLAG POINTERS
%FORMAT(10I8)
       2       1
%FLAG ATOM_NAME
%FORMAT(20a4)
H1  H2
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00
%FLAG MASS
%FORMAT(5E16.8)
  1.00800000E+00  1.00800000E+00
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       1
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  4.00000000E+00
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  4.00000000E+00
"""
    )
    coords.write_text(
        """bad restart number
    2
  0.0000000  bad-token  0.0000000  1.0000000  0.0000000  0.0000000
"""
    )

    report = run_amber_openmm_mlx_parity(
        prmtop_path=prmtop,
        coords_path=coords,
        out_dir=tmp_path / "out",
    )

    assert report.status == "blocked"
    assert report.blockers == ("unsupported_terms:amber_malformed_topology",)
    assert report.unsupported_terms == ("amber_malformed_topology",)
    assert report.platform_evidence["metrics"]["blocker"] == (
        "unsupported_terms:amber_malformed_topology"
    )


def test_pme_parity_helper_rejects_non_integer_assignment_order():
    from mlx_atomistic.prep.io import synthetic_prepared_system

    config = PMEParityConfig(assignment_order=2.5)

    with pytest.raises(ValueError, match="assignment_order must be one of 2, 4, or 5"):
        _HELPER._with_pme_artifact_settings(synthetic_prepared_system(), config)


@pytest.mark.gpu
def test_charged_pme_small_fixture_passes_background_and_openmm_gate():
    payload = _CHARGED.evaluate_small_charged_fixture(platform_name="Reference")

    assert payload["passed"]
    assert payload["net_charge_e"] == pytest.approx(1.0)
    assert payload["checks"]["analytic_background_absolute"] is True
    assert payload["checks"]["analytic_background_relative"] is True
    assert payload["checks"]["energy_per_atom"] is True
    assert payload["checks"]["relative_energy"] is True
    assert payload["checks"]["force_rms"] is True
    assert payload["checks"]["force_maximum"] is True
    assert payload["pme"]["assignment_order"] == 5
    assert payload["pme"]["background_policy"] == "uniform_neutralizing_plasma"


def test_charged_pme_openmm_clone_offsets_every_supported_force_class():
    openmm = pytest.importorskip("openmm")
    unit = pytest.importorskip("openmm.unit")
    api = _CHARGED._load_openmm()
    config = _CHARGED._jac_pme_config((1, 1, 1))
    source_atom_count = 4

    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, 0.1, 100.0)
    cloned_bond = _CHARGED._clone_openmm_force(
        api,
        source_force=bond,
        source_atom_count=source_atom_count,
        replica_count=2,
        config=config,
    )
    assert cloned_bond.getNumBonds() == 2
    assert tuple(cloned_bond.getBondParameters(1)[:2]) == (4, 5)

    angle = openmm.HarmonicAngleForce()
    angle.addAngle(0, 1, 2, 1.2, 30.0)
    cloned_angle = _CHARGED._clone_openmm_force(
        api,
        source_force=angle,
        source_atom_count=source_atom_count,
        replica_count=2,
        config=config,
    )
    assert tuple(cloned_angle.getAngleParameters(1)[:3]) == (4, 5, 6)

    torsion = openmm.PeriodicTorsionForce()
    torsion.addTorsion(0, 1, 2, 3, 3, 0.2, 2.0)
    cloned_torsion = _CHARGED._clone_openmm_force(
        api,
        source_force=torsion,
        source_atom_count=source_atom_count,
        replica_count=2,
        config=config,
    )
    assert tuple(cloned_torsion.getTorsionParameters(1)[:4]) == (4, 5, 6, 7)

    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(0.9 * unit.nanometer)
    for charge in (0.5, -0.5, 0.25, -0.25):
        nonbonded.addParticle(charge, 0.2, 0.1)
    nonbonded.addException(0, 3, -0.125, 0.2, 0.05)
    cloned_nonbonded = _CHARGED._clone_openmm_force(
        api,
        source_force=nonbonded,
        source_atom_count=source_atom_count,
        replica_count=2,
        config=config,
    )
    assert cloned_nonbonded.getNumParticles() == 8
    assert cloned_nonbonded.getNumExceptions() == 2
    assert tuple(cloned_nonbonded.getExceptionParameters(1)[:2]) == (4, 7)
    assert cloned_nonbonded.getUseDispersionCorrection() is False
    alpha, nx, ny, nz = cloned_nonbonded.getPMEParameters()
    assert alpha.value_in_unit(unit.nanometer**-1) == pytest.approx(3.5)
    assert (nx, ny, nz) == (64, 64, 64)


def test_charged_pme_openmm_clone_rejects_unknown_force_class():
    openmm = pytest.importorskip("openmm")
    api = _CHARGED._load_openmm()

    with pytest.raises(_CHARGED.UnsupportedOpenMMForceError, match="CustomExternalForce"):
        _CHARGED._clone_openmm_force(
            api,
            source_force=openmm.CustomExternalForce("0"),
            source_atom_count=1,
            replica_count=1,
            config=_CHARGED._jac_pme_config((1, 1, 1)),
        )


def test_charged_pme_missing_inputs_write_blocked_report(tmp_path: Path):
    out = tmp_path / "out"

    payload = _CHARGED.run_charged_pme_parity(
        mlx_prepared=tmp_path / "missing-prepared",
        amber_prmtop=tmp_path / "missing.prmtop",
        amber_coordinates=tmp_path / "missing.inpcrd",
        replicas=(1, 1, 1),
        platform_name="Reference",
        out=out,
    )

    assert payload["status"] == "blocked"
    assert payload["passed"] is False
    assert len(payload["blockers"]) == 4
    assert (out / _CHARGED.REPORT_NAME).exists()


def test_charmm_openmm_reference_failure_writes_blocked_report(tmp_path: Path):
    psf, prm, _rtf, coords = _require_charmm_fixture()
    out_dir = tmp_path / "out"

    report = run_charmm_openmm_mlx_parity(
        psf_path=psf,
        params=[prm],
        openmm_params=[prm],
        coords_path=coords,
        out_dir=out_dir,
        fixture=DEFAULT_CHARMM_FIXTURE,
    )

    assert report.status == "blocked"
    assert report.source_kind == "charmm"
    assert report.blockers
    assert report.blockers[0].startswith("OpenMM charmm reference load/evaluation failed:")
    payload = _read_report_json(out_dir)
    assert payload["status"] == "blocked"
    assert payload["blockers"] == list(report.blockers)
    assert payload["reference_engine"] == "openmm"


def test_missing_fixture_returns_exact_blocker(tmp_path: Path):
    report = run_amber_openmm_mlx_parity(
        prmtop_path=tmp_path / "missing.prmtop",
        coords_path=tmp_path / "missing.inpcrd",
        out_dir=tmp_path / "out",
    )

    assert report.status == "blocked"
    assert not report.passed
    assert report.blockers == (f"missing AMBER prmtop: {tmp_path / 'missing.prmtop'}",)
    assert report.reference_engine == "openmm"
    assert report.platform_evidence["status"] == "blocked"
    assert report.platform_evidence["finite_outputs"] is False
    payload = _read_report_json(tmp_path / "out")
    assert payload["status"] == "blocked"
    assert payload["blockers"] == list(report.blockers)
