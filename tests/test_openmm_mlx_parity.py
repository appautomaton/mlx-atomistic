import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("openmm")
_HELPER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "openmm_mlx_parity.py"
_SPEC = importlib.util.spec_from_file_location("openmm_mlx_parity", _HELPER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HELPER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HELPER
_SPEC.loader.exec_module(_HELPER)

DEFAULT_AMBER_FIXTURE = _HELPER.DEFAULT_AMBER_FIXTURE
PMEParityConfig = _HELPER.PMEParityConfig
ParityTolerances = _HELPER.ParityTolerances
REPORT_NAME = _HELPER.REPORT_NAME
default_amber_fixture_paths = _HELPER.default_amber_fixture_paths
run_amber_openmm_mlx_parity = _HELPER.run_amber_openmm_mlx_parity


def _require_default_fixture() -> tuple[Path, Path]:
    prmtop, coords = default_amber_fixture_paths()
    if not prmtop.exists() or not coords.exists():
        pytest.skip("vendored OpenMM AMBER fixture is not present")
    return prmtop, coords


def test_default_amber_fixture_paths_are_present():
    prmtop, coords = _require_default_fixture()

    assert prmtop.exists()
    assert coords.exists()


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


def test_missing_fixture_returns_exact_blocker(tmp_path: Path):
    report = run_amber_openmm_mlx_parity(
        prmtop_path=tmp_path / "missing.prmtop",
        coords_path=tmp_path / "missing.inpcrd",
        out_dir=tmp_path / "out",
    )

    assert report.status == "blocked"
    assert not report.passed
    assert report.blockers == (f"missing AMBER prmtop: {tmp_path / 'missing.prmtop'}",)
