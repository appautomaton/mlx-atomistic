from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HELPER_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "select_production_md_fixture.py"
)
_SPEC = importlib.util.spec_from_file_location("select_production_md_fixture", _HELPER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HELPER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HELPER
_SPEC.loader.exec_module(_HELPER)

build_candidate_record = _HELPER.build_candidate_record
write_candidate_record = _HELPER.write_candidate_record


def _write_gpcrmd_cache(cache: Path) -> None:
    cache.mkdir(parents=True)
    for filename in (
        "15286_dyn_729.psf",
        "17686_dyn_729.pdb",
        "15290_prm_729.prm",
        "17687_oth_729.tar.gz",
    ):
        (cache / filename).write_text("fixture\n")


def test_gpcrmd_candidate_record_selects_large_fixture_from_local_cache(tmp_path: Path):
    cache = tmp_path / "cache"
    report = tmp_path / "gpcrmd_import_report.json"
    _write_gpcrmd_cache(cache)
    report.write_text(
        """
{
  "blockers": ["parse_failed:parameters:could not parse CHARMM parameters"],
  "compatibility_report": {
    "unsupported_physics": [
      "virtual_sites_or_hydrogen_mass_repartitioning_not_checked"
    ]
  }
}
""".strip()
    )

    record = build_candidate_record(root=tmp_path, cache_dir=cache, import_report=report)

    assert record["status"] == "selected"
    assert record["selected"] is True
    assert record["fixture"]["id"] == "gpcrmd-729-beta1-5f8u-cyanopindolol"
    assert record["scale"]["atom_count"] == 92001
    assert record["periodic_box"]["status"] == "expected_from_gpcrmd_protocol"
    assert record["protocol_relevance"]["pme_electrostatics_relevance"] == (
        "required_for_periodic_explicit_membrane"
    )
    assert record["protocol_relevance"]["npt_barostat_relevance"] == (
        "not_protocol_required; target ensemble is NVT"
    )
    categories = {item["category"] for item in record["known_pre_execution_blockers"]}
    assert {
        "preparation",
        "constraints_hmr_virtual_sites",
        "electrostatics_pme",
    }.issubset(categories)
    assert record["blockers"] == []
    assert record["downstream"]["parallel_safe_after_this_record"] is True
    assert "prepared_system.npz" in record["artifact_policy"]["do_not_commit"]


def test_missing_cache_records_artifact_source_blocker(tmp_path: Path):
    record = build_candidate_record(root=tmp_path)

    assert record["status"] == "blocked"
    assert record["selected"] is False
    assert record["blockers"][0]["category"] == "artifact_source"
    assert "missing required GPCRmd cache roles" in record["blockers"][0]["observed_result"]
    assert record["downstream"]["parallel_safe_after_this_record"] is False


def test_candidate_record_writes_stable_json(tmp_path: Path):
    cache = tmp_path / "cache"
    out = tmp_path / "evidence" / "candidate-fixture.json"
    _write_gpcrmd_cache(cache)
    record = build_candidate_record(root=tmp_path, cache_dir=cache)

    write_candidate_record(record, out)

    text = out.read_text()
    assert text.endswith("\n")
    assert '"schema_version": 1' in text
    assert "candidate-fixture.json" in text
