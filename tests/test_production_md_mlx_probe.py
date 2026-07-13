from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_HELPER_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_mlx_production_md_probe.py"
)
_SPEC = importlib.util.spec_from_file_location("run_mlx_production_md_probe", _HELPER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HELPER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HELPER
_SPEC.loader.exec_module(_HELPER)

build_mlx_probe_record = _HELPER.build_mlx_probe_record
write_mlx_probe_record = _HELPER.write_mlx_probe_record


@dataclass(frozen=True)
class _BlockedAttempt:
    def to_json_dict(self):
        return {
            "target_id": "gpcrmd-729-beta1-5f8u-cyanopindolol",
            "dynamics_id": 729,
            "out_dir": "/tmp/prepared",
            "exported": False,
            "prepared_artifact_path": None,
            "blockers": [
                "parse_failed:parameters:could not parse CHARMM topology/parameters "
                "with ParmEd: Could not find atom type for CT3"
            ],
            "required_artifact_fields": ["coordinates", "topology"],
            "compatibility_report": {
                "target_id": "gpcrmd-729-beta1-5f8u-cyanopindolol",
                "dynamics_id": 729,
                "runnable_now": False,
                "missing_input": [],
                "unsupported_physics": [
                    "virtual_sites_or_hydrogen_mass_repartitioning_not_checked"
                ],
                "runtime_risk": {
                    "system_size": "large",
                    "total_atoms": 92001,
                    "dense_pair_count": 4232046000,
                },
                "next_engine_slice": "parse_gpcrmd_constraints_hmr_or_virtual_sites_policy",
            },
        }


@dataclass(frozen=True)
class _ExportedAttempt:
    prepared_artifact_path: Path

    def to_json_dict(self):
        return {
            "target_id": "gpcrmd-729-beta1-5f8u-cyanopindolol",
            "dynamics_id": 729,
            "out_dir": str(self.prepared_artifact_path.parent),
            "exported": True,
            "prepared_artifact_path": str(self.prepared_artifact_path),
            "blockers": [],
            "required_artifact_fields": ["coordinates", "topology"],
            "compatibility_report": {
                "target_id": "gpcrmd-729-beta1-5f8u-cyanopindolol",
                "dynamics_id": 729,
                "runnable_now": True,
            },
        }


def _candidate_json(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir()
    candidate = tmp_path / "candidate-fixture.json"
    candidate.write_text(
        """
{
  "schema_version": 1,
  "selected": true,
  "fixture": {
    "id": "gpcrmd-729-beta1-5f8u-cyanopindolol",
    "dynamics_id": 729,
    "source_path": "cache"
  },
  "protocol_relevance": {
    "ensemble": "NVT",
    "time_step_fs": 4.0,
    "npt_barostat_relevance": "not_protocol_required; target ensemble is NVT"
  }
}
""".strip()
    )
    return candidate


def test_mlx_probe_records_earliest_preparation_blocker(tmp_path: Path):
    candidate = _candidate_json(tmp_path)
    out = tmp_path / "mlx-probe.json"

    record = build_mlx_probe_record(
        candidate_path=candidate,
        out_path=out,
        root=tmp_path,
        prep_importer=lambda cache, out_dir: _BlockedAttempt(),
    )

    assert record["status"] == "blocked"
    assert record["stages"]["prep"]["status"] == "blocked"
    assert record["stages"]["load"]["status"] == "pending"
    assert record["stages"]["readiness"]["status"] == "pending"
    assert record["stages"]["run"]["status"] == "pending"
    assert record["earliest_blocker"]["category"] == "preparation"
    assert "CT3" in record["earliest_blocker"]["observed_result"]
    assert record["taxonomy_blockers"][0]["prevents_bounded_pass"] is True
    assert record["finite_checks"]["positions"] is None
    assert record["runtime_performance"]["bounded_run_attempted"] is False
    assert record["platform_readiness"]["runtime"]["mlx_version"]
    assert record["dependency_boundary"]["vendor_runtime_imports"] is False


def test_mlx_probe_records_artifact_source_blocker_for_unselected_candidate(tmp_path: Path):
    candidate = tmp_path / "candidate-fixture.json"
    candidate.write_text(
        """
{
  "schema_version": 1,
  "selected": false,
  "fixture": {"id": "gpcrmd-729-beta1-5f8u-cyanopindolol"}
}
""".strip()
    )

    record = build_mlx_probe_record(
        candidate_path=candidate,
        out_path=tmp_path / "mlx-probe.json",
        root=tmp_path,
        prep_importer=lambda cache, out_dir: _BlockedAttempt(),
    )

    assert record["status"] == "blocked"
    assert record["earliest_blocker"]["category"] == "artifact_source"
    assert record["stages"]["prep"]["status"] == "blocked"


def test_mlx_probe_record_writes_stable_json(tmp_path: Path):
    candidate = _candidate_json(tmp_path)
    out = tmp_path / "mlx-probe.json"
    record = build_mlx_probe_record(
        candidate_path=candidate,
        out_path=out,
        root=tmp_path,
        prep_importer=lambda cache, out_dir: _BlockedAttempt(),
    )

    write_mlx_probe_record(record, out)

    text = out.read_text()
    assert text.endswith("\n")
    assert '"schema_version": 1' in text
    assert "mlx-probe.json" in text


def test_mlx_probe_written_evidence_redacts_temporary_prepared_paths(
    tmp_path: Path,
    monkeypatch,
):
    candidate = _candidate_json(tmp_path)
    out = tmp_path / "mlx-probe.json"

    def fake_importer(cache, out_dir):
        return _ExportedAttempt(Path(out_dir))

    def fake_loader(path, require_production):
        raise FileNotFoundError(f"missing prepared artifact at {path}")

    monkeypatch.setattr(_HELPER, "load_prepared_mlx_artifact", fake_loader)

    record = build_mlx_probe_record(
        candidate_path=candidate,
        out_path=out,
        root=tmp_path,
        prep_importer=fake_importer,
    )
    write_mlx_probe_record(record, out)

    text = out.read_text()
    assert "<mlx-production-md-probe>/prepared" in text
    assert not re.search(
        r"(?:/tmp|/var/folders)/(?:[^\s\"';,)]+/)*"
        r"mlx-production-md-probe-[^\s\"';,)]+",
        text,
    )
    persisted = json.loads(text)
    assert (
        persisted["stages"]["prep"]["prepared_artifact_path"]
        == "<mlx-production-md-probe>/prepared"
    )
    assert (
        persisted["earliest_blocker"]["smallest_reproduction_context"]
        == "prepared_artifact_path=<mlx-production-md-probe>/prepared"
    )


def test_mlx_probe_bounded_run_uses_production_neighbor_manager(monkeypatch):
    from mlx_atomistic.core import Cell
    from mlx_atomistic.forcefields import NonbondedPotential
    from mlx_atomistic.neighbors import NeighborListManager
    from mlx_atomistic.topology import Topology

    positions = np.asarray(
        [
            [0.1, 0.1, 0.1],
            [1.1, 0.1, 0.1],
            [0.1, 1.1, 0.1],
            [1.1, 1.1, 0.1],
        ],
        dtype=np.float32,
    )
    cell = Cell.cubic(5.0)
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1)],
        eager_nonbonded_pair_limit=0,
    )
    term = NonbondedPotential(
        sigma=np.ones((4,), dtype=np.float32),
        epsilon=np.full((4,), 0.01, dtype=np.float32),
        charges=np.zeros((4,), dtype=np.float32),
        topology=topology,
        cutoff=1.6,
        backend="auto",
    )
    system = SimpleNamespace(cell=cell)
    artifact = SimpleNamespace(
        arrays={
            "positions": positions,
            "velocities": np.zeros_like(positions),
            "masses": np.ones((4,), dtype=np.float32),
        },
        cell=cell,
        unit_system=None,
    )
    manager = NeighborListManager(
        cell,
        cutoff=1.6,
        skin=0.2,
        max_mlx_dense_atoms=3,
    )
    policy_calls = []

    monkeypatch.setattr(
        _HELPER,
        "build_mlx_system_from_artifact",
        lambda value: (system, [term], None),
    )

    def production_policy(value, terms, *, require_production):
        policy_calls.append((value, tuple(terms), require_production))
        return manager

    monkeypatch.setattr(_HELPER, "_production_neighbor_manager", production_policy)
    report = {
        "status": "running",
        "stages": {"run": _HELPER._stage("pending")},
        "finite_checks": _HELPER._empty_finite_checks(),
        "runtime_performance": {
            "bounded_run_attempted": False,
            "bounded_run_completed": False,
            "wall_time_seconds": None,
        },
        "taxonomy_blockers": [],
        "earliest_blocker": None,
    }

    _HELPER._run_bounded_probe(
        report,
        artifact,
        time.perf_counter(),
        prepared_artifact_evidence_path="<synthetic>/prepared",
    )

    assert policy_calls == [(system, (term,), True)]
    assert report["status"] == "passed"
    assert report["stages"]["run"]["status"] == "passed"
    assert report["runtime_performance"]["bounded_run_completed"] is True
    assert report["runtime_performance"]["backend"] == "mlx_cell_pairs"
    assert report["runtime_performance"]["fallback_reason"] is None
    assert report["runtime_performance"]["candidate_count"] is not None
    assert report["runtime_performance"]["force_evaluation_wall_seconds"] >= 0.0
    assert report["finite_checks"]["energies"] is True
    assert topology._nonbonded_pairs is None
