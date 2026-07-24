import numpy as np

import mlx_atomistic.md as md
from mlx_atomistic.core import Cell
from mlx_atomistic.md import LennardJonesPotential, SimulationConfig, simulate_nve
from mlx_atomistic.runtime import RuntimeInfo, get_platform_boundary_report


def test_platform_boundary_report_names_local_engine_concepts():
    report = get_platform_boundary_report(
        runtime_info=RuntimeInfo(
            mlx_version="0.0",
            default_device="Device(gpu, 0)",
            metal_available=True,
        )
    )
    payload = report.to_dict()
    sections = {section["name"]: section for section in payload["sections"]}

    assert payload["product_runtime"] == "mlx_atomistic"
    assert payload["runtime"]["metal_available"] is True
    assert "openmm" in payload["reference_engine_policy"]
    assert "vendors" in payload["reference_engine_policy"]
    assert "runtime_backend" in sections
    assert "system_artifact" in sections
    assert "protocol" in sections
    assert "readiness" in sections
    assert "validation" in sections
    assert "dft_qm_scope" in sections
    assert sections["runtime_backend"]["status"] == "supported"
    assert sections["readiness"]["status"] == "proof-level"
    assert "PreparedMLXArtifact" in sections["system_artifact"]["local_concepts"]
    assert "pme_readiness_report" in sections["readiness"]["local_concepts"]
    assert "ReferenceDFTCase" in sections["dft_qm_scope"]["local_concepts"]


def test_non_output_failure_checks_do_not_host_materialize_positions_or_velocities(monkeypatch):
    positions = np.array([[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    host_materialized_shapes = []
    original_asarray = np.asarray

    def recording_asarray(value, *args, **kwargs):
        if isinstance(value, md.mx.array) and value.ndim == 2 and value.shape[1] == 3:
            host_materialized_shapes.append(tuple(value.shape))
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(md.np, "asarray", recording_asarray)

    result = simulate_nve(
        positions,
        velocities,
        cell=Cell.cubic(6.0),
        force_terms=LennardJonesPotential(cutoff=2.5),
        config=SimulationConfig(
            dt=0.001,
            steps=2,
            sample_interval=10,
            diagnostic_interval=10,
            evaluation_interval=1,
            pressure_diagnostics=False,
        ),
    )

    assert host_materialized_shapes == []
    assert result.runtime_sync_report["runtime_sync_failure_check_count"] == 1
    assert result.runtime_sync_report["runtime_sync_diagnostic_count"] == 1
    assert result.runtime_sync_report["runtime_sync_final_state_count"] == 1
    assert result.runtime_sync_report["runtime_materialization_checkpoint_count"] == 0


def test_md_performance_output_carries_runtime_sync_reason_counts():
    from mlx_atomistic.benchmarks import md_performance

    payload = md_performance.build_payload(
        sizes=(16,),
        steps=2,
        dt=0.002,
        mode="auto",
        dense_threshold=1536,
        sample_interval=1,
        diagnostic_interval=1,
        evaluation_interval=1,
        neighbor_check_interval=1,
    )

    row = payload["cases"][0]
    assert set(row["runtime_sync_reason_counts"]) == set(md.RUNTIME_SYNC_REASONS)
    assert row["runtime_sync_reason_counts"]["explicit_user_output"] == 2
    assert row["runtime_sync_reason_counts"]["final_state"] == 1
    assert row["runtime_sync_reason_counts"]["diagnostic"] == 2
    assert row["runtime_sync_reason_counts"]["failure_check"] == 0
    assert row["runtime_materialization_reason_counts"]["checkpoint"] == 0
