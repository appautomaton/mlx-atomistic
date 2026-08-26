from __future__ import annotations

from pathlib import Path

from mlx_atomistic._artifact_identity import sha256_file
from mlx_atomistic.benchmarks import dft_silicon_phonon as phonon


def _gth_source(path: Path) -> Path:
    path.write_text(
        """Si GTH-PBE-q4 GTH-PBE
2 2
0.44000000 1 -6.26928833
2
0.43563383 2 8.95174150 -2.70627082
3.49378060
0.49794218 1 2.43127673
"""
    )
    return path


def test_silicon_phonon_reference_and_dry_run_lock_bounded_work(monkeypatch, tmp_path):
    source = _gth_source(tmp_path / "Si.gth")
    monkeypatch.setattr(phonon, "GTH_RESOURCE_SHA256", sha256_file(source))

    references = phonon.load_silicon_phonon_references()
    plan = phonon.run_silicon_phonon_validation(
        gth_source=source,
        out=tmp_path / "unused",
        dry_run=True,
    )

    assert references["reference"]["optical_frequency_cm1"] == 516.174502
    assert plan["status"] == "planned"
    assert plan["electronic_kpoint_count"] == 64
    assert plan["total_scf_count"] == 8
    assert [item["representative_dofs"] for item in plan["displacements"]] == [
        [0, 3],
        [0, 3],
    ]
    assert plan["asr_imposed"] is False
