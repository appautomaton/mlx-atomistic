from __future__ import annotations

import tarfile
from copy import deepcopy

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic._artifact_identity import canonical_json_bytes
from mlx_atomistic.benchmarks.dft_hpsi_profile import (
    LOCAL_FFT_PROFILE_STAGES,
    PROFILE_STAGES,
    _archive_capture,
    _build_profile_case,
    _finalize_report,
    _measure_case,
    _parse_vector_counts,
    _profile_contract,
    _ProfileContext,
    _stage_action,
    _validate_protocol,
)
from mlx_atomistic.dft import (
    GTHProjectorChannel,
    PeriodicGTHNonlocalOperator,
    PeriodicKohnShamOperator,
    PlaneWaveBasis,
    PseudopotentialData,
    PseudopotentialFormat,
    RealSpaceGrid,
)


def _tiny_profile_context() -> _ProfileContext:
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis.from_reduced_kpoint(
        grid,
        4.0,
        (0.25, 0.0, 0.0),
        lane_label="test:dft-hpsi-profile",
    )
    pseudo = PseudopotentialData(
        element="H",
        format=PseudopotentialFormat.GTH,
        valence_charge=1.0,
        gth_rloc=0.25,
        gth_coefficients=(-1.0,),
        gth_channels=(GTHProjectorChannel(0, 0.3, ((0.5,),)),),
    )
    nonlocal_operator = PeriodicGTHNonlocalOperator(
        pseudo,
        basis,
        ((1.0, 2.0, 3.0),),
    )
    operator = PeriodicKohnShamOperator(
        basis,
        mx.full(grid.shape, 0.2),
        nonlocal_operator,
    )
    return _ProfileContext(
        basis=basis,
        nonlocal_operator=nonlocal_operator,
        operator=operator,
        grid_shape=grid.shape,
    )


def test_vector_count_parser_and_capture_contract_fail_closed():
    assert _parse_vector_counts("1,2,8") == (1, 2, 8)
    with pytest.raises(Exception, match="unique positive integers"):
        _parse_vector_counts("1,1")
    with pytest.raises(RuntimeError, match="Metal capture requires"):
        _validate_protocol(
            vector_counts=(1, 2),
            warmups=0,
            samples=1,
            capture_stage="hpsi",
            capture_vector_count=2,
            capture_repetitions=1,
            selected_device="Device(cpu, 0)",
            metal_available=True,
        )


def test_profile_contract_and_report_identity_are_deterministic():
    contract = _profile_contract(
        workload_fingerprint="a" * 64,
        vector_counts=(1, 4),
        warmups=1,
        samples=3,
        capture_stage=None,
        capture_vector_count=None,
        capture_repetitions=3,
        selected_device="Device(cpu, 0)",
        mlx_version="test",
        profiler_sha256="b" * 64,
    )
    assert contract["stage_order"] == list(PROFILE_STAGES)
    first = _finalize_report({"schema_version": "test", "contract": contract})
    second = _finalize_report(deepcopy({"schema_version": "test", "contract": contract}))
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


def test_capture_bundle_is_dereferenced_into_one_regular_archive(tmp_path):
    capture = tmp_path / "metal.gputrace"
    capture.mkdir()
    payload = capture / "payload.bin"
    payload.write_bytes(b"metal-capture")
    (capture / "buffer-link").symlink_to(payload.name)

    name, capture_format = _archive_capture(capture)

    archived = tmp_path / name
    assert capture_format == "tar-gzip-wrapped-gputrace"
    assert archived.is_file() and not archived.is_symlink()
    assert not capture.exists()
    with tarfile.open(archived, mode="r:gz") as bundle:
        linked = bundle.extractfile("metal.gputrace/buffer-link")
        assert linked is not None
        assert linked.read() == b"metal-capture"


def test_cpu_stage_profiler_preserves_hpsi_result_and_runtime_selection():
    context = _tiny_profile_context()
    selected_device = mx.default_device()
    selected_stream = mx.default_stream(selected_device)
    try:
        case = _build_profile_case(context, 2)
        before = context.operator._apply_compact(
            case.state,
            prepared_batch=case.batch,
        ).values
        mx.eval(before)
        measured = _measure_case(case, warmups=0, samples=1)
        split_local = _stage_action(case, "gather")()[0]
        complete_local = _stage_action(case, "local-fft")()[0]
        mx.eval(split_local, complete_local)
        after = context.operator._apply_compact(
            case.state,
            prepared_batch=case.batch,
        ).values
        mx.eval(after)

        assert set(measured["stages"]) == set(PROFILE_STAGES)
        assert set(measured["attribution"]["local_fft_substages_over_hpsi"]) == set(
            LOCAL_FFT_PROFILE_STAGES
        )
        assert all(
            stage["median_seconds"] > 0.0 and stage["output"]["finite"]
            for stage in measured["stages"].values()
        )
        np.testing.assert_allclose(
            np.asarray(after),
            np.asarray(before),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(split_local),
            np.asarray(complete_local),
            rtol=1e-6,
            atol=1e-6,
        )
        assert mx.default_device() == selected_device
        assert mx.default_stream(selected_device) == selected_stream
    finally:
        context.nonlocal_operator.close()
