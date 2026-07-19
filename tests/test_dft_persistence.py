from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic._artifact_identity import (
    ArtifactIntegrityError,
    AtomicGeneration,
    canonical_json_bytes,
    inspect_generation,
    sha256_bytes,
)
from mlx_atomistic.dft import (
    PERIODIC_SCF_CHECKPOINT_KIND,
    PERIODIC_SCF_CHECKPOINT_SCHEMA,
    GTHProjectorChannel,
    KPoint,
    KPointMesh,
    PeriodicDavidsonConfig,
    PeriodicDFTSystem,
    PeriodicSCFConfig,
    PeriodicSCFExecutionIdentity,
    PlaneWaveBasis,
    PseudopotentialData,
    PseudopotentialFormat,
    ReciprocalGrid,
    admit_time_reversal_bases,
    build_time_reversal_ownership,
    inspect_periodic_scf_checkpoint,
    load_periodic_scf_checkpoint,
    periodic_scf_calculation_contract,
    periodic_scf_execution_settings,
    periodic_scf_initialization_identity,
    publish_periodic_scf_checkpoint,
    run_periodic_scf,
    run_periodic_scf_checkpointed,
)
from mlx_atomistic.dft._runtime_observer import RuntimeObserver
from mlx_atomistic.dft.artifacts import PERIODIC_SCF_CHECKPOINT_PAYLOAD


def _stalled_checkpoint_worker(queue, _manifest, _gth_source, state_root):
    isolated = False
    if hasattr(os, "setsid"):
        os.setsid()
        isolated = True
    queue.put({"type": "ready", "process_group_isolated": isolated})
    generation = AtomicGeneration(
        Path(state_root) / "checkpoint",
        PERIODIC_SCF_CHECKPOINT_KIND,
        PERIODIC_SCF_CHECKPOINT_SCHEMA,
    )
    generation.__enter__()
    generation.write_bytes("payload.bin", b"partial")
    queue.put(
        {
            "type": "progress",
            "event": {"event": "checkpoint_partial", "status": "started"},
        }
    )
    while True:
        time.sleep(1.0)


def _problem(*, mixer: str = "diis"):
    pseudo = PseudopotentialData(
        element="H",
        format=PseudopotentialFormat.GTH,
        valence_charge=1.0,
        gth_rloc=0.25,
        gth_coefficients=(-1.0,),
        gth_channels=(GTHProjectorChannel(0, 0.3, ((0.5,),)),),
    )
    system = PeriodicDFTSystem(
        (8.0, 8.0, 8.0),
        (10, 10, 10),
        ((2.5, 4.0, 4.0), (5.5, 4.0, 4.0)),
        pseudo,
    )
    mesh = KPointMesh(
        (
            KPoint((-0.125, 0.0, 0.0), weight=0.5, coordinate_system="reduced"),
            KPoint((0.125, 0.0, 0.0), weight=0.5, coordinate_system="reduced"),
        )
    )
    config = PeriodicSCFConfig(
        max_iterations=4,
        min_iterations=4,
        density_tolerance=1e-12,
        energy_tolerance=1e-12,
        orbital_tolerance=5e-3,
        mixing_beta=0.5,
        mixer=mixer,
        kpoint_batch_size=2,
        davidson=PeriodicDavidsonConfig(
            max_iterations=10,
            tolerance=5e-3,
            max_subspace_size=10,
        ),
    )
    return system, mesh, config


def _execution_context(
    system,
    mesh,
    config,
    *,
    initial_density=None,
    initial_coefficients=None,
):
    calculation = periodic_scf_calculation_contract(
        system,
        cutoff_hartree=4.0,
        kpoint_mesh=mesh,
        n_bands=1,
        config=config,
    )
    contract = {
        "schema_version": "mlx-atomistic.dft-runtime-execution-contract.v1",
        "command_kind": "periodic-scf",
        "workload_fingerprint": "a" * 64,
        "protocol_fingerprint": "b" * 64,
        "runtime_fingerprint": "c" * 64,
        "solver": {
            "davidson": calculation["config"]["davidson"],
            "scf": {
                "density_tolerance": calculation["config"]["density_tolerance"],
                "energy_tolerance_hartree": calculation["config"][
                    "energy_tolerance_hartree"
                ],
                "max_iterations": calculation["config"]["max_iterations"],
                "min_iterations": calculation["config"]["min_iterations"],
                "mixer": calculation["config"]["mixer"],
                "mixing_beta": calculation["config"]["mixing_beta"],
                "orbital_tolerance": calculation["config"]["orbital_tolerance"],
            },
        },
        "initialization": periodic_scf_initialization_identity(
            initial_density=initial_density,
            initial_coefficients=initial_coefficients,
        ),
        "settings_override": periodic_scf_execution_settings(config),
        "lock": {"path": "uv.lock", "byte_size": 1, "sha256": "d" * 64},
        "environment": {
            "python_version": "3.13.0",
            "python_implementation": "CPython",
            "mlx_version": "test",
            "default_device": str(mx.default_device()),
            "metal_available": True,
            "selected_device": str(mx.default_device()),
            "precision": "complex64/float32",
            "full_grid_precision": "complex64/float32",
            "projected_eigensolve_device": "cpu",
            "projected_eigensolve_backend": "numpy-lapack-cpu-complex128",
            "projected_eigensolve_precision": "complex128",
            "projected_eigensolve_output_precision": "float32/complex64",
        },
        "host_protocol": {
            "model": "test-model",
            "model_identifier": "test-id",
            "chip": "test-host",
            "machine": "arm64",
            "macos": {
                "ProductName": "macOS",
                "ProductVersion": "test",
                "BuildVersion": "test",
            },
            "power_source": "Battery Power",
            "low_power_mode": 1,
        },
        "synchronization": "explicit-test-boundaries",
    }
    return {
        "execution_contract": contract,
        "execution_contract_fingerprint": sha256_bytes(canonical_json_bytes(contract)),
        "protocol_fingerprint": contract["protocol_fingerprint"],
        "runtime_fingerprint": contract["runtime_fingerprint"],
        "git": {"commit": "e" * 40, "dirty": True},
    }


def _mutated_context(context, dotted_field: str, value: object):
    mutated = json.loads(json.dumps(context))
    target = mutated["execution_contract"]
    parts = dotted_field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    if dotted_field == "workload_fingerprint":
        pass
    elif dotted_field == "protocol_fingerprint":
        mutated["protocol_fingerprint"] = value
    elif dotted_field == "runtime_fingerprint":
        mutated["runtime_fingerprint"] = value
    mutated["execution_contract_fingerprint"] = sha256_bytes(
        canonical_json_bytes(mutated["execution_contract"])
    )
    return mutated


def _republish_checkpoint(
    source: Path,
    destination: Path,
    *,
    mutate_metadata=lambda metadata: metadata,
    mutate_envelope=lambda metadata: metadata,
    artifact_schema: str = PERIODIC_SCF_CHECKPOINT_SCHEMA,
):
    manifest = inspect_generation(source)
    metadata = json.loads((source / PERIODIC_SCF_CHECKPOINT_PAYLOAD).read_bytes())
    metadata = mutate_metadata(metadata)
    envelope_metadata = mutate_envelope(json.loads(json.dumps(manifest["metadata"])))
    with AtomicGeneration(
        destination,
        PERIODIC_SCF_CHECKPOINT_KIND,
        artifact_schema,
        identity=manifest["identity"],
        metadata=envelope_metadata,
    ) as generation:
        for record in manifest["files"]:
            path = record["path"]
            if path == PERIODIC_SCF_CHECKPOINT_PAYLOAD:
                generation.write_json(path, metadata)
            else:
                generation.write_bytes(path, (source / path).read_bytes())
        return generation.publish()


def _run_to_checkpoint(tmp_path: Path, *, mixer: str = "diis"):
    system, mesh, config = _problem(mixer=mixer)
    context = _execution_context(system, mesh, config)
    destination = tmp_path / f"{mixer}-checkpoint"
    partial = run_periodic_scf_checkpointed(
        system,
        cutoff_hartree=4.0,
        kpoint_mesh=mesh,
        n_bands=1,
        config=config,
        execution_context=context,
        checkpoint_to=destination,
        checkpoint_iteration=2,
        provenance=context["git"],
    )
    return system, mesh, config, context, destination, partial


@pytest.mark.parametrize("mixer", ["linear", "diis"])
def test_resume_trajectory_equivalence_and_timing_admission(tmp_path, mixer):
    system, mesh, config, context, checkpoint, partial = _run_to_checkpoint(
        tmp_path,
        mixer=mixer,
    )
    uninterrupted = run_periodic_scf(
        system,
        cutoff_hartree=4.0,
        kpoint_mesh=mesh,
        n_bands=1,
        config=config,
    )
    resumed = run_periodic_scf_checkpointed(
        system,
        cutoff_hartree=4.0,
        kpoint_mesh=mesh,
        n_bands=1,
        config=config,
        execution_context=context,
        resume_from=checkpoint,
    )

    assert partial.status == "checkpointed"
    assert partial.iterations == 2
    assert partial.timing_admission_status == "ineligible_checkpointed"
    assert resumed.iterations == uninterrupted.iterations == 4
    assert resumed.resume_integrity_status == "validated"
    assert resumed.timing_admission_status == "ineligible_resumed_state"
    assert resumed.lineage == (inspect_generation(checkpoint)["manifest_sha256"],)
    assert len(resumed.history) == len(uninterrupted.history) == 4
    for resumed_row, uninterrupted_row in zip(
        resumed.history,
        uninterrupted.history,
        strict=True,
    ):
        assert resumed_row.keys() == uninterrupted_row.keys()
        for key in resumed_row:
            if isinstance(resumed_row[key], float):
                assert resumed_row[key] == pytest.approx(
                    uninterrupted_row[key],
                    abs=2e-6,
                    rel=0.0,
                )
            else:
                assert resumed_row[key] == uninterrupted_row[key]
    assert resumed.total_energy == pytest.approx(
        uninterrupted.total_energy,
        abs=2e-6,
        rel=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(resumed.density),
        np.asarray(uninterrupted.density),
        atol=2e-6,
        rtol=0.0,
    )
    for resumed_point, uninterrupted_point in zip(
        resumed.kpoints,
        uninterrupted.kpoints,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(resumed_point.eigen.eigenvalues),
            np.asarray(uninterrupted_point.eigen.eigenvalues),
            atol=2e-6,
            rtol=0.0,
        )


def test_shared_envelope_and_shared_publisher_checkpoint_serialization(tmp_path):
    system, mesh, config, context, checkpoint, partial = _run_to_checkpoint(tmp_path)
    manifest = inspect_generation(checkpoint)
    summary = inspect_periodic_scf_checkpoint(
        checkpoint,
        expected_execution_context=context,
    )
    metadata = json.loads((checkpoint / PERIODIC_SCF_CHECKPOINT_PAYLOAD).read_bytes())

    assert manifest["artifact_kind"] == PERIODIC_SCF_CHECKPOINT_KIND
    assert manifest["artifact_schema_version"] == PERIODIC_SCF_CHECKPOINT_SCHEMA
    assert manifest["complete"] is True
    assert summary["status"] == "ok"
    assert summary["completed_iteration"] == partial.iterations == 2
    assert metadata["resume_eligible"] is True
    assert metadata["next_iteration"] == 3
    assert metadata["statuses"] == {
        "numerical_status": "accepted_iteration",
        "resume_integrity_status": "fresh",
        "timing_admission_status": "not_a_timing_sample",
    }
    assert len(metadata["owned_lanes"]) == 1
    assert metadata["ownership"]["explicit_count"] == 2
    coefficient_files = [
        path.name for path in checkpoint.joinpath("owned").glob("*.npy")
    ]
    assert coefficient_files == ["0000-coefficients.npy"]
    assert not any("partner" in record["path"] for record in manifest["files"])
    assert len(metadata["history"]) == 2
    assert metadata["mixer"]["stored"] <= metadata["mixer"]["history_size"]

    second = tmp_path / "second"
    republished = publish_periodic_scf_checkpoint(
        second,
        partial,
        system=system,
        cutoff_hartree=4.0,
        kpoint_mesh=mesh,
        n_bands=1,
        config=config,
        execution_context=context,
    )
    assert republished["complete"] is True


def test_atomic_collision_and_concurrent_checkpoint_publishers(tmp_path):
    system, mesh, config, context, checkpoint, partial = _run_to_checkpoint(tmp_path)
    before = {
        path.relative_to(checkpoint): path.read_bytes()
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError):
        publish_periodic_scf_checkpoint(
            checkpoint,
            partial,
            system=system,
            cutoff_hartree=4.0,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            execution_context=context,
        )
    after = {
        path.relative_to(checkpoint): path.read_bytes()
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    assert after == before

    destination = tmp_path / "concurrent"

    def publish():
        return publish_periodic_scf_checkpoint(
            destination,
            partial,
            system=system,
            cutoff_hartree=4.0,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            execution_context=context,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(publish), executor.submit(publish))
        outcomes = [future.exception() for future in futures]
    assert sum(outcome is None for outcome in outcomes) == 1
    assert sum(isinstance(outcome, FileExistsError) for outcome in outcomes) == 1
    assert inspect_generation(destination)["complete"] is True
    assert not list(tmp_path.glob(".concurrent.tmp-*"))


@pytest.mark.parametrize("stage", ["after_payload_sync", "after_manifest"])
def test_fault_injection_before_completion_is_atomic(tmp_path, stage):
    system, mesh, config, context, _, partial = _run_to_checkpoint(tmp_path)
    destination = tmp_path / f"fault-{stage}"

    def fail(observed_stage):
        if observed_stage == stage:
            raise RuntimeError(f"fault at {stage}")

    with pytest.raises(RuntimeError, match="fault at"):
        publish_periodic_scf_checkpoint(
            destination,
            partial,
            system=system,
            cutoff_hartree=4.0,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            execution_context=context,
            fault_hook=fail,
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(f".{destination.name}.tmp-*"))


def test_fault_injection_after_completion_preserves_valid_generation(tmp_path):
    system, mesh, config, context, _, partial = _run_to_checkpoint(tmp_path)
    destination = tmp_path / "fault-after-rename"

    def fail(stage):
        if stage == "after_rename":
            raise RuntimeError("fault after rename")

    with pytest.raises(RuntimeError, match="fault after rename"):
        publish_periodic_scf_checkpoint(
            destination,
            partial,
            system=system,
            cutoff_hartree=4.0,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            execution_context=context,
            fault_hook=fail,
        )
    assert inspect_generation(destination)["complete"] is True


def test_stale_temp_is_ignored_and_direct_resume_is_rejected(tmp_path):
    system, mesh, config, context, checkpoint, _ = _run_to_checkpoint(tmp_path)
    stale = tmp_path / ".unrelated.tmp-stale"
    stale.mkdir()
    (stale / "artifact-manifest.json").write_text("{}")
    assert inspect_periodic_scf_checkpoint(checkpoint)["status"] == "ok"
    with pytest.raises(ArtifactIntegrityError, match="temporary|schema"):
        load_periodic_scf_checkpoint(
            stale,
            system=system,
            cutoff_hartree=4.0,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            execution_context=context,
        )


def test_timeout_cleanup_leaves_no_completed_checkpoint(tmp_path):
    destination = tmp_path / "timeout"
    with (
        pytest.raises(TimeoutError),
        AtomicGeneration(
            destination,
            PERIODIC_SCF_CHECKPOINT_KIND,
            PERIODIC_SCF_CHECKPOINT_SCHEMA,
        ) as generation,
    ):
        generation.write_bytes("payload.bin", b"partial")
        raise TimeoutError("supervised worker timeout")
    assert not destination.exists()
    assert not list(tmp_path.glob(".timeout.tmp-*"))


def test_timeout_cleanup_kills_child_and_publishes_no_checkpoint(tmp_path):
    from mlx_atomistic.benchmarks.dft_runtime_core import supervise_full_scf_worker

    outcome = supervise_full_scf_worker(
        manifest_path=tmp_path / "unused-manifest",
        gth_source=tmp_path / "unused-gth",
        state_root=tmp_path,
        timeout_seconds=0.2,
        worker=_stalled_checkpoint_worker,
    )
    assert outcome["status"] == "timeout"
    assert outcome["timed_out"] is True
    assert outcome["worker_alive_after_cleanup"] is False
    assert not (tmp_path / "checkpoint").exists()
    stale = list(tmp_path.glob(".checkpoint.tmp-*"))
    assert len(stale) <= 1
    if stale:
        with pytest.raises(ArtifactIntegrityError, match="temporary"):
            inspect_generation(stale[0])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workload_fingerprint", "1" * 64),
        ("protocol_fingerprint", "2" * 64),
        ("runtime_fingerprint", "3" * 64),
        ("lock.sha256", "4" * 64),
        ("environment.selected_device", "Device(gpu, 99)"),
        ("environment.precision", "float64/complex128"),
        ("solver.scf.mixing_beta", 0.25),
        ("settings_override.periodic_scf.batch_policy.kpoint_batch_size", 1),
    ],
)
def test_resume_rejects_execution_identity_mismatch_before_array_decode(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    import mlx_atomistic.dft.artifacts as checkpoint_module

    system, mesh, config, context, checkpoint, _ = _run_to_checkpoint(tmp_path)
    mismatched = _mutated_context(context, field, value)

    def forbidden_decode(*_args, **_kwargs):
        raise AssertionError("array decoding crossed the identity gate")

    monkeypatch.setattr(checkpoint_module.np, "load", forbidden_decode)
    with pytest.raises(ArtifactIntegrityError, match="identity|settings"):
        load_periodic_scf_checkpoint(
            checkpoint,
            system=system,
            cutoff_hartree=4.0,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            execution_context=mismatched,
        )


def test_resume_rejects_solver_batch_and_pseudopotential_mismatch(tmp_path, monkeypatch):
    import mlx_atomistic.dft.artifacts as checkpoint_module

    system, mesh, config, context, checkpoint, _ = _run_to_checkpoint(tmp_path)

    def forbidden_decode(*_args, **_kwargs):
        raise AssertionError("array decoding crossed the calculation gate")

    monkeypatch.setattr(checkpoint_module.np, "load", forbidden_decode)
    for mismatched_config in (
        replace(config, mixing_beta=0.25),
        replace(config, kpoint_batch_size=1),
        replace(config, davidson=replace(config.davidson, tolerance=1e-4)),
    ):
        with pytest.raises(ArtifactIntegrityError, match="settings"):
            load_periodic_scf_checkpoint(
                checkpoint,
                system=system,
                cutoff_hartree=4.0,
                kpoint_mesh=mesh,
                n_bands=1,
                config=mismatched_config,
                execution_context=context,
            )

    pseudo = replace(system.pseudopotential, gth_coefficients=(-0.9,))
    mismatched_system = PeriodicDFTSystem(
        np.asarray(system.grid.lengths),
        system.grid.shape,
        system.positions,
        pseudo,
    )
    with pytest.raises(ArtifactIntegrityError, match="settings"):
        load_periodic_scf_checkpoint(
            checkpoint,
            system=mismatched_system,
            cutoff_hartree=4.0,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            execution_context=context,
        )


def test_resume_rejects_incomplete_execution_contract_before_running(tmp_path):
    system, mesh, config = _problem()
    context = _execution_context(system, mesh, config)
    del context["execution_contract"]["lock"]
    context["execution_contract_fingerprint"] = sha256_bytes(
        canonical_json_bytes(context["execution_contract"])
    )
    destination = tmp_path / "incomplete-context"
    with pytest.raises(ValueError, match="required object"):
        run_periodic_scf_checkpointed(
            system,
            cutoff_hartree=4.0,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            execution_context=context,
            checkpoint_to=destination,
            checkpoint_iteration=2,
        )
    assert not destination.exists()


def test_resume_rejects_initialization_identity_mismatch_before_running(tmp_path):
    system, mesh, config = _problem(mixer="linear")
    context = _execution_context(system, mesh, config)
    density = mx.full(system.grid.shape, system.electron_count / system.grid.volume)
    destination = tmp_path / "wrong-initialization"
    with pytest.raises(ArtifactIntegrityError, match="initialization"):
        run_periodic_scf_checkpointed(
            system,
            cutoff_hartree=4.0,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            execution_context=context,
            initial_density=density,
            checkpoint_to=destination,
            checkpoint_iteration=2,
        )
    assert not destination.exists()


def test_checksum_completion_and_unknown_schema_fail_before_array_decode(
    tmp_path,
    monkeypatch,
):
    import mlx_atomistic.dft.artifacts as checkpoint_module

    system, mesh, config, context, checkpoint, _ = _run_to_checkpoint(tmp_path)
    coefficient = next(checkpoint.joinpath("owned").glob("*.npy"))
    coefficient.write_bytes(coefficient.read_bytes() + b"tamper")

    def forbidden_decode(*_args, **_kwargs):
        raise AssertionError("array decoding crossed the integrity gate")

    monkeypatch.setattr(checkpoint_module.np, "load", forbidden_decode)
    with pytest.raises(ArtifactIntegrityError, match="inventory|checksum"):
        load_periodic_scf_checkpoint(
            checkpoint,
            system=system,
            cutoff_hartree=4.0,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            execution_context=context,
        )


def test_completion_envelope_and_checkpoint_payload_must_agree(tmp_path):
    system, mesh, config, context, checkpoint, _ = _run_to_checkpoint(tmp_path)
    inconsistent = tmp_path / "inconsistent-envelope"

    def mutate(envelope):
        envelope["completed_iteration"] = 99
        return envelope

    _republish_checkpoint(checkpoint, inconsistent, mutate_envelope=mutate)
    with pytest.raises(ArtifactIntegrityError, match="envelope.*payload"):
        load_periodic_scf_checkpoint(
            inconsistent,
            system=system,
            cutoff_hartree=4.0,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            execution_context=context,
        )

    _, _, _, _, valid, _ = _run_to_checkpoint(tmp_path / "valid")
    unknown = tmp_path / "unknown-schema"
    _republish_checkpoint(valid, unknown, artifact_schema="unknown-checkpoint.v9")
    with pytest.raises(ArtifactIntegrityError, match="supported"):
        load_periodic_scf_checkpoint(
            unknown,
            system=system,
            cutoff_hartree=4.0,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            execution_context=context,
        )

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(ArtifactIntegrityError, match="manifest"):
        load_periodic_scf_checkpoint(
            incomplete,
            system=system,
            cutoff_hartree=4.0,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            execution_context=context,
        )


@pytest.mark.parametrize("invalid_path", ["../density.npy", "/density.npy", "a/../b.npy"])
def test_confinement_rejects_absolute_parent_and_noncanonical_payload_refs(
    tmp_path,
    invalid_path,
):
    system, mesh, config, context, checkpoint, _ = _run_to_checkpoint(tmp_path)
    invalid = tmp_path / "invalid-reference"

    def mutate(metadata):
        metadata["density_file"] = invalid_path
        return metadata

    _republish_checkpoint(checkpoint, invalid, mutate_metadata=mutate)
    with pytest.raises(ArtifactIntegrityError, match="undeclared|confined"):
        load_periodic_scf_checkpoint(
            invalid,
            system=system,
            cutoff_hartree=4.0,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            execution_context=context,
        )


def test_confinement_rejects_symlink_payload(tmp_path):
    system, mesh, config, context, checkpoint, _ = _run_to_checkpoint(tmp_path)
    density = checkpoint / "density.npy"
    outside = tmp_path / "outside.npy"
    outside.write_bytes(density.read_bytes())
    density.unlink()
    density.symlink_to(outside)
    with pytest.raises(ArtifactIntegrityError, match="regular|inventory|checksum"):
        load_periodic_scf_checkpoint(
            checkpoint,
            system=system,
            cutoff_hartree=4.0,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            execution_context=context,
        )


def test_path_independent_identity_excludes_output_resume_locator_and_git_state(tmp_path):
    system, mesh, config, context, first, partial = _run_to_checkpoint(tmp_path)
    second = tmp_path / "elsewhere" / "second"
    second.parent.mkdir()
    second_context = json.loads(json.dumps(context))
    second_context["git"] = {"commit": "f" * 40, "dirty": False}
    publish_periodic_scf_checkpoint(
        second,
        partial,
        system=system,
        cutoff_hartree=4.0,
        kpoint_mesh=mesh,
        n_bands=1,
        config=config,
        execution_context=second_context,
        provenance=second_context["git"],
    )
    first_manifest = inspect_generation(first)
    second_manifest = inspect_generation(second)
    first_metadata = json.loads((first / PERIODIC_SCF_CHECKPOINT_PAYLOAD).read_bytes())
    second_metadata = json.loads((second / PERIODIC_SCF_CHECKPOINT_PAYLOAD).read_bytes())
    assert first_manifest["identity"] == second_manifest["identity"]
    assert (
        first_metadata["calculation_fingerprint"]
        == second_metadata["calculation_fingerprint"]
    )
    semantic_bytes = canonical_json_bytes(
        {
            "identity": first_manifest["identity"],
            "calculation": first_metadata["calculation_contract"],
        }
    )
    assert str(tmp_path).encode() not in semantic_bytes


def test_path_independent_execution_identity_revalidates_nested_mutation(tmp_path):
    system, mesh, config = _problem()
    context = _execution_context(system, mesh, config)
    identity = PeriodicSCFExecutionIdentity.from_context(context)
    original_fingerprint = identity.execution_contract_fingerprint
    context["execution_contract"]["environment"]["precision"] = "mutated-caller"
    assert identity.execution_contract["environment"]["precision"] == (
        "complex64/float32"
    )
    identity.execution_contract["environment"]["precision"] = "mutated-instance"
    with pytest.raises(ValueError, match="fingerprint"):
        run_periodic_scf_checkpointed(
            system,
            cutoff_hartree=4.0,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            execution_context=identity,
        )
    assert identity.execution_contract_fingerprint == original_fingerprint


def test_implicit_resume_is_never_attempted_and_invalid_explicit_has_no_fallback(
    tmp_path,
    monkeypatch,
):
    import mlx_atomistic.dft.artifacts as checkpoint_module

    system, mesh, config = _problem(mixer="linear")
    short_config = replace(config, max_iterations=1, min_iterations=1)
    context = _execution_context(system, mesh, short_config)

    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("implicit resume was attempted")

    monkeypatch.setattr(checkpoint_module, "load_periodic_scf_checkpoint", forbidden_load)
    fresh = run_periodic_scf_checkpointed(
        system,
        cutoff_hartree=4.0,
        kpoint_mesh=mesh,
        n_bands=1,
        config=short_config,
        execution_context=context,
    )
    assert fresh.resume_integrity_status == "fresh"

    invalid = tmp_path / "legacy.npz"
    np.savez(invalid, density=np.zeros((2, 2, 2), dtype=np.float32))

    def forbidden_run(*_args, **_kwargs):
        raise AssertionError("invalid explicit resume fell back to a fresh run")

    monkeypatch.undo()
    monkeypatch.setattr(checkpoint_module, "_run_periodic_scf_controlled", forbidden_run)
    with pytest.raises(ArtifactIntegrityError, match="manifest"):
        run_periodic_scf_checkpointed(
            system,
            cutoff_hartree=4.0,
            kpoint_mesh=mesh,
            n_bands=1,
            config=short_config,
            execution_context=context,
            resume_from=invalid,
        )


def test_implicit_resume_and_serialization_do_not_touch_ordinary_runtime(
    monkeypatch,
):
    import mlx_atomistic.dft.periodic_scf as periodic_scf_module

    system, mesh, config = _problem(mixer="linear")
    short_config = replace(config, max_iterations=1, min_iterations=1)
    context = _execution_context(system, mesh, short_config)

    def forbidden_capture(*_args, **_kwargs):
        raise AssertionError("ordinary SCF materialized checkpoint state")

    monkeypatch.setattr(
        periodic_scf_module,
        "_continuation_state_from_boundary",
        forbidden_capture,
    )
    ordinary = run_periodic_scf(
        system,
        cutoff_hartree=4.0,
        kpoint_mesh=mesh,
        n_bands=1,
        config=short_config,
    )
    artifact_aware = run_periodic_scf_checkpointed(
        system,
        cutoff_hartree=4.0,
        kpoint_mesh=mesh,
        n_bands=1,
        config=short_config,
        execution_context=context,
    )
    assert ordinary.iterations == artifact_aware.iterations == 1
    assert ordinary._checkpoint_state is None
    assert artifact_aware._checkpoint_state is None


def test_timing_admission_persistence_event_precedes_state_serialization(
    tmp_path,
    monkeypatch,
):
    import mlx_atomistic.dft.periodic_scf as periodic_scf_module

    system, mesh, config = _problem(mixer="linear")
    context = _execution_context(system, mesh, config)
    delivered_events = []
    observer = RuntimeObserver(
        synchronize=mx.synchronize,
        callback=delivered_events.append,
    )
    original = periodic_scf_module._continuation_state_from_boundary
    saw_started_event = []

    def observed_materialization(*args, **kwargs):
        saw_started_event.append(
            any(
                event["event"] == "persistence" and event["status"] == "started"
                for event in delivered_events
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(
        periodic_scf_module,
        "_continuation_state_from_boundary",
        observed_materialization,
    )
    run_periodic_scf_checkpointed(
        system,
        cutoff_hartree=4.0,
        kpoint_mesh=mesh,
        n_bands=1,
        config=config,
        execution_context=context,
        observer=observer,
        checkpoint_to=tmp_path / "timed-checkpoint",
        checkpoint_iteration=2,
    )
    snapshot = observer.snapshot()
    assert saw_started_event == [True]
    assert snapshot["phase_seconds"]["persistence"] > 0.0
    persistence_events = [
        event for event in snapshot["events"] if event["event"] == "persistence"
    ]
    assert [event["status"] for event in persistence_events] == [
        "started",
        "completed",
    ]


def test_resume_checkpoint_at_iteration_limit_is_rejected_as_ineligible(tmp_path):
    system, mesh, config = _problem()
    context = _execution_context(system, mesh, config)
    destination = tmp_path / "never-resumable"
    with pytest.raises(ValueError, match="precede"):
        run_periodic_scf_checkpointed(
            system,
            cutoff_hartree=4.0,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            execution_context=context,
            checkpoint_to=destination,
            checkpoint_iteration=config.max_iterations,
        )
    assert not destination.exists()


def test_resume_trajectory_equivalence_preserves_initial_mismatch_fallback(tmp_path):
    system, mesh, config = _problem(mixer="linear")
    reciprocal = ReciprocalGrid.from_real_space(system.grid)
    bases = [
        PlaneWaveBasis.from_reduced_kpoint(
            system.grid,
            4.0,
            point.vector,
            reciprocal_grid=reciprocal,
            lane_label=f"kpoint:{index}",
        )
        for index, point in enumerate(mesh.points)
    ]
    topology = admit_time_reversal_bases(
        build_time_reversal_ownership(mesh),
        bases,
    )
    permutation = np.asarray(topology.entries[0].time_reversal_permutation)
    owner_values = np.zeros((1, bases[0].active_count), dtype=np.complex64)
    partner_values = np.zeros((1, bases[1].active_count), dtype=np.complex64)
    owner_values[0, 0] = 1.0
    partner_values[0, (int(permutation[0]) + 1) % bases[1].active_count] = 1.0
    initial = (
        bases[0]._layout.unpack_fresh(mx.array(owner_values)),
        bases[1]._layout.unpack_fresh(mx.array(partner_values)),
    )
    context = _execution_context(
        system,
        mesh,
        config,
        initial_coefficients=initial,
    )

    uninterrupted = run_periodic_scf(
        system,
        cutoff_hartree=4.0,
        kpoint_mesh=mesh,
        n_bands=1,
        config=config,
        initial_coefficients=initial,
    )
    checkpoint = tmp_path / "fallback-checkpoint"
    partial = run_periodic_scf_checkpointed(
        system,
        cutoff_hartree=4.0,
        kpoint_mesh=mesh,
        n_bands=1,
        config=config,
        execution_context=context,
        initial_coefficients=initial,
        checkpoint_to=checkpoint,
        checkpoint_iteration=2,
    )
    resumed = run_periodic_scf_checkpointed(
        system,
        cutoff_hartree=4.0,
        kpoint_mesh=mesh,
        n_bands=1,
        config=config,
        execution_context=context,
        resume_from=checkpoint,
    )
    expected_fallback = {
        0: "initial_coefficients_time_reversal_mismatch",
        1: "initial_coefficients_time_reversal_mismatch",
    }
    assert partial.time_reversal_ownership.fallback_reasons == expected_fallback
    assert resumed.time_reversal_ownership.fallback_reasons == expected_fallback
    assert resumed.time_reversal_ownership.owned_indices == (0, 1)
    assert len(resumed.owned_kpoints) == 2
    metadata = json.loads((checkpoint / PERIODIC_SCF_CHECKPOINT_PAYLOAD).read_bytes())
    assert len(metadata["owned_lanes"]) == 2
    assert len(list(checkpoint.joinpath("owned").glob("*.npy"))) == 2
    np.testing.assert_allclose(
        np.asarray(resumed.density),
        np.asarray(uninterrupted.density),
        atol=2e-6,
        rtol=0.0,
    )
    assert resumed.total_energy == pytest.approx(
        uninterrupted.total_energy,
        abs=2e-6,
        rel=0.0,
    )


def test_legacy_restart_is_not_reinterpreted_as_periodic_resume(tmp_path):
    system, mesh, config = _problem()
    context = _execution_context(system, mesh, config)
    legacy = tmp_path / "legacy-dense-restart.npz"
    np.savez_compressed(
        legacy,
        density=np.zeros((2, 2, 2), dtype=np.float32),
        orbitals=np.zeros((1, 2, 2, 2), dtype=np.complex64),
        positions=np.zeros((1, 3)),
        cell_lengths=np.ones(3),
        metadata_json=np.asarray("{}"),
    )
    with pytest.raises(ArtifactIntegrityError, match="manifest"):
        load_periodic_scf_checkpoint(
            legacy,
            system=system,
            cutoff_hartree=4.0,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            execution_context=context,
        )


def test_dft_artifacts_cli_serialization_is_read_only_and_json_stable(tmp_path):
    _, _, _, context, checkpoint, _ = _run_to_checkpoint(tmp_path)
    context_path = tmp_path / "execution-context.json"
    context_path.write_bytes(canonical_json_bytes(context) + b"\n")
    before = {
        path.relative_to(checkpoint): (path.stat().st_mtime_ns, path.read_bytes())
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_atomistic.benchmarks.dft_artifacts",
            "inspect",
            "--artifact",
            str(checkpoint),
            "--expected-execution-context",
            str(context_path),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert completed.stdout.encode() == canonical_json_bytes(payload) + b"\n"
    after = {
        path.relative_to(checkpoint): (path.stat().st_mtime_ns, path.read_bytes())
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    assert after == before

    mismatched = _mutated_context(context, "runtime_fingerprint", "9" * 64)
    context_path.write_bytes(canonical_json_bytes(mismatched) + b"\n")
    blocked = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_atomistic.benchmarks.dft_artifacts",
            "inspect",
            "--artifact",
            str(checkpoint),
            "--expected-execution-context",
            str(context_path),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert blocked.returncode == 2
    assert json.loads(blocked.stdout)["status"] == "blocked"

    context_path.write_text("[]\n")
    malformed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_atomistic.benchmarks.dft_artifacts",
            "inspect",
            "--artifact",
            str(checkpoint),
            "--expected-execution-context",
            str(context_path),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert malformed.returncode == 2
    assert json.loads(malformed.stdout)["status"] == "blocked"
    assert malformed.stderr == ""


def test_docs_only_is_identity_stable_and_hot_path_mismatch_is_runtime_scoped(
    tmp_path,
    monkeypatch,
):
    import mlx_atomistic.benchmarks.dft_runtime_contract as contract_module

    root = tmp_path / "checkout"
    (root / "src/mlx_atomistic/dft").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (root / "protocol.py").write_text("PROTOCOL = 1\n")
    (root / "runtime.py").write_text("RUNTIME = 1\n")
    (root / "src/mlx_atomistic/dft/hot.py").write_text("HOT = 1\n")
    (root / "docs/readme.md").write_text("docs v1\n")
    monkeypatch.setattr(contract_module, "PROTOCOL_SOURCE_PATHS", ("protocol.py",))
    monkeypatch.setattr(contract_module, "RUNTIME_SOURCE_PATHS", ("runtime.py",))
    monkeypatch.setattr(
        contract_module,
        "RUNTIME_SOURCE_ROOTS",
        ("src/mlx_atomistic/dft",),
    )
    original = contract_module.build_source_fingerprints(root)
    (root / "docs/readme.md").write_text("docs v2\n")
    assert contract_module.build_source_fingerprints(root) == original
    (root / "src/mlx_atomistic/dft/hot.py").write_text("HOT = 2\n")
    hot = contract_module.build_source_fingerprints(root)
    assert hot["protocol_fingerprint"] == original["protocol_fingerprint"]
    assert hot["runtime_fingerprint"] != original["runtime_fingerprint"]
    (root / "protocol.py").write_text("PROTOCOL = 2\n")
    changed_protocol = contract_module.build_source_fingerprints(root)
    assert changed_protocol["protocol_fingerprint"] != original["protocol_fingerprint"]
