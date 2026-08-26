from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from mlx_atomistic._artifact_identity import sha256_bytes
from mlx_atomistic.dft import (
    ATOMIC_MASS_UNIT_TO_ELECTRON_MASS,
    HARTREE_TO_WAVENUMBER_CM1,
    PERIODIC_PHONON_SAMPLES_SCHEMA,
    KPoint,
    KPointMesh,
    PeriodicDFTSystem,
    PeriodicPhononConfig,
    PeriodicPhononSample,
    PeriodicPhononSampleSet,
    PeriodicPhononSymmetry,
    PseudopotentialData,
    PseudopotentialFormat,
    assemble_periodic_phonons,
    compare_periodic_phonon_displacements,
    cubic_reciprocal_symmetry_operations,
    evaluate_periodic_phonon_sample,
    periodic_phonon_displaced_system,
    plan_periodic_phonon_displacements,
    read_periodic_phonon_samples,
    write_periodic_phonon_samples,
)


def _system() -> PeriodicDFTSystem:
    pseudo = PseudopotentialData(
        element="He",
        format=PseudopotentialFormat.GTH,
        valence_charge=2.0,
        gth_rloc=0.3,
        gth_coefficients=(-1.0,),
    )
    return PeriodicDFTSystem(
        (4.0, 4.0, 4.0),
        (4, 4, 4),
        ((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
        pseudo,
    )


def _cubic_symmetries() -> tuple[PeriodicPhononSymmetry, ...]:
    return tuple(
        PeriodicPhononSymmetry(operation, label=f"cubic-{index}")
        for index, operation in enumerate(cubic_reciprocal_symmetry_operations())
    )


def _harmonic_matrix(force_constant: float = 0.8) -> np.ndarray:
    identity = np.eye(3)
    return force_constant * np.block([[identity, -identity], [-identity, identity]])


def _samples(
    plan,
    force_constants,
    *,
    cubic_coefficient: float = 0.0,
    representatives: tuple[int, ...] | None = None,
) -> PeriodicPhononSampleSet:
    selected = plan.representative_dofs if representatives is None else representatives
    result = PeriodicPhononSampleSet.empty(plan)
    for dof in selected:
        displacement = np.zeros(plan.dof_count)
        displacement[dof] = plan.displacement_bohr

        def forces(values):
            linear = -(force_constants @ values)
            if cubic_coefficient:
                relative = values[:3] - values[3:]
                nonlinear = cubic_coefficient * relative**3
                linear[:3] -= nonlinear
                linear[3:] += nonlinear
            return linear.reshape(plan.atom_count, 3)

        result = result.with_sample(
            PeriodicPhononSample(
                representative_dof=dof,
                minus_forces_hartree_per_bohr=forces(-displacement),
                plus_forces_hartree_per_bohr=forces(displacement),
                minus_calculation_fingerprint=sha256_bytes(f"minus-{dof}".encode()),
                plus_calculation_fingerprint=sha256_bytes(f"plus-{dof}".encode()),
            )
        )
    return result


def test_cubic_symmetry_reduces_displacements_and_reconstructs_force_constants():
    system = _system()
    config = PeriodicPhononConfig(displacement_bohr=0.01)
    plan = plan_periodic_phonon_displacements(
        system,
        config=config,
        symmetry_operations=_cubic_symmetries(),
    )
    expected = _harmonic_matrix()
    samples = _samples(plan, expected)

    result = assemble_periodic_phonons(plan, samples, (4.0, 8.0), config=config)

    assert plan.representative_dofs == (0, 3)
    assert len(plan.symmetry_operations) == 48
    np.testing.assert_allclose(
        result.force_constants_hartree_per_bohr2,
        expected,
        atol=2.0e-14,
    )
    assert result.force_constants_passed is True
    assert result.acoustic_passed is True
    assert result.stable is True
    assert result.valid is True


def test_periodic_phonon_modes_have_expected_acoustic_and_optical_frequencies():
    system = _system()
    config = PeriodicPhononConfig(displacement_bohr=0.01)
    plan = plan_periodic_phonon_displacements(system, config=config)
    force_constant = 0.8
    masses = np.asarray((4.0, 8.0))
    result = assemble_periodic_phonons(
        plan,
        _samples(plan, _harmonic_matrix(force_constant)),
        masses,
        config=config,
    )

    expected_optical = (
        np.sqrt(
            force_constant
            * (1.0 / masses[0] + 1.0 / masses[1])
            / ATOMIC_MASS_UNIT_TO_ELECTRON_MASS
        )
        * HARTREE_TO_WAVENUMBER_CM1
    )
    assert result.frequencies_cm1 is not None
    np.testing.assert_allclose(result.frequencies_cm1[:3], 0.0, atol=1.0e-4)
    np.testing.assert_allclose(
        result.frequencies_cm1[3:],
        expected_optical,
        atol=1.0e-9,
    )
    assert result.acoustic_translation_overlap_minimum == pytest.approx(1.0)
    np.testing.assert_allclose(
        result.mass_weighted_eigenvectors.T @ result.mass_weighted_eigenvectors,
        np.eye(6),
        atol=2.0e-15,
    )


def test_periodic_phonons_reject_incomplete_samples_and_diagnose_sum_rule_failure():
    system = _system()
    config = PeriodicPhononConfig(displacement_bohr=0.01)
    plan = plan_periodic_phonon_displacements(system, config=config)
    expected = _harmonic_matrix()
    partial = _samples(plan, expected, representatives=(0,))

    with pytest.raises(ValueError, match="incomplete"):
        assemble_periodic_phonons(plan, partial, (4.0, 8.0), config=config)

    complete = _samples(plan, expected)
    first = complete.samples[0]
    corrupted_plus = np.array(first.plus_forces_hartree_per_bohr, copy=True)
    corrupted_plus[0, 0] += 1.0e-3
    corrupted = PeriodicPhononSample(
        representative_dof=first.representative_dof,
        minus_forces_hartree_per_bohr=first.minus_forces_hartree_per_bohr,
        plus_forces_hartree_per_bohr=corrupted_plus,
        minus_calculation_fingerprint=first.minus_calculation_fingerprint,
        plus_calculation_fingerprint=first.plus_calculation_fingerprint,
    )
    broken = PeriodicPhononSampleSet(
        complete.plan_fingerprint,
        complete.atom_count,
        (corrupted, *complete.samples[1:]),
    )
    result = assemble_periodic_phonons(plan, broken, (4.0, 8.0), config=config)

    assert result.force_constants_passed is False
    assert result.frequencies_cm1 is None
    assert result.valid is False
    assert result.right_sum_rule_residual_hartree_per_bohr2 > (
        config.sum_rule_tolerance_hartree_per_bohr2
    )


def test_periodic_phonon_displacement_convergence_uses_two_distinct_steps():
    system = _system()
    coarse_config = PeriodicPhononConfig(displacement_bohr=0.02)
    fine_config = PeriodicPhononConfig(displacement_bohr=0.01)
    force_constants = _harmonic_matrix()
    coarse_plan = plan_periodic_phonon_displacements(system, config=coarse_config)
    fine_plan = plan_periodic_phonon_displacements(system, config=fine_config)
    coarse = assemble_periodic_phonons(
        coarse_plan,
        _samples(coarse_plan, force_constants, cubic_coefficient=0.1),
        (4.0, 8.0),
        config=coarse_config,
    )
    fine = assemble_periodic_phonons(
        fine_plan,
        _samples(fine_plan, force_constants, cubic_coefficient=0.1),
        (4.0, 8.0),
        config=fine_config,
    )
    comparison = compare_periodic_phonon_displacements(
        coarse,
        fine,
        config=PeriodicPhononConfig(
            frequency_convergence_tolerance_cm1=5.0,
            eigenvalue_convergence_tolerance_au=1.0e-8,
        ),
    )

    assert comparison.maximum_frequency_drift_cm1 > 0.0
    assert comparison.passed is True
    with pytest.raises(ValueError, match="exceed"):
        compare_periodic_phonon_displacements(fine, coarse)


def test_periodic_phonon_displaced_system_preserves_cell_and_exact_step():
    system = _system()
    plan = plan_periodic_phonon_displacements(system)
    displaced = periodic_phonon_displaced_system(system, plan, 0, 1)

    np.testing.assert_array_equal(displaced.grid.cell.matrix, system.grid.cell.matrix)
    assert displaced.positions[0, 0] == pytest.approx(
        system.positions[0, 0] + plan.displacement_bohr
    )
    np.testing.assert_array_equal(displaced.positions[1], system.positions[1])


def test_periodic_phonon_sample_evaluator_runs_exactly_two_displaced_scfs(monkeypatch):
    import mlx_atomistic.dft.periodic_phonons as module

    system = _system()
    plan = plan_periodic_phonon_displacements(system)
    calls = []

    def run_scf(displaced, **_kwargs):
        calls.append(np.array(displaced.positions, copy=True))
        return SimpleNamespace(converged=True)

    def forces(displaced, _result):
        displacement = np.asarray(displaced.positions) - np.asarray(system.positions)
        return SimpleNamespace(forces=-displacement)

    monkeypatch.setattr(module, "run_periodic_scf", run_scf)
    monkeypatch.setattr(module, "periodic_scf_forces", forces)
    sample = evaluate_periodic_phonon_sample(
        system,
        plan,
        0,
        cutoff_hartree=4.0,
        kpoint_mesh=KPointMesh(
            (KPoint((0.0, 0.0, 0.0), coordinate_system="reduced"),)
        ),
    )

    assert len(calls) == 2
    assert calls[0][0, 0] == pytest.approx(-plan.displacement_bohr)
    assert calls[1][0, 0] == pytest.approx(plan.displacement_bohr)
    assert sample.minus_calculation_fingerprint != sample.plus_calculation_fingerprint
    assert sample.minus_forces_hartree_per_bohr[0, 0] == pytest.approx(
        plan.displacement_bohr
    )


def test_periodic_phonon_samples_round_trip_partial_restart_and_full_result(tmp_path):
    system = _system()
    plan = plan_periodic_phonon_displacements(system)
    expected = _harmonic_matrix()
    partial = _samples(plan, expected, representatives=(0, 1))
    path = tmp_path / "partial.npz"

    write_periodic_phonon_samples(path, plan, partial)
    resumed = read_periodic_phonon_samples(path, plan)

    assert resumed.missing_representatives(plan) == (2, 3, 4, 5)
    for dof in resumed.missing_representatives(plan):
        sample = next(
            item
            for item in _samples(plan, expected).samples
            if item.representative_dof == dof
        )
        resumed = resumed.with_sample(sample)
    uninterrupted = _samples(plan, expected)
    restarted_result = assemble_periodic_phonons(plan, resumed, (4.0, 8.0))
    uninterrupted_result = assemble_periodic_phonons(
        plan,
        uninterrupted,
        (4.0, 8.0),
    )

    np.testing.assert_array_equal(
        restarted_result.force_constants_hartree_per_bohr2,
        uninterrupted_result.force_constants_hartree_per_bohr2,
    )
    np.testing.assert_array_equal(
        restarted_result.frequencies_cm1,
        uninterrupted_result.frequencies_cm1,
    )
    with pytest.raises(FileExistsError):
        write_periodic_phonon_samples(path, plan, partial)


def test_periodic_phonon_samples_reject_malformed_no_pickle_archive(tmp_path):
    system = _system()
    plan = plan_periodic_phonon_displacements(system)
    path = tmp_path / "malformed.npz"
    metadata = {
        "schema_version": PERIODIC_PHONON_SAMPLES_SCHEMA,
        "plan_fingerprint": plan.fingerprint,
        "system_fingerprint": plan.system_fingerprint,
        "atom_count": plan.atom_count,
        "force_unit": "hartree/bohr",
        "samples": [],
    }
    np.savez_compressed(
        path,
        metadata_json=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
        representative_dofs=np.asarray([], dtype=np.int32),
        minus_forces_hartree_per_bohr=np.empty((0, 2, 3), dtype=np.float64),
        plus_forces_hartree_per_bohr=np.empty((0, 2, 3), dtype=np.float64),
    )

    with pytest.raises(ValueError, match="representative array"):
        read_periodic_phonon_samples(path, plan)
