from dataclasses import replace

import numpy as np
import pytest

from mlx_atomistic.artifacts import build_mlx_system_from_artifact, load_prepared_mlx_artifact
from mlx_atomistic.forcefields import NonbondedPotential, SoftCoreNonbondedPotential
from mlx_atomistic.nonbonded import EwaldReferenceConfig
from mlx_atomistic.prep.io import save_prepared_system, synthetic_prepared_system


def test_soft_core_nonbonded_is_finite_at_overlap():
    positions = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    term = NonbondedPotential(
        sigma=[1.0, 1.0],
        epsilon=[0.2, 0.2],
        charges=[1.0, -1.0],
        cutoff=None,
        lj_shift=False,
        lambda_lj=0.5,
        lambda_electrostatics=0.5,
    )

    energy, forces, derivatives = term.energy_forces_dlambda(positions)

    assert np.isfinite(float(np.asarray(energy)))
    assert np.all(np.isfinite(np.asarray(forces)))
    assert np.isfinite(float(np.asarray(derivatives["lambda_lj"])))
    assert np.isfinite(float(np.asarray(derivatives["lambda_electrostatics"])))


def test_soft_core_endpoint_matches_hard_core_nonbonded():
    positions = np.asarray([[0.0, 0.0, 0.0], [1.35, 0.2, 0.0]], dtype=np.float32)
    kwargs = {
        "sigma": [1.0, 1.2],
        "epsilon": [0.3, 0.4],
        "charges": [0.7, -0.5],
        "cutoff": None,
        "lj_shift": False,
    }
    hard = NonbondedPotential(**kwargs)
    soft_endpoint = NonbondedPotential(
        **kwargs,
        lambda_lj=1.0,
        lambda_electrostatics=1.0,
    )

    hard_energy, hard_forces = hard.energy_forces(positions)
    soft_energy, soft_forces = soft_endpoint.energy_forces(positions)

    np.testing.assert_allclose(np.asarray(soft_energy), np.asarray(hard_energy), atol=1e-7)
    np.testing.assert_allclose(np.asarray(soft_forces), np.asarray(hard_forces), atol=1e-7)


def test_soft_core_dlambda_matches_finite_difference():
    positions = np.asarray([[0.0, 0.0, 0.0], [1.4, 0.1, 0.0]], dtype=np.float32)
    kwargs = {
        "sigma": [1.0, 1.1],
        "epsilon": [0.25, 0.35],
        "charges": [0.6, -0.4],
        "cutoff": None,
        "lj_shift": False,
    }
    term = NonbondedPotential(
        **kwargs,
        lambda_lj=0.6,
        lambda_electrostatics=0.7,
    )
    _, _, derivatives = term.energy_forces_dlambda(positions)

    delta = 1e-3
    lj_plus = NonbondedPotential(
        **kwargs,
        lambda_lj=0.6 + delta,
        lambda_electrostatics=0.7,
    ).energy_forces(positions)[0]
    lj_minus = NonbondedPotential(
        **kwargs,
        lambda_lj=0.6 - delta,
        lambda_electrostatics=0.7,
    ).energy_forces(positions)[0]
    coulomb_plus = NonbondedPotential(
        **kwargs,
        lambda_lj=0.6,
        lambda_electrostatics=0.7 + delta,
    ).energy_forces(positions)[0]
    coulomb_minus = NonbondedPotential(
        **kwargs,
        lambda_lj=0.6,
        lambda_electrostatics=0.7 - delta,
    ).energy_forces(positions)[0]

    np.testing.assert_allclose(
        np.asarray(derivatives["lambda_lj"]),
        (np.asarray(lj_plus) - np.asarray(lj_minus)) / (2.0 * delta),
        atol=2e-4,
    )
    np.testing.assert_allclose(
        np.asarray(derivatives["lambda_electrostatics"]),
        (np.asarray(coulomb_plus) - np.asarray(coulomb_minus)) / (2.0 * delta),
        atol=2e-4,
    )


def test_soft_core_nonbonded_wrapper_delegates_lambda_derivatives():
    base = NonbondedPotential(
        sigma=[1.0, 1.0],
        epsilon=[0.2, 0.2],
        charges=[1.0, -1.0],
        cutoff=None,
        lj_shift=False,
    )
    wrapped = SoftCoreNonbondedPotential(base, lambda_lj=0.5, lambda_electrostatics=0.75)
    positions = np.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np.float32)

    energy, forces, derivatives = wrapped.energy_forces_dlambda(positions)

    assert wrapped.potential.lambda_lj == 0.5
    assert wrapped.potential.lambda_electrostatics == 0.75
    assert np.isfinite(float(np.asarray(energy)))
    assert np.all(np.isfinite(np.asarray(forces)))
    assert set(derivatives) == {"lambda_lj", "lambda_electrostatics"}


def test_default_component_energies_do_not_include_lambda_derivatives():
    term = NonbondedPotential(
        sigma=[1.0, 1.0],
        epsilon=[0.2, 0.2],
        charges=[1.0, -1.0],
        cutoff=None,
        lj_shift=False,
    )
    positions = np.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np.float32)

    _, _, components = term.energy_forces_with_components(positions)

    assert set(components) == {"lj", "coulomb"}


def test_artifact_accepts_lambda_scaled_nonbonded_term(tmp_path):
    prepared = synthetic_prepared_system()
    metadata = replace(
        prepared.metadata,
        compatibility_report={
            "production_force_field": False,
            "supported_terms": ["lambda_scaled_nonbonded"],
            "required_terms": ["lambda_scaled_nonbonded"],
            "unsupported_terms": [],
        },
    )
    prepared = replace(
        prepared,
        metadata=metadata,
        bonds=np.empty((0, 2), dtype=np.int32),
        bond_k=np.asarray([], dtype=np.float32),
        bond_length=np.asarray([], dtype=np.float32),
    )
    save_prepared_system(prepared, tmp_path)

    artifact = load_prepared_mlx_artifact(tmp_path, require_production=False)
    artifact.metadata["lambda_lj"] = 0.5
    artifact.metadata["lambda_electrostatics"] = 0.5
    _, terms, _ = build_mlx_system_from_artifact(artifact)

    assert isinstance(terms[-1], SoftCoreNonbondedPotential)
    assert terms[-1].potential.lambda_lj == 0.5
    assert terms[-1].potential.lambda_electrostatics == 0.5


def test_artifact_reads_soft_core_lj_term_scoped_lambda_metadata(tmp_path):
    prepared = synthetic_prepared_system()
    metadata = replace(
        prepared.metadata,
        compatibility_report={
            "production_force_field": False,
            "supported_terms": ["soft_core_lj"],
            "required_terms": ["soft_core_lj"],
            "unsupported_terms": [],
        },
    )
    prepared = replace(
        prepared,
        metadata=metadata,
        bonds=np.empty((0, 2), dtype=np.int32),
        bond_k=np.asarray([], dtype=np.float32),
        bond_length=np.asarray([], dtype=np.float32),
    )
    save_prepared_system(prepared, tmp_path)

    artifact = load_prepared_mlx_artifact(tmp_path, require_production=False)
    artifact.metadata["soft_core_lj"] = {
        "lambda_lj": 0.25,
        "lambda_electrostatics": 0.75,
    }
    _, terms, _ = build_mlx_system_from_artifact(artifact)

    assert isinstance(terms[-1], SoftCoreNonbondedPotential)
    assert terms[-1].potential.lambda_lj == 0.25
    assert terms[-1].potential.lambda_electrostatics == 0.75


def test_energy_forces_dlambda_fails_closed_for_non_cutoff_electrostatics():
    term = NonbondedPotential(
        sigma=[1.0, 1.0],
        epsilon=[0.0, 0.0],
        charges=[1.0, -1.0],
        cutoff=5.0,
        electrostatics="ewald_reference",
        ewald_config=EwaldReferenceConfig(alpha=0.25, real_cutoff=5.0, reciprocal_cutoff=1),
    )

    with pytest.raises(ValueError, match="dU/dlambda.*cutoff electrostatics"):
        term.energy_forces_dlambda(
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        )
