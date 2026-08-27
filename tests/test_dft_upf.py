from __future__ import annotations

import numpy as np
import pytest

import mlx_atomistic.dft.periodic_upf as periodic_upf
from mlx_atomistic.dft import (
    PeriodicDFTSystem,
    PlaneWaveBasis,
    PseudopotentialData,
    PseudopotentialFormat,
    RadialGrid,
    RealSpaceGrid,
    gth_local_reciprocal_coefficients,
    periodic_gth_local_forces,
    periodic_upf_local_forces,
    read_upf,
    upf_local_potential_grid,
    upf_local_reciprocal_coefficients,
)


def _upf_text(*, dij: str = "2.0 0.4 0.4 4.0", weights: str = "0.01 0.09 0.9") -> str:
    return f"""<UPF version="2.0.1">
<PP_HEADER element="Mg" z_valence="2" pseudo_type="NC"
 relativistic="scalar" is_ultrasoft="F" is_paw="F" has_so="F"
 core_correction="F" functional="PBE"/>
<PP_MESH>
  <PP_R>0.01 0.1 1.0</PP_R>
  <PP_RAB>{weights}</PP_RAB>
</PP_MESH>
<PP_LOCAL>-2.0 -1.0 -0.2</PP_LOCAL>
<PP_NONLOCAL>
  <PP_BETA.1 index="1" angular_momentum="0" cutoff_radius="0.8">
    0.0 0.2 0.0
  </PP_BETA.1>
  <PP_BETA.2 index="2" angular_momentum="1" cutoff_radius="0.9">
    0.0 0.3 0.0
  </PP_BETA.2>
  <PP_DIJ>{dij}</PP_DIJ>
</PP_NONLOCAL>
</UPF>
"""


def test_upf_parser_preserves_radial_quadrature_and_full_dij(tmp_path):
    source = tmp_path / "Mg.nc.UPF"
    source.write_text(_upf_text())

    pseudo = read_upf(source)

    assert pseudo.format == PseudopotentialFormat.UPF
    assert pseudo.periodic_upf_compatible is True
    np.testing.assert_allclose(
        pseudo.local_grid.integration_weights,
        (0.01, 0.09, 0.9),
    )
    assert pseudo.nonlocal_coupling_matrix == ((1.0, 0.2), (0.2, 2.0))
    assert pseudo.nonlocal_projectors[0].coefficients == (1.0, 0.2)
    assert pseudo.nonlocal_projectors[1].coefficients == (0.2, 2.0)
    assert pseudo.nonlocal_projectors[0].coupling == pytest.approx(1.0)
    assert pseudo.nonlocal_projectors[1].coupling == pytest.approx(2.0)
    assert all(
        projector.metadata["radial_representation"] == "r_beta"
        for projector in pseudo.nonlocal_projectors
    )


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ('pseudo_type="NC"', 'pseudo_type="US"'),
        ('relativistic="scalar"', 'relativistic="full"'),
        ('has_so="F"', 'has_so="T"'),
        ('core_correction="F"', 'core_correction="T"'),
    ),
)
def test_upf_periodic_boundary_rejects_unsupported_physics(tmp_path, old, new):
    source = tmp_path / "unsupported.UPF"
    source.write_text(_upf_text().replace(old, new))

    assert read_upf(source).periodic_upf_compatible is False


@pytest.mark.parametrize(
    ("dij", "message"),
    (
        ("2.0 0.4 4.0", "size must match"),
        ("2.0 0.3 0.4 4.0", "finite and symmetric"),
    ),
)
def test_upf_parser_rejects_invalid_dij(tmp_path, dij, message):
    source = tmp_path / "invalid-dij.UPF"
    source.write_text(_upf_text(dij=dij))

    with pytest.raises(ValueError, match=message):
        read_upf(source)


def test_upf_parser_rejects_mismatched_radial_quadrature(tmp_path):
    source = tmp_path / "invalid-grid.UPF"
    source.write_text(_upf_text(weights="0.01 0.99"))

    with pytest.raises(ValueError, match="sizes do not match"):
        read_upf(source)


def test_upf_parser_rejects_ambiguous_projector_order(tmp_path):
    source = tmp_path / "duplicate-projector.UPF"
    source.write_text(_upf_text().replace('PP_BETA.1 index="1"', 'PP_BETA.1 index="2"'))

    with pytest.raises(ValueError, match="unique and contiguous"):
        read_upf(source)


def _matched_gth_and_numerical_upf():
    gth = PseudopotentialData(
        element="Si",
        format=PseudopotentialFormat.GTH,
        valence_charge=4.0,
        gth_rloc=0.44,
        gth_coefficients=(-6.26928833,),
    )
    radii = np.linspace(0.0, 12.0, 6001, dtype=np.float64)
    spacing = float(radii[1] - radii[0])
    upf = PseudopotentialData(
        element="Si",
        format=PseudopotentialFormat.UPF,
        valence_charge=4.0,
        local_grid=RadialGrid(
            radii,
            gth.local_potential(radii),
            integration_weights=np.full(radii.shape, spacing),
        ),
    )
    return gth, upf


def test_periodic_upf_local_transform_matches_qe_gth_oracle():
    gth, upf = _matched_gth_and_numerical_upf()
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis(grid, 4.0)
    position = ((1.1, 2.2, 3.3),)

    expected = np.asarray(
        gth_local_reciprocal_coefficients(gth, basis, position)
    )
    observed = np.asarray(
        upf_local_reciprocal_coefficients(upf, basis, position)
    )
    potential = np.asarray(upf_local_potential_grid(upf, basis, position))

    np.testing.assert_allclose(observed, expected, rtol=2.0e-5, atol=2.0e-7)
    assert np.isfinite(potential).all()


def test_periodic_upf_local_force_matches_qe_gth_oracle(monkeypatch):
    gth, upf = _matched_gth_and_numerical_upf()
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis(grid, 4.0)
    positions = ((1.1, 2.2, 3.3), (5.0, 4.0, 2.0))
    coordinates = np.asarray(grid.coordinates())
    density = (
        0.02
        + 0.004 * np.cos(2.0 * np.pi * coordinates[..., 0] / 8.0)
        + 0.003 * np.sin(2.0 * np.pi * coordinates[..., 1] / 8.0)
    ).astype(np.float32)
    transform = periodic_upf._upf_local_radial_transform
    transform_calls = 0

    def observed_transform(*args, **kwargs):
        nonlocal transform_calls
        transform_calls += 1
        return transform(*args, **kwargs)

    monkeypatch.setattr(
        periodic_upf,
        "_upf_local_radial_transform",
        observed_transform,
    )

    expected = np.asarray(
        periodic_gth_local_forces(density, gth, basis, positions)
    )
    observed = np.asarray(
        periodic_upf_local_forces(density, upf, basis, positions)
    )

    np.testing.assert_allclose(observed, expected, rtol=3.0e-5, atol=2.0e-7)
    assert transform_calls == 1


def test_periodic_upf_identity_is_content_bound_and_path_independent(tmp_path):
    first_path = tmp_path / "first.UPF"
    second_path = tmp_path / "nested" / "second.UPF"
    changed_path = tmp_path / "changed.UPF"
    second_path.parent.mkdir()
    first_path.write_text(_upf_text())
    second_path.write_text(_upf_text())
    changed_path.write_text(_upf_text(dij="2.0 0.6 0.6 4.0"))

    def system(path):
        return PeriodicDFTSystem(
            (8.0, 8.0, 8.0),
            (8, 8, 8),
            ((1.0, 2.0, 3.0),),
            read_upf(path),
        )

    assert system(first_path).fingerprint == system(second_path).fingerprint
    assert system(first_path).fingerprint != system(changed_path).fingerprint
