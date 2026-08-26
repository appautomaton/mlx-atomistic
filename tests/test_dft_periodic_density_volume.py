from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.core import Cell
from mlx_atomistic.dft import (
    PERIODIC_DENSITY_VOLUME_SCHEMA,
    PeriodicDFTSystem,
    PeriodicSCFResult,
    PseudopotentialData,
    PseudopotentialFormat,
    periodic_density_volume,
    read_periodic_density_volume,
    write_periodic_density_volume,
)


def _system() -> PeriodicDFTSystem:
    pseudo = PseudopotentialData(
        element="He",
        format=PseudopotentialFormat.GTH,
        valence_charge=2.0,
        gth_rloc=0.3,
        gth_coefficients=(-1.0,),
    )
    cell = Cell.triclinic(
        (
            (6.0, 0.0, 0.0),
            (0.8, 5.5, 0.0),
            (0.3, 0.6, 6.2),
        )
    )
    fractional = np.asarray(((0.2, 0.3, 0.4),), dtype=np.float64)
    return PeriodicDFTSystem(
        cell,
        (4, 5, 6),
        fractional @ np.asarray(cell.matrix),
        pseudo,
    )


def _result(
    system: PeriodicDFTSystem,
    *,
    converged: bool = True,
    system_fingerprint: str | None = None,
    magnetization: float | None = None,
) -> PeriodicSCFResult:
    density = mx.full(
        system.grid.shape,
        system.electron_count / system.grid.volume,
        dtype=mx.float32,
    )
    magnetization_density = (
        None
        if magnetization is None
        else mx.full(
            system.grid.shape,
            magnetization / system.grid.volume,
            dtype=mx.float32,
        )
    )
    return PeriodicSCFResult(
        converged=converged,
        status="converged" if converged else "max_iterations",
        iterations=2,
        total_energy=-1.0,
        electron_count=system.electron_count,
        density_residual=1.0e-7,
        energy_delta=1.0e-8,
        density=density,
        kpoints=(),
        energy_by_term={"total": -1.0},
        history=(),
        timings={},
        system_fingerprint=(
            system.fingerprint
            if system_fingerprint is None
            else system_fingerprint
        ),
        integrated_magnetization=magnetization,
        magnetization_density=magnetization_density,
    )


def test_periodic_density_volume_round_trips_full_rank_geometry(tmp_path):
    system = _system()
    source = _result(system)
    path = tmp_path / "density.npz"

    published = write_periodic_density_volume(path, system, source)
    loaded = read_periodic_density_volume(published)

    assert loaded.grid_shape == system.grid.shape
    assert loaded.symbols == ("He",)
    assert loaded.system_fingerprint == system.fingerprint
    assert loaded.electron_count == pytest.approx(2.0)
    assert loaded.integrated_magnetization is None
    np.testing.assert_array_equal(loaded.cell_matrix_bohr, system.grid.cell.matrix)
    np.testing.assert_array_equal(loaded.positions_bohr, system.positions)
    assert not loaded.charge_density_electron_per_bohr3.flags.writeable
    assert loaded.to_dict()["schema_version"] == PERIODIC_DENSITY_VOLUME_SCHEMA
    with pytest.raises(FileExistsError):
        write_periodic_density_volume(path, system, source)


def test_periodic_density_volume_round_trips_spin_magnetization(tmp_path):
    system = _system()
    source = _result(system, magnetization=0.5)

    loaded = read_periodic_density_volume(
        write_periodic_density_volume(tmp_path / "spin-density.npz", system, source)
    )

    assert loaded.integrated_magnetization == pytest.approx(0.5)
    assert loaded.magnetization_density_electron_per_bohr3 is not None
    observed = (
        np.sum(
            loaded.magnetization_density_electron_per_bohr3,
            dtype=np.float64,
        )
        * loaded.cell_volume_bohr3
        / np.prod(loaded.grid_shape)
    )
    assert observed == pytest.approx(0.5, abs=1.0e-6)


def test_periodic_density_volume_rejects_unmatched_or_unconverged_source():
    system = _system()

    with pytest.raises(ValueError, match="converged"):
        periodic_density_volume(system, _result(system, converged=False))
    with pytest.raises(ValueError, match="fingerprint"):
        periodic_density_volume(system, _result(system, system_fingerprint="0" * 64))


def test_periodic_density_volume_rejects_malformed_no_pickle_archive(tmp_path):
    path = tmp_path / "malformed.npz"
    metadata = {
        "schema_version": PERIODIC_DENSITY_VOLUME_SCHEMA,
        "has_magnetization_density": False,
        "density_unit": "electron/bohr^3",
        "grid_order": "fractional-cell-axis-index",
    }
    np.savez_compressed(
        path,
        metadata_json=np.frombuffer(
            json.dumps(metadata).encode("utf-8"),
            dtype=np.uint8,
        ),
        cell_matrix_bohr=np.eye(3, dtype=np.float64),
        positions_bohr=np.zeros((1, 3), dtype=np.float64),
        charge_density_electron_per_bohr3=np.ones((2, 2, 2), dtype=np.float64),
    )

    with pytest.raises(ValueError, match="float32"):
        read_periodic_density_volume(path)
