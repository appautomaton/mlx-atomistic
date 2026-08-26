"""Portable periodic charge and magnetization density volumes."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlx_atomistic._artifact_identity import canonical_json_bytes
from mlx_atomistic.dft._periodic_models import PeriodicDFTSystem, PeriodicSCFResult

PERIODIC_DENSITY_VOLUME_SCHEMA = "mlx-atomistic.periodic-density-volume.v1"


@dataclass(frozen=True)
class PeriodicDensityVolume:
    """Validated periodic scalar fields and their full-rank geometry."""

    cell_matrix_bohr: np.ndarray
    positions_bohr: np.ndarray
    symbols: tuple[str, ...]
    charge_density_electron_per_bohr3: np.ndarray
    magnetization_density_electron_per_bohr3: np.ndarray | None
    electron_count: float
    integrated_magnetization: float | None
    system_fingerprint: str

    def __post_init__(self) -> None:
        cell = np.array(self.cell_matrix_bohr, dtype=np.float64, copy=True)
        positions = np.array(self.positions_bohr, dtype=np.float64, copy=True)
        charge = np.array(
            self.charge_density_electron_per_bohr3,
            dtype=np.float32,
            copy=True,
        )
        magnetization = (
            None
            if self.magnetization_density_electron_per_bohr3 is None
            else np.array(
                self.magnetization_density_electron_per_bohr3,
                dtype=np.float32,
                copy=True,
            )
        )
        symbols = tuple(str(symbol) for symbol in self.symbols)
        if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
            raise ValueError("density volume cell must be a finite 3 x 3 matrix")
        volume = float(np.linalg.det(cell))
        if not np.isfinite(volume) or volume <= 0.0:
            raise ValueError("density volume cell must be right-handed and full-rank")
        if (
            positions.shape != (len(symbols), 3)
            or not symbols
            or any(not symbol for symbol in symbols)
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError("density volume positions and symbols are invalid")
        if (
            charge.ndim != 3
            or any(size <= 0 for size in charge.shape)
            or not np.all(np.isfinite(charge))
            or float(np.min(charge)) < -1.0e-7
        ):
            raise ValueError("charge density must be a finite non-negative 3D field")
        if magnetization is not None and (
            magnetization.shape != charge.shape
            or not np.all(np.isfinite(magnetization))
        ):
            raise ValueError("magnetization density must match the charge field")
        if not np.isfinite(self.electron_count) or self.electron_count <= 0.0:
            raise ValueError("density volume electron count must be finite and positive")
        dv = volume / float(charge.size)
        observed_electrons = float(np.sum(charge, dtype=np.float64) * dv)
        if abs(observed_electrons - self.electron_count) > 1.0e-4:
            raise ValueError("charge density normalization differs from electron count")
        if magnetization is None:
            if self.integrated_magnetization is not None:
                raise ValueError("scalar density volume cannot declare magnetization")
        else:
            observed_magnetization = float(
                np.sum(magnetization, dtype=np.float64) * dv
            )
            if (
                self.integrated_magnetization is None
                or not np.isfinite(self.integrated_magnetization)
                or abs(observed_magnetization - self.integrated_magnetization) > 1.0e-4
            ):
                raise ValueError("magnetization density normalization is inconsistent")
        if len(self.system_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.system_fingerprint
        ):
            raise ValueError("density volume system fingerprint must be lowercase SHA-256")
        for values in (cell, positions, charge):
            values.setflags(write=False)
        if magnetization is not None:
            magnetization.setflags(write=False)
        object.__setattr__(self, "cell_matrix_bohr", cell)
        object.__setattr__(self, "positions_bohr", positions)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "charge_density_electron_per_bohr3", charge)
        object.__setattr__(
            self,
            "magnetization_density_electron_per_bohr3",
            magnetization,
        )

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """Return the stored fractional-grid shape."""

        return tuple(int(value) for value in self.charge_density_electron_per_bohr3.shape)

    @property
    def cell_volume_bohr3(self) -> float:
        """Return the right-handed cell volume in bohr cubed."""

        return float(np.linalg.det(self.cell_matrix_bohr))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe volume summary without field arrays."""

        return {
            "schema_version": PERIODIC_DENSITY_VOLUME_SCHEMA,
            "grid_shape": list(self.grid_shape),
            "cell_matrix_bohr": self.cell_matrix_bohr.tolist(),
            "cell_volume_bohr3": self.cell_volume_bohr3,
            "symbols": list(self.symbols),
            "positions_bohr": self.positions_bohr.tolist(),
            "electron_count": self.electron_count,
            "integrated_magnetization": self.integrated_magnetization,
            "has_magnetization_density": (
                self.magnetization_density_electron_per_bohr3 is not None
            ),
            "density_unit": "electron/bohr^3",
            "grid_order": "fractional-cell-axis-index",
            "system_fingerprint": self.system_fingerprint,
        }


def periodic_density_volume(
    system: PeriodicDFTSystem,
    source: PeriodicSCFResult,
) -> PeriodicDensityVolume:
    """Build a portable density volume from a matching converged periodic SCF.

    Args:
        system: Periodic system that produced ``source``.
        source: Converged scalar or collinear-spin SCF result.

    Returns:
        Validated immutable charge and optional magnetization volume.

    Raises:
        TypeError: If inputs use unsupported types.
        ValueError: If convergence, fingerprint, shape, or normalization differ.
    """

    if not isinstance(system, PeriodicDFTSystem):
        raise TypeError("system must be PeriodicDFTSystem")
    if not isinstance(source, PeriodicSCFResult):
        raise TypeError("source must be PeriodicSCFResult")
    if not source.converged:
        raise ValueError("density volume requires a converged SCF result")
    if source.system_fingerprint != system.fingerprint:
        raise ValueError("density volume system fingerprint differs from SCF source")
    if not np.isclose(
        source.electron_count,
        system.electron_count,
        atol=1.0e-8,
        rtol=0.0,
    ):
        raise ValueError("density volume electron count differs from periodic system")
    charge = np.asarray(source.density, dtype=np.float32)
    if charge.shape != system.grid.shape:
        raise ValueError("SCF density shape differs from the periodic system grid")
    magnetization = (
        None
        if source.magnetization_density is None
        else np.asarray(source.magnetization_density, dtype=np.float32)
    )
    return PeriodicDensityVolume(
        cell_matrix_bohr=np.asarray(system.grid.cell.matrix, dtype=np.float64),
        positions_bohr=system.positions,
        symbols=system.symbols,
        charge_density_electron_per_bohr3=charge,
        magnetization_density_electron_per_bohr3=magnetization,
        electron_count=source.electron_count,
        integrated_magnetization=(
            None if magnetization is None else source.integrated_magnetization
        ),
        system_fingerprint=system.fingerprint,
    )


def write_periodic_density_volume(
    path: str | Path,
    system: PeriodicDFTSystem,
    source: PeriodicSCFResult,
) -> Path:
    """Atomically publish a portable no-pickle periodic density volume.

    Args:
        path: Previously absent output NPZ path.
        system: Periodic system that produced ``source``.
        source: Converged scalar or collinear-spin SCF result.

    Returns:
        Resolved published path.

    Raises:
        FileExistsError: If the destination already exists.
        ValueError: If the source cannot form a valid density volume.
    """

    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    volume = periodic_density_volume(system, source)
    metadata = canonical_json_bytes(volume.to_dict())
    payloads = {
        "metadata_json": np.frombuffer(metadata, dtype=np.uint8),
        "cell_matrix_bohr": volume.cell_matrix_bohr,
        "positions_bohr": volume.positions_bohr,
        "charge_density_electron_per_bohr3": (
            volume.charge_density_electron_per_bohr3
        ),
    }
    if volume.magnetization_density_electron_per_bohr3 is not None:
        payloads["magnetization_density_electron_per_bohr3"] = (
            volume.magnetization_density_electron_per_bohr3
        )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.tmp-",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, **payloads)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def read_periodic_density_volume(path: str | Path) -> PeriodicDensityVolume:
    """Load and validate a portable periodic density volume without pickle.

    Args:
        path: Existing density-volume NPZ file.

    Returns:
        Validated immutable periodic density volume.

    Raises:
        ValueError: If schema, inventory, metadata, arrays, or normalization
            are invalid.
    """

    source = Path(path).expanduser().resolve()
    try:
        with np.load(source, allow_pickle=False) as archive:
            names = set(archive.files)
            if "metadata_json" not in names:
                raise ValueError("density volume metadata is missing")
            metadata_array = archive["metadata_json"]
            if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
                raise ValueError("density volume metadata array is invalid")
            metadata_bytes = metadata_array.tobytes()
            metadata = json.loads(metadata_bytes.decode("utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("density volume metadata must be a JSON object")
            has_magnetization = metadata.get("has_magnetization_density")
            if type(has_magnetization) is not bool:
                raise ValueError("density volume magnetization flag is invalid")
            expected_names = {
                "metadata_json",
                "cell_matrix_bohr",
                "positions_bohr",
                "charge_density_electron_per_bohr3",
            }
            if has_magnetization:
                expected_names.add("magnetization_density_electron_per_bohr3")
            if names != expected_names:
                raise ValueError("density volume array inventory is inconsistent")
            if metadata.get("schema_version") != PERIODIC_DENSITY_VOLUME_SCHEMA:
                raise ValueError("unsupported periodic density volume schema")
            if metadata.get("density_unit") != "electron/bohr^3" or metadata.get(
                "grid_order"
            ) != "fractional-cell-axis-index":
                raise ValueError("density volume units or grid order are unsupported")
            cell = archive["cell_matrix_bohr"]
            positions = archive["positions_bohr"]
            charge = archive["charge_density_electron_per_bohr3"]
            magnetization = (
                archive["magnetization_density_electron_per_bohr3"]
                if has_magnetization
                else None
            )
            if cell.dtype != np.float64 or positions.dtype != np.float64:
                raise ValueError("density volume geometry arrays must use float64")
            if charge.dtype != np.float32 or (
                magnetization is not None and magnetization.dtype != np.float32
            ):
                raise ValueError("density volume field arrays must use float32")
            volume = PeriodicDensityVolume(
                cell_matrix_bohr=cell,
                positions_bohr=positions,
                symbols=tuple(metadata.get("symbols", ())),
                charge_density_electron_per_bohr3=charge,
                magnetization_density_electron_per_bohr3=magnetization,
                electron_count=float(metadata["electron_count"]),
                integrated_magnetization=(
                    None
                    if metadata.get("integrated_magnetization") is None
                    else float(metadata["integrated_magnetization"])
                ),
                system_fingerprint=str(metadata["system_fingerprint"]),
            )
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("periodic density volume metadata is invalid") from error
    if volume.to_dict() != metadata:
        raise ValueError("density volume metadata differs from decoded arrays")
    return volume
