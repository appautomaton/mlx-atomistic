"""Pseudopotential and ion models for the DFT prototype."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from math import erf, pi, sqrt
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import as_mx_array
from mlx_atomistic.dft.grids import RealSpaceGrid


class PseudopotentialFormat(StrEnum):
    """Supported pseudopotential file families."""

    UPF = "upf"
    GTH = "gth"


@dataclass(frozen=True)
class RadialGrid:
    """Radial samples for a local pseudopotential channel."""

    radii: np.ndarray
    values: np.ndarray

    def __init__(self, radii: Sequence[float], values: Sequence[float]):
        radii_np = np.asarray(radii, dtype=np.float64)
        values_np = np.asarray(values, dtype=np.float64)
        if radii_np.ndim != 1 or values_np.ndim != 1:
            msg = "radial grid radii and values must be one-dimensional"
            raise ValueError(msg)
        if radii_np.shape != values_np.shape:
            msg = "radial grid radii and values must have matching shapes"
            raise ValueError(msg)
        if len(radii_np) < 2:
            msg = "radial grid requires at least two samples"
            raise ValueError(msg)
        if np.any(np.diff(radii_np) <= 0.0):
            msg = "radial grid radii must be strictly increasing"
            raise ValueError(msg)
        object.__setattr__(self, "radii", radii_np)
        object.__setattr__(self, "values", values_np)

    @property
    def size(self) -> int:
        """Number of radial samples."""

        return int(self.radii.size)

    def interpolate(self, r: np.ndarray, *, tail_charge: float | None = None) -> np.ndarray:
        """Interpolate values at radius ``r`` with a Coulomb tail fallback."""

        radius = np.asarray(r, dtype=np.float64)
        values = np.interp(
            radius,
            self.radii,
            self.values,
            left=self.values[0],
            right=self.values[-1],
        )
        if tail_charge is not None:
            mask = radius > self.radii[-1]
            values = np.where(mask, -tail_charge / np.maximum(radius, 1e-12), values)
        return values

    def derivative(self, r: np.ndarray, *, tail_charge: float | None = None) -> np.ndarray:
        """Interpolate ``dV/dr`` at radius ``r``."""

        derivatives = np.gradient(self.values, self.radii, edge_order=1)
        radius = np.asarray(r, dtype=np.float64)
        values = np.interp(
            radius,
            self.radii,
            derivatives,
            left=derivatives[0],
            right=derivatives[-1],
        )
        if tail_charge is not None:
            mask = radius > self.radii[-1]
            values = np.where(mask, tail_charge / np.maximum(radius, 1e-12) ** 2, values)
        return values


@dataclass(frozen=True)
class NonlocalProjectorData:
    """Parsed nonlocal projector metadata for ion-aware operator application."""

    angular_momentum: int
    values: tuple[float, ...] = ()
    radial_grid: RadialGrid | None = None
    cutoff_radius: float | None = None
    coefficients: tuple[float, ...] = ()
    coupling: float = 0.0
    metadata: dict[str, str | int | float] | None = None


@dataclass(frozen=True)
class GTHProjectorChannel:
    """Complete GTH nonlocal channel with its symmetric coupling matrix."""

    angular_momentum: int
    radius: float
    coupling_matrix: tuple[tuple[float, ...], ...]

    def __init__(
        self,
        angular_momentum: int,
        radius: float,
        coupling_matrix: Sequence[Sequence[float]],
    ):
        if angular_momentum < 0:
            msg = "GTH angular momentum must be non-negative"
            raise ValueError(msg)
        if radius <= 0.0:
            msg = "GTH projector radius must be positive"
            raise ValueError(msg)
        matrix = np.asarray(coupling_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
            msg = "GTH coupling matrix must be non-empty and square"
            raise ValueError(msg)
        if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
            msg = "GTH coupling matrix must be symmetric"
            raise ValueError(msg)
        object.__setattr__(self, "angular_momentum", int(angular_momentum))
        object.__setattr__(self, "radius", float(radius))
        object.__setattr__(
            self,
            "coupling_matrix",
            tuple(tuple(float(value) for value in row) for row in matrix),
        )

    @property
    def projector_count(self) -> int:
        """Number of radial projectors in this channel."""

        return len(self.coupling_matrix)


@dataclass(frozen=True)
class PseudopotentialData:
    """Parsed pseudopotential data used by ion-centered local fields."""

    element: str
    format: PseudopotentialFormat
    valence_charge: float
    local_grid: RadialGrid | None = None
    gth_rloc: float | None = None
    gth_coefficients: tuple[float, ...] = ()
    gth_channels: tuple[GTHProjectorChannel, ...] = ()
    nonlocal_projectors: tuple[NonlocalProjectorData, ...] = ()
    metadata: dict[str, str | int | float | bool] | None = None

    @property
    def nonlocal_available(self) -> bool:
        """Whether the source file contained nonlocal projector metadata."""

        return bool(self.gth_channels or self.nonlocal_projectors)

    def local_potential(self, radius: np.ndarray) -> np.ndarray:
        """Evaluate the local potential in Hartree on radii in bohr."""

        radius_np = np.asarray(radius, dtype=np.float64)
        if self.format == PseudopotentialFormat.GTH:
            return _gth_local_potential(
                radius_np,
                valence_charge=self.valence_charge,
                rloc=_required(self.gth_rloc, "gth_rloc"),
                coefficients=self.gth_coefficients,
            )
        if self.local_grid is None:
            msg = "UPF pseudopotential is missing a local radial grid"
            raise ValueError(msg)
        return self.local_grid.interpolate(radius_np, tail_charge=self.valence_charge)

    def local_derivative(self, radius: np.ndarray) -> np.ndarray:
        """Evaluate ``dV_local/dr`` in Hartree/bohr."""

        radius_np = np.asarray(radius, dtype=np.float64)
        if self.format == PseudopotentialFormat.GTH:
            return _gth_local_derivative(
                radius_np,
                valence_charge=self.valence_charge,
                rloc=_required(self.gth_rloc, "gth_rloc"),
                coefficients=self.gth_coefficients,
            )
        if self.local_grid is None:
            msg = "UPF pseudopotential is missing a local radial grid"
            raise ValueError(msg)
        return self.local_grid.derivative(radius_np, tail_charge=self.valence_charge)


@dataclass(frozen=True)
class Ion:
    """Ion center with a parsed pseudopotential."""

    symbol: str
    position: mx.array
    pseudopotential: PseudopotentialData

    def __init__(
        self,
        symbol: str,
        position: Sequence[float],
        pseudopotential: PseudopotentialData,
    ):
        if len(position) != 3:
            msg = "ion position must have three coordinates"
            raise ValueError(msg)
        if symbol != pseudopotential.element:
            msg = "ion symbol must match pseudopotential element"
            raise ValueError(msg)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "position", as_mx_array(position))
        object.__setattr__(self, "pseudopotential", pseudopotential)

    @property
    def charge(self) -> float:
        """Valence charge represented by this pseudopotential."""

        return self.pseudopotential.valence_charge


@dataclass(frozen=True)
class IonCollection:
    """Collection of ions and their pseudopotentials."""

    ions: tuple[Ion, ...]

    def __init__(self, ions: Sequence[Ion]):
        if not ions:
            msg = "IonCollection requires at least one ion"
            raise ValueError(msg)
        object.__setattr__(self, "ions", tuple(ions))

    @property
    def centers(self) -> mx.array:
        """Ion center coordinates."""

        return mx.stack([ion.position for ion in self.ions], axis=0)

    @property
    def charges(self) -> tuple[float, ...]:
        """Valence charges."""

        return tuple(float(ion.charge) for ion in self.ions)

    @property
    def symbols(self) -> tuple[str, ...]:
        """Ion symbols."""

        return tuple(ion.symbol for ion in self.ions)

    @property
    def valence_electron_count(self) -> float:
        """Neutral valence electron count."""

        return float(sum(self.charges))

    @property
    def nonlocal_available(self) -> bool:
        """Whether any ion has parsed nonlocal projectors."""

        return any(ion.pseudopotential.nonlocal_available for ion in self.ions)

    @property
    def formats(self) -> tuple[str, ...]:
        """Pseudopotential formats in ion order."""

        return tuple(str(ion.pseudopotential.format) for ion in self.ions)

    def with_positions(self, positions: Sequence[Sequence[float]]) -> IonCollection:
        """Return a copy with updated positions."""

        positions_np = np.asarray(positions, dtype=np.float64)
        if positions_np.shape != (len(self.ions), 3):
            msg = "positions must have shape (n_ions, 3)"
            raise ValueError(msg)
        return IonCollection(
            [
                Ion(ion.symbol, positions_np[index], ion.pseudopotential)
                for index, ion in enumerate(self.ions)
            ]
        )


@dataclass(frozen=True)
class LocalPseudopotentialField:
    """Real-space local potential generated by an `IonCollection`."""

    ions: IonCollection

    @property
    def centers(self) -> mx.array:
        """Ion center coordinates."""

        return self.ions.centers

    @property
    def nonlocal_available(self) -> bool:
        """Whether nonlocal metadata is present."""

        return self.ions.nonlocal_available

    def field(self, grid: RealSpaceGrid) -> mx.array:
        """Evaluate the total local ion potential on a real-space grid."""

        coordinates = np.array(grid.coordinates(), dtype=np.float64)
        potential = np.zeros(grid.shape, dtype=np.float64)
        for ion in self.ions.ions:
            center = np.array(ion.position, dtype=np.float64)
            displacement = np.array(grid.cell.minimum_image(coordinates - center), dtype=np.float64)
            radius = np.linalg.norm(displacement, axis=-1)
            potential += ion.pseudopotential.local_potential(radius)
        return mx.array(potential.astype(np.float32))

    __call__ = field

    def forces(self, density: mx.array, grid: RealSpaceGrid) -> mx.array:
        """Return local electron-ion forces from a density."""

        coordinates = np.array(grid.coordinates(), dtype=np.float64)
        rho = np.array(density, dtype=np.float64)
        forces = []
        for ion in self.ions.ions:
            center = np.array(ion.position, dtype=np.float64)
            displacement = np.array(grid.cell.minimum_image(coordinates - center), dtype=np.float64)
            radius = np.linalg.norm(displacement, axis=-1)
            derivative = ion.pseudopotential.local_derivative(radius)
            unit = np.divide(
                displacement,
                radius[..., None],
                out=np.zeros_like(displacement),
                where=radius[..., None] > 1e-12,
            )
            force = np.sum(rho[..., None] * derivative[..., None] * unit, axis=(0, 1, 2))
            forces.append(force * grid.dv)
        return mx.array(np.asarray(forces, dtype=np.float32))


def read_upf(path: str | Path) -> PseudopotentialData:
    """Read a UPF v2-style pseudopotential file."""

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as err:
        msg = f"failed to parse UPF file {path}"
        raise ValueError(msg) from err
    header = root.find("PP_HEADER")
    if header is None:
        msg = "UPF file is missing PP_HEADER"
        raise ValueError(msg)
    element = _required_str(header.attrib.get("element"), "UPF element")
    valence = float(_required_str(header.attrib.get("z_valence"), "UPF z_valence"))
    radii_node = root.find("./PP_MESH/PP_R")
    local_node = root.find("PP_LOCAL")
    if radii_node is None or local_node is None:
        msg = "UPF file is missing PP_R or PP_LOCAL"
        raise ValueError(msg)
    radii = _numbers(radii_node.text)
    # QE UPF local potentials are conventionally stored in Ry; convert to Hartree.
    local = 0.5 * _numbers(local_node.text)
    if radii.shape != local.shape:
        msg = "UPF PP_R and PP_LOCAL sizes do not match"
        raise ValueError(msg)
    radial_grid = RadialGrid(radii, local)
    projectors = []
    for node in root.iter():
        if not node.tag.startswith("PP_BETA"):
            continue
        angular = int(
            _required_str(node.attrib.get("angular_momentum"), "PP_BETA angular_momentum")
        )
        values = tuple(float(item) for item in _numbers(node.text))
        metadata = {
            "index": int(node.attrib.get("index", len(projectors) + 1)),
            "source": "upf",
        }
        cutoff = node.attrib.get("cutoff_radius")
        projectors.append(
            NonlocalProjectorData(
                angular_momentum=angular,
                values=values,
                radial_grid=radial_grid,
                cutoff_radius=None if cutoff is None else float(cutoff),
                metadata=metadata,
            )
        )
    dij_node = root.find("./PP_NONLOCAL/PP_DIJ")
    if dij_node is not None and projectors:
        dij = 0.5 * _numbers(dij_node.text)
        size = int(round(np.sqrt(float(dij.size))))
        if size * size == dij.size and size >= len(projectors):
            matrix = dij.reshape((size, size))
            projectors = [
                replace(
                    projector,
                    coupling=float(matrix[index, index]),
                    coefficients=(float(matrix[index, index]),),
                    metadata={
                        **({} if projector.metadata is None else projector.metadata),
                        "dij_diagonal_hartree": float(matrix[index, index]),
                    },
                )
                for index, projector in enumerate(projectors)
            ]
    metadata: dict[str, str | int | float | bool] = {
        "source_path": str(path),
        "version": root.attrib.get("version", ""),
        "pseudo_type": header.attrib.get("pseudo_type", ""),
        "functional": header.attrib.get("functional", ""),
        "nonlocal_applied": False,
    }
    return PseudopotentialData(
        element=element,
        format=PseudopotentialFormat.UPF,
        valence_charge=valence,
        local_grid=radial_grid,
        nonlocal_projectors=tuple(projectors),
        metadata=metadata,
    )


def read_gth(
    path_or_database: str | Path,
    *,
    element: str | None = None,
    name: str | None = None,
) -> PseudopotentialData:
    """Read a single GTH file or one entry from a CP2K-style GTH database."""

    path = Path(path_or_database)
    payload = _gth_payload(path.read_text().splitlines())
    if not payload:
        msg = "GTH source is empty"
        raise ValueError(msg)
    if payload[0].lower().startswith("goedecker"):
        parsed_element, parsed_name, valence, functional, rloc, coefficients, channels = (
            _parse_standalone_gth(path, payload, element=element)
        )
    else:
        parsed_element, parsed_name, valence, functional, rloc, coefficients, channels = (
            _parse_database_gth(payload, element=element, name=name)
        )
    projectors = tuple(
        NonlocalProjectorData(
            angular_momentum=channel.angular_momentum,
            cutoff_radius=channel.radius,
            coefficients=tuple(channel.coupling_matrix[index]),
            coupling=float(channel.coupling_matrix[index][index]),
            metadata={
                "source": "gth",
                "n_projectors": channel.projector_count,
                "projector_index": index,
            },
        )
        for channel in channels
        for index in range(channel.projector_count)
    )
    metadata: dict[str, str | int | float | bool] = {
        "source_path": str(path),
        "name": parsed_name,
        "functional": functional,
        "nonlocal_applied": False,
    }
    return PseudopotentialData(
        element=parsed_element,
        format=PseudopotentialFormat.GTH,
        valence_charge=valence,
        gth_rloc=rloc,
        gth_coefficients=coefficients,
        gth_channels=channels,
        nonlocal_projectors=projectors,
        metadata=metadata,
    )


def _gth_payload(lines: Sequence[str]) -> list[str]:
    return [
        line
        for raw in lines
        if (line := raw.split("#", 1)[0].strip())
    ]


def _upper_matrix(
    first_values: Sequence[float],
    remaining_lines: Sequence[str],
    count: int,
) -> tuple[tuple[float, ...], ...]:
    rows = [list(float(value) for value in first_values)]
    rows.extend(_leading_floats(line.split()) for line in remaining_lines)
    matrix = np.zeros((count, count), dtype=np.float64)
    for row_index, values in enumerate(rows):
        expected = count - row_index
        if len(values) != expected:
            msg = "GTH coupling matrix row length does not match projector count"
            raise ValueError(msg)
        matrix[row_index, row_index:] = values
        matrix[row_index:, row_index] = values
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _parse_channels(
    payload: Sequence[str],
    cursor: int,
    count: int,
    *,
    standalone: bool,
) -> tuple[tuple[GTHProjectorChannel, ...], int]:
    channels: list[GTHProjectorChannel] = []
    for angular in range(count):
        if cursor >= len(payload):
            msg = "GTH nonlocal projector block is incomplete"
            raise ValueError(msg)
        parts = payload[cursor].split()
        cursor += 1
        if len(parts) < 2:
            msg = "GTH nonlocal projector line is incomplete"
            raise ValueError(msg)
        radius = float(parts[0])
        projector_count = int(parts[1])
        if projector_count < 0:
            msg = "GTH projector count must be non-negative"
            raise ValueError(msg)
        if projector_count == 0:
            continue
        if len(parts) < 3:
            msg = "GTH nonlocal projector line is incomplete"
            raise ValueError(msg)
        remaining = payload[cursor : cursor + projector_count - 1]
        if len(remaining) != projector_count - 1:
            msg = "GTH coupling matrix is incomplete"
            raise ValueError(msg)
        matrix = _upper_matrix(_leading_floats(parts[2:]), remaining, projector_count)
        cursor += projector_count - 1
        channels.append(GTHProjectorChannel(angular, radius, matrix))
        if standalone and angular > 0:
            # Standalone QE GTH files carry an additional spin-orbit coupling
            # matrix after h_ij. The selected scalar-relativistic path records
            # no spin-orbit operator, but the rows still have to be consumed.
            cursor += projector_count
            if cursor > len(payload):
                msg = "GTH spin-orbit coupling matrix is incomplete"
                raise ValueError(msg)
    return tuple(channels), cursor


def _parse_standalone_gth(
    path: Path,
    payload: Sequence[str],
    *,
    element: str | None,
) -> tuple[str, str, float, str, float, tuple[float, ...], tuple[GTHProjectorChannel, ...]]:
    if len(payload) < 5:
        msg = "standalone GTH entry is incomplete"
        raise ValueError(msg)
    parsed_element = element or _element_from_filename(path)
    charge_line = _numbers(payload[1])
    if charge_line.size < 2:
        msg = "standalone GTH charge line is incomplete"
        raise ValueError(msg)
    valence = float(charge_line[1])
    metadata_parts = payload[2].split()
    if len(metadata_parts) < 2:
        msg = "standalone GTH metadata line is incomplete"
        raise ValueError(msg)
    functional = {
        1: "PZ",
        7: "PW",
        11: "PBE",
        18: "BLYP",
        -101130: "PBE",
    }.get(int(metadata_parts[1]), "unknown")
    local_parts = payload[3].split()
    local_count = int(local_parts[1])
    coefficients = tuple(_leading_floats(local_parts[2:]))
    if len(coefficients) != local_count:
        msg = "GTH local coefficient count does not match its declaration"
        raise ValueError(msg)
    channel_count = int(payload[4].split()[0])
    channels, _ = _parse_channels(payload, 5, channel_count, standalone=True)
    return (
        parsed_element,
        path.name,
        valence,
        functional,
        float(local_parts[0]),
        coefficients,
        channels,
    )


def _parse_database_gth(
    payload: Sequence[str],
    *,
    element: str | None,
    name: str | None,
) -> tuple[str, str, float, str, float, tuple[float, ...], tuple[GTHProjectorChannel, ...]]:
    if element is None and name is None:
        msg = "element or name is required when reading a GTH database"
        raise ValueError(msg)
    for index, line in enumerate(payload):
        parts = line.split()
        if not parts:
            continue
        if element is not None and parts[0] != element:
            continue
        if name is not None and name not in parts[1:]:
            continue
        cursor = index + 1
        charge_shells = tuple(int(value) for value in payload[cursor].split())
        cursor += 1
        local_parts = payload[cursor].split()
        cursor += 1
        local_count = int(local_parts[1])
        coefficients = tuple(float(value) for value in local_parts[2:])
        if len(coefficients) != local_count:
            msg = "GTH local coefficient count does not match its declaration"
            raise ValueError(msg)
        channel_count = int(payload[cursor].split()[0])
        cursor += 1
        channels, _ = _parse_channels(payload, cursor, channel_count, standalone=False)
        selected_name = name or (parts[1] if len(parts) > 1 else "unknown")
        upper_name = selected_name.upper()
        functional = (
            "PBE"
            if "PBE" in upper_name
            else "BLYP"
            if "BLYP" in upper_name
            else "PW"
            if "PADE" in upper_name or "LDA" in upper_name
            else "unknown"
        )
        return (
            parts[0],
            selected_name,
            float(sum(charge_shells)),
            functional,
            float(local_parts[0]),
            coefficients,
            channels,
        )
    msg = "requested GTH entry was not found"
    raise ValueError(msg)


def _gth_local_potential(
    radius: np.ndarray,
    *,
    valence_charge: float,
    rloc: float,
    coefficients: Sequence[float],
) -> np.ndarray:
    radius_np = np.asarray(radius, dtype=np.float64)
    x = radius_np / rloc
    erf_term = np.empty_like(radius_np)
    small = radius_np < 1e-10
    erf_term[small] = -valence_charge * sqrt(2.0 / pi) / rloc
    erf_term[~small] = (
        -valence_charge * np.vectorize(erf)(x[~small] / sqrt(2.0)) / radius_np[~small]
    )
    polynomial = np.zeros_like(radius_np)
    for power, coefficient in enumerate(coefficients):
        polynomial += float(coefficient) * x ** (2 * power)
    return erf_term + np.exp(-0.5 * x * x) * polynomial


def _gth_local_derivative(
    radius: np.ndarray,
    *,
    valence_charge: float,
    rloc: float,
    coefficients: Sequence[float],
) -> np.ndarray:
    radius_np = np.asarray(radius, dtype=np.float64)
    x = radius_np / rloc
    derivative = np.zeros_like(radius_np)
    small = radius_np < 1e-8
    large = ~small
    if np.any(large):
        r = radius_np[large]
        xl = x[large]
        erf_values = np.vectorize(erf)(xl / sqrt(2.0))
        gaussian = np.exp(-0.5 * xl * xl)
        coulomb = -valence_charge * (
            sqrt(2.0 / pi) * gaussian * r / rloc - erf_values
        ) / (r * r)
        polynomial = np.zeros_like(r)
        dpolynomial_dx = np.zeros_like(r)
        for power, coefficient in enumerate(coefficients):
            polynomial += float(coefficient) * xl ** (2 * power)
            if power > 0:
                dpolynomial_dx += float(coefficient) * (2 * power) * xl ** (2 * power - 1)
        local = gaussian * (dpolynomial_dx - xl * polynomial) / rloc
        derivative[large] = coulomb + local
    return derivative


def _numbers(text: str | None) -> np.ndarray:
    if text is None:
        return np.array([], dtype=np.float64)
    normalized = text.replace("D", "E").replace("d", "e")
    values = re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[Ee][-+]?\d+)?", normalized)
    return np.asarray([float(value) for value in values], dtype=np.float64)


def _leading_floats(tokens: Sequence[str]) -> list[float]:
    values = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError:
            break
    return values


def _first_nonzero(values: Sequence[float], *, default: float) -> float:
    for value in values:
        if abs(value) > 1e-12:
            return float(value)
    return default


def _required(value: float | None, name: str) -> float:
    if value is None:
        msg = f"{name} is required"
        raise ValueError(msg)
    return value


def _required_str(value: str | None, name: str) -> str:
    if value is None or not value.strip():
        msg = f"{name} is required"
        raise ValueError(msg)
    return value.strip()


def _element_from_filename(path: Path) -> str:
    match = re.match(r"([A-Z][a-z]?)", path.name)
    if match is None:
        msg = "element is required for single GTH files without an element prefix"
        raise ValueError(msg)
    return match.group(1)
