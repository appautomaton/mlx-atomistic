"""Controlled finite-difference stress for periodic plane-wave DFT."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

import numpy as np

from mlx_atomistic.dft._periodic_frozen_energy import (
    _evaluate_periodic_frozen_energy,
)
from mlx_atomistic.dft._periodic_models import (
    PeriodicDFTSystem,
    PeriodicSCFConfig,
    PeriodicSCFResult,
)
from mlx_atomistic.dft._runtime_observer import RuntimeObserver
from mlx_atomistic.dft.kpoints import KPointMesh
from mlx_atomistic.dft.periodic_scf import (
    _run_periodic_scf_fixed_topology,
    run_periodic_scf,
)
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional

HARTREE_PER_BOHR3_TO_GPA = 29421.02648438959

PeriodicStressMode = Literal["isotropic", "diagonal", "symmetric"]
PeriodicStressElectronicResponse = Literal["frozen_variational", "reconverged"]

_STRAIN_COMPONENTS = (
    ("xx", 0, 0),
    ("yy", 1, 1),
    ("zz", 2, 2),
    ("yz", 1, 2),
    ("xz", 0, 2),
    ("xy", 0, 1),
)


@dataclass(frozen=True)
class PeriodicStressConfig:
    """Controls for periodic numerical stress.

    Args:
        mode: `isotropic`, `diagonal`, or complete `symmetric` stress.
        strain_step: Dimensionless central-difference strain.
        electronic_response: `frozen_variational` or diagnostic `reconverged`.
        variational_energy_tolerance: Maximum base frozen-functional mismatch
            in Hartree.
        stress_consistency_tolerance: Maximum stress disagreement between
            primary and doubled frozen-variational strain steps.
        reuse_scf_state: Seed diagnostic reconverged SCFs from the converged
            base density and compact orbitals.
        require_fixed_basis_topology: Transport the base integer-G topology to
            every strained SCF instead of reselecting at the cutoff.
    """

    mode: PeriodicStressMode = "symmetric"
    strain_step: float = 1.0e-3
    electronic_response: PeriodicStressElectronicResponse = "frozen_variational"
    variational_energy_tolerance: float = 5.0e-5
    stress_consistency_tolerance: float = 2.0e-5
    reuse_scf_state: bool = True
    require_fixed_basis_topology: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"isotropic", "diagonal", "symmetric"}:
            raise ValueError("stress mode must be 'isotropic', 'diagonal', or 'symmetric'")
        if (
            isinstance(self.strain_step, (bool, np.bool_))
            or not np.isfinite(self.strain_step)
            or not 0.0 < self.strain_step < 0.1
        ):
            raise ValueError("strain_step must be finite and lie in (0, 0.1)")
        if self.electronic_response not in {"frozen_variational", "reconverged"}:
            raise ValueError(
                "electronic_response must be 'frozen_variational' or 'reconverged'"
            )
        if (
            isinstance(self.variational_energy_tolerance, (bool, np.bool_))
            or not np.isfinite(self.variational_energy_tolerance)
            or self.variational_energy_tolerance <= 0.0
        ):
            raise ValueError("variational_energy_tolerance must be finite and positive")
        if (
            isinstance(self.stress_consistency_tolerance, (bool, np.bool_))
            or not np.isfinite(self.stress_consistency_tolerance)
            or self.stress_consistency_tolerance <= 0.0
        ):
            raise ValueError("stress_consistency_tolerance must be finite and positive")
        if type(self.reuse_scf_state) is not bool:
            raise ValueError("reuse_scf_state must be bool")
        if type(self.require_fixed_basis_topology) is not bool:
            raise ValueError("require_fixed_basis_topology must be bool")
        if (
            self.electronic_response == "frozen_variational"
            and not self.require_fixed_basis_topology
        ):
            raise ValueError(
                "frozen_variational response requires fixed basis topology"
            )

    def to_dict(self) -> dict[str, object]:
        """Return canonical JSON-safe stress controls."""

        return {
            "mode": self.mode,
            "strain_step": self.strain_step,
            "electronic_response": self.electronic_response,
            "variational_energy_tolerance_hartree": (
                self.variational_energy_tolerance
            ),
            "stress_consistency_tolerance_hartree_per_bohr3": (
                self.stress_consistency_tolerance
            ),
            "reuse_scf_state": self.reuse_scf_state,
            "require_fixed_basis_topology": self.require_fixed_basis_topology,
        }


@dataclass(frozen=True)
class PeriodicStressSample:
    """One converged strained-energy sample."""

    component: str
    level: str
    sign: int
    strain: np.ndarray
    energy: float
    energy_by_term: dict[str, float]
    volume: float
    scf_iterations: int | None
    density_residual: float | None
    energy_delta: float | None
    used_density_continuation: bool
    active_counts: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe strained sample."""

        return {
            "component": self.component,
            "level": self.level,
            "sign": self.sign,
            "strain": np.asarray(self.strain, dtype=np.float64).tolist(),
            "energy_hartree": self.energy,
            "energy_by_term_hartree": dict(self.energy_by_term),
            "volume_bohr3": self.volume,
            "scf_iterations": self.scf_iterations,
            "density_residual": self.density_residual,
            "energy_delta_hartree": self.energy_delta,
            "used_density_continuation": self.used_density_continuation,
            "active_counts": list(self.active_counts),
        }


@dataclass(frozen=True)
class PeriodicStressResult:
    """Compression-positive stress from converged periodic free energies."""

    stress: np.ndarray
    pressure: float
    base_scf: PeriodicSCFResult
    samples: tuple[PeriodicStressSample, ...]
    config: PeriodicStressConfig
    elapsed_ms: float
    scf_evaluations: int
    continuation_density_uses: int
    effective_strain_steps: dict[str, float] = field(default_factory=dict)
    base_variational_energy_error: float | None = None
    stress_consistency_errors: dict[str, float] = field(default_factory=dict)

    @property
    def stress_gpa(self) -> np.ndarray:
        """Return the compression-positive stress tensor in GPa."""

        return np.asarray(self.stress, dtype=np.float64) * HARTREE_PER_BOHR3_TO_GPA

    @property
    def pressure_gpa(self) -> float:
        """Return hydrostatic pressure in GPa."""

        return self.pressure * HARTREE_PER_BOHR3_TO_GPA

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe stress report without dense electronic arrays."""

        return {
            "stress_hartree_per_bohr3": np.asarray(
                self.stress,
                dtype=np.float64,
            ).tolist(),
            "stress_gpa": self.stress_gpa.tolist(),
            "pressure_hartree_per_bohr3": self.pressure,
            "pressure_gpa": self.pressure_gpa,
            "sign_convention": "compression-positive",
            "energy_convention": (
                "helmholtz_free_energy"
                if self.base_scf.smearing_width_hartree is not None
                else "total_energy"
            ),
            "config": self.config.to_dict(),
            "elapsed_ms": self.elapsed_ms,
            "scf_evaluations": self.scf_evaluations,
            "continuation_density_uses": self.continuation_density_uses,
            "effective_strain_steps": dict(self.effective_strain_steps),
            "base_variational_energy_error_hartree": (
                self.base_variational_energy_error
            ),
            "stress_consistency_errors_hartree_per_bohr3": dict(
                self.stress_consistency_errors
            ),
            "samples": [sample.to_dict() for sample in self.samples],
        }


def _component_strain(component: tuple[str, int, int]) -> np.ndarray:
    _name, first, second = component
    strain = np.zeros((3, 3), dtype=np.float64)
    if first == second:
        strain[first, second] = 1.0
    else:
        strain[first, second] = 0.5
        strain[second, first] = 0.5
    return strain


def _requested_components(mode: PeriodicStressMode) -> tuple[tuple[str, int, int], ...]:
    if mode == "diagonal":
        return _STRAIN_COMPONENTS[:3]
    if mode == "symmetric":
        return _STRAIN_COMPONENTS
    return ()


def _active_integer_sets(result: PeriodicSCFResult) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(point.basis.active_integer_g, dtype=np.int32)
        for point in result.kpoints
    )


def _active_counts(result: PeriodicSCFResult) -> tuple[int, ...]:
    return tuple(int(point.basis.active_count) for point in result.kpoints)


def _validate_scf_state(
    system: PeriodicDFTSystem,
    result: PeriodicSCFResult,
    *,
    label: str,
) -> None:
    if not isinstance(result, PeriodicSCFResult):
        raise TypeError(f"{label} must be PeriodicSCFResult")
    if (
        not result.converged
        or not np.isfinite(result.total_energy)
        or not np.isfinite(result.electron_count)
        or abs(result.electron_count - system.electron_count) > 1.0e-4
        or result.system_fingerprint != system.fingerprint
    ):
        raise ValueError(f"{label} must be a converged finite state for the exact system")


def _topology_matches(
    reference: tuple[np.ndarray, ...],
    candidate: PeriodicSCFResult,
) -> bool:
    observed = _active_integer_sets(candidate)
    return len(reference) == len(observed) and all(
        np.array_equal(left, right)
        for left, right in zip(reference, observed, strict=True)
    )


def periodic_finite_difference_stress(
    system: PeriodicDFTSystem,
    *,
    cutoff_hartree: float,
    kpoint_mesh: KPointMesh,
    n_bands: int | None = None,
    config: PeriodicStressConfig | None = None,
    scf_config: PeriodicSCFConfig | None = None,
    xc_functional: ExchangeCorrelationFunctional | None = None,
    observer: RuntimeObserver | None = None,
    base_result: PeriodicSCFResult | None = None,
) -> PeriodicStressResult:
    """Evaluate compression-positive periodic stress by central strain.

    Args:
        system: Periodic GTH system at the unstrained cell.
        cutoff_hartree: Fixed plane-wave kinetic cutoff in Hartree.
        kpoint_mesh: Fixed reduced-coordinate k-point mesh.
        n_bands: Fixed computed band count.
        config: Numerical strain and topology controls.
        scf_config: Exact periodic SCF controls.
        xc_functional: Exchange-correlation functional.
        observer: Optional shared runtime observer.
        base_result: Optional converged SCF state for the exact base system.

    Returns:
        Compression-positive stress, pressure, and every strained sample.

    Raises:
        TypeError: If public inputs have unsupported types.
        ValueError: If SCF convergence, identity, or plane-wave topology fails.
    """

    if not isinstance(system, PeriodicDFTSystem):
        raise TypeError("system must be PeriodicDFTSystem")
    if not isinstance(kpoint_mesh, KPointMesh):
        raise TypeError("kpoint_mesh must be KPointMesh")
    if (
        isinstance(cutoff_hartree, (bool, np.bool_))
        or not np.isfinite(cutoff_hartree)
        or cutoff_hartree <= 0.0
    ):
        raise ValueError("cutoff_hartree must be finite and positive")
    resolved = PeriodicStressConfig() if config is None else config
    resolved_scf = PeriodicSCFConfig() if scf_config is None else scf_config
    if not isinstance(resolved, PeriodicStressConfig):
        raise TypeError("config must be PeriodicStressConfig")
    if not isinstance(resolved_scf, PeriodicSCFConfig):
        raise TypeError("scf_config must be PeriodicSCFConfig")
    started = perf_counter()
    scf_evaluations = 0
    continuation_uses = 0
    if base_result is None:
        base_result = run_periodic_scf(
            system,
            cutoff_hartree=cutoff_hartree,
            kpoint_mesh=kpoint_mesh,
            n_bands=n_bands,
            config=resolved_scf,
            xc_functional=xc_functional,
            observer=observer,
        )
        scf_evaluations += 1
    _validate_scf_state(system, base_result, label="base_result")
    base_variational_error = None
    if resolved.electronic_response == "frozen_variational":
        frozen_base = _evaluate_periodic_frozen_energy(
            system,
            base_result,
            cutoff_hartree=cutoff_hartree,
            kpoint_mesh=kpoint_mesh,
            config=resolved_scf,
            xc_functional=xc_functional,
            observer=observer,
        )
        base_variational_error = abs(
            frozen_base.total_energy - base_result.total_energy
        )
        if base_variational_error > resolved.variational_energy_tolerance:
            raise ValueError(
                "base frozen variational energy differs from the converged SCF by "
                f"{base_variational_error:.6g} Hartree"
            )
    reference_topology = _active_integer_sets(base_result)
    base_matrix = np.asarray(system.grid.cell.matrix, dtype=np.float64)
    base_volume = float(system.grid.volume)
    density_seed = base_result.density if resolved.reuse_scf_state else None
    coefficient_seed = (
        base_result.continuation_coefficients if resolved.reuse_scf_state else None
    )
    samples: list[PeriodicStressSample] = []
    effective_steps: dict[str, float] = {}
    consistency_errors: dict[str, float] = {}

    def strained_system(sign: int, strain: np.ndarray, step: float) -> PeriodicDFTSystem:
        deformation = np.eye(3) + sign * step * strain
        return system.with_cell(base_matrix @ deformation, scale_positions=True)

    def evaluate(
        component: str,
        level: str,
        sign: int,
        strain: np.ndarray,
        step: float,
        strained: PeriodicDFTSystem,
    ) -> tuple[float, PeriodicStressSample, bool]:
        nonlocal scf_evaluations, continuation_uses
        if resolved.electronic_response == "frozen_variational":
            frozen = _evaluate_periodic_frozen_energy(
                strained,
                base_result,
                cutoff_hartree=cutoff_hartree,
                kpoint_mesh=kpoint_mesh,
                config=resolved_scf,
                xc_functional=xc_functional,
                observer=observer,
            )
            if abs(frozen.electron_count - system.electron_count) > 1.0e-4:
                raise ValueError("frozen variational state changes the electron count")
            sample = PeriodicStressSample(
                component=component,
                level=level,
                sign=sign,
                strain=sign * step * strain,
                energy=frozen.total_energy,
                energy_by_term=frozen.energy_by_term,
                volume=float(strained.grid.volume),
                scf_iterations=None,
                density_residual=None,
                energy_delta=None,
                used_density_continuation=False,
                active_counts=frozen.active_counts,
            )
            return frozen.total_energy, sample, True
        scf_kwargs = {
            "cutoff_hartree": cutoff_hartree,
            "kpoint_mesh": kpoint_mesh,
            "n_bands": n_bands,
            "config": resolved_scf,
            "xc_functional": xc_functional,
            "initial_density": density_seed,
            "observer": observer,
        }
        if resolved.require_fixed_basis_topology:
            result = _run_periodic_scf_fixed_topology(
                strained,
                basis_integer_g=reference_topology,
                initial_coefficients=coefficient_seed,
                **scf_kwargs,
            )
        else:
            result = run_periodic_scf(strained, **scf_kwargs)
        scf_evaluations += 1
        continuation_uses += int(density_seed is not None)
        _validate_scf_state(strained, result, label=f"{component}:{sign:+d}")
        sample = PeriodicStressSample(
            component=component,
            level=level,
            sign=sign,
            strain=sign * step * strain,
            energy=float(result.total_energy),
            energy_by_term=dict(result.energy_by_term),
            volume=float(strained.grid.volume),
            scf_iterations=int(result.iterations),
            density_residual=float(result.density_residual),
            energy_delta=(
                None if result.energy_delta is None else float(result.energy_delta)
            ),
            used_density_continuation=density_seed is not None,
            active_counts=_active_counts(result),
        )
        return result.total_energy, sample, _topology_matches(reference_topology, result)

    def evaluate_pair(
        component: str,
        strain: np.ndarray,
        *,
        stress_denominator: float,
    ) -> tuple[float, float, float]:
        step = resolved.strain_step
        plus, plus_sample, plus_matches = evaluate(
            component,
            "primary",
            1,
            strain,
            step,
            strained_system(1, strain, step),
        )
        minus, minus_sample, minus_matches = evaluate(
            component,
            "primary",
            -1,
            strain,
            step,
            strained_system(-1, strain, step),
        )
        if resolved.require_fixed_basis_topology and not (
            plus_matches and minus_matches
        ):
            raise ValueError(f"{component} fixed integer-G topology was not preserved")
        samples.extend((plus_sample, minus_sample))
        if resolved.electronic_response == "frozen_variational":
            outer_step = 2.0 * step
            outer_plus, outer_plus_sample, _ = evaluate(
                component,
                "doubled",
                1,
                strain,
                outer_step,
                strained_system(1, strain, outer_step),
            )
            outer_minus, outer_minus_sample, _ = evaluate(
                component,
                "doubled",
                -1,
                strain,
                outer_step,
                strained_system(-1, strain, outer_step),
            )
            samples.extend((outer_plus_sample, outer_minus_sample))
            primary_value = -(plus - minus) / (2.0 * step * stress_denominator)
            doubled_value = -(
                outer_plus - outer_minus
            ) / (2.0 * outer_step * stress_denominator)
            error = abs(primary_value - doubled_value)
            consistency_errors[component] = error
            if error > resolved.stress_consistency_tolerance:
                raise ValueError(
                    f"{component} frozen stress changes by {error:.6g} "
                    "Hartree/bohr^3 across strain scales"
                )
        effective_steps[component] = step
        return plus, minus, step

    stress = np.zeros((3, 3), dtype=np.float64)
    if resolved.mode == "isotropic":
        strain = np.eye(3, dtype=np.float64)
        plus, minus, step = evaluate_pair(
            "isotropic",
            strain,
            stress_denominator=3.0 * base_volume,
        )
        derivative = (plus - minus) / (2.0 * step)
        pressure = -float(derivative) / (3.0 * base_volume)
        np.fill_diagonal(stress, pressure)
    else:
        for component in _requested_components(resolved.mode):
            name, first, second = component
            strain = _component_strain(component)
            plus, minus, step = evaluate_pair(
                name,
                strain,
                stress_denominator=base_volume,
            )
            derivative = (plus - minus) / (2.0 * step)
            value = -float(derivative) / base_volume
            stress[first, second] = value
            stress[second, first] = value
        pressure = float(np.trace(stress) / 3.0)
    return PeriodicStressResult(
        stress=stress,
        pressure=pressure,
        base_scf=base_result,
        samples=tuple(samples),
        config=resolved,
        elapsed_ms=(perf_counter() - started) * 1000.0,
        scf_evaluations=scf_evaluations,
        continuation_density_uses=continuation_uses,
        effective_strain_steps=effective_steps,
        base_variational_energy_error=base_variational_error,
        stress_consistency_errors=consistency_errors,
    )
