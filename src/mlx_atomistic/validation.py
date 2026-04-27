"""Validation helpers for force terms and MD diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import pi
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.forcefields import (
    CoulombPotential,
    HarmonicAnglePotential,
    HarmonicBondPotential,
    PeriodicDihedralPotential,
)
from mlx_atomistic.md import ForceTerm, LennardJonesPotential


@dataclass(frozen=True)
class ForceValidationCase:
    """One finite-difference force validation case."""

    name: str
    term: ForceTerm
    positions: object
    seed: int
    epsilon: float = 1e-3
    tolerance: float = 5e-3
    cell: Cell | None = None
    pairs: object | None = None

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            msg = "epsilon must be positive"
            raise ValueError(msg)
        if self.tolerance < 0.0:
            msg = "tolerance must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class ForceValidationResult:
    """Finite-difference force validation result."""

    case_name: str
    term_name: str
    seed: int
    atom_count: int
    coordinate_count: int
    epsilon: float
    tolerance: float
    energy: float
    max_abs_error: float
    rms_abs_error: float
    max_force_abs: float
    failing_atom: int
    failing_axis: int
    finite: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON- and CSV-friendly representation."""

        return {
            "case_name": self.case_name,
            "term_name": self.term_name,
            "seed": self.seed,
            "atom_count": self.atom_count,
            "coordinate_count": self.coordinate_count,
            "epsilon": self.epsilon,
            "tolerance": self.tolerance,
            "energy": self.energy,
            "max_abs_error": self.max_abs_error,
            "rms_abs_error": self.rms_abs_error,
            "max_force_abs": self.max_force_abs,
            "failing_atom": self.failing_atom,
            "failing_axis": self.failing_axis,
            "finite": self.finite,
            "passed": self.passed,
        }


def _energy_scalar(
    term: ForceTerm,
    positions: np.ndarray,
    *,
    cell: Cell | None,
    pairs: object | None,
) -> float:
    energy, _ = term.energy_forces(as_mx_array(positions), cell=cell, pairs=pairs)
    mx.eval(energy)
    return float(np.asarray(energy))


def _finite_difference_forces(
    term: ForceTerm,
    positions: np.ndarray,
    *,
    epsilon: float,
    cell: Cell | None,
    pairs: object | None,
) -> np.ndarray:
    forces = np.zeros_like(positions, dtype=np.float64)
    for atom in range(positions.shape[0]):
        for axis in range(positions.shape[1]):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom, axis] += epsilon
            minus[atom, axis] -= epsilon
            e_plus = _energy_scalar(term, plus, cell=cell, pairs=pairs)
            e_minus = _energy_scalar(term, minus, cell=cell, pairs=pairs)
            forces[atom, axis] = -(e_plus - e_minus) / (2.0 * epsilon)
    return forces


def validate_force_term(
    term: ForceTerm,
    positions,
    *,
    case_name: str | None = None,
    seed: int = 0,
    epsilon: float = 1e-3,
    tolerance: float = 5e-3,
    cell: Cell | None = None,
    pairs: object | None = None,
) -> ForceValidationResult:
    """Compare analytic/autodiff forces against central finite differences."""

    positions_np = np.asarray(positions, dtype=np.float32)
    if positions_np.ndim != 2 or positions_np.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    if positions_np.shape[0] == 0:
        msg = "positions must contain at least one atom"
        raise ValueError(msg)
    if epsilon <= 0.0:
        msg = "epsilon must be positive"
        raise ValueError(msg)
    if tolerance < 0.0:
        msg = "tolerance must be non-negative"
        raise ValueError(msg)

    energy, forces = term.energy_forces(as_mx_array(positions_np), cell=cell, pairs=pairs)
    mx.eval(energy, forces)
    analytical = np.asarray(forces, dtype=np.float64)
    finite_difference = _finite_difference_forces(
        term,
        positions_np,
        epsilon=epsilon,
        cell=cell,
        pairs=pairs,
    )

    abs_error = np.abs(analytical - finite_difference)
    failing_index = np.unravel_index(int(np.argmax(abs_error)), abs_error.shape)
    finite = bool(
        np.all(np.isfinite(analytical))
        and np.all(np.isfinite(finite_difference))
        and np.all(np.isfinite(abs_error))
        and np.isfinite(float(np.asarray(energy)))
    )
    max_abs_error = float(np.max(abs_error))
    rms_abs_error = float(np.sqrt(np.mean(abs_error * abs_error)))

    return ForceValidationResult(
        case_name=case_name or str(getattr(term, "name", type(term).__name__)),
        term_name=str(getattr(term, "name", type(term).__name__)),
        seed=seed,
        atom_count=int(positions_np.shape[0]),
        coordinate_count=int(positions_np.size),
        epsilon=epsilon,
        tolerance=tolerance,
        energy=float(np.asarray(energy)),
        max_abs_error=max_abs_error,
        rms_abs_error=rms_abs_error,
        max_force_abs=float(np.max(np.abs(analytical))),
        failing_atom=int(failing_index[0]),
        failing_axis=int(failing_index[1]),
        finite=finite,
        passed=bool(finite and max_abs_error <= tolerance),
    )


def _jitter(base: np.ndarray, rng: np.random.Generator, *, scale: float = 0.03) -> np.ndarray:
    return (base + rng.normal(scale=scale, size=base.shape)).astype(np.float32)


def _case_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.int32).max))


def default_force_validation_cases(
    *,
    seed: int = 7,
    cases_per_term: int = 1,
    epsilon: float = 1e-3,
    tolerance: float = 5e-3,
) -> tuple[ForceValidationCase, ...]:
    """Return seeded force-validation cases for the currently supported terms."""

    if cases_per_term <= 0:
        msg = "cases_per_term must be positive"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    cases: list[ForceValidationCase] = []
    base_bond = np.array([[0.0, 0.0, 0.0], [1.18, 0.08, 0.02]], dtype=np.float32)
    base_angle = np.array(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.25, 0.92, 0.08]],
        dtype=np.float32,
    )
    base_dihedral = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.2, 0.95, 0.05], [1.45, 1.08, 0.78]],
        dtype=np.float32,
    )
    base_lj = np.array(
        [[0.0, 0.0, 0.0], [1.35, 0.1, 0.0], [0.2, 1.42, 0.1], [1.45, 1.36, 0.28]],
        dtype=np.float32,
    )
    base_coulomb = np.array(
        [[0.0, 0.0, 0.0], [1.15, 0.15, 0.05], [0.35, 1.25, 0.08], [1.5, 1.2, 0.25]],
        dtype=np.float32,
    )

    for index in range(cases_per_term):
        bond_seed = _case_seed(rng)
        cases.append(
            ForceValidationCase(
                name=f"bond-{index}",
                term=HarmonicBondPotential([(0, 1)], k=5.0, length=1.0),
                positions=_jitter(base_bond, np.random.default_rng(bond_seed)),
                seed=bond_seed,
                epsilon=epsilon,
                tolerance=tolerance,
            )
        )

        angle_seed = _case_seed(rng)
        cases.append(
            ForceValidationCase(
                name=f"angle-{index}",
                term=HarmonicAnglePotential([(0, 1, 2)], k=2.0, angle=pi / 2.0),
                positions=_jitter(base_angle, np.random.default_rng(angle_seed)),
                seed=angle_seed,
                epsilon=epsilon,
                tolerance=tolerance,
            )
        )

        dihedral_seed = _case_seed(rng)
        cases.append(
            ForceValidationCase(
                name=f"dihedral-{index}",
                term=PeriodicDihedralPotential(
                    [(0, 1, 2, 3)],
                    k=0.4,
                    periodicity=3.0,
                    phase=0.1,
                ),
                positions=_jitter(base_dihedral, np.random.default_rng(dihedral_seed)),
                seed=dihedral_seed,
                epsilon=epsilon,
                tolerance=tolerance,
            )
        )

        lj_seed = _case_seed(rng)
        cases.append(
            ForceValidationCase(
                name=f"lj-{index}",
                term=LennardJonesPotential(cutoff=None, shift=False),
                positions=_jitter(base_lj, np.random.default_rng(lj_seed), scale=0.02),
                seed=lj_seed,
                epsilon=epsilon,
                tolerance=tolerance,
            )
        )

        coulomb_seed = _case_seed(rng)
        cases.append(
            ForceValidationCase(
                name=f"coulomb-{index}",
                term=CoulombPotential(charges=[1.0, -0.5, 0.25, -0.75], cutoff=None),
                positions=_jitter(base_coulomb, np.random.default_rng(coulomb_seed), scale=0.02),
                seed=coulomb_seed,
                epsilon=epsilon,
                tolerance=tolerance,
            )
        )

    return tuple(cases)


def run_force_validation_suite(
    cases: Iterable[ForceValidationCase] | None = None,
    *,
    seed: int = 7,
    cases_per_term: int = 1,
    epsilon: float = 1e-3,
    tolerance: float = 5e-3,
) -> tuple[ForceValidationResult, ...]:
    """Run a seeded finite-difference force-validation suite."""

    if cases is None:
        cases = default_force_validation_cases(
            seed=seed,
            cases_per_term=cases_per_term,
            epsilon=epsilon,
            tolerance=tolerance,
        )

    return tuple(
        validate_force_term(
            case.term,
            case.positions,
            case_name=case.name,
            seed=case.seed,
            epsilon=case.epsilon,
            tolerance=case.tolerance,
            cell=case.cell,
            pairs=case.pairs,
        )
        for case in cases
    )


def summarize_validation_results(results: Iterable[ForceValidationResult]) -> dict[str, Any]:
    """Summarize force-validation results for CLIs and notebooks."""

    result_tuple = tuple(results)
    if not result_tuple:
        msg = "results must not be empty"
        raise ValueError(msg)

    worst = max(result_tuple, key=lambda result: result.max_abs_error)
    failed = [result for result in result_tuple if not result.passed]
    return {
        "total_cases": len(result_tuple),
        "passed_cases": len(result_tuple) - len(failed),
        "failed_cases": len(failed),
        "all_passed": len(failed) == 0,
        "pass_rate": (len(result_tuple) - len(failed)) / len(result_tuple),
        "max_abs_error": max(result.max_abs_error for result in result_tuple),
        "max_rms_abs_error": max(result.rms_abs_error for result in result_tuple),
        "worst_case": worst.to_dict(),
        "failed_case_names": [result.case_name for result in failed],
    }


__all__ = [
    "ForceValidationCase",
    "ForceValidationResult",
    "default_force_validation_cases",
    "run_force_validation_suite",
    "summarize_validation_results",
    "validate_force_term",
]
