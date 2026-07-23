"""Material-independent equation-of-state fitting and validation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

HARTREE_TO_EV = 27.211386245988
EV_PER_ANGSTROM3_TO_GPA = 160.2176634

VERIFIED_THRESHOLDS = {
    "delta_mev_per_atom": 3.0,
    "lattice_relative": 0.005,
    "bulk_modulus_relative": 0.10,
    "bulk_derivative_relative": 0.15,
}
EXCELLENT_THRESHOLDS = {
    "delta_mev_per_atom": 1.0,
    "lattice_relative": 0.002,
    "bulk_modulus_relative": 0.05,
    "bulk_derivative_relative": 0.10,
}
CONVERGENCE_THRESHOLDS = {
    "curve_max_mev_per_atom": 1.0,
    "lattice_relative": 0.001,
    "bulk_modulus_relative": 0.03,
    "bulk_derivative_relative": 0.10,
}


def _relative_error(value: float, reference: float) -> float:
    if reference == 0.0:
        raise ValueError("relative error reference must be nonzero")
    return abs(value - reference) / abs(reference)


def birch_murnaghan_energy(
    volume_angstrom3: np.ndarray | float,
    energy0_ev: float,
    volume0_angstrom3: float,
    bulk_modulus_ev_angstrom3: float,
    bulk_derivative: float,
) -> np.ndarray:
    """Evaluate the third-order Birch-Murnaghan energy equation."""

    volume = np.asarray(volume_angstrom3, dtype=np.float64)
    ratio = (volume0_angstrom3 / volume) ** (2.0 / 3.0)
    strain = ratio - 1.0
    correction = strain**3 * bulk_derivative + strain**2 * (6.0 - 4.0 * ratio)
    return energy0_ev + 9.0 * volume0_angstrom3 * bulk_modulus_ev_angstrom3 * correction / 16.0


def fit_birch_murnaghan(
    volumes_angstrom3_per_atom: Sequence[float],
    energies_ev_per_atom: Sequence[float],
) -> dict[str, Any]:
    """Fit and guard a seven-point third-order Birch-Murnaghan EOS."""

    volumes = np.asarray(volumes_angstrom3_per_atom, dtype=np.float64)
    energies = np.asarray(energies_ev_per_atom, dtype=np.float64)
    if volumes.shape != energies.shape or volumes.ndim != 1 or volumes.size != 7:
        msg = "Birch-Murnaghan fitting requires seven matching one-dimensional samples"
        raise ValueError(msg)
    if not np.isfinite(volumes).all() or not np.isfinite(energies).all():
        msg = "Birch-Murnaghan fitting inputs must be finite"
        raise ValueError(msg)
    if np.any(volumes <= 0.0) or np.any(np.diff(volumes) <= 0.0):
        msg = "Birch-Murnaghan volumes must be positive and strictly increasing"
        raise ValueError(msg)

    observed_index = int(np.argmin(energies))
    if observed_index in {0, volumes.size - 1}:
        return {
            "status": "blocked",
            "blocker": "observed_eos_minimum_not_interior",
            "observed_minimum_index": observed_index,
        }
    initial = (
        float(energies[observed_index]),
        float(volumes[observed_index]),
        0.55,
        4.0,
    )
    lower = (-np.inf, float(volumes[0]), 1.0e-8, 0.0)
    upper = (np.inf, float(volumes[-1]), 10.0, 10.0)
    try:
        values, covariance = curve_fit(
            birch_murnaghan_energy,
            volumes,
            energies,
            p0=initial,
            bounds=(lower, upper),
            maxfev=50_000,
        )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        return {
            "status": "blocked",
            "blocker": "birch_murnaghan_fit_failed",
            "error": str(error),
            "observed_minimum_index": observed_index,
        }
    energy0, volume0, bulk_modulus, bulk_derivative = (float(value) for value in values)
    fitted = birch_murnaghan_energy(volumes, *values)
    residual = fitted - energies
    interior = float(volumes[0]) < volume0 < float(volumes[-1])
    finite = np.isfinite(values).all() and np.isfinite(covariance).all()
    status = "ok" if interior and finite else "blocked"
    return {
        "status": status,
        "blocker": None if status == "ok" else "eos_fit_not_finite_or_interior",
        "energy0_ev_per_atom": energy0,
        "equilibrium_volume_angstrom3_per_atom": volume0,
        "bulk_modulus_ev_angstrom3": bulk_modulus,
        "bulk_modulus_gpa": bulk_modulus * EV_PER_ANGSTROM3_TO_GPA,
        "bulk_derivative": bulk_derivative,
        "rmse_mev_per_atom": float(np.sqrt(np.mean(residual * residual)) * 1000.0),
        "max_residual_mev_per_atom": float(np.max(np.abs(residual)) * 1000.0),
        "observed_minimum_index": observed_index,
        "observed_minimum_volume_angstrom3_per_atom": float(volumes[observed_index]),
    }


def fit_cubic_eos(
    lattice_constants_angstrom: Sequence[float],
    total_energies_hartree: Sequence[float],
    *,
    atom_count: int,
) -> dict[str, Any]:
    """Fit a cubic-cell EOS from total cell energies."""

    lattice = np.asarray(lattice_constants_angstrom, dtype=np.float64)
    energies = np.asarray(total_energies_hartree, dtype=np.float64)
    if atom_count <= 0:
        raise ValueError("atom_count must be positive")
    if lattice.shape != energies.shape:
        raise ValueError("lattice constants and total energies must have matching shapes")
    fit = fit_birch_murnaghan(
        lattice**3 / atom_count,
        energies * HARTREE_TO_EV / atom_count,
    )
    if "equilibrium_volume_angstrom3_per_atom" in fit:
        fit["equilibrium_lattice_constant_angstrom"] = (
            fit["equilibrium_volume_angstrom3_per_atom"] * atom_count
        ) ** (1.0 / 3.0)
    fit["atom_count"] = atom_count
    fit["equation"] = "third-order Birch-Murnaghan"
    return fit


def reference_fit(reference: Mapping[str, Any]) -> dict[str, float]:
    """Normalize a diamond-structure reference fit to per-atom volume."""

    primitive_atoms = int(reference["primitive_cell_atoms"])
    values = reference["fit"]
    primitive_volume = float(values["equilibrium_volume_angstrom3"])
    return {
        "equilibrium_volume_angstrom3_per_atom": primitive_volume / primitive_atoms,
        "equilibrium_lattice_constant_angstrom": (4.0 * primitive_volume) ** (1.0 / 3.0),
        "bulk_modulus_ev_angstrom3": float(values["bulk_modulus_ev_angstrom3"]),
        "bulk_modulus_gpa": float(values["bulk_modulus_ev_angstrom3"])
        * EV_PER_ANGSTROM3_TO_GPA,
        "bulk_derivative": float(values["bulk_derivative"]),
    }


def delta_factor_mev_per_atom(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
) -> float:
    """Return the RMS EOS energy difference over ±6% reference volume."""

    reference_volume = float(reference["equilibrium_volume_angstrom3_per_atom"])
    volumes = np.linspace(0.94 * reference_volume, 1.06 * reference_volume, 4097)
    candidate_curve = birch_murnaghan_energy(
        volumes,
        0.0,
        float(candidate["equilibrium_volume_angstrom3_per_atom"]),
        float(candidate["bulk_modulus_ev_angstrom3"]),
        float(candidate["bulk_derivative"]),
    )
    reference_curve = birch_murnaghan_energy(
        volumes,
        0.0,
        reference_volume,
        float(reference["bulk_modulus_ev_angstrom3"]),
        float(reference["bulk_derivative"]),
    )
    difference = candidate_curve - reference_curve
    return float(
        np.sqrt(np.trapezoid(difference * difference, volumes) / (volumes[-1] - volumes[0]))
        * 1000.0
    )


def compare_fit_to_reference(
    candidate: Mapping[str, Any],
    reference: Mapping[str, float],
) -> dict[str, Any]:
    """Compare an EOS fit with the primary all-electron reference."""

    required = {
        "equilibrium_volume_angstrom3_per_atom",
        "equilibrium_lattice_constant_angstrom",
        "bulk_modulus_ev_angstrom3",
        "bulk_derivative",
    }
    if candidate.get("status") != "ok" or not required.issubset(candidate):
        return {
            "status": "blocked",
            "blocker": "candidate_eos_fit_not_admissible",
            "verified": False,
            "excellent": False,
        }
    metrics = {
        "delta_mev_per_atom": delta_factor_mev_per_atom(candidate, reference),
        "lattice_relative": _relative_error(
            float(candidate["equilibrium_lattice_constant_angstrom"]),
            float(reference["equilibrium_lattice_constant_angstrom"]),
        ),
        "bulk_modulus_relative": _relative_error(
            float(candidate["bulk_modulus_ev_angstrom3"]),
            float(reference["bulk_modulus_ev_angstrom3"]),
        ),
        "bulk_derivative_relative": _relative_error(
            float(candidate["bulk_derivative"]),
            float(reference["bulk_derivative"]),
        ),
    }

    def passes(thresholds: Mapping[str, float]) -> bool:
        return all(metrics[key] <= limit for key, limit in thresholds.items())

    return {
        "status": "ok",
        "metrics": metrics,
        "verified_thresholds": VERIFIED_THRESHOLDS,
        "excellent_thresholds": EXCELLENT_THRESHOLDS,
        "verified": passes(VERIFIED_THRESHOLDS),
        "excellent": passes(EXCELLENT_THRESHOLDS),
    }


def compare_eos_convergence(
    baseline_fit: Mapping[str, Any],
    candidate_fit: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two local EOS fits for cutoff or k-point convergence."""

    if baseline_fit.get("status") != "ok" or candidate_fit.get("status") != "ok":
        return {"status": "blocked", "blocker": "convergence_fit_not_admissible", "passed": False}
    reference_volume = float(baseline_fit["equilibrium_volume_angstrom3_per_atom"])
    volumes = np.linspace(0.94 * reference_volume, 1.06 * reference_volume, 4097)
    baseline_curve = birch_murnaghan_energy(
        volumes,
        0.0,
        reference_volume,
        float(baseline_fit["bulk_modulus_ev_angstrom3"]),
        float(baseline_fit["bulk_derivative"]),
    )
    candidate_curve = birch_murnaghan_energy(
        volumes,
        0.0,
        float(candidate_fit["equilibrium_volume_angstrom3_per_atom"]),
        float(candidate_fit["bulk_modulus_ev_angstrom3"]),
        float(candidate_fit["bulk_derivative"]),
    )
    metrics = {
        "curve_max_mev_per_atom": float(np.max(np.abs(candidate_curve - baseline_curve)) * 1000.0),
        "lattice_relative": _relative_error(
            float(candidate_fit["equilibrium_lattice_constant_angstrom"]),
            float(baseline_fit["equilibrium_lattice_constant_angstrom"]),
        ),
        "bulk_modulus_relative": _relative_error(
            float(candidate_fit["bulk_modulus_ev_angstrom3"]),
            float(baseline_fit["bulk_modulus_ev_angstrom3"]),
        ),
        "bulk_derivative_relative": _relative_error(
            float(candidate_fit["bulk_derivative"]),
            float(baseline_fit["bulk_derivative"]),
        ),
    }
    passed = all(metrics[key] <= limit for key, limit in CONVERGENCE_THRESHOLDS.items())
    return {
        "status": "ok",
        "metrics": metrics,
        "thresholds": CONVERGENCE_THRESHOLDS,
        "passed": passed,
    }
