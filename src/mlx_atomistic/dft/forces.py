"""DFT force consistency checks."""

from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft.scf import SCFConfig, SCFResult, run_scf
from mlx_atomistic.dft.system import DFTSystem
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional


@dataclass(frozen=True)
class SCFForceConsistencyResult:
    """SCF total-energy finite-difference force check."""

    result: SCFResult
    analytic_forces: mx.array
    finite_difference_forces: mx.array
    max_abs_error: float
    rms_abs_error: float
    displacement: float
    samples: list[dict[str, float | int]]

    def to_dict(self) -> dict:
        """Return a JSON-safe summary."""

        return {
            "max_abs_error": self.max_abs_error,
            "rms_abs_error": self.rms_abs_error,
            "displacement": self.displacement,
            "analytic_forces": np.array(self.analytic_forces).tolist(),
            "finite_difference_forces": np.array(self.finite_difference_forces).tolist(),
            "samples": list(self.samples),
            "result": self.result.to_dict(),
        }


def _default_force_check_config() -> SCFConfig:
    return SCFConfig(
        max_iterations=6,
        tolerance=1e-8,
        mixing=0.45,
        solver="dense",
        seed=19,
        convergence_mode="either",
    )


def scf_total_energy_forces(
    system: DFTSystem,
    *,
    config: SCFConfig | None = None,
    xc_functional: ExchangeCorrelationFunctional | None = None,
    displacement: float = 1e-3,
    reuse_scf_state: bool = True,
) -> SCFForceConsistencyResult:
    """Compare SCF Hellmann-Feynman forces with total-energy finite differences.

    The analytic force is the force reported by :func:`run_scf`, which includes
    electronic local-pseudopotential forces and center-center repulsion for a
    :class:`DFTSystem`. The finite-difference force is
    ``-[E(R + δ) - E(R - δ)] / 2δ`` after re-running SCF for each displacement.
    """

    if displacement <= 0.0:
        msg = "displacement must be positive"
        raise ValueError(msg)
    config = _default_force_check_config() if config is None else config
    base = run_scf(system, config=config, xc_functional=xc_functional)
    if base.forces is None:
        msg = "SCF result did not include center forces"
        raise ValueError(msg)

    centers = np.array(system.centers, dtype=np.float64)
    finite_difference = np.zeros_like(centers)
    samples: list[dict[str, float | int]] = []
    initial_density = base.density if reuse_scf_state else None
    initial_orbitals = base.orbitals if reuse_scf_state else None

    for center_index in range(system.center_count):
        for axis in range(3):
            plus_centers = centers.copy()
            minus_centers = centers.copy()
            plus_centers[center_index, axis] += displacement
            minus_centers[center_index, axis] -= displacement
            plus = run_scf(
                system.with_centers(plus_centers),
                config=config,
                initial_density=initial_density,
                initial_orbitals=initial_orbitals,
                xc_functional=xc_functional,
            )
            minus = run_scf(
                system.with_centers(minus_centers),
                config=config,
                initial_density=initial_density,
                initial_orbitals=initial_orbitals,
                xc_functional=xc_functional,
            )
            force = -(plus.total_energy - minus.total_energy) / (2.0 * displacement)
            finite_difference[center_index, axis] = force
            samples.append(
                {
                    "center": center_index,
                    "axis": axis,
                    "energy_plus": plus.total_energy,
                    "energy_minus": minus.total_energy,
                    "force": float(force),
                }
            )

    analytic = np.array(base.forces, dtype=np.float64)
    delta = analytic - finite_difference
    max_abs_error = float(np.max(np.abs(delta)))
    rms_abs_error = float(np.sqrt(np.mean(delta * delta)))
    summary = {
        "max_abs_error": max_abs_error,
        "rms_abs_error": rms_abs_error,
        "displacement": displacement,
        "analytic_forces": analytic.tolist(),
        "finite_difference_forces": finite_difference.tolist(),
    }
    checked_result = replace(base, force_consistency=summary)
    return SCFForceConsistencyResult(
        result=checked_result,
        analytic_forces=mx.array(analytic.astype(np.float32)),
        finite_difference_forces=mx.array(finite_difference.astype(np.float32)),
        max_abs_error=max_abs_error,
        rms_abs_error=rms_abs_error,
        displacement=displacement,
        samples=samples,
    )
