"""Apple Silicon-native atomistic simulation tools built on MLX."""

from importlib.metadata import version

from mlx_atomistic.core import Atoms, Cell
from mlx_atomistic.forcefields import (
    CoulombPotential,
    HarmonicAnglePotential,
    HarmonicBondPotential,
    PeriodicDihedralPotential,
)
from mlx_atomistic.topology import Topology
from mlx_atomistic.units import LJ_REDUCED_UNITS, LennardJonesReducedUnits
from mlx_atomistic.validation import (
    ForceValidationCase,
    ForceValidationResult,
    run_force_validation_suite,
    summarize_validation_results,
    validate_force_term,
)

__version__ = version("mlx-atomistic")

__all__ = [
    "Atoms",
    "Cell",
    "CoulombPotential",
    "HarmonicAnglePotential",
    "HarmonicBondPotential",
    "LJ_REDUCED_UNITS",
    "LennardJonesReducedUnits",
    "PeriodicDihedralPotential",
    "ForceValidationCase",
    "ForceValidationResult",
    "Topology",
    "__version__",
    "run_force_validation_suite",
    "summarize_validation_results",
    "validate_force_term",
]
