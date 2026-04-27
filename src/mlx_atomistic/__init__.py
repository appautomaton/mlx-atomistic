"""Apple Silicon-native atomistic simulation tools built on MLX."""

from importlib.metadata import version

from mlx_atomistic.constraints import DistanceConstraints
from mlx_atomistic.core import Atoms, Cell
from mlx_atomistic.forcefields import (
    CoulombPotential,
    HarmonicAnglePotential,
    HarmonicBondPotential,
    NonbondedPotential,
    PeriodicDihedralPotential,
)
from mlx_atomistic.io import (
    TrajectoryRecord,
    load_npz_trajectory,
    read_xyz,
    restart_state_from_trajectory,
    save_npz_trajectory,
    write_xyz,
)
from mlx_atomistic.mm import (
    AngleParameter,
    AtomType,
    BondParameter,
    DihedralParameter,
    ForceField,
    MMSystem,
    NonbondedParameter,
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
    "DistanceConstraints",
    "AngleParameter",
    "HarmonicAnglePotential",
    "HarmonicBondPotential",
    "AtomType",
    "BondParameter",
    "DihedralParameter",
    "ForceField",
    "LJ_REDUCED_UNITS",
    "LennardJonesReducedUnits",
    "MMSystem",
    "NonbondedParameter",
    "NonbondedPotential",
    "PeriodicDihedralPotential",
    "ForceValidationCase",
    "ForceValidationResult",
    "TrajectoryRecord",
    "Topology",
    "__version__",
    "load_npz_trajectory",
    "read_xyz",
    "restart_state_from_trajectory",
    "run_force_validation_suite",
    "save_npz_trajectory",
    "summarize_validation_results",
    "validate_force_term",
    "write_xyz",
]
