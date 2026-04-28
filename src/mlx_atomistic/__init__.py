"""Apple Silicon-native atomistic simulation tools built on MLX."""

from importlib.metadata import version

from mlx_atomistic.constraints import DistanceConstraints
from mlx_atomistic.core import Atoms, Cell
from mlx_atomistic.dft import (
    DFTSystem,
    DiracExchange,
    ExchangeCorrelationFunctional,
    LDACorrelationPZ81,
    LDAExchangeCorrelation,
    LinearMixer,
    LocalGaussianPseudopotential,
    PulayDIISMixer,
    RealSpaceGrid,
    ReciprocalGrid,
    SCFConfig,
    SCFResult,
    density_from_orbitals,
    hartree_potential,
    lda_exchange_energy_potential,
    local_pseudopotential_forces,
    normalize_orbitals,
    run_scf,
)
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
    "DFTSystem",
    "DistanceConstraints",
    "DiracExchange",
    "AngleParameter",
    "ExchangeCorrelationFunctional",
    "HarmonicAnglePotential",
    "HarmonicBondPotential",
    "AtomType",
    "BondParameter",
    "DihedralParameter",
    "ForceField",
    "LDACorrelationPZ81",
    "LJ_REDUCED_UNITS",
    "LDAExchangeCorrelation",
    "LennardJonesReducedUnits",
    "LinearMixer",
    "MMSystem",
    "NonbondedParameter",
    "NonbondedPotential",
    "PeriodicDihedralPotential",
    "LocalGaussianPseudopotential",
    "PulayDIISMixer",
    "RealSpaceGrid",
    "ReciprocalGrid",
    "SCFConfig",
    "SCFResult",
    "ForceValidationCase",
    "ForceValidationResult",
    "TrajectoryRecord",
    "Topology",
    "__version__",
    "load_npz_trajectory",
    "density_from_orbitals",
    "hartree_potential",
    "lda_exchange_energy_potential",
    "local_pseudopotential_forces",
    "normalize_orbitals",
    "read_xyz",
    "restart_state_from_trajectory",
    "run_scf",
    "run_force_validation_suite",
    "save_npz_trajectory",
    "summarize_validation_results",
    "validate_force_term",
    "write_xyz",
]
