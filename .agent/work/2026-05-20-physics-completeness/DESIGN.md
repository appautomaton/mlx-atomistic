# DESIGN: Phase 3 Physics Completeness and Phase 4 Production Hardening

## Architecture Approach

Extend the established prepared-system pipeline: parsers produce `PreparedSystem`, artifacts validate and round-trip it, `build_mlx_system_from_artifact` creates runtime terms, and OpenMM remains test-only.

New force terms follow `ForceTerm` protocol with `energy_forces(positions, cell, pairs)`. Each adds a class to `forcefields.py` (or a dedicated module) and a term entry to `artifacts.py`. `NonbondedPotential` is the only existing class that needs internal modification (virtual-site redistribution and soft-core parameters).

## New Module Boundaries

- `virtual_sites.py`: VirtualSite base classes, geometry computation, force redistribution manager.
- `gbsa.py`: GB-OBC implicit solvent force term and ACE surface-area approximation.
- `custom_force.py`: CustomForcePotential with symbolic expression evaluation.
- `replica_exchange.py` (or new top-level module): Multi-copy driver, temperature/Hamiltonian swap logic.
- Soft-core and lambda scaling: modifications within `nonbonded.py` for `NonbondedPotential`, plus new lambda-scaled wrapper.

## Force Redistribution Architecture

Virtual-site forces flow through a redistribution layer:

1. Before force evaluation: reconstruct virtual-site positions from parent atoms.
2. During force evaluation: `NonbondedPotential` and other terms compute forces on all atoms (real + virtual).
3. After force evaluation: redistribute virtual-site forces to parent atoms using geometry-derived weight matrices.
4. Constraints (SETTLE, DistanceConstraints) operate on real atoms only.

This requires a `VirtualSiteManager` that:
- Computes virtual-site positions from parent positions at the start of each step.
- Provides force-redistribution weights to the MD driver.
- Integrates with `simulate_nvt`/`simulate_npt` without modifying their signatures (add an optional `virtual_sites` parameter).

## Custom Force Expression Architecture

`CustomForcePotential` evaluates symbolic expressions:

- Accepts a string expression (e.g., `"k * (r - r0)^2"`) and per-particle or per-pair parameters.
- Uses `mlx.core` operations only (no Python `eval`; safe and MLX-compatible).
- Expression is parsed into an operation DAG at construction time, then evaluated as MLX ops.
- GBSA/OBC uses this framework for its GB energy term if the expression framework is general enough, or GBSA operates as a standalone `ForceTerm` with internal MLX kernels for the surface-area computation.

## GBSA/OBC Architecture

GB-OBC implicit solvent is a `ForceTerm`:

- `GBSAForcePotential` computes GB energy using the OBC screening function.
- ACE surface-area approximation uses pairwise neighbor lists for efficiency.
- The surface-area kernel is initially Python/MLX with correctness tests; performance optimization is a later concern.
- Parameters: solute dielectric, solvent dielectric, cutoff, surface-area tension.

## Soft-Core and Lambda Architecture

Soft-core modifications are internal to `NonbondedPotential`:

- `NonbondedPotential` accepts optional `lambda_lj` and `lambda_electrostatics` parameters (default `1.0`).
- When `lambda < 1`, LJ and Coulomb use soft-core potentials with shifted `r` values.
- `dU/dlambda` is computed analytically alongside energy and forces.
- `SoftCoreNonbondedPotential` wraps `NonbondedPotential` with lambda parameters for artifact construction.

## Replica Exchange Architecture

Replica exchange is a new simulation driver, not a modification of existing drivers:

- `simulate_replica_exchange` manages N replicas, each with its own `SimulationState` and force terms.
- Temperature ladder is configurable.
- Hamiltonian exchange: lambda-scaled `NonbondedPotential` parameters vary across replicas.
- Swap acceptance uses the standard Metropolis criterion.
- This is a new top-level function in a dedicated module, not a modification of `simulate_nvt`/`simulate_npt`.

## Artifact Pipeline Changes

Each new capability adds terms to `SUPPORTED_FORCE_TERMS` and removes entries from `FAIL_CLOSED_TERMS`:

- `virtual_site` → moved from `FAIL_CLOSED_TERMS` to `SUPPORTED_FORCE_TERMS`
- `tip4p` → moved (or mapped to `virtual_site`)
- `custom_force` → added to `SUPPORTED_FORCE_TERMS`
- `gbsa` → removed from `FAIL_CLOSED_TERMS`, added to `SUPPORTED_FORCE_TERMS`
- `soft_core_lj` → added to `SUPPORTED_FORCE_TERMS`
- `lambda_scaled_nonbonded` → added to `SUPPORTED_FORCE_TERMS`
- `replica_exchange` → added to `SUPPORTED_FORCE_TERMS` for configuration metadata

Each is done in its owning slice to avoid cross-slice merge conflicts in `artifacts.py`.

## Fail-Closed Boundary

Unsupported Phase 5+ features — polarizable force fields (Drude, AMOEBA), machine-learned potentials, reactive force fields, materials potentials, CUDA/ROCm — remain blockers. `FAIL_CLOSED_TERMS` entries for `drude`, `polarizable`, `reactive`, `qm_mm` continue to block partial execution.