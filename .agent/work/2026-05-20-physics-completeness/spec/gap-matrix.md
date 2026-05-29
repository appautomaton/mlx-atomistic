# Gap Matrix: Phase 3 Physics Completeness

This file is normative detail linked from `SPEC.md`.

## Virtual Sites

| Gap ID | Area | Current Evidence | Required Closure | Acceptance Link |
|--------|------|------------------|-------------------|-----------------|
| P3-VS-01 | Virtual site base | `artifacts.py` line 73-86: `FAIL_CLOSED_TERMS` blocks `virtual_site`, `virtual_sites`, `tip4p`, `opc`, `advanced_water`. No `VirtualSite` class exists. | Add `VirtualSite` base classes (TwoParticleAverage, ThreeParticleAverage, OutOfPlane, LocalCoordinates) with parent-atom geometry computation. | AC-01 |
| P3-VS-02 | Force redistribution | `nonbonded.py` (1,092 lines) computes forces on real atoms only. No redistribution path from virtual to parent atoms exists. | Virtual-site forces redistribute to parent atoms using local-coordinate weights; redistributed forces match analytically derived reference. | AC-01 |
| P3-VS-03 | Virtual-site constraints | `constraints.py` has DistanceConstraints and SETTLE only. No virtual-site position reconstruction from parent atoms. | Virtual sites reconstruct positions from parent atoms each timestep before force evaluation. | AC-01 |
| P3-VS-04 | TIP4P-Ew water | No water model chooses virtual sites. No TIP4P parameter set or fixture exists. | TIP4P-Ew water model uses virtual-site geometry, produces correct geometry, and matches OpenMM energetics within tolerance. | AC-02 |
| P3-VS-05 | Topology virtual-site fields | `topology.py` `Topology` has no virtual-site fields. | `Topology` accepts virtual-site definitions; parsers populate them for supported water models. | AC-01, AC-02 |
| P3-VS-06 | Artifact virtual-site surface | `artifacts.py` blocks `virtual_site` terms and `tip4p`/`opc`/`advanced_water` models. | Artifact validation accepts declared virtual-site terms and water models; undeclared or unsupported types remain blocked. | AC-01, AC-02 |

## Custom Force Expressions

| Gap ID | Area | Current Evidence | Required Closure | Acceptance Link |
|--------|------|------------------|-------------------|-----------------|
| P3-CF-01 | Custom force expression | `forcefields.py` has no `CustomForcePotential` or expression parser. | `CustomForcePotential` accepts a symbolic expression string, per-particle or per-pair parameters, and computes energy and forces matching finite-difference reference. | AC-03 |
| P3-CF-02 | Custom force artifact | `artifacts.py` has no `custom_force` term. | Artifact validation accepts `custom_force` terms with expression metadata; `build_mlx_system_from_artifact` constructs `CustomForcePotential` from artifact data. | AC-03 |

## GBSA/OBC Implicit Solvent

| Gap ID | Area | Current Evidence | Required Closure | Acceptance Link |
|--------|------|------------------|-------------------|-----------------|
| P3-GB-01 | GB energy/forces | `forcefields.py` has no implicit solvent term. `artifacts.py` blocks `gbsa`. | GB-OBC energy and forces computed on MLX for neutral and charged periodic fixtures, matching analytical reference within tolerance. | AC-04 |
| P3-GB-02 | ACE surface area | No surface-area calculation exists on MLX. | ACE surface-area approximation computed efficiently; energy contribution validated against analytical surface integral within tolerance. | AC-04 |
| P3-GB-03 | GB artifact | `artifacts.py` blocks `gbsa`. | `GBSAForcePotential` is a supported artifact term; artifact load/save preserves GB parameters. | AC-04 |

## Soft-Core Potentials and Lambda Scaling

| Gap ID | Area | Current Evidence | Required Closure | Acceptance Link |
|--------|------|------------------|-------------------|-----------------|
| P3-SC-01 | Soft-core LJ | `nonbonded.py` `NonbondedPotential` has no soft-core path. LJ uses hard-core `1/r^6` throughout. | Soft-core LJ potential with `lambda_lj` scaling produces finite energies at `r=0` for `lambda < 1` and matches hard-core LJ at `lambda = 1`. | AC-05 |
| P3-SC-02 | Soft-core Coulomb | `nonbonded.py` has no soft-core electrostatics path. | Soft-core Coulomb with `lambda_electrostatics` scaling produces finite charges at `lambda = 0` and matches standard Coulomb at `lambda = 1`. | AC-05 |
| P3-SC-03 | Lambda-scaled nonbonded | `NonbondedPotential` has no lambda parameter. | `NonbondedPotential` accepts lambda parameters; energy and forces interpolate smoothly between end states; dU/dlambda is available for thermodynamic integration. | AC-05 |
| P3-SC-04 | Lambda artifact | No artifact surface for lambda scaling or soft-core parameters. | Artifact validation accepts `soft_core_lj` and `lambda_scaled_nonbonded` terms with lambda parameters. | AC-05 |

## Replica Exchange

| Gap ID | Area | Current Evidence | Required Closure | Acceptance Link |
|--------|------|------------------|-------------------|-----------------|
| P3-RE-01 | Multi-copy driver | `md.py` drives single-trajectory simulation only. No replica container, swap logic, or Hamiltonian exchange path. | Replica exchange driver manages N replicas with distinct temperatures or Hamiltonians; swap attempts occur at specified intervals; acceptance probability matches the Metropolis criterion. | AC-06 |
| P3-RE-02 | Temperature ladder | No temperature schedule or Hamiltonian schedule exists. | Temperature ladder and Hamiltonian schedule are configurable; replica exchange produces correct Boltzmann sampling across temperatures (verified by energy histogram overlap). | AC-06 |
| P3-RE-03 | Replica artifact | No artifact surface for replica-exchange configurations. | Artifact validation accepts `replica_exchange` configurations; prepared-system metadata includes replica parameters. | AC-06 |

## Production Hardening (Phase 4)

| Gap ID | Area | Current Evidence | Required Closure | Acceptance Link |
|--------|------|------------------|-------------------|-----------------|
| P4-CI-01 | CI pipeline | No `.github/workflows/` directory exists. | GitHub Actions CI runs `pytest` and `ruff check` on every PR; test failures and lint errors block merge. | AC-07 |
| P4-CI-02 | Version | `pyproject.toml` has `version = "0.1.0"`. | Version bumped to next 0.x minor. | AC-07 |
| P4-CI-03 | Docs | 12+ markdown docs exist but are not linked from README or organized. | Docs are restructured and linked from README; new capabilities have minimal API docs. | AC-07 |