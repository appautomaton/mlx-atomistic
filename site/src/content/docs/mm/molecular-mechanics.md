---
title: "Molecular Mechanics Core"
---


The MD engine now supports a programmatic molecular mechanics surface. It is not
a complete AMBER/CHARMM feature clone yet, but it is no longer limited to an LJ
fluid demo.

## Topology

`Topology` stores validated atom connectivity:

```text
bonds      shape (n_bonds, 2)
angles     shape (n_angles, 3)
dihedrals  shape (n_dihedrals, 4)
impropers  shape (n_impropers, 4)
exclusions shape (n_exclusions, 2)
charges    shape (n_atoms,)
```

`Topology.from_sequences(...)` is the intended v1 construction helper. Bonded
pairs are excluded from nonbonded interactions by default. Dihedral endpoints are
used as default 1-4 pairs for optional LJ/Coulomb scaling.

## Force Terms

Supported terms:

- `LennardJonesPotential`
- `CoulombPotential`
- `NonbondedPotential`
- `HarmonicBondPotential`
- `HarmonicAnglePotential`
- `PeriodicDihedralPotential`
- `ImproperDihedralPotential`
- `RBDihedralPotential`
- `PositionalRestraintPotential`

All terms implement `energy_forces(positions, cell=None, pairs=None)` and can be
passed together to `simulate_nve()` or `simulate_nvt()`.

## Nonbonded Rules

LJ and Coulomb support topology-aware nonbonded pairs:

- bonded-pair exclusions
- explicit exclusions and nonbonded exception overrides
- optional 1-4 scaling / overrides
- direct cutoff support
- orthorhombic periodic minimum-image behavior

`PMEConfig` supports PME assignment orders `2`, `4`, and `5`. Prepared artifacts
must carry explicit PME arrays/metadata before the runtime PME path is accepted;
unsupported or partial PME requests still fail closed.

PME charge handling is explicit:

- `reject_non_neutral` is the default and the backward-compatible behavior for
  existing artifacts without a policy field;
- `uniform_neutralizing_plasma` opts into the OpenMM-compatible scalar
  background correction for charged periodic systems;
- unknown policies, metadata/array disagreement, and charged reject-mode
  artifacts are errors.

`PMEExecutionPlan` owns the reciprocal state that is invariant for a fixed
cell. A plan has an inspectable fingerprint, setup/reuse counters, estimated
resident bytes, and strict validation across cell, mesh, alpha, cutoff,
assignment order, deconvolution, Coulomb constant, dtype/backend/device, and
background policy. `NonbondedPotential.bind_pme_plan(...)` binds one plan to
all PME force scopes; unbound direct API calls retain one-shot behavior. No
process-global PME cache is used.

Production fixed-cell PME is currently admitted only for supported
orthorhombic configurations within the measured 100,000-atom and 1,048,576
mesh-point checks. Large lazy-topology PME uses shared
`mlx_cell_blocks`/`NeighborBlocks` for LJ and direct-space Coulomb and refuses a
dense fallback. Two measured workloads now anchor this boundary:

- the 94,232-atom charged AMBER20 JAC envelope with explicit
  `uniform_neutralizing_plasma`, documented in
  [`scalable-charged-pme-runtime-m5max.md`](../benchmarks/scalable-charged-pme-runtime-m5max.md);
- the neutral 92,001-atom GPCRmd 729 CHARMM membrane fixture with
  `reject_non_neutral`, independent parity, bounded source-protocol NVT, and
  checkpoint restart, documented in
  [`gpcrmd-729-pme-runtime-m5max.md`](../benchmarks/gpcrmd-729-pme-runtime-m5max.md).

These are workload-specific fixed-cell results. Production NPT/cell changes,
analytic PME virial, triclinic PME, production-length stability, and broad
membrane readiness remain outside this boundary.

`NonbondedPotential` is the production-oriented direct pair path. It combines
mixed LJ and direct Coulomb terms, supports explicit nonbonded exceptions and
independent LJ/Coulomb 1-4 scaling, and reports component energies as
`nonbonded.lj` and `nonbonded.coulomb` in MD diagnostics.

## Force-Field Imports

The optional prep layer exposes native accepted-subset importers:

```python
from mlx_atomistic.prep import (
    import_amber_prmtop,
    import_charmm_psf,
    import_gromacs_top_gro,
)
```

These importers produce prepared-system artifacts for AMBER `prmtop`/`inpcrd`,
CHARMM PSF/parameter, and GROMACS `.top`/`.gro` inputs. The production artifact
gate remains fail-closed for unsupported terms such as virtual sites, advanced
water models, polarizable terms, or incomplete PME metadata.

## Energy Decomposition

NVE and NVT results expose:

```text
potential_energy
potential_energy_by_term
```

`potential_energy` is the total potential energy. `potential_energy_by_term` maps
term names such as `bond`, `angle`, `dihedral`, `coulomb`, and `lj` to dense
per-step energy series.

PME component diagnostics additionally expose real, reciprocal, self,
exclusion/exception, and named `nonbonded.coulomb_background` contributions.
For fixed cell the uniform-background component changes scalar energy and has
zero coordinate force; it must not be interpreted as an analytic virial.

The diagnostic helper `summarize_md_result()` includes final and mean term-energy
summaries for notebook and benchmark output.
