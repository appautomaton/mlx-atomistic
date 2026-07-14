# Molecular Mechanics Core

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

The validated PME production envelope currently reaches 24,488 atoms for a
neutral orthorhombic TIP3P/0.15 M NaCl fixture with a 9 Angstrom cutoff,
OpenMM-resolved `56 x 56 x 56` mesh, alpha `0.292028987 Angstrom^-1`, and
fifth-order assignment. The production direct-space path uses
`mlx_cell_pairs` with no dense or tiled fallback. See
[the DHFR-scale neutral PME report](./benchmarks/dhfr-scale-neutral-pme-validation-m5max.md).
This boundary does not cover non-neutral background conventions, triclinic
cells, analytic PME virial/NPT, or a protein/membrane production workflow.

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

The diagnostic helper `summarize_md_result()` includes final and mean term-energy
summaries for notebook and benchmark output.
