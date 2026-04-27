# Molecular Mechanics Core

The MD engine now supports a programmatic molecular mechanics surface. It is not
a full AMBER/CHARMM/OpenMM replacement yet, but it is no longer limited to an LJ
fluid demo.

## Topology

`Topology` stores validated atom connectivity:

```text
bonds      shape (n_bonds, 2)
angles     shape (n_angles, 3)
dihedrals  shape (n_dihedrals, 4)
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
- `HarmonicBondPotential`
- `HarmonicAnglePotential`
- `PeriodicDihedralPotential`

All terms implement `energy_forces(positions, cell=None, pairs=None)` and can be
passed together to `simulate_nve()` or `simulate_nvt()`.

## Nonbonded Rules

LJ and Coulomb support topology-aware nonbonded pairs:

- bonded-pair exclusions
- explicit exclusions
- optional 1-4 scaling
- direct cutoff support
- orthorhombic periodic minimum-image behavior

PME/Ewald and force-field file parsers are intentionally out of scope for this
milestone.

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
