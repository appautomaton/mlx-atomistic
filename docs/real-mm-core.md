# Real Molecular Mechanics Core

This milestone adds the first system-level molecular mechanics layer. The engine
can now represent typed systems, assign force-field parameters, run a combined
LJ+Coulomb nonbonded path, apply pair-distance constraints, and save/load simple
structures and trajectories.

## Systems And Parameters

`MMSystem` stores the molecular state:

```text
symbols, atom_names, atom_types
masses, charges
positions, velocities
topology, optional cell
```

`ForceField` assigns parameters from atom types:

- `AtomType`
- `NonbondedParameter`
- `BondParameter`
- `AngleParameter`
- `DihedralParameter`

Bond parameters match atom-type pairs without order sensitivity. Angle and
dihedral parameters match forward or reverse type tuples. Missing parameters
raise errors with atom indices and type tuples so mistakes are visible.

## Production Nonbonded Path

`NonbondedPotential` combines Lennard-Jones and direct Coulomb terms in one pair
loop. It supports:

- per-atom `σ`, `ε`, and charge
- Lorentz-Berthelot mixing
- topology exclusions
- independent LJ and Coulomb 1-4 scaling
- cutoff and minimum-image behavior
- optional LJ and Coulomb energy shifts

NVE/NVT diagnostics record component terms as:

```text
nonbonded.lj
nonbonded.coulomb
```

This gives a realistic target for the future custom Metal pair kernel.

## Constraints

`DistanceConstraints` supports fixed pair distances. The current implementation
uses iterative position projection and velocity projection:

- position correction after position updates
- velocity correction after force/velocity updates
- dense `constraint_max_error` diagnostics for every step

This is intentionally limited to pair-distance constraints. It is enough for
water-like toy systems and for measuring constraint overhead.

## I/O

The I/O layer is deliberately small:

- `read_xyz(...)`
- `write_xyz(...)`
- `save_npz_trajectory(...)`
- `load_npz_trajectory(...)`
- `restart_state_from_trajectory(...)`

XYZ is for quick structure interchange. NPZ is the native trajectory format and
stores sampled frames, scalar diagnostics, energy decomposition, optional cell,
symbols, and JSON metadata.

## Examples

Programmatic examples now include:

- water-like constrained molecule
- butane-like torsion system
- small ionic cluster
- mixed typed LJ fluid

The notebook `notebooks/05-real-mm-core.ipynb` demonstrates the full workflow:
system construction, force-field assignment, constraints, I/O round trip,
restart, diagnostics, and benchmark rows.

## Remaining Boundaries

Still out of scope:

- PME/Ewald
- NPT/barostat
- full AMBER/CHARMM/GROMACS parser
- DFT
- custom Metal kernels

The next optimization decision should be based on benchmark evidence from the
combined nonbonded and constraint paths.
