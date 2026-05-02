# DESIGN: PME/Ewald Periodic Electrostatics

## Current State

`NonbondedPotential` currently evaluates LJ + Coulomb through direct pair, dense, and tiled paths. Periodic electrostatics uses minimum-image direct Coulomb with optional cutoff/shift, not Ewald or PME. Prepared artifacts therefore fail closed on `pme_ewald_periodic_electrostatics`.

## Design Direction

Add periodic electrostatics as an explicit mode, not as a silent behavior change.

```text
prepared artifact metadata
  -> electrostatics mode validation
  -> NonbondedPotential
  -> short-range LJ/exceptions + electrostatics backend
  -> diagnostics / benchmarks
```

Modes:

- `cutoff`: existing direct short-range Coulomb path.
- `ewald_reference`: real-space erfc term + reciprocal-space k-sum + self correction for neutral orthorhombic cells.
- `pme`: reserved until particle-mesh assignment, FFT solve, interpolation, and force path are implemented.

## Ewald Reference Contract

Supported first:

- orthorhombic cells;
- neutral systems within tolerance;
- real units already supported by `MDUnitSystem`;
- topology exclusions and 1-4 exceptions;
- energy/forces with per-term diagnostics.

Rejected first:

- triclinic cells;
- non-neutral systems unless an explicit neutralizing-background policy is added later;
- virtual sites/HMR requirements;
- PME mesh labels without mesh implementation.

## Force Accounting

Ewald Coulomb must not double-count bonded exceptions:

- excluded pairs are excluded from normal real-space Ewald;
- 1-4 exceptions remain explicit pair corrections using artifact exception parameters;
- LJ remains on the existing short-range path.

## Validation Strategy

Small fixtures are the source of truth first:

- two/three charge neutral cells;
- translated/wrapped equivalents;
- small NaCl-like periodic box;
- tiny water/ion box after artifact import supports it.

Acceptance is finite forces, finite-difference agreement, no self-force, translational invariance, and convergence as `alpha`, real cutoff, and reciprocal cutoff are tightened.

## Runtime Strategy

Reference Ewald may be slower than direct cutoff. That is acceptable for correctness. PME mesh acceleration is only started after Ewald reference tests pass.
