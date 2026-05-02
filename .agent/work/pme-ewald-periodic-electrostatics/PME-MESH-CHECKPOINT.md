# PME Mesh Checkpoint

## Completed In This Change

- Added explicit electrostatics modes: `cutoff`, `ewald_reference`, and reserved `pme`.
- Added neutral orthorhombic Ewald reference Coulomb energy with real-space, reciprocal-space, and self terms.
- Added analytical Ewald reference forces validated against finite differences.
- Integrated `ewald_reference` into `NonbondedPotential` while keeping LJ on the existing short-range path.
- Preserved bonded-exception accounting: exclusions are removed from normal Ewald Coulomb and 1-4/explicit exceptions are applied as pair corrections.
- Wired prepared artifacts so `electrostatics=ewald_reference` can load only when cell lengths are finite and charges are neutral.
- Kept `electrostatics=pme` fail-closed with an explicit mesh-not-implemented error.
- Updated GPCRmd compatibility reporting so small soluble periodic fixtures can use Ewald reference, while full membrane targets still report mesh PME and non-electrostatic blockers.
- Added `mlx_atomistic.benchmarks.ewald_reference` for JSON/CSV-safe Ewald diagnostics.

## Current Electrostatics Boundary

`ewald_reference` is an internal correctness backend and future PME oracle. It is not the scalable PME implementation needed for GPCRmd-scale membrane/water systems.

Supported now:

- neutral systems;
- orthorhombic periodic cells;
- real-space Ewald term;
- reciprocal-space direct k-sum;
- self correction;
- finite energy/force diagnostics;
- exclusions and 1-4 exception corrections through `NonbondedPotential`.

Still rejected:

- `pme` / mesh PME labels;
- non-neutral periodic systems;
- triclinic cells;
- virtual-site water models;
- Drude/polarizable terms;
- CMAP force terms;
- NPT/barostat claims.

## Remaining PME Mesh Tasks

1. Charge assignment from particles onto a 3D mesh.
2. Mesh-size, spline-order, and alpha policy with explicit metadata.
3. Forward FFT of assigned charge density.
4. Reciprocal-space influence function.
5. Inverse FFT potential/field solve.
6. Particle force interpolation from the mesh.
7. PME energy decomposition compatible with existing diagnostics.
8. Self, exclusion, and 1-4 correction parity against Ewald reference.
9. Error-control tests comparing PME mesh against Ewald reference on neutral fixtures.
10. Performance benchmarks for mesh PME versus Ewald reference and direct cutoff.

## Remaining GPCRmd Blockers

For GPCRmd target 729 / PDB 5F8U, the exact blockers after this change are:

- mesh PME for the full periodic water/membrane system;
- membrane/lipid force-field terms;
- POPC topology and parameters;
- CHARMM CMAP terms;
- large periodic neighbor-list and nonbonded scaling;
- virtual-site or hydrogen-mass-repartitioning policy check for large time steps;
- NPT/barostat support if the selected protocol requires pressure coupling.

Full GPCRmd simulation is not supported yet. This change only removes the generic PME/Ewald blocker for small Ewald-reference-compatible fixtures.

## Next Recommendation

Start a new implementation slice for PME mesh only after keeping the current Ewald reference tests green. The first PME mesh acceptance target should be numerical agreement against Ewald reference on small neutral periodic fixtures, not GPCRmd runtime.
