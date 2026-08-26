# DFT General Periodic Cell Geometry

This document records the completed scientific and engineering contract for
Phase 3 of the [DFT roadmap](./dft-roadmap.md). It replaces the periodic
runtime's orthorhombic-only geometry assumptions with one full-rank cell model
while preserving the accepted orthorhombic numerical path.

## Objective

Allow the periodic PBE/GTH runtime to operate on any finite, right-handed
`3 x 3` cell matrix admitted by the project-level `Cell`. This phase changes
fixed-cell geometry only. It does not add stress or variable-cell relaxation.

The implementation must extend the existing `Cell`, `RealSpaceGrid`,
`ReciprocalGrid`, and `PeriodicDFTSystem` contracts. It must not introduce a
second DFT-specific cell type.

## Matrix Convention

The direct cell matrix `A` stores lattice vectors as rows. Fractional row
coordinates `s` map to Cartesian coordinates as

```text
r = s A
```

The reciprocal matrix is

```text
B = 2 pi (A^-1)^T
```

and therefore satisfies `A B^T = 2 pi I`. An integer FFT index row `n` maps to
the Cartesian reciprocal vector `G = n B`; a reduced k-point `k` maps to
`k_cartesian = k B`. Cell volume is `det(A)`, which is positive by the `Cell`
contract.

These conventions match the matrix-first reciprocal-grid construction used by
Quantum ESPRESSO and CP2K. Those projects remain design references only and do
not enter the MLX runtime path.

## Runtime Contract

`RealSpaceGrid` accepts either three orthorhombic lengths, a full matrix, or an
existing `Cell`. Grid points remain uniformly spaced in fractional coordinates:

```text
s_ijk = ((i + 1/2) / N1, (j + 1/2) / N2, (k + 1/2) / N3)
r_ijk = s_ijk A
```

`ReciprocalGrid` retains exact NumPy FFT integer ordering and maps those
integers through `B`. Plane-wave cutoffs continue to apply to
`0.5 |G + k|^2` in Cartesian reciprocal space.

`PeriodicDFTSystem` preserves the complete cell matrix across immutable
position updates, calculation fingerprints, SCF checkpoints, outer relaxation
checkpoints, and reports. APIs that historically accept three lengths continue
to do so.

## Electrostatics And Forces

The ion-ion Ewald implementation operates on direct and reciprocal lattice
matrices rather than component-wise lengths. Direct translations are integer
rows mapped through `A`; reciprocal vectors are integer rows mapped through
`B`. Enumeration bounds use lattice-vector and reciprocal-vector norms and
retain the spherical real- and reciprocal-space cutoff checks.

Local and nonlocal GTH operators already consume Cartesian reciprocal vectors
and cell volume. They require no cell-specific branch once their grids and
bases are correct. Analytic ionic forces must remain consistent with
finite-difference energy derivatives in a skew cell.

## Compatibility Boundary

The existing orthorhombic route is a compatibility oracle. Diagonal cell input
must retain its FFT ordering, reciprocal values, basis membership,
fingerprints, SCF energies, and forces within the existing locked tolerances.
The general formulas must not silently replace an orthorhombic fast path where
that would perturb an accepted float32 trajectory.

Unsupported work remains fail-closed. Phase 3 does not admit singular or
left-handed cells, variable FFT topology during one calculation, stress,
variable-cell optimization, crystallographic symmetry reduction, or slab and
isolated electrostatics.

## Acceptance Criteria

The implementation is complete only when all of the following pass:

- direct/reciprocal duality and fractional/Cartesian round trips for cubic,
  hexagonal, and low-symmetry cells;
- exact FFT integer ordering and correct Cartesian reciprocal vectors;
- determinant volume, uniform fractional coordinates, and density
  normalization;
- reduced k-point conversion through the reciprocal matrix;
- Ewald translation invariance and analytic-versus-finite-difference forces in
  a skew cell;
- periodic GTH local and nonlocal energy/force execution in a skew cell;
- SCF and checkpoint identity preserve the complete matrix;
- existing orthorhombic targeted tests remain unchanged;
- one source-bound hexagonal crystal and one bounded low-symmetry numerical
  case close their declared energy and force gates.

Only targeted unit and compatibility tests ran during implementation. The
material-level calculation ran after the runtime source and protocol were
frozen; remote CPU CI carries the full regression suite.

## Accepted Result

The current-verified material case is ideal 2H-Silicon in the lonsdaleite
`A_hP4_194_f-001` prototype. The `P63/mmc` four-atom cell uses `z = 1/16`, the
accepted Silicon lattice as its source scale, PBE-PW92, `Si GTH-PBE-q4`, a
`25 Ha` cutoff, a `40 x 40 x 64` FFT grid, and a `6 x 6 x 4`
Monkhorst-Pack mesh. The fixed cell remained immutable while the Phase 2
optimizer relaxed the symmetry-allowed internal coordinate.

The calculation converged in three accepted ionic steps, four SCF evaluations,
and three line-search evaluations. The final energy was
`-15.762112189664592 Ha`, maximum force was
`1.715483631414827e-5 Ha/bohr`, net-force norm was
`1.7555596748712108e-7 Ha/bohr`, and the final maximum step was
`1.6593486505125634e-4 bohr`. The maximum translation-aligned displacement
from the ideal source structure was `0.003351 A`.

Complete wall time was `28.744 s`; peak physical memory was `4.856 GB`, and
the memory plateau gate passed. The run used AC power with low-power mode
disabled. Its workload fingerprint is
`cebd61b0baeae25935cef6d27acdad46b2f9a455b0cc1c5de42839660ef74931`, and its
runtime fingerprint is
`625f216ce19def219758c2d49159667977619e302496c2cc050a0019a036387a`.

Hexagonal and low-symmetry numerical oracles additionally lock reciprocal
duality, FFT ordering, Ewald basis invariance, analytic force derivatives,
GTH lattice-translation invariance, and complete matrix identity in SCF state
metadata. The result establishes ordinary fixed full-rank cell execution; it
does not establish stress or variable-cell relaxation.

## Delivery

Delivered in this phase:

1. General real and reciprocal grid geometry and reduced k-point mapping.
2. Full-matrix `PeriodicDFTSystem` construction and immutable updates.
3. Full-rank periodic Ewald energy and analytic forces.
4. Matrix identity through SCF, forces, checkpoints, and reports.
5. Targeted orthorhombic compatibility and nonorthogonal correctness tests.
6. Source-bound 2H-Silicon evidence. Repository CI remains the merge gate.

## Out Of Scope

Stress, cell derivatives, cell optimization, coupled ion/cell relaxation,
phonons, spin, and new exchange-correlation or pseudopotential families remain
separate roadmap phases.
