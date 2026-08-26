# DFT Core Observables And Convergence

This document is the scientific and engineering contract for Phase 6 of the
[DFT roadmap](./dft-roadmap.md). It adds analysis and reproducibility surfaces
above converged periodic SCF results without changing the Hamiltonian or SCF
controller.

## Density Of States

The public density-of-states result uses one shared energy grid and reports:

- state DOS, whose integral is the computed spin-degenerate or spin-resolved
  state capacity;
- occupation-weighted DOS, whose integral is the SCF electron count;
- separate up/down channels when collinear spin is active;
- the exact Gaussian broadening and energy-window convention;
- shared, per-channel, or occupied-edge Fermi-level metadata.

For Fermi-Dirac calculations, the SCF chemical potential is authoritative.
Fixed-occupation calculations without an empty computed band report the
highest occupied computed state, not an invented mid-gap value. Projected DOS
is outside this phase.

`periodic_density_of_states` returns `PeriodicDOSResult`; each physical
channel is a `PeriodicDOSChannel`. The sampled arrays remain MLX arrays so
downstream analysis can stay on the selected MLX device.

## Periodic Density Volume

The portable volume is one compressed NPZ file decoded with
`allow_pickle=False`. It stores the full-rank cell matrix in bohr, fractional
grid order, charge density in electrons per bohr cubed, optional
magnetization density in the same units, electron and moment integrals,
symbols, positions, and the periodic-system fingerprint.

Publication is atomic through a same-directory temporary file. Loading checks
the schema, dtypes, shapes, finite values, positivity of charge density,
cell volume, and declared normalization. The format is a deterministic data
interchange surface; it is not a checkpoint and contains no orbitals or mixer
state.

`write_periodic_density_volume` refuses to overwrite an existing destination.
`read_periodic_density_volume` validates the exact array inventory and dtypes
before constructing an immutable `PeriodicDensityVolume`.

## Reusable Convergence Reports

One axis-agnostic comparison contract covers cutoff, k-point, smearing, and
future numerical axes. A report binds two exact calculation identities,
parameter values, declared observables, absolute and relative tolerances, raw
signed differences, and pass/fail status. Missing or non-finite observables
fail closed.

Material runners may add scientific interpretation, but they must not
reimplement the numerical comparison or discard a failed criterion. A
smearing study compares the same thermodynamic observable, normally Helmholtz
free energy, at two declared widths.

`compare_periodic_convergence` consumes two `PeriodicConvergencePoint` values
and unique `PeriodicConvergenceCriterion` values. Every point carries separate
calculation, source, and runtime SHA-256 fingerprints. The same comparator
accepts scalar cutoff and smearing values or vector k-point meshes; missing or
non-finite observables raise instead of producing a passing report.

## Acceptance Criteria

- state and occupation-weighted DOS integrals recover their expected counts;
- scalar and collinear-spin DOS use one result contract;
- fixed and smeared Fermi conventions are explicit and deterministic;
- charge and magnetization volumes round-trip cell, grid, units, and integrals;
- malformed or mismatched volumes fail closed without pickle decoding;
- cutoff, k-point, and smearing examples use the same convergence comparator;
- deterministic tests cover all semantics before any material workflow is
  rerun.

## Evidence Boundary

The deterministic gates establish observable semantics and artifact
integrity. Existing Silicon, Aluminum, and Iron reports may be summarized
through the new contracts after implementation; they are not rerun merely to
exercise formatting code. No claim of projected-state, charge-partitioning,
or broad spectroscopy accuracy is introduced.
