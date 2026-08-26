# DFT Metallic Validation

This document is the completed scientific specification and evidence summary
for Phase 1 of the [DFT roadmap](./dft-roadmap.md). It closes one
material-level metallic validation without expanding the runtime beyond the
existing unpolarized PBE/GTH and Fermi-Dirac capability.

## Objective

Promote fcc Aluminum from an execution smoke test to a source-bound metallic
equation-of-state validation. The accepted workload must exercise a global
chemical potential, fractional occupations, electronic entropy, and the
Helmholtz free energy `F = E - TS` over a converged weighted k-point mesh.

The work must also make the dense metallic mesh practical. It may add general
k-point construction and explicit symmetry-reduction utilities, but it must
not add a reference engine, automatic space-group detection, a C++ extension,
or a new dependency.

## Source Contract

The primary source is the Materials Cloud ACWF verification dataset, record
`yf0rj-w3r97`, DOI `10.24435/materialscloud:s4-3h`. The downloaded archive has
MD5 `6bd97a883b439507d0be4638c1bc7514`. Only the compact Aluminum values and
their provenance are committed; the source archive remains external.

The exact source records are:

- `Al-FCC.xsf`, SHA-256
  `c4446a6d46475e4cca8067169330284c17c031b97477010b2a433bbf85be25ac`;
- `results-unaries-verification-PBE-v1-AE-average.json`, SHA-256
  `d7844caa127eae860fe5087ead42f80d1b6b5eb952a686ff4b912ddaed7db48b`;
- `results-unaries-verification-PBE-v1-cp2k_TZV2P.json`, SHA-256
  `3a787c7197cc50e003b38494c4b489b72eeb553e8ca0da3a4505ed08f96e9b6e`.

The all-electron FLEUR/WIEN2k average is the primary scientific target:

- equilibrium volume: `16.49535905981626 A^3/atom`;
- conventional fcc lattice: `4.040861093109186 A`;
- bulk modulus: `0.4837908557795412 eV/A^3`;
- bulk derivative: `4.623179033235038`.

CP2K Quickstep is the same-pseudopotential-family diagnostic. Its
`TZV2P-MOLOPT-PBE-GTH-q3` basis uses `Al GTH-PBE-q3` and gives:

- equilibrium volume: `16.437017660642258 A^3/atom`;
- conventional fcc lattice: `4.036091509687826 A`;
- bulk modulus: `0.48951774655729935 eV/A^3`;
- bulk derivative: `4.514335571200874`.

Against the all-electron target, that CP2K fit has a project Delta factor of
`0.9954359681 meV/atom` and passes the existing excellent thresholds. Absolute
energies are not comparable because CP2K uses a Gaussian orbital basis while
the MLX runtime uses plane waves.

The historical protocol is reconstructed from the ACWF 1.0 release line. The
`v1.0.1` tag resolves to commit
`c08c3f2f7babcb78c7b0a1ddaa28f2fb1d0d8d39` and records PBE, a
`0.06 A^-1` maximum k-point spacing, Fermi-Dirac smearing at `710.5 K`, 20
additional molecular orbitals, and `Al GTH-PBE-q3`. This reconstruction is
protocol evidence, not a substitute for the source result files above.

## Local Protocol

The source primitive cell is represented locally as the equivalent four-atom
conventional cubic cell because the current DFT grid is orthorhombic. The
seven volumes are the source lattice scaled by volume factors
`0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06` around
`4.040422065345 A`.

The locked physics is:

- unpolarized PBE-PW92;
- `Al GTH-PBE-q3`, extracted and fingerprinted from the CP2K database;
- 12 valence electrons per conventional cell;
- Fermi-Dirac width `0.00225 Ha`, equivalent to the ACWF electronic
  temperature within the recorded precision;
- the stationary Helmholtz free energy as the EOS energy;
- a Gamma-centered mesh with maximum conventional-cell reciprocal spacing no
  greater than `0.06 A^-1`.

A numerical CP2K GPW grid cutoff is not transferable to a plane-wave kinetic
cutoff. The local cutoff, FFT shape, band capacity, and k-point density are
therefore admitted by representation-appropriate convergence gates rather
than by copying unlike numeric inputs.

## Efficient K-Point Contract

The runtime will retain explicit weighted `KPointMesh` as the integration
contract and add two composable utilities:

1. a Gamma-centered regular grid, distinct from the existing even half-shifted
   `MonkhorstPackGrid`;
2. deterministic reduction by caller-supplied reciprocal-space symmetry
   operations.

The reducer validates finite reduced coordinates, integer unimodular
operations, mesh closure, unique transformed matches, and equal weights within
each orbit. It retains the full orbit mapping used by SCF density
reconstruction and never infers that a symmetry is valid for a Hamiltonian.
Workload schema v2 persists that mapping so reloading a mesh cannot silently
degrade it to scalar weight aggregation.

For conventional fcc Aluminum, the 48 signed permutation operations of the
cubic point group reduce the source-density `27 x 27 x 27` Gamma-centered mesh
from 19,683 explicit points to 560 weighted representatives, a 97.2% lane
reduction. The odd mesh keeps the maximum reciprocal spacing below
`0.06 A^-1` at every EOS volume. A full-mesh path remains available as the
correctness oracle.

## Acceptance Criteria

Numerical admission requires every retained point to pass:

- converged SCF and finite free energy, internal energy, entropy, and chemical
  potential;
- electron-count error no greater than `1e-4` per cell;
- maximum orbital residual no greater than `1e-6`;
- maximum orthonormality error no greater than `1e-4`;
- maximum occupation of the highest computed band no greater than `1e-6`;
- consistency of `F = E - TS` within `5e-6 Ha`.

Cutoff and k-point admission reuse the existing EOS convergence thresholds:
maximum curve change `1 meV/atom`, lattice change `0.1%`, bulk-modulus change
`3%`, and bulk-derivative change `10%`. The final seven-point fit must pass the
existing verified material thresholds: Delta no greater than `3 meV/atom`,
lattice error `0.5%`, bulk-modulus error `10%`, and bulk-derivative error `15%`.
Thresholds are fixed before the production run and are not relaxed afterward.

The symmetry path must reproduce a full-grid invariant quadrature exactly and
a bounded full-versus-reduced SCF within the established numerical gates. The
final report records complete wall time, peak memory, explicit point count,
representative count, and work counters. Existing fixed-occupation and
time-reversal tests must not regress.

## Accepted Result

The admitted profile is `c15-k15-b11`: a `15 Ha` plane-wave cutoff, `36 x 36 x
36` FFT grid, `15 x 15 x 15` Gamma-centered k-point mesh reduced to 120 weighted
representatives, 11 bands, and the locked `0.00225 Ha` Fermi-Dirac width. The
band-capacity gate was evaluated on the denser locked `27 x 27 x 27` mesh at
the largest EOS volume. Ten bands failed because the highest occupation was
`2.93e-6`; 11 bands passed at `1.30e-18`.

The selected seven-volume fit gives:

- conventional fcc lattice: `4.039885108 A`;
- bulk modulus: `76.630636 GPa`;
- bulk derivative: `4.583841965`;
- Delta factor against the all-electron reference: `0.229587 meV/atom`.

The corresponding relative errors are `0.0242%` for the lattice, `1.14%` for
the bulk modulus, and `0.851%` for the bulk derivative. All locked numerical,
convergence, and scientific gates pass. A bounded `4 x 4 x 4` full-grid oracle
and its ten-representative symmetry reduction differ by `8.45e-6 Ha/atom`,
below the fixed `5e-5 Ha/atom` gate.

On an Apple M5 Max connected to AC power with Low Power Mode disabled, the
accepted seven-point curve took `48.03 s` complete wall time. Individual
points took `6.37-7.62 s`; maximum process physical memory was `3.06 GB`.
These measurements are current-verified for this workload and power state,
not a cross-device performance claim. Raw reports remain under gitignored
`results/`.

## Delivery Plan

1. Commit the compact, hash-guarded Aluminum reference bundle and portable
   workload preparation contract.
2. Add and test Gamma-centered meshes and explicit reciprocal-symmetry
   reduction without changing existing `MonkhorstPackGrid` behavior.
3. Add a bounded Aluminum point runner and fail-early admission ladder for band
   capacity, cutoff, k-point density, and the seven-volume EOS.
4. Run the ladder on Metal, retain only generated evidence under `results/`,
   and promote the smallest profile that passes every locked gate.
5. Commit the accepted scientific summary, runtime measurement, known
   boundary, and roadmap status; keep raw calculations gitignored.

All five delivery steps are complete.

## Out Of Scope

This phase does not add spin polarization, general cells, stress, ionic
relaxation, projected observables, new exchange-correlation functionals,
automatic crystal-symmetry discovery, or broad Aluminum chemistry. Those
claims remain governed by later roadmap phases.
