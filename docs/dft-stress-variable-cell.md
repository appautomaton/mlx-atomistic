# DFT Stress And Variable-Cell Relaxation

This document is the scientific and engineering contract for Phase 4 of the
[DFT roadmap](./dft-roadmap.md). It adds periodic stress and restartable cell
optimization on the full-rank cell foundation delivered by Phase 3.

## Objective

Provide a reliable periodic stress tensor and use it to optimize cell geometry
at fixed external pressure. The implementation must support cell-only and
coupled ion/cell workflows without weakening SCF convergence, force, identity,
or artifact-integrity requirements.

The legacy `finite_difference_stress` teaching surface is not the controller
for this phase. Its orthorhombic diagonal scope and historical pressure sign
remain a separate proof-level boundary.

## Stress Convention

The public periodic tensor is compression-positive, matching the convention
printed by Quantum ESPRESSO and used as an internal pressure tensor by CP2K.
For a symmetric Cartesian strain `epsilon` applied to row-vector cells and
positions as

```text
A' = A (I + epsilon)
r' = s A'
```

the tensor is

```text
sigma_ij = -(1 / V) dF / d epsilon_ij
pressure = trace(sigma) / 3
```

where `F` is the converged SCF total energy for fixed occupations and the
Helmholtz electronic free energy when Fermi-Dirac smearing is active. Units are
Hartree/bohr cubed; reports also expose GPa.

The six independent components use Voigt order `xx, yy, zz, yz, xz, xy`.
Off-diagonal perturbations use engineering shear: the two symmetric matrix
entries receive half the scalar perturbation, so the resulting derivative maps
directly to one tensor component without an implicit factor of two.

## Numerical Stress Oracle

The first retained implementation is a frozen-variational central-difference
oracle around one converged periodic SCF state. Electronic stationarity means
the first-order cell derivative can hold the converged electronic state fixed.
The runtime transports its compact orbitals by exact integer-G identity,
rebuilds their density, and reevaluates the complete energy functional on each
strained cell without starting a new SCF. It supports three explicit strain
modes:

- `isotropic`: two strained energy evaluations and one hydrostatic pressure;
- `diagonal`: six strained energy evaluations and three normal stresses;
- `symmetric`: twelve strained energy evaluations and the complete symmetric
  tensor.

All strained evaluations keep fractional ion coordinates, FFT shape,
pseudopotentials, electron count, reduced k-points, bands, functional, and
occupations fixed. They remap the base compact orbitals, rebuild and normalize
their density, then reevaluate kinetic, local and nonlocal GTH, Hartree,
exchange-correlation, and Ewald terms. The base frozen functional must reproduce
the converged SCF energy within a locked tolerance before any derivative is
reported. The frozen derivative is evaluated at both the requested strain and
twice that strain; their stress values must agree within a locked tolerance.
This multiscale gate rejects basis-set cusps and other non-smooth cell energies
before they can enter an accepted optimization trajectory. A reconverged
finite-difference response remains an explicit diagnostic mode, not the
production default.

The exact active integer plane-wave set from the base state is transported to
the plus and minus cells for every k-point. This fixed-topology derivative avoids
cutoff crossings without shrinking the strain until SCF noise overwhelms the
energy signal. A transported set that is not representable or is not preserved
fails closed. Every retained sample must converge and remain finite. Effective
strain, wall time, and SCF work counts are reported explicitly.

This oracle establishes semantics and validates later analytic stress. An
analytic implementation may replace its cost only after every energy
contribution agrees with the oracle, including kinetic, Hartree, PBE gradient,
local and nonlocal GTH, Ewald, occupation, and finite-cutoff effects.

## Analytic Stress Contract

The production path differentiates the frozen stationary energy functional
with respect to one symmetric Cartesian strain tensor. The strain is an MLX
value, while the FFT shape, active integer-G sets, reduced k-points,
occupations, pseudopotentials, and Ewald summation topology remain fixed. For a
deformation `F = I + epsilon`, the differentiable geometry is

```text
A' = A F
G' = G F^-T
V' = V det(F)
rho'(s) = rho(s) / det(F)
```

This produces the electronic first derivative directly from the existing
energy definitions. It does not subtract nearby float32 total energies. The
MLX graph contains kinetic, local GTH, nonlocal GTH, Hartree, and PBE
exchange-correlation including its density-gradient response. The ion-ion
Ewald tensor is evaluated analytically with float64 real and reciprocal sums
and added once. The converged electronic coefficients and occupations are held
fixed because their first-order response vanishes at a stationary free-energy
solution.

The graph rebuilds its density from the retained compact orbitals rather than
mixing those orbitals with the final SCF input density. Its zero-strain energy
is anchored to the admitted frozen-functional value by a strain-independent
correction. That correction is reported and fails closed above
`1e-3 Ha/electron`; it cannot alter stress.

The analytic graph must reproduce the base SCF energy within the existing
variational tolerance before its derivative is admitted. Every electronic term
is tested against stable isotropic and shear central differences on a smooth
bounded system; the float64 Ewald tensor has its own full-rank cell derivative
gate. The total tensor must agree with the retained frozen oracle wherever that
oracle passes its multiscale gate. Unsupported exchange-correlation functionals
fail closed rather than silently omitting their cell derivative.

Fixed integer-G differentiation removes cutoff-crossing discontinuities, but
it does not claim that a finite plane-wave cutoff is complete. Material
admission therefore compares analytic stress at the locked production cutoff
and a higher cutoff. A trajectory is not verified when this Pulay convergence
check exceeds its declared tolerance.

## Current Evidence

Deterministic CPU tests recover analytic diagonal and shear tensors, preserve
translation and equivalent-cell invariance, reject topology and multiscale
drift, converge cell-only and coupled elastic models, and reproduce
uninterrupted results across an accepted-cell checkpoint.

The source-bound 2H-Silicon material gate is closed. On Apple M5 Max, AC power,
and normal power mode, the cell started at `0.995` of the accepted scale,
accepted one cell step, and converged at scale `0.9981142`. The final pressure
was `2.66553e-6 Ha/bohr³`, the source-lattice relative error was `0.1886%`, and
cell relaxation took `22.707 s`. The complete validation, including the
higher-cutoff check, took `30.956 s`.

The final 25 Ha stress was compared with a fresh 35 Ha SCF and analytic stress
on the same cell. The higher-cutoff pressure was
`2.88255e-6 Ha/bohr³`; their `2.17019e-7 Ha/bohr³` difference passed the locked
`5e-6 Ha/bohr³` Pulay gate in `8.248 s`. The final frozen-functional error was
`4.64563e-7 Ha`, and the reported stationary correction was
`0.00581397 Ha`, or `3.63373e-4 Ha/electron`.

The current runtime, workload, and implementation fingerprints are
`3f70bd6eb973feabf80ab8ba949975c1e73199528e0237791f3be601496f6f60`,
`44353da8e872b4d270bd9e2ad9b15662b14570a733da58cd16491de2d982d205`, and
`a96880a3ee3a73ee0516a5f7d972a8eb6922db751fec70bbf7fbcd450633e3b2`.
This verifies the locked 2H-Silicon path, not broad stress or cell-relaxation
transferability.

## Variable-Cell Workflow

The workflow minimizes generalized enthalpy

```text
H = F + P_external V
```

with scalar external pressure. The cell descent residual is
`sigma - P_external I`. A bounded compliance maps that residual to a trial
strain, and backtracking accepts only a converged finite state satisfying an
enthalpy Armijo condition. Trial cells must remain right-handed and above a
minimum volume ratio.

The integer-G topology selected at the initial cell remains fixed for the
complete trajectory. Candidate SCFs and nested fixed-cell ionic relaxations
remap accepted compact orbitals into that topology. This avoids variational
energy jumps when individual plane waves cross the nominal kinetic cutoff;
the physical kinetic energies still change with the cell.

Supported modes are:

- `cell`: optimize the cell while holding fractional ion coordinates fixed;
- `ions_and_cell`: alternate bounded fixed-cell ionic relaxation with one
  accepted cell step until force, stress, ionic displacement, and strain gates
  all pass.

The workflow supports isotropic, diagonal, and symmetric cell freedom. It does
not silently reduce a requested tensor mode. Accepted cell steps retain the
SCF density and compact orbitals for the next ionic or stress evaluation;
rejected cells cannot seed later work.

## Checkpoint Contract

An outer checkpoint is published only after an accepted cell step. It stores
the accepted cell matrix, fractional and Cartesian positions, density seed,
energy or free energy, stress, force state when present, accepted history,
work counters, and next-step cursor.

Resume binds the original system, FFT shape, pseudopotentials, electron count,
k-points, bands, functional, SCF controls, stress controls, optimizer controls,
and external pressure. The trajectory basis is reconstructed deterministically
from that original calculation contract. Inner SCF, fixed-cell ionic, and
variable-cell checkpoints remain distinct artifact types.

## Acceptance Criteria

The phase closes only when all of the following pass:

- quadratic analytic energy models recover exact diagonal and shear stress;
- isotropic, diagonal, and symmetric modes agree on shared components;
- stress is invariant to lattice translation and equivalent cell bases;
- analytic periodic forces and numerical stress use one converged free-energy
  state and preserve electron count;
- cutoff crossings, inconsistent strain scales, unconverged samples, singular
  cells, and identity drift fail closed;
- cell-only and coupled workflows converge deterministic harmonic or elastic
  oracles;
- checkpoint/resume reproduces uninterrupted status, accepted-step count,
  cell, positions, and enthalpy within locked tolerances;
- existing fixed-cell and orthorhombic trajectories do not regress;
- one source-bound Silicon workload recovers its accepted equilibrium lattice
  and near-zero pressure from a displaced initial cell;
- complete wall time and peak physical memory are recorded once after source
  and thresholds are frozen.

Only affected unit modules run during implementation. The source-bound
material calculation runs once at the final gate; remote CPU CI carries the
complete regression suite.

## Delivery Order

1. Implement the controlled periodic stress oracle and topology gate.
2. Add immutable full-matrix cell updates with fractional-position scaling.
3. Add cell-only enthalpy minimization and accepted-state continuation.
4. Compose bounded ionic and cell steps for coupled relaxation.
5. Add atomic outer checkpoint/resume.
6. Close targeted numerical gates and the source-bound Silicon validation.
7. Update capability docs and merge only after repository CI passes.

The analytic closure follows the same contract in four bounded slices: build
one differentiable frozen energy graph, validate every term, make the cell
controller consume the admitted derivative, then rerun the source-bound
material and higher-cutoff Pulay gates once.

## Out Of Scope

External anisotropic pressure, symmetry-constrained lattice families,
transition-state cell paths, molecular or slab boundary conditions, phonons,
and periodic spin remain separate work.
