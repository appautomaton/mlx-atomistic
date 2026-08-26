# Finite-Displacement Phonons

This document is the scientific and engineering contract for Phase 8 of the
[DFT roadmap](./dft-roadmap.md). The first production boundary is a finite-
displacement, Gamma-point harmonic core for a caller-supplied periodic cell or
supercell. General q-point interpolation, non-analytic LO-TO splitting, Raman
and infrared intensities, and electron-phonon coupling remain out of scope.

## Force-Constant Convention

For displaced degree of freedom `i alpha` and response force `j beta`, the
stored force constant is

```text
Phi[j beta, i alpha]
    = -d F[j beta] / d u[i alpha]
    ~= -(F_plus[j beta] - F_minus[j beta]) / (2 delta)
```

This matches the force and force-constant sign convention documented by
[Phonopy](https://phonopy.github.io/phonopy/formulation.html). Every production
sample is central: both displacement signs must be present and bound to exact
calculation fingerprints. One-sided differences are not admitted.

## Symmetry Boundary

The planner accepts explicit affine crystal symmetries: an orthogonal
Cartesian rotation plus a fractional translation. It verifies that each
operation preserves the cell lattice, maps atoms bijectively within the same
element, and maps Cartesian displacement axes through signed permutations.
It then chooses one displacement per exact degree-of-freedom orbit and records
the operation used to reconstruct every omitted column.

The runtime does not guess a space group. With no supplied operations, the
identity operation produces all `3N` Cartesian displacements. Rotations that
mix a Cartesian axis into a non-axis direction fail closed in this phase.

Electronic k-point reduction is a separate symmetry decision. A displacement
lowers the Hamiltonian symmetry, so an equilibrium point-group-reduced mesh is
not automatically valid for a displaced SCF. The accepted Silicon workload
uses its full 4-cubed Gamma-centered electronic mesh for every displacement;
MLX may still apply exact time-reversal reuse after active-basis admission.

## Dynamical Matrix And Units

The Gamma-point dynamical matrix is

```text
D[j beta, i alpha]
    = Phi[j beta, i alpha] / sqrt(m_j m_i)
```

with caller-supplied positive masses in atomic mass units. Eigenvalues are
angular-frequency squared in atomic units after conversion to electron masses.
Signed frequencies in inverse centimeters use the Hartree-to-wavenumber
factor; negative values denote imaginary modes. Mass-weighted eigenvectors and
normalized Cartesian displacement eigenvectors are both retained.

## Diagnostics, Not Repairs

The raw force-constant matrix is never modified to manufacture a pass. Before
diagonalization, the implementation reports and gates:

- reciprocity residual `max(abs(Phi - Phi.T))`;
- right and left translational residuals;
- finite values and complete symmetry-reconstructed columns.

Only roundoff-level reciprocity averaging is used after the raw residual has
passed its locked tolerance. The Acoustic Sum Rule (ASR) is not imposed. This
follows the distinction in the
[Quantum ESPRESSO phonon guidance](https://www.quantum-espresso.org/faq/phonons/):
small nonzero acoustic frequencies diagnose approximate translational
invariance, while large residuals require better input convergence rather than
silent correction.

The three acoustic candidates are the modes with smallest absolute
eigenvalues. Admission requires both bounded absolute frequencies and overlap
with the exact mass-weighted translation subspace. A non-acoustic imaginary
mode outside tolerance fails the stability gate.

## Restart And Convergence

The sample artifact is a compressed NPZ file loaded with
`allow_pickle=False`. It binds the displacement-plan fingerprint, sorted
representative degrees of freedom, plus/minus forces, and both calculation
fingerprints. Publication is atomic and refuses an existing destination.
Loading validates the exact schema, inventory, dtypes, shapes, finite values,
and plan identity.

Restart means loading a partial sample set, evaluating only missing
representatives, and publishing a new immutable artifact. Interrupted and
uninterrupted assembly must be exactly equivalent. Displacement convergence
compares two complete results for the same system, masses, and symmetry plan at
different displacement magnitudes using locked frequency and eigenvalue
drifts.

## Efficient Validation Order

Validation proceeds without repeated material work:

1. analytic harmonic and anharmonic force oracles establish signs, units,
   modes, symmetry reconstruction, ASR rejection, and displacement convergence;
2. no-pickle partial/full artifacts establish restart equivalence;
3. only then may one bounded source-bound crystal run execute the minimum
   symmetry-independent central displacements;
4. a material run is not repeated to reformat its report.

## Accepted Silicon Gate

The source-bound diamond-Silicon workload uses the verified `5.460859 Å`
lattice, a two-atom FCC primitive cell, PBE-PW92, hash-pinned Si GTH-PBE-q4,
25 Ha, a 32-cubed FFT grid, and the full 4-cubed Gamma-centered electronic
mesh. Six exact axis permutations reduce the ionic displacement plan to two
representatives per magnitude. The complete two-magnitude gate therefore uses
eight SCFs.

At `0.01 bohr`, the uncorrected acoustic modes are `3.739`, `4.269`, and
`7.517 cm^-1`; their minimum translation-subspace overlap exceeds
`0.9999999997`. The optical triplet is `509.325`, `512.097`, and
`512.103 cm^-1`, with mean `511.175 cm^-1`. The mean differs from the locked
Quantum ESPRESSO tutorial context by `4.999 cm^-1`, within the predeclared
`60 cm^-1` method-difference boundary.

The `0.02 -> 0.01 bohr` maximum frequency drift is `2.379 cm^-1`, below the
locked `5 cm^-1` gate; maximum dynamical-eigenvalue drift is `1.579e-9`, below
`1e-7`. Raw reciprocity and left/right ASR residuals pass without modifying the
force constants. The two displacement sets took `11.850` and `11.868 s`, with
`23,820,912` peak temporary bytes. Partial-sample reload produces bitwise-
identical force constants and frequencies in the deterministic restart gate.

This closes the bounded Gamma-point Phase 8 gate. It does not establish phonon
dispersion, supercell range convergence, polar non-analytic corrections, or
finite-temperature lattice dynamics.
