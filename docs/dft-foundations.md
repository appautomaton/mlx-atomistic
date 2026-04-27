# DFT Foundations

This first DFT slice is a small spin-unpolarized Γ-point plane-wave prototype.
It is meant to make the numerical building blocks inspectable before we try to
make them chemically broad or highly optimized.

## What It Models

The prototype works with one total electron density `ρ(r)`.
For closed-shell systems, each spatial orbital is doubly occupied:

```text
ρ(r) = 2Σᵢ |ψᵢ(r)|²
```

Odd or fractional electron counts are allowed for toy examples, but the code
does not yet model separate `ρ↑(r)` and `ρ↓(r)` spin densities.

## Units

DFT internals use atomic units:

```text
ℏ = 1
m_e = 1
e = 1
4πε₀ = 1
```

Coordinates and cell lengths are in bohr, energies are in hartree, and the
electron density integrates to electron count over the cell.

## Numerical Pieces

- `RealSpaceGrid` stores an orthorhombic periodic grid.
- `ReciprocalGrid` stores FFT-compatible `G` vectors and `|G|²`.
- `normalize_orbitals(...)` enforces `∫ |ψᵢ(r)|² dr = 1`.
- `density_from_orbitals(...)` builds `ρ(r)` from occupied orbitals.
- `LocalGaussianPseudopotential` provides a toy local external potential.
- `hartree_potential(...)` solves the periodic Poisson equation in reciprocal
  space, with the `G = 0` term set to zero.
- `lda_exchange_energy_potential(...)` implements Dirac LDA exchange only.
- `run_scf(...)` iterates density, effective potential, and orbitals with
  simple density mixing.

Programmatic toy systems are available as `toy_one_electron_dft_example()` and
`toy_closed_shell_dft_example()` from `mlx_atomistic.examples`.

## Current Limits

This is not production DFT. It intentionally excludes k-points, spin
polarization, real pseudopotential formats, correlation functionals, forces,
geometry optimization, and custom Metal kernels.

The current value is correctness and observability: density normalization,
energy decomposition, SCF residuals, FFT behavior, and small benchmark evidence.
The next DFT milestones can add LDA correlation, force calculations, better SCF
mixing, and then performance work once the hot paths are measured.
