---
title: "DFT Foundations"
---


This first DFT slice is a small spin-unpolarized Γ-point plane-wave prototype.
It is meant to make the numerical building blocks inspectable before we try to
make them chemically broad or highly optimized.

## What It Models

The prototype works with one total electron density `ρ(r)`.
For closed-shell systems, each spatial orbital is doubly occupied:

```text
ρ(r) = 2Σᵢ |ψᵢ(r)|²
```

Odd or fractional electron counts are allowed for toy examples. Separate
`ρ↑(r)` and `ρ↓(r)` spin-density helpers exist as diagnostics, but the primary
SCF path remains spin-unpolarized for `0.0.1`.

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
- `DiracExchange`, `LDACorrelationPZ81`, `LDAExchangeCorrelation`, and the
  public-alpha `PBEExchangeCorrelation` expose the first exchange-correlation
  layer.
- `run_scf(...)` iterates density, effective potential, and orbitals with
  linear or Pulay DIIS density mixing.
- `read_upf(...)`, `read_gth(...)`, and `NonlocalPseudopotentialOperator` expose
  proof-level ion-backed pseudopotential paths.

Programmatic toy systems are available as `toy_one_electron_dft_example()` and
`toy_closed_shell_dft_example()` from `mlx_atomistic.examples`.

## Current Limits

This is not production DFT. The primary SCF path is still Γ-point and
spin-unpolarized. K-point support is non-SCF diagnostics, spin/occupation
support is diagnostic, real pseudopotential formats and nonlocal projectors are
proof-level, and force, stress, and geometry optimization paths are prototype
surfaces. Production materials validation, cell relaxation, and custom Metal
kernels remain out of scope for `0.0.1`.

The current value is correctness and observability: density normalization,
energy decomposition, SCF residuals, FFT behavior, pseudopotential diagnostics,
force provenance, and small benchmark evidence.
