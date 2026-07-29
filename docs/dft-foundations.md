# DFT Foundations

This page documents the legacy `DFTSystem`/`run_scf` teaching and
dense-reference surface: a small spin-unpolarized Γ-point plane-wave prototype.
It is intentionally separate from the periodic production path built around
`PeriodicDFTSystem` and `run_periodic_scf`.

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

This legacy path is not production DFT. It remains Γ-point and
spin-unpolarized; its spin, occupation, pseudopotential, force, stress, and
geometry surfaces are diagnostic. The separate periodic path supports
Monkhorst-Pack integration, reciprocal-space GTH operators, Davidson solves,
frozen-density band paths, and analytic periodic forces. Its validated
materials and limits are recorded in the
[DFT material-validation summary](./benchmarks/dft-material-validation.md).

The current value is correctness and observability: density normalization,
energy decomposition, SCF residuals, FFT behavior, pseudopotential diagnostics,
force provenance, and small benchmark evidence.
