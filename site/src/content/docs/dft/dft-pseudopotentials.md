---
title: "DFT Pseudopotentials"
---


The DFT layer includes an ion-model surface while keeping the engine small and
inspectable. The code supports parsed UPF and GTH pseudopotential inputs for
local-potential SCF plus proof-level nonlocal projector application.

## What Is Implemented

- `PseudopotentialData` stores parsed local potential data, valence charge, and
  nonlocal metadata.
- `Ion` and `IonCollection` place parsed pseudopotentials at periodic ion
  centers.
- `LocalPseudopotentialField` builds `V_local(r)` on a real-space DFT grid.
- `NonlocalPseudopotentialOperator` applies ion-aware separable projectors when
  parsed projector metadata is available.
- `DFTSystem` accepts `IonCollection` and defaults the electron count to the
  sum of valence charges for neutral systems.
- `run_scf(...)` records pseudopotential diagnostics:
  `pseudopotential_format`, `ion_count`, `valence_electron_count`,
  `nonlocal_available`, and `nonlocal_applied`.

## UPF

`read_upf(path)` reads UPF v2-style XML files from Quantum ESPRESSO-style
sources. The current path uses:

- `PP_HEADER` for element and valence charge.
- `PP_MESH/PP_R` for radial samples.
- `PP_LOCAL` for the local potential.
- `PP_BETA.*` tags for nonlocal projector metadata.

The local UPF potential is interpolated onto the periodic real-space grid. UPF
nonlocal projectors are parsed and applied by the ion-aware operator when
`SCFConfig(apply_nonlocal=True)` is active.

## GTH

`read_gth(path, element=..., name=...)` reads both single GTH files and CP2K
database entries. The local GTH potential is evaluated analytically:

```text
V_local(r) = -Z_ion erf(r / √2 r_loc) / r
             + exp[-0.5(r/r_loc)²] Σᵢ cᵢ(r/r_loc)²ⁱ
```

The derivative of this local form is used for fixed-density ion-force checks.
GTH nonlocal channel metadata is parsed and applied by the same separable
operator path when projector metadata is present.

## Forces

For ion-backed systems, reported forces include:

```text
F_total = F_local electron-ion + F_center-center
```

The force validation in this milestone checks fixed-density local forces and
SCF total-energy finite differences. This is a consistency check for the current
local-potential model, not a claim of production DFT force accuracy.

## Current Limits

- Nonlocal projectors are a proof-level Hermitian separable operator path, not a
  chemically certified reproduction of every UPF/GTH convention.
- Fixed-cell geometry optimization, spin/k-point diagnostics, and
  finite-difference stress exist as prototype surfaces; production materials validation
  and cell relaxation remain out of scope.
- Vendor checkouts remain reference material only; the package does not import
  Quantum ESPRESSO or CP2K code.

## Benchmark

Run:

```bash
uv run python -m mlx_atomistic.benchmarks.dft_pseudopotential --json
```

The benchmark compares compact Gaussian, UPF-local, and GTH-local SCF cases and
reports timing plus pseudopotential diagnostics when explicit pseudopotential
files are supplied. Without extra inputs, the installed package runs only the
self-contained Gaussian case:

```bash
uv run python -m mlx_atomistic.benchmarks.dft_pseudopotential --json
uv run python -m mlx_atomistic.benchmarks.dft_pseudopotential --upf path/to/pseudo.upf --gth path/to/pseudo.gth --gth-element H --json
```
