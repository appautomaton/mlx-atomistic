# DFT Pseudopotentials

Milestone 4 adds a real ion-model layer while keeping the DFT engine small and
inspectable. The code now supports parsed UPF and GTH pseudopotential inputs for
local-potential SCF calculations.

## What Is Implemented

- `PseudopotentialData` stores parsed local potential data, valence charge, and
  nonlocal metadata.
- `Ion` and `IonCollection` place parsed pseudopotentials at periodic ion
  centers.
- `LocalPseudopotentialField` builds `V_local(r)` on a real-space DFT grid.
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
nonlocal projectors are parsed and stored, but not applied to orbitals yet.

## GTH

`read_gth(path, element=..., name=...)` reads both single GTH files and CP2K
database entries. The local GTH potential is evaluated analytically:

```text
V_local(r) = -Z_ion erf(r / √2 r_loc) / r
             + exp[-0.5(r/r_loc)²] Σᵢ cᵢ(r/r_loc)²ⁱ
```

The derivative of this local form is used for fixed-density ion-force checks.
GTH nonlocal channel metadata is parsed and stored, but nonlocal application is
still intentionally disabled.

## Forces

For ion-backed systems, reported forces include:

```text
F_total = F_local electron-ion + F_center-center
```

The force validation in this milestone checks fixed-density local forces and
SCF total-energy finite differences. This is a consistency check for the current
local-potential model, not a claim of production DFT force accuracy.

## Current Limits

- No nonlocal projector application yet.
- No spin, k-points, stress tensor, geometry optimizer, or real production
  pseudopotential validation.
- Vendor checkouts remain reference material only; the package does not import
  Quantum ESPRESSO or CP2K code.

## Benchmark

Run:

```bash
uv run python -m mlx_atomistic.benchmarks.dft_pseudopotential --json
```

The benchmark compares compact Gaussian, UPF-local, and GTH-local SCF cases and
reports timing plus pseudopotential diagnostics.
