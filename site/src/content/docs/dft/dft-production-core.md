---
title: "DFT Production-Core Milestone"
---


This milestone consolidates the next DFT phases into one correctness-first release. The library now has working prototype surfaces for nonlocal pseudopotentials, iterative Kohn-Sham solving, spin/occupation diagnostics, k-point and band-path diagnostics, finite-difference stress, dense SCF restart persistence, and reference comparison.

The implementation is still not chemically certified production DFT. It is a production-core infrastructure milestone: the main algorithms exist, have tests, and expose diagnostics, but the numerical models remain intentionally conservative.

## Nonlocal Pseudopotentials

UPF and GTH nonlocal metadata is converted into normalized real-space separable projectors. The operator applies:

```text
V̂_NL ψ = Σᵢ |βᵢ⟩ Dᵢ ⟨βᵢ|ψ⟩
```

For UPF, diagonal `PP_DIJ` values are used as projector couplings after Ry-to-Hartree conversion. For GTH, parsed projector coefficients seed the coupling. This gives a Hermitian validation path, but it is not yet a full chemically faithful reproduction of every format convention.

SCF applies nonlocal projectors by default when available. `SCFConfig(apply_nonlocal=False)` keeps the old local-only path available for debugging and comparison.

## Solvers

Dense diagonalization remains the tiny-grid reference. Above the dense cutoff, `auto` SCF selects the Davidson-style path. The current Davidson implementation uses a kinetic preconditioner and residual iteration for larger grids, while still falling back to dense reference on tiny validation grids.

Diagnostics expose residuals, orthonormality error, solver metadata, subspace size, and restart count.

## Spin, Occupations, k-Points, And Bands

The new spin layer is collinear only:

- `unpolarized`: one total density `ρ(r)`.
- `polarized`: separate `ρ↑(r)` and `ρ↓(r)` diagnostics.

Occupation models include fixed occupations and Fermi-Dirac occupations. k-point abstractions support Γ-point meshes, Monkhorst-Pack-style meshes, and non-SCF band paths. The kinetic operator supports `0.5|G + k|²`, and Γ remains the default.

## Stress, Relaxation, And Restart

Finite-difference stress estimates diagonal orthorhombic stress by changing cell lengths and rerunning SCF. Geometry optimization remains ion-position-first by default, with config fields now prepared for cell and coupled relaxation modes.

Dense SCF restart files store density, orbitals, ion positions, cell lengths, spin metadata, and Γ k-point metadata for small-system continuation workflows.

## Reference Validation

Reference comparison is intentionally static and lightweight. Fixtures are JSON summaries; QE/CP2K are not imported, built, or required in CI. The comparison helper records observed energy, expected energy, error, and pass/fail against a documented tolerance.

## DFT/QM Platform Scope

`get_dft_qm_scope_report()` classifies local DFT/QM capability against CP2K and
Quantum ESPRESSO reference families without changing the runtime dependency
boundary.

| Feature | Local Status | Reference Family |
| --- | --- | --- |
| Plane-wave SCF core | proof-level | CP2K Quickstep, QE PWscf |
| UPF/GTH pseudopotentials and nonlocal projectors | proof-level | QE UPF, CP2K GTH |
| Geometry relaxation and finite-difference stress | proof-level | CP2K MOTION/GEO_OPT, QE relax |
| Static reference comparison | supported | static CP2K/QE fixture summaries |
| QM/MM force-environment orchestration | deferred | CP2K FORCE_EVAL/QMMM |
| PH/EPW/NEB/TDDFT/MPI/offload suite breadth | deferred | QE and CP2K production suites |
| Importing, wrapping, building, or running CP2K/QE | anti-goal | external executables |

`dft_qm_scope_readiness_report()` returns a shared readiness payload for these
features. Deferred, anti-goal, and unknown features report blockers before any
production-suite claim can be emitted.

## Hot-Path Recommendation

The first future custom Metal kernel should target **Hamiltonian application**, specifically the combined kinetic + local + nonlocal application path used by Davidson and band calculations.

Reason:

- Dense diagonalization is a reference path and should not be optimized first.
- SCF and band workloads repeatedly apply `Hψ`.
- Nonlocal projector application adds many grid reductions and scatter-like projector accumulations.
- A fused Metal path can reduce Python-loop overhead before deeper eigensolver work.

Second-tier targets are projector construction/interpolation and orthonormalization. FFT/Hartree should be measured carefully before replacing MLX/Accelerate-backed paths.
