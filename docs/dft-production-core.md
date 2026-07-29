# DFT Production-Core Milestone

The DFT package contains two intentionally different surfaces. The legacy
`DFTSystem`/`run_scf` surface supplies tiny Γ-point teaching, dense-reference,
spin, occupation, finite-difference stress, and restart diagnostics. The
periodic `PeriodicDFTSystem`/`run_periodic_scf` surface supplies the
materials-workload path: PBE-PW92, reciprocal-space GTH operators,
Monkhorst-Pack integration, block-Davidson/Rayleigh-Ritz solves,
frozen-density band paths, and periodic forces.

The periodic implementation has verified results for specific workloads, but
is not broadly chemically certified. Capability claims are tied to the
[material-validation summary](./benchmarks/dft-material-validation.md).

## Nonlocal Pseudopotentials

UPF and GTH nonlocal metadata is converted into normalized real-space separable projectors. The operator applies:

```text
V̂_NL ψ = Σᵢ |βᵢ⟩ Dᵢ ⟨βᵢ|ψ⟩
```

For UPF, diagonal `PP_DIJ` values are used as projector couplings after Ry-to-Hartree conversion. For GTH, parsed projector coefficients seed the coupling. This gives a Hermitian validation path, but it is not yet a full chemically faithful reproduction of every format convention.

SCF applies nonlocal projectors by default when available. `SCFConfig(apply_nonlocal=False)` keeps the old local-only path available for debugging and comparison.

## Solvers

Dense diagonalization remains the tiny-grid reference for the legacy path.
Periodic SCF and bands use the MLX-native block-Davidson/Rayleigh-Ritz solver
without building the full plane-wave Hamiltonian. Diagnostics expose residuals,
orthonormality error, subspace work, and convergence metadata.

## Spin, Occupations, k-Points, And Bands

The new spin layer is collinear only:

- `unpolarized`: one total density `ρ(r)`.
- `polarized`: separate `ρ↑(r)` and `ρ↓(r)` diagnostics.

The legacy layer exposes fixed and Fermi-Dirac occupation diagnostics. The
periodic layer uses reduced-coordinate Monkhorst-Pack meshes and
`0.5|G + k|²`, including Bloch-phase local and nonlocal GTH evaluation.
`run_periodic_band_structure` reuses a converged SCF density and solves
non-self-consistently along a high-symmetry path.

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
| Plane-wave SCF core | verified (bulk-Si EOS) | CP2K Quickstep, QE PWscf |
| UPF/GTH pseudopotentials and nonlocal projectors | proof-level | QE UPF, CP2K GTH |
| Geometry relaxation and finite-difference stress | proof-level | CP2K MOTION/GEO_OPT, QE relax |
| Static reference comparison | supported | static CP2K/QE fixture summaries |
| QM/MM force-environment orchestration | deferred | CP2K FORCE_EVAL/QMMM |
| PH/EPW/NEB/TDDFT/MPI/offload suite breadth | deferred | QE and CP2K production suites |
| Importing, wrapping, building, or running CP2K/QE | anti-goal | external executables |

Plane-wave SCF core is `verified` only for the bulk-silicon PBE equation of state
against an all-electron (FLEUR/WIEN2k) reference (Lejaeghere Δ factor
1.942 meV/atom); broader chemistry stays proof-level, and the separate
MLX-versus-QE PWscf cross-engine parity is still diagnostic, not closed.

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
