# DFT Production-Core Milestone

The DFT package contains two intentionally different surfaces. The legacy
`DFTSystem`/`run_scf` surface supplies tiny Γ-point teaching, dense-reference,
spin, occupation, finite-difference stress, and restart diagnostics. The
periodic `PeriodicDFTSystem`/`run_periodic_scf` surface supplies the
materials-workload path: PBE-PW92, reciprocal-space GTH operators,
Monkhorst-Pack integration, block-Davidson/Rayleigh-Ritz solves,
fixed or Fermi-Dirac occupations, frozen-density band paths, and periodic
forces.

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

Periodic SCF reports one device-inclusive `effective_potential` timing for the
independent Hartree and exchange-correlation branches. They share a single MLX
materialization boundary so phase accounting does not serialize the runtime.

Adaptive periodic SCF uses paired subspace residuals only while its requested
eigensolver tolerance is looser than the final Davidson tolerance. It restores
direct-operator residual validation at the final tolerance, and SCF convergence
requires that directly validated result. Fixed-tolerance and standalone
eigensolves always retain direct validation.

## Spin, Occupations, k-Points, And Bands

The new spin layer is collinear only:

- `unpolarized`: one total density `ρ(r)`.
- `polarized`: separate `ρ↑(r)` and `ρ↓(r)` diagnostics.

The legacy layer exposes fixed and Fermi-Dirac occupation diagnostics. The
periodic layer supports two complete occupation paths:

- The default fixed path computes exactly `N/2` doubly occupied bands. It
  retains the cached-density fast path used by the verified insulating
  workloads.
- `PeriodicFermiDiracSmearing(width_hartree=...)` resolves one global chemical
  potential over the weighted k-point mesh. The caller supplies enough
  computed bands that `2 * n_bands > electron_count`.

The smeared density and band energy use the resolved occupation of every band,
not a post-hoc scalar density correction. SCF convergence follows the
variational electronic free energy `F = E - (k_B T) S`; `PeriodicSCFResult`
separately reports internal energy, chemical potential, dimensionless electronic
entropy,
and smearing width. Checkpoints bind the smearing method and width and reproduce
the same occupations after resume. Periodic nonlocal forces also consume those
occupations, so a converged smeared result yields the stationary free-energy
force.

Both paths use reduced-coordinate Monkhorst-Pack meshes and `0.5|G + k|²`,
including Bloch-phase local and nonlocal GTH evaluation.
`run_periodic_band_structure` reuses a converged SCF density and solves
non-self-consistently along a high-symmetry path.

## Stress, Relaxation, And Restart

Finite-difference stress estimates diagonal orthorhombic stress by changing
cell lengths and rerunning SCF. Geometry optimization supports fixed-cell ion
relaxation only. `GeometryOptimizationConfig` rejects variable-cell and coupled
ion/cell modes instead of silently accepting settings the optimizer cannot use.

Dense SCF restart files store density, orbitals, ion positions, cell lengths, spin metadata, and Γ k-point metadata for small-system continuation workflows.

## Reference Validation

Reference comparison is intentionally static and lightweight. Fixtures are JSON summaries; QE/CP2K are not imported, built, or required in CI. The comparison helper records observed energy, expected energy, error, and pass/fail against a documented tolerance.

## DFT/QM Platform Scope

`get_dft_qm_scope_report()` classifies local DFT/QM capability against CP2K and
Quantum ESPRESSO reference families without changing the runtime dependency
boundary.

| Feature | Local Status | Reference Family |
| --- | --- | --- |
| Plane-wave SCF core | verified for fixed-occupation bulk-Si EOS; Fermi-Dirac path numerically validated | CP2K Quickstep, QE PWscf |
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
Periodic Fermi-Dirac occupations remove the former fixed-occupation algorithmic
blocker for metals and odd electron counts. They do not, by themselves,
establish a material-level metallic accuracy claim; that requires a
source-bound metal workload with matching pseudopotential, smearing convention,
k-point mesh, and free-energy definition.

`dft_qm_scope_readiness_report()` returns a shared readiness payload for these
features. Deferred, anti-goal, and unknown features report blockers before any
production-suite claim can be emitted.

## Hot-Path Recommendation

Hamiltonian application remains the largest periodic DFT phase, but retained
evidence does not justify another narrow custom Metal wrapper. One-dimensional
and three-dimensional scatter/gather kernels improved isolated boundaries but
did not clear the complete-run retention gate. A compiled local FFT wrapper
also changed the convergence trajectory and regressed complete wall time.

The current route instead removes algorithmic work. Final-tolerance Davidson
inverse FFTs are reused by density construction, while earlier adaptive cycles
defer direct validation and perform only the inverse FFT needed for density.
Further Hpsi work should likewise reduce FFT applications or useful
vector-equivalents while preserving final direct-residual validation.
Orthogonalization is the second measured target. The current measurements and
rejected boundaries are maintained in the
[DFT performance decision ledger](./benchmarks/dft-performance-decisions-m5max.md).
