---
title: "DFT Numerics"
---


Milestone 3 makes the DFT layer more numerically inspectable. The primary SCF
path is still small, Γ-point, and alpha/proof-level, but the code now has
explicit checks for the Kohn-Sham operator, orbital residuals, energy
decomposition, nonlocal projector diagnostics, and total-energy force
consistency.

## Kohn-Sham Operator

The Kohn-Sham equation is an eigenproblem for auxiliary one-electron orbitals:

```text
H_KS[ρ] ψᵢ = εᵢ ψᵢ
```

For the current prototype:

```text
H_KS[ρ] = T + V_local + V_nonlocal + V_H[ρ] + V_xc[ρ]
```

- `T` is the plane-wave kinetic operator, applied in reciprocal space as
  `0.5 |G|²`.
- `V_local` is the toy Gaussian local pseudopotential.
- `V_nonlocal` is included only when ion-backed proof-level nonlocal projectors
  are active.
- `V_H[ρ]` is the Hartree potential from the reciprocal-space Poisson solve.
- `V_xc[ρ]` is the exchange-correlation potential.

`KohnShamOperator` exposes these pieces separately so tests and notebooks can
inspect operator action directly instead of treating SCF as a black box.

## Dense Reference Vs Operator Application

For tiny grids, the package can explicitly build the dense Hamiltonian matrix by
applying `H_KS` to every grid basis vector. That path is intentionally not a
production solver; it is a reference implementation.

The operator path applies `H_KS` without building the dense matrix. Milestone 3
compares the two:

```text
dense_matrix @ ψ  ≈  KohnShamOperator.apply_hamiltonian(ψ)
```

This is the main guardrail before replacing the tiny dense solver with a real
iterative eigensolver.

## Orbital Residuals

SCF density residual alone is not enough. A density can stop changing while the
orbitals are still poor eigenvectors of the current Hamiltonian. The new
diagnostic computes:

```text
||Hψᵢ - εᵢψᵢ||
```

Small residuals mean the orbital is close to an eigenvector of the current
Kohn-Sham operator. `SCFResult` now records orbital eigenvalues, per-orbital
residuals, and the maximum orthonormality error.

## Energy Decomposition

The SCF electronic energy is separated from center-center repulsion:

```text
E_total = E_electronic + E_center-center
```

`E_electronic` includes kinetic, local, nonlocal pseudopotential, Hartree, and
XC terms when those terms are active. The center-center term is the current
ion-center Coulomb contribution; it prevents the total energy from hiding an
important physical contribution.

## Force Checks

`run_scf(...)` reports center forces for `DFTSystem` calculations. These combine
the local electron-ion force, the center-center force, and the fixed-orbital
nonlocal finite-difference correction when nonlocal projectors are active.

`scf_total_energy_forces(...)` reruns SCF after displacing each center and
compares:

```text
F_A ≈ -[E(R_A + δ) - E(R_A - δ)] / 2δ
```

This checks consistency between the reported force and the total-energy surface.
It does not prove production-grade DFT forces because the nonlocal correction is
an alpha finite-difference path and production materials validation, cell
relaxation, and custom kernels remain out of scope.

### Periodic plane-wave forces

The production periodic path has a separate fixed-cell force implementation:

```text
Fᵢ = Fᵢ(local GTH) + Fᵢ(nonlocal GTH) + Fᵢ(Ewald ions)
```

`periodic_scf_forces(system, result)` requires a converged
`PeriodicSCFResult` with an exact system fingerprint. The local term
differentiates the reciprocal-space ionic phase against the converged density;
the nonlocal term differentiates each GTH projector phase and sums occupied
states with their k-point integration weights; and the ion-ion term uses the
analytic Ewald derivative.

At fixed cell, the plane-wave basis does not depend on ionic positions, so
there is no ionic Pulay-force term. Bounded fixed-state and reconverged
two-species finite-difference tests cover the implementation.

The full eight-atom MgO validation uses the accepted 70 Ha, 6×6×6 PBE-GTH
workload and 48 reconverged SCFs at ±0.01 bohr. At the unchanged
`1e-4 Ha/bohr` gate, 21 of 24 analytic-versus-central-difference components
pass. The remaining O 6-x and O 7-y/z components are recorded as a known
float32 total-energy precision limitation, with a maximum deviation of
`2.246e-4 Ha/bohr`; the threshold is not weakened. Their analytic equilibrium
forces remain symmetry-correct and near zero, while tighter SCF continuation
shifts the finite-difference values non-monotonically without a systematic
atom-axis pattern.

## Benchmark Evidence

The DFT benchmark reports SCF timing sections. The new operator benchmark adds:

- Kohn-Sham operator construction time.
- Direct operator application time.
- Dense Hamiltonian build time.
- Dense diagonalization time.
- Prototype subspace solve time.
- Dense-vs-operator matrix-vector error.

Use:

```bash
uv run python -m mlx_atomistic.benchmarks.dft_operator --json
```

The intended decision point is whether the next optimization target should be
Hamiltonian application, FFT/Hartree work, eigensolver iteration, or a future
custom Metal kernel.
