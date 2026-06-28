# DFT Numerics

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
