# DFT Performance Decision Ledger on Apple M5 Max

Date: 2026-08-14

This ledger preserves the transferable performance decisions for the periodic
Density Functional Theory (DFT) runtime. It replaces separate Silicon runtime,
dual-runtime baseline, and custom-kernel boundary narratives. Scientific
admission remains in the
[DFT material-validation ledger](./dft-material-validation.md).

Measurements from different dates and power states are not direct A/B pairs.
Generated reports remain under gitignored `results/`; retired long-form reports
remain available through Git history.

## Production Baseline

The complete eight-atom Silicon workload uses PBE/GTH-q4, a 25 Hartree cutoff,
a 56-by-56-by-56 fast Fourier transform (FFT) grid, and a 6-by-6-by-6 k-point
mesh. Time reversal reduces 216 explicit k-points to 108 calculated
representatives.

| Production state | Complete SCF wall | Cycles | Decision |
| --- | ---: | ---: | --- |
| Earlier retained implementation | 152.291 s | 13 | Historical baseline |
| Adaptive-tolerance implementation | 73.743 s | 14 | Superseded baseline |
| Finite Hpsi shape scheduler | 59.231 s | 14 | Retained production route |

The finite scheduler, introduced in `2a56533`, maps variable Davidson batches
onto 12 reusable Metal shapes: lane capacities 1, 2, 4, or 8 crossed with
vector capacities 4, 8, or 16. It improved the immediately preceding complete
run by 19.68% without reducing self-consistent field (SCF) work or changing the
cycle count.

The accepted complete-run attribution was:

| Phase | Time | Share |
| --- | ---: | ---: |
| Hpsi applications | 29.32 s | 49.5% |
| Orthogonalization | 11.46 s | 19.3% |
| Projected Rayleigh-Ritz solve | 6.11 s | 10.3% |
| Eigensolver control | 5.66 s | 9.6% |
| Density | 2.43 s | 4.1% |
| Setup, mixing, persistence, and unaccounted | 4.25 s | 7.2% |

Hpsi is the Hamiltonian applied to a batch of wavefunctions. It remains the
largest phase, but the measurements below show that not every Hpsi boundary is
large enough to justify a new runtime route.

## Retained Infrastructure

| Change | Decision evidence | Commit |
| --- | --- | --- |
| Finite Hpsi shape scheduling | Reused a bounded set of GPU shapes and reduced complete Silicon SCF wall from 73.743 to 59.231 seconds. | `2a56533` |
| Scientific EOS and band gates | Separated runtime convergence from equation-of-state (EOS) and band validation; Silicon admission was later recorded against all-electron references. | `78d4b9d`, `1972dce` |
| Hpsi stage profiler | Separates local FFT, compact scatter/gather, kinetic, and Goedecker-Teter-Hutter (GTH) pseudopotential work using stable captured inputs. Profiles are diagnostic rather than production timings. | `9cd4ef6` |

The stable 64-vector profiler attributed 72.60% of independently synchronized
Hpsi time to the local FFT path, 23.30% to compact scatter, and 5.60% to GTH.
These isolated medians are not additive. A later decomposition measured the
inverse and forward FFTs as 95.17% of the complete local-FFT median.

## Rejected Directions

| Candidate | Why it was rejected | Historical source |
| --- | --- | --- |
| Padded multi-lane CholeskyQR2 | Fixed-Hamiltonian wall regressed from 1.890 to 2.590 seconds and one residual check failed. | pre-`2a56533` scheduler experiments |
| Larger GTH overlap chunks and compiled contraction | Davidson/Hpsi work increased or the bounded probe slowed. | pre-`2a56533` scheduler experiments |
| Predictive Gram admission and ragged projected solves | Bounded timings regressed or introduced extra Hpsi work. | pre-`2a56533` scheduler experiments |
| Smaller Davidson subspace, RMM-DIIS, and converged-subspace locking | Iteration counts or complete bounded wall increased. | pre-`2a56533` scheduler experiments |
| One-dimensional compact Hpsi Metal boundary | Fixed-density wall improved 14.49%, narrowly below the frozen 14.72% dispersion gate. The candidate was removed. | `831e077` |
| Three-dimensional scatter/gather Metal boundary | The isolated Hpsi boundary improved 17.33%, but complete fixed-density wall improved only 6.40%, below the same retention gate. The candidate was removed. | `9cd4ef6` |

The custom-kernel results establish a useful boundary: Metal can accelerate
scatter and gather locally, but that boundary is too diluted in the complete
calculation to maintain a separate production route.

## Current Direction

The next DFT performance candidate should begin with a fresh complete-run
profile and reduce useful FFT work: padded vector capacity, submission shapes,
or avoidable FFT vector equivalents. Another C++ extension or narrow
scatter/gather kernel is not justified by the retained evidence.

Retrieve the retired detailed reports when auditing a historical result:

```bash
git show 8899994:site/src/content/docs/benchmarks/mlx-dft-silicon-m5max.md
git show 8899994:docs/benchmarks/dft-hpsi-metal-boundary-m5max.md
git show 8899994:docs/benchmarks/dual-runtime-baselines-m5max.md
```
