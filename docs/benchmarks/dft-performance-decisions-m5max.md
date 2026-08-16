# DFT Performance Decision Ledger on Apple M5 Max

Date: 2026-08-16

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
| Residual-aware CholeskyQR1/2 | 55.384 s paired median | 14 | Retained route |
| Davidson-to-density FFT reuse | 54.839 s paired diagnostic | 14 | Retained route |
| Deferred adaptive residual validation | 44.914 s paired diagnostic | 14 | Current route |

The finite scheduler, introduced in `2a56533`, maps variable Davidson batches
onto 12 reusable Metal shapes: lane capacities 1, 2, 4, or 8 crossed with
vector capacities 4, 8, or 16. It improved the immediately preceding complete
run by 19.68% without reducing self-consistent field (SCF) work or changing the
cycle count.

The residual-aware orthonormalization route uses one CholeskyQR pass only while
the Davidson residual target is loose, validates the resulting overlap, and
restores CholeskyQR2 before the target approaches the complex64 rank scale. Two
interleaved bounded runs reduced median wall by 2.59% and orthonormalization by
17.66%. Two complete candidate runs converged in 14 cycles with numerical
validation and a median wall of 55.384 seconds, compared with 56.855 seconds
for the two CholeskyQR2 controls. These were same-session diagnostic runs; they
are not a formal low-power publication pair.

The current route retains each final Davidson inverse FFT long enough to form
its occupied-orbital density. The density stage previously repeated the same
inverse transform immediately after direct residual validation. A same-session
complete A/B pair preserved the 14-cycle trajectory, final energy, density
residual, and electron count exactly. It reduced FFT vector-equivalents from
279,910 to 255,718, density time from 2.525 to 0.113 seconds, and complete
elapsed time from 56.305 to 54.839 seconds. This is a 2.61% wall reduction, or
1.027x speedup, under the diagnostic low-power protocol.

Adaptive SCF now defers direct-operator residual validation while the requested
Davidson tolerance is looser than its final tolerance. Those inexact cycles use
the paired subspace residual and rebuild density with a separate inverse FFT.
Fixed-tolerance solves, standalone eigensolves, and every final-tolerance SCF
cycle retain direct validation; an SCF result cannot converge without that final
direct residual. This follows a measured middle path between Quantum ESPRESSO,
which accepts projected convergence without a final Hamiltonian reapplication,
and CP2K block Davidson, which constructs direct residuals from the current
orbitals.

A same-session low-power A/B/B/A diagnostic reduced mean complete wall from
48.801 to 44.914 seconds, a 7.97% reduction or 1.087x speedup. Both candidate
runs completed 14 cycles with identical candidate results and final direct
residuals. Against the controls, the final energy changed by 2.87e-7 Hartree,
the density residual by 7.10e-9, and the maximum orbital residual by 1.81e-9.
Hpsi vector-equivalents fell from 127,859 to 110,118 and FFT
vector-equivalents from 255,718 to 237,516.

The current complete-run attribution is approximately:

| Phase | Time | Share |
| --- | ---: | ---: |
| Hpsi applications | 22.86 s | 50.9% |
| Orthogonalization | 9.59 s | 21.4% |
| Projected Rayleigh-Ritz solve | 2.27 s | 5.1% |
| Eigensolver control | 3.22 s | 7.2% |
| CPU small solves | 3.56 s | 7.9% |
| Density | 2.18 s | 4.9% |
| Setup, mixing, persistence, and unaccounted | 1.24 s | 2.8% |

Hpsi is the Hamiltonian applied to a batch of wavefunctions. It remains the
largest phase, but the measurements below show that not every Hpsi boundary is
large enough to justify a new runtime route.

## Retained Infrastructure

| Change | Decision evidence | Commit |
| --- | --- | --- |
| Finite Hpsi shape scheduling | Reused a bounded set of GPU shapes and reduced complete Silicon SCF wall from 73.743 to 59.231 seconds. | `2a56533` |
| Residual-aware CholeskyQR | Skips the second pass only for a validated loose-residual basis; paired complete diagnostics reduced median wall from 56.855 to 55.384 seconds. | `6e49877` |
| Davidson-to-density FFT reuse | Reuses the final direct-validation real-space orbitals; a complete paired diagnostic removed 24,192 FFT vector-equivalents and reduced wall by 2.61% with an identical trajectory. | `a8e0b6b` |
| Deferred adaptive residual validation | Uses paired residuals only during inexact adaptive cycles, restores direct validation at the final tolerance, and reduced complete diagnostic wall by 7.97%. | `1c92075` |
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
| Multi-lane grouped CholeskyQR2 | The bounded eight-representative run regressed from 4.801 to 5.773 seconds because small grouped Gram operations were slower. | 2026-08-16 diagnostic |
| Unconditional CholeskyQR1 | The bounded gate improved, but the complete run reached 80 cycles without meeting the final residual. | 2026-08-16 diagnostic |
| Newton-Schulz orthogonal refinement | Cholesky normalizer calls fell from 703 to 525, but bounded median wall regressed from 5.816 to 5.868 seconds. | 2026-08-16 diagnostic |
| Smaller Davidson subspace, RMM-DIIS, and converged-subspace locking | Iteration counts or complete bounded wall increased. | pre-`2a56533` scheduler experiments |
| One-dimensional compact Hpsi Metal boundary | Fixed-density wall improved 14.49%, narrowly below the frozen 14.72% dispersion gate. The candidate was removed. | `831e077` |
| Three-dimensional scatter/gather Metal boundary | The isolated Hpsi boundary improved 17.33%, but complete fixed-density wall improved only 6.40%, below the same retention gate. The candidate was removed. | `9cd4ef6` |
| Flattened leading FFT dimensions | The dominant 8-lane, 16-vector local FFT regressed from 27.57 to 33.23 milliseconds. | 2026-08-16 diagnostic |
| Compiled local FFT wrapper | A compiled graph reduced time per Hpsi, but changed complex64 rounding and increased complete-run Hpsi calls from 1,388 to 1,449; wall regressed from 72.408 to 75.046 seconds. | 2026-08-16 diagnostic |

The custom-kernel results establish a useful boundary: Metal can accelerate
scatter and gather locally, but that boundary is too diluted in the complete
calculation to maintain a separate production route.

## Current Direction

The current route deliberately trades early direct Hpsi applications for
density-only inverse FFTs. Density therefore rises from 0.2% to 4.9% of wall,
while the more expensive Hpsi phase falls by about four seconds and remains the
largest phase at 50.9%. The next high-ceiling target is correction-vector Hpsi,
which accounted for 16.15 seconds in the control attribution, followed by
orthogonalization at 21.4% of current wall. Reshaping or wrapping the same FFTs
did not improve the complete calculation, so another C++ extension or narrow
scatter/gather kernel is not justified by the retained evidence.

Retrieve the retired detailed reports when auditing a historical result:

```bash
git show 8899994:site/src/content/docs/benchmarks/mlx-dft-silicon-m5max.md
git show 8899994:docs/benchmarks/dft-hpsi-metal-boundary-m5max.md
git show 8899994:docs/benchmarks/dual-runtime-baselines-m5max.md
```
