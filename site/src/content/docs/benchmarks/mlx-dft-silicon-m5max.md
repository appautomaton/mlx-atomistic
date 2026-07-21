---
title: "MLX periodic DFT: 8-atom silicon on M5 Max"
---


This is the canonical performance ledger for the eight-atom conventional
silicon cell. The production target is the complete 6 × 6 × 6 k-point SCF,
not a small fixed-Hamiltonian or partial-k-point development probe.

## Workload identity

| Field | Value |
| --- | --- |
| Engine | `mlx_atomistic` MLX/Metal runtime |
| Host | Apple M5 Max MacBook Pro, low-power mode |
| Cell | 8-atom conventional silicon periodic cell |
| Electrons / requested bands | 32 valence electrons / 16 occupied bands |
| Functional / pseudopotential | PBE / Si GTH-PBE-q4 |
| Plane-wave cutoff / FFT grid | 25 Ha / 56 × 56 × 56 |
| k-point mesh | 6 × 6 × 6: 216 explicit, 108 calculated representatives after time reversal |
| Precision | complex64/float32 full grid; complex128 CPU projected eigensolve |
| Eigensolver | block Davidson–Rayleigh–Ritz, at most 64 subspace vectors |
| Runtime boundary | complete SCF timing includes setup and total-energy cycle; raw report publication is outside the measured SCF kernel |

## Accepted complete-SCF results

| State | Wall time | SCF cycles | Representative k-points | Verdict |
| --- | ---: | ---: | ---: | --- |
| Earlier retained implementation | 152.291 s | 13 | 108 | Numerically valid historical baseline |
| Adaptive-tolerance implementation | 73.743 s | 14 | 108 | Latest complete result; current optimization baseline |

The current baseline is about 2.07× faster than the earlier retained result.
It is still above the near-term acceptance target of 66.37 seconds and the
60-second stretch target.

Raw complete-run evidence:

- Earlier diagnostic family: `results/mlx-dft-runtime-architecture/diagnostics/`
- Current report: `results/mlx-dft-runtime-architecture/diagnostics/20260721-adaptive-tolerance/full-scf/report.json`

## Current complete-run profile

| Phase | Time | Share of 73.743 s |
| --- | ---: | ---: |
| Hψ applications | 36.93 s | 50.1% |
| Orthogonalization | 13.35 s | 18.1% |
| Rayleigh–Ritz | 8.32 s | 11.3% |
| Everything else | 15.14 s | 20.5% |

The largest remaining cost is applying the Hamiltonian. Converged-subspace
locking is worth testing only if it reduces total repeated solver work; moving
work between the other two eigensolver phases is not sufficient.

## Bounded development evidence

These rows are intentionally not comparable to the complete production result.

| Probe | Scope | Result | What it proves |
| --- | --- | ---: | --- |
| Fixed-Hamiltonian A/B | First 8 representatives, one eigensolve, no density loop | 1.890 → 1.752 s | Restart changes removed redundant Davidson work |
| Adaptive SCF diagnostic | First 8 representatives, complete density loop for only those points | 7.469 s, 13 cycles | Tail lanes repeatedly carry large active subspaces after most bands converge |
| One-representative adaptive gate | One representative, partial-zone SCF | 5.990 s, 14 cycles | Fast fail-early baseline, not a production timing |

The direct fixed-Hamiltonian A/B reduced CholeskyQR2 attempts from 336 to 241,
orthogonalized vectors from 3,193 to 1,918, and Hψ vector equivalents from
2,344 to 2,206. All eight lanes passed residual and overlap checks. Later
absolute samples varied, so the work counters and matched A/B are stronger
evidence than the isolated 1.752-second value.

## Reproducer

The supported partial-zone SCF gate requires explicit science inputs and labels
its output as non-production:

```sh
uv run python -m mlx_atomistic.benchmarks.dft_scf_smell \
  --manifest results/mlx-dft-runtime-architecture/workload-slice1/manifest.json \
  --gth-source results/mlx-dft-runtime-architecture/workload-slice1/resources/Si-GTH-PBE-q4.gth \
  --mode adaptive --representatives 8 \
  --out results/mlx-dft-runtime-architecture/smell/adaptive-8.json --json
```

Run it through the repository's bounded process-tree wrapper when collecting
performance evidence. The hard process-tree memory limit is 40 GB. A one-point
gate precedes three interleaved eight-point baseline/candidate pairs. A candidate
must improve every pair and the paired median by at least 10%, without increasing
SCF cycles or Hψ vector work, before any complete 108-representative run is
allowed.

## Rejected experiments

| Experiment | Bounded outcome | Decision |
| --- | --- | --- |
| Padded multi-lane CholeskyQR2 | 1.890 → 2.590 s; one residual failure | Removed |
| GTH overlap chunk 1024 → 2048 | 2.199 s; more Davidson/Hψ iterations | Removed |
| Predictive Gram admission | 2.055 and 2.013 s | Removed |
| Ragged padded projected eigensolves | 2.010 s; one extra Hψ round | Removed |
| Compiled GTH contraction | 2.169 s; Hψ rose to 1.176 s | Removed |
| Davidson maximum subspace 64 → 48 | 2.192 s; more iterations and Hψ | Removed |
| Hybrid RMM-DIIS prototype | One point: 5.550 → 8.458 s; 14 → 21 cycles | Removed |

The symbols associated with rejected implementations must remain absent:
`finite_certified`, `projected_eigh_ragged`, `prediction_tolerance`,
`predicted_gram`, `measure_locked_overlap`, and `compiled_gth_contraction`.
