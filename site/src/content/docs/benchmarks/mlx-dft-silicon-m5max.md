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
| Adaptive-tolerance implementation | 73.743 s | 14 | 108 | Previous complete baseline |
| Finite Hψ shape scheduler | 59.231 s | 14 | 108 | Latest complete result; retained default |

The retained scheduler result is 19.68% faster than the immediately preceding
complete run and about 2.57× faster than the 152.291-second historical result.
It passes the 67.844-second full-run retention gate and reaches the 60-second
stretch target without changing the SCF cycle count.

Raw complete-run evidence:

- Earlier diagnostic family: `results/mlx-dft-runtime-architecture/diagnostics/`
- Previous report: `results/mlx-dft-runtime-architecture/diagnostics/20260721-adaptive-tolerance/full-scf/report.json`
- Current report: `results/mlx-dft-runtime-architecture/finite-buckets/full-candidate-r1/report.json`
- Current process-memory trace: `results/mlx-dft-runtime-architecture/finite-buckets/full-candidate-r1-memory.json`

## Current complete-run profile

| Phase | Time | Share of 59.233 s observed |
| --- | ---: | ---: |
| Hψ applications | 29.32 s | 49.5% |
| Orthogonalization | 11.46 s | 19.3% |
| Projected Rayleigh–Ritz, including CPU small solves | 6.11 s | 10.3% |
| Eigensolver control | 5.66 s | 9.6% |
| Density | 2.43 s | 4.1% |
| Setup, persistence, mixing, and unaccounted | 4.25 s | 7.2% |

The finite scheduler maps variable Davidson batches onto only 12 reusable Metal
shapes: lane capacities 1, 2, 4, or 8 crossed with vector capacities 4, 8, or
16. This avoids repeatedly compiling and dispatching many nearly unique tail
shapes. Against the previous complete run, Hψ time fell from 36.93 to 29.32
seconds and orthogonalization from 13.35 to 11.46 seconds, while logical Hψ work
rose only 0.57%. The speedup is therefore primarily better GPU execution shape,
not less accurate physics or fewer SCF cycles.

The largest remaining cost is still applying the Hamiltonian. The next
optimization should reduce useful solver work, not add more scheduler shapes.

## Bounded development evidence

These rows are intentionally not comparable to the complete production result.

| Probe | Scope | Result | What it proves |
| --- | --- | ---: | --- |
| Fixed-Hamiltonian A/B | First 8 representatives, one eigensolve, no density loop | 1.890 → 1.752 s | Restart changes removed redundant Davidson work |
| Adaptive SCF diagnostic | First 8 representatives, complete density loop for only those points | 7.469 s, 13 cycles | Tail lanes repeatedly carry large active subspaces after most bands converge |
| One-representative adaptive gate | One representative, partial-zone SCF | 5.990 s, 14 cycles | Fast fail-early baseline, not a production timing |
| Finite-shape one-point gate | One representative, partial-zone SCF | 4.904 → 1.135 s, 14 cycles both | Large shape-dispatch benefit without cycle drift |
| Finite-shape paired gate | First 8 representatives, three interleaved SCF pairs | 18.38%, 16.45%, and 9.83% faster; 16.45% median | Candidate was faster in every pair and passed numerical/work gates |

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

Add `--shape-profile` only for scheduler-design diagnostics. It enables detailed
batch events, emits an aggregated Hψ shape inventory and one-tail replay table,
and therefore does not produce an admissible timing sample.

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
| Safe converged-subspace locking | One point: 5.180 → 6.902 s; 14 → 17 cycles; 4.09 → 3.92 GB peak | Removed before the eight-point gate |
| One main plus one Hψ tail shape | Stable profile: 17,024 submitted versus 8,980 logical vector equivalents; the best candidate within the call-growth bound removed 23.0%, below the 25% implementation gate | Stopped before scheduler implementation or timing gates |

The earlier power-of-two shape prototype was initially removed because its
memory tradeoff had not been validated end to end. The reconstructed finite
policy is now retained: its complete run peaked at 7.86 GB, passed the 8 GB
candidate gate, and showed a stable late-run memory plateau under the hard
40 GB process-tree limit.

The symbols associated with rejected implementations must remain absent:
`finite_certified`, `projected_eigh_ragged`, `prediction_tolerance`,
`predicted_gram`, `measure_locked_overlap`, and `compiled_gth_contraction`.
