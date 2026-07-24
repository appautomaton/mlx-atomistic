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

## Scientific EOS validation

Runtime convergence is not scientific validation. The repository now carries a
separate, source-backed equation-of-state (EOS) gate for the existing
zero-temperature, fixed-occupation silicon method. It fits seven energies over
`0.94–1.06 V₀` to a third-order Birch–Murnaghan equation and reports:

| Quantity | Primary all-electron PBE reference |
| --- | ---: |
| Conventional lattice constant | 5.469916 Å |
| Bulk modulus | 88.511 GPa |
| Pressure derivative `B₀′` | 4.3118 |

The primary values are the curated FLEUR/WIEN2k average from the
[Materials Cloud ACWF verification archive](https://archive.materialscloud.org/records/yf0rj-w3r97).
The pinned bundle also includes the CP2K Quickstep TZV2P/GTH curve from the same
archive as a same-pseudopotential-family check, Quantum ESPRESSO SSSP as context,
and the NIST experimental lattice parameter as context only. The reference JSON
is hash-locked and retains source filenames, DOI, URLs, and CC-BY attribution.

The gate deliberately does **not** claim exact ACWF protocol parity. ACWF uses
0.0045 Ry Fermi–Dirac smearing and the `E-TS` free energy; the current product
solver uses 16 fixed occupied bands for insulating silicon. This first gate asks
whether that existing method produces the correct silicon energy curve.

| Admission tier | Δ factor | Lattice | Bulk modulus | `B₀′` |
| --- | ---: | ---: | ---: | ---: |
| Verified | ≤ 3 meV/atom | ≤ 0.5% | ≤ 10% | ≤ 15% |
| Excellent | ≤ 1 meV/atom | ≤ 0.2% | ≤ 5% | ≤ 10% |

The independent numerical-convergence gates are ≤ 1 meV/atom maximum EOS-curve
change, ≤ 0.1% lattice drift, ≤ 3% bulk-modulus drift, and ≤ 10% `B₀′` drift
for the complete 30 Ha / 64³ cutoff curve. A separate three-volume 8³ spot
check requires its central energy shape to remain within 1 meV/atom of the 6³
baseline. A combined 30 Ha / 64³ / 8³ profile remains an optional stress
diagnostic and is not part of the admitted scientific claim. Tolerances are
fixed; a failure is reported rather than widened.

Each geometry runs in a fresh supervised process with a 40 GB process-tree
ceiling. Baseline and cutoff points time out at 180 seconds; 8³ points time out
at 240 seconds. The runner stops on the first numerical failure or nonphysical
central energy triad. Fingerprinted point artifacts make the longer admission
ladder resumable without silently accepting mismatched inputs.

The 56³ profiles retain the production solver's 512 MiB logical compact-batch
ceiling. The 64³ convergence profiles use 768 MiB, matching the 1.49× FFT-volume
increase; this changes only workspace admission, not the Hamiltonian or physics.
The external 40 GB process-tree ceiling remains authoritative.

Inspect the exact three-point screen without running SCF:

```sh
uv run python -m mlx_atomistic.benchmarks.dft_silicon validate-eos \
  --manifest results/mlx-dft-science/workload/manifest.json \
  --gth-source results/mlx-dft-science/workload/resources/Si-GTH-PBE-q4.gth \
  --level screen --dry-run \
  --out results/mlx-dft-science/eos-screen --json
```

Remove `--dry-run` to execute the bounded three-point screen. Use
`--level admission` only after that screen passes. Admission schedules 17
points: complete seven-point baseline and cutoff curves plus the three central
8³ spot checks. Add `--include-combined` only to append the optional three-point
64³/8³ interaction stress profile; it is not required for admission.
Use `--summarize-only` to rebuild the partial scientific report from existing
fingerprinted point artifacts and bounded-failure traces without launching SCF.

The real baseline screen passed on 2026-07-22:

| Volume factor | Lattice | Relative energy | SCF cycles | Wall time | Peak process tree |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.98 | 5.433491 Å | 1.158 meV/atom | 14 | 59.96 s | 7.12 GB |
| 1.00 | 5.470205 Å | 0 | 14 | 60.64 s | 7.85 GB |
| 1.02 | 5.506433 Å | 3.286 meV/atom | 14 | 61.88 s | 7.60 GB |

All three points passed the residual, overlap, electron-count, timeout, and
memory gates. The center is lower than both neighbors. A diagnostic quadratic
through only these three points places the minimum near 5.4613 Å, 0.158% below
the all-electron PBE reference. This is encouraging screening evidence, not an
admitted EOS: seven points are still required for the Birch–Murnaghan fit,
bulk modulus, pressure derivative, Δ factor, and basis convergence.
Raw evidence is under `results/mlx-dft-science/eos-screen/`.

The subsequent admission run completed all 17 required points.
The first 64³ point exposed a 512 MiB compact-workspace ceiling inherited from
the 56³ production profile. The science harness now assigns 768 MiB only to
64³ profiles, proportional to their 1.49× FFT-volume increase. A focused rerun
passed in 119.49 seconds with 8.28 GB peak process-tree memory and a stable
plateau.

All three central convergence triads then passed:

| Profile | Diagnostic minimum | Maximum central-curve change vs baseline |
| --- | ---: | ---: |
| 25 Ha / 56³ / 6³ | 5.461259 Å | — |
| 30 Ha / 64³ / 6³ | 5.461136 Å | 0.0215 meV/atom |
| 25 Ha / 56³ / 8³ | 5.461261 Å | 0.0010 meV/atom |

The complete seven-point baseline curve is scientifically verified against the
all-electron PBE reference:

| Quantity | MLX result | Difference from reference |
| --- | ---: | ---: |
| Conventional lattice constant | 5.460859 Å | 0.166% |
| Bulk modulus | 88.306 GPa | 0.232% |
| Pressure derivative `B₀′` | 4.3052 | 0.153% |
| Lejaeghere Δ factor | 1.942 meV/atom | verified tier |

The Birch–Murnaghan fit has 0.0051 meV/atom RMSE and 0.0103 meV/atom maximum
residual. It passes every verified-tier threshold, but not the stricter
1 meV/atom excellent Δ threshold.

The complete seven-point 30 Ha / 64³ / 6³ curve also passes numerical
convergence against the baseline:

| Convergence metric | Observed | Limit |
| --- | ---: | ---: |
| Maximum EOS-curve change | 0.0583 meV/atom | 1 meV/atom |
| Lattice drift | 0.00086% | 0.1% |
| Bulk-modulus drift | 0.0793% | 3% |
| `B₀′` drift | 1.019% | 10% |

The three-point 8³ spot check passes and tracks the 6³ baseline central curve
within 0.0010 meV/atom, far inside its 1 meV/atom limit. Together with the
complete cutoff curve and all-electron comparison, this admits the 6³ EOS for
the stated practical scope. A full seven-point 8³ curve is deliberately outside
that scope and is not a pending task.

One additional outer 8³ diagnostic is retained as supporting evidence. It first
reached the 240-second limit in low-power mode, then completed an identical
full-power run in 123.19 seconds and 14 SCF cycles with a 9.20 GB peak. No other
outer 8³ points are scheduled.

The optional combined 30 Ha / 64³ / 8³ stress point independently reached the
same 240-second limit, at an 11.44 GB peak with a stable memory plateau. The
timeout was not raised into the excluded 300–445 second range or relabeled as a
scientific pass because this optional profile is outside the admission claim.
The persisted report therefore says **scientifically verified baseline,
cutoff-converged, and 8³ spot-check passed**. Artifacts are under
`results/mlx-dft-science/eos-admission/`.

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
