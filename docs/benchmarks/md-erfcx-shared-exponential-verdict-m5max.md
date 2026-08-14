# MD Shared-Exponential `erfcx` Verdict on M5 Max

Date: 2026-08-12

## Decision

Reject the force-only direct-space `erfcx` approximation. It reduced the
isolated spatial-tile kernel in one stable JAC sample, but it did not improve
the complete production critical path. Two equal-rebuild 75-step JAC pairs
were neutral to slightly slower, and the longer 750-step candidate was 4.00%
slower while also triggering one additional neighbor rebuild. The candidate
source and its test were removed. No runtime behavior changed.

## Candidate

The production spatial-tile kernel evaluates both
`erfc(alpha * distance)` and `exp(-(alpha * distance)^2)` for every valid
charged pair. The candidate kept the existing low-argument path and replaced
the high-argument force-only path with

```text
erfc(x) = exp(-x^2) * erfcx(x)
```

where `erfcx(x)` used a normalized degree-eight polynomial on
`[0.927734375, 3.2]`. This let both screened-Coulomb force terms share one
exponential. The energy and diagnostic kernels and the compact-pair force
kernel remained unchanged.

The polynomial was generated offline with NumPy and introduced no package or
runtime dependency. A 1,000,001-point float32 validation grid measured a
maximum `erfcx` relative error of `3.103e-6`, maximum `erfc` absolute error of
`3.098e-7`, and maximum final force-coefficient relative error of `4.901e-7`.

## Workload coverage and correctness

A deterministic sample of 250,000 spatial tiles from each production artifact
confirmed that the high branch covered nearly all cutoff pairs:

| Workload | Alpha | `alpha * cutoff` | Sampled cutoff pairs | High-branch fraction |
| --- | ---: | ---: | ---: | ---: |
| 5DFR, 23,558 atoms | 0.292029 | 2.628261 | 509,057 | 96.19% |
| JAC, 94,232 atoms | 0.350000 | 3.150000 | 503,678 | 98.03% |

The sampled argument medians were `2.0891` on 5DFR and `2.5054` on JAC. The
99th percentiles were `2.6195` and `3.1395`, respectively. The approximation
therefore targeted the measured production distribution rather than an
arbitrary broad range.

Metal tests covered arguments `0.95`, `1.5`, `2.0`, `2.6`, and `3.14` and
passed against the analytic screened-Coulomb formula. At `3.14`, the candidate
was closer to the analytic value than the retained MLX-derived `erfc`
approximation, so the later rejection is not a numerical-accuracy failure.

On complete immutable production bindings, candidate-minus-control direct
forces had the following deltas:

| Workload | RMS delta, kJ/mol/angstrom | Maximum delta, kJ/mol/angstrom |
| --- | ---: | ---: |
| 5DFR | `2.91e-5` | `2.75e-4` |
| JAC | `2.74e-5` | `3.05e-4` |

## Performance evidence

All measurements ran on the Apple M5 Max under AC Low Power Mode. Extended
back-to-back kernel work moved the host through several performance states, so
absolute times are not comparable to the normal historical baseline. Controls
and candidates were interleaved or position-balanced within each decision
group.

A 12-call batched, 12-sample same-process JAC kernel comparison measured
`4.0554 ms` for the control and `3.9471 ms` for the candidate, a 2.67% isolated
improvement. A later throttled repeat showed a larger JAC difference but also a
30.30% 5DFR regression, demonstrating that the isolated result was not stable
enough to decide retention.

The complete 75-step JAC path was stable and decisive:

| Pair | Control | Candidate | Candidate result | Rebuilds |
| --- | ---: | ---: | ---: | ---: |
| Control first | 6.302974 s | 6.305974 s | 0.05% slower | 2 / 2 |
| Candidate first | 6.350454 s | 6.357553 s | 0.11% slower | 2 / 2 |
| Two-run median | 6.326714 s | 6.331764 s | 0.08% slower | equal |

The candidate reduced the separately reported force-evaluation probe in both
75-step pairs, but that work was hidden by the asynchronous production
pipeline. Removing arithmetic inside one kernel did not shorten the complete
wall-time critical path.

The 750-step control completed in `75.162928 s` with 21 rebuilds. The candidate
completed in `78.167471 s` with 22 rebuilds, 4.00% slower. Because the rebuild
counts differ, this row is supporting evidence rather than a clean speed ratio.
It did not rescue the already-failed equal-rebuild 75-step gate. The planned
1,500-step run was therefore skipped.

## Reproducer and boundary

The complete-wall command shape was:

```text
uv run --no-sync python -m mlx_atomistic.benchmarks.charged_pme runtime \
  --prepared results/larger-system-scaling/jac-2x2x1-modern/prepared \
  --warmups 10 --steps 75 --dt-ps 0.004 --temperature-k 300 --seed 17 \
  --neighbor-skin 5.5 --neighbor-check-interval 1 \
  --sample-interval 75 --diagnostic-interval 75 \
  --neighbor-backend mlx_cell_tiles \
  --out results/md-erfcx-shared-exp/jac-candidate-75-1.json
```

The 750-step runs changed the step, sample, and diagnostic intervals together.
Control runs came from detached commit `7f7b878`. Raw reports remain local and
gitignored under `results/md-erfcx-shared-exp/`.

Do not revive this degree-eight shared-exponential design without evidence that
the direct kernel has moved onto the synchronous critical path. The next MD
optimization target should be the independently synchronized reciprocal PME
route, not another approximation inside direct-space arithmetic.
