# Final force-kick fusion verdict on M5 Max

Status: rejected on 2026-08-12.

## Question

The production constrained NVT loop forms the final force kick as an `N x 3`
MLX value and then projects that velocity through the dense composite
SETTLE/SHAKE path. The candidate moved the kick into three existing Metal
stages:

- the analytical SETTLE velocity-delta kernel;
- the disjoint SHAKE-cluster velocity-delta kernel;
- the final dense constraint-delta application kernel.

The intended saving was one full-size intermediate and one elementwise Metal
dispatch per constrained step. CPU execution, overlapping constraints,
non-dense constraint layouts, and runtime profiling retained the existing MLX
fallback.

## Correctness

The candidate passed the complete default pytest suite and all 28 tests in the
opt-in Metal kernel file. A fixed-input comparison against the original MLX
force-kick expression measured a maximum velocity difference of
`1.1175871e-08` and an RMS difference of `1.0709164e-09` on 5DFR. That is much
smaller than the approximately `1.34e-05` constraint-error scale in the
sustained runs. Every admitted 75-, 750-, and 1,500-step production run stayed
finite and passed its runtime checks.

The small difference is expected from regrouping float32 operations. It was
large enough to change some neighbor-list rebuild steps, so sustained timings
must not treat rebuild-count differences as kernel speedups.

## Performance evidence

The host was the admitted Apple M5 Max with 128 GB unified memory, macOS
26.5.2, and Low Power Mode active. Production runs used a 5.5 Angstrom
neighbor skin, a check interval of one step, seed 17, ten warmup steps, and only
final-step sampling and diagnostics. Four-sample values are medians.

| Workload and scope | Control | Candidate | Candidate change | Rebuild evidence |
| --- | ---: | ---: | ---: | --- |
| 5DFR fixed-input projection, 100 interleaved calls | 0.2390 ms | 0.2279 ms | 4.65% faster | none |
| JAC fixed-input projection, 100 interleaved calls | 0.2623 ms | 0.2531 ms | 3.51% faster | none |
| 5DFR, 75 production steps | 0.100909 s | 0.097765 s | 3.12% faster | 2 in every sample |
| JAC, 75 production steps | 0.382514 s | 0.378478 s | 1.06% faster | 2 in every sample |
| 5DFR, 750 production steps | 1.326886 s | 1.295726 s | 2.35% faster | control 22/24/23/23; candidate 21/22/22/22 |
| JAC, 750 production steps | 5.871825 s | 5.884622 s | 0.22% slower | control 22/21/23/21; candidate 21/23/22/22 |
| JAC, 1,500 production steps | 12.305593 s | 12.191351 s | 0.93% faster | control 45/45/46/45; candidate 46/45/45/45 |

The fixed-input and short-run results show that the dispatch fusion itself was
real. It did not become a stable sustained wall-time improvement. Among the
1,500-step samples with 45 rebuilds, one paired run made the candidate 0.51%
slower and another made it 2.41% faster. The execution-order effect was larger
than the retained median. The 750-step JAC median also slightly favored the
control.

An attempted fixed neighbor-check cadence was excluded from the decision.
Changing the check interval from one caused the final diagnostic path to
materialize the full pair list, so it no longer measured the admitted
production route.

Memory was neutral. The 75-step candidate peaks were 228.8 MB for 5DFR and
962.5 MB for JAC, compared with 229.0 MB and 963.3 MB for the controls. The
1,500-step JAC peaks were 1.140 GB for the candidate and 1.141 GB for the
control.

## Decision

Reject the candidate and keep the original MLX force kick. The implementation
required about 580 changed lines, three additional Metal kernel variants, new
dispatch plumbing, and a private composite-constraint route. A small isolated
kernel win that is neutral within sustained JAC wall-time noise does not cover
that maintenance cost.

Revisit this boundary only as part of a broader integration-and-constraint
fusion that removes at least two full-array dispatches, or after a synchronized
profile demonstrates that the final kick has become a material fraction of the
production critical path. Do not revive this exact three-variant design from
the isolated microbenchmark alone.

## Reproducer and raw evidence

The production command shape was:

```text
uv run --no-sync python -m mlx_atomistic.benchmarks.charged_pme runtime \
  --prepared results/dhfr-npt-closure/prepared \
  --warmups 10 --steps 75 --dt-ps 0.004 --temperature-k 300 --seed 17 \
  --neighbor-skin 5.5 --neighbor-check-interval 1 \
  --sample-interval 75 --diagnostic-interval 75 \
  --neighbor-backend mlx_cell_tiles \
  --out results/md-final-kick/5dfr-candidate-1.json
```

JAC used `results/larger-system-scaling/jac-2x2x1-modern/prepared`. The 750-
and 1,500-step runs changed the step, sample, and diagnostic intervals together.
Control runs came from detached commit `df7d065`. Raw JSON reports remain local
and gitignored under `results/md-final-kick/`.
