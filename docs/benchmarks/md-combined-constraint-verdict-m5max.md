# Combined position/velocity constraint verdict on M5 Max

Status: rejected on 2026-08-12.

## Question

The constrained production NVT loop projects predicted positions, writes the
full position array, derives a velocity correction from the position change,
projects non-SETTLE velocities through RATTLE, and writes the full velocity
array. For the dense disjoint SETTLE/SHAKE route this is approximately six
dependent Metal dispatches.

The candidate changed the SETTLE and SHAKE position kernels to emit both
position and pre-force velocity deltas. The SHAKE kernel continued directly
into its RATTLE iteration while the final constrained positions were still in
registers. A dual-output dense kernel then wrote constrained positions and
velocities together. The intended production route had three dispatches and no
full-size position-correction intermediate. CPU execution, overlapping
constraints, non-dense layouts, and synchronized runtime profiling retained
the original path.

## Correctness

Ruff and Python compilation passed. All 15 CPU constraint tests passed, and
three targeted Metal tests covered periodic numerical parity, production-route
selection, and overlap/profiling fallback.

On deterministic fixed inputs, candidate positions were bit-identical to the
control on both workloads. The maximum velocity difference was
`5.9604645e-08`; RMS differences were `9.0294359e-09` on 5DFR and
`7.5153332e-09` on JAC. Every admitted production sample stayed finite and
passed its runtime checks.

## Performance evidence

The host was the admitted Apple M5 Max with 128 GB unified memory, macOS
26.5.2, and Low Power Mode active. Production runs used a 5.5 Angstrom
neighbor skin, a check interval of one step, seed 17, ten warmup steps, and
only final-step sampling and diagnostics. Four-sample values are medians.

| Workload and scope | Control | Candidate | Candidate change | Rebuild evidence |
| --- | ---: | ---: | ---: | --- |
| 5DFR fixed-input constraint boundary, 100 interleaved calls | 0.2864 ms | 0.2376 ms | 17.06% faster | none |
| JAC fixed-input constraint boundary, 100 interleaved calls | 0.3150 ms | 0.2623 ms | 16.74% faster | none |
| 5DFR, 75 production steps | 0.102356 s | 0.101897 s | 0.45% faster | 2 in every sample |
| JAC, 75 production steps | 0.377238 s | 0.375831 s | 0.37% faster | 2 in every sample |
| 5DFR, 750 production steps | 1.603715 s | 1.491160 s | 7.02% faster | control 23/23/22/23; candidate 22/23/23/23 |
| JAC, 750 production steps | 6.039981 s | 6.096518 s | 0.94% slower | 22 in every sample |

The isolated dispatch reduction was real, but it did not transfer consistently
to the complete asynchronous production critical path. The short-run gains
were below one percent. More importantly, JAC held rebuild count constant in
all eight sustained samples and the candidate regressed in three of four
paired runs. The 1,500-step extension was stopped because the equal-rebuild
JAC retention gate had already failed.

Memory was neutral: 75-step peak Metal allocation changed from 229.04 MB to
229.03 MB on 5DFR and from 963.32 MB to 963.34 MB on JAC. At 750 JAC steps the
candidate peak was about 0.84 MB higher.

## Decision

Reject the candidate and retain the existing staged constraint route. The
prototype required 633 changed source lines, three additional Metal kernels,
and dual-output routing. A 17% isolated boundary win does not justify that
cost when the larger equal-rebuild production workload regresses by 0.94%.

The result also narrows the next optimization search: dispatch count alone is
not the governing JAC bottleneck while the force pipeline is submitted
asynchronously. Future constraint work should begin from a synchronized
critical-path profile or fuse the constraint step with an adjacent integration
stage, rather than combining only the already-dependent constraint kernels.

## Reproducer and raw evidence

The production command shape was:

```text
uv run --no-sync python -m mlx_atomistic.benchmarks.charged_pme runtime \
  --prepared results/dhfr-npt-closure/prepared \
  --warmups 10 --steps 75 --dt-ps 0.004 --temperature-k 300 --seed 17 \
  --neighbor-skin 5.5 --neighbor-check-interval 1 \
  --sample-interval 75 --diagnostic-interval 75 \
  --neighbor-backend mlx_cell_tiles \
  --out results/md-combined-constraint/5dfr-candidate-1.json
```

JAC used `results/larger-system-scaling/jac-2x2x1-modern/prepared`. The
750-step runs changed the step, sample, and diagnostic intervals together.
Control runs came from detached commit `9efb38b`. Raw reports remain local and
gitignored under `results/md-combined-constraint/`.
