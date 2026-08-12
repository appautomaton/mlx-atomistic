# MD Reciprocal PME Complex-Grid Verdict on M5 Max

Date: 2026-08-12

## Decision

Retain the order-five, force-only reciprocal PME graph simplification. The
interpolation Metal kernel now consumes the unscaled complex inverse-FFT grid
directly and applies the reciprocal grid-size scale to the per-atom force
outputs. The prepared route also stops wrapping positions before calling two
stages that already apply periodic wrapping themselves.

The isolated reciprocal route was effectively neutral on 5DFR and 1.54%
faster on JAC. Two position-balanced 750-step JAC pairs improved complete
production throughput by 3.65% and 1.31%, with 22 neighbor rebuilds in every
run. Their mean wall times were `6.736882 s` for the control and `6.574773 s`
for the candidate, a 2.47% throughput improvement. All runtime checks and the
complete CPU and Metal PME test lanes passed.

## Candidate

The retained order-five GPU force route previously constructed the potential
grid as

```text
real(ifftn(phi_hat)) * grid_size
```

over the complete mesh before its Metal interpolation kernel read 125 grid
points per atom. The JAC mesh contains 1,048,576 points. The candidate passes
the complex `ifftn(phi_hat)` result directly to a force-only interpolation
variant, reads the real component at the required points, and applies
`grid_size` once to each of the three force components. This preserves the
same formula while avoiding a full-grid real projection and scale operation.

The prepared reciprocal input helper also used to wrap every position before
charge assignment and interpolation. Both order-five Metal kernels already
wrap positions while computing fractional coordinates, so the outer wrap was
redundant. Removing it does not change the periodic-coordinate contract.

The energy-and-force route retains the scaled real grid because it also needs
the interpolated scalar potential for energy. Non-order-five and CPU routes
are unchanged. The implementation adds no package or runtime dependency.

## Profile and workload coverage

Synchronized stage profiling on the 94,232-atom JAC artifact measured the
following representative medians before the candidate:

| Reciprocal stage | Median |
| --- | ---: |
| Prepared position wrap | 0.252 ms |
| Order-five charge spread | 0.360 ms |
| Forward FFT | 0.323 ms |
| Influence, inverse FFT, real projection, and scale | 0.427 ms |
| Potential-derivative interpolation | 0.247 ms |
| Complete compiled force-only graph | 0.875 ms |

Stage values are individually synchronized and must not be added to predict
the compiled graph. MLX can collapse dispatch and dependency overhead across
the complete graph. The profile instead identified the only remaining
full-mesh elementwise transform and the duplicated position wrapping as safe
graph work to remove.

A 12-call batched, 12-sample same-process comparison then measured:

| Workload | Atoms | Mesh | Control | Candidate | Result |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5DFR | 23,558 | `64 x 64 x 64` | 0.332396 ms | 0.332276 ms | 0.04% faster |
| JAC | 94,232 | `128 x 128 x 64` | 0.833326 ms | 0.820505 ms | 1.54% faster |

The small-system result is deliberately described as neutral. The removed
full-grid transform becomes more relevant as the PME mesh grows.

## Correctness

The old graph scaled each float grid value before interpolation. The retained
graph interpolates normalized real components and scales the final force.
These expressions are mathematically equivalent but change float32 rounding
order. On immutable production bindings, candidate-minus-control reciprocal
forces measured:

| Workload | RMS delta, kJ/mol/angstrom | Maximum delta, kJ/mol/angstrom |
| --- | ---: | ---: |
| 5DFR | `2.924e-6` | `2.193e-5` |
| JAC | `2.753e-6` | `2.213e-5` |

The Metal unit test compares the complex-grid and retained real-grid kernels
against the MLX numerical route. The compiled production test also verifies
that forces are invariant when positions are shifted by whole box lengths.

Verification commands and outcomes were:

```text
uv run --no-sync pytest -q tests/test_pme.py --run-gpu
57 passed

uv run --no-sync pytest -q tests/test_pme.py
54 passed, 3 skipped

uv run --no-sync ruff check src tests
All checks passed!
```

## End-to-end retention gate

Short 75-step runs showed substantial host and neighbor-update variation. The
three-run medians were `0.427663 s` for the control and `0.423031 s` for the
candidate, a 1.09% throughput improvement, but one control was an obvious
neighbor-update outlier. That result only triggered the longer gate; it was
not used alone to retain the code.

The 750-step position-balanced pairs were consistent:

| Pair order | Control | Candidate | Candidate throughput | Rebuilds |
| --- | ---: | ---: | ---: | ---: |
| Control first | 6.727580 s | 6.490785 s | 3.65% faster | 22 / 22 |
| Candidate first | 6.746184 s | 6.658760 s | 1.31% faster | 22 / 22 |
| Two-run mean | 6.736882 s | 6.574773 s | 2.47% faster | equal |

Individual neighbor update and rebuild timers still varied between runs, so
2.47% is an observed end-to-end result rather than a claim that every point
comes from the reciprocal graph. The important retention evidence is that the
isolated JAC route improved, both long complete simulations moved in the same
direction, rebuild counts matched, and the small 5DFR workload did not regress.

## Reproducer and boundary

The complete-wall command shape was:

```text
uv run --no-sync python -m mlx_atomistic.benchmarks.charged_pme runtime \
  --prepared results/larger-system-scaling/jac-2x2x1-modern/prepared \
  --warmups 10 --steps 750 --dt-ps 0.004 --temperature-k 300 --seed 17 \
  --neighbor-skin 5.5 --neighbor-check-interval 1 \
  --sample-interval 750 --diagnostic-interval 750 \
  --neighbor-backend mlx_cell_tiles \
  --out results/md-reciprocal-complex-grid/jac-candidate-750-1.json
```

Control runs came from detached commit `91b7b0b`. Raw reports remain local and
gitignored under `results/md-reciprocal-complex-grid/`.

Do not generalize this result into a complex-grid energy path. Energy requires
the scalar potential and has different data-flow economics. The next
reciprocal PME candidate should be selected from a fresh synchronized profile,
with charge-spread launch geometry as the leading kernel-level question.
