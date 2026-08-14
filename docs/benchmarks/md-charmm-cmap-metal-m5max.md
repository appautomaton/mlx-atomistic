# CHARMM CMAP Metal Force Kernel on Apple M5 Max

Date: 2026-08-14

## Decision

Retain the prepared CHARMM bonded-force expansion. Urey-Bradley 1-3 distance
terms now join the harmonic-bond records, and CHARMM correction map (CMAP)
forces run as a fifth family in the recurring fused bonded Metal dispatch. On
the 92,001-atom GPCRmd 729 workload, the combined change reduces median clean
step time by 20.94%, from 5.8741 to 4.6439 ms/step. That is a 26.49% throughput
increase, from 58.83 to 74.42 ns/day.

The result is a general CHARMM force-path optimization, not a GPCRmd-specific
parameter choice. The runtime consumes prepared term arrays and coefficient
tables for any admitted orthorhombic Metal system. The CPU and diagnostic
energy paths keep the existing MLX formulas.

## Removed recurring work

The original GPCRmd force-only path evaluated 49,223 Urey-Bradley terms with a
standalone MLX scatter and 317 CMAP terms through MLX automatic differentiation
on every force evaluation. It then added each full 92,001-by-3 force array into
the running total.

The retained path changes that ownership:

- Urey-Bradley records reuse the harmonic-bond formula inside the bonded
  kernel. Their angle-triplet endpoints become one 1-3 bond record.
- One CMAP worker evaluates both signed dihedrals, selects one precomputed
  periodic bicubic patch, differentiates the patch analytically with respect
  to both angles, and scatters both four-atom force contributions.
- Standard five-atom CHARMM CMAP overlap remains correct because the two
  dihedrals use atomic accumulation into the same output.
- A CMAP-only force pipeline is eligible for the Metal route. Unsupported
  devices and non-orthorhombic cells retain the existing MLX fallback.
- Diagnostics still use the established energy implementation. The optimized
  route is force-only and does not alter reported energy semantics.

This removes the recurring automatic-differentiation graph, the standalone
CMAP dispatch, and one full-force-array aggregation. The preceding
Urey-Bradley fusion separately removes another dispatch and aggregation.

## Clean interleaved A/B

The machine was an Apple M5 Max with AC Low Power Mode disabled, MLX 0.31.2,
and Python 3.13.12. Each process used the same prepared GPCRmd 729 artifact,
seed 17, 4 fs timestep, 9 A cutoff, 5.5 A neighbor skin, Metal spatial tiles,
10 warmup steps, and 750 measured fixed-cell NVT steps. Sampling and
diagnostics occurred only at the measurement boundary.

The order was control 1, candidate 1, candidate 2, control 2, control 3,
candidate 3 so process position did not favor one implementation.

| Sample | Control ms/step | Candidate ms/step | Throughput increase |
| --- | ---: | ---: | ---: |
| 1 | 5.8741 | 4.6884 | 25.29% |
| 2 | 5.9003 | 4.6439 | 27.06% |
| 3 | 5.8480 | 4.6393 | 26.05% |
| Median | 5.8741 | 4.6439 | 26.49% |

All six runs passed their finite-state, constraint, memory, fixed-cell,
neighbor-representation, lazy-topology, and PME-plan reuse checks. The raw
outputs are under `results/md-suite/cmap-ab/`.

The control is commit `4f22a86`, before either CHARMM fusion. A separate
position-balanced Urey-Bradley-only A/B measured a 0.75% median throughput
increase. Nearly all of the combined result therefore belongs to eliminating
the recurring CMAP automatic-differentiation path, but the clean table above
intentionally reports the exactly interleaved combined change.

## Synchronized structural profile

The synchronized profiler is intrusive and is not a throughput measurement.
It does show that the intended route disappeared:

| Bonded route | Wall over 749 force evaluations | Force aggregations |
| --- | ---: | ---: |
| Original, standalone Urey-Bradley and CMAP | 1.3689 s | 2,247 |
| Urey-Bradley fused, standalone CMAP | 1.2284 s | 1,498 |
| Urey-Bradley and CMAP fused | 0.1962 s | 749 |

The final route has one `bonded_fused` call per ordinary force evaluation and
no standalone `urey_bradley` or `charmm_cmap` call. Its raw profile is
`results/md-suite/gpcr-cmap-fused-profile.json`.

## Correctness evidence

Six Metal parity cases compare the fused result with the existing MLX force
formulas. They cover CMAP-only execution, two map identifiers, a non-separable
two-angle surface, the standard five-atom overlapping topology, and periodic
boundary crossing. All pass at 3e-5 relative and absolute tolerance on the
small deterministic systems.

The real GPCRmd prepared coordinates add 317 CMAP terms over 92,001 atoms. The
candidate-versus-existing-MLX comparison produced:

| Metric | Result |
| --- | ---: |
| Relative L2 force delta | 8.08e-6 |
| RMS component delta | 1.30e-5 |
| Maximum component delta | 0.00280 |
| Maximum reference component | 79.91 |
| Components above 0.001 | 11 of 276,003 |

Both arrays were finite and their net forces remained within 4e-5 of zero per
axis. The small difference comes from analytical evaluation and atomic
accumulation order rather than a physics change. The raw result is
`results/md-suite/cmap-gpcr-force-parity.json`.

## Reproduction

The whole-step profile was:

```bash
uv run python -m mlx_atomistic.benchmarks.md_suite profile \
  --case gpcrmd-729-pme \
  --warmup-steps 10 \
  --measured-steps 750 \
  --out results/md-suite/gpcr-cmap-fused-profile.json
```

Each clean A/B sample used:

```bash
uv run python -m mlx_atomistic.benchmarks.charged_pme runtime \
  --prepared results/gpcrmd-pme-runtime-closure/prepared \
  --warmups 10 \
  --steps 750 \
  --neighbor-backend mlx_cell_tiles \
  --sample-interval 750 \
  --diagnostic-interval 750 \
  --out results/md-suite/cmap-ab/SAMPLE.json
```

The retained conclusion is bounded to recurring force-only execution on the
orthorhombic Metal path. It does not claim a triclinic route, change energy or
virial diagnostics, or replace complete-force and trajectory validation.
