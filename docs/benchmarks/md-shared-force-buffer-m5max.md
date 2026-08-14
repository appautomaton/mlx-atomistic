# Shared Bonded and Sparse PME Force Buffer on Apple M5 Max

Date: 2026-08-14

## Decision

Retain one shared Metal force buffer for fused bonded interactions and sparse
Particle Mesh Ewald (PME) corrections. The placement reduces median complete
trajectory wall time by 1.77% on 5DFR, 0.90% on JAC, and 0.68% on GPCRmd while
preserving the existing Direct Space kernel, diagnostic ownership, fallbacks,
and memory envelope.

## Root cause

The prepared PME path previously produced three full atom-force arrays for
Direct Space, reciprocal PME, and sparse topology corrections, then added them.
The force pipeline separately produced one fused bonded array and added that to
the nonbonded result. Sparse corrections contain only topology-owned exclusions,
exceptions, and 1-4 pairs, but their standalone dispatch still initialized an
`atom_count x 3` output and required another complete-array aggregation.

The retained route changes ownership only when all of the following are true:

- Metal is active;
- the prepared PME binding exposes its validated sparse correction inventory;
- the force pipeline already has a fused bonded Metal binding.

The correction pairs become a final work family in the fused bonded dispatch.
Both families use atomic accumulation into its existing output. The bound PME
term then returns Direct Space plus reciprocal PME without emitting the
standalone correction array. CPU execution, unsupported pipelines, force terms
without a fused bonded owner, energy diagnostics, and explicit correction APIs
retain the previous route.

This is the same shared-force-buffer principle used by mature GPU runtimes, but
the implementation remains entirely inside the project-owned MLX and Metal
runtime.

## Placement experiment

The first candidate appended correction threadgroups to the recurring spatial
Direct Space kernel. Its synchronized profile removed the standalone 182.7 ms
correction stage over 750 JAC steps, but it enlarged the hottest kernel. Three
clean interleaved runs rejected that placement:

| JAC placement | Median seconds/step | Result |
| --- | ---: | ---: |
| Existing separate correction dispatch | 0.00341405 | control |
| Correction work appended to Direct Space | 0.00343016 | 0.47% slower |

Two of three paired directions regressed. That source was removed before the
retained placement was implemented. Raw outputs remain under
`results/md-suite/fused-direct-corrections/`.

Appending the same work to the smaller bonded dispatch leaves Direct Space
unchanged. In the synchronized JAC profile, the previous bonded and sparse
correction routes totaled 347.9 ms over 750 evaluations. The retained combined
route took 190.2 ms, a 45.3% reduction at that explicit completion boundary.
The standalone `pme_exceptions_corrections` route is absent. These intrusive
stage times establish work ownership, not complete-trajectory speed.

## Interleaved complete-trajectory A/B

Each arm ran 10 warmup and 750 measured fixed-cell NVT steps with seed 17, a
4 fs timestep, 9 Angstrom cutoff, 5.5 Angstrom neighbor skin, Metal spatial
tiles, and boundary-only sampling and diagnostics. The order for each workload
was control 1, candidate 1, candidate 2, control 2, control 3, candidate 3.

| Workload | Atoms | Control median | Candidate median | Step-wall reduction | Throughput increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5DFR | 23,558 | 1.1489 ms | 1.1285 ms | 1.77% | 1.80% |
| JAC 4-cell | 94,232 | 3.6574 ms | 3.6246 ms | 0.90% | 0.90% |
| GPCRmd 729 | 92,001 | 4.2750 ms | 4.2459 ms | 0.68% | 0.69% |

All three paired directions improved on 5DFR and JAC. Two of three improved on
GPCRmd; its final candidate ran after the local frequency trend had declined.
Every one of the 18 runs passed finite-state, constraint, memory, fixed-cell,
lazy-topology, neighbor-representation, and PME plan-reuse checks. Rebuild
counts matched within every pair. Median MLX peak allocation changed by less
than 1.1 MiB on every workload.

Raw outputs are under
`results/md-suite/fused-bonded-corrections/final-ab/`. The structural profiles
are `results/md-suite/fused-direct-corrections/jac-control-stage-profile.json`
and
`results/md-suite/fused-bonded-corrections/jac-candidate-stage-profile.json`.

## Correctness

Fixed-position parity covers a spatial-tile PME binding with exclusions, 1-4
scales, an explicit exception, fused bond and angle work, and the shared sparse
correction buffer. The combined result matches the previous independently
evaluated terms inside the established float32 atomic tolerance.

The full 36-case fused bonded/nonbonded Metal suite passes. The 99-case CPU
force-runtime, forcefield, neighbor, and MD-suite regression set also passes.
CPU paths never select the shared Metal ownership.

## Reproducer

One arm was run with:

```bash
uv run --no-sync python -m mlx_atomistic.benchmarks.charged_pme runtime \
  --prepared results/larger-system-scaling/jac-2x2x1-modern/prepared \
  --warmups 10 \
  --steps 750 \
  --dt-ps 0.004 \
  --temperature-k 300 \
  --seed 17 \
  --neighbor-skin 5.5 \
  --neighbor-check-interval 1 \
  --sample-interval 750 \
  --diagnostic-interval 750 \
  --neighbor-backend mlx_cell_tiles \
  --out results/md-suite/fused-bonded-corrections/final-ab/jac-candidate-1.json
```

The control worktree was detached at `4625326`. Candidate runs used the same
environment and prepared artifacts with only the uncommitted shared-buffer
change present.
