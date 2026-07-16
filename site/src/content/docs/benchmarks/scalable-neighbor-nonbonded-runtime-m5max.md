---
title: "Scalable Neighbor-Listed Nonbonded Runtime (M5 Max)"
---


Date: 2026-07-13

Status: `diagnostic`. This report validates the MLX neighbor/nonbonded runtime
through 92,001 atoms on a synthetic orthorhombic system. This row does not
claim a GPCRmd production run, PME-at-scale support, or an OpenMM/LAMMPS
performance ratio; charged PME is validated separately on the JAC workload.

## Result

`mlx_cell_pairs` matched the tiled all-pairs MLX oracle at every requested size:
1,000, 4,000, 16,000, 50,000, and 92,001 atoms. The topology-bearing 1,000-atom
case included two exclusions (one explicit and one bond-derived) and one 1-4
pair with LJ/Coulomb scales of 0.5/0.75. The lazy topology's dense pair cache was
never materialized.

| atoms | compact pairs | candidates | waste | build (ms) | pair force (ms) | tiled oracle (s) | abs dE | rel dE | max abs dF |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 34,278 | 236,788 | 85.52% | 90.48 | 32.28 | 0.025 | 1.22e-4 | 1.05e-7 | 2.35e-6 |
| 4,000 | 108,000 | 998,000 | 89.18% | 22.94 | 3.59 | 0.038 | 0 | 0 | 5.36e-7 |
| 16,000 | 425,760 | 3,482,556 | 87.77% | 86.87 | 8.15 | 0.517 | 1.95e-3 | 1.12e-7 | 7.00e-7 |
| 50,000 | 1,314,708 | 10,517,242 | 87.50% | 267.24 | 24.35 | 12.955 | 7.81e-3 | 1.35e-7 | 9.54e-7 |
| 92,001 | 2,432,693 | 16,902,196 | 85.61% | 545.01 | 67.69 | 112.055 | 4.69e-2 | 4.56e-7 | 8.49e-7 |

The 92,001-atom explicitly synchronized static decomposition was 89.0% neighbor
build and 11.0% compact pair-force evaluation. The tiled all-pairs oracle took
1,655x as long as the compact pair evaluation for the same energy and forces.
The unusually high first-row timings include MLX compilation/warm-up work; the
larger rows show the steady scaling shape.

## Two-step runtime row

A separate 92,001-atom synthetic Langevin run exercised the full MD loop with a
1.2 neighbor skin:

| atoms | steps | wall (s) | steps/s | backend | compact pairs | candidates | waste | rebuild (s) | max RSS | finite |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 92,001 | 2 | 1.143 | 1.749 | `dynamic-neighbor+mlx_cell_pairs` | 7,869,217 | 52,355,335 | 84.97% | 0.411 | 340 MiB | yes |

The run completed with no fallback and relative total-energy drift of
`-3.55e-7`. The MD report's `force_eval_wall_s` measures asynchronous enqueue
time, so it is not used as the compute-time split above. The synchronized static
parity row supplies the build-versus-force wall times.

## Provenance

- Host: Apple M5 Max, 18 CPU cores, 128 GB unified memory.
- Runtime: Python 3.13.12, MLX 0.31.2, `Device(gpu, 0)`, Metal available.
- Commit used for measurement: `9c43d83`.
- Cell: orthorhombic FCC lattice at reduced density 0.8.
- Nonbonded model: heterogeneous LJ parameters and alternating direct-cutoff
  charges; cutoff 2.5; no PME.
- Pair path: `mlx_cell_pairs`, CPU `argwhere` compaction, unsorted output, no
  fallback.
- Reference path: tiled MLX all-pairs evaluation. Triclinic compact execution
  remains fail-closed.

Raw outputs (gitignored):

- `results/scalable-neighbor-nonbonded-runtime/parity.json`
- `results/scalable-neighbor-nonbonded-runtime/synthetic-runtime-92001.json`

## Reproduce

```bash
uv run python -m mlx_atomistic.benchmarks.neighbor_nonbonded_parity \
  --sizes 1000,4000,16000,50000,92001 \
  --out results/scalable-neighbor-nonbonded-runtime/parity.json

uv run python -m mlx_atomistic.benchmarks.md_performance \
  --sizes 92001 --steps 2 --mode dynamic-neighbor \
  --sample-interval 2 --diagnostic-interval 2 --evaluation-interval 2 \
  --json-out results/scalable-neighbor-nonbonded-runtime/synthetic-runtime-92001.json
```

The GPCRmd cache was absent when this 2026-07-13 measurement was made, so this
report contains no real-fixture result. A later source-backed run uses the
source protocol and the production `NeighborBlocks` PME path; its commands and
evidence are recorded separately in
[`gpcrmd-729-pme-runtime-m5max.md`](./gpcrmd-729-pme-runtime-m5max.md).

## Comparison status and next blocker

This row is `diagnostic`: the new parity workload uses heterogeneous LJ+Coulomb
parameters and topology semantics, while the existing OpenMM/LAMMPS scaling row
uses a uniform reduced-unit LJ fluid. A ratio would mix physics and is therefore
not reported.

The neighbor/topology axis is proven at the target atom count. A separate
94,232-atom charged JAC report now validates fixed-cell orthorhombic PME with an
explicit neutralizing-plasma policy; see
[`scalable-charged-pme-runtime-m5max.md`](./scalable-charged-pme-runtime-m5max.md).
That does not turn this synthetic row into a GPCRmd result. The membrane fixture
has since passed a separate bounded fixed-cell NVT parity/runtime/restart gate.
Production NPT, analytic PME virial, triclinic PME, production-length stability,
and broad membrane readiness remain deferred.
