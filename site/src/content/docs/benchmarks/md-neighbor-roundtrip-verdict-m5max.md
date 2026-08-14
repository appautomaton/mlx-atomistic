---
title: "MD neighbor round-trip verdict on Apple M5 Max"
---

The production `mlx_cell_tiles` route now defers compact exact-pair diagnostics
when every active force term can consume or ignore the tile schedule. Exact
pairs remain available through explicit access and for unsupported custom-force
fallbacks, but they no longer occupy the recurring fixed-cell Particle Mesh
Ewald (PME) force and non-pressure diagnostic path.

## Retained result

| Workload and metric | Control | Candidate | Change |
| --- | ---: | ---: | ---: |
| 23,558-atom 5DFR wall median | 0.124154 s | 0.103989 s | -16.24% |
| 5DFR rebuild median | 16.121 ms | 8.803 ms | -45.39% |
| 5DFR Metal peak allocation | 962.3 MB | 102.8 MB | -89.31% |
| 5DFR resident neighbor estimate | 129.37 MB | 11.77 MB | -90.90% |
| 94,232-atom JAC wall median | 0.774136 s | 0.303246 s | -60.83% |
| JAC Metal peak allocation | 3.988 GB | 0.441 GB | -88.95% |
| JAC resident neighbor estimate | 534.33 MB | 50.28 MB | -90.59% |

The JAC control was noisy. Comparing the candidate median with the fastest
control sample still gives a conservative 14.42% improvement. All eight formal
5DFR samples and all eight formal JAC samples passed their science, route, and
bounded-memory checks.

## Design boundary

An earlier on-demand attempt was 1.8% slower because force binding and
molecular-dynamics diagnostics still requested exact pairs every generation.
The retained design removes those downstream consumers: tile-aware terms state
whether they consume or ignore tiles, non-pressure energy diagnostics use a
Metal tile energy-and-force kernel, and unknown terms fall back to pairs.

The small tile and exact-pair inventory counts now share one host transfer.
Capacity-sized output buffers were not added because historical capacity and
fused-output candidates regressed, while removing the duplicate exact-pair
representation already eliminated the dominant allocation.

The comparison used independent Python 3.13.12 environments, MLX 0.31.2, and
baseline commit `64b5f4ff10f11a91bd52f373424cb5cc33d25057` on Apple M5 Max with
macOS 26.5.2. Raw reports are local and gitignored under
`results/md-neighbor-roundtrip-phase4/`.
