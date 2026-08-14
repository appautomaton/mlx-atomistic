# Hybrid spatial-tile kernel verdict on Apple M5 Max

Date: 2026-08-11

## Decision

The MLX/Metal molecular-dynamics runtime now retains a hybrid spatial-tile
layout. Neighbor membership is evaluated in coarse 8-by-8 atom tiles, and the
same Metal kernel packs each coarse result into its non-empty 4-by-4 subtiles.
One compaction pass emits the exact 4-by-4 representation consumed by the
direct-force kernel.

This design keeps the smaller force-kernel padding without multiplying the
neighbor search and prefix-scan work by four. OpenMM and LAMMPS remain reference
surfaces and do not enter this runtime path. No additional package is required;
the implementation uses the existing Python, MLX, and Metal Shading Language
stack.

## Why the hybrid was selected

The 23,558-atom 5DFR inventory contains 14,699,933 exact pairs. The former
8-by-8 execution layout scheduled 41,405,760 lanes at 35.50% active occupancy.
The retained 4-by-4 execution layout schedules 27,697,296 lanes at 53.07%
occupancy, a 33.11% reduction in padded lanes.

A first candidate changed both search and execution to 4-by-4 tiles. Its
isolated direct kernel reached about 0.598 ms, but rebuild time rose to about
22.03 ms and its complete 75-step median was about 106.80 ms. It was removed.
The retained hybrid performs the search in 8-by-8 tiles and directly compacts
their four 4-by-4 quadrants. In the contemporaneous profile it measured 0.629
ms for the direct kernel against 0.734 ms for the detached-HEAD control, a
14.20% reduction. Rebuild time was 5.54% higher in that synchronized profile,
so complete-wall and sustained measurements, rather than the kernel result
alone, decide retention.

## Complete-wall evidence

All short runs used 10 warmups, 75 measured steps, a 0.004 ps timestep, 300 K,
seed 17, a 9 A cutoff, a 5.5 A neighbor skin, and one-step neighbor checks.
Four independent processes per arm were ordered in two control-candidate-
candidate-control blocks.

| Workload | Control median | Hybrid median | Result |
| --- | ---: | ---: | ---: |
| 5DFR, 23,558 atoms, 75 steps | 0.106646 s | 0.101136 s | 5.17% faster |
| JAC, 94,232 atoms, 75 steps | 0.565313 s | 0.394179 s | Bimodal; not used as the primary claim |

The JAC hybrid samples were 0.291501, 0.496856, 0.290919, and 0.514486
seconds. To avoid claiming the unusually fast mode, the median of only the two
slower hybrid samples is 0.505671 seconds, still 10.55% below the control
median.

The longer paired runs are the primary cross-regime result:

| Workload | Control | Hybrid | Result |
| --- | ---: | ---: | ---: |
| 5DFR, 750 steps | 1.669350 s | 1.555150 s | 6.84% faster |
| JAC, 750 steps | 6.781678 s | 6.557215 s | 3.31% faster |

Every short and sustained runtime sample passed its finite-state, constraint,
route, lazy-topology, neighbor-backend, and no-pair-materialization checks.
The focused Metal suite passed, including periodic-edge, dense, topology,
one-four scaling, energy, and force comparisons against the compact-pair path.

The synchronized pair-versus-tile profile reported a final-state consistency
blocker in both the hybrid run and the detached-HEAD control run in this
session. Immediate force parity remained within the declared gates. The hybrid
maximum force delta was 0.0008545 kJ mol^-1 A^-1 and its root-mean-square delta
was 0.0000682 kJ mol^-1 A^-1. The shared blocker is therefore recorded as
existing atomic-summation trajectory nondeterminism, not treated as evidence
for or against this layout.

## Memory and settling

The smaller execution tiles trade metadata for fewer force lanes.

| Workload | Control Metal peak | Hybrid Metal peak | Resident neighbor estimate |
| --- | ---: | ---: | ---: |
| 5DFR, 750 steps | 116.0 MB | 199.6 MB | 11.8 to 22.7 MB |
| JAC, 750 steps | 493.8 MB | 839.2 MB | 50.4 to 93.9 MB |

The reported process peaks did not increase: the JAC pair measured about 452 MB
and 448 MB. A separate 1,500-step JAC hybrid run completed 47 rebuilds and
passed every runtime check. Its Metal peak was 838.4 MB, active memory was
217.1 MB, allocator cache was 4.062 GB, and process peak was 449.9 MB. The
peak did not grow relative to the 750-step run, and the allocator cache changed
by only about 8 MB across 25 additional rebuilds.

Raw generated evidence is under `results/md-direct-tile4/` and remains
gitignored. The detached control source is commit `1573757`.
