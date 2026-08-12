# Active-column spatial kernel verdict on Apple M5 Max

Date: 2026-08-11

## Decision

The MLX/Metal molecular-dynamics runtime now compacts every non-empty right-atom
column from its retained 4-by-4 force tiles. A 32-lane single-instruction,
multiple-data (SIMD) group consumes up to 32 compact column descriptors that
share one left atom block. The descriptor is an `int32` value encoding
`4 * tile_index + right_column`.

The layout is retained because it reduces complete-wall time on both the
23,558-atom 5DFR workload and the 94,232-atom JAC workload. It adds no package
dependency. The implementation uses the existing Python, MLX, and Metal Shading
Language stack. OpenMM and LAMMPS remain reference surfaces and are not on the
runtime path.

## Kernel and rebuild evidence

The 5DFR geometry contains 1,731,081 non-empty 4-by-4 tiles and 14,699,933 exact
pairs. The former schedule dispatched 7,015,712 column lanes in 219,241 SIMD
groups. Column compaction retained 5,068,119 active columns and dispatched
5,169,152 lanes in 161,536 groups. This is a 26.3% reduction in dispatched
column lanes and force groups.

In a synchronized route profile, 74 direct-force evaluations fell from
44.716 ms to 40.147 ms, a 10.2% reduction. The isolated direct kernel fell from
0.967 ms to 0.534 ms. Rebuild time increased from 20.896 ms to 22.078 ms because
the builder emits the additional descriptor array.

An initial builder prototype counted active columns over every coarse 8-by-8
candidate before tile compaction. That broadcast-heavy implementation raised
the measured rebuild to about 183 ms and was removed. The retained builder
counts only the already-compacted, non-empty 4-by-4 tiles. It accepts one small
host count synchronization rather than materializing work over the much larger
coarse candidate inventory.

## Complete-wall evidence

All sustained samples used independent processes, 10 warmups, 750 measured
steps, a 0.004 ps timestep, 300 K, seed 17, a 5.5 A neighbor skin, and one-step
neighbor checks. Four samples per arm were interleaved as two
control-candidate-candidate-control blocks.

| Workload | Control median | Active-column median | Result |
| --- | ---: | ---: | ---: |
| 5DFR, 23,558 atoms | 1.459423 s | 1.370360 s | 6.10% faster |
| JAC, 94,232 atoms | 5.965862 s | 5.850388 s | 1.94% faster |

The 5DFR active-column samples were tightly grouped between 1.362531 and
1.380809 seconds. The controls ranged from 1.327786 to 1.531581 seconds because
the host entered two performance modes. JAC showed the same host behavior: the
active-column samples ranged from 5.484679 to 6.177216 seconds and controls
ranged from 5.742649 to 6.223132 seconds. Medians across the symmetric,
interleaved samples are therefore the retained complete-wall comparison.

Every standalone runtime sample passed its finite-state, fixed-cell,
constraint-route, neighbor-backend, lazy-topology, no-pair-materialization,
memory, and plan-reuse checks. The focused Metal suite passed 27 tests,
including periodic-edge, dense, topology, one-four scaling, energy, and force
comparisons with the compact-pair path.

The synchronized profile retained the previously recorded final-state
consistency blocker caused by atomic-summation trajectory nondeterminism. The
immediate force comparison remained within its declared gates: the maximum
delta was 0.000641 kJ mol^-1 A^-1 and the root-mean-square delta was 0.000068
kJ mol^-1 A^-1. Standalone runtime checks, rather than that nondeterministic
trajectory comparison, are the scientific acceptance surface for this change.

## Memory and settling

The compact descriptors trade memory for less direct-force work.

| Workload | Control Metal peak | Active-column Metal peak | Process peak |
| --- | ---: | ---: | ---: |
| 5DFR, 750 steps | about 200 MB | about 273 MB | about 192 MB |
| JAC, 750 steps | about 839 MB | about 1.141 GB | about 446 MB |

The 5DFR tile-geometry estimate rose from 22.6 MB to 42.5 MB. On JAC, active
MLX memory rose from about 217 MB to about 297 MB, while the process peak stayed
slightly below the control. A separate 1,500-step JAC run completed 46 rebuilds
and passed every runtime check. Its Metal peak was 1.143 GB, active memory was
297.0 MB, allocator cache was 4.011 GB, and process peak was 446.3 MB. The peak
did not grow relative to the 750-step samples.

Raw generated evidence is under `results/md-active-columns/` and remains
gitignored. The detached control source is commit `999a632`.

## Follow-up threadgroup sweep

A follow-up sweep on 2026-08-12 compared one, two, four, and eight SIMD groups
per Metal threadgroup without changing the active-column geometry or force
arithmetic. Isolated direct-kernel measurements eliminated one and eight
groups because neither produced a repeatable advantage across 5DFR and JAC.
Two groups showed the best isolated JAC result and advanced to complete-wall
testing against the retained four-group configuration.

Four independent 75-step samples per arm used the same 10 warmups, timestep,
temperature, seed, neighbor skin, and check interval as the sustained tests
above. On 5DFR, two groups had a 98.079 ms median against 98.933 ms for four
groups, a 0.86% reduction. On JAC, two groups had a 379.187 ms median against
376.750 ms for four groups, a 0.65% regression. Every run passed its runtime
checks, and peak Metal memory was unchanged within allocator noise.

The two-group candidate was rejected because its small 5DFR result did not
transfer to JAC and both differences were below the observed host-mode
variation. Four SIMD groups per threadgroup remains the runtime configuration.
The raw sweep is under `results/md-threadgroup-tuning/` and remains gitignored.
