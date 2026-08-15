# MLX-First MD Acceleration

This page describes the current production Molecular Dynamics (MD) execution
path. Historical experiments and their retain/reject decisions live in the
[MD performance decision ledger](./benchmarks/md-performance-decisions-m5max.md).
Raw benchmark output remains local under gitignored `results/`.

## Scope

`mlx_atomistic` is the runtime. OpenMM and LAMMPS are reference and validation
surfaces only. The optimized path stays within Python, MLX, and focused Metal
kernels on Apple Silicon.

The primary performance target is recurring fixed-cell, constrained,
Particle Mesh Ewald (PME) simulation. Energy diagnostics, unsupported force
terms, CPU execution, and non-orthorhombic cells retain conservative fallback
routes.

## Production Step

A constrained Langevin step currently follows this ownership order:

1. Drift positions and apply position constraints.
2. Apply the pre-force velocity projection.
3. Admit or rebuild the Neighbor generation.
4. Rebind the prepared force pipeline only when that generation changes.
5. Submit direct nonbonded, reciprocal PME, bonded, and sparse correction work.
6. Apply the final kick and velocity constraints.
7. Materialize state only for an explicit diagnostic, sample, failure check, or
   final result.

Prepared PME plans, constraint schedules, force parameters, and topology
records persist across steps. Runtime synchronization is recorded by reason so
an apparent Neighbor wait can be separated from completion of earlier Metal
work.

## Neighbor Representations

| Backend | Intended use | Representation |
| --- | --- | --- |
| `mlx_dense_pairs` | small periodic systems | exact explicit pairs |
| `mlx_cell_pairs` | general large orthorhombic systems | device-built exact pairs |
| `mlx_cell_blocks` | fixed-shape compatibility and diagnostics | padded blocks |
| `mlx_cell_tiles` | measured Metal PME route | exact masked 4-by-4 force tiles |
| `mlx_interaction32` | opt-in PME performance candidate | retained device-built 32-atom schedule |

The general `auto` policy selects dense pairs below its atom limit and cell
pairs above it. Performance runners select `mlx_cell_tiles` explicitly only
when the force path and validation contract admit tiles.

The tile builder now has four distinct granularities:

- a spatial cell template prunes geometry;
- coarse 8-by-8 atom tiles evaluate exact cutoff-plus-skin membership;
- non-empty 4-by-4 tiles become the recurring execution representation;
- one 32-lane single-instruction, multiple-data (SIMD) group consumes up to 32
  active right-atom columns sharing one four-atom left block.

Cell occupancy, task counts, candidate offsets, exact membership, and force
schedule construction remain on device. A 32-lane Metal membership kernel
loads the 16 coarse-tile atoms once and constructs four exact masks without
threadgroup scratch.

Small exact-tile inventories compact and sort by left block. Above three
million coarse candidates, normally occupied cells use a left-grouped Metal
count/scatter path that avoids a global `mx.argsort`. Cells with more than
eight coarse blocks use the older parallel route to prevent one thread from
serializing pathological occupancy. See the
[adaptive scatter report](./benchmarks/md-left-grouped-neighbor-scatter-m5max.md).

Exact diagnostic pairs are deferred. They are materialized only when a public
pair consumer, pressure diagnostic, or unsupported force term requests them.

## Direct and Reciprocal Forces

The production Direct Space kernel fuses Lennard-Jones and screened Coulomb
work over spatial tiles. It uses:

- packed column descriptors carrying four membership bits;
- atom-local compressed sparse row (CSR) exclusion and 1-4 lookup;
- CHARMM pair-specific NBFIX overrides where present;
- four-left-atom register accumulation and SIMD reduction before global force
  writes.

The order-five reciprocal PME route retains one charge-spread Metal dispatch,
MLX fast Fourier transforms, influence multiplication, and one analytic
B-spline derivative interpolation dispatch. Its force-only route consumes the
complex inverse-transform grid directly and avoids redundant outer position
wrapping. Alternative spread launch geometries did not improve the complete
reciprocal graph and were rejected.

Sparse PME exclusions, exceptions, and 1-4 corrections share the fused bonded
Metal force buffer when that owner is available. The hottest Direct Space
kernel is deliberately not enlarged by this work.

## Bonded and Constraint Work

The recurring bonded Metal dispatch covers standard bonds, angles, periodic
torsions, impropers, Urey-Bradley 1-3 terms, and prepared CHARMM correction map
(CMAP) forces. The CMAP worker evaluates both signed dihedrals and differentiates
the prepared periodic bicubic patch analytically. Diagnostic energy retains the
established MLX implementation.

Constraint topology is partitioned once:

- rigid water uses analytical SETTLE;
- artifacts without molecule identifiers recover rigid water only when the
  water mask and constraint graph prove a complete set of disjoint O-H-H
  triangles; ambiguous graphs fail closed to the existing generic route;
- disjoint central-atom clusters use specialized SHAKE/RATTLE kernels;
- independent components up to four atoms and three edges use a component-owned
  Metal solver;
- larger or overlapping graphs retain the generic MLX fallback.

Combining already-dependent constraint dispatches produced isolated wins but
did not transfer reliably to sustained JAC trajectories. Those prototypes are
closed in the decision ledger.

The GPCRmd 729 artifact has no molecule identifiers but does have a complete
water mask and constraint graph. Topology recovery proves 19,944 disjoint
rigid waters, covering all 59,832 marked water atoms, then leaves 19,064
non-water constraint pairs in 9,709 SHAKE clusters. This replaces one
29,653-component, 20-iteration route with analytical SETTLE plus the existing
dense composite path. A same-context position-balanced 1,000-step diagnostic
improved both directions by 5.31% and 32.37%, with an 18.82% balanced median.
Because the machine changed Metal performance states between independent
processes, the retained claim is the conservative 5.1-5.3% complete-wall gain.
The passing profile and A/B artifacts are under
`results/md-suite/gpcrmd-water-topology-settle-{profile,shared-context}-2026-08-15/`.

## Current Evidence

The latest clean, position-balanced 750-step comparison on 2026-08-15 measured
the opt-in `mlx_interaction32` backend against the default production tiles:

| Workload | Atoms | Control | Current | Result |
| --- | ---: | ---: | ---: | ---: |
| 5DFR | 23,558 | 1.7496 ms/step | 1.5992 ms/step | 8.60% faster |
| JAC 4-cell | 94,232 | 6.8322 ms/step | 5.9439 ms/step | 13.00% faster |
| GPCRmd 729 | 92,001 | 7.8329 ms/step | 7.4761 ms/step | 4.56% faster |

The comparison used two processes per arm, ten warmup steps, Low Power Mode,
and a balanced control/candidate/candidate/control order. It is not a same-date
OpenMM ratio. Reference-engine comparisons must use a matched
manifest, platform, precision, protocol, power state, and measurement window.

The canonical local performance gate is documented in
[`md-suite.md`](./benchmarks/md-suite.md). Long-form physics and reference
evidence remains in the JAC, GPCRmd, and same-workload reports indexed from the
[benchmark directory](./benchmarks/README.md).

## Measurement Rules

- Measure complete trajectory wall before retaining a kernel optimization.
- Use independent processes and a position-balanced order such as
  `control, candidate, candidate, control, control, candidate`.
- Hold seed, timestep, cutoff, skin, diagnostics, sampling, Neighbor cadence,
  prepared artifact, and power state fixed.
- Require every arm to pass finite-state, constraints, topology, memory,
  Neighbor representation, and PME-plan reuse checks.
- Treat synchronized stage profiles as structural attribution, not clean
  throughput.
- Report rebuild counts and memory with wall time. Floating-point reduction
  order can alter later rebuild timing in chaotic trajectories.

## Current Bottleneck Direction

The 32-atom Direct Space route and its Neighbor builder beat the production
tile route on 5DFR, JAC, and GPCRmd. An immutable topology snapshot first
removed repeated host topology preparation. A subsequent fresh whole-step
profile identified Direct Space as 16.87%, 32.94%, and 36.11% of synchronized
wall on the three systems. The Neighbor lifecycle remained material at 11.79%,
14.18%, and 15.03%, but was not the leading stage.

Builder attribution then showed that ordinary count/prefix plus ordinary
scatter owned about 78% of 5DFR rebuild wall and 91% of JAC and GPCRmd rebuild
wall. Both stages recomputed the same periodic block and atom memberships. The
count kernel now retains each membership mode in two bits, and scatter decodes
that temporary cache instead of repeating the geometry. Median rebuild time
fell from 6.47 to 2.91 ms on 5DFR, 27.20 to 12.41 ms on JAC, and 26.73 to
12.61 ms on GPCRmd. Position-balanced 750-step complete walls improved in both
directions on every system, by 1.53-4.80%.

The cache is admitted only when it is at most 64 MiB. Larger systems retain the
original sparse two-pass builder, preserving the runtime's scalable memory
boundary. Capacity admission remains a microsecond-scale host operation and
does not justify a native C++ MLX primitive. A new whole-step profile, rather
than another builder micro-optimization, selected constraints as the next
tractable target. After recovered SETTLE, a passing GPCRmd profile ranks Direct
Space first at 38.71% of synchronized wall, followed by the Neighbor lifecycle
at 14.06%, with PME and constraints both near 12.78%. The next experiment must
therefore return to Direct Space or PME rather than extending the builder.

The experimental 32-atom engine is not the default production route. Its
device-built schedule now passes the initial 5DFR, JAC, and NBFIX-bearing
GPCRmd end-to-end gate. It remains opt-in while broader stability coverage and
whole-step profiling continue. See
[`metal-interaction-engine.md`](./metal-interaction-engine.md).
