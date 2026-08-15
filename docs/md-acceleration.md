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
- disjoint central-atom clusters use specialized SHAKE/RATTLE kernels;
- independent components up to four atoms and three edges use a component-owned
  Metal solver;
- larger or overlapping graphs retain the generic MLX fallback.

Combining already-dependent constraint dispatches produced isolated wins but
did not transfer reliably to sustained JAC trajectories. Those prototypes are
closed in the decision ledger.

## Current Evidence

The latest clean, position-balanced 750-step comparison at commit `8899994`
measured:

| Workload | Atoms | Control | Current | Result |
| --- | ---: | ---: | ---: | ---: |
| 5DFR | 23,558 | 1.0881 ms/step | 1.0910 ms/step | 0.26% noise; no material regression |
| JAC 4-cell | 94,232 | 3.3983 ms/step | 3.2597 ms/step | 4.08% faster |
| GPCRmd 729 | 92,001 | 3.9123 ms/step | 3.8200 ms/step | 2.36% faster |

This comparison isolates the adaptive left-grouped scatter. It is not a
same-date OpenMM ratio. Reference-engine comparisons must use a matched
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

Large-system profiles consistently identify Direct Space and the Neighbor
generation lifecycle as the shared costs. GPCRmd additionally exposes CHARMM
bonded work. The next performance candidates should therefore change one of
these structural boundaries:

1. reduce host admission or allocation boundaries without hiding correctness
   checks;
2. improve device locality shared by Direct Space and PME;
3. reduce global force accumulation while preserving a cross-system win.

The experimental 32-atom engine is not the production route. Sustained
force-only blocks now pass on 5DFR, JAC, and NBFIX-bearing GPCRmd, reopening its
device-builder gate. Its next required result is a device-built schedule with
an end-to-end trajectory win, not another host-built kernel microbenchmark. See
[`metal-interaction-engine.md`](./metal-interaction-engine.md).
