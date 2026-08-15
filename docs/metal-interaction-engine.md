# Device-Resident 32-Atom Metal Interaction Engine

## Status

The 32-atom engine is an experimental architecture surface, not the production
Molecular Dynamics (MD) route. Production currently uses coarse 8-by-8
Neighbor search tiles, exact 4-by-4 force tiles, and 32 active-column Metal
groups. That path has device-built schedules, broad force-field coverage, and
the accepted end-to-end benchmarks.

The experimental 32-atom kernels remain useful only if they can demonstrate a
larger structural improvement than the production 4-by-4 engine. A faster
isolated kernel is insufficient. The required result is a device-built,
device-resident schedule that improves a complete constrained Particle Mesh
Ewald (PME) trajectory.

No C++ extension or additional package is justified at the current boundary.
MLX custom Metal kernels can express the required prototypes. A native
extension becomes relevant only if a measured allocation, synchronization, or
dispatch boundary cannot be removed through MLX and Metal.

## Why 32 Atoms

OpenMM organizes its CUDA nonbonded work around 32-lane warps and 32-atom
blocks. A recurring interaction tile is usually one fixed left block plus 32
right atoms compacted from multiple spatial candidates, not simply two dense
32-atom blocks. Each lane owns an atom, positions and parameters rotate through
warp shuffle operations, and partial forces accumulate in registers before
global writes.

Metal exposes 32-lane SIMD groups on the target Apple hardware, so the same
ownership principle is relevant. CUDA launch heuristics, fixed-point force
buffers, multi-GPU partitioning, and host/device copy machinery are not direct
templates for an MLX runtime on unified memory.

The transferable ideas are:

- one SIMD group owns a coherent interaction record;
- exact membership is compacted before recurring force evaluation;
- pair work accumulates in registers before global atomics;
- sparse topology interactions take an explicit side path;
- Neighbor admission, capacity, and overflow remain explicit correctness
  contracts.

## Current Production Baseline

The retained spatial engine already implements the most important pieces at a
smaller left-block granularity:

| Stage | Production layout |
| --- | --- |
| Spatial pruning | cached periodic cell-pair template |
| Membership | one 32-lane group per coarse 8-by-8 tile |
| Exact execution | non-empty 4-by-4 membership masks |
| Force schedule | up to 32 active right columns per four-atom left block |
| Topology | atom-local exclusion, 1-4, and NBFIX records |
| Force work | fused Lennard-Jones and Direct Space PME |
| Schedule ownership | device inventory with sized host boundaries |

The experimental engine must beat this complete baseline, including rebuild
amortization and memory, rather than compare only against an old explicit-pair
kernel.

## Proposed Data Model

A serious 32-atom candidate needs persistent device arrays for:

- a canonical-to-packed atom permutation and its inverse;
- packed position, charge, Lennard-Jones, and type records;
- one left-block identifier per interaction record;
- 32 compacted right-atom identifiers;
- exact 32-by-32 membership words;
- sparse explicit pairs for exclusions, exceptions, 1-4 terms, and unusual
  topology;
- block centers, bounds, old positions, and rebuild predicates;
- capacity, logical count, overflow, and generation state.

Public state remains in canonical atom order. Packing is an internal device
view and must not require a host round trip for positions or velocities.

## Kernel Family

The minimum useful family is:

1. `pack_atom_records_32` builds or refreshes the internal packed view.
2. `check_rebuild_and_bounds_32` combines the half-skin displacement predicate
   with per-block spatial bounds.
3. `find_interaction_blocks_32` prunes candidate blocks, evaluates exact
   membership, compacts 32 right atoms, and reports overflow without host-built
   schedules.
4. `compute_ordinary_interactions_32` evaluates recurring Lennard-Jones and
   screened Coulomb work with SIMD rotation and register accumulation.
5. `compute_special_interactions_32` handles sparse topology-owned work.
6. `scatter_ordered_force_32` is optional for diagnostics when the force buffer
   uses packed atom order.

The first prototype may use global atomics, but they must occur after useful
register accumulation. A no-atomic owner-computes variant is not automatically
better: the previous prototype evaluated each pair twice, ran 2.2-2.5 times
slower, and accumulated unacceptable float32 error along long owner lists.

## Capacity and Correctness

Dynamic output sizing must not become a hidden host synchronization. The
builder should use retained capacity, write logical counts and overflow state
on device, and make overflow fail closed. A production force evaluation may
retry after growing capacity, but it must never consume a truncated schedule.

Every admitted generation must preserve:

- exact cutoff-plus-skin membership;
- unique pair ownership;
- periodic minimum-image behavior;
- exclusion, 1-4, exception, and NBFIX semantics;
- force and energy parity inside declared float32 tolerances;
- bounded memory under repeated rebuilds;
- canonical public atom indexing.

## Promotion Gates

The candidate advances only in this order:

### Gate A: Inventory

Record schedule size, occupancy, sparse-pair fraction, bytes, and rebuild time
on small Lennard-Jones, 5DFR, JAC, and GPCRmd systems. A host-built schedule is
diagnostic only and cannot pass this gate.

### Gate B: Force Kernel

Compare the production 4-by-4 route, a 32-atom atomic route, and any
owner-computes variant on the same immutable schedule. Measure complete force
wall, global writes, force error, and memory. Reject a kernel that wins only on
synthetic dense occupancy.

### Gate C: Device Builder

Build and reuse the schedule entirely on device. Include half-skin admission,
periodic boundary cases, concentrated occupancy, logical-capacity overflow,
and generation rebinding. The amortized rebuild plus force cost must beat the
production engine.

### Gate D: Full Trajectory

Run position-balanced, independent-process comparisons on 5DFR, JAC, and
GPCRmd. Require stable complete-wall improvement, force parity, finite state,
constraint checks, Neighbor completeness, memory bounds, and acceptable
constant-particle-number, volume, and energy drift.

Only Gate D permits promotion to the runtime path.

## Explicit Non-Goals

- OpenMM or LAMMPS code does not enter the runtime.
- The first candidate does not add multi-GPU partitioning.
- It does not replace MLX fast Fourier transforms or the PME plan.
- It does not change public atom ordering or trajectory formats.
- It does not add a native build dependency before a measured MLX boundary
  requires one.

## References

- Production architecture: [`md-acceleration.md`](./md-acceleration.md)
- Current Neighbor result:
  [`md-left-grouped-neighbor-scatter-m5max.md`](./benchmarks/md-left-grouped-neighbor-scatter-m5max.md)
- Historical decisions:
  [`md-performance-decisions-m5max.md`](./benchmarks/md-performance-decisions-m5max.md)
- Local reference sources: `vendors/openmm/` and `vendors/lammps/`
