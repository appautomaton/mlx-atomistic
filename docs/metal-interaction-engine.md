# Device-Resident 32-Atom Metal Interaction Engine

## Status

The 32-atom engine is now the default Molecular Dynamics (MD) backend for the
measured fixed-cell Metal PME envelope. It is named `mlx_interaction32` and
uses a retained device-built schedule. Production keeps `mlx_cell_tiles` and
`mlx_cell_pairs` as explicit, checkpoint-compatible fallback routes.

The backend owns retained device capacity, fail-closed overflow,
Neighbor generation identity, the recurring direct-force schedule, and lazy
diagnostic tiles. It has improved complete constrained Particle Mesh Ewald
(PME) trajectories across proteins, pure water, a membrane protein, a
protein-lipid-water system, AMBER, and CHARMM NBFIX. The promotion envelope is
deliberately bounded; unsupported devices, cells, PME settings, atom counts,
or topology surfaces retain the previous routes.

The sustained force gate was reopened on 2026-08-14. Earlier 16-call timing
blocks showed only a small advantage and correctly blocked production work at
that time. New 750-call, direction-balanced blocks expose a repeatable benefit
under the same sustained memory and power pressure as long trajectories. The
`fused_half32` candidate now also applies the production CHARMM NBFIX type
table, so the force surface covers 5DFR, JAC, and GPCRmd.

| Workload | Production direct force | `fused_half32` | Reduction | RMS / maximum force delta |
| --- | ---: | ---: | ---: | ---: |
| 5DFR | 0.8897 ms | 0.7527 ms | 15.40% | `6.84e-5` / `5.19e-4` kJ/mol/A |
| JAC | 3.7297 ms | 2.7425 ms | 26.47% | `5.18e-5` / `3.97e-4` kJ/mol/A |
| GPCRmd 729 with NBFIX | 4.3586 ms | 3.1922 ms | 26.76% | `5.84e-5` / `5.49e-4` kJ/mol/A |

Each row used eight samples, 750 individually synchronized calls per sample,
and both control-first and candidate-first directions. Every direction was
positive. These historical force-only results authorized the device builder.
The host SciPy `cKDTree` schedule remains an oracle only and never enters the
runtime.

The command shape for each prepared artifact was:

```bash
uv run python scripts/benchmark_interaction32.py <prepared> \
  --architecture fused_half32 \
  --ordinary-tiles-per-group 3 \
  --warmups 4 \
  --samples 8 \
  --timing-block-count 750 \
  --out results/md-suite/round1-sustained-interaction32/<case>.json
```

The raw payloads remain local and gitignored under `results/`. The benchmark
schema is `mlx_atomistic.interaction32_force_benchmark.v4`; it records whether
the production NBFIX type table was active.

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
   screened Coulomb work with SIMD rotation and register accumulation. Its
   canonical-record specialization applies the same optional NBFIX type table
   as the production tile kernel.
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

## Device Builder Contract

Gate C is split into two measured boundaries. C1 proves the builder algorithm
with MLX custom Metal kernels. C2 removes the remaining scalar sizing boundary
only if C1 shows that it is material.

### C1: device-built payload with sized allocation

1. Wrap positions, assign fine spatial keys, and produce a canonical-to-packed
   32-atom order with MLX `argsort`.
2. One SIMD group per packed block computes its center, radius, and axis-aligned
   half extent without materializing a block-by-atom position tensor.
3. Static canonical exclusion and 1-4 pairs map through the inverse order.
   Sorted sparse block-pair codes define diagonal and topology-bearing special
   work. Ordinary traversal uses binary search rather than a quadratic dense
   block-pair flag table.
4. The ordinary count kernel assigns one SIMD group to each left block. It
   rejects right blocks by sphere and axis-aligned bounds, skips special block
   pairs, and counts unique right atoms in three buckets: first half only,
   second half only, and both halves.
5. MLX prefix scans the three count vectors. One synchronized inventory read
   obtains ordinary rows, force groups, and the special-block count. No
   atom identifiers, positions, bounds, or membership payloads cross to the
   host.
6. The ordinary scatter kernel repeats the exact geometric test and writes
   compact 32-right-atom rows directly into final left-block and half-mode
   order. Existing grouped-work emission builds runs of at most three rows.
7. Each compact special block conservatively emits two 16-atom rows. The force
   kernel applies the exact distance test, while device-built topology masks
   retain production exclusion and 1-4 semantics, including diagonal ownership.

The two-pass ordinary search deliberately trades repeated distance-only tests
for deterministic compact output and avoids global atomic allocation. Its
complete rebuild wall, not its kernel time alone, decides whether it survives.

### C1 measured checkpoint

The first device-builder checkpoint passes synthetic periodic, topology, and
complete-force oracles and three real fixed-coordinate artifacts. All runs used
Low Power Mode, eight position-balanced samples, and 750 synchronized force
calls per timing block. Generated JSON remains under
`results/md-suite/round2-device-builder/` and is not a package input.

| System | Atoms | Device build | Production force | Candidate force | Force speedup | RMS force delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5DFR | 23,558 | 11.75 ms | 0.839 ms | 0.756 ms | 9.9% | `6.85e-5` |
| JAC | 94,232 | 37.18 ms | 3.577 ms | 2.646 ms | 26.0% | `5.19e-5` |
| GPCRmd | 92,001 | 50.63 ms | 4.189 ms | 3.121 ms | 25.5% | `5.83e-5` |

The device builder reproduces the host prototype's size-ordered ordinary
inventory. This ordering is performance-critical: a correct block-ID traversal
retained nearly the same logical work but changed force locality enough to lose
the candidate benefit. Special work currently over-allocates at most one empty
half per special block; that small conservative cost is measured before adding
another geometry compaction stage.

The force-only rebuild break-even is about 142 steps for 5DFR, 40 for JAC, and
47 for GPCRmd. These are not trajectory claims. Promotion still requires the
capacity/generation contract and measured rebuild frequency in moving NVT
trajectories.

### Capacity, overflow, and generation

The first admitted generation reserves at least 125% row and group capacity,
rounded to a 64-record allocation quantum. Later generations retain that
capacity until an inventory grows beyond it. Count kernels expose every logical
count before scatter. Scatter is forbidden when any count exceeds capacity. An
overflowed candidate generation never replaces the previous valid schedule; a
safe host boundary grows capacity and retries the already-counted inventory.

Generation identity includes atom count, box, search radius, topology digest,
capacity, and a monotonic generation value. Force bindings rebind only after a
complete new generation passes capacity admission. Tests prove that an
overflowed generation leaves the prior schedule active and that a force binding
rejects the wrong generation. Public positions and forces remain in canonical
order.

### Moving NVT checkpoint

On 2026-08-15 the retained-capacity builder was connected to the prepared force
pipeline through the opt-in `mlx_interaction32` Neighbor backend. Ordinary steps
use only the 32-atom schedule. Energy and virial diagnostics lazily build exact
`NeighborTiles` for the same reference generation; they never materialize the
60-million-pair GPCRmd shell. The backend participates in the same asynchronous
force-submission path as production tiles.

The clean comparison used two independent processes per arm, ten warmup steps,
750 measured steps, seed 17, skin 5.5 A, unchanged diagnostics, and Low Power
Mode. Order was balanced as control/candidate/candidate/control, reversed for
JAC. Raw JSON is under
`results/md-suite/round2-device-builder/moving-ab-v2/`.

| Workload | Production tiles | `mlx_interaction32` | Complete-wall speedup | Candidate rebuild wall / count | Metal peak memory change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5DFR | 1.7496 ms/step | 1.5992 ms/step | 8.60% | 0.3038 s / 21.5 | -16.2% |
| JAC 4-cell | 6.8322 ms/step | 5.9439 ms/step | 13.00% | 1.2362 s / 21 | -18.5% |
| GPCRmd 729 | 7.8329 ms/step | 7.4761 ms/step | 4.56% | 2.1509 s / 26.5 | -18.7% |

The candidate builder is still slower than the production tile builder: about
14.1 versus 11.2 ms per rebuild on 5DFR, 58.9 versus 46.6 ms on JAC, and 81.2
versus 44.6 ms on GPCRmd. The complete trajectory wins because the recurring
direct force is cheaper. This makes builder stage attribution the next target;
it does not yet justify a C++ MLX primitive because the remaining scalar
inventory boundary has not been isolated as the cause.

### Topology snapshot checkpoint

The follow-up attribution added an opt-in, sequentially synchronized
Interaction32 rebuild profile. It separates geometry, topology preparation,
geometry-dependent special-block inventory, ordinary count/prefix readback,
capacity admission, ordinary scatter, special scatter, and completion. The
profiler is inactive on the clean path and adds no device synchronization when
disabled.

Before the optimization, fixed topology processing dominated every system:

| Workload | Profiled builder | Fixed topology share | Fixed topology median |
| --- | ---: | ---: | ---: |
| 5DFR | 14.84 ms/rebuild | 61.3% | 8.64 ms |
| JAC 4-cell | 59.50 ms/rebuild | 59.9% | 34.46 ms |
| GPCRmd 729 | 82.59 ms/rebuild | 72.3% | 57.60 ms |

Exclusion and one-four pairs, topology owner offsets, neighbor classes, and the
topology digest depend on topology rather than positions. They are now prepared
once as a manager-owned immutable snapshot and reused across spatial
generations. Reassigning either topology source invalidates the snapshot, and a
cell candidate inherits it. Special block codes still rebuild from the current
atom ordering, so dynamic geometry is not frozen.

With the snapshot active, steady-state topology lookup is 0.005-0.006 ms and
the remaining geometry-dependent special inventory is 0.39-0.64 ms. The same
750-step synchronized profiles measured these builder reductions:

| Workload | Before | After | Builder reduction |
| --- | ---: | ---: | ---: |
| 5DFR | 14.84 ms/rebuild | 6.52 ms/rebuild | 56.1% |
| JAC 4-cell | 59.50 ms/rebuild | 29.02 ms/rebuild | 51.2% |
| GPCRmd 729 | 82.59 ms/rebuild | 27.22 ms/rebuild | 67.0% |

A separate clean production/candidate/candidate/production comparison used two
independent processes per arm, ten warmup steps, 750 measured steps, seed 17,
skin 5.5 A, and unchanged Low Power Mode. Every arm passed finite-state,
constraint, topology, memory, Neighbor-representation, and PME-plan reuse
checks.

| Workload | Production tiles | Interaction32 | Complete-wall speedup | Production builder | Interaction32 builder |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5DFR | 1.7922 ms/step | 1.4448 ms/step | 19.39% | 11.72 ms/rebuild | 5.78 ms/rebuild |
| JAC 4-cell | 6.9320 ms/step | 5.3844 ms/step | 22.33% | 48.92 ms/rebuild | 27.34 ms/rebuild |
| GPCRmd 729 | 7.8009 ms/step | 5.9995 ms/step | 23.09% | 46.10 ms/rebuild | 26.37 ms/rebuild |

Raw JSON is under
`results/md-suite/interaction32-builder-topology-cache-2026-08-15/`. The old
builder control is commit `25adc7d`. These results close the builder regression
that remained at the moving-NVT checkpoint. The next decision must use a fresh
whole-step profile. If the builder is still selected, ordinary count/prefix and
ordinary scatter are its next shared targets. Capacity admission is already a
microsecond-scale host boundary, so this checkpoint does not authorize a C++
MLX primitive.

### C1 packed membership checkpoint

The post-snapshot whole-step profile selected Direct Space as the largest
shared route and retained the Neighbor lifecycle as a material secondary cost.
Inside rebuilds, ordinary count/prefix and ordinary scatter represented about
78% of 5DFR wall and 91% of JAC and GPCRmd wall. Inspection confirmed that
scatter repeated the count kernel's periodic block rejection and 32-by-32 atom
membership calculation.

The retained builder now packs each right atom's membership mode into two bits
during count. Sixteen modes fit in one `uint32`, so each upper-triangular block
pair uses two words. Scatter decodes those words and performs only prefix-local
schedule writes. The packed path is admitted only when its temporary storage is
at most 64 MiB; larger systems use the unchanged sparse two-pass path. The
measured caches were 2.17 MB on 5DFR, 34.68 MB on JAC, and 33.07 MB on GPCRmd.

Synchronized rebuild medians changed as follows:

| Workload | Snapshot builder | Packed-mode builder | Reduction |
| --- | ---: | ---: | ---: |
| 5DFR | 6.47 ms | 2.91 ms | 55.1% |
| JAC 4-cell | 27.20 ms | 12.41 ms | 54.4% |
| GPCRmd 729 | 26.73 ms | 12.61 ms | 52.8% |

A separate `control, candidate, candidate, control` run used independent
processes, ten warmup steps, 750 measured steps, seed 17, skin 5.5 A, and an
unchanged power state. Every arm passed finite-state, constraint, topology,
memory, Neighbor-representation, and PME-plan reuse checks.

| Workload | Control range | Candidate range | Directional speedups |
| --- | ---: | ---: | ---: |
| 5DFR | 0.9547-0.9777 ms/step | 0.9282-0.9307 ms/step | 2.78%, 4.80% |
| JAC 4-cell | 2.9726-2.9862 ms/step | 2.9254-2.9270 ms/step | 1.53%, 2.04% |
| GPCRmd 729 | 3.2004-3.2174 ms/step | 3.1182-3.1199 ms/step | 2.57%, 3.03% |

Raw JSON is under
`results/md-suite/interaction32-packed-mode-cache-{screen,formal}-2026-08-15/`.
The control is `e2040e2`; the retained implementation is `0066b58`. MLX and
process peak memory remained within run-to-run noise. This closes the repeated
ordinary traversal target without changing the C2 decision: capacity admission
is still too small to justify a native primitive.

### C2: optional native state boundary

`mx.fast.metal_kernel` requires host-provided output shapes and direct dispatch
sizes. C1 therefore retains one scalar inventory synchronization per rebuild.
That is acceptable for the algorithm gate but is not called fully
device-resident. A C++ MLX `Primitive` becomes authorized only when C1 beats the
production builder plus force path and profiling shows this scalar boundary is
material. The native version must use the active MLX stream, functional state
tokens, validated buffer donation, indirect dispatch, and the same fail-closed
overflow contract. It must not create a private command queue or mutable global
device state.

Fresh clean host-clock attribution does not cross that gate. On three
independent 1,500-step Low Power Mode runs, main-thread CPU was 0.408 ms/step on
5DFR and 0.464 ms/step on JAC, or 31.9% and 10.3% of wall respectively. Those
values include Python, MLX graph construction, dispatch, and synchronous host
work and overlap GPU execution. The production-scale JAC ceiling is therefore
too small and too broad to justify a nanobind/C++ boundary without a subsequent
trace proving host queue starvation or a specific non-overlapped route.

## Promotion Gates

The candidate advances only in this order:

### Gate A: Inventory

Status: passed for the bounded production backend.

Record schedule size, occupancy, sparse-pair fraction, bytes, and rebuild time
on small Lennard-Jones, 5DFR, JAC, and GPCRmd systems. A host-built schedule is
diagnostic only and cannot pass this gate.

### Gate B: Force Kernel

Status: passed and integrated into the production runtime backend.

Compare the production 4-by-4 route, a 32-atom atomic route, and any
owner-computes variant on the same immutable schedule. Measure complete force
wall, global writes, force error, and memory. Reject a kernel that wins only on
synthetic dense occupancy.

### Gate C: Device Builder

Status: passed for the production runtime backend.

Build and reuse the schedule entirely on device. Include half-skin admission,
periodic boundary cases, concentrated occupancy, logical-capacity overflow,
and generation rebinding. The amortized rebuild plus force cost must beat the
production engine.

### Gate D: Full Trajectory

Status: passed; bounded default promotion completed.

Run position-balanced, independent-process comparisons on 5DFR, JAC, and
GPCRmd. Require stable complete-wall improvement, force parity, finite state,
constraint checks, Neighbor completeness, memory bounds, and acceptable
constant-particle-number, volume, and energy drift.

The promotion follow-up ran all six release workloads for 750 measured steps.
Every run passed the finite-state, constraint, topology, Neighbor,
Particle Mesh Ewald plan-reuse, and memory checks, with 20--40 measured
Neighbor rebuilds per system and no fallback. Direct-force parity across those
six systems had root-mean-square deltas of `5.18e-5`--`6.85e-5` and maximum
deltas of `4.27e-4`--`4.88e-4` kJ/mol/A.

Independent low-power control/candidate/candidate/control runs added the
previously uncovered release systems:

| Workload | Direction 1 | Direction 2 |
| --- | ---: | ---: |
| 30k TIP3P water | 33.2% faster | 28.2% faster |
| 90k TIP3P water | 30.9% faster | 33.8% faster |
| ApoA1 | 27.6% faster | 17.4% faster |

The 47,116-atom JAC midpoint was directionally non-regressing but unstable
(0.5% and 23.1%), so it remains on `mlx_cell_tiles`. This is why the production
selector admits the measured 23,000--31,000 and 90,000--100,000 atom windows
rather than claiming one continuous range. Old checkpoints pin their recorded
backend; new eligible runs select `mlx_interaction32`.

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
