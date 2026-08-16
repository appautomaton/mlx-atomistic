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
| `mlx_cell_tiles` | explicit/checkpoint-compatible Metal fallback | exact masked 4-by-4 force tiles |
| `mlx_interaction32` | measured fixed-cell Metal PME default | retained device-built 32-atom schedule |

The general `auto` policy selects dense pairs below its atom limit and cell
pairs above it. The prepared production selector chooses `mlx_interaction32`
only inside the validated fixed-cell PME envelope. Unsupported devices, cells,
PME settings, topology surfaces, and atom counts retain tiles or compact pairs.

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
B-spline derivative interpolation dispatch. Its recurring force-only route now
uses a real forward transform, stores only the last-axis half-spectrum, applies
the matching half-spectrum influence view, and produces a normalized real
inverse-transform grid. The Metal interpolation kernel applies the transform
normalization once per atom instead of scaling the complete grid. Energy and
diagnostic paths retain the conservative full complex transform. Alternative
spread launch geometries did not improve the complete reciprocal graph and
were rejected.

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

The initial clean, position-balanced 750-step comparison on 2026-08-15 measured
`mlx_interaction32` against the former production tile route:

| Workload | Atoms | Control | Current | Result |
| --- | ---: | ---: | ---: | ---: |
| 5DFR | 23,558 | 1.7496 ms/step | 1.5992 ms/step | 8.60% faster |
| JAC 4-cell | 94,232 | 6.8322 ms/step | 5.9439 ms/step | 13.00% faster |
| GPCRmd 729 | 92,001 | 7.8329 ms/step | 7.4761 ms/step | 4.56% faster |

The comparison used two processes per arm, ten warmup steps, Low Power Mode,
and a balanced control/candidate/candidate/control order. It is not a same-date
OpenMM ratio. Reference-engine comparisons must use a matched
manifest, platform, precision, protocol, power state, and measurement window.

A fresh current-main snapshot on commit `1bd476b` measured sustained absolute
throughput under Low Power Mode. Each row used a 75-step rehearsal, three
independent repeats, ten warmup steps, and 750 measured steps:

| Workload | Atoms | Wall | Throughput | Timing spread |
| --- | ---: | ---: | ---: | ---: |
| 5DFR | 23,558 | 1.2533 ms/step | 275.75 ns/day | 1.60% |
| JAC 4-cell | 94,232 | 4.2450 ms/step | 81.41 ns/day | 1.23% |
| TIP3P 30k | 30,000 | 1.5720 ms/step | 219.85 ns/day | 1.11% |
| TIP3P 90k | 90,000 | 4.3517 ms/step | 79.42 ns/day | 6.81% |
| ApoA1 | 92,224 | 4.8183 ms/step | 71.73 ns/day | 5.18% |
| GPCRmd 729 | 92,001 | 5.3984 ms/step | 64.02 ns/day | 0.97% |

Five rows passed in the release-suite run. JAC narrowly missed the spread gate
at 10.42% because its first sample was slow; an immediate three-repeat
supplement produced the passing row above. These numbers describe one
throttled measurement window, not an expected peak-throughput claim.

A separate manifest-matched JAC comparison in the same Low Power Mode window
used 94,232 atoms, a 4 fs timestep, a 9 A cutoff, a `128x128x64` PME mesh, the
same constraints and seed, and 750 measured steps. MLX measured 4.6834 ms/step
and OpenMM OpenCL measured 2.4161 ms/step across two independent processes per
engine. The remaining matched wall ratio was `1.9385x`, so MLX delivered 51.6%
of OpenMM throughput on that protocol. This ratio is not transferred to any
other row or power state.

The promotion follow-up added force parity and a 750-step trajectory smoke gate
for all six release systems. Every trajectory passed with no fallback and
20--40 measured Neighbor rebuilds. Isolated, position-balanced low-power runs
for the newly covered systems measured 28.2--33.2% improvements on 30k TIP3P
water, 30.9--33.8% on 90k TIP3P water, and 17.4--27.6% on ApoA1. The unstable
47,116-atom JAC midpoint remains on `mlx_cell_tiles`, so default promotion is
bounded to the measured 23,000--31,000 and 90,000--100,000 atom windows.

The canonical local performance gate is documented in
[`md-suite.md`](./benchmarks/md-suite.md). Long-form physics and reference
evidence remains in the JAC, GPCRmd, and same-workload reports indexed from the
[benchmark directory](./benchmarks/README.md).

The subsequent real-transform PME checkpoint reduced a `128x128x64` JAC force
spectrum to `128x128x33`. Same-process old/new reciprocal-graph ABBA measured
1.4438 to 0.6359 ms on JAC and 0.6349 to 0.5115 ms on GPCRmd, improvements of
55.96% and 19.44%. Maximum force deltas against the former complex path were
`2.50e-5` and `3.72e-5 kJ/(mol A)`, respectively. Even- and odd-sized mesh GPU
parity tests also pass against the unchanged energy-plus-force path.

Two independent 750-step canonical comparisons passed the complete-wall gate
in both directions. Directional gains were 2.91% and 23.29% on 5DFR, and 28.09%
and 19.86% on JAC. Absolute walls changed substantially with the machine's
Metal performance state, so these figures are retained as directional evidence
and are not averaged into a single expected speedup. GPCRmd whole-step arms
were even less stable, ranging from 11.92 to 27.63 ms/step and changing
direction across balanced pairs. Its reciprocal-graph improvement is retained,
but no GPCRmd whole-step claim is made from that run. Raw JSON is under
`results/md-suite/pme-real-fft-*-2026-08-15/`.

The next Direct audit closed the remaining easy branches. Full,
Lennard-Jones-only, and Coulomb-only kernels attributed 66--84% of elapsed time
to common geometry, traversal, reduction, and force ownership. Morton ordering
increased scheduled lanes by about 5%, while all six linear axis permutations
stayed within 0.21% and did not select one cross-system winner. A useful-pair
compaction prepass and cross-tile reduction batching both reduced their target
work but lost more time to an extra pass or register pressure. The next bounded
target is therefore the remaining reciprocal FFT cost, not another small Direct
formula branch.

A fresh reciprocal profile then measured the complete force-only graph at
0.615 ms on JAC and 0.510 ms on GPCRmd. Synchronized charge spread, forward
FFT, inverse FFT, and interpolation probes all sat near the same 0.26--0.31 ms
launch-and-completion floor, so none remained a dominant substage. Sharing
B-spline weights across spread workers produced no directional win. Splitting
each atom's 125 interpolation reads across five workers regressed JAC by
2.4--5.6% and GPCRmd by 5.2--13.0%. The retained one-thread interpolation and
five-thread spread therefore remain the measured layouts.

The apparent diagnostics share in a short whole-step profile is not a new
per-step hotspot. `diagnostics_reporting` runs only at the initial and final
boundaries and intentionally evaluates the full energy/force report. Its
per-step contribution falls by roughly an order of magnitude in the 750-step
canonical gate.

A subsequent six-system release profile changed the next-priority decision.
Direct Space was the largest synchronized stage on every release workload and
owned a 34.19% cross-system median share. Reciprocal PME followed at 14.04%,
the Neighbor lifecycle at 11.31%, constraints at 10.95%, and integration at
9.93%. Integration/constraint fusion therefore remains a secondary candidate.

The next Direct experiment targeted complete SIMD work items rather than
repeating rejected lane branches or formula specialization. Release schedules
contained 41.1--44.4% more padded lanes than exact cutoff-plus-skin membership,
but that builder padding was not itself removable force work. At a 9 A force
cutoff, rebuild positions left 26.3--27.7% of ordinary groups fully inactive,
or 14.8--15.3% of their scheduled lanes. Weighting the same test by the real
750-step Neighbor lifecycle reduced the mean fully inactive-group lane share to
1.84% on JAC and 1.98% on GPCRmd; the medians were only 0.05--0.06%. A static
group-skip kernel would therefore have less than about 0.7% whole-step upside
before paying for metadata and branching. The candidate is closed, and the
next bounded attribution returns to the integration/constraint interface.

That attribution split the three formerly aggregated integration barriers.
On JAC, drift/thermostat and the final force kick measured 0.253 and 0.219
ms/step, while the three constraint projections totaled 0.746 ms/step. On
GPCRmd they measured 0.348, 0.267, and 1.017 ms/step respectively. The post-step
state assembly was only 0.008--0.014 ms/step on those workloads. Replacing the
pre-force dense write with a sparse SHAKE scatter changed direction in an ABBA
screen and was rejected.

5DFR exposed a separate protocol-specific cost because it runs
`CMMotionRemover` every step. Its post-step route measured 0.224 ms/step.
Reusing the invariant total mass removed one repeated reduction. Two
independent-process 3,000-step pairs improved complete wall by 4.20% and 2.26%,
with a 3.24% median reduction from 1.3702 to 1.3258 ms/step. This gain applies
only when center-of-mass motion removal is enabled; it is not attributed to JAC
or GPCRmd.

Three execution-layout follow-ups did not pass. Generation-owned arrays for
stable Interaction32 parameters and dispatch counts regressed a fixed-input JAC
median by 3.17%. Transient spatial packing made the interaction itself about
0.086 ms faster, but pack and scatter added about 0.50 ms; a same-schedule ABBA
regressed in both positions. Finally, assigning Direct Space and reciprocal PME
to separate MLX Metal streams increased fixed-input JAC calls by 27--65%.
OpenMM's persistent records and separate PME stream remain architectural
references, but neither maps to a per-step MLX copy or manual stream split.

Constraint dispatch fusion also reached its boundary. Fixed-input attribution
put the theoretical SETTLE/SHAKE launch-saving ceiling near 0.14--0.15 ms/step
on JAC and GPCRmd. A combined Metal kernel produced bitwise-identical deltas,
but complete 750-step ABBA runs did not retain that ceiling. GPCRmd's candidate
median regressed 0.31%, while JAC's warm adjacent pair regressed 0.17%; the
larger JAC median movement was a first-control power-state artifact. Separate
family kernels therefore remain the production path under MLX's lazy graph.

The same rule applies to force-array aggregation. Folding the Direct force
array into the order-five PME interpolation write preserved force parity and
removed one explicit full-array addition, but it made the interpolation depend
on Direct Space earlier in the graph. Complete ABBA runs regressed 4.7% on
5DFR, produced only a 0.8% warm adjacent gain on JAC, and were neutral on
GPCRmd. The explicit lazy addition remains because it gives MLX more scheduling
freedom than the apparently more fused kernel.

Approximate arithmetic is also screened at the complete-kernel boundary before
it reaches trajectories. `metal::fast::exp` in the shared Ewald helper changed
sampled forces by only about 3e-7 relative to the largest force, but fixed-input
timing regressed in both adjacent JAC comparisons and was unstable on GPCRmd.
The precise `metal::exp` remains canonical because the approximation supplied no
portable performance benefit.

### Host and native-extension boundary

The charged-PME runtime now records main-thread and whole-process CPU clocks
around the clean measured interval. CPU clocks exclude time blocked on Metal
completion, but overlap GPU execution. They are upper bounds on host activity,
not additive pieces of trajectory wall and not direct speedup estimates.

After a reported device-state change, a fresh Low Power Mode boundary used
three independent 1,500-step repeats and twenty warmup steps:

| Workload | Wall | Main-thread CPU | Main-thread/wall | Process CPU | Process/wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5DFR | 1.2883 ms/step | 0.4082 ms/step | 31.9% | 0.6376 ms/step | 49.9% |
| JAC 4-cell | 4.4883 ms/step | 0.4635 ms/step | 10.3% | 0.7324 ms/step | 16.3% |

The nearly fixed main-thread cost matters proportionally on the smaller 5DFR
system, but it is not the primary production-scale JAC bottleneck. The JAC
main-thread value includes Python, MLX graph construction, kernel submission,
and synchronous MLX host work; eliminating all of it would still cap the
theoretical gain near 10%, while a real C++ primitive could remove only part of
that value. A Python `cProfile` run also assigned Metal completion waits inside
`_needs_rebuild_mlx_scalar` to its Python caller, demonstrating why ordinary
cumulative profiles cannot separate Python from device time. These measurements
do not authorize a nanobind/C++ extension. Revisit that boundary only when a
clean trace identifies host queue starvation or a specific Python/MLX graph
route with a material, non-overlapped wall ceiling.

One feature-specific Direct Space screen followed. GPCRmd has 83 atom types but
only six participate in its five NBFIX overrides. Compressing the two dense
NBFIX tables from `83x83` to `7x7` preserved the lookup contract, but a
`control, candidate, candidate, control` 750-step run was neutral: the candidate
was 0.16% slower in the first adjacent pair and 0.17% faster in the second. All
arms passed, and the candidate was removed. The table footprint was not a
measurable bottleneck on this workload; compaction is not the next Direct lever.

An Xcode GPU Replay then resolved the current Interaction32 compiler profile.
Two JAC force dispatches took 1.95 ms each, and the 382-instruction profile was
dominated by arithmetic logic unit work rather than memory, control flow, or
synchronization. Xcode did not expose occupancy or a usable live-register peak
for the just-in-time MLX custom kernel. Removing one derived-index threadgroup
buffer nevertheless tested the memory hypothesis directly. It preserved force
parity and improved fixed-schedule blocks on all three systems, but an isolated
1,500-step JAC trajectory improved only 0.46% against a contemporaneous control.
The source change was removed because it missed the 3% whole-step gate. The
next Direct candidate must reduce useful-pair arithmetic or complete SIMD work,
not merely move index traffic out of threadgroup memory.

Three follow-up arithmetic screens tightened that boundary. Factoring the
Lennard-Jones expression in OpenMM's form regressed the fixed-schedule JAC
kernel by 0.97%. Replacing the shared screened-Coulomb expression with a 4,097
entry, 16 KiB linear-interpolation table regressed it by 0.83%, despite a
`9.8e-8` offline maximum force-factor error. Hoisting the invariant
`coulomb_constant * right_charge` product regressed by 0.52%, indicating that
the compiler already performs the useful loop-invariant motion. Every prototype
preserved the existing force-parity scale and was removed before trajectory
testing.

The earlier component attribution explains the result: geometry, traversal,
cutoff checks, SIMD reductions, and writes account for 66-84% of Direct time,
whereas Lennard-Jones accounts for 13-23% and Coulomb for 3-15%. Atomic output
attribution caps global force writes at 0.9-3.2%. A larger design therefore had
to remove complete scheduled work. The retained two-level Verlet lifecycle
keeps the current 5.5 A outer schedule for correctness and compacts a 3.0 A
inner force schedule once at the same generation reference. The force path uses
the inner schedule only while the maximum displacement is at most 1.5 A, then
falls back to the same-generation outer schedule until the ordinary 2.75 A
rebuild boundary. It never derives a fresh inner list from stale outer
candidates.

The compactor is two Metal passes: one classifies outer right entries at the
inner radius and caches their half-interaction mode, and one scatters retained
entries into the generation-owned capacity. Both schedules share atom order,
special topology, and generation identity. This removes about 40% of ordinary
pair lanes from the early part of each large-system generation without adding
a C++ build dependency.

Generation lifetime is the runtime admission boundary. After at least three
complete two-level generations, the engine evaluates their cumulative mean. If
they average fewer than 24 updates, later rebuilds return to the original outer
builder. This changed the
90k TIP3P water result from an 8.2% regression to a 0.33% difference from
control, while retaining sustained improvements on slower-generation systems.
In 750-step runs, adaptive JAC measured 4.091 ms/step against adjacent controls
at 4.362 and 4.403 ms/step; GPCRmd measured 5.267 ms/step against controls at
5.412 ms/step. ApoA1 improved by 2.5% before the adaptive follow-up. The 5DFR
case is below the 80,000-atom admission boundary and remains on its unchanged
single schedule.

A fresh post-promotion whole-step profile at `e334398` keeps Direct Space as
the next target. Its synchronized wall fractions were 19.06%, 29.17%, and
32.34% on 5DFR, JAC, and GPCRmd. The corresponding reciprocal PME fractions
were 10.26%, 10.79%, and 9.08%; Neighbor lifecycle fractions were 9.13%,
10.01%, and 14.85%. Direct Space therefore has a 29.17% cross-system median,
about 2.8 times the 10.26% reciprocal-PME median.

The same run now reports the adaptive boundary explicitly. A 750-step probe
kept JAC admitted at 34.76 updates per observed generation. GPCRmd and 90k
TIP3P water returned to the single outer schedule after means of 21.67 and
20.00 updates. Their close lifetimes rule out lowering the 24-update threshold
to retain GPCRmd without risking the measured water regression. The next
Direct experiment must therefore improve work shared by the admitted inner and
production outer schedules; another admission-threshold adjustment is not a
valid optimization target.

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

A follow-up Direct Space component profile separated ordinary and special work
on the same retained schedule. After subtracting a separate zero-output
baseline, ordinary work remained about three times larger than special work on
both 5DFR and NBFIX-bearing GPCRmd. A zero-epsilon LJ arithmetic branch then
improved isolated 5DFR and 30k TIP3P blocks, but regressed ApoA1 and GPCRmd and
changed sign across the two JAC 94k directions. The prototype was removed.
Together with the earlier rejected owner-computes, lane-rotation, grouping, and
special-write variants, this closes another Direct Space screen without
claiming a runtime gain. That evidence selected Reciprocal PME as the next
bounded profiling target.

That bounded PME target is now complete. The retained force-only route uses a
real half-spectrum and a normalized-real Metal consumer. The subsequent
three-system whole-step profile ranks Direct Space first at a 22.41% median
instrumented share and reciprocal PME second at 17.74%.

Follow-up attribution prevents several false starts. A three-force-array sum is
only about 0.13-0.16 ms in isolation, so its larger synchronized profile share
mostly belongs to a preceding barrier. An arithmetic-preserving checksum shows
that Direct force atomics account for only 0.87-3.18% of the kernel. Spatially
ordered static parameters, PME mesh sorting after amortization, and right-lane
AABB culling all fail the cross-system transfer gate. The AABB test is the most
instructive: it can reject 52-54% of logical pair lanes, yet still regresses all
three systems because the SIMD loop and reductions remain. Future Direct work
must remove complete SIMD work items or improve useful-pair arithmetic rather
than add divergent lane-level branches. Builder capacity admission remains too
small to justify native C++ work.

The 32-atom engine is now the default inside the bounded, measured fixed-cell
Metal PME envelope. Existing checkpoints continue to pin their recorded
backend, and every unsupported configuration retains the previous fallback.
See
[`metal-interaction-engine.md`](./metal-interaction-engine.md).
