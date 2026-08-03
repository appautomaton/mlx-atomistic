---
title: "MLX-First MD Acceleration"
---


This milestone keeps Python as the user-facing API but moves the MD nonbonded
hot path into MLX arrays and focused custom Metal kernels. The current priority
is recurring fixed-cell PME throughput on Apple Silicon while preserving the
existing force, constraint, and restart validation gates.

## Backends

- `mlx_dense` evaluates all pair interactions with dense MLX arrays. This is a
  serious baseline on M-series machines with large unified memory.
- `mlx_tiled` evaluates row blocks against all particles, reducing peak memory
  while keeping the pair math in MLX.
- `mlx_pairs` evaluates an explicit pair list with MLX gather/scatter. This is
  the path used with prebuilt neighbor lists.
- `mlx_dense_pairs` is a small-system neighbor-list backend selected by the
  neighbor manager's `auto` policy. It evaluates the dense periodic distance
  mask in MLX, then records explicit CPU `argwhere` compaction metadata because
  this MLX runtime does not expose dynamic `argwhere`/`nonzero` pair emission.
- `mlx_cell_pairs` is the large-system neighbor-list backend selected by the
  neighbor manager's `auto` policy above the dense-pair atom limit. On Metal it
  bins and stably orders atoms in MLX, prunes a fine cell grid by bounding-box
  distance, emits bounded candidate batches with a custom Metal kernel, and
  compacts exact pairs by prefix scan. Only small cell-occupancy and task
  metadata reaches the CPU; coordinates and compact pairs remain in MLX. The
  CPU backend retains the established NumPy cell-list implementation.
- `mlx_cell_blocks` keeps the periodic cell/bin candidate search in a
  fixed-shape block representation. It remains available for fixed-shape
  consumers and historical benchmark reproduction.
- `mlx_cell_tiles` spatially sorts atoms into eight-atom blocks, retains exact
  cutoff-plus-skin membership masks over non-empty 8x8 tiles, and groups tiles
  sharing a left block for the prepared orthorhombic PME force route. It keeps
  compact pairs as a diagnostic oracle during this development stage.
- `python_neighbor` means the Python/NumPy cell-list builder is included in the
  benchmark before MLX pair evaluation.
- `auto` uses dense MLX when no pair list is supplied and the dense memory
  estimate fits the configured budget; otherwise it falls back to tiled MLX.
  For neighbor-list managers, `auto` selects `mlx_dense_pairs` for supported
  small systems and `mlx_cell_pairs` above the small-system limit. The
  charged-PME performance runner explicitly selects `mlx_cell_tiles`.
  Production fixed-cell PME also selects tiles only inside the measured Metal
  envelope: 90,000--100,000 atoms, order-5 assignment, 9 A cutoff, 5.5 A skin,
  orthorhombic cell, and no NBFIX. Every other PME case keeps compact pairs.

## Current Hot-Path Recommendation

Measure with `python -m mlx_atomistic.benchmarks.md_acceleration --json` before
changing a force path. The large-system neighbor emitter and recurring direct
force path are now Metal-native. The first canonical-ID 8x8 atom-tile
implementation was parity-correct but slower end to end and was removed. The
later retained route instead creates blocks from spatial order and groups
same-left tiles, while preserving canonical atom indices at force scatter.

The current production path also uses specialized rigid-water/constraint
kernels and one fused standard-bonded force dispatch. The charged-PME
development runner now selects spatial tiles inside its narrow supported
envelope and fails its profile gate on a compact-pair fallback. The production
runner uses the same conservative envelope and records the selected backend in
checkpoints; resume pins that backend rather than silently changing the force
representation. General
large-system execution remains pair-oriented until the tile envelope is
broadened deliberately. The next performance work should reduce remaining
direct atomics and host-controlled rebuild work rather than add another wrapper
around the same 8x8 schedule.

For GPCRmd-scale periodic systems, dense all-pairs is not viable. The current
large-system route uses `mlx_cell_pairs`. The historical implementation used
CPU-side periodic bins and candidate arrays; the current Metal route uses a
fine device-resident grid and an AABB-pruned canonical half-stencil, then emits
and compacts candidates on Metal for the existing MLX pair force path. A
92,001-atom GPCRmd 729 short-range proof run under
`/tmp/mlx-atomistic-gpcrmd-729-mlx-cell-pairs` selected
`nonbonded_runtime.backend=mlx_cell_pairs` with no fallback. It tested
`candidate_count=390934237` local candidates, emitted `pair_count=48933140`,
and recorded `elapsed_wall_seconds=23.05953904206399`,
`neighbor_rebuild_wall_seconds=1.3890800829976797`,
`force_evaluation_wall_seconds=11.277963876025751`, `skin=2.5`, and one
rebuild. Those figures describe the historical hybrid implementation. The
under-5-second GPCRmd stretch target remains blocked by force evaluation,
per-step synchronization, and the cost of rebuilding and consuming a global
explicit pair array. Dynamic compaction is no longer a CPU bottleneck on Metal.

### Spatial Metal neighbor result

The retained 2026-07-31 JAC fixed-cell PME ladder used the same 9 A cutoff,
5.5 A skin, ten warmup steps, and 750 measured steps as its immediate control.
These are one-time engineering measurements on an M5 Max, not a CI gate.

| Atoms | Prior MLX | Spatial MLX | Time reduction | OpenMM artifact | Peak process memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 23,558 | 12.6688 s | 8.7643 s | 30.8% | 0.6810 s | 1.76 GB |
| 47,116 | 20.5108 s | 13.9166 s | 32.1% | 1.0945 s | 2.76 GB |
| 94,232 | 40.4986 s | 24.3424 s | 39.9% | 1.6861 s | 4.80 GB |

At 94,232 atoms the current 25x25x12 grid emits 184.1 million candidates,
down 60.7% from 468.3 million, and retains about 60.5 million exact pairs. All
three runs were finite, passed the existing constraint behavior, stayed below
40 GB, and had passing late-memory plateaus. The result is a substantial
retained gain but does not meet the one-order-of-magnitude OpenMM stretch
target. The next large-system lever is spatial tiles or cell-pair tasks that
feed the direct-space force kernel without materializing and repeatedly
scanning the full explicit pair array.

The OpenMM values in this historical ladder are provisional context. The local
artifact does not bind integration, timestep, thermostat, constraints,
completion, timing boundary, and workload identity strongly enough for a
publishable cross-engine ratio. Its 1.6861-second value therefore defines only
the provisional 16.8615-second ten-times stretch target.

### Prepared recurring-force result

A later fixed-cell pass retained four additional changes without changing the
cutoff, PME mesh, timestep, thermostat, or constraint tolerances:

1. Fixed cell-pair topology is cached while dynamic occupancies and exact pair
   membership are recomputed at every rebuild.
2. Box reciprocals and Lorentz-Berthelot particle invariants are prepared once
   for the recurring direct-space kernel.
3. The first of two mathematically redundant SETTLE velocity projections is
   skipped only for disjoint rigid-water groups; mixed or overlapping
   constraints retain the full projection.
4. Each prepared Metal worker processes eight adjacent compact pairs and
   accumulates repeated left-atom forces in registers before global atomics.

The fixed-topology cache first reduced the complete 94,232-atom, 750-step run
from 24.3424 to 19.1477 seconds. The retained prepared-force and constraint
changes then produced a complete 18.0356-second run: 25.9% below the original
spatial result, with a 3.43 GB peak. This is the latest stable complete number
for the retained stack before the eight-pair worker change. No formal
MLX/OpenMM ratio is reported because the OpenMM artifact did not pass the
current comparability manifest gate.

Because later complete samples were thermally throttled, the eight-pair worker
was admitted with position-balanced A/B evidence instead of an unmatched full
run. On the 60.5-million-pair 94,232-atom list it reduced the isolated direct
kernel by about 25%. Alternating 75-step trajectories reduced median measured
time from 5.1243 to 4.8785 seconds (4.8%) while keeping constraint residuals
near 1.5e-5 A. A bounded 92,001-atom GPCRmd 729 run also passed all finite-state,
constraint, HMR, PME-plan reuse, trajectory, and checkpoint checks with a
5.60 GB process peak. At that point, the provisional one-order-of-magnitude
target still required a stable complete time at or below 16.8615 seconds and
had not yet been demonstrated.

### Complete-step pipeline result

The 2026-08-01 pipeline pass used the same 94,232-atom JAC 2x2x1 system,
9 A cutoff, 128x128x64 PME mesh, 4 fs timestep, 300 K target, 1/ps friction,
ten warmup steps, and 75 measured steps throughout. It retained two changes:

1. The 89,160 constraints are partitioned once into 28,092 exact three-site
   SETTLE waters and 3,160 disjoint SHAKE clusters. The recurring position and
   velocity projections use their specialized batched Metal kernels; no JAC
   constraint remains on the generic path.
2. Ordinary force-only steps combine the standard bond, angle, periodic
   torsion, and improper families into one atomic Metal output. This removes
   four dense 94,232x3 force outputs, their four outer additions, and the
   equivalent of 13 separate scatter stages. Energy, virial, and component
   diagnostics still use the original independently tested force terms.

| Retained state | 75-step wall time | Change from corrected control |
| --- | ---: | ---: |
| Corrected pipeline control | 1.588653 s | baseline |
| Specialized constraints | 1.142599--1.190142 s | 25.1--28.1% lower |
| Constraints plus fused bonded forces, two-run median | 1.003149 s | 36.9% lower |
| Matched OpenMM/OpenCL, single precision | 0.102947 s | reference |

The final MLX/OpenMM ratio is `9.7586x`, so this named workload now meets the
one-order-of-magnitude stretch target, narrowly. The manifest-bound comparison
matches atom and force-term inventory, cell, PME method/cutoff/alpha/mesh,
constraints, fixed-cell Langevin-middle protocol, timestep, warmups, measured
steps, and timing boundary. Both engines include explicit final-device
completion in the timer and exclude initialization and I/O. Their random-number
implementations differ, so this is matched protocol throughput rather than
trajectory identity.

The two fused MLX samples were 1.001684 and 1.004613 seconds. The latter had a
`3.21e-5 A` maximum constraint residual and a 3.86 GB process-tree peak. A
fixed-coordinate production-force comparison gave `1.90e-4 kJ/mol/A` RMS and
`0.01692 kJ/mol/A` maximum absolute force delta against a
`437.69 kJ/mol/A` maximum reference force, consistent with the existing
float32 atomic-scatter envelope. The OpenMM process tree peaked at 0.401 GB.

Two attractive-looking experiments were removed after complete-step tests:

- The atom-tile direct kernel improved isolated force time from 0.005970 to
  0.004718 seconds, but its padded representation increased the full 75-step
  run from 1.190142 to 1.506723 seconds and raised peak memory from about 3.90
  to 6.06 GB.
- A constraint-aware compiled trajectory block reduced neighbor rebuilding,
  but repeated a much larger lazy graph. Even after rebuilds fell from nine to
  two, it took 1.87596 seconds versus the 1.190142-second control.

A synchronized post-constraint profile assigned 35.91% of wall time to direct
LJ plus screened Coulomb, 14.48% to the then-unfused bonded terms, 12.12% to
neighbor update/rebuild, 8.53% to constraints, 7.36% to force aggregation,
6.16% to diagnostics, 6.06% to reciprocal PME, and 4.34% to PME corrections.
The fused bonded route addressed the second item. The next pass therefore
changed the direct-space work layout instead of adding another Python-loop
cleanup.

### Retained spatial-direct increment

The retained 2026-08-01 development candidate is structurally different from
the rejected atom-tile experiment below. It builds spatially sorted eight-atom
blocks, evaluates exact cutoff membership for each 8x8 block pair on Metal,
and compacts only accepted tiles. Accepted tiles are sorted by their left block
and scheduled in groups of up to four, allowing one threadgroup to reuse the
left atom coordinates and parameters and to reduce its atomic force writes.
Small cell-count and task metadata still cross the host boundary during a
rebuild. The existing exact compact-pair inventory remains the correctness
oracle.

The same pass combines sparse PME exclusions, Coulomb exceptions, 1-4 terms,
and LJ exceptions into one Metal force buffer. This is a force-only production
optimization; diagnostic energy and component paths remain independently
observable.

| Bounded development check | Before | Retained candidate | Change |
| --- | ---: | ---: | ---: |
| Isolated direct-space force | 5.52227 ms | 3.79931 ms | 31.20% lower |
| Sparse PME correction force | 0.568062 ms | 0.262083 ms | 53.86% lower |
| Complete 75-step trajectory, median | 0.933094 s | 0.601954 s | 35.49% lower |

The spatial inventory contains 60,502,167 exact pairs in 2,751,445 tiles, or
176,092,480 padded lanes with 34.36% occupancy. Direct-force agreement with the
compact-pair route was `6.32e-5 kJ/mol/A` RMS and `6.71e-4 kJ/mol/A` maximum;
the fused correction comparison was `4.68e-5 kJ/mol/A` RMS and
`3.05e-4 kJ/mol/A` maximum. The complete trajectory remained finite, reused
its PME plan, ended below a `3.51e-5 A` maximum constraint residual, and peaked
at 5.69 GB across the process tree with a passing late-memory plateau check.
The current development representation intentionally retains the 484 MB exact
pair array as a diagnostic oracle; tile geometry, topology masks, and that
oracle total about 578 MB of persistent indexed state. Per-pair LJ scales are
no longer constructed for an admitted tile route.

The complete gate used the position-balanced order pairs, tiles, tiles, pairs.
The pair samples were 1.045140 and 0.821048 seconds; the tile samples were
0.577860 and 0.626048 seconds. This clears the 0.85-second development target,
but the candidate does not replace the existing manifest-bound 1.003149-second
MLX result or establish a new OpenMM ratio. The main remaining costs are
direct-space work, neighbor update/rebuild, constraints, reciprocal PME, force
aggregation, and Python-side launch orchestration.

### Production routing follow-up

A fresh raw 75-step bundle on 2026-08-01 reproduced the retained route at a
0.558220-second tile median versus a 0.859394-second compact-pair control
median, 35.04% lower. All force, inventory, constraint, finite-state, PME-plan,
and 40 GB memory gates passed. The recurring order-5 reciprocal path now omits
particle-energy output and reduction during force-only steps. Its synchronized
route time fell from 0.081776 to 0.067008 seconds across 74 calls, 18.06%, while
complete clean wall remained statistically flat at 0.560536 seconds.

Two bounded direct-force experiments were rejected. Raising the same-left tile
group width from four to eight improved the isolated tile kernel by 5.53% but
changed complete wall from 0.558220 to 0.559019 seconds. A two-pass non-atomic
right-block reducer produced a 264 MB temporary tile-force buffer, reduced the
direct advantage over pairs to 8.52%, and raised complete wall to 0.734520
seconds. Both experiments were removed. The retained tile route is now selected
by production only inside the measured envelope above; compact pairs remain the
fallback, and checkpoint resume preserves the originally recorded backend.

### 23,558-atom 5DFR transfer result

A second fixed-cell PME workload confirmed that the retained optimizations are
not specific to the 94,232-atom JAC replication. The 5DFR system contains
23,558 atoms, 7,023 SETTLE waters, and 790 disjoint SHAKE clusters. It uses a
56x56x56 order-5 PME mesh, 9 A cutoff, 5.5 A skin, 4 fs timestep, ten warmups,
and 750 measured NVT steps.

Three general runtime corrections were retained. Standard bonded-force fusion
now accepts any unique set of at least two supported families, so 5DFR's bond,
angle, and periodic-torsion terms no longer miss the Metal route merely because
the artifact has no improper torsions. Force accumulation adopts the first
force output directly instead of allocating a zero array and adding it. MD
execution also applies a scoped 4 GB MLX cache limit and restores the caller's
allocator policy afterward; this replaces a full cache clear after every
spatial rebuild while leaving direct neighbor-manager use fail-safe.

| Complete 750-step 5DFR result | Wall time | Throughput |
| --- | ---: | ---: |
| Earlier compact-pair result | 7.046576 s | 36.7838 ns/day |
| Bonded fusion plus direct force seeding | 3.988908 s | 64.9802 ns/day |
| Scoped 4 GB cache, compact pairs | 3.682865 s | 70.3800 ns/day |
| Scoped cache plus spatial tiles | 2.979058 s | 87.0074 ns/day |

The final tile result is 57.72% lower than the earlier compact-pair result. It
remained finite, reused one PME plan, ended at a `1.51e-5 A` maximum constraint
error, peaked at 2.25 GB across the process tree, and passed the late-memory
plateau check. A fresh OpenMM/OpenCL reference measured 475.947 ns/day, making
the contextual throughput gap about 5.47x. That ratio is not a formal
manifest-matched comparison. The upstream `pme` benchmark performs five
initial integration steps and a pre-timer energy query, then includes its final
energy query inside the timed interval. It does not minimize this workload.
OpenMM also enables center-of-mass motion removal by default, while the
historical MLX row above omitted that operation. The OpenMM result has no
persisted mesh, Ewald alpha, or timing-boundary manifest, so the ratio remains
contextual and the omitted MLX operation slightly favored MLX. The charged-PME
runner now derives the center-of-mass removal cadence from the prepared
artifact so future measurements do not repeat that mismatch.

The fixed-coordinate tile inventory exactly matched all 14,699,933 compact
pairs. Direct-force differences were `6.10e-5 kJ/mol/A` RMS and
`6.71e-4 kJ/mol/A` maximum against a `671.76 kJ/mol/A` maximum reference force.
Sparse-correction differences were `4.61e-5 kJ/mol/A` RMS and
`2.44e-4 kJ/mol/A` maximum. The tile direct kernel was 23.95% faster than the
pair kernel in the interleaved probe.

Production tile routing is therefore limited to two measured atom-count
windows, 23,000--24,000 and 90,000--100,000, with the existing fixed-cell,
orthorhombic, order-5 PME, 9 A cutoff, 5.5 A skin, and no-NBFIX gates. The gap
between those windows continues to use compact pairs rather than extrapolating
from unmeasured sizes.

### Rejected two-step neighbor admission

The 2026-08-02 experiment delayed the constrained NVT neighbor-displacement
decision for two steps, then either committed both steps or rolled back and
replayed them with the authoritative per-step path. The matched 5DFR runs used
the same prepared artifact, Metal route, 5.5 A skin, and per-step
center-of-mass motion removal on both sides.

| 750-step 5DFR route | Run 1 | Run 2 | Median wall | Median throughput |
| --- | ---: | ---: | ---: | ---: |
| Exact per-step admission | 2.326254 s | 2.348854 s | 2.337554 s | 110.888 ns/day |
| Two-step transaction | 2.356621 s | 2.425385 s | 2.391003 s | 108.429 ns/day |

The transaction was 2.29% slower by median wall time and was therefore
removed. It accepted 338--339 epochs but replayed 42--44 steps. Combining two
checks reduced host decision count without reducing displacement work: the
single materialization evaluated a two-step lazy graph, while rejected epochs
also repeated integration and one force evaluation. Median measured neighbor
update time consequently rose instead of falling. All runs stayed finite,
constraint errors remained below `1.70e-5 A`, and the hard-bounded process-tree
peak stayed below 2.52 GB, so this was a performance rejection rather than a
correctness or memory failure.

### Rejected fused neighbor-admission producer

The 2026-08-03 experiment kept exact per-step admission and its single host
decision, but replaced the steady MLX finite/minimum-image/maximum chain with
one custom Metal threadgroup reduction. The kernel produced the maximum squared
displacement and a non-finite flag; the host restored the float32 distance and
made the unchanged strict rebuild-threshold comparison after the existing
materialization.

Direct graph capture fell from 34 eager primitives to 8 with one CustomKernel.
Both routes used one `mx.eval`, returned the same `0.30671805 A` maximum
displacement, and made the same rebuild decision. CPU/fallback tests, real-Metal
partial-group and non-finite oracles, exact-threshold/nextafter branches, forced
rebuilds, and variable-cell parity all passed before timing.

| 75-step 5DFR gate | C1 | A1 | A2 | C2 | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact MLX admission | 0.143159 s | -- | -- | 0.144269 s | 0.143714 s |
| Fused Metal producer | -- | 0.144530 s | 0.144230 s | -- | 0.144380 s |

The candidate was 0.46% slower by median wall time, and the paired directions
disagreed (`-0.96%` and `+0.03%`). All four complete runs passed their finite,
constraint, route, and workload checks; process-tree peaks were 0.97--1.10 GB
under the 40 GB limit. The candidate therefore failed the immediate-rejection
gate and was removed. JAC and 750-step transfer runs were not performed. The
result shows that this primitive reduction did not pay for its custom
reduction/output handling while the load-bearing admission sync remained; it
does not establish which internal Metal cost dominated or rule out a different
producer design.

### Retained 32-lane spatial direct-force schedule

The 2026-08-02 follow-up replaced the spatial direct kernel's 64-thread,
pair-scratch schedule with one 32-lane SIMD group for up to four tiles sharing
an eight-atom left block. Each lane retains one right atom in registers and
walks the eight left atoms. SIMD reductions produce the left-atom forces, while
each lane writes its accumulated right-atom force once. This preserves the
exact tile membership and topology masks while removing the repeated per-tile
barriers, 64-pair threadgroup scratch arrays, and repeated right-atom writes.

| Complete constrained NVT gate | Control | 32-lane schedule | Wall reduction |
| --- | ---: | ---: | ---: |
| 23,558-atom 5DFR, 75-step median | 0.174411 s | 0.152943 s | 12.31% |
| 23,558-atom 5DFR, 750 steps | 2.337554 s | 2.066392 s | 11.60% |
| 94,232-atom JAC, matched 75-step sample | 0.931896 s | 0.850194 s | 8.77% |

The 750-step 5DFR run reached 125.436 ns/day, remained finite, ended at a
`1.62e-5 A` maximum constraint residual, and peaked at 2.10 GB across the
bounded process tree. The JAC transfer retained its complete SETTLE/SHAKE
partition, ended at `3.33e-5 A`, and peaked at 5.21 GB. All 22 Metal parity
tests for the fused direct, topology, constraint, and neighbor kernels passed.
The JAC row is a back-to-back code A/B; its absolute wall time should not be
compared with older samples collected under different machine conditions.

Four adjacent ideas were measured and removed. On-demand exact-pair decoding
made the complete 5DFR path 1.8% slower, a speculative second Metal stream was
6.6% slower, omitting the disjoint SHAKE pre-force projection was 1.7% slower,
and a handwritten fused BAOAB drift improved the median by only 0.5%. None met
the 5% complete-wall retention threshold. The retained change is therefore the
kernel work schedule only; the neighbor lifecycle, integrator sequence,
constraints, and scientific workload are unchanged.

### Guarded intra-step force submission

The next retained change overlaps host graph construction with Metal execution
without changing the integrator or adding a synchronization barrier. On an
ordinary constrained Langevin step with a prepared spatial-tile force pipeline,
the runtime submits the completed force graph through `mx.async_eval` before it
builds the final kick and constraint graph. CPU execution, diagnostics, the
final step, synchronized route profiling, non-Langevin dynamics, non-tile
neighbors, and unsupported force paths keep their previous synchronous route.

The evidence categories are intentionally separate:

| Evidence type | Result |
| --- | --- |
| Measured structural census | One steady 5DFR step contained 181 MLX primitive nodes, including 28 in the force subgraph. This is not a Metal-dispatch count. |
| Measured host opportunity | Post-force host graph construction took about 0.361 ms/step; asynchronous submission took about 0.023 ms. |
| Estimated ceiling | At most about 17.8% of a 5DFR step could be hidden if all eligible host work overlapped. This was a design bound, not a predicted result. |
| Measured 75-step 5DFR gate | Control median 0.158528 s; candidate median 0.143178 s; 9.68% lower. Both C-to-A comparisons agreed. |
| Measured 75-step JAC transfer | Control median 0.754487 s; candidate median 0.765292 s; 1.43% higher and inside the 2% neutrality limit. One 1.244 s candidate outlier prevents a broader claim. |
| Measured 750-step 5DFR confirmation | 2.037116 s to 1.956978 s; 3.93% lower, or 127.239 to 132.449 ns/day. |

The complete candidate run remained finite, ended at the same `1.35e-5 A`
constraint residual as its paired control, reused the same PME plan, and peaked
at 2.60 GB across the bounded process tree. The long-run gain is smaller than
the short gate, so this is classified as a modest scheduling improvement. It
does not change the scientific operations and does not close the remaining
OpenMM throughput gap.

A second experiment built fixed atom-owner maps for disjoint SETTLE and SHAKE
families and replaced their recurring sparse-scatter application chains with
one dense per-atom Metal write. Metal parity covered periodic positions,
pre-force velocities, final velocities, unconstrained atoms, noncontiguous
indices, and one-to-three SHAKE peripherals; CPU, overlap, and profiler
fallbacks also passed. The complete 75-step 5DFR median changed from 0.142829
to 0.139473 seconds, only 2.35%, with inconsistent paired gains. The specialized
kernel, maps, and tests were therefore removed under the 5% retention rule.

Two later pipeline experiments were also rejected by the same bounded 75-step
5DFR gate. Coalescing SETTLE and SHAKE sparse constraint writes changed complete
wall time from 0.202904 to 0.574724 seconds, a 2.83x regression, because this
system constrains nearly every atom and the added concatenation and scatter
work outweighed fewer full-position additions. Flattening the prepared force
pipeline and replacing its nested MLX additions with one cached Metal
force-buffer sum changed 0.158592 to 0.159115 seconds, 0.33% slower. Both paths
remained finite and inside the constraint and 40 GB memory gates, but neither
earned a repeat, JAC transfer, or 750-step run. Their implementation and tests
were removed. The second result also narrows the remaining diagnosis:
force-buffer addition is measurable under synchronized profiling, but it is
not a material complete-trajectory bottleneck.

An ordinary-step graph-pruning experiment then separated SETTLE/SHAKE position
projection from its discarded error result and hoisted three invocation- or
topology-invariant MLX values. Checked CPU and Metal parity passed, but a
steady-step graph capture shrank only from 153 to 149 MLX primitives: two
`ExpandDims` and two `Reshape` nodes. The error result was already absent from
the materialized Metal graph, so this was primarily a small host-construction
change rather than the expected compute deletion. In the real-Metal
`C1 -> A1 -> A2 -> C2` gate, control walls were 0.144676 and 0.148414 seconds;
candidate walls were 0.143181 and 0.143799 seconds. The candidate median was
2.08% lower, with both paired directions favorable, while all finite-state,
constraint, and 40 GB memory checks passed. Because that result was below the
3% immediate-rejection line and the 5% retention threshold, the implementation
and candidate-only tests were removed. No JAC or 750-step run was performed.

A follow-up staged-compilation experiment placed separate `mx.compile` units
around the ordinary constrained-Langevin work before and after the retained
asynchronous force submission. Compiled/eager Metal parity passed across
forced neighbor rebuilds, NPT and unsupported routes stayed eager, and no
additional synchronization caller appeared. Graph capture reduced the same
steady step from 153 primitives when eager to 112 with pre-force compilation,
147 with post-force compilation, and 106 with both; the category signatures
were unchanged across a rebuild. Structural fusion nevertheless did not
improve the complete trajectory. In the real-Metal `C1 -> A1 -> A2 -> C2`
gate, control walls were 0.143491 and 0.143358 seconds, while candidate walls
were 0.142737 and 0.144577 seconds. The candidate median was 0.16% slower and
the paired directions disagreed. All finite-state, constraint, route, and
40 GB memory gates passed, but the result triggered immediate rejection. The
compiled units, eager extraction, eligibility plumbing, and candidate-only
tests were removed. JAC and 750-step runs were not performed. This result is
specific to the current 23,558-atom staged route; it does not claim that
`mx.compile` is generally ineffective elsewhere in MLX Atomistic.

### Rejected atom-tile result

The 2026-07-31 canonical-ID atom-tile experiment tested the full route instead
of assuming that a faster kernel would make the trajectory faster. Its tiles
were derived from the compact-pair list, unlike the retained spatially built
route above. Its isolated 8x8 Metal
kernel reduced median direct-force latency from 0.005548 to 0.004257 seconds,
a 30.33% win, and matched the explicit-pair force within the established
float32 tolerance. Initial integration lost badly because tile construction
and CPU topology binding raised the 94,232-atom, 75-step wall time from the
1.850210-second explicit-pair control to 34.429230 seconds.

One bounded lifecycle correction then derived tiles from the exact compact-
pair oracle and packed topology masks on Metal. It reduced the atom-tile wall
time from 34.821962 to 2.963867 seconds, geometry construction from 7.933098 to
0.280091 seconds, and topology binding from 24.648279 to 0.012191 seconds. All
finite-state, constraint, neighbor, PME-reuse, route, and no-fallback checks
passed. The corrected route nevertheless remained 60.19% slower than explicit
pairs and increased the short-run process-tree peak from 4.17 to 8.06 GB. Its
9,664,362 tiles represented 60,504,316 exact pairs through 618,519,168 padded
lanes, so the construction fix did not make the complete representation
competitive.

The conditional repeat and 750-step proof were not run because the first exact
comparison was decisive and projected about 29.6 seconds for 750 steps, above
both the retained complete result and the 16.8615-second stretch target. The
candidate tile kernel, production routing, experimental selector, and metadata
were removed. Exact compact-pair ownership remains the verified correctness
foundation for the later spatial route.

A fresh synthetic orthorhombic parity ladder now validates this route at
1k/4k/16k/50k/92,001 atoms against the tiled all-pairs MLX oracle. At 92,001
atoms, the compact build took 0.545 s, the explicitly synchronized pair-force
evaluation took 0.068 s, and the tiled oracle took 112.1 s; relative energy
delta was `4.56e-7` and maximum absolute force delta was `8.49e-7`. This
2026-07-13 result remains diagnostic because the local real-fixture cache was
unavailable for that measurement. A later source-backed GPCRmd 729 run now
passes a separate bounded fixed-cell parity/NVT/restart gate using
`mlx_cell_blocks`/`NeighborBlocks`; it does not change the classification of the
synthetic neighbor row. The production runner has since moved to compact
`mlx_cell_pairs`. Retained NPT diagnostic reuse, reciprocal-PME graph
compilation, and fused parameterized LJ/direct-PME Metal kernels then reduced a
matched 75-step DHFR NPT prefix from 142.87 to a repeated median of 13.77
seconds. Process-tree peak memory fell from 27.33 GB to 5.18--6.11 GB across
the retained samples, and the same numerical gates passed. Order-five
reciprocal PME now also uses dedicated Metal charge-spread and
potential-derivative interpolation kernels, reducing the 2,269-atom alanine
50-step fixed-cell median from 0.853 to 0.537 seconds without pressure
diagnostics and from 1.313 to 0.987 seconds with analytic pressure diagnostics.
The artifact loader now also routes complete three-site water constraint
triangles to a batched MLX rigid-water projector while preserving a fail-closed
generic path for incomplete or mixed geometries. That reduced the same
fixed-cell median from 0.537 to 0.419 seconds and the analytic-pressure median
from 0.987 to 0.863 seconds. The complete 100-step NVT plus 1,000-step NPT
alanine check then passed all 16 unchanged science gates in 15.899 seconds,
down from 23.110 seconds, with a `3.34e-6` A maximum constraint error and a
0.94 GB process-tree peak. Fixed-coordinate OpenMM parity remained within the
accepted energy and force gates. These measurements were made on an M5 Max in
low-power mode. This is bounded optimization and one-picosecond stability
evidence, not a production-length validation. Charged fixed-cell PME also has
a separate 94,232-atom JAC validation. See
[`docs/benchmarks/scalable-neighbor-nonbonded-runtime-m5max.md`](../benchmarks/scalable-neighbor-nonbonded-runtime-m5max.md)
and
[`docs/benchmarks/scalable-charged-pme-runtime-m5max.md`](../benchmarks/scalable-charged-pme-runtime-m5max.md),
and
[`docs/benchmarks/gpcrmd-729-pme-runtime-m5max.md`](../benchmarks/gpcrmd-729-pme-runtime-m5max.md).

For the active solvated ligand-receptor notebook system, the near-term GPU
occupancy lever is independent replica batching. The system is only a few
hundred atoms, so one trajectory does not expose much work to the GPU. Use the
prep APIs to advance multiple physically independent velocity seeds in one MLX
loop:

```python
from mlx_atomistic.prep.replicas import run_ligand_receptor_replicas

run_ligand_receptor_replicas(
    "notebooks/ligand-receptor-motion/data/mlx-real-md/example-200ps-r4",
    replicas=4,
    selected_replica=0,
    steps=200000,
    dt=0.001,
    sample_interval=100,
)
```

For repeatable profiling across durations and replica counts:

```python
from mlx_atomistic.prep.replicas import profile_ligand_receptor_performance

profile_ligand_receptor_performance(
    "notebooks/ligand-receptor-motion/data/perf/replica-profile",
    durations_ps=[5, 50, 200],
    replica_counts=[1, 4, 8, 16],
    dt=0.001,
    sample_interval=100,
)
```

The profile reports wall time, per-replica and aggregate steps/s, aggregate
ps/s, GPU-visible atoms, dense pair slots, force-evaluation cost, constraint
projection cost, diagnostic cost, max constraint error, and artifact size.

## Interpreting The Benchmark

The benchmark reports `ms_per_eval`, `neighbor_rebuild_ms_per_eval`,
`force_eval_ms_per_eval`, an estimated dense memory footprint,
`ns_per_day_at_dt_0_002`, and force/energy deltas relative to dense MLX. The
`ns_per_day` number is only a throughput-style indicator because the MD engine
uses reduced internal units. Use the separated rebuild and force timings when
deciding whether the current bottleneck is pair construction or pair evaluation.
