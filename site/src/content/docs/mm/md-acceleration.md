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
  sharing a left block for the prepared orthorhombic PME force route. Compact
  diagnostic pairs are deferred and materialized only when an explicit pair
  consumer requests them.
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
and compacts candidates on Metal for the existing MLX pair force path. The
under-5-second GPCRmd stretch target remains blocked by force evaluation,
per-step synchronization, and the cost of rebuilding and consuming a global
explicit pair array. Dynamic compaction is no longer a CPU bottleneck on Metal.
Current GPCRmd 729 figures live in
[`docs/benchmarks/gpcrmd-729-pme-runtime-m5max.md`](../benchmarks/gpcrmd-729-pme-runtime-m5max.md).

## Measurement Noise Floor

Read this before any result below. These M5 Max measurements were made in
low-power mode, and the machine does not hold a stable clock under load.

Twelve complete 5DFR runs at 23,558 atoms, 9 A cutoff, 5.5 A skin, seed 17, and
two fixed sources span:

| Measured steps | Runs | Range | Spread |
| ---: | ---: | --- | ---: |
| 75 | 4 | 1.6828--1.8273 ms/step | 8.6% |
| 750 | 8 | 2.3140--2.8436 ms/step | 22.9% |

Two consequences constrain how the numbers below may be read.

A short gate and a production-length run measure different machine states. One
unchanged source ran at 1.6828 ms/step over 75 steps and 2.5072 ms/step over
750 steps, a 49% difference with no code change. Short-gate percentages
therefore do not transfer, which is consistent with short-gate gains in this
document shrinking on their 750-step and JAC confirmations.

Single-pair wall-time differences below roughly 10% are not resolvable here.
Several entries under [Closed directions](#closed-directions) were decided at
0.16% to 3%. Those decisions are sound as "no measurable gain" and the
implementations are correctly removed, but they do not establish the sign of
the effect. Prefer exact counts, interleaved medians with rotating order, or
ratios that cancel clock drift.

## Retained Results

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

Two attractive-looking experiments were removed after complete-step tests. Both
are recorded under [Closed directions](#closed-directions).

A synchronized post-constraint profile assigned 35.91% of wall time to direct
LJ plus screened Coulomb, 14.48% to the then-unfused bonded terms, 12.12% to
neighbor update/rebuild, 8.53% to constraints, 7.36% to force aggregation,
6.16% to diagnostics, 6.06% to reciprocal PME, and 4.34% to PME corrections.
The fused bonded route addressed the second item. The next pass therefore
changed the direct-space work layout instead of adding another Python-loop
cleanup.

Synchronized route shares are structural upper bounds and must not be read as
clean wall shares. The profiler completes every route boundary with `mx.eval`,
so a 5DFR step that runs in about 2.31 ms clean takes about 4.82 ms
instrumented across roughly 17 evaluation boundaries. Routes with many
boundaries and little queued work absorb the added synchronization: force
aggregation measures 384 us/step under instrumentation, of which 4.9 us/step is
graph and host work and the remainder is three command-buffer round trips. Use
these shares to rank boundaries structurally, never to size a candidate.

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

Two bounded direct-force experiments measured in the same pass were removed and
are recorded under [Closed directions](#closed-directions). The retained tile
route is now selected by production only inside the measured envelope above;
compact pairs remain the fallback, and checkpoint resume preserves the
originally recorded backend.

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

Four adjacent ideas were measured in the same pass and removed; they appear
under [Closed directions](#closed-directions). The retained change is therefore
the kernel work schedule only; the neighbor lifecycle, integrator sequence,
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

Four experiments measured against this parent were removed and are recorded
under [Closed directions](#closed-directions): dense constraint owner maps,
coalesced sparse constraint writes, a cached Metal force-buffer sum, and
ordinary-step graph pruning. Two of them were later revisited under composition
and retained; see [Cumulative interaction stack](#cumulative-interaction-stack).

A staged-compilation experiment placed separate `mx.compile` units around the
ordinary constrained-Langevin work before and after the retained asynchronous
force submission. Compiled/eager Metal parity passed across forced neighbor
rebuilds, NPT and unsupported routes stayed eager, and no additional
synchronization caller appeared. Graph capture reduced the same steady step
from 153 primitives when eager to 112 with pre-force compilation, 147 with
post-force compilation, and 106 with both. The complete trajectory did not
improve: the candidate median was 0.16% slower with disagreeing paired
directions, so the compiled units, eager extraction, and eligibility plumbing
were removed. This result is specific to the current 23,558-atom staged route
and does not claim that `mx.compile` is generally ineffective elsewhere in MLX
Atomistic. It is consistent with the step budget in [Where the time
goes](#where-the-time-goes): the ordinary step spends most of its wall waiting
for queued device work, which leaves little host graph construction for
compilation to remove.

### Neighbor rebuild lifecycle experiments

The 2026-08-03 phase tested six independent ways to shorten the complete
spatial-tile rebuild lifecycle. Every candidate first preserved the 23,558-atom
5DFR inventory exactly: 14,699,933 pairs, 646,965 non-empty tiles, 163,090
force groups, and six byte-identical array digests. Process-tree peaks remained
between 0.85 and 1.00 GB under the 40 GB limit. No candidate reached the 10%
local admission gate, so no JAC or 750-step run was performed and all candidate
code and tests were removed. The six results appear under [Closed
directions](#closed-directions).

The structural findings still narrow future work. A race-free stable linear
placement was possible, but scanning each left-block row cost more than MLX
`argsort` plus gathers. The prefix-tail experiment showed that slices created
after an `mx.eval` can add scalar work, but its benefit did not reproduce the
10% gate. Schedule validation and topology-mask waits could be moved safely,
yet complete-boundary timing showed that the waits were either too small or
merely transferred to first force evaluation. The finite certificate removed
one reduction but added more host bookkeeping than it saved.

### Cumulative interaction stack

A later 2026-08-03 phase revisited the decision method, not the scientific
workload. The earlier fixed 3%, 5%, and 10% per-candidate thresholds were useful
for rejecting obvious regressions, but they could not answer whether several
small changes improve the same graph when composed. The cumulative phase
therefore reconstructed exact source variants, measured interactions directly,
and retained no result by adding isolated percentages.

The final ordinary-step stack contains three changes:

1. projection-only ordinary SETTLE/SHAKE work that omits discarded error graphs
   while checked diagnostic steps keep the public fail-closed path;
2. immutable dense atom-owner maps and one Metal write for eligible disjoint
   SETTLE/SHAKE position and velocity updates, with the prior fallbacks intact;
3. one fused Metal BAOAB drift operation before the retained asynchronous force
   boundary.

Together they reduce the captured steady-step graph from 126 to 101 MLX
primitives. More importantly, the complete runtime results are favorable:

| Evidence | Control | Retained stack | Direct result |
|---|---:|---:|---:|
| 5DFR, 23,558 atoms, 75 steps, C-A-A-C medians | 0.143160 s | 0.129063 s | 9.85% faster; paired gains 10.63% and 9.06% |
| Fresh 5DFR confirmation pair | 0.135353 s | 0.130723 s | 3.42% faster |
| Remove fused BAOAB from the final graph | 0.143556 s | 0.129007 s | Complete stack 10.13% faster |
| Disable dense ownership in the final graph | 0.143125 s | 0.129982 s | Complete stack 9.18% faster |
| Restore checked ordinary constraint work | 0.144781 s | 0.129209 s | Complete stack 10.76% faster |
| JAC, 94,232 atoms, 75 steps, C-A-A-C medians | 0.756679 s | 0.730588 s | 3.45% faster; paired gains 6.45% and 0.24% |
| 5DFR, 750-step paired confirmation | 2.019479 s | 1.807516 s | 10.50% faster; 128.35 to 143.40 ns/day |

Every run used the same prepared artifact, seed, timestep, cutoff, skin,
per-step admission, PME plan, sampling cadence, and spatial-tile route. All
finite-state and constraint gates passed. The largest process-tree peaks were
1.36 GB for the short composition work, 5.22 GB for JAC, and 2.29 GB for the
750-step pair, all below the 40 GB limit. These M5 Max measurements were made
in low-power mode.

The larger leave-one-out effects do not mean each component independently
provides about 10%. Earlier isolated results ranged from regression to a few
percent. They mean the three changes alter the same ordinary constraint and
integration graph and are valuable as one interaction-tested stack.

The rebuild substack remained empty. Four rebuild candidates measured in this
phase appear under [Closed directions](#closed-directions). The retained result
is therefore the three-component ordinary-step stack plus the existing neighbor
rebuild path.

### Earlier retained kernel and parity evidence

These predate the spatial-tile work above and remain part of the retained
runtime. All are M5 Max, low-power mode, bounded optimization and
one-picosecond stability evidence rather than production-length validation.

- **Compact neighbor parity.** A synthetic orthorhombic ladder at
  1k/4k/16k/50k/92,001 atoms validates the compact route against the tiled
  all-pairs MLX oracle. At 92,001 atoms the compact build took 0.545 s and the
  explicitly synchronized pair-force evaluation 0.068 s, against 112.1 s for
  the oracle, at `4.56e-7` relative energy delta and `8.49e-7` maximum absolute
  force delta. This 2026-07-13 result stays diagnostic because the local
  real-fixture cache was unavailable. A later source-backed GPCRmd 729 run
  passes a separate bounded fixed-cell parity/NVT/restart gate using
  `mlx_cell_blocks`/`NeighborBlocks` and does not change that classification.
  The production runner has since moved to compact `mlx_cell_pairs`.
- **NPT diagnostic reuse, reciprocal-PME graph compilation, and fused
  parameterized LJ/direct-PME kernels.** A matched 75-step DHFR NPT prefix fell
  from 142.87 s to a repeated median of 13.77 s, with process-tree peak memory
  down from 27.33 GB to 5.18--6.11 GB and the same numerical gates passing.
- **Order-five reciprocal PME charge-spread and potential-derivative kernels.**
  The 2,269-atom alanine 50-step fixed-cell median fell from 0.853 to 0.537 s
  without pressure diagnostics, and from 1.313 to 0.987 s with analytic
  pressure diagnostics.
- **Batched MLX rigid-water projector.** The artifact loader routes complete
  three-site water constraint triangles to it while keeping a fail-closed
  generic path for incomplete or mixed geometries. The same medians fell from
  0.537 to 0.419 s and from 0.987 to 0.863 s.
- **Complete alanine science gate.** The 100-step NVT plus 1,000-step NPT check
  passed all 16 unchanged gates in 15.899 s, down from 23.110 s, at a
  `3.34e-6 A` maximum constraint error and a 0.94 GB process-tree peak.
  Fixed-coordinate OpenMM parity stayed inside the accepted energy and force
  gates.

Charged fixed-cell PME also has a separate 94,232-atom JAC validation. See
[`docs/benchmarks/scalable-neighbor-nonbonded-runtime-m5max.md`](../benchmarks/scalable-neighbor-nonbonded-runtime-m5max.md)
and
[`docs/benchmarks/scalable-charged-pme-runtime-m5max.md`](../benchmarks/scalable-charged-pme-runtime-m5max.md),
and
[`docs/benchmarks/gpcrmd-729-pme-runtime-m5max.md`](../benchmarks/gpcrmd-729-pme-runtime-m5max.md).

## Where The Time Goes

The 5DFR ordinary step is bounded by two exact measurements. Neither is a
single wall-time sample, so both survive the noise floor above.

The step waits on the device far more than it builds graphs. Exact per-step
admission materializes the maximum displacement once, and that is the only
blocking evaluation in an ordinary step, so every queued device operation
drains there. Across eight complete 750-step 5DFR runs the non-rebuild part of
`NeighborListManager.update` accounts for 58.1% to 62.2% of clean wall,
mean 60.4%, while the runs themselves span 2.3140 to 2.8436 ms/step. That
fraction is a ratio and therefore stable where the absolute times are not.
Rebuild takes a further 18.6% to 21.7%, leaving under a quarter of the step for
host graph construction and every other boundary combined. This is the
structural reason host-side candidates keep failing to move complete wall.

The direct-force kernel schedules far more atom-pair lanes than the cutoff
admits. One threadgroup owns four tiles and always runs all of them, so the
scheduled count is `force_group_count * 4 * 64`. At production settings:

| Lane level | Lanes per step | Share |
| --- | ---: | ---: |
| Scheduled by the kernel | 41,751,040 | 100.00% |
| Force-group padding | 345,280 | 0.83% |
| Tile padding | 26,705,827 | 63.96% |
| Inside cutoff plus skin only | 11,191,063 | 26.80% |
| Inside the 9 A cutoff | 3,508,870 | 8.40% |

A geometric model reproduces both ends of that ladder:

    scheduled ~ (N/2) (4/3) pi (cutoff + skin + 2 r_block)^3 rho
    useful    ~ (N/2) (4/3) pi (cutoff)^3 rho

At `rho = 0.09776` atoms/A^3 and `r_block = 2.693 A` for eight-atom blocks, it
predicts 3,516,119 useful lanes against 3,508,870 measured, and a
scheduled-to-useful ratio of `((9.0 + 5.5 + 5.39)/9.0)^3 = 10.79` against 11.90
measured. The three terms are the physical cutoff, the skin, and the block
extent set by `DEFAULT_MLX_CELL_TILE_BLOCK_SIZE`.

Reducing that ratio does not pay proportionally. Holding atoms, cutoff, and
in-cutoff pairs fixed and varying only the skin moves scheduled lanes by 2.211x
while interleaved-median direct-force time moves by 1.381x:

| Skin | Scheduled lanes | Direct force | Glane/s |
| ---: | ---: | ---: | ---: |
| 2.0 A | 26,209,280 | 0.747 ms | 35.10 |
| 3.5 A | 33,144,832 | 0.840 ms | 39.47 |
| 5.5 A | 41,751,040 | 0.900 ms | 46.41 |
| 8.0 A | 57,946,368 | 1.031 ms | 56.20 |

Time scales as roughly the 0.41 power of scheduled lanes and lane throughput
rises with lane count, so the kernel is partly latency-bound rather than
work-bound at the production skin. Extrapolating that exponent, removing every
padded and skin-only lane would take the direct route from about 0.900 to about
0.329 ms against a 2.31 ms step. No reachable design approaches that bound:
the smallest scheduled count anywhere in the skin sweep is 26,209,280, worth
0.153 ms, and it costs 143 rebuilds instead of 22. Lane occupancy is therefore
not a viable lever on this workload, which is consistent with smaller skins
being closed on complete wall.

Those route times include a fixed cost that is not kernel work. Replacing the
kernel body with an immediate return still costs about 0.41 ms through the same
timing boundary, which is dispatch, output initialization, and one
synchronization round trip. A real MD step batches its submission and pays that
once for the whole step, not once per force route, so isolated route times
overstate the kernel and their absolute values do not transfer between thermal
states. Use them for ratios between arms, never as a share of the step.

### Stacking the exact direct-kernel savings

Ablation showed that no single part of the kernel dominates. Each arm below
removes one cost and keeps the rest:

| Removed | Share of kernel work |
| --- | ---: |
| Ewald erfc and exp | 16% |
| Lennard-Jones branch | 15% |
| Atomic force writes | 15% |
| Per-iteration mask loads | 11% |

That distribution is the finding. The engine is not slow because of one
bottleneck, it is slow because several similar costs compose, so the productive
response is to remove them together rather than to search for a dominant one.
Five exact transformations were retained. Each preserves every float the kernel
reads and every arithmetic operation it performs on a live path:

| Transformation | Kernel speedup, cumulative |
| --- | ---: |
| Read the three per-tile mask words once per tile instead of once per left slot | 1.101x |
| Carry `_TILE_PME_GROUPS_PER_THREADGROUP` force groups per threadgroup | 1.165x |
| Hoist the box constants and hot `params` entries above the loop | 1.248x |
| Skip the `unswitched_energy` chain when switching is disabled | 1.383x |
| Return `erfc` from its own polynomial instead of one minus `erf` | 1.566x |

The last two deserve a note. `unswitched_energy` reaches the force only through
a product with `switch_derivative`, which is identically zero when switching is
off, so the whole chain including the shift correction was dead work. And on
the branch above 0.927734375, which dominates a 9 A cutoff, the existing `erf`
forms `1 - exp(r)` through `expm1`; `erfc` is `exp(r)` from the same
polynomial, so the complementary form removes one transcendental round trip and
a cancelling subtraction.

Two interleaved `C1 -> A1 -> A2 -> C2` gates at 750 steps confirm the stack end
to end:

| Round | Control medians | Candidate medians | Complete wall |
| --- | --- | --- | ---: |
| 1 | 2.5100, 2.3713 ms/step | 2.0707, 2.0126 ms/step | 16.3% lower |
| 2 | 2.4748, 2.2951 ms/step | 2.0642, 2.0387 ms/step | 14.0% lower |

Both rounds and all four paired directions agree, and the gap exceeds the
control spread, so this clears the noise floor above. Pooled medians are
2.4231 against 2.0515 ms/step, 15.3% lower, or 141 to 168 ns/day. All eight
runs stayed finite and passed their constraint and memory gates. Direct-force
agreement against the previous kernel was `3.95e-5 kJ/mol/A` RMS and
`3.05e-4 kJ/mol/A` maximum against a `671.76 kJ/mol/A` maximum reference,
inside the established tile route envelope.

A 1.566x kernel speedup producing a 15.3% complete-wall gain places the direct
route at about 40% of the ordinary step, which is the most reliable estimate of
that share in this document because it comes from a complete-wall A/B rather
than an instrumented boundary.

### The rest of the step, measured

Every route below was timed at one call and at nine calls inside a single
evaluation over distinct inputs. The marginal cost is the difference divided by
eight, which cancels the fixed dispatch and synchronization cost instead of
assuming a value for it. The nonbonded parts reconcile with a directly measured
nonbonded total to within 1%.

| Block | Marginal cost | Share of a 2.05 ms step |
| --- | ---: | ---: |
| Spatial tile direct force | 0.581 ms | 28% |
| Neighbor rebuild, amortized | 0.49 ms | 24% |
| SETTLE and SHAKE projections | 0.165 ms | 8% |
| Prepared reciprocal PME | 0.144 ms | 7% |
| Sparse corrections and fused bonded | 0.052 ms | 2.5% |
| BAOAB, Langevin noise, centre-of-mass removal | 0.06 ms | 3% |

The residual is inter-kernel serialization and the single per-step host round
trip, not removable work: the marginal measurements let independent calls
pipeline, while a real step is a dependency chain.

Four directions were measured and closed in that pass, and they are recorded
here so they are not rederived:

- **Integration arithmetic is not a target.** The full BAOAB chain is 0.0339
  ms, centre-of-mass removal 0.0242 ms, the thermostat with its noise 0.0225
  ms. Hoisting the constant mass reduction out of centre-of-mass removal, which
  looks like obvious waste, saves 0.0025 ms, or 0.12% of a step.
- **The constraint routes hold no waste of the direct-kernel class.** SHAKE
  runs 790 clusters at eight iterations, so its inner-loop divisions total
  about 32,000 per step; SETTLE is analytic.
- **Rebuild host time is algorithmic.** Of about 5.8 ms per rebuild,
  `_spatial_cell_pair_tasks` is 1.2 ms, nine prefix-scan readbacks that size the
  next allocation are 1.2 ms, and `cumsum` is 0.6 ms. The one true constant, the
  cell-lengths evaluation and readback at the top of every rebuild, is 0.3% of a
  step and would require caching inside `Cell`.
- **The former 117.6 MB exact-pair array was unused by the tile force route.**
  The first on-demand experiment still decoded it through force binding and MD
  diagnostics and was 1.8% slower. The retained design removes those eager
  consumers: tile-aware force terms declare whether they consume or ignore the
  tile schedule, and the non-pressure energy diagnostic uses the same tile
  membership directly. Exact pairs now remain absent unless an explicit pair
  API or an unsupported custom force requests them.

That redesign reduced the independent 23,558-atom 5DFR complete-wall median
from 0.124154 to 0.103989 seconds, a 16.24% improvement. Median rebuild time
fell 45.39%, estimated resident neighbor storage fell from 129.37 to 11.77 MB,
and Metal peak allocation fell from 962.3 to 102.8 MB. Four control and four
candidate runs all passed the science and route checks. A 94,232-atom JAC
transfer also passed all checks; its candidate was stable at a 0.303246-second
median and remained 14.42% faster even against the fastest control sample.
See
[`md-neighbor-roundtrip-verdict-m5max.md`](../benchmarks/md-neighbor-roundtrip-verdict-m5max.md)
for the complete protocol and the deliberately conservative interpretation of
the noisier JAC control.

One pure duplication was removed. `NeighborListManager.rebuild` rebuilt the tile
geometry through `dataclasses.replace` only to stamp a generation counter, and
because `NeighborTiles` is frozen that re-ran `__post_init__`, including its
force-group schedule reduction and a blocking evaluation. The generation is now
a `build_neighbor_list` parameter, so validation runs once per rebuild instead
of twice. That is worth about 0.3% of a step, below what a complete-wall gate
can resolve, so it is verified by counting validations per rebuild and by an
interleaved four-run gate that shows no regression.

The tile builder also has a known granularity limit at small search radius.
Cells are sized at one third of the search radius, so at 5.5 A skin a cell
holds about 11 atoms and an eight-atom block fits inside one cell, while at
1.0 A skin a cell holds about 3.6 atoms and blocks span several cells. Tile
count is consequently non-monotonic in skin, with a minimum at 2.0 A: 472,848
tiles at 1.0 A against 402,910 at 2.0 A. `DEFAULT_MLX_CELL_TILE_BLOCK_SIZE` and
`DEFAULT_MLX_SPATIAL_CELL_SUBDIVISION` are independent constants and become
inconsistent below about 3 A of skin.

## Closed Directions

Closed means do not repeat that implementation. It does not ban a future design
that removes the named cause. Product code and candidate-only tests for every
row below were removed; the rows exist so the work is not repeated.

Rows with a structural cause:

| Direction | Measured result | Cause | Reopening condition |
| --- | --- | --- | --- |
| Canonical-ID atom tiles derived from the compact pair list | 60.19% slower complete; peak 4.17 to 8.06 GB | 9,664,362 tiles carried 60,504,316 exact pairs through 618,519,168 padded lanes | A spatially native, occupancy-bounded design that quantifies scheduled lanes before implementation |
| Two-pass non-atomic right-block reducer | Complete wall 0.558220 to 0.734520 s | 264 MB temporary tile-force buffer; the direct advantage over pairs fell to 8.52% | Reduce global atomics without a proportional global temporary |
| Fused Metal neighbor-admission producer | 0.46% slower; paired directions disagreed | The direct graph fell from 34 primitives to 8, but the load-bearing admission sync remained and the custom reduction and output cost erased the saving | Eliminate or overlap a real consumer boundary, not the reduction chain alone |
| Two-step neighbor admission transaction | 2.29% slower over 750 steps | Fewer host decisions but the same displacement work; 42--44 replayed steps repeated integration and one force evaluation | Reduce displacement work, not decision count |
| Coalesced SETTLE/SHAKE sparse writes | 2.83x regression, 0.202904 to 0.574724 s | This system constrains nearly every atom, so added concatenation and scatter outweighed fewer full-position additions | -- |
| Constraint-aware compiled trajectory block | 1.87596 s against a 1.190142 s control | Rebuilds fell from nine to two but a much larger lazy graph was repeated | -- |
| Left-major rebuild row compaction | 6.60% slower, both directions regressive | The device descriptor, count, prefix, and compaction pipeline cost more than the removed argsort, permutation, and two gathers | Remove those added traversals and buffers with a unique-writer ordering proof |
| Stable linear force-group placement | 8.73% slower | Scanning each left-block row cost more than MLX `argsort` plus gathers | -- |
| Same-position finite certificate | 5.54% slower | Removed one reduction but added more host bookkeeping than it saved | -- |
| Second Metal stream | 6.6% slower | Speculative concurrency on an already device-bound step | -- |
| Smaller neighbor skins for 5DFR | Complete-wall optimum is the production 5.5 A | Rebuild cost rises faster than direct-force lanes fall; see [Where the time goes](#where-the-time-goes) | A rebuild whose fixed cost is small enough to change that balance |

Rows decided inside the noise floor. These are recorded as "no measurable
gain", not as established regressions:

| Direction | Measured result |
| --- | --- |
| Unified schedule-validation ownership | 0.017% faster; paired directions disagreed |
| Staged pre/post `mx.compile` reconstruction | 0.16% slower; primitives fell from 153 to 106 |
| Flattened and cached force-buffer summation | 0.33% slower |
| Deferred topology-mask materialization | 0.37% faster; the wait moved into first force evaluation |
| Omitting the disjoint SHAKE pre-force projection | 1.7% slower |
| Partial on-demand exact-pair decoding with eager downstream consumers | 1.8% slower |
| Ordinary-step graph pruning | 2.08% faster; only two `ExpandDims` and two `Reshape` nodes were removed |
| Fused tile compaction and pair emission | 2.64% slower |
| Prefix tails, fresh extraction | 3.10% slower |
| One-transfer prefix tails | At most about 4.2% after settling |
| Eight-tile same-left grouping | Isolated kernel 5.53% faster; complete wall 0.558220 to 0.559019 s |
| Admission intervals greater than one | No complete-wall gain |

The retained deferred exact-pair route does not repeat the closed partial
experiment. It removes the force-binding and diagnostic consumers that caused
the delayed decode to re-enter the recurring path. The small prefix-tail
inventory values are also transferred together, but capacity-sized tile
buffers were not added: historical capacity and fused-output candidates
regressed, while removing the unnecessary exact-pair allocation already
eliminated the dominant storage and synchronization cost.

Two directions were closed in isolation and later retained under composition,
because they alter the same ordinary constraint and integration graph: dense
constraint owner maps, measured alone at 2.35% with inconsistent paired gains,
and a handwritten fused BAOAB drift, measured alone at 0.5%. See [Cumulative
interaction stack](#cumulative-interaction-stack).

## Replica Batching For Small Systems

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
