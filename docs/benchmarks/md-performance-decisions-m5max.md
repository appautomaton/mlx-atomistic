# MD Performance Decision Ledger on Apple M5 Max

Date: 2026-08-15

This ledger is the compact history of production Molecular Dynamics (MD)
performance work. It replaces one report per micro-optimization while
preserving the decision, transferable evidence, source commit, and reason a
direction is open or closed.

The runtime in every row is `mlx_atomistic` on MLX and Metal. OpenMM and LAMMPS
are reference surfaces only. Percentages from different rows are incremental
experiments against different historical controls and must not be multiplied
to estimate current performance.

Detailed retired reports remain available in Git history:

```bash
git show <commit>:docs/benchmarks/<historical-file>.md
```

## Retained Architecture

| Date | Change | End-to-end decision evidence | Commit |
| --- | --- | --- | --- |
| 2026-08-11 | Deferred exact diagnostic pairs | Removed the redundant per-rebuild pair array, about 118 MB on 5DFR and 484 MB on JAC; complete-wall benefit transferred to both systems. Pair materialization remains available on demand. | `20199e4` |
| 2026-08-11 | Hybrid search/execution tiles | Coarse 8-by-8 membership plus exact 4-by-4 force tiles improved 750-step wall by 6.84% on 5DFR and 3.31% on JAC. | `999a632` |
| 2026-08-11 | Active-column force schedule | Compacting only non-empty right columns improved complete wall by 6.10% on 5DFR and 1.94% on JAC. | `ff94d19` |
| 2026-08-13 | Packed column descriptor | Stored the four column membership bits in the existing `int32` descriptor, removing an indirect mask load without another schedule or dispatch. Cross-system gates passed. | `ff94d19` |
| 2026-08-12 | Reciprocal complex-grid force path | Removed a full-grid real projection/scale and redundant position wrapping. Two equal-rebuild JAC pairs averaged a 2.47% throughput increase; 5DFR was neutral. | `f0f818c` |
| 2026-08-15 | Reciprocal real half-spectrum force path | Replaced the recurring full complex transform with `rfftn`/`irfftn`, a last-axis half-spectrum influence view, and per-atom normalization in the Metal interpolation kernel. Same-process reciprocal-graph ABBA improved JAC by 55.96% and GPCRmd by 19.44%. Two 750-step canonical comparisons passed in both directions: 5DFR improved 2.91% and 23.29%, while JAC improved 28.09% and 19.86%. | `6d2a03d` |
| 2026-08-12 | Neighbor wait attribution | Established that 72.5% of the 5DFR and 88.5% of the JAC admission boundary was completion of earlier GPU work, not displacement arithmetic. The attribution remains part of profiler interpretation. | `7f7b878` |
| 2026-08-12 | Small constraint components | A component-owned Metal solver replaced roughly 60 full-array SHAKE/RATTLE layers per JAC step. The 750-step JAC median fell by 52.3%; unsupported graphs retain MLX fallback. | `d3b264b` |
| 2026-08-13 | Spatial cell-task reuse | Reused immutable dense task geometry and coalesced independent schedule prefixes. Warm rebuild medians improved about 4% on both 5DFR and JAC; sustained gains were 2.03% and 0.31%. | `4da45d5` |
| 2026-08-14 | Atom-local topology masks | Compressed sparse row exclusion and 1-4 lookup improved throughput by 6.01% on 5DFR, 7.96% on JAC, and 7.41% on GPCRmd without changing tile masks. | `a9da192` |
| 2026-08-14 | Device-resident Neighbor inventory | Kept cell, block, and task inventory on device. Throughput improved 4.29% on JAC and 4.00% on GPCRmd; longer 5DFR runs established non-regression. | `4625326` |
| 2026-08-14 | 32-lane membership kernel | Replaced 64-thread scratch/barrier assembly. Membership medians improved 19.11% on 5DFR and 13.41% on JAC while preserving exact 4-by-4 output. | `4122577` |
| 2026-08-14 | CHARMM Urey-Bradley and correction map | Moved Urey-Bradley into bond records and differentiated prepared correction map (CMAP) patches analytically in the fused bonded dispatch. GPCRmd throughput improved 26.49% against the pre-fusion control. | `ced64a4` |
| 2026-08-14 | Shared bonded/correction force buffer | Sparse Particle Mesh Ewald (PME) corrections joined the bonded buffer instead of allocating another full force array. Wall improved 1.77% on 5DFR, 0.90% on JAC, and 0.68% on GPCRmd. | `3f281e0` |
| 2026-08-14 | Adaptive left-grouped exact-tile scatter | Large inventories avoid the global exact-tile sort; small or pathologically occupied cells retain compact-and-sort. Final wall improved 4.08% on JAC and 2.36% on GPCRmd with 0.26% 5DFR noise. | `8899994` |
| 2026-08-15 | Bounded Interaction32 mode cache | The count kernel retains each right atom's two-bit half-membership mode, so ordinary scatter no longer repeats periodic geometry and 32-by-32 membership work. A 64 MiB admission limit preserves the original sparse fallback. Position-balanced 750-step walls improved 2.78-4.80% on 5DFR, 1.53-2.04% on JAC, and 2.57-3.03% on GPCRmd. | `0066b58` |
| 2026-08-15 | Topology-recovered rigid water | When molecule identifiers are absent, a fail-closed constraint-graph proof recovers only complete disjoint O-H-H water triangles. GPCRmd moves 19,944 waters from a 20-iteration small-component route to analytical SETTLE. Same-context 1,000-step pairs improved 5.31% and 32.37%; the retained claim is the conservative 5.1-5.3% complete-wall gain because independent processes changed Metal performance states. | `7b93553` |
| 2026-08-15 | Reused center-of-mass total mass | A run segment now computes the invariant total mass once instead of reducing the mass vector in every `CMMotionRemover` step. Independent-process 3,000-step ABBA pairs on 5DFR improved complete wall by 4.20% and 2.26%; the median fell from 1.3702 to 1.3258 ms/step, or 3.24%. Every arm passed the runtime checks, with 97-99 Neighbor rebuilds. | `6dea505` |
| 2026-08-17 | Packed Interaction32 mode collectives | Three mutually exclusive mode counts and ranks now share one six-bit-field SIMD collective; the mode cache is packed by two disjoint reductions instead of sixteen shuffles per lane. Low-power 750-step A/B/B/A walls improved 0.01% and 3.68% on 5DFR, 2.24% and 6.08% on JAC, and 1.99% and 3.13% on GPCRmd. Rebuild counts matched in every adjacent pair, while measured rebuild wall fell 5.7-6.3%, 22.4-29.3%, and 27.0-29.6%, respectively. | `97ea8d2` |
| 2026-08-19 | Inline active-right compaction | The fused-half Direct kernel now proves right atoms outside the current left-slice AABB, compacts surviving records inside each SIMD group, and transposes the ordinary pair loop so column reductions follow active work rather than padded width. Fixed-input Direct blocks improved 24.8% on 5DFR, 24.6% on JAC, 25.3% on GPCRmd, 23.1-28.3% on 30k/90k water, and 23.6% on ApoA1. Both 750-step A-B-A comparisons passed: throughput improved 12.78%/7.24% on 5DFR and 3.93%/8.86% on JAC. The six-system release run passed with at most 2.9% within-case spread. | `acd05c6` |
| 2026-08-20 | Speculative Neighbor admission overlap | Ordinary constrained Metal steps now submit the device displacement probe before the current-generation force graph. An unchanged Neighbor object reuses the queued force; an inner/outer transition or rebuild discards it and recomputes against the exact binding. Three-repeat 750-step throughput improved 37.92% on 5DFR, 8.77% on JAC, and 7.75% on GPCRmd. All 53 related Metal tests and the six-system release run passed. A fresh matched JAC run measured 1.944 ms/step for MLX and 1.227 ms/step for OpenMM, a 1.584 MLX/OpenMM wall ratio. | `f18c2e8` |
| 2026-08-23 | Constant-time Interaction32 special membership | A generation-owned bitset replaced the ordinary builder's binary search through sorted topology-bearing block codes. Fixed-input count kernels improved 62.48% on JAC and 63.19% on GPCRmd; complete rebuild medians improved 38.90% and 42.29%. Both adjacent three-repeat 750-step comparisons passed: throughput improved 2.59%/3.36% on 5DFR, 4.74%/6.80% on JAC, and 6.57%/7.02% on GPCRmd. | `360219a` |

The adaptive scatter retains its own
[boundary report](./md-left-grouped-neighbor-scatter-m5max.md) because it is the
current Neighbor producer and documents the active runtime thresholds.

## Retained Measurement Infrastructure

The canonical stage map separates a clean trajectory from an intrusive,
synchronized attribution plane. Its first three-system result identified
Direct Space and the Neighbor lifecycle as the shared large-system costs, with
an additional CHARMM bonded cost on GPCRmd. Profiles must not be reported as
clean throughput because their barriers remove normal lazy overlap.

The Neighbor rebuild profiler records validation, cell sorting, task inventory,
membership, exact scatter/order, force inventory, force-schedule scatter, and
result assembly. The adaptive path also records occupancy and the selected
scatter route.

These capabilities were introduced with `6dd355d`, `4122577`, and later
Neighbor commits. The runnable contract is documented in
[`md-suite.md`](./md-suite.md).

A 2026-08-16 Xcode GPU Replay of the current JAC Interaction32 force path
measured two identical 2,307,200-thread dispatches at 1.95 ms each. Xcode
reported 382 compiler instructions and an instruction mix dominated by
arithmetic logic unit work, with much smaller memory, control-flow, and
synchronization components. It did not report Maximum Theoretical Occupancy or
a usable live-register peak for this just-in-time MLX custom kernel. These are
compiler-profile boundaries, not runtime utilization percentages; they direct
new work toward useful-pair arithmetic rather than another memory or barrier
micro-optimization.

## Current Verified Snapshot

The 2026-08-24 post-bitset normal-power baseline at `a219226` passed three
independent 750-step repeats per system. Median throughput was 0.5694 ms/step
on 5DFR, 1.8169 ms/step on JAC, and 1.9479 ms/step on GPCRmd; every timing
spread was below 1%. The synchronized structural profile ranks Direct Space
first at 25.75% of JAC and 28.36% of GPCRmd wall, followed by constraints and
reciprocal PME. These fractions are not additive clean-wall timings.

A fresh manifest-matched JAC comparison used the middle MLX baseline repeat
and an OpenMM 8.5.1.dev-f7fa0c2 single-precision OpenCL run with the same ten
warmups, 750 measured 4 fs steps, PME mesh, alpha, cutoff, thermostat, and
fixed cell. MLX measured 1.8150 ms/step and OpenMM measured 1.2221 ms/step, a
1.485 MLX/OpenMM wall ratio. Raw evidence is under
`results/md-suite/post-bitset-current-2026-08-24/`.

## Rejected Directions

| Candidate | Why it was rejected | Historical commit |
| --- | --- | --- |
| Dense affine constraint inputs | The four-run 5DFR median was 9.62% slower and failed both adjacent-direction gates. Source was removed. | `64b5f4f` |
| Final force-kick fusion | Fixed-input projection was 3.5-4.7% faster, but sustained JAC results were neutral and order-sensitive; 750-step median was 0.22% slower. | `9efb38b` |
| Combined position/velocity constraint kernel | Isolated constraint calls improved about 17%, but equal-rebuild 750-step JAC wall regressed 0.94% for 633 changed lines. | `5b7bc13` |
| Sparse pre-force constraint write | The pre-force projection skips SETTLE, so a sparse SHAKE scatter appeared cheaper than the dense all-atom write. A synchronized JAC ABBA screen changed direction: candidate route averages were 0.251 and 0.305 ms/step against adjacent controls at 0.318 and 0.249 ms/step. The apparent aggregate difference was only 1.9%, so the dense production route remains. | uncommitted diagnostic; `results/md-suite/sparse-preforce-screen-2026-08-15/` |
| Shared-exponential `erfcx` approximation | Numerical parity passed and one direct-kernel sample improved, but the 750-step JAC candidate was 4.00% slower and rebuilt once more. | `91b7b0b` |
| Sparse corrections appended to Direct Space | The synchronized nonbonded route shrank, but the placement enlarged the hottest kernel and did not transfer to 5DFR. Corrections were later retained in the bonded buffer instead. | `a4ac6ef`, `3f281e0` |
| Reciprocal charge-spread launch variants | Two 25-lane layouts regressed the graph by 23-31% on 5DFR and 28% on JAC. The conservative eight-lane variant improved less than the declared gate. | `f0f818c` |
| Additional asynchronous velocity submission | Dependent command-buffer boundaries caused queue backpressure; some JAC samples grew from about 0.4 seconds to more than 4.5 seconds. | `7f7b878` |
| Explicit Direct/PME MLX streams | OpenMM overlaps reciprocal PME on a separate stream, but manually assigning Direct Space and PME to two MLX Metal streams regressed fixed-input JAC in both ABBA positions. Parallel calls grew from 2.434 to 4.015 ms and from 3.844 to 4.893 ms. The default lazy graph remains the scheduler. | uncommitted diagnostic; `results/md-suite/direct-pme-stream-overlap-2026-08-15/jac-94k-pme-64.json` |
| Fixed-capacity schedule tails | Isolated tail handling improved, but reserved output writes and exact logical slicing made production rebuilds slower. | `4da45d5` |
| Direct fixed-grid consumer variants | Every measured JAC configuration regressed by at least 12.43%, despite some 5DFR wins. | `c1caefc` |
| Unified iterative constraint route | Replacing specialized constraint ownership was 15-22% slower on 5DFR and 25-51% slower on JAC. | `c1caefc` |
| Atomic grouped-column ownership | Isolated grouped work improved, but the complete pipeline regressed 5.50% on 5DFR and 21.15% on JAC. | `c1caefc` |
| Constrained multi-step device block | A fixed-generation preflight improved, but real Neighbor admission made 5DFR 10.73% slower and JAC 3.04 times slower. | `c1caefc` |
| Pre-force SHAKE deferral | Correctness passed, but the long paired median was 0.39% slower and only one of four pairs improved. | `c1caefc` |
| Native 4-by-4 search | It reduced Direct Space padding but multiplied candidate search and prefix work. The retained hybrid uses 8-by-8 search and 4-by-4 execution. | `999a632` |
| No-atomic 32-atom owner-computes | It evaluated each pair twice, ran 2.2-2.5 times slower, and accumulated unacceptable float32 error along long owner lists. | design prototype; see `docs/metal-interaction-engine.md` |
| Direct-kernel layout variants | Changing ordinary group width or SIMD-group count did not transfer across systems. OpenMM-style lane rotation was about 80% slower on 5DFR, and single-periodic-copy was neutral to slower while increasing JAC force error. | uncommitted prototypes; local results dated 2026-08-15 |
| Zero-epsilon LJ branch | Skipping Lennard-Jones arithmetic when the mixed epsilon was zero improved synchronized Direct Space blocks on 5DFR and 30k TIP3P water, but regressed ApoA1 and all-positive-epsilon GPCRmd. JAC 94k changed sign across the two balanced directions. Static zero-epsilon fraction therefore did not provide a transferable specialization boundary, and the prototype was removed. | uncommitted prototype; `results/md-suite/direct-zero-lj-screen-2026-08-15/` |
| Special-half write fusion | Serial 32-atom special work helped GPCRmd but regressed 5DFR and JAC. A parallel paired-write variant preserved force parity but regressed all three systems by 0.8-3.6%. | uncommitted prototypes; local results dated 2026-08-15 |
| Per-cluster position-constraint early exit | Synchronized GPCRmd position attribution improved 5.9%, but a control/candidate/candidate/control clean gate was neutral to 0.30% slower. The branch did not transfer through the production lazy schedule, so the prototype was removed. | uncommitted prototype; local results dated 2026-08-15 |
| Spatially ordered Direct parameters | Reordering static charge and Lennard-Jones parameter reads with the spatial atom order preserved parity but slowed the JAC steady kernel by 8.06%; both timing directions regressed. Canonical parameter gathers remain cheaper than the altered register and occupancy footprint. | uncommitted prototype; `results/md-suite/direct-ordered-static-parameter-screen-2026-08-15/` |
| Transient packed Direct records | Packing positions and parameters into spatial order and scattering forces back preserved parity, but the complete fixed-input JAC path regressed in both ABBA positions by 81.5% and 7.8%; its median was 40.4% slower. Packing plus scatter cost about 0.50 ms while the interaction itself saved only about 0.086 ms. | existing diagnostic path; `results/md-suite/record-layout-{boundary,fixed}-2026-08-15/` |
| PME mesh-sorted charge spread | Exact mesh ordering changed the spread result only at float32 atomic-summation scale, but its timing did not transfer: batched JAC regressed 2.55%, and GPCRmd's 0.026 ms spread gain was smaller than its 28-step amortized 0.032 ms sort cost. Reusing the free Interaction32 order regressed spread by 1-4%. | uncommitted prototype; `results/md-suite/pme-spread-order-*-2026-08-15.json` |
| Partial-force or segmented atomic replacement | An arithmetic-preserving checksum kernel made Direct Space only 3.18% faster on 5DFR, 0.87% on JAC, and 1.24% on GPCRmd. Global force writes are therefore too small to repay a large partial buffer and a second reduction dispatch. | uncommitted diagnostic; `results/md-suite/direct-atomic-write-attribution-2026-08-15.json` |
| Right-lane point-to-AABB culling | The geometric test could reject 52-54% of scheduled ordinary pair lanes, but SIMD-divergent execution retained the loop reductions while adding bounds work. The kernel regressed 4.00% on 5DFR, 3.39% on JAC, and 2.26% on GPCRmd in 128-call blocks. | uncommitted prototype; `results/md-suite/direct-point-aabb-cull-screen-2026-08-15.json` |
| Conservative Verlet-shell group skip | At rebuild positions, 26.3-27.7% of ordinary groups were fully outside the 9 A force cutoff. Across the real 750-step Neighbor lifecycle, displacement expansion reduced their mean scheduled-lane share to 1.84% on JAC and 1.98% on GPCRmd, with medians of only 0.05-0.06%. The resulting whole-step ceiling was below about 0.7% before metadata and branch cost. | uncommitted diagnostic; `results/md-suite/current-direct-shell-*-2026-08-15.json` |
| Prepared Interaction32 argument arrays | Moving stable force parameters and dispatch counts into the generation-owned binding preserved Metal parity but did not reduce fixed-input JAC cost. In 256-call ABBA blocks, the candidate median was 3.386 ms/call against a 3.282 ms/call control, a 3.17% regression, and the two adjacent directions disagreed. MLX's transient constants remain cheaper. | uncommitted prototype; `results/md-suite/prepared-direct-args-fixed-2026-08-15/jac-94k-pme-256.json` |
| Combined SETTLE/SHAKE dispatch | One Metal dispatch produced bitwise-identical SETTLE and SHAKE deltas, but the saved launch did not transfer through the complete lazy graph. GPCRmd regressed from a 5.074 to 5.090 ms/step median, with opposite adjacent directions (+0.75% and -0.12%). JAC's apparent 6.18% median gain came entirely from a slow first control; its warm adjacent pair regressed by 0.17%. The prototype was removed. | uncommitted prototype; `results/md-suite/fused-constraint-family-2026-08-15/` |
| PME interpolation adds Direct force | Reading the Direct force in the order-five PME interpolation kernel removed a separate full-array addition and preserved force parity. It also introduced an earlier Direct-to-PME graph dependency. The 750-step median regressed 4.7% on 5DFR, JAC's warm adjacent gain was only 0.8%, and GPCRmd changed direction (+0.44% and -0.37%) with a neutral 0.04% median. The prototype was removed. | uncommitted prototype; `results/md-suite/pme-inline-aggregation-2026-08-15/` |
| Fast exponential in the shared Ewald helper | Replacing `metal::exp` with `metal::fast::exp` preserved finite forces with a maximum sampled absolute delta of 1.83e-4, about 3e-7 relative to the largest force. Fixed-schedule ABBA timing did not transfer across systems: JAC regressed in both adjacent directions, and GPCRmd had one large regression plus one drift-sized gain. The candidate never entered the product source or full trajectories. | uncommitted diagnostic; `results/md-suite/interaction32-fast-exp-2026-08-15/` |
| Fast reciprocal square root in Direct Space | `metal::fast::rsqrt` improved fixed-schedule calls by 2.3-6.5% across 5DFR, JAC, and GPCRmd with maximum sampled force deltas below `3.4e-4 kJ/(mol A)`. The 750-step JAC C/A/C trajectory changed direction and the candidate mean was 0.33% slower, so the exact reciprocal square root remains. | uncommitted diagnostic; `results/md-suite/direct-fast-rsqrt-*-2026-08-17/` |
| Forced two-level schedules for short generations | Lowering the 24-update admission threshold made GPCRmd build an inner schedule that could not amortize its compaction cost. Both adjacent 750-step directions regressed, including one large 42.6% loss, so the motion-based admission boundary remains unchanged. | uncommitted diagnostic; `results/md-suite/lane-aware-admission-screen-2026-08-17/` |
| Active-right prepass and compaction | Compacting useful right entries made the force kernel 10.5% faster on 5DFR, but the 1.416 ms prepass cost exceeded the 0.253 ms saved from the force call. The complete two-pass candidate regressed 18.6%. | uncommitted prototype; `results/md-suite/direct-common-work-screen-2026-08-15.json` |
| Direct arithmetic-only specialization | Full, Lennard-Jones-only, and Coulomb-only kernels attributed 66-84% of Direct time to shared geometry, pair traversal, reduction, and writes. The isolated formulas were too small and too system-dependent to justify another specialization. | uncommitted diagnostic; `results/md-suite/direct-common-work-screen-2026-08-15.json` |
| Morton and axis-permuted atom order | Morton ordering enlarged scheduled lanes by 4.68-5.26% across 5DFR, JAC, and GPCRmd. The best linear-axis permutation changed scheduled lanes by only 0.02-0.21%, with a different winner per system, so the existing periodic cell order remains canonical. | uncommitted diagnostic; `results/md-suite/axis-ordering-screen-2026-08-15/` |
| Cross-tile SIMD reduction batching | Accumulating two or three right tiles before reducing left forces lowered the nominal reduction count but increased register pressure. Both timing directions regressed by 17.6-20.3% on 5DFR; a vector-form `simd_sum` was also neutral to slower. | uncommitted prototypes; `results/md-suite/direct-common-work-screen-2026-08-15.json` |
| Shared-weight PME spread | Computing order-five B-spline weights once per atom removed four of five repeated weight evaluations, but the required threadgroup barrier did not transfer. Directional timings changed sign on all three systems, and the combined change stayed below 1% on 5DFR and GPCRmd. | uncommitted prototype; `results/md-suite/pme-shared-spread-screen-2026-08-15/` |
| Five-thread PME interpolation | Splitting each atom's 125 grid reads across five z-slice workers preserved force parity within `2.3e-5 kJ/(mol A)`, but two barriers and shared-memory traffic regressed 5DFR by 0.4-0.9%, JAC by 2.4-5.6%, and GPCRmd by 5.2-13.0%. | uncommitted prototype; `results/md-suite/pme-parallel-interpolation-screen-2026-08-15/` |
| Computed left-order indices | Replacing a 256-512 byte per-threadgroup index buffer with the equivalent block/slice/slot expression preserved Interaction32, device-built fused-half, and CHARMM NBFIX force parity. Fixed-schedule blocks improved 15.6% on 5DFR, 1.5-6.7% on JAC, and 0.9-2.1% on GPCRmd, but the gain did not transfer to the trajectory gate: an isolated 1,500-step JAC candidate improved only 0.46% against a contemporaneous control, below the required 3%, while 5DFR was effectively neutral against the fresh pre-change baseline. The prototype was removed. | uncommitted prototype; `results/md-suite/direct-left-index-*-2026-08-16*` |
| Compile-time Lennard-Jones switch variants | Specializing the Interaction32 fused-half kernel into switch and no-switch variants preserved ordinary and NBFIX force parity and improved fixed-input calls by 15.9% on 5DFR, 21.0% on JAC, and 1.8% on GPCRmd. The first stable 750-step control/candidate comparison transferred only 0.75% on 5DFR and 0.26% on JAC, below the 3% JAC retention gate. A following control shifted by 15.3% and 10.5%, respectively, so its apparent larger gain was ineligible. Four extra kernel variants were not retained. | unmerged prototype; `results/md-suite/switch-specialization-admission-2026-08-19/` |
| Local Direct arithmetic rewrites | Three ALU-motivated JAC screens all preserved force parity but failed the fixed-schedule gate. OpenMM-style Lennard-Jones factorization regressed 0.97%, a 4,097-entry screened-Coulomb table with a 16 KiB footprint and `9.8e-8` offline maximum factor error regressed 0.83%, and hoisting the invariant right-charge scale regressed 0.52%. The table traded arithmetic for irregular loads, while the compiler already optimized the simple invariant. All prototypes were removed before trajectory testing. | uncommitted prototypes; `results/md-suite/direct-{lj-factor,ewald-table,coulomb-charge-hoist}-screen-2026-08-16/` |
| Fast Ewald reciprocal | OpenMM's Apple OpenCL backend can select `native_recip`, but replacing the shared Ewald helper's float division with Metal `fast::divide` did not transfer. Same-process JAC A/B regressed the 12 A inner schedule by 0.96%; the 14.5 A outer aggregate improved 0.77%, but its two timing directions disagreed at -1.27% and +1.83%. Force parity passed, the prototype was removed, and no trajectory gate was run. | uncommitted diagnostic; `results/md-suite/direct-fast-recip-screen-2026-08-16/` |
| Bounded Ewald force polynomial | A degree-17 float32 polynomial removed the per-pair exponential and reciprocal while keeping the JAC maximum force delta at `3.43e-4 kJ/(mol A)`. The current fixed-input Interaction32 call regressed from 1.1128 to 1.1657 ms, or 4.75%, so the prototype was removed before trajectory testing. | uncommitted prototype; `results/md-suite/direct-bounded-ewald-polynomial-2026-08-24/` |
| Fused position/pre-force constraint pipeline | A generation-independent Metal path removed one dense full-atom write and one critical-path dispatch. Fixed-input JAC improved 11.64%, with bitwise-identical positions and a `1.19e-7` maximum velocity delta. The complete 750-step screen transferred only 2.76% on 5DFR and 1.00% on JAC, while GPCRmd regressed 1.55%. The roughly 450-line prototype was removed. | uncommitted prototype; `results/md-suite/fused-position-pre-force-screen-2026-08-24/` |
| Generation-bound compiled force graph | `mx.compile` successfully captured bonded, Direct, PME, correction, and aggregation custom-kernel work, but each new Neighbor binding owned a new compiled closure. A 750-step JAC screen regressed from the 1.8169 ms/step current baseline to 1.8920 ms/step, or 4.14%; trace cost did not amortize over the generation lifetime. Cross-generation dynamic state, not another Python binding wrapper, is the remaining compilation boundary. | uncommitted prototype; `results/md-suite/compiled-force-graph-screen-2026-08-24/` |

## Interaction32 Promotion

The atomic `fused_half32` force path is distinct from the rejected no-atomic
owner-computes design. It now has a Metal device builder, retained capacity,
generation ownership, NBFIX support, and an immutable topology snapshot.
Against production tiles, the topology-snapshot checkpoint improved complete
750-step walls by 19.39% on 5DFR, 22.33% on JAC, and 23.09% on GPCRmd.

The retained two-level Verlet schedule addresses the arithmetic-dominated
Direct kernel by removing complete scheduled lanes rather than adding a
divergent in-kernel test. At each eligible outer rebuild, two Metal passes
compact a 3.0 A-skin inner schedule that shares the outer generation's atom
order and special topology. The manager switches from inner to outer at 1.5 A
maximum displacement and rebuilds both at the unchanged 2.75 A outer boundary.
Three observed generations form a motion-based admission test: systems
averaging fewer than 24 updates per generation return to the single outer
schedule. Sustained 750-step JAC improved by 6.15% and 4.72% in the two balanced
directions; GPCRmd improved by 3.83% and 2.96%. ApoA1 improved 2.50%. The first
atom-count-only version regressed 90k TIP3P water by 8.2%, while adaptive
admission reduced the final difference to 0.33%. Direct-force parity and the
inner-to-outer lifecycle GPU tests passed. Implementation checkpoints are
`e4c631d` and `8c029c2`; raw local evidence is under
`results/md-suite/two-level-interaction32-2026-08-16/`.

The post-promotion profile kept Direct Space dominant at 19.06% on 5DFR,
29.17% on JAC, and 32.34% on GPCRmd, against 9.08--10.79% reciprocal PME.
Explicit admission diagnostics measured 34.76 updates per JAC generation,
21.67 for GPCRmd, and 20.00 for 90k TIP3P water. The 24-update threshold is
retained: lowering it cannot separate GPCRmd from the water regression with a
safe margin. The active schedule decision is reported as `adaptation_reason`;
`fallback_reason` remains reserved for an actual backend failure. Raw evidence
is under `results/md-suite/post-two-level-whole-step-profile-2026-08-16/`.

Current fixed-input component attribution confirms that ordinary traversal is
the remaining Direct target. On JAC, the 12 A inner schedule carried 72.1
million scheduled pair lanes and measured 2.530 ms/call, while the 14.5 A
outer schedule carried 108.3 million lanes and measured 3.170 ms/call. After
subtracting the same empty-output baseline, ordinary work measured 1.597 ms
against 0.455 ms of special work on the inner schedule, and 1.995 ms against
0.458 ms on the outer schedule. Special work is nearly invariant; the outer
growth belongs to ordinary pair traversal. The reusable benchmark attribution
is emitted as `component_timing`; raw evidence is under
`results/md-suite/direct-current-attribution-2026-08-16/`.

A fresh manifest-matched JAC 94,232-atom comparison at `97ea8d2` used 10
warmup steps and 750 measured 4 fs fixed-cell NVT steps while macOS Low Power
Mode was enabled. Current MLX with the packed-collective adaptive Interaction32
route measured 4.813 ms/step (71.81 ns/day); OpenMM 8.5.1.dev-f7fa0c2 with
single-precision OpenCL measured 2.699 ms/step (128.03 ns/day). Every workload
and runtime check passed, giving an MLX/OpenMM wall-time ratio of 1.783. The
comparison is matched protocol throughput, not trajectory identity, because
the engines use independent random-number implementations. Raw local evidence
is under `results/md-suite/current-matched-openmm-97ea8d2-2026-08-17/`.

The bounded mode cache in `0066b58` removes the remaining repeated ordinary
membership traversal for systems whose packed cache is at most 64 MiB. Larger
systems retain the exact sparse two-pass builder, so this promotion does not
turn the general runtime into a quadratic-memory design. Interaction32 remains
bounded to the measured Metal fixed-cell PME envelope. A six-system release
gate, cross-system force parity, and isolated low-power comparisons on 30k/90k
water and ApoA1 completed the default-promotion evidence. The unstable 47k JAC
midpoint remains on production tiles, while eligible 23k--31k and 90k--100k
runs now default to Interaction32. See
[`metal-interaction-engine.md`](../metal-interaction-engine.md) for the builder
contract and promotion evidence.

The subsequent whole-step profile moved attention away from builder internals.
GPCRmd's missing molecule identifiers had forced 59,832 known water atoms
through the general small-component constraint solver. Commit `7b93553`
recovers rigid waters only when the water mask, element identities, and exact
three-edge constraint components prove the partition. Any incomplete,
cross-linked, duplicated, or non-O-H-H component keeps the previous fallback.
After this change Direct Space remains the dominant GPCRmd stage; constraints
are approximately level with PME rather than the next independent target.

## Current Interpretation

Three patterns recur across the rejected work:

1. Reducing dispatch count does not guarantee a critical-path win under MLX
   lazy scheduling.
2. A kernel-local win can be erased by schedule construction, synchronization,
   force aggregation, or changed Neighbor rebuild timing.
3. A 5DFR-only improvement is not sufficient for a general large-system route.

New candidates should therefore start from a current whole-step profile and
must pass the position-balanced 5DFR/JAC canonical gate. GPCRmd is an
additional required coverage surface when its machine-state spread admits a
comparison; otherwise its whole-step claim remains explicitly blocked.

The real half-spectrum PME change passed the required 5DFR/JAC canonical gate.
GPCRmd force parity and its short same-process reciprocal graph also passed,
but sustained whole-step attribution is blocked under the current machine
state: repeated arms ranged from 11.92 to 27.63 ms/step and changed direction
across the balanced pairs. No GPCRmd whole-step speedup or regression is claimed
from that run. Raw local evidence is under
`results/md-suite/pme-real-fft-*-2026-08-15/`.
