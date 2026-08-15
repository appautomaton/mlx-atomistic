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

## Rejected Directions

| Candidate | Why it was rejected | Historical commit |
| --- | --- | --- |
| Dense affine constraint inputs | The four-run 5DFR median was 9.62% slower and failed both adjacent-direction gates. Source was removed. | `64b5f4f` |
| Final force-kick fusion | Fixed-input projection was 3.5-4.7% faster, but sustained JAC results were neutral and order-sensitive; 750-step median was 0.22% slower. | `9efb38b` |
| Combined position/velocity constraint kernel | Isolated constraint calls improved about 17%, but equal-rebuild 750-step JAC wall regressed 0.94% for 633 changed lines. | `5b7bc13` |
| Shared-exponential `erfcx` approximation | Numerical parity passed and one direct-kernel sample improved, but the 750-step JAC candidate was 4.00% slower and rebuilt once more. | `91b7b0b` |
| Sparse corrections appended to Direct Space | The synchronized nonbonded route shrank, but the placement enlarged the hottest kernel and did not transfer to 5DFR. Corrections were later retained in the bonded buffer instead. | `a4ac6ef`, `3f281e0` |
| Reciprocal charge-spread launch variants | Two 25-lane layouts regressed the graph by 23-31% on 5DFR and 28% on JAC. The conservative eight-lane variant improved less than the declared gate. | `f0f818c` |
| Additional asynchronous velocity submission | Dependent command-buffer boundaries caused queue backpressure; some JAC samples grew from about 0.4 seconds to more than 4.5 seconds. | `7f7b878` |
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
| PME mesh-sorted charge spread | Exact mesh ordering changed the spread result only at float32 atomic-summation scale, but its timing did not transfer: batched JAC regressed 2.55%, and GPCRmd's 0.026 ms spread gain was smaller than its 28-step amortized 0.032 ms sort cost. Reusing the free Interaction32 order regressed spread by 1-4%. | uncommitted prototype; `results/md-suite/pme-spread-order-*-2026-08-15.json` |
| Partial-force or segmented atomic replacement | An arithmetic-preserving checksum kernel made Direct Space only 3.18% faster on 5DFR, 0.87% on JAC, and 1.24% on GPCRmd. Global force writes are therefore too small to repay a large partial buffer and a second reduction dispatch. | uncommitted diagnostic; `results/md-suite/direct-atomic-write-attribution-2026-08-15.json` |
| Right-lane point-to-AABB culling | The geometric test could reject 52-54% of scheduled ordinary pair lanes, but SIMD-divergent execution retained the loop reductions while adding bounds work. The kernel regressed 4.00% on 5DFR, 3.39% on JAC, and 2.26% on GPCRmd in 128-call blocks. | uncommitted prototype; `results/md-suite/direct-point-aabb-cull-screen-2026-08-15.json` |
| Active-right prepass and compaction | Compacting useful right entries made the force kernel 10.5% faster on 5DFR, but the 1.416 ms prepass cost exceeded the 0.253 ms saved from the force call. The complete two-pass candidate regressed 18.6%. | uncommitted prototype; `results/md-suite/direct-common-work-screen-2026-08-15.json` |
| Direct arithmetic-only specialization | Full, Lennard-Jones-only, and Coulomb-only kernels attributed 66-84% of Direct time to shared geometry, pair traversal, reduction, and writes. The isolated formulas were too small and too system-dependent to justify another specialization. | uncommitted diagnostic; `results/md-suite/direct-common-work-screen-2026-08-15.json` |
| Morton and axis-permuted atom order | Morton ordering enlarged scheduled lanes by 4.68-5.26% across 5DFR, JAC, and GPCRmd. The best linear-axis permutation changed scheduled lanes by only 0.02-0.21%, with a different winner per system, so the existing periodic cell order remains canonical. | uncommitted diagnostic; `results/md-suite/axis-ordering-screen-2026-08-15/` |
| Cross-tile SIMD reduction batching | Accumulating two or three right tiles before reducing left forces lowered the nominal reduction count but increased register pressure. Both timing directions regressed by 17.6-20.3% on 5DFR; a vector-form `simd_sum` was also neutral to slower. | uncommitted prototypes; `results/md-suite/direct-common-work-screen-2026-08-15.json` |

## Interaction32 Promotion

The atomic `fused_half32` force path is distinct from the rejected no-atomic
owner-computes design. It now has a Metal device builder, retained capacity,
generation ownership, NBFIX support, and an immutable topology snapshot.
Against production tiles, the topology-snapshot checkpoint improved complete
750-step walls by 19.39% on 5DFR, 22.33% on JAC, and 23.09% on GPCRmd.

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
