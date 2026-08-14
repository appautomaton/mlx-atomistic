# MD dense affine constraint verdict on Apple M5 Max

Date: 2026-08-11

This record closes Phase 3 of `.agent/steering/ROADMAP.md`. It evaluates the
private dense affine constraint input fusion introduced by commit `2b8df0f`
against the already-retained explicit position-correction, final-kick, and
constraint fallback. OpenMM remains a reference surface and is not on either
MLX runtime path.

## Source states and protocol

The candidate was commit `831e077520f590963f652ca67cc966df1f3709bf`.
Two temporary detached worktrees isolated the comparison. The candidate
worktree used that source unchanged. The control worktree changed only
`_dense_affine_constraint_candidate_enabled()` to return `False`, forcing the
existing fallback without adding a product or workload selector.

Every sample used the prepared 23,558-atom 5DFR workload under
`results/dhfr-npt-closure/prepared`, 10 warmups, 75 measured steps, a 0.004 ps
timestep, 300 K, seed 17, a 9 A prepared cutoff, a 5.5 A neighbor skin, and the
`mlx_cell_tiles` backend. Samples ran as independent processes in
`C1 -> A1 -> A2 -> C2` order with 40 GB and 90-second process-tree bounds.

The host was an Apple M5 Max with MLX 0.31.2 and macOS 26.5.2. Immediately
after the matrix, `pmset` reported AC Power and `lowpowermode 0`, while the
battery status still said discharging. The interleaved result is used directly;
no cross-power comparison or absolute hardware claim is made.

## Complete-wall result

| Sample | Route | Measured wall |
| --- | --- | ---: |
| C1 | affine disabled | 0.162681667 s |
| A1 | affine enabled | 0.119971625 s |
| A2 | affine enabled | 0.225087583 s |
| C2 | affine disabled | 0.152092334 s |

The control median was 0.157387000 seconds and the candidate median was
0.172529604 seconds. The affine candidate was therefore 9.6213% slower. The
first adjacent control-to-candidate effect favored the candidate by 26.2538%,
but the second regressed by 47.9940%. This fails the frozen rule that the
candidate median must be lower and both adjacent effects must be nonnegative.
The protocol does not permit repeating the matrix to search through timing
noise.

All four runs completed 75 steps, remained finite, selected
`mlx_cell_tiles`, performed two neighbor rebuilds, preserved all 22,290
constraint pairs, and passed every emitted science and route check. Bounded
process-tree peaks were 1.328 GB, 0.962 GB, 1.224 GB, and 0.959 GB respectively,
all below 40 GB. The negative verdict is therefore a performance decision, not
a correctness, route, timeout, or memory failure.

## Source decision

The decision is `removed`. The source reverses the affine additions from
`2b8df0f` while preserving the later `993f9b4` spatial direct-force work.
Affine admission, cached affine parameters, affine SETTLE/SHAKE inputs, the
dense affine apply kernel, and candidate-only tests are gone. The retained
specialized SETTLE, SHAKE, dense owner-map, asynchronous force submission, and
ordinary explicit integration/constraint paths remain.

The post-removal focused CPU suite passed 45 tests with 27 GPU skips. The full
focused Apple Metal suite passed 72 tests, and a bounded post-removal 5DFR
smoke passed its science and route checks. Because the primary 5DFR gate
rejected the candidate, the conditional JAC transfer and 750-step confirmation
were not run.

Raw reports and memory traces are local and gitignored under
`results/md-affine-verdict/2026-08-11/`. The next roadmap phase is the measured
neighbor-rebuild host readback and diagnostic exact-pair lifetime, not another
affine constraint variant.
