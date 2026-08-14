# MD neighbor-rebuild SIMD membership verdict on Apple M5 Max

Date: 2026-08-14

Status: retained. The production `mlx_cell_tiles` rebuild now evaluates each
coarse 8-by-8 candidate tile with one 32-lane SIMD group. The change preserves
the exact 4-by-4 `NeighborTiles` consumer contract and adds no dependency,
fallback, persistent buffer, C++ extension, or reference-engine runtime path.

## Boundary and candidate

A new opt-in structural profiler split the exact-shape Metal tile builder into
nine synchronized stages. Across 750 measured steps, candidate membership was
the largest stage on both canonical workloads: 26.2% of profiled 5DFR rebuild
wall and 30.3% of profiled JAC rebuild wall. Host task inventory was second at
about 25.3% on both workloads.

The previous membership kernel assigned 64 threads to one coarse tile. Every
thread loaded both atom indices and both positions, so the 16 unique atoms were
read repeatedly. The threads wrote 64 flags to threadgroup memory, crossed a
barrier, and left lane zero to assemble four masks serially.

The retained kernel uses one 32-lane SIMD group:

- its first 16 lanes load the eight left and eight right atoms once;
- `simd_shuffle` broadcasts atom indices and coordinates within the SIMD group;
- each lane evaluates one top-half and one bottom-half pair;
- `simd_sum` directly constructs the four non-overlapping mask words and exact
  member count, removing threadgroup scratch and the barrier.

The mask bit order, exact counts, tile compaction order, force columns, force
groups, and recurring Direct Space consumer are unchanged.

## Structural profile

Both control and candidate used 10 warmups, 750 measured fixed-cell NVT steps,
a 0.004 ps timestep, 300 K, seed 17, a 9 A cutoff, a 5.5 A skin, per-step
neighbor admission, and final-only diagnostics. The host was on AC power with
Low Power Mode disabled when the evidence was collected. MLX was 0.31.2 on the
Apple M5 Max Metal GPU.

| Workload | Rebuilds, control/candidate | Membership median | Membership total | Profiled rebuild wall | Complete profiled wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5DFR, 23,558 atoms | 22 / 22 | 2.4589 -> 1.9890 ms, 19.11% faster | 55.768 -> 45.842 ms, 17.80% faster | 212.558 -> 203.688 ms, 4.17% faster | 937.308 -> 931.073 ms, 0.67% faster |
| JAC, 94,232 atoms | 21 / 21 | 11.4183 -> 9.8867 ms, 13.41% faster | 244.061 -> 208.336 ms, 14.64% faster | 806.518 -> 786.207 ms, 2.52% faster | 3004.701 -> 2991.146 ms, 0.45% faster |

The complete-wall rows are directional structural-profile observations, not a
clean throughput claim: the diagnostic intentionally inserts completion
boundaries between rebuild stages. The retention claim is the transferable
membership reduction plus the equal-rebuild reduction in profiled rebuild
wall. Clean canonical suite runs remain the whole-runtime regression gate.

## Clean canonical suite gate

The uninstrumented local suite then ran one 75-step rehearsal and three
independent 750-step samples per workload. Both cases passed every runtime,
artifact-fingerprint, finite-state, neighbor-backend, and timing-spread check.

| Workload | Seconds/step samples | Median | Median throughput | Relative spread |
| --- | --- | ---: | ---: | ---: |
| 5DFR | 0.00121548; 0.00120764; 0.00118402 | 0.00120764 | 286.18 ns/day | 2.61% |
| JAC | 0.00393699; 0.00396847; 0.00398717 | 0.00396847 | 87.09 ns/day | 1.26% |

These rows establish stable current-candidate operation. They are not compared
against older suite JSON collected under a different power and frequency state.

## Correctness and profiler contract

The fixed-position Metal gate passed six periodic edge-case, tile-inventory,
and complete direct-force parity tests. A dedicated GPU integration test also
confirmed that every stage emitted exactly one sample, the stage totals
reconciled with complete build wall, and the reported tile and pair inventories
matched the constructed `NeighborTiles` object.

The profiler is disabled by default. `--neighbor-rebuild-profile` activates it
only around the measured simulation, adds synchronized builder boundaries, and
persists raw samples plus cell, tile, pair, column, and force-group inventory.
Its output fails the benchmark if no rebuild is recorded, the stage accounting
does not reconcile, or the recorded rebuild count differs from the manager's
measured count.

## Rejected host-allocation follow-up

A follow-up attempted to remove four large temporary `int64` arrays from the
host cell-task inventory by multiplying gathered counts in place. It did not
transfer. The JAC host-stage median improved by only 0.2%, while the 5DFR
host-stage median regressed by 6.2%. The apparent JAC complete improvement also
contained one fewer rebuild and was therefore inadmissible. The follow-up was
removed.

## Reproducer and evidence

The JAC candidate command was:

```text
uv run --no-sync python -m mlx_atomistic.benchmarks.charged_pme runtime \
  --prepared results/larger-system-scaling/jac-2x2x1-modern/prepared \
  --warmups 10 --steps 750 --dt-ps 0.004 --temperature-k 300 --seed 17 \
  --neighbor-skin 5.5 --neighbor-check-interval 1 \
  --sample-interval 750 --diagnostic-interval 750 \
  --neighbor-backend mlx_cell_tiles --neighbor-rebuild-profile \
  --out results/md-suite/rebuild-profile-jac-750-simd-membership.json
```

5DFR changes the prepared path to `results/dhfr-npt-closure/prepared`. Raw
control and candidate JSON is gitignored under `results/md-suite/`:

- `rebuild-profile-5dfr-750-8ad66a2.json`
- `rebuild-profile-5dfr-750-simd-membership.json`
- `rebuild-profile-jac-750-8ad66a2.json`
- `rebuild-profile-jac-750-simd-membership.json`
- `simd-membership-current.json`

## Next boundary

No single remaining stage dominates. The next investigation should separate
coarse candidate emission from distance membership, then examine the exact-tile
scatter/order stage without changing its recurring force-access locality. A
device-capacity builder, fixed-grid consumer, 32-atom interaction engine, or
C++ primitive remains outside the authorized boundary until an exact consumer
gate demonstrates complete-wall headroom.
