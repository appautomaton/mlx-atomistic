# MD spatial-cell task reuse verdict on Apple M5 Max

Date: 2026-08-13

Status: retained. The production `mlx_cell_tiles` rebuild now avoids copying a
cached dense cell-pair template and submits the three independent force-schedule
prefix reductions at one synchronization boundary. The change adds no buffer,
dependency, fallback, or reference-engine runtime path.

## Bottleneck and candidate

A synchronized profile of the 94,232-atom JAC workload attributed about 23% of
750-step measured wall time to 22 neighbor rebuilds. A warm rebuild contained
eight `mx.eval` calls. The membership inventory and force-schedule prefix
boundaries contributed most of the device wait, while the host also rebuilt two
4,675,860-entry cell-pair index arrays even though the fixed simulation cell had
no empty task.

The retained candidate makes two independent changes:

- active-column, per-left-block column, and force-group prefixes are constructed
  before one `mx.eval` call because none depends on the compacted force columns;
- when every cell-pair task has at least one atom candidate,
  `_spatial_cell_pair_tasks` returns the immutable cached template directly
  instead of boolean-indexing and copying both index arrays.

Sparse occupancy still follows the existing filtering path. Candidate counts
are recomputed from the current cell occupancy on every rebuild, so cached
geometry cannot make atom counts stale.

## Rejected fixed-capacity route

A device-resident schedule-tail prototype allocated outputs from the preceding
rebuild and read only the final logical counts. Its isolated schedule microbench
was encouraging: a 125% capacity reduced the schedule-tail median by 36.83% on
5DFR and 26.22% on JAC. It did not survive production integration. Writing the
reserved tail and then exposing exact logical slices made warm JAC rebuilds
about 4% slower than the already coalesced exact-allocation path. Reusing the
previous exact count removed the extra capacity but remained slower in the
direct production comparison. The bounded route and its Metal write guards were
therefore reverted.

This is also the current answer to the custom C++ primitive gate: the measured
boundary can be reduced safely in the existing MLX/Metal path, while the native
extension design has not yet demonstrated enough additional value to justify a
new build surface.

## Isolated rebuild result

Each arm ran in two independent processes. The first rebuild in each process was
discarded, leaving 14 warm samples per arm and workload.

| Workload | Control median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| 5DFR, 23,558 atoms | 17.556 ms | 16.850 ms | 4.02% faster |
| JAC, 94,232 atoms | 61.870 ms | 59.443 ms | 3.92% faster |

The arrays contained the same tile and active-column counts in both arms:
1,731,081 tiles and 5,068,119 columns for 5DFR; 7,182,592 tiles and 20,932,623
columns for JAC.

## Complete-wall gates

The common protocol used 10 warmups, fixed-cell NVT, a 0.004 ps timestep, 300 K,
seed 17, a 5.5 A neighbor skin, per-step displacement checks, and the
`mlx_cell_tiles` backend. Each row is the median of two control and two candidate
processes in `C1 -> A1 -> A2 -> C2` order.

| Workload | Steps | Control wall | Candidate wall | Change | Rebuild result |
| --- | ---: | ---: | ---: | ---: | --- |
| 5DFR | 75 | 135.147 ms | 129.123 ms | 4.46% faster | 9.21% lower median cumulative rebuild wall; 2 rebuilds in every run |
| JAC | 75 | 375.984 ms | 370.814 ms | 1.38% faster | 4.28% lower median cumulative rebuild wall; 2 rebuilds in every run |
| 5DFR | 750 | 1.454857 s | 1.425379 s | 2.03% faster | candidate completed 23 rebuilds; controls completed 22 and 23 |
| JAC | 750 | 5.723437 s | 5.705528 s | 0.31% faster | 3.92% lower median cumulative rebuild wall; 22 rebuilds in every run |

Both independent 750-step JAC candidates were faster than their corresponding
outer controls. The complete-wall JAC improvement is small, but the target
stage improved consistently, rebuild counts matched, 5DFR transferred, and the
implementation has no persistent-memory cost.

## Correctness and evidence

The focused Metal kernel suite passed all 27 tests. The cell-pair-template unit
test now covers both sparse filtering and dense identity reuse, including fresh
candidate counts after occupancy changes. Ruff and API-document generation also
passed. Every performance run completed with finite state, no neighbor fallback,
and lazy nonbonded-pair storage.

Raw reports are local and gitignored under:

- `results/md-cell-task-reuse/5dfr/`
- `results/md-cell-task-reuse/jac/`
- `results/md-cell-task-reuse/5dfr-750/`
- `results/md-cell-task-reuse/jac-750/`
- `results/md-bounded-tile-schedule/`

The 75-step command shape was:

```text
uv run --no-sync python -m mlx_atomistic.benchmarks.charged_pme runtime \
  --prepared results/dhfr-npt-closure/prepared \
  --warmups 10 --steps 75 --dt-ps 0.004 --temperature-k 300 --seed 17 \
  --neighbor-skin 5.5 --neighbor-check-interval 1 \
  --sample-interval 75 --diagnostic-interval 75 \
  --neighbor-backend mlx_cell_tiles \
  --out results/md-cell-task-reuse/5dfr/candidate-1.json
```

JAC changes the prepared path to
`results/larger-system-scaling/jac-2x2x1-modern/prepared`. The sustained gate
changes `steps`, `sample-interval`, and `diagnostic-interval` to 750. Controls
used detached commit `a4ac6ef` through an explicit `PYTHONPATH`.
