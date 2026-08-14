# Device-Resident Neighbor Inventory on Apple M5 Max

Date: 2026-08-14

## Decision

Retain device-resident cell, block, and task inventory for the Metal spatial
tile neighbor builder. The change raises median complete-trajectory throughput
by 4.29% on JAC and 4.00% on GPCRmd while preserving exact neighbor membership
and force results. A longer 5DFR follow-up confirmed non-regression in both
paired directions, but its power-state-sensitive magnitude is not used as a
retention claim.

## Root cause

The previous Metal builder assigned and sorted atoms by cell on the device,
then materialized every cell count on the host. NumPy used those dynamic counts
to build cell starts, padded eight-atom block rows, occupied cell-pair tasks,
candidate counts, and output offsets. The device then consumed the resulting
arrays. On the 94,232-atom JAC case, this host inventory stage took a median
15.081 ms per synchronized rebuild.

The retained route keeps the dynamic work on the device:

- cumulative cell counts produce cell and block starts;
- the cached axis-aligned bounding-box task template supplies immutable cell
  pairs;
- device expressions compute active task counts and prefix offsets;
- a Metal kernel scatters sorted atoms directly into eight-atom build blocks;
- the host reads only the final coarse-block, candidate-tile, raw-candidate,
  and occupied-cell counts plus one overflow guard needed by the existing sized
  outputs and reports.

Atom-block storage uses the static upper bound
`(atom_count + 7 * cell_count) // 8`. Unused rows remain `-1` and are never
referenced by candidate tiles. This removes an extra sizing readback at the
cost of bounded temporary capacity. Exact 4-by-4 tile masks, force-column
ordering, topology binding, and the recurring direct-force kernel are
unchanged.

## Synchronized rebuild profile

The JAC control was commit `a9da192`. Each arm ran 10 warmup and 750 measured
steps with synchronized rebuild-stage profiling. Both arms recorded 21
rebuilds, and the table uses per-rebuild medians.

| Metric | Control | Device inventory | Reduction |
| --- | ---: | ---: | ---: |
| Inventory stage | 15.081 ms | 3.235 ms | 78.55% |
| Complete profiled rebuild | 64.042 ms | 52.118 ms | 18.62% |

Raw profiles are
`results/md-suite/device-inventory/jac-{control,candidate}-profile.json`.

## Interleaved complete-trajectory A/B

Each process ran 10 warmup and 750 measured fixed-cell NVT steps with seed 17,
a 4 fs timestep, 9 A cutoff, 5.5 A skin, Metal spatial tiles, and boundary-only
sampling and diagnostics. The order was control 1, candidate 1, candidate 2,
control 2, control 3, candidate 3. AC Low Power Mode was enabled, so absolute
rates must not be compared with non-low-power benchmark reports. Interleaving
keeps each local control/candidate decision in the same power policy.

| Workload | Control median | Candidate median | Step-wall reduction | Throughput increase |
| --- | ---: | ---: | ---: | ---: |
| JAC 4-cell | 7.7212 ms | 7.4032 ms | 4.12% | 4.29% |
| GPCRmd 729 | 9.2088 ms | 8.8546 ms | 3.85% | 4.00% |

Two of three complete-wall directions improved on each large workload. The
remaining JAC direction was 0.24% slower, inside the local noise floor. The
remaining GPCRmd candidate was 3.00% slower and performed one additional
rebuild. Every candidate still reduced total measured rebuild wall.

Median measured rebuild wall fell by 11.50% on JAC and 15.83% on GPCRmd. All 12
raw runs passed finite-state, constraint, memory, fixed-cell, lazy-topology,
neighbor-representation, and Particle Mesh Ewald (PME) plan-reuse gates.
Outputs are under `results/md-suite/device-inventory/final-ab/`.

Short 5DFR samples exposed a strong Low Power Mode ramp, so the non-regression
follow-up used 750 warmup and 3,000 measured steps in the balanced order
control 1, candidate 1, candidate 2, control 2. Both paired directions improved
complete wall. Candidate rebuild wall per rebuild fell by 27.3% and 25.9%, and
all four runs passed. Because that complete-wall magnitude exceeds the isolated
rebuild saving and still contains frequency movement, it is evidence against a
small-system regression, not a separate speed claim. Raw outputs are under
`results/md-suite/device-inventory/final-ab-long/`.

## Memory and correctness

The static block capacity and device prefix intermediates increased median MLX
peak memory by about 12 MiB on 5DFR, 59 MiB on JAC, and 60 MiB on GPCRmd. Peak
usage remained approximately 0.29, 1.21, and 1.22 GiB respectively, far below
the 40 GB production gate.

The Metal edge-case suite covers empty and single-atom systems, periodic
boundaries, aliased periodic tasks, dense cells, exact unique membership, and
repeatability. Tile-versus-compact direct-force parity, NBFIX, and synchronized
inventory reconciliation pass. The full 30-case Metal neighbor/nonbonded suite
and the 44-case CPU neighbor/force-runtime suite also pass.

This slice removes host ownership of dynamic task inventory. It does not make
the entire neighbor lifecycle conditional on the device: exact output sizes
still cross the host boundary before later sized Metal kernels launch.
