# MD neighbor round-trip and deferred-pair verdict on Apple M5 Max

Date: 2026-08-11

The decision is `retained`. The production `mlx_cell_tiles` route no longer
builds a compact exact-pair array on every neighbor rebuild when all active
force terms can consume or ignore the tile schedule. The complete 5DFR runtime
became faster, the result transferred to the larger JAC workload, and all
science, route, and bounded-memory checks passed. OpenMM and LAMMPS remain
reference surfaces and are not on either measured MLX runtime path.

## What changed

The previous tile builder held two equivalent representations of the same
neighbor generation:

1. occupancy-bounded 8x8 tile membership used by the direct-force kernel; and
2. a compact `int32` exact-pair array used as a diagnostic oracle.

The second representation occupied about 117.6 MB for 5DFR and about 484 MB
for JAC. An earlier on-demand prototype was 1.8% slower because
`_PreparedForcePipeline.bind()` and molecular-dynamics diagnostics still
requested the array on every generation, merely moving its decode later.

The retained design changes ownership instead of only changing timing:

- `NeighborList` accepts a deferred diagnostic-pair representation and caches
  the exact pairs only after an explicit consumer requests them.
- Tile-aware force terms declare `consume` or `ignore`; unknown custom terms
  retain the compact-pair fallback.
- Non-pressure energy and force diagnostics consume the exact tile membership
  directly through a Metal energy-and-force variant. Pressure diagnostics
  retain their existing pair path.
- Explicit access to `pairs` still materializes a sorted exact-pair array and
  is parity-tested against the pair builder.
- The tile and exact-pair prefix tails are packed into one small inventory
  transfer. Capacity-sized tile buffers were deliberately not added because
  earlier capacity and fused-output candidates regressed, while eliminating
  the recurring exact-pair allocation removed the dominant cost.

## Source states and protocol

The control was detached at commit
`64b5f4ff10f11a91bd52f373424cb5cc33d25057`. It had its own Python 3.13.12
virtual environment and editable install. The candidate was the working tree
that implements the deferred-pair route. This separation prevents one `uv`
sync from redirecting both processes to the same checkout.

Both states used MLX 0.31.2 on Apple M5 Max with macOS 26.5.2. Each sample was
an independent process with a 40 GB physical-memory ceiling and a 90-second
limit for 5DFR or 180-second limit for JAC. The common simulation protocol was
10 warmups, 75 measured fixed-cell NVT steps, 0.004 ps timestep, 300 K, seed
17, 9 A cutoff, 5.5 A neighbor skin, one-step neighbor checks, and the
`mlx_cell_tiles` backend.

The 5DFR matrix used the 23,558-atom fixture under
`results/dhfr-npt-closure/prepared` and ran in
`C1 -> A1 -> A2 -> C2 -> C3 -> A3 -> A4 -> C4` order. The JAC transfer used
the performance-admissible 94,232-atom fixture under
`results/larger-system-scaling/jac-2x2x1-modern/prepared` and the same order.
The older JAC fixture without molecule IDs was rejected by its existing generic
constraint-route admission check and was not included in the result.

## Complete-wall result

| Workload and metric | Control | Candidate | Change |
| --- | ---: | ---: | ---: |
| 5DFR measured wall, median | 0.124154 s | 0.103989 s | -16.24% |
| 5DFR measured rebuild, median | 16.121 ms | 8.803 ms | -45.39% |
| 5DFR measured neighbor update, median | 84.509 ms | 72.671 ms | -14.01% |
| JAC measured wall, median | 0.774136 s | 0.303246 s | -60.83% |
| JAC measured rebuild, median | 115.096 ms | 36.522 ms | -68.27% |
| JAC measured neighbor update, median | 603.335 ms | 255.914 ms | -57.58% |

The four 5DFR control walls were 0.188689, 0.116039, 0.132025, and 0.116283
seconds. The candidate walls were 0.103951, 0.153590, 0.104028, and 0.103619
seconds. One candidate sample was a timing outlier, but the median agrees with
the earlier independent interleaved matrix, where the candidate median was
0.104599 seconds against a 0.123245-second control.

The JAC control was noisier: 0.825167, 0.354330, 0.817885, and 0.730387
seconds. Candidate samples were stable at 0.302208, 0.306800, 0.300934, and
0.304284 seconds. The 60.83% median improvement should therefore not be read as
a clock-independent hardware ratio. A conservative comparison between the
candidate median and the fastest control sample is still a 14.42% improvement,
which establishes transfer without relying on the slow control samples.

## Memory result

| Workload and metric | Control median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| 5DFR Metal peak allocation | 962.3 MB | 102.8 MB | -89.31% |
| 5DFR estimated resident neighbor storage | 129.37 MB | 11.77 MB | -90.90% |
| JAC Metal peak allocation | 3.988 GB | 0.441 GB | -88.95% |
| JAC estimated resident neighbor storage | 534.33 MB | 50.28 MB | -90.59% |

The candidate reports zero estimated compact-pair bytes throughout the tile
runtime. A dedicated runtime check now also requires diagnostic pairs to remain
unmaterialized for a passing tile benchmark. Explicit pair access remains
available and is tested separately.

## Correctness and compatibility

All eight formal 5DFR samples and all eight formal JAC samples completed 75
steps, remained finite, reused one prepared Particle Mesh Ewald (PME) plan,
selected the expected tile representation, avoided neighbor fallback, retained
lazy topology, and passed the constraint-route inventory. The candidate also
passed:

- the default CPU pytest suite;
- all GPU-marked tests, 40 passed and 1 skipped;
- the focused Metal direct-kernel suite, 27 passed;
- explicit tile-to-pair decoding parity;
- tile versus compact-pair energy, component-energy, and complete-force parity;
- custom force fallback and pair-oriented compatibility tests.

The CPU result remains the continuous-integration reference. The Metal tests
are ad-hoc optimization evidence, consistent with the repository test policy.

## Reproducer and raw evidence

The candidate command shape was:

```text
uv run --no-sync python scripts/run_bounded_process.py \
  --max-bytes 40000000000 --timeout-seconds 90 \
  --trace-out results/md-neighbor-roundtrip-phase4/final-independent/candidate-1-memory.json \
  -- python -m mlx_atomistic.benchmarks.charged_pme runtime \
  --prepared results/dhfr-npt-closure/prepared \
  --warmups 10 --steps 75 --dt-ps 0.004 --temperature-k 300 --seed 17 \
  --neighbor-skin 5.5 --neighbor-check-interval 1 \
  --sample-interval 75 --diagnostic-interval 75 \
  --neighbor-backend mlx_cell_tiles \
  --out results/md-neighbor-roundtrip-phase4/final-independent/candidate-1.json
```

The JAC command changes the prepared path to
`results/larger-system-scaling/jac-2x2x1-modern/prepared`, the output directory
to `jac-modern-transfer`, and the timeout to 180 seconds. Control commands use
the same arguments from the detached baseline worktree.

Raw reports and memory traces are local and gitignored under
`results/md-neighbor-roundtrip-phase4/`. The source decision is to retain the
deferred exact-pair route and keep capacity allocation closed until a future
design removes a measured boundary without reintroducing proportional buffers.
