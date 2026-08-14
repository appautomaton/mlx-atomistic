# Current-Main Sustained MD and OpenMM Baseline on M5 Max

Date: 2026-08-14

## Decision

The current `main` branch establishes a new manifest-matched 750-step JAC
baseline at a two-run ratio of medians of `MLX/OpenMM = 2.7895x`. The MLX
samples were 5.8192 and 6.0093 seconds. The OpenMM/OpenCL single-precision
samples were 2.1054 and 2.1350 seconds. Both runtime comparisons passed every
manifest, protocol, PME, constraint, timing-boundary, and completion check.

This sustained result supersedes the old `5.2701x` ratio as the active
optimization baseline, but it does not rewrite that historical 75-step result.
The protocols have different measured durations, and the host remained in Low
Power Mode throughout this refresh.

An opt-in fixed-grid Direct Space consumer passed force parity and selected
5DFR timings, but regressed every measured JAC configuration by 12.43% to
21.54%. This fails the cross-workload consumer gate. The prototype was removed
before implementing the dependent capacity-aware neighbor builder. The result
does not reopen the rejected 32-atom interaction engine, owner-computes force
path, or fixed-capacity exact-slice prototype.

A constraint-aware speculative block also failed the sustained gate. A
synchronized fixed-generation preflight appeared to favor two-step blocks, but
it omitted the production loop's existing asynchronous force submission. In
the real runtime, the final reduced candidate regressed 5DFR by 10.73% in an
adjacent pair and made JAC 3.04 times slower. The complete prototype was
removed. The result closes Python-built multi-step lazy graphs as the next
production-residence mechanism.

## Provenance and protocol

| Field | Value |
| --- | --- |
| Project commit | `cdf5ce949626ed890cb4d8706f41a70f9f56fd7f` |
| MLX | 0.31.2, Metal GPU 0 |
| OpenMM | `8.5.1.dev-f7fa0c2` |
| OpenMM platform | OpenCL, Apple M5 Max, single precision |
| Host | Apple M5 Max, macOS 26.5.2, Python 3.13.12 |
| Power state | AC profile with Low Power Mode enabled |
| Ensemble | fixed-cell Langevin-middle NVT |
| Timestep | 0.004 ps |
| Temperature / friction | 300 K / 1 per ps |
| Warmup / measured steps | 10 / 750 |
| Neighbor policy | 9 A cutoff, 5.5 A skin, check every step |
| Output cadence | final-only sample and diagnostics |

The matched JAC runtime used 94,232 atoms, 89,160 constraints, a
128-by-128-by-64 order-five PME mesh, alpha 0.35 per angstrom, and an explicit
final-device-completion boundary in both engines. The random streams are
engine-native, so this is matched protocol throughput rather than trajectory
identity.

## Clean sustained results

| Workload | Samples, seconds | Median | Rebuilds | Median measured rebuild wall | Peak Metal |
| --- | --- | ---: | --- | ---: | ---: |
| 5DFR, 23,558 atoms | 1.4883; 1.8242; 1.8798; 1.5850 | 1.7046 s | 23; 23; 22; 22 | 0.3415 s | 288.1 MB |
| JAC, 94,232 atoms | 5.8192; 6.0093 | 5.9142 s | 22; 21 | 1.3285 s | 1.203 GB |
| OpenMM JAC | 2.1054; 2.1350 | 2.1202 s | internal | -- | -- |

The 5DFR spread is too large for a narrow point estimate. All four samples are
retained rather than selecting the fastest host performance state. The JAC
samples differ by 3.27%. The ratio of the JAC medians is:

```text
5.9142071881 / 2.1201796876 = 2.7894839398
```

Measured rebuild wall is 20.0% of the 5DFR median and 22.5% of the JAC median.
Unlike `NeighborListManager.update_wall_seconds`, these rebuild counters do not
assign all upstream GPU completion to neighbor construction. They therefore
select rebuild architecture more reliably than the inclusive update timer.

The formal production artifacts contain molecule identities. Both workloads
use the specialized SETTLE plus disjoint SHAKE-cluster route. The recently
retained small-component constraint kernel is not selected by these runs, so
its generic-constraint gain cannot be applied to this baseline.

## Synchronized structural profile

One 75-step synchronized profile per workload inserted completion barriers
between named routes. The shares diagnose dependency structure and must not be
read as clean asynchronous wall shares.

| Route family | 5DFR share | JAC share |
| --- | ---: | ---: |
| Dense SETTLE/SHAKE constraints and validation | 28.58% | 18.77% |
| Direct Space spatial tiles | 13.17% | 24.76% |
| Neighbor update and rebuild | 10.16% | 16.01% |
| Reciprocal PME | 8.44% | 12.54% |
| Integration and thermostat | 16.00% | 7.46% |
| PME and outer force aggregation | 8.43% | 5.71% |
| Sparse PME corrections | 5.07% | 4.01% |
| Fused bonded force | 5.01% | 3.48% |
| Neighbor force binding | 1.80% | 4.00% |

The profile does not identify one dominant force kernel. JAC Direct Space is
the largest named synchronized route, but the retained tile kernel is already
68.84% faster than the compact-pair control in the same profile. Constraint
work is collectively large, but two recent integration-adjacent candidates
failed sustained equal-rebuild JAC retention. Sustained rebuild wall is the
largest remaining independently measured component on both workloads.

## Rejected fixed-grid consumer gate

The proposed neighbor-engine design depended on a Direct Space consumer that
could accept capacity-sized arrays plus a device logical count. The smallest
independent test therefore added a grid-stride loop to the retained 4-by-4
force-only tile kernel and launched a fixed number of threadgroups. Production
dispatch remained unchanged during the experiment.

All tested configurations passed direct-force parity. On 5DFR, 8,192 and
16,384 threadgroups passed the 3% isolated-consumer gate, with measured
aggregate improvements of 1.06% and 4.15%. Smaller grids regressed by 8.34% to
27.7%, and the 32,768-threadgroup sample regressed by 4.26%.

The larger JAC workload rejected the design:

| Fixed threadgroups | JAC isolated consumer regression |
| ---: | ---: |
| 4,096 | 17.93% |
| 8,192 | 15.52% |
| 16,384 | 21.54% |
| 32,768 | 12.43% |

The force root-mean-square differences were approximately `3.2e-5` to
`4.2e-5 kJ/mol/A`, and maximum differences were `4.27e-4` to
`4.88e-4 kJ/mol/A`. Correctness therefore passed while the required consumer
performance failed decisively.

The capacity-aware builder was not implemented because it had no admissible
consumer. The experimental kernel, API switch, benchmark script, and test
extension were removed. Raw measurements remain under the gitignored
`fixed-grid/` result directory.

## Rejected unified constraint route

The synchronized profile assigns 18.77% of JAC and 28.58% of 5DFR structural
time to constraints and their validation. Production currently uses separate
specialized SETTLE and SHAKE routes, while the retained small-component Metal
kernel had previously only been measured against the generic MLX constraint
control. An artifact-matched iteration sweep compared that existing kernel
directly with production SETTLE/SHAKE.

The candidate lost the complete isolated constraint pipeline at every tested
iteration count:

| Workload | Candidate iterations | Pipeline regression |
| --- | --- | ---: |
| 5DFR | 4, 8, 12, 20 | 15.09% to 21.63% |
| JAC | 4, 8, 12, 20 | 24.63% to 51.14% |

The residual non-water pre-force projection alone regressed by 47.44% on 5DFR
and 96.63% on JAC. At 20 iterations, the 5DFR candidate still had a
`1.70e-3 A` maximum position residual and a `4.06e-1 A/ps` maximum constrained
velocity residual, versus production values of `1.37e-5 A` and
`2.58e-5 A/ps`. JAC converged more closely, but its 20-iteration pipeline was
still 32.06% slower.

The small-component route remains valuable for eligible generic
`DistanceConstraints`; it is not a replacement for the specialized production
SETTLE/SHAKE route. No production code changed. Raw measurements remain under
the gitignored `constraint-routes/` result directory.

## Rejected atomic grouped-column route

Boundary attribution measured the global sort over 1,731,081 retained 5DFR
tiles at 1.80 ms, or 13.21% of an instrumented warm rebuild. The corresponding
7,182,592-tile JAC sort took 4.97 ms, or 11.60%. Reconstructing a validated
`NeighborTiles` object after the builder's final completion boundary added
another 3.3% to 4.0%.

An exact-shape counting-scatter prototype removed the sort and grouped packed
force-column descriptors into the same left-block ranges with atomic offsets.
Inventory counts matched exactly, and direct-force differences remained inside
the existing float32 atomic envelope:

| Workload | Rebuild result | Direct Space result | Force RMS / maximum delta |
| --- | ---: | ---: | ---: |
| 5DFR | 10.55% faster | 5.50% slower | `3.26e-5` / `3.36e-4` |
| JAC | 15.92% faster | 21.15% slower | `3.21e-5` / `4.58e-4` |

Atomic allocation destroyed the deterministic spatial order inside each left
block, making recurring right-side reads less coherent. The unsorted
inventories contained 947,271 and 3,966,532 contiguous left-block runs, only
1.83 and 1.81 tiles per run. Sorting runs would therefore retain 55% of the
original sort inventory while adding more prefix and scatter work. The design
was closed rather than carrying a slower consumer or a marginal segmented-sort
successor. All prototype source was removed. Raw results remain under the
gitignored `rebuild-boundaries/` and `grouped-columns/` directories.

## Rejected constrained device-block gate

The matched OpenMM comparison and source study point to execution residence,
not another isolated arithmetic approximation. The current constrained
production loop already has project-owned SETTLE, SHAKE, Direct Space, bonded,
integration, and PME Metal kernels, but it returns to a host neighbor admission
boundary every step. A constraint-aware prototype chained those existing
kernels for two, four, or eight steps against one prepared tile generation. It
tracked the maximum displacement and replayed exact per-step neighbor admission
when a proposed block crossed the Verlet threshold. It also preserved
per-step center-of-mass removal.

The synchronized fixed-generation preflight was positive:

| Workload | 2-step | 4-step | 8-step |
| --- | ---: | ---: | ---: |
| 5DFR | 25.18% faster | 27.65% faster | 27.28% faster |
| JAC | 8.93% faster | 7.85% faster | 2.41% faster |

Every preflight block remained below the 2.75 A rebuild threshold. The largest
measured displacement was 1.112 A on 5DFR and 0.709 A on JAC. Constraint errors
remained in the production float32 tolerance envelope.

That preflight was not a valid production admission test. Its control forced a
complete synchronization after every step, while the retained constrained loop
already calls `mx.async_eval` for ordinary prepared force submissions. The
first production prototype also added redundant whole-state finite reductions;
removing those reductions fixed that local mistake but did not rescue the
architecture:

| Workload | Per-step control | Reduced 2-step candidate | Result |
| --- | ---: | ---: | ---: |
| 5DFR, adjacent pair | 1.4808 s | 1.6397 s | 10.73% slower |
| JAC, adjacent pair | 7.0427 s | 21.4391 s | 3.04x slower |

JAC's candidate spent 17.93 seconds in the inclusive block-admission path,
versus 5.63 seconds in the control's inclusive neighbor updates. Building two
complete constrained force graphs in Python increased graph and device-memory
pressure and defeated the existing asynchronous pipeline. The rebuild counts
remained comparable at 22 and 21, so rebuild frequency does not explain the
regression.

The production code, command-line switch, and GPU tests were removed. Raw
preflight and sustained results remain under the gitignored
`device-block-preflight/` and `device-block-production/` directories.

## Rejected pre-force SHAKE deferral

Launch-level attribution split the retained dense constraint route into
specialized SETTLE deltas, disjoint SHAKE deltas, and the dense owner-map apply.
An interleaved unique-input JAC profile measured median synchronized walls of
1.845 ms for the position pipeline, 1.045 ms for the pre-force SHAKE velocity
pipeline, and 1.115 ms for the final SETTLE/SHAKE velocity pipeline. Frequency
variation remained large, so these values describe launch structure rather
than clean per-step wall.

The pre-force SHAKE projection is algebraically eligible for deferral: force
evaluation depends on positions, not velocities, and the final velocity
projection uses the same constrained positions. A monkey-patched preflight
therefore omitted only the dense-route pre-force projection and retained the
final projection. After 20 steps, candidate-versus-control deltas stayed inside
the independent control-versus-control atomic-scatter envelope on both formal
artifacts:

| Workload | Candidate position RMS / max | Control-repeat position RMS / max | Candidate constraint error |
| --- | ---: | ---: | ---: |
| 5DFR | `4.34e-5` / `5.34e-4 A` | `4.34e-5` / `5.72e-4 A` | `1.36e-5 A` |
| JAC | `7.69e-5` / `9.92e-4 A` | `7.75e-5` / `1.05e-3 A` | `3.75e-5 A` |

Velocity and force deltas were likewise no larger than their control-repeat
envelopes. Reapplying the final velocity projector changed the candidate by at
most `1.13e-5 A/ps` on 5DFR and `2.21e-5 A/ps` on JAC.

The performance gate kept Low Power Mode enabled and used independent processes
in alternating control/candidate and candidate/control order. Six paired
750-step samples split evenly: the candidate won three pairs and lost three,
with paired changes ranging from 42.96% faster to 11.46% slower. The 1--2 second
measurement windows were too short relative to frequency variation, so they
were not used for admission.

The measurement window was then increased tenfold to 7,500 steps. All four
crossed pairs passed the runtime checks:

| Pair order | Control | Candidate | Candidate change |
| --- | ---: | ---: | ---: |
| control / candidate | 18.4603 s | 20.9525 s | 13.50% slower |
| candidate / control | 15.0394 s | 15.1190 s | 0.53% slower |
| control / candidate | 14.9652 s | 15.0016 s | 0.24% slower |
| candidate / control | 15.1201 s | 14.9678 s | 1.01% faster |

The median paired ratio was 0.39% slower, and only one of four long pairs was
positive. Neighbor rebuild counts differed by at most two per pair, so rebuild
work did not explain away the result. The deferral is therefore correct but
has no sustained performance benefit on 5DFR. It failed the first workload
gate and was not transferred to JAC. No production code or test changed. Raw
measurements remain under the gitignored `constraint-launches/crossed-low-power/`
directory.

## Retained analytical SHAKE velocity kernel

The disjoint SHAKE-cluster velocity kernel previously applied eight Jacobi-like
pairwise projection rounds to each one-to-three-peripheral star. The retained
kernel instead forms the star's symmetric mass-weighted constraint matrix and
solves the padded three-by-three system analytically inside one Metal thread.
Position SHAKE remains iterative because its constraints are nonlinear. The CPU
fallback is unchanged.

Interleaved unique-input kernel measurements showed a transferable reduction:

| Workload | Iterative velocity kernel | Analytical velocity kernel | Kernel change |
| --- | ---: | ---: | ---: |
| 5DFR | 107.927 us | 71.294 us | 33.94% faster |
| JAC | 73.872 us | 60.344 us | 18.31% faster |

On 5DFR the analytical result was bit-identical to the eight-round result. On
JAC the maximum velocity difference was `2.38e-7 A/ps`, while the maximum bond
velocity residual improved from `2.20e-7` to `1.19e-7 A/ps`. The CPU constraint
suite passed 18/18 tests and the targeted Metal physics-lock suite passed 4/4.

Four independent 7,500-step 5DFR crossed pairs were intentionally retained as
noise evidence rather than summarized as a precise whole-runtime speedup. The
candidate won two pairs and lost two; paired changes ranged from 12.84% faster
to 34.21% slower, with a paired-ratio median of 1.55% faster. Neighbor rebuild
counts and Low Power Mode frequency variation were larger than the expected
whole-runtime effect. A 7,500-step JAC stability run passed with finite state
and a `3.28e-5 A` maximum constraint error, below the `3.45e-5` to `3.58e-5 A`
errors in the earlier retained JAC baselines. The admission claim is therefore
the directly measured kernel reduction and improved projection residual, not a
stable end-to-end percentage.

Three narrower tuning candidates were rejected before this kernel was retained:

- Packing the three dense owner maps into one integer map had exact parity but
  did not transfer: a persisted rerun was 23.02% faster on 5DFR and 4.44% slower
  on JAC, with highly inconsistent paired samples.
- Changing only solver or dense-apply threadgroup widths produced roughly
  1% to 5% direct-stage changes, but four 5DFR long pairs split two wins and two
  losses. The production width remains 256.
- Reducing SHAKE from eight to six iterations met the sampled residual target,
  but the combined position/velocity microbenchmark was 4.5% slower on 5DFR
  while 7.7% faster on JAC. The retained analytical velocity solve avoids that
  non-transferable approximation and leaves position iterations unchanged.

Raw evidence is gitignored under `constraint-launches/`, including
`analytic-shake-velocity-*.json`, `analytic-shake-runtime/`,
`dense-apply-widths-*.json`, `solver-widths-*.json`, and
`shake-6-vs-8-*.json`.

## Next architecture gate

The next action is a synchronized re-profile of the retained analytical runtime.
The dense owner-map write is now measured at about 29 us in batched dispatch and
is not a large enough lever by itself. The next architecture candidate should
come from the refreshed Direct Space or neighbor-rebuild boundary, preserve the
specialized constraint solvers, and avoid another multi-step lazy graph. This
investigation does not initially require a new package, C++ extension, or
reference-engine dependency.

## Reproducer

The MLX command shape was:

```bash
uv run --no-sync python -m mlx_atomistic.benchmarks.charged_pme runtime \
  --prepared results/larger-system-scaling/jac-2x2x1-modern/prepared \
  --warmups 10 --steps 750 --dt-ps 0.004 --temperature-k 300 --seed 17 \
  --neighbor-skin 5.5 --neighbor-check-interval 1 \
  --sample-interval 750 --diagnostic-interval 750 \
  --neighbor-backend mlx_cell_tiles \
  --out results/current-main-md-baseline-2026-08-14/clean/jac-1.json
```

The synchronized profile changed the command to `charged_pme profile` with 75
measured steps. The matched reference command was:

```bash
uv run --with openmm python scripts/run_openmm_charged_pme_runtime.py \
  --mlx-prepared results/larger-system-scaling/jac-2x2x1-modern/prepared \
  --mlx-runtime results/current-main-md-baseline-2026-08-14/clean/jac-1.json \
  --amber-prmtop results/inputs/Amber20_Benchmark_Suite/PME/Topologies/JAC.prmtop \
  --amber-coordinates results/inputs/Amber20_Benchmark_Suite/PME/Coordinates/JAC.inpcrd \
  --replicas 2,2,1 --platform OpenCL --precision single \
  --warmups 10 --steps 750 --dt-ps 0.004 --temperature-k 300 \
  --friction-per-ps 1.0 --constraint-tolerance 1e-5 --seed 17 \
  --out results/current-main-md-baseline-2026-08-14/openmm/run-1
```

All raw JSON remains gitignored under
`results/current-main-md-baseline-2026-08-14/`.
