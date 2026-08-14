# Whole-Step MD Stage Map on Apple M5 Max

Date: 2026-08-14

## Decision

The next optimization must target the complete Molecular Dynamics (MD) step,
not another isolated neighbor-membership microkernel. A new canonical stage-map
command now pairs clean end-to-end controls with synchronized structural
attribution across multiple prepared systems. The first 750-step map identifies
Direct Space and the neighbor-generation lifecycle as the shared large-system
costs. The GPCRmd case also exposes a separate CHARMM bonded/other-force cost
that is invisible in the 5DFR/JAC acceptance pair.

The production runtime remains MLX and Metal only. OpenMM and LAMMPS informed
the execution-boundary analysis but do not enter the measured runtime path.

## Measurement contract

The command was:

```bash
uv run python -m mlx_atomistic.benchmarks.md_suite profile \
  --case dhfr-5dfr-pme \
  --case jac-94k-pme \
  --case gpcrmd-729-pme \
  --warmup-steps 10 \
  --measured-steps 750 \
  --out results/md-suite/stage-profile-generation-binding.json
```

Each case runs two planes:

1. A clean production control provides the authoritative end-to-end
   throughput.
2. A synchronized structural profile inserts completion barriers after named
   stages so device work has exclusive ownership.

The synchronized plane is intrusive. Its fractions are shares of the
instrumented wall, not the clean wall. It preserves the production constraint
implementation, including dense composite Metal routes, but it cannot preserve
normal lazy force overlap. Final-state closeness is diagnostic only because
changed floating-point reduction order can separate long chaotic trajectories.
Both raw runtimes must independently pass their finite-state, constraint,
neighbor, topology, and execution checks.

## Clean throughput

All three clean controls passed:

| Workload | Atoms | Seconds/step | ns/day |
| --- | ---: | ---: | ---: |
| 5DFR | 23,558 | 0.00120679 | 286.38 |
| JAC 4-cell | 94,232 | 0.00401662 | 86.04 |
| GPCRmd 729 | 92,001 | 0.00594068 | 58.18 |

## Structural stage shares

| Stage | 5DFR | JAC | GPCRmd |
| --- | ---: | ---: | ---: |
| Neighbor lifecycle | 17.62% | 28.78% | 24.54% |
| Direct nonbonded | 15.77% | 26.20% | 21.78% |
| Constraints | 18.18% | 10.84% | 9.09% |
| Reciprocal PME | 9.86% | 13.15% | 6.99% |
| Integration and thermostat | 17.27% | 6.98% | 4.74% |
| Force aggregation | 9.72% | 5.64% | 7.53% |
| Bonded and other forces | 5.34% | 3.69% | 21.48% |
| PME sparse corrections | 5.31% | 4.07% | 2.78% |

The neighbor row is an inclusive synchronization boundary. The per-step
displacement admission waits for previously submitted force work, so the row
must not be interpreted as pure neighbor arithmetic. Earlier wait attribution
found that upstream completion owns most of that boundary. The actual rebuild
and force-binding work is still material at scale, but replacing the scalar
displacement reduction alone cannot remove the inclusive wait.

The reusable conclusions are:

- Direct nonbonded is the largest independently synchronized project-owned
  recurring stage on both large systems.
- GPCRmd cannot be represented by the 5DFR/JAC pair alone. Its CHARMM
  bonded/other-force route is as large as Direct Space.
- Constraint and integration costs matter more on the smaller 5DFR system,
  where fixed launch and whole-array costs occupy a larger fraction.
- Reciprocal Particle Mesh Ewald (PME) is material but no longer the dominant
  cross-system target.

## Generation-aware force binding

The MD loops previously called `_PreparedForcePipeline.bind()` on every step,
even when `NeighborListManager.update()` returned the same immutable neighbor
object. The loops now retain the binding until object identity changes. The
synchronized binding counts fell from 751 calls per case to 22 on 5DFR, 22 on
JAC, and 27 on GPCRmd. Those counts equal the initial generation plus measured
rebuilds.

Most binding time belongs to real generation changes, not cached calls. Three
clean 750-step repeats therefore show only a small directional result against
the immediately preceding retained working state:

| Workload | Previous median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| 5DFR | 0.00120764 s/step | 0.00118469 s/step | 1.94% faster |
| JAC | 0.00396847 s/step | 0.00394811 s/step | 0.52% faster |

The JAC result is below the suite's 3% improvement gate. The change is retained
as a generation-ownership correction with no large performance claim.

## Architecture boundary and next work

The stage map rejects three narrow next moves:

- Do not optimize the scalar neighbor check from its inclusive timer.
- Do not revive the rejected shared-exponential Direct Space approximation.
- Do not retune the existing PME charge-spread launch geometry.

The next architecture slice must reduce recurring complete-step work while
remaining useful across force fields:

1. Separate actual rebuild work from the upstream completion sink using only
   existing production synchronization epochs.
2. Attribute GPCRmd's two `other_force_terms` independently and determine
   whether their arrays can join the prepared bonded execution plan.
3. Design a persistent interaction schedule whose producer and consumer share
   one spatial ordering. A 32-lane consumer is eligible only with a
   device-built schedule and complete-step amortization; the rejected
   host-built experimental 32-atom path is not a starting point.
4. Evaluate capacity-owned device buffers and overflow/retry semantics before
   any C++ extension. Exact-shape host readback remains the producer boundary,
   but a native extension is justified only after a successful Metal
   producer/consumer design exists.

The raw JSON remains gitignored under `results/md-suite/`.
