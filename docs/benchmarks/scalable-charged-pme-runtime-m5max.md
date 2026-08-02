# Scalable Charged PME Runtime (M5 Max)

Date: 2026-07-15; runtime comparison updated 2026-08-01

Status: `validated-envelope`. The MLX/Metal product runtime passed independent
OpenMM energy/force parity and a bounded fixed-cell NVT gate for the charged
94,232-atom 2x2x1 replication of the AMBER20 JAC DHFR preparation. `JAC` is
the AMBER20 file stem for DHFR, not a different protein. This is the validated
workload; it is not the separately prepared OpenMM 5DFR case, a GPCRmd
membrane run, or a general PME production certification. A later
manifest-bound 75-step MLX/OpenMM run established a `9.7586x` throughput ratio
for this exact workload.

## Raw evidence

Every measured value in this report comes from one of these gitignored JSON
files:

- **[parity]**
  `results/scalable-charged-pme-runtime/jac-2x2x1/charged_pme_parity_report.json`
- **[runtime]**
  `results/scalable-charged-pme-runtime/jac-2x2x1/runtime.json`
- **[profile]**
  `results/scalable-charged-pme-runtime/jac-2x2x1/profile/pme-profile.json`
- **[matched runtime]**
  `results/larger-system-scaling/jac-2x2x1-modern/matched-runtime-v2/`

The complete force arrays are stored at
`results/scalable-charged-pme-runtime/jac-2x2x1/complete_force_comparison.npz`,
as recorded by **[parity]**.

## Validated workload

Source for every value in this table: **[parity]**.

| Field | Validated value |
| --- | --- |
| System | DHFR · AMBER20 JAC preparation, deterministic 2x2x1 replication |
| Operation | Fixed-coordinate total energy and complete forces |
| Atoms | 94,232 |
| Cell | Orthorhombic, 123.2894 x 123.2894 x 61.6447 A |
| Net charge | -44.0 e in the canonical workload manifest |
| Constraints | 89,160 |
| Nonbonded exceptions | 138,836 |
| PME cutoff | 9.0 A |
| Ewald alpha | 0.35 A^-1 |
| Mesh | 128x128x64, 1,048,576 points |
| Assignment | Cardinal B-spline order 5, deconvolution enabled |
| Charge convention | `uniform_neutralizing_plasma` |
| MLX execution | `float32`, `mlx_fft_cic`, `Device(gpu, 0)` |
| OpenMM reference | OpenCL, single precision, Apple M5 Max |
| OpenMM version | `8.5.1.dev-f7fa0c2` |
| MLX version | `0.31.2` |
| Python / host | Python 3.13.12, macOS 26.5.2 arm64 |

The independent OpenMM builder starts from the AMBER `JAC.prmtop` and
`JAC.inpcrd`, clones only the observed supported force classes, and compares a
manifest covering atom and coordinate order, masses, charges, LJ parameters,
constraints, exclusions/exceptions, cell, PME parameters, and force-term
counts. The MLX and OpenMM manifests matched before numerical metrics were
accepted. **[parity]**

## Charged-system convention

Charged PME is opt-in. For total charge `Q`, fixed volume `V`, Ewald parameter
`alpha`, and Coulomb prefactor `k_e`, the runtime uses the OpenMM-compatible
uniform-background term

```text
E_background = -k_e * pi * Q^2 / (2 * V * alpha^2)
```

For this workload, MLX reported `Q=-43.999996 e`,
`V=937017.5 A^3`, and `E_background=-36.809040 kJ/mol`. The term changes scalar
energy and has zero coordinate force at fixed cell. That statement does not
provide an analytic cell derivative or virial. **[parity]**

Artifacts without an explicit supported policy retain the fail-closed
`reject_non_neutral` behavior. Unknown policies, policy disagreement between
metadata and arrays, and non-neutral reject-mode artifacts remain errors.

## Energy and force parity

Source for every measured value and threshold in this table: **[parity]**.

| Check | Measured | Gate | Result |
| --- | ---: | ---: | --- |
| Total-energy absolute error per atom | 0.00015749 kJ/mol/atom | <= 0.005 kJ/mol/atom | pass |
| Total-energy relative error | 1.22186e-5 | <= 5e-5 | pass |
| Nonbonded-energy absolute error per atom | 0.00015799 kJ/mol/atom | <= 0.005 kJ/mol/atom | pass |
| Nonbonded-energy relative error | 1.18161e-5 | <= 5e-5 | pass |
| Complete-force RMS absolute error | 0.10544 kJ/mol/nm | <= 3 kJ/mol/nm | pass |
| Complete-force maximum absolute error | 1.18618 kJ/mol/nm | <= 12 kJ/mol/nm | pass |

The complete MLX, OpenMM, and delta force arrays each have shape
`(94232, 3)` and are hashed in **[parity]**. Component comparison also passed
for bonds, angles, torsions, and the combined nonbonded term.

## Reusable plan and scalable neighbors

Source for every value in this table: **[runtime]**.

| Field | Measured value |
| --- | ---: |
| Plan build count | 1 |
| Plan reuse count after the bounded run | 5 |
| Plan setup time | 0.002347 s |
| Estimated plan resident bytes | 16,777,216 |
| Plan fingerprint | `95ed27f16964d6cd9a83510653b411813eacd8c7673efde6da089c09d5dd052d` |
| Neighbor backend / representation | `mlx_cell_blocks` / `blocks` |
| Candidate interactions | 118,234,717 |
| Compact interactions | 15,935,779 |
| Candidate waste fraction | 86.5219% |
| Dense topology pair cache | not materialized |
| Neighbor fallback | none |

The plan fingerprint covers the fixed cell, mesh, alpha, cutoff, assignment
order, deconvolution, Coulomb constant, dtype/backend/device, and background
policy. Production PME direct space receives the same `NeighborBlocks` policy
used by LJ; it does not fall back to dense all-pairs execution.

## Bounded NVT gate

The gate ran one warmup step followed by two measured fixed-cell Langevin NVT
steps at `0.004 ps`. Source for every value in the next two tables:
**[runtime]**.

| Runtime result | Measured value |
| --- | ---: |
| Measured wall time | 16.493522 s |
| Time per measured step | 8.246761 s/step |
| Throughput | 0.121260 steps/s |
| Simulated throughput | 0.041907 ns/day |
| Warmup wall time | 15.935591 s |
| Final temperature | 263.014 K |
| Final maximum constraint error | 0.00013721 A |
| Peak resident set | 15,340.484 MB |
| MLX peak memory | 12,829,452,316 bytes |
| Finite positions, velocities, energies, forces, and temperature | yes |

| Measured-run timing counter | Wall time |
| --- | ---: |
| Neighbor update | 3.979437 s |
| Neighbor rebuild | 3.797901 s |
| Force evaluation | 12.143885 s |
| Explicit synchronization | 0.367122 s |

The neighbor update and rebuild counters overlap: rebuild time is accumulated
inside the update call, so these rows are not additive. The bounded gate proves
finite execution and plan reuse, not long-trajectory stability.

## PME timing profile

The profiler used one warmup plus two measured evaluations. Values below are
medians from **[profile]**.

| Profile stage | Median time |
| --- | ---: |
| Direct-space Coulomb | 0.469082 s |
| Reciprocal space | 0.034234 s |
| Assignment/interpolation | 0.032126 s |
| FFT/influence | 0.002416 s |
| Corrections | 0.001371 s |
| Synchronization probe | 0.000373 s |
| PME Coulomb total | 0.535213 s |
| Production nonbonded total | 0.908918 s |

Assignment/interpolation and FFT/influence are reciprocal-space sub-splits and
must not be added to reciprocal space again. The profile recorded peak RSS of
`12251.125 MB` and MLX peak memory of `8596761412` bytes. The dense O(N^2)
real-space reference was intentionally disabled above `4096` atoms; the
validated direct path used shared `mlx_cell_blocks` with no fallback.
**[profile]**

## Matched NVT performance update

The 2026-08-01 comparison closes the earlier runtime-manifest blocker without
changing the physics. Both engines used 94,232 atoms, the same force-term and
constraint inventory, fixed orthorhombic cell, 9 A cutoff, alpha 0.35 A^-1,
128x128x64 PME mesh, 4 fs Langevin-middle steps at 300 K and 1/ps friction,
ten warmups, 75 measured steps, and single-precision GPU execution. Explicit
final-state/device completion is inside both timers; setup, warmup, and I/O are
outside.

| Engine | Accelerator | 75 steps | Time per step | Process-tree peak |
| --- | --- | ---: | ---: | ---: |
| MLX | Metal, Apple M5 Max | 1.004613 s | 0.013395 s | 3.86 GB |
| OpenMM | OpenCL, Apple M5 Max | 0.102947 s | 0.001373 s | 0.401 GB |

The admitted ratio is `MLX/OpenMM = 9.7586x`, which narrowly passes the
one-order-of-magnitude stretch target for this named workload. The two final
MLX samples were 1.001684 and 1.004613 seconds, giving a 1.003149-second median.
The retained pipeline is 36.9% faster than its corrected 1.588653-second
control. The final MLX maximum constraint residual was `3.21e-5 A`; OpenMM's
was `1.19e-5 A`. Both states were finite.

The comparison manifest and static admission report pass every required field.
The engines use independent random-number implementations, so the comparison
claims matched protocol throughput, not identical stochastic trajectories.
The older **[runtime]** row correctly retains `openmm_ratio: null` because its
original artifact predates this stronger manifest; historical evidence is not
rewritten retroactively.

## Spatial-direct development update

A later bounded pass changed the recurring Metal work layout without changing
the admitted workload or physics. It builds exact-cutoff 8x8 tiles from
spatially sorted atom blocks, groups up to four tiles sharing a left block, and
reuses that block while locally reducing force writes. A second kernel fuses
sparse PME exclusions, Coulomb exceptions, 1-4 corrections, and LJ exceptions
into one force buffer.

| Development check | Existing route | Spatial/fused route | Change |
| --- | ---: | ---: | ---: |
| Direct-space force | 5.52227 ms | 3.79931 ms | 31.20% lower |
| Sparse PME corrections | 0.568062 ms | 0.262083 ms | 53.86% lower |
| Complete 75-step trajectory, median | 0.933094 s | 0.601954 s | 35.49% lower |

The tile inventory reproduced all 60,502,167 compact pairs. Direct-force error
was `6.32e-5 kJ/mol/A` RMS and `6.71e-4 kJ/mol/A` maximum; correction-force
error was `4.68e-5 kJ/mol/A` RMS and `3.05e-4 kJ/mol/A` maximum. The complete
run stayed finite, ended below a `3.51e-5 A` maximum constraint residual, and
peaked at 5.69 GB across the process tree with a passing late-memory plateau
check. The dual development representation reports 578 MB of persistent tile,
topology-mask, and diagnostic-pair state; admitted tiles no longer allocate the
additional per-pair LJ-scale array.

This is development evidence, not a replacement for the manifest-bound
1.003149-second MLX result or its `9.7586x` OpenMM ratio above. The bounded gate
used the position-balanced order pairs, tiles, tiles, pairs; pair samples were
1.045140 and 0.821048 seconds, while tile samples were 0.577860 and 0.626048
seconds. This clears the 0.85-second development target, but no new cross-engine
ratio is claimed without a fresh manifest-bound OpenMM comparison.

## Reproduce

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python \
  -m mlx_atomistic.benchmarks.charged_pme prepare \
  --source results/dhfr-artifacts/dhfr-amber20-jac-pme \
  --replicas 2,2,1 --assignment-order 5 \
  --background-policy uniform_neutralizing_plasma \
  --out results/scalable-charged-pme-runtime/jac-2x2x1/prepared

UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --with openmm python \
  scripts/run_charged_pme_parity.py \
  --mlx-prepared results/scalable-charged-pme-runtime/jac-2x2x1/prepared \
  --amber-prmtop results/inputs/Amber20_Benchmark_Suite/PME/Topologies/JAC.prmtop \
  --amber-coordinates results/inputs/Amber20_Benchmark_Suite/PME/Coordinates/JAC.inpcrd \
  --replicas 2,2,1 --platform OpenCL \
  --out results/scalable-charged-pme-runtime/jac-2x2x1

UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python \
  -m mlx_atomistic.benchmarks.charged_pme runtime \
  --prepared results/scalable-charged-pme-runtime/jac-2x2x1/prepared \
  --warmups 1 --steps 2 \
  --out results/scalable-charged-pme-runtime/jac-2x2x1/runtime.json

UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --no-sync python \
  -m mlx_atomistic.benchmarks.charged_pme runtime \
  --prepared results/larger-system-scaling/jac-2x2x1-modern/prepared \
  --warmups 10 --steps 75 --seed 17 \
  --out results/larger-system-scaling/jac-2x2x1-modern/matched-runtime-v2/mlx_runtime.json

UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --no-sync python \
  scripts/run_openmm_charged_pme_runtime.py \
  --mlx-prepared results/larger-system-scaling/jac-2x2x1-modern/prepared \
  --mlx-runtime results/larger-system-scaling/jac-2x2x1-modern/matched-runtime-v2/mlx_runtime.json \
  --amber-prmtop results/inputs/Amber20_Benchmark_Suite/PME/Topologies/JAC.prmtop \
  --amber-coordinates results/inputs/Amber20_Benchmark_Suite/PME/Coordinates/JAC.inpcrd \
  --replicas 2,2,1 --platform OpenCL --precision single \
  --warmups 10 --steps 75 --out results/larger-system-scaling/jac-2x2x1-modern/matched-runtime-v2

UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python \
  -m mlx_atomistic.benchmarks.pme_performance \
  --fixture-dir results/scalable-charged-pme-runtime/jac-2x2x1 \
  --iterations 2 --warmups 1 \
  --out-dir results/scalable-charged-pme-runtime/jac-2x2x1/profile --json
```

These commands require the local AMBER20 JAC inputs, Apple Silicon/Metal, and
OpenMM OpenCL for the reference evaluation. Missing inputs or reference
platforms produce a blocked result rather than a synthetic pass.

## Claim boundary

- Validated: this charged AMBER20 JAC 2x2x1 workload, fixed orthorhombic cell,
  94,232 atoms, 128x128x64 mesh, order-5 assignment, 9 A cutoff, and explicit
  uniform neutralizing plasma. The 75-step matched NVT comparison also supports
  a descriptive `9.7586x` MLX/OpenMM ratio for this exact protocol and host.
- Admitted but not broadly certified: fixed-cell orthorhombic PME up to the
  runtime checks of 100,000 atoms and 1,048,576 mesh points when all other
  configuration checks pass.
- Not claimed by this JAC row: the separately measured GPCRmd membrane result,
  production NPT or cell changes, analytic PME virial, triclinic PME, universal
  charged-system coverage, a long stability trajectory, or a general OpenMM
  throughput claim. The bounded GPCRmd result is documented independently in
  [`gpcrmd-729-pme-runtime-m5max.md`](./gpcrmd-729-pme-runtime-m5max.md).
