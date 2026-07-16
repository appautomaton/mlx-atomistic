---
title: "Scalable Charged PME Runtime (M5 Max)"
---


Date: 2026-07-15

Status: `validated-envelope`. The MLX/Metal product runtime passed independent
OpenMM energy/force parity and a bounded fixed-cell NVT gate for the charged
94,232-atom AMBER20 JAC 2x2x1 supercell. This is the validated workload; it is
not a GPCRmd membrane run or a general PME production certification.

## Raw evidence

Every measured value in this report comes from one of these gitignored JSON
files:

- **[parity]**
  `results/scalable-charged-pme-runtime/jac-2x2x1/charged_pme_parity_report.json`
- **[runtime]**
  `results/scalable-charged-pme-runtime/jac-2x2x1/runtime.json`
- **[profile]**
  `results/scalable-charged-pme-runtime/jac-2x2x1/profile/pme-profile.json`

The complete force arrays are stored at
`results/scalable-charged-pme-runtime/jac-2x2x1/complete_force_comparison.npz`,
as recorded by **[parity]**.

## Validated workload

Source for every value in this table: **[parity]**.

| Field | Validated value |
| --- | --- |
| System | AMBER20 JAC, deterministic 2x2x1 replication |
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

## Comparison status

Energy and complete-force parity are valid same-workload comparisons because
the fixed-coordinate manifests matched. No OpenMM/MLX NVT throughput ratio is
reported: **[runtime]** records `openmm_ratio: null` and
`not_reported_without_matching_runtime_manifest`. Existing OpenMM DHFR and
ApoA1 throughput reports use different systems or runtime schemas and are
reference context only.

## Reproduce

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python \
  -m mlx_atomistic.benchmarks.charged_pme prepare \
  --source results/dhfr-artifacts/dhfr-explicit-pme \
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
  uniform neutralizing plasma.
- Admitted but not broadly certified: fixed-cell orthorhombic PME up to the
  runtime checks of 100,000 atoms and 1,048,576 mesh points when all other
  configuration checks pass.
- Not claimed by this JAC row: the separately measured GPCRmd membrane result,
  production NPT or cell changes, analytic PME virial, triclinic PME, universal
  charged-system coverage, a long stability trajectory, or an OpenMM
  throughput ratio. The bounded GPCRmd result is documented independently in
  [`gpcrmd-729-pme-runtime-m5max.md`](./gpcrmd-729-pme-runtime-m5max.md).
