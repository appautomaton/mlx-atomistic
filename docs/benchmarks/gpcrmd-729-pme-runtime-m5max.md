# GPCRmd 729 PME Runtime Closure (M5 Max)

Date: 2026-07-15

Status: `bounded-pass`. The MLX/Metal product runtime passed independent
OpenMM fixed-coordinate parity, bounded source-protocol NVT execution, saved
trajectory/checkpoint reload, and checkpoint continuation for the real
92,001-atom GPCRmd 729 membrane fixture.

This closes the stale fixture-specific topology and PME blockers. It does not
establish production NPT, analytic PME virial, triclinic PME, production-length
stability, or broad membrane-system readiness.

## Raw evidence

Every quantitative result below comes from these gitignored artifacts:

- **[source]**
  `results/gpcrmd-pme-runtime-closure/source/fixture-manifest.json`
- **[prepared]**
  `results/gpcrmd-pme-runtime-closure/prepared/mlx-workload-manifest.json`
- **[parity]**
  `results/gpcrmd-pme-runtime-closure/parity/gpcrmd_pme_parity_report.json`
- **[runtime]**
  `results/gpcrmd-pme-runtime-closure/runtime/gpcrmd_performance.json`
- **[profile]**
  `results/gpcrmd-pme-runtime-closure/profile/pme-profile.json`
- **[matrix]**
  `results/gpcrmd-pme-runtime-closure/blocker-matrix.json`

Complete MLX, OpenMM, and difference force arrays are stored at
`results/gpcrmd-pme-runtime-closure/parity/complete_force_comparison.npz`, as
recorded by **[parity]**.

## Source and workload

The acquisition manifest records the four required official GPCRmd file IDs,
their resolved names, sizes, SHA-256 hashes, and safe protocol-archive
extraction. Official file downloads require a GPCRmd account; no cookie,
credential, or session material is persisted. **[source]**

Official simulation record:
`https://www.gpcrmd.org/dynadb/dynamics/id/729/`.

| ID | Role | Resolved file | Bytes | SHA-256 |
| ---: | --- | --- | ---: | --- |
| 15286 | topology | `15286_dyn_729.psf` | 17,137,863 | `3e20322b7a5f441eb7d07eec962c4cd17138dd562fa58b7d0a38f3d08ff5434f` |
| 17686 | model | `17686_dyn_729.pdb` | 7,268,129 | `91ae6058c6be2a837bcaa0ba14472d91f35d3b99da9c873bb246a15c2e456925` |
| 15290 | parameters | `15290_prm_729.prm` | 1,186,094 | `f6df09414454e50a22da48908d4680b9c7b695223cd8da6a030425f9294502e9` |
| 17687 | protocol/start files | `17687_oth_729.tar.gz` | 13,162,376 | `c333d83cdfd891fa50bf2acf1e29280d837da5ad47aa5f017b613bd656f07642` |

| Field | Source-faithful value |
| --- | --- |
| Fixture | `gpcrmd-729-beta1-5f8u-cyanopindolol` |
| Atoms | 92,001 |
| Selections | receptor 5,195; ligand 43; lipid 26,800; water 59,832; ions 131 |
| Cell | orthorhombic, 87.17032 × 87.15242 × 118.58050 Å |
| Ensemble | fixed-cell Langevin NVT |
| Selected source replicate | `rep_1` restart coordinates, velocities, and cell |
| Temperature / friction | 310 K / 0.10 ps⁻¹ |
| Timestep | 4 fs |
| Source production length | 125,000,000 steps; trajectory interval 50,000 steps |
| Constraints | 78,896 |
| HMR | 58,952 bonded hydrogens repartitioned to 4.032 Da |
| Nonbonded | 9 Å cutoff; switching from 7.5 Å |
| PME | α = 0.29202899 Å⁻¹; mesh 78 × 78 × 108; order 5 |
| Charge policy | source-neutral; `reject_non_neutral`; tolerance 1 × 10⁻⁴ e |

Source topology and prepared arrays agree on 91,734 bonds, 80,726 angles,
109,071 expanded proper terms, 1,214 harmonic impropers, 49,223 Urey-Bradley
terms, 317 CMAP terms, 237,483 nonbonded-exception records, and five applicable
NBFIX overrides. The prepared manifest hash is
`13b69589e48dc72abfba59232ac1a2ff913f047d892c5a2f8712c3501d93eeac`.
**[prepared]**

The analytic PME minimum in the source derivation was 78 × 78 × 106. Both
independent builders round dimensions to the OpenCL/VkFFT-supported
2/3/5/7/11/13 factor set, yielding 78 × 78 × 108. OpenMM resolved exactly that
declared grid. **[prepared] [parity]**

## Independent OpenMM parity

OpenMM was built independently from the GPCRmd PSF, PDB, parameter, and start
files. It did not consume MLX force objects. The canonical manifests matched
particles, coordinates, masses, charges/LJ data, cell, CHARMM terms,
constraints/HMR, exclusions/exceptions, switching, and PME semantics before
metrics were accepted. **[parity]**

| Check | Measured | Gate | Result |
| --- | ---: | ---: | --- |
| Total-energy error per atom | 1.49744 × 10⁻⁷ kJ/mol/atom | ≤ 5 × 10⁻³ | pass |
| Relative total-energy error | 1.57880 × 10⁻⁸ | ≤ 5 × 10⁻⁵ | pass |
| Complete-force RMS error | 0.085565 kJ/mol/nm | ≤ 3 | pass |
| Complete-force maximum error | 11.6013 kJ/mol/nm | ≤ 12 | pass |

The complete force arrays each have shape `(92001, 3)`. Every reported
component-energy bound passed. The maximum force error is inside, but close to,
the fixed 12 kJ/mol/nm gate; the bound was not relaxed. **[parity]**

Reference execution used OpenMM `8.5.1.dev-f7fa0c2`, OpenCL single precision,
and `Apple M5 Max`. MLX used version `0.31.2`, `float32`, Metal GPU device 0,
and backend `mlx_fft_cic`. **[parity]**

## Bounded source-protocol runtime and restart

The runtime used source-derived 4 fs, 310 K, 0.10 ps⁻¹ fixed-cell NVT. Prepared
velocities were projected onto the constraints without temperature rescaling.
Twenty constraint iterations were used for this real-system row. **[runtime]**

| Phase | Step range | Time range | Steps | Run wall time | Max constraint residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warmup | 0 → 1 | 0 → 0.004 ps | 1 | 36.3446 s | 3.3855 × 10⁻⁵ Å |
| Measured | 1 → 3 | 0.004 → 0.012 ps | 2 | 38.2100 s | 5.5552 × 10⁻⁵ Å |
| Restart | 3 → 4 | 0.012 → 0.016 ps | 1 | 35.6137 s | 4.9829 × 10⁻⁵ Å |

Measured throughput was `0.052342 steps/s`, equivalent to
`2.09369 × 10⁻⁴ ps/s`. No OpenMM runtime ratio is reported because no matching
OpenMM NVT runtime manifest was produced. **[runtime]**

Every phase reported finite positions, velocities, potential/kinetic/total
energy, forces, temperature, and constraint diagnostics. The fixed cell, HMR
state, thermostat RNG step offset, lazy topology, neighbor policy, and PME-plan
metadata were preserved. Trajectory and checkpoint files reloaded, and restart
continued without minimization or equilibration. **[runtime]**

Each phase built one PME plan and recorded reuse. The runtime used
`mlx_cell_blocks`/`NeighborBlocks`, shared LJ and direct-space PME neighbors,
and no dense or tiled fallback. The process-wide cumulative RSS high-water mark
reported by `ru_maxrss` rose from 25,479 MB in warmup to 39,538 MB after
restart; these are not phase-local allocation measurements. **[runtime]**

## PME timing profile

The profiler used the passing parity report and prepared artifact as explicit
inputs, then rechecked atom count, manifest integrity, PME configuration, lazy
topology, shared NeighborBlocks, and no-fallback policy before timing. It ran
one warmup plus two measured evaluations. Values are medians from **[profile]**.

| Profile stage | Median time |
| --- | ---: |
| Direct-space Coulomb | 0.468097 s |
| Reciprocal space | 0.043819 s |
| Assignment/interpolation | 0.058718 s |
| FFT/influence | 0.002230 s |
| Corrections | 0.002716 s |
| Synchronization probe | 0.000371 s |
| PME Coulomb total | 0.524680 s |
| Production nonbonded total | 1.141916 s |

Assignment/interpolation and FFT/influence are independently timed diagnostic
decompositions of reciprocal work. They must not be added to the separately
timed reciprocal-space row to reconstruct wall time. The profile recorded one
plan build, nine reuses, 8.763 GB peak MLX memory, 12.766 GB peak process RSS,
and no runtime fallback. The dense O(N²) reference lane was intentionally
disabled at this atom count; the production direct path remained block-neighbor
based. **[profile]**

## Closure decision

The regenerated blocker matrix marks all in-boundary categories passed. The
`npt_barostat` category is an explicit anti-goal rather than a hidden blocker.
The stale `topology_terms` and `electrostatics_pme` observations are not reused.
**[matrix]**

The evidence supports this statement only:

> MLX/Metal can prepare, parity-check, execute, save, and restart the fixed-cell
> orthorhombic NVT GPCRmd 729 workload for this bounded four-step protocol.

It does not support a claim of production-length stability, production NPT,
cell-changing dynamics, analytic PME virial, triclinic PME, general GPCRmd
coverage, or broad membrane-production readiness.

## Reproduce

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python \
  scripts/acquire_gpcrmd_fixture.py \
  --target-id gpcrmd-729-beta1-5f8u-cyanopindolol \
  --cache notebooks/ligand-receptor-motion/data/gpcrmd-cache/729 \
  --manifest results/gpcrmd-pme-runtime-closure/source/fixture-manifest.json

UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python \
  -m mlx_atomistic.prep.gpcrmd prepare \
  --target-id gpcrmd-729-beta1-5f8u-cyanopindolol \
  --cache notebooks/ligand-receptor-motion/data/gpcrmd-cache/729 \
  --source-manifest results/gpcrmd-pme-runtime-closure/source/fixture-manifest.json \
  --out results/gpcrmd-pme-runtime-closure/prepared \
  --report results/gpcrmd-pme-runtime-closure/preparation-report.json

UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --with openmm python \
  scripts/run_gpcrmd_pme_parity.py \
  --source-manifest results/gpcrmd-pme-runtime-closure/source/fixture-manifest.json \
  --cache notebooks/ligand-receptor-motion/data/gpcrmd-cache/729 \
  --mlx-prepared results/gpcrmd-pme-runtime-closure/prepared \
  --platform OpenCL --out results/gpcrmd-pme-runtime-closure/parity

UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python \
  -m mlx_atomistic.prep.gpcrmd_benchmark \
  --target-id gpcrmd-729-beta1-5f8u-cyanopindolol \
  --prepared results/gpcrmd-pme-runtime-closure/prepared \
  --protocol-manifest \
  results/gpcrmd-pme-runtime-closure/prepared/mlx-workload-manifest.json \
  --warmups 1 --measured-steps 2 --checkpoint-restart \
  --out results/gpcrmd-pme-runtime-closure/runtime --force --json

UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python \
  -m mlx_atomistic.benchmarks.pme_performance \
  --parity-report \
  results/gpcrmd-pme-runtime-closure/parity/gpcrmd_pme_parity_report.json \
  --prepared results/gpcrmd-pme-runtime-closure/prepared \
  --iterations 2 --warmups 1 \
  --out-dir results/gpcrmd-pme-runtime-closure/profile --json

UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python \
  scripts/build_production_md_blocker_matrix.py \
  --candidate results/gpcrmd-pme-runtime-closure/source/fixture-manifest.json \
  --openmm \
  results/gpcrmd-pme-runtime-closure/parity/gpcrmd_pme_parity_report.json \
  --mlx results/gpcrmd-pme-runtime-closure/runtime/gpcrmd_performance.json \
  --out results/gpcrmd-pme-runtime-closure/blocker-matrix.json \
  --report results/gpcrmd-pme-runtime-closure/final-readiness-report.md
```

Acquisition needs access to the official GPCRmd downloads. The parity and live
runtime/profile commands need the local source/prepared artifacts, Apple
Silicon/Metal, and OpenMM OpenCL for the reference evaluation.
