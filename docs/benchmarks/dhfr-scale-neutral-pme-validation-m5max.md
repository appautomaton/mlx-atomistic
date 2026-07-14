# DHFR-Scale Neutral PME Validation (M5 Max)

Date: 2026-07-13

Status: `validated` for neutral orthorhombic PME through 24,488 atoms. This is a
controlled TIP3P/NaCl workload in the DHFR atom-count class, not a DHFR protein
simulation and not a GPCRmd production claim.

## Result

The MLX/Metal production path passed three separate gates on the same target
fixture and OpenMM-resolved PME parameters:

| Gate | Result |
| --- | --- |
| Fixed-coordinate accuracy | Three target configurations passed OpenMM parity; worst normalized RMS/maximum force errors were `9.88e-5`/`1.34e-4`, and worst energy error was `3.28e-4 kJ mol^-1 atom^-1`. |
| Short dynamics | The 1.0/0.5 fs NVE maximum drifts were `0.00928`/`0.00530 kJ mol^-1 atom^-1`; maximum constraint errors were `1.11e-5`/`1.00e-5 nm`. The 1 ps NVT mean/final temperatures were `329.57`/`302.54 K`. |
| Target-scale operation | Production nonbonded PME completed with `mlx_cell_pairs`, `3,755,924` compact pairs, no fallback, `0.06008 s` median evaluation time, and no sustained memory growth across five repeated evaluations. |

This closes the previous 4,096-atom neutral PME readiness ceiling for the
validated envelope below. It does not validate charged-system conventions,
triclinic PME, analytic PME virial/NPT, a neutralized DHFR artifact, or the
92,001-atom GPCRmd membrane fixture.

## System And PME Contract

| Field | Value |
| --- | --- |
| Initial solvent sites | 8,192 |
| Composition | 8,148 TIP3P waters, 22 Na+, 22 Cl- |
| Atom count | 24,488 |
| Net charge | 0 e |
| Ionic strength | 0.148779 M |
| Cubic box | 62.6195 Angstrom per side |
| Fixture seed | 20260713 |
| Parameter source | `amber14_tip3p_joung_cheatham_ions`; OpenMM `amber14/tip3p.xml` parameter reference |
| Real-space cutoff | 9.0 Angstrom |
| Ewald error-tolerance request | `5e-4` |
| OpenMM-resolved alpha | 0.292028987 Angstrom^-1 |
| PME mesh | 56 x 56 x 56 |
| Assignment order | 5 |
| Target fixture hash | `d21d1ad3f0fe4623ae1dbcd11a967f9c0e963a1e851d8be8c7026ba81753b22a` |
| Target parameter-manifest hash | `a3e5d99a4d6517ccf20df17817a638cf8e5057040c594883e4a3e55de83f1fe4` |

OpenMM selected the alpha and mesh from its Context on the Reference platform
in double precision. MLX then used those resolved values explicitly in its
float32 production path. OpenMM remains a reference engine; trajectory
generation and target profiling use `mlx_atomistic`.

## Accuracy Evidence

Direct Ewald supplies an independent mathematical reference on the small
fixtures. OpenMM supplies the matched cross-engine PME reference on the small
and target fixtures.

| Fixture/reference | Configurations | Last-ladder normalized RMS | Worst MLX normalized RMS | Worst normalized maximum | Worst energy error (`kJ mol^-1 atom^-1`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Water / direct Ewald | 5 | `1.67e-8` | `9.28e-4` | `1.50e-3` | `7.46e-3` |
| Salt / direct Ewald | 5 | `1.92e-8` | `8.55e-4` | `1.74e-3` | `3.00e-3` |
| Water / OpenMM PME | 5 | N/A | `9.77e-5` | `1.35e-4` | `2.96e-4` |
| Salt / OpenMM PME | 5 | N/A | `1.02e-4` | `1.07e-4` | `3.54e-4` |
| Target / OpenMM PME | 3 | N/A | `9.88e-5` | `1.34e-4` | `3.28e-4` |

Every row passed the approved force and energy thresholds. Whole-box wrapping
also passed. The target wrapping check reported `1.65e-5` normalized RMS force
difference, `7.83e-5` normalized maximum, and zero energy difference per atom.

Small-fixture hashes:

- water: `750ab9da58a8bce9ade438a030d4e9ece198cdb709ed38272667e1688a3b1c74`
- salt: `f216f5d9ac683faa268326c24e30eb73a81d72b97e42bec64736141f3a16a5ed`

## Dynamics Evidence

The stability command starts from a deterministic 100-step constrained
minimization and uses the same minimized state and velocity seed for both NVE
timesteps. Rigid TIP3P geometry uses tolerance-aware SHAKE/RATTLE projection
with a `1e-4 Angstrom` position tolerance.

| Ensemble | Timestep | Steps | Maximum drift per atom | Maximum constraint error | Temperature |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVE | 1.0 fs | 1,000 | `0.00928 kJ mol^-1` | `1.11e-5 nm` | mean `556.79 K` |
| NVE | 0.5 fs | 2,000 | `0.00530 kJ mol^-1` | `1.00e-5 nm` | mean `558.36 K` |
| Langevin NVT | 1.0 fs | 1,000 | N/A | `1.00e-5 nm` | mean `329.57 K`; final `302.54 K` |

The deterministic lattice remains a validation fixture rather than an
equilibrated liquid. The bounded minimization reduced the energy from
`-37,558.48` to `-212,671.48 kJ/mol` but reached its 100-step limit with a
projected maximum force of `117.04 kJ mol^-1 Angstrom^-1`. The hot NVE mean is
therefore reported as a fixture-relaxation caveat, not hidden as an equilibrated
temperature claim. The required energy drift, constraint, and NVT-temperature
criteria still pass.

## Timing And Memory

The profiler uses explicit `mx.eval` barriers around each measured stage. The
target dense quadratic real-space oracle is not executed; it remains limited to
fixtures of at most 4,096 atoms.

| Stage | Median time |
| --- | ---: |
| Charge assignment | `0.01344 s` |
| Potential/field interpolation | `0.03117 s` |
| Forward FFT | `0.000262 s` |
| Influence work | `0.000553 s` |
| Inverse FFT and fields | `0.000567 s` |
| Reciprocal-space full | `0.03706 s` |
| Direct-space Coulomb | `0.00771 s` |
| Coulomb corrections | `0.000792 s` |
| PME Coulomb total | `0.04847 s` |
| Production LJ + PME nonbonded total | `0.06008 s` |

Memory evidence:

- peak process RSS: `510.69 MB`;
- peak MLX allocation: `582.58 MB`;
- largest observed MLX cache: `1,548.36 MB`;
- five-evaluation RSS growth: `0.016 MB`;
- five-evaluation active-Metal growth: `0 MB`.

The exact in-function synchronization split remains unavailable without deeper
runtime instrumentation and is recorded as an unsupported timing split.

## Comparison Boundary

No OpenMM/MLX performance ratio is reported. The available OpenMM target
artifact is a double-precision correctness reference for PME electrostatic
energy and forces. The MLX timing row is a float32 production LJ + PME
nonbonded evaluation. Their operation, precision, and timing metrics do not
match, so the comparison is `diagnostic` and the ratio is suppressed.

## Provenance

- Run host: Apple M5 Max / arm64, macOS 26.5.2.
- Python: 3.13.12.
- MLX: 0.31.2, `Device(gpu, 0)`, Metal available.
- OpenMM reference: 8.5.1 development build, Reference platform, double
  precision.
- Profile commit: `4d899d3776fc526ba3059d007c4bd25a323c2a98`.
- Raw output root: `results/dhfr-scale-neutral-pme-validation/` (gitignored).

Primary raw outputs:

- `ewald-water-small.json`, `ewald-salt-small.json`
- `openmm-water-small/`, `openmm-salt-small/`, `openmm-target/`
- `mlx-water-small/mlx_parity.json`, `mlx-salt-small/mlx_parity.json`,
  `mlx-target/mlx_parity.json`
- `stability/stability.json` and its NVE/NVT NPZ files
- `profile/pme-profile.json`

## Reproduce

```bash
uv run python -m mlx_atomistic.benchmarks.pme_validation \
  --case water-small \
  --out results/dhfr-scale-neutral-pme-validation/ewald-water-small.json

uv run python -m mlx_atomistic.benchmarks.pme_validation \
  --case salt-small \
  --out results/dhfr-scale-neutral-pme-validation/ewald-salt-small.json

uv run --with openmm python scripts/run_openmm_pme_validation.py \
  --case target --configurations 3 \
  --out results/dhfr-scale-neutral-pme-validation/openmm-target

uv run python -m mlx_atomistic.benchmarks.pme_validation \
  --case target \
  --reference results/dhfr-scale-neutral-pme-validation/openmm-target \
  --out results/dhfr-scale-neutral-pme-validation/mlx-target

uv run python -m mlx_atomistic.benchmarks.pme_stability \
  --case target \
  --reference results/dhfr-scale-neutral-pme-validation/openmm-target \
  --nve-ps 1.0 --nve-dt-fs 1.0,0.5 \
  --nvt-ps 1.0 --nvt-dt-fs 1.0 --temperature-k 300 \
  --out results/dhfr-scale-neutral-pme-validation/stability

uv run python -m mlx_atomistic.benchmarks.pme_performance \
  --fixture-dir results/dhfr-scale-neutral-pme-validation/mlx-target/prepared \
  --iterations 5 --warmups 2 \
  --out-dir results/dhfr-scale-neutral-pme-validation/profile --json
```

The original PME and smooth-PME papers motivate the method's `N log N`
accuracy/scaling design: <https://doi.org/10.1063/1.464397> and
<https://doi.org/10.1063/1.470117>. OpenMM's PME theory documentation describes
its fifth-order B-spline implementation and empirical Ewald error tolerance:
<https://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#coulomb-interaction-with-particle-mesh-ewald>.
