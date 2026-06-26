# OpenMM OpenCL — Amber20 Cellulose & STMV on Apple M5 Max

Engine: `openmm-reference`. Large-system measurement of OpenMM's OpenCL
backend on the M5 Max for the two heavy AMBER 20 benchmark systems.

## Result

| Test | Atoms | M5 Max OpenCL (ns/day) | A100 (ns/day)² | H100 1× (ns/day)² | H100 4× (ns/day)² | B200 (ns/day)² | M5 Max / A100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cellulose PME | 408,609 | **55.89** | 131.85 | 216.98 | 342.21 | 276.41 | **42%** |
| STMV PME | 1,067,095 | **18.14** | 46.05 | 70.16 | 147.09 | 116.91 | **39%** |

² [openmm.org/benchmarks](https://openmm.org/benchmarks), OpenMM 8.4.

## The full M5 Max scaling curve, all benchmarks so far

Combining this report with [`openmm-opencl-dhfr.md`](./openmm-opencl-dhfr.md)
and [`openmm-opencl-apoa1.md`](./openmm-opencl-apoa1.md):

| System | Atoms | PME? | M5 Max ns/day | M5 Max / A100 |
|---|---:|:---:|---:|---:|
| DHFR Implicit | ~2.5k | ❌ | 1762.1 | 91% |
| DHFR Explicit-RF | 23.6k | ❌ | 1018.4 | 70% |
| DHFR Explicit-PME | 23.6k | ✅ | 752.5 | 58% |
| ApoA1 PME | 92.2k | ✅ | 231.1 | 48% |
| Cellulose PME | 408.6k | ✅ | 55.9 | 42% |
| STMV PME | 1,067k | ✅ | 18.1 | 39% |

### What this tells us

The M5 Max / A100 ratio on PME-heavy systems **decelerates as system size
grows**, approaching an asymptote near ~35–40%:

- 23.6k → 92.2k (≈4×): 58% → 48%, a 10-point drop.
- 92.2k → 408.6k (≈4×): 48% → 42%, a 6-point drop.
- 408.6k → 1.07M (≈2.6×): 42% → 39%, a 3-point drop.

The ratio is not crashing toward zero. **M5 Max is not memory-pressure-limited
at STMV (1M atoms)**: unified memory holds the system fine, and the relative
deficit vs A100 stabilizes rather than blowing up. This is the most
informative thing this run produced — Apple's unified memory does carry its
weight at large scale; the gap stays bound by the per-step PME-FFT deficit,
not by data movement.

For reference, even a single A100 only reaches 46 ns/day on STMV; the OpenMM
table shows you need 3–4× H100 to make STMV throughput meaningful (130 to
147 ns/day). M5 Max at 18 ns/day is in the regime where single-GPU NVIDIA is
also a struggle.

## Provenance

- Engine: OpenMM 8.5.1.dev-f7fa0c2, run as an internal reference checkout
  outside the package runtime
- Platform: OpenCL
- OpenCL platform name: Apple
- Device: Apple M5 Max (DeviceIndex 0)
- Host: `AppCubics-MacBook-Pro.local`, Darwin arm64
- Date: 2026-05-15
- Raw output: `results/openmm-opencl-amber20-m5max.json` (gitignored)

## External inputs

These tests require the AMBER 20 Benchmark Suite (not shipped with OpenMM):

| Field | Value |
|---|---|
| Source | `https://ambermd.org/Amber20_Benchmark_Suite.tar.gz` |
| Tarball size | ~75 MB |
| Extracted size | ~411 MB |
| Local path | `results/inputs/Amber20_Benchmark_Suite/` (gitignored) |
| Fetched | 2026-05-15 |

`results/inputs/README.md` records what is in that directory and why. The
benchmark script handles the download automatically on first run; subsequent
runs reuse the local copy. See the reproducer below.

## Config

| Parameter | Cellulose | STMV |
|---|---|---|
| Force field | AMBER (suite default) | AMBER (suite default) |
| Integrator | Langevin (NVT) | Langevin (NVT) |
| Timestep | 4 fs | 4 fs |
| Constraints | HBonds | HBonds |
| Hydrogen mass | 1.5 amu | 1.5 amu |
| PME cutoff | 0.9 nm | 0.9 nm |
| Precision | single | single |
| Target wall time | 30 s | 30 s |

Identical to the configuration used at `openmm.org/benchmarks`.

## Reproducer

This is an internal reference run, not a package workflow. Use the repository
reference harness to recreate the OpenMM command plan; it keeps downloaded
AMBER inputs and raw outputs under gitignored `results/`.

## Open questions for follow-up benchmarks

- **`--disable-pme-stream`** on STMV: would directly test whether the
  PME-stream overlap path is the deficit, or whether the FFT itself is.
- **Cellulose with mixed precision**: large-system mixed-precision benefits
  on Apple GPU are unmeasured publicly.
- **mlx-atomistic at the same scales** — once feasible, an MLX run on a
  comparable system would land alongside these and tell us where the MLX
  runtime fits relative to OpenMM's OpenCL ceiling on this hardware.
