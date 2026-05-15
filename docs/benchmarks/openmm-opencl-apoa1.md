# OpenMM OpenCL — ApoA1 on Apple M5 Max

Engine: `openmm-reference`. Not a product runtime path; this is a reference
ceiling for what OpenMM extracts from the M5 Max via its OpenCL backend.

## Result

| Test | M5 Max OpenCL (ns/day) | M1 Max OpenCL (ns/day)¹ | A100 (ns/day)² | H100 (ns/day)² | B200 (ns/day)² |
|---|---:|---:|---:|---:|---:|
| ApoA1 RF | **331.8** | 41.7 | 615.9 | 921.8 | 1000.8 |
| ApoA1 PME | **231.1** | 31.7 | 479.7 | 742.1 | 875.9 |
| ApoA1 LJPME | **172.9** | 25.4 | 356.7 | 553.8 | 655.1 |

¹ philipturner, [openmm/openmm#3847](https://github.com/openmm/openmm/issues/3847) (2022, OpenMM dev branch).
² [openmm.org/benchmarks](https://openmm.org/benchmarks), OpenMM 8.4.

### Derived ratios

- M5 Max vs M1 Max ApoA1 PME: **7.3×** speedup across 4–5 GPU generations.
- M5 Max vs A100 ApoA1 PME: **48%** of A100 throughput.
- M5 Max vs H100 ApoA1 PME: **31%** of H100 throughput.
- Per-watt (rough, ≤80 W vs 400 W for A100): M5 Max ≈ **2.4×** A100 on ApoA1 PME.

## Provenance

- Engine: OpenMM 8.5.1.dev-f7fa0c2 (vendored at `vendors/openmm/`, run from the
  upstream stock benchmark script)
- Platform: OpenCL
- OpenCL platform name: Apple
- Device: Apple M5 Max (DeviceIndex 0)
- Host: `AppCubics-MacBook-Pro.local`, Darwin arm64
- Date: 2026-05-15
- Raw output: `results/openmm-opencl-apoa1-m5max.json` (gitignored)

## Config

All three tests share the OpenMM public-benchmark config exactly:

| Parameter | Value |
|---|---|
| Force field | AMBER14 |
| Integrator | Langevin (NVT) |
| Timestep | 4 fs |
| Constraints | HBonds |
| Hydrogen mass | 1.5 amu |
| PME cutoff | 0.9 nm (RF uses 1.0 nm) |
| Precision | single |
| Target wall time | 30 s per test |

This matches the configuration on `openmm.org/benchmarks`, so the M5 Max
column is directly comparable to the NVIDIA columns in that table.

## Reproducer

```bash
cd vendors/openmm/examples/benchmarks
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --project ../../../.. \
  python benchmark.py \
    --platform OpenCL \
    --test apoa1rf,apoa1pme,apoa1ljpme \
    --seconds 30 \
    --precision single \
    --outfile ../../../../results/openmm-opencl-apoa1-m5max.json
```

OpenCL device access on macOS requires running outside the default Claude
Code sandbox; from a normal terminal session no special permission is
needed. See `docs/runtime-boundaries.md` for the broader OpenMM-as-reference
boundary statement.

## Notes worth keeping

- **OpenCL ICD overhead is still present.** philipturner
  [(openmm/openmm#3924)](https://github.com/openmm/openmm/issues/3924) shows that
  reimplementing `findBlocksWithInteractions` in native Metal Shading Language —
  including `simd_prefix_inclusive_sum` and `half`-compressed position buffers —
  yields **+58% to +73%** over the current OpenCL kernel on Apple GPUs. A
  hypothetical OpenMM Metal backend would push M5 Max ApoA1 PME toward 300–380
  ns/day, in the same range as A100.
- **Apple GPUs have no native FP64.** Single-precision is the only realistic
  GPU path; double-precision asks fall back to CPU or emulation.
- **GROMACS does not plan to add Metal.** See
  [t/gpu-acceleration-on-mac-m1-mini/2938](https://gromacs.bioexcel.eu/t/gpu-acceleration-on-mac-m1-mini/2938).
  GROMACS on Apple Silicon is OpenCL-only with `GMX_GPU_DISABLE_COMPATIBILITY_CHECK=1`.

## External comparison context

- Same script, same systems, same config as the rows at
  [openmm.org/benchmarks](https://openmm.org/benchmarks).
- HECBioSim publishes a parallel benchmark suite at
  [hecbiosim.ac.uk/access-hpc/hpc-benchmarking](https://www.hecbiosim.ac.uk/access-hpc/hpc-benchmarking)
  with standardized 21k / 61k / 465k / 1.4M / 3M-atom systems and energy-per-ns
  figures; their tooling is at [github.com/HECBioSim/hpcbench](https://github.com/HECBioSim/hpcbench).
- AMBER's source benchmark page is [ambermd.org/GPUPerformance.php](https://ambermd.org/GPUPerformance.php);
  OpenMM's DHFR/ApoA1/Cellulose/STMV input sets are imported from there.

## Open questions for follow-up benchmarks

- **DHFR (23k atoms)** — same script with `--test rf,pme` exercises the
  smaller hello-world system. Would let M5 Max land in *every* column of the
  OpenMM official table, not just ApoA1.
- **Mixed precision on Apple GPU** — `--precision mixed` to test whether
  Apple's FP16 path measurably helps; most NVIDIA reference numbers use
  mixed by default.
- **STMV (1M atoms)** — large-system scaling on Apple Silicon. Likely the
  point where unified memory either pays off or hits a wall vs A100 80 GB.
- **mlx-atomistic on the same system** — once the `mlx_atomistic` runtime
  can run ApoA1, that result lives at
  `docs/benchmarks/mlx-atomistic-apoa1.md` and is directly comparable to
  this file.
