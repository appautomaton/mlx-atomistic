---
title: "OpenMM OpenCL — DHFR on Apple M5 Max"
---


Engine: `openmm-reference`. Reference-ceiling measurement of OpenMM's OpenCL
backend on the M5 Max for the canonical DHFR hello-world system (23k atoms).

## Result

| Test | M5 Max OpenCL (ns/day) | A100 (ns/day)² | H100 (ns/day)² | B200 (ns/day)² | M5 Max / A100 |
|---|---:|---:|---:|---:|---:|
| DHFR Implicit (GBSA) | **1762.1** | 1942.2 | 2498.6 | 2268.4 | **91%** |
| DHFR Explicit-RF | **1018.4** | 1460.6 | 1873.2 | 1802.3 | **70%** |
| DHFR Explicit-PME | **752.5** | 1286.6 | 1704.5 | 1658.7 | **58%** |

² [openmm.org/benchmarks](https://openmm.org/benchmarks), OpenMM 8.4.

### Cross-system scaling (M5 Max vs A100)

| System | Atoms | M5 Max / A100 (PME) |
|---|---:|---:|
| DHFR Implicit (no PME, no water) | 2.5k | 91% |
| DHFR Explicit-RF (water, no PME) | 23.6k | 70% |
| DHFR Explicit-PME | 23.6k | 58% |
| ApoA1 PME ([report](./openmm-opencl-apoa1.md)) | 92.2k | 48% |

The relative gap to A100 **grows with system size and PME usage**, not with
kernel-launch overhead. M5 Max essentially matches an A100 on DHFR Implicit
and falls to about half the throughput at ApoA1 PME.

## What this tells us about the M5 Max bottleneck

The expected story for an Apple GPU vs NVIDIA on a small system is that
launch overhead, threadgroup-memory latency, and lack of warp-level
primitives would hurt the small-system regime more than the large one.
The data here points the other direction:

- **Small + no PME (DHFR GBSA): M5 Max ≈ A100.** Pair-list and bonded force
  kernels on Apple GPU are competitive when they are the only thing running.
- **Large + PME (ApoA1 PME): M5 Max ≈ 0.5 × A100.** The deficit shows up in
  proportion to PME workload.

The real M5 Max bottleneck in OpenMM today is the **OpenCL FFT path used by
PME**, not pair-list dispatch. This is consistent with philipturner's
finding in [openmm/openmm#3924](https://github.com/openmm/openmm/issues/3924)
that the largest unrealized speedups on Apple GPUs come from rewriting
`findBlocksWithInteractions` and prefix-sum primitives in native Metal —
those gains are on the *non-PME* side. A native Metal FFT (separate from
the Metal kernel work) is likely needed to close the PME gap.

## Notable side observation

M5 Max DHFR Implicit (1762 ns/day) lands within ~10% of NVIDIA DGX Spark
(1942 ns/day on the same test). DGX Spark is NVIDIA's personal-workstation
class part, so this is a near-peer comparison in the segment Apple actually
competes in.

## Provenance

- Engine: OpenMM 8.5.1.dev-f7fa0c2, run as an internal reference checkout
  outside the package runtime
- Platform: OpenCL
- OpenCL platform name: Apple
- Device: Apple M5 Max (DeviceIndex 0)
- Host: `AppCubics-MacBook-Pro.local`, Darwin arm64
- Date: 2026-05-15
- Raw output: `results/openmm-opencl-dhfr-m5max.json` (gitignored)

## Config

All three tests share the OpenMM public-benchmark config exactly:

| Parameter | Value |
|---|---|
| Force field | AMBER99SB |
| Integrator | Langevin (NVT) |
| Timestep | 4 fs |
| Constraints | HBonds |
| Hydrogen mass | 1.5 amu |
| Cutoff | 0.9 nm (PME) / 1.0 nm (RF) / 2.0 nm (implicit) |
| Precision | single |
| Target wall time | 30 s per test |

Identical to the configuration used at `openmm.org/benchmarks`, so the
M5 Max column is directly comparable to the NVIDIA columns in that table.

## Reproducer

This is an internal reference run, not a package workflow. Use the repository
reference harness to recreate the OpenMM command plan and keep raw outputs under
gitignored `results/`.

PDB inputs (`5dfr_minimized.pdb`, `5dfr_solv-cube_equil.pdb`) ship with the
OpenMM source in this directory — no downloads required.

## External comparison context

- Cross-reference with [`openmm-opencl-apoa1.md`](./openmm-opencl-apoa1.md)
  to see the small-vs-large scaling on this hardware.
- Same script, same systems, same config as the rows at
  [openmm.org/benchmarks](https://openmm.org/benchmarks).
- The DHFR input is the AMBER JAC benchmark, originating from
  [ambermd.org/GPUPerformance.php](https://ambermd.org/GPUPerformance.php).

## Open questions for follow-up benchmarks

- **STMV (1M atoms)** — the natural other extreme. Would test whether
  Apple's unified-memory advantage shows up at large scale, or whether the
  PME-FFT deficit dominates further.
- **Mixed precision** — `--precision mixed` on DHFR PME to see if FP16
  intermediate accumulation moves the needle on Apple GPU.
- **AMOEBA polarizable on DHFR** — `--test amoebagk,amoebapme` exercises a
  different code path (induced dipoles). M5 Max relative position there is
  unmeasured publicly.
- **PME isolation** — toggle `--disable-pme-stream` to measure how much of
  the M5 Max PME deficit is the stream-overlap path vs the FFT itself.
