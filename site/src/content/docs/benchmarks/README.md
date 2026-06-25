---
title: "Benchmarks"
---


This directory collects benchmark results in a form that is comparable across
machines, runs, and engines. Each file documents one benchmark: what was run,
on what hardware, with what config, and how to reproduce it.

## Engine label convention

Per the runtime-boundaries doc, every result carries an engine tag:

- `mlx_atomistic` — the project's MLX/Metal runtime (product output)
- `openmm-reference` — OpenMM, used as a reference ceiling, not a product path
- `lammps-reference` — LAMMPS, used as a reference for GPU/OpenCL semantics

Filenames lead with the engine tag plus the platform and system, e.g.
`openmm-opencl-apoa1.md`.

## File template

Each result file should answer, in order:

1. **Result table** — ns/day (and any other primary metric) for each test,
   with one column per platform variant if applicable.
2. **Provenance** — engine version, device, host, date, commit if relevant.
3. **Config** — timestep, cutoff, constraints, precision, ensemble. Match
   OpenMM's public benchmark config when comparing against `openmm.org/benchmarks`.
4. **Reproducer** — exact shell command that regenerates the JSON, plus the
   path to the raw JSON output (kept under gitignored `results/`).
5. **External comparison** — links to public reference numbers, with the
   same config caveats called out.

## Index

| File | Engine | System | Platform | Host |
|---|---|---|---|---|
| [inventory-gap-matrix.md](./inventory-gap-matrix.md) | mlx_atomistic | benchmark inventory and Phase 3 gaps | N/A | N/A |
| [benchmark-ladder.md](./benchmark-ladder.md) | mlx_atomistic/openmm-reference/lammps-reference | benchmark ladder and row decision value | Metal/OpenCL where available | local |
| [same-workload-comparison-matrix.md](./same-workload-comparison-matrix.md) | mlx_atomistic/openmm-reference | planned same-workload comparison pairs | Metal/OpenCL where available | local |
| [same-workload-openmm-comparison.md](./same-workload-openmm-comparison.md) | mlx_atomistic/openmm-reference | refreshed controlled same-workload comparison report | Metal/OpenCL where available | local |
| [same-workload-dhfr-stretch.md](./same-workload-dhfr-stretch.md) | mlx_atomistic/openmm-reference | DHFR stretch status | Metal/OpenCL where available | local |
| [performance-audit-baseline.md](./performance-audit-baseline.md) | mlx_atomistic | fast baseline audit and ranked backlog | Metal/OpenCL where available | local |
| [m5max-reference-engines.md](./m5max-reference-engines.md) | openmm-reference/lammps-reference | M5 Max reference-engine manifest overview | OpenCL | Apple M5 Max |
| [openmm-opencl-dhfr.md](./openmm-opencl-dhfr.md) | openmm-reference | DHFR (23k atoms) | OpenCL | Apple M5 Max |
| [openmm-opencl-apoa1.md](./openmm-opencl-apoa1.md) | openmm-reference | ApoA1 (92k atoms) | OpenCL | Apple M5 Max |
| [openmm-opencl-amber20.md](./openmm-opencl-amber20.md) | openmm-reference | Cellulose (409k) + STMV (1.07M atoms) | OpenCL | Apple M5 Max |
| [lammps-opencl-m5max.md](./lammps-opencl-m5max.md) | lammps-reference | official LAMMPS five-case benchmark set | OpenCL | Apple M5 Max |

The inventory appears first. Result files are ordered by system size, smallest
first, so the scaling story reads top-to-bottom.

## Command Matrix

Fast developer commands are routine local checks. They must not require OpenMM,
LAMMPS, OpenCL, large downloaded fixtures, or committed raw outputs.

| Command | Engine | Tier | Output |
| --- | --- | --- | --- |
| `uv run pytest tests/test_benchmarks.py -q` | mlx_atomistic | fast developer | pytest stdout; temporary test files only |
| `uv run python -m mlx_atomistic.benchmarks.mm_force_terms --evaluations 1 --particles 16 --json` | mlx_atomistic | fast developer | normalized JSON on stdout |
| `uv run python -m mlx_atomistic.benchmarks.md_acceleration --sizes 16 --evaluations 1 --json` | mlx_atomistic | fast developer | normalized JSON on stdout |
| `uv run python -m mlx_atomistic.benchmarks.md_performance --sizes 32 --steps 1 --sample-interval 1 --diagnostic-interval 1 --evaluation-interval 1 --json` | mlx_atomistic | fast developer | normalized JSON on stdout |
| `uv run python -m mlx_atomistic.benchmarks.pme_performance --fixture-dir results/missing-pme-fixture --iterations 1 --warmups 0 --json` | mlx_atomistic | fast developer blocked-path smoke | normalized blocked JSON on stdout |

Opt-in performance commands are non-CI and non-routine. They may need Apple
Silicon/Metal, prepared fixtures, OpenMM/LAMMPS from the `dev` group, OpenCL, or
downloaded inputs. Raw JSON/CSV belongs under gitignored `results/`; committed
Markdown summaries should cite those raw paths and reproduction commands.

| Command | Engine | Tier | Output |
| --- | --- | --- | --- |
| `uv run python -m mlx_atomistic.benchmarks.md_performance --include-large --steps 100 --json > results/mlx-md-performance.json` | mlx_atomistic | opt-in performance | raw JSON under `results/` |
| `uv run python -m mlx_atomistic.benchmarks.md_acceleration --include-large --evaluations 10 --json > results/mlx-md-acceleration.json` | mlx_atomistic | opt-in performance | raw JSON under `results/` |
| `uv run python -m mlx_atomistic.benchmarks.pme_performance --out-dir results/pme-performance --json` | mlx_atomistic | opt-in performance | raw JSON under `results/pme-performance/` |
| `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --steps 1 --json > results/same-workload-openmm-comparison/mlx-dhfr-implicit.json` | mlx_atomistic | opt-in runnable stretch smoke | normalized runnable JSON under `results/` |
| `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --steps 1 --json > results/same-workload-openmm-comparison/mlx-dhfr-explicit-pme.json` | mlx_atomistic | opt-in blocked-path stretch | normalized blocked JSON under `results/` until PME neutrality policy is resolved |
| `uv run python scripts/benchmark_openmm_opencl.py --platform OpenCL --particles 4096 --steps 1000 --json --csv results/openmm-opencl-synthetic.csv > results/openmm-opencl-synthetic.json` | openmm-reference | opt-in reference | raw JSON/CSV under `results/` |
| `uv run python scripts/benchmark_openmm_dhfr.py --case dhfr-implicit --platform Reference --steps 1 --json > results/same-workload-openmm-comparison/openmm-dhfr-implicit.json` | openmm-reference | opt-in reference shape check | raw JSON under `results/` |
| `uv run python scripts/benchmark_openmm_dhfr.py --case dhfr-explicit-pme --platform Reference --steps 1 --json > results/same-workload-openmm-comparison/openmm-dhfr-explicit-pme.json` | openmm-reference | opt-in reference shape check | raw JSON under `results/` |
| `uv run python scripts/benchmark_openmm_opencl.py --platform DefinitelyMissing --particles 16 --steps 1 --json` | openmm-reference | fast blocked-path smoke | normalized blocked JSON on stdout |
| `uv run python scripts/benchmark_lammps_opencl.py --particles 16 --steps 1 --json` | lammps-reference | opt-in reference / blocked-path smoke | normalized JSON or blocked JSON on stdout |
| `uv run python scripts/benchmark_m5max_reference.py environment --json` | openmm-reference/lammps-reference | reference environment probe | normalized JSON on stdout |
| `uv run python scripts/benchmark_m5max_reference.py openmm --dry-run --json` | openmm-reference | opt-in reference command plan | raw path plan under `results/m5max-reference/openmm/` |
| `uv run python scripts/benchmark_m5max_reference.py lammps --classify-only --json` | lammps-reference | opt-in official case classification | normalized diagnostic JSON on stdout |
| `uv run python scripts/benchmark_m5max_reference.py run --seconds 30 --json` | openmm-reference/lammps-reference | host-only reference benchmark suite | raw manifest under `results/m5max-reference/` |
| `uv run python scripts/benchmark_m5max_reference.py validate --manifest results/m5max-reference/manifest.json --json` | openmm-reference/lammps-reference | reference manifest validation | validation JSON on stdout |

## External inputs

Some benchmarks pull input data from upstream sources. The reproducer
commands handle the download automatically. Downloaded data lands in
`results/inputs/` (gitignored), with a one-line provenance record in
`results/inputs/README.md`. Re-running a reproducer is the recommended way
to refresh; nothing in `results/inputs/` needs to be committed.

## Raw outputs

Raw JSON/CSV produced by the benchmark scripts is written to `results/`,
which is gitignored. The synthesized markdown report in this directory is
the committed record; rerunning the reproducer should reproduce the JSON.

## Reference Summaries

The existing OpenMM reports are committed normalized summaries over raw
reference inputs. Their raw JSON files remain under gitignored `results/` and
may come from either this repository's synthetic fail-soft script or OpenMM's
stock upstream benchmark script:

| Summary | Raw reference input | Normalized fields |
| --- | --- | --- |
| [openmm-opencl-dhfr.md](./openmm-opencl-dhfr.md) | `results/openmm-opencl-dhfr-m5max.json` from `vendors/openmm/examples/benchmarks/benchmark.py` | engine, fixture/system, atom count, timing metric, runtime, hardware, raw output path |
| [openmm-opencl-apoa1.md](./openmm-opencl-apoa1.md) | `results/openmm-opencl-apoa1-m5max.json` from `vendors/openmm/examples/benchmarks/benchmark.py` | engine, fixture/system, atom count, timing metric, runtime, hardware, raw output path |
| [openmm-opencl-amber20.md](./openmm-opencl-amber20.md) | `results/openmm-opencl-amber20-m5max.json` from `vendors/openmm/examples/benchmarks/benchmark.py` | engine, fixture/system, atom count, timing metric, runtime, hardware, raw output path |
| [m5max-reference-engines.md](./m5max-reference-engines.md) | `results/m5max-reference/manifest.json` from `scripts/benchmark_m5max_reference.py` | engine provenance, required case coverage, OpenMM rows, LAMMPS statuses, raw output paths |
| [lammps-opencl-m5max.md](./lammps-opencl-m5max.md) | `results/m5max-reference/lammps/*.json` from `scripts/benchmark_m5max_reference.py` | official input paths, style mapping, acceleration classification, loop time or blocker |

`scripts/benchmark_openmm_opencl.py` and
`scripts/benchmark_lammps_opencl.py` emit the shared normalized JSON schema
directly. When the optional reference engine, OpenCL platform, fixture, or GPU
support is unavailable, they return `status: "blocked"` with a concrete
`blocker` instead of turning reference-engine availability into a routine test
failure.
