# Benchmarks

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
| [openmm-opencl-dhfr.md](./openmm-opencl-dhfr.md) | openmm-reference | DHFR (23k atoms) | OpenCL | Apple M5 Max |
| [openmm-opencl-apoa1.md](./openmm-opencl-apoa1.md) | openmm-reference | ApoA1 (92k atoms) | OpenCL | Apple M5 Max |
| [openmm-opencl-amber20.md](./openmm-opencl-amber20.md) | openmm-reference | Cellulose (409k) + STMV (1.07M atoms) | OpenCL | Apple M5 Max |

The index is ordered by system size, smallest first, so the scaling story
reads top-to-bottom.

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
