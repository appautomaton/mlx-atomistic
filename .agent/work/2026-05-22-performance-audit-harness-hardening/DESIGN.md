# DESIGN: Performance Audit and Harness Hardening

## Architecture Approach

The benchmark surface should be organized around three durable layers:

1. Shared benchmark metadata and row normalization in `src/mlx_atomistic/benchmarks/`.
2. Product benchmark entrypoints for MLX-owned behavior in `src/mlx_atomistic/benchmarks/`.
3. Reference-engine scripts in `scripts/`, with OpenMM and LAMMPS kept outside package runtime dependencies.

This keeps product performance tracking importable through `mlx_atomistic`, while OpenMM/LAMMPS remain opt-in command surfaces that can report `blocked` without breaking local development.

## Normalized Output Contract

Every benchmark payload should expose the same top-level fields where applicable:

- `schema_version`
- `engine`
- `benchmark_name`
- `fixture`
- `system_name`
- `atom_count`
- `step_count` or `evaluation_count`
- timing metric fields such as `steps_per_s`, `ns_per_day`, or `ms_per_eval`
- `hardware`
- `runtime` or reference-engine version metadata
- `finite`
- `status`
- `blocker`
- `raw_output_path`
- `command`
- `commit`

Rows may keep benchmark-specific diagnostic fields, but the shared fields must be present so reports can compare MLX, OpenMM, and LAMMPS rows without per-script adapters.

## Benchmark Tiers

Fast developer tier:
- Small synthetic systems.
- No mandatory OpenMM, LAMMPS, OpenCL, or large downloaded fixture.
- Covered by `uv run pytest tests/test_benchmarks.py`.

Opt-in performance tier:
- Larger Apple Silicon runs, prepared production fixtures, and reference-engine context.
- Writes raw outputs under gitignored `results/`.
- Fails soft with `status: blocked` and a concrete `blocker` when a reference engine, fixture, platform, or accelerator is unavailable.

## Script Placement

- Keep package benchmark modules under `src/mlx_atomistic/benchmarks/`.
- Keep standalone reference-engine launchers under `scripts/`.
- Commit summary reports under `docs/benchmarks/`.
- Keep raw benchmark outputs under `results/` and reference them from committed docs.

## Multi-Agent Ownership

The implementation should split into disjoint write scopes:

- Audit/documentation agent: benchmark inventory and command matrix docs.
- Schema/harness agent: shared normalization helpers and product benchmark payload updates.
- Phase 3 coverage agent: MLX benchmark rows for virtual sites, TIP4P-Ew, GBSA/OBC, soft-core/lambda, and replica exchange.
- Reference-engine agent: OpenMM normalization and LAMMPS fail-soft reference path.
- Synthesis owner: baseline run artifacts, ranked backlog, and final verification.

The schema slice should complete before other code slices that emit normalized rows. After that, Phase 3 coverage and reference-engine work are parallel-safe if their write sets stay disjoint.
