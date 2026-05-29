# Benchmark Context and Current Surface

This detail file supports `SPEC.md` for `2026-05-22-performance-audit-harness-hardening`.

## Current Repo Evidence

### MLX Product Benchmark Surface

- `src/mlx_atomistic/benchmarks/lj_md.py`: LJ MD benchmark with JSON/CSV support.
- `src/mlx_atomistic/benchmarks/md_performance.py`: full MD benchmark with backend policy, neighbor, cadence, synchronization, memory, finite-output, and throughput fields.
- `src/mlx_atomistic/benchmarks/md_acceleration.py`: neighbor/force split benchmark with backend and representation metadata.
- `src/mlx_atomistic/benchmarks/cadence_sensitivity.py`: reporting/synchronization cadence sensitivity benchmark.
- `src/mlx_atomistic/benchmarks/mm_force_terms.py`: focused MM force-term rows for bonded autodiff, neighbor-list build, LJ pair evaluation, direct Coulomb, combined nonbonded, and constraints.
- `src/mlx_atomistic/benchmarks/pme_performance.py`: PME stage profiling against a prepared parity fixture, with blocked status when fixture data is absent.
- `src/mlx_atomistic/benchmarks/stability.py`: NVE/NVT stability diagnostics.
- `src/mlx_atomistic/benchmarks/validation_gauntlet.py`: finite-difference force validation CLI.
- `tests/test_benchmarks.py`: smoke tests for benchmark JSON/CSV schemas, blocked reference paths, cadence fields, PME blocked payloads, and DFT benchmark scripts.

### Existing Reference-Engine Surface

- `pyproject.toml` keeps OpenMM and LAMMPS in the `dev` dependency group, not `project.dependencies`.
- `pyproject.toml` configures LAMMPS as `no-binary-package = ["lammps"]` and sets CMake GPU/OpenCL options through `tool.uv.config-settings-package`.
- `scripts/benchmark_openmm_opencl.py` is a standalone OpenMM/OpenCL reference benchmark, intentionally outside the package runtime path.
- No LAMMPS reference benchmark script is currently visible under `scripts/`.
- `docs/benchmarks/README.md` defines engine labels: `mlx_atomistic`, `openmm-reference`, and `lammps-reference`.

## User Concerns Captured For Framing

- Reference engines must be tested in a clean boundary: OpenMM and LAMMPS are reference/dev surfaces, not product runtime dependencies.
- The spec must decide whether reference benchmarks use the existing `uv` dev group or a separately documented reference setup. Current repo evidence supports `uv --group dev` as the starting point, with fail-soft benchmark behavior when a reference engine or platform is unavailable.
- Existing MLX performance tracking is non-empty, but the current surface does not yet clearly cover recent Phase 3 features such as virtual sites, TIP4P-Ew, GBSA/OBC, soft-core lambda scaling, and replica exchange.

## External Benchmark Context

- OpenMM benchmark discussions and public result tables commonly report `ns/day`, use named systems such as DHFR, ApoA1, Cellulose, and STMV, and record platform, precision, constraints, hydrogen mass, timestep, ensemble, and backend.
- LAMMPS official benchmarks cover LJ liquid, polymer, metal/EAM, granular, and rhodopsin protein systems. The official page notes many historical numbers are old but still useful for relative strong/weak scaling behavior. LAMMPS reports include atoms, timesteps, neighbor counts, precision/backend/package context, and loop time or atom-timestep metrics.
- OpenBenchmarking.org has a maintained LAMMPS test profile updated in 2026. Its current public configuration reports `ns/day` for Rhodopsin Protein, 20k atoms, and 61k atoms, with run repetition and standard-deviation reporting.
- MLX benchmark practice in public repos focuses on operation-level timing across Apple Silicon devices and differentiates MLX GPU, MLX CPU, MPS, CPU, CUDA, and compiled variants. For this project, MLX context mainly supports disciplined runtime/device metadata and synchronization-aware timing, not direct atomistic performance targets.

## Research Sources

- LAMMPS official benchmarks: `https://www.lammps.org/bench.html`
- OpenBenchmarking LAMMPS profile: `https://openbenchmarking.org/test/pts/lammps`
- OpenMM benchmark issue context: `https://github.com/openmm/openmm/issues/4854`
- OpenMM historical DHFR benchmark page: `https://simtk.org/plugins/moinmoin/openmm/BenchmarkOpenMMDHFR`
- MLX operation benchmark repo: `https://github.com/TristanBilot/mlx-benchmark`

## Benchmark Design Implications

- Use `ns/day` for end-to-end MD comparisons when timestep and simulated time are meaningful.
- Use `steps/s`, `ms/eval`, neighbor counts, memory, synchronization/materialization counts, and blocked-status payloads for diagnostic MLX rows.
- Keep reference-engine comparisons opt-in and fail-soft so local development and CI do not depend on OpenMM/OpenCL or LAMMPS/OpenCL availability.
- Separate routine developer checks from larger local Apple Silicon performance runs.
- Require benchmark outputs to include enough provenance to compare runs: hardware, MLX/runtime version, engine label, command, config, commit when available, and raw output path.
