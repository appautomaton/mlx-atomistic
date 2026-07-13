# Benchmark Inventory And Gap Matrix

Date: 2026-05-22

Scope: `.agent/work/2026-05-22-performance-audit-harness-hardening`.
This inventory maps the benchmark surface and tracks which Phase 3 coverage
gaps were closed by the harness-hardening change.
External OpenMM, LAMMPS, OpenBenchmarking, and MLX material is used only as
benchmark-design context, not as a pass/fail target for `mlx_atomistic`.

## Tier Rules

| Tier | Purpose | Required availability | Result location |
| --- | --- | --- | --- |
| Fast developer | Smoke-check importable MLX benchmark modules and JSON/CSV shape. | `uv run` environment only; no mandatory OpenMM, LAMMPS, OpenCL, or large fixture. | Temporary pytest paths, stdout JSON, or optional local CSV. |
| Opt-in performance | Larger Apple Silicon runs, prepared production fixtures, and reference-engine context. | Local accelerator, optional OpenMM/LAMMPS/dev reference setup, and optional gitignored fixtures. | Raw JSON/CSV under gitignored `results/`; committed summaries under `docs/benchmarks/`. |

## Current MLX Benchmark Modules

| Module | Current coverage | Test/doc evidence | Result/raw-output location | Tier | Phase 3 gap |
| --- | --- | --- | --- | --- | --- |
| `src/mlx_atomistic/benchmarks/lj_md.py` | LJ MD modes and CSV output. | `tests/test_benchmarks.py::test_lj_benchmark_csv_smoke` | Caller-provided `--csv`; stdout otherwise. | Fast developer | Does not cover virtual sites, TIP4P-Ew, GBSA/OBC, soft-core/lambda, or replica exchange. |
| `src/mlx_atomistic/benchmarks/md_performance.py` | End-to-end synthetic LJ MD throughput, neighbor policy, cadence, synchronization, finite output, and MLX runtime metadata. | `tests/test_benchmarks.py` benchmark smoke and JSON-file output coverage. | Stdout JSON/CSV or caller-provided `--json-out`; raw outputs belong in `results/`. | Fast developer now; opt-in for larger sizes. | No Phase 3 feature-specific row. |
| `src/mlx_atomistic/benchmarks/neighbor_nonbonded_parity.py` | Compact `mlx_cell_pairs` versus tiled all-pairs energy/force parity, topology semantics, candidate waste, and build/evaluation timing through caller-selected sizes. | `tests/test_benchmarks.py::test_neighbor_nonbonded_parity_command_writes_validated_row`; [scalable-neighbor-nonbonded-runtime-m5max.md](./scalable-neighbor-nonbonded-runtime-m5max.md). | Caller-provided `--out`; at-scale raw JSON under `results/scalable-neighbor-nonbonded-runtime/`. | Fast smoke; opt-in at scale. | Validates the real-space neighbor axis; PME scale remains separate. |
| `src/mlx_atomistic/benchmarks/md_acceleration.py` | Neighbor build versus force evaluation split, backend policy, pair representation, and waste counters. | `tests/test_benchmarks.py` benchmark smoke coverage. | Stdout JSON/CSV when invoked by module. | Fast developer now; opt-in for larger sizes. | No virtual-site, GBSA/OBC, soft-core/lambda, or replica-exchange overhead row. |
| `src/mlx_atomistic/benchmarks/cadence_sensitivity.py` | Reporter/evaluation cadence and synchronization/materialization counts. | `tests/test_benchmarks.py` benchmark smoke coverage. | Stdout JSON/CSV when invoked by module. | Fast developer | Can inform future replica exchange history/materialization timing, but does not cover replica exchange today. |
| `src/mlx_atomistic/benchmarks/mm_force_terms.py` | Bonded autodiff, neighbor-list build, LJ pair eval, direct Coulomb, combined nonbonded, constraints, and TIP4P-Ew virtual-site reconstruction/force redistribution micro-rows. | `tests/test_benchmarks.py::test_force_term_benchmark_includes_profile_rows` | Caller-provided `--csv`; stdout JSON with `--json`. | Fast developer | Still a microbenchmark surface, not a production-scale advanced-water workload. |
| `src/mlx_atomistic/benchmarks/phase3_physics.py` | Fast normalized rows for virtual sites, TIP4P-Ew M-site reconstruction, GBSA/OBC energy/forces and surface-area term, soft-core/lambda derivative grid, and two-replica exchange. | `tests/test_benchmarks.py::test_phase3_physics_benchmark_covers_required_feature_rows` | Caller-provided `--csv`; stdout JSON with `--json`; raw audit outputs under `results/performance-audit-harness-hardening/`. | Fast developer | Synthetic probes only; larger opt-in rows are still needed before optimization claims. |
| `src/mlx_atomistic/benchmarks/pme_performance.py` | PME stage profiling against a prepared parity fixture; blocked payload when fixture data is absent. | `tests/test_benchmarks.py` benchmark smoke coverage. | Default fixture under `results/md-engine-structural-gap-closure/pme-parity`; default raw output under `results/md-engine-structural-gap-closure/baseline/pme-profile.json`. | Opt-in performance; blocked-path smoke is fast. | Adjacent to TIP4P-Ew water work but does not benchmark TIP4P-Ew virtual-site overhead. |
| `src/mlx_atomistic/benchmarks/ewald_reference.py` | Small-system Ewald correctness/backend timing, explicitly not GPCRmd-scale PME. | `tests/test_benchmarks.py::test_ewald_reference_benchmark_json_and_csv_smoke` | Caller-provided `--csv`; stdout JSON with `--json`. | Fast developer | Does not cover soft-core/lambda derivatives or advanced water models. |
| `src/mlx_atomistic/benchmarks/stability.py` | NVE/NVT stability diagnostics for small systems. | `tests/test_benchmarks.py::test_stability_cli_json_and_csv_smoke` | Caller-provided `--csv`; stdout JSON with `--json`. | Fast developer | Could catch stability impacts after Phase 3 rows exist, but no Phase 3 timing row today. |
| `src/mlx_atomistic/benchmarks/validation_gauntlet.py` | Finite-difference force validation cases. | `tests/test_benchmarks.py::test_validation_gauntlet_cli_json_and_csv` | Caller-provided `--csv`; stdout JSON with `--json`. | Fast developer | Validation-oriented, not a performance row for Phase 3 features. |
| `src/mlx_atomistic/benchmarks/schema.py` | Shared normalized benchmark fields and default command helpers. | Imported by current benchmark modules and exercised by `tests/test_benchmarks.py`. | N/A helper module. | Fast developer support | New benchmark rows should use this helper to keep report joins simple. |
| `src/mlx_atomistic/benchmarks/gpcrmd_runtime.py` | Shared GPCRmd runtime reporting helpers for output directory size, resident memory, PME mesh summary, and diagnostic reductions. | Consumed by the tested `mlx_atomistic.prep.gpcrmd_benchmark` function and module CLI. | Helper-only; GPCRmd benchmark consumers write JSON/CSV under caller-provided `results/` paths. | Opt-in performance support | Reporting support; real GPCRmd execution still requires gitignored fixture data. |
| DFT benchmark modules: `dft_scf.py`, `dft_operator.py`, `dft_pseudopotential.py`, `dft_geometry.py`, `dft_nonlocal.py`, `dft_solver.py`, `dft_spin_kpoints.py`, `dft_relaxation.py` | DFT operation and solver smoke timings. | DFT smoke tests in `tests/test_benchmarks.py`. | Caller-provided `--csv`; stdout JSON with `--json`. | Fast developer | Out of the Phase 3 MD physics gap set. |

## Reference Scripts And Docs

| Surface | Current coverage | Test/doc evidence | Result/raw-output location | Tier | Boundary |
| --- | --- | --- | --- | --- | --- |
| `scripts/benchmark_openmm_opencl.py` | Synthetic OpenMM/OpenCL LJ reference benchmark with blocked payload for unavailable platform. | `tests/test_benchmarks.py::test_openmm_opencl_unavailable_platform_non_json_does_not_crash`; docs under `docs/benchmarks/openmm-opencl-*.md`. | Caller-provided `--csv`; raw OpenMM report examples under `results/openmm-opencl-*.json`. | Opt-in reference; blocked smoke is fast. | `openmm-reference`, design context only. |
| `scripts/run_openmm_mlx_parity.py` | OpenMM versus MLX parity workflow support. | Existing parity tests outside this slice; not a benchmark smoke in `tests/test_benchmarks.py`. | Local run artifacts under `results/` when invoked. | Opt-in reference/context | Reference parity context, not product runtime dependency. |
| `scripts/run_openmm_mlx_npt_parity.py` | NPT parity workflow support. | Existing NPT/parity tests outside this slice. | Local run artifacts under `results/` when invoked. | Opt-in reference/context | Reference parity context, not product runtime dependency. |
| `scripts/run_openmm_production_md_reference.py` | Production MD reference command surface. | Existing production reference tests outside this slice. | Local run artifacts under `results/` when invoked. | Opt-in reference/context | OpenMM context only; not a pass/fail throughput target. |
| `scripts/run_mlx_production_md_probe.py` | MLX production-probe command surface. | Existing production probe tests outside this slice. | Local run artifacts under `results/` when invoked. | Opt-in performance | MLX product probe, but not a routine fast gate. |
| `scripts/benchmark_lammps_opencl.py` | Synthetic LAMMPS/OpenCL reference benchmark with normalized ok or blocked payloads. | `tests/test_benchmarks.py::test_lammps_opencl_reference_payload_is_normalized`; docs command matrix and baseline audit. | Caller-provided `--csv`; stdout JSON with `--json`; raw audit output under `results/performance-audit-harness-hardening/lammps-fast.json`. | Opt-in reference; blocked smoke is fast. | `lammps-reference`, design context only. |
| `docs/benchmarks/README.md` | Engine labels, file template, index, external input policy, raw-output policy. | This inventory is linked from README. | Committed Markdown in `docs/benchmarks/`. | Documentation | Defines `mlx_atomistic`, `openmm-reference`, and `lammps-reference`. |
| `docs/benchmarks/openmm-opencl-dhfr.md`, `openmm-opencl-apoa1.md`, `openmm-opencl-amber20.md` | OpenMM OpenCL summary reports for DHFR, ApoA1, Cellulose, and STMV on Apple M5 Max. | Indexed from `docs/benchmarks/README.md`. | Raw JSON paths named under gitignored `results/`. | Opt-in reference documentation | External/reference comparison only. |

## Phase 3 Coverage Gaps

| Feature | Current implementation/test evidence | Current benchmark placement | Remaining benchmark gap |
| --- | --- | --- | --- |
| virtual sites | `tests/test_virtual_sites.py` covers virtual-site position reconstruction, force redistribution, artifact round trip, runner propagation, and simulation configuration plumbing. | `phase3_physics.py` covers reconstruction and force redistribution; `mm_force_terms.py` adds synchronized TIP4P-Ew micro-rows. | Larger opt-in advanced-water workload. |
| TIP4P-Ew | `tests/test_virtual_sites.py` covers `tip4p_ew_virtual_site`, reference geometry, prepared-system round trip, and artifact build exposure. | `phase3_physics.py` emits a TIP4P-Ew M-site row; `mm_force_terms.py` emits TIP4P-Ew reconstruction and redistribution rows. | Couple the row to a larger nonbonded/PME workload before optimization claims. |
| GBSA/OBC | `tests/test_gbsa.py` covers GBSA surface-area energy, OBC force finite-difference behavior, OpenMM OBC reference energy, artifact loading, and save/load parameters. | `phase3_physics.py` emits OBC energy/force and surface-area rows. | Scaling row over larger atom counts. |
| soft-core/lambda | `tests/test_soft_core.py` covers finite overlap, endpoint equivalence, finite-difference `energy_forces_dlambda`, wrapper delegation, artifact metadata, and fail-closed non-cutoff electrostatics. | `phase3_physics.py` emits an `energy_forces_dlambda` lambda-grid row. | Larger lambda-grid sweep. |
| replica exchange | `tests/test_replica_exchange.py` covers Metropolis probability, adjacent swaps, lambda-scaled Hamiltonians, odd/even pairing, metadata validation, and unsupported runtime inputs. | `phase3_physics.py` reports per-replica throughput, swap counts, acceptance rate, and history materialization count. | Larger opt-in multi-replica workload. |

## External Context Caveats

| Context source | Useful design signal | Caveat |
| --- | --- | --- |
| OpenMM | Public reports commonly use `ns/day` on named systems such as DHFR, ApoA1, Cellulose, and STMV, with platform, precision, timestep, constraints, hydrogen mass, ensemble, and cutoff/PME settings recorded. | OpenMM numbers are reference context only. They are not direct pass/fail targets for MLX because hardware, backend, precision, and engine semantics differ. |
| LAMMPS | Official benchmark families cover LJ liquid, polymers, metals/EAM, granular systems, and protein/rhodopsin-style systems with atom counts, timesteps, packages, precision/backend, and loop-time style metrics. | LAMMPS coverage should stay opt-in and fail-soft; historical public numbers are context for scaling behavior, not target thresholds. |
| OpenBenchmarking | The LAMMPS profile records repeatable run metadata and reports `ns/day` for rhodopsin/protein-sized systems with variance reporting. | OpenBenchmarking helps shape provenance and repetition fields; it is not an apples-to-apples MLX acceptance gate. |
| MLX | Public MLX benchmark practice emphasizes operation-level timing, Apple Silicon device/runtime metadata, synchronization-aware measurements, and CPU/GPU/backend distinctions. | MLX context informs measurement hygiene for `mlx_atomistic`; it does not provide atomistic throughput targets by itself. |

## Slice 1 Findings

- The fast benchmark gate is currently centered on `tests/test_benchmarks.py`.
- Current committed benchmark docs include OpenMM reference reports, the benchmark README, this inventory, and the baseline audit report.
- Raw benchmark outputs should remain under gitignored `results/`; committed Markdown in `docs/benchmarks/` should summarize reproducible rows and cite raw output paths.
- The Phase 3 named features now have normalized fast benchmark rows; remaining gaps are larger opt-in workloads for optimization validation.
