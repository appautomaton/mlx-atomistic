# EVIDENCE INDEX: MD Engine Capability Gap Matrix

## Scope

This index records the live repo surfaces used to classify `mlx_atomistic` MD
engine capability. OpenMM, GROMACS, and LAMMPS are comparison/reference systems,
not product dependencies.

Generated trajectories, benchmark JSON/CSV, and large science outputs remain
under ignored `results/` paths or temporary directories. This audit did not
create trajectory outputs.

## Status Vocabulary

- `implemented`: source and tests expose a working capability in the current
  product surface.
- `partial`: source exposes a prototype, gated path, limited path, or lower-level
  primitive but not full production coverage.
- `missing`: no product-facing implementation surface was found.
- `unverified`: source suggests capability, but the audit did not find enough
  tests, executable probes, or parity evidence to classify it as working.

## Local Executability

| Engine | Current audit use | Local evidence |
| --- | --- | --- |
| OpenMM | Executable reference/parity engine | `uv run python -c ...` reports `8.5.1.dev-f7fa0c2` and platforms `['Reference', 'CPU', 'OpenCL']`. `scripts/benchmark_openmm_opencl.py` and `scripts/run_openmm_gpcrmd_*.py` are present. |
| GROMACS | Source-only reference | `command -v gmx` returned no path. `vendors/gromacs/` is present and ignored; no build was run. |
| LAMMPS | Reference/runtime-adjacent | `command -v lmp` returned no path; `uv run python -c "import importlib.util; ..."` found the `lammps` Python module. `STATUS.md` records a `uv` local build with GPU package support, but sandboxed MPI/runtime checks are out of scope here. |

## Product Runtime Surfaces

| Track | Inspected surfaces |
| --- | --- |
| T1 Core MD physics | `src/mlx_atomistic/md.py`, `constraints.py`, `pme.py`, `nonbonded.py`, `core.py`, `cell_list.py`, `protocols.py`; tests `test_md.py`, `test_nve.py`, `test_nvt.py`, `test_pme.py`, `test_ewald_reference.py`, `test_protocols.py`, `test_virial_pressure.py`, `test_core.py`, `test_neighbors.py`; docs `production-md.md`, `real-mm-core.md`, `validation-and-performance.md`. |
| T2 Force-field/artifact coverage | `src/mlx_atomistic/artifacts.py`, `forcefields.py`, `charmm_terms.py`, `prep/schema.py`, `prep/topology_import.py`, `prep/production_pocket.py`, `prep/t4l_benzene.py`, `prep/solvated_example.py`, `prep/gpcrmd.py`; tests `test_forcefields.py`, `test_charmm_terms.py`, `test_production_artifacts.py`, `test_mlx_prep.py`. |
| T3 Runtime production usability | `src/mlx_atomistic/io.py`, `trajectory_adapters.py`, `prep/runner.py`, `prep/notebook.py`; tests `test_trajectory_adapters.py`, `test_mlx_prep.py`, `test_diagnostics.py`, `test_runtime.py`; docs `runtime-boundaries.md`, `production-md.md`, `real-mm-core.md`. |
| T4 Validation/parity | `src/mlx_atomistic/validation.py`, `src/mlx_atomistic/benchmarks/`, `scripts/benchmark_openmm_opencl.py`, `scripts/run_openmm_gpcrmd_preview.py`, `scripts/run_openmm_gpcrmd_charmm_md.py`; tests `test_validation.py`, `test_benchmarks.py`, `test_nonbonded_acceleration.py`, `test_gpcrmd_registry.py`; docs `docs/benchmarks/*.md`. |
| T5 Performance/backend | `src/mlx_atomistic/nonbonded.py`, `cell_list.py`, `pme.py`, `benchmarks/md_acceleration.py`, `benchmarks/md_performance.py`; OpenMM, GROMACS, and LAMMPS vendor source trees listed below. |
| T6 Prep/workflow | `src/mlx_atomistic/prep/`, notebooks/docs references, `runtime-boundaries.md`, `production-md.md`; tests `test_mlx_prep.py`, `test_gpcrmd_registry.py`, `test_ligand_receptor_motion.py`. |

## Reference Engine Surfaces

| Engine | Inspected surfaces | Why useful |
| --- | --- | --- |
| OpenMM | `vendors/openmm/openmmapi/include/openmm`, `vendors/openmm/openmmapi/src`, `vendors/openmm/platforms/opencl`, `vendors/openmm/platforms/common`, `vendors/openmm/wrappers/python/openmm/app` | Reference for `System`, `Context`, `State`, `Force`, integrators, PME/nonbonded, barostats, virtual sites, reporters, checkpoints, and Python app workflow. |
| GROMACS | `vendors/gromacs/CMakeLists.txt`, `vendors/gromacs/src/gromacs/nbnxm`, `vendors/gromacs/src/gromacs/ewald`, `vendors/gromacs/src/gromacs/fft`, `vendors/gromacs/src/gromacs/mdrun`, `vendors/gromacs/docs/user-guide/mdrun-performance.rst` | Reference for mature NBNXM pair kernels, GPU backend selection, PME/FFT split, GPU-resident execution, checkpoint/restart, and performance tuning. |
| LAMMPS | `vendors/lammps/src`, `vendors/lammps/src/GPU`, `vendors/lammps/lib/gpu`, `vendors/lammps/src/EXTRA-DUMP` | Reference for modular fixes/integrators, NPT/NVT implementations, PPPM/kspace, GPU/OpenCL paths, and DCD/XTC dump support. |

## Runnable Product Probes Available

- `uv run pytest tests/test_md.py tests/test_nve.py tests/test_nvt.py`
- `uv run pytest tests/test_pme.py tests/test_ewald_reference.py`
- `uv run pytest tests/test_production_artifacts.py tests/test_mlx_prep.py`
- `uv run pytest tests/test_validation.py tests/test_benchmarks.py`
- `uv run python -m mlx_atomistic.benchmarks.validation_gauntlet --json`
- `uv run python -m mlx_atomistic.benchmarks.stability --json`
- `uv run python -m mlx_atomistic.benchmarks.md_performance --json`
- `uv run python scripts/benchmark_openmm_opencl.py ...`

The audit did not run the full benchmark/test matrix because the requested slice
is classification and backlog, not fresh trajectory generation.

## Evidence Notes

- `src/mlx_atomistic/pme.py` sets `PME_PRODUCTION_EXECUTABLE = False`; therefore
  PME is classified as `partial`, not absent.
- `src/mlx_atomistic/protocols.py` fails closed for NPT/barostat requests before
  integration; therefore NPT/barostat is classified as `missing` at product level.
- `src/mlx_atomistic/charmm_terms.py` and `prep/topology_import.py` expose CHARMM
  CMAP/NBFIX surfaces; therefore CHARMM is `partial/unverified`, not absent.
- `src/mlx_atomistic/io.py` has `restart_state_from_trajectory(...)`; therefore
  restart is `partial`, not absent. Runner-level checkpoint/restart remains absent.
- `vendors/` is reference-only by project guidance and `docs/runtime-boundaries.md`.
