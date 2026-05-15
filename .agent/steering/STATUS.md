---
active_change: md-engine-gap-closure
stage: execute
---

# Status

## Current Change

- active change: `md-engine-gap-closure`
- current stage: `execute`

## What Is True Now

- Office-hours direction is approved: keep the repo lean by making `mlx_atomistic` the product runtime and treating OpenMM, LAMMPS, and `vendors/` as reference or validation surfaces only.
- OpenMM in the project `.venv` was removed and reinstalled through `uv`; it resolves from the PyPI `openmm==8.5.1` macOS arm64 wheel and exposes `Reference`, `CPU`, and `OpenCL` platforms.
- LAMMPS is configured as a `uv` local build from the upstream PyPI source package with `PKG_GPU=ON`, `GPU_API=opencl`, and `GPU_PREC=single`; the installed runtime reports `has_GPU=True`.
- OpenMM provenance was rechecked as `uv`/PyPI `openmm==8.5.1` with platforms `Reference`, `CPU`, and `OpenCL`.
- LAMMPS provenance was rechecked as a `uv` local build with runtime version `20250722` and GPU package support `True`.
- `mlx_atomistic.prep` is now the only preparation/import package surface; the legacy package shim and prep console command have been removed.
- Current setup docs use the full notebook/prep/viz environment: `uv sync --extra notebook --extra prep --extra viz --group dev`.
- Local `main` tracks `origin/main` at `https://github.com/appautomaton/mlx-atomistic.git`.
- Historical `.agent/work/**` records are retained as provenance and are not expected to be rewritten for every package/command rename.
- Docs-hygiene verification passed: old prep CLI wording is absent from active docs, `OpenMM.ipynb` has no code-cell outputs, source/test/script Ruff passed, `uv lock --check` passed, and the targeted MLX regression suite reported `154 passed in 6.75s` on an approved unsandboxed run.
- The MD engine capability audit has produced `.agent/work/md-engine-capability-gap-matrix/EVIDENCE-INDEX.md`, `GAP-MATRIX.md`, and `BACKLOG.md`.
- The audit classifies PME as partial/gated, NPT/barostat as missing, CHARMM CMAP/NBFIX as partial rather than absent, and runner checkpoint/restart as partial rather than absent.
- The recommended next SPEC is `production-artifact-openmm-parity-fixture`; it should create the shared MLX/OpenMM fixture and force/energy parity harness before PME/NPT implementation.
- The active frame is now `.agent/work/md-engine-gap-closure/SPEC.md`, an umbrella gap-closure contract that keeps Phase 1 as the OpenMM parity fixture and then sequences PME, reporters/checkpoint, HMR/virtual-site policy, NPT, DCD/XTC, and performance work.
- The active frame now links `spec/mature-framework-gap-comparison.md`, which classifies mature-framework gaps as build now, validate now, or defer for this project.
- The active plan is now `.agent/work/md-engine-gap-closure/PLAN.md`; Slice 1 is the next executable unit and focuses on the production artifact OpenMM-vs-MLX parity fixture.
- Engineering review is `approved_with_risks`; the known risk is that Slice 1 crosses fixture selection, OpenMM reference construction, MLX term mapping, tests, scripts, and generated results, so execution must stay narrowly scoped and stop at the parity decision checkpoint.
- Slice 1 is complete at its decision checkpoint. The selected fixture is `amber-alanine-dipeptide-implicit`, the parity report is `results/md-engine-gap-closure/parity-fixture/openmm_mlx_parity_report.json`, and the run passed with total energy error `0.0007589909347984758` kJ/mol and no unsupported terms.
- During Slice 1, the Angstrom-space Coulomb constant was corrected from `COULOMB_CONSTANT_KJ_MOL_NM / 10` to `COULOMB_CONSTANT_KJ_MOL_NM * 10`, which was required for AMBER nonbonded parity.
- Slice 2 is complete at its decision checkpoint. PME readiness now reports the executable `mlx_fft_cic` backend for valid orthorhombic configurations, and the selected fixture passed OpenMM PME parity at `results/md-engine-gap-closure/pme-parity/openmm_mlx_parity_report.json`.
- Slice 2 PME parity metrics: total energy abs error `0.022714817898133788` kJ/mol, nonbonded component abs error `0.023510570876212` kJ/mol, force max abs error `8.634396488319567` kJ/mol/nm, and force RMS abs error `2.4919619801391475` kJ/mol/nm.
- PME virial status is explicit: finite-difference cell-strain diagnostics are available, while analytic PME virial is not implemented.
- Slice 3 is complete. Runtime reporters now observe sampled frames and
  diagnostic intervals through `ReporterEvent`, `RuntimeTraceReporter`, direct
  NVE/NVT runs, and `prep.run_mlx` production NVT without changing NPZ output.
- Slice 4 is complete. `prep.run_mlx` can write and resume runner-level
  checkpoints, and deterministic NVT restart uses seed plus RNG step cursor.
- Slice 5 is complete. Distance constraints have a focused stability check,
  HMR is accepted only when represented by declared artifact masses, and
  virtual-site/TIP4P-style models fail closed.
- Slice 6 is complete. `ensemble=NPT` with `barostat=monte_carlo` now reaches a
  supported orthorhombic NPT path, final NPT cell/checkpoint metadata is
  persisted, and the shared AMBER PME fixture passed the short OpenMM-vs-MLX
  volume comparison with ratio delta `0.0075521811451760845` under a `0.25`
  bound.
- Slice 7 is complete. `prep.run_mlx` can write DCD and XTC alongside native
  NPZ output through MDTraj, reusing or emitting `view.pdb` as topology, and
  missing optional writer dependencies fail with `uv sync --extra viz` guidance.
- Slice 8 is complete at its decision checkpoint. MLX synthetic LJ at 2000
  atoms/1000 steps measured `729.5651978507578` steps/s; OpenMM OpenCL at 2000
  atoms/10000 steps measured `19088.50935994495` steps/s on Apple M5 Max.
  The measured follow-up target is nonbonded pair/neighbor behavior plus
  diagnostic synchronization, not unscoped custom Metal work.

## Next Step

The active `md-engine-gap-closure` execution plan is complete. Decide whether
to create a focused optimization spec or commit the completed gap-closure wave.

## Open Risks

- The active OpenMM exploratory notebook must stay output-free or be archived later; the current cleanup clears outputs in place.
- Dependency metadata must stay lean without deleting useful reference-engine workflows that are still valuable for validation.
- LAMMPS runtime checks require unsandboxed execution on this machine because MPI initialization is blocked by sandbox network-interface policy.
- PME is only proven for the small orthorhombic AMBER fixture family in this
  slice. Larger systems, triclinic boxes, and analytic PME virial remain future
  scope.
- Runtime reporters currently observe production NVT only when used through
  `run_minimize_then_nvt`; equilibration reporter hooks remain future scope.
- Slice 6 is a first supported NPT proof path, not a mature long-production
  barostat implementation with analytic PME virial.
- Performance profiling was LJ-focused. PME FFT performance still needs its own
  PME-specific profile before any custom-kernel work is justified.
