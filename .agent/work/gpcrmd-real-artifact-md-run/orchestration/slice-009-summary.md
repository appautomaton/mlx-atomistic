# Slice 009 Summary

- Slice: End-To-End GPCRmd Short MLX Run
- Status: completed
- Execution route: direct
- Stop reason: Slice 9 has `Auto-continue: yes`; execution can continue to Slice 10.

## What Changed

- Added a large-system pressure diagnostic opt-out for `run_mlx` so GPCRmd-scale proof runs do not evaluate finite-difference virials over 92k atoms and CHARMM CMAP terms.
- The run metadata records `pressure_diagnostics=false` and `pressure_diagnostics_reason=disabled_large_system_finite_difference_virial`.
- Generated ignored GPCRmd proof outputs under `notebooks/ligand-receptor-motion/data/gpcrmd-mlx/729-proof/`.

## Plan Correction

- Slice 8 made explicit electrostatics selection mandatory before a non-production proof run, so Slice 9 used `--electrostatics short-range-prototype`.
- Before the neighbor-list acceleration pass, the planned 10-step command exceeded a 600-second execution window before writing a report.
- Before the neighbor-list acceleration pass, a 1-step target-path proof was used as the smallest honest trajectory. That 1-step target run took about 372 seconds and wrote 2 sampled frames.
- Post-slice optimization superseded this artifact with a 10-step, 11-frame proof trajectory using threaded periodic cell-list neighbors, `skin=2.5`, and sparse preview diagnostics.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run mlx_atomistic.prep Python API run-gpcrmd-mlx --cache notebooks/ligand-receptor-motion/data/gpcrmd-cache/729 --out notebooks/ligand-receptor-motion/data/gpcrmd-mlx/729-proof --steps 10 --sample-interval 1 --dt 0.0005 --minimize-steps 0 --equilibration-steps 0 --electrostatics short-range-prototype --force`: timed out after 600 seconds before writing a report.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run mlx_atomistic.prep Python API run-gpcrmd-mlx --cache notebooks/ligand-receptor-motion/data/gpcrmd-cache/729 --out /tmp/mlx-atomistic-gpcrmd-729-proof-1step --steps 1 --sample-interval 1 --dt 0.0005 --minimize-steps 0 --equilibration-steps 0 --electrostatics short-range-prototype --force`: ran; 2 frames, 2 diagnostics, finite total energy and temperature, max constraint error `0.009981155395507812` A.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run mlx_atomistic.prep Python API run-gpcrmd-mlx --cache notebooks/ligand-receptor-motion/data/gpcrmd-cache/729 --out notebooks/ligand-receptor-motion/data/gpcrmd-mlx/729-proof --steps 1 --sample-interval 1 --dt 0.0005 --minimize-steps 0 --equilibration-steps 0 --electrostatics short-range-prototype --force`: ran; wrote `trajectory.npz` and `gpcrmd_mlx_run_report.json`.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_gpcrmd_registry.py::test_gpcrmd_run_mlx_exports_tiny_amber_fixture_trajectory tests/test_gpcrmd_registry.py::test_gpcrmd_run_mlx_blocks_existing_trajectory_before_reimporting_different_target tests/test_virial_pressure.py -q`: 10 passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/prep/runner.py src/mlx_atomistic/prep/ src/mlx_atomistic/md.py tests/test_gpcrmd_registry.py tests/test_virial_pressure.py`: passed.

## Evidence

- Run report: `notebooks/ligand-receptor-motion/data/gpcrmd-mlx/729-proof/gpcrmd_mlx_run_report.json`.
- Trajectory: `notebooks/ligand-receptor-motion/data/gpcrmd-mlx/729-proof/trajectory.npz`.
- Report status: `ran`.
- `trajectory_written`: `true`.
- `engine`: `mlx_atomistic`.
- `electrostatics_route`: `short-range-prototype`.
- `electrostatics_production_ready`: `false`.
- Current `sampled_frame_count`: `11`.
- Current `steps`: `10`.
- Current `elapsed_wall_seconds`: `13.944710792042315`.
- Current `integration_steps_per_second`: `0.7171177767061757`.
- Current `nonbonded_runtime.skin`: `2.5`.
- Current `nonbonded_runtime.rebuild_count`: `1`.
- `total_energy_finite`: `true`.
- `temperature_finite`: `true`.
- `pressure_finite`: `true`.
- `max_constraint_error_A`: `0.009981155395507812`.

## Next

Slice 10 should make the notebook consume this generated trajectory and clearly label it as a short-range prototype proof, not production PME or long-timescale GPCR dynamics.
