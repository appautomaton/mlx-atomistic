# Slice 011 Summary

- Slice: Full Verification And Handoff
- Status: completed
- Execution route: direct
- Stop reason: Slice 11 has `Auto-continue: no`; execution stage is complete and ready for `auto-verify`.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests notebooks/ligand-receptor-motion`: Ruff passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest`: pytest passed with 342 tests after the neighbor-list tuning pass.

## Run Record

- Run report: `notebooks/ligand-receptor-motion/data/gpcrmd-mlx/729-proof/gpcrmd_mlx_run_report.json`.
- Trajectory: `notebooks/ligand-receptor-motion/data/gpcrmd-mlx/729-proof/trajectory.npz`.
- Status: `ran`.
- Engine: `mlx_atomistic`.
- Workflow: `run_gpcrmd_mlx`.
- Target: `gpcrmd-729-beta1-5f8u-cyanopindolol`.
- Dynamics ID: `729`.
- Steps: `10`.
- Frames: `11`.
- Electrostatics route: `short-range-prototype`.
- Production electrostatics ready: `false`.
- Nonbonded backend: `periodic_cell_list`.
- Neighbor skin: `2.5`.
- Neighbor rebuild count: `1`.
- Neighbor pair count: `48933140`.
- Estimated pair memory bytes: `391465120`.
- Elapsed wall seconds: `13.944710792042315`.
- Integration steps per second: `0.7171177767061757`.
- Pressure diagnostics: disabled for large-system finite-difference virial; finite zero pressure diagnostics are stored.
- Max constraint error A: `0.009981155395507812`.

## Notes

- The original 10-step Slice 9 command exceeded a 600-second execution window before writing a report because neighbor-list construction was still effectively too slow. The current artifact supersedes that first proof with a 10-step run using the optimized periodic cell-list route.
- Production PME remains blocked on `pme_backend_not_production_executable:current_backend=numpy_reference`.
- Generated GPCRmd data and trajectory outputs were not staged for commit. `git status --short` shows a dirty worktree with many source/artifact changes, but no staged entries.

## Next

Run `auto-verify` to check the completed change against user-visible outcomes and produce the final verification handoff.
