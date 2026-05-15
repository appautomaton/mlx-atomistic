# Slice 7: Production DCD/XTC Output

## Status

Complete.

## What changed

- Added `write_mdtraj_trajectory(...)` to the trajectory adapter layer.
- Added `trajectory_record_from_result(...)` to build a native record from an
  in-memory simulation result without forcing a NPZ reload.
- Added `prep.run_mlx(..., dcd_out=..., xtc_out=..., topology_out=...)`.
- `run_mlx` reuses prepared `view.pdb` when available, or writes one beside the
  requested output for in-memory prepared systems.
- Missing MDTraj writer dependency now raises the existing
  `OptionalTrajectoryDependencyError` with the `uv sync --extra viz` guidance.

## Evidence

- `uv run pytest tests/test_trajectory_adapters.py tests/test_runner_outputs.py -q`
  passed: `7 passed`.
- `uv run pytest tests/test_mlx_prep.py tests/test_checkpoint_restart.py tests/test_runtime_reporters.py tests/test_trajectory_adapters.py tests/test_runner_outputs.py -q`
  passed: `39 passed`.
- `uv run ruff check src/mlx_atomistic/trajectory_adapters.py src/mlx_atomistic/io.py src/mlx_atomistic/prep/runner.py src/mlx_atomistic/__init__.py tests/test_trajectory_adapters.py tests/test_runner_outputs.py`
  passed.

## Decision

The production output polish requested in this slice is complete. NPZ remains
the native diagnostic format; DCD/XTC are first-class runner outputs when the
optional viz stack is installed.
