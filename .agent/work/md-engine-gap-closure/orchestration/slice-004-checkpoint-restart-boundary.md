# Slice 4: Checkpoint/Restart Boundary

## Status

Complete.

## What Changed

- Added `initial_step` and `initial_time` to `SimulationConfig`.
- Added deterministic Langevin RNG cursor support through
  `LangevinThermostat.rng_step_offset`.
- Added `SimulationCheckpoint`, `save_simulation_checkpoint`, and
  `load_simulation_checkpoint`.
- Added `prep.run_mlx` checkpoint output and resume input parameters.
- Kept default simulations unchanged when no checkpoint/resume options are
  provided.

## Evidence

Verification:

```sh
uv run pytest tests/test_checkpoint_restart.py -q
uv run pytest tests/test_checkpoint_restart.py tests/test_mlx_prep.py -q
uv run ruff check src/mlx_atomistic/md.py src/mlx_atomistic/io.py src/mlx_atomistic/protocols.py src/mlx_atomistic/prep/runner.py tests/test_checkpoint_restart.py tests/test_runtime_reporters.py
```

All verification commands passed.

## Notes

- The deterministic restart contract uses seed plus RNG step cursor. A resumed
  NVT segment from checkpoint matches the continuous run final state.
- `prep.run_mlx` resume skips minimization/equilibration and continues
  production from the checkpoint state.
