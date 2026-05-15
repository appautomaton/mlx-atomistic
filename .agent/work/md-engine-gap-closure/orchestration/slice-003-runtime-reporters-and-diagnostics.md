# Slice 3: Runtime Reporters And Diagnostics

## Status

Complete.

## What Changed

- Added `ReporterEvent` and `RuntimeReporter` in `md.py`.
- Added reporter callbacks to `simulate_nve` and `simulate_nvt` for sampled
  frames and diagnostic intervals.
- Added `RuntimeTraceReporter` in `io.py` for scalar parity/diagnostic traces.
- Threaded reporters through `prep.run_mlx` and production
  `run_minimize_then_nvt` without changing integration behavior.

## Evidence

Verification:

```sh
uv run pytest tests/test_runtime_reporters.py -q
uv run pytest tests/test_mlx_prep.py tests/test_diagnostics.py tests/test_runtime_reporters.py -q
uv run ruff check src/mlx_atomistic/md.py src/mlx_atomistic/protocols.py src/mlx_atomistic/prep/runner.py src/mlx_atomistic/io.py tests/test_runtime_reporters.py
```

All verification commands passed.

## Notes

- Reporter callbacks receive MLX arrays and scalar diagnostics; native NPZ
  trajectory output remains unchanged.
- The reporter surface observes production NVT in `run_minimize_then_nvt`; it
  does not emit equilibration callbacks in this slice.
