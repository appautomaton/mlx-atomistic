# Slice 008 Spec Review

- Slice: PME Readiness Gate For GPCRmd
- Status: approved after targeted fixes

## Findings

- Default GPCRmd PME is fail-closed because the current PME backend is not production-executable.
- The explicit short-range prototype route is exposed through the API and forwarded to the runner.
- Prototype runs force cutoff electrostatics at runtime and write non-production prototype metadata.
- Benchmark completed rows report the actual runnable request as `short-range-prototype`.

## Requested Fixes

- Pass `args.electrostatics` from API into `run_gpcrmd_mlx` so explicit prototype requests do not default to PME.
- Keep tiny trajectory-producing callers and benchmark callers explicit about prototype electrostatics.
- Ensure prototype mode does not execute artifact-declared PME.

## Evidence

- `src/mlx_atomistic/pme.py`: readiness validates executable backend, mesh shape, alpha, cutoff, neutrality, box, exclusions, 1-4 corrections, and explicit exceptions.
- `src/mlx_atomistic/prep/runner.py`: electrostatics readiness is checked before dynamics; PME only becomes production-ready if readiness status is `ready`.
- `src/mlx_atomistic/prep/runner.py`: `short-range-prototype` forces runtime cutoff electrostatics and marks output as `short_range_electrostatics_prototype` with `production_ready=false`.
- `src/mlx_atomistic/prep/`: `--electrostatics {pme,short-range-prototype}` is exposed and forwarded.
- `tests/test_production_artifacts.py`: verifies default PME blockers and prototype cutoff runtime behavior for PME artifacts.
- Required command passed: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_pme.py tests/test_production_artifacts.py -k "pme or electrostatics or gpcrmd or blocker"`: 39 passed, 15 deselected.
