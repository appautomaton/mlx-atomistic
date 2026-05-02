# Slice 008 Implementer

- Slice: PME Readiness Gate For GPCRmd
- Status: completed with review-requested fixes
- Route: subagent implementation

## Implementation Outcome

- Added PME production-readiness reporting and validation.
- Marked the current `numpy_reference` PME backend as not production-executable.
- Added GPCRmd electrostatics gating so default PME blocks before dynamics when production PME is not executable.
- Added explicit `--electrostatics short-range-prototype` routing and metadata.
- Forced short-range prototype runs to build cutoff electrostatics at runtime instead of executing artifact-declared PME.
- Updated tiny/prototype tests and benchmark callers to request prototype electrostatics explicitly.

## Files Changed

- `src/mlx_atomistic/pme.py`: PME readiness reporting and validation.
- `src/atomistic_prep/runner.py`: GPCRmd electrostatics pre-run gate, runtime prototype cutoff override, and report metadata.
- `src/atomistic_prep/cli.py`: explicit `--electrostatics` option and runner forwarding.
- `src/atomistic_prep/gpcrmd_benchmark.py`: benchmark rows label actual prototype execution.
- `tests/test_pme.py`: PME readiness blocker coverage.
- `tests/test_production_artifacts.py`: GPCRmd PME blocker, prototype metadata, and PME-artifact cutoff runtime regression coverage.
- `tests/test_gpcrmd_registry.py`: explicit prototype callers and benchmark-label assertions.

## Implementer Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_pme.py tests/test_production_artifacts.py -k "pme or electrostatics or gpcrmd or blocker"`: 38 passed, 15 deselected before follow-up fixes; 39 passed, 15 deselected after final verification.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_gpcrmd_registry.py::test_gpcrmd_run_mlx_cli_json_blocks_incomplete_cache_without_trajectory`: passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_gpcrmd_registry.py::test_gpcrmd_run_mlx_exports_tiny_amber_fixture_trajectory tests/test_gpcrmd_registry.py::test_gpcrmd_run_mlx_blocks_existing_trajectory_before_reimporting_different_target tests/test_gpcrmd_registry.py::test_gpcrmd_runtime_benchmark_writes_json_csv_for_tiny_fixture`: 3 passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_production_artifacts.py::test_gpcrmd_short_range_prototype_pme_artifact_runs_cutoff_not_pme tests/test_gpcrmd_registry.py::test_gpcrmd_runtime_benchmark_writes_json_csv_for_tiny_fixture`: 2 passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/pme.py src/mlx_atomistic/forcefields.py src/mlx_atomistic/nonbonded.py src/mlx_atomistic/artifacts.py src/atomistic_prep/runner.py src/atomistic_prep/cli.py src/atomistic_prep/gpcrmd_benchmark.py tests/test_pme.py tests/test_production_artifacts.py tests/test_gpcrmd_registry.py`: passed.

## Residual Concern

Production PME remains intentionally blocked. Slice 9 must run the proof route explicitly as `short-range-prototype` or report the production PME blocker.
