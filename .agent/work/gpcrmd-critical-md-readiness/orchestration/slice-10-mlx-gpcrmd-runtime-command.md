# Slice 10: MLX GPCRmd Runtime Command

## Result

- Status: completed
- Route: subagent implementer with spec and quality review
- Auto-continue: yes

## Scope

Added the `run-gpcrmd-mlx` runtime path. The command imports or loads a GPCRmd prepared artifact, runs the short MLX NVT proof protocol through `run_mlx`, and writes a run report for both completed and blocked paths.

## Files Changed

- `src/mlx_atomistic/prep/runner.py`: added `run_gpcrmd_mlx`, `GPCRMD_RUN_REPORT_NAME`, run-report payload helpers, finite diagnostic summary, and stale-output guards.
- `src/mlx_atomistic/prep/`: added `run-gpcrmd-mlx` parser and command handler.
- `tests/test_gpcrmd_registry.py`: added GPCRmd runtime tests for runnable tiny AMBER fixtures, blocked incomplete caches, and reused-output stale artifact prevention.

## Review Loop

- Implementer: `DONE`
- Spec review 1: `APPROVED`
- Quality review 1: `CHANGES_REQUESTED`
  - Issue: cache-backed runs imported into `--out` before checking whether `trajectory.npz` already existed, which could pair a new prepared artifact with an old trajectory.
  - Fix: cache-backed runs now block on existing `trajectory.npz` before import writes when `force=False`; regression coverage preserves existing prepared artifact, import report, and trajectory bytes.
- Spec review 2: `APPROVED`
- Quality review 2: `APPROVED`

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "gpcrmd and run"`
  - Result: `3 passed, 296 deselected`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_gpcrmd_registry.py`
  - Result: `25 passed`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/prep/runner.py src/mlx_atomistic/prep/ tests/test_gpcrmd_registry.py`
  - Result: `All checks passed!`
- Fixture API smoke:
  - `uv run mlx_atomistic.prep Python API run-gpcrmd-mlx ... --json`
  - Result: `status=ran`, `trajectory_written=True`, `blockers=[]`, finite positions, and `trajectory.npz` written.
- External-engine/process scan over touched runtime sources:
  - Result: no external MD engine or process-spawn calls found.

## Remaining Risks

- The selected real GPCRmd package can still block during import if required source files or unsupported parsed terms are missing. The runtime command reports those blockers and does not fabricate a trajectory.
