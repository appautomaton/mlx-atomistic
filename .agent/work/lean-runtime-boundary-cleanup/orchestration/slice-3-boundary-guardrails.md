# Slice 3: Boundary Guardrails And Validation Evidence

## Status

complete

## Route

direct

## Files Changed

- `tests/test_runtime_boundaries.py`: added AST-based import guardrails, dependency-surface checks, and boundary-document label checks.
- `.agent/work/lean-runtime-boundary-cleanup/VERIFY.md`: records execution and provenance evidence for the cleanup.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_runtime_boundaries.py tests/test_mlx_prep.py` passed with `32 passed in 6.31s` on an approved unsandboxed rerun.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts` passed with `All checks passed!`.
- OpenMM provenance command reported `metadata_version 8.5.1`, `installer uv`, `direct_url None`, and platforms `['Reference', 'CPU', 'OpenCL']`.
- LAMMPS runtime provenance command reported version `20250722` and GPU package support `True` on an approved unsandboxed run.
- Import scan found OpenMM imports only in the two documented preview scripts, plus one test assertion string. No OpenMM or LAMMPS import exists in `src/mlx_atomistic/`.

## Notes

- The first sandboxed pytest run failed with `No Metal device available`; the unsandboxed rerun passed.
- The first sandboxed LAMMPS runtime check failed during MPI initialization on `utun6`; the unsandboxed rerun passed.
