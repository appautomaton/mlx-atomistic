# Slice 003 Implementer: PME Runtime Assignment Orders 4 And 5

## Verdict

DONE

## Scope

- Added runtime support for PME assignment orders `2`, `4`, and `5`.
- Added generalized cardinal B-spline charge assignment and interpolation helpers.
- Preserved public and private CIC compatibility wrappers as order-2 shims.
- Threaded `PMEConfig.assignment_order` through reciprocal charge assignment, field/potential interpolation, influence-function deconvolution, diagnostics, and readiness metadata.
- Added targeted PME and nonbonded PME tests for supported orders and diagnostics.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_pme.py tests/test_forcefields.py -k "pme"` passed outside the sandbox: `41 passed, 23 deselected`.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/pme.py src/mlx_atomistic/forcefields.py src/mlx_atomistic/benchmarks/pme_performance.py tests/test_pme.py tests/test_forcefields.py` passed.
