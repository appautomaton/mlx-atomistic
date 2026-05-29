# Slice 003 Summary: PME Runtime Assignment Orders 4 And 5

## Status

Complete.

## Route

Subagent implementation with spec review and code-quality review.

## Acceptance

- `PMEConfig.assignment_order` accepts only `2`, `4`, and `5`.
- Charge assignment uses generalized cardinal B-spline weights and conserves charge for supported orders, including wrapped positions.
- PME reciprocal assignment, potential interpolation, field interpolation, and influence-function deconvolution all use the configured assignment order.
- Order-2 public and private CIC wrappers remain available and match the generalized order-2 path.
- PME diagnostics and readiness metadata expose the selected assignment order.
- The PME benchmark profiler now uses the generalized assignment/interpolation helpers with the runtime config order.

## Verification

- Initial sandboxed verification failed before exercising the slice because MLX could not access a Metal device: `[metal::load_device] No Metal device available`.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_pme.py tests/test_forcefields.py -k "pme"` passed outside the sandbox: `41 passed, 23 deselected`.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/pme.py src/mlx_atomistic/forcefields.py src/mlx_atomistic/benchmarks/pme_performance.py tests/test_pme.py tests/test_forcefields.py` passed.

## Review Verdicts

- Spec review: `APPROVED`.
- Code-quality review: `APPROVED`.
