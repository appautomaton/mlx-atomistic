# Slice 003 Quality Review

- Status: approved after one correction
- Reviewer route: subagent

## Initial Issue

The first quality review found that `src/mlx_atomistic/artifacts.py::_validate_declared_term_arrays` only treated legacy `nbfix_pairs` as `nbfix_pair_overrides`. New compact arrays could be present while metadata omitted `nbfix_pair_overrides`, bypassing the undeclared-array guard.

## Correction

- Added `nbfix_type_pairs`, `nbfix_type_sigma`, and `nbfix_type_epsilon` to the undeclared-array guard.
- Added regression tests proving hidden compact NBFIX arrays are rejected and declared compact NBFIX artifacts load until runtime construction.

## Final Evidence

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_production_artifacts.py -k "nbfix or undeclared"`: 3 passed, 35 deselected.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_atomistic_prep.py tests/test_gpcrmd_registry.py tests/test_production_artifacts.py -k "nbfix or charmm or gpcrmd"`: 43 passed, 53 deselected.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/artifacts.py tests/test_production_artifacts.py`: passed.
- Re-review approved with no remaining issues.
