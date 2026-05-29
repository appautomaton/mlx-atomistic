# Slice 9: CI Pipeline and Version Bump

## Status: DONE

## Summary

Added GitHub Actions CI, bumped project version from `0.1.0` to `0.2.0`, and replaced README scattered doc references with a concise documentation index.

## Files Changed

- `.github/workflows/ci.yml`: New CI workflow running `uv sync`, `ruff check`, and `pytest` on PRs and pushes to main/master.
- `pyproject.toml`: Minor version bump to `0.2.0`.
- `README.md`: Documentation index linking core MD/DFT/runtime docs.

## Verification

- `.github/workflows/ci.yml` exists.
- `pyproject.toml` contains `version = "0.2.0"`.
- `README.md` links the docs index.
- `uv run ruff check src tests scripts && uv run pytest`: ruff passed, `736 passed`.
- `git diff --check`: passed.

## Unresolved Risks

- CI cannot be executed on GitHub from this local environment; workflow syntax and local commands were validated.
