# Release checklist

This checklist keeps the PyPI release focused on the scientific Python package.
The repository also contains docs, site, notebooks, tests, and reference tooling,
but the PyPI artifacts should ship only `mlx_atomistic` plus package metadata.

## Local gates

Run these from the repository root on an Apple Silicon machine with usable Metal:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv build
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --with twine twine check dist/*
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --group test ruff check src tests scripts
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --no-project --with griffe --python 3.13 python scripts/gen_api_docs.py
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --group test pytest -m "not slow and not integration and not reference and not data and not gpu"
```

The full pytest suite requires Metal access. A headless or sandboxed macOS
session may fail with `No Metal device available`; rerun on the host if that
happens.

Reference-engine and vendor-data validation is not a PyPI gate. Run it only
after explicitly provisioning the local reference surfaces:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --group dev python -m pytest --run-reference -m reference
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --group dev python -m pytest --run-data -m data
```

## Artifact inspection

Confirm the wheel contains the package, metadata, license, entry points, and the
`py.typed` marker:

```bash
unzip -l dist/mlx_atomistic-*.whl
```

Confirm the source distribution excludes monorepo surfaces such as `.github/`,
`site/`, `notebooks/`, `tests/`, `scripts/`, `docs/`, `AGENTS.md`, `CLAUDE.md`,
and `uv.lock`. Hatchling may still include the root `.gitignore` as sdist build
provenance:

```bash
tar -tzf dist/mlx_atomistic-*.tar.gz
```

Install the wheel outside the checkout and verify the import:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --isolated --with dist/mlx_atomistic-*-py3-none-any.whl python -c "import mlx_atomistic as ma; print(ma.__version__)"
```

## Publishing

PyPI Trusted Publishing is configured through a pending publisher for:

- Repository: `appautomaton/mlx-atomistic`
- Workflow: `workflow.yml`
- Environment: `pypi`
- Project: `mlx-atomistic`

Publishing happens when a GitHub release is published. The pending publisher does
not reserve the project name until the first successful publish, so avoid a long
delay between final validation and the first release.
