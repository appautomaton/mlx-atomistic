# Release checklist

This checklist keeps the PyPI release focused on the scientific Python package.
The repository also contains docs, site, notebooks, tests, and reference tooling,
but the PyPI artifacts should ship only `mlx_atomistic` plus package metadata.

## Local gates

Run these from the repository root on an Apple Silicon machine with usable Metal:

```bash
RELEASE_DIST=/tmp/mlx-atomistic-release-dist
rm -rf "$RELEASE_DIST"
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv lock --check
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv sync --locked --no-default-groups --extra prep --group test
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --locked --no-default-groups --group test ruff check src tests scripts
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --no-project --with griffe --python 3.13.12 python scripts/gen_api_docs.py
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --locked --no-default-groups --group test python -m pytest -m "not slow and not integration and not reference and not data and not gpu"
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --locked --no-default-groups --group test python -m pytest -m "not reference and not data and not gpu" --cov=mlx_atomistic --cov-report=term-missing --cov-fail-under=80
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv build --out-dir "$RELEASE_DIST"
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --no-project --with twine twine check "$RELEASE_DIST"/*
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --no-project --python 3.13.12 python scripts/check_dist_contents.py "$RELEASE_DIST"/*
```

The full pytest suite requires a local Apple Silicon runtime with stable MLX
execution. GitHub-hosted macOS runners are used only for lint, package-boundary,
documentation, build, and publish checks because hosted MLX runtime tests can
abort inside the MLX backend.

Reference-engine and vendor-data validation is not a PyPI gate. Run it only
after explicitly provisioning the local reference surfaces:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --locked --no-default-groups --group dev python -m pytest --run-reference -m reference
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --locked --no-default-groups --group dev python -m pytest --run-data -m data
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --locked --no-default-groups --group test python -m pytest --run-gpu -m gpu
```

## Artifact inspection

Confirm the wheel contains the package, metadata, license, entry points, and the
`py.typed` marker:

```bash
unzip -l /tmp/mlx-atomistic-release-dist/mlx_atomistic-*.whl
```

Confirm the source distribution excludes monorepo surfaces such as `.github/`,
`site/`, `notebooks/`, `tests/`, `scripts/`, `docs/`, `AGENTS.md`, `CLAUDE.md`,
and `uv.lock`. Hatchling may still include the root `.gitignore` as sdist build
provenance:

```bash
tar -tzf /tmp/mlx-atomistic-release-dist/mlx_atomistic-*.tar.gz
```

Install the wheel outside the checkout and verify the import:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --no-project --isolated --with /tmp/mlx-atomistic-release-dist/mlx_atomistic-*-py3-none-any.whl python -c "import mlx_atomistic as ma; print(ma.__version__)"
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
