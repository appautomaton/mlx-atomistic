# Slice 10 Summary: Phase 2 Regression And Minimal API Docs

Status: complete

Route: coordinator implementation with read-only docs discovery and independent spec/code-quality reviews.

Implemented:
- Exported `PMEConfig` from the root package and covered the export.
- Updated minimal docs to describe Phase 2 parser entry points, RB torsion support, and PME assignment orders `2`, `4`, and `5`.
- Removed stale docs claims that all PME/Ewald and AMBER/CHARMM/GROMACS parser work is out of scope.
- Clarified reduced-unit low-level kernels versus physical-unit prepared artifacts.
- Hardened public `PMEConfig` finite-value validation.
- Fixed the tiny GPCRmd AMBER fixture metadata so restart box values are recognized as periodic box data.

Changed paths:
- `README.md`
- `docs/molecular-mechanics.md`
- `docs/production-md.md`
- `docs/real-mm-core.md`
- `docs/units.md`
- `src/mlx_atomistic/__init__.py`
- `src/mlx_atomistic/pme.py`
- `tests/test_forcefields.py`
- `tests/test_pme.py`
- `tests/test_gpcrmd_registry.py`

Verification:
- `uv run pytest tests/test_pme.py -k "invalid_mesh_settings or non_finite_public_values or non_finite_charges"` passed outside the sandbox: `9 passed, 27 deselected`.
- `uv run pytest tests/test_forcefields.py -k "pme or rb"` passed outside the sandbox: `15 passed, 20 deselected`.
- `uv run pytest tests/test_gpcrmd_registry.py -k "tiny_amber or runtime_benchmark_writes_json_csv"` passed outside the sandbox: `3 passed, 20 deselected`.
- `uv run pytest` passed outside the sandbox: `636 passed`.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts` passed.
- `git diff --check` passed.

Reviews:
- Spec review: `APPROVED`.
- Code quality review: `APPROVED` after fixing public PME validation and docs consistency issues.

Concerns:
- Runtime tests require Metal access; sandboxed MLX tests fail on device loading, while outside-sandbox gates pass.
