# Slice 10 Implementer: Phase 2 Regression And Minimal API Docs

Status: `DONE`

Route: coordinator implementation with read-only docs discovery and independent spec/code-quality reviews.

Implemented:
- Added `PMEConfig` to the top-level `mlx_atomistic` public export surface and covered it with an export test.
- Updated minimal public docs for Phase 2:
  - README now mentions RB torsions, bounded PME, and accepted AMBER/CHARMM/GROMACS imports.
  - `docs/molecular-mechanics.md` documents `RBDihedralPotential`, PME assignment orders `2`, `4`, and `5`, and prep-layer importer entry points.
  - `docs/production-md.md` documents accepted importer entry points and PME/RB artifact-gate behavior without linking to untracked evidence files.
  - `docs/real-mm-core.md` no longer says all PME/Ewald and AMBER/CHARMM/GROMACS parser support is out of scope.
  - `docs/units.md` clarifies low-level reduced-unit kernels versus physical-unit prepared-system artifacts.
- Hardened public `PMEConfig` validation so non-finite `alpha`, `real_cutoff`, and `charge_tolerance` fail at construction.
- Updated the tiny GPCRmd AMBER test fixture so its periodic restart box values match AMBER `IFBOX` metadata.

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

Concerns:
- MLX runtime tests require Metal access; sandboxed focused runs fail on device loading, while outside-sandbox gates pass.
