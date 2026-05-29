# Slice 9 Summary: OpenMM Parity Harness For Accepted Imports

Status: complete

Route: subagent implementer with coordinator verification and independent spec/code-quality reviews.

Implemented:
- Extended OpenMM-vs-MLX fixed-coordinate parity beyond AMBER to accepted CHARMM and GROMACS imports.
- Added source-aware parity reports with reference engine role, readiness, platform evidence, unsupported terms, component errors, force max/RMS errors, and pass/block status.
- Added nonzero CHARMM component parity for bond, angle, torsion, Urey-Bradley, CMAP, and nonbonded terms.
- Added GROMACS accepted-subset parity with RB component mapping.
- Added blocked-report JSON serialization and CLI routing for `amber`, `charmm`, and `gromacs`.
- Restored the runtime/prep boundary by moving shared compatibility normalization into `src/mlx_atomistic/compatibility.py`.

Changed paths:
- `scripts/openmm_mlx_parity.py`
- `scripts/run_openmm_mlx_parity.py`
- `src/mlx_atomistic/compatibility.py`
- `src/mlx_atomistic/artifacts.py`
- `src/mlx_atomistic/prep/schema.py`
- `tests/test_openmm_mlx_parity.py`
- `tests/fixtures/amber/`
- `tests/fixtures/charmm/`
- `tests/fixtures/gromacs/`

Verification:
- `uv run pytest tests/test_openmm_mlx_parity.py -k "amber or charmm or gromacs or pme"` passed outside the sandbox: `20 passed, 6 deselected`.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_runtime_boundaries.py` passed: `8 passed` with the known MLX Metal atexit warning.
- `uv run pytest tests/test_mlx_prep.py tests/test_production_artifacts.py -k "charmm or cmap or urey or nbfix"` passed outside the sandbox: `39 passed, 140 deselected`.
- `uv run pytest tests/test_production_artifacts.py tests/test_mlx_prep.py -k "artifact or compatibility or unsupported or rb or pme or charmm or gromacs"` passed outside the sandbox: `117 passed, 62 deselected`.
- Targeted Ruff passed for Slice 9 and compatibility-boundary files.
- `git diff --check` passed for Slice 9 and compatibility-boundary files.

Reviews:
- Spec review: `APPROVED` after correcting CHARMM component parity.
- Code quality review: `APPROVED` after correcting blocked-report serialization, component mapping ambiguity, CLI routing, and the prep/runtime boundary.

Concerns:
- Runtime parity gates require Metal access; sandboxed runs fail on MLX device loading, while the same gates pass outside the sandbox.
