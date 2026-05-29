# Slice 8 Summary: Cross-Format Artifact Compatibility Gate

Status: complete

Route: subagent implementer with coordinator verification and independent spec/code-quality reviews.

Implemented:
- Added normalized compatibility metadata for supported, required, unsupported, rejected, blocker, parser-provenance, and term-count fields.
- Wired normalization through prepared-system metadata, artifact save, artifact load, and compatibility validation.
- Hardened fail-closed behavior for declared blockers, unsupported/rejected terms, virtual-site aliases, advanced-water aliases, hidden force-term arrays, and term-count metadata mismatches.
- Preserved PME assignment orders 4/5, RB arrays, CHARMM-specific arrays, exceptions, constraints, parser provenance, and blockers through artifact round trips.
- Added AMBER, CHARMM, and GROMACS representative artifact tests for normalized metadata and expected runtime term lists.

Changed paths:
- `src/mlx_atomistic/artifacts.py`
- `src/mlx_atomistic/prep/schema.py`
- `src/mlx_atomistic/prep/io.py`
- `tests/test_production_artifacts.py`

Verification:
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_production_artifacts.py tests/test_mlx_prep.py -k "artifact or compatibility or unsupported or rb or pme or charmm or gromacs"` passed outside the sandbox: `117 passed, 62 deselected`.
- Focused regression test passed in sandbox: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_production_artifacts.py -k "term_count_metadata_mismatch or supported_only_rb_arrays or rb_arrays_cannot_be_hidden"` -> `5 passed, 68 deselected` with the known MLX Metal atexit warning.
- Targeted Ruff passed for Slice 8 files.
- `git diff --check` passed for Slice 8 files.

Reviews:
- Spec review: `APPROVED`.
- Code quality review: `APPROVED` after correcting shared term-count validation and supported-only RB array handling.

Concerns:
- Runtime artifact construction tests require Metal access; sandboxed pytest fails on MLX device loading, while the same gate passes outside the sandbox.
