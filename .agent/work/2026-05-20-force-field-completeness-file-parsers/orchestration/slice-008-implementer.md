# Slice 8 Implementer: Cross-Format Artifact Compatibility Gate

Status: `DONE`

Route: subagent implementer with one quality-review correction pass.

Implemented:
- Added compatibility report normalization that preserves parser-specific fields while adding normalized supported, required, unsupported, rejected, blocker, provenance, and term-count metadata.
- Normalized prepared-system metadata at save/load boundaries so persisted artifacts carry array-derived counts and parser provenance.
- Hardened the shared artifact compatibility gate so declared blockers, virtual-site aliases, advanced-water aliases, unsupported/rejected terms, and term-count mismatches fail closed.
- Preserved PME order 4/5, RB arrays, CHARMM CMAP/Urey/NBFIX arrays, nonbonded exceptions, constraints, parser provenance, and blockers through artifact round trips.
- Added cross-format AMBER, CHARMM, and GROMACS artifact tests for normalized metadata and runtime term construction.

Quality-review corrections:
- Moved term-count mismatch validation into `validate_mlx_compatibility(..., arrays=...)` so in-memory and readiness paths share the same gate as `load_prepared_mlx_artifact`.
- Removed the supported-only RB exception so RB arrays are accepted only when normalized required terms require `rb_dihedral`; legacy artifacts without `required_terms` still fall back to normalized supported terms.

Changed paths:
- `src/mlx_atomistic/artifacts.py`
- `src/mlx_atomistic/prep/schema.py`
- `src/mlx_atomistic/prep/io.py`
- `tests/test_production_artifacts.py`

Verification:
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_production_artifacts.py -k "term_count_metadata_mismatch or supported_only_rb_arrays or rb_arrays_cannot_be_hidden"` passed in sandbox: `5 passed, 68 deselected` with the known MLX Metal atexit warning.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_production_artifacts.py tests/test_mlx_prep.py -k "artifact or compatibility or unsupported or rb or pme or charmm or gromacs"` passed outside the sandbox: `117 passed, 62 deselected`.
- Targeted Ruff passed for the touched Slice 8 files.
- `git diff --check` passed for the touched Slice 8 files.

Concerns:
- The canonical gate needs Metal access for runtime construction tests; the sandboxed run failed only with `RuntimeError: [metal::load_device] No Metal device available`, and the same command passed outside the sandbox.
