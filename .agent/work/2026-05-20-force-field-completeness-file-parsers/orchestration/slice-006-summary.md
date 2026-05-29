# Slice 6 Summary: Native CHARMM PSF/Parameter Import

Status: complete

Route: subagent implementer with coordinator corrections and independent spec/code-quality reviews.

Implemented:
- Added native `import_charmm_psf` entry point exported from `mlx_atomistic.prep`.
- Parsed supported PSF atom, bond, angle, dihedral, CMAP, coordinate, and box records without ParmEd.
- Parsed CHARMM parameter BONDS, ANGLES with Urey-Bradley, DIHEDRALS, CMAP, NONBONDED, and NBFIX records into existing `PreparedSystem` arrays.
- Added explicit blockers for unsupported PSF records, virtual sites, unsupported water models, non-empty HBOND records, malformed NBFIX, non-finite and nonphysical numeric values, float32-overflowing values, malformed parameters, and missing supported-subset parameters.
- Preserved existing ParmEd compatibility path.
- Added matching float32-overflow blockers for ParmEd compatibility CMAP/NBFIX helpers.

Changed paths:
- `src/mlx_atomistic/prep/topology_import.py`
- `src/mlx_atomistic/prep/__init__.py`
- `tests/test_mlx_prep.py`
- `tests/fixtures/charmm/native-mini.psf`
- `tests/fixtures/charmm/native-mini.prm`
- `tests/fixtures/charmm/native-mini.pdb`

Verification:
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_mlx_prep.py -k "charmm_psf"` passed in sandbox: `24 passed, 80 deselected` with the known MLX Metal atexit warning.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_mlx_prep.py -k "charmm_parmed"` passed in sandbox: `4 passed, 102 deselected` with the known MLX Metal atexit warning.
- Required pytest command fails inside the sandbox when MLX force-term tests need Metal, then passed outside the sandbox: `69 passed, 152 deselected`.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/prep/topology_import.py src/mlx_atomistic/prep/__init__.py tests/test_mlx_prep.py tests/test_charmm_terms.py tests/test_production_artifacts.py tests/test_gpcrmd_registry.py` passed.

Reviews:
- Spec review: `APPROVED` after correcting non-empty HBOND and malformed NBFIX blocker gaps.
- Code quality review: `APPROVED` after adding physical numeric validation and float32-overflow blockers across native and ParmEd compatibility CHARMM CMAP/NBFIX paths.

Concerns:
- The native parser intentionally covers the declared supported CHARMM36 subset used by the fixture and fails closed for unsupported records instead of claiming broad CHARMM grammar coverage.
