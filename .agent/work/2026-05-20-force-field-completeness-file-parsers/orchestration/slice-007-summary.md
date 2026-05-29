# Slice 7 Summary: Native GROMACS Top/Gro Import

Status: complete

Route: subagent implementer with coordinator verification and independent spec/code-quality reviews.

Implemented:
- Added native `import_gromacs_top_gro` exported from `mlx_atomistic.prep`.
- Added `src/mlx_atomistic/prep/gromacs.py` for the declared supported GROMACS subset.
- Parsed `[defaults]`, `[atomtypes]`, `[moleculetype]`, `[atoms]`, `[bonds]`, `[angles]`, `[dihedrals]`, `[pairs]`, `[exclusions]`, `[system]`, `[molecules]`, `.gro` coordinates, molecule expansion, and periodic box metadata for the accepted fixture.
- Mapped GROMACS function-type-3 RB torsions into existing `PreparedSystem` RB arrays.
- Converted GROMACS nm coordinates, sigma, bond lengths, and boxes to angstrom, and harmonic bond force constants from kJ/mol/nm^2 to kJ/mol/angstrom^2.
- Added fail-closed blockers for preprocessing directives, unsupported sections, unsupported combination rules, unsafe generated-pair cases, virtual sites, B-state atom records, unsupported function types, malformed records, invalid boxes, and `[ pairs ]` when `gen-pairs = no`.
- Updated GPCRmd `.top` routing to distinguish GROMACS topology files from AMBER `%FLAG` topology files.

Changed paths:
- `src/mlx_atomistic/prep/gromacs.py`
- `src/mlx_atomistic/prep/topology_import.py`
- `src/mlx_atomistic/prep/__init__.py`
- `src/mlx_atomistic/prep/gpcrmd.py`
- `tests/fixtures/gromacs/native-mini.top`
- `tests/fixtures/gromacs/native-mini.gro`
- `tests/test_gromacs_import.py`
- `tests/test_gpcrmd_registry.py`

Verification:
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_gromacs_import.py tests/test_mlx_prep.py tests/test_gpcrmd_registry.py -k "gromacs or rb"` passed in sandbox: `12 passed, 123 deselected` with the known MLX Metal atexit warning.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/prep/gromacs.py src/mlx_atomistic/prep/topology_import.py src/mlx_atomistic/prep/__init__.py src/mlx_atomistic/prep/gpcrmd.py tests/test_gromacs_import.py tests/test_gpcrmd_registry.py` passed.
- `git diff --check` passed for the touched Slice 7 files.

Reviews:
- Spec review: `APPROVED`.
- Code quality review: `APPROVED` after correcting the `gen-pairs = no` with `[ pairs ]` fail-closed gap.

Concerns:
- The parser intentionally rejects GROMACS preprocessing, includes, generated 1-4 pair inference without explicit `[ pairs ]`, virtual sites, constraints/SETTLE, unsupported function types, explicit pair parameter rows, `[pairtypes]`, and combination rules other than 2.
