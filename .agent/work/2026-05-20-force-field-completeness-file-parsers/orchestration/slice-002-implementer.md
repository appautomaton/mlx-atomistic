# Slice 002 Implementer: RB Prepared-System And Artifact Integration

## Verdict

DONE

## Scope

- Added `rb_dihedrals` plus `rb_c0` through `rb_c5` to prepared-system schema validation.
- Added RB arrays to prepared-system NPZ save/load defaults.
- Added `rb_dihedral` artifact compatibility term support, aliases, declared-array checks, requested-array validation, and finite coefficient validation.
- Updated artifact runtime construction to include RB torsions in topology 1-4 handling and append `RBDihedralPotential`.
- Added RB round-trip, runtime construction, and fail-closed artifact/schema tests.

## Correction Pass

Code-quality review requested two corrections:

- Do not let every `supported_terms` entry declare advanced arrays; only `rb_dihedral` is folded in because this slice validates/builds RB arrays from array presence.
- Reject `charmm_force_switch_nonbonded` artifacts when RB torsions are present.

Both corrections were implemented with focused regressions.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_production_artifacts.py tests/test_mlx_prep.py -k "dihedral or rb or artifact"` passed outside the sandbox: `67 passed, 23 deselected`.
- After correction, `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_production_artifacts.py tests/test_mlx_prep.py -k "dihedral or rb or artifact or charmm"` passed outside the sandbox: `72 passed, 20 deselected`.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/prep/schema.py src/mlx_atomistic/prep/io.py src/mlx_atomistic/artifacts.py tests/test_production_artifacts.py tests/test_mlx_prep.py` passed.
