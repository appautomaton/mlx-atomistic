# Slice 004 Implementer: PME Schema, Artifact, And Readiness Integration

## Verdict

DONE

## Scope

- Updated prepared-system PME validation to accept assignment orders `2`, `4`, and `5`.
- Updated artifact PME metadata and array validation to accept only assignment orders `2`, `4`, and `5`.
- Made partial PME array configs fail closed instead of falling back to metadata.
- Added round-trip and artifact-build tests for PME assignment orders `4` and `5`.
- Added parity-helper coverage proving configured assignment order is preserved in metadata, arrays, and readiness.

## Correction Pass

Code-quality review requested two corrections:

- Validate `PMEParityConfig` before writing PME metadata/arrays so non-integer assignment orders cannot be truncated.
- Use one supported-assignment-order source of truth.

Both corrections were implemented: the parity helper validates through `PMEConfig`, and schema/artifact validation import `PME_SUPPORTED_ASSIGNMENT_ORDERS` from runtime PME.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_production_artifacts.py tests/test_mlx_prep.py tests/test_openmm_mlx_parity.py -k "pme or assignment_order or prepared"` passed outside the sandbox after correction: `46 passed, 66 deselected`.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/prep/schema.py src/mlx_atomistic/prep/io.py src/mlx_atomistic/artifacts.py scripts/openmm_mlx_parity.py tests/test_production_artifacts.py tests/test_mlx_prep.py tests/test_openmm_mlx_parity.py` passed.
