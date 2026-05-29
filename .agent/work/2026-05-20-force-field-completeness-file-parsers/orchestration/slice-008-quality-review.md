# Slice 8 Code Quality Review: Cross-Format Artifact Compatibility Gate

Status: `APPROVED`

Summary:
- Initial review requested changes for two fail-open paths.
- Re-review approved after both corrections landed.

Issues fixed:
- `important`: Term-count mismatch validation was loader-only. It now runs in the shared compatibility path whenever arrays are supplied, so readiness and in-memory paths cannot bypass it.
- `important`: Supported-only RB arrays could be treated as declared/executable when `required_terms` omitted `rb_dihedral`. RB now follows normalized required terms, with legacy no-`required_terms` artifacts handled by normalization.

Evidence:
- `src/mlx_atomistic/artifacts.py` calls `_validate_term_count_metadata` from `validate_mlx_compatibility` when arrays are supplied.
- `artifact_readiness_report` delegates through `validate_mlx_compatibility`, inheriting term-count blocking.
- `_validate_declared_term_arrays` no longer adds supported-only RB terms to declared arrays.
- `build_mlx_system_from_artifact` only builds RB when `rb_dihedral` is in requested/required terms.
- `tests/test_production_artifacts.py` covers shared validation and readiness term-count blocking plus supported-only RB array rejection.

Verification:
- Focused regression tests passed: `5 passed, 68 deselected`.
- Broader Slice 8 gate passed outside the sandbox: `117 passed, 62 deselected`.
- Targeted Ruff and `git diff --check` passed.

Residual risk:
- None beyond the existing requirement that MLX runtime construction tests need a Metal-capable environment.
