# Slice 001 Implementer: RB Dihedral Force Term

## Verdict

DONE

## Scope

- Added `RBDihedralPotential` in `src/mlx_atomistic/forcefields.py`.
- Exported `RBDihedralPotential` from `src/mlx_atomistic/__init__.py`.
- Added RB export, reference-expression, finite-value, and finite-difference coverage in `tests/test_forcefields.py`.

## Notes

- The RB convention is `E = sum(Cn * cos(phi - pi)^n)`, where `phi` is the existing package/OpenMM-style periodic dihedral angle.
- The implementation reuses the existing periodic dihedral geometry convention.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_forcefields.py -k "dihedral or rb"` passed outside the sandbox: `5 passed, 27 deselected`.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/forcefields.py src/mlx_atomistic/__init__.py tests/test_forcefields.py` passed.
