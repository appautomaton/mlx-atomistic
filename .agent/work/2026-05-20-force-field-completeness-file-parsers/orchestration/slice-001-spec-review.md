# Slice 001 Spec Review: RB Dihedral Force Term

## Verdict

APPROVED

## Evidence

- `RBDihedralPotential` implements the `cos(phi - pi)` RB convention and energy/force paths in `src/mlx_atomistic/forcefields.py`.
- The package root imports and exports `RBDihedralPotential` in `src/mlx_atomistic/__init__.py`.
- `tests/test_forcefields.py` covers export, finite values, the `periodic_phi - pi` reference expression, finite-difference forces, and the existing periodic/improper dihedral tests.

## Verification Basis

- Review was read-only.
- Host verification passed: `5 passed, 27 deselected`; targeted Ruff passed.
