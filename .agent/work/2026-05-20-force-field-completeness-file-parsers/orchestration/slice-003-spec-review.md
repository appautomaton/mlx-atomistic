# Slice 003 Spec Review: PME Runtime Assignment Orders 4 And 5

## Verdict

APPROVED

## Evidence

- `PMEConfig.assignment_order` is constrained to supported orders `2`, `4`, and `5`.
- Reciprocal PME uses `config.assignment_order` for assignment, deconvolution, potential interpolation, and field interpolation.
- Public and private CIC wrappers remain as order-2 shims over the generalized B-spline helpers.
- PME diagnostics and readiness metadata expose the selected assignment order.
- Tests cover accepted/rejected orders, charge conservation for orders `2`, `4`, and `5`, order-2 CIC compatibility, higher-order finite PME, deconvolution order selection, and nonbonded PME diagnostics.

## Verification Basis

- Review was read-only.
- Host verification passed: `41 passed, 23 deselected`; targeted Ruff passed.
