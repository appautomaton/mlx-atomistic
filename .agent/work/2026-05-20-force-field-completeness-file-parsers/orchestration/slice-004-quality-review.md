# Slice 004 Quality Review: PME Schema, Artifact, And Readiness Integration

## Initial Verdict

CHANGES_REQUESTED

## Findings

- The parity helper could coerce a non-integer `assignment_order` through `np.int32` before schema validation.
- Supported PME assignment orders were duplicated across runtime, schema, and artifact modules.

## Corrections

- `_with_pme_artifact_settings` now validates through `PMEConfig` before writing PME metadata and arrays.
- `prep/schema.py` and `artifacts.py` now import `PME_SUPPORTED_ASSIGNMENT_ORDERS` from runtime PME.
- A regression test covers non-integer parity assignment-order rejection.

## Final Verdict

APPROVED

## Verification Basis

- Review was read-only.
- Host verification passed: `46 passed, 66 deselected`; targeted Ruff passed.
