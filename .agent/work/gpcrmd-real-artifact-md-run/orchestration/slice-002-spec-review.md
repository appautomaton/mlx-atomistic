# Slice 002 Spec Review

- Slice: Strict CHARMM Term Export
- Final status: approved

## First Review

Status: changes requested.

Issue found: the blocked GPCRmd import report still exposed the target-level compatibility shape and did not list prepared `supported_terms`, `required_terms`, `rejected_terms`, `rejection_reasons`, or `term_counts`.

## Re-Review

Status: approved.

The reviewer confirmed:

- Urey-Bradley arrays are exported when present.
- CMAP terms, grid indices, and grids are exported when present.
- NBFIX pair overrides are rejected with a precise blocker for the current runtime semantics.
- Blocked GPCRmd import reports now surface prepared force-term fields.
- Metadata-only hiding of CHARMM arrays is rejected.

## Evidence

- Focused Slice 2 tests: `71 passed, 53 deselected`.
- Focused Ruff over touched files: passed.
- Real GPCRmd 729 import probe: exit 0, `exported=false`, blocker `rejected_terms:nbfix_pair_overrides`.
- Real GPCRmd 729 report includes `charmm_cmap_terms=317`, `urey_bradley_terms=49223`, and `nbfix_pair_overrides=37`.
