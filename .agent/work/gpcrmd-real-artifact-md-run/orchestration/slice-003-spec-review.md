# Slice 003 Spec Review

- Status: approved
- Reviewer route: subagent

## Summary

Slice 3 satisfies the export/schema acceptance criteria. NBFIX is exported as compact atom-type pairs with converted LJ parameters, GPCRmd 729 reports concrete values, JSON/NPZ round-trip works through the prepared-system schema, and runtime NBFIX semantics remain Slice 4.

## Evidence

- Focused tests passed.
- Ruff passed on touched prep/schema/test files.
- Real GPCRmd 729 import exported a `/tmp` artifact with `nbfix_type_pairs` shape `(37, 2)` and legacy `nbfix_pairs` shape `(0, 2)`.
- First real NBFIX value: `BRGR1`/`NC2`, sigma `3.260689...` angstrom, epsilon `4.6024` kJ/mol.
- Compatibility report includes `required_terms` containing `nbfix_pair_overrides`, `term_counts.nbfix_pair_overrides = 37`, and concrete `term_details.atom_type_pairs`.
- Runtime risk remains explicit through `runnable_now: false` and large-system runtime metadata.

## Issues

None blocking. Runtime MLX system construction still needs Slice 4 NBFIX semantics.
