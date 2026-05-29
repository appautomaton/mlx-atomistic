# Slice 6 Spec Review

## Status

APPROVED

## Summary

- Slice 6 satisfies the native CHARMM PSF/parameter import acceptance criteria.
- Prior spec findings for non-empty HBOND records and malformed NBFIX records were corrected with explicit blockers and regression tests.

## Issues

- none

## Evidence

- `src/mlx_atomistic/prep/__init__.py` imports and exports `import_charmm_psf`.
- `src/mlx_atomistic/prep/topology_import.py` implements native `import_charmm_psf`, maps PSF atoms, charges, masses, bonds, angles, dihedrals, Urey-Bradley, CMAP, NBFIX type overrides, coordinates, cell lengths, and cell matrix metadata into `PreparedSystem`.
- `src/mlx_atomistic/prep/topology_import.py` preserves `import_charmm_with_parmed` and the `_prepared_from_parmed_structure` compatibility helpers.
- `tests/test_mlx_prep.py` covers native CHARMM fixture mapping, artifact round trip, virtual-site and water-model blockers, HBOND blockers, malformed NBFIX blockers, physical numeric blockers, float32 overflow blockers, and fake-ParmEd CMAP/NBFIX overflow blockers.
- Required Slice 6 pytest passed outside the sandbox: `69 passed, 152 deselected`.
- Targeted Ruff passed for the touched Slice 6 files.

## Prior Findings Resolved

- Non-empty `HBOND` parameter sections now raise `TopologyImportError("unsupported_terms:charmm_parameter_hbond")`.
- Malformed `NBFIX` records now raise `TopologyImportError("unsupported_terms:nbfix_pair_overrides:malformed_entries")` instead of being silently dropped.
