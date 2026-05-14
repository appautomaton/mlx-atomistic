# Slice 001 Spec Review

- Slice: GPCRmd Cache And Protocol Normalization
- Status: APPROVED
- Route: explorer subagent

## Summary

- Required Slice 1 behavior is present: deterministic role paths, PSF-derived MASS prelude, protocol `.xsc` box parsing, import-report provenance, and prepared `cell_lengths` propagation.
- Targeted verification passed.
- Real GPCRmd import now blocks on expected CHARMM terms rather than CT3 MASS parsing.

## Issues

- None.

## Evidence

- `src/mlx_atomistic/prep/gpcrmd.py:146`: reports `resolved_role_paths`.
- `src/mlx_atomistic/prep/gpcrmd.py:1173`: sorts role paths deterministically.
- `src/mlx_atomistic/prep/topology_import.py:306`: builds PSF-derived MASS prelude.
- `src/mlx_atomistic/prep/gpcrmd.py:1119`: prepends MASS prelude for CHARMM/ParmEd import when needed.
- `src/mlx_atomistic/prep/gpcrmd.py:1201`: adds `protocol_box` to import details.
- `src/mlx_atomistic/prep/gpcrmd.py:1374`: applies protocol box metadata into `PreparedSystem.cell_lengths`.
- `tests/test_gpcrmd_registry.py:772`: covers MASS prelude wiring, `input.xsc` parsing, report metadata, and prepared `cell_lengths`.
