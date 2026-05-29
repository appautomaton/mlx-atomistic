# Slice 7 Spec Review: Native GROMACS Top/Gro Import

## Status

APPROVED

## Summary

- Slice 7 matches the requested native GROMACS `.top`/`.gro` import scope.
- The entry point is exported, the accepted fixture subset is parsed and expanded, RB torsions are carried into existing RB arrays, unsupported GROMACS features fail closed, and `.top` routing distinguishes GROMACS from AMBER-style files.

## Issues

- none

## Evidence

- `src/mlx_atomistic/prep/__init__.py` exports `import_gromacs_top_gro` from `mlx_atomistic.prep`.
- `src/mlx_atomistic/prep/topology_import.py` defines the public wrapper and includes it in `__all__`.
- `src/mlx_atomistic/prep/gromacs.py` covers `[defaults]`, `[atomtypes]`, `[moleculetype]`, `[atoms]`, `[bonds]`, `[angles]`, `[dihedrals]`, `[pairs]`, `[exclusions]`, molecule expansion, `.gro` coordinates, and box parsing.
- `src/mlx_atomistic/prep/gromacs.py` parses GROMACS RB dihedral function type 3 and maps coefficients to `PreparedSystem` RB arrays.
- `src/mlx_atomistic/prep/gromacs.py` rejects preprocessor directives, unsupported directives, unsupported combination rules, unsafe generated-pair cases, unsupported function types, virtual sites, and malformed records.
- `src/mlx_atomistic/prep/gpcrmd.py` implements GROMACS-vs-AMBER `.top` detection, with routing coverage in `tests/test_gpcrmd_registry.py`.
- Required Slice 7 pytest passed: `12 passed, 123 deselected`.

## Residual Scope

- Broad GROMACS preprocessing, `[pairtypes]`, explicit pair parameters, generated-pair inference without explicit `[ pairs ]`, constraints/SETTLE, and virtual sites remain explicit blockers.
