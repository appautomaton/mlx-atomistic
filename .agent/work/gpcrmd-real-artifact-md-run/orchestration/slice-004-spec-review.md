# Slice 004 Spec Review

- Status: approved
- Reviewer route: subagent

## Summary

Slice 4 satisfies the NBFIX runtime semantics acceptance criteria without requiring Slice 6 large-system routing. NBFIX is applied by substituting LJ sigma/epsilon inside `NonbondedPotential`; artifacts route NBFIX into the normal nonbonded term; PME/Ewald Coulomb remains separate.

## Evidence

- `src/mlx_atomistic/forcefields.py` accepts NBFIX fields, substitutes mixed LJ pair parameters, and forces pair backend when NBFIX is active.
- Explicit exceptions remain separate from regular nonbonded evaluation.
- Ewald/PME Coulomb correction paths remain independent from NBFIX-modified LJ.
- `src/mlx_atomistic/artifacts.py` validates compact/legacy NBFIX arrays and builds a normal `NonbondedPotential`, not a separate NBFIX force term.
- Tests cover type-pair substitution, explicit-pair substitution, unknown type fail-closed, exclusions, exceptions, and PME plus NBFIX Coulomb separation.
- Focused Slice 4 tests passed: 64 passed, 32 deselected.

## Issues

None. The GPCRmd dense topology build failure is assigned to Slice 6.
