# Slice 004 Quality Review

- Status: approved
- Reviewer route: subagent

## Summary

The NBFIX runtime path is maintainable enough for this slice. NBFIX lives in `NonbondedPotential` as LJ parameter substitution, explicit exceptions remain separate, and artifacts avoid adding a duplicate NBFIX force term.

## Evidence

- `NonbondedPotential.mixed_pair_parameters()` applies NBFIX sigma/epsilon to matching explicit atom pairs or atom-type pairs.
- Regular nonbonded evaluation removes exception pairs before adding exception terms.
- Ewald/PME paths compute LJ and Coulomb through separate components.
- Artifact construction passes NBFIX arrays into the normal nonbonded term.
- Tests cover the main regression risks.

## Issues

None.
