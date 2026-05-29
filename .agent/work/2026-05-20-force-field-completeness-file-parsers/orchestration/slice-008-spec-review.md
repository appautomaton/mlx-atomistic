# Slice 8 Spec Review: Cross-Format Artifact Compatibility Gate

Status: `APPROVED`

Summary:
- Slice 8 matches the requested artifact compatibility gate scope.
- Compatibility metadata is normalized before save/load/gating, artifact arrays preserve the requested cross-format fields, runtime construction covers representative AMBER/CHARMM/GROMACS artifacts, and virtual-site/advanced-water records fail closed.

Issues:
- None.

Evidence:
- `src/mlx_atomistic/prep/schema.py` normalizes supported, required, unsupported, rejected, term-count, array-count, blocker, and parser-provenance metadata.
- `src/mlx_atomistic/prep/io.py` normalizes compatibility metadata during save with the persisted NPZ arrays.
- `src/mlx_atomistic/artifacts.py` applies the normalized fail-closed compatibility gate, normalizes on artifact load, rejects term-count mismatches, and builds RB, CHARMM, exception, constraint, PME, and nonbonded runtime terms from artifacts.
- `tests/test_production_artifacts.py` covers AMBER/CHARMM/GROMACS normalized metadata, cross-format runtime term lists, and virtual-site/advanced-water blocker preservation and rejection.
- Requested Slice 8 gate passed outside the sandbox: `117 passed, 62 deselected`.

Residual risk:
- None for Slice 8 acceptance. Broader force-field parity remains owned by Slice 9.
