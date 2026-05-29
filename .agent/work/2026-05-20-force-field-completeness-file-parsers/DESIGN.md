# DESIGN: Phase 2 Force-Field Parser Completeness

## Architecture Approach

Phase 2 should extend the existing prepared-system pipeline rather than creating a second runtime model:

1. Parser entry points produce `PreparedSystem`.
2. `save_prepared_system` and `load_prepared_system` preserve the arrays and metadata.
3. `load_prepared_mlx_artifact` validates fail-closed compatibility.
4. `build_mlx_system_from_artifact` produces `MMSystem`, force terms, constraints, and PME configuration.
5. OpenMM reference paths remain test-only parity oracles.

## New Force-Term Surface

Add RB torsions as a first-class force term, not as overloaded periodic dihedrals.

- Runtime term: `RBDihedralPotential` in `src/mlx_atomistic/forcefields.py`.
- Public export: `src/mlx_atomistic/__init__.py`.
- Prepared-system arrays: `rb_dihedrals` plus six coefficient vectors, `rb_c0` through `rb_c5`, all length-matched to `rb_dihedrals`.
- Artifact term name: `rb_dihedral`.
- Build path: `build_mlx_system_from_artifact` appends `RBDihedralPotential` when `rb_dihedral` is required or arrays are present.

The implementation should keep `PeriodicDihedralPotential` unchanged and use explicit conversion only in parsers that encounter format-specific RB torsion records.
RB angle convention is a correctness risk. The force-term tests must pin the polynomial convention before any parser maps GROMACS or OpenMM RB records into these arrays.

## PME Order Surface

PME assignment order support must move together across runtime, schema, artifacts, and parity helpers.

- `PMEConfig.assignment_order` accepts only `2`, `4`, and `5`.
- Charge assignment and interpolation use the same B-spline order.
- Deconvolution uses the assignment-order window power, not a fixed CIC/order-2 assumption.
- `PMEDiagnostics`, readiness reports, prepared-system arrays, and metadata preserve the selected order.
- Existing order-2 behavior remains the compatibility baseline.
- Existing benchmark/private helper callers either keep compatibility wrappers or move to the new generalized helper names in the same slice that changes PME internals.

## Parser Surface

Keep public parser entry points under `mlx_atomistic.prep.topology_import` and exported from `mlx_atomistic.prep`.

- AMBER: complete the existing native `import_amber_prmtop` path.
- CHARMM: add a native accepted path for PSF/parameter/coordinate bundles; keep `import_charmm_with_parmed` as optional compatibility, not the normative Phase 2 parser.
- GROMACS: add a native `.top`/`.gro` path with an explicit supported subset and fail-closed unsupported directives.

Each parser must populate compatibility metadata with:

- source kind and file paths,
- supported, required, unsupported, and rejected terms,
- term counts and parser subset/version details,
- blocker reasons for unsupported records,
- parser provenance that does not imply an external MD engine runtime dependency.

## Parity Harness

Extend the current OpenMM parity helper pattern instead of adding a separate validation framework.

- Keep OpenMM imports inside tests/scripts, not package runtime paths.
- Add reusable fixed-coordinate parity runners for AMBER, CHARMM, and GROMACS accepted fixtures.
- Compare total energy, component energies where available, force shape, max force error, RMS force error, unsupported terms, blockers, and readiness reports.
- Write machine-readable reports under test or temporary output directories only.

## Fail-Closed Boundary

Unsupported Phase 3+ features, including virtual sites and advanced water models, remain blockers. Parser success requires either faithful MLX support for every required term in the accepted subset or an explicit blocker that prevents production execution.
