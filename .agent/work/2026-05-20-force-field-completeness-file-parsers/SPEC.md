# SPEC: Force Field Completeness and File Parsers

## Bounded Goal

Implement all Phase 2 force-field completeness work so MLX can parse accepted AMBER, CHARMM, and GROMACS biomolecular inputs, execute RB dihedrals and PME assignment orders 4 and 5, and validate imported-system energies against OpenMM references.

## Broader Intent

This preserves the production biomolecular MD roadmap after Phase 1 by moving from hand-built or partially imported systems to common force-field input files that can be prepared and run through MLX with reference parity.

## Work Scale And Shape

- Scale: capability-sized.
- Shape: mixed parity, parser migration, and force-term/runtime extension.
- Rationale: the work touches multiple modules, but all acceptance criteria serve one outcome: force-field input completeness for production biomolecular MD parity.

## Selected Lenses

- product
- engineering
- runtime

## Target User Or Stakeholder

Researchers and engineer-users who need AMBER, CHARMM, or GROMACS force-field files to become MLX-ready systems on Apple Silicon without depending on external MD engines at runtime.

## Linked Detail Files

- `spec/gap-matrix.md`: normative Phase 2 gap IDs, evidence, and acceptance links.

## Constraints And Risks

- Use `uv` for Python commands and keep source code under `src/mlx_atomistic/`.
- Treat `vendors/` as reference-only; do not build against or import from vendor checkouts.
- OpenMM may be used as a test/reference oracle, but MLX execution and accepted parser paths must not require OpenMM at runtime.
- The existing code already has native AMBER import coverage and a CHARMM importer that relies on ParmEd; Phase 2 should complete accepted native paths and preserve useful compatibility behavior without making ParmEd the normative parser.
- PME order support is currently hard-coded to order 2 in `PMEConfig`, `PreparedSystem` validation, and artifact validation, so orders 4 and 5 must update runtime, schema, artifact, and diagnostic surfaces together.
- Parser import must fail closed for unsupported directives, force terms, virtual sites, water models, or preprocessing features instead of silently dropping required terms.
- GROMACS `.top` grammar can include broad preprocessing behavior; the supported subset must be explicit, tested, and blocker-producing outside the subset.
- Phase 2 may expose unsupported Phase 3+ needs such as virtual sites. Those stay blockers, not partial implementations, unless directly required by this spec's acceptance criteria.

## Required Outcome

The Phase 2 implementation provides force terms, PME runtime support, native parser entry points, prepared-system schema support, and parity fixtures that together allow accepted AMBER ff14SB, CHARMM36, and GROMACS biomolecular inputs to produce validated MLX-ready systems. Each accepted import path must preserve topology, parameters, exceptions, periodic box metadata, supported-term counts, unsupported-term diagnostics, and energy parity evidence.

## Acceptance Criteria

| ID | Check |
| --- | --- |
| AC-01 | RB dihedral support exists as an executable force term with validated energy and force behavior, including finite-difference force coverage and an OpenMM-equivalent reference check. |
| AC-02 | PME accepts assignment orders 2, 4, and 5 end to end, including charge assignment, force interpolation, deconvolution, diagnostics, prepared-system validation, artifact load/save, and reference parity on a neutral periodic fixture. |
| AC-03 | AMBER `prmtop`/`inpcrd` import covers the accepted ff14SB protein fixture with validated atoms, residues, bonds, angles, periodic dihedrals, impropers, exceptions, constraints, charges, LJ parameters, and periodic box metadata. |
| AC-04 | CHARMM PSF/parameter import has a native accepted path for a CHARMM36 protein fixture, maps supported CHARMM terms to MLX force terms, and reports unsupported terms as explicit blockers. |
| AC-05 | GROMACS `.top`/`.gro` import has a tested native path for its declared supported subset, including topology directives, coordinates, molecule expansion, nonbonded defaults, bonded terms, RB dihedrals where present, exclusions or pairs, and fail-closed unsupported directive reporting. |
| AC-06 | Prepared-system and artifact round trips preserve new Phase 2 metadata and arrays, including PME assignment order 4/5 metadata, RB dihedral arrays or term records, parser provenance, supported-term counts, and unsupported-term blockers. |
| AC-07 | OpenMM-referenced parity tests cover accepted AMBER, CHARMM, and GROMACS imports; at minimum, the AMBER ff14SB and CHARMM36 protein fixtures match reference total and component energies within stated tolerances. |
| AC-08 | Existing Phase 1 MD, artifact, and parser tests continue to pass after Phase 2 changes. |

## Scope Coverage Decisions

Included:
- Ryckaert-Bellemans dihedrals.
- PME 4th and 5th order B-spline assignment and interpolation.
- Native AMBER, CHARMM, and GROMACS file parser paths.
- Prepared-system, artifact, and runtime metadata needed for the new terms and PME orders.
- OpenMM-referenced parity fixtures and tests.

Deferred:
- Virtual sites, TIP4P/5P water models, and virtual-site force redistribution remain Phase 3.
- GBSA/OBC, custom force expressions, replica exchange, alchemical methods, CI/release automation, and broad documentation restructuring remain later phases.
- Polarizable, machine-learned, reactive, and materials force fields remain out of scope.

Assumption:
- "All Phase 2" means every Phase 2 roadmap item is in this spec, while individual parser grammar edge cases may be bounded by explicit supported-subset and fail-closed diagnostics during planning.

## Anti-Goals

- Do not split Phase 2 into an AMBER-only, CHARMM-only, GROMACS-only, or PME-only spec.
- Do not add virtual-site or advanced water-model support as part of this change.
- Do not silently ignore unsupported force-field terms, parser directives, preprocessing features, or topology records.
- Do not introduce runtime dependence on OpenMM, GROMACS, CHARMM, AMBER tools, ParmEd, or vendor checkouts for accepted MLX execution.
- Do not broaden this phase into production deployment, CI/release automation, or broad documentation work.
