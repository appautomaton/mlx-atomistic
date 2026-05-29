# Intake: Force Field Completeness and File Parsers

## Work Classification

- Mode: Builder
- Work scale: capability
- Work shape: mixed, with parity as the dominant shape

## Objective Statement

Put all of Phase 2 into the next spec: Ryckaert-Bellemans dihedral support, higher-order PME interpolation with 4th and 5th order B-splines, and native AMBER, CHARMM, and GROMACS file parsers that produce MLX-ready systems and validate against OpenMM-parsed references.

## Broader Intent

Continue the production biomolecular MD roadmap after the verified Phase 1 integrator and constraint work, making common AMBER, CHARMM, and GROMACS force-field inputs runnable on MLX with reference parity.

## Target User Or Stakeholder

Researchers and engineer-users who want to run real biomolecular systems from common force-field files on Apple Silicon without relying on an external MD engine at runtime.

## Desired Outcome

A CHARMM36 protein system and an AMBER ff14SB protein system can be parsed, imported, and run with MLX energy matching OpenMM reference output within tolerance. GROMACS `.top`/`.gro` import has a tested path with fail-closed unsupported-directive handling. RB dihedrals and higher-order PME are available where those imported systems require them.

## Scope Boundary And Anti-Goals

Included:
- `RBDihedralPotential` force term.
- PME 4th and 5th order B-spline charge assignment, interpolation, deconvolution, diagnostics, and artifact/schema acceptance.
- Native AMBER `prmtop`/`inpcrd` import completion and parity validation.
- Native CHARMM PSF/parameter import path, including mapping to existing CHARMM-specific force terms.
- Native GROMACS `.top`/`.gro` import path for the supported production subset.
- Prepared-system and force-term outputs that preserve provenance, term counts, unsupported-term reporting, and OpenMM parity evidence.

Anti-goals:
- No virtual-site framework, TIP4P/5P water models, or force redistribution work.
- No implicit solvent, GBSA/OBC, custom-force expression framework, replica exchange, or alchemical free-energy work.
- No CUDA, ROCm, HPC deployment, or non-Apple-Silicon runtime target.
- No runtime dependency on OpenMM, GROMACS, CHARMM, AMBER tools, or `vendors/` checkouts.
- No broad CI, release automation, or documentation restructuring beyond the minimal API and usage updates needed for Phase 2.

## Rejected Framings

- Rejected an AMBER-only vertical slice. The user explicitly wants all of Phase 2 in the next spec.
- Rejected decomposing Phase 2 into separate specs before framing. The next spec must preserve the full Phase 2 scope and let planning order the implementation slices.

## Scope Preservation

This preserves the user's full stated intent for Phase 2 rather than decomposing it. The work is large, but it is one coherent capability: force-field input completeness for production biomolecular MD parity.

## Scope Coverage

Included:
- RB dihedrals.
- Higher-order PME.
- AMBER parser.
- CHARMM parser.
- GROMACS parser.
- OpenMM parity validation for imported systems.

Deferred:
- Phase 3 virtual sites and advanced water models.
- Phase 4 implicit solvent and enhanced sampling.
- Phase 5 alchemical free energy.
- Phase 6 CI, release automation, and broad documentation.
- Roadmap deferred items such as polarizable force fields, ML potentials, reactive force fields, and heavyweight chemistry integrations.

Needs decision:
- None. Parser grammar details may be constrained during planning, but unsupported format features must fail closed rather than silently dropping terms.

## Selected Approach

Frame the whole Phase 2 capability in one spec, with an ordered plan later splitting the work into force-term, PME, parser, artifact, and parity slices. This follows the user's direction and keeps shared parser and parity constraints in one contract.

## Key Assumptions And Risks

- The current repo has partial AMBER import support and a CHARMM path that uses ParmEd as a parser; Phase 2 should complete native accepted paths rather than erase that existing surface.
- PME assignment order is currently constrained to order 2 in runtime config and artifact/schema validation, so higher-order PME must update all validation and metadata surfaces together.
- GROMACS `.top` parsing can expose preprocessor and directive edge cases; the accepted subset must be explicit and fail closed outside it.
- OpenMM is the reference oracle for parity tests, not a runtime dependency for MLX execution.
- Parser fidelity can expose unsupported terms from real force fields; those terms must be reported as blockers unless this phase implements them.

## Deferred Scope

All Phase 3 and later roadmap work remains deferred. The next lifecycle step is `auto-frame` to turn this intake into the canonical `SPEC.md`.
