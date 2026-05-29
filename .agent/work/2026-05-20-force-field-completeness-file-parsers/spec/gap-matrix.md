# Phase 2 Gap Matrix

This file is normative detail linked from `SPEC.md`.

| Gap ID | Area | Current Evidence | Required Closure | Acceptance Link |
| --- | --- | --- | --- | --- |
| P2-FF-01 | RB dihedral | `src/mlx_atomistic/forcefields.py` has `PeriodicDihedralPotential` and `ImproperDihedralPotential`, but no Ryckaert-Bellemans term. | Add an executable RB torsion term with energy and force behavior validated against finite differences and OpenMM-equivalent reference expressions. | AC-01 |
| P2-PME-01 | PME order | `src/mlx_atomistic/pme.py`, `src/mlx_atomistic/prep/schema.py`, and `src/mlx_atomistic/artifacts.py` currently require `assignment_order=2`. | Accept and execute PME assignment orders 2, 4, and 5, with matching assignment, interpolation, deconvolution, diagnostics, and artifact round-trip validation. | AC-02 |
| P2-PARSE-01 | AMBER | `src/mlx_atomistic/prep/topology_import.py` has a native `import_amber_prmtop` path and existing AMBER parity fixtures. | Complete AMBER `prmtop`/`inpcrd` import coverage needed for ff14SB protein parity, including exceptions, constraints, impropers, box metadata, and fail-closed unsupported terms. | AC-03, AC-07 |
| P2-PARSE-02 | CHARMM | `src/mlx_atomistic/prep/topology_import.py` imports CHARMM through ParmEd; `src/mlx_atomistic/charmm_terms.py` has CHARMM-specific force terms. | Provide a native accepted CHARMM PSF/parameter path that maps supported CHARMM36 terms to MLX terms and reports unsupported terms as blockers. | AC-04, AC-07 |
| P2-PARSE-03 | GROMACS | No native `.top`/`.gro` importer is present in the exported prep surface. | Provide a native GROMACS `.top`/`.gro` import path for a declared production subset, including directive handling and fail-closed unsupported-feature diagnostics. | AC-05 |
| P2-ART-01 | Prepared artifacts | `PreparedSystem` and artifact validation are shaped around periodic dihedrals and PME order 2. | Preserve provenance, term counts, unsupported-term reporting, PME order metadata, and new force-term arrays through validation and artifact round trips. | AC-06 |
| P2-PARITY-01 | Reference parity | Phase 1 parity exists for MD/integrator behavior, but Phase 2 parser and force-field parity is not complete. | Add OpenMM-referenced energy parity checks for accepted AMBER, CHARMM, and GROMACS import paths, with tolerances stated in tests or fixtures. | AC-07 |
