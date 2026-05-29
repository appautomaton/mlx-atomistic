# PLAN: Force Field Completeness and File Parsers

## Goal

Deliver the Phase 2 SPEC: accepted AMBER, CHARMM, and GROMACS inputs become MLX-ready systems with RB dihedral support, PME assignment orders 4/5, artifact round trips, and OpenMM-referenced parity.

## Architecture Approach

Use the existing prepared-system pipeline as the integration spine: parsers produce `PreparedSystem`, artifacts validate and round-trip it, `build_mlx_system_from_artifact` creates runtime terms, and OpenMM remains a test-only reference. See `DESIGN.md` for the RB, PME, parser, parity, and fail-closed boundaries.

## Ordered Slice Sequence

### Slice 1: RB Dihedral Force Term

**Objective:** Add the runtime RB torsion term and pin its angle convention before parser integration.

**Acceptance criteria:**
- `RBDihedralPotential` evaluates finite energies and forces and is exported from the package.
- Tests cover finite-difference forces and a reference expression that locks the polynomial angle convention.
- `PeriodicDihedralPotential` and `ImproperDihedralPotential` behavior is unchanged.

**Verification:** `uv run pytest tests/test_forcefields.py -k "dihedral or rb"`

**Execution:** subagent recommended

**Touches:** `src/mlx_atomistic/forcefields.py`, `src/mlx_atomistic/__init__.py`, `tests/test_forcefields.py`

**Produces:** `P2-FF-01`, `AC-01`

**Status:** Complete. Subagent implementation, spec review, and code-quality review approved.

**Evidence:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_forcefields.py -k "dihedral or rb"` passed outside the sandbox: `5 passed, 27 deselected`; targeted Ruff passed for the touched files. See `orchestration/slice-001-summary.md`.

### Slice 2: RB Prepared-System And Artifact Integration

**Objective:** Carry RB torsions through prepared-system validation, artifact round trips, and runtime artifact construction.

**Acceptance criteria:**
- RB arrays validate in `PreparedSystem` and save/load through prepared-system artifacts.
- Artifact compatibility recognizes `rb_dihedral` as a supported production term.
- `build_mlx_system_from_artifact` appends `RBDihedralPotential` from RB arrays.
- Missing, malformed, or undeclared RB arrays fail closed.

**Verification:** `uv run pytest tests/test_production_artifacts.py tests/test_mlx_prep.py -k "dihedral or rb or artifact"`

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `src/mlx_atomistic/prep/schema.py`, `src/mlx_atomistic/prep/io.py`, `src/mlx_atomistic/artifacts.py`, tests

**Produces:** partial `P2-ART-01`, `AC-06`

**Status:** Complete. Subagent implementation, correction pass, spec re-review, and code-quality re-review approved.

**Evidence:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_production_artifacts.py tests/test_mlx_prep.py -k "dihedral or rb or artifact or charmm"` passed outside the sandbox: `72 passed, 20 deselected`; targeted Ruff passed for the touched files. See `orchestration/slice-002-summary.md`.

### Slice 3: PME Runtime Assignment Orders 4 And 5

**Objective:** Generalize PME charge assignment, interpolation, and deconvolution from CIC/order 2 to orders 2, 4, and 5.

**Acceptance criteria:**
- `PMEConfig.assignment_order` accepts only 2, 4, and 5.
- Generalized B-spline assignment conserves charge for all supported orders.
- Interpolation and deconvolution use the same configured order and preserve order-2 behavior.
- PME diagnostics report the selected assignment order.
- Existing benchmark/private CIC callers keep compatibility wrappers or are updated in the same slice.

**Verification:** `uv run pytest tests/test_pme.py tests/test_forcefields.py -k "pme"`

**Execution:** subagent recommended

**Touches:** `src/mlx_atomistic/pme.py`, `src/mlx_atomistic/forcefields.py`, `src/mlx_atomistic/benchmarks/pme_performance.py`, `tests/test_pme.py`

**Produces:** runtime part of `P2-PME-01`, `AC-02`

**Status:** Complete. Subagent implementation, spec review, and code-quality review approved.

**Evidence:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_pme.py tests/test_forcefields.py -k "pme"` passed outside the sandbox: `41 passed, 23 deselected`; targeted Ruff passed for the touched files. See `orchestration/slice-003-summary.md`.

### Slice 4: PME Schema, Artifact, And Readiness Integration

**Objective:** Make PME orders 4 and 5 usable from prepared systems and artifacts.

**Acceptance criteria:**
- `PreparedSystem` validation accepts PME assignment orders 2, 4, and 5 and rejects other values.
- Artifact metadata and array validation accept only orders 2, 4, and 5.
- PME readiness and parity helpers preserve the configured order.
- Round-trip tests prove order 4/5 metadata survives save/load and artifact construction.

**Verification:** `uv run pytest tests/test_production_artifacts.py tests/test_mlx_prep.py tests/test_openmm_mlx_parity.py -k "pme or assignment_order or prepared"`

**Execution:** subagent recommended

**Depends on:** Slice 3

**Touches:** `src/mlx_atomistic/prep/schema.py`, `src/mlx_atomistic/artifacts.py`, `src/mlx_atomistic/prep/io.py`, `scripts/openmm_mlx_parity.py`, tests

**Produces:** artifact/schema part of `P2-PME-01`, partial `P2-ART-01`, `AC-02`, `AC-06`

**Status:** Complete. Subagent implementation, correction pass, spec re-review, and code-quality re-review approved.

**Evidence:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_production_artifacts.py tests/test_mlx_prep.py tests/test_openmm_mlx_parity.py -k "pme or assignment_order or prepared"` passed outside the sandbox: `46 passed, 66 deselected`; targeted Ruff passed for the touched files. See `orchestration/slice-004-summary.md`.

### Slice 5: AMBER Import Completion And Phase 2 Metadata Alignment

**Objective:** Complete the existing native AMBER `prmtop`/`inpcrd` path for the accepted ff14SB fixture and align its metadata with Phase 2 runtime terms.

**Acceptance criteria:**
- The AMBER importer preserves atoms, residues, bonded terms, impropers, exceptions, constraints, charges, LJ parameters, and periodic box metadata.
- AMBER 1-4 scaling and exception handling are derived from topology data where available instead of relying only on hard-coded defaults.
- Unsupported AMBER records fail closed with explicit blocker text.
- The AMBER parity fixture still builds as a production artifact and can opt into PME order metadata.
- Existing tiny AMBER and OpenMM parity tests continue to pass.

**Verification:** `uv run pytest tests/test_mlx_prep.py tests/test_openmm_mlx_parity.py -k "amber"`

**Execution:** subagent recommended

**Depends on:** Slice 4

**Touches:** `src/mlx_atomistic/prep/topology_import.py`, `scripts/openmm_mlx_parity.py`, `tests/fixtures/amber/`, tests

**Produces:** `P2-PARSE-01`, partial `P2-PARITY-01`, `AC-03`

**Status:** Complete. Subagent implementation, focused correction passes, spec re-review, and code-quality re-review approved.

**Evidence:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_mlx_prep.py tests/test_openmm_mlx_parity.py -k "amber"` passed outside the sandbox: `54 passed, 46 deselected`; targeted Ruff passed for the touched files. See `orchestration/slice-005-summary.md`.

### Slice 6: Native CHARMM PSF/Parameter Import

**Objective:** Add a native accepted CHARMM parser path that maps CHARMM36 supported terms to MLX prepared-system fields without making ParmEd normative.

**Acceptance criteria:**
- A native CHARMM entry point is exported from `mlx_atomistic.prep`.
- PSF atoms, charges, masses, bonds, angles, dihedrals, Urey-Bradley, CMAP, NBFIX, coordinates, and box metadata are mapped where supported.
- Unsupported CHARMM records, virtual sites, or water models produce blockers rather than partial execution.
- Existing ParmEd compatibility tests still pass.

**Verification:** `uv run pytest tests/test_mlx_prep.py tests/test_charmm_terms.py tests/test_production_artifacts.py tests/test_gpcrmd_registry.py -k "charmm or cmap or urey or nbfix"`

**Execution:** subagent recommended

**Depends on:** Slice 2

**Touches:** `src/mlx_atomistic/prep/topology_import.py` or parser helper modules, `src/mlx_atomistic/prep/__init__.py`, CHARMM fixtures, tests

**Produces:** `P2-PARSE-02`, `AC-04`

**Status:** Complete. Subagent implementation, coordinator correction passes, spec re-review, and code-quality re-review approved.

**Evidence:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_mlx_prep.py tests/test_charmm_terms.py tests/test_production_artifacts.py tests/test_gpcrmd_registry.py -k "charmm or cmap or urey or nbfix"` passed outside the sandbox: `69 passed, 152 deselected`; targeted Ruff passed for the touched files. See `orchestration/slice-006-summary.md`.

### Slice 7: Native GROMACS Top/Gro Import

**Objective:** Add a native GROMACS `.top`/`.gro` parser for the declared supported subset, including RB torsions where present.

**Acceptance criteria:**
- A GROMACS entry point is exported from `mlx_atomistic.prep`.
- The supported subset covers molecule expansion, `[defaults]`, `[atomtypes]`, `[moleculetype]`, `[atoms]`, `[bonds]`, `[angles]`, `[dihedrals]`, `[pairs]` or exclusions, coordinates, and box vectors needed by the fixture.
- RB dihedral records map to the RB artifact/runtime surface from Slice 2.
- Unsupported directives or preprocessing features fail closed with actionable blockers.
- Existing `.top` routing heuristics distinguish GROMACS `.top` from AMBER-style topology inputs.

**Verification:** `uv run pytest tests/test_gromacs_import.py tests/test_mlx_prep.py tests/test_gpcrmd_registry.py -k "gromacs or rb"`

**Execution:** subagent recommended

**Depends on:** Slice 2

**Touches:** `src/mlx_atomistic/prep/topology_import.py` or parser helper modules, `src/mlx_atomistic/prep/__init__.py`, GROMACS fixtures, tests

**Produces:** `P2-PARSE-03`, additional `P2-FF-01`, `AC-05`

**Status:** Complete. Subagent implementation, correction pass, spec review, and code-quality re-review approved.

**Evidence:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_gromacs_import.py tests/test_mlx_prep.py tests/test_gpcrmd_registry.py -k "gromacs or rb"` passed in sandbox: `12 passed, 123 deselected`; targeted Ruff and `git diff --check` passed for the touched files. See `orchestration/slice-007-summary.md`.

### Slice 8: Cross-Format Artifact Compatibility Gate

**Objective:** Harden artifact validation and runtime construction so AMBER, CHARMM, and GROMACS imports share one fail-closed production gate.

**Acceptance criteria:**
- Compatibility reports normalize supported, required, unsupported, rejected, and term-count metadata across all parser paths.
- Artifact load/save preserves PME order 4/5, RB arrays, CHARMM-specific arrays, exceptions, constraints, parser provenance, and blockers.
- `build_mlx_system_from_artifact` constructs the expected term list for representative AMBER, CHARMM, and GROMACS artifacts.
- Virtual-site and advanced-water records remain blocked.

**Verification:** `uv run pytest tests/test_production_artifacts.py tests/test_mlx_prep.py -k "artifact or compatibility or unsupported or rb or pme or charmm or gromacs"`

**Execution:** subagent recommended

**Depends on:** Slice 5, Slice 6, Slice 7

**Touches:** `src/mlx_atomistic/artifacts.py`, `src/mlx_atomistic/prep/schema.py`, `src/mlx_atomistic/prep/io.py`, parser metadata tests

**Produces:** `P2-ART-01`, `AC-06`

**Status:** Complete. Subagent implementation, quality correction pass, spec review, and code-quality re-review approved.

**Evidence:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_production_artifacts.py tests/test_mlx_prep.py -k "artifact or compatibility or unsupported or rb or pme or charmm or gromacs"` passed outside the sandbox: `117 passed, 62 deselected`; targeted Ruff and `git diff --check` passed for the touched files. See `orchestration/slice-008-summary.md`.

### Slice 9: OpenMM Parity Harness For Accepted Imports

**Objective:** Extend fixed-coordinate OpenMM parity coverage across accepted AMBER, CHARMM, and GROMACS imports.

**Acceptance criteria:**
- AMBER ff14SB and CHARMM36 protein fixtures compare MLX and OpenMM total and component energies within stated tolerances.
- GROMACS accepted-subset fixture has an OpenMM-referenced parity path or an explicit blocker if OpenMM cannot load an equivalent fixture.
- Component mapping covers the force classes or custom torsion-style terms needed for CHARMM CMAP and GROMACS/RB parity reporting.
- Parity reports include reference engine role, unsupported terms, readiness, component errors, force max error, force RMS error, and pass/block status.
- OpenMM remains confined to tests/scripts and dev dependencies.

**Verification:** `uv run pytest tests/test_openmm_mlx_parity.py -k "amber or charmm or gromacs or pme"`

**Execution:** subagent recommended

**Depends on:** Slice 8

**Touches:** `scripts/openmm_mlx_parity.py`, parity fixtures, `tests/test_openmm_mlx_parity.py`

**Produces:** `P2-PARITY-01`, `AC-07`

**Status:** Complete. Subagent implementation, correction pass, spec re-review, and code-quality re-review approved.

**Evidence:** `uv run pytest tests/test_openmm_mlx_parity.py -k "amber or charmm or gromacs or pme"` passed outside the sandbox: `20 passed, 6 deselected`; runtime boundary gate passed: `8 passed`; targeted Ruff and `git diff --check` passed. See `orchestration/slice-009-summary.md`.

### Slice 10: Phase 2 Regression And Minimal API Docs

**Objective:** Verify the full Phase 2 implementation, update minimal public API/docs references, and leave later roadmap scope deferred.

**Acceptance criteria:**
- All Phase 2 acceptance criteria are covered by targeted tests.
- Existing Phase 1 MD, artifact, and parser tests still pass.
- Public exports and minimal docs mention the new parser entry points, PME assignment-order support, and RB term without broad documentation restructuring.
- Ruff passes on changed code.

**Verification:** `uv run ruff check src tests scripts && uv run pytest`

**Depends on:** Slice 9

**Touches:** `src/`, `tests/`, `scripts/`, minimal docs or README references as needed

**Produces:** `AC-08` and final change readiness

**Status:** Complete. Coordinator implementation, spec re-review, and code-quality re-review approved.

**Evidence:** `uv run pytest` passed outside the sandbox: `636 passed`; `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts` passed; `git diff --check` passed. See `orchestration/slice-010-summary.md`.

## Requirement Traceability

| Gap / AC | Primary slices |
| --- | --- |
| `P2-FF-01`, `AC-01` | Slice 1, Slice 2, Slice 7 |
| `P2-PME-01`, `AC-02` | Slice 3, Slice 4, Slice 9 |
| `P2-PARSE-01`, `AC-03` | Slice 5 |
| `P2-PARSE-02`, `AC-04` | Slice 6, Slice 9 |
| `P2-PARSE-03`, `AC-05` | Slice 7, Slice 9 |
| `P2-ART-01`, `AC-06` | Slice 2, Slice 4, Slice 8 |
| `P2-PARITY-01`, `AC-07` | Slice 5, Slice 9 |
| `AC-08` | Slice 10 |

## Execution Routing And Topology

- Default route: continue directly through slices after each slice verification passes.
- Subagent route: recommended for all material slices because this plan crosses shared runtime, parser, artifact, fixture, or parity surfaces.
- Required human checkpoints: none.
- Parallel-safe groups: none by default. The schema/artifact and parser metadata surfaces are shared enough that parallel implementation should happen only with explicit disjoint file ownership during `auto-execute`.
- Review gate: run `auto-eng-review` before execution because this plan changes shared schemas, parser contracts, and runtime force-term construction.

## Aggregate Verification Commands

| Stage | Command |
| --- | --- |
| RB runtime | `uv run pytest tests/test_forcefields.py -k "dihedral or rb"` |
| RB artifacts | `uv run pytest tests/test_production_artifacts.py tests/test_mlx_prep.py -k "dihedral or rb or artifact"` |
| PME runtime | `uv run pytest tests/test_pme.py tests/test_forcefields.py -k "pme"` |
| PME artifacts | `uv run pytest tests/test_production_artifacts.py tests/test_mlx_prep.py tests/test_openmm_mlx_parity.py -k "pme or assignment_order or prepared"` |
| AMBER | `uv run pytest tests/test_mlx_prep.py tests/test_openmm_mlx_parity.py -k "amber"` |
| CHARMM | `uv run pytest tests/test_mlx_prep.py tests/test_charmm_terms.py tests/test_production_artifacts.py tests/test_gpcrmd_registry.py -k "charmm or cmap or urey or nbfix"` |
| GROMACS | `uv run pytest tests/test_gromacs_import.py tests/test_mlx_prep.py tests/test_gpcrmd_registry.py -k "gromacs or rb"` |
| Artifacts | `uv run pytest tests/test_production_artifacts.py tests/test_mlx_prep.py -k "artifact or compatibility or unsupported or rb or pme or charmm or gromacs"` |
| Parity | `uv run pytest tests/test_openmm_mlx_parity.py -k "amber or charmm or gromacs or pme"` |
| Final | `uv run ruff check src tests scripts && uv run pytest` |

## Context Budget For This Change

This is multi-session work. Execution should load only the active slice, the linked SPEC/gap matrix, `DESIGN.md`, and the named touched files for that slice. Parser fixtures and parity details should be loaded only when their slice is active.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan follows the existing `PreparedSystem` to artifact to runtime pipeline and orders shared RB/PME contracts before parser and parity work.
- Concern: Slices 6 through 9 can stall or produce weak parity evidence if CHARMM36, GROMACS, and protein fixture availability or supported-subset boundaries are not resolved with explicit fail-closed blockers.
- Action: Proceed with `auto-execute` at Slice 1 and treat fixture sourcing, PME deconvolution convention, GROMACS `.top` routing, and CHARMM/GROMACS component mapping as named risks during their owning slices.
- Verified: state diagnostics, PLAN/DESIGN/SPEC alignment, slice dependencies, verification commands, and high-risk source boundaries in `pme.py`, `prep/schema.py`, `artifacts.py`, `topology_import.py`, `gpcrmd.py`, and `scripts/openmm_mlx_parity.py`.
