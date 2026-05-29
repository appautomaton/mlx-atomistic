# PLAN: Physics Completeness and Production Hardening

## Goal

Implement the Phase 3 SPEC: virtual sites, custom forces, GBSA, soft-core/lambda, replica exchange, with validated OpenMM parity, then Phase 4 CI/docs/version hardening.

## Architecture Approach

Use the existing `ForceTerm` protocol as the integration spine. Each new capability adds classes following `energy_forces(positions, cell, pairs)` and registers in the artifact pipeline. `NonbondedPotential` receives internal modifications for virtual-site redistribution and soft-core/lambda in separate slices. See `DESIGN.md` for module boundaries, redistribution architecture, and fail-closed surface changes.

## Ordered Slice Sequence

### Slice 1: Virtual-Site Geometry and Topology Integration

**Objective:** Add VirtualSite base classes and topology fields so that virtual-site geometry can be defined and carried through the prepared-system pipeline.

**Acceptance criteria:**
- `VirtualSite` base classes (TwoParticleAverage, ThreeParticleAverage, OutOfPlane, LocalCoordinates) compute positions from parent atoms.
- `Topology` accepts `virtual_sites` and `virtual_site_types` fields with backward-compatible defaults.
- `PreparedSystem` validation accepts virtual-site arrays; artifact pipeline moves `virtual_site` from `FAIL_CLOSED_TERMS` to `SUPPORTED_FORCE_TERMS` and `tip4p`/`opc`/`advanced_water` remain blocked until their owning slices.
- Virtual-site position reconstruction produces correct geometry for a test TIP4P-like configuration.
- Existing Phase 1+2 tests pass without modification.

**Verification:** `uv run pytest tests/test_virtual_sites.py tests/test_topology.py tests/test_production_artifacts.py -k "virtual_site or topology" && uv run ruff check src/mlx_atomistic/virtual_sites.py src/mlx_atomistic/topology.py`

**Execution:** subagent recommended

**Depends on:** none

**Touches:** `src/mlx_atomistic/virtual_sites.py` (new), `src/mlx_atomistic/topology.py`, `src/mlx_atomistic/artifacts.py`, `src/mlx_atomistic/prep/schema.py`, `src/mlx_atomistic/__init__.py`, `tests/test_virtual_sites.py` (new)

**Produces:** P3-VS-01, P3-VS-05, partial P3-VS-06

**Status:** Complete. Subagent implementation and self-review approved.

**Evidence:** 21 new virtual-site tests pass; 28 topology/artifact tests pass; full regression 694 passed; ruff clean. See `orchestration/slice-001-summary.md`.

### Slice 2: Custom Force Expressions

**Objective:** Add `CustomForcePotential` with symbolic expression evaluation so that downstream force terms (GBSA) can use it.

**Acceptance criteria:**
- `CustomForcePotential` accepts a string expression and per-particle or per-pair parameters; energy and forces match finite-difference reference for bond-like, angle-like, and nonbonded-like expressions.
- `CustomForcePotential` is exported from the package.
- Artifact pipeline accepts `custom_force` terms with expression metadata.
- Existing force-term tests pass.

**Verification:** `uv run pytest tests/test_custom_force.py tests/test_forcefields.py tests/test_production_artifacts.py -k "custom_force or CustomForce or expression" && uv run ruff check src/mlx_atomistic/custom_force.py src/mlx_atomistic/forcefields.py`

**Execution:** subagent recommended

**Depends on:** none (parallel-safe with Slice 1; touches disjoint files)

**Touches:** `src/mlx_atomistic/custom_force.py` (new), `src/mlx_atomistic/forcefields.py`, `src/mlx_atomistic/artifacts.py`, `src/mlx_atomistic/__init__.py`, `tests/test_custom_force.py` (new)

**Produces:** P3-CF-01, P3-CF-02

**Context budget:** ~15%

**Status:** Complete. Subagent implementation and self-review approved.

**Evidence:** 37 custom force tests pass; full regression 694 passed; ruff clean. See `orchestration/slice-002-summary.md`.

### Slice 3: Virtual-Site Force Redistribution and Constraint Integration

**Objective:** Add force redistribution from virtual sites to parent atoms and virtual-site position reconstruction in the MD timestep.

**Acceptance criteria:**
- Virtual-site forces are redistributed to parent atoms using geometry-derived weight matrices; redistributed forces match analytically derived reference.
- Virtual-site positions are reconstructed from parent atoms each timestep before force evaluation.
- `VirtualSiteManager` integrates with `simulate_nvt`/`simulate_npt` via an optional `virtual_sites` parameter; existing simulation signatures unchanged.
- SETTLE and DistanceConstraints continue to operate on real atoms only.
- Existing Phase 1+2 tests pass.

**Verification:** `uv run pytest tests/test_virtual_sites.py tests/test_md.py tests/test_constraints.py tests/test_nvt.py -k "virtual_site or redistribute or vsite" && uv run ruff check src/mlx_atomistic/virtual_sites.py src/mlx_atomistic/md.py`

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `src/mlx_atomistic/virtual_sites.py`, `src/mlx_atomistic/md.py`, `src/mlx_atomistic/nonbonded.py`

**Produces:** P3-VS-02, P3-VS-03

**Status:** Complete. Subagent implementation, spec review, and quality re-review approved.

**Evidence:** Slice verification passed: `25 passed, 18 deselected`; targeted ruff passed; full regression `698 passed`; `git diff --check` passed. See `orchestration/slice-003-summary.md`.

### Slice 4: TIP4P-Ew Water Model and Parity

**Objective:** Implement TIP4P-Ew water using virtual-site geometry, with OpenMM-referenced energy parity.

**Acceptance criteria:**
- TIP4P-Ew water model produces correct geometry (bond length, angle, virtual-site position).
- TIP4P-Ew water box energy matches OpenMM reference within stated tolerance.
- Parser entry points populate virtual-site topology for TIP4P-Ew.
- Artifact pipeline accepts `tip4p` water model; `advanced_water` and `opc` remain blocked.
- Existing Phase 1+2 tests pass.

**Verification:** `uv run pytest tests/test_virtual_sites.py tests/test_openmm_mlx_parity.py -k "tip4p or virtual_site or vsite" && uv run ruff check src/mlx_atomistic/virtual_sites.py src/mlx_atomistic/prep/`

**Execution:** subagent recommended

**Depends on:** Slice 3

**Touches:** `src/mlx_atomistic/virtual_sites.py`, `src/mlx_atomistic/prep/topology_import.py`, `src/mlx_atomistic/prep/schema.py`, `src/mlx_atomistic/prep/io.py`, `src/mlx_atomistic/prep/runner.py`, `src/mlx_atomistic/artifacts.py`, `src/mlx_atomistic/mm.py`, `src/mlx_atomistic/md.py`, `scripts/openmm_mlx_parity.py`, `tests/test_virtual_sites.py`, `tests/test_openmm_mlx_parity.py`

**Correction:** Spec review found TIP4P virtual-site arrays must be serialized by `src/mlx_atomistic/prep/io.py`; this file is in scope for Slice 4 so parser-populated virtual sites survive save/load and artifact construction.

**Correction:** Quality review found accepted TIP4P artifacts must expose runtime virtual-site handling so M sites are not integrated as ordinary massive atoms. Targeted edits to `src/mlx_atomistic/mm.py` and `src/mlx_atomistic/md.py` are in scope only to carry a `VirtualSiteManager`/real-atom view without changing `build_mlx_system_from_artifact` return shape or public `simulate_nvt`/`simulate_npt` signatures.

**Status:** Blocked after implementation/review loop. Spec review approved, but quality re-review found the same runtime-integration issue is still unresolved in runner-created `SimulationConfig` objects.

**Correction:** `src/mlx_atomistic/prep/runner.py` is in scope for Slice 4. Runner-created `SimulationConfig` objects must propagate `system.virtual_sites` for NVT, NPT, minimize/equilibration, and steered paths as applicable, without changing `simulate_nvt`/`simulate_npt` signatures or `build_mlx_system_from_artifact` return shape.

**Status:** Complete. Subagent implementation, spec review, and quality re-review approved after runner propagation correction.

**Evidence:** Slice verification passed: `31 passed, 26 deselected`; targeted ruff passed; full regression `704 passed`; `git diff --check` passed. See `orchestration/slice-004-summary.md`.

**Produces:** P3-VS-04, P3-VS-06

### Slice 5: GBSA/OBC Implicit Solvent

**Objective:** Add GB-OBC implicit solvent with ACE surface-area approximation.

**Acceptance criteria:**
- `GBSAForcePotential` computes GB-OBC energy and forces for neutral and charged periodic fixtures.
- ACE surface-area term is computed and validated against analytical reference.
- GBSA energy matches OpenMM GB-OBC reference within stated tolerance for a protein fixture.
- `gbsa` is moved from `FAIL_CLOSED_TERMS` to `SUPPORTED_FORCE_TERMS`; artifact load/save preserves GB parameters.
- Existing Phase 1+2 tests pass.

**Verification:** `uv run pytest tests/test_gbsa.py tests/test_production_artifacts.py -k "gbsa or implicit or obc" && uv run ruff check src/mlx_atomistic/gbsa.py src/mlx_atomistic/forcefields.py`

**Execution:** subagent recommended

**Depends on:** Slice 2 (CustomForcePotential used for expression evaluation)

**Touches:** `src/mlx_atomistic/gbsa.py` (new), `src/mlx_atomistic/forcefields.py`, `src/mlx_atomistic/artifacts.py`, `src/mlx_atomistic/__init__.py`, `tests/test_gbsa.py` (new)

**Produces:** P3-GB-01, P3-GB-02, P3-GB-03

**Status:** Complete. Subagent implementation, spec re-review, and quality re-review approved.

**Evidence:** Slice verification passed: `8 passed, 73 deselected`; targeted ruff passed; full regression `712 passed`; `git diff --check` passed. See `orchestration/slice-005-summary.md`.

### Slice 6: Soft-Core Potentials and Lambda Scaling

**Objective:** Add soft-core LJ and Coulomb potentials with lambda-dependent scaling and `dU/dlambda` for thermodynamic integration.

**Acceptance criteria:**
- `NonbondedPotential` accepts optional `lambda_lj` and `lambda_electrostatics` parameters; when `lambda < 1`, LJ and Coulomb use soft-core potentials with finite energies at `r=0`.
- Soft-core LJ matches hard-core LJ at `lambda_electrostatics = 1` and `lambda_lj = 1`.
- `dU/dlambda` is computed analytically alongside energy and forces.
- `SoftCoreNonbondedPotential` wraps `NonbondedPotential` with lambda for artifact construction.
- Artifact pipeline accepts `soft_core_lj` and `lambda_scaled_nonbonded` terms.
- Existing Phase 1+2 tests continue to pass (default `lambda=1.0` preserves current behavior).

**Verification:** `uv run pytest tests/test_soft_core.py tests/test_forcefields.py tests/test_production_artifacts.py -k "soft_core or lambda or alchemical" && uv run ruff check src/mlx_atomistic/nonbonded.py src/mlx_atomistic/forcefields.py`

**Execution:** subagent recommended

**Depends on:** Slice 3 (virtual-site redistribution already modified `nonbonded.py`; this slice modifies the same file in a separate region)

**Touches:** `src/mlx_atomistic/nonbonded.py`, `src/mlx_atomistic/forcefields.py`, `src/mlx_atomistic/artifacts.py`, `tests/test_soft_core.py` (new)

**Produces:** P3-SC-01, P3-SC-02, P3-SC-03, P3-SC-04

**Status:** Complete. Subagent implementation, spec review, quality re-review, and regression-fix pass approved.

**Evidence:** Slice verification passed: `8 passed, 108 deselected`; targeted ruff passed; full regression `720 passed`; `git diff --check` passed. See `orchestration/slice-006-summary.md`.

### Slice 7: Replica Exchange

**Objective:** Add a multi-copy replica exchange driver for temperature and Hamiltonian exchange.

**Acceptance criteria:**
- `simulate_replica_exchange` manages N replicas with distinct temperatures or lambda-scaled Hamiltonians.
- Swap attempts occur at configurable intervals; acceptance probability matches the Metropolis criterion (verified by energy histogram overlap across temperatures).
- Existing `simulate_nvt`/`simulate_npt` signatures are unchanged.
- Artifact pipeline accepts `replica_exchange` configuration metadata.
- Existing Phase 1+2 tests pass.

**Verification:** `uv run pytest tests/test_replica_exchange.py -k "replica or exchange or histogram" && uv run ruff check src/mlx_atomistic/replica_exchange.py src/mlx_atomistic/md.py`

**Execution:** subagent recommended

**Depends on:** Slice 6 (Hamiltonian exchange uses lambda-scaled `NonbondedPotential`)

**Touches:** `src/mlx_atomistic/replica_exchange.py` (new), `src/mlx_atomistic/md.py`, `src/mlx_atomistic/artifacts.py`, `src/mlx_atomistic/__init__.py`, `tests/test_replica_exchange.py` (new)

**Produces:** P3-RE-01, P3-RE-02, P3-RE-03

**Status:** Complete. Subagent implementation, spec re-review, and quality final re-review approved.

**Evidence:** Slice verification passed: `16 passed`; targeted ruff passed; full regression `736 passed`; `git diff --check` passed. See `orchestration/slice-007-summary.md`.

### Slice 8: Phase 3 Regression and OpenMM Parity Gate

**Objective:** Verify that all Phase 3 capabilities work together and that the full regression suite is green.

**Acceptance criteria:**
- All Phase 3 acceptance criteria (AC-01 through AC-06) are covered by targeted tests.
- Existing Phase 1+2 tests still pass.
- Public exports mention virtual sites, custom forces, GBSA, soft-core, lambda, and replica exchange.
- Ruff passes on changed code.

**Verification:** `uv run ruff check src tests scripts && uv run pytest`

**Depends on:** Slice 7

**Touches:** `src/`, `tests/`, minimal docs references

**Produces:** AC-01 through AC-06 regression evidence

**Status:** Complete. Phase 3 regression and OpenMM parity gate passed.

**Evidence:** `uv run ruff check src tests scripts && uv run pytest` passed: ruff clean, `736 passed`; `git diff --check` passed. See `orchestration/slice-008-summary.md`.

### Slice 9: CI Pipeline and Version Bump

**Objective:** Add GitHub Actions CI, bump the version, and link docs from README.

**Acceptance criteria:**
- `.github/workflows/ci.yml` runs `pytest` and `ruff check` on every PR.
- `pyproject.toml` version is bumped to the next 0.x minor.
- Docs are restructured and linked from README; new capabilities have minimal API docs.
- CI passes on the branch.

**Verification:** `.github/workflows/ci.yml` exists; `grep -q "version" pyproject.toml`; `uv run ruff check src tests scripts && uv run pytest`

**Depends on:** Slice 8 (after Phase 3 stabilizes)

**Touches:** `.github/`, `pyproject.toml`, `README.md`, `docs/`

**Produces:** P4-CI-01, P4-CI-02, P4-CI-03, AC-07

**Status:** Complete. Direct implementation verified.

**Evidence:** `.github/workflows/ci.yml` added; `pyproject.toml` version bumped to `0.2.0`; README docs index added; `uv run ruff check src tests scripts && uv run pytest` passed with `736 passed`; `git diff --check` passed. See `orchestration/slice-009-summary.md`.

### Slice 10: Phase 4 Final Regression

**Objective:** Full regression after CI and version bump.

**Acceptance criteria:**
- Full test suite passes.
- Ruff passes.
- CI workflow runs green.

**Verification:** `uv run ruff check src tests scripts && uv run pytest`

**Depends on:** Slice 9

**Touches:** minimal (CI validation only)

**Produces:** final AC-07 evidence

**Status:** Complete. Final Phase 4 regression passed.

**Evidence:** `uv run ruff check src tests scripts && uv run pytest && git diff --check` passed: ruff clean, `736 passed`, whitespace clean. See `orchestration/slice-010-summary.md`.

## Requirement Traceability

| Gap / AC | Primary slices |
|----------|----------------|
| P3-VS-01, P3-VS-05, partial P3-VS-06 | Slice 1 |
| P3-VS-02, P3-VS-03 | Slice 3 |
| P3-VS-04, P3-VS-06 | Slice 4 |
| P3-CF-01, P3-CF-02 | Slice 2 |
| P3-GB-01, P3-GB-02, P3-GB-03 | Slice 5 |
| P3-SC-01, P3-SC-02, P3-SC-03, P3-SC-04 | Slice 6 |
| P3-RE-01, P3-RE-02, P3-RE-03 | Slice 7 |
| AC-01 | Slices 1, 3, 4 |
| AC-02 | Slice 4 |
| AC-03 | Slice 2 |
| AC-04 | Slice 5 |
| AC-05 | Slice 6 |
| AC-06 | Slice 7 |
| AC-07 | Slices 9, 10 |
| P4-CI-01, P4-CI-02, P4-CI-03 | Slice 9 |

## Execution Routing and Topology

- Default route: continue directly through slices after each slice verification passes.
- Subagent route: recommended for all physics slices (Slices 1–7) because they cross subsystem boundaries or modify shared interfaces.
- Parallel-safe groups:
  - **Slices 1 and 2 can run in parallel.** Slice 1 touches `virtual_sites.py` (new), `topology.py`, `artifacts.py` (virtual_site entries). Slice 2 touches `custom_force.py` (new), `forcefields.py` (new class), `artifacts.py` (custom_force entries). No shared state; disjoint write sets except `artifacts.py` term list additions, which are in different regions. Coordinate by adding terms in their own slice with no overlap.
  - **Slices 5 and 6 can overlap partially.** Slice 5 (GBSA) touches `gbsa.py` (new), `forcefields.py` (new class), `artifacts.py` (gbsa entries). Slice 6 (soft-core) touches `nonbonded.py`, `forcefields.py` (new class), `artifacts.py` (soft_core entries). Different regions of `forcefields.py` and `artifacts.py`. Subagents must not modify the same function signatures simultaneously.
- Required human checkpoints: none.
- Review gate: run `auto-eng-review` before execution because this plan modifies shared schemas, artifact contracts, runtime force-term construction, and the `nonbonded.py` hot zone.

## Aggregate Verification Commands

| Stage | Command |
|-------|---------|
| Virtual-site geometry | `uv run pytest tests/test_virtual_sites.py tests/test_topology.py tests/test_production_artifacts.py -k "virtual_site or topology"` |
| Custom forces | `uv run pytest tests/test_custom_force.py tests/test_forcefields.py tests/test_production_artifacts.py -k "custom_force or CustomForce or expression"` |
| Virtual-site redistribution | `uv run pytest tests/test_virtual_sites.py tests/test_md.py tests/test_constraints.py tests/test_nvt.py -k "virtual_site or redistribute or vsite"` |
| TIP4P-Ew parity | `uv run pytest tests/test_virtual_sites.py tests/test_openmm_mlx_parity.py -k "tip4p or virtual_site or vsite"` |
| GBSA | `uv run pytest tests/test_gbsa.py tests/test_production_artifacts.py -k "gbsa or implicit or obc"` |
| Soft-core/lambda | `uv run pytest tests/test_soft_core.py tests/test_forcefields.py tests/test_production_artifacts.py -k "soft_core or lambda or alchemical"` |
| Replica exchange | `uv run pytest tests/test_replica_exchange.py -k "replica or exchange or histogram"` |
| Phase 3 regression | `uv run ruff check src tests scripts && uv run pytest` |
| CI/version/docs | `.github/workflows/ci.yml` exists; `grep -q "version" pyproject.toml`; `uv run ruff check src tests scripts && uv run pytest` |
| Phase 4 regression | `uv run ruff check src tests scripts && uv run pytest` |

## Context Budget for This Change

This is multi-session work. Execution should load only the active slice, the SPEC, DESIGN, and the named touched files for that slice. Virtual-site, GBSA, and replica-exchange details should be loaded only when their slice is active.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: Slices are traceable to gap IDs and keep the high-conflict surfaces (`nonbonded.py`, `artifacts.py`, `md.py`) mostly separated; Slices 1 and 2 already have regression evidence.
- Concern: Slice 3 has an API ambiguity (`simulate_nvt`/`simulate_npt` signatures unchanged vs. adding optional `virtual_sites`), and Slices 5/6 are marked as partially overlapping despite shared `forcefields.py` and `artifacts.py` write areas.
- Action: Resolve Slice 3 integration to one path before editing, preferably preserving public simulation signatures via `SimulationConfig` or a wrapper; run Slices 5/6 serially unless file ownership is explicitly split before dispatch.
- Verified: Read `STATUS.md`, `PLAN.md`, `DESIGN.md`; inspected `virtual_sites.py`, `custom_force.py`, `md.py`, `artifacts.py`, and `nonbonded.py`; verification commands cover per-slice targeted tests plus full regression gates.
