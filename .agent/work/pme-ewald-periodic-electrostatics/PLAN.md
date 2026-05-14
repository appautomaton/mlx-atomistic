# PLAN: PME/Ewald Periodic Electrostatics

## Goal

Add a validated periodic electrostatics path to `mlx_atomistic`, starting with an Ewald reference implementation and then a PME-compatible interface, so solvated periodic prepared systems can fail closed on precise remaining blockers instead of always rejecting `pme_ewald_periodic_electrostatics`.

## Architecture Approach

Introduce electrostatics mode as a first-class engine setting. Keep the existing direct cutoff path unchanged, add `ewald_reference` as a correctness backend, and reserve `pme` until the mesh algorithm is implemented. Ewald reference becomes the internal oracle for future PME mesh work and for compatibility-report progress on GPCRmd.

## Ordered Task Sequence

### Slice 1: Electrostatics Mode Contract

**Objective:** Add explicit mode/schema validation for `cutoff`, `ewald_reference`, and reserved `pme`.
**Execution:** direct
**Depends on:** none
**Touches:** `src/mlx_atomistic/nonbonded.py`, `src/mlx_atomistic/forcefields.py`, `src/mlx_atomistic/artifacts.py`, `tests/`
**Context budget:** ~8% of context window
**Produces:** Public mode types/config and fail-closed validation for unsupported PME labels.
**Acceptance criteria:**
- Existing direct-cutoff tests remain unchanged.
- `pme` mode refuses to run with a clear "PME mesh not implemented" error.
- Artifact compatibility can distinguish `ewald_reference` support from full `pme` support.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "nonbonded or artifacts or units"`
**Auto-continue:** yes
**Execution evidence:** Done in direct route. Added explicit electrostatics mode validation for `cutoff`, `ewald_reference`, and reserved `pme`; wired artifact compatibility to recognize electrostatics metadata while fail-closing on PME; verified with `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "nonbonded or artifacts or units"`.

### Slice 2: Ewald Reference Energy

**Objective:** Implement neutral orthorhombic Ewald Coulomb energy with real-space, reciprocal-space, and self terms.
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `src/mlx_atomistic/nonbonded.py`, `tests/`
**Context budget:** ~10% of context window
**Produces:** Standalone Ewald energy function/config usable before force integration.
**Acceptance criteria:**
- Zero charges produce zero energy.
- Non-neutral systems fail closed by default.
- Energy is invariant to whole-system translation and periodic wrapping.
- Energy converges as cutoffs are tightened on small neutral fixtures.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "ewald and energy"`
**Auto-continue:** yes
**Execution evidence:** Done in direct route. Added standalone neutral orthorhombic Ewald reference energy with real, reciprocal, and self components; verified zero-charge, non-neutral rejection, periodic invariance, and convergence with `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "ewald and energy"` and `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_ewald_reference.py`.

### Slice 3: Ewald Reference Forces

**Objective:** Add analytical Ewald forces and validate them against finite differences.
**Execution:** subagent recommended
**Depends on:** Slice 2
**Touches:** `src/mlx_atomistic/nonbonded.py`, `tests/`
**Context budget:** ~12% of context window
**Produces:** Ewald energy/force function for neutral orthorhombic cells.
**Acceptance criteria:**
- Forces match finite-difference gradients on small fixtures within tolerance.
- Net force is near zero for closed neutral periodic systems.
- No self-force appears for isolated symmetric cases.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "ewald and force"`
**Auto-continue:** no
**Execution evidence:** Done in direct route because the slice was subagent-recommended, not subagent-required, and implementation remained bounded to standalone Ewald functions/tests. Added analytical real-space and reciprocal-space Ewald forces; verified finite differences, near-zero net force, and no self-force with `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "ewald and force"` and `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_ewald_reference.py`.

### Slice 4: NonbondedPotential Integration

**Objective:** Integrate `ewald_reference` Coulomb into `NonbondedPotential` without changing LJ, exclusions, or explicit exceptions semantics.
**Execution:** subagent recommended
**Depends on:** Slice 3
**Touches:** `src/mlx_atomistic/forcefields.py`, `src/mlx_atomistic/nonbonded.py`, `tests/`
**Context budget:** ~14% of context window
**Produces:** `NonbondedPotential(..., electrostatics="ewald_reference")` with component energies and forces.
**Acceptance criteria:**
- Existing direct `NonbondedPotential` tests stay green.
- Ewald mode refuses missing cells.
- Topology exclusions and 1-4 exceptions are not double-counted.
- Component diagnostics separate LJ, Coulomb real/reciprocal/self, and exceptions where feasible.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "nonbonded or forcefields or ewald"`
**Auto-continue:** no
**Execution evidence:** Done in direct route because the slice was subagent-recommended, not subagent-required, and user did not explicitly request spawned agents. Integrated `ewald_reference` into `NonbondedPotential` while keeping LJ on the existing direct-pair path; added full-system Ewald Coulomb, missing-cell refusal, exclusion corrections, explicit exception corrections, 1-4 Coulomb corrections, and component diagnostics. Verified with `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "nonbonded or forcefields or ewald"` and focused Ruff on touched source/tests.

### Slice 5: Artifact And Compatibility Wiring

**Objective:** Let prepared artifacts request `ewald_reference` and update GPCRmd blocker reports to show PME/Ewald progress without overclaiming full GPCR support.
**Execution:** direct
**Depends on:** Slice 4
**Touches:** `src/mlx_atomistic/artifacts.py`, `src/mlx_atomistic/prep/gpcrmd.py`, `tests/`
**Context budget:** ~10% of context window
**Produces:** Artifact metadata parsing and compatibility reports that distinguish `ewald_reference` from mesh `pme`.
**Acceptance criteria:**
- Artifacts with `electrostatics=ewald_reference` build force terms when cells/charges are valid.
- Artifacts with `electrostatics=pme` still fail closed until mesh PME exists.
- GPCRmd target 729 no longer reports generic PME/Ewald if Ewald reference is enough for a selected small fixture, but still reports mesh/scale/lipid blockers for full GPCRmd.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "artifacts or gpcrmd or ewald"`
**Auto-continue:** yes
**Execution evidence:** Done in direct route. Added artifact metadata aliases and validation for `ewald_reference`, including cell-length and neutral-charge checks; kept PME fail-closed; updated GPCRmd compatibility reporting so small soluble periodic fixtures can use Ewald reference while full target 729 still reports mesh PME, membrane, lipid, CMAP, scale, and related blockers. Verified with `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "artifacts or gpcrmd or ewald"` and focused Ruff on touched source/tests.

### Slice 6: Benchmarks And Diagnostics

**Objective:** Add short Ewald performance and diagnostics probes for small/medium neutral periodic fixtures.
**Execution:** direct
**Depends on:** Slice 4
**Touches:** `src/mlx_atomistic/benchmarks/`, `tests/`, docs if needed
**Context budget:** ~8% of context window
**Produces:** Benchmark output with atom count, k-vector count, real cutoff, alpha, wall time, steps/s or eval/s, and energy components.
**Acceptance criteria:**
- Benchmark is short enough for development runs.
- Output is JSON/CSV friendly.
- Report states that Ewald reference is a correctness backend, not GPCRmd-scale PME.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "benchmark or ewald"`
**Auto-continue:** yes
**Execution evidence:** Done in direct route. Added `mlx_atomistic.benchmarks.ewald_reference`, a short neutral periodic Ewald reference probe with JSON/CSV-safe rows covering atom count, k-vector count, real-space shift count, alpha, real cutoff, timing, force diagnostics, and energy components. The report explicitly labels Ewald reference as a correctness backend and future PME oracle, not GPCRmd-scale PME. Verified with `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "benchmark or ewald"`, focused Ruff, and a smoke run of `uv run python -m mlx_atomistic.benchmarks.ewald_reference --atoms 4 --evaluations 1 --reciprocal-cutoff 1 --json`.

### Slice 7: PME Mesh Planning Checkpoint

**Objective:** Decide whether to start PME mesh implementation from validated Ewald reference results.
**Execution:** direct
**Depends on:** Slices 1-6
**Touches:** `.agent/work/pme-ewald-periodic-electrostatics/`, docs if needed
**Context budget:** ~6% of context window
**Produces:** A checkpoint note listing remaining PME mesh tasks: charge assignment, FFT solve, influence function, interpolation, forces, and validation against Ewald reference.
**Acceptance criteria:**
- Source tests and source lint pass.
- Remaining GPCRmd blockers are updated and exact.
- Full GPCRmd simulation is not claimed unless all other blockers are cleared.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest` and `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts`
**Auto-continue:** no
**Execution evidence:** Done in direct route. Added `PME-MESH-CHECKPOINT.md` with the completed Ewald reference boundary, remaining PME mesh tasks, and exact GPCRmd blockers. Updated steering status to point at the checkpoint/verification handoff. Verified with `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest` (`192 passed`) and `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts`.

## Execution Routing

- Slice 1: direct
- Slice 2: direct
- Slice 3: subagent recommended
- Slice 4: subagent recommended
- Slice 5: direct
- Slice 6: direct
- Slice 7: direct

Use subagents for Slice 3 and Slice 4 when explicitly authorized because force derivation and nonbonded integration are the risk points.

## Verification Commands

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "nonbonded or artifacts or units"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "ewald and energy"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "ewald and force"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "nonbonded or forcefields or ewald"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "artifacts or gpcrmd or ewald"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "benchmark or ewald"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts`

## Context Budget For This Change

Estimated total: ~68% of one large context window if executed end-to-end. Recommended checkpoint after Slice 3 because force validation is the mathematical risk; checkpoint again after Slice 6 before starting mesh PME.

## Recommended Next Skill

Run `auto-eng-review` before execution. The plan is buildable, but Ewald force derivation and exception accounting should be reviewed before code starts.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan isolates periodic electrostatics behind explicit modes, keeps the existing cutoff path intact, and sequences Ewald energy before forces before `NonbondedPotential` integration.
- Concern: The risky parts are analytical Ewald forces and bonded-exception accounting, especially ensuring excluded pairs and 1-4 exceptions are not double-counted when Ewald replaces direct Coulomb.
- Action: Start with Slice 1 mode-gate tests, then stop for a checkpoint after Slice 3 before integrating Ewald into `NonbondedPotential`.
- Verified: Reviewed STATUS.md, SPEC.md, DESIGN.md, PLAN.md, current nonbonded/artifact/unit boundaries, risk points, slice dependencies, and verification commands.
