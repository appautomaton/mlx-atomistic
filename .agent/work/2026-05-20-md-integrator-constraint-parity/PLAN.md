# PLAN: MD Integrator and Constraint Parity

## Goal

Implement the Phase 1 parity surface defined in `SPEC.md`: stronger minimization, SETTLE/HMR, triclinic PBC, Nose-Hoover NVT, anisotropic or membrane-aware MC NPT, and OpenMM-backed bounded validation.

## Architecture Approach

Follow `DESIGN.md`. Preserve the existing public runtime shape and add parity capabilities behind explicit cell, constraint, thermostat, and barostat contracts.

## Requirement Traceability

- P1.1 / AC2: Slice 3.
- P1.2 / AC3 / AC4: Slices 4 and 5.
- P1.3 / AC5: Slices 1 and 2.
- P1.4 / AC6: Slice 6.
- P1.5 / AC7: Slice 7.
- P1.6 / AC8 / AC9: Slice 8.
- AC1 / AC10: All slices, with final verification in Slice 8.

## Ordered Slice Sequence

### Slice 1: Cell Matrix Primitive

**Objective:** Extend `Cell` to represent full 3x3 triclinic matrices while preserving cubic and orthorhombic APIs.

**Acceptance criteria:**
- `Cell.cubic()` and `Cell.orthorhombic()` keep current behavior and tests.
- A full-matrix cell can wrap coordinates, convert fractional/cartesian coordinates, compute volume, and apply minimum-image displacements.
- Invalid cell shapes and singular matrices fail clearly.

**Verification:** `uv run pytest tests/test_core.py tests/test_triclinic_cell.py`

**Touches:** `src/mlx_atomistic/core.py`, focused cell tests.

**Context budget:** ~8%.

**Produces:** Matrix-capable `Cell` compatibility layer.

**Execution evidence:** Completed via direct route. `uv run pytest tests/test_core.py tests/test_triclinic_cell.py` passed with `7 passed in 0.13s`.

### Slice 2: Triclinic Runtime Propagation

**Objective:** Carry matrix-cell behavior through neighbor lists, nonbonded force paths, PME readiness, virial/pressure diagnostics, artifacts, and trajectory/checkpoint metadata.

**Acceptance criteria:**
- Neighbor candidates and nonbonded displacements use `Cell.minimum_image()` correctly for triclinic fixtures or fail closed where a compact path is not yet supported.
- PME and pressure diagnostics no longer assume `volume = prod(lengths)` when a matrix cell is present.
- Artifact and checkpoint/trajectory records preserve triclinic cell shape without breaking orthorhombic records.

**Verification:** `uv run pytest tests/test_triclinic_cell.py tests/test_neighbors.py tests/test_nonbonded_acceleration.py tests/test_pme.py tests/test_virial_pressure.py tests/test_checkpoint_restart.py`

**Execution:** subagent recommended.

**Depends on:** Slice 1.

**Touches:** `src/mlx_atomistic/core.py`, `neighbors.py`, `nonbonded.py`, `pme.py`, `md.py`, `io.py`, `artifacts.py`, related tests.

**Context budget:** ~15%.

**Produces:** Runtime-safe triclinic propagation and fail-closed unsupported paths.

**Execution evidence:** Completed through subagent route. Implementer status `DONE`; spec review `APPROVED`; quality review `APPROVED` after one prepared-system `cell_matrix` persistence fix. Coordinator verification passed: `uv run pytest tests/test_triclinic_cell.py tests/test_neighbors.py tests/test_nonbonded_acceleration.py tests/test_pme.py tests/test_virial_pressure.py tests/test_checkpoint_restart.py tests/test_mlx_prep.py` -> `125 passed in 3.63s`.

### Slice 3: L-BFGS And Conjugate-Gradient Minimizers

**Objective:** Add selectable L-BFGS and conjugate-gradient minimization paths with convergence diagnostics and OpenMM-comparable fixture proof.

**Acceptance criteria:**
- Existing steepest-descent minimization remains available and compatible.
- L-BFGS and conjugate-gradient reduce potential energy and max force on deterministic fixtures.
- Minimizers handle neighbor-list-backed force terms and report convergence reason, steps, energy history, and force history.
- At least one small OpenMM reference fixture shows comparable minimized energy within an explicit tolerance.

**Verification:** `uv run pytest tests/test_minimize.py tests/test_openmm_mlx_parity.py -k "minimize or parity"`

**Touches:** `src/mlx_atomistic/minimize.py`, parity helper scripts, minimizer tests.

**Context budget:** ~10%.

**Produces:** Selectable minimization API and parity-backed minimization tests.

**Execution evidence:** Completed via direct route. `uv run pytest tests/test_minimize.py tests/test_openmm_mlx_parity.py -k "minimize or parity"` passed with `9 passed in 0.35s`. `uv run ruff check src/mlx_atomistic/minimize.py tests/test_minimize.py tests/test_openmm_mlx_parity.py` passed.

### Slice 4: SETTLE Water Constraints

**Objective:** Add analytical SETTLE support for supported water triplets while preserving generic pair-distance constraints.

**Acceptance criteria:**
- SETTLE keeps O-H and H-H distances within declared tolerance for water fixtures.
- Velocity projection removes constrained relative components.
- Generic `DistanceConstraints` tests still pass.
- Malformed or unsupported water topology fails with an explicit error.
- Constraint DOF and reporter max-error behavior remain correct when SETTLE is active.

**Verification:** `uv run pytest tests/test_constraints.py tests/test_real_mm_core.py tests/test_nvt.py -k "settle or constraint or dof"`

**Depends on:** Slice 1.

**Touches:** `src/mlx_atomistic/constraints.py`, `md.py`, examples or prep helpers, constraint tests.

**Context budget:** ~10%.

**Produces:** Analytical water constraint path and compatibility tests.

**Execution evidence:** Completed via direct route. `uv run pytest tests/test_constraints.py tests/test_real_mm_core.py tests/test_nvt.py -k "settle or constraint or dof"` passed with `7 passed, 13 deselected in 0.42s`. `uv run ruff check src/mlx_atomistic/constraints.py src/mlx_atomistic/__init__.py tests/test_constraints.py` passed.

### Slice 5: Hydrogen Mass Repartitioning

**Objective:** Add deterministic HMR mass transformation and provenance through artifact/checkpoint boundaries.

**Acceptance criteria:**
- Selected hydrogen masses increase according to configuration and bonded heavy-atom masses decrease consistently.
- Total system mass is preserved within tolerance.
- HMR provenance records original masses, transformed masses, selected hydrogens, and policy.
- Artifacts/checkpoints can report HMR state without implying virtual-site or unsupported force-field support.

**Verification:** `uv run pytest tests/test_hmr.py tests/test_production_artifacts.py tests/test_checkpoint_restart.py`

**Execution:** subagent recommended.

**Depends on:** Slice 4.

**Touches:** `src/mlx_atomistic/artifacts.py`, `prep/`, `io.py`, constraint/runtime tests.

**Context budget:** ~12%.

**Produces:** HMR policy and serialized provenance.

**Execution evidence:** Completed through subagent route. Implementer status `DONE`; spec review `APPROVED`; quality review `APPROVED` after two focused fixes for final-dtype mass preservation and explicit-subset provenance. Coordinator verification passed: `uv run pytest tests/test_hmr.py tests/test_production_artifacts.py tests/test_checkpoint_restart.py` -> `60 passed in 0.45s`.

### Slice 6: Nose-Hoover NVT

**Objective:** Add deterministic Nose-Hoover NVT support as a separate thermostat path from Langevin BAOAB.

**Acceptance criteria:**
- Nose-Hoover simulations produce finite positions, velocities, energies, and temperatures on a small fixture.
- Thermostat metadata and reporter events identify Nose-Hoover separately from Langevin.
- Checkpoint/restart preserves deterministic continuation state.
- Existing Langevin NVT tests and zero-friction behavior remain compatible.

**Verification:** `uv run pytest tests/test_nvt.py tests/test_runtime_reporters.py tests/test_checkpoint_restart.py -k "nose or thermostat or nvt or checkpoint"`

**Execution:** subagent recommended.

**Depends on:** Slices 1 and 4.

**Touches:** `src/mlx_atomistic/md.py`, `io.py`, `runtime.py`, NVT/reporter/checkpoint tests.

**Context budget:** ~12%.

**Produces:** Nose-Hoover thermostat state and runtime integration.

**Execution evidence:** Completed through subagent route. Implementer status `DONE_WITH_CONCERNS` with non-blocking context/worktree concerns; spec review `APPROVED`; quality review `APPROVED`. Coordinator verification passed: `uv run pytest tests/test_nvt.py tests/test_runtime_reporters.py tests/test_checkpoint_restart.py -k "nose or thermostat or nvt or checkpoint"` -> `17 passed, 2 deselected in 0.34s`.

### Slice 7: Anisotropic And Membrane MC Barostats

**Objective:** Extend MC NPT with anisotropic and membrane/semi-isotropic barostat modes over the matrix-cell runtime.

**Acceptance criteria:**
- Isotropic MC NPT behavior and tests remain compatible.
- Anisotropic proposals update allowed axes independently and rescale coordinates consistently.
- Membrane/semi-isotropic proposals keep plane and normal-axis policy explicit.
- Barostat attempts, acceptances, pressure metadata, cell history, and final cell are reported.
- Unsupported virial or cell configurations fail before pressure-coupled runtime claims.

**Verification:** `uv run pytest tests/test_npt.py tests/test_virial_pressure.py tests/test_runtime_reporters.py -k "barostat or npt or pressure"`

**Execution:** subagent recommended.

**Depends on:** Slices 2, 4, and 6.

**Touches:** `src/mlx_atomistic/md.py`, `core.py`, `io.py`, protocol gates, NPT tests.

**Context budget:** ~14%.

**Produces:** Phase 1 pressure-coupling modes and fail-closed gates.

**Execution evidence:** Completed through subagent route. Implementer status `DONE`; spec review `APPROVED` after adding a real `DistanceConstraints` NPT compatibility test; quality review `APPROVED` after fixing accepted-barostat public sampled output and neighbor-managed/lazy-topology result assembly. Coordinator verification passed: `uv run pytest tests/test_npt.py tests/test_virial_pressure.py tests/test_runtime_reporters.py -k "barostat or npt or pressure"` -> `19 passed, 5 deselected in 0.94s`. Targeted Ruff passed for Slice 7 files.

### Slice 8: Phase 1 OpenMM Parity And End-To-End Proof

**Objective:** Add bounded OpenMM-backed parity fixtures and one end-to-end Phase 1 protocol proof.

**Acceptance criteria:**
- OpenMM reference evidence remains reference-only and reproducible through `uv run`.
- Parity checks cover minimized energy, constrained water geometry, triclinic periodic distances, thermostat temperature statistics, and barostat cell/volume trends.
- A bounded proof runs minimize -> Nose-Hoover NVT -> anisotropic or membrane MC NPT and records finite energies, coordinates, cell state, and no unsupported-term blockers.
- Final lint and full test suite pass.

**Verification:** `uv run pytest tests/test_openmm_mlx_parity.py tests/test_md_phase1_end_to_end.py && uv run ruff check src tests scripts && uv run pytest`

**Execution:** subagent recommended.

**Depends on:** Slices 1 through 7.

**Touches:** parity scripts, bounded fixtures, docs or evidence notes, final verification report.

**Context budget:** ~12%.

**Produces:** Phase 1 parity report and final acceptance evidence.

**Execution evidence:** Completed through subagent route. Implementer status `DONE`; spec review `APPROVED`; quality review `APPROVED` after moving the default AMBER parity fixture out of ignored `vendors/` into tracked `tests/fixtures/amber/`. Coordinator verification passed: `uv run pytest tests/test_openmm_mlx_parity.py tests/test_md_phase1_end_to_end.py` -> `10 passed in 0.62s`; `uv run ruff check src tests scripts` -> passed; `uv run pytest` -> `486 passed in 39.94s`.

## Execution Routing And Topology

- Default route: direct execution by the main agent.
- Subagent recommended: Slices 2, 5, 6, 7, and 8 because they cross shared runtime interfaces or validation surfaces.
- Subagent required: none. The user has not requested multi-agent execution.
- Checkpoints: none. Continue after each slice when its verification passes.
- Parallel-safe groups: none by default. The slices share core runtime contracts, so serial execution is the safe route.
- Engineering review: recommended before execution because the plan touches cross-cutting runtime contracts, checkpoint metadata, and parity criteria.

## Aggregate Verification Commands

| Stage | Command |
| --- | --- |
| Slice 1 | `uv run pytest tests/test_core.py tests/test_triclinic_cell.py` |
| Slice 2 | `uv run pytest tests/test_triclinic_cell.py tests/test_neighbors.py tests/test_nonbonded_acceleration.py tests/test_pme.py tests/test_virial_pressure.py tests/test_checkpoint_restart.py` |
| Slice 3 | `uv run pytest tests/test_minimize.py tests/test_openmm_mlx_parity.py -k "minimize or parity"` |
| Slice 4 | `uv run pytest tests/test_constraints.py tests/test_real_mm_core.py tests/test_nvt.py -k "settle or constraint or dof"` |
| Slice 5 | `uv run pytest tests/test_hmr.py tests/test_production_artifacts.py tests/test_checkpoint_restart.py` |
| Slice 6 | `uv run pytest tests/test_nvt.py tests/test_runtime_reporters.py tests/test_checkpoint_restart.py -k "nose or thermostat or nvt or checkpoint"` |
| Slice 7 | `uv run pytest tests/test_npt.py tests/test_virial_pressure.py tests/test_runtime_reporters.py -k "barostat or npt or pressure"` |
| Slice 8 / final | `uv run pytest tests/test_openmm_mlx_parity.py tests/test_md_phase1_end_to_end.py && uv run ruff check src tests scripts && uv run pytest` |

## Context Budget For This Change

This phase is likely multi-session work. Load the active `SPEC.md`, `DESIGN.md`, and the slice currently being executed first, then load source/test files named in that slice. Avoid reloading later-slice context until dependencies have passed.
