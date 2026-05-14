# PLAN: GPCRmd-Critical MLX MD Readiness

## Goal

Implement the GPCRmd-critical missing engine and import capabilities so a complete GPCRmd membrane/solvent system can be imported, validated, simulated by `mlx_atomistic`, and visualized from an MLX-generated trajectory.

## Architecture Approach

Build readiness as gated engine capabilities, not broad MD feature parity. First inventory the selected GPCRmd target and freeze the exact required terms. Then implement PME mesh, CHARMM/GPCR force terms, scalable periodic nonbonded execution, import/export wiring, virial/pressure, protocol support, and notebook consumption. Each gate fails closed until the real selected target can run through MLX.

## Ordered Task Sequence

### Slice 1: GPCRmd Target Inventory Gate

**Objective:** Inspect the selected GPCRmd target package and produce the exact MLX readiness inventory.
**Execution:** subagent recommended
**Depends on:** none
**Touches:** `src/mlx_atomistic/prep/gpcrmd.py`, `src/mlx_atomistic/prep/`, `tests/test_gpcrmd_registry.py`, `.agent/work/gpcrmd-critical-md-readiness/`
**Context budget:** ~8% of context window
**Produces:** A target inventory/report naming files, force terms, water/lipid models, box, constraints, exceptions, protocol requirements, and blockers.
**Acceptance criteria:**
- The selected target is fixed or replaced with a documented lower-blocker GPCRmd target.
- The report distinguishes required terms from optional analysis features.
- The report lists exact first engine blockers instead of generic “production validation”.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "gpcrmd and inventory"` plus a fixture `uv run mlx_atomistic.prep Python API gpcrmd-inspect --target gpcrmd-729-beta1-5f8u-cyanopindolol --cache <fixture-cache> --compatibility --json`
**Auto-continue:** no

**Status:** completed. Evidence: `.agent/work/gpcrmd-critical-md-readiness/SLICE-1-GPCRMD-INVENTORY.md` and `.agent/work/gpcrmd-critical-md-readiness/orchestration/slice-1-gpcrmd-target-inventory-gate.md`.
**Verified:** `tests/test_gpcrmd_registry.py` passed (`18 passed`), `tests -k "gpcrmd and inventory"` passed (`3 passed, 194 deselected`), targeted Ruff passed, and the fixture `gpcrmd-inspect --compatibility --json` smoke emitted `mlx_readiness_inventory` with exact blockers.
**Correction recorded:** JSON manifest entries now count as present only when their resolved local file exists; missing model/topology/parameters/protocol files block MLX readiness, while missing reference trajectories remain optional analysis inputs.

### Slice 2: PME Mesh Standalone Backend

**Objective:** Add a standalone PME mesh electrostatics backend validated against Ewald reference fixtures.
**Execution:** subagent recommended
**Depends on:** Slice 1
**Touches:** `src/mlx_atomistic/pme.py`, `tests/test_pme.py`, `src/mlx_atomistic/benchmarks/`
**Context budget:** ~14% of context window
**Produces:** Mesh charge assignment, FFT solve, influence function, interpolation, PME energy/forces, and diagnostics for neutral orthorhombic fixtures.
**Acceptance criteria:**
- PME energy/forces match Ewald reference within stated tolerances on small neutral periodic fixtures.
- PME refuses unsupported cells, non-neutral policy gaps, or invalid mesh settings.
- Benchmark output compares PME mesh against Ewald reference.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "pme or ewald"` and `uv run python -m mlx_atomistic.benchmarks.ewald_reference --atoms 4 --evaluations 1 --json`
**Auto-continue:** yes

**Status:** completed. Evidence: `.agent/work/gpcrmd-critical-md-readiness/orchestration/slice-2-pme-mesh-standalone-backend.md`.
**Verified:** `tests -k "pme or ewald"` passed (`27 passed, 181 deselected`), targeted Ruff passed, and the requested Ewald benchmark JSON emitted finite PME comparison fields.
**Correction recorded:** PME input validation now rejects non-finite positions/charges and fractional mesh dimensions instead of producing NaNs or silently truncating mesh sizes.

### Slice 3: CHARMM/GPCR Force-Term Primitives

**Objective:** Implement the selected target’s required CHARMM/GPCR force-term primitives outside the artifact importer.
**Execution:** subagent recommended
**Depends on:** Slice 1
**Touches:** `src/mlx_atomistic/charmm_terms.py`, `src/mlx_atomistic/forcefields.py`, `tests/test_charmm_terms.py`
**Context budget:** ~12% of context window
**Produces:** CMAP and any required Urey-Bradley, force-switch, NBFIX/pair-override, or lipid term primitives with finite force tests.
**Acceptance criteria:**
- Each implemented term has finite energy/force tests and finite-difference checks where practical.
- Terms not required by the selected GPCRmd target remain fail-closed rather than stubbed.
- Public names match artifact-loader terminology.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "charmm or cmap or nbfix"`
**Auto-continue:** yes

**Status:** completed. Evidence: `.agent/work/gpcrmd-critical-md-readiness/orchestration/slice-3-charmm-gpcr-force-term-primitives.md`.
**Verified:** `tests/test_charmm_terms.py` passed (`31 passed`), `tests -k "charmm or cmap or nbfix"` passed (`31 passed, 208 deselected`), and targeted Ruff passed.
**Correction recorded:** CHARMM force-switch now implements a dedicated LJ force-switch primitive with reference tests; NBFIX restricted pair evaluation fails closed; CMAP tests include grid nodes, periodic seam, and multiple maps; non-finite/invalid CHARMM inputs are rejected before they can produce NaNs.

### Slice 4: Scalable Periodic Nonbonded Execution

**Objective:** Add a periodic cell-list or pair-list path suitable for GPCRmd-scale LJ/direct real-space work.
**Execution:** subagent recommended
**Depends on:** Slice 1
**Touches:** `src/mlx_atomistic/cell_list.py`, `src/mlx_atomistic/neighbors.py`, `tests/test_nonbonded_acceleration.py`, `src/mlx_atomistic/benchmarks/`
**Context budget:** ~12% of context window
**Produces:** A scalable periodic pair-construction backend with deterministic pair counts, rebuild policy, memory estimates, and benchmark rows.
**Acceptance criteria:**
- Pair construction avoids dense all-pairs memory for large periodic fixtures.
- Energies/forces match dense reference on small fixtures.
- Benchmarks report pair count, rebuild count, memory estimate, and backend choice.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "neighbor or nonbonded_acceleration"`
**Auto-continue:** yes

**Status:** completed. Evidence: `.agent/work/gpcrmd-critical-md-readiness/orchestration/slice-4-scalable-periodic-nonbonded-execution.md`.
**Verified:** `tests/test_neighbors.py tests/test_nonbonded_acceleration.py` passed (`18 passed`), `tests -k "neighbor or nonbonded_acceleration"` passed (`21 passed, 223 deselected`), targeted Ruff passed, and the `md_acceleration` JSON smoke emitted backend, pair count, rebuild count, pair-memory estimate, and dense-memory estimate fields.
**Correction recorded:** `NeighborListManager.needs_rebuild()` now rejects invalid or non-finite positions before displacement math or interval skipping, preventing stale pair-list reuse.

### Slice 5: PME NonbondedPotential Integration

**Objective:** Wire the standalone PME backend into production nonbonded force evaluation.
**Execution:** subagent recommended
**Depends on:** Slices 2-4
**Touches:** `src/mlx_atomistic/nonbonded.py`, `src/mlx_atomistic/forcefields.py`, `tests/test_forcefields.py`, `tests/test_pme.py`
**Context budget:** ~12% of context window
**Produces:** `NonbondedPotential(electrostatics="pme")` with LJ/direct real-space handling, exclusions, exceptions, 1-4 corrections, PME components, and missing-cell/mesh validation.
**Acceptance criteria:**
- Existing `cutoff` and `ewald_reference` nonbonded tests stay green.
- PME mode refuses missing cells, missing mesh settings, unsupported cells, and unsupported charge policy.
- PME mode reports component energies and avoids double-counting exclusions, exceptions, and 1-4 corrections.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "pme or nonbonded or forcefields"`
**Auto-continue:** yes

**Status:** completed. Evidence: `.agent/work/gpcrmd-critical-md-readiness/orchestration/slice-5-pme-nonbondedpotential-integration.md`.
**Verified:** `tests/test_forcefields.py -k "pme or ewald" tests/test_pme.py` passed (`20 passed, 12 deselected`), `tests -k "pme or nonbonded or forcefields"` passed (`54 passed, 196 deselected`), and targeted Ruff passed.
**Correction recorded:** PME mode now fails closed for invalid Coulomb/PME config values and has nonzero exception-charge override tests for total/component/force correction accounting.

### Slice 6: Artifact Schema For GPCRmd Terms

**Objective:** Extend strict prepared artifacts to represent PME, CHARMM terms, lipids, constraints, exceptions, and protocol metadata without silent drops.
**Execution:** subagent recommended
**Depends on:** Slices 3-5
**Touches:** `src/mlx_atomistic/artifacts.py`, `src/mlx_atomistic/prep/schema.py`, `src/mlx_atomistic/prep/io.py`, `tests/test_production_artifacts.py`
**Context budget:** ~12% of context window
**Produces:** Versioned artifact schema and loader support for the GPCRmd-critical terms implemented so far.
**Acceptance criteria:**
- Artifacts request `pme` only when PME arrays/settings are present and valid.
- Required CHARMM/lipid/exception/constraint arrays round-trip with shape checks.
- Unsupported terms still raise `MLXCompatibilityError` with exact names.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "artifacts or schema or pme or charmm"`
**Auto-continue:** yes

**Status:** completed. Evidence: `.agent/work/gpcrmd-critical-md-readiness/orchestration/slice-6-artifact-schema-for-gpcrmd-terms.md`.
**Verified:** `tests/test_production_artifacts.py tests/test_mlx_prep.py` passed (`55 passed`), `tests -k "artifacts or schema or pme or charmm"` passed (`83 passed, 192 deselected`), and targeted Ruff passed.
**Correction recorded:** PME scalar and mesh settings now fail at artifact validation/save time before lossy casts or later build-time errors.

### Slice 7: GPCRmd Topology/Parameter Import

**Objective:** Import the selected GPCRmd topology, parameters, coordinates, box, masks, and protocol metadata into strict MLX artifacts.
**Execution:** subagent recommended
**Depends on:** Slice 6
**Touches:** `src/mlx_atomistic/prep/gpcrmd.py`, `src/mlx_atomistic/prep/topology_import.py`, `src/mlx_atomistic/prep/`, `tests/test_gpcrmd_registry.py`, `tests/test_mlx_prep.py`
**Context budget:** ~14% of context window
**Produces:** `mlx_atomistic.prep Python API gpcrmd-import` path that writes `prepared_system.json`, `prepared_system.npz`, and `view.pdb` or blocks exactly.
**Acceptance criteria:**
- Water, ions, lipids, receptor, ligand, box, constraints, exclusions, exceptions, and masks are exported.
- Parameter counts match topology counts and unsupported terms are not dropped.
- The selected GPCRmd target either imports or produces a precise remaining blocker report.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "gpcrmd or topology_import or mlx_atomistic.prep"` plus a fixture `uv run mlx_atomistic.prep Python API gpcrmd-import ... --json`
**Auto-continue:** no

**Status:** completed. Evidence: `.agent/work/gpcrmd-critical-md-readiness/orchestration/slice-7-gpcrmd-topology-parameter-import.md`.
**Verified:** `tests/test_gpcrmd_registry.py tests/test_mlx_prep.py` passed (`44 passed`), `tests -k "gpcrmd or topology_import or mlx_atomistic.prep"` passed (`44 passed, 236 deselected`), targeted Ruff passed, fixture `gpcrmd-import --json` exported prepared artifacts, and selected-target missing-cache import emitted exact blockers.
**Correction recorded:** CHARMM/ParmEd imports now fail closed for unexported CHARMM-specific terms instead of silently dropping CMAP/Urey-Bradley/NBFIX-style terms, and blocked imports remove stale generated prepared artifacts from reused output directories.

### Slice 8: Virial And Pressure Diagnostics

**Objective:** Add virial/pressure diagnostics needed for GPCRmd protocol checks and any later barostat.
**Execution:** subagent recommended
**Depends on:** Slices 2-7
**Touches:** `src/mlx_atomistic/forcefields.py`, `src/mlx_atomistic/nonbonded.py`, `src/mlx_atomistic/md.py`, `tests/`
**Context budget:** ~10% of context window
**Produces:** Per-frame virial/pressure diagnostics for supported bonded, nonbonded, PME, and restraint terms.
**Acceptance criteria:**
- Virial/pressure arrays are finite for periodic fixtures.
- Diagnostics are saved and reloaded through trajectory artifacts.
- Unsupported terms report missing virial support before NPT/barostat use.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "virial or pressure or trajectory"`
**Auto-continue:** yes

**Status:** completed. Evidence: `.agent/work/gpcrmd-critical-md-readiness/orchestration/slice-8-virial-and-pressure-diagnostics.md`.
**Verified:** `tests -k "virial or pressure or trajectory"` passed (`16 passed, 272 deselected`), focused NVE/NVT/energy/core/virial-pressure tests passed (`29 passed`), and targeted Ruff passed.
**Correction recorded:** Periodic virial diagnostics now use explicit-support-gated orthorhombic cell-strain finite differences with fractional coordinates held fixed, instead of wrapped absolute-coordinate virials. This slice remains diagnostic-only: off-diagonal periodic virials and barostat moves are deferred.

### Slice 9: NPT Or Membrane-Barostat Runtime Gate

**Objective:** Implement only the barostat path required by the selected GPCRmd protocol, or prove short NVT is the accepted first proof.
**Execution:** subagent recommended
**Depends on:** Slice 8
**Touches:** `src/mlx_atomistic/protocols.py`, `src/mlx_atomistic/md.py`, `src/mlx_atomistic/core.py`, `tests/`
**Context budget:** ~12% of context window
**Produces:** NPT/membrane barostat support if required, otherwise a documented NVT-only GPCRmd proof gate.
**Acceptance criteria:**
- If implemented, volume/box moves preserve finite energy and constraints on small periodic fixtures.
- If deferred, GPCRmd run commands refuse NPT protocols and accept only documented short NVT proof mode.
- Protocol metadata records ensemble and barostat status.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "npt or barostat or protocol"`
**Auto-continue:** no

**Status:** completed. Evidence: `.agent/work/gpcrmd-critical-md-readiness/orchestration/slice-9-npt-or-membrane-barostat-runtime-gate.md`.
**Verified:** `tests -k "npt or barostat or protocol"` passed (`10 passed, 286 deselected`), focused protocol/runner tests passed (`8 passed, 22 deselected`), and targeted Ruff passed.
**Correction recorded:** The selected GPCRmd proof path is NVT, so barostat mechanics were deferred. Runner paths now validate artifact protocol metadata before system construction and save normalized protocol metadata, so NPT/barostat requests fail closed before any MLX integration.

### Slice 10: MLX GPCRmd Runtime Command

**Objective:** Add one runtime command that imports or loads the selected GPCRmd artifact and runs the short MLX protocol.
**Execution:** subagent recommended
**Depends on:** Slices 7-9
**Touches:** `src/mlx_atomistic/prep/`, `src/mlx_atomistic/prep/runner.py`, `src/mlx_atomistic/protocols.py`, `tests/`
**Context budget:** ~10% of context window
**Produces:** `uv run mlx_atomistic.prep Python API run-gpcrmd-mlx --target <id> --out <dir> ...` with artifact, trajectory, diagnostics, and blocker outputs.
**Acceptance criteria:**
- The command never calls external MD engines.
- Runnable fixtures produce `trajectory.npz` with finite diagnostics.
- Blocked GPCRmd targets exit with exact blocker JSON and no fake trajectory.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "gpcrmd and run"` plus one fixture API smoke command.
**Auto-continue:** yes

**Status:** completed. Evidence: `.agent/work/gpcrmd-critical-md-readiness/orchestration/slice-10-mlx-gpcrmd-runtime-command.md`.
**Verified:** `tests -k "gpcrmd and run"` passed (`3 passed, 296 deselected`), `tests/test_gpcrmd_registry.py` passed (`25 passed`), targeted Ruff passed, and a fixture `run-gpcrmd-mlx --json` smoke wrote `trajectory.npz` with finite diagnostics.
**Correction recorded:** Cache-backed runtime runs now check existing `trajectory.npz` before import writes when `force=False`, preventing mixed stale prepared artifacts and old trajectories in reused output directories.

### Slice 11: Notebook GPCRmd MLX Main Path

**Objective:** Make `notebooks/ligand-receptor-motion/` consume the MLX-generated GPCRmd trajectory as the main result.
**Execution:** subagent recommended
**Depends on:** Slice 10
**Touches:** `notebooks/ligand-receptor-motion/`, `tests/test_ligand_receptor_motion.py`
**Context budget:** ~10% of context window
**Produces:** Notebook/helper path that runs or loads `run-gpcrmd-mlx`, visualizes MLX trajectory, and labels GPCRmd reference data as comparison only.
**Acceptance criteria:**
- No downloaded GPCRmd trajectory is displayed as the main MLX result.
- Missing or blocked artifacts stop before MD visualization and show blocker JSON.
- The viewer/analysis uses saved MLX trajectory diagnostics.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "ligand_receptor_motion or notebook or gpcrmd"` and `rg -n "downloaded.*main|fake|benzene|steered" notebooks/ligand-receptor-motion`
**Auto-continue:** yes

**Status:** completed. Evidence: `.agent/work/gpcrmd-critical-md-readiness/orchestration/slice-11-notebook-gpcrmd-mlx-main-path.md`.
**Verified:** `tests/test_ligand_receptor_motion.py` passed (`15 passed`), `tests -k "ligand_receptor_motion or notebook or gpcrmd"` passed (`47 passed, 258 deselected`), targeted Ruff passed, forbidden wording scan found no active downloaded/fake/benzene/steered main-path references, and notebook code cells parse.
**Correction recorded:** Cached notebook bundles now fail closed when the requested target does not match the saved run report, when `trajectory.npz` is corrupt, when the prepared artifact is missing/corrupt, or when prepared identity/dimensions disagree with saved trajectory metadata.

### Slice 12: GPCRmd Performance And Scale Gate

**Objective:** Add repeatable performance gates for the selected GPCRmd run path.
**Execution:** direct
**Depends on:** Slice 10
**Touches:** `src/mlx_atomistic/benchmarks/`, `src/mlx_atomistic/prep/`, `tests/`
**Context budget:** ~8% of context window
**Produces:** JSON/CSV benchmark for import time, atom count, PME mesh size, pair count, wall time, steps/s, ps/s, memory, and artifact size.
**Acceptance criteria:**
- Benchmark defaults are short enough for development.
- Reports compare cutoff/Ewald/PME where applicable without claiming biological sampling.
- Blocked systems emit blockers instead of timing placeholders.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "benchmark or performance or gpcrmd"`
**Auto-continue:** yes

**Status:** completed. Evidence: `.agent/work/gpcrmd-critical-md-readiness/orchestration/slice-12-gpcrmd-performance-and-scale-gate.md`.
**Verified:** `tests/test_gpcrmd_registry.py -k "benchmark"` passed (`3 passed, 25 deselected`), `tests -k "benchmark or performance or gpcrmd"` passed (`60 passed, 248 deselected`), `tests/test_gpcrmd_registry.py` passed (`28 passed`), targeted Ruff passed, and a blocked API smoke emitted blocker JSON for an unknown target.
**Correction recorded:** Benchmark rows now use actual MLX run metadata for runnable systems and explicit blocker rows for missing/unsupported systems. Electrostatics comparison requests do not mutate artifacts; cutoff/Ewald/PME variants block unless they match the prepared artifact.

### Slice 13: Readiness Verification And Handoff

**Objective:** Verify the completed readiness path and document any remaining exact blockers.
**Execution:** direct
**Depends on:** Slices 1-12
**Touches:** `.agent/work/gpcrmd-critical-md-readiness/`, `docs/` if needed
**Context budget:** ~6% of context window
**Produces:** Verification/handoff note stating whether the selected GPCRmd target runs through MLX or which blockers remain.
**Acceptance criteria:**
- Source tests and source lint pass.
- The final status does not claim full GPCRmd readiness unless the selected target imports and runs through MLX.
- Remaining blockers, if any, map to named engine/import capabilities.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest` and `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts`
**Auto-continue:** no

**Status:** completed. Evidence: `.agent/work/gpcrmd-critical-md-readiness/orchestration/slice-13-readiness-verification-and-handoff.md` and `.agent/work/gpcrmd-critical-md-readiness/READINESS-HANDOFF.md`.
**Verified:** full test suite passed (`308 passed`), `ruff check src tests scripts` passed, selected-target empty-cache inspection reported exact missing inputs, and selected-target empty-cache `run-gpcrmd-mlx` blocked with JSON and no trajectory.
**Correction recorded:** The final status does not claim the selected real GPCRmd target has run. It states that the MLX path is implemented and fixture-verified, while the real target still requires the GPCRmd package files and may next expose HMR/virtual-site parsing blockers.

## Execution Routing And Topology

- Slice 1: subagent recommended; checkpoint before engine work because it fixes the real target requirements.
- Parallel-safe group A after Slice 1: Slices 2, 3, and 4 may run in parallel if workers keep disjoint ownership: PME standalone worker owns `pme.py`/PME tests, CHARMM worker owns `charmm_terms.py`/CHARMM tests, neighbor worker owns cell-list/neighbor files.
- Slice 5 is a serial integration checkpoint that wires PME into `NonbondedPotential` before artifacts or runtime can request `pme`.
- Slices 6-11 are serial integration work because they share schemas, artifact loading, API/protocol surfaces, and notebook behavior.
- Slice 12 may run in parallel with Slice 11 after Slice 10 if benchmark files and notebook files remain disjoint.
- Slice 13 is the final direct verification checkpoint.

Use subagents for Slices 1-11 when the user authorizes multi-agent execution. Each worker must be told it is not alone in the codebase, must not revert unrelated edits, and must keep to its assigned write set.

## Verification Commands

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "gpcrmd and inventory"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "pme or ewald"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "pme or nonbonded or forcefields"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "charmm or cmap or nbfix"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "neighbor or nonbonded_acceleration"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "artifacts or schema or pme or charmm"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "gpcrmd or topology_import or mlx_atomistic.prep"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "virial or pressure or trajectory"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "npt or barostat or protocol"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "gpcrmd and run"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "ligand_receptor_motion or notebook or gpcrmd"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "benchmark or performance or gpcrmd"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts`

## Context Budget For This Change

Estimated total: ~140% of one large context window if executed end-to-end. This should be executed as checkpointed slices, with a hard checkpoint after Slice 1, Slice 5, and Slice 7, and another after Slice 9 if NPT or membrane barostat support is required.

## Recommended Next Skill

Run `auto-eng-review` before execution. The plan is intentionally large and needs engineering review on dependency order, subagent write-set boundaries, and whether the selected GPCRmd protocol requires barostat support for the first MLX proof.

## Plan Correction

Engineering review found that the previous plan built PME as standalone math but did not wire it into production nonbonded force evaluation before artifact/runtime slices depended on `pme`. Slice 5 now owns PME `NonbondedPotential` integration and serializes access to `src/mlx_atomistic/nonbonded.py` and `src/mlx_atomistic/forcefields.py`.

## Review: Engineering

- Verdict: needs_correction
- Strength: The plan has the right gated shape: target inventory first, disjoint PME/CHARMM/neighbor worker lanes, serial artifact/import/runtime integration, and explicit fail-closed notebook behavior.
- Concern: Slice 2 builds PME as a standalone backend, but no later slice explicitly wires PME into `NonbondedPotential` or the production force-evaluation path before Slice 5 artifacts and Slice 9 runtime commands depend on `pme`.
- Action: Return to `auto-plan` and add an explicit PME/nonbonded integration slice, with write ownership for `src/mlx_atomistic/nonbonded.py`, `src/mlx_atomistic/forcefields.py`, and matching tests, before artifact-schema/runtime slices.
- Verified: Read STATUS.md, current state, PLAN.md, DESIGN.md, review template, and checked slice dependencies, write-set boundaries, PME data flow, artifact/runtime dependencies, and verification commands.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The corrected plan now has an executable dependency chain from target inventory through standalone PME, PME `NonbondedPotential` integration, artifact schema, import, runtime command, notebook, and final verification.
- Concern: The first GPCRmd inventory slice may expose additional selected-target terms or protocol requirements that force the plan to pause before the parallel PME, CHARMM, and neighbor workers start.
- Action: Start with Slice 1 only, record the concrete target inventory, and proceed to the parallel Slice 2-4 workers only if the inventory confirms their write sets and acceptance criteria remain accurate.
- Verified: Read STATUS.md, current state, corrected PLAN.md, DESIGN.md, review template, and checked PME integration placement, slice dependencies, subagent write-set boundaries, checkpoint placement, fail-closed behavior, and verification commands.
