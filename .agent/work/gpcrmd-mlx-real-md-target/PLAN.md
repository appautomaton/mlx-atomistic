# PLAN: GPCRmd-Backed MLX Real MD Target

## Goal

Make GPCRmd the first real-system target for `notebooks/ligand-receptor-motion/` by using GPCRmd data as reference input/validation context while requiring any newly generated project-result trajectory to come from `mlx_atomistic`.

## Architecture Approach

Build a fail-closed GPCRmd target workflow before attempting large physics work:

- Add a GPCRmd target registry/cache-inspection layer in `mlx_atomistic.prep`.
- Add a compatibility report that says exactly whether the selected GPCRmd system is MLX-runnable today.
- Keep `mlx_atomistic` as the only trajectory generator and make unsupported physics explicit.
- Update the active notebook so it either runs/loads an MLX trajectory or stops at the report.
- Add short benchmark probes for any runnable imported system; do not start long notebook runs during development.

The plan intentionally does not implement PME, NPT, or full membrane production in this change. If those are the blockers, this change produces the concrete evidence needed for the next engine slice.

## Ordered Task Sequence

### Slice 1: GPCRmd Target Registry And Selection Gate

**Objective:** Add a small source-backed target registry and selection gate for candidate GPCRmd systems.
**Execution:** direct
**Depends on:** none
**Touches:** `src/mlx_atomistic/prep/`, `tests/`, `.agent/work/gpcrmd-mlx-real-md-target/`
**Context budget:** ~8% of context window
**Produces:** A registry/manifest surface that names candidate GPCRmd targets and records required files, source URLs, and selection reasons.
**Acceptance criteria:**
- At least one ligand-bound GPCRmd target candidate is represented with coordinates, topology/parameter expectations, water/ion/box expectations, and reference trajectory metadata fields.
- The registry supports an offline fixture so tests do not require downloading large GPCRmd files.
- Selection failure says exactly which required metadata is missing.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k gpcrmd`
**Auto-continue:** yes

**Execution evidence:** Done in direct route. Added `src/mlx_atomistic/prep/gpcrmd.py`, exported registry APIs from `src/mlx_atomistic/prep/__init__.py`, added `tests/test_gpcrmd_registry.py`, and recorded source evidence in `SLICE-1-EVIDENCE.md`. Verified with `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k gpcrmd` and targeted Ruff on touched source/test files.

### Slice 2: GPCRmd Cache Inspection API

**Objective:** Add API support to inspect a local GPCRmd target package or manifest without running simulation.
**Execution:** subagent recommended
**Depends on:** Slice 1
**Touches:** `src/mlx_atomistic/prep/`, `src/mlx_atomistic/prep/`, `tests/`
**Context budget:** ~10% of context window
**Produces:** A command such as `uv run mlx_atomistic.prep Python API gpcrmd-inspect --target <id> --cache <path>` that prints JSON/table status for files, formats, atom counts if available, and missing inputs.
**Acceptance criteria:**
- The command never calls external MD engines.
- The command works on a tiny fixture package.
- Missing files produce actionable errors, not tracebacks.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "gpcrmd and inspect"` plus a fixture API smoke command.
**Auto-continue:** yes

**Execution evidence:** Done in direct route because host subagents require an explicit user request. Added `gpcrmd-inspect` API, local cache/manifest inspection, JSON/table output, `--require-complete`, and fixture tests. Verified with `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "gpcrmd and inspect"`, targeted Ruff on touched files, and a fixture `uv run mlx_atomistic.prep Python API gpcrmd-inspect ...` smoke command.

### Slice 3: Compatibility Report For MLX-Runnable GPCRmd Systems

**Objective:** Convert inspected target data into a fail-closed MLX compatibility report.
**Execution:** subagent recommended
**Depends on:** Slice 2
**Touches:** `src/mlx_atomistic/prep/`, `src/mlx_atomistic/artifacts.py`, `src/mlx_atomistic/validation.py`, `tests/`
**Context budget:** ~12% of context window
**Produces:** A structured report with `supported_now`, `missing_input`, `unsupported_physics`, `runtime_risk`, and `next_engine_slice`.
**Acceptance criteria:**
- Unsupported GPCR features such as PME/Ewald, CMAP, virtual sites, unsupported water/lipid models, missing box vectors, or NPT-only protocols are listed explicitly.
- A raw reference trajectory alone cannot pass as an MLX-runnable prepared system.
- The report is serializable and usable by the notebook.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "compatibility or gpcrmd"`
**Auto-continue:** yes

**Execution evidence:** Done in direct route because host subagents require an explicit user request. Added fail-closed GPCRmd MLX compatibility reports with `supported_now`, `missing_input`, `unsupported_physics`, `runtime_risk`, and `next_engine_slice`; extended `gpcrmd-inspect --compatibility`; verified complete-cache and trajectory-only cases. Verified with `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "compatibility or gpcrmd"`, targeted Ruff, and a fixture API compatibility smoke command.

### Slice 4: Import Attempt To Prepared MLX Artifact

**Objective:** Attempt conversion from supported GPCRmd topology/coordinate formats into the existing prepared artifact schema, and fail closed when terms are unsupported.
**Execution:** subagent recommended
**Depends on:** Slice 3
**Touches:** `src/mlx_atomistic/prep/topology_import.py`, `src/mlx_atomistic/prep/`, `src/mlx_atomistic/artifacts.py`, `tests/`
**Context budget:** ~14% of context window
**Produces:** `prepared_system.json` / `prepared_system.npz` export for compatible fixture data, or a precise blocker report for the selected GPCRmd target.
**Acceptance criteria:**
- Water, ions, hydrogens, box vectors, masks, constraints, exclusions, and exceptions are either exported or reported missing.
- Unsupported terms do not get silently dropped.
- Existing AMBER/CHARMM import behavior stays green.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "topology_import or artifacts or gpcrmd"`
**Auto-continue:** no

**Execution evidence:** Done in direct route because host subagents require an explicit user request. Added `gpcrmd-import` as a fail-closed import-attempt command that writes `gpcrmd_import_report.json` and does not create `prepared_system.*` when GPCRmd blockers exist. Verified the selected GPCRmd target reports PME/Ewald, membrane/lipid terms, POPC parameters, CHARMM CMAP, scale, and virtual-site/HMR uncertainty blockers. Verified with `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "topology_import or artifacts or gpcrmd"`, targeted Ruff, and a fixture `uv run mlx_atomistic.prep Python API gpcrmd-import ...` smoke command.

### Slice 5: Notebook MLX-Only GPCRmd Target Status

**Objective:** Update `notebooks/ligand-receptor-motion/` to present the GPCRmd target status and only visualize MLX-generated trajectories.
**Execution:** subagent recommended
**Depends on:** Slice 4
**Touches:** `notebooks/ligand-receptor-motion/`, `src/mlx_atomistic/prep/`, `tests/`
**Context budget:** ~12% of context window
**Produces:** Notebook/helper flow that loads a compatibility report, runs a short MLX probe only when compatible, and otherwise stops before MD visualization.
**Acceptance criteria:**
- The notebook labels GPCRmd as reference context, not the active trajectory source.
- No downloaded GPCRmd frames, benzene pull, or toy trajectory appear as the main result.
- The notebook shows exact blockers when MLX cannot run the selected target.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "notebook or gpcrmd"` and `rg -n "benzene|public trajectory|fake|steered" notebooks/ligand-receptor-motion`
**Auto-continue:** no

### Slice 6: Short-Run Performance Probe For Runnable Targets

**Objective:** Add short, repeatable runtime probes for any MLX-runnable imported target.
**Execution:** direct
**Depends on:** Slice 4
**Touches:** `src/mlx_atomistic/prep/`, `src/mlx_atomistic/benchmarks/`, `tests/`
**Context budget:** ~8% of context window
**Produces:** A probe command/report with wall time, steps/s, ps/s, atom count, pair count, constraint error, artifact size, and backend metadata.
**Acceptance criteria:**
- Probe defaults are short enough for development runs.
- If the GPCRmd target is not runnable, the probe exits with the compatibility blocker report.
- Output is JSON/CSV friendly for comparing later engine changes.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "performance or gpcrmd"` plus one fixture probe command.
**Auto-continue:** yes

### Slice 7: Source/Test Hygiene And Handoff

**Objective:** Run the completion gates that are meaningful for this change and document any deferred engine blockers.
**Execution:** direct
**Depends on:** Slices 1-6
**Touches:** `.agent/work/gpcrmd-mlx-real-md-target/`, `docs/` if needed
**Context budget:** ~6% of context window
**Produces:** A concise blocker/next-engine-slice note if the selected GPCRmd system cannot yet run in MLX.
**Acceptance criteria:**
- Source tests and source lint pass.
- Full-repo Ruff is not claimed green unless notebook lint policy is addressed separately.
- Deferred blockers are named as engine capabilities, not vague “production validation” language.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest` and `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts`
**Auto-continue:** no

## Execution Routing

- Slice 1: direct
- Slice 2: subagent recommended
- Slice 3: subagent recommended
- Slice 4: subagent recommended
- Slice 5: subagent recommended
- Slice 6: direct
- Slice 7: direct

Use subagents for Slices 2-5 because they cross API, prep schema, artifact validation, and notebook boundaries. Keep Slices 1, 6, and 7 direct because they are bounded registry/probe/verification work.

## Verification Commands

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k gpcrmd`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "gpcrmd and inspect"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "compatibility or gpcrmd"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "topology_import or artifacts or gpcrmd"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "notebook or gpcrmd"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "performance or gpcrmd"`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts`

## Context Budget For This Change

Estimated total: ~70% of one large context window if executed end-to-end. Recommended execution is slice-by-slice, with a checkpoint after Slice 4 because that is where the chosen GPCRmd target will either become runnable or produce the concrete engine-blocker list.

## Recommended Next Skill

Run `auto-eng-review` before execution. The risk is not product scope now; it is engineering order: we need to ensure the first implementation slice does not turn into hidden PME/NPT or full membrane-force-field work.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan cleanly separates GPCRmd reference data, mlx_atomistic.prep import/reporting, mlx_atomistic simulation, and notebook visualization, with fail-closed behavior before any misleading trajectory output.
- Concern: The selected GPCRmd target is not fixed yet and may immediately require missing engine capabilities such as PME/Ewald, lipid terms, scalable periodic nonbonded handling, or NPT/barostat support.
- Action: Execute Slice 1 first, then checkpoint after Slice 4 with the concrete runnable-or-blocked compatibility report before touching notebook visualization or longer performance probes.
- Verified: Reviewed STATUS.md, PLAN.md, DESIGN.md, data flow, slice dependencies, verification commands, vendor boundary, external-engine boundary, and notebook-output boundary.
