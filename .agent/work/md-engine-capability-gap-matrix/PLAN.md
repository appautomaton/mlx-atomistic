# PLAN: MD Engine Capability Gap Matrix

## Goal

Audit `mlx_atomistic` as a general MD simulation engine, classify each production
capability as implemented, partial, missing, or unverified against mature MD engine
expectations, and produce an ordered capability backlog.

## Architecture Approach

This is an evidence-producing audit, not an implementation change. The work writes
reloadable planning artifacts under `.agent/work/md-engine-capability-gap-matrix/`
and keeps source files read-only. Each capability row must point to live repo
evidence, a runnable probe, or a reference-engine source/doc comparison.

## Requirement Traceability

- `T1`: Core MD physics.
- `T2`: Force-field and artifact coverage.
- `T3`: Runtime production usability.
- `T4`: Validation and parity.
- `T5`: Performance and backend.
- `T6`: Prep and workflow.
- `R1`: Distinguish `implemented`, `partial`, `missing`, and `unverified`.
- `R2`: Make at least one OpenMM, GROMACS, or LAMMPS comparison per major track
  where useful.
- `R3`: Produce an ordered first-wave backlog plus deferred list.
- `R4`: Change no source implementation files.

## Ordered Task Sequence

### Slice 1: Evidence Inventory

**Objective:** Build the source/doc/test/reference inventory used by every matrix row.
**Execution:** direct
**Depends on:** none
**Touches:** `.agent/work/md-engine-capability-gap-matrix/EVIDENCE-INDEX.md`; read-only scan of `src/`, `tests/`, `docs/`, `scripts/`, `vendors/`
**Context budget:** ~8% of context window
**Produces:** `EVIDENCE-INDEX.md` listing inspected repo surfaces, reference-engine surfaces, runnable probes available, and evidence rules.
**Acceptance criteria:**
- The index names exact files or directories inspected for each track.
- The index records which reference engines are executable locally versus source-only references.
- The index states that generated outputs remain under ignored `results/` or temporary paths.
**Verification:** `test -f .agent/work/md-engine-capability-gap-matrix/EVIDENCE-INDEX.md && rg -n "OpenMM|GROMACS|LAMMPS|src/mlx_atomistic|tests/" .agent/work/md-engine-capability-gap-matrix/EVIDENCE-INDEX.md`
**Checkpoint after:** none
**Checkpoint reason:** none
**Detail:** none

### Slice 2: Core MD Physics Matrix

**Objective:** Classify core MD physics capabilities and blockers.
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `.agent/work/md-engine-capability-gap-matrix/GAP-MATRIX.md`; read-only scan of `src/mlx_atomistic/{md,protocols,constraints,pme,nonbonded,core}.py`, tests, docs
**Context budget:** ~12% of context window
**Produces:** `GAP-MATRIX.md` section for `T1` covering NVE/NVT/NPT, integrators, thermostats, constraints, PME, virial/pressure, cells, virtual sites, and water models.
**Acceptance criteria:**
- Each `T1` row has status, evidence, production impact, next action, and recommended order.
- PME and NPT are classified from live code gates, not prior assumptions.
- Orthorhombic/triclinic and virtual-site policy are explicitly classified.
**Verification:** `rg -n "T1|PME|NPT|barostat|constraint|triclinic|virtual" .agent/work/md-engine-capability-gap-matrix/GAP-MATRIX.md`
**Checkpoint after:** none
**Checkpoint reason:** none
**Detail:** none

### Slice 3: Force-Field, Artifact, And Prep Matrix

**Objective:** Classify force-field, artifact, and prep/workflow capabilities.
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `.agent/work/md-engine-capability-gap-matrix/GAP-MATRIX.md`; read-only scan of `src/mlx_atomistic/prep/`, `src/mlx_atomistic/artifacts.py`, `src/mlx_atomistic/charmm_terms.py`, tests, docs
**Context budget:** ~12% of context window
**Produces:** `GAP-MATRIX.md` sections for `T2` and `T6` covering AMBER, CHARMM, CMAP, NBFIX, exceptions, HMR, ligands, production artifact gates, raw topology inputs, and prepared artifacts.
**Acceptance criteria:**
- Existing implemented or gated CHARMM/CMAP/PME artifact support is not mislabeled as absent.
- Production artifact gates and known fixture limitations are recorded.
- The previous production-artifact evidence baseline appears as a possible evidence slice, not the active goal.
**Verification:** `rg -n "T2|T6|AMBER|CHARMM|CMAP|NBFIX|HMR|production artifact|PreparedSystem" .agent/work/md-engine-capability-gap-matrix/GAP-MATRIX.md`
**Checkpoint after:** none
**Checkpoint reason:** none
**Detail:** none

### Slice 4: Runtime Usability And Validation Matrix

**Objective:** Classify runner usability, long-run diagnostics, and validation/parity surfaces.
**Execution:** direct
**Depends on:** Slices 1-3
**Touches:** `.agent/work/md-engine-capability-gap-matrix/GAP-MATRIX.md`; read-only scan of `src/mlx_atomistic/io.py`, `trajectory_adapters.py`, `validation.py`, `benchmarks/`, runner code, scripts, docs
**Context budget:** ~12% of context window
**Produces:** `GAP-MATRIX.md` sections for `T3` and `T4` covering reporters, checkpoint/restart, DCD/XTC, diagnostics, OpenMM parity, GROMACS/LAMMPS reference comparisons, and validation harnesses.
**Acceptance criteria:**
- Low-level restart support is distinguished from runner-level checkpoint/restart.
- Native `.npz` output and analysis adapters are distinguished from first-class DCD/XTC runner output.
- Existing validation/benchmark surfaces are classified before adding new ones.
**Verification:** `rg -n "T3|T4|reporter|checkpoint|restart|DCD|XTC|OpenMM|GROMACS|LAMMPS|parity" .agent/work/md-engine-capability-gap-matrix/GAP-MATRIX.md`
**Checkpoint after:** none
**Checkpoint reason:** none
**Detail:** none

### Slice 5: Performance And Backend Matrix

**Objective:** Classify performance/backend gaps against local MLX behavior and reference-engine patterns.
**Execution:** direct
**Depends on:** Slices 1-4
**Touches:** `.agent/work/md-engine-capability-gap-matrix/GAP-MATRIX.md`; read-only scan of `src/mlx_atomistic/{neighbors,cell_list,nonbonded,pme}.py`, benchmark docs/scripts, `vendors/gromacs`, `vendors/lammps`, `vendors/openmm`
**Context budget:** ~12% of context window
**Produces:** `GAP-MATRIX.md` section for `T5` covering neighbor-list scaling, PME FFT, MLX/Metal kernel bottlenecks, OpenMM OpenCL baseline, and reference implementation patterns worth studying.
**Acceptance criteria:**
- Performance rows separate measured evidence from hypotheses.
- GROMACS and LAMMPS are treated as reference source only unless an executable probe is already available.
- The first performance follow-up depends on correctness/parity gates unless evidence shows a pure runtime blocker.
**Verification:** `rg -n "T5|neighbor|PME FFT|OpenCL|Metal|GROMACS|LAMMPS|performance|backend" .agent/work/md-engine-capability-gap-matrix/GAP-MATRIX.md`
**Checkpoint after:** none
**Checkpoint reason:** none
**Detail:** none

### Slice 6: Ordered Backlog And Handoff

**Objective:** Convert matrix findings into a first-wave implementation backlog and deferred list.
**Execution:** direct
**Depends on:** Slices 2-5
**Touches:** `.agent/work/md-engine-capability-gap-matrix/GAP-MATRIX.md`, `.agent/work/md-engine-capability-gap-matrix/BACKLOG.md`
**Context budget:** ~10% of context window
**Produces:** `BACKLOG.md` with ordered first-wave candidates, deferred work, and recommended next SPEC.
**Acceptance criteria:**
- The backlog maps every first-wave item back to matrix row IDs.
- The first wave is ordered by production impact and dependency order, not preference.
- Deferred items are explicitly named and not lost.
- No source implementation files are changed.
**Verification:** `test -f .agent/work/md-engine-capability-gap-matrix/BACKLOG.md && rg -n "First Wave|Deferred|Next SPEC|T[1-6]" .agent/work/md-engine-capability-gap-matrix/BACKLOG.md && git diff --name-only -- src tests scripts pyproject.toml`
**Checkpoint after:** decision
**Checkpoint reason:** The backlog selects the next implementation SPEC; pause before starting implementation.
**Detail:** none

## Execution Routing And Topology

- Route all slices direct. The write set is limited to audit artifacts under
  `.agent/work/md-engine-capability-gap-matrix/`.
- Parallel-safe groups: none. Slices share `GAP-MATRIX.md`, so execute them
  serially to avoid merge churn.
- Slices 4 and 5 depend on the earlier matrix sections.
- Continuation is the default after Slices 1-5 once verification passes.
- Slice 6 is the decision checkpoint and must not continue into implementation.

## Verification Commands

- Slice 1: `test -f .agent/work/md-engine-capability-gap-matrix/EVIDENCE-INDEX.md && rg -n "OpenMM|GROMACS|LAMMPS|src/mlx_atomistic|tests/" .agent/work/md-engine-capability-gap-matrix/EVIDENCE-INDEX.md`
- Slice 2: `rg -n "T1|PME|NPT|barostat|constraint|triclinic|virtual" .agent/work/md-engine-capability-gap-matrix/GAP-MATRIX.md`
- Slice 3: `rg -n "T2|T6|AMBER|CHARMM|CMAP|NBFIX|HMR|production artifact|PreparedSystem" .agent/work/md-engine-capability-gap-matrix/GAP-MATRIX.md`
- Slice 4: `rg -n "T3|T4|reporter|checkpoint|restart|DCD|XTC|OpenMM|GROMACS|LAMMPS|parity" .agent/work/md-engine-capability-gap-matrix/GAP-MATRIX.md`
- Slice 5: `rg -n "T5|neighbor|PME FFT|OpenCL|Metal|GROMACS|LAMMPS|performance|backend" .agent/work/md-engine-capability-gap-matrix/GAP-MATRIX.md`
- Slice 6: `test -f .agent/work/md-engine-capability-gap-matrix/BACKLOG.md && rg -n "First Wave|Deferred|Next SPEC|T[1-6]" .agent/work/md-engine-capability-gap-matrix/BACKLOG.md && git diff --name-only -- src tests scripts pyproject.toml`

## Context Budget For This Change

Estimated total: ~66% of a context window if done in one pass.

- Slice 1: ~8%
- Slice 2: ~12%
- Slice 3: ~12%
- Slice 4: ~12%
- Slice 5: ~12%
- Slice 6: ~10%

Stop after any slice if evidence contradicts the spec assumptions, especially around
PME, CHARMM, GPCRmd artifact status, or existing validation coverage.

## Execution Evidence

- Slice 1 completed: wrote `EVIDENCE-INDEX.md` with inspected source, docs, tests,
  reference-engine surfaces, executable/reference status, and output policy.
- Slice 2 completed: wrote `GAP-MATRIX.md` `T1` rows for NVE/NVT/NPT, PME,
  constraints, virial/pressure, cells, virtual sites, water, and dispersion.
- Slice 3 completed: added `T2` and `T6` rows for AMBER, CHARMM, CMAP, NBFIX,
  exceptions, HMR, ligand parameters, production artifact gates, and prep workflow.
- Slice 4 completed: added `T3` and `T4` rows for NPZ output, reporters,
  checkpoint/restart, DCD/XTC, diagnostics, OpenMM parity, and validation surfaces.
- Slice 5 completed: added `T5` rows for neighbor-list scaling, PME FFT/backend,
  OpenMM OpenCL targets, GROMACS/LAMMPS source references, and custom-kernel timing.
- Slice 6 completed: wrote `BACKLOG.md` with the ordered first wave, deferred list,
  and recommended next SPEC: `production-artifact-openmm-parity-fixture`.
- Verification passed for all slice `rg`/file checks. `git diff --name-only -- src
  tests scripts pyproject.toml` produced no paths, so no source implementation files
  changed.
- Checkpoint reached: decision. Do not start implementation until the next SPEC is
  selected.
