# PLAN: DHFR Benchmark Artifact And Runtime Path

## Goal

Implement the approved DHFR benchmark spec: make implicit DHFR and explicit PME DHFR represented as MLX/OpenMM comparison rows with runnable, diagnostic, or concrete blocked status.

## Requirement Traceability

| Requirement | Covered by slices |
| --- | --- |
| AC-01 DHFR artifact readiness | Slice 1, Slice 3, Slice 5 |
| AC-02 implicit DHFR MLX JSON | Slice 4 |
| AC-03 explicit PME DHFR MLX JSON | Slice 5 |
| AC-04 OpenMM DHFR reference JSON | Slice 2 |
| AC-05 comparison gate | Slice 6 |
| AC-06 docs/report refresh | Slice 7 |
| AC-07 runtime boundary | Slice 2, Slice 4, Slice 5, Slice 8 |
| AC-08 focused gate | Slice 8 |

## Architecture Approach

Use `DESIGN.md`: add a dedicated MLX DHFR benchmark module, a dedicated OpenMM DHFR reference script, and extend the existing same-workload comparison registry. Use current AMBER import, artifact readiness, PME readiness, and runtime paths before introducing new data models.

## Ordered Slice Sequence

### Slice 1: DHFR Input Resolver And Readiness Surface

**Objective:** Add a DHFR input/readiness layer that reports local implicit and explicit PME input availability before runtime.

**Acceptance criteria:**
- Resolves local OpenMM DHFR PDB inputs and Amber20/JAC `prmtop`/`inpcrd` inputs when present.
- Emits normalized JSON for `dhfr-implicit` and `dhfr-explicit-pme` readiness with concrete blockers when inputs are missing.
- Identifies atom count, solvent model target, force-field/source family, cell metadata availability, and raw input paths.
- Does not download inputs.

**Verification:** `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --readiness --json && uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --readiness --json && uv run pytest tests/test_benchmarks.py -q`

**Execution:** subagent recommended

**Depends on:** none

**Touches:** `src/mlx_atomistic/benchmarks/dhfr.py`, `src/mlx_atomistic/benchmarks/__init__.py`, `tests/test_benchmarks.py`

**Produces:** DHFR MLX benchmark/readiness CLI skeleton and input blocker contract.

**Status:** complete
**Evidence:** changed `src/mlx_atomistic/benchmarks/dhfr.py` and `tests/test_benchmarks.py`; `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --readiness --json` returned `status=ok`, `atom_count=2489`, `solvent_model=implicit`, `electrostatics_model=gbsa_obc`, and no downloads; `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --readiness --json` returned `status=ok`, `atom_count=23558`, `solvent_model=explicit`, `electrostatics_model=pme`, Amber20/JAC paths present, and no downloads; sandbox pytest failed with `No Metal device available`, then Metal-visible `uv run pytest tests/test_benchmarks.py -q` passed 43 tests; `uv run ruff check src/mlx_atomistic/benchmarks/dhfr.py tests/test_benchmarks.py` passed.
**Risks / next:** readiness uses local/gitignored `results/inputs` when present and reports concrete blockers when absent.

### Slice 2: OpenMM DHFR Reference Runner

**Objective:** Add normalized OpenMM reference output for implicit DHFR and explicit PME DHFR.

**Acceptance criteria:**
- OpenMM code stays in `scripts/`, outside product runtime imports.
- `dhfr-implicit` and `dhfr-explicit-pme` emit normalized JSON with `status`, `fixture`, `atom_count`, `timing_metric`, `timing_value`, `ns_per_day` where runnable, command, and raw input metadata.
- Missing OpenMM inputs/platforms return concrete `blocked` payloads.
- Invalid numeric arguments fail nonzero before OpenMM platform/import handling.

**Verification:** `uv run python scripts/benchmark_openmm_dhfr.py --case dhfr-implicit --platform Reference --steps 1 --json && uv run python scripts/benchmark_openmm_dhfr.py --case dhfr-explicit-pme --platform Reference --steps 1 --json && uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q`

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `scripts/benchmark_openmm_dhfr.py`, `tests/test_benchmarks.py`, `tests/test_runtime_boundaries.py`

**Produces:** OpenMM DHFR normalized reference surface.

**Status:** complete
**Evidence:** subagent implementer changed `scripts/benchmark_openmm_dhfr.py`, `tests/test_benchmarks.py`, and `tests/test_runtime_boundaries.py`; `uv run python scripts/benchmark_openmm_dhfr.py --case dhfr-implicit --platform Reference --steps 1 --json` returned normalized `status=ok`, `fixture=dhfr_implicit`, `atom_count=2489`, `timing_metric=ns_per_day`, finite output, and raw input metadata; `uv run python scripts/benchmark_openmm_dhfr.py --case dhfr-explicit-pme --platform Reference --steps 1 --json` returned normalized `status=ok`, `fixture=dhfr_explicit_pme`, `atom_count=23558`, `timing_metric=ns_per_day`, finite output, PME setup metadata, and raw input metadata; `uv run ruff check scripts/benchmark_openmm_dhfr.py tests/test_benchmarks.py tests/test_runtime_boundaries.py` passed; Metal-visible `uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q` passed 55 tests; spec review `APPROVED`; quality review `APPROVED` after E501 line-wrap fix.
**Risks / next:** Reference timings used `Reference` platform and 1 step for shape verification only; performance reporting still needs controlled raw-output refresh later.

### Slice 3: DHFR Artifact Import And Compatibility Gate

**Objective:** Convert available DHFR AMBER inputs into a prepared MLX artifact or a precise compatibility blocker.

**Acceptance criteria:**
- Uses existing AMBER import/prep/artifact APIs where possible.
- Records required arrays, unsupported terms, electrostatics model, GBSA/OBC metadata for implicit, and PME metadata for explicit.
- If GBSA radii/scales or PME prerequisites are missing, reports the exact missing capability.
- Saves generated artifacts only under gitignored `results/`.

**Verification:** `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --prepare --json && uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --prepare --json && uv run pytest tests/test_benchmarks.py -q`

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `src/mlx_atomistic/benchmarks/dhfr.py`, `src/mlx_atomistic/prep/topology_import.py`, `src/mlx_atomistic/artifacts.py`, `tests/test_benchmarks.py`

**Produces:** DHFR prepared-artifact readiness path for both target rows.

**Status:** complete
**Evidence:** subagent changed `src/mlx_atomistic/benchmarks/dhfr.py`, `src/mlx_atomistic/prep/topology_import.py`, and `tests/test_benchmarks.py`; `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --prepare --json` returned normalized `status=blocked` with exact missing GBSA/OBC arrays `gbsa_radius` and `gbsa_scale`; `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --prepare --json` returned normalized `status=blocked`, NetCDF coordinate format, PME config metadata, and `unsupported_terms=["amber_10_12_nonbonded"]`; `uv run pytest tests/test_benchmarks.py -q -k dhfr` passed 8 tests; `uv run ruff check src/mlx_atomistic/benchmarks/dhfr.py src/mlx_atomistic/prep/topology_import.py tests/test_benchmarks.py` passed; spec review `APPROVED`; quality review `APPROVED`.
**Risks / next:** runtime slices should preserve these concrete blockers until GBSA/OBC parameter import and AMBER 10-12 handling are addressed.

### Slice 4: Implicit DHFR MLX Runtime Benchmark

**Objective:** Run the implicit DHFR MLX benchmark or emit a normalized blocker that names the remaining runtime gap.

**Acceptance criteria:**
- Emits normalized JSON for `dhfr-implicit` with `comparison_pair_id`, `atom_count`, solvent/electrostatics metadata, timing metric, status, and blocker if any.
- If runnable, performs a bounded short run/evaluation and reports finite output.
- If blocked, distinguishes GBSA parameter gap, neighbor/lazy-topology gap, runtime artifact gap, or Metal environment gap.
- Does not compute or claim OpenMM comparison ratio inside the MLX benchmark itself.

**Verification:** `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --steps 1 --json && uv run pytest tests/test_benchmarks.py -q`

**Execution:** subagent recommended

**Depends on:** Slice 3

**Touches:** `src/mlx_atomistic/benchmarks/dhfr.py`, possible runtime/artifact support files, `tests/test_benchmarks.py`

**Produces:** MLX implicit DHFR benchmark row.

**Status:** complete
**Evidence:** changed `src/mlx_atomistic/benchmarks/dhfr.py` and `tests/test_benchmarks.py`; `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --steps 1 --json` returned normalized `status=blocked`, `comparison_pair_id=dhfr-implicit`, `atom_count=2489`, `step_count=1`, `runtime_attempted=false`, and `runtime_blocker_category=gbsa_parameter_gap`; blocker names missing `gbsa_radius` and `gbsa_scale`; `uv run pytest tests/test_benchmarks.py -q -k dhfr` passed 10 tests; `uv run ruff check src/mlx_atomistic/benchmarks/dhfr.py tests/test_benchmarks.py` passed.
**Risks / next:** implicit DHFR cannot run until a DHFR topology/parameter source provides GBSA/OBC radii and scales.

### Slice 5: Explicit PME DHFR MLX Runtime Or Readiness Blocker

**Objective:** Represent explicit PME DHFR/JAC as a normalized MLX row with a runnable result or concrete PME/runtime blocker.

**Acceptance criteria:**
- Emits normalized JSON for `dhfr-explicit-pme` with PME config/readiness metadata.
- If runnable, performs a bounded short run/evaluation and reports finite output.
- If blocked, names whether the blocker is input absence, PME atom-count envelope, orthorhombic/neutrality/cell readiness, lazy-neighbor runtime, or Metal environment.
- Keeps NPT/barostat validation out unless already required by the selected benchmark command.

**Verification:** `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --steps 1 --json && uv run pytest tests/test_benchmarks.py -q`

**Execution:** subagent recommended

**Depends on:** Slice 3

**Touches:** `src/mlx_atomistic/benchmarks/dhfr.py`, possible PME/artifact support files, `tests/test_benchmarks.py`

**Produces:** MLX explicit PME DHFR benchmark or readiness-blocked row.

**Status:** complete
**Evidence:** changed `src/mlx_atomistic/benchmarks/dhfr.py` and `tests/test_benchmarks.py`; `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --steps 1 --json` returned normalized `status=blocked`, `comparison_pair_id=dhfr-explicit-pme`, `atom_count=23558`, `step_count=1`, `runtime_attempted=false`, `runtime_blocker_category=amber_import_unsupported_terms`, PME config metadata, NetCDF coordinate format, and `unsupported_terms=["amber_10_12_nonbonded"]`; `uv run pytest tests/test_benchmarks.py -q -k dhfr` passed 10 tests; `uv run ruff check src/mlx_atomistic/benchmarks/dhfr.py tests/test_benchmarks.py` passed.
**Risks / next:** explicit PME DHFR cannot reach PME readiness/runtime until AMBER 10-12 nonbonded terms are either supported or explicitly mapped/filtered with a scientifically valid policy.

### Slice 6: DHFR Comparison Classification

**Objective:** Extend same-workload comparison so DHFR rows produce ratios only when semantics match.

**Acceptance criteria:**
- Adds `dhfr-implicit` and `dhfr-explicit-pme` pair specs.
- Indexes MLX and OpenMM DHFR rows by pair ID/case.
- Suppresses ratios for mismatched atom count, solvent model, PME status, timing metric, timestep semantics, blocked status, or diagnostic status.
- Existing LJ, GBSA/OBC, and TIP4P-Ew comparisons remain unchanged.

**Verification:** `uv run pytest tests/test_benchmarks.py -q && uv run python -m mlx_atomistic.benchmarks.same_workload_compare --mlx-json results/same-workload-openmm-comparison/mlx-dhfr-implicit.json --openmm-json results/same-workload-openmm-comparison/openmm-dhfr-implicit.json --json`

**Execution:** subagent recommended

**Depends on:** Slice 2, Slice 4, Slice 5

**Touches:** `src/mlx_atomistic/benchmarks/same_workload_compare.py`, `tests/test_benchmarks.py`

**Produces:** strict DHFR comparison gate.

**Status:** complete
**Evidence:** changed `src/mlx_atomistic/benchmarks/same_workload_compare.py` and `tests/test_benchmarks.py`; added `dhfr-implicit` and `dhfr-explicit-pme` pair specs, OpenMM case aliases, `ns/day` step-count gating, and DHFR solvent/electrostatics/timestep semantic checks; `uv run pytest tests/test_benchmarks.py -q -k 'same_workload_comparison'` passed 6 tests; `uv run ruff check src/mlx_atomistic/benchmarks/same_workload_compare.py tests/test_benchmarks.py` passed; regenerated `results/same-workload-openmm-comparison/summary.json` with 5 pairs, 3 comparable rows, and 2 blocked DHFR rows; both DHFR rows suppress ratios because MLX is blocked while OpenMM Reference one-step shape checks run.
**Risks / next:** DHFR rows intentionally remain blocked until MLX artifacts become runnable; OpenMM Reference one-step timings are shape checks, not OpenCL performance claims.

### Slice 7: DHFR Report Refresh

**Objective:** Refresh benchmark docs and local raw outputs so DHFR is no longer an unexplained blocked stretch row.

**Acceptance criteria:**
- Updates ladder and same-workload DHFR docs with implicit and explicit PME rows.
- Cites raw output paths under `results/same-workload-openmm-comparison/`.
- Includes reproducer commands for MLX and OpenMM DHFR rows.
- States comparable, diagnostic, or blocked status without broad framework claims.
- Keeps existing OpenMM DHFR context separate from same-workload rows unless semantics match.

**Verification:** `rg -n "DHFR|dhfr-implicit|dhfr-explicit-pme|comparable|diagnostic|blocked|raw output|Reproducer" docs/benchmarks && uv run python -c "from pathlib import Path; text='\\n'.join(p.read_text() for p in Path('docs/benchmarks').glob('*.md')); bad=('beats OpenMM','loses to OpenMM','leaderboard'); assert not any(item in text for item in bad), [item for item in bad if item in text]"`

**Execution:** subagent recommended

**Depends on:** Slice 6

**Touches:** `docs/benchmarks/`, raw outputs under `results/same-workload-openmm-comparison/`

**Produces:** refreshed human-readable DHFR benchmark report.

**Status:** complete
**Evidence:** updated `docs/benchmarks/same-workload-dhfr-stretch.md`, `docs/benchmarks/same-workload-openmm-comparison.md`, `docs/benchmarks/benchmark-ladder.md`, `docs/benchmarks/same-workload-comparison-matrix.md`, and `docs/benchmarks/README.md`; docs now split old `dhfr-stretch` into `dhfr-implicit` and `dhfr-explicit-pme`, cite raw DHFR MLX/OpenMM JSON paths, and include reproducers; `rg -n "DHFR|dhfr-implicit|dhfr-explicit-pme|comparable|diagnostic|blocked|raw output|Reproducer" docs/benchmarks` found the refreshed rows and reproducers; `uv run python -c "from pathlib import Path; text='\n'.join(p.read_text() for p in Path('docs/benchmarks').glob('*.md')); bad=('beats OpenMM','loses to OpenMM','leaderboard'); assert not any(item in text for item in bad), [item for item in bad if item in text]"` passed.
**Risks / next:** committed docs cite gitignored raw output paths; rerun reproducers to refresh the local JSON when hardware, OpenMM version, or DHFR artifact support changes.

### Slice 8: Regression Gate And Handoff

**Objective:** Verify DHFR benchmark behavior, runtime boundaries, docs, and focused regression suite.

**Acceptance criteria:**
- Focused lint passes or records concrete blockers.
- Focused pytest passes in a Metal-visible environment, or sandbox-only Metal failures are recorded separately.
- OpenMM imports remain in scripts/tests only.
- Plan evidence records whether each DHFR row is runnable, diagnostic, or blocked.

**Verification:** `uv run ruff check src/mlx_atomistic scripts tests/test_benchmarks.py tests/test_runtime_boundaries.py && uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q`

**Execution:** direct

**Depends on:** Slice 7

**Touches:** `.agent/work/2026-05-23-dhfr-benchmark-artifact-runtime/PLAN.md`

**Produces:** final verification evidence for auto-verify.

**Status:** complete
**Evidence:** `uv run ruff check src/mlx_atomistic scripts tests/test_benchmarks.py tests/test_runtime_boundaries.py` passed; sandbox `uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q` failed only with repeated `RuntimeError: [metal::load_device] No Metal device available`; Metal-visible rerun of the same focused pytest command passed 61 tests.
**Risks / next:** final outcome is implementation-complete for the approved slice scope, but DHFR runtime remains scientifically blocked by known artifact/import gaps, not by missing benchmark plumbing.

## Execution Routing And Topology

Default route: execute slices in order and continue after each verified slice.

Subagent routing:

- Slice 1: subagent recommended for input/readiness contract and test coverage.
- Slice 2: subagent recommended because it owns OpenMM reference behavior with disjoint script/tests scope.
- Slice 3: subagent recommended because it crosses AMBER import, artifact validation, and readiness semantics.
- Slice 4: subagent recommended because implicit DHFR may require runtime and GBSA fixes.
- Slice 5: subagent recommended because explicit PME may hit PME/readiness/runtime boundaries.
- Slice 6: subagent recommended because comparison classification is shared behavior.
- Slice 7: subagent recommended because benchmark docs are high-factual-risk reporting.

Parallel-safe groups:

- After Slice 1, Slice 2 and Slice 3 can run in parallel if their test edits are coordinated; write scopes are mostly disjoint except `tests/test_benchmarks.py`.
- After Slice 3, Slice 4 and Slice 5 can run in parallel if they keep case-specific helpers isolated inside `dhfr.py` and avoid conflicting support-file edits.
- Slice 6 waits for Slices 2, 4, and 5.
- Slice 7 waits for Slice 6.
- Slice 8 waits for Slice 7.

Coordination rule for multiagent execution: assign one worker as integration owner for `tests/test_benchmarks.py` and `dhfr.py` if Slices 4 and 5 run in parallel. Other workers must not revert concurrent edits.

Checkpoints: none. If DHFR cannot run, the approved outcome is a concrete normalized `blocked` or `diagnostic` row plus docs, not a stop for human decision.

Recommended review: run `auto-eng-review` before execution because the plan crosses benchmark schema, OpenMM reference behavior, AMBER artifact import, PME readiness, runtime boundaries, and high-factual-risk reporting.

## Aggregate Verification Commands

| Scope | Command |
| --- | --- |
| DHFR readiness | `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --readiness --json && uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --readiness --json` |
| OpenMM reference | `uv run python scripts/benchmark_openmm_dhfr.py --case dhfr-implicit --platform Reference --steps 1 --json && uv run python scripts/benchmark_openmm_dhfr.py --case dhfr-explicit-pme --platform Reference --steps 1 --json` |
| MLX DHFR rows | `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --steps 1 --json && uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --steps 1 --json` |
| Comparison/docs | `uv run pytest tests/test_benchmarks.py -q && rg -n "DHFR|dhfr-implicit|dhfr-explicit-pme|comparable|diagnostic|blocked|raw output|Reproducer" docs/benchmarks` |
| Focused gate | `uv run ruff check src/mlx_atomistic scripts tests/test_benchmarks.py tests/test_runtime_boundaries.py && uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q` |
