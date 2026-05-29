# PLAN: DHFR Runnable Benchmarks

## Goal

Implement the approved spec so `dhfr-implicit` and `dhfr-explicit-pme` become real runnable MLX benchmark rows, then compare them against OpenMM only when semantics match.

## Requirement Traceability

| Requirement | Covered by slices |
| --- | --- |
| AC-01 implicit OpenMM-derived artifact data | Slice 2, Slice 4 |
| AC-02 implicit GBSA metadata/provenance | Slice 2, Slice 4 |
| AC-03 explicit PME zero-LJ water O/H policy | Slice 1, Slice 4 |
| AC-04 explicit PME unsupported-term fail-closed tests | Slice 1 |
| AC-05 both MLX DHFR runtime rows return normalized `ok` payloads | Slice 5 |
| AC-06 OpenMM reference boundary stays outside product runtime | Slice 2, Slice 6, Slice 8 |
| AC-07 comparison ratios only for matching `ok` rows | Slice 6 |
| AC-08 benchmark docs refreshed | Slice 7 |
| AC-09 focused lint/tests pass | Slice 8 |

## Architecture Approach

Use `DESIGN.md`: first make prepared artifacts scientifically valid, then run short MLX runtime rows from saved artifacts. OpenMM may be used for explicit reference/prep extraction only; runtime benchmark execution must load artifacts and run through MLX.

## Ordered Slice Sequence

### Slice 1: AMBER Zero-LJ Water Pair Import Policy

**Objective:** Relax the explicit PME AMBER importer only for the local JAC water O/H negative `NONBONDED_PARM_INDEX` case that is equivalent to zero LJ.

**Acceptance criteria:**
- Local Amber20/JAC import no longer blocks on `unsupported_terms:amber_10_12_nonbonded` for OW/HW negative pair entries with zero mixed LJ epsilon.
- Compatibility metadata records the allowed negative-pair policy and affected type pairs.
- Nonzero HBOND/10-12 terms still block.
- `LENNARD_JONES_CCOEF`, modified nonzero LJ pair parameters, and non-water negative pair cases still block with concrete `unsupported_terms`.

**Verification:** `uv run pytest tests/test_openmm_mlx_parity.py tests/test_benchmarks.py -q -k 'amber or dhfr_explicit' && uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --prepare --json`

**Execution:** subagent recommended

**Depends on:** none

**Touches:** `src/mlx_atomistic/prep/topology_import.py`, `tests/test_openmm_mlx_parity.py`, `tests/test_benchmarks.py`

**Produces:** Explicit PME DHFR can prepare or reach the next concrete PME/artifact gate without the false 10-12 blocker.

**Status:** complete
**Evidence:** changed `src/mlx_atomistic/prep/topology_import.py` and `tests/test_openmm_mlx_parity.py`; added a narrowly-scoped zero-LJ water O/H negative `NONBONDED_PARM_INDEX` policy and tests for allowed water pairs plus blocked non-water pairs; `uv run ruff check src/mlx_atomistic/prep/topology_import.py tests/test_openmm_mlx_parity.py` passed; sandbox `uv run pytest tests/test_openmm_mlx_parity.py tests/test_benchmarks.py -q -k 'amber or dhfr_explicit'` failed only with `No Metal device available`; Metal-visible rerun passed 15 tests; `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --prepare --json` now reaches a concrete PME neutrality blocker `pme artifacts must be neutral; net_charge=-11` instead of `unsupported_terms:amber_10_12_nonbonded`; direct import metadata reports `allowed_zero_lj_water_pairs` with 2 affected OW/HW pairs.
**Risks / next:** explicit PME still cannot become runnable until the neutrality policy is addressed; that is now the real Slice 3/4 blocker rather than an AMBER 10-12 import blocker.

### Slice 2: OpenMM-Derived Implicit DHFR Prepared Artifact

**Objective:** Add a prep/extraction path that converts the OpenMM implicit DHFR `System` into a validated MLX `PreparedSystem`.

**Acceptance criteria:**
- Extracts particles, masses, positions, velocities, atom/residue names, charges, sigma, epsilon, bonded terms, torsions, constraints, and GBSA/OBC radii/scales.
- Stores GBSA metadata: solvent dielectric, solute dielectric, surface-area energy when available, nonbonded method/cutoff, and parameter source.
- Saves the artifact under `results/dhfr-artifacts/dhfr-implicit`.
- OpenMM imports remain isolated to script/prep/test surfaces and do not become normal product runtime imports.

**Verification:** `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --prepare --json && uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q -k 'dhfr_implicit or runtime_boundaries'`

**Execution:** subagent recommended

**Depends on:** none

**Touches:** `scripts/` or `src/mlx_atomistic/prep/`, `src/mlx_atomistic/benchmarks/dhfr.py`, `tests/test_benchmarks.py`, `tests/test_runtime_boundaries.py`

**Produces:** Implicit DHFR artifact with GBSA/OBC arrays and provenance.

**Status:** complete
**Evidence:** added `scripts/prepare_openmm_dhfr_implicit.py` and wired `src/mlx_atomistic/benchmarks/dhfr.py` so `dhfr-implicit --prepare` calls the script without importing OpenMM in product runtime modules; saved `results/dhfr-artifacts/dhfr-implicit` with 2,489 atoms, GBSA/OBC `gbsa_radius`/`gbsa_scale`, bonded terms, torsions, constraints, nonbonded exceptions, and GBSA metadata from OpenMM `amber99sb.xml`/`amber99_obc.xml`; `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --prepare --json` returns `status: "ok"` and artifact readiness `ready`; `uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q -k 'dhfr_implicit or runtime_boundaries'` passed 10 tests; `uv run ruff check scripts/prepare_openmm_dhfr_implicit.py src/mlx_atomistic/benchmarks/dhfr.py tests/test_benchmarks.py tests/test_runtime_boundaries.py` passed.
**Risks / next:** implicit DHFR now has a valid prepared artifact; it still needs Slice 4 runtime execution before it becomes a real `ns/day` benchmark row.

### Slice 3: DHFR Artifact Readiness Gate For Both Cases

**Objective:** Make `dhfr --prepare` for both cases validate saved artifacts and report artifact readiness rather than stale blockers.

**Acceptance criteria:**
- `dhfr-implicit --prepare --json` returns `status: "ok"` with artifact files, GBSA arrays present, and readiness metadata.
- `dhfr-explicit-pme --prepare --json` returns `status: "ok"` with artifact files, PME arrays present, and readiness metadata, unless a new downstream blocker is specific and scientifically justified.
- Both payloads retain raw input paths, atom counts, solvent/electrostatics model, and artifact paths.

**Verification:** `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --prepare --json && uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --prepare --json && uv run pytest tests/test_benchmarks.py -q -k dhfr`

**Execution:** direct

**Depends on:** Slice 1, Slice 2

**Touches:** `src/mlx_atomistic/benchmarks/dhfr.py`, `tests/test_benchmarks.py`

**Produces:** Prepared-artifact readiness is green for both DHFR rows or exposes only newly discovered blockers.

**Status:** complete
**Evidence:** `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --prepare --json` returns `status: "ok"`, saved artifact files, present GBSA arrays, and artifact readiness `ready`; `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --prepare --json` exits cleanly but returns `status: "blocked"` with the concrete PME neutrality blocker `pme artifacts must be neutral; net_charge=-11`, after the previous AMBER 10-12 blocker was removed; `uv run pytest tests/test_benchmarks.py -q -k dhfr` passed 12 tests.
**Risks / next:** explicit PME remains blocked until the benchmark policy handles the charged Amber20/JAC system, for example by adding neutralizing ions/artifact preparation or by documenting a scientifically valid charged-PME path.

### Slice 4: Bounded MLX DHFR Runtime

**Objective:** Replace the placeholder runtime blocker with a short MLX runtime path that loads DHFR artifacts and computes finite `ns/day`.

**Acceptance criteria:**
- Loads prepared DHFR artifacts with `load_prepared_mlx_artifact(require_production=True)`.
- Builds force terms through `build_mlx_system_from_artifact()`.
- Runs a bounded step count through existing MLX MD primitives.
- Reports finite timing, `dt_ps`, `step_count`, `simulated_ns`, energy/temperature evidence when available, and `timing_metric: "ns_per_day"`.
- Fails closed with concrete blocker only for real runtime capability gaps.

**Verification:** `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --steps 1 --json && uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --steps 1 --json && uv run pytest tests/test_benchmarks.py -q -k dhfr`

**Execution:** subagent recommended

**Depends on:** Slice 3

**Touches:** `src/mlx_atomistic/benchmarks/dhfr.py`, possible runtime/artifact support files, `tests/test_benchmarks.py`

**Produces:** Runnable MLX DHFR benchmark rows.

**Status:** complete
**Evidence:** changed `src/mlx_atomistic/benchmarks/dhfr.py` to load prepared artifacts, build MLX force terms through `build_mlx_system_from_artifact()`, run bounded NVT with physical unit scaling, and report finite `ns/day`, `dt_ps`, `simulated_ns`, wall time, energy, temperature, and constraint evidence; changed `src/mlx_atomistic/artifacts.py` so artifact builds can opt into eager nonbonded pair materialization and read nonbonded cutoff from protocol metadata; updated `scripts/prepare_openmm_dhfr_implicit.py` to preserve OpenMM zero-LJ hydrogens by replacing only `sigma=0, epsilon=0` particles with inert `sigma=1.0 Å` and recording the count; `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --steps 1 --json` returns `status: "ok"` with a finite `0.004 ps` one-step row; `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --steps 1 --json` remains blocked at artifact prep with `pme artifacts must be neutral; net_charge=-11`; `uv run pytest tests/test_benchmarks.py -q -k dhfr` passed 12 tests; targeted ruff passed.
**Risks / next:** implicit DHFR is runnable but the one-step number is a smoke/timing row, not an optimized performance claim; explicit PME needs a neutralized artifact policy before it can produce a runtime row.

### Slice 5: Raw Output Refresh For DHFR MLX And OpenMM

**Objective:** Regenerate local raw JSON rows for both MLX DHFR cases and both OpenMM DHFR reference cases.

**Acceptance criteria:**
- Writes MLX raw outputs under `results/same-workload-openmm-comparison/mlx-dhfr-implicit.json` and `mlx-dhfr-explicit-pme.json`.
- Writes OpenMM raw outputs under `results/same-workload-openmm-comparison/openmm-dhfr-implicit.json` and `openmm-dhfr-explicit-pme.json`.
- Raw MLX outputs are normalized, `status: "ok"`, finite, and include artifact/input metadata unless a named unavoidable blocker is present.
- OpenMM outputs are normalized `ok` or concrete `blocked` payloads.

**Verification:** `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --steps 1 --json > results/same-workload-openmm-comparison/mlx-dhfr-implicit.json && uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --steps 1 --json > results/same-workload-openmm-comparison/mlx-dhfr-explicit-pme.json && uv run python scripts/benchmark_openmm_dhfr.py --case dhfr-implicit --platform Reference --steps 1 --json > results/same-workload-openmm-comparison/openmm-dhfr-implicit.json && uv run python scripts/benchmark_openmm_dhfr.py --case dhfr-explicit-pme --platform Reference --steps 1 --json > results/same-workload-openmm-comparison/openmm-dhfr-explicit-pme.json`

**Execution:** direct

**Depends on:** Slice 4

**Touches:** `results/same-workload-openmm-comparison/`

**Produces:** Fresh DHFR raw evidence for comparison and docs.

**Status:** complete
**Evidence:** regenerated `results/same-workload-openmm-comparison/mlx-dhfr-implicit.json` (`status: "ok"`, current verification value `0.3094726757540178 ns/day`), `results/same-workload-openmm-comparison/mlx-dhfr-explicit-pme.json` (`status: "blocked"`, `pme artifacts must be neutral; net_charge=-11`), `results/same-workload-openmm-comparison/openmm-dhfr-implicit.json` (`status: "ok"`, current verification value `1.3136035206533072 ns/day` on OpenMM Reference), and `results/same-workload-openmm-comparison/openmm-dhfr-explicit-pme.json` (`status: "ok"`, current verification value `0.4990235699739706 ns/day` on OpenMM Reference); reran the raw-output commands with escalation after sandboxed `uv` cache access failed at `/Users/ac/.cache/uv`.
**Risks / next:** Reference-platform OpenMM one-step numbers are correctness/reference context, not OpenCL performance numbers; comparison still needs to suppress explicit PME because MLX is blocked.

### Slice 6: DHFR Comparison Ratio Gate Refresh

**Objective:** Ensure same-workload comparison computes DHFR ratios only for matching runnable rows and suppresses anything else.

**Acceptance criteria:**
- DHFR comparison rows use matching pair IDs and raw output paths.
- Ratios are computed for DHFR only when both rows are `ok`, `ns_per_day`, same atom count, same step count, same `dt_ps`, same solvent model, and same electrostatics model.
- Blocked/diagnostic rows preserve concrete reasons and no ratio.
- Existing LJ, GBSA/OBC, and TIP4P-Ew comparison behavior remains unchanged.

**Verification:** `uv run python -m mlx_atomistic.benchmarks.same_workload_compare --mlx-json results/same-workload-openmm-comparison/mlx-lj-synthetic-loop.json --mlx-json results/same-workload-openmm-comparison/mlx-phase3-controlled.json --mlx-json results/same-workload-openmm-comparison/mlx-dhfr-implicit.json --mlx-json results/same-workload-openmm-comparison/mlx-dhfr-explicit-pme.json --openmm-json results/same-workload-openmm-comparison/openmm-lj-synthetic-loop.json --openmm-json results/same-workload-openmm-comparison/openmm-gbsa-obc-small.json --openmm-json results/same-workload-openmm-comparison/openmm-tip4p-ew-water.json --openmm-json results/same-workload-openmm-comparison/openmm-dhfr-implicit.json --openmm-json results/same-workload-openmm-comparison/openmm-dhfr-explicit-pme.json --out results/same-workload-openmm-comparison/summary.json --json && uv run pytest tests/test_benchmarks.py -q -k same_workload_comparison`

**Execution:** direct

**Depends on:** Slice 5

**Touches:** `src/mlx_atomistic/benchmarks/same_workload_compare.py`, `tests/test_benchmarks.py`, `results/same-workload-openmm-comparison/summary.json`

**Produces:** Updated same-workload summary with honest DHFR ratio handling.

**Status:** complete
**Evidence:** reran `uv run python -m mlx_atomistic.benchmarks.same_workload_compare ... --out results/same-workload-openmm-comparison/summary.json --json`; summary marks `dhfr-implicit` `comparison_status: "comparable"` with MLX `0.3094726757540178 ns/day`, OpenMM Reference `1.3136035206533072 ns/day`, and `openmm_to_mlx_ratio: 4.244651058297036`; summary marks `dhfr-explicit-pme` `comparison_status: "blocked"` with no ratio because MLX raw output is blocked on `pme artifacts must be neutral; net_charge=-11`; existing LJ, GBSA/OBC, and TIP4P-Ew comparison rows remain comparable; `uv run pytest tests/test_benchmarks.py -q -k same_workload_comparison` passed 6 tests.
**Risks / next:** the implicit ratio is for a one-step MLX row and OpenMM Reference, so docs must label it as a smoke/reference comparison rather than a broad framework performance claim.

### Slice 7: Benchmark Documentation Refresh

**Objective:** Update benchmark docs so DHFR status, results, caveats, raw paths, and reproducers are human-readable and current.

**Acceptance criteria:**
- Docs show whether each DHFR row is runnable, comparable, diagnostic, or blocked.
- Docs cite the raw MLX and OpenMM output paths.
- Docs include reproduction commands for MLX, OpenMM, and same-workload comparison summary.
- Docs avoid broad framework claims and do not imply ratios where the comparison gate suppresses them.

**Verification:** `rg -n "DHFR|dhfr-implicit|dhfr-explicit-pme|ns/day|raw output|Reproducer|comparable|diagnostic|blocked" docs/benchmarks && uv run python -c "from pathlib import Path; text='\\n'.join(p.read_text() for p in Path('docs/benchmarks').glob('*.md')); bad=('beats OpenMM','loses to OpenMM','leaderboard'); assert not any(item in text for item in bad), [item for item in bad if item in text]"`

**Execution:** direct

**Depends on:** Slice 6

**Touches:** `docs/benchmarks/`

**Produces:** Refreshed benchmark report language.

**Status:** complete
**Evidence:** updated `docs/benchmarks/same-workload-openmm-comparison.md`, `docs/benchmarks/benchmark-ladder.md`, `docs/benchmarks/same-workload-dhfr-stretch.md`, `docs/benchmarks/same-workload-comparison-matrix.md`, and `docs/benchmarks/README.md` so `dhfr-implicit` is shown as a runnable/comparable one-step smoke row and `dhfr-explicit-pme` is shown as blocked on PME neutrality (`net_charge=-11`); docs cite raw MLX/OpenMM output paths, summary path, and reproducers; `rg -n "DHFR|dhfr-implicit|dhfr-explicit-pme|ns/day|raw output|Reproducer|comparable|diagnostic|blocked" docs/benchmarks` passed; `uv run python -c "from pathlib import Path; text='\\n'.join(p.read_text() for p in Path('docs/benchmarks').glob('*.md')); bad=('beats OpenMM','loses to OpenMM','leaderboard'); assert not any(item in text for item in bad), [item for item in bad if item in text]"` passed with escalation after sandboxed `uv` cache access failed.
**Risks / next:** docs intentionally frame DHFR implicit as one-step smoke/reference evidence, not broad MLX-vs-OpenMM performance ranking.

### Slice 8: Regression Gate And Handoff

**Objective:** Run focused verification and record final evidence for the runnable-DHFR change.

**Acceptance criteria:**
- Focused lint passes.
- Focused pytest passes in a Metal-visible environment, or sandbox-only Metal failures are separated from real failures.
- Runtime-boundary tests confirm OpenMM remains outside product runtime paths.
- Plan evidence records final DHFR runnable/comparison status.

**Verification:** `uv run ruff check src/mlx_atomistic scripts tests/test_benchmarks.py tests/test_openmm_mlx_parity.py tests/test_runtime_boundaries.py && uv run pytest tests/test_benchmarks.py tests/test_openmm_mlx_parity.py tests/test_runtime_boundaries.py -q`

**Execution:** direct

**Depends on:** Slice 7

**Touches:** `.agent/work/2026-05-23-dhfr-runnable-benchmarks/PLAN.md`

**Produces:** Final verification evidence ready for auto-verify.

**Status:** complete
**Evidence:** `uv run ruff check src/mlx_atomistic scripts tests/test_benchmarks.py tests/test_openmm_mlx_parity.py tests/test_runtime_boundaries.py` passed; `uv run pytest tests/test_benchmarks.py tests/test_openmm_mlx_parity.py tests/test_runtime_boundaries.py -q` passed 100% of the focused suite; runtime-boundary coverage includes the new OpenMM DHFR prep script as a documented reference surface, while `src/mlx_atomistic` remains free of OpenMM imports.
**Risks / next:** execution is complete; final auto-verify should summarize the accepted explicit PME blocker and confirm no ratio is emitted for that row.

## Execution Routing And Topology

Default route: execute slices in order and continue after each verified slice.

Subagent recommended:

- Slice 1 crosses AMBER import semantics and compatibility policy.
- Slice 2 crosses OpenMM extraction, artifact schema, and runtime-boundary constraints.
- Slice 4 crosses artifact loading, force-term construction, MD runtime, and benchmark schema.

Parallel-safe groups:

- Slices 1 and 2 may run in parallel if coordinated because their main write sets are mostly disjoint. Both touch `tests/test_benchmarks.py`, so one worker must own integration of shared tests.
- Slices 3 through 8 are serial.

Checkpoints: none. The approved outcome is runnable DHFR rows. If a newly discovered blocker prevents a row from becoming `ok`, execution should record the exact scientific/runtime blocker and continue only if the blocker is unavoidable within this spec.

Recommended review: run `auto-eng-review` before execution because this plan changes importer semantics, OpenMM-derived prep, artifact readiness, runtime benchmarks, comparison reporting, and docs.

## Aggregate Verification Commands

| Scope | Command |
| --- | --- |
| DHFR prepare | `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --prepare --json && uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --prepare --json` |
| DHFR runtime | `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --steps 1 --json && uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --steps 1 --json` |
| OpenMM reference | `uv run python scripts/benchmark_openmm_dhfr.py --case dhfr-implicit --platform Reference --steps 1 --json && uv run python scripts/benchmark_openmm_dhfr.py --case dhfr-explicit-pme --platform Reference --steps 1 --json` |
| Comparison summary | `uv run python -m mlx_atomistic.benchmarks.same_workload_compare --mlx-json results/same-workload-openmm-comparison/mlx-lj-synthetic-loop.json --mlx-json results/same-workload-openmm-comparison/mlx-phase3-controlled.json --mlx-json results/same-workload-openmm-comparison/mlx-dhfr-implicit.json --mlx-json results/same-workload-openmm-comparison/mlx-dhfr-explicit-pme.json --openmm-json results/same-workload-openmm-comparison/openmm-lj-synthetic-loop.json --openmm-json results/same-workload-openmm-comparison/openmm-gbsa-obc-small.json --openmm-json results/same-workload-openmm-comparison/openmm-tip4p-ew-water.json --openmm-json results/same-workload-openmm-comparison/openmm-dhfr-implicit.json --openmm-json results/same-workload-openmm-comparison/openmm-dhfr-explicit-pme.json --out results/same-workload-openmm-comparison/summary.json --json` |
| Focused gate | `uv run ruff check src/mlx_atomistic scripts tests/test_benchmarks.py tests/test_openmm_mlx_parity.py tests/test_runtime_boundaries.py && uv run pytest tests/test_benchmarks.py tests/test_openmm_mlx_parity.py tests/test_runtime_boundaries.py -q` |
