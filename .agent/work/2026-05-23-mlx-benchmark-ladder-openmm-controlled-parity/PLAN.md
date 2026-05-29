# PLAN: MLX Benchmark Ladder With OpenMM Controlled Parity

## Goal

Implement the approved SPEC: create an MLX-centered benchmark ladder and replace OpenMM GBSA/OBC and TIP4P-Ew placeholder blockers with controlled reference behavior where semantics allow.

## Requirement Traceability

| Requirement | Covered by slices |
| --- | --- |
| AC-01 ladder taxonomy | Slice 1 |
| AC-02 commands, metrics, raw paths, comparability | Slice 1, Slice 5 |
| AC-03 OpenMM GBSA/OBC controlled case | Slice 2 |
| AC-04 OpenMM TIP4P-Ew controlled case | Slice 3 |
| AC-05 ratio gate remains strict | Slice 4 |
| AC-06 refreshed comparison report | Slice 5 |
| AC-07 runtime boundary | Slice 2, Slice 3, Slice 6 |
| AC-08 focused gate | Slice 6 |

## Architecture Approach

Use the boundary in `DESIGN.md`: MLX benchmark code stays under `src/mlx_atomistic/benchmarks/`, OpenMM reference code stays under `scripts/`, raw outputs stay under `results/`, and committed interpretation stays under `docs/benchmarks/`. `same_workload_compare.py` remains the ratio gate.

## Content Constraints

- Audience: `mlx_atomistic` maintainers and performance reviewers.
- Thesis: MLX benchmarks should be organized by decision value first; OpenMM/LAMMPS comparisons appear only when semantics and metrics line up.
- Channel: benchmark docs.
- Source policy: repo-generated JSON, existing benchmark docs, and SPEC-listed reference URLs only.
- Factual risk: high for performance claims; numeric claims must trace to raw output paths or existing committed docs.
- Format: concise Markdown tables with row status, reproducer commands, caveats, and next-optimization guidance.
- Anti-goals: no framework leaderboard, no broad MLX-vs-OpenMM claim, no unexplained metric mixing.

## Ordered Slice Sequence

### Slice 1: Benchmark Ladder Doc

**Objective:** Add a committed benchmark ladder document that organizes MLX rows by decision value and maps fair reference-engine coverage.

**Acceptance criteria:**
- Defines micro/kernel, controlled MD, feature physics, scaling, reference parity, and stretch layers.
- Names must-ship rows for LJ, GBSA/OBC, TIP4P-Ew, soft-core/lambda, replica exchange, scaling, and DHFR stretch status.
- Each must-ship row includes MLX command, reference command or deferred mapping, metric family, raw output path, comparability status, and decision value.
- LAMMPS is documented as deferred except for simple LJ-like mapping notes.
- Content constraints are preserved: concise docs, no broad claims, no unit mixing.

**Verification:** `rg -n "micro|controlled MD|feature physics|scaling|reference parity|stretch|decision value|MLX command|OpenMM command|LAMMPS|metric|raw output|comparable|diagnostic|blocked|deferred" docs/benchmarks`

**Execution:** subagent recommended

**Depends on:** none

**Touches:** `docs/benchmarks/`

**Produces:** benchmark ladder doc linked from `docs/benchmarks/README.md`

**Status:** complete
**Evidence:** changed `docs/benchmarks/benchmark-ladder.md` and `docs/benchmarks/README.md`; `rg -n "micro|controlled MD|feature physics|scaling|reference parity|stretch|decision value|MLX command|OpenMM command|LAMMPS|metric|raw output|comparable|diagnostic|blocked|deferred" docs/benchmarks` passed; subagent spec review `APPROVED`; subagent quality review `APPROVED`.
**Risks / next:** none.

### Slice 2: OpenMM GBSA/OBC Controlled Case

**Objective:** Replace the OpenMM `gbsa-obc-small` placeholder blocker with a controlled GBSA/OBC reference payload where OpenMM can represent the MLX operation.

**Acceptance criteria:**
- OpenMM code remains in `scripts/benchmark_openmm_opencl.py`.
- `--case gbsa-obc-small --platform Reference --particles 4 --steps 1 --json` emits normalized JSON with `fixture=gbsa_obc_small`.
- If implemented as `ok`, payload records concrete OBC force setup, finite output, `ms_per_eval`, atom count, evaluation count, and timing value.
- If exact setup cannot be represented, payload remains `blocked` with a concrete semantic blocker rather than the current placeholder text.
- Invalid numeric inputs still exit nonzero before OpenMM import/platform handling.
- Tests cover `ok` or concrete `blocked` payload shape and invalid input behavior.

**Verification:** `uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q && uv run python scripts/benchmark_openmm_opencl.py --case gbsa-obc-small --platform Reference --particles 4 --steps 1 --json && uv run ruff check scripts tests/test_benchmarks.py tests/test_runtime_boundaries.py`

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `scripts/benchmark_openmm_opencl.py`, `tests/test_benchmarks.py`, `tests/test_runtime_boundaries.py`

**Produces:** OpenMM GBSA/OBC controlled reference surface

**Status:** complete
**Evidence:** changed `scripts/benchmark_openmm_opencl.py` and `tests/test_benchmarks.py`; `uv run python scripts/benchmark_openmm_opencl.py --case gbsa-obc-small --platform Reference --particles 4 --steps 1 --json` returned normalized `status=ok`, `fixture=gbsa_obc_small`, finite output, `ms_per_eval`, `evaluation_count=1`, and concrete `obc_force_setup`; non-JSON GBSA command returned `ms/eval`; `uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q` passed 45 tests; `uv run ruff check scripts tests/test_benchmarks.py tests/test_runtime_boundaries.py` passed; subagent spec review `APPROVED`; subagent quality review `APPROVED` after the non-JSON formatter fix.
**Risks / next:** none.

### Slice 3: OpenMM TIP4P-Ew Controlled Case

**Objective:** Replace the OpenMM `tip4p-ew-water` placeholder blocker with a controlled TIP4P-Ew reference payload or a concrete diagnostic/blocker when semantics do not match the MLX operation.

**Acceptance criteria:**
- OpenMM code remains in `scripts/benchmark_openmm_opencl.py`.
- `--case tip4p-ew-water --platform Reference --particles 4 --steps 1 --json` emits normalized JSON with `fixture=tip4p_ew_water`.
- Payload records the operation semantics, such as virtual-site reconstruction or full water force evaluation.
- If comparable to the MLX virtual-site row, status is `ok` with `ms_per_eval` and finite output.
- If not comparable, status is `diagnostic` or `blocked` with a concrete reason and no implied ratio.
- Invalid numeric inputs still exit nonzero before OpenMM import/platform handling.
- Tests cover `ok`, `diagnostic`, or concrete `blocked` payload shape and invalid input behavior.

**Verification:** `uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q && uv run python scripts/benchmark_openmm_opencl.py --case tip4p-ew-water --platform Reference --particles 4 --steps 1 --json && uv run ruff check scripts tests/test_benchmarks.py tests/test_runtime_boundaries.py`

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `scripts/benchmark_openmm_opencl.py`, `tests/test_benchmarks.py`, `tests/test_runtime_boundaries.py`

**Produces:** OpenMM TIP4P-Ew controlled reference surface

**Status:** complete
**Evidence:** changed `scripts/benchmark_openmm_opencl.py` and `tests/test_benchmarks.py`; `uv run python scripts/benchmark_openmm_opencl.py --case tip4p-ew-water --platform Reference --particles 4 --steps 1 --json` returned normalized `status=ok`, `fixture=tip4p_ew_water`, finite output, `ms_per_eval`, and `operation_semantics=virtual_site_reconstruction` / `openmm_operation=Context.computeVirtualSites`; invalid TIP4P particle count exited nonzero before OpenMM import handling; `uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q` passed 46 tests; `uv run ruff check scripts tests/test_benchmarks.py tests/test_runtime_boundaries.py` passed; subagent spec review `APPROVED`; subagent quality review `APPROVED`.
**Risks / next:** ratio remains valid only for the virtual-site reconstruction row, not full water force evaluation.

### Slice 4: Comparison Gate Update

**Objective:** Update comparison aggregation so new GBSA/OBC and TIP4P-Ew OpenMM statuses produce ratios only when both sides are truly comparable.

**Acceptance criteria:**
- `same_workload_compare.py` preserves strict ratio rules for `ok` rows only.
- `diagnostic` rows retain concrete reasons and no ratio.
- `blocked` rows retain concrete blockers and no ratio.
- Tests cover comparable, diagnostic, and blocked classification for controlled pairs.
- Existing LJ ratio behavior is preserved.

**Verification:** `uv run pytest tests/test_benchmarks.py -q && uv run ruff check src/mlx_atomistic/benchmarks tests/test_benchmarks.py`

**Execution:** subagent recommended

**Depends on:** Slice 2, Slice 3

**Touches:** `src/mlx_atomistic/benchmarks/same_workload_compare.py`, `tests/test_benchmarks.py`

**Produces:** updated comparison gate for controlled OpenMM parity rows

**Status:** complete
**Evidence:** changed `src/mlx_atomistic/benchmarks/same_workload_compare.py` and `tests/test_benchmarks.py`; ratios now require `comparable` classification after `ok` status, matching metric/value/count checks, and GBSA/TIP4P operation parity checks; diagnostic and blocked rows retain reasons and suppress ratios; `uv run pytest tests/test_benchmarks.py -q` passed 41 tests; `uv run ruff check src/mlx_atomistic/benchmarks tests/test_benchmarks.py` passed; subagent spec review `APPROVED`; subagent quality review `APPROVED`.
**Risks / next:** none.

### Slice 5: Refreshed Comparison Report

**Objective:** Refresh the benchmark report so LJ, GBSA/OBC, and TIP4P-Ew statuses reflect the ladder and controlled OpenMM reference behavior.

**Acceptance criteria:**
- Report lives under `docs/benchmarks/` and links from `docs/benchmarks/README.md`.
- Report labels LJ, GBSA/OBC, TIP4P-Ew, and DHFR stretch as comparable, diagnostic, blocked, or deferred.
- Every numeric performance claim traces to `results/` raw output or an existing committed benchmark doc.
- Report includes reproducer commands for the controlled rows.
- Report states the next optimization target only from measured comparable or clearly diagnostic evidence.
- Anti-claim scan passes for broad framework claims and promotional language.

**Verification:** `rg -n "lj-synthetic-loop|gbsa-obc-small|tip4p-ew-water|DHFR|ratio|diagnostic|blocked|deferred|raw output|Reproducer|next optimization" docs/benchmarks && uv run python -c "from pathlib import Path; text='\\n'.join(p.read_text() for p in Path('docs/benchmarks').glob('*.md')); bad=('beats OpenMM','loses to OpenMM','leaderboard','pivotal','crucial','serves as','Let\\'s dive'); assert not any(item in text for item in bad), [item for item in bad if item in text]"`

**Execution:** subagent recommended

**Depends on:** Slice 4

**Touches:** `docs/benchmarks/`, optional raw outputs under `results/`

**Produces:** refreshed human-readable comparison report

**Status:** complete
**Evidence:** changed `docs/benchmarks/same-workload-openmm-comparison.md`, `docs/benchmarks/benchmark-ladder.md`, and `docs/benchmarks/README.md`; refreshed row outputs under `results/same-workload-openmm-comparison/` include `summary.json`, OpenMM GBSA/TIP4P row files, and MLX GBSA/TIP4P row extracts pointing back to `mlx-phase3-controlled.json`; report labels LJ, GBSA/OBC, and TIP4P-Ew as `comparable` and DHFR stretch as `blocked`; required docs `rg` passed; anti-claim scan passed; subagent spec review `APPROVED`; subagent quality review `APPROVED` after row-extract traceability fix.
**Risks / next:** raw `results/` files remain gitignored local evidence; committed docs cite their paths and reproducers.

### Slice 6: Regression Gate And Handoff

**Objective:** Verify the ladder, OpenMM controlled parity, comparison gate, docs, and runtime boundary against the SPEC.

**Acceptance criteria:**
- Focused lint and tests pass, or failures are recorded with concrete Metal/OpenCL/import blockers.
- Runtime boundary remains intact: OpenMM imports stay in reference scripts/tests only.
- Docs prove AC-01 through AC-06.
- Sandbox/Metal/OpenCL requirements are documented.
- Plan evidence is recorded for completed slices.

**Verification:** `uv run ruff check src/mlx_atomistic/benchmarks scripts tests/test_benchmarks.py tests/test_runtime_boundaries.py && uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q`

**Depends on:** Slice 5

**Touches:** `PLAN.md` evidence notes only if verification uncovers gaps

**Produces:** final verification evidence for `auto-verify`

**Status:** complete
**Evidence:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/benchmarks scripts tests/test_benchmarks.py tests/test_runtime_boundaries.py` passed; `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q` passed 49 tests when run with Metal-visible execution; sandbox pytest failed only with `No Metal device available`; summary/control row JSON check passed for 3 comparable rows and required row files.
**Risks / next:** Metal access is required for benchmark tests and MLX raw-output refresh.

## Execution Routing And Topology

Default route: execute slices in order and continue after each slice verifies.

Subagent routing:
- Slice 1: subagent recommended because it is high-factual-risk documentation and sets benchmark taxonomy.
- Slice 2: subagent recommended because it touches OpenMM reference behavior, normalized schema output, validation, and runtime-boundary tests.
- Slice 3: subagent recommended because it touches OpenMM virtual-site semantics and may need diagnostic classification.
- Slice 4: subagent recommended because it updates shared comparison classification behavior.
- Slice 5: subagent recommended because it is high-factual-risk benchmark reporting.

Parallel-safe groups:
- Slice 2 and Slice 3 may run in parallel after Slice 1 if their edits to `scripts/benchmark_openmm_opencl.py` and `tests/test_benchmarks.py` are coordinated at merge time; direct serial execution is safer in one session.
- Slice 4 waits for Slice 2 and Slice 3.
- Slice 5 waits for Slice 4.
- Slice 6 waits for Slice 5.

Checkpoints: none. If OpenMM semantics do not support exact parity, the approved outcome is a concrete `diagnostic` or `blocked` row, not a human decision.

Recommended review: run `auto-eng-review` before execution because the plan crosses benchmark taxonomy, OpenMM physics setup, normalized schema semantics, comparison ratio rules, and high-factual-risk docs.

## Aggregate Verification Commands

| Scope | Command |
| --- | --- |
| Ladder docs | `rg -n "micro|controlled MD|feature physics|scaling|reference parity|stretch|decision value" docs/benchmarks` |
| Controlled OpenMM cases | `uv run python scripts/benchmark_openmm_opencl.py --case gbsa-obc-small --platform Reference --particles 4 --steps 1 --json && uv run python scripts/benchmark_openmm_opencl.py --case tip4p-ew-water --platform Reference --particles 4 --steps 1 --json` |
| Benchmark tests | `uv run pytest tests/test_benchmarks.py -q` |
| Runtime boundary | `uv run pytest tests/test_runtime_boundaries.py -q` |
| Focused lint | `uv run ruff check src/mlx_atomistic/benchmarks scripts tests/test_benchmarks.py tests/test_runtime_boundaries.py` |

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan has a clean execution boundary: ladder/docs under `docs/benchmarks/`, OpenMM reference work in `scripts/benchmark_openmm_opencl.py`, ratio logic in `same_workload_compare.py`, and runtime-boundary tests guarding product imports.
- Concern: TIP4P-Ew semantics may not match MLX virtual-site reconstruction if OpenMM exposes a fuller water-system operation, so execution must preserve diagnostic/blocker classification rather than forcing an `ok` ratio.
- Action: Proceed to `auto-execute`, starting with Slice 1 and treating Slice 2 and Slice 3 as serial unless a subagent route coordinates the shared script/test edits.
- Verified: context state, canonical SPEC/DESIGN/PLAN pointers, OpenMM case routing, GBSA OpenMM reference precedent in tests, TIP4P OpenMM parity helper, comparison ratio gate, runtime-boundary allowlist, and per-slice verification commands.
