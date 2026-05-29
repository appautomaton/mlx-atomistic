# PLAN: Same-Workload MLX/OpenMM Benchmark Comparison

## Goal

Implement the approved SPEC: build runnable same-workload benchmark pairs for `mlx_atomistic` and OpenMM, then produce a first comparison report that separates real performance gaps from incompatible metrics.

## Requirement Traceability

| Requirement | Covered by slices |
| --- | --- |
| AC-01 comparison design artifact | Slice 1 |
| AC-02 three controlled runnable or blocked pairs | Slice 2, Slice 3, Slice 4 |
| AC-03 at least one measured same-metric pair | Slice 3, Slice 4 |
| AC-04 real-system stretch measured or blocked | Slice 5 |
| AC-05 no global claims from unmatched microbenchmarks | Slice 6, Slice 7 |
| AC-06 OpenMM reference/dev boundary | Slice 2, Slice 7 |
| AC-07 normalized schema for new outputs | Slice 2, Slice 3, Slice 4, Slice 6 |
| AC-08 Metal/sandbox requirements recorded | Slice 5, Slice 6, Slice 7 |

## Architecture Approach

Use the boundary in `DESIGN.md`: MLX benchmark code stays under `src/mlx_atomistic/benchmarks/`, OpenMM execution stays under `scripts/`, raw outputs stay under gitignored `results/`, and committed interpretation stays under `docs/benchmarks/`.

## Ordered Slice Sequence

### Slice 1: Comparison Pair Matrix

**Objective:** Add a committed pair matrix that defines each MLX/OpenMM workload pair, commands, physics assumptions, metric family, output path, and comparable status.

**Acceptance criteria:**
- The matrix defines synthetic LJ, GBSA/OBC, TIP4P-Ew, and DHFR-style stretch rows.
- Each row names MLX command, OpenMM command, expected metric, raw output path, and caveat/blocker policy.
- LAMMPS is explicitly deferred beyond the existing reference smoke path.
- Content constraints are preserved: short technical docs, no broad framework claims, no unit mixing.

**Verification:** `rg -n "synthetic LJ|GBSA/OBC|TIP4P-Ew|DHFR|OpenMM command|MLX command|comparable|blocked|LAMMPS" docs/benchmarks .agent/work/2026-05-23-same-workload-openmm-benchmark-comparison`

**Touches:** `docs/benchmarks/`, `.agent/work/2026-05-23-same-workload-openmm-benchmark-comparison/`

**Produces:** comparison pair matrix or design doc linked from benchmark docs

**Status:** complete
**Evidence:** added `docs/benchmarks/same-workload-comparison-matrix.md` and linked it from `docs/benchmarks/README.md`; verification `rg -n "synthetic LJ|GBSA/OBC|TIP4P-Ew|DHFR|OpenMM command|MLX command|comparable|blocked|LAMMPS" docs/benchmarks .agent/work/2026-05-23-same-workload-openmm-benchmark-comparison` returned the required pair, command, status, and deferral hits. Anti-goal scan for `beats OpenMM|loses to OpenMM|leaderboard|pivotal|crucial|vital|serves as|Let's dive` returned no hits.
**Risks / next:** Slice 2 must either implement the placeholder OpenMM `--case` commands or return concrete normalized blockers for unsupported controlled physics.

### Slice 2: OpenMM Controlled Reference Surface

**Objective:** Add or extend OpenMM reference scripts so controlled LJ, GBSA/OBC, and TIP4P-Ew workloads can emit normalized `ok` or `blocked` payloads.

**Acceptance criteria:**
- OpenMM reference code remains in `scripts/` and does not move into product package imports.
- New reference outputs use the existing normalized benchmark schema fields.
- Invalid numeric inputs still fail as validation errors instead of being hidden as blocked reference availability.
- Missing OpenMM platform, import, or unsupported physics returns `status: "blocked"` with a concrete blocker.
- Tests cover normalized ok or blocked payload shape for each controlled pair.

**Verification:** `uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q && uv run ruff check scripts tests/test_benchmarks.py tests/test_runtime_boundaries.py`

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `scripts/`, `tests/test_benchmarks.py`, `tests/test_runtime_boundaries.py`, `docs/benchmarks/`

**Produces:** OpenMM controlled-reference command surface

**Status:** complete
**Evidence:** extended `scripts/benchmark_openmm_opencl.py` with `--case` support for `synthetic-lj-periodic`, `gbsa-obc-small`, and `tip4p-ew-water`; controlled GBSA/OBC and TIP4P-Ew currently emit normalized `blocked` OpenMM reference payloads with concrete blockers while preserving existing LJ behavior. Added tests in `tests/test_benchmarks.py` for normalized controlled-pair payloads and invalid-input failure. Verification `uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q` passed with 43 tests; `uv run ruff check scripts tests/test_benchmarks.py tests/test_runtime_boundaries.py` passed.
**Risks / next:** Slice 3 must align MLX benchmark rows to the same comparison pair ids without hiding the benchmark-specific detail already present.

### Slice 3: MLX Controlled Pair Output Alignment

**Objective:** Ensure MLX controlled benchmark rows expose pair ids, metric fields, command provenance, and raw-output paths needed for same-workload comparison.

**Acceptance criteria:**
- Existing MLX LJ/nonbonded, GBSA/OBC, and TIP4P-Ew rows can be selected or summarized by comparison pair id.
- Existing benchmark-specific detail remains available.
- Rows retain normalized schema fields and finite/status/blocker semantics.
- Tests prove the comparison-facing fields exist without requiring OpenMM.

**Verification:** `uv run pytest tests/test_benchmarks.py -q && uv run ruff check src/mlx_atomistic/benchmarks tests/test_benchmarks.py`

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `src/mlx_atomistic/benchmarks/`, `tests/test_benchmarks.py`

**Produces:** MLX comparison-ready controlled rows

**Status:** complete
**Evidence:** added comparison-facing metadata to MLX normalized rows: `md_performance` synthetic LJ rows now expose `comparison_pair_id=lj-synthetic-loop`, and `phase3_physics` GBSA/OBC plus TIP4P-Ew rows expose `comparison_pair_id`, MLX role, metric family, command provenance, and planned raw output paths while retaining existing benchmark-specific fields. Added assertions in `tests/test_benchmarks.py`. Verification `uv run pytest tests/test_benchmarks.py -q` passed with 35 tests; `uv run ruff check src/mlx_atomistic/benchmarks tests/test_benchmarks.py` passed.
**Risks / next:** Slice 4 must classify rows using these fields without treating diagnostic `ms/eval` feature probes as direct full-MD throughput comparisons.

### Slice 4: Comparison Aggregator And Raw Run Harness

**Objective:** Add a runner or report helper that consumes MLX/OpenMM normalized JSON, writes raw outputs under `results/`, and classifies each pair as comparable, diagnostic, or blocked.

**Acceptance criteria:**
- The helper can run or ingest the three controlled pairs and produce one normalized comparison summary.
- At least one controlled pair computes a measured same-metric ratio when both sides are `ok`.
- Blocked or diagnostic rows retain concrete reasons and do not produce ratios.
- The helper records command, hardware/runtime, raw output paths, and sandbox/Metal/OpenCL caveats.

**Verification:** `uv run pytest tests/test_benchmarks.py -q && uv run python -m mlx_atomistic.benchmarks.phase3_physics --evaluations 1 --waters 1 --atoms 4 --replica-steps 1 --json && uv run python scripts/benchmark_openmm_opencl.py --platform DefinitelyMissing --particles 16 --steps 1 --json`

**Execution:** subagent recommended

**Depends on:** Slice 2, Slice 3

**Touches:** `src/mlx_atomistic/benchmarks/`, `scripts/`, `tests/test_benchmarks.py`, `results/`

**Produces:** comparison summary JSON and raw controlled-pair outputs

**Status:** complete
**Evidence:** added `src/mlx_atomistic/benchmarks/same_workload_compare.py`, an ingest helper/CLI that indexes normalized MLX rows by `comparison_pair_id`, maps OpenMM controlled payloads by case/fixture, emits normalized comparison summary fields, classifies rows as comparable/diagnostic/blocked, and computes `openmm_to_mlx_ratio` only when both sides are `ok` with matching metric, atom count, and step semantics. Added `tests/test_benchmarks.py` coverage proving a comparable LJ ratio and suppressed ratios for blocked/missing rows. Verification `uv run pytest tests/test_benchmarks.py -q` passed with 36 tests; `uv run python -m mlx_atomistic.benchmarks.phase3_physics --evaluations 1 --waters 1 --atoms 4 --replica-steps 1 --json` emitted comparison metadata; `uv run python scripts/benchmark_openmm_opencl.py --platform DefinitelyMissing --particles 16 --steps 1 --json` emitted a normalized blocked OpenMM payload; `uv run ruff check src/mlx_atomistic/benchmarks scripts tests/test_benchmarks.py tests/test_runtime_boundaries.py` passed.
**Risks / next:** Slice 5 should add the DHFR stretch row as measured only if matching MLX physics exists; otherwise it must be an explicit blocked row that does not affect controlled-pair usefulness.

### Slice 5: Real-System Stretch Probe

**Objective:** Add a DHFR-style real-system comparison row that either measures MLX/OpenMM with matching physics or records a concrete blocker.

**Acceptance criteria:**
- The row names the intended system, physics, metric, and OpenMM reference.
- If measured, MLX and OpenMM use compatible timestep/cutoff/force-field assumptions and report `ns/day` or a clearly justified metric.
- If blocked, the blocker names the missing MLX capability, fixture, prep path, or parity reason.
- The controlled-pair comparison summary remains useful if this row is blocked.

**Verification:** `rg -n "DHFR|real-system|ns/day|blocked|missing|fixture|parity" docs/benchmarks .agent/work/2026-05-23-same-workload-openmm-benchmark-comparison`

**Execution:** subagent recommended

**Depends on:** Slice 4

**Touches:** `docs/benchmarks/`, optional `scripts/`, optional `src/mlx_atomistic/benchmarks/`

**Produces:** real-system measured or blocked row

**Status:** complete
**Evidence:** added `docs/benchmarks/same-workload-dhfr-stretch.md` and linked it from `docs/benchmarks/README.md`; the row names DHFR explicit PME, AMBER99SB/HBonds/4 fs/PME/single-precision physics, `ns/day`, the existing OpenMM raw output `results/openmm-opencl-dhfr-m5max.json`, and a concrete MLX blocker: missing DHFR prepared artifact plus matching parity/runtime path. Verification `rg -n "DHFR|real-system|ns/day|blocked|missing|fixture|parity" docs/benchmarks .agent/work/2026-05-23-same-workload-openmm-benchmark-comparison` returned required hits.
**Risks / next:** Slice 6 must report DHFR as blocked context only and avoid using the OpenMM DHFR number as a direct MLX/OpenMM result.

### Slice 6: First Comparison Report

**Objective:** Commit a human-readable benchmark report that explains measured pairs, blocked rows, ratios, caveats, and the next MLX optimization target.

**Acceptance criteria:**
- Report lives under `docs/benchmarks/` and links from `docs/benchmarks/README.md`.
- Channel: benchmark docs; format: Markdown report with tables and reproduction commands.
- Source policy: repo-generated benchmark JSON plus existing repo benchmark docs only.
- Factual risk: high; every performance claim must trace to a raw output path or an existing benchmark doc.
- Report labels each row as comparable, diagnostic, or blocked.
- Report includes the DHFR-style real-system stretch status from Slice 5.
- Report avoids global MLX-vs-OpenMM claims unless supported by same-workload data.
- Report identifies the next optimization target from measured evidence, or says the evidence is insufficient.

**Verification:** `rg -n "comparable|diagnostic|blocked|raw output|Reproducer|OpenMM|mlx_atomistic|ratio|next optimization" docs/benchmarks && uv run python -c "from pathlib import Path; text='\\n'.join(p.read_text() for p in Path('docs/benchmarks').glob('*.md')); bad=('beats OpenMM','loses to OpenMM','leaderboard'); assert not any(item in text for item in bad), [item for item in bad if item in text]"`

**Execution:** subagent recommended

**Depends on:** Slice 5

**Touches:** `docs/benchmarks/`

**Produces:** committed comparison report

**Status:** complete
**Evidence:** ran controlled raw outputs under `results/same-workload-openmm-comparison/` and generated `summary.json`; added `docs/benchmarks/same-workload-openmm-comparison.md` and linked it from `docs/benchmarks/README.md`. The report includes one measured same-metric row (`lj-synthetic-loop`: MLX 37.8469 steps/s, OpenMM 727.9128 steps/s, ratio 19.2331 OpenMM/MLX), blocked GBSA/OBC and TIP4P-Ew rows with OpenMM blockers, the blocked DHFR stretch status, reproducer commands, raw output paths, and a narrow next optimization target. Verification `rg -n "comparable|diagnostic|blocked|raw output|Reproducer|OpenMM|mlx_atomistic|ratio|next optimization" docs/benchmarks` passed; anti-claim scan for `beats OpenMM`, `loses to OpenMM`, and `leaderboard` passed.
**Risks / next:** Slice 7 must run the regression gate; the full test suite may require Metal access outside the default sandbox.

### Slice 7: Regression Gate And Handoff

**Objective:** Verify the comparison capability against the SPEC and record residual limitations for execution handoff.

**Acceptance criteria:**
- Full lint and tests pass, or failures are recorded with concrete blockers.
- Runtime boundary remains intact: OpenMM imports stay in allowed reference scripts/tests only.
- Docs prove AC-01 through AC-08.
- Sandbox/Metal/OpenCL requirements are documented.

**Verification:** `uv run ruff check src tests scripts && uv run pytest`

**Depends on:** Slice 6

**Touches:** `PLAN.md` evidence notes only if verification uncovers gaps

**Produces:** final verification evidence for `auto-verify`

**Status:** complete
**Evidence:** regression gate passed. `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts` returned `All checks passed!`; `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest` passed with 751 tests in 84.05s. Runtime boundary remains covered by `tests/test_runtime_boundaries.py`, and OpenMM additions stayed in `scripts/benchmark_openmm_opencl.py` plus tests/docs.
**Risks / next:** continue into final verification against SPEC acceptance criteria.

## Execution Routing And Topology

Default route: execute slices in order and continue after each slice passes verification.

Subagent routing:
- Slice 2: subagent recommended because it touches OpenMM reference code, runtime-boundary tests, and fail-soft behavior.
- Slice 3: subagent recommended because it changes shared MLX benchmark payload fields.
- Slice 4: subagent recommended because it integrates both sides and raw-output handling.
- Slice 5: subagent recommended because real-system feasibility is uncertain and should not block controlled-pair delivery.
- Slice 6: subagent recommended because it is a high-factual-risk technical report.

Parallel-safe groups:
- Slice 2 and Slice 3 may run in parallel after Slice 1 because their primary write sets are disjoint.
- Slice 4 waits for Slice 2 and Slice 3.
- Slice 5 waits for Slice 4.
- Slice 6 waits for Slice 5 so the report can include the measured or blocked real-system status.

Checkpoints: none. The approved scope already accepts blocked rows when blockers are concrete.

Engineering review: recommended before execution because the plan crosses product benchmark schemas, OpenMM reference scripts, raw output orchestration, runtime-boundary tests, and high-factual-risk reporting.

## Aggregate Verification Commands

| Scope | Command |
| --- | --- |
| Benchmark smoke | `uv run pytest tests/test_benchmarks.py -q` |
| Runtime boundary | `uv run pytest tests/test_runtime_boundaries.py -q` |
| Focused lint | `uv run ruff check src/mlx_atomistic/benchmarks scripts tests/test_benchmarks.py tests/test_runtime_boundaries.py` |
| Full gate | `uv run ruff check src tests scripts && uv run pytest` |

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan has a safe execution boundary: MLX code stays in `src/mlx_atomistic/benchmarks/`, OpenMM stays in `scripts/`, raw outputs stay under `results/`, and committed interpretation stays in `docs/benchmarks/`.
- Concern: Slice 2 and Slice 4 depend on OpenMM controlled-pair behavior that may vary by platform/import/runtime availability, so implementation must preserve explicit `blocked` payloads and invalid-input failures rather than forcing every pair to look comparable.
- Action: Proceed to `auto-execute`, starting with Slice 1 and keeping Slice 2 and Slice 3 parallel-safe only until their shared `tests/test_benchmarks.py` changes are integrated.
- Verified: context state, canonical SPEC/DESIGN/PLAN pointers, product review, slice dependencies, normalized schema helper, OpenMM reference script boundary, runtime-boundary tests, and benchmark smoke test surfaces.
