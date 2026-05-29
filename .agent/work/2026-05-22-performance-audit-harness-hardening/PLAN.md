# PLAN: Performance Audit and Harness Hardening

## Goal

Implement the bounded goal in `SPEC.md`: create a benchmark-driven performance audit and hardened benchmark harness, without doing throughput optimization in this change.

## Requirement Traceability

| Requirement | Covered by slices |
| --- | --- |
| AC-01 benchmark inventory | Slice 1 |
| AC-02 external benchmark context | Slice 1, Slice 6 |
| AC-03 reference-engine policy | Slice 4 |
| AC-04 normalized output schema | Slice 2, Slice 4, Slice 5 |
| AC-05 fast developer tier | Slice 2, Slice 3, Slice 7 |
| AC-06 opt-in performance tier | Slice 3, Slice 4, Slice 6 |
| AC-07 Phase 3 feature coverage | Slice 5 |
| AC-08 ranked optimization backlog | Slice 6 |
| AC-09 validation and regression gates | Slice 7 |

## Ordered Slice Sequence

### Slice 1: Benchmark Inventory And Gap Matrix

**Objective:** Add a committed benchmark inventory that maps current scripts, tests, docs, result locations, fast/opt-in tier placement, and Phase 3 coverage gaps.

**Acceptance criteria:**
- Inventory references current MLX benchmark modules, reference scripts, benchmark docs, and tests.
- Inventory explicitly covers virtual sites, TIP4P-Ew, GBSA/OBC, soft-core/lambda, and replica exchange.
- External OpenMM, LAMMPS, OpenBenchmarking, and MLX context is summarized as design context, not pass/fail targets.

**Verification:** `uv run python -m pytest tests/test_benchmarks.py -q && rg -n "virtual sites|TIP4P|GBSA|soft-core|replica exchange|OpenBenchmarking|LAMMPS|OpenMM" docs/benchmarks .agent/work/2026-05-22-performance-audit-harness-hardening`

**Execution:** subagent required

**Depends on:** none

**Touches:** `docs/benchmarks/`, `.agent/work/2026-05-22-performance-audit-harness-hardening/`

**Produces:** committed inventory or audit artifact plus any active-change evidence note needed by later slices

**Status:** complete
**Evidence:** added `docs/benchmarks/inventory-gap-matrix.md`; linked it from `docs/benchmarks/README.md`; recorded current MLX benchmark modules, reference scripts/docs/tests, raw-output locations, fast/opt-in tier placement, and Phase 3 gaps for virtual sites, TIP4P-Ew, GBSA/OBC, soft-core/lambda, and replica exchange. Verification: `uv run python -m pytest tests/test_benchmarks.py -q` passed with 24 tests; `rg -n "virtual sites|TIP4P|GBSA|soft-core|replica exchange|OpenBenchmarking|LAMMPS|OpenMM" docs/benchmarks .agent/work/2026-05-22-performance-audit-harness-hardening` passed.
**Risks / next:** later slices still need to add normalized benchmark rows and reference-engine fail-soft coverage.

### Slice 2: Shared Benchmark Schema And Metadata Helpers

**Objective:** Add a shared schema helper for benchmark payloads and update fast product benchmark payloads to emit common provenance fields.

**Acceptance criteria:**
- Shared helper defines the normalized fields required by AC-04.
- Existing fast MLX benchmark payloads include engine label, benchmark name, fixture/system, atom/evaluation or step count, timing metric, hardware/runtime metadata, finite/status/blocker fields, command, commit, and raw output path where applicable.
- Existing benchmark-specific fields remain available.
- Existing `fixture`, `case`, `test`, `steps`, `step_count`, `evaluations`, `evaluation_count`, `mean_s`, `median_s`, and benchmark-specific timing fields are mapped into the normalized contract without deleting useful local detail.

**Verification:** `uv run pytest tests/test_benchmarks.py -q`

**Execution:** subagent required

**Depends on:** none

**Touches:** `src/mlx_atomistic/benchmarks/`, `tests/test_benchmarks.py`

**Produces:** schema helper and updated product benchmark smoke coverage

**Status:** complete
**Evidence:** changed `src/mlx_atomistic/benchmarks/schema.py`, `src/mlx_atomistic/benchmarks/__init__.py`, `src/mlx_atomistic/benchmarks/md_acceleration.py`, `src/mlx_atomistic/benchmarks/md_performance.py`, `src/mlx_atomistic/benchmarks/cadence_sensitivity.py`, `src/mlx_atomistic/benchmarks/pme_performance.py`, `src/mlx_atomistic/benchmarks/mm_force_terms.py`, and `tests/test_benchmarks.py`; coordinator verification `uv run pytest tests/test_benchmarks.py -q` passed with 24 tests and `uv run ruff check src/mlx_atomistic/benchmarks tests/test_benchmarks.py` passed; spec review `APPROVED`; quality review `APPROVED`.
**Risks / next:** later reference-engine and Phase 3 slices must adapt to the normalized helper and coordinate any additional `tests/test_benchmarks.py` edits.

### Slice 3: Benchmark Tier Command Matrix

**Objective:** Document and test the fast developer benchmark tier and the opt-in Apple Silicon/reference tier.

**Acceptance criteria:**
- Docs identify routine fast commands and larger opt-in commands.
- Opt-in commands are marked non-CI/non-routine and write raw outputs under `results/`.
- Command matrix separates MLX product runs from OpenMM/LAMMPS reference context.

**Verification:** `uv run pytest tests/test_benchmarks.py -q && rg -n "fast developer|opt-in|results/|openmm-reference|lammps-reference" docs/benchmarks`

**Execution:** direct

**Depends on:** Slice 1

**Touches:** `docs/benchmarks/`, `tests/test_benchmarks.py`

**Produces:** documented command matrix with smoke-verified fast commands

**Status:** complete
**Evidence:** updated `docs/benchmarks/README.md` with fast developer and opt-in Apple Silicon/reference command matrices; verification `uv run pytest tests/test_benchmarks.py -q` passed with 24 tests and `rg -n "fast developer|opt-in|results/|openmm-reference|lammps-reference" docs/benchmarks` returned the required command-matrix and inventory hits.
**Risks / next:** Slice 4 must add the documented `scripts/benchmark_lammps_opencl.py` fail-soft path so the LAMMPS command row is executable.

### Slice 4: Reference Engine Fail-Soft Harness

**Objective:** Normalize reference-engine benchmark outputs and add a LAMMPS opt-in path or blocked-status script without adding runtime dependencies.

**Acceptance criteria:**
- OpenMM reference output follows the shared schema and still fails soft on missing platform.
- OpenMM import-time failures are converted into blocked payloads instead of crashing before `main()` can report status.
- Existing stock OpenMM raw outputs are either wrapped by a normalized summary or documented as raw reference inputs with normalized committed summaries.
- LAMMPS has an explicit opt-in command path that reports `blocked` when import, OpenCL, fixture, or platform support is unavailable.
- `pyproject.toml` keeps OpenMM/LAMMPS in the dev/reference boundary, not `project.dependencies`.

**Verification:** `uv run pytest tests/test_benchmarks.py -q && uv run python scripts/benchmark_openmm_opencl.py --platform DefinitelyMissing --particles 16 --steps 1 --json && uv run python scripts/benchmark_lammps_opencl.py --particles 16 --steps 1 --json`

**Execution:** subagent required

**Depends on:** Slice 2

**Touches:** `scripts/benchmark_openmm_opencl.py`, `scripts/benchmark_lammps_opencl.py`, `docs/benchmarks/`, `tests/test_benchmarks.py`, `pyproject.toml`

**Produces:** normalized reference outputs and LAMMPS fail-soft reference surface

**Status:** complete
**Evidence:** changed `scripts/benchmark_openmm_opencl.py`, added `scripts/benchmark_lammps_opencl.py`, updated `docs/benchmarks/README.md`, and extended `tests/test_benchmarks.py`; verification `uv run pytest tests/test_benchmarks.py -q && uv run python scripts/benchmark_openmm_opencl.py --platform DefinitelyMissing --particles 16 --steps 1 --json && uv run python scripts/benchmark_lammps_opencl.py --particles 16 --steps 1 --json` passed. Ruff passed for the reference scripts and benchmark tests. Spec review approved; quality review issues were fixed by preserving invalid-input failures, recording LAMMPS requested device separately, and adding regression coverage.
**Risks / next:** LAMMPS device selection is recorded as requested but not applied because this installed GPU package accepts `platform` but not a `device` keyword.

### Slice 5: Phase 3 Physics Benchmark Coverage

**Objective:** Add fast benchmark rows or explicit blocked/deferred benchmark entries for virtual sites, TIP4P-Ew, GBSA/OBC, soft-core/lambda, and replica exchange.

**Acceptance criteria:**
- Each named Phase 3 feature has a benchmark row with timing/provenance or a documented blocked/deferred rationale.
- Rows use the normalized schema and remain fast enough for developer smoke tests.
- Coverage exercises MLX-owned code paths rather than only OpenMM parity helpers.
- Virtual-site rows measure reconstruction and force redistribution overhead, likely in `mm_force_terms.py`.
- TIP4P-Ew coverage emits benchmark JSON/CSV or a documented blocker around the existing parity helper.
- GBSA/OBC coverage measures OBC pair accumulation, surface-area term, force cost, or scaling.
- Soft-core/lambda coverage measures lambda-grid or `energy_forces_dlambda` overhead.
- Replica-exchange coverage reports per-replica throughput, swap overhead, acceptance rate, or history/materialization cost.

**Verification:** `uv run pytest tests/test_benchmarks.py tests/test_virtual_sites.py tests/test_gbsa.py tests/test_soft_core.py tests/test_replica_exchange.py -q`

**Execution:** subagent required

**Depends on:** Slice 2

**Touches:** `src/mlx_atomistic/benchmarks/`, `tests/test_benchmarks.py`, Phase 3 feature tests as needed

**Produces:** Phase 3 benchmark coverage rows or explicit benchmark blockers

**Status:** complete
**Evidence:** added `src/mlx_atomistic/benchmarks/phase3_physics.py`; added virtual-site reconstruction and force-redistribution rows to `src/mlx_atomistic/benchmarks/mm_force_terms.py`; extended `tests/test_benchmarks.py`. Verification `uv run pytest tests/test_benchmarks.py tests/test_virtual_sites.py tests/test_gbsa.py tests/test_soft_core.py tests/test_replica_exchange.py -q` passed. `uv run python -m mlx_atomistic.benchmarks.phase3_physics --evaluations 1 --waters 1 --atoms 4 --replica-steps 1 --json` returned seven normalized ok rows across virtual sites, TIP4P-Ew, GBSA/OBC, soft-core/lambda, and replica exchange. Ruff and `git diff --check` passed. Spec review approved; quality review timing issue was fixed by synchronizing each MLX virtual-site timing iteration.
**Risks / next:** Rows are fast synthetic probes, not production-scale performance claims.

### Slice 6: Baseline Audit Report And Ranked Backlog

**Objective:** Run the fast baseline suite and synthesize a committed audit report with a ranked optimization backlog tied to measured rows.

**Acceptance criteria:**
- Report cites fresh benchmark rows, metrics, reproduction commands, hardware/runtime metadata, and raw output locations.
- Backlog items are ranked by measured evidence and identify the likely next optimization spec.
- Reference-engine context is caveated and never treated as a direct pass/fail gate.

**Verification:** `uv run pytest tests/test_benchmarks.py -q && rg -n "Ranked optimization backlog|Reproducer|raw output|blocker|ns/day|ms/eval|steps/s" docs/benchmarks`

**Execution:** direct

**Depends on:** Slice 3, Slice 4, Slice 5

**Touches:** `docs/benchmarks/`, `results/` for gitignored raw outputs

**Produces:** benchmark audit report and evidence-backed optimization backlog

**Status:** complete
**Evidence:** added `docs/benchmarks/performance-audit-baseline.md` and raw gitignored JSON under `results/performance-audit-harness-hardening/`; verification `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_benchmarks.py -q` passed with 32 tests when rerun outside the sandbox for Metal access, and `rg -n "Ranked optimization backlog|Reproducer|raw output|blocker|ns/day|ms/eval|steps/s" docs/benchmarks` returned the required audit/report hits.
**Risks / next:** The committed backlog is based on fast synthetic rows; larger opt-in benchmark runs remain a follow-on validation step before committing to kernel-level optimization work.

### Slice 7: Regression Gate And Lifecycle Closeout

**Objective:** Verify the full change against the spec acceptance criteria and record any residual benchmark limitations.

**Acceptance criteria:**
- Full lint and test gate passes or any failure is recorded with a concrete blocker.
- Audit artifacts prove AC-01 through AC-08.
- No runtime dependency boundary regression is introduced.

**Verification:** `uv run ruff check src tests scripts && uv run pytest`

**Execution:** direct

**Depends on:** Slice 6

**Touches:** `PLAN.md` evidence notes only if verification uncovers gaps

**Produces:** final verification evidence for handoff to `auto-verify`

**Status:** complete
**Evidence:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts` passed; `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest` passed outside the sandbox for Metal access with 747 tests. The only full-gate regression found was the missing `scripts/benchmark_lammps_opencl.py` reference-script allowlist entry in `tests/test_runtime_boundaries.py`; it was fixed and the focused boundary test passed before the full rerun.
**Risks / next:** Full verification requires Metal access; sandbox-only runs still fail MLX array construction with `No Metal device available`.

## Execution Routing And Topology

Default route: continue slice-by-slice after each verification passes. No human checkpoint is required by this plan.

Subagent routing:
- Slice 1: documentation/audit subagent required.
- Slice 2: schema/harness subagent required.
- Slice 4: reference-engine subagent required.
- Slice 5: Phase 3 coverage subagent required.

Parallel-safe groups:
- Slice 1 and Slice 2 may start in parallel because their write sets are distinct except for later doc references.
- After Slice 2 passes, Slice 4 and Slice 5 may run in parallel if both workers treat `tests/test_benchmarks.py` changes as integration-sensitive and report their edits before merge.
- Slice 3 should run after Slice 1 so the command matrix reflects the inventory.
- Slice 6 must run after Slice 3, Slice 4, and Slice 5 so the backlog cites complete baseline rows.

Risk handling from product review:
- The plan does not serialize independent audit, schema, reference-engine, and Phase 3 coverage work.
- It keeps optimization deferred until Slice 6 produces measured backlog evidence.
- It preserves the reference-engine boundary and fail-soft behavior as explicit acceptance criteria.

## Aggregate Verification Commands

| Scope | Command |
| --- | --- |
| Benchmark smoke suite | `uv run pytest tests/test_benchmarks.py -q` |
| Phase 3 benchmark-adjacent tests | `uv run pytest tests/test_virtual_sites.py tests/test_gbsa.py tests/test_soft_core.py tests/test_replica_exchange.py -q` |
| OpenMM blocked smoke | `uv run python scripts/benchmark_openmm_opencl.py --platform DefinitelyMissing --particles 16 --steps 1 --json` |
| LAMMPS blocked smoke | `uv run python scripts/benchmark_lammps_opencl.py --particles 16 --steps 1 --json` |
| Full gate | `uv run ruff check src tests scripts && uv run pytest` |

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan has clear slice ordering, disjoint ownership for schema, docs, reference engines, and Phase 3 coverage, and concrete `uv run` verification commands tied back to AC-01 through AC-09.
- Concern: Slice 2, Slice 4, and Slice 5 all converge on `tests/test_benchmarks.py` and shared payload fields, so integration can drift unless the schema helper lands first and later workers adapt to that contract.
- Action: Start execution with Slice 2 as the schema contract owner, then merge Slice 1 in parallel only where it touches docs and active-change evidence.
- Verified: current Automaton diagnostics, canonical SPEC/DESIGN/PLAN pointers, PLAN slice dependencies, benchmark package surfaces, reference-engine policy, and Phase 3 coverage gaps.
