# SPEC: Performance Audit and Harness Hardening

## Bounded Goal

Create a benchmark-driven performance audit and hardened benchmark harness that measures current MLX atomistic performance, includes opt-in reference-engine context, and produces a prioritized optimization backlog without performing the optimization work itself.

## Broader Intent

After physics completeness, performance work should be evidence-led: the project needs reproducible measurements, coverage for recent atomistic features, and enough external context to choose the next optimization target instead of guessing.

## Work Classification

- Work scale: capability
- Work shape: mixed audit / coverage / performance-feature

## Selected Lenses

- product
- engineering
- runtime

## Target User Or Stakeholder

- Primary: project maintainers deciding where to invest optimization effort.
- Secondary: contributors running local Apple Silicon performance checks and comparing against reference-engine context.

## Linked Detail Files

- `spec/benchmark-context.md`: current benchmark surface, reference-engine boundary, external benchmark context, and design implications.

## Constraints And Risks

- Use `uv run ...` for Python execution.
- OpenMM and LAMMPS stay reference/dev surfaces; they must not be added to product runtime dependencies.
- The existing `uv` dev group is the default reference-engine setup unless implementation finds a concrete need for a separate documented environment.
- Reference-engine benchmark commands must fail soft with explicit blocker/status payloads when OpenMM, LAMMPS, OpenCL, fixtures, or platform support are unavailable.
- Benchmark design must separate fast developer checks from larger opt-in Apple Silicon performance runs.
- Existing benchmark outputs and docs should be normalized where needed, not replaced wholesale.
- Actual throughput optimization and custom kernel rewrites are deferred until the audit identifies a measured target.
- External benchmark numbers are context, not direct pass/fail targets, because hardware, engine semantics, and system configurations may not match MLX/Metal.

## Required Outcome

The completed change must answer these audit questions with fresh repo evidence and benchmark outputs:

- Which existing MLX benchmark scripts track product performance today, and what gaps remain for recent Phase 3 features?
- How are OpenMM and LAMMPS reference benchmarks installed, invoked, skipped, and reported without turning them into runtime dependencies?
- Which benchmark rows belong in the fast development gate versus larger opt-in performance runs?
- Which metrics and provenance fields are required for repeatable comparison across local runs, reference engines, and future optimization specs?
- What ranked optimization backlog follows from the measured baseline?

## Acceptance Criteria

| ID | Criterion | Verification |
|----|-----------|--------------|
| AC-01 | Benchmark inventory exists and maps current MLX scripts, tests, docs, outputs, and missing Phase 3 coverage. | Read committed audit artifact and verify it references existing files plus gaps for virtual sites, TIP4P-Ew, GBSA/OBC, soft-core/lambda, and replica exchange. |
| AC-02 | External benchmark context is incorporated into benchmark design with sources and caveats. | Read `spec/benchmark-context.md` or successor doc and verify OpenMM, LAMMPS, OpenBenchmarking, and MLX context are cited as context rather than pass/fail targets. |
| AC-03 | Reference-engine policy is explicit: OpenMM/LAMMPS use dev/reference setup, remain outside runtime dependencies, and fail soft when unavailable. | Inspect `pyproject.toml`, benchmark docs, and reference benchmark tests; run unavailable-platform smoke tests where applicable. |
| AC-04 | Benchmark harness outputs use a normalized schema for engine label, benchmark name, fixture/system, atom count, step/evaluation count, timing metric, hardware/runtime metadata, finite/status/blocker fields, and raw output path when applicable. | Run fast benchmark smoke tests and inspect JSON/CSV payloads for required fields. |
| AC-05 | Fast developer benchmark tier is defined and covered by tests that complete locally without heavyweight fixtures or mandatory reference engines. | Run `uv run pytest tests/test_benchmarks.py` and the documented fast benchmark commands. |
| AC-06 | Opt-in performance tier is defined for larger Apple Silicon runs and reference-engine context, including OpenMM and a LAMMPS decision path. | Read docs/benchmark command matrix and verify opt-in commands are marked non-CI/non-routine and have blocker behavior. |
| AC-07 | Recent Phase 3 physics features have benchmark coverage or explicit blocked/deferred rationale. | Verify committed benchmark rows or documented blockers for virtual sites, TIP4P-Ew, GBSA/OBC, soft-core/lambda, and replica exchange. |
| AC-08 | The audit produces a ranked optimization backlog from measured evidence, not intuition. | Read final audit/report artifact and verify each backlog item cites a benchmark row, metric, and reproduction command. |
| AC-09 | Existing validation and regression gates remain green. | Run `uv run ruff check src tests scripts && uv run pytest`. |

## Scope Coverage Decisions

- Included: benchmark harness audit, benchmark script hardening, new or extended coverage for recent atomistic physics features, external benchmark/context research, baseline measurement plan, reporting format, and prioritized optimization backlog.
- Deferred: actual throughput optimization, deeper kernel rewrites, and direct performance target commitments against OpenMM/LAMMPS numbers.
- Anti-goals: guessing hot paths, optimizing before measuring, making OpenMM or LAMMPS product runtime dependencies, treating external benchmarks as exact apples-to-apples targets, or benchmarking only toy LJ workloads while ignoring production-relevant physics additions.

## Assumptions

- The existing `uv` dev group is sufficient for OpenMM reference setup and for building LAMMPS from source with the configured OpenCL options unless execution proves otherwise.
- LAMMPS reference coverage may initially be a documented opt-in command path or blocked-status benchmark if a reliable local OpenCL run is not available.
- Representative large systems may require gitignored `results/` inputs and should not be required in routine CI.

## Anti-Goals

- Do not implement the optimization backlog in this change.
- Do not add heavyweight chemistry or ML helper packages for benchmark convenience without a concrete need.
- Do not make benchmark runs depend on `vendors/` as build inputs.
- Do not remove existing benchmark scripts unless an equivalent normalized replacement is committed and tested.
- Do not require OpenMM, LAMMPS, OpenCL, or large downloaded fixtures for the fast test suite.

## Framing Notes

- Existing MLX tracking is already present through `lj_md`, `md_performance`, `md_acceleration`, `cadence_sensitivity`, `mm_force_terms`, `pme_performance`, `stability`, and `validation_gauntlet` plus smoke coverage in `tests/test_benchmarks.py`.
- OpenMM currently has a standalone reference script at `scripts/benchmark_openmm_opencl.py`.
- LAMMPS is configured as a dev/reference dependency built from source with OpenCL settings, but a dedicated LAMMPS benchmark script is not currently visible and should be handled by the spec as an explicit gap.

## Review: Product

- Verdict: approved_with_risks
- Strength: The bet is clear: maintainers will use measured benchmark evidence to choose the next optimization target because current performance work spans MLX hot paths, reference engines, and new physics features that cannot be prioritized safely by intuition.
- Concern: The spec is plan-ready but does not itself encode agentic orchestration, so the plan must make parallel workstreams explicit instead of serializing independent audit, research, harness, and reporting tasks.
- Action: Proceed to auto-plan and require independent slices for benchmark inventory, external context, schema normalization, reference-engine policy, Phase 3 benchmark coverage, baseline audit, and backlog synthesis, with parallel-safe ownership called out.
- De-scoped: throughput optimization, custom kernel rewrites, direct pass/fail commitments against OpenMM or LAMMPS numbers
