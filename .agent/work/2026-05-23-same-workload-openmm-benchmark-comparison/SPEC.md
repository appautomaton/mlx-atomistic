# SPEC: Same-Workload MLX/OpenMM Benchmark Comparison

## Bounded Goal

Build runnable same-workload benchmark pairs for `mlx_atomistic` and OpenMM, then produce a first comparison report that separates real performance gaps from incompatible metrics.

## Broader Intent

Preserve the benchmark-harness work already completed while moving from internal MLX bottleneck probes to fair external comparison. The larger goal is to know where `mlx_atomistic` is slower, comparable, or not yet comparable against a mature engine before choosing optimization work.

## Work Scale And Shape

- Scale: capability-sized follow-on to the benchmark harness.
- Shape: benchmark comparison and reporting capability, with controlled workload pairs first and one real-system stretch target when feasible.

## Selected Lenses

- product
- engineering
- runtime
- content

## Target User Or Stakeholder

Engineers deciding the next `mlx_atomistic` performance objective. They know the current MLX benchmark rows are useful internally, but need same-workload comparisons before making OpenMM/LAMMPS-style performance claims.

## Content Direction

- Audience: local project engineers and the user reviewing performance direction.
- Thesis: `mlx_atomistic` should only be compared to OpenMM when the workload, physics, hardware, and metric are aligned; otherwise the report must label the row as diagnostic or blocked.
- Voice: short, direct engineering prose. Use tables for comparison, plain caveats for non-comparable rows, and no promotional language.
- Content anti-goals: no broad "MLX beats/loses to OpenMM" claim without same-workload evidence; no hiding blocked rows; no inflated language; no unexplained unit mixing.

## Scope Coverage Decisions

- Outcome: run benchmark pairs and produce a first comparison report.
- Workloads: staged approach, controlled pairs first, with one real-system stretch target if feasible.
- Reference engines: OpenMM first.
- Deferred: LAMMPS comparison beyond existing smoke/reference path. It may remain documented as deferred unless a controlled workload maps cleanly without expanding setup risk.

## Constraints And Risks

- Use `uv run ...` and Python 3.13.
- MLX product benchmark code remains under `src/mlx_atomistic/benchmarks/`.
- OpenMM reference commands remain under `scripts/` and outside `project.dependencies`.
- Raw benchmark JSON/CSV goes under gitignored `results/`; committed interpretation lives under `docs/benchmarks/`.
- Actual MLX execution may require local Metal access outside restrictive sandboxes.
- OpenMM/OpenCL availability must fail soft with normalized `status: "blocked"` payloads.
- The comparison must not mix `ns/day`, `steps/s`, and `ms/eval` as if they were equivalent.
- Real-system comparison is a stretch target until the MLX prep/runtime path can load and run the matching system with comparable physics.

## Required Outcome

The change must add a comparison layer that can run or clearly block MLX/OpenMM benchmark pairs. It must include controlled same-workload rows for MLX-owned code paths and OpenMM reference paths, plus a report that explains:

- what was compared,
- what was not comparable,
- what unit was used,
- what hardware/runtime was used,
- what the measured gap was when a valid pair exists,
- and which `mlx_atomistic` bottleneck should be optimized next.

Candidate controlled pairs:

| Workload | MLX side | OpenMM side | Expected metric |
| --- | --- | --- | --- |
| Synthetic LJ nonbonded/full loop | existing MLX MD/nonbonded benchmarks | synthetic OpenMM LJ benchmark | `steps/s` or normalized `ns/day` when timestep is meaningful |
| GBSA/OBC small system | MLX Phase 3 GBSA/OBC row | OpenMM OBC reference row | `ms/eval` or per-step timing |
| TIP4P-Ew water/virtual-site row | MLX Phase 3 or force-term row | OpenMM TIP4P-Ew reference row | `ms/eval` or per-step timing |
| Real-system stretch | MLX prepared DHFR-style candidate if feasible | existing OpenMM DHFR-style reference | `ns/day` |

## Acceptance Criteria

| ID | Criterion | Verification |
| --- | --- | --- |
| AC-01 | A comparison design artifact defines each workload pair, physics assumptions, metric, and comparable/non-comparable status. | Read committed docs/spec artifact and verify each pair has MLX command, OpenMM command, metric, and caveat. |
| AC-02 | At least three controlled MLX/OpenMM pairs are runnable or return normalized blocked payloads with concrete blockers. | Run the documented commands through `uv run ...` and inspect normalized JSON. |
| AC-03 | At least one controlled pair produces measured MLX and OpenMM values in the same metric family. | Inspect raw JSON and comparison report for a computed ratio or explicitly marked non-comparable reason. |
| AC-04 | The real-system stretch target is either measured or blocked with a specific missing capability, fixture, or parity reason. | Run or inspect the real-system command path and verify report status is `ok` or `blocked` with evidence. |
| AC-05 | The report does not claim global framework performance from unmatched microbenchmarks. | Read `docs/benchmarks/` report and verify unmatched rows are labeled diagnostic/context only. |
| AC-06 | OpenMM remains a reference/dev surface and is not added to product runtime dependencies. | Inspect `pyproject.toml`, import boundaries, and reference-script tests. |
| AC-07 | All new benchmark outputs use the normalized schema fields from the existing benchmark harness. | Run benchmark smoke tests and inspect representative JSON rows. |
| AC-08 | Verification records Metal/sandbox requirements instead of treating environment failure as code failure. | Run or document MLX/OpenMM execution path and confirm any sandbox-specific failure is reported as environment-specific. |

## Anti-Goals

- Do not optimize `mlx_atomistic` in this change.
- Do not add LAMMPS as a first-class comparison unless it maps cleanly without changing the approved OpenMM-first scope.
- Do not treat existing OpenMM DHFR/ApoA1/STMV numbers as directly comparable to current tiny MLX probes.
- Do not add OpenMM or LAMMPS to product runtime dependencies.
- Do not hide failed or blocked benchmark rows.
- Do not publish a leaderboard-style conclusion without same-workload evidence.

## Blocking Questions Or Assumptions

- Assumption: OpenMM is the first reference engine; LAMMPS remains deferred unless a low-risk controlled pair falls out naturally.
- Assumption: controlled pairs must land before the real-system stretch target.
- Assumption: a blocked real-system stretch target is acceptable if the blocker is concrete and reproducible.

## Review: Product

- Verdict: approved_with_risks
- Strength: The spec targets the exact user decision that is blocked today: choosing MLX optimization work from same-workload evidence instead of mixing OpenMM production `ns/day` with MLX diagnostic probes.
- Concern: The real-system stretch target may still block if MLX cannot load or run matching DHFR-style physics, so the first report must remain valuable even if only controlled pairs produce measured ratios.
- Action: Proceed to `auto-plan` with controlled MLX/OpenMM benchmark pairs as the must-ship path and real-system comparison as a documented stretch or blocked row.
- De-scoped: LAMMPS first-class comparison, MLX optimization, leaderboard-style framework claims
