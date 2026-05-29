# SPEC: DHFR Benchmark Artifact And Runtime Path

## Bounded Goal

Make DHFR a runnable `mlx_atomistic` benchmark by adding the missing prepared artifact/runtime path for both implicit DHFR and explicit PME DHFR, then refresh the MLX/OpenMM comparison report with honest runnable, blocked, or diagnostic results.

## Broader Intent

Move the benchmark ladder from controlled toy rows toward a real biomolecular benchmark without losing the strict comparability rules established for MLX/OpenMM performance reporting.

## Work Scale And Work Shape

- Work scale: capability
- Work shape: parity and benchmark enablement

## Selected Lenses

- product
- engineering
- runtime

## Target Stakeholder

`mlx_atomistic` maintainers deciding whether the next performance work should target real-system preparation/runtime gaps, PME scaling, force-term coverage, or lower-level kernel overhead.

## Scope Coverage Decisions

- Included: implicit DHFR benchmark path as the lower-risk first gate.
- Included: explicit PME DHFR benchmark path as the required second gate.
- Included: MLX prepared artifact creation or loading for DHFR, including structure, topology, coordinates, parameters, cell metadata where applicable, and runtime metadata.
- Included: OpenMM reference execution for matching DHFR cases, using normalized benchmark output.
- Included: comparison report refresh that labels DHFR rows as comparable, diagnostic, or blocked with concrete reasons.
- Included: tests for artifact readiness, runtime-boundary behavior, blocker reporting, and comparison classification.
- Deferred: ApoA1, Cellulose, STMV, GPCRmd, and other larger real-system benchmarks.
- Deferred: actual MLX performance optimization after DHFR becomes measurable.
- Deferred: LAMMPS DHFR mapping unless a low-risk reference path already exists.
- Anti-goal: no broad MLX-vs-OpenMM claim from DHFR alone.
- Anti-goal: no forcing a ratio when force fields, solvent model, PME behavior, atom count, or timing metric do not match.

## Constraints And Risks

- Use `uv run ...` for Python execution.
- Keep MLX benchmark/product code under `src/mlx_atomistic/`.
- Keep OpenMM reference execution under `scripts/`; OpenMM must remain a reference/dev surface, not a product runtime import.
- Keep raw benchmark outputs under gitignored `results/`.
- Keep committed interpretation under `docs/benchmarks/`.
- Prefer existing repo artifact, topology import, prep, PME, and benchmark schema paths over introducing a separate benchmark-only data model.
- Do not add heavyweight chemistry or ML helper packages unless a concrete DHFR input requirement proves unavoidable.
- If DHFR inputs are not already local, the implementation must fail closed with a concrete acquisition/preparation blocker instead of silently downloading uncontrolled data.
- Explicit PME DHFR may expose missing PME, topology, constraint, virtual-site, long-range correction, barostat, or artifact-readiness gaps.
- Metal/OpenCL benchmark execution may require a non-sandboxed local run; sandbox failures must be recorded as environment blockers, not product benchmark results.

## Required Outcome

The change must create a DHFR benchmark path that answers four operational questions:

- Can `mlx_atomistic` prepare or load a DHFR artifact with the force terms and metadata needed for runtime?
- Can MLX run an implicit DHFR benchmark and emit normalized timing/status output?
- Can MLX run an explicit PME DHFR benchmark and emit normalized timing/status output?
- Can the comparison layer report DHFR against OpenMM without mixing incompatible systems, solvent models, metrics, or blocker states?

The final report must show each DHFR row as one of:

- `comparable`: MLX and OpenMM ran the same DHFR semantics with compatible metrics.
- `diagnostic`: both sides ran, but semantics differ enough that no ratio should be trusted.
- `blocked`: one side cannot run, with a concrete blocker and next implementation decision.

## Acceptance Criteria

| ID | Criterion | Verification |
| --- | --- | --- |
| AC-01 | The repo has a DHFR artifact readiness path that identifies required input files, atom count, solvent model, force-field terms, cell metadata, and unsupported terms before benchmark execution. | `uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q` plus a DHFR readiness command documented in `docs/benchmarks/` |
| AC-02 | Implicit DHFR MLX benchmark emits normalized JSON with `status`, `fixture`, `atom_count`, `timing_metric`, `timing_value` or concrete `blocker`, and raw output path guidance. | `uv run python -m mlx_atomistic.benchmarks.<dhfr-module> --case dhfr-implicit --json` or the final planned command |
| AC-03 | Explicit PME DHFR MLX benchmark emits normalized JSON with PME/runtime metadata or a concrete blocker that names the missing artifact/runtime capability. | `uv run python -m mlx_atomistic.benchmarks.<dhfr-module> --case dhfr-explicit-pme --json` or the final planned command |
| AC-04 | OpenMM reference DHFR commands emit normalized outputs for matching implicit and explicit PME cases, or concrete blockers when reference inputs/platforms are unavailable. | `uv run python scripts/<openmm-dhfr-benchmark>.py --case dhfr-implicit --json` and `--case dhfr-explicit-pme --json` or final planned equivalents |
| AC-05 | The comparison gate produces DHFR ratios only when MLX/OpenMM semantics and metrics match; otherwise it preserves `diagnostic` or `blocked` reasons with no ratio. | `uv run pytest tests/test_benchmarks.py -q` |
| AC-06 | `docs/benchmarks/benchmark-ladder.md` and the same-workload comparison report are updated so DHFR no longer appears as an unexplained missing row. | `rg -n "DHFR|dhfr-implicit|dhfr-explicit-pme|comparable|diagnostic|blocked|raw output|Reproducer" docs/benchmarks` |
| AC-07 | Runtime boundary remains intact: OpenMM imports stay in reference scripts/tests and do not enter product runtime modules. | `uv run pytest tests/test_runtime_boundaries.py -q` |
| AC-08 | Focused lint/tests pass, or failures are recorded with concrete DHFR input, PME, Metal/OpenCL, or reference-engine blockers. | `uv run ruff check src/mlx_atomistic scripts tests/test_benchmarks.py tests/test_runtime_boundaries.py && uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q` |

## Anti-Goals

- Do not optimize MLX performance in this change.
- Do not add ApoA1, Cellulose, STMV, or GPCRmd benchmark scope.
- Do not claim DHFR proves general framework performance.
- Do not download or vendor large benchmark inputs without an explicit, documented source and reproducible preparation path.
- Do not hide missing DHFR capabilities behind generic benchmark failures.
- Do not add OpenMM to product runtime imports.

## Blocking Questions Or Assumptions

- User decision: both implicit DHFR and explicit PME DHFR are in scope.
- Assumption: implementation may sequence implicit DHFR before explicit PME DHFR, but completion requires both to be represented as runnable, diagnostic, or concretely blocked rows.
- Assumption: if exact OpenMM stock benchmark inputs are unavailable locally, the implementation can use a documented reproducible DHFR source or emit a concrete input-acquisition blocker.
