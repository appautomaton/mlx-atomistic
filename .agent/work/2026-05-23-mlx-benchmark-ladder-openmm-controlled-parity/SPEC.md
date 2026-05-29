# SPEC: MLX Benchmark Ladder With OpenMM Controlled Parity

## Bounded Goal

Design and implement an MLX-centered benchmark ladder, then replace the current OpenMM `blocked` placeholders for GBSA/OBC and TIP4P-Ew with controlled reference cases where the operation semantics are comparable.

## Broader Intent

Create benchmark evidence that lets us choose the next `mlx_atomistic` performance target without mixing MLX microbenchmarks, OpenMM production `ns/day`, LAMMPS reference rows, and blocked real-system rows as if they were equivalent.

## Work Classification

- Mode: Builder
- Work scale: capability
- Work shape: mixed: benchmark design, audit, parity

## Selected Lenses

- product
- engineering
- content
- runtime

## Target Stakeholder

Engineers deciding the next `mlx_atomistic` performance objective. They need a small, repeatable benchmark ladder that separates MLX product bottlenecks from reference-engine context.

## Content Target

- Audience: `mlx_atomistic` maintainers and performance reviewers who already know the existing benchmark docs and need a clearer decision surface.
- Thesis: MLX benchmarks should be organized by decision value first, and OpenMM/LAMMPS comparisons should appear only where operation semantics and metrics line up.
- Voice direction: short engineering prose, tables where they clarify row status, concrete caveats, no promotional framing.
- Content anti-goals: no broad framework claims, no leaderboard language, no unexplained metric mixing, no hiding blocked or diagnostic rows.

## Constraints And Risks

- Use `uv run ...` for Python execution.
- Keep MLX benchmark code under `src/mlx_atomistic/benchmarks/`.
- Keep OpenMM reference execution under `scripts/`; OpenMM remains a dev/reference dependency.
- Do not add OpenMM or LAMMPS to product runtime dependencies.
- Keep raw benchmark outputs under gitignored `results/`.
- Keep committed interpretation under `docs/benchmarks/`.
- Use the normalized benchmark schema for new outputs.
- Invalid numeric inputs must continue to fail as validation errors, not become blocked reference rows.
- Missing OpenMM imports/platforms may return normalized `blocked` payloads with concrete blockers.
- Metal/OpenCL execution may require local non-sandboxed runs; docs and outputs must state that boundary.
- TIP4P-Ew may remain `diagnostic` if the OpenMM side measures full water force evaluation while MLX measures only virtual-site reconstruction.

## Reference Context

- OpenMM benchmark families include GBSA, RF, PME, ApoA1, AMOEBA, and Amber20 systems including DHFR, Cellulose, and STMV. Source: `https://sources.debian.org/src/openmm/8.1.2%2Bdfsg-12/examples/benchmark.py`
- LAMMPS benchmark families include LJ, Chain, EAM, Chute, and Rhodopsin/protein workloads. Source: `https://www.lammps.org/bench.html`
- OpenMMTools test systems include small reusable systems such as alanine dipeptide implicit, PME water boxes, and DHFR explicit. Source: `https://openmmtools.readthedocs.io/en/stable/testsystems.html`

These sources guide benchmark selection; they are not pass/fail targets for `mlx_atomistic`.

## Required Outcome

The change must produce a clear benchmark ladder with row taxonomy and decision value:

- Micro/kernel rows: isolated force, neighbor, GBSA/OBC, TIP4P-Ew, virtual-site, and synchronization costs.
- Controlled MD rows: tiny same-workload rows that can be run quickly and compared to OpenMM where semantics match.
- Feature-physics rows: MLX features such as GBSA/OBC, TIP4P-Ew, soft-core/lambda, and replica exchange, labeled by whether they are directly comparable or diagnostic.
- Scaling rows: opt-in size sweeps that show whether a bottleneck is fixed-size overhead, neighbor-list overhead, memory pressure, or force-evaluation cost.
- Reference-parity rows: OpenMM controlled references for LJ, GBSA/OBC, and TIP4P-Ew with normalized `ok`, `diagnostic`, or `blocked` semantics.
- Stretch rows: DHFR or other real-system rows remain explicitly blocked or deferred unless matching MLX prepared artifacts and runtime parity already exist.

The OpenMM GBSA/OBC and TIP4P-Ew controlled cases must move beyond placeholder blockers when operation semantics can be implemented safely. If implementation reveals a mismatch, the row must be classified as `diagnostic` or `blocked` with the reason.

## Scope Coverage Decisions

- Included: MLX benchmark ladder design and committed documentation.
- Included: row taxonomy, decision value, commands, metrics, raw paths, and comparability status.
- Included: OpenMM controlled GBSA/OBC reference case.
- Included: OpenMM controlled TIP4P-Ew reference case or concrete diagnostic/blocker if exact semantic match is not possible.
- Included: comparison helper/report updates so new OpenMM rows can produce ratios only when valid.
- Included: tests for normalized `ok`, `diagnostic`, and `blocked` behavior where applicable.
- Deferred: LAMMPS first-class mapping beyond simple LJ-like notes unless it falls out without expanding setup risk.
- Deferred: DHFR or other real-system MLX prepared-artifact/runtime parity.
- Deferred: actual MLX performance optimization.
- Anti-goal: no broad MLX-vs-OpenMM/LAMMPS performance claim.
- Anti-goal: no treating `steps/s`, `ms/eval`, and `ns/day` as one metric family.
- Anti-goal: no adding reference engines to product runtime imports.

## Acceptance Criteria

| ID | Criterion | Verification |
| --- | --- | --- |
| AC-01 | A committed benchmark ladder doc classifies rows by micro, controlled MD, feature physics, scaling, reference parity, and stretch purpose. | `rg -n "micro|controlled MD|feature physics|scaling|reference parity|stretch|decision value" docs/benchmarks` |
| AC-02 | The ladder names MLX commands, intended OpenMM/LAMMPS reference mapping, metric family, raw output path, and comparability status for each must-ship row. | `rg -n "MLX command|OpenMM command|LAMMPS|metric|raw output|comparable|diagnostic|blocked" docs/benchmarks` |
| AC-03 | OpenMM GBSA/OBC controlled case emits normalized `ok` or `blocked` JSON with concrete force setup or blocker, and invalid numeric input still exits nonzero. | `uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q` plus `uv run python scripts/benchmark_openmm_opencl.py --case gbsa-obc-small --platform Reference --particles 4 --steps 1 --json` |
| AC-04 | OpenMM TIP4P-Ew controlled case emits normalized `ok`, `diagnostic`, or `blocked` JSON with concrete operation semantics, and invalid numeric input still exits nonzero. | `uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q` plus `uv run python scripts/benchmark_openmm_opencl.py --case tip4p-ew-water --platform Reference --particles 4 --steps 1 --json` |
| AC-05 | The comparison helper computes ratios only for pairs where both sides are `ok`, metrics match, and atom/step semantics are compatible. | `uv run pytest tests/test_benchmarks.py -q` |
| AC-06 | The first refreshed comparison report includes at least LJ plus GBSA/OBC and TIP4P-Ew statuses, with ratios only for valid rows and blocker/diagnostic reasons otherwise. | `rg -n "lj-synthetic-loop|gbsa-obc-small|tip4p-ew-water|ratio|diagnostic|blocked|raw output" docs/benchmarks` |
| AC-07 | Runtime boundary remains intact: OpenMM imports stay in reference scripts/tests, not product runtime code. | `uv run pytest tests/test_runtime_boundaries.py -q` |
| AC-08 | Full focused gate passes, or failures are recorded with concrete Metal/OpenCL/import blockers. | `uv run ruff check src/mlx_atomistic/benchmarks scripts tests/test_benchmarks.py tests/test_runtime_boundaries.py && uv run pytest tests/test_benchmarks.py tests/test_runtime_boundaries.py -q` |

## Anti-Goals

- Do not implement MLX optimization changes.
- Do not build DHFR or other real-system MLX parity in this spec.
- Do not make LAMMPS a first-class comparison surface unless a simple existing row maps cleanly without scope expansion.
- Do not force a ratio for GBSA/OBC or TIP4P-Ew if the OpenMM and MLX operations differ.
- Do not hide blocked rows or reference-engine availability failures.
- Do not broaden the report into a framework ranking.

## Assumptions

- The current normalized schema is sufficient for the ladder and controlled OpenMM outputs.
- Existing `phase3_physics`, `md_performance`, and `same_workload_compare` code provide the starting MLX/comparison surfaces.
- OpenMM `Reference` platform is enough for correctness/shape tests; measured performance rows may still use OpenCL where locally available.
- LAMMPS and DHFR stay deferred unless planning discovers a low-risk same-workload row already present.
