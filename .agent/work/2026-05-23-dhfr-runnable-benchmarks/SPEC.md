# SPEC: DHFR Runnable Benchmarks

## Bounded Goal

Make the DHFR MLX benchmark rows runnable end-to-end for the implicit GBSA/OBC case and the explicit PME case, then produce honest MLX/OpenMM comparison rows with ratios only when semantics match.

## Broader Intent

The larger goal is to turn DHFR from a diagnostic benchmark-plumbing exercise into a real performance benchmark for `mlx_atomistic`, while preserving the existing rule that OpenMM is reference/prep evidence and not a product runtime dependency.

## Work Scale And Shape

- Scale: medium-to-large engineering change.
- Shape: parity/runtime enablement with benchmark verification.

## Selected Lenses

- product
- engineering
- runtime

## Target User Or Stakeholder

Project maintainer evaluating whether `mlx_atomistic` can run real DHFR molecular-dynamics workloads and compare them responsibly against OpenMM reference behavior.

## Scope Coverage Decisions

- Included: implicit DHFR GBSA/OBC artifact creation from a real parameter source.
- Included: explicit DHFR PME AMBER/JAC artifact import by handling the known water O/H negative LJ pair case scientifically.
- Included: short bounded MLX benchmark execution for both DHFR rows, producing normalized JSON.
- Included: OpenMM reference shape/performance rows only as comparison/reference evidence.
- Included: docs and comparison summary updates that distinguish runnable, diagnostic, and blocked cases.
- Deferred: broad performance optimization after the rows run.
- Deferred: ApoA1, Cellulose, STMV, LAMMPS parity, and general protein benchmark expansion.
- Anti-goal: reporting a DHFR speed ratio if MLX/OpenMM semantics do not match.

## Constraints And Risks

- OpenMM may be used in `scripts/`, tests, or explicit prep/import tooling, but OpenMM must not become a product runtime dependency under `src/mlx_atomistic/` normal execution paths.
- Implicit DHFR must not fake GBSA/OBC arrays. The source must be OpenMM `ForceField("amber99sb.xml", "amber99_obc.xml")` or an equivalent traceable force-field source.
- The implicit extractor must capture enough of the OpenMM `System` to build a real MLX artifact, not only `gbsa_radius` and `gbsa_scale`.
- The explicit PME AMBER importer may relax the current `amber_10_12_nonbonded` blocker only for the observed zero-LJ water O/H case: negative `NONBONDED_PARM_INDEX`, zero HBOND coefficients, and standard mixed LJ epsilon equal to zero.
- Any nonzero AMBER 10-12/HBOND coefficients, 12-6-4 coefficients, modified nonzero LJ pair parameters, or non-water negative pair cases must remain blocked with a concrete reason.
- Full tests and benchmark runs may require Metal-visible execution outside restrictive sandboxes.
- If a new downstream blocker appears after artifact creation, the implementation must either fix it inside this scope or record a concrete scientific/runtime blocker; it must not silently downgrade the goal.

## Required Outcome

The completed change must provide a working DHFR benchmark path:

- `dhfr-implicit` prepares a real MLX artifact containing bonded, nonbonded, constraint, and GBSA/OBC data from a traceable parameter source.
- `dhfr-explicit-pme` prepares a real MLX artifact from the Amber20/JAC inputs, including PME metadata and a documented policy for zero-LJ water O/H negative `NONBONDED_PARM_INDEX` entries.
- `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --steps 1 --json` returns `status: "ok"` with finite timing unless a newly discovered blocker is scientifically unavoidable and documented.
- `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --steps 1 --json` returns `status: "ok"` with finite timing unless a newly discovered blocker is scientifically unavoidable and documented.
- The same-workload comparison helper includes both DHFR rows and computes ratios only for matching `ok` rows with matching atom count, metric, timestep, solvent model, and electrostatics model.
- Benchmark docs explain the actual DHFR status in human-readable language and cite raw output paths and reproducers.

## Acceptance Criteria

| ID | Check |
| --- | --- |
| AC-01 | Implicit DHFR prep extracts OpenMM GBSA/OBC radii/scales, charges, LJ parameters, bonded terms, constraints, masses, positions, and metadata into a validated MLX prepared artifact. |
| AC-02 | Implicit DHFR artifact records GBSA metadata including solvent dielectric, solute dielectric, nonbonded method/cutoff, parameter source, and raw input provenance. |
| AC-03 | Explicit PME AMBER import succeeds for the local JAC fixture by treating only zero-LJ water O/H negative pair entries as zero LJ and recording that policy in compatibility metadata. |
| AC-04 | Explicit PME AMBER import still blocks nonzero 10-12/HBOND terms, 12-6-4 terms, modified nonzero pair LJ parameters, and non-water negative pair cases with tests. |
| AC-05 | Both DHFR MLX benchmark commands with `--steps 1 --json` return normalized payloads with `status: "ok"`, finite outputs, atom counts, timing metric `ns_per_day`, and raw artifact/input metadata, unless a newly discovered unavoidable blocker is captured as a named follow-up gap. |
| AC-06 | OpenMM reference DHFR commands still run or fail closed with normalized blocked payloads; OpenMM imports remain outside product runtime paths. |
| AC-07 | Same-workload comparison emits DHFR ratios only when both MLX and OpenMM rows are `ok` and semantics match; otherwise it suppresses ratios with concrete blockers. |
| AC-08 | Docs under `docs/benchmarks/` are refreshed to show DHFR runnable status, results, remaining caveats, raw output paths, and reproduction commands without broad framework claims. |
| AC-09 | Focused lint and tests pass, including importer policy tests, DHFR benchmark tests, comparison tests, and runtime-boundary tests. |

## Anti-Goals

- Do not optimize DHFR performance before the benchmark is runnable and semantically valid.
- Do not report a fake or partial DHFR speed ratio.
- Do not ignore AMBER terms just to make the benchmark run.
- Do not add heavyweight chemistry packages unless a concrete implementation slice proves they are necessary.
- Do not make OpenMM a product runtime dependency for `mlx_atomistic`.
- Do not broaden this change to non-DHFR systems.

## Blocking Questions Or Assumptions

- Assumption: using OpenMM in an explicit prep/reference script is acceptable because the existing project boundary already allows OpenMM under `scripts/` and tests.
- Assumption: “benchmark working” means both DHFR MLX rows should return `status: "ok"` with finite `ns_per_day` timing for short bounded runs, not merely better blocked diagnostics.
