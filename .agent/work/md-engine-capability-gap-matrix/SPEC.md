# SPEC: MD Engine Capability Gap Matrix

## Bounded Goal

Audit `mlx_atomistic` as a general MD simulation engine, classify each production
capability as implemented, partial, missing, or unverified against mature MD engine
expectations, and produce an ordered capability backlog.

## Broader Intent

Make `mlx_atomistic` more robust and production-ready as an MLX-native MD engine.
This spec preserves the full goal by defining the missing pieces before starting
PME, NPT, checkpoint, force-field, or performance implementation work.

## Work Scale And Shape

- scale: roadmap-sized objective framed as one audit/specification change
- shape: parity + audit

## Selected Lenses

- product
- engineering
- runtime

## Constraints And Risks

- Use repo evidence first: source, docs, tests, scripts, generated local evidence,
  and reference-only vendor trees.
- Treat OpenMM, GROMACS, and LAMMPS as comparison/reference systems, not runtime
  dependencies.
- Use `uv run ...` for runnable probes.
- Do not mark a capability "missing" when source shows a prototype or gated path;
  classify it as partial or unverified and record the evidence.
- Do not assume PME, CHARMM, GPCRmd, or DFT-era notes are current without checking
  live files.
- Keep generated trajectories, downloaded data, and large artifacts under ignored
  `results/` or external temporary paths.
- This change should produce decisions and evidence, not feature implementation.

## Required Outcome

Produce a gap matrix that covers these MD engine tracks:

1. **Core MD physics:** NVE/NVT/NPT, integrators, thermostats, constraints, PME,
   long-range corrections, pressure/virial, triclinic cells, virtual sites, water
   models.
2. **Force-field and artifact coverage:** AMBER, CHARMM, CMAP, NBFIX, 1-4
   exceptions, HMR policy, ligand/small-molecule parameters, production artifact
   gates.
3. **Runtime production usability:** reporters, checkpoint/restart, DCD/XTC output,
   long-run diagnostics, failure messages, reproducibility metadata.
4. **Validation and parity:** OpenMM parity targets, GROMACS reference behavior where
   useful, finite-difference force checks, stability checks, benchmark fixtures.
5. **Performance and backend:** neighbor-list scaling, PME FFT path, MLX/Metal
   kernel bottlenecks, OpenMM OpenCL comparison, GROMACS/LAMMPS implementation
   patterns worth studying.
6. **Prep and workflow:** raw topology/coordinate inputs, prepared artifacts,
   fail-closed metadata, notebook/script entrypoints.

Each capability row must include:

- status: `implemented`, `partial`, `missing`, or `unverified`
- evidence: file/function/test/doc path or runnable probe output
- production impact: blocker, important, or deferred
- next action: implement, validate, benchmark, document, or defer
- recommended order relative to PME, parity, NPT, usability, and performance tracks

## Acceptance Criteria

- The audit reads the live repo and does not rely only on previous chat summaries.
- The gap matrix distinguishes missing capabilities from partial/gated/unverified
  capabilities.
- At least one reference comparison is made for each major track using OpenMM,
  GROMACS, or LAMMPS source/docs where useful.
- The final backlog has an ordered first wave of implementation candidates and a
  separate deferred list.
- The previous narrow production-artifact baseline is retained as a possible evidence
  slice inside the matrix, not as the active goal.
- No source implementation files are changed.

## Linked Detail Files

None for framing. The implementation plan may create `GAP-MATRIX.md` or per-track
detail files if needed.

## Blocking Questions Or Assumptions

- Assumption: MD capability is the priority; DFT remains a separate roadmap/audit.
- Assumption: Apple Silicon remains the primary local runtime target.
- Assumption: OpenMM is the primary executable parity reference, while GROMACS and
  LAMMPS are mainly source/architecture references unless the plan explicitly adds a
  runnable reference command.

## Anti-Goals

- Do not implement PME, NPT/barostat, reporters, checkpointing, DCD/XTC, or kernel
  optimizations in this change.
- Do not build GROMACS or make GROMACS a runtime dependency.
- Do not broaden into DFT capability work.
- Do not create a marketing roadmap; every claim must point to code, docs, tests, or
  a runnable probe.
