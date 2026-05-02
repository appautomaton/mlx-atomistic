---
active_change: gpcrmd-real-artifact-md-run
stage: verify
---

# Status

## Current Change

- active change: `gpcrmd-real-artifact-md-run`
- current stage: `verify`

## What Is True Now

- The previous GPCRmd-critical readiness change completed its planned slices and left the runtime honestly blocked until real files were available.
- The selected GPCRmd 729 files are now downloaded locally under `notebooks/ligand-receptor-motion/data/gpcrmd-cache/729/` and ignored by git.
- A direct `run-gpcrmd-mlx` smoke attempt reached the real CHARMM import path and blocked before integration on `Could not find atom type for CT3`.
- Manual probing showed the PSF/PRM can be parsed when a PSF-derived MASS prelude is supplied, revealing Urey-Bradley, CMAP, and protocol box handling as the next concrete import requirements.
- This new change is framed as: real GPCRmd artifacts -> strict MLX artifact -> short MLX NVT trajectory -> notebook visualization/analysis.
- `PLAN.md` and `DESIGN.md` now define the ordered execution path: completed GPCRmd cache/protocol normalization, completed strict CHARMM term export, completed NBFIX export/runtime semantics, GPCRmd prepared artifact probe, lazy large-topology contract, scalable periodic nonbonded gate, PME readiness gate, short MLX proof run, notebook artifact consumption, and verification.
- Engineering review verdict is `approved_with_risks`; after the plan refresh, Slice 6 through Slice 8 should be re-reviewed before code execution because they define the large-system runtime contract.
- Slice 1 is complete: GPCRmd cache inspection now reports deterministic role paths, CHARMM import gets a PSF-derived MASS prelude, and protocol `input.xsc` boxes are parsed into import details and prepared `cell_lengths`.
- Slice 1 verification passed: focused GPCRmd/CHARMM pytest, focused Ruff, and real-cache `gpcrmd-import` now blocks on `charmm_cmap_terms`, `nbfix_pair_overrides`, and `urey_bradley_terms` instead of `CT3`.
- Slice 2 is complete: Urey-Bradley and CMAP terms are exported from parsed CHARMM structures, and CHARMM arrays cannot be hidden by metadata-only edits.
- Real GPCRmd 729 import now reaches a precise fail-closed blocker: `rejected_terms:nbfix_pair_overrides`.
- The blocked GPCRmd 729 report now includes prepared force-term counts: `charmm_cmap_terms=317`, `urey_bradley_terms=49223`, and `nbfix_pair_overrides=37`.
- Slice 3 is complete: GPCRmd 729 now exports and loads a strict prepared artifact with 37 compact NBFIX type-pair overrides and concrete converted sigma/epsilon values.
- Slice 4 is complete: NBFIX now applies inside `NonbondedPotential` as LJ sigma/epsilon substitution while preserving topology exclusions, explicit nonbonded exceptions, and PME/Ewald Coulomb separation.
- A no-dynamics GPCRmd build probe no longer rejects NBFIX. It now reaches the planned large-system topology blocker: dense nonbonded pair materialization for 92,001 atoms.
- Slice 5 is complete: real GPCRmd 729 exports a strict prepared artifact under `/tmp/mlx-atomistic-gpcrmd-729-prepared`, and `load_prepared_mlx_artifact(..., require_production=True)` loads it with `92001` atoms and `37` NBFIX type-pair overrides.
- Slice 6 is complete: large topologies now defer dense nonbonded pair arrays, keep compact exclusions/exceptions/1-4 metadata available, and report pair policy plus counts.
- Real GPCRmd 729 `MMSystem` build now reaches the planned lazy nonbonded pair-provider gate instead of failing in topology pair materialization.
- Slice 7 is complete: compact periodic neighbor pairs feed large-system real-space nonbonded evaluation, topology exclusions/exceptions filter those pairs without dense matrices, NBFIX works on neighbor-list pairs, and lazy topologies refuse dense/tiled fallback without a compact provider.
- Slice 8 is complete: production PME remains blocked on the current `numpy_reference` backend, while explicit `short-range-prototype` runs force cutoff electrostatics and write non-production metadata.
- The next proof run can proceed only as the explicit short-range prototype unless production PME is implemented first.
- Slice 9 is complete: a real GPCRmd 729 `mlx_atomistic` trajectory and run report exist under `notebooks/ligand-receptor-motion/data/gpcrmd-mlx/729-proof/`.
- The current artifact is a 10-step, 11-frame short-range prototype proof with `sample_interval=1`, `dt=0.0005`, `simulated_time_ps=0.005`, `elapsed_wall_seconds=13.944710792042315`, and `integration_steps_per_second=0.7171177767061757`.
- The original GPCRmd proof path was dominated by neighbor-list construction. The active route now uses periodic cell-list neighbors, vectorized pair construction, unsorted compact pairs, up to 8 NumPy worker threads, `GPCRMD_NEIGHBOR_SKIN=2.5`, scalar unit pair scales when 1-4 scaling is inactive, and sparse preview diagnostics.
- The current run report records `nonbonded_runtime.backend=periodic_cell_list`, `skin=2.5`, `rebuild_count=1`, and `pair_count=48933140`.
- Slice 10 is complete: the ligand-receptor notebook path consumes only MLX GPCRmd artifacts, displays blocker JSON when needed, uses bounded large-system viewer defaults, and labels the trajectory as a short-range prototype proof.
- Slice 11 is complete: Ruff passed for `src`, `tests`, and `notebooks/ligand-receptor-motion`; full pytest passed with `342 passed`.

## Next Step

Continue the MD efficiency pass from the optimized 10-step artifact. The highest-leverage remaining targets are topology filtering copies, CMAP force evaluation cost, diagnostic synchronization, streaming/strided trajectory persistence for longer previews, and a possible MLX-native or hybrid neighbor pair emitter.

## Open Risks

- The 92k-atom system must not fall back to dense all-pairs construction.
- Production PME for the real GPCRmd 729 artifact remains blocked on `pme_backend_not_production_executable:current_backend=numpy_reference`.
- The next proof trajectory must be explicitly labeled `short-range-prototype` if it runs before production PME exists.
- The notebook must present the trajectory as a short-range prototype proof, not production PME or long-timescale GPCR dynamics.
- The previous broad large-system runtime gate is now split into lazy topology, scalable periodic nonbonded, and PME readiness so each blocker is visible.
- The current PME implementation must not be treated as automatically ready for 92k-atom per-step production dynamics.
- CHARMM terms must not be silently dropped to make the run pass.
- The first successful trajectory is a short NVT proof, not a production-timescale GPCRmd result.
- Production PME remains unavailable until the `numpy_reference` PME path is replaced or validated as production-executable.
- The generated trajectory is a short-range prototype proof, not production PME or long-timescale GPCR dynamics.
