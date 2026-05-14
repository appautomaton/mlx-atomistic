# SPEC: MLX Native Neighbor Pairs

## Bounded Goal

Replace the cutoff-MD neighbor rebuild path with an MLX/device-heavy tile or pair backend that materially reduces CPU rebuild work while preserving current neighbor-list physics and topology semantics across small and large systems.

## Selected Lenses

- engineering
- runtime

## Constraints

- Use `vendors/` only as reference material; do not import, vendor, build against, or copy vendor source into package code.
- Use OpenMM GPU nonbonded tile/block architecture as the primary blueprint and LAMMPS bin/stencil/half-list behavior as the correctness model.
- Preserve the current CPU `periodic_cell_list` route as an oracle and explicit fallback until the MLX backend passes correctness and performance gates.
- Do not reintroduce dense all-atom O(N^2) pair materialization for GPCRmd-scale lazy topologies.
- Keep topology filtering semantics intact: exclusions, explicit nonbonded exceptions, NBFIX, 1-4 scaling, cutoff, skin, periodic minimum image, and rebuild safety.
- Use `uv run ...` with the project environment for all validation.

## Required Behavior

- A new MLX-native or MLX-dominant neighbor backend is available for cutoff nonbonded MD and can be selected through the existing neighbor-manager/nonbonded runtime path.
- Supported cutoff systems use the new backend by default once it passes the acceptance gates; unsupported cases fail closed or explicitly fall back to the CPU oracle with metadata explaining why.
- The runtime report distinguishes `periodic_cell_list` from the new MLX backend and records pair/tile counts, cutoff, skin, rebuild count, and enough timing data to separate rebuild work from force evaluation.
- The backend should avoid requiring a giant flat `(i, j)` pair array if a tile/block representation is the more MLX-compatible shape.
- The GPCRmd notebook/report path continues to label electrostatics honestly as `short-range-prototype` unless production PME is separately implemented.

## Acceptance Criteria

- Small periodic systems produce the same neighbor pair set as the CPU `periodic_cell_list` oracle, ignoring pair order, or produce nonbonded energies and forces within existing tolerances where representation differs.
- Large lazy-topology systems remain blocked from dense all-pairs fallback and can run through the new backend or emit a precise unsupported-backend blocker.
- Existing nonbonded behavior with topology exclusions, explicit exceptions, NBFIX type-pair overrides, and 1-4 scaling remains covered by regression tests.
- A 10-step GPCRmd 729 short-range prototype proof run can use the new backend and records a runtime target of under 5 seconds on the current machine, or verification documents the measured bottleneck that prevents that target.
- Rebuild CPU work is visibly reduced versus the current threaded NumPy builder through a benchmark/report that separates neighbor rebuild time from force-evaluation time.
- Full test suite status is known before completion.

## Blocking Questions Or Assumptions

- Assumption: MLX may not expose a direct dynamic `nonzero(mask)` compaction primitive, so a tile/block interaction representation is acceptable if it preserves nonbonded semantics and improves runtime.
- Assumption: pair order is not part of the public contract; exact set or force/energy equivalence is the correctness bar.
- Assumption: under-5-second GPCRmd 10-step runtime is the stretch target, not a guarantee if MLX API limitations make compaction or scatter the true bottleneck.

## Anti-Goals

- Do not implement production PME in this change.
- Do not optimize unrelated bonded terms, CMAP, constraints, notebook rendering, or trajectory streaming unless required to verify the neighbor backend.
- Do not add custom Metal kernels unless a later spec explicitly expands scope.
- Do not remove the CPU neighbor-list implementation before it has served as an oracle and fallback for this change.
- Do not claim production-timescale GPCR dynamics from the short-range proof benchmark.
