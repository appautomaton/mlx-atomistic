# PLAN: MLX Native Neighbor Pairs

## Goal

Replace the cutoff-MD neighbor rebuild path with an MLX/device-heavy tile or pair backend that materially reduces CPU rebuild work while preserving current neighbor-list physics and topology semantics across small and large systems.

## Architecture Approach

Keep the existing CPU `periodic_cell_list` path as an oracle and fallback, then add a backend contract around neighbor rebuilds before introducing MLX-specific candidate generation. The implementation should prefer a bounded tile/block representation when MLX dynamic compaction cannot efficiently emit a giant flat `(i, j)` pair array. Dense all-pairs fallback remains forbidden for GPCRmd-scale lazy topologies.

`DESIGN.md` records the contract and decision boundaries. The main risk is representation shape: if the MLX path cannot produce compact pairs efficiently, force evaluation may need a tile adapter instead of only the current explicit-pairs API.

## Ordered Task Sequence

### Slice 1: Backend Contract And Capability Checkpoint

**Objective:** Add the neighbor-backend selection/reporting contract and prove the current CPU backend still behaves exactly as before.
**Execution:** direct
**Depends on:** none
**Touches:** `src/mlx_atomistic/neighbors.py`, `src/mlx_atomistic/md.py`, `tests/test_neighbors.py`, `tests/test_nonbonded_acceleration.py`
**Context budget:** ~10% of context window
**Produces:** Stable backend metadata and explicit backend routing for the existing `periodic_cell_list` path.
**Acceptance criteria:**
- Existing callers can continue to use `NeighborListManager(...)` without new arguments.
- CPU oracle reports `backend=periodic_cell_list`, cutoff, skin, rebuild count, pair count, and estimated memory.
- Unsupported backend names fail closed with a precise error.
- No dense all-pairs fallback is introduced for lazy topologies.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_neighbors.py tests/test_nonbonded_acceleration.py`
**Auto-continue:** yes

**Execution evidence:** Completed 2026-05-02. Added explicit validation/routing for the existing `periodic_cell_list` neighbor backend, preserved default caller behavior, extended runtime metadata with cell-list memory, and covered invalid backend names plus compact-backend report metadata in tests. Verification passed with `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_neighbors.py tests/test_nonbonded_acceleration.py` (`25 passed`) and `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/neighbors.py src/mlx_atomistic/md.py tests/test_neighbors.py tests/test_nonbonded_acceleration.py` (`All checks passed`).

### Slice 2: MLX Tile Candidate Prototype

**Objective:** Implement the first MLX-dominant periodic candidate backend for small systems and compare its pair set against the CPU oracle.
**Execution:** subagent recommended
**Depends on:** Slice 1
**Touches:** `src/mlx_atomistic/neighbors.py`, optional new neighbor backend module under `src/mlx_atomistic/`, `tests/test_neighbors.py`
**Context budget:** ~15% of context window
**Produces:** A selectable MLX neighbor backend that emits either compact unique pairs or a documented tile/candidate representation for small periodic systems.
**Acceptance criteria:**
- Small periodic systems match the CPU oracle pair set ignoring order, or document a representation-equivalence check when tiles are used.
- Periodic minimum image, cutoff, and skin behavior match the oracle.
- Pair/tile metadata records backend name, representation kind, pair or candidate count, and estimated memory.
- If MLX compaction is not viable, the backend returns a precise unsupported/fallback reason instead of pretending to be accelerated.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_neighbors.py -k "neighbor"`
**Auto-continue:** no

**Execution evidence:** Completed 2026-05-02 using the direct route because host subagent dispatch requires an explicit user request for multi-agent work. Added selectable `mlx_dense_pairs` backend for small periodic systems. The backend computes dense periodic distance candidates in MLX, emits compact unique pairs after CPU `argwhere` mask compaction, records `representation_kind=pairs`, candidate count, candidate memory, compaction backend, and `fallback_reason=mlx_argwhere_or_nonzero_unavailable`, and fails closed above the configured small-system atom limit. Checkpoint decision: proceed with compact pairs for this prototype and do not change default runtime selection yet. Verification passed with `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_neighbors.py -k "neighbor"` (`11 passed`), targeted Ruff (`All checks passed`), and the broader neighbor/nonbonded regression command (`28 passed`).

### Slice 3: Nonbonded Semantics Integration

**Objective:** Wire the new backend through cutoff nonbonded MD while preserving topology filtering and pair overrides.
**Execution:** subagent recommended
**Depends on:** Slice 2
**Touches:** `src/mlx_atomistic/forcefields.py`, `src/mlx_atomistic/nonbonded.py`, `src/mlx_atomistic/md.py`, `tests/test_nonbonded_acceleration.py`
**Context budget:** ~15% of context window
**Produces:** New neighbor backend can drive nonbonded force evaluation for supported cutoff systems, with CPU oracle/fallback still available.
**Acceptance criteria:**
- Exclusions, explicit exceptions, NBFIX, and 1-4 LJ/Coulomb scaling remain covered by regression tests.
- Lazy topologies still refuse dense/tiled all-pairs fallback when a runtime neighbor provider is required.
- Energies and forces match existing tolerances for small systems.
- Unsupported representation or physics combinations fail closed with reportable metadata.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_nonbonded_acceleration.py`
**Auto-continue:** yes

**Execution evidence:** Completed 2026-05-02 using the direct route because host subagent dispatch requires an explicit user request for multi-agent work and the compact-pair integration reused the existing explicit-pairs force path. Extended nonbonded runtime reports with representation, candidate, compaction, and fallback metadata. Added regression coverage showing `mlx_dense_pairs` preserves lazy topology exclusions, explicit exceptions, NBFIX type-pair overrides, and 1-4 scaling against the CPU compact-pair oracle, and that `NeighborListManager(..., backend="mlx_dense_pairs")` drives `simulate_nvt` without dense fallback. Verification passed with `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_nonbonded_acceleration.py` (`19 passed`), targeted Ruff (`All checks passed`), and `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_neighbors.py tests/test_nonbonded_acceleration.py` (`30 passed`).

### Slice 4: Runtime Selection And Reporting

**Objective:** Make supported cutoff systems select the new backend by default while exposing clear fallback metadata and separated timings.
**Execution:** subagent recommended
**Depends on:** Slice 3
**Touches:** `src/mlx_atomistic/neighbors.py`, `src/mlx_atomistic/md.py`, `src/mlx_atomistic/prep/runner.py`, `src/mlx_atomistic/benchmarks/md_acceleration.py`, tests covering runtime reports
**Context budget:** ~15% of context window
**Produces:** Runtime reports distinguish `periodic_cell_list` from the MLX backend and separate rebuild time from force-evaluation time.
**Acceptance criteria:**
- Supported cutoff runs use the MLX backend by default, or report an explicit fallback reason.
- GPCRmd setup no longer hardcodes CPU neighbor construction when the MLX backend is supported.
- Reports include backend, representation, cutoff, skin, rebuild count, pair/tile count, rebuild timing, force timing, and fallback/blocker metadata.
- The GPCRmd electrostatics label remains `short-range-prototype`.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_mlx_prep.py tests/test_nonbonded_acceleration.py`
**Auto-continue:** yes

**Execution evidence:** Completed 2026-05-02 using the direct route because host subagent dispatch requires an explicit user request for multi-agent work. `NeighborListManager` now defaults to an `auto` policy that selects `mlx_dense_pairs` for supported small systems and falls back to `periodic_cell_list` with explicit fallback metadata when the small-system limit is exceeded. Runtime reports now include backend representation, candidate counts, candidate memory, compaction backend, fallback reason, neighbor update/rebuild wall time, and force-evaluation wall time. GPCRmd production neighbor-manager construction now inherits the auto backend policy instead of pinning CPU construction. Verification passed with `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_mlx_prep.py tests/test_nonbonded_acceleration.py` (`48 passed`), targeted Ruff (`All checks passed`), and focused neighbor/NVE/NVT/nonbonded regression (`41 passed`).

### Slice 5: Benchmark Evidence And GPCRmd Proof

**Objective:** Produce benchmark evidence showing reduced CPU rebuild work and run or block the GPCRmd 729 10-step short-range proof.
**Execution:** direct
**Depends on:** Slice 4
**Touches:** `src/mlx_atomistic/benchmarks/md_acceleration.py`, `docs/md-acceleration.md`, `notebooks/ligand-receptor-motion/data/gpcrmd-mlx/729-proof/`
**Context budget:** ~10% of context window
**Produces:** JSON/report evidence for rebuild timing, force timing, backend selection, and GPCRmd proof runtime or precise blocker.
**Acceptance criteria:**
- Benchmark separates neighbor rebuild work from force evaluation.
- New backend shows visibly reduced CPU rebuild work versus `periodic_cell_list`, or documents the measured bottleneck.
- GPCRmd 729 10-step proof uses the new backend and records under-5-second runtime, or documents why the target remains blocked.
- Report still states the proof is short-range, not production PME.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -m mlx_atomistic.benchmarks.md_acceleration --sizes 128,512 --evaluations 2 --json`
**Auto-continue:** no

**Execution evidence:** Completed 2026-05-02. `md_acceleration` now reports `neighbor_rebuild_ms_per_eval` separately from `force_eval_ms_per_eval`, and `docs/md-acceleration.md` records the current `mlx_dense_pairs` small-system policy plus GPCRmd-scale fallback. The required benchmark command passed. At 128 particles, `python_neighbor` reported about `1.09 ms` rebuild and `2.04 ms` force evaluation per eval, while `mlx_pairs` reported about `0.51 ms` one-time rebuild amortized per eval and `1.05 ms` force evaluation per eval; at 512 particles, `python_neighbor` reported about `8.39 ms` rebuild and `1.60 ms` force evaluation per eval, while `mlx_pairs` reported about `4.10 ms` one-time rebuild amortized per eval and `1.21 ms` force evaluation per eval. A fresh 10-step GPCRmd 729 short-range proof run completed under `/tmp/mlx-atomistic-gpcrmd-729-slice5` with `status=ran`, `elapsed_wall_seconds=24.147015874972567`, `integration_steps_per_second=0.4141298474220414`, `electrostatics_route=short-range-prototype`, `nonbonded_runtime.backend=periodic_cell_list`, `fallback_reason=mlx_dense_pairs_atom_limit_exceeded:n_atoms=92001:max_mlx_dense_atoms=4096`, `neighbor_rebuild_wall_seconds=2.80373049993068`, `force_evaluation_wall_seconds=11.339769957237877`, `pair_count=48933140`, and `rebuild_count=1`. The under-5-second target remains blocked by the small-system MLX pair emitter limit and remaining force-evaluation cost. Verification also passed with `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_nonbonded_acceleration.py` (`20 passed`) and targeted Ruff (`All checks passed`).

### Slice 6: Final Regression Gate

**Objective:** Verify the completed change against source tests, lint, and full-suite status.
**Execution:** direct
**Depends on:** Slice 5
**Touches:** test and lint surfaces only
**Context budget:** ~8% of context window
**Produces:** Final verification status and any remaining known blockers.
**Acceptance criteria:**
- Full pytest status is known.
- Source/test/script Ruff status is known.
- Any full-repo Ruff notebook failures are reported separately from source health.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest && UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts`
**Auto-continue:** no

**Execution evidence:** Completed 2026-05-02. Full-suite verification passed with `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest` (`350 passed in 34.99s`). Source/test/script lint passed with `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts` (`All checks passed`). Full-repo Ruff was also checked separately with `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check .` and remains blocked only by notebook lint findings: 53 errors across archived/workflow notebooks, mainly import sorting, unused imports, and line-length issues.

## Execution Routing And Topology

- Direct slices: 1, 5, 6.
- Subagent recommended slices: 2, 3, 4, because they cross backend representation, nonbonded semantics, and runtime-reporting boundaries.
- Subagent required slices: none.
- Auto-continue chain: Slice 1 -> Slice 2; Slice 3 -> Slice 4 -> Slice 5 after each verification passes.
- Checkpoints: after Slice 2, decide whether the implementation proceeds as compact pairs or tile/block representation; after Slice 5, decide whether performance target is met or bottleneck documentation is the accepted outcome.
- Parallel-safe groups: none. The write sets overlap through `neighbors.py`, `nonbonded.py`, `forcefields.py`, and runtime reports.

## Verification Commands

- Slice 1: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_neighbors.py tests/test_nonbonded_acceleration.py`
- Slice 2: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_neighbors.py -k "neighbor"`
- Slice 3: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_nonbonded_acceleration.py`
- Slice 4: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_mlx_prep.py tests/test_nonbonded_acceleration.py`
- Slice 5: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -m mlx_atomistic.benchmarks.md_acceleration --sizes 128,512 --evaluations 2 --json`
- Slice 6: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest && UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts`

## Context Budget For This Change

Estimated total execution context: ~73% of a context window across all slices.

Expected loading pattern:

- Always load `SPEC.md`, `PLAN.md`, and `DESIGN.md`.
- Load `neighbors.py` first for Slices 1-2.
- Load `forcefields.py`, `nonbonded.py`, and `md.py` only for Slices 3-4.
- Load `mlx_atomistic.prep/runner.py`, benchmark code, docs, and GPCRmd proof artifacts only for Slices 4-5.

## Recommended Next Skill

Run `auto-verify` to check the completed change against `PLAN.md` and user-visible outcomes.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan preserves the CPU oracle and fail-closed lazy-topology behavior while sequencing backend contract, MLX candidate generation, force integration, runtime reporting, and benchmark evidence in a testable order.
- Concern: Slice 2 may force a material representation decision because compact MLX pair emission may be unavailable or slower than a tile/block path that requires force-evaluation adaptation.
- Action: Run `auto-execute` starting with Slice 1 and treat the Slice 2 compact-pairs-versus-tiles checkpoint as mandatory before changing default runtime selection.
- Verified: state pointers, PLAN.md, DESIGN.md, current neighbor/nonbonded source surfaces, execution topology, verification commands, and engineering risk matrix checked.
