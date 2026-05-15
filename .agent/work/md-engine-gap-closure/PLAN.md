# PLAN: MD Engine Gap Closure

## Goal

Close the important `mlx_atomistic` production-MD engine gaps through a gated
sequence: first build an OpenMM-vs-MLX parity fixture, then use that fixture to
make PME, observability/restart, HMR or virtual-site policy, NPT, production
trajectory output, and performance work evidence-driven.

## Architecture Approach

Use a phase-gated capability plan, not a rewrite. The prepared artifact remains
the product boundary. OpenMM is the executable reference. GROMACS and LAMMPS
remain source/design references only.

The first implementation slice creates the shared fixture and tolerance harness.
Every later feature must prove itself against that fixture family or explicitly
record why a different fixture is required. Large generated trajectories,
reports, and benchmark outputs stay under ignored `results/` paths. No GROMACS
build/runtime dependency and no DFT scope are part of this plan.

No separate `DESIGN.md` is created for this plan stage. Architecture decisions
that require source-level design are deferred to the slice that first changes
that subsystem, with the parity fixture as the gate.

## Requirement Traceability

| Spec phase | Matrix rows | Plan slice |
| --- | --- | --- |
| Production artifact parity fixture | `T2-02`, `T2-03`, `T2-06`, `T4-04`, `T6-02`, `T6-07` | Slice 1 |
| PME production readiness | `T1-05`, `T1-06`, `T1-08`, `T4-04`, `T5-03` | Slice 2 |
| Runtime observability | `T3-02`, `T3-05`, `T4-05`, `T5-04` | Slice 3 |
| Checkpoint/restart boundary | `T3-03`, `T3-07`, `T4-05` | Slice 4 |
| Constraints, HMR, virtual-site policy | `T1-07`, `T1-11`, `T2-07`, `T6-05` | Slice 5 |
| NPT/barostat | `T1-04`, `T1-08`, `T2-01`, `T3-02` | Slice 6 |
| Production DCD/XTC output | `T3-04`, `T3-02`, `T6-06` | Slice 7 |
| Performance pass | `T5-01`, `T5-02`, `T5-04`, `T5-05`, `T5-06`, `T5-07` | Slice 8 |

Deferred rows stay out of scope for this wave: `T1-03`, `T1-10`, `T1-13`,
`T2-09`, `T4-06`, `T5-08`, `T6-03`, plus free energy, REMD, metadynamics,
GBSA, and polarizable-force-field expansion.

## Ordered Task Sequence

### Slice 1: Production Artifact Parity Fixture

**Objective:** Select one small real AMBER or CHARMM prepared artifact and build
a repeatable OpenMM-vs-MLX force/energy parity harness around it.
**Execution:** direct
**Depends on:** none
**Touches:** `src/mlx_atomistic/prep/`, `src/mlx_atomistic/validation.py` or a
new validation helper, `tests/`, `scripts/`, `results/md-engine-gap-closure/`
**Context budget:** ~15% of context window
**Produces:** selected fixture record, parity harness, focused tests, and a
local parity result under `results/md-engine-gap-closure/parity-fixture/`
**Acceptance criteria:**
- A selected fixture loads through the production artifact path or records an
  exact metadata blocker.
- MLX terms and an OpenMM reference are built from the same topology/parameter
  source.
- Total energy, component energies where mappable, and force arrays are compared
  at the same coordinates with explicit tolerances.
- Unsupported terms are reported exactly and fail closed.
**Verification:**
`UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_openmm_mlx_parity.py`
`UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/run_openmm_mlx_parity.py --fixture <selected-fixture> --out results/md-engine-gap-closure/parity-fixture`
**Checkpoint after:** decision
**Checkpoint reason:** If parity fails before PME, the next work is artifact
conversion/term mapping, not PME. If it passes, Slice 2 can start.
**Detail:** `slices/slice-001-production-artifact-parity-fixture.md`

### Slice 2: PME Production Readiness

**Objective:** Make PME production-executable for the selected fixture family
and prove energy/force parity against OpenMM PME.
**Execution:** subagent recommended
**Depends on:** Slice 1 parity outcome
**Touches:** `src/mlx_atomistic/pme.py`, nonbonded/virial integration points,
PME tests, parity harness outputs
**Context budget:** ~15% of context window
**Produces:** green `pme_readiness_report` semantics for the fixture, PME parity
tests, and updated local PME parity results
**Acceptance criteria:**
- `pme_readiness_report` is green only for configurations that the runtime can
  execute without NumPy-reference production fallback.
- OpenMM PME total energy and forces pass explicit tolerances on the selected
  fixture family.
- PME virial contributions needed by later pressure work are either validated or
  explicitly blocked.
**Verification:**
`UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_pme.py tests/test_openmm_mlx_parity.py`
`UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/run_openmm_mlx_parity.py --fixture <selected-fixture> --pme --out results/md-engine-gap-closure/pme-parity`
**Checkpoint after:** decision
**Checkpoint reason:** NPT and performance work depend on whether PME and virial
paths are credible.
**Detail:** none

### Slice 3: Runtime Reporters And Diagnostics

**Objective:** Add a minimal step/interval reporter surface for frames, scalar
state data, diagnostics, and parity traces without changing physics.
**Execution:** subagent recommended
**Depends on:** Slice 1; may proceed after Slice 2 if PME trace fields are needed
**Touches:** `src/mlx_atomistic/md.py`, `src/mlx_atomistic/prep/runner.py`,
`src/mlx_atomistic/io.py`, reporter tests
**Context budget:** ~10% of context window
**Produces:** reporter/callback API used by NVT runs and `prep.run_mlx`
**Acceptance criteria:**
- Reporters can observe sampled positions, energies, temperatures, diagnostics,
  and step/time without altering integration results.
- Existing NPZ output still works.
- Parity traces can be produced without ad hoc runner edits.
**Verification:**
`UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_mlx_prep.py tests/test_diagnostics.py tests/test_runtime_reporters.py`
**Checkpoint after:** none
**Checkpoint reason:** none
**Detail:** none

### Slice 4: Checkpoint/Restart Boundary

**Objective:** Serialize and restore the state needed for long runs at the
runner level.
**Execution:** subagent recommended
**Depends on:** Slice 3
**Touches:** `src/mlx_atomistic/io.py`, `src/mlx_atomistic/md.py`,
`src/mlx_atomistic/prep/runner.py`, checkpoint tests
**Context budget:** ~12% of context window
**Produces:** checkpoint schema/API and restart/resume tests
**Acceptance criteria:**
- Checkpoints include positions, velocities, cell, step/time,
  thermostat/RNG state or deterministic seed/cursor, neighbor-list policy,
  force-term metadata, and diagnostic cursor.
- A stopped run can resume and produce the same sampled trajectory as a
  continuous run under the documented determinism contract.
**Verification:**
`UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_checkpoint_restart.py tests/test_mlx_prep.py`
**Checkpoint after:** none
**Checkpoint reason:** none
**Detail:** none

### Slice 5: Constraints, HMR, And Virtual-Site Policy

**Objective:** Make common 2-4 fs production workflow support explicit by either
implementing required HMR/virtual-site support or rejecting artifacts with exact
metadata-driven blockers.
**Execution:** subagent recommended
**Depends on:** Slice 1; Slice 2 if PME fixtures expose relevant water/ion terms
**Touches:** artifact validation, prep schema/import checks, constraint tests,
GPCRmd blockers
**Context budget:** ~12% of context window
**Produces:** validated constraint/HMR/virtual-site policy and tests
**Acceptance criteria:**
- Existing distance constraints remain stable under focused long-run tests.
- HMR is detected and either represented correctly or rejected with an exact
  blocker.
- Virtual-site/TIP4P-style artifacts fail closed unless full support lands.
**Verification:**
`UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_protocols.py tests/test_production_artifacts.py tests/test_constraints.py tests/test_gpcrmd_registry.py`
**Checkpoint after:** decision
**Checkpoint reason:** If virtual-site implementation is required rather than
rejection-only policy, it should be framed as its own implementation slice.
**Detail:** none

### Slice 6: NPT / Barostat

**Objective:** Add pressure coupling after PME and virial diagnostics are
credible, starting with Monte Carlo barostat unless fresh evidence says
otherwise.
**Execution:** subagent recommended
**Depends on:** Slice 2 and Slice 5
**Touches:** `src/mlx_atomistic/md.py`, `src/mlx_atomistic/protocols.py`,
barostat support, virial/pressure tests
**Context budget:** ~15% of context window
**Produces:** NPT runner/protocol path and OpenMM density/volume comparison on a
small fixture
**Acceptance criteria:**
- NPT requests no longer fail closed for supported orthorhombic PME fixtures.
- Volume scaling respects constraints and metadata gates.
- Density/volume behavior is compared with OpenMM under explicit statistical
  acceptance bounds.
**Verification:**
`UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_virial_pressure.py tests/test_protocols.py tests/test_npt.py`
`UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/run_openmm_mlx_npt_parity.py --fixture <selected-fixture> --out results/md-engine-gap-closure/npt-parity`
**Checkpoint after:** decision
**Checkpoint reason:** If density/volume parity fails, performance and output
work should wait until the physical behavior is understood.
**Detail:** none

### Slice 7: Production DCD/XTC Output

**Objective:** Expose DCD/XTC as first-class runner outputs through the reporter
surface while keeping NPZ as the native diagnostic format.
**Execution:** direct
**Depends on:** Slice 3
**Touches:** `src/mlx_atomistic/io.py`, `src/mlx_atomistic/trajectory_adapters.py`,
`src/mlx_atomistic/prep/runner.py`, output tests
**Context budget:** ~8% of context window
**Produces:** runner-level DCD/XTC output option and round-trip/readability tests
**Acceptance criteria:**
- Runner output accepts native NPZ plus DCD and/or XTC where optional writer
  dependencies are available.
- Missing optional writer dependencies produce explicit, actionable errors.
- Output is readable by MDAnalysis/MDTraj in tests.
**Verification:**
`UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_trajectory_adapters.py tests/test_runner_outputs.py`
**Checkpoint after:** none
**Checkpoint reason:** none
**Detail:** none

### Slice 8: Performance Hot-Path Profile

**Objective:** Profile the parity fixture after correctness gates and choose
measured optimization targets for neighbor, pair, PME FFT, or MLX/Metal paths.
**Execution:** direct for profiling; subagent recommended only if a measured
hot-path optimization is approved
**Depends on:** Slice 2; Slice 6 if NPT throughput is the target
**Touches:** benchmarks, scripts, optional source hot paths only after profiling
**Context budget:** ~10% of context window
**Produces:** local profile report and either a bounded optimization patch or a
new focused optimization SPEC
**Acceptance criteria:**
- Throughput is measured against OpenMM OpenCL on the same fixture/config family.
- Hot paths are ranked from profiler evidence.
- No custom Metal/MLX kernel work is started without measured justification.
**Verification:**
`UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -m mlx_atomistic.benchmarks.md_performance --json`
`UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_openmm_opencl.py --help`
**Checkpoint after:** decision
**Checkpoint reason:** Optimization scope depends on measured hotspot and may
need a narrower implementation plan.
**Detail:** none

## Execution Routing And Topology

Default continuation path is serial: Slice 1 -> Slice 2 -> Slice 3 -> Slice 4
-> Slice 5 -> Slice 6 -> Slice 7 -> Slice 8.

Explicit decision checkpoints:

- After Slice 1: decide whether to fix artifact conversion/term mapping or start
  PME readiness.
- After Slice 2: decide whether PME/virial evidence is strong enough for NPT.
- After Slice 5: decide whether virtual-site support is required now or exact
  rejection is sufficient.
- After Slice 6: decide whether physical behavior is credible enough for output
  polish and performance.
- After Slice 8: choose a measured optimization target or create a focused
  optimization spec.

Parallel-safe groups: none for the first execution wave. Later, Slice 7 can run
in parallel with Slice 6 only after Slice 3 lands and if write sets stay limited
to output surfaces.

## Verification Commands

Plan-stage verification:

`test -f .agent/work/md-engine-gap-closure/PLAN.md`

`rg -n "T1-05|T1-04|T3-02|T4-04|T6-02|Production Artifact Parity Fixture|PME Production Readiness|NPT / Barostat" .agent/work/md-engine-gap-closure/PLAN.md`

Material slice verification commands are listed inside each slice above.

## Context Budget For This Change

Total estimated context consumption across the full gap-closure wave is ~97% of
a large context window, so execution should proceed one slice at a time and
reload only the slice detail plus touched source files. Slice 1 is the next
executable unit.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan correctly makes a real OpenMM-vs-MLX parity fixture the first executable gate before PME, NPT, checkpointing, or performance work.
- Concern: Slice 1 is executable but crosses fixture selection, OpenMM reference construction, MLX term mapping, tests, scripts, and generated results, so implementation must stay narrowly scoped and stop at the parity decision checkpoint.
- Action: Run `auto-execute` for Slice 1 and stop after the parity fixture result if artifact conversion or term mapping fails before PME.
- Verified: current state, STATUS.md, canonical PLAN.md, missing DESIGN.md boundary, slice dependency order, gap-ID traceability, verification commands, and checkpoint routing were checked.

## Execution Evidence

### Slice 1: Production Artifact Parity Fixture

- Status: complete at decision checkpoint.
- Route: direct.
- Selected fixture: `amber-alanine-dipeptide-implicit`.
- Local report: `results/md-engine-gap-closure/parity-fixture/openmm_mlx_parity_report.json`.
- Evidence: `orchestration/slice-001-production-artifact-parity-fixture.md`.
- Decision: parity passes for the small AMBER fixture, so Slice 2 can start.
