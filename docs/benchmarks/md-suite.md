# Canonical MD Performance Suite

The canonical molecular-dynamics performance suite makes optimization decisions
from repeated, end-to-end prepared-system runs. It is separate from correctness
tests and isolated kernel microbenchmarks.

## Decision contract

The default `local` suite contains two required cases:

| Case | Atoms | Neighbor backend | Role |
| --- | ---: | --- | --- |
| `dhfr-5dfr-pme` | 23,558 | `mlx_interaction32` | Non-regression gate for fixed overhead, constraints, neighbor admission, and PME. |
| `jac-94k-pme` | 94,232 | `mlx_interaction32` | Improvement gate for production-scale direct space, PME, memory traffic, and scheduling. |

One case is deliberately insufficient. A candidate passes the default
comparison only when 5DFR does not regress by more than 3% and JAC improves by
at least 3%. Thresholds are command-line options, but both required cases must
remain present and comparable.

Every case first runs an unmeasured 75-step rehearsal. This reaches an ordinary
neighbor rebuild on the local contracts and warms the recurring Metal shapes
before three recorded 750-step repeats, each with ten warmup steps. The longer
window covers recurring Neighbor generations rather than one favorable phase of
the Verlet lifecycle. The persisted metric is the median end-to-end seconds per
measured step. A case is blocked when its recorded timing range exceeds 10% of
the median. This prevents cold compilation, thermal transitions, or frequency
drift from becoming an eligible speedup.

The 750-step default replaced the former 75-step measurement after four
position-balanced JAC arms each produced about 40% spread at the same repeat
position and covered only one measured rebuild. Contemporaneous 750-step arms
covered 19--41 rebuilds across the large-system cases and stayed below 5.1%
spread. The 75-step window remains appropriate for intrusive structural
profiles, but not for sustained throughput claims.

A comparison is blocked when the case contract, rehearsal and measured step
counts, repeat count, spread limit, or neighbor backend does not match. It also
fingerprints prepared artifact contents and requires matching hardware and MLX
runtime metadata, so differently prepared systems cannot produce an eligible
speedup ratio under the same case name.

## Commands

List registered cases and local artifact availability:

```bash
uv run python -m mlx_atomistic.benchmarks.md_suite list
```

Run the baseline and candidate from their respective commits or worktrees:

```bash
uv run python -m mlx_atomistic.benchmarks.md_suite run \
  --suite local \
  --repeats 3 \
  --rehearsal-steps 75 \
  --warmup-steps 10 \
  --measured-steps 750 \
  --out results/md-suite/baseline.json

uv run python -m mlx_atomistic.benchmarks.md_suite run \
  --suite local \
  --repeats 3 \
  --rehearsal-steps 75 \
  --warmup-steps 10 \
  --measured-steps 750 \
  --out results/md-suite/candidate.json

uv run python -m mlx_atomistic.benchmarks.md_suite compare \
  --baseline results/md-suite/baseline.json \
  --candidate results/md-suite/candidate.json \
  --out results/md-suite/comparison.json
```

Evaluate the previous tile route as an explicit control without changing the
committed production case contracts:

```bash
uv run python -m mlx_atomistic.benchmarks.md_suite run \
  --suite local \
  --neighbor-backend mlx_cell_tiles \
  --repeats 3 \
  --rehearsal-steps 75 \
  --warmup-steps 10 \
  --measured-steps 750 \
  --out results/md-suite/legacy-tiles.json
```

Attribute non-overlapping synchronized stages inside its measured rebuilds:

```bash
uv run python -m mlx_atomistic.benchmarks.charged_pme runtime \
  --prepared results/dhfr-npt-closure/prepared \
  --neighbor-backend mlx_interaction32 \
  --neighbor-rebuild-profile \
  --warmups 10 \
  --steps 750 \
  --out results/md-suite/interaction32-rebuild-profile.json
```

This mode deliberately synchronizes geometry, topology preparation, special
inventory, count/prefix, admission, scatter, and completion boundaries. Use it
for attribution only; run the same workload without the flag for throughput.

Build a whole-step performance map before choosing an optimization target:

```bash
uv run python -m mlx_atomistic.benchmarks.md_suite profile \
  --suite local \
  --warmup-steps 10 \
  --measured-steps 75 \
  --out results/md-suite/stage-profile.json
```

The stage profile runs one clean end-to-end control and one synchronized,
instrumented sample for every selected case. It groups the instrumented routes
into neighbor lifecycle, direct nonbonded, reciprocal PME, constraints,
bonded/other forces, integration, force aggregation, sparse corrections, and
diagnostic work. `cross_case_stage_ranking` ranks the structural shares across
all successful cases.

Only the clean sample is a throughput measurement. The instrumented sample
adds completion barriers so stages own non-overlapping work. It preserves the
production constraint implementation, including the dense composite Metal
route, but it cannot preserve the ordinary lazy force schedule or asynchronous
overlap. Its fractions are therefore structural attribution inside the
instrumented wall, not shares of the clean wall.

The final clean and instrumented states are compared as a diagnostic only.
Added synchronization can change floating-point reduction order, and long
chaotic trajectories can separate while both runs remain individually valid.
Each raw runtime must still pass its own finite-state, constraint, topology,
neighbor, and execution checks.

Clean charged-PME payloads also record `main_thread_cpu_seconds` and
`process_cpu_seconds` for the measured interval, plus each value divided by
wall time. The main-thread clock includes Python and synchronous MLX host work.
The process clock additionally includes MLX runtime and driver worker threads.
Both clocks exclude blocked CPU time and can overlap Metal execution, so they
are host-activity upper bounds rather than additive stage timings or predicted
C++ speedups.

Keep power mode, thermal state, neighbor backend, and reporting cadence fixed.
Low Power Mode is valid for a local relative comparison when it remains
unchanged for both runs. Its absolute throughput must not be compared with a
normal-power result.

## Extended cases

The committed registry also defines:

- a 23,558/47,116/94,232-atom JAC scaling ladder;
- deterministic 30,000- and 90,000-atom TIP3P water boxes;
- the official 92,224-atom OpenMM ApoA1 PME workload;
- the 92,001-atom GPCRmd 729 CHARMM workload.

The neighbor backend is part of each case contract. All release PME cases use
the promoted `mlx_interaction32` device-built 32-atom force schedule, including
the CHARMM type-pair NBFIX specialization used by GPCRmd. It retains capacity
across Neighbor generations and creates production tiles lazily only at energy
or virial diagnostic boundaries. The 47,116-atom JAC scaling midpoint remains
on `mlx_cell_tiles` because its low-power promotion measurement was unstable;
the 23,558- and 94,232-atom endpoints use `mlx_interaction32`.
`--neighbor-backend` is an explicit diagnostic override and changes comparison
semantics.

Generate the deterministic water artifacts without a reference engine:

```bash
uv run python -m mlx_atomistic.benchmarks.tip3p_water \
  --preset 30k \
  --out results/md-benchmarks/tip3p-water-30k/prepared

uv run python -m mlx_atomistic.benchmarks.tip3p_water \
  --preset 90k \
  --out results/md-benchmarks/tip3p-water-90k/prepared
```

Prepare ApoA1 through the reference-only OpenMM construction boundary:

```bash
uv run --with openmm python scripts/prepare_openmm_dhfr_explicit.py \
  --case apoa1 \
  --json
```

OpenMM is not used after preparation. The saved artifact is consumed only by
the MLX runtime.

## Persistence

The case registry is committed at
`src/mlx_atomistic/benchmarks/data/md_suite_cases.json`. Raw repeat payloads,
prepared artifacts, and comparisons stay under gitignored `results/`.
Committed benchmark reports summarize those raw results and record the exact
command, commit, hardware, power mode, and protocol.

Performance retention never replaces correctness gates. Energy and complete
force parity, neighbor completeness, constraint residuals, and trajectory
stability must be validated separately before a candidate is retained.
