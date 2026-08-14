# Canonical MD Performance Suite

The canonical molecular-dynamics performance suite makes optimization decisions
from repeated, end-to-end prepared-system runs. It is separate from correctness
tests and isolated kernel microbenchmarks.

## Decision contract

The default `local` suite contains two required cases:

| Case | Atoms | Neighbor backend | Role |
| --- | ---: | --- | --- |
| `dhfr-5dfr-pme` | 23,558 | `mlx_cell_tiles` | Non-regression gate for fixed overhead, constraints, neighbor admission, and PME. |
| `jac-94k-pme` | 94,232 | `mlx_cell_tiles` | Improvement gate for production-scale direct space, PME, memory traffic, and scheduling. |

One case is deliberately insufficient. A candidate passes the default
comparison only when 5DFR does not regress by more than 3% and JAC improves by
at least 3%. Thresholds are command-line options, but both required cases must
remain present and comparable.

Every case runs three times by default after ten warmup steps. The persisted
metric is the median end-to-end seconds per measured step. A comparison is
blocked when the case contract, step counts, repeat count, or neighbor backend
does not match. It also fingerprints prepared artifact contents and requires
matching hardware and MLX runtime metadata, so differently prepared systems
cannot produce an eligible speedup ratio under the same case name.

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
  --warmup-steps 10 \
  --measured-steps 75 \
  --out results/md-suite/baseline.json

uv run python -m mlx_atomistic.benchmarks.md_suite run \
  --suite local \
  --repeats 3 \
  --warmup-steps 10 \
  --measured-steps 75 \
  --out results/md-suite/candidate.json

uv run python -m mlx_atomistic.benchmarks.md_suite compare \
  --baseline results/md-suite/baseline.json \
  --candidate results/md-suite/candidate.json \
  --out results/md-suite/comparison.json
```

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

The neighbor backend is part of each case contract. GPCRmd uses
`mlx_cell_pairs` because its CHARMM NBFIX path is not tile-owned; the other
release PME cases use `mlx_cell_tiles`. `--neighbor-backend` is an explicit
diagnostic override and changes comparison semantics.

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
