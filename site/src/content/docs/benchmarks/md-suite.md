---
title: "Canonical MD Performance Suite"
---

The canonical molecular-dynamics performance suite makes optimization decisions
from repeated, end-to-end prepared-system runs. It is separate from correctness
tests and isolated kernel microbenchmarks.

## Local acceptance pair

| Case | Atoms | Neighbor backend | Role |
| --- | ---: | --- | --- |
| `dhfr-5dfr-pme` | 23,558 | `mlx_cell_tiles` | Non-regression gate for fixed overhead, constraints, neighbor admission, and PME. |
| `jac-94k-pme` | 94,232 | `mlx_cell_tiles` | Improvement gate for production-scale direct space, PME, memory traffic, and scheduling. |

One case is deliberately insufficient. The default comparison requires 5DFR
to remain within 3% of the baseline and JAC to improve by at least 3%. Each
case runs three times by default and reports the median end-to-end seconds per
measured step. Comparisons require matching case contracts, prepared artifact
contents, hardware metadata, MLX runtime, step counts, and neighbor backend.

```bash
uv run python -m mlx_atomistic.benchmarks.md_suite list

uv run python -m mlx_atomistic.benchmarks.md_suite run \
  --suite local \
  --out results/md-suite/current.json
```

The release registry additionally includes a JAC scaling ladder, deterministic
30,000- and 90,000-atom TIP3P water boxes, the official 92,224-atom OpenMM
ApoA1 PME workload, and GPCRmd 729.

Neighbor backend is a per-case contract. GPCRmd uses `mlx_cell_pairs` for its
CHARMM NBFIX path; the other release PME cases use `mlx_cell_tiles`.

Raw repeat payloads and prepared artifacts remain under gitignored `results/`.
The committed registry is
`src/mlx_atomistic/benchmarks/data/md_suite_cases.json`.

See the repository report `docs/benchmarks/md-suite.md` for preparation,
comparison, persistence, and correctness-gate details.
