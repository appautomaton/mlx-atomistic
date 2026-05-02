# Slice 005 Summary: GPCRmd Prepared Artifact Unblock

## Status

- Implementer route: direct
- Result: completed
- Date: 2026-05-02T15:24:15Z

## Scope

Slice 5 was a no-code execution probe. It verified that the real GPCRmd 729 cache can export a strict prepared artifact after NBFIX support.

## Evidence

Command:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run atomistic-prep gpcrmd-import \
  --cache notebooks/ligand-receptor-motion/data/gpcrmd-cache/729 \
  --out /tmp/mlx-atomistic-gpcrmd-729-prepared \
  --json
```

Result:

- Exit code: 0
- Exported: true
- Output directory: `/tmp/mlx-atomistic-gpcrmd-729-prepared`
- Files written:
  - `prepared_system.json`
  - `prepared_system.npz`
  - `view.pdb`
  - `gpcrmd_import_report.json`

Strict loader probe:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python - <<'PY'
from pathlib import Path
from mlx_atomistic.artifacts import load_prepared_mlx_artifact
artifact = load_prepared_mlx_artifact(Path('/tmp/mlx-atomistic-gpcrmd-729-prepared'), require_production=True)
print('loaded', artifact.atom_count)
print('nbfix_type_pairs', artifact.arrays.get('nbfix_type_pairs').shape)
print('required_terms', len(artifact.metadata.get('compatibility_report', {}).get('required_terms', [])))
PY
```

Result:

```text
loaded 92001
nbfix_type_pairs (37, 2)
required_terms 16
```

## Artifact Facts

- Target: `gpcrmd-729-beta1-5f8u-cyanopindolol`
- Atom count: `92001`
- Hydrogens present: yes, `58952`
- Water present: yes
- Ions present: yes
- Lipids present: yes
- Ligand present: yes
- Receptor present: yes
- Periodic box present: yes
- Constraints present: yes, `78896`
- Nonbonded exceptions: `152516`
- NBFIX type-pair overrides: `37`
- Urey-Bradley terms: `49223`
- CMAP terms: `317`

## Current Blocker

Slice 5 itself is unblocked. The next planned blocker remains Slice 6: the runtime must build `MMSystem` for this 92,001 atom artifact without eager dense nonbonded pair materialization.

The import report still marks `runnable_now: false` because the runtime path has not yet implemented the large-system topology/nonbonded/PME gates covered by later slices.

## Next Slice

Proceed to Slice 6, `Lazy Large-Topology Contract`.
