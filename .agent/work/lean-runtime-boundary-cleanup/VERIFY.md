# VERIFY: Lean Runtime Boundary Cleanup

## Summary

The cleanup keeps `mlx_atomistic` as the product runtime and labels OpenMM,
LAMMPS, and `vendors/` as reference or validation surfaces. No runtime physics
or engine behavior was changed.

## Files Changed

- `README.md`: added the central runtime-boundary pointer and clarified source layout.
- `docs/runtime-boundaries.md`: added the authoritative runtime-boundary document.
- `pyproject.toml`: documented dev/reference engine roles while keeping OpenMM and LAMMPS outside core dependencies.
- `notebooks/README.md`: labels active MLX output as `mlx_atomistic` and OpenMM workflows as `openmm-reference`.
- `notebooks/ligand-receptor-motion/README.md`: labels OpenMM artifacts as `openmm-reference` and not production runtime output.
- `scripts/run_openmm_gpcrmd_preview.py`: adds `artifact_label="openmm-reference"` to generated metadata and reports.
- `scripts/run_openmm_gpcrmd_charmm_md.py`: adds `artifact_label="openmm-reference"` to generated metadata and reports.
- `.gitignore`: clarifies that generated MLX and OpenMM reference outputs are ignored.
- `tests/test_runtime_boundaries.py`: adds scan-based guardrails for external-engine imports and dependency boundaries.

## Slice Evidence

### Slice 1

- `rg -n "primary trajectory generator|reference|OpenMM|LAMMPS|vendors" README.md docs/runtime-boundaries.md pyproject.toml` passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -c 'import tomllib; ...'` passed after an approved unsandboxed rerun. The first sandboxed attempt failed while fetching `hatchling` for local package rebuild.

### Slice 2

- `rg -n "mlx_atomistic|openmm-reference|reference preview|generated.*ignored|not production" notebooks/README.md notebooks/ligand-receptor-motion/README.md scripts/run_openmm_gpcrmd_preview.py scripts/run_openmm_gpcrmd_charmm_md.py .gitignore` passed.
- `git check-ignore notebooks/ligand-receptor-motion/data/openmm-md/example/trajectory.npz notebooks/ligand-receptor-motion/data/openmm-preview/example/trajectory.npz notebooks/ligand-receptor-motion/data/gpcrmd-mlx/example/trajectory.npz` passed.

### Slice 3

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_runtime_boundaries.py tests/test_mlx_prep.py` passed with `32 passed in 6.31s` on an approved unsandboxed rerun. The first sandboxed attempt failed because MLX could not access Metal.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts` passed with `All checks passed!`.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -c 'import openmm, importlib.metadata as md; ...'` reported OpenMM `metadata_version 8.5.1`, `installer uv`, `direct_url None`, and platforms `['Reference', 'CPU', 'OpenCL']`.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -c 'from lammps import lammps as L; ...'` reported LAMMPS version `20250722` and `has_package("GPU") == True` on an approved unsandboxed rerun. The first sandboxed attempt failed during MPI initialization on `utun6`.
- `rg -n "import openmm|from openmm|import lammps|from lammps" src/mlx_atomistic src/mlx_atomistic/prep scripts tests --glob '!vendors/**'` found OpenMM imports only in `scripts/run_openmm_gpcrmd_preview.py` and `scripts/run_openmm_gpcrmd_charmm_md.py`, plus the existing assertion string in `tests/test_mlx_prep.py`.

## Residual Risk

- Full-repo notebook lint remains outside this cleanup; source/test/script Ruff is clean.
- LAMMPS runtime checks need unsandboxed execution on this machine because MPI initialization is blocked by the sandbox network-interface policy.

## Verification: Lean Runtime Boundary Cleanup

**Date:** 2026-05-14
**Verifier:** Codex auto-verify

### Criterion 1: A new contributor can answer from repo docs and metadata that the primary trajectory generator is `mlx_atomistic`, not OpenMM or LAMMPS.

- **Result:** PASS
- **Evidence:** `rg -n "primary trajectory generator|reference|OpenMM|LAMMPS|vendors" README.md docs/runtime-boundaries.md pyproject.toml` found `README.md` stating "`mlx_atomistic` is the primary trajectory generator and product runtime" and `docs/runtime-boundaries.md` stating OpenMM and LAMMPS are reference engines.
- **Gap:** none

### Criterion 2: `pyproject.toml` keeps core runtime dependencies lean and keeps OpenMM/LAMMPS in non-core dependency surfaces.

- **Result:** PASS
- **Evidence:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -c 'import tomllib; ...'` exited 0 after asserting OpenMM and LAMMPS are not in `project.dependencies`.
- **Gap:** none

### Criterion 3: A repository scan finds no unintended OpenMM/LAMMPS runtime imports in `src/mlx_atomistic/`.

- **Result:** PASS
- **Evidence:** `rg -n "import openmm|from openmm|import lammps|from lammps" src/mlx_atomistic src/mlx_atomistic/prep scripts tests --glob '!vendors/**'` found OpenMM imports only in `scripts/run_openmm_gpcrmd_preview.py` and `scripts/run_openmm_gpcrmd_charmm_md.py`, plus an assertion string in `tests/test_mlx_prep.py`.
- **Gap:** none

### Criterion 4: Notebook docs label OpenMM outputs as reference or preview artifacts and MLX outputs as `mlx_atomistic` outputs.

- **Result:** PASS
- **Evidence:** `rg -n "mlx_atomistic|openmm-reference|reference preview|generated.*ignored|not production" ...` found `mlx_atomistic` labels in both notebook READMEs and `openmm-reference` labels in the ligand-receptor README plus both OpenMM scripts.
- **Gap:** none

### Criterion 5: `.gitignore` or local artifact documentation covers generated OpenMM/LAMMPS/MLX trajectory outputs that should not be committed.

- **Result:** PASS
- **Evidence:** `git check-ignore notebooks/ligand-receptor-motion/data/openmm-md/example/trajectory.npz notebooks/ligand-receptor-motion/data/openmm-preview/example/trajectory.npz notebooks/ligand-receptor-motion/data/gpcrmd-mlx/example/trajectory.npz` printed all three ignored paths.
- **Gap:** none

### Criterion 6: Validation records current OpenMM and LAMMPS provenance clearly.

- **Result:** PASS
- **Evidence:** OpenMM command printed `8.5.1` and `['Reference', 'CPU', 'OpenCL']`. LAMMPS command printed `20250722` and `True` for GPU package support.
- **Gap:** none

### Criterion 7: Source/test/script validation status is known after the cleanup.

- **Result:** PASS
- **Evidence:** `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_runtime_boundaries.py tests/test_mlx_prep.py` passed with `32 passed in 4.39s`; `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts` printed `All checks passed!`.
- **Gap:** none

### Content Checks

- **Result:** PASS
- **Audience:** new contributors and future agents; `README.md` points them to `docs/runtime-boundaries.md`, and the doc directly names runtime, reference-engine, and vendor roles.
- **Thesis:** `mlx_atomistic` is the product runtime; OpenMM, LAMMPS, and `vendors/` are reference or validation surfaces. Each changed doc section supports that boundary.
- **Voice:** concise repo documentation, factual and direct.
- **Source policy and factual risk:** technical provenance claims are backed by fresh `uv run` OpenMM/LAMMPS commands and `pyproject.toml` assertions.
- **Anti-slop scan:** `rg -n "successfully|crucial|important improvement|groundbreaking|revolutionary|Let's dive|serves as|stands as|highlighting|ensuring" ...` returned no matches.
- **Format:** docs, test, script metadata, and ignore-file changes match the plan targets.

### Summary

- **Overall:** PASS
- **Passed:** 7 of 7 criteria
- **Remaining gaps:** none
- **Recommended next skill:** none; the cleanup is verified.
