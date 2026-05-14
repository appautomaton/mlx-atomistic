# PLAN: Lean Runtime Boundary Cleanup

## Goal

Make the repository structure and project metadata clearly communicate that `mlx_atomistic` is the product runtime, while OpenMM, LAMMPS, and vendor checkouts are reference or validation surfaces only.

## Architecture Approach

Use documentation, dependency metadata, script labels, and scan-based guardrails to make the boundary obvious without changing runtime behavior. Keep the core package dependency list lean; keep OpenMM and LAMMPS in dev/reference-engine surfaces; keep `vendors/` reference-only; and make every external-engine artifact read as reference or preview output.

No `DESIGN.md` is required for this change because it introduces no new runtime architecture or data model. The executable design is the existing repo layout plus stricter labels and verification checks.

## Ordered Task Sequence

### Slice 1: Central Runtime Boundary

**Objective:** Add a single authoritative boundary statement for runtime ownership and dependency roles.
**Execution:** direct
**Depends on:** none
**Touches:** `README.md`, `docs/runtime-boundaries.md`, `pyproject.toml`
**Context budget:** ~8% of context window
**Produces:** Repo-level documentation that says `mlx_atomistic` is the primary trajectory generator and OpenMM/LAMMPS are reference tools, plus inline dependency-role clarity for dev/reference engines.
**Acceptance criteria:**
- `README.md` points contributors to the runtime boundary.
- `docs/runtime-boundaries.md` records the OpenMM PyPI-wheel/OpenCL status, the LAMMPS local-build/OpenCL status, and the `vendors/` reference-only rule.
- `pyproject.toml` still keeps core dependencies limited to the MLX runtime surface, with OpenMM/LAMMPS remaining outside core package dependencies.
**Verification:**
```bash
rg -n "primary trajectory generator|reference|OpenMM|LAMMPS|vendors" README.md docs/runtime-boundaries.md pyproject.toml
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -c 'import tomllib; data=tomllib.load(open("pyproject.toml","rb")); assert "openmm>=8.5.1" not in data["project"]["dependencies"]; assert "lammps>=2025.7.22.4.0" not in data["project"]["dependencies"]'
```
**Auto-continue:** yes

### Slice 2: Reference Workflow Labels And Artifact Hygiene

**Objective:** Make notebooks, OpenMM scripts, and generated artifact paths impossible to confuse with product runtime output.
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `notebooks/README.md`, `notebooks/ligand-receptor-motion/README.md`, `scripts/run_openmm_gpcrmd_preview.py`, `scripts/run_openmm_gpcrmd_charmm_md.py`, `.gitignore`
**Context budget:** ~10% of context window
**Produces:** Consistent `mlx_atomistic`, `openmm-reference`, and local-generated-artifact labels across the active notebook workflow and OpenMM preview scripts.
**Acceptance criteria:**
- Active MLX notebook/docs label MLX output as `mlx_atomistic`.
- OpenMM preview notebooks/scripts label their outputs as reference or preview artifacts, not production runtime output.
- `.gitignore` or notebook documentation covers generated OpenMM/MLX trajectory directories that should remain local.
**Verification:**
```bash
rg -n "mlx_atomistic|openmm-reference|reference preview|generated.*ignored|not production" notebooks/README.md notebooks/ligand-receptor-motion/README.md scripts/run_openmm_gpcrmd_preview.py scripts/run_openmm_gpcrmd_charmm_md.py .gitignore
git check-ignore notebooks/ligand-receptor-motion/data/openmm-md/example/trajectory.npz notebooks/ligand-receptor-motion/data/openmm-preview/example/trajectory.npz notebooks/ligand-receptor-motion/data/gpcrmd-mlx/example/trajectory.npz
```
**Auto-continue:** yes

### Slice 3: Boundary Guardrails And Validation Evidence

**Objective:** Add or refresh scan-based checks that prevent accidental external-engine runtime imports and record current engine provenance.
**Execution:** direct
**Depends on:** Slice 2
**Touches:** `tests/test_runtime_boundaries.py` or existing focused tests, `.agent/work/lean-runtime-boundary-cleanup/VERIFY.md`
**Context budget:** ~10% of context window
**Produces:** A focused boundary test plus a verification note proving OpenMM and LAMMPS provenance and support status.
**Acceptance criteria:**
- `src/mlx_atomistic/` has no `import openmm`, `from openmm`, `import lammps`, or `from lammps` runtime dependency.
- External-engine usage outside `vendors/` is limited to documented reference/prep/script/notebook/test surfaces.
- OpenMM provenance is recorded as `uv`/PyPI wheel with `OpenCL` available.
- LAMMPS provenance is recorded as `uv` local build from upstream PyPI source with GPU/OpenCL enabled.
- Source/test/script validation status is recorded.
**Verification:**
```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_runtime_boundaries.py tests/test_mlx_prep.py
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -c 'import openmm, importlib.metadata as md; print(md.version("openmm")); print([openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())])'
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -c 'from lammps import lammps as L; lmp=L(cmdargs=["-echo","none","-log","none","-screen","none"]); print(lmp.version()); print(lmp.has_package("GPU")); lmp.close()'
```
**Auto-continue:** no

## Execution Routing And Topology

- Slice 1: direct.
- Slice 2: direct.
- Slice 3: direct.
- Auto-continue chain: Slice 1 -> Slice 2 -> Slice 3 after each slice passes its verification.
- Checkpoints: Slice 3 is the completion checkpoint because it records validation evidence and may expose environment-specific LAMMPS/MPI constraints.
- Parallel-safe groups: none. The edits are small, but they overlap in documentation language and should stay serial for consistency.
- Subagents: none recommended. The work is bounded, low-risk, and mostly docs/tests in one repo.

## Verification Commands

Run the slice-specific verification commands after each slice. Before completion, run:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_runtime_boundaries.py tests/test_mlx_prep.py
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts
rg -n "import openmm|from openmm|import lammps|from lammps" src/mlx_atomistic src/mlx_atomistic/prep scripts tests --glob '!vendors/**'
```

If the LAMMPS runtime provenance check fails with MPI interface sandboxing, rerun that single command with approved unsandboxed execution and record the reason in `VERIFY.md`.

## Context Budget For This Change

Estimated total execution context: ~28% of the context window.

Expected source loading:
- Slice 1: `README.md`, `pyproject.toml`, new boundary doc.
- Slice 2: active notebook README, OpenMM preview scripts, `.gitignore`.
- Slice 3: focused test file and verification output only.
