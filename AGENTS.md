# Project Guidance

`mlx_atomistic` is the Apple-Silicon-native MD + DFT runtime built on MLX/Metal —
the product. OpenMM, LAMMPS, and everything under `vendors/` are reference and
validation surfaces only; they never replace the MLX runtime path.

## Environment

- `uv` runs everything. Prefer `uv run ...` / `uv sync ...` over system `python3`
  or global tools.
- Python 3.13, pinned by `.python-version`.
- Package source under `src/mlx_atomistic/`; prep/import tooling under
  `src/mlx_atomistic/prep/`.
- Notebooks live under `notebooks/` and import the package from the `uv` env.
- Treat `vendors/` as reference only — not dependencies, package inputs, or build
  targets unless a task explicitly moves that boundary.
- Do not add heavyweight chemistry or ML helper packages without a concrete need.

## Repository layout

    mlx-atomistic/        standard checkout and project root
    ├── .git/             normal Git metadata
    ├── src/              product package source
    ├── tests/            pytest suite
    ├── docs/             project and benchmark documentation
    ├── results/          local generated outputs, gitignored
    └── vendors/          local reference trees, gitignored

- Run Git, builds, tests, and edits from the repository root.
- `results/` contains local generated benchmark/science outputs and stays
  gitignored.
- `vendors/` contains local reference source trees and stays read-mostly unless
  a task explicitly moves that boundary.

## Worktree workflow

Use worktrees only when parallel branch isolation is worth the extra setup. From
the repository root:

    git worktree add ../mlx-atomistic-feat-x -b feat/x
    ln -s ../mlx-atomistic/vendors ../mlx-atomistic-feat-x/vendors
    cd ../mlx-atomistic-feat-x && uv venv --python 3.13 && uv sync --group dev

Tear down with `git worktree remove ../mlx-atomistic-feat-x` (never `rm -rf`).
Each worktree carries its own `.venv` and its own committed copy of this file.

## Multi-agent / parallel work

- Instruction context is per-agent, resolved from each agent's worktree root.
- Isolate agents that edit files in parallel into their own worktrees so changes
  never collide; merge back through normal Git history.
- Keep the `vendors/` reference-only boundary in every worktree.

## Communication

- Address the user as `My Love`.
