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

## Repository layout — bare repo + linked worktrees

    mlx-atomistic/        container root — NOT a checkout (.git -> ./.bare)
    ├── .bare/            the bare repository (objects + refs); read-mostly storage
    ├── main/             worktree for the `main` branch   <- work happens here
    └── vendors/          shared reference trees, untracked (scripts/fetch-vendors)

- The container root has no working tree: `git status` there fatals by design.
  Never edit, build, or run from it — only from inside a worktree.
- `vendors/` exists once at the container root and is shared into each worktree
  via a `vendors -> ../vendors` symlink (untracked).

## Worktree workflow

One branch <-> one worktree <-> (typically) one agent or session. From inside any
worktree:

    git worktree add ../feat-x -b feat/x       # new sibling worktree + branch
    ln -s ../vendors ../feat-x/vendors          # share the reference trees
    cd ../feat-x && uv venv --python 3.13 && uv sync --group dev

Tear down with `git worktree remove ../feat-x` (never `rm -rf`). Each worktree
carries its own `.venv` and its own committed copy of this file, so a branch's
rules always travel with its checkout.

## Multi-agent / parallel work

- Instruction context is per-agent, resolved from each agent's worktree root.
  Because this file is committed in-branch, every spun-up worktree materializes it
  automatically and it dies with the worktree on teardown.
- Isolate agents that edit files in parallel into their own worktrees so changes
  never collide; merge back through the shared `.bare` history.
- Keep the `vendors/` reference-only boundary in every worktree.

## Communication

- Address the user as `My Love`.
