# Project Guidance

`mlx_atomistic` is an Apple-Silicon-native molecular dynamics and DFT runtime
built on MLX and Metal. That runtime is the product. OpenMM, LAMMPS, and
everything under `vendors/` are reference and validation surfaces only. They
never sit on the MLX runtime path and never replace it.

This file is the authoritative, cross-tool source of truth for working in this
repository. Read it in full at the start of a session. Narrative documentation
lives under `docs/` and in the docs site (`site/src/content/docs/`). Reach for
it when a section here points you deeper.

## Ground rules

Read this section first. These hold everywhere in the repo.

- `uv` runs everything. Prefer `uv run ...` and `uv sync ...` over system
  `python3` or global tools.
- Python is 3.13, pinned by `.python-version`.
- Run Git, builds, tests, and edits from the repository root so this file loads
  in full.
- The MLX runtime is the product. Reference engines (OpenMM, LAMMPS) validate
  it and never replace it.
- `vendors/` is local reference source, read-mostly, and gitignored.
  `results/` is local generated output and gitignored. Neither is a dependency,
  package input, or build target unless a task explicitly moves that boundary.
- Do not add heavyweight chemistry or ML packages without a concrete need.

## Repository layout

    mlx-atomistic/        project root and standard checkout
    ├── src/              product package source (mlx_atomistic/, incl. prep/)
    ├── tests/            pytest suite
    ├── scripts/          reference runners and the API-doc generator
    ├── notebooks/        Jupyter notebooks that import the package from uv
    ├── docs/             project and benchmark documentation
    ├── site/             docs + landing site (narrative + generated API reference)
    ├── results/          local generated outputs, gitignored
    └── vendors/          local reference trees, gitignored

Package source lives under `src/mlx_atomistic/`. Import and prep tooling lives
under `src/mlx_atomistic/prep/`.

## Environment and dependencies

Dependency groups keep heavy tooling separate from light:

- `test` (installed by default): pytest and ruff. The fast, engine-free lane.
- `reference`: OpenMM and LAMMPS. Heavy, only for reference and parity runs.
- `docs`: Griffe, for the API-reference generator.
- `dev`: `test` and `reference` combined, the full local environment.

Install with `uv sync` and the group you need, for example `uv sync --group dev`.

## Testing and CI

The suite runs on the MLX **CPU backend** by default. `tests/conftest.py` sets
`MLX_ATOMISTIC_DEVICE=cpu` and pins the default device and stream to CPU, and
`core.py` falls back to CPU when Metal is unavailable. Correctness is therefore
verifiable without a GPU, which is what lets cheap, remote, CPU-only CI carry
the regression safety net.

The GPU is a development and optimization instrument, not a test tier. Use
Apple Silicon Metal when you write or optimize the runtime and when you run real
workloads. Do not build a routine GPU test gate, and do not assume a GPU is
available. Local development often runs in low-power mode, which throttles
Metal.

Working guidance:

- Let remote CPU CI catch regressions. Do not expect anyone to run the full
  suite locally on every change.
- Run the GPU-marked tests (`--run-gpu`) only when you touch the Metal path,
  such as `metal_kernels.py` or `dft/_compact.py`. They are ad-hoc self-checks,
  not merge gates.
- The CPU result is the reference. Where a GPU path exists its numbers should
  match the CPU path, so an occasional CPU-versus-GPU parity check is enough.

Test markers (see `pyproject.toml` and `conftest.py`):

- `slow`, `integration`: descriptive markers for long or multi-component tests.
- `reference`, `data`, `gpu`: opt-in. They are skipped unless you pass
  `--run-reference`, `--run-data`, or `--run-gpu`.

## Documentation and docstrings

The docs site under `site/` has two parts. Hand-written narrative lives under
`site/src/content/docs/` (overview, foundations, mm, dft, benchmarks, project).
The API reference under `.../api/` is generated from the package by
`scripts/gen_api_docs.py`, a static Griffe parse with no import and no MLX,
git-ignored and rebuilt on deploy. The build also emits `llms.txt` and
`llms-full.txt` for agentic consumption.

Never hand-edit the API pages. Edit the docstrings and let the generator rebuild
them.

House style for public docstrings:

- Google style. `src/mlx_atomistic/core.py` is the exemplar to copy. Keep one
  blank line after the docstring.
- Most docstrings are one-line summaries today. Enrich them toward full `Args:`
  and `Returns:` and the page renders richer automatically.

Three CI guards keep the docs honest, and all must stay green:

- `ruff` **D101/D102/D103** require a docstring on every public class, method,
  and function on the docs surface.
- `ruff` **D417** fails when a documented `Args:` omits a parameter.
- `gen_api_docs.py` exits non-zero when a docstring documents a parameter the
  signature no longer has.

Presence rules are scoped to the documented public API. `prep/`, `benchmarks/`,
`tests/`, and `scripts/` are excluded (see `[tool.ruff.lint.per-file-ignores]`),
which matches the generator's SKIP set. Run `ruff check src` and the generator
before you push.

## Worktrees and parallel work

Read this section when you run parallel or isolated work. Instruction context is
per-agent, resolved from each agent's worktree root, so every worktree carries
its own `.venv` and its own committed copy of this file.

Isolate agents that edit files in parallel into their own worktrees so changes
never collide, then merge back through normal Git history. Keep the `vendors/`
boundary in every worktree. Use worktrees only when parallel branch isolation is
worth the extra setup.

From the repository root:

    git worktree add ../mlx-atomistic-feat-x -b feat/x
    ln -s ../mlx-atomistic/vendors ../mlx-atomistic-feat-x/vendors
    cd ../mlx-atomistic-feat-x && uv venv --python 3.13 && uv sync --group dev

Tear down with `git worktree remove ../mlx-atomistic-feat-x`. Never use
`rm -rf`.

## Communication

- Address the user as `My Love`.
- Prefer plain, direct sentences. Avoid overusing mid-sentence em-dashes and
  semicolons.
