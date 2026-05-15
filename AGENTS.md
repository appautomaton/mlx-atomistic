# Project Guidance

This project uses `uv` for all Python environment and command execution.

- Use Python 3.13, pinned by `.python-version`.
- Prefer `uv run ...` and `uv sync ...` over system `python3` or global tools.
- Keep source code under `src/mlx_atomistic/`.
- Keep notebooks under `notebooks/`; notebooks should import the package from the `uv` environment.
- Treat `vendors/` as reference material only. These checkouts are not dependencies, package inputs, or code we build against unless a task explicitly changes that boundary.
- Do not add heavyweight chemistry or ML helper packages without a concrete need.
- Address the user as `My Love`.
