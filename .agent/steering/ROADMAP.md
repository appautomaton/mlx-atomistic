# Roadmap

## Sequencing Notes

- The repository is already a functioning package with green tests, active notebooks, package entry points, and scaffold-level steering now replaced by evidence-backed onboarding (`README.md`, `pyproject.toml`, `.agent/wiki/REPO-MAP.md`).
- The first real change should be framed before implementation because the active Automaton state is still `bootstrap` / `frame` (`.agent/.automaton/state/current.json`).

## Phase 1: Frame the First Concrete Slice

- objective: choose one bounded first change after onboarding, such as notebook lint policy, preparation workflow hardening, benchmark reporting, or a focused DFT/MD capability slice.
- why now: repo steering is now current, but the active work state is still bootstrap/frame and no specific implementation objective is accepted (`.agent/.automaton/state/current.json`, `.agent/wiki/REPO-MAP.md`).
- likely outputs: `.agent/work/<change>/SPEC.md` and an accepted scope for planning.
- evidence: `.agent/.automaton/state/current.json`, `.agent/wiki/REPO-MAP.md`
- exit signal: a SPEC with observed/inferred/unknown requirements and a clear non-goal boundary.

## Phase 2: Resolve Verification Policy Drift

- objective: make verification expectations unambiguous, especially the mismatch between green source/test lint and failing full-repo notebook lint.
- why now: `uv run pytest` and source lint are green, while `uv run ruff check .` fails on notebook findings (`pyproject.toml`, verified commands).
- likely outputs: either notebook lint fixes, a deliberate Ruff exclusion/selection policy, or a documented split between source lint and notebook hygiene.
- evidence: `pyproject.toml`, `notebooks/README.md`, `.agent/wiki/REPO-MAP.md`
- exit signal: later plans can name exactly which lint command is required for completion.

## Phase 3: Stabilize User-Facing Workflow Surfaces

- objective: harden the surfaces users actually run: `atomistic-prep`, `mlx-atomistic-benchmark`, active notebooks, and setup commands.
- why now: these are declared entry points and documented workflows, so they are the highest-leverage operational surfaces (`pyproject.toml`, `README.md`, `src/atomistic_prep/cli.py`, `notebooks/README.md`).
- likely outputs: focused CLI tests, workflow smoke checks, benchmark output contracts, and notebook regeneration checks where needed.
- evidence: `pyproject.toml`, `src/atomistic_prep/cli.py`, `notebooks/README.md`
- exit signal: the chosen workflow can be run from a fresh `uv sync` with documented commands and no hidden manual steps.

## Phase 4: Deepen Scientific Capability Behind Existing Boundaries

- objective: extend MD/DFT capability only where tests, notebooks, and docs can keep the behavior validated.
- why now: the public API already exposes MD, DFT, force fields, validation, trajectories, and visualization surfaces, but the README keeps scope intentionally lightweight (`README.md`, `src/mlx_atomistic/__init__.py`).
- likely outputs: a focused MD or DFT increment with tests, benchmark evidence, and a notebook or docs update when user-facing.
- evidence: `README.md`, `src/mlx_atomistic/__init__.py`, `tests/`
- exit signal: new capability has a regression test, a documented command or notebook path, and no new heavyweight dependency without justification.

## Deferred or Not Now

- Broad production DFT-engine claims are deferred because the README explicitly frames the repo as lightweight validated building blocks (`README.md`).
- Turning `vendors/` into build inputs or dependencies is deferred unless a task explicitly changes that boundary (`AGENTS.md`, `README.md`).
- Adding broad chemistry/ML helper packages is deferred until tied to a concrete accepted feature (`AGENTS.md`, `pyproject.toml`).
