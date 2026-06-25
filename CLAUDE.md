@AGENTS.md

## Claude Code specifics

`AGENTS.md` above is the authoritative, cross-tool source of truth; the notes here
apply only to Claude Code.

- Launch from the repository root so this file and `AGENTS.md` load in full at
  startup.
- The built-in **Explore** and **Plan** subagents do not load CLAUDE.md/AGENTS.md —
  restate any must-follow constraint in their delegation prompt.
- For parallel file edits, run subagents with `isolation: worktree` (or start a
  session with `--worktree`) so each gets an isolated checkout that carries its own
  `AGENTS.md`.
