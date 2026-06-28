"""Validate PyPI distribution archives contain only release-safe surfaces."""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

FORBIDDEN_PARTS = {
    "__pycache__",
    ".github",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "artifacts",
    "docs",
    "notebooks",
    "outputs",
    "results",
    "scratch",
    "scripts",
    "site",
    "tests",
    "tmp",
    "vendors",
}
FORBIDDEN_NAMES = {
    ".coverage",
    "AGENTS.md",
    "CLAUDE.md",
    "uv.lock",
}
FORBIDDEN_SUFFIXES = {
    ".pyc",
    ".pyo",
}
REQUIRED_WHEEL_MEMBERS = {
    "mlx_atomistic/__init__.py",
    "mlx_atomistic/py.typed",
}
REQUIRED_SDIST_MEMBERS = {
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "src/mlx_atomistic/__init__.py",
    "src/mlx_atomistic/py.typed",
}
REQUIRED_DIST_INFO_FILES = {
    "METADATA",
    "WHEEL",
    "entry_points.txt",
    "LICENSE",
}


def _archive_kind(path: Path) -> str:
    if path.suffix == ".whl":
        return "wheel"
    if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix in {".tgz", ".tar"}:
        return "sdist"
    if path.suffix == ".zip":
        return "zip"
    msg = f"unsupported distribution archive: {path}"
    raise ValueError(msg)


def _archive_members(path: Path) -> list[str]:
    kind = _archive_kind(path)
    if kind in {"wheel", "zip"}:
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    if kind == "sdist":
        with tarfile.open(path, "r:*") as archive:
            return archive.getnames()
    msg = f"unsupported distribution archive: {path}"
    raise ValueError(msg)


def _strip_sdist_root(member: str) -> str:
    path = PurePosixPath(member)
    parts = path.parts
    if len(parts) > 1 and parts[0].startswith("mlx_atomistic-"):
        return str(PurePosixPath(*parts[1:]))
    return str(path)


def _member_offenses(member: str) -> list[str]:
    path = PurePosixPath(member)
    parts = set(path.parts)
    offenses = sorted(parts & FORBIDDEN_PARTS)
    if path.name in FORBIDDEN_NAMES:
        offenses.append(path.name)
    if path.suffix in FORBIDDEN_SUFFIXES:
        offenses.append(path.suffix)
    return offenses


def _has_dist_info_file(members: set[str], name: str) -> bool:
    for member in members:
        path = PurePosixPath(member)
        if path.name != name:
            continue
        if any(part.endswith(".dist-info") for part in path.parts[:-1]):
            return True
    return False


def _required_member_violations(path: Path, members: list[str]) -> list[str]:
    kind = _archive_kind(path)
    member_set = set(members)
    violations = []
    if kind == "wheel":
        for member in sorted(REQUIRED_WHEEL_MEMBERS):
            if member not in member_set:
                violations.append(f"{path.name}: missing required wheel member {member}")
        for name in sorted(REQUIRED_DIST_INFO_FILES):
            if not _has_dist_info_file(member_set, name):
                violations.append(f"{path.name}: missing required dist-info file {name}")
    elif kind == "sdist":
        normalized_members = {_strip_sdist_root(member) for member in members}
        for member in sorted(REQUIRED_SDIST_MEMBERS):
            if member not in normalized_members:
                violations.append(f"{path.name}: missing required sdist member {member}")
    return violations


def check_archive(path: Path) -> list[str]:
    """Return release-safety violations for one archive."""

    violations = []
    members = _archive_members(path)
    for member in members:
        offenses = _member_offenses(member)
        if offenses:
            violations.append(f"{path.name}: {member} ({', '.join(offenses)})")
    violations.extend(_required_member_violations(path, members))
    return violations


def main(argv: list[str] | None = None) -> int:
    """Run archive content checks."""

    paths = [Path(value) for value in (sys.argv[1:] if argv is None else argv)]
    if not paths:
        print("usage: check_dist_contents.py DIST_ARCHIVE [...]", file=sys.stderr)
        return 2

    violations = []
    for path in paths:
        violations.extend(check_archive(path))
    if violations:
        print("distribution content violations:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    print(f"checked {len(paths)} distribution archive(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
