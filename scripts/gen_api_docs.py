#!/usr/bin/env python
"""Generate Starlight API-reference pages from mlx_atomistic docstrings.

Static-only: this uses Griffe's AST loader, so it never imports the package and
needs neither MLX nor a Metal GPU. That keeps it runnable on Linux Pages CI.

    # Apple Silicon, with the project's docs group:
    uv run --group docs python scripts/gen_api_docs.py

    # Anywhere (Linux CI), isolated, without installing the project / MLX:
    uv run --no-project --with griffe python scripts/gen_api_docs.py

Output lands in site/src/content/docs/api/ and is git-ignored — it is a build
artifact, regenerated from source every time.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import griffe
from griffe import ParameterKind

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "site" / "src" / "content" / "docs" / "api"
PACKAGE = "mlx_atomistic"

# Subpackages whose internals we do not surface as reference pages.
SKIP = {"benchmarks", "prep"}

# Curated lead ordering; anything unlisted falls in afterwards, alphabetically.
ORDER_HINT = [
    "core",
    "units",
    "topology",
    "forcefields",
    "nonbonded",
    "neighbors",
    "md",
    "minimize",
    "runtime",
]

_VAR_POSITIONAL = ParameterKind.var_positional
_VAR_KEYWORD = ParameterKind.var_keyword


def summary(obj) -> str:
    """First paragraph of a docstring, collapsed to a single line."""
    if obj.docstring is None:
        return ""
    text = obj.docstring.value.strip()
    if not text:
        return ""
    para = text.split("\n\n", 1)[0]
    return " ".join(line.strip() for line in para.splitlines())


def param_list(func, *, is_method: bool = False) -> str:
    parts: list[str] = []
    for p in func.parameters:
        if is_method and p.name in ("self", "cls"):
            continue
        tok = p.name
        if p.kind is _VAR_POSITIONAL:
            tok = "*" + tok
        elif p.kind is _VAR_KEYWORD:
            tok = "**" + tok
        if p.annotation is not None:
            tok += f": {p.annotation}"
        if p.default is not None:
            tok += f" = {p.default}"
        parts.append(tok)
    return "(" + ", ".join(parts) + ")"


def render_function(func, *, level: int = 3) -> list[str]:
    sig = f"def {func.name}{param_list(func)}"
    if func.returns is not None:
        sig += f" -> {func.returns}"
    lines = [f"{'#' * level} `{func.name}`", "", "```python", sig, "```", ""]
    if func.docstring and func.docstring.value.strip():
        lines.append(func.docstring.value.strip())
        lines.append("")
    return lines


def render_class(cls, *, level: int = 3) -> list[str]:
    bases = [str(b) for b in cls.bases] if cls.bases else []
    header = f"class {cls.name}" + (f"({', '.join(bases)})" if bases else "")
    lines = [f"{'#' * level} `{cls.name}`", "", "```python", header]

    init = cls.members.get("__init__")
    if init is not None and getattr(init, "is_function", False):
        lines.append(f"    def __init__{param_list(init, is_method=True)}")
    lines.append("```")
    lines.append("")

    if cls.docstring and cls.docstring.value.strip():
        lines.append(cls.docstring.value.strip())
        lines.append("")

    methods = [
        m
        for m in cls.members.values()
        if not m.is_alias
        and getattr(m, "is_function", False)
        and m.is_public
        and m.name != "__init__"
    ]
    if methods:
        lines.append("**Methods**")
        lines.append("")
        for m in sorted(methods, key=lambda x: x.name):
            sig = param_list(m, is_method=True)
            desc = summary(m)
            suffix = f" — {desc}" if desc else ""
            lines.append(f"- `{m.name}{sig}`{suffix}")
        lines.append("")
    return lines


def public(obj, predicate) -> list:
    out = [
        m
        for m in obj.members.values()
        if not m.is_alias and m.is_public and predicate(m)
    ]
    return sorted(out, key=lambda x: x.name)


def render_module(mod, order: int) -> str | None:
    classes = public(mod, lambda m: getattr(m, "is_class", False))
    functions = public(mod, lambda m: getattr(m, "is_function", False))
    if not classes and not functions:
        return None

    title = mod.path.replace(f"{PACKAGE}.", "")
    lines = [
        "---",
        f"title: {title}",
        f"description: API reference for {mod.path}.",
        "sidebar:",
        f"  order: {order}",
        "---",
        "",
    ]
    mod_summary = summary(mod)
    if mod_summary:
        lines += [mod_summary, ""]
    lines += [f"> `import {mod.path}`", ""]

    if classes:
        lines += ["## Classes", ""]
        for cls in classes:
            lines += render_class(cls)
    if functions:
        lines += ["## Functions", ""]
        for func in functions:
            lines += render_function(func)

    return "\n".join(lines).rstrip() + "\n"


def iter_target_modules(package):
    """Yield the package's documentable submodules (one level into subpackages)."""
    for member in package.members.values():
        if member.is_alias or not getattr(member, "is_module", False):
            continue
        if member.name.startswith("_") or member.name in SKIP:
            continue
        # A subpackage (e.g. dft): descend one level into its modules.
        if member.is_package or member.members:
            sub = [
                s
                for s in member.members.values()
                if not s.is_alias
                and getattr(s, "is_module", False)
                and not s.name.startswith("_")
            ]
            if sub and member.is_package:
                yield from sorted(sub, key=lambda x: x.name)
                continue
        yield member


def order_for(name: str) -> int:
    short = name.replace(f"{PACKAGE}.", "")
    leaf = short.split(".")[-1]
    if leaf in ORDER_HINT:
        return ORDER_HINT.index(leaf)
    return 100 + sum(ord(c) for c in leaf[:3])


def main() -> None:
    package = griffe.load(PACKAGE, search_paths=[str(SRC)])

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    modules = sorted(set(iter_target_modules(package)), key=lambda m: order_for(m.path))

    written: list[str] = []
    for mod in modules:
        page = render_module(mod, order_for(mod.path))
        if page is None:
            continue
        slug = mod.path.replace(f"{PACKAGE}.", "").replace(".", "-")
        (OUT / f"{slug}.md").write_text(page, encoding="utf-8")
        written.append(mod.path)

    index = [
        "---",
        "title: API Reference",
        "description: Auto-generated reference for the mlx_atomistic public API.",
        "sidebar:",
        "  order: 0",
        "---",
        "",
        "Reference pages below are generated directly from the package source with",
        "[Griffe](https://mkdocstrings.github.io/griffe/) — signatures and docstrings",
        "stay in lockstep with the code, regenerated on every build.",
        "",
        "## Modules",
        "",
    ]
    for path in sorted(written):
        short = path.replace(f"{PACKAGE}.", "")
        slug = short.replace(".", "-")
        index.append(f"- [`{short}`](./{slug}/)")
    (OUT / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")

    print(f"Generated {len(written)} API page(s) into {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
