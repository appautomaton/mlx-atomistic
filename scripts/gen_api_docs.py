#!/usr/bin/env python
"""Generate Starlight API-reference pages from mlx_atomistic docstrings.

Static-only: this uses Griffe's AST loader, so it never imports the package and
needs neither MLX nor a Metal GPU. That keeps it runnable on Linux Pages CI.

    # Apple Silicon, with the project's docs group:
    uv run --group docs python scripts/gen_api_docs.py

    # Anywhere (Linux CI), isolated, without installing the project / MLX:
    uv run --no-project --with griffe python scripts/gen_api_docs.py

Docstrings are parsed as Google style: ``Args:`` descriptions are merged with the
real signature (types/defaults come from the code), and ``Returns:``/``Raises:``/
``Examples:`` render as their own sections. Output lands in
site/src/content/docs/api/ and is git-ignored — a build artifact regenerated
from source every time.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import griffe
from griffe import DocstringSectionKind as Kind
from griffe import ParameterKind, Parser

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
_KEYWORD_ONLY = ParameterKind.keyword_only


def esc(text: str) -> str:
    """Make a string safe for a single Markdown table cell."""
    return " ".join(str(text).split()).replace("|", "\\|")


def parse_doc(obj) -> dict:
    """Pull the structured pieces out of an object's Google-style docstring."""
    out = {"summary": "", "body": [], "params": {}, "returns": [], "raises": [], "examples": []}
    doc = getattr(obj, "docstring", None)
    if doc is None:
        return out
    try:
        sections = doc.parse(Parser.google)
    except Exception:
        text = doc.value.strip()
        out["summary"] = text.split("\n\n", 1)[0]
        return out

    first_text = True
    for s in sections:
        if s.kind is Kind.text:
            text = s.value.strip()
            if first_text:
                parts = text.split("\n\n", 1)
                out["summary"] = parts[0]
                if len(parts) > 1:
                    out["body"].append(parts[1])
                first_text = False
            else:
                out["body"].append(text)
        elif s.kind is Kind.parameters:
            for p in s.value:
                out["params"][p.name] = (p.description or "").strip()
        elif s.kind is Kind.returns:
            for r in s.value:
                ann = str(r.annotation) if r.annotation is not None else ""
                out["returns"].append((ann, (r.description or "").strip()))
        elif s.kind is Kind.raises:
            for r in s.value:
                ann = str(r.annotation) if r.annotation is not None else ""
                out["raises"].append((ann, (r.description or "").strip()))
        elif s.kind is Kind.examples:
            for item in s.value:
                content = item[1] if isinstance(item, tuple) else str(item)
                out["examples"].append(content.strip())
    return out


def param_list(func, *, is_method: bool = False) -> str:
    parts: list[str] = []
    star_done = False
    for p in func.parameters:
        if is_method and p.name in ("self", "cls"):
            continue
        if p.kind is _KEYWORD_ONLY and not star_done:
            parts.append("*")
            star_done = True
        tok = p.name
        if p.kind is _VAR_POSITIONAL:
            tok = "*" + tok
            star_done = True
        elif p.kind is _VAR_KEYWORD:
            tok = "**" + tok
        if p.annotation is not None:
            tok += f": {p.annotation}"
        if p.default is not None:
            tok += f" = {p.default}"
        parts.append(tok)
    return "(" + ", ".join(parts) + ")"


def render_params_table(func, descriptions: dict, *, is_method: bool = False) -> list[str]:
    params = [
        p for p in func.parameters if not (is_method and p.name in ("self", "cls"))
    ]
    if not params:
        return []
    lines = ["**Parameters**", "", "| Name | Type | Default | Description |", "|---|---|---|---|"]
    for p in params:
        name = p.name
        if p.kind is _VAR_POSITIONAL:
            name = "*" + name
        elif p.kind is _VAR_KEYWORD:
            name = "**" + name
        typ = f"`{esc(p.annotation)}`" if p.annotation is not None else ""
        default = f"`{esc(p.default)}`" if p.default is not None else ""
        desc = esc(descriptions.get(p.name, ""))
        lines.append(f"| `{name}` | {typ} | {default} | {desc} |")
    lines.append("")
    return lines


def render_returns_raises(doc: dict, func) -> list[str]:
    lines: list[str] = []
    if doc["returns"]:
        lines += ["**Returns**", ""]
        for ann, desc in doc["returns"]:
            ann = ann or (str(func.returns) if func.returns is not None else "")
            tag = f"`{ann}` — " if ann else ""
            lines.append(f"- {tag}{desc}".rstrip(" —"))
        lines.append("")
    elif func.returns is not None:
        lines += ["**Returns**", "", f"- `{func.returns}`", ""]
    if doc["raises"]:
        lines += ["**Raises**", ""]
        for ann, desc in doc["raises"]:
            tag = f"`{ann}` — " if ann else ""
            lines.append(f"- {tag}{desc}".rstrip(" —"))
        lines.append("")
    return lines


def render_examples(doc: dict) -> list[str]:
    if not doc["examples"]:
        return []
    lines = ["**Examples**", ""]
    for ex in doc["examples"]:
        lines += ["```python", ex, "```", ""]
    return lines


def render_function(func, *, level: int = 3, is_method: bool = False) -> list[str]:
    doc = parse_doc(func)
    sig = f"def {func.name}{param_list(func, is_method=is_method)}"
    if func.returns is not None:
        sig += f" -> {func.returns}"
    lines = [f"{'#' * level} `{func.name}`", "", "```python", sig, "```", ""]
    if doc["summary"]:
        lines += [doc["summary"], ""]
    for block in doc["body"]:
        lines += [block, ""]
    lines += render_params_table(func, doc["params"], is_method=is_method)
    lines += render_returns_raises(doc, func)
    lines += render_examples(doc)
    return lines


def render_class(cls, *, level: int = 3) -> list[str]:
    doc = parse_doc(cls)
    bases = [str(b) for b in cls.bases] if cls.bases else []
    header = f"class {cls.name}" + (f"({', '.join(bases)})" if bases else "")
    lines = [f"{'#' * level} `{cls.name}`", "", "```python", header]
    init = cls.members.get("__init__")
    if init is not None and getattr(init, "is_function", False):
        lines.append(f"    def __init__{param_list(init, is_method=True)}")
    lines.append("```")
    lines.append("")
    if doc["summary"]:
        lines += [doc["summary"], ""]
    for block in doc["body"]:
        lines += [block, ""]
    if init is not None and getattr(init, "is_function", False):
        init_doc = parse_doc(init)
        params = {**init_doc["params"], **doc["params"]}
        lines += render_params_table(init, params, is_method=True)

    # Properties render compactly (no parameters): name, type, summary.
    props = [
        m
        for m in cls.members.values()
        if not m.is_alias
        and m.is_public
        and getattr(m, "is_attribute", False)
        and "property" in (m.labels or set())
        and not m.name.startswith("_")
    ]
    if props:
        lines += ["**Properties**", ""]
        for p in sorted(props, key=lambda x: x.name):
            ann = f" `{p.annotation}`" if p.annotation is not None else ""
            desc = parse_doc(p)["summary"]
            tail = f" — {desc}" if desc else ""
            lines.append(f"- `{p.name}`{ann}{tail}")
        lines.append("")

    # Methods render as full subsections so their Args/Returns/Raises show.
    methods = [
        m
        for m in cls.members.values()
        if not m.is_alias
        and getattr(m, "is_function", False)
        and m.is_public
        and not m.name.startswith("__")
    ]
    if methods:
        lines += ["**Methods**", ""]
        for m in sorted(methods, key=lambda x: x.name):
            lines += render_function(m, level=level + 1, is_method=True)
    return lines


def public(obj, predicate) -> list:
    out = [m for m in obj.members.values() if not m.is_alias and m.is_public and predicate(m)]
    return sorted(out, key=lambda x: x.name)


def render_module(mod, order: int) -> str | None:
    classes = public(mod, lambda m: getattr(m, "is_class", False))
    functions = public(mod, lambda m: getattr(m, "is_function", False))
    if not classes and not functions:
        return None

    title = mod.path.replace(f"{PACKAGE}.", "")
    doc = parse_doc(mod)
    lines = [
        "---",
        f"title: {title}",
        f"description: API reference for {mod.path}.",
        "sidebar:",
        f"  order: {order}",
        "---",
        "",
    ]
    if doc["summary"]:
        lines += [doc["summary"], ""]
    for block in doc["body"]:
        lines += [block, ""]
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
    leaf = name.replace(f"{PACKAGE}.", "").split(".")[-1]
    if leaf in ORDER_HINT:
        return ORDER_HINT.index(leaf)
    return 100 + sum(ord(c) for c in leaf[:3])


class _DriftHandler(logging.Handler):
    """Capture Griffe warnings that signal docstring/signature drift."""

    PHRASE = "does not appear in the function signature"

    def __init__(self) -> None:
        super().__init__()
        self.drift: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if self.PHRASE in message:
            self.drift.append(message)


def main() -> None:
    # Fail the build if a docstring documents a parameter the signature no longer
    # has — the rename-style drift that structural auto-sync cannot catch.
    drift = _DriftHandler()
    griffe_logger = logging.getLogger("griffe")
    griffe_logger.addHandler(drift)
    griffe_logger.setLevel(logging.WARNING)

    package = griffe.load(PACKAGE, search_paths=[str(SRC)], docstring_parser=Parser.google)

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

    if drift.drift:
        print(
            "\nDocstring drift detected — documented parameters absent from the "
            "signature:",
            file=sys.stderr,
        )
        for message in sorted(set(drift.drift)):
            print(f"  - {message}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
