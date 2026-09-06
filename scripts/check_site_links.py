"""Check that internal links in the generated site resolve to built files."""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parent.parent
SITE = "https://appautomaton.renocrypt.com/mlx-atomistic/"


class LinkParser(HTMLParser):
    """Collect anchor destinations without interpreting scripts or examples."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self.links.extend(value for key, value in attrs if key == "href" and value)


def check_links(dist: Path) -> list[str]:
    """Return internal links whose destinations are absent from the build."""

    pages = sorted(dist.rglob("*.html"))
    if not pages:
        return [f"no HTML pages found in {dist}"]
    failures = []
    site = urlsplit(SITE)
    for page in pages:
        relative = page.relative_to(dist).as_posix()
        page_url = SITE + relative.removesuffix("index.html")
        parser = LinkParser()
        parser.feed(page.read_text())
        for href in sorted(set(parser.links)):
            destination = urlsplit(urljoin(page_url, href))
            if destination.netloc != site.netloc:
                continue
            if not destination.path.startswith(site.path):
                continue
            route = unquote(destination.path.removeprefix(site.path))
            target = dist / route
            if not target.is_file() and not (target / "index.html").is_file():
                failures.append(f"{relative}: {href}")
    return failures


def main() -> int:
    """Fail a site build when its published navigation contains missing pages."""

    failures = check_links(ROOT / "site" / "dist")
    if failures:
        print("Broken published links:\n" + "\n".join(failures), file=sys.stderr)
        return 1
    print("Verified internal links in the generated site")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
