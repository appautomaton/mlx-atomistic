"""Regression checks for links between published documentation pages."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.check_site_links import check_links
from scripts.sync_site_docs import DOCS, _published_sources, _site_link


class PublishedLinkTests(TestCase):
    def test_sibling_cross_section_and_index_urls(self):
        published = _published_sources()
        cases = [
            (
                "dft-production-core.md",
                "dft-roadmap.md#next",
                "/mlx-atomistic/dft/dft-roadmap/#next",
            ),
            (
                "dft-geometry-optimization.md",
                "dft-periodic-relaxation.md",
                "/mlx-atomistic/dft/dft-periodic-relaxation/",
            ),
            ("dft-production-core.md", "benchmarks/README.md", "/mlx-atomistic/benchmarks/"),
            ("benchmarks/README.md", "md-suite.md", "/mlx-atomistic/benchmarks/md-suite/"),
        ]
        for source, target, expected in cases:
            with self.subTest(source=source, target=target):
                source_path = (DOCS / source).resolve()
                self.assertEqual(
                    _site_link(
                        target,
                        source=source_path,
                        site_target=published[source_path],
                        published=published,
                    ),
                    expected,
                )

    def test_build_check_catches_markdown_and_nested_relative_links(self):
        with TemporaryDirectory() as directory:
            dist = Path(directory)
            page = dist / "dft" / "core" / "index.html"
            page.parent.mkdir(parents=True)
            target = dist / "dft" / "roadmap" / "index.html"
            target.parent.mkdir(parents=True)
            target.write_text("<h1>Roadmap</h1>")
            page.write_text(
                '<a href="./roadmap.md">broken file link</a>'
                '<a href="./roadmap/">wrong nesting</a>'
                '<a href="/mlx-atomistic/dft/roadmap/#next">valid published link</a>'
                '<a href="https://github.com/org/repo/blob/main/README.md">source</a>'
                '<a href="/">organization home</a>'
            )
            self.assertEqual(len(check_links(dist)), 2)
