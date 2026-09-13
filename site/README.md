# mlx-atomistic site

Astro + Starlight site for https://appautomaton.com/mlx-atomistic.

## Local dev

Use Node 24 for reproducible site builds.

```bash
cd site
npm ci
cd ..
uv run --no-project --python 3.13.12 python scripts/sync_site_docs.py
uv run --no-project --with griffe --python 3.13.12 python scripts/gen_api_docs.py
cd site
npm run dev      # http://localhost:4321/mlx-atomistic/
npm run build    # outputs to dist/
npm run preview  # preview the build
```

## Structure

- `src/pages/index.astro` — custom landing page (floating nav + bento grid + hero)
- `src/styles/custom.css` — 2026 palette overrides for Starlight
- `src/content/docs/` — generated from canonical `../docs/` plus package docstrings
- `astro.config.mjs` — site config, sidebar, base path

## Central publication

The public website is published by `appautomaton/appautomaton.github.io` from
`sites/mlx-atomistic/public/`. This repository retains the canonical technical
documentation, package docstrings, generators, and Astro inputs so that future
documentation updates remain reproducible. It no longer deploys GitHub Pages.

Update `../docs/` or package docstrings, run both generators, and build using
the commands above. Validate the generated output, then replace the complete
central `sites/mlx-atomistic/public/` tree with this build's `site/dist/` in a
reviewed central PR. Copy the full Pagefind index together with its entry and
metadata files; never mix files from different builds. Existing central
baseline hashes describe the historical import, so document intentional later
content changes rather than silently replacing that baseline.

Links between narrative pages are emitted as published `/mlx-atomistic/.../`
URLs, including directory indexes and fragments. Links to repository-only
files continue to point at GitHub. After building, run this from the repository
root to catch internal destinations missing from the output:

```bash
uv run --no-project --python 3.13.12 python scripts/check_site_links.py
```

Run this check before proposing a central content refresh. Central CI validates
the assembled artifact before publishing; runtime changes alone do not update
the published documentation snapshot.
