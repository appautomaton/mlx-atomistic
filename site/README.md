# mlx-atomistic site

Astro + Starlight site for https://appautomaton.github.io/mlx-atomistic.

## Local dev

Use Node 24, matching the Pages workflow.

```bash
cd site
npm ci
cd ..
uv run --no-project --with griffe --python 3.13.12 python scripts/gen_api_docs.py
cd site
npm run dev      # http://localhost:4321/mlx-atomistic/
npm run build    # outputs to dist/
npm run preview  # preview the build
```

## Structure

- `src/pages/index.astro` — custom landing page (floating nav + bento grid + hero)
- `src/styles/custom.css` — 2026 palette overrides for Starlight
- `src/content/docs/` — migrated from `../docs/` with Starlight frontmatter
- `astro.config.mjs` — site config, sidebar, base path

## Deploy

`.github/workflows/deploy-site.yml` builds and deploys to GitHub Pages on push
to `main` when `site/**`, `src/**`, `scripts/gen_api_docs.py`, `pyproject.toml`,
or the deploy workflow changes.
