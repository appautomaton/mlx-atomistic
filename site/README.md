# mlx-atomistic site

Astro + Starlight site for https://appautomaton.renocrypt.com/mlx-atomistic.

## Local dev

Use Node 24, matching the Pages workflow.

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

## Deploy

`.github/workflows/deploy-site.yml` generates narrative and API pages, builds,
and deploys to GitHub Pages. Narrative pages are not edited here: update
`../docs/`, then run `scripts/sync_site_docs.py`.
