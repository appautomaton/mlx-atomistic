# mlx-atomistic site

Astro + Starlight site for https://appautomaton.github.io/mlx-atomistic.

## Local dev

```bash
cd site
npm install
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
to `main` when `site/**` changes.
