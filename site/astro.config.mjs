import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import sitemap from "@astrojs/sitemap";
import starlightLlmsTxt from "starlight-llms-txt";
import { SITE, socialHead } from "./src/seo.mjs";

export default defineConfig({
  site: "https://appautomaton.renocrypt.com",
  base: "/mlx-atomistic",
  trailingSlash: "ignore",
  integrations: [
    starlight({
      title: "mlx-atomistic",
      description:
        "Apple Silicon-native atomistic simulation: MLX + Metal DFT and MD runtime.",
      // Social-card image + structured-data (@graph) on every docs page; the
      // entity graph lives in src/seo.mjs so it stays identical to the landing.
      head: socialHead,
      logo: {
        src: "./src/assets/logo.svg",
        replacesTitle: false,
      },
      favicon: "/favicon.svg",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/appautomaton/mlx-atomistic",
        },
      ],
      customCss: ["./src/styles/custom.css"],
      // Emits /llms.txt, /llms-full.txt, /llms-small.txt so an agentic harness
      // can ingest the whole library (narrative + auto-generated API) in one fetch.
      plugins: [
        starlightLlmsTxt({
          description:
            "Apple Silicon-native atomistic simulation: an MLX + Metal DFT and MD runtime that runs the GPU on your Mac.",
          promote: ["overview", "foundations/**"],
        }),
      ],
      sidebar: [
        { label: "Overview", slug: "overview" },
        {
          label: "Foundations",
          items: [{ autogenerate: { directory: "foundations" } }],
        },
        {
          label: "Molecular Mechanics",
          items: [{ autogenerate: { directory: "mm" } }],
        },
        {
          label: "Density Functional Theory",
          items: [{ autogenerate: { directory: "dft" } }],
        },
        {
          label: "Benchmarks",
          items: [{ autogenerate: { directory: "benchmarks" } }],
        },
        {
          label: "Project",
          items: [{ autogenerate: { directory: "project" } }],
        },
        {
          label: "API Reference",
          collapsed: true,
          items: [{ autogenerate: { directory: "api" } }],
        },
      ],
    }),
    // A sitemap is a list of addresses worth indexing, so two kinds of page
    // are kept out of it.
    //
    // The first is the slashless base. Under trailingSlash: "ignore" the base
    // index is emitted twice, as /mlx-atomistic/ and again without the slash,
    // and Pages answers the slashless form with a 301 to the other. Listing it
    // spends a crawl on an address that was never the destination, while the
    // page's own canonical has named the slashed form all along.
    //
    // The second is the generated API reference under /api/. Those pages carry
    // noindex, and asking a crawler to fetch a page in order to be told not to
    // index it is the whole cost with none of the benefit. They stay reachable
    // through the sidebar and the /api/ index, which is hand-written and stays
    // listed here.
    sitemap({
      filter: (page) =>
        page !== SITE && (!page.startsWith(`${SITE}/api/`) || page === `${SITE}/api/`),
    }),
  ],
});
