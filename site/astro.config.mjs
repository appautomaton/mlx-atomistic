import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import starlightLlmsTxt from "starlight-llms-txt";

export default defineConfig({
  site: "https://appautomaton.github.io",
  base: "/mlx-atomistic",
  trailingSlash: "ignore",
  integrations: [
    starlight({
      title: "mlx-atomistic",
      description:
        "Apple Silicon-native atomistic simulation: MLX + Metal DFT and MD runtime.",
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
  ],
});
