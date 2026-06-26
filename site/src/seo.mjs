// Shared SEO/GEO constants — imported by both astro.config.mjs (Starlight pages)
// and src/pages/index.astro (the standalone landing page) so the structured-data
// entity graph is defined exactly once and never drifts between pages.

export const ORIGIN = "https://appautomaton.github.io";
export const SITE = `${ORIGIN}/mlx-atomistic`;
export const REPO = "https://github.com/appautomaton/mlx-atomistic";
export const OG_IMAGE = `${SITE}/og.png`;

const DESCRIPTION =
  "Apple Silicon-native molecular dynamics and density-functional-theory " +
  "runtime built directly on MLX and Metal — it runs the GPU on your Mac, " +
  "with no CUDA, server, or cloud.";

const KEYWORDS = [
  "MLX",
  "Apple Silicon",
  "Metal",
  "molecular dynamics",
  "density functional theory",
  "DFT",
  "computational chemistry",
  "atomistic simulation",
  "GPU",
  "Python",
];

// Stable @id fragments connect the entities into one knowledge graph (the
// @graph + @id pattern recommended for 2026: define core entities once,
// reference by @id elsewhere).
const ORG_ID = `${ORIGIN}/#organization`;
const SITE_ID = `${SITE}/#website`;
const SOFTWARE_ID = `${SITE}/#software`;

export const jsonLd = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "Organization",
      "@id": ORG_ID,
      name: "App Automaton",
      url: ORIGIN,
      description:
        "Open skills, harnesses, and on-device tools for engineering with AI coding agents.",
    },
    {
      "@type": "WebSite",
      "@id": SITE_ID,
      url: `${SITE}/`,
      name: "mlx-atomistic",
      description: DESCRIPTION,
      inLanguage: "en",
      publisher: { "@id": ORG_ID },
      isPartOf: { "@id": ORG_ID },
    },
    {
      "@type": "SoftwareSourceCode",
      "@id": SOFTWARE_ID,
      name: "mlx-atomistic",
      description: DESCRIPTION,
      url: `${SITE}/`,
      codeRepository: REPO,
      programmingLanguage: "Python",
      runtimePlatform: "MLX / Metal",
      operatingSystem: "macOS (Apple Silicon)",
      license: "https://opensource.org/licenses/MIT",
      keywords: KEYWORDS,
      author: { "@id": ORG_ID },
      maintainer: { "@id": ORG_ID },
      isPartOf: { "@id": ORG_ID },
    },
  ],
};

// Default social-card meta shared across pages. Starlight already emits
// og:title/og:description/twitter:title/twitter:description per page; these add
// the image + identity tags it does not set by default.
export const socialHead = [
  { tag: "meta", attrs: { property: "og:site_name", content: "mlx-atomistic" } },
  { tag: "meta", attrs: { property: "og:image", content: OG_IMAGE } },
  { tag: "meta", attrs: { property: "og:image:width", content: "1200" } },
  { tag: "meta", attrs: { property: "og:image:height", content: "630" } },
  { tag: "meta", attrs: { property: "og:image:alt", content: "mlx-atomistic — Apple Silicon atomistic simulation" } },
  { tag: "meta", attrs: { name: "twitter:card", content: "summary_large_image" } },
  { tag: "meta", attrs: { name: "twitter:image", content: OG_IMAGE } },
  { tag: "script", attrs: { type: "application/ld+json" }, content: JSON.stringify(jsonLd) },
];
