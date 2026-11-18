import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import sitemap from "@astrojs/sitemap";

// Replace with the custom domain once one is set up.
export default defineConfig({
  site: "https://brakenews.pages.dev",
  output: "static",
  integrations: [
    // The 3D explorer (src/components/three) is imported on demand from a plain script, not
    // rendered as an Astro island, so the dev server never adds React's Fast Refresh preamble
    // and the import fails ("can't detect preamble"). Keeping those files out of the React
    // plugin compiles their JSX normally, without the refresh wrapper. Production is unaffected.
    react({ exclude: ["**/components/three/**"] }),
    sitemap(),
  ],
});
