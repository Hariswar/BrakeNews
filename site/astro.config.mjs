import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import sitemap from "@astrojs/sitemap";

// Replace with the custom domain once one is set up.
export default defineConfig({
  site: "https://brakenews.pages.dev",
  output: "static",
  integrations: [react(), sitemap()],
});
