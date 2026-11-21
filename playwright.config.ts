import { defineConfig } from "@playwright/test";

// Browser UI verification against the mock IPC layer (no Rust needed).
// Uses the installed Microsoft Edge, so no browser download is required.
export default defineConfig({
  testDir: "e2e",
  timeout: 30_000,
  fullyParallel: false,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:1420",
    channel: "msedge",
    viewport: { width: 1480, height: 940 },
    screenshot: "only-on-failure",
  },
  webServer: {
    command: "npm run dev:mock",
    url: "http://localhost:1420",
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
