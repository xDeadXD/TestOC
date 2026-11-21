import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";
import { installMockIpc } from "./lib/mockIpc";
import { isTauri } from "./lib/ipc";

// Enable mock IPC when launched with `VITE_MOCK_IPC=1` (e.g. browser/Playwright verification).
if (import.meta.env.VITE_MOCK_IPC === "1" || !isTauri()) {
  installMockIpc();
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
