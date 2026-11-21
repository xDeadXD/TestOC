import { expect, test } from "@playwright/test";

const SHOT_DIR =
  (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env
    ?.SHOT_DIR || "test-results/screenshots";

test.describe("TestOC Dashboard (Mock IPC)", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("kpi-strip")).toBeVisible();
    await expect(page.getByTestId("core-tile-15")).toBeVisible();
  });

  test("1. renders TestOC header, all panels, 16 core tiles, and 4 uPlot canvases", async ({ page }) => {
    await expect(page.locator("header")).toContainText("TESTOC");
    await expect(page.locator("header")).toContainText("xDead");

    for (const id of [
      "kpi-strip",
      "stress-panel",
      "core-grid",
      "gpu-panel",
      "board-panel",
      "session-panel",
      "safety-panel",
      "chart-cpu-load",
      "chart-temps",
      "chart-clocks",
      "chart-power-mem",
    ]) {
      await expect(page.getByTestId(id)).toBeVisible();
    }
    await expect(page.locator('[data-testid^="core-tile-"]')).toHaveCount(16);

    const canvases = page.locator("canvas");
    await expect(canvases).toHaveCount(4);
    for (let i = 0; i < 4; i++) {
      const box = await canvases.nth(i).boundingBox();
      expect(box && box.width > 150 && box.height > 80).toBeTruthy();
    }

    await page.screenshot({ path: `${SHOT_DIR}/01-dashboard-idle.png`, fullPage: false });
  });

  test("2. live stream advances and 1m / 5m / 15m window switching works", async ({ page }) => {
    const counter = page.getByTestId("sample-counter");
    const n1 = Number((await counter.innerText()).replace(/\D/g, ""));
    await expect
      .poll(async () => Number((await counter.innerText()).replace(/\D/g, "")), { timeout: 3000 })
      .toBeGreaterThan(n1);

    await page.getByTestId("window-1m").click();
    await expect(page.getByTestId("window-1m")).toHaveAttribute("aria-pressed", "true");
    await page.getByTestId("window-15m").click();
    await expect(page.getByTestId("window-15m")).toHaveAttribute("aria-pressed", "true");
    await page.getByTestId("window-5m").click();
    await expect(page.getByTestId("window-5m")).toHaveAttribute("aria-pressed", "true");
  });

  test("3. per-core series toggles update uPlot legend series", async ({ page }) => {
    const loadLegendSeries = page.getByTestId("chart-cpu-load").locator(".u-legend .u-series");
    // Default: Time + Total + 4 selected cores = 6 legend entries
    await expect(loadLegendSeries).toHaveCount(6);

    await page.getByTestId("core-toggle-5").click();
    await expect(page.getByTestId("core-toggle-5")).toHaveAttribute("aria-pressed", "true");
    await expect(loadLegendSeries).toHaveCount(7);

    await page.getByTestId("cores-none-btn").click();
    await expect(loadLegendSeries).toHaveCount(2); // Time + Total

    await page.getByTestId("cores-all-btn").click();
    await expect(loadLegendSeries).toHaveCount(18); // Time + Total + 16 cores
  });

  test("4. temp ramp triggers warning banner; critical alert shows Aborted (CPU Temp Limit) + throttle pills", async ({ page }) => {
    await page.evaluate(() => {
      window.__TESTOC_MOCK__?.setGpuThrottles(["Power Cap", "Thermal Slowdown"]);
      window.__TESTOC_MOCK__?.setCpuTemp(91.2);
    });

    await expect(page.getByTestId("gpu-throttle-pills")).toContainText("Thermal Slowdown");
    const banner = page.getByTestId("alert-banner");
    await expect(banner).toBeVisible();
    await expect(banner).toHaveAttribute("data-level", "warning");
    await expect(banner).toContainText("91.2");

    await page.evaluate(() => {
      window.__TESTOC_MOCK__?.emitAlert(
        {
          level: "critical",
          metric: "cpuTemp",
          value: 96.1,
          threshold: 95,
          action: "aborted",
          timestampMs: Date.now(),
          message: "CPU temperature 96.1 °C reached abort limit 95 °C",
        },
        "cpuTemp"
      );
    });

    await expect(banner).toHaveAttribute("data-level", "critical");
    await expect(page.getByTestId("aborted-badge")).toContainText("Aborted (CPU Temp Limit)");
    await page.screenshot({ path: `${SHOT_DIR}/02-thermal-abort-alert.png`, fullPage: false });
  });

  test("5. ABORT click and Esc hotkey both call emergency_abort", async ({ page }) => {
    await page.getByTestId("abort-btn").click();
    await page.keyboard.press("Escape");

    const abortCount = await page.evaluate(
      () => window.__TESTOC_MOCK__?.calls.filter((c) => c.cmd === "emergency_abort").length ?? 0
    );
    expect(abortCount).toBe(2);
  });

  test("6. declined sidecar shows retry + NoThermalModal passes acknowledgeNoThermal: true", async ({ page }) => {
    await page.evaluate(() => {
      window.__TESTOC_MOCK__?.setSidecarState("declined", "Elevation was declined");
    });

    await expect(page.getByTestId("sidecar-badge")).toContainText("UAC Declined");
    await expect(page.getByTestId("retry-sidecar-btn")).toBeVisible();

    await page.getByTestId("start-stress-btn").click();
    await expect(page.getByTestId("no-thermal-modal")).toBeVisible();
    await page.getByTestId("confirm-no-thermal-btn").click();
    await expect(page.getByTestId("no-thermal-modal")).toBeHidden();

    const ackCall = await page.evaluate(() =>
      window.__TESTOC_MOCK__?.calls.find(
        (c) => c.cmd === "start_stress" && (c.args as Record<string, unknown>)?.acknowledgeNoThermal === true
      )
    );
    expect(ackCall).toBeTruthy();

    await page.getByTestId("abort-btn").click();
    await page.getByTestId("retry-sidecar-btn").click();
    await expect(page.getByTestId("sidecar-badge")).toContainText("OK");
  });

  test("7. Absolut 32-step RAM profile, CPU Extreme/FFT/CoreCycler, GPU profiles & Power/VRAM match StressConfig", async ({ page }) => {
    // Reject out-of-range thread count client-side
    await page.getByTestId("threads-input").fill("99");
    await page.getByTestId("start-stress-btn").click();
    await expect(page.getByTestId("form-validation-error")).toContainText("1–16");

    // Verify CPU options: Extreme default, FFT Torture presets (Smallest-Large-Blend), CoreCycler, clean Stop
    await expect(page.getByTestId("cpu-workload")).toHaveValue("extreme");
    await expect(page.getByTestId("fft-preset-smallest")).toBeVisible();
    await expect(page.getByTestId("fft-preset-large")).toBeVisible();
    await page.getByTestId("cpu-workload").selectOption("fftTorture");
    await page.getByTestId("fft-preset-smallest").click();
    await page.getByTestId("cpu-workload").selectOption("extreme");
    await page.getByTestId("fft-preset-small").click();
    await page.getByTestId("cpu-mode").selectOption("coreCycler");
    await expect(page.getByTestId("cpu-core-cycle-s")).toHaveValue("3");
    await page.getByTestId("cpu-mode").selectOption("steady");
    await page.getByTestId("cpu-dataset").selectOption("medium");
    await page.getByTestId("physical-cores-checkbox").check();
    await expect(page.getByTestId("threads-input")).toHaveValue("8");
    await page.getByTestId("start-stress-btn").click();
    await expect(page.getByTestId("stop-stress-btn")).toHaveText("Stop");
    await page.getByTestId("stop-stress-btn").click();
    await expect(page.getByTestId("stress-phase-badge")).toContainText("stopped");
    await page.getByTestId("physical-cores-checkbox").uncheck();

    // Verify CPU+RAM exposes FFT presets (Smallest -> Large -> Blend) alongside Absolut RAM
    await page.getByTestId("kind-combined").click();
    await expect(page.getByTestId("fft-preset-blend")).toBeVisible();
    await expect(page.getByTestId("mem-profile-absolut")).toBeVisible();
    await page.getByTestId("fft-preset-large").click();
    await page.getByTestId("fft-preset-blend").click();

    // Switch to GPU test -> verify Extreme 3D default, Heavy/Standard profiles, & 3D Adaptive mode
    await page.getByTestId("kind-gpu").click();
    await expect(page.getByTestId("gpu-profile-extreme")).toBeVisible();
    await expect(page.getByTestId("gpu-profile-heavy")).toBeVisible();
    await expect(page.getByTestId("gpu-profile-standard")).toBeVisible();
    await page.getByTestId("gpu-mode").selectOption("variable");
    await page.getByTestId("gpu-var-min").fill("30");
    await page.getByTestId("gpu-var-max").fill("95");
    await page.getByTestId("gpu-var-step").fill("4");
    await page.getByTestId("dur-preset-15").click();
    await page.waitForTimeout(200);
    await page.screenshot({ path: `${SHOT_DIR}/04-gpu-preferences.png`, fullPage: false });
    await page.getByTestId("start-stress-btn").click();
    await expect(page.getByTestId("stress-phase-badge")).toContainText("running");

    const gpuCall = await page.evaluate(() =>
      window.__TESTOC_MOCK__?.calls.find(
        (c) => c.cmd === "start_stress" && ((c.args as { config?: { kind?: string } })?.config?.kind === "gpu")
      )
    );
    expect(gpuCall?.args).toMatchObject({
      config: {
        kind: "gpu",
        mode: "variable",
        profile: "extreme",
        intensity: 100,
        variableMinPct: 30,
        variableMaxPct: 95,
        variableStepS: 4,
        errorCheck: true,
        stopOnError: false,
        maxErrors: null,
        durationS: 900,
      },
      acknowledgeNoThermal: false,
    });
    await page.getByTestId("stop-stress-btn").click();
    await expect(page.getByTestId("stress-phase-badge")).toContainText("stopped");

    // Switch to Power test (CPU Extreme + GPU Extreme 3D) with duration reset to Unlimited
    await page.getByTestId("kind-power").click();
    await page.getByTestId("dur-preset-inf").click();
    await page.getByTestId("threads-input").fill("16");
    await page.getByTestId("cpu-dataset").selectOption("small");
    await page.getByTestId("power-gpu-profile-extreme").click();
    await page.getByTestId("max-errors-input").fill("10");
    await page.getByTestId("start-stress-btn").click();
    await expect(page.getByTestId("stress-phase-badge")).toContainText("running");

    await page.waitForTimeout(700);
    await page.screenshot({ path: `${SHOT_DIR}/03-power-stress-running.png`, fullPage: false });

    const powerCall = await page.evaluate(() =>
      window.__TESTOC_MOCK__?.calls.find(
        (c) => c.cmd === "start_stress" && ((c.args as { config?: { kind?: string } })?.config?.kind === "power")
      )
    );
    expect(powerCall?.args).toMatchObject({
      config: {
        kind: "power",
        cpuThreads: 16,
        cpuIsa: "auto",
        cpuDataset: "small",
        cpuWorkload: "extreme",
        gpuProfile: "extreme",
        gpuIntensity: 100,
        stopOnError: false,
        maxErrors: 10,
        durationS: null,
      },
      acknowledgeNoThermal: false,
    });

    await page.getByTestId("stop-stress-btn").click();
    await expect(page.getByTestId("stress-phase-badge")).toContainText("stopped");

    // Switch to VRAM test with 100% target preset & pattern preference
    await page.getByTestId("kind-vram").click();
    await page.getByTestId("vram-preset-100").click();
    await page.getByTestId("vram-pattern").selectOption("random");
    await page.getByTestId("stop-on-error-checkbox").check();
    await page.getByTestId("start-stress-btn").click();
    const vramCall = await page.evaluate(() =>
      window.__TESTOC_MOCK__?.calls.find(
        (c) => c.cmd === "start_stress" && ((c.args as { config?: { kind?: string } })?.config?.kind === "vram")
      )
    );
    expect(vramCall?.args).toMatchObject({
      config: { kind: "vram", targetPct: 100, pattern: "random", stopOnError: true, maxErrors: null, durationS: null },
    });
    await page.getByTestId("stop-stress-btn").click();

    // Switch to Memory test -> Absolut (32-Step) default at 100% RAM, 3C preset, live stage + RAM Alloc tile
    await page.getByTestId("kind-memory").click();
    await expect(page.getByTestId("mem-pct-slider")).toHaveValue("100");
    await expect(page.getByTestId("mem-profile-absolut")).toBeVisible();
    await expect(page.getByTestId("mem-profile-standard")).toBeVisible();
    await page.getByTestId("mem-cycles-3").click();
    await expect(page.getByTestId("match-ram-cap-btn")).toBeHidden();
    await page.getByTestId("start-stress-btn").click();
    await expect(page.getByTestId("allocated-bytes-tile")).toBeVisible();
    await expect(page.getByTestId("stress-live-stage")).toContainText("Absolut C1/3");
    const memAbsCall = await page.evaluate(() =>
      window.__TESTOC_MOCK__?.calls.find(
        (c) => c.cmd === "start_stress" && ((c.args as { config?: { kind?: string } })?.config?.kind === "memory")
      )
    );
    expect(memAbsCall?.args).toMatchObject({
      config: { kind: "memory", targetPct: 100, profile: "absolut", cycles: 3 },
    });
    await page.getByTestId("stop-stress-btn").click();
    await expect(page.getByTestId("stress-phase-badge")).toContainText("stopped");

    // Verify profile switcher to Standard exposes selectable pattern dropdown
    await page.getByTestId("mem-profile-standard").click();
    await page.getByTestId("mem-pattern").selectOption("movingInversions");
    await page.getByTestId("mem-profile-absolut").click();

    // Verify Safety Limits label has no internal "(Rust)" suffix and accepts enthusiast limits
    await expect(page.getByTestId("safety-panel")).toContainText("Safety Limits");
    await expect(page.getByTestId("safety-panel")).not.toContainText("Rust");
    await page.getByTestId("cpu-abort-input").fill("105");
    await page.getByTestId("gpu-abort-input").fill("100");
    await page.getByTestId("thermal-abort-toggle").uncheck();
    await page.getByTestId("save-limits-btn").click();
    const limCall = await page.evaluate(() =>
      window.__TESTOC_MOCK__?.calls.filter((c) => c.cmd === "set_safety_limits").at(-1)
    );
    expect(limCall?.args).toMatchObject({
      limits: { cpuAbortC: 105, gpuAbortC: 100, memMaxPct: 100, thermalAbortEnabled: false },
    });
  });

  test("8. CSV export invokes save dialog and export_session with chosen path", async ({ page }) => {
    await page.getByTestId("session-start-btn").click();
    await page.waitForTimeout(700);
    await page.getByTestId("export-csv-btn").click();
    await expect(page.getByTestId("export-result")).toContainText("Exported");

    const expCall = await page.evaluate(() => window.__TESTOC_MOCK__?.calls.find((c) => c.cmd === "export_session"));
    expect(expCall?.args).toEqual({ csvPath: "C:\\Temp\\testoc-mock.csv" });
  });
});
