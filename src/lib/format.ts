export const fmt = {
  num(v: number | null | undefined, digits = 0, unit = ""): string {
    if (v == null || !Number.isFinite(v)) return "—";
    return `${v.toFixed(digits)}${unit}`;
  },
  mhz(v: number | null | undefined): string {
    if (v == null || !Number.isFinite(v)) return "—";
    return v >= 1000 ? `${(v / 1000).toFixed(2)} GHz` : `${v.toFixed(0)} MHz`;
  },
  bytes(v: number | null | undefined): string {
    if (v == null || !Number.isFinite(v)) return "—";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let i = 0;
    let x = v;
    while (x >= 1024 && i < units.length - 1) {
      x /= 1024;
      i++;
    }
    return `${x.toFixed(i >= 3 ? 1 : 0)} ${units[i]}`;
  },
  duration(sec: number | null | undefined): string {
    if (sec == null || !Number.isFinite(sec) || sec < 0) return "—";
    const s = Math.floor(sec);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const r = s % 60;
    const mm = String(m).padStart(2, "0");
    const ss = String(r).padStart(2, "0");
    return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
  },
  throughput(mbps: number | null | undefined): string {
    if (mbps == null || !Number.isFinite(mbps)) return "—";
    return mbps >= 1000 ? `${(mbps / 1000).toFixed(1)} GB/s` : `${mbps.toFixed(0)} MB/s`;
  },
};
