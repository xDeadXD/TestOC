import { describe, expect, it } from "vitest";
import { SlidingColumns } from "./ringBuffer";

describe("SlidingColumns", () => {
  it("returns oldest-first windows", () => {
    const c = new SlidingColumns(5);
    for (let i = 0; i < 3; i++) c.push({ t: i, v: i * 10 });
    expect(c.window("t", 10)).toEqual([0, 1, 2]);
    expect(c.window("v", 2)).toEqual([10, 20]);
  });

  it("wraps around without losing order and caps at capacity", () => {
    const c = new SlidingColumns(4);
    for (let i = 0; i < 23; i++) c.push({ t: i });
    expect(c.window("t", 100)).toEqual([19, 20, 21, 22]);
    expect(c.window("t", 2)).toEqual([21, 22]);
    expect(c.latest("t")).toBe(22);
    expect(c.length).toBeLessThanOrEqual(8);
  });

  it("back-fills late columns and maps non-finite to null", () => {
    const c = new SlidingColumns(10);
    c.push({ t: 0 });
    c.push({ t: 1, temp: 50 });
    c.push({ t: 2, temp: Number.NaN });
    c.push({ t: 3 });
    expect(c.window("temp", 10)).toEqual([null, 50, null, null]);
    expect(c.window("missing", 3)).toEqual([null, null, null]);
  });

  it("supports window sizes for 1/5/15 min at 2 Hz", () => {
    const c = new SlidingColumns(1800);
    for (let i = 0; i < 5000; i++) c.push({ t: i });
    expect(c.window("t", 120)).toHaveLength(120);
    expect(c.window("t", 600)[0]).toBe(4400);
    expect(c.window("t", 1800)).toHaveLength(1800);
    expect(c.window("t", 1800).at(-1)).toBe(4999);
  });
});
