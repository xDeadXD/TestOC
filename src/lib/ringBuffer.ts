/**
 * Append-only column store with a sliding window, sized for uPlot.
 *
 * Backing arrays hold up to 2x capacity; when full, the newest `capacity` rows are shifted to the
 * front with `copyWithin` (amortized O(1) per push), so a window is always one contiguous slice.
 */
export type Cell = number | null;

export class SlidingColumns {
  readonly capacity: number;
  private cols = new Map<string, Cell[]>();
  private len = 0;

  constructor(capacity: number) {
    if (!(capacity > 0)) throw new Error("capacity must be > 0");
    this.capacity = capacity;
  }

  get length(): number {
    return this.len;
  }

  has(key: string): boolean {
    return this.cols.has(key);
  }

  keys(): string[] {
    return [...this.cols.keys()];
  }

  /** Push one row. Columns not present in `row` get `null`; new columns are back-filled with `null`. */
  push(row: Record<string, Cell>): void {
    if (this.len === this.capacity * 2) {
      for (const col of this.cols.values()) col.copyWithin(0, this.capacity, this.capacity * 2);
      this.len = this.capacity;
    }
    for (const key of Object.keys(row)) {
      if (!this.cols.has(key)) this.cols.set(key, new Array<Cell>(this.capacity * 2).fill(null));
    }
    for (const [key, col] of this.cols) {
      const v = row[key];
      col[this.len] = v === undefined || (typeof v === "number" && !Number.isFinite(v)) ? null : v;
    }
    this.len++;
  }

  /** Last `n` rows of `key` (oldest first). Unknown keys yield an all-null column. */
  window(key: string, n: number): Cell[] {
    const count = Math.min(n, this.len, this.capacity);
    const col = this.cols.get(key);
    if (!col) return new Array<Cell>(count).fill(null);
    return col.slice(this.len - count, this.len);
  }

  latest(key: string): Cell {
    if (this.len === 0) return null;
    return this.cols.get(key)?.[this.len - 1] ?? null;
  }

  clear(): void {
    this.cols.clear();
    this.len = 0;
  }
}
