// Typed access to the Node API, plus the small numeric helpers the screens share.

export const CATS = ["traffic", "industry", "construction", "burning"] as const;
export type Cat = (typeof CATS)[number];
export const CAT_LABEL: Record<string, string> = {
  traffic: "Traffic",
  industry: "Industry",
  construction: "Construction",
  burning: "Burning",
  regional: "Regional",
  unmapped: "Not on the map",
};
export const CAT_COLOR: Record<string, string> = {
  traffic: "#60a5fa",
  industry: "#f59e0b",
  construction: "#a78bfa",
  burning: "#f87171",
  regional: "#94a3b8",
  unmapped: "#38bdf8",
};

export interface Monitor { name: string; full: string; provider: string; x: number; y: number; iy: number; ix: number; mean: number }
export interface MapData {
  width_km: number; height_km: number; nx: number; ny: number; block: number;
  roads: { w: number; p: [number, number][] }[];
  landuse: Record<string, [number, number][][]>;
  monitors: Monitor[];
}
export interface PriorityRow {
  by: number; bx: number; area?: string; cat: string; likely: string; g_s: number;
  hist?: (boolean | null)[]; action?: string; sigma?: number; p?: number;
}
export interface RealMeta {
  n: number; nx: number; ny: number; nb: number; window: number; scale: number;
  local_hour: number[]; valid: boolean[]; times: string[]; bg: number[]; bg_std: number[];
  u: number[]; v: number[]; H: number[]; readings: (number | null)[][]; priority: PriorityRow[][];
}
export interface RealWindow { kriging: Float32Array; c_cat: Float32Array[]; std_blocks: Float32Array[]; E_blocks: Float32Array[] }

export async function getJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof body.error === "string" ? body.error : typeof body.detail === "string" ? body.detail : "Request failed");
  return body as T;
}

export function postJSON<T>(url: string, payload: unknown, signal?: AbortSignal): Promise<T> {
  return getJSON<T>(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload), signal });
}

const windowCache = new Map<number, Promise<RealWindow>>();

/** One window of the real month, decoded from uint16 (see real_meta.layout). Keeps the latest 96 windows. */
export function getWindow(meta: RealMeta, i: number): Promise<RealWindow> {
  const hit = windowCache.get(i);
  if (hit) return hit;
  const p = fetch(`/api/real/window/${i}`).then(async (r) => {
    if (!r.ok) throw new Error("Couldn't load this hour");
    const raw = new Uint16Array(await r.arrayBuffer());
    const full = meta.nx * meta.ny, blocks = meta.nb * meta.nb;
    const take = (o: number, n: number) => Float32Array.from(raw.subarray(o, o + n), (v) => v / meta.scale);
    let o = 0;
    const kriging = take(o, full); o += full;
    const c_cat = CATS.map(() => { const a = take(o, full); o += full; return a; });
    const std_blocks = CATS.map(() => { const a = take(o, blocks); o += blocks; return a; });
    const E_blocks = CATS.map(() => { const a = take(o, blocks); o += blocks; return a; });
    return { kriging, c_cat, std_blocks, E_blocks };
  });
  windowCache.set(i, p);
  if (windowCache.size > 96) windowCache.delete(windowCache.keys().next().value as number);
  return p;
}

/** IST end time of a real window: station period k covers [t + 0:30, t + 1:30) UTC, so it ends at t + 7 h IST. */
export function windowEnd(t: string): { date: string; time: string } {
  const d = new Date(`${t.slice(0, 13)}:00:00Z`);
  d.setUTCHours(d.getUTCHours() + 7);
  return {
    date: d.toLocaleDateString("en-IN", { day: "numeric", month: "short", timeZone: "UTC" }),
    time: d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "UTC" }),
  };
}

export const flat2 = (rows: number[][]): Float32Array => Float32Array.from(rows.flat());

export function median(values: number[]): number {
  if (!values.length) return NaN;
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export interface ShareBar { key: string; pct: number; lo: number; hi: number }

/** Shares at a point from its parts; the band is first-order error propagation of each part's s.d., clipped to 0-100%. */
export function sharesFrom(parts: number[], stds: number[], bg: number, bgStd: number): ShareBar[] {
  const vals = [...parts.map((v) => Math.max(v, 0)), Math.max(bg, 0)];
  const sds = [...stds, bgStd];
  const total = vals.reduce((a, b) => a + b, 0) || 1;
  const keys = [...CATS, "regional"];
  return keys.map((key, k) => {
    const pct = (100 * vals[k]) / total;
    const sd = (100 * sds[k]) / total;
    return { key, pct, lo: Math.max(pct - 1.645 * sd, 0), hi: Math.min(pct + 1.645 * sd, 100) };
  }).sort((a, b) => b.pct - a.pct);
}

/** Indian NAQI PM2.5 bands (24 h breakpoints, used here as a guide for hourly values). */
export function aqiBand(v: number): { label: string; color: string } {
  if (v <= 30) return { label: "Good", color: "#34d399" };
  if (v <= 60) return { label: "Satisfactory", color: "#a3e635" };
  if (v <= 90) return { label: "Moderate", color: "#facc15" };
  if (v <= 120) return { label: "Poor", color: "#fb923c" };
  if (v <= 250) return { label: "Very poor", color: "#f87171" };
  return { label: "Severe", color: "#e879f9" };
}
