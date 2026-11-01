// Map on the model grid (x, y in km), drawn on a 2D canvas. All geometry comes from the cached OSM data, so no tiles or
// network are needed, and no WebGL: browsers cap WebGL contexts, which the Scenario lab's several maps would run into.
import { useEffect, useRef, useState } from "react";
import type { MapData } from "../lib/data";

/** Magma-like ramp; perceptually ordered and readable on a projector. */
const STOPS: [number, [number, number, number]][] = [
  [0, [12, 14, 40]], [0.2, [60, 26, 110]], [0.4, [135, 42, 120]], [0.6, [205, 66, 90]], [0.8, [248, 130, 70]], [1, [252, 230, 170]],
];
const LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 0;
    while (k < STOPS.length - 2 && t > STOPS[k + 1][0]) k++;
    const f = (t - STOPS[k][0]) / (STOPS[k + 1][0] - STOPS[k][0]);
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = STOPS[k][1][c] + f * (STOPS[k + 1][1][c] - STOPS[k][1][c]);
  }
  return lut;
})();
export const rampCss = `linear-gradient(90deg, ${STOPS.map(([p, c]) => `rgb(${c.join(",")}) ${p * 100}%`).join(",")})`;

/** Bilinear upsample then colour map, so colours never blend across the ramp. Row 0 of ``field`` is south. */
function fieldImage(field: Float32Array, n: number, vmax: number, up = 4): ImageData {
  const N = n * up;
  const img = new ImageData(N, N);
  const scale = 255 / vmax;
  for (let j = 0; j < N; j++) {
    const fy = Math.min(Math.max((N - 1 - j + 0.5) / up - 0.5, 0), n - 1.001);
    const y0 = fy | 0, ty = fy - y0;
    for (let i = 0; i < N; i++) {
      const fx = Math.min(Math.max((i + 0.5) / up - 0.5, 0), n - 1.001);
      const x0 = fx | 0, tx = fx - x0;
      const a = y0 * n + x0;
      const v = (field[a] * (1 - tx) + field[a + 1] * tx) * (1 - ty) + (field[a + n] * (1 - tx) + field[a + n + 1] * tx) * ty;
      const c = Math.min(Math.max(v * scale, 0), 255) | 0;
      const o = (j * N + i) * 4;
      img.data[o] = LUT[c * 3];
      img.data[o + 1] = LUT[c * 3 + 1];
      img.data[o + 2] = LUT[c * 3 + 2];
      img.data[o + 3] = 255;
    }
  }
  return img;
}

export interface Point { x: number; y: number; color?: [number, number, number]; radius?: number; label?: string }
export interface Block { by: number; bx: number; color: [number, number, number]; selected?: boolean }

const rgb = (c: [number, number, number], a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

/** Roads and land use, cached per map and pixel size because they rarely change and take the longest to draw. */
const baseCache = new Map<string, HTMLCanvasElement>();
function baseLayer(map: MapData, px: number): HTMLCanvasElement {
  const key = `${map.roads.length}-${px}`;
  const hit = baseCache.get(key);
  if (hit) return hit;
  const c = document.createElement("canvas");
  c.width = c.height = px;
  const g = c.getContext("2d")!;
  const s = px / map.width_km;
  g.setTransform(s, 0, 0, -s, 0, px);
  g.fillStyle = "rgba(255,255,255,0.10)";
  for (const poly of (map.landuse.industrial ?? []).concat(map.landuse.landfill ?? [])) {
    g.beginPath();
    poly.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
    g.fill();
  }
  g.strokeStyle = "rgba(226,232,240,0.30)";
  g.lineJoin = "round";
  for (const r of map.roads) {
    g.lineWidth = ((r.w >= 1 ? 1.6 : r.w >= 0.7 ? 1 : 0.5) * px) / 600 / s;
    g.beginPath();
    r.p.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
    g.stroke();
  }
  baseCache.set(key, c);
  return c;
}

export function FieldMap({ map, field, vmax = 500, points = [], blocks = [], hollow = [], marks = [], wind, picked, onPick, onHover, cursor, showRoads = true, label, compact }: {
  map: MapData;
  field?: Float32Array | null;
  vmax?: number;
  points?: Point[];
  blocks?: Block[];
  hollow?: Point[];
  marks?: Point[];
  wind?: { u: number; v: number } | null;
  picked?: { x: number; y: number } | null;
  onPick?: (x: number, y: number) => void;
  onHover?: (p: Point | null) => void;
  cursor?: string;
  showRoads?: boolean;
  label?: string;
  compact?: boolean;
}) {
  const W = map.width_km;
  const bkm = (map.block * W) / map.nx;
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState(0);
  useEffect(() => {
    if (!box.current) return;
    const r = box.current.getBoundingClientRect(); // measure now: the observer only fires on a rendered frame
    setSize(Math.floor(Math.min(r.width, r.height)));
    const ro = new ResizeObserver(([e]) => setSize(Math.floor(Math.min(e.contentRect.width, e.contentRect.height))));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const c = canvas.current;
    if (!c || size < 10) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const px = Math.round(size * dpr);
    if (c.width !== px) c.width = c.height = px;
    const g = c.getContext("2d")!;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.fillStyle = "#0c1120";
    g.fillRect(0, 0, px, px);
    if (field) {
      const img = fieldImage(field, map.nx, vmax);
      const tmp = document.createElement("canvas");
      tmp.width = tmp.height = img.width;
      tmp.getContext("2d")!.putImageData(img, 0, 0);
      g.imageSmoothingEnabled = true;
      g.drawImage(tmp, 0, 0, px, px);
    }
    if (showRoads) g.drawImage(baseLayer(map, px), 0, 0);
    const s = px / W;
    const X = (x: number) => x * s, Y = (y: number) => px - y * s;
    for (const b of blocks) {
      const a = bkm * s;
      g.fillStyle = rgb(b.color, b.selected ? 0.25 : 0.1);
      g.strokeStyle = rgb(b.color);
      g.lineWidth = (b.selected ? 3 : 1.5) * dpr;
      g.fillRect(X(b.bx * bkm), Y((b.by + 1) * bkm), a, a);
      g.strokeRect(X(b.bx * bkm), Y((b.by + 1) * bkm), a, a);
    }
    for (const p of hollow) {
      g.strokeStyle = rgb(p.color ?? [248, 113, 113]);
      g.lineWidth = 2 * dpr;
      g.beginPath();
      g.arc(X(p.x), Y(p.y), 6 * dpr, 0, 2 * Math.PI);
      g.stroke();
    }
    for (const p of points) {
      g.fillStyle = rgb(p.color ?? [232, 236, 244]);
      g.strokeStyle = "#080d17";
      g.lineWidth = dpr;
      g.beginPath();
      g.arc(X(p.x), Y(p.y), (p.radius ?? (compact ? 2.5 : 4)) * dpr, 0, 2 * Math.PI);
      g.fill();
      g.stroke();
    }
    for (const m of marks) {
      const r = (compact ? 4 : 6) * dpr;
      g.strokeStyle = rgb(m.color ?? [45, 212, 191]);
      g.lineWidth = 2.5 * dpr;
      g.beginPath();
      g.moveTo(X(m.x) - r, Y(m.y) - r);
      g.lineTo(X(m.x) + r, Y(m.y) + r);
      g.moveTo(X(m.x) - r, Y(m.y) + r);
      g.lineTo(X(m.x) + r, Y(m.y) - r);
      g.stroke();
    }
    if (picked) {
      g.strokeStyle = "#2dd4bf";
      g.lineWidth = 2.5 * dpr;
      g.beginPath();
      g.arc(X(picked.x), Y(picked.y), (compact ? 6 : 9) * dpr, 0, 2 * Math.PI);
      g.stroke();
    }
  }, [size, field, vmax, map, points, blocks, hollow, marks, picked, showRoads, compact, W, bkm]);

  const at = (e: React.MouseEvent) => {
    const r = canvas.current!.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * W, y: (1 - (e.clientY - r.top) / r.height) * W, kmPerPx: W / r.width };
  };
  return (
    <div ref={box} className="relative flex h-full w-full items-center justify-center overflow-hidden rounded-lg">
      <div className="relative" style={{ width: size, height: size }}>
        <canvas ref={canvas} className="block rounded-lg" style={{ width: size, height: size, cursor: cursor ?? (onPick ? "crosshair" : "default") }}
          onClick={(e) => { if (onPick) { const q = at(e); onPick(Math.min(Math.max(q.x, 0), W), Math.min(Math.max(q.y, 0), W)); } }}
          onMouseMove={onHover ? (e) => {
            const q = at(e);
            let best: Point | null = null, bd = (8 * q.kmPerPx) ** 2;
            for (const p of points) { const d = (p.x - q.x) ** 2 + (p.y - q.y) ** 2; if (d < bd) { bd = d; best = p; } }
            onHover(best);
          } : undefined} onMouseLeave={onHover ? () => onHover(null) : undefined} />
        {label && <div className="pointer-events-none absolute left-2 top-2 rounded bg-ink/80 px-2 py-0.5 text-xs">{label}</div>}
        {wind && <WindBadge {...wind} />}
      </div>
    </div>
  );
}

function WindBadge({ u, v }: { u: number; v: number }) {
  const speed = Math.hypot(u, v);
  const toDeg = (Math.atan2(u, v) * 180) / Math.PI;
  const fromDeg = (toDeg + 540) % 360;
  const names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  return (
    <div className="pointer-events-none absolute bottom-2 left-2 flex items-center gap-2 rounded bg-ink/80 px-2 py-1 text-xs">
      <svg width="16" height="16" viewBox="-8 -8 16 16" style={{ transform: `rotate(${toDeg}deg)` }} aria-hidden="true"><path d="M0 7 V-6 M-3.5 -2 L0 -6.5 L3.5 -2" stroke="#2dd4bf" strokeWidth="1.6" fill="none" /></svg>
      <span className="num">{speed.toFixed(1)} m/s</span><span className="text-dim">from {names[Math.round(fromDeg / 45) % 8]}</span>
    </div>
  );
}

export function Legend({ vmax = 500, title }: { vmax?: number; title: string }) {
  return (
    <div className="text-xs">
      <div className="mb-1 text-dim">{title}</div>
      <div className="h-2 w-44 rounded" style={{ background: rampCss }} />
      <div className="num mt-0.5 flex w-44 justify-between text-dim"><span>0</span><span>{Math.round(vmax / 2)}</span><span>{vmax}+</span></div>
    </div>
  );
}
