// Real November 2025: kriging for PM2.5 (the most accurate on held-out stations), FNO for source estimates, clearly labelled.
import { useEffect, useMemo, useRef, useState } from "react";
import { Note, PriorityTable, Shares, downloadCsv } from "../components/Bits";
import { FieldMap, Legend, type Block, type Point } from "../components/FieldMap";
import { CATS, CAT_COLOR, aqiBand, getJSON, getWindow, median, sharesFrom, windowEnd, type MapData, type PriorityRow, type RealMeta, type RealWindow } from "../lib/data";

type Layer = "pm" | "fno" | "local" | "sources";
const LAYERS: [Layer, string][] = [["pm", "PM2.5 (kriging)"], ["fno", "PM2.5 (FNO)"], ["local", "Local part"], ["sources", "Sources"]];
const hex = (c: string): [number, number, number] => [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)];

export function LiveMap({ map }: { map: MapData }) {
  const [meta, setMeta] = useState<RealMeta | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [i, setI] = useState(0);
  const [win, setWin] = useState<RealWindow | null>(null);
  const [layer, setLayer] = useState<Layer>("pm");
  const [playing, setPlaying] = useState(false);
  const [picked, setPicked] = useState<{ x: number; y: number } | null>({ x: 24, y: 24 });
  const [hover, setHover] = useState<Point | null>(null);
  const [selBlock, setSelBlock] = useState<string | null>(null);
  const [fullList, setFullList] = useState(false);
  const [listStats, setListStats] = useState<{ k: number; precision_at_k: Record<string, number> } | null>(null);

  useEffect(() => {
    getJSON<RealMeta>("/api/real/meta").then((m) => { setMeta(m); setI(m.times.findIndex((t) => t.startsWith("2025-11-12T18"))); }).catch((e) => setError(e.message));
  }, []);
  useEffect(() => { getJSON<{ priority: typeof listStats }>("/api/evidence").then((e) => setListStats(e.priority)).catch(() => undefined); }, []);
  useEffect(() => {
    if (!meta) return;
    let live = true;
    getWindow(meta, i).then((w) => { if (live) setWin(w); }).catch((e) => setError(e.message));
    return () => { live = false; };
  }, [meta, i]);
  const timer = useRef<number>();
  useEffect(() => {
    if (!playing || !meta) return;
    timer.current = window.setInterval(() => setI((k) => (k + 1) % meta.n), 280);
    return () => window.clearInterval(timer.current);
  }, [playing, meta]);

  const medians = useMemo(() => meta?.times.map((_, k) => median(meta.readings.map((r) => r[k]).filter((v): v is number => v != null))) ?? [], [meta]);
  const n = map.nx;
  const field = useMemo(() => {
    if (!win || !meta) return null;
    if (layer === "pm") return win.kriging;
    if (layer === "sources") {
      const nb = meta.nb, f = new Float32Array(n * n), up = n / nb;
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const b = ((y / up) | 0) * nb + ((x / up) | 0);
        f[y * n + x] = win.E_blocks[1][b] + win.E_blocks[2][b] + win.E_blocks[3][b];
      }
      return f;
    }
    const f = new Float32Array(n * n);
    const bg = layer === "fno" ? meta.bg[i] : 0;
    for (let k = 0; k < n * n; k++) f[k] = win.c_cat[0][k] + win.c_cat[1][k] + win.c_cat[2][k] + win.c_cat[3][k] + bg;
    return f;
  }, [win, layer, meta, i, n]);
  if (error) return <div className="p-6"><Note tone="warn">{error}</Note></div>;
  if (!meta) return <div className="p-6 text-dim">Loading November 2025…</div>;

  const vmax = layer === "sources" ? 60 : layer === "local" ? 200 : 500;
  const unit = layer === "sources" ? "Emission, non-traffic [g/s per block]" : layer === "local" ? "Local PM2.5 [µg/m³]" : "PM2.5 [µg/m³]";
  const end = windowEnd(meta.times[i]);
  const rows = meta.priority[i];
  const points: Point[] = map.monitors.map((m, s) => {
    const v = meta.readings[s][i];
    return { x: m.x, y: m.y, label: m.name, color: v == null ? [70, 80, 100] : hex(aqiBand(v).color) };
  });
  const blocks: Block[] = rows.slice(0, 5).map((r) => ({ by: r.by, bx: r.bx, color: hex(CAT_COLOR[r.cat]), selected: selBlock === `${r.by}-${r.bx}` }));
  const hoverIdx = hover ? map.monitors.findIndex((m) => m.name === hover.label) : -1;

  const pick = picked && win ? (() => {
    const ix = Math.min((picked.x / map.width_km * n) | 0, n - 1), iy = Math.min((picked.y / map.width_km * n) | 0, n - 1);
    const bi = ((iy / map.block) | 0) * meta.nb + ((ix / map.block) | 0), c = iy * n + ix;
    const parts = CATS.map((_, k) => win.c_cat[k][c]);
    return { pm: win.kriging[c], fno: parts.reduce((a, b) => a + b, 0) + meta.bg[i], bars: sharesFrom(parts, CATS.map((_, k) => win.std_blocks[k][bi]), meta.bg[i], meta.bg_std[i]) };
  })() : null;
  const band = pick ? aqiBand(pick.pm) : null;
  const W = 100 / meta.n;
  const top = Math.max(...medians.filter(Number.isFinite));

  return (
    <div className="flex h-full min-h-0 flex-col gap-3 p-3">
      <div className="flex min-h-0 flex-1 gap-3 max-lg:flex-col max-lg:overflow-y-auto">
        <div className="panel relative flex min-h-[420px] min-w-0 flex-1 flex-col p-2">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap gap-1.5">{LAYERS.map(([k, l]) => <button key={k} className="btn" aria-pressed={layer === k} onClick={() => setLayer(k)}>{l}</button>)}</div>
            <Legend vmax={vmax} title={unit} />
          </div>
          <div className="relative min-h-0 flex-1">
            <FieldMap map={map} field={field} vmax={vmax} points={points} blocks={blocks} picked={picked} wind={{ u: meta.u[i], v: meta.v[i] }} onPick={(x, y) => setPicked({ x, y })} onHover={setHover} />
            {hover && hoverIdx >= 0 && (
              <div className="pointer-events-none absolute right-2 top-2 rounded bg-ink/90 px-2 py-1 text-xs">
                <div>{hover.label}</div><div className="num text-dim">{meta.readings[hoverIdx][i] ?? "no reading"}{meta.readings[hoverIdx][i] != null ? " µg/m³" : ""}</div>
              </div>
            )}
          </div>
        </div>
        <div className="flex w-[370px] shrink-0 flex-col gap-3 overflow-y-auto max-lg:w-full">
          <div className="panel p-3">
            <div className="text-xs text-dim">Selected point · {picked ? `${picked.x.toFixed(1)}, ${picked.y.toFixed(1)} km` : "click the map"}</div>
            {pick && band && <>
              <div className="mt-1 flex items-baseline gap-2"><span className="num text-3xl font-medium">{pick.pm.toFixed(0)}</span><span className="text-xs text-dim">µg/m³ kriging · FNO {pick.fno.toFixed(0)}</span></div>
              <div className="mb-3 text-xs" style={{ color: band.color }}>{band.label} <span className="text-dim">(NAQI 24 h bands, as a guide)</span></div>
              <Shares bars={pick.bars} band="Lighter band: spread between ensemble members, widened with the synthetic calibration. It isn't a reliable 90% interval on real data." />
              <div className="mt-2"><Note tone="warn">Model estimate. Shares can't be checked against real measurements here, so don't read them as findings about Delhi.</Note></div>
            </>}
          </div>
          <div className="panel min-h-0 flex-1">
            <div className="flex items-center justify-between px-3 pt-2.5"><span className="text-sm font-medium">Where to look first</span>
              <div className="flex gap-1.5"><button className="btn" onClick={() => setFullList(true)}>Expand</button><button className="btn" onClick={() => downloadCsv(`airtrace-priority-${meta.times[i].slice(0, 13)}.csv`, rows)}>CSV</button></div></div>
            <div className="px-3 pb-1 text-[11px] text-dim">Ranked by model emission. A ranking only, not odds and not proof of a violation.</div>
            <PriorityTable compact mode="real" rows={rows.slice(0, 5)} selected={selBlock} onSelect={(r: PriorityRow) => { setSelBlock(`${r.by}-${r.bx}`); setPicked({ x: (r.bx + 0.5) * 3, y: (r.by + 0.5) * 3 }); }} />
          </div>
        </div>
      </div>
      <div className="panel flex shrink-0 items-center gap-3 px-3 py-2">
        <button className="btn" aria-label={playing ? "Pause" : "Play"} onClick={() => setPlaying(!playing)}>{playing ? "Pause" : "Play"}</button>
        <div className="relative min-w-0 flex-1">
          <svg className="block h-8 w-full" viewBox={`0 0 ${meta.n} ${top}`} preserveAspectRatio="none" aria-hidden="true">
            <polyline fill="none" stroke="#2dd4bf" strokeOpacity="0.7" strokeWidth="1.5" vectorEffect="non-scaling-stroke" points={medians.map((v, k) => (Number.isFinite(v) ? `${k},${top - v}` : "")).join(" ")} />
            <line x1={i} x2={i} y1={0} y2={top} stroke="#e8ecf4" strokeWidth="1" vectorEffect="non-scaling-stroke" />
          </svg>
          <input type="range" aria-label="Time" min={0} max={meta.n - 1} value={i} onChange={(e) => { setPlaying(false); setI(+e.target.value); }} style={{ marginLeft: `${W / 2}%` }} />
        </div>
        <div className="w-32 text-right"><div className="num text-sm">{end.time} IST</div><div className="text-xs text-dim">{end.date} 2025 · network median <span className="num">{Number.isFinite(medians[i]) ? medians[i].toFixed(0) : "–"}</span></div></div>
      </div>
      {fullList && (
        <div className="fixed inset-0 z-20 flex items-center justify-center bg-ink/70 p-6" onClick={() => setFullList(false)}>
          <div className="panel max-h-full w-full max-w-4xl overflow-y-auto p-3" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Priority list">
            <div className="mb-2 flex items-center justify-between"><span className="text-base font-medium">Where to look first · window ending {end.date}, {end.time} IST</span><button className="btn" onClick={() => setFullList(false)}>Close</button></div>
            <PriorityTable mode="real" rows={rows} />
            <div className="mt-3"><Note>Screening aid: areas to check first, not evidence against anyone. Each row is a 3 km block, so it can't single out a site. {listStats && `On simulated cities, ${Math.round(100 * listStats.precision_at_k["AirTrace (FNO ranking)"])}% of the top ${listStats.k} blocks were near a true strong source, against ${Math.round(100 * listStats.precision_at_k["OSM map alone"])}% for the OSM map alone and ${Math.round(100 * listStats.precision_at_k["Random block"])}% at random. `}On real data the list hasn't been validated.</Note></div>
          </div>
        </div>
      )}
    </div>
  );
}
