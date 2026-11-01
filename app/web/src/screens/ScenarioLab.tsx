// Held-out simulated cities, where the truth is known: change the sensor network and watch each method respond.
import { useEffect, useMemo, useRef, useState } from "react";
import { Note, PriorityTable, Shares } from "../components/Bits";
import { FieldMap, Legend, type Block, type Point } from "../components/FieldMap";
import { CATS, CAT_COLOR, flat2, getJSON, postJSON, sharesFrom, type MapData, type PriorityRow } from "../lib/data";

interface LabMeta { scenarios: { id: number; end_hour: number; wind_speed: number; wind_from: number; H: number; median_pm25: number; sources: { x: number; y: number }[] }[] }
interface Result {
  cells: [number, number][]; n: number; truth: number[][]; fno: number[][]; fno_sigma: number[][]; kriging: number[][]; nnls: number[][];
  c_cat: number[][][]; c_cat_std: number[][][]; bg: number; bg_std: number; truth_c_cat: number[][][]; truth_bg: number;
  E_truth: number[][]; E_fno: number[][]; E_nnls: number[][];
  rmse: { fno: number; kriging: number; nnls: number }; ms: { fno: number; kriging: number }; priority: PriorityRow[];
}
interface WhatIf { frames: number[][][]; minutes: number[]; ms: number }
const hex = (c: string): [number, number, number] => [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)];

export function ScenarioLab({ map }: { map: MapData }) {
  const [meta, setMeta] = useState<LabMeta | null>(null);
  const [scenario, setScenario] = useState(0);
  const [nSensors, setNSensors] = useState(40);
  const [noise, setNoise] = useState(15);
  const [seed, setSeed] = useState(0);
  const [cells, setCells] = useState<[number, number][] | null>(null);
  const [view, setView] = useState<"pm" | "sources">("pm");
  const [res, setRes] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<{ x: number; y: number } | null>({ x: 24, y: 24 });
  const [editing, setEditing] = useState(false);
  const [windFrom, setWindFrom] = useState(90);
  const [windSpeed, setWindSpeed] = useState(3);
  const [whatIf, setWhatIf] = useState<WhatIf | null>(null);
  const [frame, setFrame] = useState(0);
  const [wiBusy, setWiBusy] = useState(false);

  useEffect(() => {
    let retry: number;
    const load = () => getJSON<LabMeta>("/api/lab/scenarios").then(setMeta).catch((e) => { setError(e.message); retry = window.setTimeout(load, 2000); });
    load();
    return () => window.clearTimeout(retry);
  }, []);
  const body = useMemo(() => ({ scenario, n_sensors: nSensors, noise: noise / 100, seed, cells }), [scenario, nSensors, noise, seed, cells]);
  const abort = useRef<AbortController>();
  useEffect(() => {
    if (!meta) return;
    const t = window.setTimeout(() => {
      abort.current?.abort();
      abort.current = new AbortController();
      setBusy(true);
      postJSON<Result>("/api/lab/predict", body, abort.current.signal)
        .then((r) => { setRes(r); setError(null); setBusy(false); })
        .catch((e) => { if (e.name !== "AbortError") { setError(e.message); setBusy(false); } });
    }, 160);
    return () => window.clearTimeout(t);
  }, [body, meta]);
  useEffect(() => { setWhatIf(null); }, [body]);
  useEffect(() => {
    if (!whatIf) return;
    setFrame(0);
    const id = window.setInterval(() => setFrame((f) => (f + 1) % whatIf.frames.length), 700);
    return () => window.clearInterval(id);
  }, [whatIf]);

  const fields = useMemo(() => {
    if (!res) return null;
    const up = (rows: number[][]) => { // block map to cell map, for the source view
      const n = map.nx, nb = rows.length, s = n / nb, out = new Float32Array(n * n);
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) out[y * n + x] = rows[(y / s) | 0][(x / s) | 0];
      return out;
    };
    return view === "pm"
      ? { truth: flat2(res.truth), fno: flat2(res.fno), kriging: flat2(res.kriging), nnls: flat2(res.nnls), sigma: flat2(res.fno_sigma) }
      : { truth: up(res.E_truth), fno: up(res.E_fno), kriging: null, nnls: up(res.E_nnls), sigma: flat2(res.fno_sigma) };
  }, [res, view, map.nx]);

  if (!meta) return <div className="p-6"><Note>{error ?? "Loading scenarios…"}</Note></div>;
  const s = meta.scenarios[scenario];
  const km = map.width_km / map.nx;
  const sensors: Point[] = (res?.cells ?? []).map(([iy, ix]) => ({ x: (ix + 0.5) * km, y: (iy + 0.5) * km, radius: 2.5 }));
  const truthMarks: Point[] = view === "sources" ? s.sources.map((q) => ({ ...q, color: [248, 113, 113] })) : [];
  const vmax = view === "pm" ? Math.max(300, s.median_pm25 * 1.8) : 80;
  const pickAt = (x: number, y: number) => {
    if (!editing) return setPicked({ x, y });
    const c: [number, number] = [Math.min((y / km) | 0, map.ny - 1), Math.min((x / km) | 0, map.nx - 1)];
    const base = cells ?? res?.cells ?? [];
    const has = base.some((b) => b[0] === c[0] && b[1] === c[1]);
    const next = has ? base.filter((b) => b[0] !== c[0] || b[1] !== c[1]) : [...base, c];
    if (next.length) setCells(next);
  };
  const pick = picked && res ? (() => {
    const ix = Math.min((picked.x / km) | 0, map.nx - 1), iy = Math.min((picked.y / km) | 0, map.ny - 1);
    return {
      model: sharesFrom(CATS.map((_, k) => res.c_cat[k][iy][ix]), CATS.map((_, k) => res.c_cat_std[k][iy][ix]), res.bg, res.bg_std),
      truth: sharesFrom(CATS.map((_, k) => res.truth_c_cat[k][iy][ix]), [0, 0, 0, 0], res.truth_bg, 0), pm: res.fno[iy][ix], tr: res.truth[iy][ix], sig: res.fno_sigma[iy][ix],
    };
  })() : null;
  const blocks: Block[] = (res?.priority ?? []).slice(0, 5).map((r) => ({ by: r.by, bx: r.bx, color: hex(CAT_COLOR[r.cat]) }));
  const runWhatIf = () => {
    setWiBusy(true);
    postJSON<WhatIf>("/api/lab/whatif", { ...body, wind_from: windFrom, wind_speed: windSpeed, hours: 3 })
      .then((r) => { setWhatIf(r); setError(null); }).catch((e) => setError(e.message)).finally(() => setWiBusy(false));
  };
  // A plain function, not a component: a component defined here would remount its WebGL map on every render.
  const tile = (title: string, f?: Float32Array | null, extra?: string, ring?: boolean) => (
    <div key={title} className={`flex min-h-0 flex-col ${ring ? "rounded-lg outline outline-1 outline-accent" : ""}`}>
      <div className="flex justify-between px-1 pb-1 text-xs"><span>{title}</span><span className="num text-dim">{extra}</span></div>
      <div className="aspect-square min-h-0">{f
        ? <FieldMap compact map={map} field={f} vmax={vmax} points={title !== "Truth" && title !== "NNLS + prior" ? sensors : []} marks={truthMarks} picked={picked} onPick={pickAt} blocks={title === "FNO ensemble" && view === "sources" ? blocks : []} cursor={editing ? "cell" : undefined} />
        : <div className="flex h-full items-center justify-center rounded-lg bg-panel-2 p-3 text-center text-xs text-dim">Kriging only interpolates PM2.5. It says nothing about sources.</div>}</div>
    </div>
  );

  return (
    <div className="flex h-full min-h-0 gap-3 overflow-y-auto p-3 max-lg:flex-col">
      <div className="min-w-0 flex-1">
        <div className="panel mb-3 flex flex-wrap items-center gap-4 px-3 py-2">
          <label className="text-xs text-dim">Scenario <select className="ml-1" value={scenario} onChange={(e) => { setScenario(+e.target.value); setCells(null); }}>
            {meta.scenarios.map((q) => <option key={q.id} value={q.id}>#{q.id + 1} · {String(q.end_hour).padStart(2, "0")}:00 · wind {q.wind_speed} m/s · {q.median_pm25} µg/m³</option>)}
          </select></label>
          <div className="flex gap-1.5"><button className="btn" aria-pressed={view === "pm"} onClick={() => setView("pm")}>PM2.5</button><button className="btn" aria-pressed={view === "sources"} onClick={() => setView("sources")}>Sources</button></div>
          <Legend vmax={Math.round(vmax)} title={view === "pm" ? "PM2.5 [µg/m³]" : "Emission [g/s per block], × true sources"} />
          <span className={`text-xs ${busy ? "text-accent" : "text-dim"}`}>{busy ? "Running on CPU…" : res ? `FNO ${res.ms.fno} ms · kriging ${res.ms.kriging} ms` : ""}</span>
        </div>
        {error && <div className="mb-3"><Note tone="warn">{error}</Note></div>}
        {fields && <div className="grid grid-cols-4 gap-2 max-xl:grid-cols-2">
          {tile("Truth", fields.truth, "simulated")}
          {tile("FNO ensemble", fields.fno, view === "pm" ? `RMSE ${res?.rmse.fno}` : "", true)}
          {tile("Kriging", fields.kriging, view === "pm" ? `RMSE ${res?.rmse.kriging}` : "")}
          {tile("NNLS + prior", fields.nnls, view === "pm" ? `RMSE ${res?.rmse.nnls}` : "")}
        </div>}
        <div className="mt-1 text-[11px] text-dim">RMSE in µg/m³ at cells without a sensor. NNLS was precomputed on this scenario's default 40-sensor network (its physics solve takes too long for a slider), so it doesn't react to the controls.</div>
        <div className="mt-3 grid grid-cols-2 gap-3 max-xl:grid-cols-1">
          <div className="panel p-3">
            <div className="text-sm font-medium">What if the wind changes?</div>
            <div className="mb-2 text-[11px] text-dim">Runs the physics solver forward 3 h from the FNO's inferred sources under a new steady wind.</div>
            <label className="block text-xs text-dim">From <span className="num text-text">{windFrom}°</span><input type="range" min={0} max={359} value={windFrom} onChange={(e) => setWindFrom(+e.target.value)} /></label>
            <label className="block text-xs text-dim">Speed <span className="num text-text">{windSpeed.toFixed(1)} m/s</span><input type="range" min={0.5} max={8} step={0.5} value={windSpeed} onChange={(e) => setWindSpeed(+e.target.value)} /></label>
            <button className="btn mt-2" onClick={runWhatIf} disabled={wiBusy}>{wiBusy ? "Solving…" : "Run the solver"}</button>
            {whatIf && <span className="num ml-2 text-xs text-dim">{whatIf.ms} ms · +{whatIf.minutes[frame]} min</span>}
            {whatIf && <div className="mt-2 aspect-square max-w-72"><FieldMap compact map={map} field={flat2(whatIf.frames[frame])} vmax={vmax} /></div>}
          </div>
          <div className="panel"><div className="px-3 pt-2.5 text-sm font-medium">Where to look first</div>
            <div className="px-3 pb-1 text-[11px] text-dim">Hit rate: how often blocks with a similar score really held a source above 15 g/s, in held-out test runs.</div>
            <PriorityTable mode="lab" rows={(res?.priority ?? []).slice(0, 6)} /></div>
        </div>
      </div>
      <div className="flex w-72 shrink-0 flex-col gap-3 max-lg:w-full">
        <div className="panel p-3">
          <label className="block text-xs text-dim">Sensors <span className="num text-text">{cells ? `${cells.length} (custom)` : nSensors}</span>
            <input type="range" min={5} max={40} value={cells ? cells.length : nSensors} disabled={!!cells} onChange={(e) => setNSensors(+e.target.value)} /></label>
          <label className="mt-2 block text-xs text-dim">Sensor noise <span className="num text-text">{noise}%</span><input type="range" min={0} max={40} value={noise} onChange={(e) => setNoise(+e.target.value)} /></label>
          <div className="mt-2 flex flex-wrap gap-1.5">
            <button className="btn" onClick={() => { setCells(null); setSeed(seed + 1); }}>New layout</button>
            <button className="btn" aria-pressed={editing} onClick={() => { setEditing(!editing); if (!editing && res) setCells(res.cells); }}>{editing ? "Done editing" : "Edit sensors"}</button>
            {cells && <button className="btn" onClick={() => { setCells(null); setEditing(false); }}>Reset</button>}
          </div>
          {editing && <div className="mt-2 text-[11px] text-accent">Click a map to add a sensor, or click one to remove it.</div>}
        </div>
        {pick && <div className="panel p-3">
          <div className="text-xs text-dim">Selected point</div>
          <div className="flex items-baseline gap-2"><span className="num text-2xl font-medium">{pick.pm.toFixed(0)}</span><span className="text-xs text-dim">± {pick.sig.toFixed(0)} · truth {pick.tr.toFixed(0)} µg/m³</span></div>
          <div className="mb-1 mt-3 text-xs">FNO shares</div>
          <Shares bars={pick.model} band="Lighter band: calibrated 90% interval (simulated data)." />
          <div className="mb-1 mt-3 text-xs">True shares</div>
          <Shares bars={pick.truth} band="" />
        </div>}
      </div>
    </div>
  );
}
