// Waits for the experiment: everything shown here comes from reports/phase7_hidden.json.
import { useEffect, useMemo, useState } from "react";
import { Note } from "../components/Bits";
import { FieldMap } from "../components/FieldMap";
import { getJSON, type MapData } from "../lib/data";

const COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
const windFrom = ([u, v]: [number, number]) => COMPASS[Math.round(((Math.atan2(-u, -v) * 180) / Math.PI + 360) / 45) % 8];

interface Run { trial: number; truth_block: [number, number]; truth_xy_km: [number, number]; strength_g_s: number; winds: [number, number][]; maps: { hidden: number[][][]; control: number[][][] } }
interface Story { trials: number; nights: number; chance_hit: number; stats: Record<string, { median_error_km: number; hit_within_1_block: number; hidden_beats_control: number; top_score_hidden_median: number; top_score_control_median: number }>; runs: Run[] }

export function HiddenStory({ map }: { map: MapData }) {
  const [s, setS] = useState<Story | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [nights, setNights] = useState(1);
  const [runIdx, setRunIdx] = useState(0);
  useEffect(() => { getJSON<Story>("/api/hidden").then((d) => { setS(d); setRunIdx(0); }).catch((e) => setErr(e.message)); }, []);
  const run = s?.runs[runIdx];
  const avg = useMemo(() => {
    if (!run) return null;
    const nb = run.maps.hidden[0].length, n = map.nx, up = n / nb;
    const m = new Float32Array(n * n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      let t = 0;
      for (let k = 0; k < nights; k++) t += run.maps.hidden[k][(y / up) | 0][(x / up) | 0];
      m[y * n + x] = t / nights;
    }
    return m;
  }, [run, nights, map.nx]);
  if (err) return <div className="p-6"><Note>The hidden-emitter experiment hasn't finished yet, so this screen has nothing to show. Run scripts/phase7_hidden.py, then phase7_precompute.py --only hidden.</Note></div>;
  if (!s || !run || !avg) return <div className="p-6 text-dim">Loading…</div>;
  const combos = Object.keys(s.stats);
  return (
    <div className="flex h-full gap-3 overflow-y-auto p-3 max-lg:flex-col">
      <div className="panel min-h-[420px] flex-1 p-2">
        <FieldMap map={map} field={avg} vmax={80} marks={[{ x: run.truth_xy_km[0], y: run.truth_xy_km[1], color: [45, 212, 191] }]} label={`Inverted emission, averaged over ${nights} night${nights > 1 ? "s" : ""}. × true location`} />
      </div>
      <div className="flex w-96 shrink-0 flex-col gap-3 max-lg:w-full">
        <div className="panel p-3 text-sm">
          <div className="font-medium">Can changing winds locate an unmapped factory?</div>
          <p className="mt-1 text-xs text-dim">One night only says "somewhere upwind along this line", and a mapped source on that line explains the reading just as well. When the wind changes between nights, the lines can cross near the source. Here a night-only factory that's on no map is observed by Delhi's real monitor locations, using physics inversion rather than the FNO.</p>
        </div>
        <div className="panel p-3">
          <label className="block text-xs text-dim">Nights combined <span className="num text-text">{nights}</span><input type="range" min={1} max={s.nights} value={nights} onChange={(e) => setNights(+e.target.value)} /></label>
          <label className="mt-2 block text-xs text-dim">Trial <select className="ml-1" value={runIdx} onChange={(e) => setRunIdx(+e.target.value)}>{s.runs.map((r, k) => <option key={r.trial} value={k}>#{k + 1} · {r.strength_g_s.toFixed(0)} g/s</option>)}</select></label>
          <div className="mt-2 text-xs text-dim">Wind from: <span className="num">{run.winds.slice(0, nights).map(windFrom).join(" → ")}</span></div>
        </div>
        <div className="panel p-3">
          <table className="w-full text-xs"><thead><tr className="text-left text-dim"><th className="font-normal">Nights</th><th className="text-right font-normal">Median error</th><th className="text-right font-normal">Hit</th><th className="text-right font-normal">Beats control</th></tr></thead>
            <tbody>{combos.map((k) => <tr key={k} className="border-t border-line"><td className="num py-1">{k}</td><td className="num text-right">{s.stats[k].median_error_km.toFixed(1)} km</td><td className="num text-right">{(100 * s.stats[k].hit_within_1_block).toFixed(0)}%</td><td className="num text-right">{(100 * s.stats[k].hidden_beats_control).toFixed(0)}%</td></tr>)}</tbody></table>
          <p className="mt-2 text-[11px] text-dim">Over {s.trials} trials. Hit: the top block is within one block of the factory; a random block manages {(100 * s.chance_hit).toFixed(0)}%. Beats control: how often the top score is higher with the factory than without it.</p>
        </div>
        <Note tone="warn">Promising, not solved. More nights bring the estimate closer, but with only {s.trials} trials of strong sources the numbers are rough. The score also can't reliably tell whether a hidden source exists at all, so this finds where to look, assuming there's something to find.</Note>
      </div>
    </div>
  );
}
