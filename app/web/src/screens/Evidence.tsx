// Every number here is read from reports/ through evidence.json, none are typed in.
import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, LabelList, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Note } from "../components/Bits";
import { getJSON } from "../lib/data";

/* eslint-disable @typescript-eslint/no-explicit-any */
const COLORS: Record<string, string> = { "FNO ensemble": "#2dd4bf", "FNO + residual kriging": "#f59e0b", Kriging: "#a78bfa", IDW: "#60a5fa", "Network median": "#94a3b8", "NNLS-L1 + OSM prior": "#f87171" };
const axis = { stroke: "#98a3b8", fontSize: 11 };
const tip = { contentStyle: { background: "#101725", border: "1px solid #263146", fontSize: 12 }, labelStyle: { color: "#e8ecf4" } };

function Card({ title, verdict, children }: { title: string; verdict: string; children: React.ReactNode }) {
  return (
    <section className="panel p-4">
      <h2 className="text-sm font-medium">{title}</h2>
      <p className="mb-3 mt-1 text-xs text-dim">{verdict}</p>
      {children}
    </section>
  );
}

export function Evidence() {
  const [ev, setEv] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { getJSON("/api/evidence").then(setEv).catch((e) => setErr(e.message)); }, []);
  if (err) return <div className="p-6"><Note tone="warn">{err}</Note></div>;
  if (!ev) return <div className="p-6 text-dim">Loading…</div>;

  const s1 = ev.synthetic_v1, c = s1.curves, r1 = ev.real_v1, r2 = ev.real_v2, pv = ev.priority;
  const sensors = c.n_sensors.map((n: number, i: number) => ({ n, FNO: c["FNO ensemble"].n_field_rmse[i], Kriging: c.Kriging.n_field_rmse[i], NNLS: c["NNLS-L1 + OSM prior"].n_field_rmse[i] }));
  const noise = c.noise.map((n: number, i: number) => ({ n, FNO: c["FNO ensemble"].noise_field_rmse[i], Kriging: c.Kriging.noise_field_rmse[i], NNLS: c["NNLS-L1 + OSM prior"].noise_field_rmse[i] }));
  const synth = ["FNO ensemble", "NNLS-L1 + OSM prior", "Kriging", "IDW"].map((m) => { const r = s1.table[`general|${m}`]; return { m, rmse: r.field_rmse, recall: r.recall ?? null, precision: r.precision ?? null }; });
  const methods = ["FNO ensemble", "FNO + residual kriging", "Kriging", "IDW", "Network median"];
  const real = methods.map((m) => ({ m: m.replace("FNO + residual kriging", "FNO + res. kriging"), "First model": r1.all[m].rmse, "Recalibrated": r2.all[m].rmse }));
  const cov = s1.levels.map((l: number, i: number) => ({ l, raw: ev.synthetic_v2.coverage.field.raw[i], calibrated: ev.synthetic_v2.coverage.field.calibrated[i] }));
  const stations = Object.entries(r2.per_station_rmse as Record<string, any>).map(([name, v]) => ({ name: name.split(",")[0].split(" - ")[0], gain: +(v.Kriging - v["FNO ensemble"]).toFixed(1) })).sort((a, b) => b.gain - a.gain);
  const prec = pv ? Object.entries(pv.precision_at_k as Record<string, number>).map(([m, v]) => ({ m, pct: +(100 * v).toFixed(0) })) : [];
  const f = s1.table["general|FNO ensemble"], kr = s1.table["general|Kriging"];

  return (
    <div className="h-full overflow-y-auto p-3">
      <div className="mx-auto grid max-w-6xl grid-cols-2 gap-3 max-lg:grid-cols-1">
        <div className="col-span-2 max-lg:col-span-1"><Note>Short version: where the physics matches (simulated cities), the FNO beats the baselines and handles sparse sensors far better. On real Delhi data it doesn't beat kriging, and its source shares can't be checked.</Note></div>

        <Card title="Simulated cities: accuracy (100 random held-out scenarios)" verdict={`FNO ${f.field_rmse.toFixed(1)} vs kriging ${kr.field_rmse.toFixed(1)} µg/m³. Recall ${f.recall.toFixed(2)} at similar precision to NNLS. First dataset (synth_v1).`}>
          <ResponsiveContainer width="100%" height={220}><BarChart data={synth}><CartesianGrid stroke="#263146" vertical={false} /><XAxis dataKey="m" {...axis} interval={0} tickFormatter={(v) => v.replace("NNLS-L1 + OSM prior", "NNLS + prior").replace(" ensemble", "")} /><YAxis {...axis} unit="" /><Tooltip {...tip} cursor={{ fill: "#172033" }} /><Bar isAnimationActive={false} dataKey="rmse" name="Field RMSE [µg/m³]" fill="#2dd4bf" /></BarChart></ResponsiveContainer>
        </Card>
        <Card title="Simulated cities: fewer sensors" verdict="Field error as the network shrinks. The FNO was trained on randomly thinned networks, and it shows.">
          <ResponsiveContainer width="100%" height={220}><LineChart data={sensors}><CartesianGrid stroke="#263146" /><XAxis dataKey="n" {...axis} label={{ value: "sensors", position: "insideBottomRight", fill: "#98a3b8", fontSize: 11, dy: 8 }} /><YAxis {...axis} domain={["auto", "auto"]} /><Tooltip {...tip} /><Legend wrapperStyle={{ fontSize: 11 }} /><Line isAnimationActive={false} dot={{ r: 3 }} dataKey="FNO" stroke="#2dd4bf" strokeWidth={2} /><Line isAnimationActive={false} dot={{ r: 3 }} dataKey="Kriging" stroke="#a78bfa" /><Line isAnimationActive={false} dot={{ r: 3 }} dataKey="NNLS" stroke="#f87171" /></LineChart></ResponsiveContainer>
        </Card>
        <Card title="Simulated cities: noisier sensors" verdict="Field error as multiplicative sensor noise grows.">
          <ResponsiveContainer width="100%" height={200}><LineChart data={noise}><CartesianGrid stroke="#263146" /><XAxis dataKey="n" {...axis} /><YAxis {...axis} domain={["auto", "auto"]} /><Tooltip {...tip} /><Legend wrapperStyle={{ fontSize: 11 }} /><Line isAnimationActive={false} dot={{ r: 3 }} dataKey="FNO" stroke="#2dd4bf" strokeWidth={2} /><Line isAnimationActive={false} dot={{ r: 3 }} dataKey="Kriging" stroke="#a78bfa" /><Line isAnimationActive={false} dot={{ r: 3 }} dataKey="NNLS" stroke="#f87171" /></LineChart></ResponsiveContainer>
        </Card>
        <Card title="Uncertainty on simulated data" verdict="The raw ensemble is overconfident. One inflation factor, fitted on validation, restores nominal coverage on the test set. Current model.">
          <ResponsiveContainer width="100%" height={200}><LineChart data={cov}><CartesianGrid stroke="#263146" /><XAxis dataKey="l" {...axis} type="number" domain={[0, 1]} /><YAxis {...axis} domain={[0, 1]} /><Tooltip {...tip} /><Legend wrapperStyle={{ fontSize: 11 }} /><ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#98a3b8" strokeDasharray="3 3" /><Line isAnimationActive={false} dot={{ r: 3 }} dataKey="raw" stroke="#f87171" /><Line isAnimationActive={false} dot={{ r: 3 }} dataKey="calibrated" stroke="#2dd4bf" strokeWidth={2} /></LineChart></ResponsiveContainer>
        </Card>
        <Card title="Real Delhi: predicting a hidden station" verdict={`November 2025, ${r2.n_stations} stations, ${r2.n_predictions.toLocaleString()} held-out predictions. Recalibrating the simulator removed the FNO's bias (${r1.all["FNO ensemble"].bias.toFixed(0)} → ${r2.all["FNO ensemble"].bias > 0 ? "+" : ""}${r2.all["FNO ensemble"].bias.toFixed(0)}), but kriging stays best at ${r2.all.Kriging.rmse.toFixed(1)} µg/m³.`}>
          <ResponsiveContainer width="100%" height={240}><BarChart data={real}><CartesianGrid stroke="#263146" vertical={false} /><XAxis dataKey="m" {...axis} interval={0} /><YAxis {...axis} domain={[0, 100]} /><Tooltip {...tip} cursor={{ fill: "#172033" }} /><Legend wrapperStyle={{ fontSize: 11 }} /><Bar isAnimationActive={false} dataKey="First model" fill="#475569"><LabelList dataKey="First model" position="top" fill="#98a3b8" fontSize={10} formatter={(v: number) => v.toFixed(1)} /></Bar><Bar isAnimationActive={false} dataKey="Recalibrated" fill="#2dd4bf"><LabelList dataKey="Recalibrated" position="top" fill="#e8ecf4" fontSize={10} formatter={(v: number) => v.toFixed(1)} /></Bar></BarChart></ResponsiveContainer>
          <div className="text-[11px] text-dim">RMSE in µg/m³. Kriging, IDW and the median don't depend on the model.</div>
        </Card>
        <Card title="Real Delhi: FNO against kriging, per station" verdict={`Bars above zero: the FNO beat kriging there (${(100 * r2.fno_beats_kriging_station_fraction).toFixed(0)}% of stations). Real 90% intervals cover only ${(100 * r2.coverage90.calibrated).toFixed(0)}% of readings.`}>
          <ResponsiveContainer width="100%" height={240}><BarChart data={stations}><CartesianGrid stroke="#263146" vertical={false} /><XAxis dataKey="name" hide /><YAxis {...axis} /><Tooltip {...tip} cursor={{ fill: "#172033" }} /><ReferenceLine y={0} stroke="#98a3b8" /><Bar isAnimationActive={false} dataKey="gain" name="Kriging RMSE − FNO RMSE" fill="#2dd4bf" /></BarChart></ResponsiveContainer>
        </Card>
        {pv && <Card title="Is the priority list useful? (simulated, truth known)" verdict={`Share of top-${pv.k} blocks within one block of a true source above ${pv.tau_g_s} g/s, over ${pv.n_test} held-out scenarios. Only slightly better than ranking by the OSM map alone, and clearly better than chance.`}>
          <ResponsiveContainer width="100%" height={200}><BarChart data={prec} layout="vertical" margin={{ left: 60 }}><CartesianGrid stroke="#263146" horizontal={false} /><XAxis type="number" {...axis} domain={[0, 100]} unit="%" /><YAxis type="category" dataKey="m" {...axis} width={120} /><Tooltip {...tip} cursor={{ fill: "#172033" }} /><Bar isAnimationActive={false} dataKey="pct" name="Precision at 5" fill="#2dd4bf" /></BarChart></ResponsiveContainer>
        </Card>}
        <Card title="What hasn't been shown" verdict="The limits to state up front.">
          <ul className="list-disc space-y-1.5 pl-4 text-xs text-dim">
            <li>Source shares on real data have no ground truth here. That would need chemical filter-based source apportionment.</li>
            <li>The model has no secondary aerosol chemistry, and no smoke transported from outside the domain. Those end up in "regional".</li>
            <li>The 150 m floor on mixing depth is a calibrated parameter, fitted on December 2025.</li>
            <li>The priority list hasn't been validated on real data, and it can't identify a specific site or serve as evidence.</li>
            <li>The FNO's win over baselines on simulated data was measured on the first dataset (synth_v1). No baselines were rerun on the current one.</li>
          </ul>
        </Card>
      </div>
    </div>
  );
}
