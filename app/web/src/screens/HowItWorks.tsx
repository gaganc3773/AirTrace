// The equation term by term, with a small live solver, then the pipeline from sensors to shares.
import { useEffect, useRef, useState } from "react";
import { rampCss } from "../components/FieldMap";

const TERMS = [
  { sym: "∂c/∂t", name: "Change", text: "How fast PM2.5 changes at a point." },
  { sym: "u·∇c", name: "Advection", text: "The wind carries pollution downwind. This is why a plume points away from its source." },
  { sym: "∇·(K∇c)", name: "Mixing", text: "Turbulence spreads the plume sideways. K is the eddy diffusivity, about 75 m²/s." },
  { sym: "λc", name: "Removal", text: "Deposition and dilution. λ also grows while the mixed layer deepens each morning." },
  { sym: "S", name: "Sources", text: "Emissions divided by mixing depth. The same factory causes more PM2.5 under a shallow night layer." },
];
const STEPS = [
  ["Monitors", "6 hours of hourly PM2.5 from any number of sensors"],
  ["Weather", "Wind and mixing depth from ERA5"],
  ["Map", "Roads, industry and landfills from OpenStreetMap"],
  ["FNO × 5", "A neural operator trained on 17,000 simulated cities; five copies give a spread"],
  ["Output", "PM2.5 per source type, source maps, regional background"],
];

/** A small 2D upwind solver with a point source, run in the browser. */
function Plume() {
  const ref = useRef<HTMLCanvasElement>(null);
  const [speed, setSpeed] = useState(2.5);
  const [dir, setDir] = useState(300);
  const [K, setK] = useState(75);
  const [lam, setLam] = useState(2);
  const p = useRef({ speed, dir, K, lam });
  p.current = { speed, dir, K, lam };
  useEffect(() => {
    const n = 96, dx = 500;
    let c = new Float32Array(n * n), c2 = new Float32Array(n * n);
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(n, n);
    let raf = 0;
    const src = (n >> 1) * n + (n >> 1);
    const tick = () => {
      const { speed: s, dir: d, K: kk, lam: l } = p.current;
      const rad = (d * Math.PI) / 180, u = -s * Math.sin(rad), v = -s * Math.cos(rad);
      const dt = Math.min((0.4 * dx) / Math.max(Math.abs(u) + Math.abs(v), 0.1), (0.2 * dx * dx) / kk);
      for (let step = 0; step < 4; step++) {
        for (let y = 1; y < n - 1; y++) for (let x = 1; x < n - 1; x++) {
          const i = y * n + x, cc = c[i];
          const ax = u > 0 ? u * (cc - c[i - 1]) : u * (c[i + 1] - cc);
          const ay = v > 0 ? v * (cc - c[i - n]) : v * (c[i + n] - cc);
          const diff = kk * (c[i - 1] + c[i + 1] + c[i - n] + c[i + n] - 4 * cc) / dx;
          c2[i] = cc + (dt / dx) * (diff - ax - ay) - dt * l * 1e-4 * cc;
        }
        c2[src] += dt * 0.02;
        [c, c2] = [c2, c];
      }
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const val = Math.min(Math.sqrt(Math.max(c[(n - 1 - y) * n + x], 0)) * 2.1, 1), o = (y * n + x) * 4;
        img.data[o] = 12 + 240 * Math.min(val * 1.4, 1);
        img.data[o + 1] = 14 + 216 * val * val;
        img.data[o + 2] = 40 + 90 * Math.sin(val * Math.PI);
        img.data[o + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);
  const slider = (label: string, v: number, set: (x: number) => void, min: number, max: number, step: number, show: string) => (
    <label className="mb-2 block text-xs text-dim">{label} <span className="num text-text">{show}</span><input type="range" min={min} max={max} step={step} value={v} onChange={(e) => set(+e.target.value)} /></label>
  );
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_220px] gap-4 max-md:grid-cols-1">
      <canvas ref={ref} width={96} height={96} className="aspect-square w-full max-w-md rounded-lg" style={{ imageRendering: "auto" }} aria-label="Live plume from one source" />
      <div>
        {slider("Wind from", dir, setDir, 0, 359, 1, `${dir}°`)}
        {slider("Wind speed", speed, setSpeed, 0.3, 6, 0.1, `${speed.toFixed(1)} m/s`)}
        {slider("Mixing K", K, setK, 10, 300, 5, `${K} m²/s`)}
        {slider("Removal λ", lam, setLam, 0, 10, 0.5, `${lam.toFixed(1)} ×10⁻⁴ /s`)}
        <p className="mt-3 text-xs text-dim">One source on a 48 km box, solved live in your browser with a simple upwind scheme. Illustrative only; AirTrace's own solver is more accurate and runs in Python.</p>
      </div>
    </div>
  );
}

export function HowItWorks() {
  const [term, setTerm] = useState(1);
  return (
    <div className="h-full overflow-y-auto p-3">
      <div className="mx-auto flex max-w-5xl flex-col gap-3">
        <section className="panel p-5">
          <h2 className="text-sm font-medium">The physics</h2>
          <div className="my-4 flex flex-wrap items-baseline justify-center gap-x-3 gap-y-2 font-mono text-xl">
            {TERMS.map((t, k) => (
              <span key={t.name} className="flex items-baseline gap-3">
                {k === 1 && <span className="text-dim">+</span>}{k === 2 && <span className="text-dim">=</span>}{k === 3 && <span className="text-dim">−</span>}{k === 4 && <span className="text-dim">+</span>}
                <button onClick={() => setTerm(k)} onMouseEnter={() => setTerm(k)} className={`rounded px-1.5 py-0.5 transition-colors ${term === k ? "bg-accent text-ink" : "hover:bg-panel-2"}`}>{t.sym}</button>
              </span>
            ))}
          </div>
          <div className="mx-auto max-w-xl text-center text-sm"><span className="font-medium">{TERMS[term].name}.</span> <span className="text-dim">{TERMS[term].text}</span></div>
        </section>
        <section className="panel p-5"><h2 className="mb-3 text-sm font-medium">Try the equation</h2><Plume /></section>
        <section className="panel p-5">
          <h2 className="text-sm font-medium">Why attribution is well defined</h2>
          <p className="mt-2 max-w-3xl text-sm text-dim">The equation is linear in c and S. Run it once per source type and the results add up to the full field: c = c<sub>traffic</sub> + c<sub>industry</sub> + c<sub>construction</sub> + c<sub>burning</sub> + c<sub>regional</sub>. So "who is responsible here" has an exact answer in the model, and each share is a ratio of concentrations. The inverse problem, going from sparse sensors back to those fields, is the hard and ill-posed part.</p>
          <div className="mt-3 h-2 w-64 rounded" style={{ background: rampCss }} aria-hidden="true" />
        </section>
        <section className="panel p-5">
          <h2 className="mb-3 text-sm font-medium">From sensors to shares</h2>
          <div className="grid grid-cols-5 gap-2 max-md:grid-cols-1">
            {STEPS.map(([a, b], k) => (
              <div key={a} className={`rounded-lg border p-3 ${k === 3 ? "border-accent" : "border-line"}`}><div className="text-sm font-medium">{a}</div><div className="mt-1 text-xs text-dim">{b}</div></div>
            ))}
          </div>
          <p className="mt-3 text-xs text-dim">Three inverse methods share one forward model: Green's-function inversion (exact physics, ~9 s), a PINN (physics in the loss, ~90 s per case) and the FNO ensemble (~0.4 s on a laptop CPU).</p>
        </section>
      </div>
    </div>
  );
}
