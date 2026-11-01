import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";
import { getJSON, type MapData } from "./lib/data";
import { Evidence } from "./screens/Evidence";
import { HiddenStory } from "./screens/HiddenStory";
import { HowItWorks } from "./screens/HowItWorks";
import { LiveMap } from "./screens/LiveMap";
import { ScenarioLab } from "./screens/ScenarioLab";

const TABS = [["live", "Live map"], ["lab", "Scenario lab"], ["hidden", "Hidden emitter"], ["evidence", "Evidence"], ["how", "How it works"]] as const;
type Tab = (typeof TABS)[number][0];

export function App() {
  const [tab, setTab] = useState<Tab>(() => {
    const h = location.hash.slice(1);
    return (TABS.some(([k]) => k === h) ? h : "live") as Tab;
  });
  const [map, setMap] = useState<MapData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState("starting");
  useEffect(() => { getJSON<MapData>("/api/map").then(setMap).catch((e) => setError(e.message)); }, []);
  useEffect(() => { location.hash = tab; }, [tab]);
  useEffect(() => {
    const onHash = () => { const h = location.hash.slice(1); if (TABS.some(([k]) => k === h)) setTab(h as Tab); };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => {
    const check = () => getJSON<{ inference: string }>("/api/status").then((s) => setStatus(s.inference)).catch(() => setStatus("down"));
    check();
    const id = window.setInterval(check, 3000);
    return () => window.clearInterval(id);
  }, []);
  return (
    <div className="flex h-full flex-col">
      <header className="flex shrink-0 items-center gap-6 border-b border-line px-4 py-2">
        <div className="text-base font-semibold tracking-tight">Air<span className="text-accent">Trace</span></div>
        <nav className="flex flex-wrap gap-4" aria-label="Screens">
          {TABS.map(([k, label]) => (
            <button key={k} onClick={() => setTab(k)} aria-current={tab === k ? "page" : undefined}
              className={`cursor-pointer border-b-2 pb-0.5 text-sm transition-colors ${tab === k ? "border-accent text-text" : "border-transparent text-dim hover:text-text"}`}>{label}</button>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-1.5 text-xs text-dim">
          <span className={`h-1.5 w-1.5 rounded-full ${status === "ready" ? "bg-accent" : status === "starting" ? "bg-warn" : "bg-burning"}`} />
          {status === "ready" ? "Model ready · offline" : status === "starting" ? "Loading model…" : "Model offline"}
        </div>
      </header>
      <main className="min-h-0 flex-1">
        {error && <div className="p-6 text-warn">{error}</div>}
        {map && (
          <AnimatePresence mode="wait">
            <motion.div key={tab} className="h-full" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.12 }}>
              {tab === "live" && <LiveMap map={map} />}
              {tab === "lab" && <ScenarioLab map={map} />}
              {tab === "hidden" && <HiddenStory map={map} />}
              {tab === "evidence" && <Evidence />}
              {tab === "how" && <HowItWorks />}
            </motion.div>
          </AnimatePresence>
        )}
      </main>
      <footer className="shrink-0 border-t border-line px-4 py-1 text-[11px] text-dim">
        Map data © <a className="underline" href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap contributors</a> (ODbL) · Monitor data via <a className="underline" href="https://openaq.org" target="_blank" rel="noreferrer">OpenAQ</a> · <a className="underline" href="https://open-meteo.com/" target="_blank" rel="noreferrer">Weather data by Open-Meteo.com</a> · Contains modified Copernicus Climate Change Service information 2026
      </footer>
    </div>
  );
}
