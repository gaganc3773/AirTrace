// Small shared pieces: share bars, the priority table and status notes.
import { motion } from "framer-motion";
import type { ReactNode } from "react";
import { CAT_COLOR, CAT_LABEL, type PriorityRow, type ShareBar } from "../lib/data";

export function Shares({ bars, band }: { bars: ShareBar[]; band: string }) {
  return (
    <div>
      {bars.map((b) => (
        <div key={b.key} className="mb-2">
          <div className="flex justify-between text-xs"><span>{CAT_LABEL[b.key]}</span><span className="num">{b.pct.toFixed(0)}%</span></div>
          <div className="relative h-2 rounded-full bg-panel-2">
            <div className="absolute inset-y-0 rounded-full opacity-35" style={{ left: `${b.lo}%`, width: `${Math.max(b.hi - b.lo, 0)}%`, background: CAT_COLOR[b.key] }} />
            <motion.div className="absolute inset-y-0 left-0 rounded-full" animate={{ width: `${b.pct}%` }} transition={{ duration: 0.25 }} style={{ background: CAT_COLOR[b.key] }} />
          </div>
        </div>
      ))}
      <div className="text-[11px] text-dim">{band}</div>
    </div>
  );
}

export function Note({ tone = "dim", children }: { tone?: "dim" | "warn"; children: ReactNode }) {
  return <div className={`rounded-md border px-2.5 py-1.5 text-xs ${tone === "warn" ? "border-warn/40 bg-warn/10 text-warn" : "border-line text-dim"}`}>{children}</div>;
}

export function Chip({ cat }: { cat: string }) {
  return <span className="inline-block whitespace-nowrap rounded px-1.5 py-0.5 text-[11px]" style={{ color: CAT_COLOR[cat], background: `${CAT_COLOR[cat]}22` }}>{CAT_LABEL[cat]}</span>;
}

export function PriorityTable({ rows, selected, onSelect, mode, compact }: { rows: PriorityRow[]; selected?: string | null; onSelect?: (r: PriorityRow) => void; mode: "real" | "lab"; compact?: boolean }) {
  if (!rows.length) return <div className="p-3 text-xs text-dim">No block is above the 15 g/s screening level in this window.</div>;
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-left text-dim">
          <th className="px-2 py-1 font-normal">#</th>
          <th className="px-2 py-1 font-normal">{mode === "real" ? "Area" : "Block"}</th>
          <th className="px-2 py-1 font-normal">Likely</th>
          <th className="px-2 py-1 text-right font-normal whitespace-nowrap">Model est.</th>
          {mode === "lab" ? <th className="px-2 py-1 text-right font-normal">Hit rate</th> : <th className="px-2 py-1 font-normal">Last 7 days</th>}
          {mode === "real" && !compact && <th className="px-2 py-1 font-normal">Next step</th>}
        </tr>
      </thead>
      <tbody>
        {rows.map((r, k) => {
          const id = `${r.by}-${r.bx}`;
          return (
            <tr key={id} tabIndex={onSelect ? 0 : undefined} onClick={() => onSelect?.(r)} onKeyDown={(e) => { if (e.key === "Enter") onSelect?.(r); }}
              className={`border-t border-line ${onSelect ? "cursor-pointer hover:bg-panel-2" : ""} ${selected === id ? "bg-panel-2" : ""}`}>
              <td className="num px-2 py-1.5">{k + 1}</td>
              <td className="px-2 py-1.5">{r.area ?? `row ${r.by}, col ${r.bx}`}</td>
              <td className="px-2 py-1.5"><Chip cat={r.cat} /></td>
              <td className="num px-2 py-1.5 text-right whitespace-nowrap">{r.g_s.toFixed(0)} g/s</td>
              {mode === "lab" ? <td className="num px-2 py-1.5 text-right">{r.p == null ? "–" : `${Math.round(r.p * 100)}%`}</td> : (
                <td className="px-2 py-1.5"><div className="flex gap-0.5" aria-label={`Flagged on ${(r.hist ?? []).filter(Boolean).length} of the last 7 days at this hour`}>
                  {(r.hist ?? []).map((h, i) => <span key={i} className="h-2 w-2 rounded-sm" style={{ background: h == null ? "transparent" : h ? "#f59e0b" : "#263146", border: h == null ? "1px solid #263146" : undefined }} />)}
                </div></td>
              )}
              {mode === "real" && !compact && <td className="px-2 py-1.5 text-dim">{r.action}</td>}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export function downloadCsv(name: string, rows: PriorityRow[]) {
  const head = "rank,area,likely_source,model_estimate_g_s,days_flagged_of_last_7,next_step\n";
  const esc = (s: string) => `"${s.replace(/"/g, '""')}"`;
  const body = rows.map((r, k) => [k + 1, esc(r.area ?? `${r.by},${r.bx}`), r.cat, r.g_s, (r.hist ?? []).filter(Boolean).length, esc(r.action ?? "")].join(",")).join("\n");
  const url = URL.createObjectURL(new Blob([head + body + "\n"], { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}
