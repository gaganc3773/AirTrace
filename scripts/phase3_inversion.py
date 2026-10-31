"""Phase 3: linear Green's-function inversion baseline on held-out synthetic scenarios.

Usage:
    python scripts/phase3_inversion.py --n-test 100 --n-easy 30
    python scripts/phase3_inversion.py --pilot 4          # quick look at a few scenarios
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from airtrace.config import load_config
from airtrace.data.dataset import SyntheticDataset
from airtrace.eval.interpolation import idw, kriging
from airtrace.eval.plotting import FIELD_CMAP, savefig, set_style
from airtrace.eval.scenario import load_truth, score, summarize
from airtrace.grid import Grid
from airtrace.models.greens import block_sum, response_matrix
from airtrace.models.inversion import invert, prior_block_score, prior_weights

BLOCK, TAU, SRC_MIN = 4, 15.0, 10.0  # 3 km blocks; hotspot threshold and "strong source" cut-off [g/s]
TEST_START = 18000  # last 2,000 scenarios are the held-out test split for every phase
METHODS = ("NNLS-L1", "NNLS-L1 + OSM prior")


def is_easy(ds: SyntheticDataset, i: int) -> bool:
    """Moderate steady wind, a well-populated network and low noise."""
    d = ds.get(i, ("wind_u", "wind_v", "sensor_readings", "sensor_noise"))
    speed = np.hypot(d["wind_u"][-6:], d["wind_v"][-6:]).mean()
    return bool(1.5 <= speed <= 4.5 and np.isfinite(d["sensor_readings"][:, -1]).sum() >= 30 and d["sensor_noise"] <= 0.15)


def run_scenario(ds, i, grid, window, device, cache_dir) -> dict:
    truth = load_truth(ds, i, grid, BLOCK, window, SRC_MIN)
    cache = cache_dir / f"G_{i:05d}.npz"
    t0 = time.perf_counter()
    if cache.exists():
        z = np.load(cache)
        G, F = z["sensors"], z["field_last"]
    else:
        resp = response_matrix(truth.specs, ds.layouts, grid, truth.cells, window, BLOCK, device)
        G, F = resp.sensors, resp.field_last
        np.savez_compressed(cache, sensors=G, field_last=F)
    t_green = time.perf_counter() - t0
    nb = grid.ny // BLOCK
    weights = {"NNLS-L1": None, "NNLS-L1 + OSM prior": prior_weights(prior_block_score(ds.layouts[truth.specs.layout_id[0]], BLOCK))}
    rows, maps = {}, {}
    for name, w in weights.items():
        r = invert(G, truth.readings, sigma_rel=truth.noise, sigma_abs=1.0, weights=w)
        c_hat = np.einsum("jyx,j->yx", F, r.E) + r.bg[-1]
        E_hat = r.E.reshape(nb, nb)
        rows[name] = {**score(truth, grid, BLOCK, TAU, E_hat, c_hat, float(r.bg[-1])),
                      "seconds": r.seconds, "beta": r.beta, "chi2": r.chi2}
        maps[name] = (E_hat, c_hat)
    online = np.isfinite(truth.readings[:, -1])
    pts = (truth.cells[online][:, ::-1] + 0.5) * grid.dx
    vals = truth.readings[online, -1]
    X, Y = np.meshgrid((np.arange(grid.nx) + 0.5) * grid.dx, (np.arange(grid.ny) + 0.5) * grid.dy)
    tgt = np.column_stack([X.ravel(), Y.ravel()])
    for name, fn in (("IDW", lambda: idw(pts, vals, tgt)), ("Kriging", lambda: kriging(pts, vals, tgt, truth.noise))):
        t1 = time.perf_counter()
        c_hat = fn().reshape(grid.ny, grid.nx)
        rows[name] = {**score(truth, grid, BLOCK, TAU, c_hat=c_hat), "seconds": time.perf_counter() - t1}
        maps[name] = (None, c_hat)
    return {"index": i, "truth": truth, "rows": rows, "maps": maps, "t_green": t_green}


def example_figure(results: list[dict], grid: Grid, out, name: str) -> None:
    """True vs estimated block emissions and fields for a few scenarios."""
    n = len(results)
    fig, axes = plt.subplots(n, 6, figsize=(21, 3.5 * n), layout="constrained", squeeze=False)
    ext = grid.extent_km()
    for row, res in zip(axes, results):
        t = res["truth"]
        vmax = max(t.E_blocks.max(), 1)
        cells = t.cells[np.isfinite(t.readings[:, -1])]
        sx, sy = (cells[:, 1] + 0.5) * grid.dx / 1e3, (cells[:, 0] + 0.5) * grid.dy / 1e3
        panels = [("true emission [g/s/block]", t.E_blocks, "E"),
                  ("NNLS-L1", res["maps"]["NNLS-L1"][0], "E"),
                  ("NNLS-L1 + OSM prior", res["maps"]["NNLS-L1 + OSM prior"][0], "E"),
                  ("true PM2.5, last hour", t.c_last, "c"),
                  ("NNLS-L1 + prior reconstruction", res["maps"]["NNLS-L1 + OSM prior"][1], "c"),
                  ("kriging", res["maps"]["Kriging"][1], "c")]
        cn = LogNorm(vmin=max(t.c_last.min(), 1), vmax=t.c_last.max())
        for ax, (title, arr, kind) in zip(row, panels):
            if kind == "E":
                im = ax.imshow(arr, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax)
                ax.plot(t.source_xy[:, 0] / 1e3, t.source_xy[:, 1] / 1e3, "r+", ms=9, mew=1.5)
            else:
                im = ax.imshow(np.clip(arr, cn.vmin, None), origin="lower", extent=ext, cmap=FIELD_CMAP, norm=cn)
                ax.scatter(sx, sy, s=7, c="white", edgecolors="black", linewidths=0.4)
            ax.set_title(title, fontsize=9)
            ax.grid(False)
            fig.colorbar(im, ax=ax, shrink=0.8)
        row[0].set_ylabel(f"#{res['index']}")
    print("saved", savefig(fig, out, name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--n-easy", type=int, default=30)
    parser.add_argument("--pilot", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_style()
    grid = Grid.from_config(cfg.domain)
    ds = SyntheticDataset(cfg.path("artifacts") / "datasets" / cfg.data.name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = cfg.path("artifacts") / "greens"
    cache_dir.mkdir(exist_ok=True)
    out = cfg.path("figures")
    rng = np.random.default_rng(cfg.seed)
    test = np.arange(TEST_START, len(ds))
    easy_pool = [int(i) for i in rng.permutation(test) if is_easy(ds, int(i))]

    if args.pilot:
        res = [run_scenario(ds, i, grid, cfg.data.window_hours, device, cache_dir) for i in easy_pool[: args.pilot]]
        for r in res:
            print(f"#{r['index']}  Green's fn {r['t_green']:.1f}s  true total {r['truth'].E_blocks.sum():.0f} g/s")
            for m, row in r["rows"].items():
                print(f"   {m:22s} " + "  ".join(f"{k}={v:.3g}" for k, v in row.items()))
        example_figure(res, grid, out, "phase3_pilot")
        return

    easy = easy_pool[: args.n_easy]
    general = [int(i) for i in rng.choice(test, args.n_test, replace=False)]
    results: dict[int, dict] = {}
    for k, i in enumerate(sorted(set(easy) | set(general))):
        results[i] = run_scenario(ds, i, grid, cfg.data.window_hours, device, cache_dir)
        print(f"[{k + 1}/{len(set(easy) | set(general))}] #{i} G {results[i]['t_green']:.1f}s", flush=True)

    table = {}
    for subset, idx in (("easy", easy), ("all", general)):
        for m in METHODS + ("IDW", "Kriging"):
            table[f"{subset}|{m}"] = summarize([results[i]["rows"][m] for i in idx])
    t_green = float(np.mean([r["t_green"] for r in results.values()]))
    report = cfg.path("artifacts").parent / "reports"
    (report / "phase3_metrics.json").write_text(json.dumps({"table": table, "t_green_s": t_green,
                                                            "easy": easy, "general": general}, indent=2))
    cols = [("precision", "Precision", "{:.2f}"), ("recall", "Recall", "{:.2f}"),
            ("loc_km_median", "Loc. err [km]", "{:.1f}"), ("total_emission_rel_err", "|ΔQ|/Q", "{:.2f}"),
            ("field_rmse", "Field RMSE [µg/m³]", "{:.1f}"), ("field_rel_rmse", "Rel. RMSE", "{:.2f}"),
            ("local_share_err_pp", "Local-share err [pp]", "{:.1f}"), ("seconds", "Solve time [s]", "{:.2f}")]
    lines = [f"Hotspot threshold {TAU:g} g/s per 3 km block; match tolerance 1 block. Green's functions: "
             f"{t_green:.1f} s per scenario on {device}.", "",
             "| Subset | Method | " + " | ".join(c[1] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for key, s in table.items():
        subset, m = key.split("|")
        lines.append(f"| {subset} (n={len(easy) if subset == 'easy' else len(general)}) | {m} | "
                     + " | ".join(fmt.format(s[k]) if k in s and np.isfinite(s[k]) else "–" for k, _, fmt in cols) + " |")
    (report / "phase3_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    example_figure([results[i] for i in easy[:4]], grid, out, "phase3_examples_easy")
    example_figure([results[i] for i in general[:4]], grid, out, "phase3_examples_general")


if __name__ == "__main__":
    main()
