"""Phase 7: build the offline data pack for the web app (app/data/).

  map.json            roads, land use, monitors and named areas in km on the model grid
  real_meta.json      hour axis, monitor readings, weather, background, priority lists
  real_fields.bin     one record per 6 h window (uint16, 0.25 µg/m³ units; see REAL_LAYOUT)
  lab_scenarios.npz   held-out synthetic scenarios with truth and precomputed NNLS results
  lab_meta.json       scenario descriptions and calibration factors
  evidence.json       every number shown on the Evidence page, read from reports/
  hidden_story.json   hidden-emitter experiment (reports/phase7_hidden.json), when available

Priority-list blocks are described relative to public monitoring stations only, never by the names
of businesses in OSM: a 3 km block can't single out a company, and the list must not imply that it does.

Usage:  python scripts/phase7_precompute.py [--only map,real,validate,lab,evidence,hidden]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

from airtrace.attribution.contrib import Z90, calibrate_kappa
from airtrace.config import Config, load_config, project_root
from airtrace.data.dataset import SyntheticDataset
from airtrace.data.geo import LocalProjection
from airtrace.data.osm import landuse_polygons, road_lines
from airtrace.data.real import RealData, RealWindows
from airtrace.data.synthetic import last_hour_background
from airtrace.eval.interpolation import kriging
from airtrace.eval.metrics import detect, window_source_strength
from airtrace.eval.scenario import load_truth
from airtrace.grid import Grid
from airtrace.models.fno_data import BG, C_SLICE, S_SLICE, GPUData
from airtrace.models.fno_train import load_members, predict, stored_networks
from airtrace.models.greens import block_sum, response_matrix
from airtrace.models.inversion import invert, prior_block_score, prior_weights

BLOCK, TAU = 4, 15.0
SCALE = 4.0  # uint16 = value * SCALE
CATS = ("traffic", "industry", "construction", "burning")


def data_dir() -> Path:
    d = project_root() / "app" / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def dump(name: str, obj) -> None:
    path = data_dir() / name
    path.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {path.name}: {path.stat().st_size / 1e6:.2f} MB", flush=True)


def build_map(cfg: Config, data: RealData) -> list[dict]:
    """map.json; returns the named areas used to label blocks."""
    proj = LocalProjection.from_config(cfg.domain)
    osm = cfg.path("artifacts") / "cache" / "osm"
    load = lambda n: json.load(gzip.open(osm / f"{n}.json.gz", "rt", encoding="utf-8"))  # noqa: E731
    roads, landuse = load("roads"), load("landuse")
    lines, weights = road_lines(roads, proj)
    out_roads = [{"w": round(float(w), 2), "p": np.round(ln[:: max(len(ln) // 40, 1)] / 1e3, 2).tolist()}
                 for ln, w in zip(lines, weights) if w >= 0.4 and len(ln) > 1]
    polys = landuse_polygons(landuse, proj)
    keep = {k: [np.round(p[:: max(len(p) // 24, 1)] / 1e3, 2).tolist() for p in v if len(p) > 2] for k, v in polys.items()
            if k in ("industrial", "landfill", "construction")}
    x, y = proj.to_xy(data.lonlat[:, 0], data.lonlat[:, 1])
    monitors = [{"name": str(n).split(",")[0].split(" - ")[0].strip(), "full": str(n), "provider": str(p), "x": round(float(a) / 1e3, 2),
                 "y": round(float(b) / 1e3, 2), "iy": int(c[0]), "ix": int(c[1]), "mean": round(float(np.nanmean(v)), 1)}
                for n, p, a, b, c, v in zip(data.names, data.providers, x, y, data.cells, data.pm25)]
    dump("map.json", {"width_km": cfg.domain.width_m / 1e3, "height_km": cfg.domain.height_m / 1e3, "nx": cfg.domain.nx, "ny": cfg.domain.ny,
                      "block": BLOCK, "roads": out_roads, "landuse": keep, "monitors": monitors})
    return monitors


def interior_mask(nb: int) -> np.ndarray:
    """Blocks away from the domain edge. Edge blocks absorb inflow and background error, so they are never ranked."""
    m = np.zeros((nb, nb), bool)
    m[1:-1, 1:-1] = True
    return m


def block_label(by: int, bx: int, monitors: list[dict], grid: Grid) -> str:
    """A block's place relative to the nearest public monitoring station, e.g. ``3 km NW of Rohini``."""
    bkm = BLOCK * grid.dx / 1e3
    x, y = (bx + 0.5) * bkm, (by + 0.5) * bkm
    a = min(monitors, key=lambda r: (r["x"] - x) ** 2 + (r["y"] - y) ** 2)
    d = float(np.hypot(a["x"] - x, a["y"] - y))
    if d < 1.5:
        return f"Around {a['name']}"
    compass = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")[int(round(np.degrees(np.arctan2(y - a["y"], x - a["x"])) / 45)) % 8]
    return f"{d:.0f} km {compass} of {a['name']}"


def suggested_action(cat: str, mapped: bool, end_hour: int) -> str:
    if not mapped:
        return "Portable monitor first"
    if cat == "burning":
        return "Fire watch"
    if cat == "construction":
        return "Dust-control check"
    return "Night patrol, stack check" if end_hour >= 20 or end_hour < 7 else "Site inspection"


def build_real(cfg: Config, data: RealData, areas: list[dict], models, dev: str, chunk: int = 32) -> None:
    grid = Grid.from_config(cfg.domain)
    priors = np.load(cfg.path("artifacts") / "delhi_priors.npy")
    W = cfg.data.window_hours
    win = RealWindows(data, W, priors, cfg.physics.wind_speed_range, (cfg.physics.mixing_height_floor, np.inf), dev)
    proj = LocalProjection.from_config(cfg.domain)
    xs, ys = proj.to_xy(data.lonlat[:, 0], data.lonlat[:, 1])
    xy = np.column_stack([xs, ys])
    X, Y = np.meshgrid((np.arange(grid.nx) + 0.5) * grid.dx, (np.arange(grid.ny) + 0.5) * grid.dy)
    tgt = np.column_stack([X.ravel(), Y.ravel()])
    ends = np.arange(W - 1, len(data.times))
    score_nt = prior_block_score(priors, BLOCK, road_weight=0.0)
    nb = grid.ny // BLOCK
    E_all = np.zeros((len(ends), 4, nb, nb), np.float32)
    bg, bg_std, valid = np.zeros(len(ends)), np.zeros(len(ends)), np.zeros(len(ends), bool)
    t0 = time.perf_counter()
    with open(data_dir() / "real_fields.bin", "wb") as fh:
        for c0 in range(0, len(ends), chunk):
            sel = ends[c0:c0 + chunk]
            rasters = [win.rasters(int(e), grid) for e in sel]
            win.set_batch(sel)
            out = predict(models, win, torch.arange(len(sel), device=dev), torch.as_tensor(np.stack([r[0] for r in rasters]), device=dev),
                          torch.as_tensor(np.stack([r[1] for r in rasters]), device=dev)).float().cpu().numpy()
            for n, e in enumerate(sel):
                k = c0 + n
                last = data.pm25[:, e]
                ok = np.isfinite(last)
                valid[k] = ok.sum() >= 10
                krig = kriging(xy[ok], last[ok], tgt).reshape(grid.ny, grid.nx) if valid[k] else np.zeros((grid.ny, grid.nx))
                cc = out[n, :, C_SLICE]  # (M, 4, ny, nx)
                bg[k], bg_std[k] = out[n, :, BG, 0, 0].mean(), out[n, :, BG, 0, 0].std(ddof=1)
                g = grid.cell_area * win.H[e] / 3600.0 / 1e6  # S [µg m^-3 h^-1] -> g/s per cell
                E_all[k] = block_sum((out[n, :, S_SLICE] * g).mean(0), BLOCK)
                std_blocks = block_sum(cc.std(0, ddof=1), BLOCK) / BLOCK**2
                rec = np.concatenate([krig.ravel(), cc.mean(0).ravel(), std_blocks.ravel(), E_all[k].ravel()])
                fh.write(np.clip(np.round(rec * SCALE), 0, 65535).astype(np.uint16).tobytes())
            print(f"  real windows {c0 + len(sel)}/{len(ends)} ({time.perf_counter() - t0:.0f} s)", flush=True)

    # priority lists: rank by non-traffic emission, persistence at the same clock hour on earlier days
    E_nt = E_all[:, 1:].sum(1)
    inner = interior_mask(nb)
    flagged = (E_nt > TAU) & inner
    lists = []
    for k, e in enumerate(ends):
        order = np.argsort(-(E_nt[k] * inner).ravel())[:8]
        rows = []
        for j in order:
            by, bx = divmod(int(j), nb)
            if E_nt[k, by, bx] <= TAU or not inner[by, bx]:
                break
            cat = CATS[1 + int(np.argmax(E_all[k, 1:, by, bx]))]
            mapped = bool(score_nt[by, bx] >= 0.05)
            hist = [bool(flagged[k - 24 * d, by, bx]) if k - 24 * d >= 0 else None for d in range(7, 0, -1)]
            rows.append({"by": by, "bx": bx, "area": block_label(by, bx, areas, grid), "cat": cat if mapped else "unmapped",
                         "likely": cat, "g_s": round(float(E_nt[k, by, bx]), 1), "hist": hist,
                         "action": suggested_action(cat, mapped, int(data.local_hour[e]))})
        lists.append(rows)
    round_list = lambda a, n=1: [None if not np.isfinite(v) else round(float(v), n) for v in a]  # noqa: E731
    dump("real_meta.json", {
        "start": str(data.times[0]), "window": W, "scale": SCALE, "n": len(ends), "nx": grid.nx, "ny": grid.ny, "nb": nb,
        "layout": {"kriging": 1, "c_cat": 4, "std_blocks": 4, "E_blocks": 4},
        "local_hour": data.local_hour[ends].tolist(), "valid": valid.tolist(), "ends": ends.tolist(),
        "times": [str(t) for t in data.times[ends]], "bg": round_list(bg), "bg_std": round_list(bg_std),
        "u": round_list(win.u[ends], 2), "v": round_list(win.v[ends], 2), "H": round_list(data.H[ends], 0),
        "readings": [round_list(data.pm25[s, ends], 0) for s in range(len(data.names))], "priority": lists,
    })


@torch.no_grad()
def validate_priority(cfg: Config, models, dev: str) -> dict:
    """Is the ranked list useful where the truth is known? Hit rates on held-out synth_v2, against the OSM map alone."""
    grid, fc = Grid.from_config(cfg.domain), cfg.fno
    ds = SyntheticDataset(cfg.path("artifacts") / "datasets" / cfg.data.name)
    data = GPUData(ds, cfg.data.window_hours, dev)
    nb = grid.ny // BLOCK

    def blocks(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vals, masks = stored_networks(ds, idx, grid, dev)
        out = predict(models, data, torch.as_tensor(idx, device=dev), vals, masks).float().cpu().numpy()
        H = np.array([ds.get(int(i), ("mixing_height",))["mixing_height"][-1] for i in idx])
        g = grid.cell_area * H[:, None, None, None] / 3600.0 / 1e6
        E = block_sum(out[:, :, 1:4].sum(2) * g, BLOCK)  # (N, M, nb, nb), last-hour non-traffic
        S = np.stack([ds.get(int(i), ("S_cat",))["S_cat"][1:].sum(0).astype(np.float64) for i in idx])
        return E, block_sum(S * g[:, 0], BLOCK)

    val = np.arange(*fc.val_range)[:400]
    Ev, Tv = blocks(val)
    m, s = Ev.mean(1), Ev.std(1, ddof=1)
    active = (m > 5) | (Tv > 5)
    kappa = calibrate_kappa((m - Tv)[active], s[active])
    test = np.arange(*fc.test_range)[:600]
    E, T = blocks(test)
    m, s = E.mean(1), E.std(1, ddof=1)
    inner = interior_mask(nb)
    P = norm.sf((TAU - m) / np.maximum(kappa * s, 1e-6))[:, inner]
    edges = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    rel = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (P >= a) & (P < b + 1e-9)
        if sel.sum() >= 20:
            rel.append({"p_mean": float(P[sel].mean()), "observed": float((T[:, inner][sel] > TAU).mean()), "n": int(sel.sum())})
    rng = np.random.default_rng(1)
    strengths = window_source_strength(ds.specs(test), cfg.data.window_hours)
    K = 5
    hits = {"AirTrace (FNO ranking)": [], "OSM map alone": [], "Random block": []}
    for n, i in enumerate(test):
        specs = ds.specs([int(i)])
        src = specs.sources[0]
        strong = strengths[n] >= TAU
        truth = np.unique((src[strong][:, [2, 1]] // (BLOCK * grid.dx)).astype(int), axis=0).reshape(-1, 2)
        if len(truth) == 0:
            continue
        score = prior_block_score(ds.layouts[specs.layout_id[0]], BLOCK, road_weight=0.0)
        ranks = {"AirTrace (FNO ranking)": m[n].ravel(), "OSM map alone": score.ravel() + 1e-6 * rng.random(nb * nb),
                 "Random block": rng.random(nb * nb)}
        ranks = {k: np.where(inner.ravel(), r, -1.0) for k, r in ranks.items()}
        for name, r in ranks.items():
            top = np.array(np.unravel_index(np.argsort(-r)[:K], (nb, nb))).T
            d = detect(top, truth, tol=1)
            hits[name].append(d.tp_pred / K)
    res = {"k": K, "tau_g_s": TAU, "kappa_E": float(kappa), "n_test": len(hits["Random block"]), "precision_at_k": {k: float(np.mean(v)) for k, v in hits.items()},
           "reliability": rel, "z90": Z90}
    print(json.dumps(res, indent=1), flush=True)
    (cfg.path("artifacts").parent / "reports" / "phase7_priority.json").write_text(json.dumps(res, indent=2))
    return res


def build_lab(cfg: Config, n_scen: int, dev: str) -> None:
    """Held-out Delhi scenarios with truth, plus NNLS on each scenario's default sensor network."""
    grid, fc = Grid.from_config(cfg.domain), cfg.fno
    ds = SyntheticDataset(cfg.path("artifacts") / "datasets" / cfg.data.name)
    priors = np.load(cfg.path("artifacts") / "delhi_priors.npy")
    idx = [int(i) for i in range(*fc.test_range) if int(ds.get(int(i), ("layout_id",))["layout_id"]) == 0][:n_scen]
    arrays: dict[str, list] = {}
    meta = []
    w = prior_weights(prior_block_score(priors, BLOCK))
    nb = grid.ny // BLOCK
    for n, i in enumerate(idx):
        t0 = time.perf_counter()
        d = ds.get(i, ("c_hourly", "c_cat", "S_cat", "wind_pat", "wind_u", "wind_v", "pert_amp", "mixing_height", "c_bg", "K", "lam", "end_hour",
                       "sources", "sensor_cells", "sensor_readings", "sensor_noise", "c_T"))
        tr = load_truth(ds, i, grid, BLOCK, cfg.data.window_hours, 10.0)
        G = response_matrix(tr.specs, ds.layouts, grid, tr.cells, cfg.data.window_hours, BLOCK, dev, entrainment=True)
        r = invert(G.sensors, tr.readings, sigma_rel=tr.noise, sigma_abs=1.0, weights=w)
        rec = dict(c_hourly=d["c_hourly"], c_cat=d["c_cat"], S_cat=d["S_cat"], wind_pat=d["wind_pat"], wind_u=d["wind_u"][-6:], wind_v=d["wind_v"][-6:],
                   pert_amp=d["pert_amp"][-6:], H=d["mixing_height"][-6:], bg_last=np.array(last_hour_background(d["c_bg"])), K=d["K"], lam=d["lam"], c_T=d["c_T"],
                   E_blocks=tr.E_blocks, nnls_E=r.E.reshape(nb, nb), nnls_c=np.einsum("jyx,j->yx", G.field_last, r.E) + r.bg[-1],
                   default_cells=d["sensor_cells"], default_readings=d["sensor_readings"])
        for k, v in rec.items():
            arrays.setdefault(k, []).append(np.asarray(v, np.float32))
        speed = np.hypot(d["wind_u"][-1], d["wind_v"][-1])
        direction = float(np.degrees(np.arctan2(-d["wind_u"][-1], -d["wind_v"][-1])) % 360)
        meta.append({"id": n, "index": i, "end_hour": int(d["end_hour"]), "wind_speed": round(float(speed), 1), "wind_from": round(direction),
                     "H": round(float(d["mixing_height"][-1])), "median_pm25": round(float(np.median(d["c_hourly"][-1]))),
                     "sources": [{"x": round(float(a) / 1e3, 2), "y": round(float(b) / 1e3, 2)} for a, b in tr.source_xy]})
        print(f"  lab scenario {n + 1}/{len(idx)} #{i} ({time.perf_counter() - t0:.0f} s)", flush=True)
    np.savez_compressed(data_dir() / "lab_scenarios.npz", priors=priors.astype(np.float32),
                        **{k: np.stack(v) for k, v in arrays.items()})
    cal = json.loads((cfg.path("artifacts") / "fno" / "calibration.json").read_text())
    dump("lab_meta.json", {"scenarios": meta, "calibration": cal, "block": BLOCK, "tau_g_s": TAU, "window": cfg.data.window_hours})


def build_evidence(cfg: Config, priority: dict | None) -> None:
    rep = cfg.path("artifacts").parent / "reports"
    load = lambda n: json.loads((rep / n).read_text(encoding="utf-8"))  # noqa: E731
    v1, p6v1, p6v2, s5v2 = load("phase5_metrics.json"), load("phase6_metrics_synth_v1.json"), load("phase6_metrics.json"), load("phase5_metrics_synth_v2.json")
    if priority is None and (rep / "phase7_priority.json").exists():
        priority = load("phase7_priority.json")
    dump("evidence.json", {"synthetic_v1": {k: v1[k] for k in ("table", "curves", "timing", "coverage", "levels", "calibration", "share_err_pp", "share_err_pp_mean")},
                           "synthetic_v2": {k: s5v2[k] for k in ("table", "coverage", "levels", "calibration", "share_err_pp", "share_err_pp_mean")},
                           "real_v1": p6v1, "real_v2": p6v2, "priority": priority})


def build_hidden(cfg: Config) -> None:
    path = cfg.path("artifacts").parent / "reports" / "phase7_hidden.json"
    if not path.exists():
        print("no hidden-emitter results yet")
        return
    dump("hidden_story.json", json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--only", default="map,real,validate,lab,evidence,hidden")
    parser.add_argument("--n-lab", type=int, default=8)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(args.config)
    parts = set(args.only.split(","))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = RealData.load(cfg.path("artifacts") / "real" / f"delhi_{cfg.real.start}_{cfg.real.end}.npz")
    areas = build_map(cfg, data) if parts & {"map", "real"} else []
    models = load_members(sorted((cfg.path("artifacts") / "fno").glob("member_*.pt")), dev) if parts & {"real", "validate"} else None
    if "real" in parts:
        build_real(cfg, data, areas, models, dev)
    priority = validate_priority(cfg, models, dev) if "validate" in parts else None
    if "lab" in parts:
        build_lab(cfg, args.n_lab, dev)
    if "evidence" in parts:
        build_evidence(cfg, priority)
    if "hidden" in parts:
        build_hidden(cfg)


if __name__ == "__main__":
    main()
