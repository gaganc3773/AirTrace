"""Phase 7: can combining nights with different winds locate an unmapped night-only emitter?

For each trial, an inventory on the Delhi layout gets one extra off-prior, night-only source. The
same inventory is observed by the real monitor cells over consecutive 20:00-02:00 IST windows of
ERA5 weather from a smog season. Each night is inverted separately with prior-free NNLS, and block
emissions are averaged over 1, 2, 3 and 5 nights. Control runs repeat every trial without the
hidden source, so the score can be read against false alarms.

Usage:  python scripts/phase7_hidden.py [--trials 12 --nights 5]
Writes reports/phase7_hidden.json (statistics plus the maps of every trial).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

import numpy as np
import torch

from airtrace.config import load_config
from airtrace.data.geo import LocalProjection
from airtrace.data.openmeteo import met_bank
from airtrace.data.real import RealData
from airtrace.data.sensors import log_error
from airtrace.data.synthetic import ForcingBuilder, ScenarioSpecs, sample_specs, simulate
from airtrace.grid import Grid
from airtrace.models.greens import response_matrix
from airtrace.models.inversion import invert, prior_block_score
from airtrace.solver import ADRSolver

BLOCK = 4
END_LOCAL = 2  # windows end at 02:00 IST
NIGHT_SCHEDULE = 2
COMBOS = (1, 2, 3, 5)


def night_specs(base: ScenarioSpecs, met, start: int, T: int, H_floor: float, bg0: float, alpha: float) -> ScenarioSpecs:
    """``base`` (one scenario) under the ERA5 segment beginning at bank index ``start``."""
    mid = lambda a: 0.5 * (a[start:start + T] + a[start + 1:start + T + 1])  # noqa: E731
    H = np.maximum(mid(met.H), H_floor)
    utc_end = int((met.times[start + T] - met.times[start + T].astype("datetime64[D]")).astype(int))
    return dataclasses.replace(base, end_hour=np.array([(utc_end + 5) % 24]), wind_u=mid(met.u)[None], wind_v=mid(met.v)[None],
                               mixing_height=H[None], c_bg=(bg0 * (500.0 / H) ** alpha)[None],
                               pert_amp=(0.1 * np.hypot(mid(met.u), mid(met.v)))[None])


def readings(sp: ScenarioSpecs, layouts, grid, solver, cells, W, dev, rng) -> np.ndarray:
    with torch.no_grad():
        c = simulate(ForcingBuilder(sp, layouts, grid, dev, entrainment=True), solver, W)["c_hourly"][0].cpu().numpy()
    truth = c[:, cells[:, 0], cells[:, 1]].T.astype(np.float64)
    return truth * np.exp(log_error(rng, truth.shape, 0.15, 0.15, 0.7))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--nights", type=int, default=5)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid, proj = Grid.from_config(cfg.domain), LocalProjection.from_config(cfg.domain)
    dc = cfg.data.model_copy(update={"delhi_fraction": 1.0})
    W, T = dc.window_hours, dc.sim_hours
    layouts = np.load(cfg.path("artifacts") / "datasets" / dc.name / "layouts.npy")
    real = RealData.load(cfg.path("artifacts") / "real" / f"delhi_{cfg.real.start}_{cfg.real.end}.npz")
    cells = np.unique(real.cells, axis=0).astype(np.int16)
    met = met_bank(proj, dc.met_seasons, cfg.path("artifacts") / "cache" / "real", cfg.real.met_points)
    starts = met.starts(T + 24 * args.nights)
    hour = lambda i: int((met.times[i] - met.times[i].astype("datetime64[D]")).astype(int))  # noqa: E731
    starts = np.array([s for s in starts if (hour(s + T) + 5) % 24 == END_LOCAL])
    solver = ADRSolver.from_config(grid, cfg.solver)
    rng = np.random.default_rng(cfg.seed + 7)
    nb = grid.ny // BLOCK
    score = prior_block_score(layouts[0], BLOCK, road_weight=0.0)
    interior = np.zeros((nb, nb), bool)
    interior[2:-2, 2:-2] = True
    candidates = np.argwhere((score < 0.05) & interior)

    out = []
    t0 = time.perf_counter()
    for trial in range(args.trials):
        base = sample_specs(rng, 1, layouts, grid, dc, cfg.physics, met)
        by, bx = candidates[rng.integers(len(candidates))]
        x, y = (bx + rng.uniform(0.25, 0.75)) * BLOCK * grid.dx, (by + rng.uniform(0.25, 0.75)) * BLOCK * grid.dy
        strength = float(rng.uniform(30.0, 80.0))  # g/s daily mean; 2.4x that while running at night
        with_src = base.sources.copy()
        row = int(np.argmin(with_src[0, :, 7]))  # first unused row
        with_src[0, row] = (1, x, y, 300.0, strength * 1e6, NIGHT_SCHEDULE, 1.0, 1.0)
        s0 = int(rng.choice(starts))
        bg0, alpha = float(np.exp(rng.uniform(*np.log(cfg.physics.background_range)))), float(rng.uniform(*dc.bg_blh_exponent))
        maps = {"hidden": [], "control": []}
        winds = []
        for k in range(args.nights):
            sp0 = night_specs(base, met, s0 + 24 * k, T, cfg.physics.mixing_height_floor, bg0, alpha)
            G = response_matrix(sp0, layouts, grid, cells, W, BLOCK, dev, entrainment=True).sensors
            winds.append([float(sp0.wind_u[0, -W:].mean()), float(sp0.wind_v[0, -W:].mean())])
            for name, table in (("hidden", with_src), ("control", base.sources)):
                y_obs = readings(dataclasses.replace(sp0, sources=table), layouts, grid, solver, cells, W, dev, rng)
                maps[name].append(invert(G, y_obs, sigma_rel=0.2, sigma_abs=1.0).E.reshape(nb, nb))
        truth = (int(by), int(bx))
        entry = {"trial": trial, "truth_block": truth, "truth_xy_km": [x / 1e3, y / 1e3], "strength_g_s": strength, "winds": winds,
                 "maps": {k: np.round(np.array(v), 1).tolist() for k, v in maps.items()}}
        out.append(entry)
        print(f"trial {trial + 1}/{args.trials} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

    off = (score < 0.05) & interior  # blocks at the domain edge absorb background mismatch in the inversion
    block_km = BLOCK * grid.dx / 1e3
    stats = {}
    for k in COMBOS:
        if k > args.nights:
            continue
        errs, hits, hid_top, ctl_top = [], [], [], []
        for e in out:
            m = np.array(e["maps"]["hidden"])[:k].mean(0)
            c = np.array(e["maps"]["control"])[:k].mean(0)
            mo, co = np.where(off, m, -1.0), np.where(off, c, -1.0)
            j = np.array(np.unravel_index(np.argmax(mo), mo.shape))
            d = float(np.hypot(*(j - np.array(e["truth_block"])))) * block_km
            errs.append(d), hits.append(bool(np.abs(j - np.array(e["truth_block"])).max() <= 1))
            hid_top.append(float(mo.max())), ctl_top.append(float(co.max()))
        hid_top, ctl_top = np.array(hid_top), np.array(ctl_top)
        stats[str(k)] = {"median_error_km": float(np.median(errs)), "hit_within_1_block": float(np.mean(hits)),
                         "top_score_hidden_median": float(np.median(hid_top)), "top_score_control_median": float(np.median(ctl_top)),
                         "hidden_beats_control": float(np.mean(hid_top[:, None] > ctl_top[None]))}
    # chance of a random off-prior block landing within one block of the truth
    chance = []
    for e in out:
        by, bx = e["truth_block"]
        near = np.zeros((nb, nb), bool)
        near[max(by - 1, 0):by + 2, max(bx - 1, 0):bx + 2] = True
        chance.append((near & off).sum() / off.sum())
    summary = {"trials": args.trials, "nights": args.nights, "chance_hit": float(np.mean(chance)), "stats": stats, "runs": out}
    path = cfg.path("artifacts").parent / "reports" / "phase7_hidden.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("trials", "chance_hit", "stats")}, indent=2))


if __name__ == "__main__":
    main()
