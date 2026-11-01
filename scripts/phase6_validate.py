"""Phase 6: leave-one-station-out (LOSO) validation on real Delhi data, and source shares over the month.

For every window end (every ``real.eval_stride_hours``) and every station with a valid
last-hour reading, the station (and any other station in its grid cell) is removed from the
network; the FNO ensemble, ordinary kriging, IDW and the network median then predict its
last-hour PM2.5. The full network is also run once per window for the attribution time series.

Usage:  python scripts/phase6_validate.py
Writes reports/phase6_metrics.{md,json} and reports/figures/phase6_*.png.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from airtrace.attribution.contrib import SHARE_NAMES, Z90, receptor_shares
from airtrace.config import load_config
from airtrace.data.geo import LocalProjection
from airtrace.data.real import RealData, RealWindows
from airtrace.eval.interpolation import idw, kriging
from airtrace.eval.plotting import PALETTE, savefig, set_style
from airtrace.grid import Grid
from airtrace.models.fno_data import BG, C_SLICE
from airtrace.models.fno_train import load_members, predict

METHODS = ("FNO ensemble", "FNO + residual kriging", "Kriging", "IDW", "Network median")
H_TRAIN = (175.0, 1560.0)  # mixing heights spanned by the synthetic training scenarios [m]


def metrics(pred: np.ndarray, obs: np.ndarray) -> dict:
    e = pred - obs
    return {"rmse": float(np.sqrt(np.mean(e**2))), "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e)),
            "r": float(np.corrcoef(pred, obs)[0, 1]), "rel_rmse": float(np.sqrt(np.mean(e**2)) / np.mean(obs)), "n": int(len(obs))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--chunk", type=int, default=16, help="windows per GPU batch")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(args.config)
    set_style()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid = Grid.from_config(cfg.domain)
    proj = LocalProjection.from_config(cfg.domain)
    data = RealData.load(cfg.path("artifacts") / "real" / f"delhi_{cfg.real.start}_{cfg.real.end}.npz")
    priors = np.load(cfg.path("artifacts") / "delhi_priors.npy")
    models = load_members(sorted((cfg.path("artifacts") / "fno").glob("member_*.pt")), dev)
    cal = json.loads((cfg.path("artifacts") / "fno" / "calibration.json").read_text())
    W, S = cfg.data.window_hours, len(data.names)
    win = RealWindows(data, W, priors, cfg.physics.wind_speed_range, H_TRAIN, dev)
    print(f"{S} stations, {len(data.times)} h; met clipped to training range: {win.clipped_fraction}")

    xs, ys = proj.to_xy(data.lonlat[:, 0], data.lonlat[:, 1])
    xy = np.column_stack([xs, ys])
    same_cell = (data.cells[:, None] == data.cells[None]).all(-1)
    ends = np.arange(W - 1, len(data.times), cfg.real.eval_stride_hours)

    rec = {k: [] for k in ("end", "station", "obs", *METHODS)}
    fno_members, shares_t, shares_end = [], [], []
    t0 = time.perf_counter()
    for c0 in range(0, len(ends), args.chunk):
        vals, masks, sample_end, sample_station, sample_others = [], [], [], [], []
        for end in ends[c0:c0 + args.chunk]:
            last = data.pm25[:, end]
            test = np.flatnonzero(np.isfinite(last))
            if len(test) < 10:
                continue
            for s in [-1, *test]:  # -1: full network
                v, m = win.rasters(end, grid, None if s < 0 else same_cell[s])
                others = np.flatnonzero(np.isfinite(last) & ~(same_cell[s] if s >= 0 else False))
                vals.append(v), masks.append(m), sample_end.append(end), sample_station.append(s)
                sample_others.append(others)
                if s < 0:
                    continue
                rec["end"].append(int(end)), rec["station"].append(int(s)), rec["obs"].append(float(last[s]))
                rec["Kriging"].append(float(kriging(xy[others], last[others], xy[s:s + 1])[0]))
                rec["IDW"].append(float(idw(xy[others], last[others], xy[s:s + 1])[0]))
                rec["Network median"].append(float(np.median(last[others])))
        if not vals:
            continue
        win.set_batch(np.array(sample_end))
        out = predict(models, win, torch.arange(len(vals), device=dev), torch.as_tensor(np.stack(vals), device=dev),
                      torch.as_tensor(np.stack(masks), device=dev)).float().cpu().numpy()  # (N, M, 11, ny, nx)
        c_cat, bg = out[:, :, C_SLICE], out[:, :, BG, 0, 0]
        for n, s in enumerate(sample_station):
            if s >= 0:
                iy, ix = data.cells[s]
                fno_members.append(c_cat[n, :, :, iy, ix].sum(-1) + bg[n])
                # regression kriging: krige the FNO misfit at the remaining stations onto the held-out one
                c_hat = c_cat[n].sum(1).mean(0) + bg[n].mean()
                o = sample_others[n]
                resid = data.pm25[o, sample_end[n]] - c_hat[data.cells[o, 0], data.cells[o, 1]]
                rec["FNO + residual kriging"].append(float(c_hat[iy, ix] + kriging(xy[o], resid, xy[s:s + 1])[0]))
            else:  # station-averaged shares, per member
                ok = np.isfinite(data.pm25[:, sample_end[n]])
                sh = np.stack([receptor_shares(c_cat[n], bg[n], iy, ix) for iy, ix in data.cells[ok]])  # (S', M, 5)
                shares_t.append(sh.mean(0)), shares_end.append(sample_end[n])
        print(f"  windows {c0 + len(ends[c0:c0 + args.chunk])}/{len(ends)}  ({time.perf_counter() - t0:.0f} s)", flush=True)

    fm = np.array(fno_members)  # (N, M)
    rec["FNO ensemble"] = fm.mean(1).tolist()
    obs = np.array(rec["obs"])
    station, end = np.array(rec["station"]), np.array(rec["end"])
    local = data.local_hour[end]
    night = (local >= 20) | (local < 8)
    res = {m: metrics(np.array(rec[m]), obs) for m in METHODS}
    res_night = {m: metrics(np.array(rec[m])[night], obs[night]) for m in METHODS}
    res_day = {m: metrics(np.array(rec[m])[~night], obs[~night]) for m in METHODS}
    sigma = fm.std(1, ddof=1)
    err = fm.mean(1) - obs
    cov_raw = float(np.mean(np.abs(err) <= Z90 * sigma))
    cov_cal = float(np.mean(np.abs(err) <= Z90 * cal["kappa_field"] * sigma))
    per_station = {str(data.names[s]): {m: float(np.sqrt(np.mean((np.array(rec[m])[station == s] - obs[station == s]) ** 2)))
                                        for m in METHODS} for s in range(S) if np.any(station == s)}
    wins = float(np.mean([v["FNO ensemble"] < v["Kriging"] for v in per_station.values()]))
    wins_rk = float(np.mean([v["FNO + residual kriging"] < v["Kriging"] for v in per_station.values()]))
    sh = np.array(shares_t)  # (Nw, M, 5)
    share_mean = {n: float(sh[:, :, k].mean()) for k, n in enumerate(SHARE_NAMES)}
    summary = {"n_stations": S, "n_windows": len(shares_end), "n_predictions": len(obs), "met_clipped": win.clipped_fraction,
               "all": res, "night_20_08": res_night, "day_08_20": res_day,
               "coverage90": {"raw": cov_raw, "calibrated": cov_cal, "kappa_field": cal["kappa_field"]},
               "fno_beats_kriging_station_fraction": wins,
               "fno_rk_beats_kriging_station_fraction": wins_rk, "per_station_rmse": per_station, "mean_shares": share_mean}
    report = cfg.path("artifacts").parent / "reports"
    (report / "phase6_metrics.json").write_text(json.dumps(summary, indent=2))
    write_markdown(summary, report, cfg)
    np.savez_compressed(cfg.path("artifacts") / "real" / "phase6_predictions.npz", end=end, station=station, obs=obs,
                        fno_members=fm, **{m.replace(" ", "_"): np.array(rec[m]) for m in METHODS[1:]},
                        shares=sh, shares_end=np.array(shares_end))
    figures(data, rec, fm, station, end, sh, np.array(shares_end), per_station, cal, cfg)


def write_markdown(s: dict, report, cfg) -> None:
    L = [f"## Leave-one-station-out, {cfg.real.start} to {cfg.real.end} ({s['n_stations']} stations, "
         f"{s['n_predictions']} held-out predictions)", "",
         "| Hours (IST) | Method | RMSE [µg/m³] | MAE | Bias | r | RMSE / mean |", "|---|---|---|---|---|---|---|"]
    for key, lab in (("all", "all"), ("night_20_08", "20-08 (night)"), ("day_08_20", "08-20 (day)")):
        for m, r in s[key].items():
            L.append(f"| {lab} | {'**' + m + '**' if 'FNO' in m else m} | {r['rmse']:.1f} | {r['mae']:.1f} | "
                     f"{r['bias']:+.1f} | {r['r']:.2f} | {r['rel_rmse']:.2f} |")
    c = s["coverage90"]
    L += ["", f"Per-station RMSE: FNO beats kriging at {s['fno_beats_kriging_station_fraction']:.0%} of stations, "
          f"FNO + residual kriging at {s['fno_rk_beats_kriging_station_fraction']:.0%}. "
          f"Nominal 90% FNO intervals cover {c['raw']:.0%} of held-out readings raw and {c['calibrated']:.0%} with the "
          f"synthetic-data κ = {c['kappa_field']:.2f}.", "",
          f"Meteorology clipped to the training range: wind speed {s['met_clipped']['wind_speed']:.0%} of hours, "
          f"mixing height {s['met_clipped']['H']:.0%} of hours.", "",
          "## Mean source shares at the monitors (FNO ensemble, full network)", "",
          "| " + " | ".join(SHARE_NAMES) + " |", "|" + "---|" * len(SHARE_NAMES),
          "| " + " | ".join(f"{100 * v:.0f}%" for v in s["mean_shares"].values()) + " |"]
    (report / "phase6_metrics.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def figures(data, rec, fm, station, end, sh, sh_end, per_station, cal, cfg) -> None:
    out = cfg.path("figures")
    obs = np.array(rec["obs"])
    colors = dict(zip(METHODS, (PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3], "grey")))
    # 1. predicted vs observed
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), layout="constrained", sharex=True, sharey=True)
    lim = np.percentile(obs, 99.5) * 1.05
    for ax, m in zip(axes, METHODS[:4]):
        p = np.array(rec[m])
        ax.hexbin(obs, p, gridsize=50, extent=(0, lim, 0, lim), mincnt=1, cmap="Blues", bins="log")
        ax.plot([0, lim], [0, lim], "k:", lw=1)
        r = metrics(p, obs)
        ax.set(title=f"{m}: RMSE {r['rmse']:.0f}, r = {r['r']:.2f}", xlabel="observed PM2.5 [µg/m³]", xlim=(0, lim), ylim=(0, lim))
    axes[0].set_ylabel("predicted at the held-out station")
    print("saved", savefig(fig, out, "phase6_loso_scatter"))

    # 2. per-station RMSE
    names = sorted(per_station, key=lambda n: per_station[n]["Kriging"])
    fig, ax = plt.subplots(figsize=(15, 4.8), layout="constrained")
    xk = np.arange(len(names))
    for j, m in enumerate(METHODS[:3]):
        ax.bar(xk + 0.27 * (j - 1), [per_station[n][m] for n in names], 0.27, color=colors[m], label=m)
    ax.plot(xk, [per_station[n]["Network median"] for n in names], "_", color="black", ms=8, label="network median")
    ax.set_xticks(xk, [n.split(",")[0][:22] for n in names], rotation=75, fontsize=7)
    ax.set(ylabel="LOSO RMSE [µg/m³]", title="Held-out error per station (sorted by kriging error)")
    ax.legend(fontsize=8)
    print("saved", savefig(fig, out, "phase6_loso_stations"))

    # 3. time series at two held-out stations (best and median FNO-vs-kriging gain)
    gain = {n: per_station[n]["Kriging"] - per_station[n]["FNO + residual kriging"] for n in per_station}
    order = sorted(gain, key=gain.get)
    pick = [order[-1], order[len(order) // 2]]
    t = data.times.astype(object)
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), layout="constrained", sharex=True)
    for ax, name in zip(axes, pick):
        s = int(np.flatnonzero(data.names == name)[0])
        sel = station == s
        tt = np.array([t[e] for e in end[sel]])
        mu, sd = fm[sel].mean(1), fm[sel].std(1, ddof=1)
        ax.plot(t, data.pm25[s], color="black", lw=0.9, alpha=0.8, label="observed (hourly)")
        ax.fill_between(tt, mu - Z90 * cal["kappa_field"] * sd, mu + Z90 * cal["kappa_field"] * sd, color=PALETTE[0],
                        alpha=0.2, label="FNO 90% interval (calibrated)")
        ax.plot(tt, mu, ".-", color=PALETTE[0], ms=3, lw=0.8, label="FNO, station held out")
        ax.plot(tt, np.array(rec["Kriging"])[sel], ".-", color=PALETTE[2], ms=3, lw=0.8, label="kriging, station held out")
        ax.plot(tt, np.array(rec["FNO + residual kriging"])[sel], ".-", color=PALETTE[1], ms=3, lw=0.8,
                label="FNO + residual kriging")
        ax.set(ylabel="PM2.5 [µg/m³]", title=f"{name}: RMSE FNO {per_station[name]['FNO ensemble']:.0f}, FNO + residual "
                   f"kriging {per_station[name]['FNO + residual kriging']:.0f}, kriging {per_station[name]['Kriging']:.0f}")
    axes[0].legend(fontsize=8, ncol=4)
    print("saved", savefig(fig, out, "phase6_loso_timeseries"))

    # 4. attribution over the month and by hour of day
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(16, 4.6), layout="constrained", gridspec_kw={"width_ratios": [2.4, 1]})
    m = sh.mean(1) * 100  # (Nw, 5)
    a0.stackplot(np.array([t[e] for e in sh_end]), m.T, labels=SHARE_NAMES, colors=PALETTE[:5], alpha=0.85)
    a0.set(ylabel="share of PM2.5 at the monitors [%]", ylim=(0, 100), title="FNO source attribution, November 2025")
    a0.legend(loc="upper left", fontsize=8, ncol=5)
    lh = data.local_hour[sh_end]
    hours = np.unique(lh)
    for k, n in enumerate(SHARE_NAMES):
        a1.plot(hours, [m[lh == h, k].mean() for h in hours], "o-", color=PALETTE[k], label=n, ms=3)
    a1.set(xlabel="local hour (IST, end of window)", ylabel="mean share [%]", title="Diurnal cycle of shares", xticks=range(0, 24, 3))
    print("saved", savefig(fig, out, "phase6_attribution"))


if __name__ == "__main__":
    main()
