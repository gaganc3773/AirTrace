"""Phase 6: fetch and cache real Delhi PM2.5 (OpenAQ) and meteorology (Open-Meteo) for ``cfg.real``.

Usage:  python scripts/phase6_fetch.py
Needs an OpenAQ key ($OPENAQ_API_KEY or .openaq_key) only the first time; afterwards
everything replays from artifacts/cache. Writes artifacts/real/delhi_<start>_<end>.npz,
reports/phase6_stations.csv and reports/figures/phase6_data.png.
"""

from __future__ import annotations

import argparse
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from airtrace.config import load_config, project_root
from airtrace.data.openaq import api_key
from airtrace.data.real import build_real, write_station_table
from airtrace.eval.plotting import PALETTE, savefig, set_style
from airtrace.grid import Grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(args.config)
    set_style()
    data = build_real(cfg, api_key(project_root()))
    out = cfg.path("artifacts") / "real"
    out.mkdir(exist_ok=True)
    path = out / f"delhi_{cfg.real.start}_{cfg.real.end}.npz"
    data.save(path)
    report = cfg.path("artifacts").parent / "reports"
    write_station_table(data, report / "phase6_stations.csv")
    cov = np.isfinite(data.pm25).mean()
    print(f"saved {path}: {len(data.names)} stations x {len(data.times)} h, {cov:.0%} of station-hours valid; "
          f"PM2.5 median {np.nanmedian(data.pm25):.0f}, 95th pct {np.nanpercentile(data.pm25, 95):.0f} µg/m³")

    grid = Grid.from_config(cfg.domain)
    layout = np.load(cfg.path("artifacts") / "delhi_priors.npy")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(16, 5.2), layout="constrained", gridspec_kw={"width_ratios": [1, 2]})
    a0.imshow(layout[0], origin="lower", extent=grid.extent_km(), cmap="Greys", vmax=layout[0].max() * 0.6)
    a0.imshow(np.ma.masked_less(layout[1:].max(0), 0.05), origin="lower", extent=grid.extent_km(), cmap="Oranges",
              alpha=0.7, vmin=0, vmax=1)
    xs, ys = (data.cells[:, 1] + 0.5) * grid.dx / 1e3, (data.cells[:, 0] + 0.5) * grid.dy / 1e3
    sc = a0.scatter(xs, ys, c=np.nanmean(data.pm25, 1), cmap="magma_r", s=60, edgecolors="black", linewidths=0.6)
    fig.colorbar(sc, ax=a0, shrink=0.8, label=f"mean PM2.5, {cfg.real.start} to {cfg.real.end} [µg/m³]")
    a0.set(title=f"{len(data.names)} reference monitors (roads grey, industry/landfill orange)", xlabel="km", ylabel="km")
    t = data.times.astype("datetime64[h]").astype(object)
    a1.fill_between(t, np.nanpercentile(data.pm25, 10, 0), np.nanpercentile(data.pm25, 90, 0), color=PALETTE[0],
                    alpha=0.25, label="10-90% across stations")
    a1.plot(t, np.nanmedian(data.pm25, 0), color=PALETTE[0], lw=1.2, label="station median")
    a1.set(ylabel="PM2.5 [µg/m³]", title=f"Hourly PM2.5, {cfg.real.start} to {cfg.real.end} (UTC)")
    a1.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2 = a1.twinx()
    ax2.plot(t, data.H, color=PALETTE[2], lw=0.7, alpha=0.8, label="boundary-layer height (ERA5)")
    ax2.set(ylabel="BLH [m]", yscale="log")
    a1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    print("saved", savefig(fig, cfg.path("figures"), "phase6_data"))


if __name__ == "__main__":
    main()
