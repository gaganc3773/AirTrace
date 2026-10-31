"""Phase 2 figures: 16-scenario sanity grid, prior maps, dataset statistics.

Usage:  python scripts/phase2_figures.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from airtrace.config import load_config
from airtrace.data.dataset import SyntheticDataset
from airtrace.data.layouts import CATEGORIES
from airtrace.data.synthetic import PROFILES, SCHEDULES
from airtrace.eval.plotting import FIELD_CMAP, PALETTE, savefig, set_style
from airtrace.grid import Grid

CAT_COLORS = {1: PALETTE[0], 2: PALETTE[4], 3: PALETTE[3]}
CAT_MARKERS = {1: "s", 2: "^", 3: "o"}


def sanity_grid(ds: SyntheticDataset, grid: Grid, out, seed: int) -> None:
    """Final-hour concentration for 16 random scenarios, with true sources, sensors and wind."""
    idx = np.sort(np.random.default_rng(seed).choice(len(ds), 16, replace=False))
    fig, axes = plt.subplots(4, 4, figsize=(15, 15.5), layout="constrained")
    ext = grid.extent_km()
    for ax, i in zip(axes.ravel(), idx):
        d = ds[int(i)]
        c = d["c_hourly"][-1].astype(np.float32)
        bg = float(d["c_bg"][-1])
        im = ax.imshow(c, origin="lower", extent=ext, cmap=FIELD_CMAP, norm=LogNorm(vmin=max(c.min(), 1), vmax=c.max()))
        ax.grid(False)
        if d["layout_id"] == 0:
            ax.contour(ds.layouts[0, 0], levels=[0.35], colors="white", linewidths=0.4, alpha=0.6, extent=ext)
        for row in d["sources"][d["sources"][:, 7] > 0]:
            cat, x, y, off = int(row[0]), row[1] / 1e3, row[2] / 1e3, bool(row[6])
            ax.plot(x, y, CAT_MARKERS[cat], color=CAT_COLORS[cat], ms=6, mec="black", mew=0.6)
            if off:
                ax.plot(x, y, "x", color="red", ms=11, mew=2.0)
        cells, r = d["sensor_cells"], d["sensor_readings"]
        on = np.isfinite(r[:, -1])
        sx, sy = (cells[:, 1] + 0.5) * grid.dx / 1e3, (cells[:, 0] + 0.5) * grid.dy / 1e3
        ax.scatter(sx[on], sy[on], s=10, c="white", edgecolors="black", linewidths=0.5)
        ax.scatter(sx[~on], sy[~on], s=10, facecolors="none", edgecolors="grey", linewidths=0.6)
        u, v = d["wind_u"][-1], d["wind_v"][-1]
        speed = float(np.hypot(u, v))
        ax.quiver(5, 43, u / speed, v / speed, color="#2a6f97", scale=9, width=0.014)
        place = "Delhi" if d["layout_id"] == 0 else f"city {d['layout_id']}"
        ax.set_title(f"#{i} {place} · {int(d['end_hour']):02d}:00 · {speed:.1f} m/s · bg {bg:.0f}", fontsize=9)
        ax.set_xticks([0, 24, 48])
        ax.set_yticks([0, 24, 48])
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    handles = [Line2D([], [], ls="", marker=CAT_MARKERS[k], color=CAT_COLORS[k], mec="black", label=CATEGORIES[k])
               for k in (1, 2, 3)]
    handles += [Line2D([], [], ls="", marker="x", color="red", mew=2, label="off-prior (unknown) source"),
                Line2D([], [], ls="", marker="o", color="white", mec="black", label="sensor (online)"),
                Line2D([], [], ls="", marker="o", mfc="none", mec="grey", label="sensor (offline)"),
                Line2D([], [], color="white", lw=1, label="major roads (Delhi)")]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.02), facecolor="#ccc",
               frameon=True)
    fig.suptitle("Synthetic scenarios: last-hour mean PM2.5 [µg/m³], true sources and sensor network", fontsize=13)
    print("saved", savefig(fig, out, "phase2_sanity_grid"))


def prior_maps(ds: SyntheticDataset, grid: Grid, out) -> None:
    """Delhi OSM priors (top row) and four random synthetic cities (bottom row)."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), layout="constrained")
    for ax, name, m in zip(axes[0], CATEGORIES, ds.layouts[0]):
        im = ax.imshow(m, origin="lower", extent=grid.extent_km(), cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"Delhi OSM prior: {name}")
        ax.grid(False)
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    for ax, k in zip(axes[1], [1, 2, 3, 4]):
        L = ds.layouts[k]
        rgb = np.stack([np.clip(L[1] + L[3], 0, 1), np.clip(L[0], 0, 1) * 0.8, np.clip(L[2] * 2, 0, 1)], -1)
        ax.imshow(rgb, origin="lower", extent=grid.extent_km())
        ax.set_title(f"synthetic city {k} (G roads, R industry+burning, B constr.)", fontsize=9)
        ax.grid(False)
    print("saved", savefig(fig, out, "phase2_prior_maps"))


def statistics(ds: SyntheticDataset, out) -> None:
    """Distributions of concentrations, local share, wind, inventory and diurnal profiles."""
    n = min(len(ds), 4000)
    rows = [ds.get(i, ("c_hourly", "c_bg", "wind_u", "wind_v", "sources", "traffic_emission", "layout_id"))
            for i in np.linspace(0, len(ds) - 1, n).astype(int)]
    c_all = np.concatenate([r["c_hourly"][-1].ravel()[::7] for r in rows]).astype(np.float32)
    mean_c = np.array([r["c_hourly"][-1].astype(np.float32).mean() for r in rows])
    bg = np.array([r["c_bg"][-1] for r in rows])
    speed = np.array([np.hypot(r["wind_u"][-1], r["wind_v"][-1]) for r in rows])
    inv = np.array([[r["traffic_emission"]] + [r["sources"][r["sources"][:, 0] == k, 4].sum() for k in (1, 2, 3)]
                    for r in rows]) / 1e9
    off = np.concatenate([r["sources"][r["sources"][:, 7] > 0, 6] for r in rows])

    fig, ax = plt.subplots(2, 3, figsize=(15, 8.2), layout="constrained")
    ax[0, 0].hist(c_all, bins=np.logspace(0.8, 3.6, 60))
    ax[0, 0].set(xscale="log", xlabel="c [µg/m³] (all cells, last hour)", ylabel="count",
                 title=f"Concentration: median {np.median(c_all):.0f}, p99 {np.percentile(c_all, 99):.0f}")
    local = 1 - bg / mean_c
    ax[0, 1].hist(local * 100, bins=40)
    ax[0, 1].set(xlabel="domain-mean local (non-background) share [%]", title="Local vs regional signal")
    ax[0, 2].hist(speed, bins=40)
    ax[0, 2].set(xlabel="wind speed, last hour [m/s]", title="Wind")
    order = np.argsort(inv.sum(1))
    ax[1, 0].stackplot(np.arange(n), (inv[order] / inv[order].sum(1, keepdims=True)).T * 100,
                       labels=CATEGORIES, colors=PALETTE[:4])
    ax[1, 0].set(xlabel="scenario (sorted by total emission)", ylabel="share of emission [%]",
                 title=f"Inventory: total {inv.sum(1).min():.2f}-{inv.sum(1).max():.2f} kg/s", ylim=(0, 100))
    ax[1, 0].legend(fontsize=8, loc="upper left")
    ax[1, 1].bar(["on-prior", "off-prior"], [(off == 0).sum(), (off == 1).sum()], color=PALETTE[:2])
    ax[1, 1].set(title=f"Discrete sources ({len(off) / n:.1f} per scenario)", ylabel="count")
    h = np.arange(24) + 0.5
    for k, name in enumerate(SCHEDULES):
        ax[1, 2].step(h, PROFILES[k], where="mid", label=name)
    ax[1, 2].step(h, PROFILES[4], where="mid", label="traffic", color="black", lw=1.8)
    ax[1, 2].set(xlabel="local hour", ylabel="relative emission (daily mean 1)", title="Emission schedules",
                 xticks=range(0, 25, 3))
    ax[1, 2].legend(fontsize=8)
    print(f"local share: median {np.median(local) * 100:.0f}%  |  Delhi layouts {np.mean([r['layout_id'] == 0 for r in rows]) * 100:.0f}%")
    print("saved", savefig(fig, out, "phase2_dataset_stats"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_style()
    grid = Grid.from_config(cfg.domain)
    ds = SyntheticDataset(cfg.path("artifacts") / "datasets" / cfg.data.name)
    print(f"dataset: {len(ds)} scenarios in {len(ds.shards)} shards")
    out = cfg.path("figures")
    sanity_grid(ds, grid, out, cfg.seed)
    prior_maps(ds, grid, out)
    statistics(ds, out)


if __name__ == "__main__":
    main()
