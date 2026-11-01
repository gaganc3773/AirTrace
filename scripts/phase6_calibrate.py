"""Calibrate the synthetic generator's magnitudes against real station statistics (a month that is NOT the test month).

Simulates Delhi-layout scenarios with sensors at the real monitor cells and compares
summary statistics with the calibration month (default December 2025; November 2025 is held
out for the leave-one-station-out test):

  level      p10 / p50 / p90 of the hourly network median
  spread     median over hours of (p90 - p10) / median across stations
  night/day  ratio of network medians, 20-08 vs 08-20 IST
  change     median |hour-to-hour change| of the network median, relative

Usage:
  python scripts/phase6_calibrate.py --set physics.background_range=40,250 data.total_emission_kg_s=1,6
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from airtrace.config import Config, load_config
from airtrace.data.geo import LocalProjection
from airtrace.data.openmeteo import met_bank
from airtrace.data.real import RealData
from airtrace.data.sensors import log_error
from airtrace.data.synthetic import ForcingBuilder, sample_specs, simulate
from airtrace.grid import Grid
from airtrace.solver import ADRSolver


def stats(pm: np.ndarray, local_end_hour: np.ndarray) -> dict[str, float]:
    """``pm``: (S, N) station values for N hours; ``local_end_hour``: (N,) IST hour at period end.

    For synthetic data the hours come in consecutive groups of 6 (one window per scenario).
    """
    med = np.nanmedian(pm, 0)
    q = np.nanpercentile(pm, [10, 90], axis=0)
    night = (local_end_hour >= 20) | (local_end_hour < 8)
    return {"p10": float(np.percentile(med, 10)), "p50": float(np.median(med)), "p90": float(np.percentile(med, 90)),
            "spread": float(np.median((q[1] - q[0]) / med)), "night/day": float(np.median(med[night]) / np.median(med[~night]))}


def change(med: np.ndarray, group: int | None) -> float:
    d = np.abs(np.diff(med)) / med[:-1]
    if group:  # drop jumps between different scenarios
        d = d[(np.arange(len(d)) + 1) % group != 0]
    return float(np.median(d))


def apply_overrides(cfg: Config, items: list[str]) -> Config:
    raw = cfg.model_dump()
    for kv in items:
        key, val = kv.split("=", 1)
        *path, name = key.split(".")
        node = raw
        for k in path:
            node = node[k]
        parts = [float(x) for x in val.split(",")]
        node[name] = tuple(parts) if isinstance(node[name], (tuple, list)) else parts[0]
    return Config.model_validate(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--month", default="2025-12-01_2025-12-31")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=V[,V]")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = apply_overrides(load_config(args.config), args.set)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid, proj = Grid.from_config(cfg.domain), LocalProjection.from_config(cfg.domain)
    real = RealData.load(cfg.path("artifacts") / "real" / f"delhi_{args.month}.npz")
    ok = np.isfinite(real.pm25).sum(0) >= 20  # hours with most of the network reporting
    r = stats(real.pm25[:, ok], real.local_hour[ok])
    med = np.nanmedian(real.pm25[:, ok], 0)
    r["change"] = change(med, None)

    layouts = np.load(cfg.path("artifacts") / "datasets" / "synth_v1" / "layouts.npy")
    dc = cfg.data.model_copy(update={"delhi_fraction": 1.0})
    met = met_bank(proj, dc.met_seasons, cfg.path("artifacts") / "cache" / "real", cfg.real.met_points)
    solver = ADRSolver.from_config(grid, cfg.solver)
    rng = np.random.default_rng(123)
    vals, hours = [], []
    for start in range(0, args.n, 250):
        specs = sample_specs(rng, min(250, args.n - start), layouts, grid, dc, cfg.physics, met)
        with torch.no_grad():
            out = simulate(ForcingBuilder(specs, layouts, grid, dev, entrainment=cfg.physics.entrainment), solver, dc.window_hours)
        c = out["c_hourly"].cpu().numpy()[:, :, real.cells[:, 0], real.cells[:, 1]]  # (B, W, S)
        sc = cfg.data.sensors  # station error model of the generator's stored networks
        for b in range(len(c)):
            e = log_error(rng, (c.shape[2], c.shape[1]), rng.uniform(*sc.noise_range),
                          rng.uniform(*sc.bias_range), sc.noise_rho)
            c[b] *= np.exp(e.T)
        vals.append(c.reshape(-1, c.shape[-1]).T)
        lh = specs.end_hour[:, None] - dc.window_hours + 1 + np.arange(dc.window_hours)[None]
        hours.append((lh % 24).ravel())
    pm = np.concatenate(vals, 1)
    s = stats(pm, np.concatenate(hours))
    s["change"] = change(np.median(pm, 0), dc.window_hours)
    print(f"{'':10s}" + "".join(f"{k:>11s}" for k in r))
    print(f"{'real':10s}" + "".join(f"{v:11.2f}" for v in r.values()))
    print(f"{'synthetic':10s}" + "".join(f"{v:11.2f}" for v in s.values()))


if __name__ == "__main__":
    main()
