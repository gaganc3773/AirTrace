"""Phase 5 data: category-resolved concentration fields and wind patterns for every scenario.

Adds two keys to each existing shard:
  c_cat     (4, ny, nx) float16  last-hour mean concentration due to each source category
                                 (traffic, industry, construction, burning); the regional part
                                 is exactly the uniform c_bg, so c ≈ sum(c_cat) + c_bg
  wind_pat  (2, ny, nx) float16  unit spatial wind perturbation at cell centres (u, v)

Uses the exactly linear scheme so the decomposition is additive.
Usage:  python scripts/phase5_contributions.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from airtrace.config import load_config
from airtrace.data.dataset import SyntheticDataset
from airtrace.data.synthetic import ForcingBuilder
from airtrace.grid import Grid
from airtrace.solver import ADRSolver, Forcing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--batch", type=int, default=400)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid = Grid.from_config(cfg.domain)
    ds = SyntheticDataset(cfg.path("artifacts") / "datasets" / cfg.data.name)
    solver = ADRSolver(grid, "linear3")
    entrain = ds.meta["config"]["physics"].get("entrainment", False)
    n_cat, sub = 4, 6
    t0, done = time.perf_counter(), 0
    for shard_idx, shard in enumerate(ds.shards):
        if (shard / "c_cat.npy").exists():
            done += int(ds._offsets[shard_idx + 1] - ds._offsets[shard_idx])
            continue
        lo, hi = int(ds._offsets[shard_idx]), int(ds._offsets[shard_idx + 1])
        c_cat, pat = [], []
        for start in range(lo, hi, args.batch):
            idx = np.arange(start, min(start + args.batch, hi))
            specs = ds.specs(idx)
            B, T = len(idx), specs.wind_u.shape[1]
            with torch.no_grad():
                fb = ForcingBuilder(specs, ds.layouts, grid, dev, entrainment=entrain)
                K = fb.K.repeat_interleave(n_cat)
                c = torch.zeros(B * n_cat, grid.ny, grid.nx, device=dev)
                zero = torch.zeros(B * n_cat, device=dev)
                for h in range(T):
                    met = fb.forcing(h)
                    S = fb.source_by_category(h).reshape(B * n_cat, grid.ny, grid.nx)
                    f = Forcing(met.u.repeat_interleave(n_cat, 0), met.v.repeat_interleave(n_cat, 0), S, zero)
                    _, snaps = solver.run(c, lambda t: f, K, fb.lam_eff(h).repeat_interleave(n_cat), sub, 3600.0 / sub)
                    if h == T - 1:
                        last = torch.trapezoid(snaps, dim=1) / sub
                    c = snaps[:, -1]
                c_cat.append(last.reshape(B, n_cat, grid.ny, grid.nx).cpu().numpy().astype(np.float16))
                up = 0.5 * (fb.up[..., :-1] + fb.up[..., 1:])
                vp = 0.5 * (fb.vp[..., :-1, :] + fb.vp[..., 1:, :])
                pat.append(torch.stack([up, vp], 1).cpu().numpy().astype(np.float16))
            done += B
            rate = done / (time.perf_counter() - t0) * 60
            print(f"[{done:6d}/{len(ds)}] {rate:,.0f} scenarios/min", flush=True)
        np.save(shard / "wind_pat.npy", np.concatenate(pat))
        np.save(shard / "c_cat.npy", np.concatenate(c_cat))  # written last: marks the shard complete
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
