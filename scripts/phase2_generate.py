"""Phase 2: build the layout bank and generate the synthetic scenario dataset.

Usage:  python scripts/phase2_generate.py [--config configs/default.yaml] [--n 20000]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from airtrace.config import load_config
from airtrace.data.dataset import write_shard
from airtrace.data.geo import LocalProjection
from airtrace.data.layouts import CATEGORIES, build_layouts
from airtrace.data.sensors import sample_network
from airtrace.data.synthetic import SCHEDULES, SRC_FIELDS, ForcingBuilder, sample_specs, simulate
from airtrace.grid import Grid
from airtrace.solver import ADRSolver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n", type=int, default=None, help="override number of scenarios")
    args = parser.parse_args()
    cfg = load_config(args.config)
    dc = cfg.data
    n_total = args.n or dc.n_scenarios
    device = "cuda" if torch.cuda.is_available() else "cpu"

    grid = Grid.from_config(cfg.domain)
    proj = LocalProjection.from_config(cfg.domain)
    root = cfg.path("artifacts") / "datasets" / dc.name
    root.mkdir(parents=True, exist_ok=True)

    layouts = build_layouts(grid, proj, cfg.path("artifacts") / "cache" / "osm", dc.n_random_layouts, cfg.seed)
    np.save(root / "layouts.npy", layouts)
    np.save(cfg.path("artifacts") / "delhi_priors.npy", layouts[0])
    (root / "meta.json").write_text(json.dumps({
        "n_scenarios": n_total, "categories": CATEGORIES, "schedules": SCHEDULES, "source_fields": SRC_FIELDS,
        "S_units": "ug m^-3 h^-1", "c_units": "ug m^-3", "config": cfg.model_dump(mode="json"),
    }, indent=2), encoding="utf-8")

    solver = ADRSolver.from_config(grid, cfg.solver)
    batches_per_shard = dc.shard_size // dc.batch_size
    n_batches = -(-n_total // dc.batch_size)
    t0 = time.perf_counter()
    shard: dict[str, list[np.ndarray]] = {}
    for b in range(n_batches):
        rng = np.random.default_rng([cfg.seed, b])
        n = min(dc.batch_size, n_total - b * dc.batch_size)
        specs = sample_specs(rng, n, layouts, grid, dc, cfg.physics)
        with torch.no_grad():
            out = simulate(ForcingBuilder(specs, layouts, grid, device), solver, dc.window_hours)
        out = {k: v.cpu().numpy() for k, v in out.items()}
        nets = [sample_network(rng, out["c_hourly"][i], grid, dc.sensors) for i in range(n)]
        arrays = {**out, **specs.as_dict(),
                  "sensor_cells": np.stack([x[0] for x in nets]),
                  "sensor_readings": np.stack([x[1] for x in nets]),
                  "sensor_noise": np.array([x[2] for x in nets])}
        for k, v in arrays.items():
            shard.setdefault(k, []).append(v)
        if (b + 1) % batches_per_shard == 0 or b == n_batches - 1:
            write_shard(root, b // batches_per_shard, {k: np.concatenate(v) for k, v in shard.items()})
            shard = {}
        done = min((b + 1) * dc.batch_size, n_total)
        rate = done / (time.perf_counter() - t0) * 60
        q = np.percentile(out["c_hourly"], [0, 1, 50, 99, 100])
        print(f"[{done:6d}/{n_total}] {rate:,.0f} scenarios/min  c percentiles 0/1/50/99/100: "
              + " / ".join(f"{x:.0f}" for x in q), flush=True)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min -> {root}")


if __name__ == "__main__":
    main()
