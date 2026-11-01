"""Phase 4: PINN reconstruction of one held-out scenario, compared with kriging, IDW and NNLS.

Usage:  python scripts/phase4_pinn.py [--scenario 19627] [--iters 15000]
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
from airtrace.eval.plotting import FIELD_CMAP, PALETTE, savefig, set_style
from airtrace.eval.scenario import load_truth, score
from airtrace.grid import Grid
from airtrace.models.greens import block_sum, response_matrix
from airtrace.models.inversion import invert, prior_block_score, prior_weights
from airtrace.models.pinn import (C0, AirPINN, ContinuousMet, Observations, emission_map, predict_hour_mean,
                                  train_pinn)

BLOCK, TAU, SRC_MIN = 4, 15.0, 10.0


def baselines(ds, truth, grid, window, device, cache_dir) -> dict[str, tuple[np.ndarray | None, np.ndarray]]:
    """Kriging, IDW and NNLS-L1 + OSM prior: ``name -> (E_blocks or None, last-hour field)``."""
    online = np.isfinite(truth.readings[:, -1])
    pts = (truth.cells[online][:, ::-1] + 0.5) * grid.dx
    X, Y = np.meshgrid((np.arange(grid.nx) + 0.5) * grid.dx, (np.arange(grid.ny) + 0.5) * grid.dy)
    tgt = np.column_stack([X.ravel(), Y.ravel()])
    out = {"Kriging": (None, kriging(pts, truth.readings[online, -1], tgt, truth.noise).reshape(grid.ny, grid.nx)),
           "IDW": (None, idw(pts, truth.readings[online, -1], tgt).reshape(grid.ny, grid.nx))}
    cache = cache_dir / f"G_{truth.index:05d}.npz"
    if cache.exists():
        z = np.load(cache)
        G, F = z["sensors"], z["field_last"]
    else:
        r = response_matrix(truth.specs, ds.layouts, grid, truth.cells, window, BLOCK, device)
        G, F = r.sensors, r.field_last
    w = prior_weights(prior_block_score(ds.layouts[truth.specs.layout_id[0]], BLOCK))
    inv = invert(G, truth.readings, sigma_rel=truth.noise, sigma_abs=1.0, weights=w)
    nb = grid.ny // BLOCK
    out["NNLS-L1 + prior"] = (inv.E.reshape(nb, nb), np.einsum("jyx,j->yx", F, inv.E) + inv.bg[-1])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--scenario", type=int, default=19627)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override PINN config fields")
    parser.add_argument("--tag", default="", help="suffix for output files (sweeps)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.physics.entrainment:
        parser.error("The PINN assumes synth_v1 physics (no entrainment) and its cached results: use --config configs/synth_v1.yaml")
    overrides = {k: float(v) for k, v in (kv.split("=") for kv in args.set)}
    if args.iters:
        overrides["iters"] = args.iters
    pc = cfg.pinn.model_validate({**cfg.pinn.model_dump(), **overrides})
    set_style()
    torch.manual_seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid = Grid.from_config(cfg.domain)
    ds = SyntheticDataset(cfg.path("artifacts") / "datasets" / cfg.data.name)
    W = cfg.data.window_hours
    truth = load_truth(ds, args.scenario, grid, BLOCK, W, SRC_MIN)
    T_sim, n_h = truth.specs.wind_u.shape[1], int(pc.window_hours)
    spin = n_h - W  # PINN hours before the first observed hour

    met = ContinuousMet.from_specs(truth.specs, ds.layouts, grid, T_sim - n_h, n_h, dev)
    ok = np.isfinite(truth.readings)
    s_idx, k_idx = np.nonzero(ok)
    vals = truth.readings[ok]
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    obs = Observations(x=t((truth.cells[s_idx, 1] + 0.5) / grid.nx), y=t((truth.cells[s_idx, 0] + 0.5) / grid.ny),
                       hour=t(k_idx + spin).long(), value=t(vals / C0), sigma=t((truth.noise * vals + 1.0) / C0))

    base = baselines(ds, truth, grid, W, dev, cfg.path("artifacts") / "greens")
    base_scores = {k: score(truth, grid, BLOCK, TAU, E, c) for k, (E, c) in base.items()}
    for k, s in base_scores.items():
        print(f"{k:18s} field RMSE {s['field_rmse']:.2f} µg/m³")

    def eval_fn(model: AirPINN) -> dict[str, float]:
        c_hat = predict_hour_mean(model, grid, n_h - 1, dev)
        return {"val_rmse": score(truth, grid, BLOCK, TAU, c_hat=c_hat)["field_rmse"]}

    model = AirPINN(pc, n_h, float(np.nanmedian(truth.readings))).to(dev)
    t0 = time.perf_counter()
    hist = train_pinn(model, met, obs, pc, eval_fn, eval_every=500, seed=cfg.seed)
    train_s = time.perf_counter() - t0

    c_pinn = predict_hour_mean(model, grid, n_h - 1, dev)
    E_pinn = block_sum(emission_map(model, grid, dev), BLOCK)
    s_pinn = score(truth, grid, BLOCK, TAU, E_pinn, c_pinn)
    print(f"PINN               field RMSE {s_pinn['field_rmse']:.2f} µg/m³  (trained {train_s / 60:.1f} min)")

    art = cfg.path("artifacts") / "pinn"
    art.mkdir(exist_ok=True)
    torch.save(model.state_dict(), art / f"pinn_{args.scenario}.pt")
    report = cfg.path("artifacts").parent / "reports"
    summary = {"scenario": args.scenario, "train_minutes": train_s / 60, "pinn": s_pinn, "baselines": base_scores,
               "config": pc.model_dump(mode="json")}
    (report / f"phase4_pinn{args.tag}.json").write_text(json.dumps({**summary, "history": hist}, indent=2))

    out = cfg.path("figures")
    # Loss curves + validation RMSE.
    it = np.array([h["iter"] for h in hist])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained")
    for key, lab in (("loss", "total"), ("data", "data misfit (χ²/n)"), ("pde", "PDE residual (causal)"),
                     ("bc", "inflow BC"), ("sparsity", "source L1")):
        a1.semilogy(it, [h[key] for h in hist], lw=1.2, label=lab)
    a1.set(xlabel="iteration", ylabel="loss", title=f"PINN training, scenario #{args.scenario}")
    a1.legend(fontsize=8)
    ev = [h for h in hist if "val_rmse" in h]
    a2.plot([h["iter"] for h in ev], [h["val_rmse"] for h in ev], marker="o", color=PALETTE[0], label="PINN")
    for (k, s), col in zip(base_scores.items(), PALETTE[1:]):
        a2.axhline(s["field_rmse"], color=col, ls="--", label=k)
    a2.set(xlabel="iteration", ylabel="RMSE at unseen cells [µg/m³]", title="Field reconstruction, last hour",
           ylim=(0, max(s["field_rmse"] for s in base_scores.values()) * 2))
    a2.legend(fontsize=8)
    print("saved", savefig(fig, out, f"phase4_pinn_loss{args.tag}"))

    # Fields and sources.
    ext = grid.extent_km()
    norm = LogNorm(vmin=max(truth.c_last.min(), 1), vmax=truth.c_last.max())
    on = np.isfinite(truth.readings[:, -1])
    sx, sy = (truth.cells[on, 1] + 0.5) * grid.dx / 1e3, (truth.cells[on, 0] + 0.5) * grid.dy / 1e3
    fields = [("truth", truth.c_last), ("PINN", c_pinn), ("NNLS-L1 + prior", base["NNLS-L1 + prior"][1]),
              ("kriging", base["Kriging"][1])]
    sources = [("true emission", truth.E_blocks), ("PINN", E_pinn), ("NNLS-L1 + prior", base["NNLS-L1 + prior"][0])]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), layout="constrained")
    for ax, (name, f) in zip(axes[0], fields):
        rm = "" if name == "truth" else f"  RMSE {score(truth, grid, BLOCK, TAU, c_hat=f)['field_rmse']:.1f}"
        im = ax.imshow(np.clip(f, norm.vmin, None), origin="lower", extent=ext, cmap=FIELD_CMAP, norm=norm)
        ax.scatter(sx, sy, s=8, c="white", edgecolors="black", linewidths=0.4)
        ax.set_title(f"{name}{rm}", fontsize=10)
        ax.grid(False)
    fig.colorbar(im, ax=axes[0], label="PM2.5, last hour [µg/m³]", shrink=0.85)
    vmax = truth.E_blocks.max()
    for ax, (name, E) in zip(axes[1], sources):
        im = ax.imshow(E, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax)
        ax.plot(truth.source_xy[:, 0] / 1e3, truth.source_xy[:, 1] / 1e3, "r+", ms=10, mew=1.5)
        ax.set_title(f"{name} [g/s per 3 km block]  Σ={E.sum():.0f}", fontsize=10)
        ax.grid(False)
    fig.colorbar(im, ax=axes[1, :3], shrink=0.85)
    err = c_pinn - truth.c_last
    lim = np.abs(err).max()
    im = axes[1, 3].imshow(err, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[1, 3].set_title("PINN − truth [µg/m³]", fontsize=10)
    axes[1, 3].grid(False)
    fig.colorbar(im, ax=axes[1, 3], shrink=0.85)
    print("saved", savefig(fig, out, f"phase4_pinn_reconstruction{args.tag}"))


if __name__ == "__main__":
    main()
