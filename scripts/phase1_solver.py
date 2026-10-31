"""Phase 1 deliverables: solver verification figure, rotating-wind plume GIF, throughput benchmark.

Usage:  python scripts/phase1_solver.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm

from airtrace.config import load_config
from airtrace.eval.plotting import FIELD_CMAP, savefig, set_style
from airtrace.grid import Grid
from airtrace.solver import ADRSolver, Forcing, uniform_wind
from airtrace.solver.analytic import gaussian_puff

LIMITERS = ["upwind1", "vanleer", "koren", "linear3"]
PUFF = dict(amplitude=100.0, x0=12e3, y0=12e3, sigma0=3000.0, u=2.4, v=1.8, K=75.0, lam=2e-5)
PUFF_T = 3 * 3600.0


def puff_run(grid: Grid, limiter: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Numerical and analytic puff fields at ``PUFF_T`` on ``grid``."""
    X, Y = grid.centers()
    c0 = gaussian_puff(X, Y, 0.0, **PUFF)[None]
    u, v = PUFF["u"], PUFF["v"]
    uf, vf = uniform_wind(grid, math.hypot(u, v), math.degrees(math.atan2(-u, -v)) % 360)
    f = Forcing(uf, vf, torch.zeros_like(c0), torch.zeros(1, dtype=torch.float64))
    _, snaps = ADRSolver(grid, limiter).run(c0, lambda t: f, PUFF["K"], PUFF["lam"], int(PUFF_T / 600), 600.0)
    return snaps[0, -1], gaussian_puff(X, Y, PUFF_T, **PUFF)


def verification_figure(grid: Grid, out_dir) -> None:
    """Cross-section along the plume axis plus grid-convergence of the relative L2 error."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    X, Y = grid.centers()
    # Plume axis passes through (x0, y0) along (u, v); sample cells within half a cell of it.
    d = ((X - PUFF["x0"]) * PUFF["v"] - (Y - PUFF["y0"]) * PUFF["u"]).abs() / math.hypot(PUFF["u"], PUFF["v"])
    on_axis = d < grid.dx * 0.75
    s = ((X - PUFF["x0"]) * PUFF["u"] + (Y - PUFF["y0"]) * PUFF["v"]) / math.hypot(PUFF["u"], PUFF["v"])
    order = torch.argsort(s[on_axis])
    exact = None
    for lim in LIMITERS:
        num, exact = puff_run(grid, lim)
        ax1.plot(s[on_axis][order] / 1e3, num[on_axis][order], marker=".", ms=3, lw=1.2, label=lim)
    ax1.plot(s[on_axis][order] / 1e3, exact[on_axis][order], "k--", lw=1.5, label="analytic")
    ax1.set(xlabel="distance along wind from release point [km]", ylabel="c [µg/m³]",
            title=f"Gaussian puff after {PUFF_T / 3600:.0f} h, Δx = {grid.dx:.0f} m")
    ax1.legend(fontsize=8)

    ns = [32, 64, 128]
    print("\nRelative L2 error of the Gaussian puff:")
    print(f"{'limiter':>9} " + " ".join(f"{48 / n * 1000:>8.0f}m" for n in ns))
    for lim in LIMITERS:
        errs = []
        for n in ns:
            g = Grid(n, n, grid.width / n, grid.height / n)
            num, ex = puff_run(g, lim)
            errs.append(float(torch.linalg.norm(num - ex) / torch.linalg.norm(ex)))
        print(f"{lim:>9} " + " ".join(f"{e * 100:>8.2f}%" for e in errs))
        ax2.loglog([grid.width / n for n in ns], errs, marker="o", label=lim)
    h = np.array([grid.width / n for n in ns])
    ax2.loglog(h, 0.02 * (h / h[1]) ** 2, "k:", lw=1, label="slope 2")
    ax2.axhline(0.05, color="grey", lw=0.8, ls="--")
    ax2.text(h[-1], 0.054, "5% acceptance", fontsize=8, color="grey", ha="left")
    ax2.set(xlabel="Δx [m]", ylabel="relative L2 error", title="Grid convergence")
    ax2.legend(fontsize=8)
    print("saved", savefig(fig, out_dir, "phase1_puff_verification"))


def rotating_plume(grid: Grid, solver: ADRSolver, K: float, lam: float, out_dir) -> None:
    """24 h plume from a factory stack while the wind direction turns through 360°."""
    hours, frame_min, interval = 24, 15, 600.0
    H, E = 500.0, 20.0  # mixing height [m], emission [g/s]
    S = torch.zeros(1, grid.ny, grid.nx, dtype=torch.float64)
    S[0, grid.ny // 2, grid.nx // 2] = E * 1e6 / (grid.cell_area * H)

    def wind_dir(t: float) -> float:
        return (270.0 + 360.0 * t / (hours * 3600.0)) % 360.0

    def forcing(t: float) -> Forcing:
        u, v = uniform_wind(grid, 2.5, wind_dir(t))
        return Forcing(u, v, S, torch.zeros(1, dtype=torch.float64))

    every = int(frame_min * 60 / interval) or 1
    times, snaps = solver.run(torch.zeros_like(S), forcing, K, lam, int(hours * 3600 / interval), interval,
                              save_every=every)
    frames = snaps[0].numpy()
    mass = frames.sum(axis=(1, 2)) * grid.cell_area * H * 1e-9  # kg in the mixed layer
    emitted = E * times.numpy() * 1e-3

    vmax = np.percentile(frames[frames > 0], 99.9)
    norm = LogNorm(vmin=vmax * 1e-3, vmax=vmax)
    fig, (ax, axm) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    im = ax.imshow(np.clip(frames[0], vmax * 1e-3, None), origin="lower", extent=grid.extent_km(),
                   norm=norm, cmap=FIELD_CMAP)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="c [µg/m³]", shrink=0.85)
    ax.set(xlabel="x [km]", ylabel="y [km]")
    title = ax.set_title("")
    arrow = ax.quiver([6], [42], [0], [0], color="#2a6f97", scale=12, width=0.012)
    axm.plot(times / 3600, emitted, "k--", lw=1, label="emitted")
    axm.plot(times / 3600, mass, lw=1.6, label="in domain")
    (cursor,) = axm.plot([], [], "o", color="#e07a5f")
    axm.set(xlabel="time [h]", ylabel="PM2.5 mass [kg]", title="Mass budget (removal + outflow)")
    axm.legend(fontsize=8)

    def update(k: int):
        im.set_data(np.clip(frames[k], vmax * 1e-3, None))
        th = math.radians(wind_dir(float(times[k])))
        arrow.set_UVC([-math.sin(th)], [-math.cos(th)])
        title.set_text(f"t = {times[k] / 3600:5.2f} h   wind from {wind_dir(float(times[k])):5.1f}° @ 2.5 m/s")
        cursor.set_data([times[k] / 3600], [mass[k]])
        return im, arrow, title, cursor

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    path = out_dir / "phase1_rotating_plume.gif"
    anim.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)
    print("saved", path)

    src_km = ((grid.nx // 2 + 0.5) * grid.dx / 1e3, (grid.ny // 2 + 0.5) * grid.dy / 1e3)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True, layout="constrained")
    for ax, hr in zip(axes, [3, 9, 15, 21]):
        k = int(hr * 60 / frame_min)
        ax.plot(*src_km, marker="*", color="#2a6f97", ms=11, mec="white", zorder=3)
        im = ax.imshow(np.clip(frames[k], vmax * 1e-3, None), origin="lower", extent=grid.extent_km(), norm=norm,
                  cmap=FIELD_CMAP)
        ax.grid(False)
        ax.set(title=f"t = {hr} h, wind from {wind_dir(hr * 3600.0):.0f}°", xlabel="x [km]")
    axes[0].set_ylabel("y [km]")
    fig.colorbar(im, ax=axes, label="c [µg/m³]", shrink=0.9)
    print("saved", savefig(fig, out_dir, "phase1_rotating_plume_panels"))


def benchmark(grid: Grid, solver: ADRSolver) -> None:
    """Wall time for 30 simulated hours: one scenario on CPU, a batch on GPU (if available)."""
    def time_it(B: int, device: str, dtype: torch.dtype) -> float:
        g = torch.Generator(device="cpu").manual_seed(0)
        S = (torch.rand(B, grid.ny, grid.nx, generator=g) * 0.05).to(device=device, dtype=dtype)
        u, v = uniform_wind(grid, torch.full((B,), 3.0), torch.rand(B, generator=g) * 360, dtype, device)
        f = Forcing(u, v, S, torch.full((B,), 50.0, dtype=dtype, device=device))
        solver.run(torch.zeros_like(S), lambda t: f, 75.0, 2e-5, 6, 600.0)  # warm-up
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        solver.run(torch.zeros_like(S), lambda t: f, 75.0, 2e-5, 180, 600.0)
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    print("\nThroughput, 30 simulated hours at dt = 60 s (1800 SSP-RK3 steps):")
    t = time_it(1, "cpu", torch.float32)
    print(f"  CPU, 1 scenario, float32:    {t:6.2f} s")
    if torch.cuda.is_available():
        for B in (256, 1024):
            t = time_it(B, "cuda", torch.float32)
            print(f"  GPU, {B:4d} scenarios, float32: {t:6.2f} s  -> {B / t * 60:,.0f} scenarios/min")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg.seed)
    set_style()
    grid = Grid.from_config(cfg.domain)
    solver = ADRSolver.from_config(grid, cfg.solver)
    out = cfg.path("figures")
    verification_figure(grid, out)
    rotating_plume(grid, solver, cfg.physics.K, cfg.physics.lam, out)
    benchmark(grid, solver)


if __name__ == "__main__":
    main()
