"""Tests for the PINN's continuous meteorology and PDE residual."""

from __future__ import annotations

import numpy as np
import torch

from airtrace.config import DataConfig, PhysicsConfig, PINNConfig
from airtrace.data.layouts import random_layout
from airtrace.data.synthetic import ForcingBuilder, sample_specs
from airtrace.grid import Grid
from airtrace.models.pinn import T0, AirPINN, ContinuousMet, pde_residual
from airtrace.solver.analytic import gaussian_puff

GRID = Grid(64, 64, 750.0, 750.0)


def _scenario():
    rng = np.random.default_rng(0)
    layouts = np.stack([random_layout(GRID, rng)])
    cfg = DataConfig(sim_hours=4, window_hours=2, wind_pert_frac=(0.3, 0.3))
    return sample_specs(rng, 1, layouts, GRID, cfg, PhysicsConfig()), layouts


def test_continuous_wind_matches_solver_faces() -> None:
    specs, layouts = _scenario()
    met = ContinuousMet.from_specs(specs, layouts, GRID, first_hour=0, n_hours=4)
    f = ForcingBuilder(specs, layouts, GRID, dtype=torch.float64).forcing(2)
    j, i = torch.meshgrid(torch.arange(GRID.ny), torch.arange(GRID.nx + 1), indexing="ij")
    x, y = (i.float() / GRID.nx).reshape(-1), ((j.float() + 0.5) / GRID.ny).reshape(-1)
    u_hat, _ = met.wind(x, y, torch.full_like(x, 2.5))
    u = u_hat * GRID.width / T0
    err = (u.double() - f.u[0].reshape(-1)).abs().max()
    assert err < 0.02 * float(f.u.abs().max()) + 1e-3  # O(dx²) difference from the discrete gradient


def test_continuous_wind_divergence_free() -> None:
    specs, layouts = _scenario()
    met = ContinuousMet.from_specs(specs, layouts, GRID, 0, 4)
    x = torch.rand(500, requires_grad=True)
    y = torch.rand(500, requires_grad=True)
    u, v = met.wind(x, y, torch.full((500,), 1.2))
    div = torch.autograd.grad(u.sum(), x)[0] + torch.autograd.grad(v.sum(), y)[0]
    assert div.abs().max() < 1e-4 * (u.abs().max() + v.abs().max())


def test_pde_residual_vanishes_on_analytic_puff() -> None:
    """The analytic advected/diffused/decayed puff satisfies the non-dimensional operator."""
    L, K, lam, uu, vv = 48e3, 75.0, 2e-5, 2.0, -1.0
    Kh, lh, uh, vh = K * T0 / L**2, lam * T0, uu * T0 / L, vv * T0 / L

    def c_fn(x, y, t):  # t in hours, x, y in [0, 1]
        return gaussian_puff(x * L, y * L, t * T0, 1.0, 20e3, 25e3, 3000.0, uu, vv, K, lam)

    x, y, t = (torch.rand(400, dtype=torch.float64) for _ in range(3))
    x, y, t = 0.3 + 0.3 * x, 0.35 + 0.3 * y, 0.2 + 2 * t
    r = pde_residual(c_fn, x, y, t, uh, vh, Kh, lh, torch.zeros_like(x))
    scale = c_fn(x, y, t).abs().max()
    assert r.abs().max() < 1e-8 * max(float(scale), 1.0) + 1e-10


def test_pinn_shapes_and_positivity() -> None:
    m = AirPINN(PINNConfig(), n_hours=8, bg_init=80.0)
    x = torch.rand(64)
    assert (m.c(x, x, x * 8) >= 0).all() and (m.source(x, x) >= 0).all()
    assert torch.allclose(m.background(torch.tensor([0.5, 7.9])), torch.tensor([0.8, 0.8]))
