"""Tests for Green's functions, the linear inversion, interpolation baselines and metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from airtrace.config import DataConfig, PhysicsConfig
from airtrace.data.layouts import random_layout
from airtrace.data.synthetic import ForcingBuilder, sample_specs
from airtrace.eval.interpolation import idw, kriging
from airtrace.eval.metrics import detect, hotspots, localization_km
from airtrace.grid import Grid
from airtrace.models.greens import block_basis, block_sum, response_matrix
from airtrace.models.inversion import invert
from airtrace.solver import ADRSolver, Forcing

GRID = Grid(16, 16, 750.0, 750.0)
T, W, BLOCK = 4, 2, 4


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    layouts = np.stack([random_layout(GRID, rng)])
    cfg = DataConfig(sim_hours=T, window_hours=W, delhi_fraction=1.0)
    specs = sample_specs(rng, 1, layouts, GRID, cfg, PhysicsConfig())
    cells = np.array([(j, i) for j in range(0, 16, 2) for i in range(0, 16, 2)], dtype=np.int16)
    resp = response_matrix(specs, layouts, GRID, cells, W, BLOCK)
    return specs, layouts, cells, resp


def test_block_basis_partitions_grid() -> None:
    b = block_basis(GRID, BLOCK)
    assert b.shape == (16, 16, 16) and torch.all(b.sum(0) == 1)
    assert block_sum(np.ones((16, 16)), BLOCK).tolist() == [[16.0] * 4] * 4


def test_greens_superposition_matches_direct_simulation(setup) -> None:
    """G @ E reproduces a direct linear3 run with the same constant block emissions."""
    specs, layouts, cells, resp = setup
    E = np.random.default_rng(1).uniform(0, 50, 16)  # g/s per block
    fb = ForcingBuilder(specs, layouts, GRID)
    emis = torch.einsum("j,jyx->yx", torch.tensor(E, dtype=torch.float32), block_basis(GRID, BLOCK)) / BLOCK**2 * 1e6
    solver = ADRSolver(GRID, "linear3")
    c = torch.zeros(1, 16, 16)
    hourly = []
    for h in range(T):
        met = fb.forcing(h)
        S = (emis / (GRID.cell_area * float(specs.mixing_height[0, h])))[None]
        f = Forcing(met.u, met.v, S, torch.zeros(1))
        _, snaps = solver.run(c, lambda t: f, fb.K, fb.lam, 6, 600.0)
        if h >= T - W:
            hourly.append(torch.trapezoid(snaps, dim=1)[0] / 6)
        c = snaps[:, -1]
    direct = torch.stack(hourly)[:, cells[:, 0], cells[:, 1]].T.numpy()  # (n, W)
    assert np.allclose(resp.sensors @ E, direct, rtol=1e-4, atol=1e-4)
    assert np.allclose(np.einsum("jyx,j->yx", resp.field_last, E), hourly[-1].numpy(), rtol=1e-4, atol=1e-4)


def test_inversion_recovers_single_block_source(setup) -> None:
    _, _, _, resp = setup
    E_true = np.zeros(16)
    E_true[5] = 60.0
    y = resp.sensors @ E_true + np.array([40.0, 45.0])[None]
    r = invert(resp.sensors, y, sigma_rel=0.01, sigma_abs=0.05)
    assert r.E.argmax() == 5 and abs(r.E[5] - 60) < 6 and np.allclose(r.bg, [40, 45], atol=2)


def test_inversion_handles_missing_readings(setup) -> None:
    _, _, _, resp = setup
    E_true = np.zeros(16)
    E_true[10] = 30.0
    y = resp.sensors @ E_true + 20.0
    y[::3, 0] = np.nan
    r = invert(resp.sensors, y, sigma_rel=0.01, sigma_abs=0.05)
    assert r.n_obs == np.isfinite(y).sum() and r.E.argmax() == 10


def test_idw_exact_and_kriging_reasonable() -> None:
    rng = np.random.default_rng(2)
    pts = rng.uniform(0, 48e3, (30, 2))
    vals = 50 + pts[:, 0] / 1e3
    assert np.allclose(idw(pts, vals, pts), vals)
    tgt = rng.uniform(5e3, 43e3, (50, 2))
    pred = kriging(pts, vals, tgt, noise_rel=0.01)
    assert np.sqrt(np.mean((pred - (50 + tgt[:, 0] / 1e3)) ** 2)) < 3.0
    assert np.allclose(kriging(pts, np.full(30, 70.0), tgt), 70.0, atol=1e-6)


def test_hotspots_and_detection() -> None:
    E = np.zeros((8, 8))
    E[2, 2], E[2, 3], E[6, 6] = 50, 30, 20
    hs = hotspots(E, tau=15)
    assert sorted(map(tuple, hs)) == [(2, 2), (6, 6)]
    d = detect(hs, np.array([[2, 3], [0, 7]]), tol=1)
    assert (d.tp_pred, d.tp_true, d.n_pred, d.n_true) == (1, 1, 2, 2)
    km = localization_km(np.array([[2.5 * 3000, 2.5 * 3000]]), hs, 3000.0)
    assert km[0] == pytest.approx(0.0)
