"""Verification tests for the finite-volume forward model."""

from __future__ import annotations

import math

import pytest
import torch

from airtrace.grid import Grid
from airtrace.solver import ADRSolver, Forcing, divergence, uniform_wind, wind_from_streamfunction
from airtrace.solver.analytic import gaussian_puff

DT = torch.float64
GRID = Grid(nx=64, ny=64, dx=750.0, dy=750.0)


def _const_forcing(u: torch.Tensor, v: torch.Tensor, S: torch.Tensor | None = None, c_bg: float = 0.0):
    B = u.shape[0]
    S = torch.zeros(B, GRID.ny, GRID.nx, dtype=DT) if S is None else S
    f = Forcing(u, v, S, torch.full((B,), c_bg, dtype=DT))
    return lambda t: f


def _puff_error(limiter: str, grid: Grid = GRID) -> float:
    """Relative L2 error of a 3 h advected/diffused/decayed puff."""
    K, lam, u, v = 75.0, 2e-5, 2.4, 1.8
    p = dict(amplitude=100.0, x0=12e3, y0=12e3, sigma0=3000.0, u=u, v=v, K=K, lam=lam)
    X, Y = grid.centers(DT)
    c0 = gaussian_puff(X, Y, 0.0, **p)[None]
    uf, vf = uniform_wind(grid, math.hypot(u, v), math.degrees(math.atan2(-u, -v)) % 360, DT)
    S = torch.zeros(1, grid.ny, grid.nx, dtype=DT)
    f = Forcing(uf, vf, S, torch.zeros(1, dtype=DT))
    solver = ADRSolver(grid, limiter)
    _, snaps = solver.run(c0, lambda t: f, K, lam, n_intervals=18, interval=600.0)
    exact = gaussian_puff(X, Y, 3 * 3600.0, **p)
    return float(torch.linalg.norm(snaps[0, -1] - exact) / torch.linalg.norm(exact))


@pytest.mark.parametrize("limiter", ["koren", "linear3"])
def test_gaussian_puff_within_5_percent(limiter: str) -> None:
    assert _puff_error(limiter) < 0.05


def test_first_order_upwind_is_too_diffusive() -> None:
    """Documents why the default is MUSCL: donor-cell numerical diffusion ~ u dx / 2 >> K."""
    assert _puff_error("upwind1") > 5 * _puff_error("koren")


def test_uniform_wind_direction_convention() -> None:
    u, v = uniform_wind(GRID, 2.0, 270.0, DT)  # westerly: blows from the west towards +x
    assert torch.allclose(u, torch.full_like(u, 2.0)) and torch.allclose(v, torch.zeros_like(v), atol=1e-12)


def test_streamfunction_wind_is_divergence_free() -> None:
    g = torch.Generator().manual_seed(0)
    psi = torch.randn(3, GRID.ny + 1, GRID.nx + 1, generator=g, dtype=DT) * 1e4
    u, v = wind_from_streamfunction(GRID, psi)
    assert divergence(GRID, u, v).abs().max() < 1e-12


@pytest.mark.parametrize("limiter", ["koren", "linear3"])
def test_mass_conservation_interior(limiter: str) -> None:
    """With no sources/decay and the plume away from boundaries, total mass is constant."""
    X, Y = GRID.centers(DT)
    c0 = gaussian_puff(X, Y, 0.0, 50.0, 24e3, 24e3, 2000.0, 0, 0, 0, 0)[None]
    yy, xx = torch.meshgrid(torch.arange(GRID.ny + 1, dtype=DT), torch.arange(GRID.nx + 1, dtype=DT), indexing="ij")
    psi = (3e3 * torch.sin(xx / 9.0) * torch.cos(yy / 7.0) + 750.0 * 1.5 * yy)[None]  # swirl + 1.5 m/s mean
    u, v = wind_from_streamfunction(GRID, psi)
    _, snaps = ADRSolver(GRID, limiter).run(c0, _const_forcing(u, v), 75.0, 0.0, n_intervals=6, interval=600.0)
    mass = snaps[0].sum(dim=(-1, -2))
    assert torch.allclose(mass, mass[0].expand_as(mass), rtol=1e-12)


def test_mass_balance_with_source_and_decay() -> None:
    """dM/dt = Q - lam M  =>  M(t) = Q/lam (1 - exp(-lam t)) for a compact source in calm air."""
    lam = 5e-5
    S = torch.zeros(1, GRID.ny, GRID.nx, dtype=DT)
    S[0, 30:34, 30:34] = 0.2
    Q = float(S.sum())
    u, v = uniform_wind(GRID, 0.0, 0.0, DT)
    times, snaps = ADRSolver(GRID).run(torch.zeros_like(S), _const_forcing(u, v, S), 75.0, lam, 12, 600.0)
    exact = Q / lam * (1 - torch.exp(-lam * times))
    assert torch.allclose(snaps[0].sum(dim=(-1, -2)), exact, rtol=1e-8)


@pytest.mark.parametrize("limiter", ["koren", "mc", "vanleer"])
def test_positivity_tvd(limiter: str) -> None:
    """Sharp sources under strong, swirling wind never create negative concentrations."""
    g = torch.Generator().manual_seed(1)
    S = (torch.rand(2, GRID.ny, GRID.nx, generator=g, dtype=DT) > 0.97).to(DT) * 0.5
    yy, xx = torch.meshgrid(torch.arange(GRID.ny + 1, dtype=DT), torch.arange(GRID.nx + 1, dtype=DT), indexing="ij")
    psi = torch.stack([4e3 * torch.sin(xx / 5.0) * torch.sin(yy / 6.0) + 750.0 * 5.0 * yy] * 2)
    u, v = wind_from_streamfunction(GRID, psi)
    _, snaps = ADRSolver(GRID, limiter).run(torch.zeros_like(S), _const_forcing(u, v, S), 30.0, 1e-5, 12, 600.0)
    assert snaps.min() >= -1e-12


def test_linear3_superposition_exact() -> None:
    """With the linear reconstruction, c is exactly linear in (S, c_bg)."""
    g = torch.Generator().manual_seed(2)
    S1, S2 = torch.rand(2, 1, GRID.ny, GRID.nx, generator=g, dtype=DT) * 0.1
    u, v = uniform_wind(GRID, 3.0, 300.0, DT)
    solver = ADRSolver(GRID, "linear3")

    def solve(S: torch.Tensor, bg: float) -> torch.Tensor:
        return solver.run(torch.zeros_like(S), _const_forcing(u, v, S, bg), 75.0, 2e-5, 12, 600.0)[1]

    combined = solve(2.0 * S1 + 3.0 * S2, 2.0 * 40.0 + 3.0 * 10.0)
    assert torch.allclose(combined, 2.0 * solve(S1, 40.0) + 3.0 * solve(S2, 10.0), rtol=1e-10, atol=1e-10)


def test_tvd_positively_homogeneous() -> None:
    """The TVD scheme is not additive, but scaling all inputs by a > 0 scales c by a."""
    g = torch.Generator().manual_seed(3)
    S = torch.rand(1, GRID.ny, GRID.nx, generator=g, dtype=DT) * 0.1
    u, v = uniform_wind(GRID, 3.0, 300.0, DT)
    solver = ADRSolver(GRID, "koren")
    a = solver.run(torch.zeros_like(S), _const_forcing(u, v, S, 30.0), 75.0, 2e-5, 6, 600.0)[1]
    b = solver.run(torch.zeros_like(S), _const_forcing(u, v, 2.5 * S, 75.0), 75.0, 2e-5, 6, 600.0)[1]
    assert torch.allclose(b, 2.5 * a, rtol=1e-10)


def test_inflow_boundary_fills_domain_with_background() -> None:
    u, v = uniform_wind(GRID, 4.0, 225.0, DT)  # south-westerly
    c0 = torch.zeros(1, GRID.ny, GRID.nx, dtype=DT)
    _, snaps = ADRSolver(GRID).run(c0, _const_forcing(u, v, c_bg=100.0), 75.0, 0.0, 60, 600.0)
    assert torch.allclose(snaps[0, -1], torch.full_like(snaps[0, -1], 100.0), rtol=1e-3)


def test_stable_dt_respects_cfl() -> None:
    solver = ADRSolver(GRID, cfl_max=0.5, dt_max=1e9)
    u, v = uniform_wind(GRID, 6.0, 270.0, DT)
    f = Forcing(u, v, torch.zeros(1, GRID.ny, GRID.nx, dtype=DT), torch.zeros(1, dtype=DT))
    dt = solver.stable_dt(f, 75.0, 2e-5)
    assert 6.0 * dt / GRID.dx <= 0.5 + 1e-12
    assert 2 * 75.0 * dt * (2 / GRID.dx**2) <= 1.0
