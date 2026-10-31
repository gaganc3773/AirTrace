"""Batched finite-volume solver for 2D advection-diffusion-reaction.

    dc/dt + div(u c) = div(K grad c) - lam c + S

Spatial discretisation is flux form (exactly conservative): MUSCL face
reconstruction for advection and central differences for diffusion. Time
stepping is SSP-RK3 (Shu-Osher). Boundaries use two ghost cells per side:
on inflow faces the ghost value is the regional background ``c_bg``
(Dirichlet), on outflow faces it copies the edge cell (zero gradient).

Reconstruction options:
    ``koren``    TVD, positivity preserving, 3rd order where smooth (default);
                 nonlinear in c. Gaussian-puff error 4.6% at 750 m vs 8.1%
                 for van Leer, which clips narrow peaks.
    ``mc``, ``vanleer``  other TVD limiters, kept for comparison.
    ``linear3``  unlimited kappa=1/3 upwind-biased, 3rd order; exactly
                 linear in (S, c_bg, c0), so use it for Green's functions,
                 adjoints and attribution.
    ``upwind1``  1st-order donor cell; kept only as a reference.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from airtrace.config import SolverConfig
from airtrace.grid import Grid

Limiter = Literal["koren", "mc", "vanleer", "linear3", "upwind1"]
Coeff = torch.Tensor | float


@dataclass
class Forcing:
    """Forcing held constant over one forcing interval, for a batch of ``B`` scenarios.

    Attributes:
        u: x-velocity on x-faces, ``(B, ny, nx + 1)`` [m/s].
        v: y-velocity on y-faces, ``(B, ny + 1, nx)`` [m/s].
        S: source field, ``(B, ny, nx)`` [µg m^-3 s^-1].
        c_bg: inflow background concentration, ``(B,)`` [µg m^-3].
    """

    u: torch.Tensor
    v: torch.Tensor
    S: torch.Tensor
    c_bg: torch.Tensor


def _b11(x: Coeff, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar or per-scenario ``(B,)`` coefficient to ``(B, 1, 1)``."""
    t = torch.as_tensor(x, dtype=ref.dtype, device=ref.device)
    return t.reshape(-1, 1, 1)


def _koren(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Koren-limited slope ``phi(a/b) b``: the kappa=1/3 slope ``(a + 2b)/3``, capped at ``2a`` and ``2b``.

    ``a`` is the difference on the upwind side of the face, ``b`` on the downwind side.
    """
    s = torch.sign(b)
    a_, b_ = a * s, b.abs()
    return s * torch.minimum(torch.minimum(2.0 * a_, (a_ + 2.0 * b_) / 3.0), 2.0 * b_).clamp(min=0)


def _minmod3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Minmod of three arguments: smallest magnitude if all share a sign, else 0."""
    same = ((a > 0) & (b > 0) & (c > 0)) | ((a < 0) & (b < 0) & (c < 0))
    m = torch.minimum(torch.minimum(a.abs(), b.abs()), c.abs())
    return torch.where(same, torch.sign(a) * m, torch.zeros_like(a))


def _pad_with_ghosts(c: torch.Tensor, vel: torch.Tensor, c_bg: torch.Tensor, dim: int) -> torch.Tensor:
    """Append two ghost cells on each side of ``dim`` (inflow: c_bg, outflow: copy edge)."""
    n = c.shape[dim]
    first, last = c.narrow(dim, 0, 1), c.narrow(dim, n - 1, 1)
    v_lo, v_hi = vel.narrow(dim, 0, 1), vel.narrow(dim, vel.shape[dim] - 1, 1)
    bg = c_bg.reshape(-1, 1, 1)
    g_lo = torch.where(v_lo > 0, bg, first)
    g_hi = torch.where(v_hi < 0, bg, last)
    return torch.cat([g_lo, g_lo, c, g_hi, g_hi], dim=dim)


def _face_flux(
    c: torch.Tensor, vel: torch.Tensor, c_bg: torch.Tensor, K: torch.Tensor, h: float, dim: int, limiter: Limiter
) -> torch.Tensor:
    """Total (advective + diffusive) flux through the ``n + 1`` faces along ``dim``."""
    n = c.shape[dim]
    cp = _pad_with_ghosts(c, vel, c_bg, dim)  # padded index j = 0..n+3, interior 2..n+1
    d = cp.narrow(dim, 1, n + 3) - cp.narrow(dim, 0, n + 3)  # d_j = c_{j+1} - c_j
    dm, dp = d.narrow(dim, 0, n + 2), d.narrow(dim, 1, n + 2)  # backward/forward differences, cells 1..n+2
    cc = cp.narrow(dim, 1, n + 2)

    if limiter == "upwind1":
        east, west = cc, cc
    elif limiter == "koren":
        east, west = cc + 0.5 * _koren(dm, dp), cc - 0.5 * _koren(dp, dm)
    elif limiter == "mc":
        s = _minmod3(2.0 * dm, 0.5 * (dm + dp), 2.0 * dp)
        east, west = cc + 0.5 * s, cc - 0.5 * s
    elif limiter == "vanleer":
        prod = dm * dp
        pos = prod > 0
        s = torch.where(pos, 2.0 * prod / torch.where(pos, dm + dp, torch.ones_like(dm)), torch.zeros_like(dm))
        east, west = cc + 0.5 * s, cc - 0.5 * s
    elif limiter == "linear3":
        east = cc + dm / 6.0 + dp / 3.0
        west = cc - dp / 6.0 - dm / 3.0
    else:
        raise ValueError(f"unknown limiter {limiter!r}")

    # Face k (k = 0..n) sits between padded cells k+1 and k+2, i.e. cc[k] and cc[k+1].
    c_left = east.narrow(dim, 0, n + 1)
    c_right = west.narrow(dim, 1, n + 1)
    grad = (cp.narrow(dim, 2, n + 1) - cp.narrow(dim, 1, n + 1)) / h
    return vel.clamp(min=0) * c_left + vel.clamp(max=0) * c_right - K * grad


class ADRSolver:
    """Advection-diffusion-reaction solver on a fixed grid.

    ``K`` and ``lam`` are passed per call (scalar or per-scenario ``(B,)``)
    because they are sampled per scenario.
    """

    def __init__(self, grid: Grid, limiter: Limiter = "koren", cfl_max: float = 0.5, dt_max: float = 60.0):
        self.grid = grid
        self.limiter: Limiter = limiter
        self.cfl_max = cfl_max
        self.dt_max = dt_max

    @classmethod
    def from_config(cls, grid: Grid, cfg: SolverConfig, limiter: Limiter | None = None) -> ADRSolver:
        """Build from a solver config, optionally overriding the limiter."""
        return cls(grid, limiter or cfg.limiter, cfg.cfl_max, cfg.dt_max)

    def tendency(self, c: torch.Tensor, f: Forcing, K: Coeff, lam: Coeff) -> torch.Tensor:
        """Semi-discrete right-hand side ``dc/dt``, shape ``(B, ny, nx)``."""
        g = self.grid
        Kb = _b11(K, c)
        Fx = _face_flux(c, f.u, f.c_bg, Kb, g.dx, -1, self.limiter)
        Fy = _face_flux(c, f.v, f.c_bg, Kb, g.dy, -2, self.limiter)
        return -(Fx[..., 1:] - Fx[..., :-1]) / g.dx - (Fy[..., 1:, :] - Fy[..., :-1, :]) / g.dy - _b11(lam, c) * c + f.S

    def stable_dt(self, f: Forcing, K: Coeff, lam: Coeff) -> float:
        """Largest step satisfying the combined advective-CFL / diffusion / decay limit."""
        g = self.grid
        cu = torch.maximum(f.u[..., :-1].abs(), f.u[..., 1:].abs()) / g.dx
        cv = torch.maximum(f.v[..., :-1, :].abs(), f.v[..., 1:, :].abs()) / g.dy
        adv = float((cu + cv).max())
        k_max = float(torch.as_tensor(K).max())
        lam_max = float(torch.as_tensor(lam).max())
        rate = adv / self.cfl_max + 2.0 * k_max * (1.0 / g.dx**2 + 1.0 / g.dy**2) + lam_max
        return min(self.dt_max, 1.0 / rate) if rate > 0 else self.dt_max

    def step(self, c: torch.Tensor, f: Forcing, dt: float, K: Coeff, lam: Coeff) -> torch.Tensor:
        """One SSP-RK3 step of size ``dt``."""
        c1 = c + dt * self.tendency(c, f, K, lam)
        c2 = 0.75 * c + 0.25 * (c1 + dt * self.tendency(c1, f, K, lam))
        return c / 3.0 + (2.0 / 3.0) * (c2 + dt * self.tendency(c2, f, K, lam))

    def run(
        self,
        c0: torch.Tensor,
        forcing_fn: Callable[[float], Forcing],
        K: Coeff,
        lam: Coeff,
        n_intervals: int,
        interval: float,
        t0: float = 0.0,
        save_every: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate over ``n_intervals`` forcing intervals.

        ``forcing_fn(t)`` is evaluated once at the start of each interval and held
        constant over it; the interval is split into equal substeps that satisfy
        :meth:`stable_dt`.

        Returns:
            ``(times, snapshots)``: times ``(T,)`` [s] and concentrations
            ``(B, T, ny, nx)``, including the initial state.
        """
        c, t = c0, t0
        times, snaps = [t0], [c0]
        for k in range(n_intervals):
            f = forcing_fn(t)
            n_sub = max(1, math.ceil(interval / self.stable_dt(f, K, lam) - 1e-9))
            dt = interval / n_sub
            for _ in range(n_sub):
                c = self.step(c, f, dt, K, lam)
            t = t0 + (k + 1) * interval
            if (k + 1) % save_every == 0:
                times.append(t)
                snaps.append(c)
        return torch.tensor(times, dtype=c0.dtype), torch.stack(snaps, dim=1)
