"""Physics-informed neural network for single-scenario field reconstruction and source inversion.

Unknowns: the local excess ``c'(x, y, t) >= 0`` over a spatially uniform
regional background ``b(t)`` (one value per hour), and a time-constant
emission density ``e(x, y) >= 0``. Because the synthetic background term
makes a uniform ``b(t)`` an exact solution, ``c'`` obeys

    c'_t + u . grad c' = K lap c' - lam c' + e / H(t),   c' = 0 on inflow boundaries,

with the scenario's (known) wind ``u`` (divergence-free) and mixing height
``H(t)``. Readings are hourly means of ``b + c'``.

Everything is non-dimensional: x̂ = x / L (L = domain width), t̂ in hours
from the start of the PINN window, c in units of ``C0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from airtrace.config import PINNConfig
from airtrace.data.synthetic import N_MODES, ForcingBuilder, ScenarioSpecs
from airtrace.grid import Grid

T0 = 3600.0  # time unit [s]
C0 = 100.0  # concentration unit [µg/m³]
H_REF = 500.0  # mixing height at which the source net output equals S


@dataclass
class ContinuousMet:
    """Wind, mixing height and coefficients of one scenario as smooth functions (non-dimensional).

    Hour ``k`` of the PINN window is simulated hour ``first_hour + k``.
    """

    L: float
    u0: torch.Tensor  # (n_hours,) mean wind [m/s]
    v0: torch.Tensor
    amp: torch.Tensor  # (n_hours,) perturbation amplitude [m/s]
    coef: torch.Tensor  # (2, N, N) streamfunction coefficients
    scale: float  # pattern-gradient normalisation [1/m]
    H: torch.Tensor  # (n_hours,) mixing height [m]
    K_hat: float
    lam_hat: float

    @classmethod
    def from_specs(cls, specs: ScenarioSpecs, layouts: np.ndarray, grid: Grid, first_hour: int, n_hours: int,
                   device: str | torch.device = "cpu") -> ContinuousMet:
        """Continuous version of the discrete forcing that generated scenario ``specs`` (B = 1)."""
        fb = ForcingBuilder(specs, layouts, grid, dtype=torch.float64)
        sl = slice(first_hour, first_hour + n_hours)
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
        L = grid.width
        return cls(L, t(specs.wind_u[0, sl]), t(specs.wind_v[0, sl]), t(specs.pert_amp[0, sl]),
                   t(specs.pert_coef[0]), float(fb.pert_scale[0]), t(specs.mixing_height[0, sl]),
                   float(specs.K[0]) * T0 / L**2, float(specs.lam[0]) * T0)

    def _hour(self, t_hat: torch.Tensor) -> torch.Tensor:
        return t_hat.detach().floor().long().clamp(0, len(self.u0) - 1)

    def wind(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Non-dimensional velocity ``(û, v̂)`` at points (x̂, ŷ in [0, 1], t̂ in hours)."""
        k = torch.arange(1, N_MODES + 1, device=x.device, dtype=x.dtype) * math.pi
        kx, ky = k[None, :] * x[:, None], k[None, :] * y[:, None]  # (P, N)
        sx, cx, sy, cy = torch.sin(kx), torch.cos(kx), torch.sin(ky), torch.cos(ky)
        a, b = self.coef[0], self.coef[1]  # indexed [k (x-mode), l (y-mode)], as in ForcingBuilder._pattern
        # P = sum a_kl sin(k x) sin(l y) + b_kl cos(k x) cos(l y); derivatives w.r.t. x̂, ŷ
        dP_dy = torch.einsum("kl,pk,pl->p", a, sx, k[None] * cy) - torch.einsum("kl,pk,pl->p", b, cx, k[None] * sy)
        dP_dx = torch.einsum("kl,pk,pl->p", a, k[None] * cx, sy) - torch.einsum("kl,pk,pl->p", b, k[None] * sx, cy)
        h = self._hour(t)
        amp = self.amp[h] / (self.scale * self.L)
        u = self.u0[h] + amp * dP_dy
        v = self.v0[h] - amp * dP_dx
        return u * T0 / self.L, v * T0 / self.L

    def mixing_height(self, t: torch.Tensor) -> torch.Tensor:
        """Mixing height [m] at times t̂."""
        return self.H[self._hour(t)]


class FourierMLP(nn.Module):
    """MLP on random Fourier features of its inputs (fixed Gaussian frequencies)."""

    def __init__(self, scales: torch.Tensor, n_fourier: int, hidden: int, depth: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(n_fourier, scales.shape[-1], generator=g) * scales)
        layers: list[nn.Module] = []
        width = 2 * n_fourier
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.Tanh()]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * z @ self.B.T
        return self.net(torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)).squeeze(-1)


def _scales(cfg: PINNConfig, n: int, with_time: bool) -> torch.Tensor:
    if not with_time:  # source network: one coarser scale
        return torch.full((n, 2), cfg.source_sigma)
    s = torch.cat([torch.full((n // 2,), cfg.fourier_sigma[0]), torch.full((n - n // 2,), cfg.fourier_sigma[1])])
    cols = [s, s] + ([torch.full((n,), cfg.fourier_sigma_t)] if with_time else [])
    return torch.stack(cols, dim=1)


class AirPINN(nn.Module):
    """``c'(x̂, ŷ, t̂)`` and ``e(x̂, ŷ)`` networks plus hourly background."""

    def __init__(self, cfg: PINNConfig, n_hours: int, bg_init: float):
        super().__init__()
        self.T = cfg.window_hours
        self.c_net = FourierMLP(_scales(cfg, cfg.n_fourier, True), cfg.n_fourier, cfg.hidden, cfg.depth, seed=1)
        self.s_net = FourierMLP(_scales(cfg, cfg.n_fourier // 2, False), cfg.n_fourier // 2, cfg.hidden // 2, 3, seed=2)
        self.bg = nn.Parameter(torch.full((n_hours,), bg_init / C0))

    def c(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Local excess c' ≥ 0 (units of C0)."""
        return nn.functional.softplus(self.c_net(torch.stack([x, y, t / self.T], dim=-1)), beta=5.0)

    def source(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Ŝ at H = H_REF (units C0 per hour), ≥ 0."""
        return nn.functional.softplus(self.s_net(torch.stack([x, y], dim=-1)) - 3.0)

    def background(self, t: torch.Tensor) -> torch.Tensor:
        return self.bg[t.detach().floor().long().clamp(0, len(self.bg) - 1)]


def pde_residual(
    c_fn, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
    K: float, lam: float, S: torch.Tensor,
) -> torch.Tensor:
    """``c_t + u c_x + v c_y - K (c_xx + c_yy) + lam c - S`` via autograd (non-dimensional)."""
    x, y, t = (a.detach().requires_grad_(True) for a in (x, y, t))
    c = c_fn(x, y, t)
    cx, cy, ct = torch.autograd.grad(c.sum(), (x, y, t), create_graph=True)
    cxx = torch.autograd.grad(cx.sum(), x, create_graph=True)[0]
    cyy = torch.autograd.grad(cy.sum(), y, create_graph=True)[0]
    return ct + u * cx + v * cy - K * (cxx + cyy) + lam * c - S


@dataclass
class Observations:
    """Sensor data in PINN coordinates."""

    x: torch.Tensor  # (n,) x̂ of each reading
    y: torch.Tensor
    hour: torch.Tensor  # (n,) PINN-window hour index of the hourly mean
    value: torch.Tensor  # (n,) reading / C0
    sigma: torch.Tensor  # (n,) error / C0


def sample_boundary(n: int, T: float, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random boundary points with outward normals ``(x, y, t, nx, ny)``."""
    side = torch.randint(0, 4, (n,), device=device)
    s = torch.rand(n, device=device)
    x = torch.where(side == 0, 0.0, torch.where(side == 1, 1.0, s))
    y = torch.where(side == 2, 0.0, torch.where(side == 3, 1.0, s))
    nx = torch.where(side == 0, -1.0, torch.where(side == 1, 1.0, 0.0))
    ny = torch.where(side == 2, -1.0, torch.where(side == 3, 1.0, 0.0))
    return x, y, torch.rand(n, device=device) * T, nx, ny


QUAD = torch.tensor([0.125, 0.375, 0.625, 0.875])  # midpoint rule for hourly means


def train_pinn(model: AirPINN, met: ContinuousMet, obs: Observations, cfg: PINNConfig, eval_fn=None,
               eval_every: int = 500, seed: int = 0) -> list[dict[str, float]]:
    """Adam with cosine decay; causal weighting of the PDE residual over time bins.

    Training stops early (Morozov's discrepancy principle) once the running mean of
    the data chi²/n over 200 iterations falls to ``cfg.stop_chi2``: fitting the
    readings more closely than their noise only fits the noise.
    """
    torch.manual_seed(seed)
    dev = obs.x.device
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.iters, eta_min=cfg.lr * 0.02)
    T, nb = cfg.window_hours, cfg.causal_bins
    quad = QUAD.to(dev)
    history: list[dict[str, float]] = []
    recent: list[float] = []
    for it in range(cfg.iters + 1):
        # Data: hourly mean = background + mean of c' at quadrature times.
        tq = (obs.hour[:, None] + quad[None]).reshape(-1)
        cq = model.c(obs.x.repeat_interleave(4), obs.y.repeat_interleave(4), tq).reshape(-1, 4).mean(1)
        pred = model.background(obs.hour.float()) + cq
        l_data = (((pred - obs.value) / obs.sigma) ** 2).mean()

        # PDE residual at collocation points (uniform + jittered around sensors), causally weighted.
        n_near = int(cfg.n_colloc * cfg.near_sensor_frac)
        pick = torch.randint(0, len(obs.x), (n_near,), device=dev)
        x = torch.cat([torch.rand(cfg.n_colloc - n_near, device=dev),
                       (obs.x[pick] + cfg.near_sensor_sigma * torch.randn(n_near, device=dev)).clamp(0, 1)])
        y = torch.cat([torch.rand(cfg.n_colloc - n_near, device=dev),
                       (obs.y[pick] + cfg.near_sensor_sigma * torch.randn(n_near, device=dev)).clamp(0, 1)])
        t = torch.rand(cfg.n_colloc, device=dev) * T
        u, v = met.wind(x, y, t)
        S = model.source(x, y) * H_REF / met.mixing_height(t)
        r2 = pde_residual(model.c, x, y, t, u, v, met.K_hat, met.lam_hat, S) ** 2
        b = (t / T * nb).long().clamp(max=nb - 1)
        bin_loss = torch.zeros(nb, device=dev).index_add(0, b, r2) / torch.bincount(b, minlength=nb).clamp(min=1)
        w = torch.exp(-cfg.causal_eps * torch.cumsum(bin_loss.detach(), 0) + cfg.causal_eps * bin_loss.detach())
        l_pde = (w * bin_loss).mean()

        # Inflow boundary: c' = 0 where u . n < 0.
        bx, by, bt, nx, ny = sample_boundary(cfg.n_boundary, T, dev)
        bu, bv = met.wind(bx, by, bt)
        inflow = ((bu * nx + bv * ny) < 0).float()
        l_bc = (inflow * model.c(bx, by, bt) ** 2).sum() / inflow.sum().clamp(min=1)

        l_sparse = S.mean()
        loss = cfg.w_data * l_data + cfg.w_pde * l_pde + cfg.w_bc * l_bc + cfg.w_sparsity * l_sparse
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        recent = (recent + [l_data.item()])[-200:]
        stop = len(recent) == 200 and float(np.mean(recent)) <= cfg.stop_chi2
        if it % 50 == 0 or stop:
            rec = {"iter": it, "loss": loss.item(), "data": l_data.item(), "pde": l_pde.item(),
                   "bc": l_bc.item(), "sparsity": l_sparse.item(), "causal_min_w": w.min().item()}
            if eval_fn is not None and (it % eval_every == 0 or stop):
                rec.update(eval_fn(model))
            history.append(rec)
        if stop:
            history[-1]["early_stop"] = True
            break
    return history


@torch.no_grad()
def predict_hour_mean(model: AirPINN, grid: Grid, hour: int, device) -> np.ndarray:
    """Hourly-mean total concentration on cell centres [µg/m³] for PINN-window hour ``hour``."""
    xs = (torch.arange(grid.nx, device=device) + 0.5) / grid.nx
    ys = (torch.arange(grid.ny, device=device) + 0.5) / grid.ny
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    x, y = X.reshape(-1), Y.reshape(-1)
    c = sum(model.c(x, y, torch.full_like(x, hour + q)) for q in QUAD.tolist()) / len(QUAD)
    bg = model.bg[min(hour, len(model.bg) - 1)]
    return ((c + bg) * C0).reshape(grid.ny, grid.nx).cpu().numpy()


@torch.no_grad()
def emission_map(model: AirPINN, grid: Grid, device) -> np.ndarray:
    """Emission per cell [g/s]: e = Ŝ * H_REF * C0 / T0 [µg m^-2 s^-1] times cell area."""
    xs = (torch.arange(grid.nx, device=device) + 0.5) / grid.nx
    ys = (torch.arange(grid.ny, device=device) + 0.5) / grid.ny
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    s = model.source(X.reshape(-1), Y.reshape(-1)).reshape(grid.ny, grid.nx)
    return (s * H_REF * C0 / T0 * grid.cell_area / 1e6).cpu().numpy()
