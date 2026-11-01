"""FNO inputs/targets: a GPU-resident copy of the dataset, sensor-network augmentation, input assembly.

Every sample is normalised by its own scale ``s`` (median observed reading), so
inputs and targets are O(1) and, because the physics is linear, the network's
outputs times ``s`` are physical. Input channels (45):

    6   hourly sensor values / s (0 where unobserved)
    6   observation masks
    6   hourly inverse-distance-weighted interpolation of the readings / s (dense: spectral
        layers handle isolated nonzero pixels poorly)
    1   log(s / 100)
    12  hourly wind u, v at cell centres / 5 m/s
    6   hourly mixing height / 1000 m
    2   sin, cos of local hour at the window end
    4   prior maps (traffic, industry, construction, burning)
    2   x, y coordinates in [0, 1]

Output channels (11), all ≥ 0 and in units of ``s``:

    0-3  S per category during the last hour [µg m^-3 h^-1 / s]
    4-7  last-hour mean concentration due to each category
    8    regional background, last-hour mean (spatially averaged into one number)
    9    instantaneous c at the window end, 10: one hour before
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from airtrace.data.dataset import SyntheticDataset
from airtrace.data.synthetic import last_hour_background
from airtrace.grid import Grid

N_IN, N_OUT = 45, 11
S_SLICE, C_SLICE, BG, CT, CTM1 = slice(0, 4), slice(4, 8), 8, 9, 10
SPEC_WINDOW_KEYS = ("wind_u", "wind_v", "pert_amp", "mixing_height", "c_bg")


@dataclass
class Met:
    """Window meteorology for a batch (tensors on one device)."""

    wind_u: torch.Tensor  # (B, W) mean wind per window hour [m/s]
    wind_v: torch.Tensor
    pert_amp: torch.Tensor  # (B, W)
    pattern: torch.Tensor  # (B, 2, ny, nx) unit perturbation at cell centres
    H: torch.Tensor  # (B, W) mixing height [m]
    end_hour: torch.Tensor  # (B,)
    priors: torch.Tensor  # (B, 4, ny, nx)

    def wind_fields(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Cell-centred hourly wind ``(B, W, ny, nx)`` each."""
        u = self.wind_u[..., None, None] + self.pert_amp[..., None, None] * self.pattern[:, None, 0]
        v = self.wind_v[..., None, None] + self.pert_amp[..., None, None] * self.pattern[:, None, 1]
        return u, v


_IDW_KERNELS: dict[tuple, torch.Tensor] = {}


def idw_kernel(ny: int, nx: int, device, power: float = 2.0) -> torch.Tensor:
    """``(N, N)`` inverse-distance weights between cell centres (distance in cells, +1 to stay finite)."""
    key = (ny, nx, str(device), power)
    if key not in _IDW_KERNELS:
        yy, xx = torch.meshgrid(torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij")
        p = torch.stack([yy.reshape(-1), xx.reshape(-1)], 1).float()
        _IDW_KERNELS[key] = 1.0 / (torch.cdist(p, p) ** 2 + 0.25) ** (power / 2)
    return _IDW_KERNELS[key]


def idw_fields(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Dense IDW interpolation of each hour's readings, ``(B, W, ny, nx)``; 0 where an hour has no data."""
    B, W, ny, nx = values.shape
    Kmat = idw_kernel(ny, nx, values.device)
    v, m = values.reshape(B * W, -1), mask.reshape(B * W, -1)
    num, den = (v * m) @ Kmat, m @ Kmat
    return (num / den.clamp(min=1e-12)).reshape(B, W, ny, nx)


def build_inputs(values: torch.Tensor, mask: torch.Tensor, met: Met) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble the 45-channel input and the per-sample scale ``s`` (B,)."""
    B, W, ny, nx = values.shape
    obs = torch.where(mask > 0, values, torch.full_like(values, float("nan")))
    s = torch.nanmedian(obs.reshape(B, -1), dim=1).values
    s = torch.nan_to_num(s, nan=50.0).clamp(min=5.0)
    u, v = met.wind_fields()
    phase = 2 * torch.pi * met.end_hour.float() / 24
    ones = torch.ones(B, 1, ny, nx, device=values.device)
    ys = (torch.arange(ny, device=values.device) + 0.5) / ny
    xs = (torch.arange(nx, device=values.device) + 0.5) / nx
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    sc = s[:, None, None, None]
    x = torch.cat([
        values / sc * mask, mask, idw_fields(values, mask) / sc, ones * torch.log(s / 100.0)[:, None, None, None],
        u / 5.0, v / 5.0, ones * (met.H / 1000.0)[..., None, None],
        ones * torch.sin(phase)[:, None, None, None], ones * torch.cos(phase)[:, None, None, None],
        met.priors, X.expand(B, 1, ny, nx), Y.expand(B, 1, ny, nx),
    ], dim=1)
    return x, s


def sample_networks(c_hourly: torch.Tensor, n_range: tuple[int, int], noise_range: tuple[float, float],
                    dropout_range: tuple[float, float], gap_prob: float, gen: torch.Generator,
                    bias_range: tuple[float, float] = (0.0, 0.0), rho: float = 0.0
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Random sensor networks on the fly: ``n`` distinct cells per sample, lognormal noise, outages.

    The log error is a persistent per-sensor offset (s.d. drawn from ``bias_range``) plus an
    AR(1) hourly part (s.d. from ``noise_range``, lag-1 correlation ``rho``); see
    :func:`airtrace.data.sensors.log_error`.
    """
    B, W, ny, nx = c_hourly.shape
    dev = c_hourly.device
    N = ny * nx
    r = lambda *shape: torch.rand(*shape, generator=gen, device=dev)  # noqa: E731
    n = torch.randint(n_range[0], n_range[1] + 1, (B,), generator=gen, device=dev)
    rank = r(B, N).argsort(1).argsort(1)
    dropout = dropout_range[0] + (dropout_range[1] - dropout_range[0]) * r(B)
    online = (rank < n[:, None]) & (r(B, N) >= dropout[:, None])
    mask = (online[:, None, :] & (r(B, W, N) >= gap_prob)).float().reshape(B, W, ny, nx)
    sigma = (noise_range[0] + (noise_range[1] - noise_range[0]) * r(B))[:, None, None]
    beta = (bias_range[0] + (bias_range[1] - bias_range[0]) * r(B))[:, None, None]
    z = torch.randn(B, W, ny, nx, generator=gen, device=dev)
    a = [sigma * z[:, 0]]
    for k in range(1, W):
        a.append(rho * a[-1] + (1 - rho**2) ** 0.5 * sigma * z[:, k])
    e = torch.stack(a, 1) + (beta * torch.randn(B, ny, nx, generator=gen, device=dev))[:, None]
    noisy = c_hourly * torch.exp(e - ((sigma**2 + beta**2) / 2)[:, None])
    return noisy * mask, mask


def rasterize_network(cells: np.ndarray, readings: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Stored network → ``(values, mask)`` rasters ``(W, ny, nx)``; NaN readings are unobserved."""
    W = readings.shape[1]
    values = np.zeros((W, grid.ny, grid.nx), np.float32)
    mask = np.zeros_like(values)
    for k in range(W):
        ok = np.isfinite(readings[:, k])
        values[k, cells[ok, 0], cells[ok, 1]] = readings[ok, k]
        mask[k, cells[ok, 0], cells[ok, 1]] = 1.0
    return values, mask


class GPUData:
    """All fields needed for FNO training/evaluation, resident on one device (float16 fields)."""

    def __init__(self, ds: SyntheticDataset, window: int, device: str | torch.device, n_shards: int | None = None):
        self.device = device
        self.window = window
        shards = ds.shards[:n_shards]
        cat = lambda k: np.concatenate([np.load(p / f"{k}.npy") for p in shards])  # noqa: E731
        to = lambda a, dt=torch.float16: torch.as_tensor(a).to(device=device, dtype=dt)  # noqa: E731
        self.c_hourly = to(cat("c_hourly"))
        self.c_cat = to(cat("c_cat"))
        self.S_cat = to(cat("S_cat"))
        self.c_T, self.c_Tm1 = to(cat("c_T")), to(cat("c_Tm1"))
        self.pattern = to(cat("wind_pat"))
        self.layouts = to(ds.layouts, torch.float32)
        self.layout_id = to(cat("layout_id"), torch.long)
        self.end_hour = to(cat("end_hour"), torch.long)
        self.K, self.lam = to(cat("K"), torch.float32), to(cat("lam"), torch.float32)
        T = ds.meta["config"]["data"]["sim_hours"]
        self.spec = {k: to(cat(k)[:, T - window - 1:], torch.float32) for k in SPEC_WINDOW_KEYS}  # W + 1 hours
        self.n = self.c_hourly.shape[0]
        self.entrainment = bool(ds.meta["config"]["physics"].get("entrainment", False))

    def met(self, idx: torch.Tensor) -> Met:
        sp = {k: v[idx] for k, v in self.spec.items()}
        return Met(sp["wind_u"][:, 1:], sp["wind_v"][:, 1:], sp["pert_amp"][:, 1:], self.pattern[idx].float(),
                   sp["mixing_height"][:, 1:], self.end_hour[idx], self.layouts[self.layout_id[idx]])

    def targets(self, idx: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Scaled 11-channel target ``(B, 11, ny, nx)``."""
        B = len(idx)
        bg = last_hour_background(self.spec["c_bg"][idx])[:, None, None, None].expand(B, 1, *self.c_T.shape[-2:])
        y = torch.cat([self.S_cat[idx].float(), self.c_cat[idx].float(), bg,
                       self.c_T[idx, None].float(), self.c_Tm1[idx, None].float()], dim=1)
        return y / s[:, None, None, None]
