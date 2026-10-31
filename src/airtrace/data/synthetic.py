"""Synthetic PM2.5 scenarios with known ground truth.

A scenario is fully described by a compact :class:`ScenarioSpecs` record
(wind, background, mixing height, emission inventory) plus a layout index;
:class:`ForcingBuilder` turns specs into hourly solver forcing
deterministically, so any scenario can be re-simulated exactly (Phase 3 needs
the full wind history for its Green's functions).

Time convention: simulated hour ``h = 0 .. T-1`` covers ``[h, h+1)``; the
window ends at local hour ``end_hour``, so hour ``h`` starts at local hour
``(end_hour - T + h) mod 24``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np
import torch

from airtrace.config import DataConfig, PhysicsConfig
from airtrace.data.layouts import CATEGORIES
from airtrace.grid import Grid
from airtrace.solver import ADRSolver, Forcing, wind_from_streamfunction

SCHEDULES = ("continuous", "day", "night", "evening")
SRC_FIELDS = ("category", "x", "y", "radius", "emission", "schedule", "off_prior", "active")
MAX_SOURCES = 24
N_MODES = 3
PRIOR_THRESHOLD = 0.05


def _daylight(hour: np.ndarray) -> np.ndarray:
    """Solar-ish factor in [0, 1]: zero outside 07-18 h, peaking near 12:30."""
    return np.clip(np.sin(np.pi * (hour - 7.0) / 11.0), 0.0, None) * ((hour >= 7) & (hour <= 18))


def _profiles() -> np.ndarray:
    """Hourly emission profiles ``(5, 24)``, each with daily mean 1.

    Rows 0-3 follow :data:`SCHEDULES`; row 4 is traffic (morning and evening
    rush plus Delhi's night-time truck entry window, 23:00-07:00).
    """
    h = np.arange(24) + 0.5
    night = ((h >= 20) | (h < 6)).astype(float)
    rows = [
        np.ones(24),
        np.where((h >= 8) & (h < 18), 1.0, 0.05),
        night,
        0.15 + np.exp(-(((h - 20.5 + 12) % 24 - 12) / 2.0) ** 2),
        0.3 + 0.7 * np.exp(-((h - 9) / 1.5) ** 2) + 0.8 * np.exp(-((h - 19) / 2.0) ** 2) + 0.45 * ((h >= 23) | (h < 7)),
    ]
    P = np.stack(rows)
    return P / P.mean(axis=1, keepdims=True)


PROFILES = _profiles()
TRAFFIC_PROFILE = 4


@dataclass
class ScenarioSpecs:
    """Batch of ``B`` scenario descriptions (NumPy arrays, leading dim ``B``, ``T`` hours)."""

    layout_id: np.ndarray  # (B,) int
    end_hour: np.ndarray  # (B,) int, local hour at window end
    K: np.ndarray  # (B,) m²/s
    lam: np.ndarray  # (B,) 1/s
    wind_u: np.ndarray  # (B, T) mean wind, m/s
    wind_v: np.ndarray  # (B, T)
    pert_amp: np.ndarray  # (B, T) perturbation speed scale, m/s
    pert_coef: np.ndarray  # (B, 2, N_MODES, N_MODES) streamfunction pattern
    c_bg: np.ndarray  # (B, T) µg/m³
    mixing_height: np.ndarray  # (B, T) m
    traffic_emission: np.ndarray  # (B,) µg/s at daily-mean profile
    traffic_coef: np.ndarray  # (B, 2, N_MODES, N_MODES) traffic intensity modulation
    sources: np.ndarray  # (B, MAX_SOURCES, len(SRC_FIELDS))

    def __len__(self) -> int:
        return len(self.layout_id)

    def as_dict(self) -> dict[str, np.ndarray]:
        """Arrays keyed by field name (for ``np.save``)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, np.ndarray]) -> ScenarioSpecs:
        """Inverse of :meth:`as_dict`; extra keys are ignored."""
        return cls(**{f.name: np.asarray(d[f.name]) for f in fields(cls)})

    def subset(self, idx: np.ndarray | slice | list[int]) -> ScenarioSpecs:
        """Specs for a subset of scenarios."""
        return ScenarioSpecs(**{k: v[idx] for k, v in self.as_dict().items()})

    def local_hours(self) -> np.ndarray:
        """Local hour at the start of each simulated hour, ``(B, T)`` int."""
        T = self.wind_u.shape[1]
        return (self.end_hour[:, None] - T + np.arange(T)[None]) % 24


def _loguniform(rng: np.random.Generator, lo: float, hi: float, size=None) -> np.ndarray:
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size))


def _mode_coefs(rng: np.random.Generator) -> np.ndarray:
    k = np.arange(1, N_MODES + 1)
    decay = 1.0 / (k[:, None] ** 2 + k[None, :] ** 2)
    return rng.normal(size=(2, N_MODES, N_MODES)) * decay


def _place(rng: np.random.Generator, grid: Grid, prior: np.ndarray, off_prior: bool) -> tuple[float, float, bool]:
    """Sample a location ∝ prior (on-prior) or uniformly where the prior is ~0 (off-prior)."""
    on = prior >= PRIOR_THRESHOLD
    if not off_prior and on.any():
        w = np.where(on, prior, 0.0).ravel()
    else:
        off_prior = True
        interior = np.zeros_like(on)
        interior[2:-2, 2:-2] = True
        w = (~on & interior).astype(float).ravel()
    k = rng.choice(w.size, p=w / w.sum())
    iy, ix = divmod(k, grid.nx)
    return (ix + rng.random()) * grid.dx, (iy + rng.random()) * grid.dy, off_prior


def sample_specs(
    rng: np.random.Generator, n: int, layouts: np.ndarray, grid: Grid, data: DataConfig, phys: PhysicsConfig
) -> ScenarioSpecs:
    """Draw ``n`` random scenarios; layout 0 (Delhi) is chosen with ``data.delhi_fraction``."""
    T = data.sim_hours
    out: dict[str, list] = {f.name: [] for f in fields(ScenarioSpecs)}
    for _ in range(n):
        layout_id = 0 if rng.random() < data.delhi_fraction else int(rng.integers(1, len(layouts)))
        prior = layouts[layout_id]
        end_hour = int(rng.integers(0, 24))
        lh = (end_hour - T + np.arange(T)) % 24 + 0.5

        theta0 = rng.uniform(280, 340) if rng.random() < data.nw_wind_prob else rng.uniform(0, 360)
        theta = np.radians(theta0 + np.cumsum(rng.normal(0, 10, T)))
        s0 = _loguniform(rng, 0.6, 4.5)
        speed = np.clip(s0 * (0.7 + 0.6 * _daylight(lh)) * np.exp(rng.normal(0, 0.15, T)), *phys.wind_speed_range)

        c_bg = _loguniform(rng, *phys.background_range) * (1 + 0.25 * np.cos(2 * np.pi * (lh - 4) / 24))
        H = (phys.mixing_height_night + (phys.mixing_height_day - phys.mixing_height_night) * _daylight(lh))
        H = H * rng.uniform(0.7, 1.3)

        # Emission inventory: domain total split across categories, then across sources.
        Q = _loguniform(rng, *data.total_emission_kg_s) * 1e9  # µg/s
        counts = [1, rng.integers(data.n_industry[0], data.n_industry[1] + 1),
                  rng.integers(data.n_construction[0], data.n_construction[1] + 1),
                  rng.integers(data.n_burning[0], data.n_burning[1] + 1)]
        shares = rng.dirichlet(data.category_alpha) * (np.array(counts) > 0)
        shares /= shares.sum()

        table = np.zeros((MAX_SOURCES, len(SRC_FIELDS)))
        row = 0
        for cat in (1, 2, 3):
            if counts[cat] == 0:
                continue
            w = rng.lognormal(0.0, 0.8, counts[cat])
            cap = data.max_source_g_s[cat - 1] * 1e6
            for e in np.minimum(Q * shares[cat] * w / w.sum(), cap):
                off = rng.random() < data.off_prior_prob
                x, y, off = _place(rng, grid, prior[cat], off)
                if cat == 1:
                    radius = rng.uniform(150, 600)  # stacks within an industrial cluster
                    sched = 2 if off and rng.random() < data.hidden_night_prob else (0 if rng.random() < 0.7 else 1)
                elif cat == 2:
                    radius, sched = rng.uniform(250, 700), 1
                else:
                    radius = rng.uniform(300, 1000)
                    sched = 3 if off or rng.random() < 0.5 else 0
                table[row] = (cat, x, y, radius, e, sched, float(off), 1.0)
                row += 1

        for key, val in dict(
            layout_id=layout_id, end_hour=end_hour, K=_loguniform(rng, *phys.K_range),
            lam=_loguniform(rng, *phys.lam_range), wind_u=-speed * np.sin(theta), wind_v=-speed * np.cos(theta),
            pert_amp=rng.uniform(*data.wind_pert_frac) * speed, pert_coef=_mode_coefs(rng), c_bg=c_bg,
            mixing_height=H, traffic_emission=Q * shares[0], traffic_coef=_mode_coefs(rng), sources=table,
        ).items():
            out[key].append(val)
    return ScenarioSpecs(**{k: np.asarray(v) for k, v in out.items()})


class ForcingBuilder:
    """Turns :class:`ScenarioSpecs` into hourly :class:`Forcing` on a device."""

    def __init__(self, specs: ScenarioSpecs, layouts: np.ndarray, grid: Grid,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        self.specs, self.grid, self.device, self.dtype = specs, grid, device, dtype
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)  # noqa: E731
        self._t = t
        self.local_hours = specs.local_hours()
        self.K, self.lam = t(specs.K), t(specs.lam)

        # Unit-max divergence-free perturbation pattern.
        psi = self._pattern(specs.pert_coef, corners=True)
        up, vp = wind_from_streamfunction(grid, psi)
        scale = torch.maximum(up.abs().amax(dim=(1, 2)), vp.abs().amax(dim=(1, 2))).clamp(min=1e-12)
        self.up, self.vp = up / scale[:, None, None], vp / scale[:, None, None]
        self.pert_scale = scale  # [psi units per m], so continuous models can rebuild the same wind

        # Traffic: road prior x lognormal-ish modulation, normalised to unit total.
        prior = t(layouts[specs.layout_id])
        mod = self._pattern(specs.traffic_coef, corners=False)
        mod = (mod - mod.mean(dim=(1, 2), keepdim=True)) / mod.std(dim=(1, 2), keepdim=True).clamp(min=1e-12)
        road = prior[:, 0] * torch.exp(0.5 * mod)
        self.traffic_map = road / road.sum(dim=(1, 2), keepdim=True).clamp(min=1e-12) * t(specs.traffic_emission)[:, None, None]

        # Discrete sources accumulated into (B, category, schedule, ny, nx) maps [µg/s].
        X, Y = grid.centers(dtype, device)
        src = t(specs.sources)
        sigma = torch.clamp(src[..., 3], min=0.35 * grid.dx)
        r2 = (X - src[..., 1, None, None]) ** 2 + (Y - src[..., 2, None, None]) ** 2
        w = torch.exp(-r2 / (2 * sigma[..., None, None] ** 2))
        w = w / w.sum(dim=(-1, -2), keepdim=True).clamp(min=1e-30) * (src[..., 4] * src[..., 7])[..., None, None]
        onehot = torch.zeros(*src.shape[:2], len(CATEGORIES), len(SCHEDULES), dtype=dtype, device=device)
        b_idx, s_idx = torch.meshgrid(torch.arange(src.shape[0]), torch.arange(src.shape[1]), indexing="ij")
        onehot[b_idx, s_idx, src[..., 0].long(), src[..., 5].long()] = 1.0
        self.source_maps = torch.einsum("bshw,bscq->bcqhw", w, onehot)

    def _pattern(self, coef: np.ndarray, corners: bool) -> torch.Tensor:
        g = self.grid
        n = (g.ny + 1, g.nx + 1) if corners else (g.ny, g.nx)
        off = 0.0 if corners else 0.5
        x = (torch.arange(n[1], dtype=self.dtype, device=self.device) + off) * g.dx / g.width
        y = (torch.arange(n[0], dtype=self.dtype, device=self.device) + off) * g.dy / g.height
        k = torch.arange(1, N_MODES + 1, dtype=self.dtype, device=self.device) * torch.pi
        sx, cx = torch.sin(k[:, None] * x), torch.cos(k[:, None] * x)  # (modes, nx)
        sy, cy = torch.sin(k[:, None] * y), torch.cos(k[:, None] * y)
        c = self._t(coef)
        return torch.einsum("bkl,ly,kx->byx", c[:, 0], sy, sx) + torch.einsum("bkl,ly,kx->byx", c[:, 1], cy, cx)

    def emission_by_category(self, h: int) -> torch.Tensor:
        """Emission rate per cell [µg/s] during simulated hour ``h``, ``(B, 4, ny, nx)``."""
        lh = self.local_hours[:, h]
        prof = self._t(PROFILES[:, lh].T)  # (B, 5)
        traffic = self.traffic_map * prof[:, TRAFFIC_PROFILE, None, None]
        rest = torch.einsum("bcqhw,bq->bchw", self.source_maps[:, 1:], prof[:, : len(SCHEDULES)])
        return torch.cat([traffic[:, None], rest], dim=1)

    def source_by_category(self, h: int) -> torch.Tensor:
        """Volumetric source S per category [µg m^-3 s^-1] during hour ``h``."""
        H = self._t(self.specs.mixing_height[:, h])
        return self.emission_by_category(h) / (self.grid.cell_area * H[:, None, None, None])

    def regional_source(self, h: int) -> torch.Tensor:
        """Uniform regional term ``lam c_bg + dc_bg/dt`` [µg m^-3 s^-1], shape ``(B,)``."""
        bg = self.specs.c_bg
        h0, h1 = (h, h + 1) if h + 1 < bg.shape[1] else (h - 1, h)
        return self.lam * self._t(bg[:, h]) + self._t((bg[:, h1] - bg[:, h0]) / 3600.0)

    def forcing(self, h: int) -> Forcing:
        """Solver forcing for simulated hour ``h``.

        Besides local sources, S includes a uniform regional term
        ``lam * c_bg + d(c_bg)/dt`` so that a spatially uniform background
        ``c_bg(t)`` is an exact solution: removal acts on the local excess, and
        the regional haze neither decays away in calm air nor lags the inflow.
        """
        s = self.specs
        amp = self._t(s.pert_amp[:, h])[:, None, None]
        u = self._t(s.wind_u[:, h])[:, None, None] + amp * self.up
        v = self._t(s.wind_v[:, h])[:, None, None] + amp * self.vp
        c_bg = self._t(s.c_bg[:, h])
        S = self.source_by_category(h).sum(dim=1) + self.regional_source(h)[:, None, None]
        return Forcing(u, v, S, c_bg)


def simulate(builder: ForcingBuilder, solver: ADRSolver, window: int, substeps: int = 6) -> dict[str, torch.Tensor]:
    """Run all scenarios in a builder and collect the observation window.

    Returns:
        ``c_hourly`` (B, window, ny, nx) hourly-mean concentration,
        ``c_T`` / ``c_Tm1`` instantaneous fields at the window end and one hour before,
        ``S_cat`` (B, 4, ny, nx) source per category during the last hour [µg m^-3 h^-1].
    """
    s = builder.specs
    T = s.wind_u.shape[1]
    c = builder._t(s.c_bg[:, 0])[:, None, None].expand(-1, builder.grid.ny, builder.grid.nx).clone()
    hourly, out = [], {}
    for h in range(T):
        f = builder.forcing(h)
        _, snaps = solver.run(c, lambda t: f, builder.K, builder.lam, substeps, 3600.0 / substeps)
        if h >= T - window:
            hourly.append(torch.trapezoid(snaps, dim=1) / substeps)
        if h == T - 1:
            out["c_Tm1"], out["S_cat"] = snaps[:, 0], builder.source_by_category(h) * 3600.0
        c = snaps[:, -1]
    out["c_T"] = c
    out["c_hourly"] = torch.stack(hourly, dim=1)
    return out
