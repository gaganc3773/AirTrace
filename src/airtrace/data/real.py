"""Real Delhi observations on the model grid: station PM2.5 (OpenAQ) + meteorology (Open-Meteo).

OpenAQ lists many physical stations more than once (the same CPCB/DPCC monitor
under two providers, or under an old and a new location id). Records within
``dedup_m`` of each other are merged by keeping the most complete one.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from airtrace.config import Config
from airtrace.data.geo import LocalProjection
from airtrace.data.openaq import OpenAQ, Station, hourly_series, pm25_stations, quality_control
from airtrace.data.openmeteo import fetch_met
from airtrace.grid import Grid


@dataclass
class RealData:
    """Hourly station PM2.5 and domain meteorology on a common UTC hour axis."""

    times: np.ndarray  # (T,) datetime64[h] UTC; station value k covers [t_k + 30 min, t_k + 90 min)
    names: np.ndarray  # (S,) station names
    providers: np.ndarray  # (S,)
    lonlat: np.ndarray  # (S, 2)
    cells: np.ndarray  # (S, 2) grid (iy, ix)
    pm25: np.ndarray  # (S, T) µg/m³, NaN missing
    u: np.ndarray  # (T,) m/s at the middle of station period k, i.e. Open-Meteo hour t_k + 1
    v: np.ndarray
    H: np.ndarray  # (T,) boundary-layer height [m], same timing

    @property
    def local_hour(self) -> np.ndarray:
        """Local (IST, UTC+5:30) clock hour at the end of each period: [t+0:30, t+1:30) UTC ends at t+7 IST."""
        h = (self.times - self.times.astype("datetime64[D]")).astype(int)
        return (h + 7) % 24

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path) -> RealData:
        z = np.load(path, allow_pickle=False)
        return cls(**{k: z[k] for k in cls.__dataclass_fields__})


def dedupe(stations: list[Station], series: np.ndarray, proj: LocalProjection, dedup_m: float = 300.0
           ) -> tuple[list[Station], np.ndarray]:
    """Merge co-located records, keeping the one with the most valid hours."""
    x, y = proj.to_xy(np.array([s.lon for s in stations]), np.array([s.lat for s in stations]))
    order = np.argsort(-np.isfinite(series).sum(1))
    keep: list[int] = []
    for k in order:
        if np.isfinite(series[k]).sum() == 0:
            continue
        if all(np.hypot(x[k] - x[j], y[k] - y[j]) > dedup_m for j in keep):
            keep.append(int(k))
    keep.sort()
    return [stations[k] for k in keep], series[keep]


def build_real(cfg: Config, key: str | None, min_coverage: float = 0.5, log=print) -> RealData:
    """Fetch (or replay from cache) stations, hourly PM2.5 and meteorology for ``cfg.real``."""
    rc = cfg.real
    proj = LocalProjection.from_config(cfg.domain)
    grid = Grid.from_config(cfg.domain)
    cache = cfg.path("artifacts") / "cache"
    s, w, n, e = proj.bbox()
    client = OpenAQ(key, cache / "openaq")
    stations = pm25_stations(client.locations((w, s, e, n), rc.monitors_only))
    end_excl = str(np.datetime64(rc.end) + np.timedelta64(1, "D"))
    times = np.arange(np.datetime64(f"{rc.start}T00"), np.datetime64(f"{end_excl}T00"))
    series = []
    for k, st in enumerate(stations):
        series.append(hourly_series(client.sensor_hours(st.sensor_id, rc.start, end_excl), times))
        log(f"  [{k + 1}/{len(stations)}] {st.name[:45]:45s} {np.isfinite(series[-1]).mean():.0%} hours")
    pm = quality_control(np.array(series), rc.max_pm25, rc.stuck_hours)
    stations, pm = dedupe(stations, pm, proj)
    ok = np.isfinite(pm).mean(1) >= min_coverage
    stations, pm = [st for st, o in zip(stations, ok) if o], pm[ok]
    lon, lat = np.array([st.lon for st in stations]), np.array([st.lat for st in stations])
    x, y = proj.to_xy(lon, lat)
    cells = np.column_stack([np.clip((y // grid.dy).astype(int), 0, grid.ny - 1),
                             np.clip((x // grid.dx).astype(int), 0, grid.nx - 1)])
    met = fetch_met(proj, rc.start, end_excl, cache / "real", rc.met_points)  # one extra day: hour t+1 at the end
    assert np.array_equal(met.times[:len(times)], times), "Open-Meteo and OpenAQ hour axes differ"
    mid = slice(1, len(times) + 1)  # met at the middle of each station period
    return RealData(times, np.array([st.name for st in stations]), np.array([st.provider for st in stations]),
                    np.column_stack([lon, lat]), cells, pm, met.u[mid], met.v[mid], met.H[mid])


def write_station_table(data: RealData, path: Path) -> None:
    """CSV of the stations used: name, provider, lon, lat, grid cell, coverage, mean PM2.5."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["name", "provider", "lon", "lat", "iy", "ix", "coverage", "mean_pm25"])
        for k in range(len(data.names)):
            wr.writerow([data.names[k], data.providers[k], f"{data.lonlat[k, 0]:.4f}", f"{data.lonlat[k, 1]:.4f}",
                         *data.cells[k], f"{np.isfinite(data.pm25[k]).mean():.2f}", f"{np.nanmean(data.pm25[k]):.1f}"])


class RealWindows:
    """FNO inputs for windows of real data; ``met(idx)`` serves the batch as :func:`predict` expects.

    Meteorology outside the synthetic training ranges is clipped to them (wind speed,
    and above all the very shallow ERA5 night-time boundary layer); see :attr:`clipped_fraction`.
    """

    def __init__(self, data: RealData, window: int, priors: np.ndarray, speed_range: tuple[float, float],
                 H_range: tuple[float, float], device="cpu"):
        import torch

        self.data, self.window, self.device = data, window, device
        speed = np.hypot(data.u, data.v)
        scale = np.clip(speed, *speed_range) / np.maximum(speed, 1e-9)
        self.u, self.v = data.u * scale, data.v * scale
        self.H = np.clip(data.H, *H_range)
        self.clipped_fraction = {"wind_speed": float(np.mean(scale != 1.0)), "H": float(np.mean(self.H != data.H))}
        self._priors = torch.as_tensor(priors, dtype=torch.float32, device=device)
        self._ends: np.ndarray = np.zeros(0, int)

    def set_batch(self, ends: np.ndarray) -> None:
        """Window end indices (into ``data.times``) of the samples about to be predicted."""
        self._ends = np.asarray(ends)

    def met(self, idx):
        import torch

        from airtrace.models.fno_data import Met

        ends = self._ends[idx.cpu().numpy() if hasattr(idx, "cpu") else idx]
        sl = ends[:, None] + np.arange(-self.window + 1, 1)[None]
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)  # noqa: E731
        B, (ny, nx) = len(ends), self._priors.shape[-2:]
        return Met(t(self.u[sl]), t(self.v[sl]), torch.zeros(B, self.window, device=self.device),
                   torch.zeros(B, 2, ny, nx, device=self.device), t(self.H[sl]),
                   torch.as_tensor(self.data.local_hour[ends], device=self.device),
                   self._priors.expand(B, -1, -1, -1))

    def rasters(self, end: int, grid: Grid, exclude: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """``(W, ny, nx)`` values and mask for the window ending at ``end``; stations sharing a cell are averaged.

        ``exclude``: boolean ``(S,)`` stations to leave out (held-out station and its cell-mates).
        """
        W = self.window
        pm = self.data.pm25[:, end - W + 1:end + 1]
        keep = np.ones(len(pm), bool) if exclude is None else ~exclude
        tot = np.zeros((W, grid.ny, grid.nx))
        cnt = np.zeros_like(tot)
        for s in np.flatnonzero(keep):
            ok = np.isfinite(pm[s])
            iy, ix = self.data.cells[s]
            tot[ok, iy, ix] += pm[s, ok]
            cnt[ok, iy, ix] += 1
        mask = (cnt > 0).astype(np.float32)
        return (tot / np.maximum(cnt, 1)).astype(np.float32), mask
