"""Hourly meteorology from the Open-Meteo historical archive (ERA5-based; no API key), cached for offline use.

Wind is taken at 100 m, closer to the mean transport wind of the mixed layer than
the 10 m wind; the boundary-layer height plays the model's mixing height H.
Times are requested in UTC so they align with OpenAQ's hourly periods.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from airtrace.data.geo import LocalProjection
from airtrace.data.http import get_json, read_cache, write_cache

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = ("wind_speed_100m", "wind_direction_100m", "boundary_layer_height")


@dataclass
class HourlyMet:
    """Domain meteorology on an hourly UTC axis (value at the start of each hour)."""

    times: np.ndarray  # (T,) datetime64[h], UTC
    u: np.ndarray  # (T,) domain-mean eastward wind [m/s]
    v: np.ndarray  # (T,) northward
    H: np.ndarray  # (T,) domain-mean boundary-layer height [m]
    u_pts: np.ndarray  # (P, T) per sample point, for spatial-variability diagnostics
    v_pts: np.ndarray


def sample_points(proj: LocalProjection, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``n x n`` lon/lat points at the centres of an even partition of the domain."""
    f = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(f * proj.width_m, f * proj.height_m)
    return proj.to_lonlat(X.ravel(), Y.ravel())


def fetch_met(proj: LocalProjection, start: str, end: str, cache_dir: Path, n: int = 3) -> HourlyMet:
    """Hourly 100 m wind and boundary-layer height for ``start..end`` (inclusive dates, UTC)."""
    path = cache_dir / f"openmeteo_{start}_{end}_{n}x{n}.json.gz"
    raw = read_cache(path)
    if raw is None:
        lon, lat = sample_points(proj, n)
        raw = get_json(ARCHIVE_URL, {
            "latitude": ",".join(f"{x:.4f}" for x in lat), "longitude": ",".join(f"{x:.4f}" for x in lon),
            "start_date": start, "end_date": end, "hourly": ",".join(VARIABLES),
            "wind_speed_unit": "ms", "timezone": "GMT",
        })
        raw = raw if isinstance(raw, list) else [raw]
        write_cache(path, raw)
    return parse_met(raw)


@dataclass
class MetBank:
    """Concatenated hourly ERA5 records from several seasons, for sampling training meteorology."""

    times: np.ndarray  # (N,) datetime64[h] UTC
    u: np.ndarray  # (N,)
    v: np.ndarray
    H: np.ndarray
    season: np.ndarray  # (N,) int, contiguous record id

    def starts(self, length: int) -> np.ndarray:
        """Indices ``s`` such that ``s .. s + length`` lie in one season with finite values."""
        n = len(self.times)
        s = np.arange(max(n - length, 0))
        ok = self.season[s] == self.season[s + length]
        bad = ~np.isfinite(self.u + self.v + self.H)
        cum = np.concatenate([[0], np.cumsum(bad)])
        ok &= (cum[s + length + 1] - cum[s]) == 0
        return s[ok]


def met_bank(proj: LocalProjection, seasons: list[tuple[str, str]], cache_dir: Path, n: int = 3) -> MetBank:
    """Fetch (or replay) each season and concatenate."""
    parts = [fetch_met(proj, a, b, cache_dir, n) for a, b in seasons]
    cat = lambda k: np.concatenate([getattr(p, k) for p in parts])  # noqa: E731
    return MetBank(cat("times"), cat("u"), cat("v"), cat("H"),
                   np.concatenate([np.full(len(p.times), k) for k, p in enumerate(parts)]))


def parse_met(raw: list[dict]) -> HourlyMet:
    """Turn Open-Meteo multi-location responses into domain-mean hourly series."""
    times = np.array(raw[0]["hourly"]["time"], dtype="datetime64[h]")
    get = lambda k: np.array([[np.nan if x is None else x for x in r["hourly"][k]] for r in raw], float)  # noqa: E731
    speed, dirn, blh = get(VARIABLES[0]), get(VARIABLES[1]), get(VARIABLES[2])
    rad = np.radians(dirn)  # meteorological convention: direction the wind blows from
    u, v = -speed * np.sin(rad), -speed * np.cos(rad)
    return HourlyMet(times, np.nanmean(u, 0), np.nanmean(v, 0), np.nanmean(blh, 0), u, v)
