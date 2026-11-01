"""PM2.5 station data for Delhi NCR from OpenAQ v3 (which ingests CPCB, DPCC and other reference monitors).

Verified against https://api.openaq.org/openapi.json: authentication is an
``X-API-Key`` header (free key from explore.openaq.org); the free tier allows
60 requests/min and 2,000/h. Stations come from ``/v3/locations`` (bbox =
min lon, min lat, max lon, max lat) and hourly averages from
``/v3/sensors/{id}/hours`` with ``datetime_from`` / ``datetime_to``, paginated
by ``limit`` / ``page``. Every response is cached under ``artifacts/cache/openaq``
so the demo needs neither the network nor the key once fetched.

The key is read from ``$OPENAQ_API_KEY`` or a git-ignored ``.openaq_key`` file in
the project root; it is never written to the cache or logs.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from airtrace.data.http import get_json, read_cache, write_cache

API_URL = "https://api.openaq.org/v3"
PM25_PARAMETER_ID = 2
PAGE_LIMIT = 1000


def api_key(root: Path) -> str | None:
    """OpenAQ key from the environment or ``<root>/.openaq_key``, else ``None``."""
    key = os.environ.get("OPENAQ_API_KEY")
    if not key and (root / ".openaq_key").exists():
        key = (root / ".openaq_key").read_text(encoding="utf-8").strip()
    return key or None


class OpenAQ:
    """Cached, rate-limited OpenAQ v3 client."""

    def __init__(self, key: str | None, cache_dir: Path, min_interval_s: float = 1.1):
        self.key, self.cache_dir, self.min_interval = key, cache_dir, min_interval_s
        self._last = 0.0

    def _get(self, path: str, params: dict, cache_name: str) -> list[dict]:
        """All pages of ``path`` (results concatenated), from the cache when available."""
        cache = self.cache_dir / f"{cache_name}.json.gz"
        hit = read_cache(cache)
        if hit is not None:
            return hit
        if not self.key:
            raise RuntimeError("OpenAQ needs an API key: set OPENAQ_API_KEY or put it in .openaq_key "
                               "(free account at https://explore.openaq.org)")
        results, page = [], 1
        while True:
            time.sleep(max(0.0, self._last + self.min_interval - time.monotonic()))
            body = get_json(f"{API_URL}{path}", {**params, "limit": PAGE_LIMIT, "page": page}, {"X-API-Key": self.key})
            self._last = time.monotonic()
            batch = body.get("results", [])
            results += batch
            if len(batch) < PAGE_LIMIT:
                break
            page += 1
        write_cache(cache, results)
        return results

    def locations(self, bbox_lonlat: tuple[float, float, float, float], monitors_only: bool = True) -> list[dict]:
        """Locations measuring PM2.5 inside ``(min_lon, min_lat, max_lon, max_lat)``."""
        params = {"bbox": ",".join(f"{v:.5f}" for v in bbox_lonlat), "parameters_id": PM25_PARAMETER_ID}
        if monitors_only:
            params["monitor"] = "true"
        return self._get("/locations", params, f"locations_{'monitors' if monitors_only else 'all'}")

    def sensor_hours(self, sensor_id: int, start: str, end: str) -> list[dict]:
        """Hourly averages for one sensor, ``start`` <= period start < ``end`` (ISO dates/datetimes, UTC)."""
        return self._get(f"/sensors/{sensor_id}/hours", {"datetime_from": start, "datetime_to": end},
                         f"hours_{sensor_id}_{start}_{end}".replace(":", ""))


@dataclass
class Station:
    """A PM2.5 monitor: OpenAQ location and its PM2.5 sensor."""

    location_id: int
    sensor_id: int
    name: str
    provider: str
    lat: float
    lon: float


def pm25_stations(locations: list[dict]) -> list[Station]:
    """One record per PM2.5 sensor (a location can hold several, e.g. after re-registration)."""
    out = []
    for loc in locations:
        if not loc.get("coordinates"):
            continue
        for s in loc.get("sensors", []):
            if (s.get("parameter") or {}).get("id") == PM25_PARAMETER_ID:
                out.append(Station(int(loc["id"]), int(s["id"]), str(loc.get("name", "")),
                                   str((loc.get("provider") or {}).get("name", "")),
                                   float(loc["coordinates"]["latitude"]), float(loc["coordinates"]["longitude"])))
    return out


def hourly_series(results: list[dict], times: np.ndarray) -> np.ndarray:
    """Values on the UTC hour axis ``times`` (datetime64[h]), NaN where missing.

    Indian monitors report IST clock hours, i.e. periods starting at hh:30 UTC; a
    period is labelled by its start truncated to the UTC hour (19:30-20:30Z -> 19).
    """
    out = np.full(len(times), np.nan)
    pos = {t: k for k, t in enumerate(times.astype("datetime64[h]"))}
    for r in results:
        t0 = np.datetime64(r["period"]["datetimeFrom"]["utc"].replace("Z", "")[:13], "h")
        if t0 in pos and r.get("value") is not None:
            out[pos[t0]] = float(r["value"])
    return out


def quality_control(pm: np.ndarray, max_value: float = 1500.0, stuck_hours: int = 6) -> np.ndarray:
    """Blank out non-positive, implausible and stuck readings (``(S, T)``, NaN = missing).

    A run of ``stuck_hours`` or more identical consecutive values is a frozen
    analyser, not air; such runs are removed entirely.
    """
    pm = np.where((pm > 0) & (pm <= max_value), pm, np.nan)
    for s in range(pm.shape[0]):
        x, start = pm[s], 0
        for k in range(1, len(x) + 1):
            if k == len(x) or not (x[k] == x[k - 1]):
                if k - start >= stuck_hours and np.isfinite(x[start]):
                    x[start:k] = np.nan
                start = k
    return pm
