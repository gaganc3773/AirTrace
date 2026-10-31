"""Local metric projection between lon/lat and model-grid metres.

For a 48 km box a local equirectangular (tangent-plane) projection is accurate
to well under one grid cell (< 10 m), so no projection library is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from airtrace.config import DomainConfig

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class LocalProjection:
    """Maps lon/lat to metres from the domain's south-west corner and back."""

    center_lat: float
    center_lon: float
    width_m: float
    height_m: float

    @classmethod
    def from_config(cls, domain: DomainConfig) -> LocalProjection:
        """Projection for a configured domain."""
        return cls(domain.center_lat, domain.center_lon, domain.width_m, domain.height_m)

    @property
    def _kx(self) -> float:
        return EARTH_RADIUS_M * np.cos(np.radians(self.center_lat)) * np.pi / 180.0

    @property
    def _ky(self) -> float:
        return EARTH_RADIUS_M * np.pi / 180.0

    def to_xy(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """lon/lat [deg] to x/y [m] from the south-west corner."""
        x = (np.asarray(lon) - self.center_lon) * self._kx + self.width_m / 2
        y = (np.asarray(lat) - self.center_lat) * self._ky + self.height_m / 2
        return x, y

    def to_lonlat(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """x/y [m] to lon/lat [deg]."""
        lon = (np.asarray(x) - self.width_m / 2) / self._kx + self.center_lon
        lat = (np.asarray(y) - self.height_m / 2) / self._ky + self.center_lat
        return lon, lat

    def bbox(self, margin_m: float = 0.0) -> tuple[float, float, float, float]:
        """``(south, west, north, east)`` in degrees, Overpass order."""
        lon0, lat0 = self.to_lonlat(-margin_m, -margin_m)
        lon1, lat1 = self.to_lonlat(self.width_m + margin_m, self.height_m + margin_m)
        return float(lat0), float(lon0), float(lat1), float(lon1)
