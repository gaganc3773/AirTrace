"""Source-type prior maps ("layouts"): real Delhi geography and random synthetic cities.

A layout is a ``(4, ny, nx)`` array in [0, 1], one channel per emission
category in :data:`CATEGORIES` order:

* traffic       weighted major-road length density
* industry      industrial land-use coverage
* construction  construction / brownfield coverage
* burning       landfill coverage (+ 0.3 x farmland, for crop-residue burning)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from airtrace.data.geo import LocalProjection
from airtrace.data.osm import fetch_delhi_osm, landuse_polygons, road_lines
from airtrace.data.raster import ellipse, polygon_coverage, rasterize_lines
from airtrace.grid import Grid

CATEGORIES = ("traffic", "industry", "construction", "burning")


def _normalize_roads(length_per_cell: np.ndarray, grid: Grid) -> np.ndarray:
    """Weighted road length per cell → [0, 1], saturating at the 99th percentile."""
    density = length_per_cell / grid.dx
    top = np.percentile(density[density > 0], 99) if (density > 0).any() else 1.0
    return np.clip(density / top, 0.0, 1.0)


def delhi_layout(grid: Grid, proj: LocalProjection, cache_dir: Path) -> np.ndarray:
    """Prior maps for the configured Delhi domain from (cached) OpenStreetMap data."""
    roads, landuse = fetch_delhi_osm(proj, cache_dir)
    lines, weights = road_lines(roads, proj)
    polys = landuse_polygons(landuse, proj)
    traffic = _normalize_roads(rasterize_lines(grid, lines, weights), grid)
    industry = polygon_coverage(grid, polys["industrial"])
    construction = polygon_coverage(grid, polys["construction"] + polys["brownfield"])
    burning = np.clip(polygon_coverage(grid, polys["landfill"]) + 0.3 * polygon_coverage(grid, polys["farmland"]), 0, 1)
    return np.stack([traffic, industry, construction, burning]).astype(np.float32)


def random_layout(grid: Grid, rng: np.random.Generator) -> np.ndarray:
    """A plausible synthetic city: ring road, radial arterials, street grid, industrial
    estates, scattered construction sites, landfills and peri-urban farmland."""
    W, H = grid.width, grid.height
    c = np.array([W, H]) * rng.uniform(0.35, 0.65, size=2)
    lines, weights = [], []

    r_ring = rng.uniform(8e3, 16e3)
    t = np.linspace(0, 2 * np.pi, 97)
    wobble = 1 + 0.12 * np.sin(rng.integers(2, 5) * t + rng.uniform(0, 2 * np.pi))
    lines.append(c + (r_ring * wobble)[:, None] * np.column_stack([np.cos(t), np.sin(t)]))
    weights.append(1.0)

    for ang in rng.uniform(0, 2 * np.pi, size=rng.integers(3, 7)):
        s = np.linspace(0, 40e3, 41)
        bend = np.cumsum(rng.normal(0, 0.04, size=s.size))
        lines.append(c + s[:, None] * np.column_stack([np.cos(ang + bend), np.sin(ang + bend)]))
        weights.append(0.8)

    theta = rng.uniform(0, np.pi / 2)
    d1, d2 = np.array([np.cos(theta), np.sin(theta)]), np.array([-np.sin(theta), np.cos(theta)])
    for _ in range(rng.integers(6, 14)):
        offset = rng.uniform(-20e3, 20e3)
        d, n = (d1, d2) if rng.random() < 0.5 else (d2, d1)
        lines.append(c + offset * n + np.outer(np.linspace(-r_ring * 1.2, r_ring * 1.2, 2), d))
        weights.append(rng.choice([0.4, 0.2]))
    traffic = _normalize_roads(rasterize_lines(grid, lines, weights), grid)

    def blobs(n: int, r_lo: float, r_hi: float, where: str = "any") -> list[np.ndarray]:
        out = []
        for _ in range(n):
            while True:
                p = rng.uniform([2e3, 2e3], [W - 2e3, H - 2e3])
                dist = np.hypot(*(p - c))
                if where == "any" or (where == "outer" and dist > r_ring) or (where == "inner" and dist < r_ring):
                    break
            out.append(ellipse(*p, rng.uniform(r_lo, r_hi), rng.uniform(r_lo, r_hi), rng.uniform(0, np.pi)))
        return out

    industry = polygon_coverage(grid, blobs(rng.integers(2, 6), 1e3, 3e3))
    construction = polygon_coverage(grid, blobs(rng.integers(5, 16), 250.0, 700.0, "inner"))
    landfill = polygon_coverage(grid, blobs(rng.integers(1, 4), 400.0, 900.0))
    farmland = polygon_coverage(grid, blobs(rng.integers(3, 8), 2e3, 5e3, "outer"))
    burning = np.clip(landfill + 0.3 * farmland, 0, 1)
    return np.stack([traffic, industry, construction, burning]).astype(np.float32)


def build_layouts(grid: Grid, proj: LocalProjection, cache_dir: Path, n_random: int, seed: int) -> np.ndarray:
    """Layout bank: index 0 is Delhi, 1..n_random are synthetic cities."""
    rng = np.random.default_rng(seed)
    bank = [delhi_layout(grid, proj, cache_dir)] + [random_layout(grid, rng) for _ in range(n_random)]
    return np.stack(bank)
