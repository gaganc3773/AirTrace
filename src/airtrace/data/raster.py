"""Rasterising polylines and polygons (in model metres) onto the grid."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from matplotlib.path import Path

from airtrace.grid import Grid


def rasterize_lines(
    grid: Grid, lines: Iterable[np.ndarray], weights: Iterable[float], step_m: float = 25.0
) -> np.ndarray:
    """Weighted line length per cell [m].

    Each polyline ``(n, 2)`` is resampled every ``step_m`` metres and each
    sample deposits ``weight * segment_length / n_samples`` into its cell, so
    the total over the grid equals the weighted in-domain length.
    """
    out = np.zeros((grid.ny, grid.nx))
    for line, w in zip(lines, weights):
        line = np.asarray(line, dtype=float)
        for a, b in zip(line[:-1], line[1:]):
            seg = float(np.hypot(*(b - a)))
            if seg == 0.0:
                continue
            n = max(1, int(np.ceil(seg / step_m)))
            s = (np.arange(n) + 0.5) / n
            pts = a + s[:, None] * (b - a)
            ix = np.floor(pts[:, 0] / grid.dx).astype(int)
            iy = np.floor(pts[:, 1] / grid.dy).astype(int)
            ok = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
            np.add.at(out, (iy[ok], ix[ok]), w * seg / n)
    return out


def polygon_coverage(grid: Grid, polygons: Iterable[np.ndarray], sub: int = 8) -> np.ndarray:
    """Fraction of each cell covered by the union of ``polygons`` (each ``(n, 2)`` in metres).

    Estimated with a ``sub x sub`` point lattice per cell.
    """
    xs = (np.arange(grid.nx * sub) + 0.5) * grid.dx / sub
    ys = (np.arange(grid.ny * sub) + 0.5) * grid.dy / sub
    hit = np.zeros((grid.ny * sub, grid.nx * sub), dtype=bool)
    for poly in polygons:
        poly = np.asarray(poly, dtype=float)
        if len(poly) < 3:
            continue
        x0, y0 = poly.min(axis=0)
        x1, y1 = poly.max(axis=0)
        i0, i1 = np.searchsorted(xs, [x0, x1])
        j0, j1 = np.searchsorted(ys, [y0, y1])
        if i1 <= i0 or j1 <= j0:
            continue
        X, Y = np.meshgrid(xs[i0:i1], ys[j0:j1])
        inside = Path(poly).contains_points(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
        hit[j0:j1, i0:i1] |= inside
    return hit.reshape(grid.ny, sub, grid.nx, sub).mean(axis=(1, 3))


def ellipse(cx: float, cy: float, rx: float, ry: float, angle: float = 0.0, n: int = 48) -> np.ndarray:
    """Polygon approximating an ellipse (metres, angle in radians)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = rx * np.cos(t), ry * np.sin(t)
    c, s = np.cos(angle), np.sin(angle)
    return np.column_stack([cx + c * x - s * y, cy + s * x + c * y])
