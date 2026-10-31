"""Synthetic low-cost sensor networks: placement, noisy readings, rasterisation."""

from __future__ import annotations

import numpy as np

from airtrace.config import SensorConfig
from airtrace.grid import Grid


def sample_sensor_cells(rng: np.random.Generator, n: int, grid: Grid, min_dist_m: float) -> np.ndarray:
    """``n`` distinct cells ``(iy, ix)`` at least ``min_dist_m`` apart (greedy dart throwing).

    Falls back to relaxing the distance if the domain cannot fit ``n`` sensors.
    """
    cells: list[tuple[int, int]] = []
    d2 = min_dist_m**2
    tries = 0
    while len(cells) < n:
        iy, ix = int(rng.integers(0, grid.ny)), int(rng.integers(0, grid.nx))
        ok = all(((iy - j) * grid.dy) ** 2 + ((ix - i) * grid.dx) ** 2 >= d2 for j, i in cells)
        if ok:
            cells.append((iy, ix))
        tries += 1
        if tries > 200 * n:
            d2 *= 0.8
            tries = 0
    return np.array(cells, dtype=np.int16)


def observe(
    rng: np.random.Generator, c_hourly: np.ndarray, cells: np.ndarray, noise: float, dropout: float, gap_prob: float
) -> np.ndarray:
    """Noisy hourly readings ``(n, W)``; NaN where a sensor is offline or has a gap.

    Error model: unbiased multiplicative lognormal ``c * exp(N(0, σ²) - σ²/2)``.
    """
    truth = c_hourly[:, cells[:, 0], cells[:, 1]].T.astype(np.float64)  # (n, W)
    r = truth * np.exp(rng.normal(0.0, noise, truth.shape) - noise**2 / 2)
    offline = rng.random(len(cells)) < dropout
    gaps = rng.random(truth.shape) < gap_prob
    r[offline] = np.nan
    r[gaps] = np.nan
    return r


def sample_network(
    rng: np.random.Generator, c_hourly: np.ndarray, grid: Grid, cfg: SensorConfig
) -> tuple[np.ndarray, np.ndarray, float]:
    """Default network for one scenario: ``(cells, readings, noise_sigma)``."""
    cells = sample_sensor_cells(rng, cfg.n_sensors, grid, cfg.min_dist_m)
    noise = rng.uniform(*cfg.noise_range)
    readings = observe(rng, c_hourly, cells, noise, rng.uniform(*cfg.dropout_range), cfg.hourly_gap_prob)
    return cells, readings, noise


def rasterize(readings: np.ndarray, cells: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Readings ``(n, W)`` → value rasters ``(W, ny, nx)`` (0 where unobserved) and mask ``(W, ny, nx)``."""
    W = readings.shape[1]
    values = np.zeros((W, grid.ny, grid.nx), dtype=np.float32)
    mask = np.zeros_like(values)
    ok = ~np.isnan(readings)
    for k in range(W):
        sel = ok[:, k]
        values[k, cells[sel, 0], cells[sel, 1]] = readings[sel, k]
        mask[k, cells[sel, 0], cells[sel, 1]] = 1.0
    return values, mask
