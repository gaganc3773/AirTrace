"""Physics-free spatial interpolation baselines: inverse-distance weighting and ordinary kriging."""

from __future__ import annotations

import numpy as np


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))


def idw(points: np.ndarray, values: np.ndarray, targets: np.ndarray, power: float = 2.0) -> np.ndarray:
    """Inverse-distance-weighted interpolation (exact at the data points)."""
    d = _dist(targets, points)
    hit = d < 1e-9
    w = 1.0 / np.maximum(d, 1e-9) ** power
    out = (w @ values) / w.sum(axis=1)
    rows = hit.any(axis=1)
    out[rows] = values[hit[rows].argmax(axis=1)]
    return out


def kriging(
    points: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    noise_rel: float = 0.15,
    ranges_m: tuple[float, ...] = (2e3, 4e3, 8e3, 16e3, 32e3),
) -> np.ndarray:
    """Ordinary kriging (GP with unknown constant mean) and an exponential covariance.

    The range is chosen by closed-form leave-one-out error; the sill is the sample
    variance and the nugget follows the multiplicative sensor-noise level.
    """
    n = len(values)
    sill = max(float(np.var(values)), 1e-6)
    nugget = (noise_rel * float(np.mean(values))) ** 2
    D = _dist(points, points)
    best = None
    for r in ranges_m:
        K = sill * np.exp(-D / r) + nugget * np.eye(n)
        Ki = np.linalg.inv(K)
        one = np.ones(n)
        mu = (one @ Ki @ values) / (one @ Ki @ one)
        a = Ki @ (values - mu)
        loo = float(np.mean((a / np.diag(Ki)) ** 2))
        if best is None or loo < best[0]:
            best = (loo, r, Ki, mu, a)
    _, r, Ki, mu, a = best
    k = sill * np.exp(-_dist(targets, points) / r)
    return mu + k @ a
