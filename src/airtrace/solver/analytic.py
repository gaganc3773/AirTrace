"""Closed-form solutions used to verify the numerical solver."""

from __future__ import annotations

import torch


def gaussian_puff(
    X: torch.Tensor,
    Y: torch.Tensor,
    t: float,
    amplitude: float,
    x0: float,
    y0: float,
    sigma0: float,
    u: float,
    v: float,
    K: float,
    lam: float,
) -> torch.Tensor:
    """Isotropic Gaussian puff under uniform wind, constant K and first-order decay.

    Initial state ``A exp(-r²/2σ0²)``; in free space
    ``c = A (σ0²/σ²) exp(-|x - x0 - u t|²/2σ²) exp(-λ t)`` with ``σ² = σ0² + 2 K t``.
    """
    s2 = sigma0**2 + 2.0 * K * t
    r2 = (X - x0 - u * t) ** 2 + (Y - y0 - v * t) ** 2
    return amplitude * (sigma0**2 / s2) * torch.exp(-r2 / (2.0 * s2)) * torch.exp(torch.tensor(-lam * t, dtype=X.dtype))
