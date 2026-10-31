"""Staggered (face-centred) wind fields.

``u`` lives on x-faces, shape ``(B, ny, nx + 1)``; ``v`` on y-faces, shape
``(B, ny + 1, nx)``. Directions follow the meteorological convention: the
direction the wind blows FROM, in degrees clockwise from north.
"""

from __future__ import annotations

import math

import torch

from airtrace.grid import Grid


def uniform_wind(
    grid: Grid,
    speed: torch.Tensor | float,
    direction_deg: torch.Tensor | float,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spatially uniform wind for a batch of ``B`` scenarios.

    Args:
        grid: Model grid.
        speed: Wind speed(s) [m/s], scalar or shape ``(B,)``.
        direction_deg: Meteorological direction(s) [deg], scalar or ``(B,)``.

    Returns:
        ``(u, v)`` face velocities.
    """
    speed = torch.as_tensor(speed, dtype=dtype, device=device).reshape(-1)
    theta = torch.as_tensor(direction_deg, dtype=dtype, device=device).reshape(-1) * (math.pi / 180.0)
    u = (-speed * torch.sin(theta)).reshape(-1, 1, 1)
    v = (-speed * torch.cos(theta)).reshape(-1, 1, 1)
    return u.expand(-1, grid.ny, grid.nx + 1).clone(), v.expand(-1, grid.ny + 1, grid.nx).clone()


def wind_from_streamfunction(grid: Grid, psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Discretely divergence-free wind from a streamfunction on cell corners.

    ``u = dpsi/dy``, ``v = -dpsi/dx``, so the finite-volume divergence of each
    cell vanishes exactly (telescoping corner values).

    Args:
        grid: Model grid.
        psi: Streamfunction [m²/s] on corners, shape ``(B, ny + 1, nx + 1)``.
    """
    u = (psi[:, 1:, :] - psi[:, :-1, :]) / grid.dy
    v = -(psi[:, :, 1:] - psi[:, :, :-1]) / grid.dx
    return u, v


def divergence(grid: Grid, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Finite-volume divergence per cell [1/s], shape ``(B, ny, nx)``."""
    return (u[..., 1:] - u[..., :-1]) / grid.dx + (v[..., 1:, :] - v[..., :-1, :]) / grid.dy
