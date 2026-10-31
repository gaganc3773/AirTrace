"""Uniform Cartesian finite-volume grid.

Arrays are laid out as ``[..., y, x]`` with row 0 at the southern edge and
column 0 at the western edge. Cell (j, i) has its centre at
``((i + 0.5) dx, (j + 0.5) dy)`` metres from the south-west corner.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from airtrace.config import DomainConfig


@dataclass(frozen=True)
class Grid:
    """Uniform grid of ``ny x nx`` cells of size ``dx x dy`` metres."""

    nx: int
    ny: int
    dx: float
    dy: float

    @classmethod
    def from_config(cls, domain: DomainConfig) -> Grid:
        """Build the grid described by a domain config."""
        return cls(domain.nx, domain.ny, domain.width_m / domain.nx, domain.height_m / domain.ny)

    @property
    def width(self) -> float:
        """Domain extent in x [m]."""
        return self.nx * self.dx

    @property
    def height(self) -> float:
        """Domain extent in y [m]."""
        return self.ny * self.dy

    @property
    def cell_area(self) -> float:
        """Area of one cell [m²]."""
        return self.dx * self.dy

    def centers(
        self, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cell-centre coordinates ``(X, Y)``, each of shape ``(ny, nx)`` [m]."""
        x = (torch.arange(self.nx, dtype=dtype, device=device) + 0.5) * self.dx
        y = (torch.arange(self.ny, dtype=dtype, device=device) + 0.5) * self.dy
        Y, X = torch.meshgrid(y, x, indexing="ij")
        return X, Y

    def extent_km(self) -> tuple[float, float, float, float]:
        """``(left, right, bottom, top)`` in km, for ``imshow(origin="lower")``."""
        return (0.0, self.width / 1e3, 0.0, self.height / 1e3)
