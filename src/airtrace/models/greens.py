"""Source-receptor (Green's function) matrices from the linear solver.

The candidate source space is a coarse grid of ``block x block`` cell
patches. Column ``j`` of the response matrix is the concentration produced
by a constant 1 g/s emission spread uniformly over block ``j`` for the whole
simulation, converted to a volumetric source with the scenario's hourly
mixing height. Columns come from one batched forward run with the exactly
linear ``linear3`` scheme, so ``G @ E`` is exact superposition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from airtrace.data.synthetic import ForcingBuilder, ScenarioSpecs
from airtrace.grid import Grid
from airtrace.solver import ADRSolver, Forcing


def block_basis(grid: Grid, block: int, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    """Indicator masks of the ``(ny/block) * (nx/block)`` source blocks, shape ``(J, ny, nx)``."""
    nby, nbx = grid.ny // block, grid.nx // block
    ids = torch.arange(nby * nbx, device=device).reshape(nby, nbx)
    ids = ids.repeat_interleave(block, 0).repeat_interleave(block, 1)
    return (ids[None] == torch.arange(nby * nbx, device=device)[:, None, None]).to(dtype)


def block_sum(field: np.ndarray, block: int) -> np.ndarray:
    """Sum a ``(..., ny, nx)`` field over ``block x block`` patches."""
    *lead, ny, nx = field.shape
    return field.reshape(*lead, ny // block, block, nx // block, block).sum(axis=(-3, -1))


@dataclass
class Response:
    """Response of hourly-mean concentration [µg/m³] to 1 g/s per source block.

    Attributes:
        sensors: ``(n_sensors, W, J)`` at the sensor cells for each window hour.
        field_last: ``(J, ny, nx)`` full field, last window hour.
    """

    sensors: np.ndarray
    field_last: np.ndarray


def response_matrix(
    specs: ScenarioSpecs,
    layouts: np.ndarray,
    grid: Grid,
    cells: np.ndarray,
    window: int,
    block: int = 4,
    device: str | torch.device = "cpu",
    substeps: int = 6,
) -> Response:
    """Green's functions for one scenario (``len(specs) == 1``) under its own meteorology."""
    assert len(specs) == 1, "one scenario at a time"
    solver = ADRSolver(grid, "linear3")
    fb = ForcingBuilder(specs, layouts, grid, device)
    basis = block_basis(grid, block, device=device)
    J = basis.shape[0]
    per_cell = basis / basis.sum(dim=(1, 2), keepdim=True) * 1e6  # µg/s per cell for 1 g/s per block
    T = specs.wind_u.shape[1]
    iy, ix = torch.as_tensor(cells[:, 0].astype(np.int64)), torch.as_tensor(cells[:, 1].astype(np.int64))
    c = torch.zeros(J, grid.ny, grid.nx, device=device)
    zero_bg = torch.zeros(J, device=device)
    sensors, field_last = [], None
    for h in range(T):
        met = fb.forcing(h)
        S = per_cell / (grid.cell_area * float(specs.mixing_height[0, h]))
        f = Forcing(met.u.expand(J, -1, -1), met.v.expand(J, -1, -1), S, zero_bg)
        _, snaps = solver.run(c, lambda t: f, fb.K, fb.lam, substeps, 3600.0 / substeps)
        if h >= T - window:
            mean = torch.trapezoid(snaps, dim=1) / substeps
            sensors.append(mean[:, iy, ix].T.cpu())  # (n, J)
            if h == T - 1:
                field_last = mean.cpu().numpy()
        c = snaps[:, -1]
    return Response(torch.stack(sensors, dim=1).numpy(), field_last)
