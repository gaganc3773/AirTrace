"""Evaluation metrics shared by all inversion models (baseline, PINN, FNO).

Source maps are compared on the coarse block grid (``block x block`` cells,
3 km at the default resolution) in g/s per block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter

from airtrace.data.synthetic import PROFILES, ScenarioSpecs


def hotspots(E_blocks: np.ndarray, tau: float) -> np.ndarray:
    """Block indices ``(k, 2)`` of local maxima (8-neighbourhood) with emission ``>= tau``."""
    peak = (E_blocks == maximum_filter(E_blocks, size=3, mode="constant")) & (E_blocks >= tau)
    return np.argwhere(peak)


@dataclass
class Detection:
    """Counts for hotspot precision/recall with a spatial tolerance."""

    tp_pred: int  # predicted hotspots that match a true one
    tp_true: int  # true hotspots matched by a prediction
    n_pred: int
    n_true: int


def detect(pred: np.ndarray, true: np.ndarray, tol: int = 1) -> Detection:
    """Match hotspots within ``tol`` blocks (Chebyshev distance)."""
    if len(pred) == 0 or len(true) == 0:
        return Detection(0, 0, len(pred), len(true))
    d = np.abs(pred[:, None, :] - true[None, :, :]).max(-1)
    close = d <= tol
    return Detection(int(close.any(1).sum()), int(close.any(0).sum()), len(pred), len(true))


def window_source_strength(specs: ScenarioSpecs, window: int) -> np.ndarray:
    """Mean emission [g/s] of each tabulated source over the last ``window`` hours, ``(B, MAX_SOURCES)``."""
    lh = specs.local_hours()[:, -window:]
    sched = specs.sources[..., 5].astype(int)
    prof = np.stack([PROFILES[sched[b]][:, lh[b]].mean(axis=1) for b in range(len(specs))])
    return specs.sources[..., 4] * specs.sources[..., 7] * prof / 1e6


def localization_km(source_xy_m: np.ndarray, pred_blocks: np.ndarray, block_m: float) -> np.ndarray:
    """Distance [km] from each true source to the nearest predicted hotspot centre (inf if none)."""
    if len(pred_blocks) == 0:
        return np.full(len(source_xy_m), np.inf)
    centres = (pred_blocks[:, ::-1] + 0.5) * block_m  # (x, y)
    return np.sqrt(((source_xy_m[:, None] - centres[None]) ** 2).sum(-1)).min(1) / 1e3


def rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Root-mean-square difference, optionally over ``mask``."""
    d = (a - b) if mask is None else (a - b)[mask]
    return float(np.sqrt(np.mean(d**2)))
