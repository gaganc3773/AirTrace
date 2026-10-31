"""Source attribution ("who is polluting this spot?") and hidden-emitter flags from ensemble outputs.

Because the transport is linear in the sources, the concentration at a receptor
splits exactly into one part per source category plus the regional background.
The FNO predicts those per-category fields directly, so a share is a ratio of
predicted fields at the receptor; the ensemble spread gives its uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from airtrace.data.layouts import CATEGORIES

SHARE_NAMES = (*CATEGORIES, "regional")
Z90 = 1.6448536269514722  # two-sided 90% normal quantile


def receptor_shares(c_cat: np.ndarray, bg: np.ndarray, iy: int, ix: int) -> np.ndarray:
    """Shares at cell ``(iy, ix)``: ``c_cat (..., 4, ny, nx)``, ``bg (...)`` → ``(..., 5)`` summing to 1.

    Parts are clipped at zero: the exactly linear (unlimited) scheme used to build the
    reference decomposition undershoots by up to tens of µg/m³ at sharp plume edges,
    which would otherwise make shares unbounded where the parts nearly cancel.
    """
    parts = np.clip(np.concatenate([c_cat[..., iy, ix], np.asarray(bg)[..., None]], axis=-1), 0.0, None)
    return parts / np.clip(parts.sum(axis=-1, keepdims=True), 1e-9, None)


@dataclass
class Interval:
    """Ensemble mean and calibrated central interval."""

    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


def ensemble_interval(members: np.ndarray, kappa: float = 1.0, level_z: float = Z90,
                      bounds: tuple[float, float] | None = None) -> Interval:
    """Gaussian interval ``mean ± z κ σ`` over axis 0 (ensemble members), optionally clipped."""
    m, s = members.mean(axis=0), members.std(axis=0, ddof=1)
    lo, hi = m - level_z * kappa * s, m + level_z * kappa * s
    if bounds is not None:
        lo, hi = np.clip(lo, *bounds), np.clip(hi, *bounds)
    return Interval(m, lo, hi)


def calibrate_kappa(errors: np.ndarray, sigmas: np.ndarray, coverage: float = 0.9) -> float:
    """Variance inflation κ so that ``|error| <= z κ σ`` holds for ``coverage`` of held-out cases."""
    z = Z90 if coverage == 0.9 else float(np.sqrt(2) * _erfinv(coverage))
    ratio = np.abs(errors) / np.clip(sigmas, 1e-12, None)
    return float(np.quantile(ratio, coverage) / z)


def _erfinv(y: float) -> float:
    from scipy.special import erfinv

    return float(erfinv(y))


def coverage(errors: np.ndarray, sigmas: np.ndarray, kappa: float, level: float) -> float:
    """Empirical coverage of the central ``level`` Gaussian interval with inflation κ."""
    z = float(np.sqrt(2) * _erfinv(level))
    return float(np.mean(np.abs(errors) <= z * kappa * np.clip(sigmas, 1e-12, None)))


def hidden_emitters(E_members: np.ndarray, prior_score: np.ndarray, tau: float = 15.0, z: float = 2.0,
                    prior_max: float = 0.05) -> np.ndarray:
    """Blocks with confidently positive emission where the prior shows no known source.

    Args:
        E_members: ``(M, nb, nb)`` block emissions [g/s] from each ensemble member.
        prior_score: ``(nb, nb)`` OpenStreetMap prior score in [0, 1].
    Returns:
        ``(k, 2)`` block indices with ``mean - z σ > tau`` and ``prior < prior_max``.
    """
    m, s = E_members.mean(0), E_members.std(0, ddof=1)
    return np.argwhere((m - z * s > tau) & (prior_score < prior_max))
