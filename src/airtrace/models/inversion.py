"""Linear (adjoint-style) inversion: non-negative, L1 + Tikhonov regularised least squares.

Model: ``y[s, k] = sum_j G[s, k, j] E_j + b_k + noise`` with block emissions
``E >= 0`` [g/s] held constant in time and a uniform background ``b_k >= 0``
per window hour. Objective (weighted by the sensor error model)::

    0.5 ||W (A x - y)||² + 0.5 alpha ||E||² + beta * sum_j w_j E_j

The L1 weight ``beta`` is chosen by K-fold cross-validation over sensors
(predict held-out sensors' readings). Morozov's discrepancy principle on the
mean chi² is available but over-regularises here: a plume touches only a few
of ~240 readings, so explaining it barely moves the mean chi². Optional per-block
weights ``w_j`` encode the OpenStreetMap prior (cheaper to emit where a road,
factory or landfill is known).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import nnls


@dataclass
class InversionResult:
    """Estimated block emissions [g/s], hourly background [µg/m³] and diagnostics."""

    E: np.ndarray
    bg: np.ndarray
    beta: float
    chi2: float
    n_obs: int
    seconds: float
    cv_curve: np.ndarray | None = None  # (n_beta, 2): beta, held-out loss


def prior_block_score(layout: np.ndarray, block: int) -> np.ndarray:
    """Per-block prior score in [0, 1] from a ``(4, ny, nx)`` layout (roads count half)."""
    *_, ny, nx = layout.shape
    mean = layout.reshape(4, ny // block, block, nx // block, block).mean(axis=(2, 4))
    return np.clip(0.5 * mean[0] + mean[1:].max(axis=0) * 2, 0, 1)


def prior_weights(prior_blocks: np.ndarray, floor: float = 0.25) -> np.ndarray:
    """Per-block L1 weights from a prior score in [0, 1]: ``1/(floor + p)``, normalised to mean 1."""
    w = 1.0 / (floor + np.clip(prior_blocks.ravel(), 0.0, 1.0))
    return w / w.mean()


def invert(
    G: np.ndarray,
    y: np.ndarray,
    sigma_rel: float = 0.2,
    sigma_abs: float = 2.0,
    weights: np.ndarray | None = None,
    alpha_rel: float = 1e-6,
    select: str = "cv",
    chi2_target: float = 1.0,
    n_beta: int = 16,
    n_folds: int = 5,
) -> InversionResult:
    """Solve for block emissions and background from sensor readings.

    Args:
        G: response ``(n_sensors, W, J)`` [µg/m³ per g/s].
        y: readings ``(n_sensors, W)``, NaN where missing.
        sigma_rel, sigma_abs: observation error model ``sigma = sigma_rel * y + sigma_abs``
            (sensor noise plus representation/model error).
        weights: per-block L1 weights ``(J,)`` (default uniform).
        select: ``"cv"`` (sensor K-fold) or ``"discrepancy"`` for choosing ``beta``.
    """
    t0 = time.perf_counter()
    n, W, J = G.shape
    ok = np.isfinite(y)
    A = np.concatenate([G[ok], np.repeat(np.eye(W)[None], n, axis=0)[ok]], axis=1)  # (m, J + W)
    yo = y[ok]
    sensor_of = np.repeat(np.arange(n)[:, None], W, axis=1)[ok]
    sig = sigma_rel * yo + sigma_abs
    Aw, yw = A / sig[:, None], yo / sig
    m = len(yo)
    w = np.ones(J) if weights is None else weights
    alpha = alpha_rel * float(np.mean(np.sum(Aw[:, :J] ** 2, axis=0)))
    bg0 = np.full(W, np.nanmedian(y))
    beta_max = float(np.max((Aw[:, :J].T @ (yw - Aw[:, J:] @ bg0)) / w))
    betas = beta_max * np.logspace(0, -5, n_beta)

    def path(rows: np.ndarray) -> list[np.ndarray]:
        """Exact solutions along the beta path using only ``rows``.

        With x >= 0 the L1 term is linear, so each problem is the QP
        ``min 0.5 x'Qx - (b - beta w)'x``; with ``Q = L L'`` it equals the NNLS
        problem ``||L' x - L^{-1}(b - beta w)||²``, solved exactly by active set.
        """
        Ar, yr = Aw[rows], yw[rows]
        Q, b = Ar.T @ Ar, Ar.T @ yr
        Q[:J, :J] += alpha * np.eye(J)
        Q[J:, J:] += 1e-9 * np.trace(Q) / (J + W) * np.eye(W)
        L = cholesky(Q, lower=True)
        return [nnls(L.T, solve_triangular(L, b - np.concatenate([beta * w, np.zeros(W)]), lower=True),
                     maxiter=50 * (J + W))[0] for beta in betas]

    full = path(np.ones(m, dtype=bool))
    chi2s = [float(np.sum((Aw @ x - yw) ** 2) / m) for x in full]
    cv_curve = None
    if select == "cv":
        fold = np.random.default_rng(0).permutation(n)[sensor_of] % n_folds
        loss = np.zeros(n_beta)
        for f in range(n_folds):
            held = fold == f
            for k, x in enumerate(path(~held)):
                loss[k] += float(np.sum((Aw[held] @ x - yw[held]) ** 2))
        k = int(np.argmin(loss))
        cv_curve = np.column_stack([betas, loss / m])
    else:
        hit = [k for k, c in enumerate(chi2s) if c <= chi2_target]
        k = hit[0] if hit else n_beta - 1
    x = full[k]
    return InversionResult(x[:J], x[J:], float(betas[k]), chi2s[k], m, time.perf_counter() - t0, cv_curve)
