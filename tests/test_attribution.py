"""Tests for attribution shares, ensemble intervals, calibration and hidden-emitter flags."""

from __future__ import annotations

import numpy as np

from airtrace.attribution.contrib import (calibrate_kappa, coverage, ensemble_interval, hidden_emitters,
                                          receptor_shares)


def test_shares_sum_to_one_and_match_parts() -> None:
    rng = np.random.default_rng(0)
    c_cat = rng.uniform(0, 50, (5, 4, 8, 8))
    bg = rng.uniform(20, 80, 5)
    sh = receptor_shares(c_cat, bg, 3, 4)
    assert sh.shape == (5, 5) and np.allclose(sh.sum(-1), 1.0)
    assert np.allclose(sh[:, 4], bg / (c_cat[:, :, 3, 4].sum(-1) + bg))


def test_calibration_restores_nominal_coverage() -> None:
    rng = np.random.default_rng(1)
    sigma = rng.uniform(0.5, 2.0, 20000)
    err = rng.normal(0, 2.5 * sigma)  # ensemble 2.5x over-confident
    k = calibrate_kappa(err, sigma, 0.9)
    assert abs(k - 2.5) < 0.1
    assert coverage(err, sigma, 1.0, 0.9) < 0.6 and abs(coverage(err, sigma, k, 0.9) - 0.9) < 0.01


def test_ensemble_interval_bounds() -> None:
    members = np.array([[0.1, 0.9], [0.3, 0.95], [0.2, 0.99]])
    iv = ensemble_interval(members, kappa=3.0, bounds=(0.0, 1.0))
    assert np.all(iv.lo >= 0) and np.all(iv.hi <= 1) and np.allclose(iv.mean, members.mean(0))


def test_hidden_emitters_need_confidence_and_no_prior() -> None:
    E = np.zeros((5, 4, 4))
    E[:, 1, 1] = [40, 42, 38, 41, 39]  # confident, no prior -> flagged
    E[:, 2, 2] = [40, 42, 38, 41, 39]  # confident but prior says industry -> not flagged
    E[:, 3, 3] = [0, 80, 0, 90, 0]  # uncertain -> not flagged
    prior = np.zeros((4, 4))
    prior[2, 2] = 0.8
    assert hidden_emitters(E, prior).tolist() == [[1, 1]]
