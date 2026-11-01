"""Tests for the app's inference service, on the real weights and data pack (skipped when those aren't built)."""

from __future__ import annotations

import numpy as np
import pytest

from airtrace.config import project_root

root = project_root()
if not (root / "app" / "data" / "lab_scenarios.npz").exists() or not (root / "artifacts" / "fno" / "member_0.pt").exists():
    pytest.skip("data pack or weights not built", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from app.inference.service import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_predict_returns_fields_and_beats_nothing_silently(client) -> None:
    r = client.post("/lab/predict", json={"scenario": 0, "n_sensors": 20, "noise": 0.1}).json()
    assert r["n"] == 20
    for k in ("truth", "fno", "kriging", "nnls", "fno_sigma"):
        assert np.array(r[k]).shape == (64, 64)
    assert np.array(r["c_cat"]).shape == (4, 64, 64) and np.array(r["E_fno"]).shape == (16, 16)
    assert 0 < r["rmse"]["fno"] < 200 and min(np.min(r["fno_sigma"]), np.min(r["c_cat"])) >= 0


def test_same_seed_is_reproducible_and_noise_changes_the_answer(client) -> None:
    q = {"scenario": 1, "n_sensors": 15, "noise": 0.2, "seed": 5}
    a, b = client.post("/lab/predict", json=q).json(), client.post("/lab/predict", json=q).json()
    assert a["fno"] == b["fno"] and a["cells"] == b["cells"]
    assert client.post("/lab/predict", json={**q, "seed": 6}).json()["cells"] != a["cells"]


def test_custom_sensor_cells_and_bad_input(client) -> None:
    r = client.post("/lab/predict", json={"scenario": 0, "cells": [[10, 10], [20, 30], [40, 50]]}).json()
    assert sorted(map(tuple, r["cells"])) == [(10, 10), (20, 30), (40, 50)]
    assert client.post("/lab/predict", json={"scenario": 0, "cells": []}).status_code == 422
    assert client.post("/lab/predict", json={"scenario": 0, "cells": [[99, 1]]}).status_code == 422
    assert client.post("/lab/predict", json={"scenario": 999}).status_code == 404
    assert client.post("/lab/predict", json={"scenario": 0, "noise": 5}).status_code == 422


def test_priority_excludes_domain_edge(client) -> None:
    rows = client.post("/lab/predict", json={"scenario": 2}).json()["priority"]
    assert all(0 < r["by"] < 15 and 0 < r["bx"] < 15 for r in rows)
    assert [r["g_s"] for r in rows] == sorted((r["g_s"] for r in rows), reverse=True)


def test_whatif_plume_moves_downwind(client) -> None:
    """Under an easterly the excess must shift west of where a westerly puts it."""
    q = {"scenario": 0, "wind_speed": 6.0, "hours": 3}
    east = np.array(client.post("/lab/whatif", json={**q, "wind_from": 90.0}).json()["frames"][-1])
    west = np.array(client.post("/lab/whatif", json={**q, "wind_from": 270.0}).json()["frames"][-1])
    assert east[:, :32].mean() > west[:, :32].mean() and west[:, 32:].mean() > east[:, 32:].mean()
