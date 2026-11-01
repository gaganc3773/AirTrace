"""AirTrace inference service: FNO ensemble, kriging and the forward solver, on CPU, for the Scenario lab.

Loads only the data pack in ``app/data`` and the weights in ``artifacts/fno``; no network, GPU or full
dataset is needed. Started by the Node server; run alone with
``python -m uvicorn app.inference.service:app --port 8765`` from the project root.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy.stats import norm

from airtrace.attribution.contrib import Z90
from airtrace.config import load_config, project_root
from airtrace.data.sensors import log_error, sample_sensor_cells
from airtrace.eval.interpolation import kriging
from airtrace.grid import Grid
from airtrace.models.fno_data import BG, C_SLICE, S_SLICE, Met, build_inputs
from airtrace.models.fno_train import activate, load_members
from airtrace.models.greens import block_sum
from airtrace.models.inversion import prior_block_score
from airtrace.solver import ADRSolver, Forcing, uniform_wind

CATS = ("traffic", "industry", "construction", "burning")


class PredictRequest(BaseModel):
    scenario: int = Field(ge=0)
    n_sensors: int = Field(default=40, ge=3, le=64)
    noise: float = Field(default=0.15, ge=0.0, le=0.6)
    seed: int = 0
    cells: list[tuple[int, int]] | None = None  # explicit sensor cells (iy, ix), overriding n_sensors


class WhatIfRequest(PredictRequest):
    wind_from: float = Field(ge=0.0, le=360.0)
    wind_speed: float = Field(ge=0.3, le=8.0)
    hours: int = Field(default=3, ge=1, le=6)


class Lab:
    """Models and held-out scenarios, loaded once."""

    def __init__(self, root: Path):
        self.cfg = load_config(root / "configs" / "default.yaml")
        self.grid = Grid.from_config(self.cfg.domain)
        data = root / "app" / "data"
        self.meta = json.loads((data / "lab_meta.json").read_text(encoding="utf-8"))
        z = np.load(data / "lab_scenarios.npz")
        self.s = {k: z[k] for k in z.files}
        self.models = load_members(sorted((root / "artifacts" / "fno").glob("member_*.pt")), "cpu")
        torch.set_num_threads(max((os.cpu_count() or 4) // 2, 2))
        self.score_nt = prior_block_score(self.s["priors"], self.meta["block"], road_weight=0.0)
        pri = root / "reports" / "phase7_priority.json"
        pv = json.loads(pri.read_text()) if pri.exists() else {"kappa_E": 1.0, "reliability": []}
        self.kappa_E = pv["kappa_E"]
        # the Gaussian P(E > tau) is overconfident even on synthetic data, so report the hit rate measured for similar P
        self.rel_p = np.array([r["p_mean"] for r in pv["reliability"]])
        self.rel_obs = np.array([r["observed"] for r in pv["reliability"]])

    def _network(self, q: PredictRequest) -> tuple[np.ndarray, np.ndarray]:
        i, g = q.scenario, self.grid
        rng = np.random.default_rng([q.seed, i])
        if q.cells is not None:
            if not q.cells:
                raise HTTPException(422, "Add at least one sensor")
            cells = np.unique(np.array(q.cells, dtype=np.int16), axis=0)
            if (cells < 0).any() or cells[:, 0].max() >= g.ny or cells[:, 1].max() >= g.nx:
                raise HTTPException(422, "Sensor cell outside the grid")
        else:
            cells = sample_sensor_cells(rng, q.n_sensors, g, self.cfg.data.sensors.min_dist_m)
        W = self.s["c_hourly"].shape[1]
        err = log_error(rng, (len(cells), W), q.noise, 0.15 * (q.noise > 0), 0.7)
        readings = self.s["c_hourly"][i][:, cells[:, 0], cells[:, 1]].T * np.exp(err)
        return cells, readings.astype(np.float32)

    @torch.no_grad()
    def _ensemble(self, i: int, cells: np.ndarray, readings: np.ndarray) -> np.ndarray:
        """Physical outputs, ``(M, 11, ny, nx)``."""
        g, s = self.grid, self.s
        W = readings.shape[1]
        values = torch.zeros(1, W, g.ny, g.nx)
        mask = torch.zeros_like(values)
        iy, ix = torch.as_tensor(cells[:, 0].astype(np.int64)), torch.as_tensor(cells[:, 1].astype(np.int64))
        values[0, :, iy, ix] = torch.as_tensor(readings.T)
        mask[0, :, iy, ix] = 1.0
        t = lambda a: torch.as_tensor(a, dtype=torch.float32)[None]  # noqa: E731
        met = Met(t(s["wind_u"][i]), t(s["wind_v"][i]), t(s["pert_amp"][i]), t(s["wind_pat"][i]), t(s["H"][i]),
                  torch.tensor([self.meta["scenarios"][i]["end_hour"]]), t(s["priors"]))
        x, sc = build_inputs(values, mask, met)
        return torch.stack([activate(m(x))[0] * sc for m in self.models]).numpy()

    def predict(self, q: PredictRequest) -> dict:
        self._check(q.scenario)
        i, g, s = q.scenario, self.grid, self.s
        cells, readings = self._network(q)
        t0 = time.perf_counter()
        out = self._ensemble(i, cells, readings)
        fno_ms = (time.perf_counter() - t0) * 1e3
        cc = out[:, C_SLICE]
        bg = out[:, BG, 0, 0]
        c_members = cc.sum(1) + bg[:, None, None]
        cal = self.meta["calibration"]
        t1 = time.perf_counter()
        xy = (cells[:, ::-1] + 0.5) * g.dx
        X, Y = np.meshgrid((np.arange(g.nx) + 0.5) * g.dx, (np.arange(g.ny) + 0.5) * g.dy)
        tgt = np.column_stack([X.ravel(), Y.ravel()])
        krig = (kriging(xy, readings[:, -1], tgt, max(q.noise, 0.05)).reshape(g.ny, g.nx) if len(cells) >= 3
                else np.full((g.ny, g.nx), float(np.mean(readings[:, -1]))))
        krig_ms = (time.perf_counter() - t1) * 1e3
        truth = s["c_hourly"][i, -1]
        unseen = np.ones(truth.shape, bool)
        unseen[cells[:, 0], cells[:, 1]] = False
        rm = lambda a: float(np.sqrt(np.mean((a - truth)[unseen] ** 2)))  # noqa: E731
        block = self.meta["block"]
        g_cell = g.cell_area * float(s["H"][i, -1]) / 3600.0 / 1e6
        E_cat = block_sum(out[:, S_SLICE] * g_cell, block)  # (M, 4, nb, nb)
        E_nt = E_cat[:, 1:].sum(1)
        return {
            "scenario": self.meta["scenarios"][i], "cells": cells.tolist(), "n": len(cells),
            "truth": _r(truth), "fno": _r(c_members.mean(0)), "fno_sigma": _r(cal["kappa_field"] * c_members.std(0, ddof=1)),
            "kriging": _r(krig), "nnls": _r(s["nnls_c"][i]),
            "c_cat": _r(cc.mean(0)), "c_cat_std": _r(cal["kappa_share"] * cc.std(0, ddof=1)), "bg": round(float(bg.mean()), 1),
            "bg_std": round(float(cal["kappa_share"] * bg.std(ddof=1)), 1), "truth_c_cat": _r(s["c_cat"][i]), "truth_bg": round(float(s["bg_last"][i]), 1),
            "E_truth": _r(s["E_blocks"][i]), "E_fno": _r(E_cat.mean(0).sum(0)), "E_nnls": _r(s["nnls_E"][i]),
            "rmse": {"fno": round(rm(c_members.mean(0)), 1), "kriging": round(rm(krig), 1), "nnls": round(rm(s["nnls_c"][i]), 1)},
            "ms": {"fno": round(fno_ms), "kriging": round(krig_ms)}, "priority": self._priority(E_cat, E_nt),
        }

    def _priority(self, E_cat: np.ndarray, E_nt: np.ndarray) -> list[dict]:
        tau = self.meta["tau_g_s"]
        m, sd = E_nt.mean(0), E_nt.std(0, ddof=1)
        P = norm.sf((tau - m) / np.maximum(self.kappa_E * sd, 1e-6))
        nb = m.shape[0]
        inner = np.zeros((nb, nb), bool)
        inner[1:-1, 1:-1] = True  # edge blocks absorb inflow and background error, so they are never ranked
        rows = []
        for j in np.argsort(-(m * inner).ravel())[:8]:
            by, bx = divmod(int(j), nb)
            if m[by, bx] <= tau or not inner[by, bx]:
                break
            mapped = bool(self.score_nt[by, bx] >= 0.05)
            cat = CATS[1 + int(np.argmax(E_cat[:, 1:, by, bx].mean(0)))]
            rows.append({"by": by, "bx": bx, "cat": cat if mapped else "unmapped", "likely": cat, "g_s": round(float(m[by, bx]), 1),
                         "sigma": round(float(self.kappa_E * sd[by, bx] * Z90), 1),
                         "p": round(float(np.interp(P[by, bx], self.rel_p, self.rel_obs)), 2) if len(self.rel_p) else None})
        return rows

    def whatif(self, q: WhatIfRequest) -> dict:
        """Forward-run the FNO's inferred sources under a different wind (physics solver, not the network)."""
        self._check(q.scenario)
        i, g, s = q.scenario, self.grid, self.s
        cells, readings = self._network(q)
        t0 = time.perf_counter()
        out = self._ensemble(i, cells, readings).mean(0)
        S = torch.as_tensor(out[S_SLICE].sum(0) / 3600.0, dtype=torch.float64)[None]
        bg = float(out[BG, 0, 0])
        c0 = torch.as_tensor(out[C_SLICE].sum(0) + bg, dtype=torch.float64)[None]
        u, v = uniform_wind(g, q.wind_speed, q.wind_from, dtype=torch.float64)
        K, lam = float(s["K"][i]), float(s["lam"][i])
        f = Forcing(u, v, S + lam * bg, torch.tensor([bg], dtype=torch.float64))
        n = q.hours * 2
        _, snaps = ADRSolver(g, self.cfg.solver.limiter).run(c0, lambda t: f, K, lam, n, 1800.0)
        return {"frames": [_r(a) for a in snaps[0].numpy()], "minutes": [30 * k for k in range(n + 1)], "bg": round(bg, 1),
                "ms": round((time.perf_counter() - t0) * 1e3), "wind_from": q.wind_from, "wind_speed": q.wind_speed}

    def _check(self, i: int) -> None:
        if i >= len(self.meta["scenarios"]):
            raise HTTPException(404, "No such scenario")


def _r(a: np.ndarray) -> list:
    return np.round(np.asarray(a, np.float64), 1).tolist()


_lab: Lab | None = None


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    lab()  # load the models before reporting healthy
    yield


app = FastAPI(title="AirTrace inference", lifespan=_lifespan)


def lab() -> Lab:
    global _lab
    if _lab is None:
        _lab = Lab(project_root())
    return _lab


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "members": len(lab().models), "scenarios": len(lab().meta["scenarios"])}


@app.get("/lab/scenarios")
def scenarios() -> dict:
    return lab().meta


@app.post("/lab/predict")
def predict_route(q: PredictRequest) -> dict:
    return lab().predict(q)


@app.post("/lab/whatif")
def whatif_route(q: WhatIfRequest) -> dict:
    return lab().whatif(q)
