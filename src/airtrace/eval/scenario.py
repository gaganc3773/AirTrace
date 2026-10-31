"""Per-scenario ground truth and the common metric bundle every model is scored with."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from airtrace.data.dataset import SyntheticDataset
from airtrace.data.synthetic import ForcingBuilder, ScenarioSpecs
from airtrace.eval.metrics import detect, hotspots, localization_km, rmse, window_source_strength
from airtrace.grid import Grid
from airtrace.models.greens import block_sum

RECEPTOR_KM = (8.0, 24.0, 40.0)  # 3 x 3 receptor lattice for local-share errors


@dataclass
class Truth:
    """Ground truth for one held-out scenario."""

    index: int
    specs: ScenarioSpecs
    E_blocks: np.ndarray  # (nb, nb) window-mean emission [g/s]
    c_last: np.ndarray  # (ny, nx) last-hour mean concentration
    bg_last: float
    source_xy: np.ndarray  # (k, 2) strong discrete sources [m]
    cells: np.ndarray  # (n, 2) sensor cells
    readings: np.ndarray  # (n, W) noisy readings, NaN gaps
    noise: float


def load_truth(ds: SyntheticDataset, i: int, grid: Grid, block: int, window: int, src_min_g_s: float) -> Truth:
    """Assemble ground truth for scenario ``i`` (re-deriving emissions from its spec)."""
    d = ds[i]
    specs = ds.specs([i])
    fb = ForcingBuilder(specs, ds.layouts, grid)
    T = specs.wind_u.shape[1]
    emis = sum(fb.emission_by_category(h).sum(dim=1)[0].numpy() for h in range(T - window, T)) / window
    strength = window_source_strength(specs, window)[0]
    strong = strength >= src_min_g_s
    return Truth(
        index=i, specs=specs, E_blocks=block_sum(emis, block) / 1e6, c_last=d["c_hourly"][-1].astype(np.float64),
        bg_last=float(specs.c_bg[0, -1]), source_xy=specs.sources[0, strong, 1:3], cells=d["sensor_cells"],
        readings=d["sensor_readings"], noise=float(d["sensor_noise"]),
    )


def receptor_cells(grid: Grid) -> np.ndarray:
    """``(9, 2)`` receptor cells ``(iy, ix)``."""
    idx = [int(km * 1e3 / grid.dx) for km in RECEPTOR_KM]
    return np.array([(j, i) for j in idx for i in idx])


def score(
    truth: Truth,
    grid: Grid,
    block: int,
    tau: float,
    E_hat: np.ndarray | None = None,
    c_hat: np.ndarray | None = None,
    bg_hat: float | None = None,
) -> dict[str, float]:
    """Metric bundle; source metrics need ``E_hat`` (nb, nb), field metrics need ``c_hat``,
    local-share error additionally needs ``bg_hat``."""
    out: dict[str, float] = {}
    if E_hat is not None:
        true_hot, pred_hot = hotspots(truth.E_blocks, tau), hotspots(E_hat, tau)
        det = detect(pred_hot, true_hot, tol=1)
        out.update(tp_pred=det.tp_pred, tp_true=det.tp_true, n_pred=det.n_pred, n_true=det.n_true)
        loc = localization_km(truth.source_xy, pred_hot, block * grid.dx)
        out["loc_km_median"] = float(np.median(loc)) if len(loc) else np.nan
        out["total_emission_rel_err"] = float(E_hat.sum() / max(truth.E_blocks.sum(), 1e-9) - 1)
    if c_hat is not None:
        seen = np.zeros((grid.ny, grid.nx), dtype=bool)
        online = np.isfinite(truth.readings[:, -1])
        seen[truth.cells[online, 0], truth.cells[online, 1]] = True
        out["field_rmse"] = rmse(c_hat, truth.c_last, ~seen)
        out["field_rel_rmse"] = out["field_rmse"] / float(truth.c_last[~seen].mean())
        if bg_hat is not None:
            r = receptor_cells(grid)
            c_t, c_e = truth.c_last[r[:, 0], r[:, 1]], c_hat[r[:, 0], r[:, 1]]
            share_t = 1 - truth.bg_last / c_t
            share_e = 1 - bg_hat / np.maximum(c_e, 1e-6)
            out["local_share_err_pp"] = float(np.mean(np.abs(share_e - share_t)) * 100)
    return out


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    """Pool detection counts into precision/recall; average the other metrics."""
    s: dict[str, float] = {}
    if rows and "n_pred" in rows[0]:
        tp_p, tp_t = sum(r["tp_pred"] for r in rows), sum(r["tp_true"] for r in rows)
        n_p, n_t = sum(r["n_pred"] for r in rows), sum(r["n_true"] for r in rows)
        s["precision"] = tp_p / n_p if n_p else np.nan
        s["recall"] = tp_t / n_t if n_t else np.nan
        s["loc_km_median"] = float(np.nanmedian([r["loc_km_median"] for r in rows]))
        s["total_emission_rel_err"] = float(np.median(np.abs([r["total_emission_rel_err"] for r in rows])))
    for k in ("field_rmse", "field_rel_rmse", "local_share_err_pp", "seconds"):
        if rows and k in rows[0]:
            s[k] = float(np.mean([r[k] for r in rows]))
    return s
