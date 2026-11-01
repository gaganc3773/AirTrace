"""FNO training (with a physics-consistency loss), deep-ensemble inference and stored-network evaluation."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from airtrace.config import FNOConfig
from airtrace.data.dataset import SyntheticDataset
from airtrace.data.synthetic import last_hour_background
from airtrace.grid import Grid
from airtrace.models.fno import FNO2d
from airtrace.models.fno_data import (BG, C_SLICE, CT, CTM1, N_IN, N_OUT, S_SLICE, GPUData, build_inputs,
                                      rasterize_network, sample_networks)
from airtrace.solver import ADRSolver, Forcing


def make_model(cfg: FNOConfig) -> FNO2d:
    return FNO2d(N_IN, N_OUT, cfg.width, cfg.modes, cfg.layers, cfg.pad)


def activate(raw: torch.Tensor) -> torch.Tensor:
    """Non-negative outputs; the background channel becomes its spatial mean."""
    out = nn.functional.softplus(raw)
    bg = out[:, BG].mean(dim=(-1, -2), keepdim=True).expand_as(out[:, BG])
    return torch.cat([out[:, :BG], bg[:, None], out[:, BG + 1:]], dim=1)


def physics_residual(pred: torch.Tensor, data: GPUData, idx: torch.Tensor, s: torch.Tensor, grid: Grid) -> torch.Tensor:
    """Hour-integrated PDE residual of the predicted fields (scaled units), ``(B, ny, nx)``.

    ``c_T - c_Tm1 - 1800 s [L(c_Tm1) + L(c_T)] - S_tot - 3600 s * regional`` with the
    trapezoid rule over the last hour, the scenario's last-hour wind (faces by averaging
    cell centres) and its K, lambda (privileged information available at training time only).
    """
    met = data.met(idx)
    u, v = met.wind_fields()
    uc, vc = u[:, -1], v[:, -1]
    uf = torch.cat([uc[..., :1], 0.5 * (uc[..., 1:] + uc[..., :-1]), uc[..., -1:]], dim=-1)
    vf = torch.cat([vc[..., :1, :], 0.5 * (vc[..., 1:, :] + vc[..., :-1, :]), vc[..., -1:, :]], dim=-2)
    bg = data.spec["c_bg"][idx]
    K, lam = data.K[idx], data.lam[idx]
    if data.entrainment:
        H = data.spec["mixing_height"][idx]
        lam = lam + (torch.log(H[:, -1] / H[:, -2]).clamp(min=0) / 3600.0)
    bg_mean = last_hour_background(bg)
    regional = (lam * bg_mean + (bg[:, -1] - bg[:, -2]) / 3600.0) / s
    f = Forcing(uf, vf, torch.zeros_like(uc), bg_mean / s)
    solver = ADRSolver(grid, "linear3")
    L = solver.tendency(pred[:, CTM1], f, K, lam) + solver.tendency(pred[:, CT], f, K, lam)
    return (pred[:, CT] - pred[:, CTM1]) - 1800.0 * L - pred[:, S_SLICE].sum(1) - 3600.0 * regional[:, None, None]


def losses(pred: torch.Tensor, y: torch.Tensor, cfg: FNOConfig, phys: torch.Tensor | None) -> dict[str, torch.Tensor]:
    mse = lambda a, b: ((a - b) ** 2).mean()  # noqa: E731
    pool = lambda a: nn.functional.avg_pool2d(a, 4) * 4  # noqa: E731
    out = {
        "field": mse(pred[:, C_SLICE], y[:, C_SLICE]) + mse(pred[:, BG, 0, 0], y[:, BG, 0, 0])
        + mse(pred[:, CT], y[:, CT]) + mse(pred[:, CTM1], y[:, CTM1]),
        "source": mse(pred[:, S_SLICE], y[:, S_SLICE]),
        "block": mse(pool(pred[:, S_SLICE]), pool(y[:, S_SLICE])),
    }
    out["phys"] = (phys**2).mean() if phys is not None else torch.zeros((), device=pred.device)
    out["total"] = out["field"] + cfg.w_source * out["source"] + cfg.w_block * out["block"] + cfg.w_phys * out["phys"]
    return out


def stored_networks(ds: SyntheticDataset, indices: np.ndarray, grid: Grid, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterised default sensor networks for evaluation, ``(B, W, ny, nx)`` values and masks."""
    vm = [rasterize_network(*(lambda d: (d["sensor_cells"], d["sensor_readings"]))(ds.get(int(i), ("sensor_cells", "sensor_readings"))), grid)
          for i in indices]
    return (torch.as_tensor(np.stack([a for a, _ in vm]), device=device),
            torch.as_tensor(np.stack([b for _, b in vm]), device=device))


@torch.no_grad()
def predict(models: list[nn.Module], data: GPUData, idx: torch.Tensor, values: torch.Tensor, mask: torch.Tensor,
            batch: int = 64) -> torch.Tensor:
    """Ensemble prediction in physical units, ``(B, M, 11, ny, nx)``."""
    outs = []
    for k in range(0, len(idx), batch):
        sl = slice(k, k + batch)
        x, s = build_inputs(values[sl], mask[sl], data.met(idx[sl]))
        outs.append(torch.stack([activate(m(x)) for m in models], 1) * s[:, None, None, None, None])
    return torch.cat(outs)


def train_member(data: GPUData, cfg: FNOConfig, grid: Grid, seed: int, val: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                 log=print) -> tuple[nn.Module, list[dict[str, float]]]:
    """Train one ensemble member; ``val`` = (indices, values, masks) with stored networks."""
    dev = data.device
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    model = make_model(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_idx = torch.arange(*cfg.train_range, device=dev)
    steps = cfg.epochs * (len(train_idx) // cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.1)
    history = []
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), generator=gen, device=dev)]
        agg: dict[str, float] = {}
        nb = len(perm) // cfg.batch_size
        for b in range(nb):
            idx = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            values, mask = sample_networks(data.c_hourly[idx].float(), cfg.n_sensors, cfg.noise_range,
                                           cfg.dropout_range, cfg.gap_prob, gen, cfg.bias_range, cfg.noise_rho)
            x, s = build_inputs(values, mask, data.met(idx))
            pred = activate(model(x))
            phys = physics_residual(pred, data, idx, s, grid) if cfg.w_phys > 0 else None
            L = losses(pred, data.targets(idx, s), cfg, phys)
            opt.zero_grad(set_to_none=True)
            L["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in L.items():
                agg[k] = agg.get(k, 0.0) + v.item() / nb
        rec = {"epoch": epoch + 1, **{f"train_{k}": v for k, v in agg.items()}, **evaluate_val(model, data, cfg, grid, val),
               "minutes": (time.perf_counter() - t0) / 60}
        history.append(rec)
        log(f"  epoch {epoch + 1:2d}  train {agg['total']:.4f}  val {rec['val_total']:.4f}  "
            f"val field RMSE {rec['val_field_rmse']:.2f}  ({rec['minutes']:.1f} min)")
    return model, history


@torch.no_grad()
def evaluate_val(model: nn.Module, data: GPUData, cfg: FNOConfig, grid: Grid,
                 val: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    """Validation losses and last-hour field RMSE at unobserved cells [µg/m³]."""
    model.eval()
    idx, values, mask = val
    agg: dict[str, float] = {}
    se, n = 0.0, 0
    for k in range(0, len(idx), 64):
        sl = slice(k, k + 64)
        x, s = build_inputs(values[sl], mask[sl], data.met(idx[sl]))
        pred = activate(model(x))
        y = data.targets(idx[sl], s)
        L = losses(pred, y, cfg, physics_residual(pred, data, idx[sl], s, grid))
        for key, v in L.items():
            agg[f"val_{key}"] = agg.get(f"val_{key}", 0.0) + v.item() * len(idx[sl]) / len(idx)
        c_hat = (pred[:, C_SLICE].sum(1) + pred[:, BG]) * s[:, None, None]
        unseen = mask[sl, -1] == 0
        se += float((((c_hat - data.c_hourly[idx[sl], -1].float()) ** 2) * unseen).sum())
        n += int(unseen.sum())
    agg["val_field_rmse"] = (se / n) ** 0.5
    return agg


def save_member(model: nn.Module, cfg: FNOConfig, path: Path, history: list) -> None:
    torch.save({"state_dict": model.state_dict(), "config": cfg.model_dump(mode="json"), "history": history}, path)


def load_members(paths: list[Path], device="cpu") -> list[nn.Module]:
    models = []
    for p in paths:
        ck = torch.load(p, map_location=device, weights_only=False)
        m = make_model(FNOConfig.model_validate(ck["config"])).to(device)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models.append(m)
    return models
