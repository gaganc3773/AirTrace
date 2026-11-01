"""Phase 5 evaluation: FNO ensemble vs the Phase 3 baselines, calibration, sensitivity curves, timing.

Usage:  python scripts/phase5_evaluate.py
Writes reports/phase5_metrics.md, reports/phase5_metrics.json, artifacts/fno/calibration.json
and the figures reports/figures/phase5_*.png.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from airtrace.attribution.contrib import (SHARE_NAMES, Z90, calibrate_kappa, coverage, hidden_emitters,
                                          receptor_shares)
from airtrace.config import load_config
from airtrace.data.dataset import SyntheticDataset
from airtrace.data.synthetic import last_hour_background
from airtrace.eval.interpolation import idw, kriging
from airtrace.eval.metrics import detect, window_source_strength
from airtrace.eval.plotting import FIELD_CMAP, PALETTE, savefig, set_style
from airtrace.eval.scenario import load_truth, receptor_cells, score, summarize
from airtrace.grid import Grid
from airtrace.models.fno_data import BG, C_SLICE, S_SLICE, GPUData, Met, rasterize_network
from airtrace.models.fno_train import load_members, predict, stored_networks
from airtrace.models.greens import block_sum
from airtrace.models.inversion import invert, prior_block_score, prior_weights

BLOCK, TAU, SRC_MIN = 4, 15.0, 10.0
N_SENSORS = (5, 10, 20, 40)
NOISE = (0.05, 0.10, 0.20, 0.30)
HIDDEN_Z = (0.0, 1.0, 2.0)
TAG = ""  # output suffix, e.g. "_synth_v2"
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])


class Pred:
    """Ensemble outputs for a set of scenarios (CPU float32, physical units)."""

    def __init__(self, raw: torch.Tensor, H_last: np.ndarray, grid: Grid):
        r = raw.float().cpu().numpy()  # (N, M, 11, ny, nx)
        self.S = r[:, :, S_SLICE]  # µg m^-3 h^-1 per category
        self.c_cat = r[:, :, C_SLICE]
        self.bg = r[:, :, BG, 0, 0]
        self.c = self.c_cat.sum(2) + self.bg[..., None, None]  # (N, M, ny, nx)
        cell_g_s = grid.cell_area * H_last[:, None, None, None, None] / 3600.0 / 1e6  # S -> g/s per cell
        self.E_blocks = block_sum((self.S * cell_g_s).sum(2), BLOCK)  # (N, M, nb, nb)
        self.E_blocks_nt = block_sum((self.S[:, :, 1:] * cell_g_s).sum(2), BLOCK)  # non-traffic only


def run_ensemble(models, data, ds, idx, grid, dev, values=None, masks=None) -> Pred:
    idx_t = torch.as_tensor(idx, device=dev)
    if values is None:
        values, masks = stored_networks(ds, idx, grid, dev)
    raws = [predict(models, data, idx_t[k:k + 100], values[k:k + 100], masks[k:k + 100]).half().cpu()
            for k in range(0, len(idx), 100)]
    H_last = np.array([ds.get(int(i), ("mixing_height",))["mixing_height"][-1] for i in idx])
    return Pred(torch.cat(raws), H_last, grid)


def truth_shares(ds, i, receptors) -> np.ndarray:
    d = ds.get(int(i), ("c_cat", "c_bg"))
    c_cat = d["c_cat"].astype(np.float64)
    return np.stack([receptor_shares(c_cat, last_hour_background(d["c_bg"]), iy, ix) for iy, ix in receptors])  # (9, 5)


def unseen_mask(masks: torch.Tensor) -> np.ndarray:
    return (masks[:, -1] == 0).cpu().numpy()


def calibrate(models, data, ds, grid, dev, fc, receptors) -> dict:
    """Variance-inflation factors for field and share intervals, fitted on the validation split."""
    idx = np.arange(*fc.val_range)
    vals, masks = stored_networks(ds, idx, grid, dev)
    p = run_ensemble(models, data, ds, idx, grid, dev, vals, masks)
    truth_c = data.c_hourly[torch.as_tensor(idx, device=dev), -1].float().cpu().numpy()
    um = unseen_mask(masks)
    err_c, sig_c = (p.c.mean(1) - truth_c)[um], p.c.std(1, ddof=1)[um]
    sh_m = np.stack([[receptor_shares(p.c_cat[n], p.bg[n], iy, ix) for iy, ix in receptors] for n in range(len(idx))])
    sh_t = np.stack([truth_shares(ds, i, receptors) for i in idx])  # (N, 9, 5)
    err_s, sig_s = sh_m.mean(2) - sh_t, sh_m.std(2, ddof=1)
    return {"kappa_field": calibrate_kappa(err_c, sig_c), "kappa_share": calibrate_kappa(err_s.ravel(), sig_s.ravel())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--quick", action="store_true", help="debug run: 10 scenarios in the sensitivity curves")
    parser.add_argument("--no-baselines", action="store_true",
                        help="skip the Phase 3 comparisons (they were computed on synth_v1 scenarios)")
    parser.add_argument("--tag", default="", help="suffix for report and figure names")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    cfg = load_config(args.config)
    if cfg.physics.entrainment and not args.no_baselines:
        parser.error("the Phase 3 baselines were computed on synth_v1: pass --no-baselines, or use configs/synth_v1.yaml "
                     "with the weights in artifacts/fno_synth_v1")
    global TAG
    TAG = args.tag
    fc = cfg.fno
    set_style()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid = Grid.from_config(cfg.domain)
    ds = SyntheticDataset(cfg.path("artifacts") / "datasets" / cfg.data.name)
    data = GPUData(ds, cfg.data.window_hours, dev)
    fno_dir = cfg.path("artifacts") / "fno"
    paths = sorted(fno_dir.glob("member_*.pt"))
    models = load_members(paths, dev)
    report = cfg.path("artifacts").parent / "reports"
    out = cfg.path("figures")
    receptors = receptor_cells(grid)
    phase3 = None if args.no_baselines else json.loads((report / "phase3_metrics.json").read_text())
    print(f"{len(models)} members")

    # ---- calibration on the validation split
    cal = calibrate(models, data, ds, grid, dev, fc, receptors)
    (fno_dir / "calibration.json").write_text(json.dumps(cal, indent=2))
    print("calibration", cal)

    # ---- test-set predictions with the stored networks
    test = np.arange(fc.test_range[0], min(fc.test_range[1], fc.test_range[0] + args.n_test))
    vals, masks = stored_networks(ds, test, grid, dev)
    t0 = time.perf_counter()
    p = run_ensemble(models, data, ds, test, grid, dev, vals, masks)
    gpu_s = (time.perf_counter() - t0) / len(test)
    um = unseen_mask(masks)
    truth_c = data.c_hourly[torch.as_tensor(test, device=dev), -1].float().cpu().numpy()

    rows, share_err, cov_rows = {}, [], {"field": [], "share": []}
    hid = {z: {"tp_pred": 0, "tp_true": 0, "n_pred": 0, "n_true": 0} for z in HIDDEN_Z}
    for n, i in enumerate(test):
        tr = load_truth(ds, int(i), grid, BLOCK, cfg.data.window_hours, SRC_MIN)
        rows[int(i)] = score(tr, grid, BLOCK, TAU, p.E_blocks[n].mean(0), p.c[n].mean(0), float(p.bg[n].mean()))
        sh_m = np.stack([receptor_shares(p.c_cat[n], p.bg[n], iy, ix) for iy, ix in receptors])  # (9, M, 5)
        sh_t = truth_shares(ds, i, receptors)
        share_err.append(np.abs(sh_m.mean(1) - sh_t) * 100)
        cov_rows["share"].append((sh_m.mean(1) - sh_t, sh_m.std(1, ddof=1)))
        cov_rows["field"].append(((p.c[n].mean(0) - truth_c[n])[um[n]], p.c[n].std(0, ddof=1)[um[n]]))
        # hidden emitters: strong off-prior sources vs confident off-prior detections
        strength = window_source_strength(tr.specs, cfg.data.window_hours)[0]
        src = tr.specs.sources[0]
        off = (src[:, 6] > 0) & (strength >= TAU)
        true_blocks = np.unique((src[off][:, [2, 1]] // (BLOCK * grid.dx)).astype(int), axis=0)
        # "no known source" = no industry / construction / landfill prior (roads are everywhere in Delhi)
        score_nt = prior_block_score(ds.layouts[tr.specs.layout_id[0]], BLOCK, road_weight=0.0)
        for z in HIDDEN_Z:
            d = detect(hidden_emitters(p.E_blocks_nt[n], score_nt, TAU, z), true_blocks.reshape(-1, 2), tol=1)
            for k in hid[z]:
                hid[z][k] += getattr(d, k)
    share_err = np.stack(share_err)  # (N, 9, 5)

    def cov(kind: str, kappa: float, level: float) -> float:
        e = np.concatenate([a.ravel() for a, _ in cov_rows[kind]])
        s = np.concatenate([b.ravel() for _, b in cov_rows[kind]])
        return coverage(e, s, kappa, level)

    calib = {kind: {"raw": [cov(kind, 1.0, lv) for lv in LEVELS],
                    "calibrated": [cov(kind, cal[f"kappa_{kind}"], lv) for lv in LEVELS]} for kind in ("field", "share")}

    # ---- comparison on the Phase 3 subsets
    table = {}
    for subset in () if phase3 is None else ("easy", "general"):
        idx = [i for i in phase3[subset] if i in rows]
        table[f"{subset}|FNO ensemble"] = summarize([rows[i] for i in idx])
        for m in ("NNLS-L1", "NNLS-L1 + OSM prior", "IDW", "Kriging"):
            table[f"{subset}|{m}"] = phase3["table"][f"{'all' if subset == 'general' else 'easy'}|{m}"]
    table["test (n=%d)|FNO ensemble" % len(test)] = summarize(list(rows.values()))

    # ---- sensitivity curves on the Phase 3 "general" scenarios (all methods on identical inputs)
    curves, timing = {}, {}
    if phase3 is not None:
        gen = phase3["general"][:10] if args.quick else phase3["general"]
        curves = sensitivity(models, data, ds, gen, grid, dev, cfg)
        # ---- inference time
        timing = time_methods(models, data, ds, gen[0], grid, cfg, report, gpu_s, phase3)

    hidden = {str(z): {"precision": h["tp_pred"] / max(h["n_pred"], 1), "recall": h["tp_true"] / max(h["n_true"], 1),
                       "n_true": h["n_true"], "n_flagged": h["n_pred"]} for z, h in hid.items()}
    summary = {"table": table, "calibration": cal, "coverage": calib, "levels": LEVELS.tolist(),
               "share_err_pp": {n: float(share_err[..., k].mean()) for k, n in enumerate(SHARE_NAMES)},
               "share_err_pp_mean": float(share_err.mean()), "hidden_emitters": hidden, "curves": curves, "timing": timing}
    (report / f"phase5_metrics{TAG}.json").write_text(json.dumps(summary, indent=2))
    write_markdown(summary, report, len(test))
    figures(summary, p, test, ds, data, grid, out, dev, masks, receptors, models)


def sensitivity(models, data, ds, idx, grid, dev, cfg) -> dict:
    """Field RMSE / recall / localisation vs sensor count and noise for FNO, NNLS + prior, kriging."""
    rng = np.random.default_rng(cfg.seed)
    res: dict[str, dict[str, list]] = {}
    configs = [("n", n, None) for n in N_SENSORS] + [("noise", None, s) for s in NOISE]
    for kind, n_s, noise in configs:
        per = {"FNO ensemble": [], "NNLS-L1 + OSM prior": [], "Kriging": []}
        nets = []
        for i in idx:
            d = ds.get(int(i), ("sensor_cells", "sensor_readings", "c_hourly"))
            cells, r = d["sensor_cells"], d["sensor_readings"].copy()
            if kind == "n":
                cells, r = cells[:n_s], r[:n_s]
            else:
                truth = d["c_hourly"][:, cells[:, 0], cells[:, 1]].T.astype(np.float64)
                r = truth * np.exp(rng.normal(0, noise, truth.shape) - noise**2 / 2)
            nets.append((cells, r))
        vm = [rasterize_network(c, r, grid) for c, r in nets]
        v = torch.as_tensor(np.stack([a for a, _ in vm]), device=dev)
        m = torch.as_tensor(np.stack([b for _, b in vm]), device=dev)
        p = run_ensemble(models, data, ds, np.array(idx), grid, dev, v, m)
        for k, i in enumerate(idx):
            tr = load_truth(ds, int(i), grid, BLOCK, cfg.data.window_hours, SRC_MIN)
            cells, r = nets[k]
            tr.cells, tr.readings = cells, r
            per["FNO ensemble"].append(score(tr, grid, BLOCK, TAU, p.E_blocks[k].mean(0), p.c[k].mean(0)))
            z = np.load(cfg.path("artifacts") / "greens" / f"G_{int(i):05d}.npz")
            G = z["sensors"][: len(cells)]
            w = prior_weights(prior_block_score(ds.layouts[tr.specs.layout_id[0]], BLOCK))
            inv = invert(G, r, sigma_rel=noise if noise else tr.noise, sigma_abs=1.0, weights=w)
            nb = grid.ny // BLOCK
            per["NNLS-L1 + OSM prior"].append(score(tr, grid, BLOCK, TAU, inv.E.reshape(nb, nb),
                                                    np.einsum("jyx,j->yx", z["field_last"], inv.E) + inv.bg[-1]))
            ok = np.isfinite(r[:, -1])
            pts = (cells[ok][:, ::-1] + 0.5) * grid.dx
            X, Y = np.meshgrid((np.arange(grid.nx) + 0.5) * grid.dx, (np.arange(grid.ny) + 0.5) * grid.dy)
            kr = kriging(pts, r[ok, -1], np.column_stack([X.ravel(), Y.ravel()]), noise or tr.noise)
            per["Kriging"].append(score(tr, grid, BLOCK, TAU, c_hat=kr.reshape(grid.ny, grid.nx)))
        for meth, rws in per.items():
            s = summarize(rws)
            r = res.setdefault(meth, {})
            for key in ("field_rmse", "recall", "precision", "loc_km_median"):
                if key in s:
                    r.setdefault(f"{kind}_{key}", []).append(s[key])
        print(f"  sensitivity {kind}={n_s or noise}: " + ", ".join(
            f"{m} {res[m][f'{kind}_field_rmse'][-1]:.1f}" for m in per))
    return {"n_sensors": list(N_SENSORS), "noise": list(NOISE), **res}


def time_methods(models, data, ds, i, grid, cfg, report, gpu_s, phase3) -> dict:
    """Wall-clock per scenario. FNO timed on CPU (the demo laptop setting) and GPU."""
    cpu_models = [m.to("cpu") for m in models]  # moved back to the data device afterwards
    d = ds.get(int(i), ("sensor_cells", "sensor_readings"))
    v, m = rasterize_network(d["sensor_cells"], d["sensor_readings"], grid)
    sub = GPUDataSlice(data, i)
    x_v, x_m = torch.as_tensor(v)[None], torch.as_tensor(m)[None]
    idx = torch.tensor([0])

    def fno_cpu(ms):
        return predict(ms, sub, idx, x_v, x_m)

    for _ in range(3):
        fno_cpu(cpu_models[:1])
    t_one = np.median([_t(lambda: fno_cpu(cpu_models[:1])) for _ in range(20)])
    t_ens = np.median([_t(lambda: fno_cpu(cpu_models)) for _ in range(10)])
    for mm in models:
        mm.to(data.device)
    pinn = json.loads((report / "phase4_pinn.json").read_text())
    nnls_solve = phase3["table"]["all|NNLS-L1 + OSM prior"]["seconds"]
    kr = phase3["table"]["all|Kriging"]["seconds"]
    return {"FNO (1 member, CPU)": t_one, "FNO ensemble (5, CPU)": t_ens, "FNO ensemble (5, GPU, batched)": gpu_s,
            "NNLS: Green's functions (GPU)": phase3["t_green_s"], "NNLS: solve (CPU)": nnls_solve,
            "PINN training (GPU)": pinn["train_minutes"] * 60, "Kriging (CPU)": kr}


def _t(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


class GPUDataSlice:
    """A one-scenario CPU view of :class:`GPUData`, enough for :func:`predict`."""

    def __init__(self, data: GPUData, i: int):
        self.device = "cpu"
        self._data, self._i = data, i

    def met(self, idx):
        m = self._data.met(torch.tensor([self._i], device=self._data.device))
        return Met(**{k: v.cpu() for k, v in vars(m).items()})


def write_markdown(s: dict, report, n_test: int) -> None:
    cols = [("precision", "Precision", "{:.2f}"), ("recall", "Recall", "{:.2f}"), ("loc_km_median", "Loc. err [km]", "{:.1f}"),
            ("total_emission_rel_err", "|ΔQ|/Q", "{:.2f}"), ("field_rmse", "Field RMSE [µg/m³]", "{:.1f}"),
            ("field_rel_rmse", "Rel. RMSE", "{:.2f}"), ("local_share_err_pp", "Local-share err [pp]", "{:.1f}")]
    L = ["## Source and field metrics", "",
         "| Subset | Method | " + " | ".join(c[1] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for key, row in s["table"].items():
        sub, m = key.split("|")
        cells = [fmt.format(row[k]) if k in row and row[k] is not None and np.isfinite(row[k]) else "–" for k, _, fmt in cols]
        L.append(f"| {sub} | {'**' + m + '**' if 'FNO' in m else m} | " + " | ".join(cells) + " |")
    L += ["", "## Contribution shares (9 receptors per scenario, FNO ensemble)", "",
          "| " + " | ".join(SHARE_NAMES) + " | mean |", "|" + "---|" * (len(SHARE_NAMES) + 1),
          "| " + " | ".join(f"{s['share_err_pp'][n]:.1f} pp" for n in SHARE_NAMES) + f" | {s['share_err_pp_mean']:.1f} pp |",
          "", "## Uncertainty calibration (coverage of nominal 90% intervals, test set)", "",
          "| Quantity | κ (val) | raw | calibrated |", "|---|---|---|---|"]
    i90 = list(s["levels"]).index(0.9)
    for kind in ("field", "share"):
        L.append(f"| {kind} | {s['calibration'][f'kappa_{kind}']:.2f} | {s['coverage'][kind]['raw'][i90]:.2f} | "
                 f"{s['coverage'][kind]['calibrated'][i90]:.2f} |")
    hz = s["hidden_emitters"]
    first = next(iter(hz.values()))
    L += ["", "## Hidden (off-prior) emitters", "",
          f"{first['n_true']} strong (≥ 15 g/s) off-prior sources in {n_test} test scenarios. A 3 km block is flagged when "
          "its non-traffic emission has ensemble mean − zσ > 15 g/s and the OSM prior shows no industry, construction "
          "or landfill there.", "", "| z | flagged blocks | precision | recall |", "|---|---|---|---|"]
    L += [f"| {z} | {h['n_flagged']} | {h['precision']:.2f} | {h['recall']:.2f} |" for z, h in hz.items()]
    L += ["", "## Inference time per scenario", "", "| Method | Time |", "|---|---|"]
    for k, v in s["timing"].items():
        L.append(f"| {k} | {v * 1000:.0f} ms |" if v < 1 else f"| {k} | {v:.1f} s |")
    (report / f"phase5_metrics{TAG}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def figures(s, p, test, ds, data, grid, out, dev, masks, receptors, models) -> None:
    colors = {"FNO ensemble": PALETTE[0], "NNLS-L1 + OSM prior": PALETTE[1], "NNLS-L1": PALETTE[5],
              "Kriging": PALETTE[2], "IDW": PALETTE[3]}
    if s["curves"]:
        baseline_figures(s, out, colors)
    calibration_and_examples(s, p, test, ds, grid, out, masks, receptors)


def baseline_figures(s, out, colors) -> None:
    # 1. method comparison
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), layout="constrained")
    meths = ["FNO ensemble", "NNLS-L1 + OSM prior", "NNLS-L1", "Kriging", "IDW"]
    for ax, (key, lab) in zip(axes, [("precision", "hotspot precision"), ("recall", "hotspot recall"),
                                     ("loc_km_median", "localisation error [km]"), ("field_rmse", "field RMSE [µg/m³]")]):
        for j, subset in enumerate(("easy", "general")):
            vals = [s["table"][f"{subset}|{m}"].get(key, np.nan) for m in meths]
            ax.bar(np.arange(len(meths)) + 0.4 * j, vals, 0.38, color=[colors[m] for m in meths],
                   alpha=1.0 if j else 0.55, edgecolor="black", linewidth=0.4)
        ax.set_xticks(np.arange(len(meths)) + 0.2, [m.replace(" + OSM prior", "\n+ prior").replace(" ensemble", "") for m in meths],
                      fontsize=8)
        ax.set_title(lab)
    axes[0].text(0.02, 0.97, "light: easy (n=30)\ndark: random (n=100)", transform=axes[0].transAxes, va="top", fontsize=8)
    print("saved", savefig(fig, out, "phase5_method_comparison" + TAG))

    # 2. sensitivity
    c = s["curves"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), layout="constrained")
    for m in ("FNO ensemble", "NNLS-L1 + OSM prior", "Kriging"):
        axes[0].plot(c["n_sensors"], c[m]["n_field_rmse"], "o-", color=colors[m], label=m)
        axes[2].plot(c["noise"], c[m]["noise_field_rmse"], "o-", color=colors[m], label=m)
        if "n_recall" in c[m]:
            axes[1].plot(c["n_sensors"], c[m]["n_recall"], "o-", color=colors[m], label=m)
            axes[3].plot(c["noise"], c[m]["noise_recall"], "o-", color=colors[m], label=m)
    axes[0].set(xlabel="number of sensors", ylabel="field RMSE [µg/m³]", title="Field error vs network size", xticks=c["n_sensors"])
    axes[1].set(xlabel="number of sensors", ylabel="hotspot recall", title="Source recall vs network size", xticks=c["n_sensors"])
    axes[2].set(xlabel="sensor noise σ (multiplicative)", ylabel="field RMSE [µg/m³]", title="Field error vs noise")
    axes[3].set(xlabel="sensor noise σ (multiplicative)", ylabel="hotspot recall", title="Source recall vs noise")
    axes[0].legend(fontsize=8)
    print("saved", savefig(fig, out, "phase5_sensitivity" + TAG))


    # 4. timing
    fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    names, vals = list(s["timing"]), list(s["timing"].values())
    ax.barh(names, vals, color=[PALETTE[0] if "FNO" in n else PALETTE[2] for n in names])
    ax.set(xscale="log", xlabel="seconds per scenario", title="Inference time")
    ax.axvline(1.0, color="grey", ls="--", lw=0.8)
    ax.text(1.05, len(names) - 0.6, "1 s (interactive)", fontsize=8, color="grey")
    print("saved", savefig(fig, out, "phase5_inference_time" + TAG))


def calibration_and_examples(s, p, test, ds, grid, out, masks, receptors) -> None:
    # 3. calibration
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for ax, kind in zip(axes, ("field", "share")):
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.plot(s["levels"], s["coverage"][kind]["raw"], "o-", color=PALETTE[1], label="raw ensemble")
        ax.plot(s["levels"], s["coverage"][kind]["calibrated"], "o-", color=PALETTE[0],
                label=f"calibrated (κ = {s['calibration'][f'kappa_{kind}']:.2f})")
        ax.set(xlabel="nominal coverage", ylabel="empirical coverage (test)",
               title="Field at unseen cells" if kind == "field" else "Contribution shares at receptors",
               xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
    print("saved", savefig(fig, out, "phase5_calibration" + TAG))


    # 5. examples with uncertainty
    pick = [0, 1, 2, 3]
    fig, axes = plt.subplots(len(pick), 6, figsize=(21, 3.5 * len(pick)), layout="constrained", squeeze=False)
    ext = grid.extent_km()
    for row, n in zip(axes, pick):
        i = int(test[n])
        tr = load_truth(ds, i, grid, BLOCK, 6, SRC_MIN)
        norm = LogNorm(vmin=max(tr.c_last.min(), 1), vmax=tr.c_last.max())
        on = masks[n, -1].cpu().numpy() > 0
        yy, xx = np.nonzero(on)
        vmax = max(tr.E_blocks.max(), 1)
        panels = [("true PM2.5", tr.c_last, "c"), ("FNO mean", p.c[n].mean(0), "c"),
                  ("FNO σ [µg/m³]", p.c[n].std(0, ddof=1), "s"), ("true emission [g/s/block]", tr.E_blocks, "E"),
                  ("FNO emission", p.E_blocks[n].mean(0), "E"), ("FNO emission σ", p.E_blocks[n].std(0, ddof=1), "es")]
        for ax, (title, arr, kind) in zip(row, panels):
            if kind == "c":
                im = ax.imshow(np.clip(arr, norm.vmin, None), origin="lower", extent=ext, cmap=FIELD_CMAP, norm=norm)
                ax.scatter((xx + 0.5) * 0.75, (yy + 0.5) * 0.75, s=6, c="white", edgecolors="black", linewidths=0.4)
            elif kind in ("E", "es"):
                im = ax.imshow(arr, origin="lower", extent=ext, cmap="viridis" if kind == "E" else "Greys",
                               vmin=0, vmax=vmax if kind == "E" else None)
                if kind == "E":
                    ax.plot(tr.source_xy[:, 0] / 1e3, tr.source_xy[:, 1] / 1e3, "r+", ms=9, mew=1.5)
            else:
                im = ax.imshow(arr, origin="lower", extent=ext, cmap="Greys")
            ax.set_title(title, fontsize=9)
            ax.grid(False)
            fig.colorbar(im, ax=ax, shrink=0.8)
        row[0].set_ylabel(f"#{i}")
    print("saved", savefig(fig, out, "phase5_examples" + TAG))

    # 6. shares with intervals at the domain-centre receptor for 4 scenarios
    kappa = s["calibration"]["kappa_share"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), layout="constrained", sharey=True)
    iy, ix = receptors[4]
    for ax, n in zip(axes, pick):
        i = int(test[n])
        sh = receptor_shares(p.c_cat[n], p.bg[n], iy, ix) * 100  # (M, 5)
        m, sd = sh.mean(0), sh.std(0, ddof=1)
        t = truth_shares(ds, i, [(iy, ix)])[0] * 100
        xs = np.arange(5)
        ax.bar(xs, m, 0.6, yerr=Z90 * kappa * sd, capsize=4, color=PALETTE[:5], alpha=0.85, label="FNO (90% CI)")
        ax.scatter(xs, t, marker="D", color="black", zorder=3, label="truth")
        ax.set_xticks(xs, SHARE_NAMES, rotation=30, fontsize=8)
        ax.set_title(f"#{i}: share of PM2.5 at the city centre", fontsize=9)
    axes[0].set_ylabel("contribution [%]")
    axes[0].legend(fontsize=8)
    print("saved", savefig(fig, out, "phase5_shares" + TAG))

    # 7. training curves
    hist = json.loads((cfg_path(out) / "histories.json").read_text()) if (cfg_path(out) / "histories.json").exists() else {}
    if hist:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for k, h in hist.items():
            ep = [r["epoch"] for r in h]
            a1.semilogy(ep, [r["train_total"] for r in h], color=PALETTE[int(k) % 6], lw=1, label=f"member {k} train")
            a1.semilogy(ep, [r["val_total"] for r in h], color=PALETTE[int(k) % 6], lw=1, ls="--")
            a2.plot(ep, [r["val_field_rmse"] for r in h], color=PALETTE[int(k) % 6], lw=1.2, label=f"member {k}")
        a1.set(xlabel="epoch", ylabel="loss (solid train, dashed val)", title="FNO training")
        a2.set(xlabel="epoch", ylabel="val field RMSE [µg/m³]", title="Validation (stored sensor networks)")
        a2.legend(fontsize=8)
        print("saved", savefig(fig, out, "phase5_training" + TAG))


def cfg_path(figures_dir):
    return figures_dir.parent.parent / "artifacts" / "fno"


if __name__ == "__main__":
    main()
