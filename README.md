# AirTrace

Physics-informed PM2.5 source attribution from sparse sensors (Delhi NCR prototype).

> Work in progress. Phases 1-5 (solver, synthetic data, linear baseline, PINN, FNO ensemble) are complete; the full README with results lands in Phase 8.

## Setup

```bash
python -m venv .venv --system-site-packages
.venv/Scripts/python -m pip install -e .   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pytest
.venv/Scripts/python scripts/phase1_solver.py
.venv/Scripts/python scripts/phase2_generate.py   # ~20 min on a laptop GPU, 2 GB
.venv/Scripts/python scripts/phase2_figures.py
.venv/Scripts/python scripts/phase3_inversion.py   # ~20 min (Green's functions on GPU)
.venv/Scripts/python scripts/phase4_pinn.py --scenario 19627
.venv/Scripts/python scripts/phase5_contributions.py   # ~75 min: per-category concentration fields
.venv/Scripts/python scripts/phase5_train.py           # ~55 min: 5-member FNO ensemble
.venv/Scripts/python scripts/phase5_evaluate.py
```

## Phase 1: forward solver

Flux-form finite-volume solver for `∂c/∂t + ∇·(uc) = ∇·(K∇c) − λc + S` on a 48 × 48 km, 64 × 64 grid
(`src/airtrace/solver/`). Koren-limited MUSCL advection (TVD, positive), central diffusion, SSP-RK3,
inflow-Dirichlet / outflow zero-gradient boundaries, batched in PyTorch (CPU or GPU).

| Gaussian puff, 3 h, rel. L2 error | 1500 m | 750 m | 375 m |
|---|---|---|---|
| 1st-order upwind | 68.1% | 52.1% | 35.6% |
| van Leer | 26.4% | 8.1% | 1.9% |
| **Koren (default)** | 19.8% | **4.6%** | 0.7% |
| linear κ=1/3 (for Green's functions) | 13.3% | 2.3% | 0.3% |

![puff](reports/figures/phase1_puff_verification.png)
![plume](reports/figures/phase1_rotating_plume.gif)

## Phase 2: synthetic dataset

20,000 scenarios (`src/airtrace/data/`), each a 30 h simulation (24 h spin-up + 6 h observation window):

- **Geography:** 70% on real Delhi priors rasterised from OpenStreetMap (major roads, industrial,
  construction/brownfield, landfill/farmland; cached in `artifacts/cache/osm`), 30% on 256 random synthetic cities.
- **Sources:** traffic line sources along roads (rush hours + night truck window), 3-12 industrial clusters,
  0-6 construction sites (daytime), 0-5 burning sites (landfill / evening). 15% of discrete sources are placed
  where the prior shows nothing; half of the off-prior industry emits only at night (hidden emitters).
- **Meteorology:** hourly wind with a NW-sector bias, diurnal speed cycle and a divergence-free spatial
  perturbation; diurnal mixing height converts emissions to S = E / (H dx dy).
- **Regional background:** inflow BC plus a uniform term λ c_bg + dc_bg/dt, so a uniform c_bg(t) is an exact
  solution (removal acts on the local excess only).
- **Sensors:** 40 cells ≥ 1.5 km apart, 10-20% lognormal noise, 0-30% offline, 5% hourly gaps
  (training re-samples networks on the fly).
- **Exact re-simulation:** every scenario is stored with its compact spec (wind, background, inventory) so it can
  be re-run, e.g. for Green's functions.

![sanity grid](reports/figures/phase2_sanity_grid.png)

## Phase 3: linear inversion baseline

Source-receptor matrix for 256 candidate 3 km blocks (one batched forward run of the exactly linear scheme,
~8 s per scenario on GPU), then non-negative L1 + Tikhonov least squares with an hourly background, solved
exactly as NNLS after a Cholesky transform. The L1 weight is chosen by 5-fold cross-validation over sensors.
The "+ OSM prior" variant makes emitting cheaper where OpenStreetMap shows roads, industry or landfills.
Assumes perfectly known meteorology and constant-in-time emissions. Evaluated on held-out test scenarios
(indices 18000-19999).

Hotspot threshold 15 g/s per 3 km block; match tolerance 1 block. Green's functions: 8.3 s per scenario on cuda.

| Subset | Method | Precision | Recall | Loc. err [km] | |ΔQ|/Q | Field RMSE [µg/m³] | Rel. RMSE | Local-share err [pp] | Solve time [s] |
|---|---|---|---|---|---|---|---|---|---|
| easy (n=30) | NNLS-L1 | 0.47 | 0.57 | 4.5 | 0.43 | 19.5 | 0.28 | 9.6 | 0.42 |
| easy (n=30) | NNLS-L1 + OSM prior | 0.51 | 0.62 | 3.2 | 0.31 | 18.7 | 0.27 | 10.4 | 0.40 |
| easy (n=30) | IDW | – | – | – | – | 22.2 | 0.31 | – | 0.01 |
| easy (n=30) | Kriging | – | – | – | – | 21.7 | 0.30 | – | 0.01 |
| all (n=100) | NNLS-L1 | 0.48 | 0.52 | 4.5 | 0.47 | 36.0 | 0.40 | 13.1 | 0.43 |
| all (n=100) | NNLS-L1 + OSM prior | 0.53 | 0.56 | 3.8 | 0.47 | 34.6 | 0.39 | 12.1 | 0.41 |
| all (n=100) | IDW | – | – | – | – | 34.8 | 0.40 | – | 0.01 |
| all (n=100) | Kriging | – | – | – | – | 34.4 | 0.39 | – | 0.01 |

![examples](reports/figures/phase3_examples_easy.png)

## Phase 4: PINN

Networks for the local excess c'(x, y, t) ≥ 0 (random Fourier features in space-time) and a time-constant
emission density e(x, y) ≥ 0, plus an hourly uniform background. Loss = sensor misfit (hourly means,
noise-weighted) + causally weighted PDE residual (autograd, with the scenario's known divergence-free wind
and mixing height) + inflow BC + source L1. Non-dimensional (L = 48 km, hours, 100 µg/m³).

What made it work: (1) half the collocation points jittered around sensors, otherwise the network
satisfies sensors with spikes that uniform collocation never sees; (2) PDE weight 1000 (at 100 the data
chi²/n fell to 0.18, i.e. it fitted noise); (3) **Morozov early stopping**: stop when data chi²/n reaches 1,
a criterion that never looks at ground truth.

| Scenario | Kriging | IDW | **PINN** | NNLS-L1 + prior (Phase 3) | PINN train time |
|---|---|---|---|---|---|
| #19627 | 12.7 | 13.2 | **11.2** | 7.4 | 1.4 min |
| #18080 | 21.4 | 22.1 | **17.2** | 12.3 | 0.7 min |

Field RMSE [µg/m³] at unseen cells, last hour. The PINN beats physics-free interpolation by 12-20% and
recovers the total emission (826 vs 811 g/s true on #19627, where NNLS gets 477) but smears the source
along the plume axis (an up/downwind degeneracy). It is slower and less accurate than the Green's-function
inversion, which uses the exact discrete operator. Its role in AirTrace is mesh-free, single-scenario
reconstruction; the FNO (Phase 5) is the fast, amortised model.

![pinn](reports/figures/phase4_pinn_reconstruction.png)
![pinn loss](reports/figures/phase4_pinn_loss.png)

## Phase 5: FNO deep ensemble

A Fourier Neural Operator (4 layers, width 64, 16 modes, zero-padded against wrap-around; ~16M params)
maps 45 input channels (6 h of rasterised sensor readings + masks + IDW interpolations, hourly wind,
mixing height, hour of day, OSM priors, coordinates) to 11 outputs: last-hour source maps per category,
last-hour concentration **per category**, regional background, and c at T and T−1 h. Per-category
concentration targets come from a linear decomposition run (`phase5_contributions.py`), so attribution
is a ratio of predicted fields: no solve at click time. Training: 17k scenarios with **random sensor
networks re-drawn every batch** (5-40 sensors, 5-25% noise, outages), 1k validation, 2k test.
Five members with different seeds; intervals inflated by a factor κ fitted on validation.

Notes from development:
- Adding dense IDW channels and more width cut validation error by ~5% and raised the win rate vs
  kriging from 56% to 70% of scenarios (isolated non-zero pixels are hard for spectral layers).
- The physics-consistency loss (hour-integrated PDE residual of predicted c with predicted S, trapezoid
  in time) is non-zero even for the true fields (RMS ~0.5 vs source RMS ~1.4 in scaled units), so at
  weight 0.1 it blurred outputs; it is kept at 0.02 as a light regulariser.
- The reference decomposition uses the unlimited linear scheme, which undershoots by up to tens of
  µg/m³ at sharp plume edges; contributions are clipped at zero when forming shares.
- The raw ensemble is strongly over-confident (90% intervals cover 38% of field errors, 17% of share
  errors); κ ≈ 4-5 restores nominal coverage on the test set.

### Source and field metrics

| Subset | Method | Precision | Recall | Loc. err [km] | |ΔQ|/Q | Field RMSE [µg/m³] | Rel. RMSE | Local-share err [pp] |
|---|---|---|---|---|---|---|---|---|
| easy | **FNO ensemble** | 0.46 | 0.60 | 3.2 | 0.24 | 20.3 | 0.28 | 10.9 |
| easy | NNLS-L1 | 0.47 | 0.57 | 4.5 | 0.43 | 19.5 | 0.28 | 9.6 |
| easy | NNLS-L1 + OSM prior | 0.51 | 0.62 | 3.2 | 0.31 | 18.7 | 0.27 | 10.4 |
| easy | IDW | – | – | – | – | 22.2 | 0.31 | – |
| easy | Kriging | – | – | – | – | 21.7 | 0.30 | – |
| general | **FNO ensemble** | 0.52 | 0.60 | 3.6 | 0.30 | 32.5 | 0.37 | 12.3 |
| general | NNLS-L1 | 0.48 | 0.52 | 4.5 | 0.47 | 36.0 | 0.40 | 13.1 |
| general | NNLS-L1 + OSM prior | 0.53 | 0.56 | 3.8 | 0.47 | 34.6 | 0.39 | 12.1 |
| general | IDW | – | – | – | – | 34.8 | 0.40 | – |
| general | Kriging | – | – | – | – | 34.4 | 0.39 | – |
| test (n=2000) | **FNO ensemble** | 0.52 | 0.58 | 3.5 | 0.27 | 32.8 | 0.38 | 11.6 |

### Contribution shares (9 receptors per scenario, FNO ensemble)

| traffic | industry | construction | burning | regional | mean |
|---|---|---|---|---|---|
| 8.6 pp | 9.2 pp | 2.4 pp | 6.1 pp | 11.8 pp | 7.6 pp |

### Uncertainty calibration (coverage of nominal 90% intervals, test set)

| Quantity | κ (val) | raw | calibrated |
|---|---|---|---|
| field | 3.93 | 0.38 | 0.90 |
| share | 5.05 | 0.17 | 0.90 |

### Hidden (off-prior) emitters

2652 strong (≥ 15 g/s) off-prior sources in 2000 test scenarios. A 3 km block is flagged when its non-traffic emission has ensemble mean − zσ > 15 g/s and the OSM prior shows no industry, construction or landfill there.

| z | flagged blocks | precision | recall |
|---|---|---|---|
| 0.0 | 337 | 0.18 | 0.02 |
| 1.0 | 82 | 0.23 | 0.01 |
| 2.0 | 20 | 0.05 | 0.00 |

### Inference time per scenario

| Method | Time |
|---|---|
| FNO (1 member, CPU) | 76 ms |
| FNO ensemble (5, CPU) | 361 ms |
| FNO ensemble (5, GPU, batched) | 3 ms |
| NNLS: Green's functions (GPU) | 8.3 s |
| NNLS: solve (CPU) | 410 ms |
| PINN training (GPU) | 86.8 s |
| Kriging (CPU) | 8 ms |

**Reading the results.** On random held-out scenarios the FNO beats every baseline on field error
(32.5 vs 34.4 kriging, 34.6 NNLS + prior), hotspot recall (0.60 vs 0.56) and total-emission error (0.30 vs
0.47) at equal precision, and it degrades far more gracefully with fewer sensors (39.7 vs 45.1 kriging and
52.6 NNLS at 5 sensors) and more noise. On the 30 easy scenarios the Green's-function inversion, which has
the exact physics, is still slightly better on field error (18.7 vs 20.3). The ensemble answers in ~0.4 s on a
laptop CPU (5 members) versus ~8.7 s for NNLS. **Hidden-emitter detection from a single 6 h window is weak**
(recall ≤ 0.02): the network leans on the OSM prior, so off-prior emission is under-predicted.

![sensitivity](reports/figures/phase5_sensitivity.png)
![examples](reports/figures/phase5_examples.png)
![calibration](reports/figures/phase5_calibration.png)
![shares](reports/figures/phase5_shares.png)
