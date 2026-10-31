# AirTrace

Physics-informed PM2.5 source attribution from sparse sensors (Delhi NCR prototype).

> Work in progress. Phases 1-3 (solver, synthetic data, linear baseline) are complete; the full README with results lands in Phase 8.

## Setup

```bash
python -m venv .venv --system-site-packages
.venv/Scripts/python -m pip install -e .   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pytest
.venv/Scripts/python scripts/phase1_solver.py
.venv/Scripts/python scripts/phase2_generate.py   # ~20 min on a laptop GPU, 2 GB
.venv/Scripts/python scripts/phase2_figures.py
.venv/Scripts/python scripts/phase3_inversion.py   # ~20 min (Green's functions on GPU)
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
