## Source and field metrics

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

## Contribution shares (9 receptors per scenario, FNO ensemble)

| traffic | industry | construction | burning | regional | mean |
|---|---|---|---|---|---|
| 8.6 pp | 9.2 pp | 2.4 pp | 6.1 pp | 11.8 pp | 7.6 pp |

## Uncertainty calibration (coverage of nominal 90% intervals, test set)

| Quantity | κ (val) | raw | calibrated |
|---|---|---|---|
| field | 3.93 | 0.38 | 0.90 |
| share | 5.05 | 0.17 | 0.90 |

## Hidden (off-prior) emitters

2652 strong (≥ 15 g/s) off-prior sources in 2000 test scenarios. A 3 km block is flagged when its non-traffic emission has ensemble mean − zσ > 15 g/s and the OSM prior shows no industry, construction or landfill there.

| z | flagged blocks | precision | recall |
|---|---|---|---|
| 0.0 | 337 | 0.18 | 0.02 |
| 1.0 | 82 | 0.23 | 0.01 |
| 2.0 | 20 | 0.05 | 0.00 |

## Inference time per scenario

| Method | Time |
|---|---|
| FNO (1 member, CPU) | 76 ms |
| FNO ensemble (5, CPU) | 361 ms |
| FNO ensemble (5, GPU, batched) | 3 ms |
| NNLS: Green's functions (GPU) | 8.3 s |
| NNLS: solve (CPU) | 410 ms |
| PINN training (GPU) | 86.8 s |
| Kriging (CPU) | 8 ms |
