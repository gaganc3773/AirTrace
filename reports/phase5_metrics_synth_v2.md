## Source and field metrics

| Subset | Method | Precision | Recall | Loc. err [km] | |ΔQ|/Q | Field RMSE [µg/m³] | Rel. RMSE | Local-share err [pp] |
|---|---|---|---|---|---|---|---|---|
| test (n=2000) | **FNO ensemble** | 0.51 | 0.68 | 3.2 | 0.25 | 33.8 | 0.16 | 7.3 |

## Contribution shares (9 receptors per scenario, FNO ensemble)

| traffic | industry | construction | burning | regional | mean |
|---|---|---|---|---|---|
| 4.6 pp | 5.0 pp | 0.7 pp | 2.8 pp | 7.4 pp | 4.1 pp |

## Uncertainty calibration (coverage of nominal 90% intervals, test set)

| Quantity | κ (val) | raw | calibrated |
|---|---|---|---|
| field | 5.24 | 0.32 | 0.90 |
| share | 4.86 | 0.19 | 0.90 |

## Hidden (off-prior) emitters

2945 strong (≥ 15 g/s) off-prior sources in 2000 test scenarios. A 3 km block is flagged when its non-traffic emission has ensemble mean − zσ > 15 g/s and the OSM prior shows no industry, construction or landfill there.

| z | flagged blocks | precision | recall |
|---|---|---|---|
| 0.0 | 183 | 0.11 | 0.01 |
| 1.0 | 34 | 0.09 | 0.00 |
| 2.0 | 10 | 0.10 | 0.00 |

## Inference time per scenario

| Method | Time |
|---|---|
