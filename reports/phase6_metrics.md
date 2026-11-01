## Leave-one-station-out, 2025-11-01 to 2025-11-30 (51 stations, 11023 held-out predictions)

| Hours (IST) | Method | RMSE [µg/m³] | MAE | Bias | r | RMSE / mean |
|---|---|---|---|---|---|---|
| all | **FNO ensemble** | 84.8 | 59.7 | -13.0 | 0.66 | 0.40 |
| all | **FNO + residual kriging** | 80.9 | 54.2 | +4.3 | 0.70 | 0.38 |
| all | Kriging | 75.8 | 49.9 | -0.8 | 0.73 | 0.36 |
| all | IDW | 77.9 | 51.5 | -3.0 | 0.72 | 0.37 |
| all | Network median | 81.7 | 56.3 | -0.4 | 0.68 | 0.38 |
| 20-08 (night) | **FNO ensemble** | 93.6 | 68.0 | -12.1 | 0.55 | 0.39 |
| 20-08 (night) | **FNO + residual kriging** | 91.3 | 64.6 | +3.7 | 0.59 | 0.38 |
| 20-08 (night) | Kriging | 85.7 | 59.4 | -1.1 | 0.64 | 0.36 |
| 20-08 (night) | IDW | 87.9 | 60.9 | -4.4 | 0.62 | 0.37 |
| 20-08 (night) | Network median | 92.7 | 67.3 | -2.1 | 0.56 | 0.39 |
| 08-20 (day) | **FNO ensemble** | 74.9 | 51.3 | -13.8 | 0.72 | 0.41 |
| 08-20 (day) | **FNO + residual kriging** | 68.7 | 43.8 | +5.0 | 0.76 | 0.37 |
| 08-20 (day) | Kriging | 64.2 | 40.4 | -0.5 | 0.79 | 0.35 |
| 08-20 (day) | IDW | 66.2 | 42.0 | -1.6 | 0.77 | 0.36 |
| 08-20 (day) | Network median | 68.9 | 45.1 | +1.3 | 0.75 | 0.37 |

Per-station RMSE: FNO beats kriging at 22% of stations, FNO + residual kriging at 29%. Nominal 90% FNO intervals cover 25% of held-out readings raw and 72% with the synthetic-data κ = 3.93.

Meteorology clipped to the training range: wind speed 10% of hours, mixing height 68% of hours.

## Mean source shares at the monitors (FNO ensemble, full network)

| traffic | industry | construction | burning | regional |
|---|---|---|---|---|
| 11% | 5% | 1% | 2% | 80% |
