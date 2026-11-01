## Leave-one-station-out, 2025-11-01 to 2025-11-30 (51 stations, 11023 held-out predictions)

| Hours (IST) | Method | RMSE [µg/m³] | MAE | Bias | r | RMSE / mean |
|---|---|---|---|---|---|---|
| all | **FNO ensemble** | 82.9 | 58.8 | +3.4 | 0.67 | 0.39 |
| all | **FNO + residual kriging** | 81.0 | 54.0 | -0.5 | 0.69 | 0.38 |
| all | Kriging | 75.8 | 49.9 | -0.8 | 0.73 | 0.36 |
| all | IDW | 77.9 | 51.5 | -3.0 | 0.72 | 0.37 |
| all | Network median | 81.7 | 56.3 | -0.4 | 0.68 | 0.38 |
| 20-08 (night) | **FNO ensemble** | 94.2 | 69.1 | -3.0 | 0.54 | 0.39 |
| 20-08 (night) | **FNO + residual kriging** | 91.5 | 64.3 | -1.3 | 0.59 | 0.38 |
| 20-08 (night) | Kriging | 85.7 | 59.4 | -1.1 | 0.64 | 0.36 |
| 20-08 (night) | IDW | 87.9 | 60.9 | -4.4 | 0.62 | 0.37 |
| 20-08 (night) | Network median | 92.7 | 67.3 | -2.1 | 0.56 | 0.39 |
| 08-20 (day) | **FNO ensemble** | 69.7 | 48.4 | +9.9 | 0.75 | 0.38 |
| 08-20 (day) | **FNO + residual kriging** | 68.7 | 43.5 | +0.3 | 0.76 | 0.37 |
| 08-20 (day) | Kriging | 64.2 | 40.4 | -0.5 | 0.79 | 0.35 |
| 08-20 (day) | IDW | 66.2 | 42.0 | -1.6 | 0.77 | 0.36 |
| 08-20 (day) | Network median | 68.9 | 45.1 | +1.3 | 0.75 | 0.37 |

Per-station RMSE: FNO beats kriging at 22% of stations, FNO + residual kriging at 20%. Nominal 90% FNO intervals cover 12% of held-out readings raw and 54% with the synthetic-data κ = 5.24.

Meteorology clipped to the training range: wind speed 1% of hours, mixing height 63% of hours.

## Mean source shares at the monitors (FNO ensemble, full network)

| traffic | industry | construction | burning | regional |
|---|---|---|---|---|
| 9% | 5% | 0% | 2% | 84% |
