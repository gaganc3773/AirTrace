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
