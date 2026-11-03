# Full pipeline: LQR vs LQI (paired seeds 100-111)

| Condition | Controller | Success | Median dock \|dy\| (mm) | p95 dock \|dy\| (mm) | Median track RMS lateral vs truth (mm) | Median effort dω (rad/s²) | Median time (s) |
|---|---|---|---|---|---|---|---|
| none | LQR | 12/12 | 1.2 | 1.6 | 16.7 | 1.18 | 44.6 |
| none | LQI | 12/12 | 1.0 | 1.7 | 16.2 | 1.24 | 44.6 |
| motor | LQR | 12/12 | 3.3 | 4.5 | 7.7 | 1.15 | 44.8 |
| motor | LQI | 12/12 | 1.3 | 2.0 | 9.3 | 1.19 | 45.0 |
| push | LQR | 12/12 | 1.0 | 4.5 | 24.8 | 1.36 | 53.2 |
| push | LQI | 11/12 | 1.2 | 148.7 | 23.6 | 1.40 | 49.3 |
