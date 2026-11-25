# Results, v1.0 (validation only)

All numbers are pooled over the five validation folds (cv1-cv5) unless stated. The sealed test is untouched.

## Point forecast: R²_Δ (share of the change explained)
| Model | 30 min | 45 min | 1 h |
|---|---|---|---|
| Persistence | 0 | 0 | 0 |
| Seasonal baseline (mean change by time-of-day bin) | 0.0749 | 0.1130 | 0.1500 |
| Ridge AR (`ar_final`) | 0.1060 | 0.1547 | 0.1997 |
| LightGBM, AR inputs only | 0.1317 | 0.1843 | 0.2243 |
| LightGBM, 26 features | 0.1419 | 0.1935 | 0.2412 |
| LightGBM, 26 features, Huber loss | 0.1452 | 0.1920 | 0.2359 |
| **Blend (0.5 LightGBM + 0.5 Huber)** | **0.1494** | **0.2004** | **0.2458** |
| XGBoost blend (robustness check) | 0.1466 | 0.1977 | 0.2470 |

R² on the log level: persistence 0.9670 / 0.9505 / 0.9335, seasonal 0.9695 / 0.9561 / 0.9435, `ar_final` 0.9705 / 0.9581 / 0.9468, blend 0.9720 / 0.9604 / 0.9498.

Blend R²_Δ by fold:

| | cv1 | cv2 | cv3 | cv4 | cv5 |
|---|---|---|---|---|---|
| 30 min | 0.162 | 0.151 | 0.101 | 0.159 | 0.185 |
| 45 min | 0.214 | 0.211 | 0.138 | 0.214 | 0.239 |
| 1 h | 0.269 | 0.262 | 0.170 | 0.253 | 0.294 |

Squared-error reduction of the blend: vs the AR 4.9 % / 5.4 % / 5.8 %; vs the seasonal baseline 8.1 % / 9.9 % / 11.3 %.

## Is the skill real?
- AR placebo (lags shuffled among days at the same time of day): real skill vs seasonal 0.034 / 0.047 / 0.059, best of 30 placebo draws 0.004 / 0.005 / 0.006.
- LightGBM placebo (history rolled by whole days): 32-37 standard deviations above the placebo mean; recent history adds about 0.07 / 0.09 / 0.10 of R²_Δ beyond time of day.
- XGBoost reproduces LightGBM: differences at most about 0.006 of R²_Δ; prediction correlation 0.98-0.99.

## Direction (non-overlapping origins, cv2-cv5)
| | 30 min | 45 min | 1 h |
|---|---|---|---|
| Hit rate | 68.9 % | 72.3 % | 74.8 % |
| Always guess the commoner direction | 50.9 % | 50.7 % | 50.9 % |
| Top 25 % most confident forecasts | 77.9 % | 81.7 % | 85.0 % |

## Variance model (R² of |change| against forecast volatility)
| | 30 min | 45 min | 1 h |
|---|---|---|---|
| GJR-GARCH(1,1), Student-t, refit per fold | 0.1615 | 0.1499 | 0.1334 |
| EWMA | 0.1457 | 0.1330 | 0.1116 |
| Rolling 6 h | 0.1204 | 0.1169 | 0.0971 |
| Constant | 0.0001 | 0.0006 | 0.0020 |
| Ceiling if volatility were known exactly | 0.260 | 0.230 | 0.215 |

## Probabilistic band, central 90 % (cv2-cv5, all three bands on the same rows)
| | 30 min | 45 min | 1 h |
|---|---|---|---|
| Coverage: plain GARCH band | 0.925 | 0.932 | 0.935 |
| Coverage: **recalibrated** | **0.905** | **0.900** | **0.900** |
| Coverage: fixed width | 0.909 | 0.909 | 0.909 |
| Mean width in log units: plain / recalibrated / fixed | 0.76 / 0.69 / 0.72 | 0.98 / 0.83 / 0.90 | 1.17 / 0.98 / 1.04 |
| Interval score (lower is better): plain / recalibrated / fixed | 1.091 / 1.079 / 1.312 | 1.356 / 1.331 / 1.602 | 1.576 / 1.539 / 1.829 |

Weakness: the recalibrated band's breach rate varies by fold (7.2 % to 13.2 % against a 10 % target) and breaches cluster in time
(independence test rejected at every horizon).
