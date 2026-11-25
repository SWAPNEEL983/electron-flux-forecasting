# Evaluation protocol (frozen before modelling)

## Data grid
Raw samples are snapped to a regular 5-minute grid of **122,976 slots** (no rows dropped). The grid is fingerprinted with an
md5 digest of the electron and proton columns: `020ce9d715220416326ed6f07b1f0674`. A different digest means a different dataset.

## Windows
Five expanding validation folds, cut on whole-hour boundaries, plus a sealed test window.

| Window | Validation period ends |
|---|---|
| cv1 | 2017-12-19 (starts 2017-11-01) |
| cv2 | 2018-02-05 |
| cv3 | 2018-03-25 |
| cv4 | 2018-05-12 |
| cv5 | 2018-06-29 (identical to the static split) |
| sealed test | 2018-06-30 to 2018-09-01, **not scored yet** |

## Leakage rules
- **Train rows:** the *target* time must be at least **24 h (288 steps) before the validation start** (embargo).
- **Validation rows:** origin at or after the fold start, and target before the fold end.
- Features may use only the past. Flags that look at the future (outage, near-outage, spike) are named `*_noncausal`
  and are used **only to choose which rows are trained and scored**, never as model inputs. `stale_steps` is causal.
  - outage: a missing run of at least 6 steps; near-outage: within 12 steps of one; spike: up more than 2.5 in log and straight back down.
- **Leak tests:** every feature builder is perturbed (scale, NaN, shuffle, fill, crash, hold) and the causal test, fit-independence test
  and a gallery of 6 deliberate leaks are run. All 6 leaks are caught and 4 clean builders pass.

## Metrics
- Headline: **R²_Δ**, the share of the log-flux change explained; persistence scores 0.
- Also: R² on the log level and raw flux, MSE/MAE, skill vs persistence and vs a seasonal baseline,
  Diebold-Mariano tests with HAC errors, day-block bootstrap intervals, placebo tests.
- Identity linking the two: `1 - R²_log(model) = (1 - R²_log(persistence)) * (1 - R²_Δ)`.

## Protocol disclosures and amendments
1. Horizons were changed from 45 min / 6 h / 12 h to **30 min / 45 min / 1 h**.
2. The seasonal-delta baseline was **redefined** between runs (now the mean change by 30-minute time-of-day bin from training rows).
3. The final feature and lag specification (26 features, lag reach to 1 day).
4. The 0.5 / 0.5 **blend** was adopted while looking at the validation folds.
5. A proton "alternating sample" artefact (2017-09-15 to 09-18) was found; it does not appear in electron.
6. The quality mask changes tail statistics (30-min excess kurtosis 26.0 with the mask, 31.4 without).
7. The volatility features were informed by AR validation diagnostics.
8. The GJR structural choice used training windows that include earlier folds' validation periods.
9. In the probabilistic forecast, the intraday scale is pooled over all 5-minute phases and re-levelled to the Step 3 start variance.
10. Band calibration is causal (earlier folds only), so the recalibrated and fixed-width bands do not exist for cv1.
