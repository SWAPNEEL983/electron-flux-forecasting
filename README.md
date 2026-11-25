# Electron-flux forecasting (ISRO GRASP)

Short-horizon forecasts (**30 min, 45 min, 1 h**) of geostationary electron flux, with a calibrated
uncertainty band, using finance-style time-series methods: ridge AR, LightGBM, and a GJR-GARCH variance model.
Data: 5-minute electron and proton flux from ISRO's GRASP instrument, 2017-07-01 to 2018-09-01.

**Status: v1.0.** Validation results are complete. The sealed test window has **not been run yet**.

## Headline results
Share of the log-flux *change* explained (R²_Δ, pooled over five validation folds; persistence = 0):

| Model | 30 min | 45 min | 1 h |
|---|---|---|---|
| Seasonal baseline (clock only) | 0.075 | 0.113 | 0.150 |
| Ridge AR | 0.106 | 0.155 | 0.200 |
| LightGBM blend (best) | **0.149** | **0.200** | **0.246** |

- Positive in every fold at every horizon; weakest fold is cv3.
- Direction is right 69 % / 72 % / 75 % of the time (about 51 % for always guessing the commoner direction).
- Placebo tests (scrambled history) drop the skill to about zero, so the gain is not an artefact.
- GJR-GARCH with Student-t innovations beats EWMA, rolling-window and constant variance.
- Recalibrated 90 % band covers 0.905 / 0.900 / 0.900, but breaches cluster and the rate varies by fold (7-13 %).

Full tables: [`docs/results_v1.md`](docs/results_v1.md). The full story, with the maths behind each step and why it was taken: [`docs/progress.md`](docs/progress.md).

## How it was evaluated
A frozen protocol was fixed before modelling: 5 expanding validation folds (2017-11-01 to 2018-06-29),
a 24 h embargo, perturbation-based leakage tests (6 deliberate leaks caught, 4 clean feature builders pass),
and a sealed test window (2018-06-30 to 2018-09-01). Details and the list of protocol disclosures:
[`docs/protocol.md`](docs/protocol.md).

## Honest limits
- cv5 is not a clean holdout: model choices were made while looking at cv1-cv5.
- The time-of-day ablation and direction-vs-seasonal comparison are not run yet.
- One instrument, about 14 months of data.
- Part of the skill is the daily cycle that the clock-only baseline already captures.

## Repository layout
    notebooks/01_electron_flux_forecasting.ipynb   code, maths and results (v1)
    src/                core.py, runio.py, evalkit.py (protocol, loaders, evaluation kit)
    docs/               progress log (maths + reasoning), protocol, results, data note, roadmap
    data/               not tracked; see data/README.md
    CHANGELOG.md

## Run it
The notebook was run on Kaggle (Python 3.12) from top to bottom and expects `grasp_combined.csv` in
`/kaggle/working/`. Install packages with `pip install -r requirements.txt`. Interactive Plotly outputs were
stripped from the stored notebook to keep it small; re-running regenerates them.
The data is **not** included. See [`docs/data.md`](docs/data.md).

## Roadmap
More data sources (solar wind, geomagnetic indices), upgrades to the model, and a live forecaster with a
dashboard: [`docs/roadmap.md`](docs/roadmap.md).

## Data credit and licence
Flux data: ISRO GRASP, publicly available from [add portal link here]. Code: MIT (see `LICENSE`).
