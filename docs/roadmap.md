# Roadmap

## A. Finish v1
- Run the sealed test once, with everything frozen, and add it to `results_v1.md`. Tag the repo `v1.1`.
- Time-of-day ablation for the AR and LightGBM (how much of R²_Δ is the clock).
- Direction accuracy compared with the seasonal baseline's direction.
- Compare the breach-heavy days (2018-04-20, 2018-05-06, 2018-05-17, 2018-03-23, 2018-01-22) with known storm records.
- Re-run the scaling diagnostics (Hurst / DFA) with the corrected quality-mask names.
- Clean the notebook: merge the duplicated cells 43/45/46/48 and re-run top to bottom for a clean execution record.

## B. More data sources
Candidates: solar wind at L1, geomagnetic indices, electron channels from other satellites.
Rules for every new input:
1. It must be **causal** (use only the past) and pass the existing leak tests.
2. It must be **available in real time** at forecast time. Check the publication delay of each source, because this decides whether it can run live.
3. It is judged on the **same folds** with the same Diebold-Mariano tests and placebo tests.
Record the outcome (gain or no gain) in `docs/`.

## C. Model upgrades
- Put the jump state into the variance forecast.
- Address the clustering of band breaches (for example by letting the band react faster after a miss).
- Re-test the 6 h / 12 h horizons, Hawkes processes, cointegration, and deep nets under the new protocol.
- Decide a refit schedule: the GJR parameters drift over time.

## D. Live forecaster and dashboard
First question: is there a **near-real-time GRASP feed**? This is not known yet and decides everything below.
1. Ingestion job that fetches new samples and appends them to the grid.
2. Feature builder that **reuses the research feature code**, so live and backtest cannot drift apart.
3. Saved final models (trained on all available data) and a variance filter that updates with each sample.
4. Forecast plus 5-95 % band for 30 min / 45 min / 1 h.
5. Dashboard (for example Streamlit or Plotly Dash) showing actual vs forecast, the band, and a status panel.
6. Log every live forecast with its timestamp, then score it later. Live results are the real test.

## Planned repository growth
    src/electron_flux/   importable package (features, models, evaluation)
    tests/               leak tests as automated checks
    pipelines/           data ingestion and training scripts
    app/                 dashboard
    notebooks/           one notebook per experiment (02_..., 03_...)
