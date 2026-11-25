"""Shared evaluation kit: one common sample set + identical metrics for every model."""
import numpy as np, pandas as pd
from core import P
from runio import RUN, load_split

FOLDS = [f"cv{k}" for k in range(1, 6)]          # the sealed test is never touched here
HZ = list(P.horizons)
DAY = 288
LAGS = [1, 2, 3, 6, 12, 24, 36, 72, 144, 216, 288]

def metrics(y, p, y0, pbest=None):
    e = y - p; sse = np.sum(e ** 2); d = y - y0
    ey, pe = np.exp(y), np.exp(p)
    out = {"n": len(y),
           "r2_log": 1 - sse / np.sum((y - y.mean()) ** 2),
           "r2_raw": 1 - np.sum((ey - pe) ** 2) / np.sum((ey - ey.mean()) ** 2),
           "r2_delta": 1 - sse / np.sum((d - d.mean()) ** 2),
           "skill_vs_persist": 1 - sse / np.sum(d ** 2),
           "mse_log": sse / len(y), "mae_log": np.abs(e).mean()}
    if pbest is not None:
        out["skill_vs_best"] = 1 - sse / np.sum((y - pbest) ** 2)
    return out

def best_baseline():
    d = pd.read_csv(RUN / "reports" / "baselines_pooled.csv")
    d = d[d.eval_set == "scored"]
    return d.sort_values("mse_log").groupby("horizon").first()["model"].to_dict()

def common_set(window, hn):
    """Samples every model is scored on = val_scored with all three baselines defined."""
    z = np.load(RUN / "preds" / f"baselines_{window}_{hn}.npz")
    return z["pos"], z["y"], {k: z[k] for k in z.files if k not in ("pos", "y")}

def features_frame(df, H, kind="full"):
    """Full-grid features for horizon H. Row t uses data stamped <= t (plus the calendar time of t+H)."""
    Lf = np.log(df["electron"]).ffill()              # last-observation-carried-forward: backward-looking only
    cols = {"y0": Lf}
    for k in LAGS:
        cols[f"d{k}"] = Lf.shift(k) - Lf
    if kind == "full":
        mod = (df.index.hour * 60 + df.index.minute).to_numpy()
        tod = ((mod + 5 * H) % 1440) / 1440.0        # time of day of the TARGET (known at forecast time)
        for k in (1, 2, 3):
            cols[f"sin{k}"] = np.sin(2 * np.pi * k * tod)
            cols[f"cos{k}"] = np.cos(2 * np.pi * k * tod)
        cols["seas_ref"] = Lf.shift(DAY - H) - Lf    # value 24 h before the target time
        for w, mp in ((12, 6), (72, 36), (288, 144)):
            r = Lf.rolling(w, min_periods=mp)
            cols[f"m{w}"] = r.mean() - Lf
            cols[f"s{w}"] = r.std()
        cols["miss288"] = df["missing"].astype(float).rolling(DAY, min_periods=1).mean()
    return pd.DataFrame(cols, index=df.index)

def run_models(fit_predict, tag):
    """fit_predict(window, hn, H, train_pos, eval_pos) -> {model_name: log-flux predictions at eval_pos}.
    Called once per (fold, horizon); must fit on train_pos ONLY."""
    best = best_baseline()
    rows, pool = [], {}
    for hn, H in P.horizons.items():
        for w in FOLDS:
            s = load_split(w, hn)
            pos, y, base = common_set(w, hn)
            preds = fit_predict(w, hn, H, s["train"], pos)
            for m, p in preds.items():
                assert np.isfinite(p).all(), f"{m}: non-finite predictions in {w} {hn}"
            allp = {**base, **preds}
            y0, pb = base["persistence"], base[best[hn]]
            for m, p in allp.items():
                rows.append({"horizon": hn, "fold": w, "model": m, **metrics(y, p, y0, pb)})
                pool.setdefault((hn, m), []).append((y, p, y0, pb))
            np.savez_compressed(RUN / "preds" / f"{tag}_{w}_{hn}.npz", pos=pos, y=y, **preds)
    res = pd.DataFrame(rows)
    pooled = pd.DataFrame([{"horizon": hn, "model": m, **metrics(*[np.concatenate(c) for c in zip(*lst)])}
                           for (hn, m), lst in pool.items()])
    res.to_csv(RUN / "reports" / f"{tag}_per_fold.csv", index=False)
    pooled.to_csv(RUN / "reports" / f"{tag}_pooled.csv", index=False)
    return res, pooled

def show(res, pooled, metric_names=("r2_log", "r2_delta", "skill_vs_best", "mse_log")):
    pd.set_option("display.float_format", "{:.3f}".format); pd.set_option("display.width", 220)
    order = {h: i for i, h in enumerate(HZ)}
    for mt in metric_names:
        t = res.pivot_table(index=["horizon", "model"], columns="fold", values=mt)[FOLDS]
        t["mean"], t["std"] = t.mean(axis=1), t[FOLDS].std(axis=1)
        t["pooled"] = pooled.set_index(["horizon", "model"])[mt]
        print(f"\n=== {mt} ===")
        print(t.sort_index(key=lambda ix: ix.map(order) if ix.name == "horizon" else ix))