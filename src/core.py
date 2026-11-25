"""GRASP flux: data grid, frozen protocol, leak-free splits, perturbation leakage tests."""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable
import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=5)

# ------------------------------------------------------------------ protocol (frozen)
@dataclass(frozen=True)
class Protocol:
    horizons: dict = field(default_factory=lambda: {"45min": 9, "6h": 72, "12h": 144})
    embargo_steps: int = 288                       # 24 h
    first_val_start: pd.Timestamp = pd.Timestamp("2017-11-01 00:00")
    dev_end: pd.Timestamp = pd.Timestamp("2018-06-29 00:00")
    test_start: pd.Timestamp = pd.Timestamp("2018-06-30 00:00")
    n_folds: int = 5
    outage_min_steps: int = 6
    near_outage_steps: int = 12
    spike_log_jump: float = 2.5

    def to_json(self) -> str:
        d = {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in asdict(self).items()}
        return json.dumps(d, indent=2)

P = Protocol()

# ------------------------------------------------------------------ data layer
def load_raw(path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["timestamp", "electron_flux", "proton_flux", "source_file"],
                       parse_dates=["timestamp"])

def _run_length(mask: pd.Series) -> pd.Series:
    run_id = (mask != mask.shift()).cumsum()
    return mask.groupby(run_id).transform("size").where(mask, 0).astype(int)

def _stale_steps(valid: pd.Series) -> pd.Series:
    pos = pd.Series(np.arange(len(valid)), index=valid.index)
    return (pos - pos.where(valid).ffill()).fillna(np.inf)

def to_grid(raw: pd.DataFrame, proto: Protocol = P) -> pd.DataFrame:
    """Snap to 5-min grid, reindex to the FULL grid (no rows dropped), keep raw values, add flags.
    *_noncausal flags use future samples: for choosing SCORED targets only, never model inputs."""
    ts = raw["timestamp"]
    snapped = ts.dt.round("5min")
    if (ts - snapped).abs().dt.total_seconds().max() > 10:
        raise ValueError("timestamp jitter too large to snap")
    if snapped.duplicated().any():
        raise ValueError("duplicate timestamps after snapping")
    if not snapped.is_monotonic_increasing:
        raise ValueError("raw rows not time-ordered")
    s = pd.DataFrame({"electron": raw["electron_flux"].to_numpy(), "proton": raw["proton_flux"].to_numpy()},
                     index=pd.DatetimeIndex(snapped.to_numpy(), name="t"))
    g = s.reindex(pd.date_range(snapped.min(), snapped.max(), freq="5min", name="t"))

    g["proton_zero"] = g["proton"] == 0
    g.loc[g["proton"] <= 0, "proton"] = np.nan
    g.loc[g["electron"] <= 0, "electron"] = np.nan
    g["missing_e"] = g["electron"].isna()
    g["missing_p"] = g["proton"].isna()
    g["missing"] = g["missing_e"] | g["missing_p"]

    g["stale_steps"] = _stale_steps(~g["missing"])                       # CAUSAL (usable as input)

    g["gap_len_noncausal"] = _run_length(g["missing"])                   # NON-CAUSAL flags below
    g["outage_noncausal"] = g["gap_len_noncausal"] >= proto.outage_min_steps
    k = proto.near_outage_steps
    dil = g["outage_noncausal"].astype(float).rolling(2 * k + 1, center=True, min_periods=1).max() > 0
    g["qc_near_outage_noncausal"] = dil & ~g["outage_noncausal"]

    def spike(x):
        lx = np.log(x)
        return ((lx.diff() > proto.spike_log_jump) & ((lx - lx.shift(-1)) > proto.spike_log_jump)).fillna(False)
    g["qc_spike_noncausal"] = spike(g["electron"]) | spike(g["proton"])
    return g

def scoring_mask(g: pd.DataFrame) -> pd.Series:
    """Targets allowed in headline metrics (same mask for every model; also report unmasked)."""
    return ~g["missing"] & ~g["qc_near_outage_noncausal"] & ~g["qc_spike_noncausal"]

def quality_report(g: pd.DataFrame) -> dict:
    m = g["missing"]
    runs = g["gap_len_noncausal"][m].groupby((m != m.shift()).cumsum()[m]).first()
    return {"n_slots": int(len(g)), "start": str(g.index[0]), "end": str(g.index[-1]),
            "missing_slots_any": int(m.sum()), "missing_slots_electron": int(g["missing_e"].sum()),
            "missing_slots_proton": int(g["missing_p"].sum()), "proton_zero_rows": int(g["proton_zero"].sum()),
            "n_missing_runs": int(len(runs)), "longest_missing_run_steps": int(runs.max()) if len(runs) else 0,
            "outage_slots": int(g["outage_noncausal"].sum()), "near_outage_slots": int(g["qc_near_outage_noncausal"].sum()),
            "spike_slots": int(g["qc_spike_noncausal"].sum()), "scorable_slots": int(scoring_mask(g).sum())}

# ------------------------------------------------------------------ splits
@dataclass(frozen=True)
class Window:
    name: str
    val_start: pd.Timestamp      # first allowed origin
    val_end: pd.Timestamp        # exclusive: every TARGET must be < val_end

def make_folds(proto: Protocol = P) -> list[Window]:
    edges = pd.date_range(proto.first_val_start, proto.dev_end, periods=proto.n_folds + 1).round("60min")
    return [Window(f"cv{i + 1}", edges[i], edges[i + 1]) for i in range(proto.n_folds)]

def static_window(proto: Protocol = P) -> Window:
    """Static train/val = the LAST rolling fold (closest to test). Same masks, so static == rolling fold 5."""
    last = make_folds(proto)[-1]
    return Window("static", last.val_start, last.val_end)

def test_window(index: pd.DatetimeIndex, proto: Protocol = P) -> Window:
    return Window("test", proto.test_start, index[-1] + STEP)

def split_masks(index: pd.DatetimeIndex, H: int, w: Window, proto: Protocol = P):
    """Masks over ORIGINS. Split on TARGET time:
         train: target <= val_start - embargo        val: origin >= val_start and target < val_end"""
    if not (index[1:] - index[:-1] == STEP).all():
        raise ValueError("index must be a regular 5-min grid")
    n = len(index)
    in_range = np.arange(n) + H < n
    tgt = index + H * STEP
    train = in_range & (tgt <= w.val_start - proto.embargo_steps * STEP)
    val = in_range & (index >= w.val_start) & (tgt < w.val_end)
    return np.asarray(train), np.asarray(val)

def usable_origins(g, H, require_scorable=False, scorable=None):
    """Origin needs a CURRENT observation; target must be observed (+ scorable mask if requested)."""
    ok = (g["stale_steps"] == 0).to_numpy() & (~g["missing"]).shift(-H, fill_value=False).to_numpy()
    if require_scorable:
        ok &= scorable.shift(-H, fill_value=False).to_numpy()
    return ok

def describe_split(g, scorable, proto: Protocol = P) -> pd.DataFrame:
    rows = []
    for w in make_folds(proto) + [test_window(g.index, proto)]:
        for hn, H in proto.horizons.items():
            tr, va = split_masks(g.index, H, w, proto)
            ok, oks = usable_origins(g, H), usable_origins(g, H, True, scorable)
            tgt = g["electron"].shift(-H)[va & oks]
            rows.append({"window": w.name, "horizon": hn, "val_start": w.val_start, "val_end": w.val_end,
                         "n_train": int((tr & ok).sum()), "n_val_all": int((va & ok).sum()),
                         "n_val_scored": int((va & oks).sum()), "target_median": float(tgt.median()),
                         "target_p95": float(tgt.quantile(.95)), "share_gt_1000": float((tgt > 1000).mean()),
                         "log_var": float(np.log(tgt).var())})
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ leakage tests (perturbation)
class CausalityError(AssertionError):
    pass

MODES = ("scale", "nan", "shuffle", "fill", "crash", "hold")

def default_cuts(df, n_random=6, n_gap=4, seed=0):
    """Random cuts + cuts inside/early in NaN runs + cuts just before/on spikes (where gap/spike leaks show)."""
    rng = np.random.default_rng(seed)
    n = len(df)
    pos = list(rng.integers(300, n - 300, n_random))
    m = df["missing"].to_numpy()
    run_id = np.cumsum(m != np.roll(m, 1))
    runs = [np.where((run_id == r) & m)[0] for r in np.unique(run_id[m])]
    for s in [s for s in runs if len(s) >= 3 and 300 < s[0] < n - 300][:n_gap]:
        pos += [int(s[len(s) // 2]), int(s[1])]
    sp = np.where(df["qc_spike_noncausal"].to_numpy())[0]
    sp = sp[(sp > 300) & (sp < n - 300)]
    pos += [int(p) - 1 for p in sp[:3]] + [int(p) for p in sp[:3]]
    return [df.index[p] for p in sorted(set(pos))]

def perturb_future(df, cut, mode, seed=0, cols=("electron", "proton")):
    rng = np.random.default_rng(seed)
    out = df.copy()
    fut = out.index > cut
    n_f = int(fut.sum())
    for c in cols:
        v = out.loc[fut, c].to_numpy(dtype=float).copy()
        if mode == "scale":
            v = v * rng.uniform(0.2, 5.0, n_f)
        elif mode == "nan":
            v = np.full(n_f, np.nan)
        elif mode == "shuffle":
            rng.shuffle(v)
        elif mode == "fill":
            v = np.where(np.isnan(v), 1.0, v)
        elif mode == "hold":
            past = out.loc[~fut, c].dropna()
            v = np.full(n_f, past.iloc[-1] if len(past) else 1.0)
        elif mode == "crash":
            v = v * 1e-3
        else:
            raise ValueError(mode)
        out.loc[fut, c] = v
    out["missing_e"] = out["electron"].isna()
    out["missing_p"] = out["proton"].isna()
    out["missing"] = out["missing_e"] | out["missing_p"]
    return out

def _differs(a, b):
    if a.dtype == bool or b.dtype == bool:
        return a.astype(bool) != b.astype(bool)
    return pd.Series(~np.isclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-9, atol=1e-12, equal_nan=True),
                     index=a.index)

def find_causal_violations(feature_fn: Callable, df, cuts=None, modes: Iterable[str] = MODES, exclude=()):
    """{feature: first timestamp <= cut whose value changed when the future was destroyed}. Empty dict = causal.
    exclude: label columns that are future-valued by design."""
    cuts = cuts if cuts is not None else default_cuts(df)
    base = feature_fn(df).drop(columns=list(exclude), errors="ignore")
    bad = {}
    for cut in cuts:
        for mode in modes:
            f2 = feature_fn(perturb_future(df, cut, mode)).drop(columns=list(exclude), errors="ignore")
            a, b = base.loc[:cut], f2.loc[:cut]
            for col in a.columns:
                if col not in bad:
                    d = _differs(a[col], b[col])
                    if d.any():
                        bad[col] = d.index[d.to_numpy().argmax()]
    return bad

def assert_causal(feature_fn, df, cuts=None, exclude=()):
    bad = find_causal_violations(feature_fn, df, cuts, exclude=exclude)
    if bad:
        raise CausalityError("features depend on future data: " + ", ".join(f"{k} (first at {v})" for k, v in bad.items()))

def find_fit_violations(fit_fn: Callable, df, train_end, modes: Iterable[str] = MODES):
    """fit_fn(df, train_end) -> {name: array}. Fitted parameters must ignore everything after train_end."""
    base = fit_fn(df, train_end)
    bad = []
    for mode in modes:
        other = fit_fn(perturb_future(df, train_end, mode), train_end)
        for k in base:
            if k not in bad and not np.allclose(np.asarray(base[k], float), np.asarray(other[k], float),
                                                rtol=1e-9, atol=1e-12, equal_nan=True):
                bad.append(k)
    return bad

def assert_fit_independent_of_future(fit_fn, df, train_end):
    bad = find_fit_violations(fit_fn, df, train_end)
    if bad:
        raise CausalityError("fitted parameters depend on data after train_end: " + ", ".join(bad))