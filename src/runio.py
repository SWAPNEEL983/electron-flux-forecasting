"""Load the saved run (grid + frozen splits) without recomputing anything."""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd

RUN = Path("/kaggle/working/final_run")
WINDOWS = ["cv1", "cv2", "cv3", "cv4", "cv5", "static", "test"]    # 'test' is sealed: do not score it yet

def load_grid(run=RUN):
    g = pd.read_csv(run / "data" / "grasp_5min_grid.csv.gz", index_col="t", parse_dates=["t"])
    man = json.loads((run / "reports" / "manifest.json").read_text())
    assert len(g) == man["n_slots"] and str(g.index[0]) == man["start"] and str(g.index[-1]) == man["end"], "grid != manifest"
    md5 = hashlib.md5(pd.util.hash_pandas_object(g[["electron", "proton"]]).values.tobytes()).hexdigest()
    if md5 != man["grid_md5"]:
        print("WARNING: grid md5 differs from manifest (csv round-trip?):", md5, "vs", man["grid_md5"])
    return g

def load_split(window, horizon, run=RUN):
    """Integer positions into the grid: 'train', 'val' (all usable), 'val_scored' (headline metrics)."""
    z = np.load(run / "splits" / f"{window}_{horizon}.npz")
    return {k: z[k] for k in z.files}