# src/

`core.py`, `runio.py` and `evalkit.py` are **exact copies** of the three files that cells 2, 4 and 14 of the
notebook write to disk (only the `%%writefile` line was removed). They hold the frozen protocol, the data
grid and leak tests, the loaders, and the shared evaluation kit.

Things to know before reusing them:
- Paths are Kaggle paths (`/kaggle/working/final_run`). Change `RUN` in `runio.py` to run elsewhere.
- Imports are flat (`from core import P`), so put this folder on `sys.path` rather than installing it.
- `core.py` still carries the *original* horizons (45 min / 6 h / 12 h). The 30 min / 45 min / 1 h results use
  `HZ = {"30min": 6, "45min": 9, "1h": 12}` defined inside the later notebook cells.
- The notebook remains the source of truth for the published results.

Planned: turn these into an importable package, add tests, and move the model code out of the notebook (see `docs/roadmap.md`).
