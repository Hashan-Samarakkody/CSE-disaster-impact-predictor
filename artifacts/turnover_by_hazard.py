"""Turnover (Y2, five sessions) out-of-fold RMSE by hazard group: random forest vs zero benchmark.
Reads only committed artifacts: artifacts/models/results_regression.pkl and artifacts/tables/dataset.parquet.
Interval: percentile bootstrap, 10,000 event-level resamples within each group, seed 42 (exploratory, unadjusted).
Usage (repository root): python scripts/build_turnover_by_hazard.py
"""
import pickle
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
r = pickle.load(open(ROOT / "artifacts/models/results_regression.pkl", "rb"))
ds = pd.read_parquet(ROOT / "artifacts/tables/dataset.parquet")
T = "Y2_5D_Forward_AbnormalVolume_LogRatio"
def oof(m):
    v = r[m][T]
    return (np.concatenate(v["index"]), np.concatenate(v["y_true"]), np.concatenate(v["y_pred"]))
idx, y, p_rf = oof("random_forest"); idx0, _, p_0 = oof("naive_zero")
assert (idx == idx0).all()
typ = ds["disaster_type"].values[idx]
rng = np.random.default_rng(42); rows = []
for name, g in [("Flood", typ == "Flood"), ("Non-flood", typ != "Flood")]:
    e1, e0 = (p_rf[g] - y[g]) ** 2, (p_0[g] - y[g]) ** 2; n = int(g.sum())
    d = np.sqrt(e0.mean()) - np.sqrt(e1.mean())
    bs = [np.sqrt(e0[i].mean()) - np.sqrt(e1[i].mean()) for i in (rng.integers(0, n, n) for _ in range(10000))]
    rows.append(dict(group=name, n=n, rmse_rf=np.sqrt(e1.mean()), rmse_zero=np.sqrt(e0.mean()), delta_rmse=d,
                     ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5)))
out = pd.DataFrame(rows); print(out.round(3).to_string(index=False))
out.to_parquet(ROOT / "artifacts/tables/turnover_by_hazard.parquet", index=False)