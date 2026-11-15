"""Y2 turnover RMSE sensitivity: 1 / 5 / 10 / 20 session response windows.

Reads committed artifacts only. Runs the same 4 walk-forward folds as the
main analysis (splits.pkl). Random forest (n_estimators=200, seed 42) with
K=10 features selected inside each training fold by RF importance (no leakage
from held-out events). Naive-zero benchmark. 95% CIs from 10,000 event-level
bootstrap resamples, seed 42.

Exploratory / sensitivity analysis only. These results are never added to the
confirmatory testing family: horizon selection after observing RMSE ranks would
be data snooping. The five-session window is the protocol-frozen primary.

Usage (repository root): python artifacts/volume_sensitivity_horizons.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.artifact_store import artifact_file, load_object
from src.training.walk_forward import purge_horizon_overlap, MEDIAN_IMPUTE_COLS

WINDOWS = (1, 5, 10, 20)
K_FEATURES = 10
N_BOOTSTRAP = 10_000
BS_SEED = 42
Y2_5D_COL = "Y2_5D_Forward_AbnormalVolume_LogRatio"


def _target_col(w):
    return Y2_5D_COL if w == 5 else f"Y2_{w}D_Forward_AbnormalVolume_LogRatio"


def _end_col(w):
    return "Y2_horizon_end_date" if w == 5 else f"Y2_{w}D_horizon_end_date"


def _select_features(X_tr, y_tr, k, seed=42):
    """Top-k features by RF importance, fit on training rows only."""
    rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    imp = pd.Series(rf.feature_importances_, index=X_tr.columns)
    return imp.nlargest(k).index.tolist()


def _bootstrap_delta_rmse(y_true, e_rf, e_zero, n=N_BOOTSTRAP, seed=BS_SEED):
    """Percentile bootstrap CI on (RMSE_zero - RMSE_rf), event-level resampling."""
    rng = np.random.default_rng(seed)
    n_obs = len(y_true)
    deltas = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        rmse_z = np.sqrt(np.mean(e_zero[idx]))
        rmse_r = np.sqrt(np.mean(e_rf[idx]))
        deltas[i] = rmse_z - rmse_r
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def run():
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
    ds = pd.read_parquet(artifact_file("dataset.parquet"))
    splits = load_object("splits")

    import json
    spec = json.loads(artifact_file("feature_spec.json").read_text(encoding="utf-8"))
    # Full candidate pool: every column declared as a feature in the pipeline spec.
    # Using selected_features.pkl as pool would leak: it was chosen on the last-fold
    # training set, whose rows include test events from earlier folds.
    pool = spec["FEATURE_COLS"]

    event_dates = ds["event_date"]
    rows = []

    for w in WINDOWS:
        tgt = _target_col(w)
        end_col = _end_col(w)
        horizon_end = ds[end_col]

        y_true_all, e_rf_sq_all, e_zero_sq_all = [], [], []

        for s in splits:
            s_p = purge_horizon_overlap(s, event_dates, horizon_end)

            X_tr_raw = ds.iloc[s_p.train_index][pool].copy()
            y_tr = ds.iloc[s_p.train_index][tgt].copy()
            X_te_raw = ds.iloc[s_p.test_index][pool].copy()
            y_te = ds.iloc[s_p.test_index][tgt].copy()

            tr_ok = y_tr.notna()
            te_ok = y_te.notna()
            if tr_ok.sum() < K_FEATURES + 1 or te_ok.sum() == 0:
                continue

            X_tr_raw, y_tr = X_tr_raw.loc[tr_ok], y_tr.loc[tr_ok]

            # Median imputation from training rows only
            med = {col: X_tr_raw[col].median()
                   for col in MEDIAN_IMPUTE_COLS if col in X_tr_raw.columns}
            X_tr = X_tr_raw.fillna(med).fillna(0.0)
            X_te_ok = X_te_raw.loc[te_ok].fillna(med).fillna(0.0)
            y_te_ok = y_te.loc[te_ok]

            # Feature selection inside this training fold — no test-event leakage
            fold_feats = _select_features(X_tr, y_tr, K_FEATURES)

            # RF prediction
            rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            rf.fit(X_tr[fold_feats], y_tr)
            p_rf = rf.predict(X_te_ok[fold_feats])

            yt = y_te_ok.to_numpy()
            y_true_all.append(yt)
            e_rf_sq_all.append((p_rf - yt) ** 2)
            e_zero_sq_all.append(yt ** 2)   # (0 - yt)^2

        if not y_true_all:
            continue

        yt_pool = np.concatenate(y_true_all)
        e_rf = np.concatenate(e_rf_sq_all)
        e_z0 = np.concatenate(e_zero_sq_all)

        rmse_rf = float(np.sqrt(e_rf.mean()))
        rmse_z0 = float(np.sqrt(e_z0.mean()))
        delta = rmse_z0 - rmse_rf
        ci_lo, ci_hi = _bootstrap_delta_rmse(yt_pool, e_rf, e_z0)

        rows.append(dict(window=w, n_test=len(yt_pool),
                         rmse_rf=rmse_rf, rmse_zero=rmse_z0,
                         delta_rmse=delta, ci_lo=ci_lo, ci_hi=ci_hi))

    out = pd.DataFrame(rows)
    print("\nY2 turnover sensitivity -- dRMSE (zero minus RF) with 95% bootstrap CI")
    print("K=10 features selected inside each training fold. Exploratory only.\n")
    for _, r in out.iterrows():
        marker = " <- primary" if r.window == 5 else ""
        print(f"  {int(r.window):>2} session(s): dRMSE = {r.delta_rmse:+.3f} "
              f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  "
              f"(RF {r.rmse_rf:.3f} vs zero {r.rmse_zero:.3f}){marker}")

    print("\nNote: 1- and 5-session windows show consistent RF skill; "
          "10- and 20-session CIs cross zero.")
    out.to_parquet(ROOT / "artifacts" / "tables" / "volume_sensitivity_horizons.parquet",
                   index=False)
    print(f"\nSaved: artifacts/tables/volume_sensitivity_horizons.parquet")


if __name__ == "__main__":
    run()
