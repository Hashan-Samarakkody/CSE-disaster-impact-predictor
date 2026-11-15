"""Y2 turnover RMSE sensitivity: 1 / 5 / 10 / 20 session response windows.

Reads committed artifacts only. Runs the same 4 walk-forward folds as the main
analysis (splits.pkl). Random forest (n_estimators=200, seed 42, no hyperparameter
search) with the K=10 features selected for Y2_5D in the primary run. Naive-zero
and naive-train-mean benchmarks for comparison.

EXPLORATORY / SENSITIVITY only. These results are never added to the confirmatory
Holm family: selecting the horizon after seeing RMSE ranks is data snooping. The
5-session window stays primary, as pre-registered.

Usage (repository root): python artifacts/volume_sensitivity_horizons.py
"""
import pickle
import sys
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
Y2_5D_COL = "Y2_5D_Forward_AbnormalVolume_LogRatio"

def _target_col(w):
    return Y2_5D_COL if w == 5 else f"Y2_{w}D_Forward_AbnormalVolume_LogRatio"

def _end_col(w):
    return "Y2_horizon_end_date" if w == 5 else f"Y2_{w}D_horizon_end_date"


def run():
    ds = pd.read_parquet(artifact_file("dataset.parquet"))
    splits = load_object("splits")
    sel = load_object("selected_features")
    feat_cols = sel[Y2_5D_COL]       # K=10 features from primary Y2_5D run

    feat_cols = sel[Y2_5D_COL]       # K=10 features; re-assign to be sure

    event_dates = ds["event_date"]
    rows = []

    for w in WINDOWS:
        tgt = _target_col(w)
        end_col = _end_col(w)
        horizon_end = ds[end_col]

        y_true_all, y_pred_rf_all, y_pred_z0_all, y_pred_tm_all = [], [], [], []

        for s in splits:
            # Purge: drop training rows whose label bleeds into test period
            s_p = purge_horizon_overlap(s, event_dates, horizon_end)

            X_tr = ds.iloc[s_p.train_index][feat_cols].copy()
            y_tr = ds.iloc[s_p.train_index][tgt].copy()
            X_te = ds.iloc[s_p.test_index][feat_cols].copy()
            y_te = ds.iloc[s_p.test_index][tgt].copy()

            # Keep only rows where target is observed
            tr_ok = y_tr.notna()
            te_ok = y_te.notna()
            if tr_ok.sum() < 5 or te_ok.sum() == 0:
                continue

            X_tr, y_tr = X_tr.loc[tr_ok], y_tr.loc[tr_ok]

            # Median imputation from training rows only
            med = {}
            for col in MEDIAN_IMPUTE_COLS:
                if col in X_tr.columns:
                    med[col] = X_tr[col].median()
            X_tr = X_tr.fillna(med).fillna(0.0)
            X_te_ok = X_te.loc[te_ok].fillna(med).fillna(0.0)
            y_te_ok = y_te.loc[te_ok]

            # Naive baselines
            z0 = np.zeros(len(y_te_ok))
            tm = np.full(len(y_te_ok), y_tr.mean())

            # RF fit
            rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            p_rf = rf.predict(X_te_ok)

            y_true_all.append(y_te_ok.to_numpy())
            y_pred_rf_all.append(p_rf)
            y_pred_z0_all.append(z0)
            y_pred_tm_all.append(tm)

        if not y_true_all:
            continue

        yt = np.concatenate(y_true_all)
        rows.append({
            "window": w,
            "n_test": len(yt),
            "rmse_rf": np.sqrt(mean_squared_error(yt, np.concatenate(y_pred_rf_all))),
            "rmse_zero": np.sqrt(mean_squared_error(yt, np.concatenate(y_pred_z0_all))),
            "rmse_train_mean": np.sqrt(mean_squared_error(yt, np.concatenate(y_pred_tm_all))),
        })

    out = pd.DataFrame(rows)
    out["delta_vs_zero"] = out["rmse_zero"] - out["rmse_rf"]
    out["pct_improvement"] = 100 * out["delta_vs_zero"] / out["rmse_zero"]
    print("\nY2 turnover RMSE by response window (exploratory sensitivity, same 4 folds, RF K=10 features no HPO):\n")
    print(out.round(4).to_string(index=False))
    print("\nNote: w=5 is the pre-registered primary. Results at w=1/10/20 are sensitivity only.")
    out.to_parquet(ROOT / "artifacts" / "tables" / "volume_sensitivity_horizons.parquet", index=False)
    print(f"\nSaved: artifacts/tables/volume_sensitivity_horizons.parquet")


if __name__ == "__main__":
    run()
