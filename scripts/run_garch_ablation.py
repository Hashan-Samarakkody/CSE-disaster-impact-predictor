"""Y1 (ASPI log return): does the new `garch_cond_vol` feature earn its place?

Same walk-forward folds, same per-fold RF-importance feature selection, same SMOGN
augmentation as notebook 04 (see `scripts/run_survival_model.py` for the shared
rationale) -- but this only needs Ridge (this thesis's designated H1 baseline
regressor, and by far the cheapest to refit) to answer the ablation question quickly:
compare the SAME pipeline with `garch_cond_vol` included in the candidate feature pool
vs excluded, everything else identical. A full re-run of every model on every target
(`notebooks/04_modeling_regression.ipynb`) is the authoritative number for the results
table; this script is the fast, targeted check for one feature on one target.

Run after `notebooks/02_features_targets.ipynb` (needs `dataset`, `feature_spec` cached
with `garch_cond_vol` present):

    python scripts/run_garch_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import bootstrap_metric_ci, evaluate_regression, skill_score
from src.sampling.time_aware_smogn import time_aware_smogn
from src.training.walk_forward import generate_walk_forward_splits, purge_horizon_overlap

ARTIFACTS = ROOT / "artifacts"
RANDOM_STATE = 42
TARGET = "Y1_ASPI_5D_Forward_LogReturn_Pct"
TRAIN_WINDOW, TEST_WINDOW, STEP = 30, 10, 10
RIDGE_ALPHAS = np.logspace(-3, 3, 13)


def select_top_features(X_train, y_series, k=20, random_state=RANDOM_STATE):
    ranker = RandomForestRegressor(n_estimators=200, random_state=random_state)
    ranker.fit(X_train, y_series)
    ranked = pd.Series(ranker.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    return ranked.head(min(k, len(ranked))).index.tolist()


def augment_fold(X_train, y_train, dates_train, type_cols, gdp_train=None,
                  max_year_gap=5.0, random_state=RANDOM_STATE):
    complete = y_train.notna().all(axis=1)
    minority_mask = complete & (y_train["Y3_recovery_days"] > 30)
    X_aug, y_aug, report = time_aware_smogn(
        X_train, y_train, minority_mask, event_dates=dates_train,
        max_year_gap=max_year_gap, random_state=random_state,
        type_cols=type_cols, gdp_current_usd=gdp_train,
        max_synthetic_share=0.25, return_report=True,
    )
    return X_aug, y_aug, report


def run(feature_cols, X_all, y_all, dates_all, gdp_all, splits, type_cols, horizon_end_all):
    yt_all, yp_all = [], []
    for s in splits:
        # ASPI_5D_Log_Return_Pct's label reads market prices up to 5 trading days past
        # the event -- purge training events whose label horizon reaches into this
        # fold's test period (fold-boundary embargo, see walk_forward.purge_horizon_overlap).
        s = purge_horizon_overlap(s, dates_all, horizon_end_all)
        X_tr_real, X_te = X_all[feature_cols].iloc[s.train_index], X_all[feature_cols].iloc[s.test_index]
        y_tr_real, y_te = y_all.iloc[s.train_index], y_all.iloc[s.test_index]
        dates_tr = dates_all.iloc[s.train_index]
        gdp_tr = None if gdp_all is None else gdp_all.iloc[s.train_index]

        real_ok, te_ok = y_tr_real[TARGET].notna(), y_te[TARGET].notna()
        if int(real_ok.sum()) < 10 or int(te_ok.sum()) < 1:
            continue

        X_tr_aug, y_tr_aug, _ = augment_fold(X_tr_real, y_tr_real, dates_tr, type_cols, gdp_tr)
        tr_ok = y_tr_aug[TARGET].notna()
        y_tr_t, y_te_t = y_tr_aug.loc[tr_ok, TARGET], y_te.loc[te_ok, TARGET]

        feat_cols = select_top_features(X_tr_aug.loc[tr_ok], y_tr_t, k=20)
        X_tr_sel, X_te_sel = X_tr_aug.loc[tr_ok, feat_cols], X_te.loc[te_ok, feat_cols]

        real_ok_feat = X_tr_real.loc[real_ok, feat_cols]
        y_real_t = y_tr_real.loc[real_ok, TARGET]
        cv_real = TimeSeriesSplit(n_splits=max(2, min(3, len(real_ok_feat) - 1)))

        alpha_search = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS, cv=cv_real))
        alpha_search.fit(real_ok_feat, y_real_t)
        ridge = make_pipeline(
            StandardScaler(), Ridge(alpha=float(alpha_search.named_steps["ridgecv"].alpha_)))
        ridge.fit(X_tr_sel, y_tr_t)
        pred = ridge.predict(X_te_sel)

        yt_all.append(y_te_t.to_numpy())
        yp_all.append(pred)

    yt = np.concatenate(yt_all)
    yp = np.concatenate(yp_all)
    return yt, yp, evaluate_regression(yt, yp)


def main() -> None:
    dataset = pd.read_parquet(ARTIFACTS / "dataset.parquet")
    spec = json.loads((ARTIFACTS / "feature_spec.json").read_text(encoding="utf-8"))
    feature_cols, type_cols = spec["FEATURE_COLS"], spec["TYPE_COLS"]
    if "garch_cond_vol" not in feature_cols:
        raise SystemExit(
            "garch_cond_vol is not in artifacts/feature_spec.json -- re-run "
            "notebooks/02_features_targets.ipynb first.")

    X_all = dataset[feature_cols].fillna(0.0)
    y_all = dataset[["Y1_ASPI_5D_Forward_LogReturn_Pct", "Y3_recovery_days"]].copy()
    dates_all = dataset["event_date"]
    horizon_end_all = dataset["Y1_horizon_end_date"]
    gdp_all = dataset["gdp_current_usd"] if "gdp_current_usd" in dataset.columns else None
    splits = list(generate_walk_forward_splits(len(X_all), TRAIN_WINDOW, TEST_WINDOW, STEP))

    with_garch = feature_cols
    without_garch = [c for c in feature_cols if c != "garch_cond_vol"]

    yt_w, yp_w, m_w = run(with_garch, X_all, y_all, dates_all, gdp_all, splits, type_cols, horizon_end_all)
    yt_wo, yp_wo, m_wo = run(without_garch, X_all, y_all, dates_all, gdp_all, splits, type_cols, horizon_end_all)

    print("=" * 100)
    print("Y1_ASPI_5D_Forward_LogReturn_Pct (ASPI_5D_Log_Return_Pct) -- Ridge, with vs without garch_cond_vol "
          "(identical folds/SMOGN/selection, purged for label-horizon/fold-boundary overlap)")
    print("=" * 100)
    print(f"{'variant':>18s} {'n':>4s} {'RMSE':>9s} {'MAE':>9s} {'pooled_R2':>10s}")
    print(f"{'without_garch':>18s} {len(yt_wo):>4d} {m_wo['rmse']:>9.5f} {m_wo['mae']:>9.5f} {m_wo['r2']:>10.5f}")
    print(f"{'with_garch':>18s} {len(yt_w):>4d} {m_w['rmse']:>9.5f} {m_w['mae']:>9.5f} {m_w['r2']:>10.5f}")

    def rmse_fn(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    n = min(len(yt_w), len(yt_wo))
    _, lo, hi = bootstrap_metric_ci(
        yt_w[:n], yp_w[:n], lambda a, b: rmse_fn(a, yp_wo[:n]) - rmse_fn(a, b))
    print(f"\nskill of adding garch_cond_vol (RMSE) = {skill_score(m_w['rmse'], m_wo['rmse']):+.5f}  "
          f"delta_rmse={m_wo['rmse'] - m_w['rmse']:+.5f}  CI=[{lo:+.5f}, {hi:+.5f}]  "
          f"({'CI excludes zero' if lo > 0 or hi < 0 else 'not distinguishable from zero'})")


if __name__ == "__main__":
    main()
