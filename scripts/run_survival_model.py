"""Y3 (recovery days): the right-censored AFT survival model is the PRIMARY reported
result for Y3 (methodology-audit followup, item 12) -- plain point-regression (RF/
Ridge/XGBoost fit directly on Y3_recovery_days, as if every 90-day cap and
competing-risk censor were an observed recovery time) is a SECONDARY diagnostic here,
not a co-equal alternative. Treating >=90-day non-recoveries as if they equalled 90 and
minimising squared error against that fabricated observation is exactly the distortion
survival analysis exists to avoid; c-index and the survival curve are what this study
actually stands behind for Y3, and the point-regression comparison exists only to show
the SIZE of that distortion, not to compete with the survival number for primacy.

Mirrors `notebooks/04_modeling_regression.ipynb` §`run_walk_forward` for a single
target (Y3_recovery_days) so the numbers are directly comparable to
`docs/RESULTS_AUDIT.txt`: same cached `dataset.parquet`/`feature_spec.json`, same
(train_window, test_window, step) = (30, 10, 10) split, same per-fold SMOGN
augmentation (`time_aware_smogn`) and per-fold RF-importance top-20 feature selection.
The only thing that differs is the estimator itself and how it treats the 90-day cap
(right-censored, not an observed value) -- see `src/models/survival_recovery.py` and
`docs/METHODOLOGY_AUDIT.md` ("Improvement attempts") for why.

Run after the full notebook pipeline (needs `dataset`, `feature_spec` cached):

    python scripts/run_survival_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_pipeline.feature_eng import FeatureEngineeringConfig
from src.evaluation.metrics import bootstrap_metric_ci, evaluate_regression, skill_score
from src.models.survival_recovery import AFTRecoveryModel
from src.sampling.time_aware_smogn import time_aware_smogn
from src.training.walk_forward import (MEDIAN_IMPUTE_COLS, generate_walk_forward_splits,
                                       median_impute_from_train, purge_horizon_overlap)

ARTIFACTS = ROOT / "artifacts"
RANDOM_STATE = 42
TARGET = "Y3_recovery_days"
CAP = FeatureEngineeringConfig().max_recovery_days
TRAIN_WINDOW, TEST_WINDOW, STEP = 30, 10, 10  # identical to notebook 02's cached `splits`


def select_top_features(X_train, y_series, k=20, random_state=RANDOM_STATE):
    """Same RF-importance ranking as notebook 04's `select_top_features`."""
    ranker = RandomForestRegressor(n_estimators=200, random_state=random_state)
    ranker.fit(X_train, y_series)
    ranked = pd.Series(ranker.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    return ranked.head(min(k, len(ranked))).index.tolist()


def augment_fold(X_train, y_train, dates_train, type_cols, gdp_train=None,
                  max_year_gap=5.0, random_state=RANDOM_STATE):
    """Same SMOGN call as notebook 04's `augment_fold`, single-target y_train frame."""
    complete = y_train.notna().all(axis=1)
    minority_mask = complete & (y_train[TARGET] > 30)
    X_aug, y_aug, report = time_aware_smogn(
        X_train, y_train, minority_mask, event_dates=dates_train,
        max_year_gap=max_year_gap, random_state=random_state,
        type_cols=type_cols, gdp_current_usd=gdp_train,
        max_synthetic_share=0.25, return_report=True,
    )
    return X_aug, y_aug, report


def main() -> None:
    dataset = pd.read_parquet(ARTIFACTS / "dataset.parquet")
    spec = json.loads((ARTIFACTS / "feature_spec.json").read_text(encoding="utf-8"))
    feature_cols, type_cols = spec["FEATURE_COLS"], spec["TYPE_COLS"]

    # Flagged columns (methodology-audit finding #14, MEDIAN_IMPUTE_COLS) keep real NaN
    # here -- imputed per fold from train rows only, below. Everything else is a
    # safety-net zero-fill (none of them actually have missing values).
    X_all = dataset[feature_cols].copy()
    _non_median_cols = [c for c in feature_cols if c not in MEDIAN_IMPUTE_COLS]
    X_all[_non_median_cols] = X_all[_non_median_cols].fillna(0.0)
    y_all = dataset[[TARGET]].copy()
    dates_all = dataset["event_date"]
    gdp_all = dataset["gdp_current_usd"] if "gdp_current_usd" in dataset.columns else None
    # Y3's label is only "known" once recovery is confirmed (or the 90-day cap is
    # confirmed) -- see feature_eng.build_targets. This script had NO fold-boundary
    # purge at all before the 2026-09-16 methodology-audit review (finding #5): a
    # training row's Y3 label can depend on prices up to 90 trading days after its
    # event, so without this a training event close to a fold boundary could leak
    # price information from on/after the first test event.
    y3_label_end_all = dataset["Y3_label_end_date"]

    splits = list(generate_walk_forward_splits(len(X_all), TRAIN_WINDOW, TEST_WINDOW, STEP))

    store_aft = {"y_true": [], "y_pred": [], "folds": [], "censored": []}
    store_naive_zero = {"y_true": [], "y_pred": [], "folds": []}
    store_naive_mean = {"y_true": [], "y_pred": [], "folds": []}
    cidx_folds = []
    fold_detail = []

    for fold_i, s in enumerate(splits):
        s = purge_horizon_overlap(s, dates_all, y3_label_end_all)
        X_tr_real, X_te = X_all.iloc[s.train_index], X_all.iloc[s.test_index]
        y_tr_real, y_te = y_all.iloc[s.train_index], y_all.iloc[s.test_index]
        dates_tr = dates_all.iloc[s.train_index]
        gdp_tr = None if gdp_all is None else gdp_all.iloc[s.train_index]
        X_tr_real, X_te = median_impute_from_train(X_tr_real, X_te)

        real_ok, te_ok = y_tr_real[TARGET].notna(), y_te[TARGET].notna()
        if int(real_ok.sum()) < 10 or int(te_ok.sum()) < 1:
            continue

        X_tr_aug, y_tr_aug, _ = augment_fold(X_tr_real, y_tr_real, dates_tr, type_cols, gdp_tr)
        tr_ok = y_tr_aug[TARGET].notna()

        # Competing-risk censoring (methodology-audit finding #7): `dataset["Y3_censored"]`
        # is the real indicator (True for BOTH the 90-day cap and a later qualifying
        # disaster striking first), not `y >= CAP` alone. `time_aware_smogn` concatenates
        # real rows first (in X_tr_real's order) then appends synthetic rows with a fresh
        # RangeIndex (`ignore_index=True`, see src/sampling/time_aware_smogn.py), so the
        # first `len(X_tr_real)` rows of X_tr_aug/y_tr_aug are exactly the real training
        # rows in that order -- real censoring is looked up for those; synthetic rows
        # (SMOGN interpolates a numeric duration, not a competing-risk event) keep the
        # old `y >= CAP` inference, since SMOGN has no way to know about a later disaster.
        n_real_tr = len(X_tr_real)
        real_censored_tr = dataset["Y3_censored"].loc[X_tr_real.index].to_numpy()
        synthetic_censored_tr = y_tr_aug[TARGET].to_numpy()[n_real_tr:] >= CAP
        censored_tr_full = pd.Series(
            np.concatenate([real_censored_tr, synthetic_censored_tr]), index=y_tr_aug.index)

        y_tr_t, y_te_t = y_tr_aug.loc[tr_ok, TARGET], y_te.loc[te_ok, TARGET]
        censored_tr = censored_tr_full[tr_ok].to_numpy()
        censored_te = dataset["Y3_censored"].loc[y_te_t.index].to_numpy()
        feat_cols = select_top_features(X_tr_aug.loc[tr_ok], y_tr_t, k=20)
        X_tr_sel = X_tr_aug.loc[tr_ok, feat_cols].to_numpy()
        X_te_sel = X_te.loc[te_ok, feat_cols].to_numpy()

        aft = AFTRecoveryModel(cap=CAP).fit(X_tr_sel, y_tr_t.to_numpy(), censored=censored_tr)
        pred_aft = aft.predict(X_te_sel)

        store_aft["y_true"].append(y_te_t.to_numpy())
        store_aft["y_pred"].append(pred_aft)
        store_aft["folds"].append(fold_i)
        store_aft["censored"].append(censored_te)
        y_te_arr = y_te_t.to_numpy()
        fold_censored = int(censored_te.sum())
        fold_detail.append({
            "fold": fold_i, "n_test": len(y_te_arr), "n_censored_test": fold_censored,
            "n_uncensored_test": len(y_te_arr) - fold_censored,
            "rmse": float(np.sqrt(np.mean((y_te_arr - pred_aft) ** 2))),
            "mae": float(np.mean(np.abs(y_te_arr - pred_aft))),
        })
        if not aft.degenerate_ and aft.model_ is not None:
            cidx = aft.concordance_index(X_te_sel, y_te_arr, censored=censored_te)
            cidx_folds.append(cidx)
            fold_detail[-1]["c_index"] = cidx
        else:
            fold_detail[-1]["c_index"] = float("nan")

        train_mean = float(y_tr_real[TARGET].dropna().mean())
        store_naive_zero["y_true"].append(y_te_t.to_numpy())
        store_naive_zero["y_pred"].append(np.zeros(len(y_te_t)))
        store_naive_mean["y_true"].append(y_te_t.to_numpy())
        store_naive_mean["y_pred"].append(np.full(len(y_te_t), train_mean))

    yt_aft = np.concatenate(store_aft["y_true"])
    yp_aft = np.concatenate(store_aft["y_pred"])
    censored_aft = np.concatenate(store_aft["censored"])
    yp_zero = np.concatenate(store_naive_zero["y_pred"])
    yp_mean = np.concatenate(store_naive_mean["y_pred"])

    m_aft = evaluate_regression(yt_aft, yp_aft)
    m_zero = evaluate_regression(yt_aft, yp_zero)
    m_mean = evaluate_regression(yt_aft, yp_mean)

    def rmse_fn(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    _, lo_z, hi_z = bootstrap_metric_ci(
        yt_aft, yp_aft, lambda a, b: rmse_fn(a, yp_zero[: len(a)]) - rmse_fn(a, b))
    _, lo_m, hi_m = bootstrap_metric_ci(
        yt_aft, yp_aft, lambda a, b: rmse_fn(a, yp_mean[: len(a)]) - rmse_fn(a, b))

    print("=" * 100)
    print(f"Y3_recovery_days -- AFT survival model vs point-regression baselines "
          f"(n={len(yt_aft)} pooled test points)")
    print("=" * 100)
    print(f"{'model':>20s} {'n':>4s} {'RMSE':>9s} {'MAE':>9s} {'pooled_R2':>10s}")
    print(f"{'naive_zero':>20s} {len(yt_aft):>4d} {m_zero['rmse']:>9.4f} {m_zero['mae']:>9.4f} {m_zero['r2']:>10.5f}")
    print(f"{'naive_train_mean':>20s} {len(yt_aft):>4d} {m_mean['rmse']:>9.4f} {m_mean['mae']:>9.4f} {m_mean['r2']:>10.5f}")
    print(f"{'aft_survival':>20s} {len(yt_aft):>4d} {m_aft['rmse']:>9.4f} {m_aft['mae']:>9.4f} {m_aft['r2']:>10.5f}")
    print()
    print(f"skill_vs_zero (RMSE)       = {skill_score(m_aft['rmse'], m_zero['rmse']):+.5f}  "
          f"delta_rmse={m_zero['rmse'] - m_aft['rmse']:+.4f}  CI=[{lo_z:+.4f}, {hi_z:+.4f}]")
    print(f"skill_vs_train_mean (RMSE) = {skill_score(m_aft['rmse'], m_mean['rmse']):+.5f}  "
          f"delta_rmse={m_mean['rmse'] - m_aft['rmse']:+.4f}  CI=[{lo_m:+.4f}, {hi_m:+.4f}]")
    print()
    if cidx_folds:
        print(f"Harrell's C-index (censoring-aware; the metric this model is actually "
              f"fit to optimise), per-fold: {[f'{c:.3f}' for c in cidx_folds]}, "
              f"mean={np.mean(cidx_folds):.3f} std={np.std(cidx_folds):.3f} "
              f"(0.5 = chance, only >0.5 is a real signal)")
    else:
        print("No fold produced a non-degenerate AFT fit -- see AFTRecoveryModel.degenerate_.")

    # Real competing-risk censoring indicator (methodology-audit finding #7), not
    # `y >= CAP` alone -- a row censored by a later qualifying disaster has y < CAP but
    # is still not an observed recovery.
    n_censored_total = int(censored_aft.sum())
    n_cap_only = int((yt_aft >= CAP).sum())
    print()
    print(f"censoring proportion (pooled test points): {n_censored_total}/{len(yt_aft)} "
          f"= {n_censored_total / len(yt_aft):.1%} (of which {n_cap_only} hit the 90-day "
          f"cap; {n_censored_total - n_cap_only} were censored earlier by a later "
          f"qualifying disaster)")
    print(f"uncensored recoveries (pooled test points): {len(yt_aft) - n_censored_total}")

    print()
    print("per-fold detail:")
    detail_df = pd.DataFrame(fold_detail)
    print(detail_df.to_string(index=False))

    uncensored_mask = ~censored_aft
    if uncensored_mask.sum() >= 6:
        yt_u, yp_u = yt_aft[uncensored_mask], yp_aft[uncensored_mask]
        corr = float(np.corrcoef(yt_u, yp_u)[0, 1])
        # duplicates="drop" can collapse fewer than 3 distinct bin edges when many
        # predictions tie (e.g. a near-degenerate fold) -- pandas then rejects the
        # fixed 3-label list outright, so fall back to pandas' own integer bin labels
        # rather than crash the whole report over a display label.
        try:
            terciles = pd.qcut(yp_u, q=3, labels=["low_pred", "mid_pred", "high_pred"],
                               duplicates="drop")
        except ValueError:
            terciles = pd.qcut(yp_u, q=3, duplicates="drop")
        calib = pd.DataFrame({"actual": yt_u, "predicted": yp_u, "bucket": terciles}).groupby(
            "bucket", observed=True)[["actual", "predicted"]].mean()
        print()
        print(f"calibration (uncensored only, n={uncensored_mask.sum()}): "
              f"Pearson r(actual, predicted median) = {corr:+.3f}")
        print(calib.to_string())
    else:
        print("\ncalibration: fewer than 6 uncensored pooled points -- not attempted.")


if __name__ == "__main__":
    main()
