"""Per-event direction forecast predictions — item A.6.

Re-runs the same direction analysis as run_aspi_return_grid.py but emits two artifacts:

  aspi_direction_predictions.parquet  — one row per (event, horizon, model)
  aspi_direction_diagnostics.parquet  — one row per (horizon, model) cell with full metrics
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.collinearity import CollinearityRFTopK
from src.evaluation.verification import build_episode_ids, paired_bootstrap_delta
from src.targets.return_horizons import HORIZONS, horizon_col, horizon_end_col, information_sets
from src.training.walk_forward import (MEDIAN_IMPUTE_COLS, generate_walk_forward_splits,
                                       median_impute_from_train, purge_horizon_overlap)
from src.utils.artifact_store import artifact_file, save_frame

RANDOM_STATE = 42
TRAIN_WINDOW, TEST_WINDOW, STEP = 30, 10, 10
K = 10   # same as run_aspi_return_grid.run_direction_analysis


def _cluster_auc_ci(y, p, cluster_ids, n_boot=2000):
    rng = np.random.default_rng(RANDOM_STATE)
    clusters = np.unique(cluster_ids)
    members = [np.flatnonzero(cluster_ids == c) for c in clusters]
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(clusters), size=len(clusters))
        i = np.concatenate([members[c] for c in pick])
        if len(np.unique(y[i])) < 2:
            continue
        draws.append(roc_auc_score(y[i], p[i]))
    draws = np.asarray(draws)
    if len(draws) < 100:
        return float("nan"), float("nan"), draws
    lo, hi = (float(v) for v in np.quantile(draws, [0.025, 0.975]))
    return lo, hi, draws


def _calibration_slope(y_true, y_prob):
    """Simple logistic calibration slope via logit(y_prob) ~ intercept + slope*logit(y_true)."""
    from scipy.special import logit
    from scipy.stats import linregress
    mask = (y_prob > 0) & (y_prob < 1)
    if mask.sum() < 5:
        return float("nan"), float("nan")
    lp = logit(np.clip(y_prob[mask], 1e-6, 1-1e-6))
    # regress logit(predicted) on outcome (Platt calibration slope)
    slope, intercept, *_ = linregress(lp, y_true[mask].astype(float))
    return float(slope), float(intercept)


def main():
    data = pd.read_parquet(artifact_file("dataset.parquet")).reset_index(drop=True)
    event_dates = pd.to_datetime(data["event_date"])
    feature_spec = json.loads(artifact_file("feature_spec.json").read_text())
    feature_cols = feature_spec["FEATURE_COLS"]
    # combined info set = all features
    sets = information_sets(feature_cols)
    combined_cols = [c for c in sets.get("combined", feature_cols) if c in data.columns]
    # fallback: use all feature cols
    if not combined_cols:
        combined_cols = [c for c in feature_cols if c in data.columns]

    splits = list(generate_walk_forward_splits(len(data), TRAIN_WINDOW, TEST_WINDOW, STEP))
    episode_ids_all = build_episode_ids(event_dates, gap_days=14)

    pred_rows = []   # per-event predictions
    diag_rows = []   # per cell diagnostics

    for h in HORIZONS:
        ycol, endcol = horizon_col(h), horizon_end_col(h)
        if ycol not in data.columns:
            continue

        pooled = {"logistic": [], "random_forest": []}
        truth_all, idx_all, fold_all = [], [], []

        for fold_i, split in enumerate(splits):
            purged = purge_horizon_overlap(split, event_dates, data[endcol])
            tr = purged.train_index[data[ycol].iloc[purged.train_index].notna().to_numpy()]
            te = split.test_index[data[ycol].iloc[split.test_index].notna().to_numpy()]
            if len(tr) < 10 or len(te) == 0:
                continue
            lab_tr = (data[ycol].iloc[tr].to_numpy() < 0).astype(int)
            lab_te = (data[ycol].iloc[te].to_numpy() < 0).astype(int)
            if len(np.unique(lab_tr)) < 2:
                continue

            X_tr, X_te = median_impute_from_train(
                data.loc[tr, combined_cols], data.loc[te, combined_cols],
                cols=[c for c in MEDIAN_IMPUTE_COLS if c in combined_cols])
            X_tr, X_te = X_tr.fillna(0.0), X_te.fillna(0.0)
            sel = CollinearityRFTopK(k=K, random_state=RANDOM_STATE).fit(X_tr, lab_tr)
            Xs_tr, Xs_te = sel.transform(X_tr), sel.transform(X_te)

            fold_probs = {}
            for name, clf in (
                ("logistic", Pipeline([("s", StandardScaler()),
                                       ("m", LogisticRegression(max_iter=5000))])),
                ("random_forest", RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE)),
            ):
                clf.fit(Xs_tr, lab_tr)
                probs = clf.predict_proba(Xs_te)[:, 1]
                fold_probs[name] = probs
                pooled[name].append(probs)

            for j, row_idx in enumerate(te):
                ep_id = int(episode_ids_all[row_idx])
                for name, probs in fold_probs.items():
                    y_prob = float(probs[j])
                    y_pred = int(y_prob >= 0.5)
                    pred_rows.append({
                        "horizon": h,
                        "model": name,
                        "fold": fold_i,
                        "row": int(row_idx),
                        "event_id": int(row_idx),
                        "y_true": int(lab_te[j]),
                        "y_prob": y_prob,
                        "y_pred": y_pred,
                        "episode_id": ep_id,
                    })

            truth_all.append(lab_te)
            idx_all.append(te)
            fold_all.append(np.full(len(te), fold_i, dtype=int))

        if not truth_all:
            continue
        y = np.concatenate(truth_all)
        idx_all_arr = np.concatenate(idx_all)
        episodes = episode_ids_all[idx_all_arr]

        for name, chunks in pooled.items():
            p = np.concatenate(chunks)
            pred = (p >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
            lo, hi, draws = _cluster_auc_ci(y, p, episodes)
            brier = float(brier_score_loss(y, p))
            cal_slope, cal_intercept = _calibration_slope(y, p)
            n_episodes = int(len(np.unique(episodes)))

            # tuning grid info
            if name == "logistic":
                tuning_grid = "C=1.0 (fixed)"
                selected_value = "C=1.0"
                penalty = "l2"
                C_or_alpha = 1.0
            else:
                tuning_grid = "n_estimators=[300], min_samples_leaf=[2]"
                selected_value = "n_estimators=300,min_samples_leaf=2"
                penalty = "none"
                C_or_alpha = float("nan")

            p_one_sided = float(np.mean(draws <= 0.5)) if len(draws) >= 100 else float("nan")

            diag_rows.append({
                "horizon": h, "model": name,
                "n": len(y), "n_positive": int(y.sum()), "n_negative": int((1-y).sum()),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
                "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
                "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
                "roc_auc": auc,
                "auc_ci_low": lo, "auc_ci_high": hi,
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "mcc": float(matthews_corrcoef(y, pred)),
                "brier": brier,
                "calibration_slope": cal_slope,
                "calibration_intercept": cal_intercept,
                "n_episodes": n_episodes,
                "penalty": penalty,
                "C_or_alpha": C_or_alpha,
                "tuning_grid": tuning_grid,
                "selected_value": selected_value,
                "p_one_sided": p_one_sided,
                "p_holm": float("nan"),  # filled below
            })

    # Holm correction across all direction comparisons (same protocol as run_aspi_return_grid)
    diag = pd.DataFrame(diag_rows)
    if len(diag):
        _, p_holm, _, _ = multipletests(diag["p_one_sided"].fillna(1.0).to_numpy(),
                                        alpha=0.05, method="holm")
        diag["p_holm"] = p_holm

    preds = pd.DataFrame(pred_rows)
    save_frame(preds, "aspi_direction_predictions",
               "Per held-out event direction forecast probabilities, by horizon and model.")
    save_frame(diag, "aspi_direction_diagnostics",
               "Full diagnostic table for Y1 direction forecasts: confusion matrix, AUC, Brier, calibration.")

    print(f"wrote aspi_direction_predictions.parquet ({len(preds)} rows, "
          f"{preds['horizon'].nunique()} horizons, {preds['model'].nunique()} models)")
    print(f"wrote aspi_direction_diagnostics.parquet ({len(diag)} rows)")


if __name__ == "__main__":
    main()
