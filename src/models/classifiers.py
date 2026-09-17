"""Classification framing of the three targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Bounded a priori by fold size, not by score. With 30 training rows,
# min_samples_leaf >= 3 puts every leaf on at least ~10% of the fold.
RF_CLF_GRID = {"n_estimators": [300], "max_depth": [2, 3, 4], "min_samples_leaf": [3, 5]}
XGB_CLF_GRID = {"n_estimators": [100, 300], "max_depth": [2, 3],
                "learning_rate": [0.03, 0.1], "subsample": [0.8], "colsample_bytree": [0.8]}
LOGIT_CS = np.logspace(-3, 3, 13)


def _binarise(values, mask):
    """Boolean condition -> {0.0, 1.0}, but NaN wherever the source target is missing."""
    return np.asarray(values, dtype=float) * 1.0 + np.where(np.asarray(mask), 0.0, np.nan)


def label_negative_return(y, train_idx=None, dataset=None):
    """C1: Y1 < 0. The sign split, kept for continuity with the directional-accuracy
    section. No free parameter.
    """
    v = y["Y1_ASPI_5D_Forward_LogReturn_Pct"]
    return pd.Series(_binarise(v < 0, v.notna()), index=v.index)


def label_adverse_move(y, train_idx, dataset=None):
    """C1b: Y1 below the bottom tercile of THIS fold's training window.

    Training rows only, so no test information reaches the label. ~33% prevalence by
    construction in training, but measured test-fold prevalence is 10% -- the held-out
    folds are calmer, leaving ~3 positives in 30 pooled points. Report with that caveat."""
    v = y["Y1_ASPI_5D_Forward_LogReturn_Pct"]
    cut = float(np.nanquantile(v.iloc[train_idx], 1 / 3))
    return pd.Series(_binarise(v < cut, v.notna()), index=v.index)


def label_volume_spike(y, train_idx=None, dataset=None):
    """C2: Y2 > 0, i.e. volume above its own 30-day baseline.

    Zero is where the target is centred by construction (Y2 = V/V_bar - 1), so this is the
    natural cut rather than a chosen one.
    """
    # Y2 is unobserved for the 2000 archive year and for every post-2023 event
    # (countryeconomy publishes the index level, not volume). Those events must be
    # dropped, never scored as non-spikes, see _binarise.
    v = y["Y2_abnormal_volume"]
    return pd.Series(_binarise(v > 0, v.notna()), index=v.index)


def label_recovers_in_90(y, train_idx=None, dataset=None):
    """C3: Y3 < 90, recovery observed inside the window. The threshold is the design
    constant itself. This is also stage 1 of the hurdle model.
    """
    v = y["Y3_recovery_days"]
    unknown = (dataset["Y3_censor_reason"].reindex(v.index).eq("next_disaster")
               if dataset is not None and "Y3_censor_reason" in dataset.columns
               else pd.Series(False, index=v.index))
    return pd.Series(_binarise(v < 90, v.notna() & ~unknown.fillna(False)), index=v.index)


def label_adverse_move_sigma(y, train_idx=None, dataset=None):
    """Secondary sensitivity only: Y1 below one pre-event standard deviation.

    Standardised-abnormal-return form from the event-study literature, but at this N it
    leaves too few positives to support precision or recall. Never headline it.
    """
    v = y["Y1_ASPI_5D_Forward_LogReturn_Pct"]
    # rolling_std_30 is computed on the daily log_return series (unchanged units), while
    # Y1 is now a %-deviation-from-30d-mean (see feature_eng.build_targets), scale sigma
    # to percent so the comparison stays meaningful.
    sigma = (dataset["rolling_std_30"] * 100.0).replace(0, np.nan)
    return pd.Series(_binarise(v < -sigma, v.notna() & sigma.notna()), index=v.index)


# label_car5_negative (C4) REMOVED, methodology-audit finding #8 (2026-09-16): it asked


def label_slow_recovery(y, train_idx, dataset=None):
    """C3b: recovery slower than the median of THIS fold's training window."""
    v = y["Y3_recovery_days"]
    cut = float(np.nanmedian(v.iloc[train_idx]))
    unknown = (dataset["Y3_censor_reason"].reindex(v.index).eq("next_disaster")
               if dataset is not None and "Y3_censor_reason" in dataset.columns
               else pd.Series(False, index=v.index))
    return pd.Series(_binarise(v > cut, v.notna() & ~unknown.fillna(False)), index=v.index)


LABELS = {
    "C1_negative_return": label_negative_return,
    "C1b_adverse_move": label_adverse_move,
    "C2_volume_spike": label_volume_spike,
    "C3_recovers_in_90": label_recovers_in_90,
    "C3b_slow_recovery": label_slow_recovery,
}


def build_classifiers(random_state=42):
    """Three families mirroring the regression side: a penalised linear baseline (the
    classification analogue of Ridge's role as the H1 reference), and two tree ensembles
    under the same a-priori capacity bounds."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    return {
        "logistic": (make_pipeline(StandardScaler(),
                                   LogisticRegression(max_iter=5000, penalty="l2",
                                                      class_weight="balanced")),
                     {"logisticregression__C": LOGIT_CS}),
        "rf_clf": (RandomForestClassifier(random_state=random_state, n_jobs=-1,
                                          class_weight="balanced"),
                   RF_CLF_GRID),
        "xgb_clf": (XGBClassifier(random_state=random_state, n_jobs=-1,
                                  eval_metric="logloss"),
                    XGB_CLF_GRID),
    }


def classification_metrics(y_true, y_score, threshold=0.5):
    """Every metric reported together, always beside the majority baseline."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                                 f1_score, matthews_corrcoef, precision_score,
                                 average_precision_score, recall_score, roc_auc_score)

    from ..evaluation.metrics import bootstrap_auc_ci, hanley_mcneil_ci

    yt = np.asarray(y_true).astype(int)
    ys = np.asarray(y_score, dtype=float)
    yp = (ys >= threshold).astype(int)

    n_pos, n_neg = int(yt.sum()), int((1 - yt).sum())
    prevalence = n_pos / len(yt) if len(yt) else float("nan")
    majority = max(prevalence, 1 - prevalence)

    out = {
        "n": len(yt), "n_pos": n_pos, "prevalence": prevalence,
        "majority_baseline_acc": majority,
        "accuracy": accuracy_score(yt, yp),
        "balanced_accuracy": balanced_accuracy_score(yt, yp),
        "mcc": matthews_corrcoef(yt, yp) if n_pos and n_neg else float("nan"),
        "precision": precision_score(yt, yp, zero_division=0),
        "recall": recall_score(yt, yp, zero_division=0),
        "f1": f1_score(yt, yp, zero_division=0),
        "brier": brier_score_loss(yt, np.clip(ys, 0, 1)) if n_pos and n_neg else float("nan"),
    }
    if n_pos and n_neg:
        auc = roc_auc_score(yt, ys)
        lo, hi = hanley_mcneil_ci(auc, n_pos, n_neg)
        # Both intervals, because they are different instruments: Hanley-McNeil is
        # parametric and approximate at n=30, the bootstrap is not. The stricter of the
        # two decides whether chance is excluded.
        _, blo, bhi, _ = bootstrap_auc_ci(yt, ys)
        out.update({"auc": auc, "auc_lo": lo, "auc_hi": hi,
                    "auc_boot_lo": blo, "auc_boot_hi": bhi,
                    "auc_beats_chance": bool(lo > 0.5 and blo > 0.5),
                    "pr_auc": average_precision_score(yt, ys)})
    else:
        out.update({"auc": float("nan"), "auc_lo": float("nan"), "auc_hi": float("nan"),
                    "auc_boot_lo": float("nan"), "auc_boot_hi": float("nan"),
                    "auc_beats_chance": False, "pr_auc": float("nan")})
    # The honest headline: a classifier only counts if it clears the majority rule on a
    # baseline-proof metric AND its AUC interval excludes chance.
    out["beats_baseline"] = bool(out["auc_beats_chance"] and out["balanced_accuracy"] > 0.5)
    return out


def summarise(records: list) -> pd.DataFrame:
    cols = ["label", "model", "n", "n_pos", "prevalence", "majority_baseline_acc",
            "accuracy", "balanced_accuracy", "mcc", "precision", "recall", "f1",
            "pr_auc", "auc", "auc_lo", "auc_hi", "auc_boot_lo", "auc_boot_hi",
            "beats_baseline"]
    frame = pd.DataFrame(records)
    return frame[[c for c in cols if c in frame.columns]]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 64
    y = pd.DataFrame({
        "Y1_ASPI_5D_Forward_LogReturn_Pct": rng.normal(0, 0.014, n),
        "Y2_abnormal_volume": rng.normal(-0.13, 0.58, n),
        "Y3_recovery_days": np.where(rng.random(n) < 0.14, 90.0,
                                     rng.exponential(6, n).round()),
    })
    train_idx = np.arange(30)

    assert set(np.unique(label_negative_return(y))) <= {0.0, 1.0}
    adverse = label_adverse_move(y, train_idx)
    # The cut is the training tercile, so about a third of the TRAINING rows are positive.
    assert 0.2 <= adverse.iloc[train_idx].mean() <= 0.45, adverse.iloc[train_idx].mean()
    assert label_recovers_in_90(y).mean() > 0.7

    # A perfect score must be recognised, and a coin flip must not.
    truth = rng.integers(0, 2, 200)
    perfect = classification_metrics(truth, truth.astype(float))
    assert perfect["f1"] == 1.0 and perfect["balanced_accuracy"] == 1.0
    chance = classification_metrics(truth, rng.random(200))
    assert not chance["beats_baseline"], chance

    # The majority rule must score exactly 0.5 balanced accuracy, the property that makes
    # it the right yardstick for an imbalanced problem.
    lopsided = np.array([1] * 80 + [0] * 20)
    maj = classification_metrics(lopsided, np.ones(100) * 0.9)
    assert abs(maj["balanced_accuracy"] - 0.5) < 1e-9, maj["balanced_accuracy"]
    assert maj["accuracy"] == 0.8

    print("classifiers.py self-check passed")
