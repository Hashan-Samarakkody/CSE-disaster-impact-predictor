"""Evaluation metrics for multi-target regression.

Beyond the basic regression scores, this module owns the small numeric helpers that
several notebook stages and every figure need in common: mapping stored out-of-fold
predictions back to their events, finite-sample interval quantiles, and bootstrap CIs.

These lived as notebook-local defs until the pipeline was split, which meant a figure
could not reuse them without re-executing a modelling cell.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


# --------------------------------------------------------------- target bounds

# Thesis Sec. 3.2.2. Y3 is capped at 90 by construction; Y2 = V/V_bar - 1 cannot go
# below -1 because volume is non-negative. Clipping to these is Euclidean projection
# onto the target's support -- every true value already lies inside, so absolute error
# cannot increase on any point. This is definitional, not a tuned post-process.
TARGET_BOUNDS = {
    "Y1_aspi_log_return": (None, None),
    "Y2_abnormal_volume": (-1.0, None),
    "Y3_recovery_days": (0.0, 90.0),
    "Y1_car_5": (None, None),
    "Y1_car_10": (None, None),
}


def clip_to_bounds(target: str, values):
    lo, hi = TARGET_BOUNDS[target]
    return np.clip(np.asarray(values, dtype=float), lo, hi)


# --------------------------------------------------------------- pooling


def pooled_arrays(results: dict, model: str, target: str):
    """Concatenate a model/target's per-fold out-of-fold arrays into one paired vector."""
    store = results[model][target]
    return np.concatenate(store["y_true"]), np.concatenate(store["y_pred"])


def pooled_frame(results, model, target, splits, dataset, y=None, date_col="event_date"):
    """One row per pooled out-of-fold prediction, mapped back to its event.

    Everything that plots a prediction against something about the event -- severity,
    date, disaster type -- needs this mapping, and two details make the naive version
    silently wrong:

    1. Predictions were recorded AFTER a per-target NaN mask, so fold i's rows are
       `test_index[y[target].iloc[test_index].notna()]`, not `test_index`. Y2 loses
       events this way, and ignoring it shifts every later row.
    2. The stacked model forfeits fold 0 to its meta-learner, so it stores one fewer
       fold than `splits` has.
    3. A (fold, target) pair whose NaN mask leaves nothing to fit is skipped entirely.
       Y2 has no observations at all for the post-2023 events, so a short test window
       can land a fold inside an all-NaN stretch.

    Case 3 is why the fold indices are read from the store when present rather than
    inferred: a skipped fold in the MIDDLE of the sequence breaks any offset arithmetic
    silently, producing a plausible-looking but wrong mapping. The subtract-from-the-end
    inference is kept only as a fallback for results recorded before folds were tracked.

    A length assertion at the end turns any of these into a hard failure instead of a
    misaligned scatter plot.
    """
    import pandas as pd

    store = results[model][target]
    n_stored = len(store["y_true"])
    recorded = [f for f in store.get("folds", []) if f is not None]
    if len(recorded) == n_stored:
        fold_ids = recorded
    else:
        offset = len(splits) - n_stored  # 0 for normal models, 1 for the stacked model
        fold_ids = [offset + i for i in range(n_stored)]

    rows = []
    for i, fold_id in enumerate(fold_ids):
        split = splits[fold_id]
        idx = np.asarray(split.test_index)
        if y is not None and target in getattr(y, "columns", []):
            idx = idx[y[target].iloc[idx].notna().to_numpy()]

        yt = np.asarray(store["y_true"][i], dtype=float)
        yp = np.asarray(store["y_pred"][i], dtype=float)
        if len(idx) != len(yt):
            raise ValueError(
                f"{model}/{target} fold {i}: {len(idx)} event rows vs {len(yt)} stored "
                f"predictions. The NaN mask used at scoring time does not match the one "
                f"reconstructed here.")

        for row_index, true_v, pred_v in zip(idx, yt, yp):
            rec = {"fold": fold_id, "row_index": int(row_index),
                   "y_true": float(true_v), "y_pred": float(pred_v),
                   "resid": float(pred_v - true_v), "abs_resid": abs(float(pred_v - true_v))}
            if date_col in dataset.columns:
                rec[date_col] = dataset[date_col].iloc[row_index]
            if "disaster_type" in dataset.columns:
                rec["disaster_type"] = dataset["disaster_type"].iloc[row_index]
            rows.append(rec)

    frame = pd.DataFrame(rows)
    expected = sum(len(a) for a in store["y_true"])
    assert len(frame) == expected, f"pooled_frame lost rows: {len(frame)} vs {expected}"
    return frame


# --------------------------------------------------------------- intervals and CIs


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a proportion.

    Preferred to the normal approximation at these sample sizes, where the latter can
    run outside [0, 1].
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def hanley_mcneil_ci(auc, n_pos, n_neg, z=1.96):
    """Hanley-McNeil standard error for an AUC, and the resulting Wald interval."""
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
                  + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg))
    return max(0.0, auc - z * se), min(1.0, auc + z * se)


def conformal_q(resid, alpha=0.20):
    """Finite-sample conformal quantile: the ceil((n+1)(1-alpha))-th smallest absolute
    residual, not the plain (1-alpha) empirical quantile. At n=10 the plain quantile
    under-covers by construction. Falls back to the largest residual where the level
    cannot be certified at this n."""
    resid = np.sort(np.asarray(resid, dtype=float))
    n = len(resid)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(resid[min(k, n) - 1])


def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_boot=2000, alpha=0.05, random_state=42):
    """Percentile bootstrap CI for any metric of (y_true, y_pred)."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    n = len(yt)
    point = float(metric_fn(yt, yp))
    if n < 8:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, n, size=(n_boot, n))
    draws = np.array([metric_fn(yt[i], yp[i]) for i in idx])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def skill_score(rmse_model: float, rmse_reference: float) -> float:
    """1 - model/reference. Positive means the model improves on the reference."""
    if rmse_reference == 0:
        return float("nan")
    return 1.0 - rmse_model / rmse_reference


def bootstrap_auc_ci(y_true, y_score, n_boot=5000, alpha=0.05, random_state=42):
    """Stratified bootstrap CI for an AUC.

    Hanley-McNeil assumes a particular parametric form and is only approximate at n=30.
    Where an AUC is the headline result it should face the same nonparametric test the
    regression baselines faced, so both are reported side by side.

    Resamples positives and negatives separately, which keeps the class balance fixed and
    stops a draw degenerating to one class. Returns (auc, lo, hi, n_discarded).
    """
    from sklearn.metrics import roc_auc_score

    yt = np.asarray(y_true).astype(int)
    ys = np.asarray(y_score, dtype=float)
    pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan"), float("nan"), 0

    point = float(roc_auc_score(yt, ys))
    rng = np.random.default_rng(random_state)
    draws, discarded = [], 0
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        try:
            draws.append(roc_auc_score(yt[idx], ys[idx]))
        except ValueError:
            discarded += 1
    if not draws:
        return point, float("nan"), float("nan"), discarded
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi), discarded
