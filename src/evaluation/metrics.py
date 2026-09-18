"""Evaluation metrics for multi-target regression."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


# --------------------------------------------------------------- target bounds

# Thesis Sec. 3.2.2. Clipping to a target's definitional support is projection, not a
# tuned post-process: every true value already lies inside, so absolute error cannot
# increase on any point.
TARGET_BOUNDS = {
    "Y1_ASPI_5D_Forward_LogReturn_Pct": (None, None),
    "Y2_abnormal_volume": (-1.0, None),
    "Y3_recovery_days": (0.0, 90.0),
    "Y1_EventWindow_0_10_LogReturn_Pct": (None, None),
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
    """One row per pooled out-of-fold prediction, mapped back to its event."""
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


def extended_regression_metrics(y_true, y_pred, y_true_reference_std=None):
    """Secondary diagnostics for a regression target, beyond RMSE/MAE/R2: median"""
    from scipy import stats

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ref_std = float(np.std(yt)) if y_true_reference_std is None else y_true_reference_std
    pearson = float(stats.pearsonr(yt, yp)[0]) if len(yt) > 2 and np.std(yp) > 0 else float("nan")
    spearman = float(stats.spearmanr(yt, yp)[0]) if len(yt) > 2 and np.std(yp) > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    return {
        "median_abs_error": float(np.median(np.abs(yt - yp))),
        "directional_accuracy": float(np.mean(np.sign(yt) == np.sign(yp))),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "rmse_normalized": rmse / ref_std if ref_std > 0 else float("nan"),
    }


def bootstrap_auc_ci(y_true, y_score, n_boot=5000, alpha=0.05, random_state=42):
    """Stratified bootstrap CI for an AUC."""
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


if __name__ == "__main__":
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    exact = evaluate_regression(truth, truth)
    assert exact["rmse"] == 0.0 and exact["mae"] == 0.0 and exact["r2"] == 1.0, exact

    off_by_one = evaluate_regression(truth, truth + 1.0)
    assert off_by_one["rmse"] == 1.0 and off_by_one["mae"] == 1.0, off_by_one

    # Skill is the fractional error reduction against a baseline, so halving the
    # error scores 0.5 and matching the baseline scores 0.
    assert abs(skill_score(1.0, 2.0) - 0.5) < 1e-12
    assert skill_score(2.0, 2.0) == 0.0

    # Clipping is a projection onto the target's support: every true value already
    # lies inside it, so absolute error can never increase.
    clipped = clip_to_bounds("Y3_recovery_days", np.array([-5.0, 45.0, 200.0]))
    assert list(clipped) == [0.0, 45.0, 90.0], clipped

    print("metrics.py self-check passed")
