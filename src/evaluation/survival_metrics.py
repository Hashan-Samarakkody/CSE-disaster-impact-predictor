"""Censoring-aware evaluation for Y3 (recovery duration)."""

from __future__ import annotations

import numpy as np
import pandas as pd

PROB_TIMES = (10.0, 20.0, 30.0, 60.0, 90.0)


def harrell_c_index(durations, event_observed, predicted_duration) -> float:
    """Concordance between predicted and actual duration ORDER. `predicted_duration` is a
    time-like score (higher = predicted slower recovery), matching lifelines' convention
    for AFT medians, not a hazard/risk score, which points the other way."""
    from lifelines.utils import concordance_index

    d = np.asarray(durations, dtype=float)
    p = np.asarray(predicted_duration, dtype=float)
    e = np.asarray(event_observed, dtype=bool)
    if len(np.unique(p)) < 2 or e.sum() < 2:
        return float("nan")
    return float(concordance_index(d, p, event_observed=e))


def _km_censoring_survival(durations, event_observed):
    """Kaplan-Meier estimate of G(t) = P(censoring time > t), fit by flipping the event
    indicator. Returns a callable t -> G(t), floored away from zero so the IPCW weights
    stay finite at the tail."""
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter().fit(np.asarray(durations, float),
                                 event_observed=~np.asarray(event_observed, bool))

    def G(t):
        vals = np.atleast_1d(np.asarray(km.predict(np.atleast_1d(np.asarray(t, float))),
                                        dtype=float))
        return np.clip(vals, 1e-3, None)

    return G


def integrated_brier_score(durations, event_observed, survival_at, times=PROB_TIMES):
    """IPCW Brier score at each `times` point and its time-average (Graf et al., 1999)."""
    d = np.asarray(durations, dtype=float)
    e = np.asarray(event_observed, dtype=bool)
    S = np.asarray(survival_at, dtype=float)
    G = _km_censoring_survival(d, e)

    per_time = []
    for j, t in enumerate(times):
        recovered_by_t = e & (d <= t)
        at_risk = d > t
        w = np.zeros(len(d))
        w[recovered_by_t] = 1.0 / G(d[recovered_by_t])
        w[at_risk] = 1.0 / float(G(t)[0])
        # target: 1{T > t}; prediction: S(t)
        target = at_risk.astype(float)
        bs = float(np.sum(w * (target - S[:, j]) ** 2) / len(d))
        per_time.append(bs)
    return {"brier_by_time": dict(zip(map(float, times), per_time)),
            "integrated_brier_score": float(np.mean(per_time))}


def probability_calibration(durations, event_observed, cdf_at, times=PROB_TIMES):
    """Predicted mean P(T <= t) against the Kaplan-Meier observed 1 - S_KM(t)."""
    from lifelines import KaplanMeierFitter

    d = np.asarray(durations, dtype=float)
    e = np.asarray(event_observed, dtype=bool)
    km = KaplanMeierFitter().fit(d, event_observed=e)
    F = np.asarray(cdf_at, dtype=float)
    rows = []
    for j, t in enumerate(times):
        observed = float(1.0 - km.predict(t))
        predicted = float(np.mean(F[:, j]))
        rows.append({"t": float(t), "predicted_P_T_le_t": predicted,
                     "observed_KM_P_T_le_t": observed,
                     "calibration_gap": predicted - observed})
    return pd.DataFrame(rows)


def uncensored_point_errors(durations, event_observed, predicted_duration) -> dict:
    """MAE / median absolute error / RMSE over GENUINELY OBSERVED recoveries only."""
    d = np.asarray(durations, dtype=float)
    e = np.asarray(event_observed, dtype=bool)
    p = np.asarray(predicted_duration, dtype=float)
    if e.sum() == 0:
        return {"n_uncensored": 0, "mae_uncensored": np.nan,
                "median_abs_error_uncensored": np.nan, "rmse_uncensored": np.nan}
    err = p[e] - d[e]
    return {"n_uncensored": int(e.sum()),
            "mae_uncensored": float(np.mean(np.abs(err))),
            "median_abs_error_uncensored": float(np.median(np.abs(err))),
            "rmse_uncensored": float(np.sqrt(np.mean(err ** 2)))}


def cluster_bootstrap_ci(stat_fn, cluster_ids, n_boot=2000, random_state=42, alpha=0.05,
                         return_draws=False):
    """Episode-clustered bootstrap CI for any statistic computed from a row-index array
    (protocol 3.2). `stat_fn(idx) -> float`; NaN draws are dropped. With
    `return_draws=True` also returns the draw array, so a caller can read a one-sided
    bootstrap p-value off it (e.g. share of draws at or below chance concordance) without
    re-running the resampling."""
    rng = np.random.default_rng(random_state)
    cids = np.asarray(cluster_ids)
    clusters = np.unique(cids)
    members = [np.flatnonzero(cids == c) for c in clusters]
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(clusters), size=len(clusters))
        idx = np.concatenate([members[c] for c in pick])
        v = stat_fn(idx)
        if np.isfinite(v):
            draws.append(v)
    if len(draws) < 100:
        return (float("nan"), float("nan"), np.asarray(draws)) if return_draws else (
            float("nan"), float("nan"))
    lo, hi = (float(v) for v in np.quantile(draws, [alpha / 2, 1 - alpha / 2]))
    return (lo, hi, np.asarray(draws)) if return_draws else (lo, hi)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    true_t = rng.exponential(20, n)
    cens_t = rng.exponential(60, n)
    d = np.minimum(true_t, cens_t)
    e = true_t <= cens_t

    # A predictor perfectly ordered with the truth must beat a pure-noise one.
    good = harrell_c_index(d, e, true_t)
    noise = harrell_c_index(d, e, rng.normal(size=n))
    assert good > 0.95, good
    assert abs(noise - 0.5) < 0.1, noise

    # IBS: the true exponential survival curve must score better than a flat 0.5.
    times = np.array(PROB_TIMES)
    S_true = np.exp(-np.outer(np.ones(n), times) / 20.0)
    ibs_true = integrated_brier_score(d, e, S_true)["integrated_brier_score"]
    ibs_flat = integrated_brier_score(d, e, np.full((n, len(times)), 0.5))["integrated_brier_score"]
    assert ibs_true < ibs_flat, (ibs_true, ibs_flat)

    cal = probability_calibration(d, e, 1.0 - S_true)
    assert cal["calibration_gap"].abs().max() < 0.1, cal

    pe = uncensored_point_errors(d, e, true_t)
    assert pe["mae_uncensored"] < 1e-9 and pe["n_uncensored"] == int(e.sum())

    lo, hi = cluster_bootstrap_ci(lambda i: harrell_c_index(d[i], e[i], true_t[i]),
                                  np.arange(n) // 3, n_boot=300)
    assert lo > 0.5 and hi <= 1.0, (lo, hi)
    lo2, hi2, draws = cluster_bootstrap_ci(lambda i: harrell_c_index(d[i], e[i], true_t[i]),
                                           np.arange(n) // 3, n_boot=300, return_draws=True)
    assert (lo2, hi2) == (lo, hi) and len(draws) >= 100

    print(f"survival_metrics.py self-check passed (c={good:.3f}, IBS={ibs_true:.4f})")
