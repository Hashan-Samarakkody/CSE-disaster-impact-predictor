"""Survival model discrimination and calibration: item 9.

Produces two artifacts:
  survival_discrimination.parquet — Harrell's C, Uno's C, IBS with bootstrap CIs.
  survival_calibration.parquet    — predicted vs observed probabilities at each horizon.

Source: recovery_grid_predictions.parquet
  cols: fold, row, model, duration, event_observed, pred_median, P_T_le_10/20/30/60/90

Harrell's C uses lifelines.utils.concordance_index.
Uno's C is implemented directly using IPCW (Uno et al. 2011).
IBS uses the predicted P_T_le_t columns at horizons 10, 20, 30, 60, 90.
Bootstrap CIs: B=1000 resamples of rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.artifact_store import artifact_file, save_frame

RANDOM_STATE = 42
B_IBS = 1000
B_C = 1000
EVAL_HORIZON = 90
HORIZONS = [10, 20, 30, 60, 90]


def km_survival(durations: np.ndarray, event_observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier survival function: returns (times, survival_probs)."""
    order = np.argsort(durations)
    t = durations[order]
    e = event_observed[order]
    unique_times = []
    surv_probs = []
    s = 1.0
    n = len(t)
    i = 0
    while i < n:
        ti = t[i]
        # Find all rows at this time
        j = i
        while j < n and t[j] == ti:
            j += 1
        n_at_risk = n - i
        n_events = int(e[i:j].sum())
        if n_events > 0:
            s *= (1.0 - n_events / n_at_risk)
        unique_times.append(ti)
        surv_probs.append(s)
        i = j
    return np.array(unique_times), np.array(surv_probs)


def km_at_time(times: np.ndarray, surv: np.ndarray, t: float) -> float:
    """KM survival estimate at time t (step function, left-continuous limit from right)."""
    idx = np.searchsorted(times, t, side="right") - 1
    if idx < 0:
        return 1.0
    return float(surv[idx])


def km_greenwood_ci(durations: np.ndarray, event_observed: np.ndarray,
                    t: float, alpha: float = 0.05) -> tuple[float, float, float]:
    """KM estimate at time t with Greenwood formula 95% CI.
    Returns (km_estimate, ci_low, ci_high).
    """
    from scipy import stats
    order = np.argsort(durations)
    dur = durations[order]
    ev = event_observed[order]
    n = len(dur)
    s = 1.0
    greenwood_sum = 0.0
    i = 0
    s_at_t = 1.0
    gw_at_t = 0.0
    while i < n:
        ti = dur[i]
        j = i
        while j < n and dur[j] == ti:
            j += 1
        n_at_risk = n - i
        n_events = int(ev[i:j].sum())
        if n_events > 0:
            s *= (1.0 - n_events / n_at_risk)
            greenwood_sum += n_events / (n_at_risk * (n_at_risk - n_events + 1e-9))
        if ti <= t:
            s_at_t = s
            gw_at_t = greenwood_sum
        i = j

    if s_at_t <= 0 or s_at_t >= 1:
        return float(s_at_t), float(s_at_t), float(s_at_t)

    # Log-log transformation for CI
    log_log_s = np.log(-np.log(s_at_t))
    sigma_log_log = np.sqrt(gw_at_t / (np.log(s_at_t) ** 2))
    z = stats.norm.ppf(1 - alpha / 2)
    ci_low_s = float(s_at_t ** np.exp(z * sigma_log_log))
    ci_hi_s = float(s_at_t ** np.exp(-z * sigma_log_log))
    ci_low_s, ci_hi_s = min(ci_low_s, ci_hi_s), max(ci_low_s, ci_hi_s)
    return float(s_at_t), max(0.0, ci_low_s), min(1.0, ci_hi_s)


def harrell_c(duration: np.ndarray, pred_median: np.ndarray,
              event_observed: np.ndarray) -> float:
    """Harrell's C-index. pred_median is a time-like score (higher = predicted slower
    recovery), matching lifelines' convention — concordant when predicted order matches
    actual duration order. This is the same convention as src/evaluation/survival_metrics.py
    harrell_c_index."""
    from lifelines.utils import concordance_index
    return float(concordance_index(duration, pred_median, event_observed))


def ipcw_weights(duration: np.ndarray, event_observed: np.ndarray) -> np.ndarray:
    """IPCW weights: 1/G(t_i) for each uncensored observation.

    G is the KM estimate of the censoring distribution.
    """
    # Fit KM on censoring (flip event indicator)
    censoring_event = (~event_observed.astype(bool)).astype(int)
    times, surv = km_survival(duration, censoring_event)
    weights = np.ones(len(duration))
    for i, (t, e) in enumerate(zip(duration, event_observed)):
        if e:
            g = km_at_time(times, surv, t)
            weights[i] = 1.0 / max(g, 1e-9)
    return weights


def uno_c(duration: np.ndarray, pred_median: np.ndarray,
          event_observed: np.ndarray, tau: float = None) -> float:
    """Uno's C-statistic (Uno et al. 2011) using IPCW.

    pred_median is a time-like score (higher = predicted longer recovery), consistent
    with Harrell C. Concordant pair (i had event first, duration[i] < duration[j]):
    pred_median[i] < pred_median[j].
    """
    if tau is None:
        tau = np.max(duration)

    weights = ipcw_weights(duration, event_observed)
    # pred_median is time-like: higher = predicted longer recovery.
    # Concordant pair (i had event before j): pred_median[i] < pred_median[j].
    # In Uno's sum: concordant_j counted when risk_score[j] > risk_score[i],
    # so we keep risk_score = pred_median (same direction as Harrell C above).
    risk_score = pred_median

    n = len(duration)
    numerator = 0.0
    denominator = 0.0

    event_mask = (event_observed.astype(bool)) & (duration <= tau)
    event_indices = np.flatnonzero(event_mask)

    for i in event_indices:
        w_i = weights[i] ** 2
        # Compare to ALL j with duration[j] > duration[i]
        j_mask = duration > duration[i]
        if j_mask.sum() == 0:
            continue
        concordant_j = (risk_score[j_mask] > risk_score[i]).sum()
        tied_j = (risk_score[j_mask] == risk_score[i]).sum()
        denominator += w_i * j_mask.sum()
        numerator += w_i * (concordant_j + 0.5 * tied_j)

    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def compute_ibs(duration: np.ndarray, event_observed: np.ndarray,
                prob_cols: dict) -> float:
    """Integrated Brier Score using P_T_le_t predictions.

    prob_cols: dict of {horizon: predicted_prob_array}
    IBS = (1/tau) * integral_0^tau BS(t) dt, approximated by trapezoidal rule.
    """
    times = sorted(prob_cols.keys())
    brier_scores = []

    # Compute KM of censoring distribution for IPCW
    censoring_event = (~event_observed.astype(bool)).astype(int)
    km_times, km_surv = km_survival(duration, censoring_event)

    for t in times:
        p_hat = prob_cols[t]   # P(T <= t)
        I_t = (duration <= t).astype(float) * event_observed  # actual events by t
        I_survive = (duration > t).astype(float)              # actually survived past t

        # IPCW weights
        # w_i = I(T_i <= t, E_i=1) / G(T_i) + I(T_i > t) / G(t)
        g_t = km_at_time(km_times, km_surv, t)
        w = np.zeros(len(duration))
        for i, (d, e) in enumerate(zip(duration, event_observed)):
            if d <= t and e:
                g_i = km_at_time(km_times, km_surv, d)
                w[i] = 1.0 / max(g_i, 1e-9)
            elif d > t:
                w[i] = 1.0 / max(g_t, 1e-9)

        bs = float(np.mean(w * (p_hat - I_t) ** 2))
        brier_scores.append(bs)

    if len(times) < 2:
        return float("nan")

    # Trapezoidal integration, normalise by the max horizon
    t_arr = np.array(times, dtype=float)
    bs_arr = np.array(brier_scores)
    ibs = float(np.trapz(bs_arr, t_arr) / (t_arr[-1] - t_arr[0] + 1e-9) if len(t_arr) > 1 else bs_arr[0])
    return ibs


def bootstrap_c(duration: np.ndarray, pred_median: np.ndarray,
                event_observed: np.ndarray, rng, b: int = B_C,
                metric: str = "harrell") -> tuple[float, float]:
    """Bootstrap CI for Harrell's or Uno's C."""
    n = len(duration)
    draws = []
    for _ in range(b):
        idx = rng.integers(0, n, size=n)
        d, p, e = duration[idx], pred_median[idx], event_observed[idx]
        if e.sum() < 2:
            continue
        try:
            if metric == "harrell":
                draws.append(harrell_c(d, p, e))
            else:
                draws.append(uno_c(d, p, e))
        except Exception:
            continue
    if len(draws) < 10:
        return float("nan"), float("nan")
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def bootstrap_ibs(duration: np.ndarray, event_observed: np.ndarray,
                  prob_cols: dict, rng, b: int = B_IBS
                  ) -> tuple[float, float]:
    """Bootstrap CI for IBS."""
    n = len(duration)
    draws = []
    horizons = sorted(prob_cols.keys())
    for _ in range(b):
        idx = rng.integers(0, n, size=n)
        d_b = duration[idx]
        e_b = event_observed[idx]
        p_b = {h: prob_cols[h][idx] for h in horizons}
        try:
            draws.append(compute_ibs(d_b, e_b, p_b))
        except Exception:
            continue
    if len(draws) < 10:
        return float("nan"), float("nan")
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    preds = pd.read_parquet(artifact_file("recovery_grid_predictions.parquet"))
    rng = np.random.default_rng(RANDOM_STATE)

    models = preds["model"].unique()
    print(f"models: {list(models)}")
    print(f"total rows: {len(preds)}")

    # ── survival_discrimination.parquet ──────────────────────────────────────────
    disc_rows = []
    for model_name in models:
        g = preds[preds.model == model_name].copy()
        duration = g["duration"].to_numpy(float)
        event_obs = g["event_observed"].to_numpy(float)
        pred_med = g["pred_median"].to_numpy(float)
        n = len(g)
        n_events = int(event_obs.sum())
        n_censored = n - n_events
        n_distinct = int(pd.Series(pred_med).nunique())

        print(f"  {model_name}: n={n} n_events={n_events} n_censored={n_censored}")

        try:
            hc = harrell_c(duration, pred_med, event_obs)
        except Exception as e:
            print(f"    Harrell C error: {e}")
            hc = float("nan")

        try:
            uc = uno_c(duration, pred_med, event_obs, tau=EVAL_HORIZON)
        except Exception as e:
            print(f"    Uno C error: {e}")
            uc = float("nan")

        # IBS
        prob_cols = {h: g[f"P_T_le_{h}"].to_numpy(float) for h in HORIZONS}
        try:
            ibs = compute_ibs(duration, event_obs, prob_cols)
        except Exception as e:
            print(f"    IBS error: {e}")
            ibs = float("nan")

        print(f"    Harrell C={hc:.4f}, Uno C={uc:.4f}, IBS={ibs:.4f}")

        # Bootstrap CIs
        hc_lo, hc_hi = bootstrap_c(duration, pred_med, event_obs, rng, b=B_C, metric="harrell")
        uc_lo, uc_hi = bootstrap_c(duration, pred_med, event_obs, rng, b=B_C, metric="uno")
        ibs_lo, ibs_hi = bootstrap_ibs(duration, event_obs, prob_cols, rng, b=B_IBS)

        brier_per_horizon = {}
        for h in HORIZONS:
            p_hat = prob_cols[h]
            I_t = (duration <= h).astype(float) * event_obs
            brier_per_horizon[f"brier_t{h}"] = float(np.mean((p_hat - I_t) ** 2))

        disc_rows.append({
            "model": model_name,
            "n": n,
            "n_events": n_events,
            "harrell_c": hc,
            "harrell_c_lo": hc_lo,
            "harrell_c_hi": hc_hi,
            "uno_c": uc,
            "uno_c_lo": uc_lo,
            "uno_c_hi": uc_hi,
            "ibs": ibs,
            "ibs_lo": ibs_lo,
            "ibs_hi": ibs_hi,
            "eval_horizon": EVAL_HORIZON,
            "tie_rule": "min",
            "risk_score_direction": "pred_median is time-like: higher = slower recovery; concordant when pred_median[i] < pred_median[j] for i that recovered first",
            "censoring_weighting": "inverse probability of censoring (IPCW) for Uno C, unweighted for Harrell C",
            "n_distinct_predictions": n_distinct,
            **brier_per_horizon,
        })

    disc = pd.DataFrame(disc_rows)
    save_frame(disc, "survival_discrimination",
               "Harrell C, Uno C (IPCW), IBS with 1000-bootstrap CIs per model. "
               f"eval_horizon={EVAL_HORIZON}, tie_rule=min, lower pred_median = higher risk.")
    print(f"wrote survival_discrimination.parquet ({len(disc)} rows)")

    # ── survival_calibration.parquet ─────────────────────────────────────────────
    cal_rows = []
    for model_name in models:
        g = preds[preds.model == model_name].copy()
        duration = g["duration"].to_numpy(float)
        event_obs = g["event_observed"].to_numpy(float)

        for h in HORIZONS:
            p_hat_col = f"P_T_le_{h}"
            if p_hat_col not in g.columns:
                continue
            p_hat = g[p_hat_col].to_numpy(float)
            predicted_prob = float(np.nanmean(p_hat))

            n_at_risk = int((duration >= h).sum())
            n_events_h = int(((duration < h) & (event_obs.astype(bool))).sum())

            km_est, ci_low, ci_high = km_greenwood_ci(duration, event_obs.astype(int), h)
            # observed_km_prob = P(T <= h) = 1 - KM(h)
            observed_km_prob = 1.0 - km_est
            ci_low_cdf = 1.0 - ci_high   # invert: 1 - upper survival = lower CDF
            ci_high_cdf = 1.0 - ci_low

            cal_rows.append({
                "model": model_name,
                "horizon": h,
                "predicted_prob": predicted_prob,
                "observed_km_prob": observed_km_prob,
                "n_at_risk": n_at_risk,
                "n_events": n_events_h,
                "ci_low": ci_low_cdf,
                "ci_high": ci_high_cdf,
            })

    cal = pd.DataFrame(cal_rows)
    save_frame(cal, "survival_calibration",
               "Predicted P(T<=t) vs observed KM P(T<=t) at each horizon for each model. "
               "CIs: Greenwood formula for KM, inverted to CDF scale.")
    print(f"wrote survival_calibration.parquet ({len(cal)} rows)")


if __name__ == "__main__":
    main()
