"""Does a model actually beat its baseline, or is the gap inside the noise?

Two tests on every (model, target, baseline) triple: a paired bootstrap that resamples
events, and a Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample
correction.

The paired event bootstrap is the PRIMARY criterion (methodology-audit followup,
2026-09-16): it makes no assumption beyond exchangeability of events, which matches
this dataset (disasters are irregular, overlapping-horizon events, not a regularly
spaced time series). Diebold-Mariano assumes a roughly stationary, weakly dependent
h-step-ahead forecast-error series -- a fit that gets worse the more a target's
horizon overlaps neighbouring events (Y3 up to 90 trading days, Y1_EventWindow_0_10 up
to 10). Requiring DM to ALSO agree before a verdict counts (the original rule) let a
bootstrap-confirmed effect get vetoed by a test whose own assumptions are the shakier
fit here -- backwards for events this irregular. DM is still computed and reported
(`dm_agrees`) as a secondary diagnostic, not a gate."""

from __future__ import annotations

import numpy as np

_MIN_N = 8  # below this, neither test carries information worth printing


def _rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err ** 2)))


def paired_bootstrap_delta(
    y_true,
    y_pred_model,
    y_pred_baseline,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    random_state: int = 42,
    metric: str = "rmse",
):
    """Bootstrap CI for (baseline error - model error), resampling EVENTS in pairs.

    Positive delta means the model is better. The pairing matters: model and baseline
    are scored on the identical resampled events every draw, so the shared difficulty of
    an event cancels and the CI reflects only the difference between the two predictors.

    Returns a dict with the point delta, its CI, and the share of draws favouring the
    model (a one-sided bootstrap p-value in the `p_model_worse` field).
    """
    yt = np.asarray(y_true, dtype=float)
    pm = np.asarray(y_pred_model, dtype=float)
    pb = np.asarray(y_pred_baseline, dtype=float)
    if not (len(yt) == len(pm) == len(pb)):
        raise ValueError("y_true, model and baseline predictions must be the same length")

    n = len(yt)
    err_m, err_b = pm - yt, pb - yt

    if metric == "rmse":
        score = _rmse
    elif metric == "mae":
        score = lambda e: float(np.mean(np.abs(e)))  # noqa: E731
    else:
        raise ValueError("metric must be 'rmse' or 'mae'")

    point = score(err_b) - score(err_m)

    if n < _MIN_N:
        return {"n": n, "delta": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "p_model_worse": float("nan"), "significant": False,
                "note": f"n={n} too small to bootstrap"}

    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = np.array([score(err_b[i]) - score(err_m[i]) for i in idx])

    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {
        "n": n,
        "delta": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_model_worse": float(np.mean(deltas <= 0)),
        # "Beats" requires the whole interval on the favourable side -- a point estimate
        # that merely happens to be positive is not evidence at this sample size.
        "significant": bool(lo > 0),
        "note": "",
    }


def diebold_mariano(y_true, y_pred_model, y_pred_baseline, h: int = 1, power: int = 2):
    """Diebold-Mariano test on the squared- (or absolute-) error differential.

    d_t = L(baseline_t) - L(model_t); positive mean d favours the model. Uses the
    Newey-West HAC variance with the standard (h-1) lag truncation, and the Harvey,
    Leybourne & Newbold (1997) small-sample correction, which matters a great deal at
    n=30 -- the uncorrected statistic over-rejects badly here.

    Returns the corrected statistic and a two-sided p-value from the t(n-1) reference
    distribution that HLN recommend in place of the normal.
    """
    from scipy import stats

    yt = np.asarray(y_true, dtype=float)
    pm = np.asarray(y_pred_model, dtype=float)
    pb = np.asarray(y_pred_baseline, dtype=float)
    n = len(yt)

    if n < _MIN_N:
        return {"n": n, "dm_stat": float("nan"), "p_value": float("nan"),
                "significant": False, "note": f"n={n} too small for DM"}

    loss_m = np.abs(pm - yt) ** power
    loss_b = np.abs(pb - yt) ** power
    d = loss_b - loss_m
    d_bar = float(np.mean(d))

    # Newey-West long-run variance of d, truncated at h-1 lags.
    gamma0 = float(np.mean((d - d_bar) ** 2))
    lrv = gamma0
    for lag in range(1, h):
        cov = float(np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar)))
        lrv += 2.0 * (1.0 - lag / h) * cov

    if lrv <= 0:
        # Degenerate: the two predictors produce identical losses (e.g. both constant).
        return {"n": n, "dm_stat": float("nan"), "p_value": float("nan"),
                "significant": False, "note": "zero loss-differential variance"}

    dm = d_bar / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n) * dm
    p = 2 * (1 - stats.t.cdf(abs(hln), df=n - 1))

    return {"n": n, "dm_stat": float(hln), "p_value": float(p),
            "significant": bool(p < 0.05 and d_bar > 0), "note": ""}


def pooled(results: dict, model: str, target: str):
    """Concatenate a model/target's per-fold out-of-fold arrays into one paired vector."""
    store = results[model][target]
    return (np.concatenate(store["y_true"]), np.concatenate(store["y_pred"]))


def verdict_table(results: dict, target_cols, baselines=("naive_zero", "naive_train_mean")):
    """Full per-(model, target, baseline) verdict table.

    Only compares models against baselines on the *same* events. The stacked model
    forfeits fold 0 to its meta-learner, so it has 20 points where the others have 30;
    comparing its pooled vector against a 30-point baseline vector would be meaningless,
    so those pairs are aligned on length and flagged rather than silently zipped.
    """
    import pandas as pd

    rows = []
    models = [m for m in results if m not in baselines]
    for target in target_cols:
        for model in models:
            if target not in results[model]:
                continue
            yt_m, yp_m = pooled(results, model, target)
            for base in baselines:
                if base not in results or target not in results[base]:
                    continue
                yt_b, yp_b = pooled(results, base, target)
                if len(yt_b) != len(yt_m):
                    # Align on the tail: the shorter vector is always the later folds.
                    k = min(len(yt_b), len(yt_m))
                    yt_m2, yp_m2, yp_b2 = yt_m[-k:], yp_m[-k:], yp_b[-k:]
                    aligned = f"aligned to last {k} points"
                else:
                    yt_m2, yp_m2, yp_b2 = yt_m, yp_m, yp_b
                    aligned = ""

                boot = paired_bootstrap_delta(yt_m2, yp_m2, yp_b2)
                dm = diebold_mariano(yt_m2, yp_m2, yp_b2)
                rows.append({
                    "target": target, "model": model, "baseline": base, "n": boot["n"],
                    "rmse_model": _rmse(yp_m2 - yt_m2),
                    "rmse_baseline": _rmse(yp_b2 - yt_m2),
                    "delta_rmse": boot["delta"],
                    "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                    "boot_beats": boot["significant"],
                    "dm_stat": dm["dm_stat"], "dm_p": dm["p_value"],
                    "dm_beats": dm["significant"],
                    # Bootstrap is PRIMARY (see module docstring); DM is reported, not
                    # required. dm_agrees is a diagnostic flag, not part of the gate.
                    "dm_agrees": dm["significant"] == boot["significant"],
                    "verdict": ("BEATS BASELINE" if boot["significant"]
                                else "better, not distinguishable" if boot["delta"] > 0
                                else "worse than baseline"),
                    "note": " ".join(filter(None, [aligned, boot["note"], dm["note"]])),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Self-check: a model with genuinely lower error must be detected, and two predictors
    # that differ only by noise must not be.
    rng = np.random.default_rng(0)
    n = 200
    truth = rng.normal(0, 1, n)

    good = truth + rng.normal(0, 0.2, n)
    bad = truth + rng.normal(0, 1.0, n)
    r = paired_bootstrap_delta(truth, good, bad, n_boot=2000)
    assert r["significant"] and r["delta"] > 0, r
    d = diebold_mariano(truth, good, bad)
    assert d["significant"], d

    a = truth + rng.normal(0, 0.5, n)
    b = truth + rng.normal(0, 0.5, n)
    r2 = paired_bootstrap_delta(truth, a, b, n_boot=2000)
    assert not r2["significant"], r2

    small = paired_bootstrap_delta(truth[:5], good[:5], bad[:5])
    assert not small["significant"] and "too small" in small["note"]

    print("verification.py self-check passed")
