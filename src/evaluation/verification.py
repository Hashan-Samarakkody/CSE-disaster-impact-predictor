"""Does a model actually beat its baseline, or is the gap inside the noise?"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MIN_N = 8  # below this, neither test carries information worth printing


def _rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err ** 2)))


def build_episode_ids(dates, gap_days: int = 14) -> np.ndarray:
    """Group events into disaster episodes (methodology-audit followup, item 22): two"""
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    order = d.sort_values().index.to_numpy()
    sorted_dates = d.iloc[order].to_numpy()
    gap = np.diff(sorted_dates).astype("timedelta64[D]").astype(float)
    new_episode = np.concatenate([[True], gap > gap_days])
    episode_of_sorted = np.cumsum(new_episode) - 1
    episode_ids = np.empty(len(d), dtype=int)
    episode_ids[order] = episode_of_sorted
    return episode_ids


def paired_bootstrap_delta(
    y_true,
    y_pred_model,
    y_pred_baseline,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    random_state: int = 42,
    metric: str = "rmse",
    cluster_ids=None,
):
    """Bootstrap CI for (baseline error - model error), resampling EVENTS in pairs."""
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
    if cluster_ids is None:
        idx = rng.integers(0, n, size=(n_boot, n))
        deltas = np.array([score(err_b[i]) - score(err_m[i]) for i in idx])
    else:
        cids = np.asarray(cluster_ids)
        if len(cids) != n:
            raise ValueError("cluster_ids must be the same length as y_true")
        clusters = np.unique(cids)
        members = [np.flatnonzero(cids == cl) for cl in clusters]
        n_clusters = len(clusters)
        deltas = np.empty(n_boot)
        for b in range(n_boot):
            draw = rng.integers(0, n_clusters, size=n_clusters)
            i = np.concatenate([members[c] for c in draw])
            deltas[b] = score(err_b[i]) - score(err_m[i])

    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {
        "n": n,
        "delta": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_model_worse": float(np.mean(deltas <= 0)),
        # "Beats" requires the whole interval on the favourable side, a point estimate
        # that merely happens to be positive is not evidence at this sample size.
        "significant": bool(lo > 0),
        "note": "",
    }


def diebold_mariano(y_true, y_pred_model, y_pred_baseline, h: int = 1, power: int = 2):
    """Diebold-Mariano test on the squared- (or absolute-) error differential."""
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
    idx = np.concatenate(store["index"]) if store.get("index") else None
    return (np.concatenate(store["y_true"]), np.concatenate(store["y_pred"]), idx)


def verdict_table(results: dict, target_cols, baselines=("naive_zero", "naive_train_mean"),
                  event_dates=None, episode_gap_days: int = 14,
                  confirmatory_models=None):
    """Full per-(model, target, baseline) verdict table.

    `confirmatory_models` names the models selected by the same purged inner cross
    validation as each other (Revision 2, T7 and T8). When given, every row carries a
    `confirmatory` flag and the Holm correction spans the confirmatory family ONLY;
    models fitted at fixed hyperparameters are reported as exploratory and are never
    corrected jointly with it, because ranking a tuned model against an untuned one is not
    a comparison. When omitted the correction spans the whole table, the original
    behaviour.
    """
    from statsmodels.stats.multitest import multipletests

    rows = []
    models = [m for m in results if m not in baselines]
    for target in target_cols:
        for model in models:
            if target not in results[model]:
                continue
            yt_m, yp_m, idx_m = pooled(results, model, target)
            for base in baselines:
                if base not in results or target not in results[base]:
                    continue
                yt_b, yp_b, idx_b = pooled(results, base, target)
                if len(yt_b) != len(yt_m):
                    # Align on the tail: the shorter vector is always the later folds.
                    k = min(len(yt_b), len(yt_m))
                    yt_m2, yp_m2, yp_b2 = yt_m[-k:], yp_m[-k:], yp_b[-k:]
                    idx_m2 = idx_m[-k:] if idx_m is not None else None
                    aligned = f"aligned to last {k} points"
                else:
                    yt_m2, yp_m2, yp_b2, idx_m2 = yt_m, yp_m, yp_b, idx_m
                    aligned = ""

                cluster_ids = None
                if event_dates is not None and idx_m2 is not None:
                    dates = pd.Series(event_dates).reindex(idx_m2)
                    if dates.notna().all():
                        cluster_ids = build_episode_ids(dates.to_numpy(), gap_days=episode_gap_days)

                boot = paired_bootstrap_delta(yt_m2, yp_m2, yp_b2, cluster_ids=cluster_ids)
                dm = diebold_mariano(yt_m2, yp_m2, yp_b2)
                rows.append({
                    "target": target, "model": model, "baseline": base, "n": boot["n"],
                    "rmse_model": _rmse(yp_m2 - yt_m2),
                    "rmse_baseline": _rmse(yp_b2 - yt_m2),
                    "delta_rmse": boot["delta"],
                    "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                    "boot_beats": boot["significant"],
                    "p_model_worse": boot["p_model_worse"],
                    "clustered_by_episode": cluster_ids is not None,
                    "dm_stat": dm["dm_stat"], "dm_p": dm["p_value"],
                    "dm_beats": dm["significant"],
                    # Bootstrap is PRIMARY (see module docstring); DM is reported, not
                    # required. dm_agrees is a diagnostic flag, not part of the gate.
                    "dm_agrees": dm["significant"] == boot["significant"],
                    "verdict": ("BEATS BASELINE" if boot["significant"]
                                else "better, not distinguishable" if boot["delta"] > 0
                                else "worse than baseline"),
                    "confirmatory": (True if confirmatory_models is None
                                     else model in confirmatory_models),
                    "note": " ".join(filter(None, [aligned, boot["note"], dm["note"]])),
                })

    table = pd.DataFrame(rows)
    if len(table):
        # NaN p-values (n too small to bootstrap) can't be corrected, treated as
        # non-significant rather than dropped, so the row count here matches `table`.
        family = table["confirmatory"].to_numpy(bool)
        table["p_holm"] = np.nan
        if family.any():
            pvals = table.loc[family, "p_model_worse"].fillna(1.0).to_numpy()
            _, p_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")
            table.loc[family, "p_holm"] = p_holm
        table["holm_significant"] = (table["p_holm"] < 0.05) & table["boot_beats"] & family
        table.attrs["holm_family_size"] = int(family.sum())
    return table


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

    # Episode clustering: three events 5 days apart chain into one episode; a 4th event
    # 30 days later is its own episode.
    dates = pd.to_datetime(["2020-01-01", "2020-01-06", "2020-01-11", "2020-02-10"])
    eids = build_episode_ids(dates, gap_days=14)
    assert eids[0] == eids[1] == eids[2] != eids[3]

    # Cluster bootstrap must still detect a real effect, same as the plain bootstrap
    # above, when every event happens to be its own episode (well-separated dates).
    wide_dates = pd.date_range("2000-01-01", periods=n, freq="60D")
    cids = build_episode_ids(wide_dates, gap_days=14)
    rc = paired_bootstrap_delta(truth, good, bad, n_boot=2000, cluster_ids=cids)
    assert rc["significant"] and rc["delta"] > 0, rc

    print("verification.py self-check passed")
