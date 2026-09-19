"""Y3 improvement grid (ablations B2-B5),  censoring-aware recovery-duration modelling."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.collinearity import CollinearityRFTopK
from src.evaluation.survival_metrics import (PROB_TIMES, cluster_bootstrap_ci,
                                             harrell_c_index, integrated_brier_score,
                                             probability_calibration,
                                             uncensored_point_errors)
from src.evaluation.verification import build_episode_ids
from src.training.walk_forward import (MEDIAN_IMPUTE_COLS, generate_walk_forward_splits,
                                       median_impute_from_train, purge_horizon_overlap)
from src.utils.artifact_store import artifact_file

ART = ROOT / "artifacts"
RANDOM_STATE = 42
TRAIN_WINDOW, TEST_WINDOW, STEP = 30, 10, 10
CAP = 90.0
DURATION_EPS = 0.5          # same interval-censoring shift AFTRecoveryModel uses
PENALIZER = 0.1
TIME_GRID = np.arange(0.0, CAP + 1.0, 1.0)
# Pre-specified: Cox PH is fitted only where the effective event count supports it.
COX_MIN_EVENTS = 12
COX_MAX_FEATURES = 5

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ curve helpers

def _median_from_curve(surv_grid):
    """First grid time where S(t) <= 0.5, read numerically off the predicted curve.
    Uniform across every model here (AFT, two-stage mixture, Kaplan-Meier baseline), so
    no model's median is defined by a different convention than another's."""
    below = surv_grid <= 0.5
    out = np.full(surv_grid.shape[0], CAP)
    any_below = below.any(axis=1)
    out[any_below] = TIME_GRID[below[any_below].argmax(axis=1)]
    return out


def _aft_survival(model, X, grid=TIME_GRID):
    """S(t) on `grid` for each row of X. lifelines works on the shifted duration scale
    (`t + DURATION_EPS`), so the grid is shifted before the lookup and not after."""
    sf = model.predict_survival_function(X, times=grid + DURATION_EPS)
    return np.clip(sf.to_numpy().T, 0.0, 1.0)      # (n_rows, n_times)


def fit_aft(kind, X, durations, observed):
    from lifelines import LogNormalAFTFitter, WeibullAFTFitter
    from lifelines.exceptions import ConvergenceError

    fitter = (WeibullAFTFitter if kind == "weibull" else LogNormalAFTFitter)(penalizer=PENALIZER)
    df = pd.DataFrame(X).copy()
    df["_duration"] = np.asarray(durations, float) + DURATION_EPS
    df["_observed"] = np.asarray(observed, bool)
    if len(df) < 8 or df["_observed"].sum() < 4 or (~df["_observed"]).sum() < 1:
        return None                                  # degenerate fold: caller falls back to KM
    try:
        fitter.fit(df, duration_col="_duration", event_col="_observed")
    except (ConvergenceError, np.linalg.LinAlgError, ValueError):
        return None
    return fitter


def fit_cox(X, durations, observed):
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceError

    df = pd.DataFrame(X).copy()
    df["_duration"] = np.asarray(durations, float) + DURATION_EPS
    df["_observed"] = np.asarray(observed, bool)
    if df["_observed"].sum() < COX_MIN_EVENTS or df.shape[1] - 2 > COX_MAX_FEATURES:
        return None
    try:
        cph = CoxPHFitter(penalizer=PENALIZER)
        cph.fit(df, duration_col="_duration", event_col="_observed")
    except (ConvergenceError, np.linalg.LinAlgError, ValueError):
        return None
    return cph


def km_curve(durations, observed, n_rows):
    """Training-fold Kaplan-Meier baseline: the same marginal curve for every test row.
    This is the pre-declared Y3 survival baseline (protocol 3.3)."""
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter().fit(np.asarray(durations, float) + DURATION_EPS,
                                 event_observed=np.asarray(observed, bool))
    s = np.atleast_1d(np.asarray(km.predict(TIME_GRID + DURATION_EPS), dtype=float))
    return np.tile(np.clip(s, 0.0, 1.0), (n_rows, 1))


# ------------------------------------------------------------------ main

def main():
    data = pd.read_parquet(artifact_file("dataset.parquet")).reset_index(drop=True)
    feature_cols = json.loads((artifact_file("feature_spec.json")).read_text())["FEATURE_COLS"]
    event_dates = pd.to_datetime(data["event_date"])
    y = data["Y3_ASPI_Recovery_Time"].to_numpy(float)
    observed = ~data["Y3_censored"].to_numpy(bool)
    drawdown = data["Y3_drawdown_occurred"].to_numpy(bool)
    end_dates = data["Y3_label_end_date"]

    print(f"{len(data)} events | {observed.sum()} observed recoveries | "
          f"{(~observed).sum()} censored | {drawdown.sum()} with a post-event drawdown")
    print("censor reasons:", data["Y3_censor_reason"].value_counts().to_dict())

    splits = list(generate_walk_forward_splits(len(data), TRAIN_WINDOW, TEST_WINDOW, STEP))
    rows = []

    for fold_i, split in enumerate(splits):
        purged = purge_horizon_overlap(split, event_dates, end_dates)
        tr, te = purged.train_index, split.test_index
        if len(tr) < 10 or len(te) == 0:
            print(f"  fold {fold_i}: skipped (train={len(tr)})")
            continue

        X_tr_raw, X_te_raw = data.loc[tr, feature_cols], data.loc[te, feature_cols]
        X_tr, X_te = median_impute_from_train(
            X_tr_raw, X_te_raw, cols=[c for c in MEDIAN_IMPUTE_COLS if c in feature_cols])
        X_tr, X_te = X_tr.fillna(0.0), X_te.fillna(0.0)
        y_tr, obs_tr, dd_tr = y[tr], observed[tr], drawdown[tr]

        km_base = km_curve(y_tr, obs_tr, len(te))
        preds = {"km_train_baseline": km_base}
        # Constant training-fold median recovery, as a flat step curve, the second
        # pre-declared baseline (protocol 3.3).
        med = float(_median_from_curve(km_base[:1])[0])
        preds["train_median_baseline"] = np.tile(
            (TIME_GRID < med).astype(float), (len(te), 1))

        for k in (5, 10, 20):
            if k > len(feature_cols):
                continue
            sel = CollinearityRFTopK(k=k, random_state=RANDOM_STATE).fit(X_tr, y_tr)
            Xs_tr, Xs_te = sel.transform(X_tr), sel.transform(X_te)

            for kind in ("weibull", "lognormal"):
                m = fit_aft(kind, Xs_tr, y_tr, obs_tr)
                preds[f"aft_{kind}_k{k}"] = (_aft_survival(m, Xs_te) if m is not None
                                             else km_base.copy())

                # Two-stage: P(drawdown) x AFT fitted on drawdown rows only. A row with
                # no drawdown recovered at t=0 by construction, so its S(t) is 0 for
                # every t >= 0, the mixture is p * S_aft(t), not a shifted curve.
                if dd_tr.sum() >= 8 and (~dd_tr).sum() >= 2:
                    clf = Pipeline([("s", StandardScaler()),
                                    ("m", LogisticRegression(max_iter=5000, C=1.0))])
                    clf.fit(Xs_tr, dd_tr.astype(int))
                    p_dd = clf.predict_proba(Xs_te)[:, 1]
                    m2 = fit_aft(kind, Xs_tr[dd_tr], y_tr[dd_tr], obs_tr[dd_tr])
                    s_dd = (_aft_survival(m2, Xs_te) if m2 is not None
                            else km_curve(y_tr[dd_tr], obs_tr[dd_tr], len(te)))
                    preds[f"two_stage_{kind}_k{k}"] = p_dd[:, None] * s_dd

            if k <= COX_MAX_FEATURES:
                cox = fit_cox(Xs_tr, y_tr, obs_tr)
                if cox is not None:
                    sf = cox.predict_survival_function(Xs_te, times=TIME_GRID + DURATION_EPS)
                    preds[f"cox_k{k}"] = np.clip(sf.to_numpy().T, 0.0, 1.0)

        for name, surv in preds.items():
            median = _median_from_curve(surv)
            for j, row in enumerate(te):
                rec = {"fold": fold_i, "row": int(row), "model": name,
                       "duration": float(y[row]), "event_observed": bool(observed[row]),
                       "drawdown_occurred": bool(drawdown[row]),
                       "pred_median": float(median[j])}
                for t in PROB_TIMES:
                    s = float(np.interp(t, TIME_GRID, surv[j]))
                    rec[f"P_T_le_{int(t)}"] = 1.0 - s
                rec["_surv"] = [float(v) for v in np.interp(PROB_TIMES, TIME_GRID, surv[j])]
                rows.append(rec)
        print(f"  fold {fold_i}: train={len(tr)} test={len(te)} models={len(preds)}")

    oof = pd.DataFrame(rows)
    surv_mat = np.vstack(oof.pop("_surv").to_numpy())
    oof.to_parquet(artifact_file("recovery_grid_predictions.parquet"), index=False)
    print(f"\nwrote recovery_grid_predictions.parquet ({len(oof)} rows, {oof.model.nunique()} models)")

    # ----------------------------------------------------------- metrics
    episodes_all = build_episode_ids(event_dates)
    metrics, calib = [], []
    for name, g in oof.groupby("model", sort=False):
        mask = (oof["model"] == name).to_numpy()
        S = surv_mat[mask]
        d = g["duration"].to_numpy(float)
        e = g["event_observed"].to_numpy(bool)
        p = g["pred_median"].to_numpy(float)
        cl = episodes_all[g["row"].to_numpy()]

        c = harrell_c_index(d, e, p)
        lo, hi, draws = cluster_bootstrap_ci(lambda i: harrell_c_index(d[i], e[i], p[i]),
                                             cl, random_state=RANDOM_STATE, return_draws=True)
        # One-sided bootstrap p against chance concordance, for the Holm correction below.
        p_one_sided = float(np.mean(draws <= 0.5)) if len(draws) >= 100 else np.nan
        ibs = integrated_brier_score(d, e, S)
        row = {"model": name, "n": len(g), "n_recovered": int(e.sum()),
               "n_censored": int((~e).sum()), "c_index": c,
               "c_index_ci_low": lo, "c_index_ci_high": hi,
               "p_one_sided": p_one_sided,
               "c_index_beats_chance": bool(np.isfinite(lo) and lo > 0.5),
               "integrated_brier_score": ibs["integrated_brier_score"]}
        row.update({f"brier_t{int(t)}": v for t, v in ibs["brier_by_time"].items()})
        row.update(uncensored_point_errors(d, e, p))
        metrics.append(row)

        cal = probability_calibration(d, e, 1.0 - S)
        cal.insert(0, "model", name)
        calib.append(cal)

    mt = pd.DataFrame(metrics).sort_values("c_index", ascending=False)
    # Holm across the Y3 survival family (protocol 3.4), a SECONDARY strict diagnostic
    # reported beside the single-comparison CI, never folded into it.
    from statsmodels.stats.multitest import multipletests
    _, p_holm, _, _ = multipletests(mt["p_one_sided"].fillna(1.0).to_numpy(),
                                    alpha=0.05, method="holm")
    mt["p_holm"] = p_holm
    mt["holm_significant"] = (p_holm < 0.05) & mt["c_index_beats_chance"]
    mt["verdict"] = np.where(mt["c_index_beats_chance"], "A - statistically supported",
                             np.where(mt["c_index"] > 0.5, "B - suggestive but uncertain",
                                      "C - unsupported"))
    mt.to_parquet(artifact_file("recovery_grid_metrics.parquet"), index=False)
    pd.concat(calib, ignore_index=True).to_parquet(artifact_file("recovery_probability_calibration.parquet"),
                                                    index=False)
    print(f"wrote recovery_grid_metrics.parquet ({len(mt)} models), recovery_probability_calibration.parquet")

    # ----------------------------------------------------------- recovery categories
    cat = recovery_categories(oof, episodes_all)
    cat.to_parquet(artifact_file("recovery_category_metrics.parquet"), index=False)
    print(f"wrote recovery_category_metrics.parquet ({len(cat)} rows)")


def recovery_categories(oof, episodes_all, threshold=20):
    """Pre-specified category #1: `recovery <= 20 trading days`, scored directly from each
    survival model's own P(T <= 20), no separate classifier, no threshold sweep.
    Rows censored BEFORE the threshold are dropped: their status at t=20 is genuinely
    unknowable, the same rule the existing `C3_recovers_in_90` label already applies.
    """
    col = f"P_T_le_{threshold}"
    rows = []
    for name, g in oof.groupby("model", sort=False):
        known = g["event_observed"].to_numpy(bool) | (g["duration"].to_numpy(float) > threshold)
        g = g[known]
        if len(g) < 8:
            continue
        label = (g["duration"].to_numpy(float) <= threshold).astype(int)
        if len(np.unique(label)) < 2:
            continue
        prob = g[col].to_numpy(float)
        pred = (prob >= 0.5).astype(int)
        cl = episodes_all[g["row"].to_numpy()]
        auc = roc_auc_score(label, prob)
        lo, hi = cluster_bootstrap_ci(
            lambda i: (roc_auc_score(label[i], prob[i])
                       if len(np.unique(label[i])) > 1 else np.nan), cl,
            random_state=RANDOM_STATE)
        rows.append({"category": f"recovery_le_{threshold}d", "model": name, "n": len(g),
                     "n_positive": int(label.sum()), "prevalence": float(label.mean()),
                     "roc_auc": float(auc), "auc_ci_low": lo, "auc_ci_high": hi,
                     "balanced_accuracy": float(balanced_accuracy_score(label, pred)),
                     "beats_chance": bool(np.isfinite(lo) and lo > 0.5
                                          and balanced_accuracy_score(label, pred) > 0.5)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
