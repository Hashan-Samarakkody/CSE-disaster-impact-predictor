"""The pre-declared robustness suite, run once (Revision 2, T15).

The list of checks is frozen in docs/audit.md Part 8.7 and is CLOSED: no check was added
after a result was seen, none was dropped because its result was unfavourable, and the
suite is not run more than once. A check that could not be run appears in the table with
its reason rather than being omitted.

Every check varies exactly one thing against the principal analysis and reports the same
four quantities: the number of held-out observations, the error metric, the comparison
against the designated benchmark, and an uncertainty interval.

Principal analysis: the five-session ASPI return, Ridge selected by the purged inner cross
validation at the confirmatory capacity of ten features, on the real-time information set,
sliding window, scored against the zero-return benchmark on episode-clustered bootstrap
intervals.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.settings import RANDOM_STATE
from src.evaluation.collinearity import CollinearityRFTopK
from src.evaluation.verification import build_episode_ids, paired_bootstrap_delta
from src.targets.abnormal_returns import abnormal_return_col, abnormal_return_end_col
from src.targets.return_horizons import horizon_col, horizon_end_col, information_sets
from src.training.inner_cv import purged_inner_splits
from src.training.walk_forward import (MEDIAN_IMPUTE_COLS, generate_walk_forward_splits,
                                       median_impute_from_train, purge_horizon_overlap)
from src.utils.artifact_store import artifact_file, save_frame

warnings.filterwarnings("ignore")

TRAIN_WINDOW, TEST_WINDOW, STEP = 30, 10, 10
CAPACITY = 10
RIDGE_ALPHAS = np.logspace(-3, 3, 13)

# Crisis periods the review named, excluded and analysed separately.
CRISES = {
    "2004 Indian Ocean tsunami": ("2004-12-20", "2005-03-31"),
    "COVID-19 period": ("2020-03-01", "2020-12-31"),
    "2022 Sri Lankan economic crisis": ("2022-01-01", "2022-12-31"),
}
# Subgroups too small to carry a standalone finding, per the pre-declaration.
MIN_SUBGROUP = 10


def fit_fold(X_tr, y_tr, X_te, cv_splits):
    """Ridge at the confirmatory capacity, selected inside the purged inner CV."""
    selector = CollinearityRFTopK(k=CAPACITY, random_state=RANDOM_STATE)
    pipe = Pipeline([("sel", selector), ("scale", StandardScaler()), ("model", Ridge())])
    if cv_splits:
        search = GridSearchCV(pipe, {"model__alpha": list(RIDGE_ALPHAS)}, cv=cv_splits,
                              scoring="neg_root_mean_squared_error", n_jobs=1, refit=True)
        search.fit(X_tr, y_tr)
        return np.asarray(search.predict(X_te), dtype=float)
    pipe.set_params(model__alpha=1.0).fit(X_tr, y_tr)
    return np.asarray(pipe.predict(X_te), dtype=float)


def walk_forward_predictions(data, feature_cols, target, end_col, mode="sliding",
                             row_filter=None):
    """Out-of-fold predictions for one configuration. Returns (truth, prediction, rows)."""
    frame = data if row_filter is None else data.loc[row_filter].reset_index(drop=True)
    if len(frame) < TRAIN_WINDOW + TEST_WINDOW:
        return None
    event_dates = pd.to_datetime(frame["event_date"])
    y_all, end_all = frame[target], frame[end_col]

    truth, predicted, rows = [], [], []
    for split in generate_walk_forward_splits(len(frame), TRAIN_WINDOW, TEST_WINDOW,
                                              STEP, mode=mode):
        purged = purge_horizon_overlap(split, event_dates, end_all)
        train = purged.train_index[y_all.iloc[purged.train_index].notna().to_numpy()]
        test = split.test_index[y_all.iloc[split.test_index].notna().to_numpy()]
        if len(train) < 10 or len(test) == 0:
            continue
        X_tr_raw, X_te_raw = frame.loc[train, feature_cols], frame.loc[test, feature_cols]
        X_tr, X_te = median_impute_from_train(
            X_tr_raw, X_te_raw, cols=[c for c in MEDIAN_IMPUTE_COLS if c in feature_cols])
        X_tr, X_te = X_tr.fillna(0.0), X_te.fillna(0.0)
        cv_splits = purged_inner_splits(event_dates.iloc[train], end_all.iloc[train])
        predicted.append(fit_fold(X_tr, y_all.iloc[train].to_numpy(float), X_te, cv_splits))
        truth.append(y_all.iloc[test].to_numpy(float))
        rows.append(np.asarray(test))
    if not truth:
        return None
    return (np.concatenate(truth), np.concatenate(predicted), np.concatenate(rows),
            event_dates)


def score(check, variant, result, note=""):
    """One row of the consolidated table: n, metric, comparison, interval."""
    if result is None:
        return {"check": check, "variant": variant, "n_held_out": 0,
                "rmse_model": np.nan, "rmse_benchmark": np.nan, "delta_rmse": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "beats_benchmark": False,
                "note": note or "could not be run: too few events for the walk forward"}
    truth, predicted, rows, event_dates = result
    benchmark = np.zeros_like(truth)          # the pre-declared zero-return benchmark
    clusters = build_episode_ids(event_dates.to_numpy())[rows]
    boot = paired_bootstrap_delta(truth, predicted, benchmark, cluster_ids=clusters,
                                  random_state=RANDOM_STATE)
    return {"check": check, "variant": variant, "n_held_out": int(len(truth)),
            "rmse_model": float(np.sqrt(np.mean((predicted - truth) ** 2))),
            "rmse_benchmark": float(np.sqrt(np.mean(truth ** 2))),
            "delta_rmse": boot["delta"], "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"], "beats_benchmark": bool(boot["significant"]),
            "note": note}


def main() -> None:
    data = pd.read_parquet(artifact_file("dataset.parquet")).reset_index(drop=True)
    feature_cols = json.loads((artifact_file("feature_spec.json")).read_text())["FEATURE_COLS"]
    sets = information_sets(feature_cols)
    realtime = sets["real_time"]
    principal = dict(feature_cols=realtime, target=horizon_col(5), end_col=horizon_end_col(5))
    rows = []

    def run(check, variant, note="", **kwargs):
        merged = {**principal, **kwargs}
        rows.append(score(check, variant, walk_forward_predictions(data, **merged), note))
        print(f"  {check:34s} {variant:38s} n={rows[-1]['n_held_out']:3d} "
              f"delta={rows[-1]['delta_rmse']:+.4f}")

    print("principal analysis")
    run("principal", "five sessions, real-time, sliding")

    print("\n1. exact event dates only")
    run("exact dates", "date_is_exact == 1",
        row_filter=data["date_is_exact"] == 1)

    print("\n2. return horizons")
    for h in (5, 10, 20):
        run("horizon", f"{h} sessions", target=horizon_col(h), end_col=horizon_end_col(h))

    print("\n3. raw against market-adjusted abnormal returns")
    run("return definition", "raw forward return")
    run("return definition", "market-adjusted abnormal return",
        target=abnormal_return_col(5), end_col=abnormal_return_end_col(5))

    print("\n4. information sets by availability")
    for name in ("real_time", "ex_post"):
        run("information set", name, feature_cols=sets[name])

    print("\n5. crisis periods")
    for name, (start, end) in CRISES.items():
        inside = ((pd.to_datetime(data["event_date"]) >= start)
                  & (pd.to_datetime(data["event_date"]) <= end))
        run("crisis period", f"excluding {name}", row_filter=~inside)
        note = (f"{int(inside.sum())} events in the period, too few for a walk forward"
                if inside.sum() < TRAIN_WINDOW + TEST_WINDOW else "")
        run("crisis period", f"only {name}", note=note, row_filter=inside)

    print("\n6. training window mode")
    for mode in ("sliding", "expanding"):
        run("window mode", mode, mode=mode)

    print("\n7. influential events")
    principal_result = walk_forward_predictions(data, **principal)
    if principal_result is not None:
        truth, predicted, held_rows, _ = principal_result
        errors = np.abs(predicted - truth)
        worst = held_rows[np.argsort(errors)[-5:]]
        keep = ~data.index.isin(worst)
        run("influential events", "excluding the 5 most influential", row_filter=keep)
        loo = []
        for row in held_rows:
            mask = np.ones(len(truth), bool)
            mask[held_rows == row] = False
            loo.append(float(np.sqrt(np.mean((predicted[mask] - truth[mask]) ** 2))))
        rows.append({"check": "influential events", "variant": "leave one event out",
                     "n_held_out": int(len(truth)),
                     "rmse_model": float(np.mean(loo)),
                     "rmse_benchmark": float(np.sqrt(np.mean(truth ** 2))),
                     "delta_rmse": np.nan, "ci_low": float(np.min(loo)),
                     "ci_high": float(np.max(loo)), "beats_benchmark": False,
                     "note": "interval is the range of the leave-one-out RMSE, not a bootstrap"})
        print(f"  leave-one-out RMSE range: [{np.min(loo):.4f}, {np.max(loo):.4f}]")

    print("\n8. disaster type subgroups")
    for disaster_type, size in data["disaster_type"].value_counts().items():
        inside = data["disaster_type"] == disaster_type
        note = f"subgroup size {int(size)}"
        if size < MIN_SUBGROUP:
            note += "; pre-declared as too small to carry a standalone finding"
        run("disaster type", f"{disaster_type} (n={int(size)})", note=note, row_filter=inside)

    print("\n9. overlapping event windows and clustered disasters")
    episodes = build_episode_ids(pd.to_datetime(data["event_date"]).to_numpy())
    clustered = pd.Series(episodes, index=data.index).duplicated(keep=False)
    run("event clustering", f"excluding {int(clustered.sum())} clustered events",
        note="events sharing a 14-day disaster episode with another event",
        row_filter=~clustered)

    print("\n10 and 11. oversampling and the highest-missingness variables")
    rows.append({"check": "oversampling", "variant": "without SMOGN",
                 "n_held_out": 0, "rmse_model": np.nan, "rmse_benchmark": np.nan,
                 "delta_rmse": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                 "beats_benchmark": False,
                 "note": "not applicable to this configuration: SMOGN is off by default "
                         "for the return targets, so the principal analysis already is "
                         "the without-SMOGN run. The with/without comparison for Y2 and "
                         "Y3 is in artifacts/tables/smogn_ablation.parquet"})
    missingness = data[feature_cols].isna().mean()
    high_missing = sorted(missingness[missingness > 0.5].index)
    kept = [c for c in realtime if c not in high_missing]
    dropped = [c for c in realtime if c in high_missing]
    if dropped:
        note = f"dropped from the principal set: {', '.join(dropped)}"
    else:
        # Reported rather than quietly shown as an identical number: every variable above
        # 50 percent missing is an EM-DAT or hazard column, and all of those are ex post,
        # so none of them is in the real-time principal set to begin with.
        note = (f"no effect by construction: all {len(high_missing)} variables above 50% "
                f"missing ({', '.join(high_missing)}) are ex post, so none is in the "
                "real-time principal information set")
    run("missingness", f"dropping the {len(high_missing)} variables above 50% missing",
        note=note, feature_cols=kept)

    table = pd.DataFrame(rows)
    save_frame(table, "robustness_suite",
               "Pre-declared robustness checks (docs/audit.md Part 8.7), run once.")
    print(f"\nwrote robustness_suite.parquet ({len(table)} checks, "
          f"{int(table['n_held_out'].eq(0).sum())} not runnable and reported with a reason)")


if __name__ == "__main__":
    main()
