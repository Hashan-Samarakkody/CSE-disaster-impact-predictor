"""Y1 improvement grid,  everything pre-declared in docs/audit.md Part 2."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.collinearity import CollinearityRFTopK
from src.evaluation.verification import build_episode_ids, paired_bootstrap_delta
from src.targets.abnormal_returns import stage_a_expected_returns
from src.targets.return_horizons import (HORIZONS, horizon_col, horizon_end_col,
                                        information_sets)
from src.training.walk_forward import (MEDIAN_IMPUTE_COLS, generate_walk_forward_splits,
                                       median_impute_from_train, purge_horizon_overlap)
from src.utils.artifact_store import artifact_file

ART = ROOT / "artifacts"
RANDOM_STATE = 42
TRAIN_WINDOW, TEST_WINDOW, STEP = 30, 10, 10
CAPACITIES = (5, 10, 20)
INFO_SETS = ("market_only", "disaster_only", "combined", "normal_plus_residual",
             "real_time", "ex_post")
MODELS = ("ridge", "elastic_net", "random_forest", "xgboost", "mlp")
BASELINES = ("naive_zero", "naive_train_mean", "market_only_expected")

# The confirmatory family, pre-declared in docs/audit.md Part 8.4 before this grid was
# re-run (T7). Everything outside it is exploratory and is written to its own artifact.
# The grid still runs in full; only what the family-wise correction spans has changed.
CONFIRMATORY_HORIZON = 5
CONFIRMATORY_INFO_SETS = ("real_time", "ex_post")
CONFIRMATORY_CAPACITY = 10
CONFIRMATORY_MODELS = ("ridge", "random_forest", "mlp")


def is_confirmatory(horizon, info_set, k, model) -> bool:
    """A comparison counts as confirmatory only if all four conditions hold."""
    return (horizon == CONFIRMATORY_HORIZON
            and info_set in CONFIRMATORY_INFO_SETS
            and k == CONFIRMATORY_CAPACITY
            and model in CONFIRMATORY_MODELS)

RIDGE_ALPHAS = np.logspace(-3, 3, 13)
ENET_GRID = {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.2, 0.5, 0.8]}
RF_GRID = {"model__n_estimators": [200], "model__max_depth": [3, None],
           "model__min_samples_leaf": [1, 4]}
XGB_GRID = {"model__n_estimators": [100, 200], "model__max_depth": [2, 3],
            "model__learning_rate": [0.05]}
# Deliberately small: parity with the other confirmatory models, not a wider search.
MLP_GRID = {"model__hidden_layer_sizes": [(8,), (16,)], "model__alpha": [1.0, 10.0]}
# Pre-specified defaults, used when purging leaves too few rows for ANY inner split --
# protocol section 1.7: reduce splits first, fall back to these second, NEVER unpurge.
DEFAULTS = {"ridge": 1.0, "elastic_net": (0.1, 0.5), "random_forest": (200, None, 1),
            "xgboost": (100, 3, 0.05), "mlp": ((16,), 1.0)}

warnings.filterwarnings("ignore")

_SELECTOR_CACHE = Path(tempfile.mkdtemp(prefix="y1_selector_cache_"))


# inner CV

def purged_inner_splits(dates, label_end, n_splits=3):
    """Purged temporal inner CV (`notebooks/_shared.purged_inner_cv`'s rule), but
    degrading the way protocol section 1.7 requires instead of reverting to an unpurged
    split: try `n_splits`, then fewer, and return `[]` (caller uses fixed defaults) if
    even 2 splits cannot survive the purge. Leakage protection is never removed."""
    from sklearn.model_selection import TimeSeriesSplit

    n = len(dates)
    dates = pd.Series(dates).reset_index(drop=True)
    label_end = pd.Series(label_end).reset_index(drop=True)
    for k in range(min(n_splits, n - 1), 1, -1):
        purged = []
        for tr, val in TimeSeriesSplit(n_splits=k).split(np.arange(n)):
            if len(tr) == 0 or len(val) == 0:
                continue
            first_val = dates.iloc[val[0]]
            end = label_end.iloc[tr]
            keep = (end.isna() | (end < first_val)).to_numpy()
            # An inner split needs enough surviving training rows to fit anything at all.
            if keep.sum() >= 5:
                purged.append((tr[keep], val))
        if len(purged) >= 2:
            return purged
    return []


# estimators

def _pipeline(steps):
    from joblib import Memory

    return Pipeline(steps, memory=Memory(location=str(_SELECTOR_CACHE), verbose=0))


def build_search(name, k, cv_splits):
    """(estimator, needs_fit_with_grid). Selection is NESTED inside the inner CV
    (`CollinearityRFTopK` is a pipeline step), so the selector refits on every inner
    split and every hyperparameter candidate."""
    sel = CollinearityRFTopK(k=k, random_state=RANDOM_STATE)
    if name == "ridge":
        pipe = _pipeline([("sel", sel), ("scale", StandardScaler()), ("model", Ridge())])
        grid = {"model__alpha": list(RIDGE_ALPHAS)}
    elif name == "elastic_net":
        pipe = _pipeline([("sel", sel), ("scale", StandardScaler()),
                          ("model", ElasticNet(max_iter=20_000))])
        grid = ENET_GRID
    elif name == "random_forest":
        pipe = _pipeline([("sel", sel),
                          ("model", RandomForestRegressor(random_state=RANDOM_STATE))])
        grid = RF_GRID
    elif name == "xgboost":
        pipe = _pipeline([("sel", sel),
                          ("model", XGBRegressor(random_state=RANDOM_STATE, verbosity=0))])
        grid = XGB_GRID
    elif name == "mlp":
        # T8: the MLP is in the confirmatory family, so it is selected by the SAME purged
        # inner cross-validation as Ridge and Random Forest. Ranking a tuned model against
        # an untuned one is not a comparison, and this model currently carries the study's
        # only positive continuous result, so it must earn it under the same discipline.
        # The grid is the smallest one that gives parity: two widths by two penalties.
        pipe = _pipeline([("sel", sel), ("scale", StandardScaler()),
                          ("model", MLPRegressor(max_iter=4000, early_stopping=False,
                                                 random_state=RANDOM_STATE))])
        grid = MLP_GRID
    else:
        raise ValueError(name)

    if not cv_splits:
        # Pre-specified defaults (protocol 1.7 step 2).
        if name == "ridge":
            pipe.set_params(model__alpha=DEFAULTS["ridge"])
        elif name == "elastic_net":
            a, l1 = DEFAULTS["elastic_net"]
            pipe.set_params(model__alpha=a, model__l1_ratio=l1)
        elif name == "random_forest":
            n, d, leaf = DEFAULTS["random_forest"]
            pipe.set_params(model__n_estimators=n, model__max_depth=d, model__min_samples_leaf=leaf)
        elif name == "xgboost":
            n, d, lr = DEFAULTS["xgboost"]
            pipe.set_params(model__n_estimators=n, model__max_depth=d, model__learning_rate=lr)
        else:
            hidden, alpha = DEFAULTS["mlp"]
            pipe.set_params(model__hidden_layer_sizes=hidden, model__alpha=alpha)
        return pipe, None
    return GridSearchCV(pipe, grid, cv=cv_splits, scoring="neg_root_mean_squared_error",
                        n_jobs=1, refit=True), grid


# Stage A

# main grid

def main():
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))
    market = pd.read_parquet(artifact_file("market.parquet"))
    market_feats = pd.read_parquet(artifact_file("market_feats.parquet"))
    feature_cols = json.loads((artifact_file("feature_spec.json")).read_text())["FEATURE_COLS"]
    event_dates = pd.to_datetime(dataset["event_date"]).reset_index(drop=True)

    # Horizon targets come straight from dataset.parquet, which event_targets.py built
    # under the frozen protocol. Rebuilding them here would be a second definition.
    data = dataset.reset_index(drop=True)
    sets = information_sets(feature_cols)

    md = market.sort_values("date").reset_index(drop=True)
    mdates = pd.to_datetime(md["date"]).to_numpy()
    positions = []
    for d in event_dates:
        cand = np.flatnonzero(mdates >= np.datetime64(d))
        positions.append(int(cand[0]) if len(cand) and cand[0] > 0 else None)

    cache = artifact_file("aspi_expected_return_market_only.parquet")
    if cache.exists():
        cached = pd.read_parquet(cache)
        expected = {h: cached[str(h)].to_numpy(float) for h in HORIZONS}
        print(f"Stage A: reusing {cache.name}")
    else:
        print("Stage A: fitting per-event normal-market expected-return models ...")
        expected = stage_a_expected_returns(market_feats, positions, HORIZONS)
        pd.DataFrame({str(h): expected[h] for h in HORIZONS}).to_parquet(cache, index=False)
    for h in HORIZONS:
        print(f"  h={h:2d}: {np.isfinite(expected[h]).sum()}/{len(expected[h])} events with an estimate")

    splits = list(generate_walk_forward_splits(len(data), TRAIN_WINDOW, TEST_WINDOW, STEP))
    print(f"{len(splits)} outer folds, train={TRAIN_WINDOW} test={TEST_WINDOW} step={STEP}")

    oof_rows, stability_rows = [], []

    for h in HORIZONS:
        ycol, endcol = horizon_col(h), horizon_end_col(h)
        y_all = data[ycol]
        end_all = data[endcol]
        exp_h = expected[h]

        for fold_i, split in enumerate(splits):
            purged = purge_horizon_overlap(split, event_dates, end_all)
            tr = purged.train_index[y_all.iloc[purged.train_index].notna().to_numpy()]
            te = split.test_index[y_all.iloc[split.test_index].notna().to_numpy()]
            if len(tr) < 10 or len(te) == 0:
                print(f"  h={h} fold {fold_i}: skipped (train={len(tr)} test={len(te)})")
                continue

            y_tr, y_te = y_all.iloc[tr].to_numpy(float), y_all.iloc[te].to_numpy(float)
            cv_splits = purged_inner_splits(event_dates.iloc[tr], end_all.iloc[tr])

            # Baselines, identical test rows for every candidate (protocol 3.1).
            base_pred = {"naive_zero": np.zeros(len(te)),
                         "naive_train_mean": np.full(len(te), float(np.mean(y_tr))),
                         "market_only_expected": np.where(np.isfinite(exp_h[te]), exp_h[te], 0.0)}
            for bname, bp in base_pred.items():
                for j, row in enumerate(te):
                    oof_rows.append({"horizon": h, "info_set": "baseline", "k": 0,
                                     "model": bname, "fold": fold_i, "row": int(row),
                                     "y_true": y_te[j], "y_pred": float(bp[j])})

            for info in INFO_SETS:
                residual = info == "normal_plus_residual"
                cols = sets["disaster_only"] if residual else sets[info]
                # Stage A needs >= 250 prior daily rows, so the earliest few events have no
                tr_rows, y_fit = tr, y_tr
                if residual:
                    if not np.isfinite(exp_h[te]).all():
                        continue
                    keep = np.isfinite(exp_h[tr])
                    if keep.sum() < 10:
                        continue
                    tr_rows, y_fit = tr[keep], y_tr[keep]

                X_tr_raw, X_te_raw = data.loc[tr_rows, cols], data.loc[te, cols]
                X_tr, X_te = median_impute_from_train(
                    X_tr_raw, X_te_raw, cols=[c for c in MEDIAN_IMPUTE_COLS if c in cols])
                X_tr, X_te = X_tr.fillna(0.0), X_te.fillna(0.0)

                if residual:
                    fit_target = y_fit - exp_h[tr_rows]  # Stage B learns the disaster residual
                    offset = exp_h[te]
                else:
                    fit_target, offset = y_fit, np.zeros(len(te))

                # The inner splits index INTO the rows being fitted, so the residual
                # architecture (which drops train rows lacking a Stage-A estimate) needs
                # its own, not the fold-wide set.
                cv_local = (purged_inner_splits(event_dates.iloc[tr_rows], end_all.iloc[tr_rows])
                            if residual else cv_splits)

                for k in CAPACITIES:
                    if k > len(cols):
                        continue
                    # Fold-level selection record (train rows only) for the stability table.
                    selector = CollinearityRFTopK(k=k, random_state=RANDOM_STATE).fit(X_tr, fit_target)
                    for rank, feat in enumerate(selector.selected_, start=1):
                        stability_rows.append({"horizon": h, "info_set": info, "k": k,
                                               "fold": fold_i, "feature": feat, "rank": rank})

                    for model_name in MODELS:
                        est, _ = build_search(model_name, k, cv_local)
                        est.fit(X_tr, fit_target)
                        pred = np.asarray(est.predict(X_te), dtype=float) + offset
                        for j, row in enumerate(te):
                            oof_rows.append({"horizon": h, "info_set": info, "k": k,
                                             "model": model_name, "fold": fold_i,
                                             "row": int(row), "y_true": y_te[j],
                                             "y_pred": float(pred[j])})
            print(f"  h={h:2d} fold {fold_i}: train={len(tr)} test={len(te)} "
                  f"inner_splits={len(cv_splits) if cv_splits else 'defaults'}")

    oof = pd.DataFrame(oof_rows)
    oof.to_parquet(artifact_file("aspi_grid_predictions.parquet"), index=False)
    print(f"\nwrote aspi_grid_predictions.parquet ({len(oof)} rows)")

    # metrics + verdicts
    metrics, verdicts = [], []
    episode_ids_all = build_episode_ids(event_dates)

    for (h, info, k, model), g in oof.groupby(["horizon", "info_set", "k", "model"], sort=False):
        g = g.sort_values(["fold", "row"])
        yt, yp = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        zero_rmse = float(np.sqrt(np.mean(yt ** 2)))
        metrics.append({
            "horizon": h, "info_set": info, "k": k, "model": model, "n": len(g),
            "confirmatory": is_confirmatory(h, info, k, model),
            "rmse": rmse, "mae": float(np.mean(np.abs(yp - yt))),
            "pooled_r2": float(1 - np.sum((yp - yt) ** 2) / np.sum((yt - yt.mean()) ** 2)),
            "skill_vs_zero": float(1 - rmse / zero_rmse) if zero_rmse else np.nan,
        })

        if info == "baseline":
            continue
        for bname in BASELINES:
            b = oof[(oof.horizon == h) & (oof.model == bname)].set_index(["fold", "row"])
            bp = b.reindex(pd.MultiIndex.from_frame(g[["fold", "row"]]))["y_pred"].to_numpy()
            if not np.isfinite(bp).all():
                continue
            cluster = episode_ids_all[g["row"].to_numpy()]
            boot = paired_bootstrap_delta(yt, yp, bp, cluster_ids=cluster,
                                          random_state=RANDOM_STATE)
            verdicts.append({
                "horizon": h, "info_set": info, "k": k, "model": model, "baseline": bname,
                "confirmatory": is_confirmatory(h, info, k, model),
                "n": boot["n"], "rmse_model": rmse,
                "rmse_baseline": float(np.sqrt(np.mean((bp - yt) ** 2))),
                "delta_rmse": boot["delta"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                "p_one_sided": boot["p_model_worse"], "boot_beats": boot["significant"],
            })

    mt = pd.DataFrame(metrics)
    mt.to_parquet(artifact_file("aspi_grid_metrics.parquet"), index=False)

    vt = pd.DataFrame(verdicts)
    if len(vt):
        from statsmodels.stats.multitest import multipletests
        # T7: the family-wise correction spans the CONFIRMATORY family only. Correcting
        # across all 720 comparisons made the ranking indistinguishable from selection
        # noise; correcting across a family nobody pre-declared would be worse.
        vt["p_holm"] = np.nan
        confirmatory = vt["confirmatory"].to_numpy(bool)
        if confirmatory.any():
            _, p_holm, _, _ = multipletests(
                vt.loc[confirmatory, "p_one_sided"].fillna(1.0).to_numpy(),
                alpha=0.05, method="holm")
            vt.loc[confirmatory, "p_holm"] = p_holm
        vt["holm_significant"] = (vt["p_holm"] < 0.05) & vt["boot_beats"] & confirmatory
        vt["verdict"] = np.where(vt["boot_beats"], "A - statistically supported",
                                 np.where(vt["delta_rmse"] > 0, "B - suggestive but uncertain",
                                          "C - unsupported"))
        vt.loc[~confirmatory, "verdict"] = "exploratory, not corrected"

    primary = vt[vt["confirmatory"]] if len(vt) else vt
    exploratory = vt[~vt["confirmatory"]] if len(vt) else vt
    primary.to_parquet(artifact_file("aspi_grid_verdicts.parquet"), index=False)
    exploratory.to_parquet(artifact_file("aspi_grid_verdicts_exploratory.parquet"), index=False)
    print(f"wrote aspi_grid_metrics.parquet ({len(mt)} configs)")
    print(f"wrote aspi_grid_verdicts.parquet: CONFIRMATORY family of {len(primary)} "
          f"comparisons (h={CONFIRMATORY_HORIZON}, {CONFIRMATORY_INFO_SETS}, "
          f"k={CONFIRMATORY_CAPACITY}, {CONFIRMATORY_MODELS}), "
          f"{int(primary['boot_beats'].sum()) if len(primary) else 0} with a CI excluding zero, "
          f"{int(primary['holm_significant'].sum()) if len(primary) else 0} surviving Holm")
    print(f"wrote aspi_grid_verdicts_exploratory.parquet ({len(exploratory)} comparisons, "
          f"never corrected jointly with the confirmatory family)")

    # feature stability
    st = pd.DataFrame(stability_rows)
    n_folds = st.groupby(["horizon", "info_set", "k"])["fold"].nunique().rename("n_folds")
    stab = (st.groupby(["horizon", "info_set", "k", "feature"])
              .agg(selection_count=("fold", "nunique"), mean_rank=("rank", "mean"),
                   median_rank=("rank", "median"))
              .join(n_folds).reset_index())
    stab["selection_frequency"] = stab["selection_count"] / stab["n_folds"]
    stab.sort_values(["horizon", "info_set", "k", "selection_frequency", "mean_rank"],
                     ascending=[True, True, True, False, True], inplace=True)
    stab.to_parquet(artifact_file("aspi_grid_feature_stability.parquet"), index=False)
    print(f"wrote aspi_grid_feature_stability.parquet ({len(stab)} rows)")

    # direction (secondary)
    direction = run_direction_analysis(data, event_dates, splits, sets["combined"])
    direction.to_parquet(artifact_file("aspi_direction_metrics.parquet"), index=False)
    print(f"wrote aspi_direction_metrics.parquet ({len(direction)} rows)")


def run_direction_analysis(data, event_dates, splits, combined_cols, k=10):
    """Protocol 1.6, a DIFFERENT question from magnitude regression, reported as one.
    `negative_return = Y_h < 0`, same folds, same purge, combined information set, K=10."""
    rows = []
    for h in HORIZONS:
        ycol, endcol = horizon_col(h), horizon_end_col(h)
        pooled = {"logistic": [], "random_forest": []}
        truth, idx = [], []
        for split in splits:
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
            sel = CollinearityRFTopK(k=k, random_state=RANDOM_STATE).fit(X_tr, lab_tr)
            Xs_tr, Xs_te = sel.transform(X_tr), sel.transform(X_te)
            for name, clf in (("logistic", Pipeline([("s", StandardScaler()),
                                                     ("m", LogisticRegression(max_iter=5000))])),
                              ("random_forest", RandomForestClassifier(
                                  n_estimators=300, min_samples_leaf=2,
                                  random_state=RANDOM_STATE))):
                clf.fit(Xs_tr, lab_tr)
                pooled[name].append(clf.predict_proba(Xs_te)[:, 1])
            truth.append(lab_te)
            idx.append(te)
        if not truth:
            continue
        y = np.concatenate(truth)
        episodes = build_episode_ids(event_dates.iloc[np.concatenate(idx)])
        for name, chunks in pooled.items():
            p = np.concatenate(chunks)
            pred = (p >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
            lo, hi, draws = _cluster_auc_ci(y, p, episodes)
            rows.append({
                "horizon": h, "model": name, "n": len(y), "n_negative": int(y.sum()),
                "prevalence": float(y.mean()),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "roc_auc": float(auc), "pr_auc": float(average_precision_score(y, p)),
                "mcc": float(matthews_corrcoef(y, pred)),
                "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
                "auc_ci_low": lo, "auc_ci_high": hi,
                "p_one_sided": float(np.mean(draws <= 0.5)) if len(draws) >= 100 else np.nan,
                "auc_ci_excludes_chance": bool(np.isfinite(lo) and lo > 0.5),
            })
    table = pd.DataFrame(rows)
    if len(table):
        # Holm across the 8 direction comparisons (protocol 3.4), a SECONDARY strict
        # diagnostic beside the single-comparison CI, never folded into it.
        from statsmodels.stats.multitest import multipletests
        _, p_holm, _, _ = multipletests(table["p_one_sided"].fillna(1.0).to_numpy(),
                                        alpha=0.05, method="holm")
        table["p_holm"] = p_holm
        table["holm_significant"] = (p_holm < 0.05) & table["auc_ci_excludes_chance"]
    return table


def _cluster_auc_ci(y, p, cluster_ids, n_boot=2000, random_state=RANDOM_STATE):
    """Episode-clustered bootstrap CI for ROC-AUC (protocol 3.2, adjacent disasters are
    not independent draws)."""
    rng = np.random.default_rng(random_state)
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


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(_SELECTOR_CACHE, ignore_errors=True)
