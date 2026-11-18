"""Episode-level permutation test for ASPI direction classifiers.

For each of the 10 (horizon, model) combinations, runs B permutations where
y_true labels are shuffled at the episode level (preserving within-episode
assignment) and the pooled ROC AUC is recomputed. The p-value is

    p = (1 + count(perm_auc >= observed_auc)) / (1 + B)

The primary comparison is horizon=10, model=logistic (the pre-declared
confirmatory classifier). All 10 comparisons are saved so Holm correction
can be applied.

Output: artifacts/tables/aspi_direction_permutation.parquet

--refit mode (A1):
    For the primary comparison (h=10, logistic) only, each permutation also
    re-runs the entire walk-forward procedure (CollinearityRFTopK k=10,
    StandardScaler, LogisticRegression) on the permuted labels, so the
    feature selection and model fit react to the permuted signal.
    For the other 9 (horizon, model) pairs the existing fixed-prediction
    p-values are carried over unchanged.
    Holm correction is then recomputed across all 10 comparisons.

    Output: artifacts/tables/aspi_direction_permutation_refit.parquet

Usage (repo root):
    python scripts/run_direction_permutation_test.py           # B=5000
    python scripts/run_direction_permutation_test.py --b 1000  # quick test
    python scripts/run_direction_permutation_test.py --refit   # refit primary
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.collinearity import CollinearityRFTopK
from src.targets.return_horizons import (HORIZONS, horizon_col, horizon_end_col,
                                         information_sets)
from src.training.walk_forward import (MEDIAN_IMPUTE_COLS,
                                       generate_walk_forward_splits,
                                       median_impute_from_train,
                                       purge_horizon_overlap)
from src.utils.artifact_store import artifact_file, save_frame

_TRAIN_WINDOW, _TEST_WINDOW, _STEP = 30, 10, 10
_RANDOM_STATE = 42
_K = 10


def _pooled_auc(df: pd.DataFrame) -> float:
    """ROC AUC on the concatenated (y_true, y_prob) across all folds."""
    return float(roc_auc_score(df["y_true"], df["y_prob"]))


def _permuted_auc(df: pd.DataFrame, episode_ids: np.ndarray,
                  rng: np.random.Generator) -> float:
    """
    Episode-level permutation: shuffle the y_true values across episodes
    (keeping within-episode label assignments intact) and recompute AUC.

    Within each episode, every row gets the same permuted label drawn from
    the original label for that episode position.  Here episodes are
    individual events (each event appears exactly once in the test set), so
    this simplifies to permuting the y_true array by episode.
    """
    unique_eps = np.unique(episode_ids)
    # Map each episode to its rows
    ep_to_rows: dict[int, list[int]] = {}
    for idx, ep in enumerate(episode_ids):
        ep_to_rows.setdefault(int(ep), []).append(idx)

    # Build the original y_true per episode (single value per episode here)
    ep_labels = {ep: df["y_true"].iloc[rows[0]] for ep, rows in ep_to_rows.items()}

    # Permute: shuffle episode label assignments
    eps_arr = np.array(unique_eps)
    perm_labels = rng.permutation(np.array([ep_labels[e] for e in eps_arr]))
    ep_perm = dict(zip(eps_arr, perm_labels))

    # Assign permuted labels back
    perm_y_true = np.array([ep_perm[int(e)] for e in episode_ids])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(roc_auc_score(perm_y_true, df["y_prob"].to_numpy()))
        except ValueError:
            # All one class after permutation — return 0.5 (chance)
            return 0.5


def _run_one(group_key: tuple, group: pd.DataFrame, B: int, seed: int) -> dict:
    horizon, model = group_key
    obs_auc = _pooled_auc(group)
    episode_ids = group["episode_id"].to_numpy()

    rng = np.random.default_rng(seed)
    perm_aucs = np.array(
        Parallel(n_jobs=-1, backend="loky")(
            delayed(lambda s=rng.integers(0, 2**31): _permuted_auc(
                group, episode_ids, np.random.default_rng(s)
            ))()
            for _ in range(B)
        )
    )

    p_value = float((1 + np.sum(perm_aucs >= obs_auc)) / (1 + B))
    return {
        "horizon": int(horizon),
        "model": str(model),
        "n_test": len(group),
        "n_permutations": B,
        "observed_auc": obs_auc,
        "perm_auc_mean": float(perm_aucs.mean()),
        "perm_auc_sd": float(perm_aucs.std()),
        "perm_auc_p95": float(np.percentile(perm_aucs, 95)),
        "p_value": p_value,
        "primary": bool(horizon == 10 and model == "logistic"),
    }


def _load_refit_data():
    """Load data needed to run the full direction walk-forward."""
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))
    feature_cols = json.loads(artifact_file("feature_spec.json").read_text())["FEATURE_COLS"]
    data = dataset.reset_index(drop=True)
    event_dates = pd.to_datetime(data["event_date"]).reset_index(drop=True)
    sets = information_sets(feature_cols)
    combined_cols = sets["combined"]
    splits = list(generate_walk_forward_splits(len(data), _TRAIN_WINDOW, _TEST_WINDOW, _STEP))
    return data, event_dates, splits, combined_cols


def _refit_one_permutation(perm_seed: int, data: pd.DataFrame,
                            event_dates: pd.Series,
                            splits: list,
                            combined_cols: list,
                            h: int = 10) -> float:
    """
    One full-refit permutation for the direction classifier at horizon h.
    Permutes y_true at episode level across ALL events, then runs the
    complete walk-forward (feature selection, scaler, model) per fold.
    Returns pooled ROC AUC on the permuted labels.
    """
    rng = np.random.default_rng(perm_seed)
    ycol = horizon_col(h)
    endcol = horizon_end_col(h)

    # Build the full label array and episode ids for all labelled events
    all_valid_idx = np.where(data[ycol].notna().to_numpy())[0]
    all_lab = (data[ycol].iloc[all_valid_idx].to_numpy() < 0).astype(int)

    # Episode-level permutation on the full label set
    # Each event is its own episode (episode_id == row index)
    perm_lab = rng.permutation(all_lab)
    # Map original row index to permuted label
    perm_label_map = dict(zip(all_valid_idx, perm_lab))

    pooled_prob, pooled_true = [], []

    for split in splits:
        purged = purge_horizon_overlap(split, event_dates, data[endcol])
        tr_base = purged.train_index[data[ycol].iloc[purged.train_index].notna().to_numpy()]
        te = split.test_index[data[ycol].iloc[split.test_index].notna().to_numpy()]
        if len(tr_base) < 10 or len(te) == 0:
            continue

        lab_tr = np.array([perm_label_map[i] for i in tr_base])
        lab_te = np.array([perm_label_map[i] for i in te])

        if len(np.unique(lab_tr)) < 2:
            continue

        X_tr, X_te = median_impute_from_train(
            data.loc[tr_base, combined_cols], data.loc[te, combined_cols],
            cols=[c for c in MEDIAN_IMPUTE_COLS if c in combined_cols])
        X_tr, X_te = X_tr.fillna(0.0), X_te.fillna(0.0)

        sel = CollinearityRFTopK(k=_K, random_state=_RANDOM_STATE).fit(X_tr, lab_tr)
        Xs_tr = sel.transform(X_tr)
        Xs_te = sel.transform(X_te)

        clf = Pipeline([("s", StandardScaler()),
                        ("m", LogisticRegression(max_iter=5000))])
        clf.fit(Xs_tr, lab_tr)
        pooled_prob.append(clf.predict_proba(Xs_te)[:, 1])
        pooled_true.append(lab_te)

    if not pooled_prob:
        return 0.5

    y = np.concatenate(pooled_true)
    p = np.concatenate(pooled_prob)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(roc_auc_score(y, p))
        except ValueError:
            return 0.5


def _run_refit(B: int, seed: int) -> dict:
    """
    Full-refit permutation for h=10, logistic (the primary comparison).
    Returns a result dict compatible with _run_one output.
    """
    t0 = time.time()
    data, event_dates, splits, combined_cols = _load_refit_data()

    # Observed AUC from the saved predictions (unchanged)
    preds = pd.read_parquet(artifact_file("aspi_direction_predictions.parquet"))
    primary_preds = preds[(preds["horizon"] == 10) & (preds["model"] == "logistic")].reset_index(drop=True)
    obs_auc = float(roc_auc_score(primary_preds["y_true"], primary_preds["y_prob"]))

    rng = np.random.default_rng(seed)
    perm_seeds = rng.integers(0, 2**31, size=B).tolist()

    print(f"  Refit permutations (B={B}) running in parallel …", flush=True)
    perm_aucs = np.array(
        Parallel(n_jobs=-1, backend="loky")(
            delayed(_refit_one_permutation)(
                s, data, event_dates, splits, combined_cols, h=10
            )
            for s in perm_seeds
        )
    )

    runtime_s = time.time() - t0
    p_value = float((1 + np.sum(perm_aucs >= obs_auc)) / (1 + B))
    return {
        "horizon": 10,
        "model": "logistic",
        "n_test": len(primary_preds),
        "n_permutations": B,
        "observed_auc": obs_auc,
        "perm_auc_mean": float(perm_aucs.mean()),
        "perm_auc_sd": float(perm_aucs.std()),
        "perm_auc_p95": float(np.percentile(perm_aucs, 95)),
        "p_value": p_value,
        "primary": True,
        "refit": True,
        "runtime_seconds": round(runtime_s, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=5000, help="Number of permutations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refit", action="store_true",
                        help="Full-refit mode: refit walk-forward for primary (h=10, logistic)")
    args = parser.parse_args()

    B = args.b
    seed = args.seed

    preds = pd.read_parquet(artifact_file("aspi_direction_predictions.parquet"))

    rows = []
    groups = list(preds.groupby(["horizon", "model"]))
    print(f"Running {B} permutations for {len(groups)} (horizon, model) combinations …")
    for key, group in sorted(groups):
        horizon, model = key
        is_primary = (horizon == 10 and model == "logistic")
        if args.refit and is_primary:
            # Primary gets full refit — handle after the loop
            continue
        print(f"  horizon={horizon}, model={model}, n={len(group)} ", end="", flush=True)
        result = _run_one(key, group.reset_index(drop=True), B, seed + hash(key) % 10000)
        result["refit"] = False
        result["runtime_seconds"] = float("nan")
        marker = " <-- PRIMARY (fixed)" if is_primary else ""
        print(f"AUC={result['observed_auc']:.4f}  p={result['p_value']:.4f}{marker}")
        rows.append(result)

    if args.refit:
        print(f"  horizon=10, model=logistic [FULL REFIT] …", flush=True)
        refit_result = _run_refit(B, seed + hash((10, "logistic")) % 10000)
        print(f"AUC={refit_result['observed_auc']:.4f}  p={refit_result['p_value']:.4f}"
              f"  runtime={refit_result['runtime_seconds']:.0f}s  <-- PRIMARY (refit)")
        rows.append(refit_result)

    out = pd.DataFrame(rows).sort_values(["horizon", "model"]).reset_index(drop=True)

    # Holm correction across all 10 comparisons
    from statsmodels.stats.multitest import multipletests
    _, p_holm, _, _ = multipletests(out["p_value"].to_numpy(), method="holm")
    out["p_holm"] = p_holm
    out["holm_significant"] = out["p_holm"] < 0.05

    if args.refit:
        artifact_name = "aspi_direction_permutation_refit"
        note = (
            f"Full-refit permutation test, B={B}, seed={seed}. "
            "Primary (h=10, logistic): entire walk-forward refit per permutation "
            "(CollinearityRFTopK k=10, StandardScaler, LogisticRegression max_iter=5000). "
            "Other 9 comparisons: fixed-prediction episode-level permutation. "
            "Holm correction across all 10 comparisons. "
            "refit=True marks the full-refit row."
        )
    else:
        artifact_name = "aspi_direction_permutation"
        note = (
            f"Episode-level permutation test, B={B}, seed={seed}. "
            "y_true permuted at episode level (one event per episode). "
            "p = (1 + count(perm_auc >= obs_auc)) / (1 + B). "
            "primary=True marks the pre-declared horizon=10 / logistic comparison."
        )

    save_frame(out, artifact_name, note=note)
    print(f"\nSaved: artifacts/tables/{artifact_name}.parquet")

    primary = out[out["primary"]]
    if not primary.empty:
        r = primary.iloc[0]
        refit_tag = " (refit)" if args.refit else ""
        print(f"\nPrimary (h=10, logistic){refit_tag}: AUC={r.observed_auc:.4f}  "
              f"p={r.p_value:.4f}  Holm_p={r.p_holm:.4f}  B={B}")


if __name__ == "__main__":
    main()
