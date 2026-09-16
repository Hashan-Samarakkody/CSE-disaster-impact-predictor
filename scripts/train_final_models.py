"""Fit and persist the classification and hurdle models the demo app serves.

`final_rf_models.pkl` (regression) is already produced by stage 04, refit on all real
rows for SHAP. Classification and the Y3 hurdle model are never refit that way anywhere
in the pipeline -- only per-fold out-of-fold predictions are kept -- so there is nothing
for a live demo to load. This script fits one final full-data model per classification
label (whichever family scored highest AUC in the recorded walk-forward audit, per
`docs/RESULTS_AUDIT.txt` conventions) and one final hurdle model, mirroring exactly the
"refit on all real rows" pattern stage 04 already uses.

These are demonstration artifacts, not new results: no number here is a held-out score,
and nothing in `artifacts/classification_summary.parquet` changes. Run after the full
pipeline (needs `dataset`, `feature_spec`, `classification_summary` cached):

    python scripts/train_final_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.classifiers import LABELS, build_classifiers
from src.models.hurdle import HurdleRecoveryModel

ARTIFACTS = ROOT / "artifacts"
RANDOM_STATE = 42


def _load_json(name):
    import json
    return json.loads((ARTIFACTS / f"{name}.json").read_text(encoding="utf-8"))


def _select_top_features(X, target, k=20, random_state=RANDOM_STATE):
    """RF-importance top-K on the full table, mirroring each notebook's per-fold version."""
    k = min(k, X.shape[1])
    probe = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3,
                                   random_state=random_state, n_jobs=-1)
    probe.fit(X, target)
    order = np.argsort(probe.feature_importances_)[::-1][:k]
    return list(X.columns[order])


def fit_final_classifiers(X: pd.DataFrame, y: pd.DataFrame, dataset: pd.DataFrame) -> dict:
    summary = pd.read_parquet(ARTIFACTS / "classification_summary.parquet")
    summary = summary[summary["model"] != "majority_baseline"]
    best_family = summary.loc[summary.groupby("label")["auc"].idxmax()].set_index("label")

    families = build_classifiers(random_state=RANDOM_STATE)
    final = {}
    for name, label_fn in LABELS.items():
        labels_all = label_fn(y, np.arange(len(y)), dataset)
        mask = labels_all.notna().to_numpy()
        Xl, yl = X.loc[mask], labels_all.loc[mask].to_numpy()

        row = best_family.loc[name]
        family = row["model"]
        feats = _select_top_features(Xl, yl)
        model = clone(families[family][0])
        model.fit(Xl[feats], yl)

        final[name] = {
            "model": model, "features": feats, "family": family,
            "historical_auc": float(row["auc"]),
            "historical_balanced_accuracy": float(row["balanced_accuracy"]),
            "historical_prevalence": float(row["prevalence"]),
            "beats_baseline": bool(row["beats_baseline"]),
        }
        print(f"  {name:22s} family={family:9s} auc={row['auc']:.3f} "
              f"beats_baseline={bool(row['beats_baseline'])}")
    return final


def fit_final_hurdle(X: pd.DataFrame, y: pd.DataFrame, dataset: pd.DataFrame) -> dict:
    target = "Y3_recovery_days"
    ok = y[target].notna().to_numpy()
    y_ok = y.loc[ok, target].to_numpy()
    # Real competing-risk censoring indicator (methodology-audit finding #7), not
    # `y_ok < 90` alone -- see AFTRecoveryModel/HurdleRecoveryModel.fit docstrings.
    recovered_ok = ~dataset["Y3_censored"].loc[y.loc[ok].index].to_numpy()
    feats = _select_top_features(X.loc[ok], recovered_ok.astype(int))

    model = HurdleRecoveryModel(
        LogisticRegression(max_iter=5000, class_weight="balanced"),
        RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=3,
                              random_state=RANDOM_STATE, n_jobs=-1),
    ).fit(X.loc[ok, feats], y_ok, recovered=recovered_ok)
    return {"model": model, "features": feats}


def main() -> None:
    dataset = pd.read_parquet(ARTIFACTS / "dataset.parquet")
    spec = _load_json("feature_spec")
    X = dataset[spec["FEATURE_COLS"]].fillna(0.0)
    y = dataset[spec["TARGET_COLS"]].copy()

    print("Fitting final classifiers (one per label, on all real rows)...")
    final_classifiers = fit_final_classifiers(X, y, dataset)

    print("Fitting final Y3 hurdle model...")
    final_hurdle = fit_final_hurdle(X, y, dataset)

    import pickle
    with open(ARTIFACTS / "final_classifiers.pkl", "wb") as fh:
        pickle.dump(final_classifiers, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ARTIFACTS / "final_hurdle_model.pkl", "wb") as fh:
        pickle.dump(final_hurdle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print("\nSaved artifacts/final_classifiers.pkl and artifacts/final_hurdle_model.pkl")


if __name__ == "__main__":
    main()
