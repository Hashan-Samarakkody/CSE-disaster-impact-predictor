"""Fit and persist the classification and hurdle models the demo app serves."""

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

from src.evaluation.collinearity import redundant_drop_set
from src.models.classifiers import LABELS, build_classifiers
from src.models.hurdle import HurdleRecoveryModel
from src.utils.artifact_store import artifact_file

ARTIFACTS = ROOT / "artifacts"
RANDOM_STATE = 42


def _load_json(name):
    import json
    return json.loads((artifact_file(f"{name}.json")).read_text(encoding="utf-8"))


def _select_top_features(X, target, k=20, random_state=RANDOM_STATE):
    """Collinearity-drop, THEN RF-importance top-K on the full table, mirroring
    notebook 05's per-fold version (same |rho|>=0.95 pre-declared rule, see
    src/evaluation/collinearity.py)."""
    keep, _dropped, _detail = redundant_drop_set(X)
    X = X[keep]
    k = min(k, X.shape[1])
    probe = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3,
                                   random_state=random_state, n_jobs=-1)
    probe.fit(X, target)
    order = np.argsort(probe.feature_importances_)[::-1][:k]
    return list(X.columns[order])


def fit_final_classifiers(X: pd.DataFrame, y: pd.DataFrame, dataset: pd.DataFrame) -> dict:
    summary = pd.read_parquet(artifact_file("classification_summary.parquet"))
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
    target = "Y3_ASPI_Recovery_Time"
    ok = y[target].notna().to_numpy()
    y_ok = y.loc[ok, target].to_numpy()
    # Real competing-risk censoring indicator (methodology-audit finding #7), not
    # `y_ok < 90` alone, see AFTRecoveryModel/HurdleRecoveryModel.fit docstrings.
    recovered_ok = ~dataset["Y3_censored"].loc[y.loc[ok].index].to_numpy()
    feats = _select_top_features(X.loc[ok], recovered_ok.astype(int))

    model = HurdleRecoveryModel(
        LogisticRegression(max_iter=5000, class_weight="balanced"),
        RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=3,
                              random_state=RANDOM_STATE, n_jobs=-1),
    ).fit(X.loc[ok, feats], y_ok, recovered=recovered_ok)
    return {"model": model, "features": feats}


def main() -> None:
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))
    spec = _load_json("feature_spec")
    # Global (all-real-rows) median for the flagged columns (methodology-audit finding
    X = dataset[spec["FEATURE_COLS"]].fillna(spec.get("MEDIAN_IMPUTE_VALUES", {})).fillna(0.0)
    y = dataset[spec["TARGET_COLS"]].copy()

    print("Fitting final classifiers (one per label, on all real rows)...")
    final_classifiers = fit_final_classifiers(X, y, dataset)

    print("Fitting final Y3 hurdle model...")
    final_hurdle = fit_final_hurdle(X, y, dataset)

    import pickle
    with open(artifact_file("final_classifiers.pkl"), "wb") as fh:
        pickle.dump(final_classifiers, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(artifact_file("final_hurdle_model.pkl"), "wb") as fh:
        pickle.dump(final_hurdle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved {artifact_file('final_classifiers.pkl')} and "
          f"{artifact_file('final_hurdle_model.pkl')}")


if __name__ == "__main__":
    main()
