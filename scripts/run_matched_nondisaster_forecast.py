"""Matched non-disaster forecast: item 14.

Compare model performance on disaster episodes vs matched non-disaster (placebo) dates.

For non-disaster dates: load the 74 placebo session indices from the provenance JSON,
extract market features at those dates, compute Y1_5D forward log returns, then run
cross-validated Ridge and Random Forest using the same 4-fold walk-forward structure.

For disaster dates: take results from aspi_grid_predictions for h=5, info_set=real_time,
k=10, models ridge and random_forest.

Output:
  origin_type: 'disaster' or 'matched_control'
  info_set: 'real_time'
  model: model name
  n: number of predictions
  rmse, mae, pooled_r2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.artifact_store import artifact_file, save_frame
from src.training.walk_forward import generate_walk_forward_splits

RANDOM_STATE = 42
TRAIN_WINDOW = 30
TEST_WINDOW = 10
STEP = 10
K = 10  # number of features
HORIZON = 5  # sessions forward

# Top-K market-only features (from feature_spec.json market_only set, first 10)
MARKET_FEATURES = [
    "log_return", "lag_return_t-1", "lag_return_t-2", "lag_return_t-3",
    "rolling_std_20", "rolling_std_30", "garch_cond_vol",
    "price_to_sma_20", "sp500_log_return", "vol_ratio_5_30",
]


def compute_forward_return(market_feats: pd.DataFrame, session_idx: int, h: int) -> float:
    """Forward log return over h sessions starting at session_idx."""
    # Cumulative log return from session_idx to session_idx+h-1 inclusive
    end = session_idx + h
    if end > len(market_feats):
        return float("nan")
    log_rets = market_feats["log_return"].iloc[session_idx:end].to_numpy(float)
    if not np.all(np.isfinite(log_rets)):
        return float("nan")
    # Multiply by 100 to match the disaster-target scale (Y1_*_Pct is in percent)
    return float(np.sum(log_rets) * 100.0)


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def _pooled_r2(y_true, y_pred):
    yt = np.array(y_true)
    yp = np.array(y_pred)
    ss_res = np.sum((yp - yt) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def build_estimator(name: str):
    """Build unfitted estimator (no tuning, consistent with real_time info_set)."""
    if name == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    elif name == "random_forest":
        return RandomForestRegressor(n_estimators=200, max_depth=3,
                                     min_samples_leaf=4, random_state=RANDOM_STATE)
    elif name == "mlp":
        return Pipeline([("scale", StandardScaler()),
                         ("model", MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0,
                                                max_iter=4000, random_state=RANDOM_STATE))])
    raise ValueError(name)


def run_placebo_cv(market_feats: pd.DataFrame, placebo_indices: list,
                   feature_cols: list, horizon: int) -> dict:
    """Walk-forward CV on placebo sessions.

    For each of the 4 folds, use all placebo sessions before the fold boundary as
    training, placebo sessions in the fold window as test, fit models and predict.
    Returns predictions dict: model_name -> list of (y_true, y_pred).
    """
    # Compute target Y for each placebo session
    placebo_indices = sorted(placebo_indices)
    records = []
    for idx in placebo_indices:
        y = compute_forward_return(market_feats, idx, horizon)
        if not np.isfinite(y):
            continue
        feats = market_feats.iloc[idx][feature_cols]
        if feats.isna().all():
            continue
        records.append({"session_idx": idx, "y": y, **feats.to_dict()})

    if not records:
        return {}

    df = pd.DataFrame(records).sort_values("session_idx").reset_index(drop=True)
    y_all = df["y"].to_numpy(float)
    X_all = df[feature_cols].fillna(0.0).to_numpy(float)
    n = len(df)

    splits = list(generate_walk_forward_splits(n, TRAIN_WINDOW, TEST_WINDOW, STEP))
    if not splits:
        # Not enough data — use simple 60/40 split
        cut = max(int(n * 0.6), 1)
        from src.training.walk_forward import WalkForwardSplit
        splits = [WalkForwardSplit(train_index=np.arange(cut), test_index=np.arange(cut, n))]

    MODEL_NAMES = ["ridge", "random_forest", "mlp"]
    all_preds = {m: {"y_true": [], "y_pred": []} for m in MODEL_NAMES}

    for split in splits:
        tr, te = split.train_index, split.test_index
        if len(tr) < 5 or len(te) == 0:
            continue
        X_tr, X_te = X_all[tr], X_all[te]
        y_tr, y_te = y_all[tr], y_all[te]

        for name in MODEL_NAMES:
            est = build_estimator(name)
            try:
                est.fit(X_tr, y_tr)
                pred = est.predict(X_te)
                all_preds[name]["y_true"].extend(y_te.tolist())
                all_preds[name]["y_pred"].extend(pred.tolist())
            except Exception as e:
                print(f"    {name} error: {e}")

    return all_preds


def main() -> None:
    market_feats = pd.read_parquet(artifact_file("market_feats.parquet"))
    # Join sp500_log_return for completeness (not in market_feats by default)
    if "sp500_log_return" not in market_feats.columns:
        try:
            sp500 = pd.read_parquet(artifact_file("sp500.parquet"))
            from src.targets.abnormal_returns import align_market_factor
            market_feats["sp500_log_return"] = align_market_factor(
                market_feats["date"], sp500["date"], sp500["sp500_log_return"])
            print("Joined sp500_log_return into market_feats")
        except Exception as e:
            print(f"Could not join sp500_log_return: {e}")
    preds_grid = pd.read_parquet(artifact_file("aspi_grid_predictions.parquet"))

    # Load placebo session indices
    prov_path = artifact_file("placebo_event_study_provenance.json")
    if not prov_path.exists():
        raise FileNotFoundError("placebo_event_study_provenance.json not found. "
                                "Run scripts/run_placebo_event_study.py first.")
    provenance = json.loads(prov_path.read_text(encoding="utf-8"))
    placebo_indices = provenance["placebo_session_indices"]
    print(f"Loaded {len(placebo_indices)} placebo session indices")

    # Verify features exist in market_feats
    feat_cols = [f for f in MARKET_FEATURES if f in market_feats.columns]
    missing = [f for f in MARKET_FEATURES if f not in market_feats.columns]
    if missing:
        print(f"  Warning: missing features {missing} — skipping them")
    print(f"  Using {len(feat_cols)} features: {feat_cols}")

    rows = []

    # ── Non-disaster (placebo) performance ───────────────────────────────────────
    print("Running walk-forward CV on placebo sessions ...")
    placebo_preds = run_placebo_cv(market_feats, placebo_indices, feat_cols, HORIZON)

    MODEL_NAMES = ["ridge", "random_forest", "mlp"]
    for model_name in MODEL_NAMES:
        if model_name not in placebo_preds or not placebo_preds[model_name]["y_true"]:
            continue
        yt = np.array(placebo_preds[model_name]["y_true"])
        yp = np.array(placebo_preds[model_name]["y_pred"])
        rows.append({
            "origin_type": "matched_control",
            "info_set": "real_time",
            "model": model_name,
            "n": len(yt),
            "rmse": _rmse(yt, yp),
            "mae": _mae(yt, yp),
            "pooled_r2": _pooled_r2(yt, yp),
        })
        print(f"  [placebo] {model_name}: n={len(yt)} rmse={_rmse(yt, yp):.4f}")

    # ── Disaster performance (from aspi_grid_predictions) ────────────────────────
    print("Extracting disaster-date performance from aspi_grid_predictions ...")
    for model_name in MODEL_NAMES:
        g = preds_grid[
            (preds_grid.horizon == HORIZON) &
            (preds_grid.info_set == "real_time") &
            (preds_grid.k == K) &
            (preds_grid.model == model_name)
        ].copy()
        if len(g) == 0:
            print(f"  [disaster] {model_name}: no rows found")
            continue
        yt = g["y_true"].to_numpy(float)
        yp = g["y_pred"].to_numpy(float)
        rows.append({
            "origin_type": "disaster",
            "info_set": "real_time",
            "model": model_name,
            "n": len(g),
            "rmse": _rmse(yt, yp),
            "mae": _mae(yt, yp),
            "pooled_r2": _pooled_r2(yt, yp),
        })
        print(f"  [disaster] {model_name}: n={len(g)} rmse={_rmse(yt, yp):.4f}")

    out = pd.DataFrame(rows)
    save_frame(out, "matched_nondisaster_forecast",
               f"Model performance on disaster (h={HORIZON}, k={K}, real_time) vs "
               f"matched non-disaster placebo dates. "
               f"Placebo: same walk-forward structure (train={TRAIN_WINDOW}, "
               f"test={TEST_WINDOW}) with features {feat_cols}.")
    print(f"\nwrote matched_nondisaster_forecast.parquet ({len(out)} rows)")


if __name__ == "__main__":
    main()
