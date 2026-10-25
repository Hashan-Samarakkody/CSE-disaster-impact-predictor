"""Build prediction_ledger.parquet — full per-event prediction log."""

from __future__ import annotations

import pickle
import numpy as np
import pandas as pd

from src.evaluation.verification import build_episode_ids
from src.utils.artifact_store import artifact_file, save_frame


def main():
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))
    event_dates = pd.to_datetime(dataset["event_date"])
    n_events = len(dataset)

    # Compute episode IDs
    episode_ids = build_episode_ids(event_dates, gap_days=14)

    # Load splits for bench_train_mean/median per fold
    with open(artifact_file("splits.pkl"), "rb") as f:
        splits = pickle.load(f)

    # Build per-event lookup: event_id -> episode_id, event_date, disaster_type, date_is_exact
    event_meta = dataset[["event_date", "disaster_type", "date_is_exact"]].copy()
    event_meta["episode_id"] = episode_ids
    event_meta["event_id"] = np.arange(n_events)
    event_meta["event_date"] = pd.to_datetime(event_meta["event_date"])

    # --- Y1 from aspi_grid_predictions ---
    gp = pd.read_parquet(artifact_file("aspi_grid_predictions.parquet"))

    # We want only horizons matching Y1 primary (5) and 10, using 'ex_post' info_set
    # But keep all for the ledger
    rows_y1 = []
    for _, pred_row in gp.iterrows():
        row_idx = int(pred_row["row"])
        if row_idx >= n_events:
            continue

        meta = event_meta.iloc[row_idx]
        horizon = int(pred_row["horizon"])
        info_set = str(pred_row["info_set"])
        k = int(pred_row["k"])
        fold = int(pred_row["fold"])
        model = str(pred_row["model"])
        y_true = float(pred_row["y_true"])
        y_pred = float(pred_row["y_pred"])

        # Determine target name
        target = f"Y1_ASPI_{horizon}D_Forward_LogReturn_Pct"

        # bench_train_mean/median from training fold
        fold_data = splits[fold]
        train_mask = fold_data.train_index
        y1_col = target if target in dataset.columns else "Y1_ASPI_5D_Forward_LogReturn_Pct"
        if y1_col in dataset.columns:
            train_vals = dataset[y1_col].iloc[train_mask].dropna()
            bench_train_mean = float(train_vals.mean()) if len(train_vals) > 0 else float("nan")
            bench_train_median = float(train_vals.median()) if len(train_vals) > 0 else float("nan")
        else:
            bench_train_mean = float("nan")
            bench_train_median = float("nan")

        # outcome_resolution_date: Y1_horizon_end_date if available
        horizon_end_col = f"Y1_{horizon}D_horizon_end_date" if horizon != 5 else "Y1_horizon_end_date"
        if horizon_end_col in dataset.columns:
            outcome_resolution_date = dataset[horizon_end_col].iloc[row_idx]
        else:
            outcome_resolution_date = pd.NaT

        rows_y1.append({
            "event_id": row_idx,
            "event_date": meta["event_date"],
            "disaster_type": meta["disaster_type"],
            "date_is_exact": meta["date_is_exact"],
            "episode_id": int(meta["episode_id"]),
            "horizon": horizon,
            "info_set": info_set,
            "k": k,
            "fold": fold,
            "target": target,
            "model": model,
            "y_true": y_true,
            "y_pred": y_pred,
            "bench_zero": 0.0,
            "bench_train_mean": bench_train_mean,
            "bench_train_median": bench_train_median,
            "outcome_resolution_date": outcome_resolution_date,
        })

    df_y1 = pd.DataFrame(rows_y1)

    # --- Y3 from recovery_grid_predictions ---
    rgp = pd.read_parquet(artifact_file("recovery_grid_predictions.parquet"))

    rows_y3 = []
    for _, pred_row in rgp.iterrows():
        row_idx = int(pred_row["row"])
        if row_idx >= n_events:
            continue

        meta = event_meta.iloc[row_idx]
        fold = int(pred_row["fold"])
        model = str(pred_row["model"])

        # Compute bench_train_mean/median for Y3
        fold_data = splits[fold]
        train_mask = fold_data.train_index
        y3_col = "Y3_ASPI_Recovery_Time"
        if y3_col in dataset.columns:
            train_vals = dataset[y3_col].iloc[train_mask].dropna()
            bench_train_mean_y3 = float(train_vals.mean()) if len(train_vals) > 0 else float("nan")
            bench_train_median_y3 = float(train_vals.median()) if len(train_vals) > 0 else float("nan")
        else:
            bench_train_mean_y3 = float("nan")
            bench_train_median_y3 = float("nan")

        # outcome resolution date: Y3_label_end_date if available
        if "Y3_label_end_date" in dataset.columns:
            outcome_resolution_date = dataset["Y3_label_end_date"].iloc[row_idx]
        else:
            outcome_resolution_date = pd.NaT

        rows_y3.append({
            "event_id": row_idx,
            "event_date": meta["event_date"],
            "disaster_type": meta["disaster_type"],
            "date_is_exact": meta["date_is_exact"],
            "episode_id": int(meta["episode_id"]),
            "horizon": 90,  # Y3 cap
            "info_set": "ex_post",
            "k": 20,
            "fold": fold,
            "target": "Y3_ASPI_Recovery_Time",
            "model": model,
            "y_true": float(pred_row["duration"]),
            "y_pred": float(pred_row["pred_median"]),
            "bench_zero": float("nan"),  # zero not meaningful for recovery time
            "bench_train_mean": bench_train_mean_y3,
            "bench_train_median": bench_train_median_y3,
            "outcome_resolution_date": outcome_resolution_date,
        })

    df_y3 = pd.DataFrame(rows_y3)

    # Combine
    df = pd.concat([df_y1, df_y3], ignore_index=True)
    df["event_date"] = pd.to_datetime(df["event_date"])

    # Reorder columns
    cols = [
        "event_id", "event_date", "disaster_type", "date_is_exact", "episode_id",
        "horizon", "info_set", "k", "fold", "target", "model",
        "y_true", "y_pred", "bench_zero", "bench_train_mean", "bench_train_median",
        "outcome_resolution_date",
    ]
    df = df[cols]

    save_frame(df, "prediction_ledger",
               note="Full prediction log. Y1 rows from aspi_grid_predictions.parquet; "
                    "Y3 rows from recovery_grid_predictions.parquet. "
                    "bench_zero=0.0 for Y1 (NaN for Y3). episode_id from build_episode_ids "
                    "with gap_days=14.")


if __name__ == "__main__":
    main()
