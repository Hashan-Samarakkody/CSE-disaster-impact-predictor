"""Build window_spans.parquet — training and test window metadata per fold."""

from __future__ import annotations

import json
import pickle
import pandas as pd
import numpy as np

from src.utils.artifact_store import artifact_file, save_frame


def main():
    with open(artifact_file("splits.pkl"), "rb") as f:
        splits = pickle.load(f)

    dataset = pd.read_parquet(artifact_file("dataset.parquet"))
    event_dates = pd.to_datetime(dataset["event_date"])
    disaster_types = dataset["disaster_type"].fillna("Other")

    rows = []
    for fold_idx, fold in enumerate(splits):
        train_idx = fold.train_index
        test_idx = fold.test_index

        train_dates = event_dates.iloc[train_idx]
        test_dates = event_dates.iloc[test_idx]

        train_span_years = float((train_dates.max() - train_dates.min()).days / 365.25)
        test_span_years = float((test_dates.max() - test_dates.min()).days / 365.25)

        # Hazard counts by type in test fold
        test_types = disaster_types.iloc[test_idx]
        hazard_counts = {}
        for dtype in ["Flood", "Storm", "Drought"]:
            hazard_counts[dtype.lower()] = int((test_types == dtype).sum())
        other_count = int(len(test_types) - sum(hazard_counts.values()))
        hazard_counts["other"] = other_count

        rows.append({
            "fold": fold_idx,
            "train_event_first": int(train_idx[0]),
            "train_event_last": int(train_idx[-1]),
            "train_date_first": train_dates.min(),
            "train_date_last": train_dates.max(),
            "train_span_years": train_span_years,
            "test_date_first": test_dates.min(),
            "test_date_last": test_dates.max(),
            "test_span_years": test_span_years,
            "n_hazard_by_type": json.dumps(hazard_counts),
        })

    df = pd.DataFrame(rows)
    df["train_date_first"] = pd.to_datetime(df["train_date_first"])
    df["train_date_last"] = pd.to_datetime(df["train_date_last"])
    df["test_date_first"] = pd.to_datetime(df["test_date_first"])
    df["test_date_last"] = pd.to_datetime(df["test_date_last"])

    save_frame(df, "window_spans",
               note="Training and test window metadata per walk-forward fold. "
                    "n_hazard_by_type is JSON of disaster type counts in the test fold.")


if __name__ == "__main__":
    main()
