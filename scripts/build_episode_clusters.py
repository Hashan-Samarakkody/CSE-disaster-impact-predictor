"""Build episode_clusters.parquet — disaster episodes grouped by temporal proximity."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.evaluation.verification import build_episode_ids
from src.utils.artifact_store import artifact_file, save_frame


def main():
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))
    event_dates = pd.to_datetime(dataset["event_date"])

    episode_ids = build_episode_ids(event_dates, gap_days=14)
    dataset["_episode_id"] = episode_ids
    dataset["_row_idx"] = np.arange(len(dataset))

    rows = []
    for ep_id, grp in dataset.groupby("_episode_id"):
        event_ids_str = ",".join(str(i) for i in grp["_row_idx"].tolist())
        dates = pd.to_datetime(grp["event_date"])
        first_date = dates.min()
        last_date = dates.max()
        span_days = int((last_date - first_date).days)
        rows.append({
            "episode_id": int(ep_id),
            "event_ids": event_ids_str,
            "size": len(grp),
            "first_date": first_date,
            "last_date": last_date,
            "span_days": span_days,
        })

    df = pd.DataFrame(rows)
    df["first_date"] = pd.to_datetime(df["first_date"])
    df["last_date"] = pd.to_datetime(df["last_date"])

    save_frame(df, "episode_clusters",
               note="Bootstrap conditions on fitted predictions and does not propagate "
                    "model-fitting or selection uncertainty. The episode-clustered bootstrap "
                    "resamples whole episodes, not individual events. Episodes defined as "
                    "groups of disaster events within 14 days of each other (gap_days=14). "
                    "event_ids is a comma-separated string of dataset row indices.")


if __name__ == "__main__":
    main()
