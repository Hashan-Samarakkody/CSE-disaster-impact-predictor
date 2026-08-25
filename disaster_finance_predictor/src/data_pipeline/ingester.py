"""Data ingestion utilities."""

from __future__ import annotations

import pandas as pd


def merge_dataframes(high_frequency_df: pd.DataFrame, low_frequency_df: pd.DataFrame, on: str = "date") -> pd.DataFrame:
    """Merge high-frequency market data with lower-frequency disaster/macro data."""
    left = high_frequency_df.copy()
    right = low_frequency_df.copy()
    left[on] = pd.to_datetime(left[on])
    right[on] = pd.to_datetime(right[on])
    return pd.merge_asof(
        left.sort_values(on),
        right.sort_values(on),
        on=on,
        direction="backward",
    )
