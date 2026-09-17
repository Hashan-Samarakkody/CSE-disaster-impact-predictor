"""Preprocessing utilities with chronological-safe transforms."""

from __future__ import annotations

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def forward_fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing values to preserve chronological continuity."""
    return df.sort_index().ffill()


def truncate_overlapping_windows(events_df: pd.DataFrame, date_col: str = "event_date", horizon_days: int = 90) -> pd.DataFrame:
    """Truncate a primary event window if a secondary event occurs within the horizon."""
    df = events_df.copy().sort_values(date_col)
    df[date_col] = pd.to_datetime(df[date_col])
    df["window_end"] = df[date_col] + pd.to_timedelta(horizon_days, unit="D")
    for i in range(len(df) - 1):
        current_end = df.iloc[i]["window_end"]
        next_start = df.iloc[i + 1][date_col]
        if next_start <= current_end:
            df.at[df.index[i], "window_end"] = next_start - pd.Timedelta(days=1)
    return df


def build_scaler(name: str = "standard") -> TransformerMixin:
    """Create scaler objects for fold-local fitting in walk-forward validation."""
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unsupported scaler '{name}'.")
