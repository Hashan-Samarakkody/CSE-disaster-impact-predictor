"""Chronological walk-forward validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class WalkForwardSplit:
    train_index: np.ndarray
    test_index: np.ndarray


def generate_walk_forward_splits(
    n_samples: int,
    train_window: int,
    test_window: int,
    step: int,
) -> Iterator[WalkForwardSplit]:
    """Yield rolling train/test windows for time-series cross-validation."""
    start = 0
    while start + train_window + test_window <= n_samples:
        train_idx = np.arange(start, start + train_window)
        test_idx = np.arange(start + train_window, start + train_window + test_window)
        yield WalkForwardSplit(train_idx, test_idx)
        start += step


def purge_horizon_overlap(
    split: WalkForwardSplit,
    event_dates: pd.Series,
    horizon_end_dates: pd.Series,
) -> WalkForwardSplit:
    """Drop training rows whose label horizon reaches into the test period.

    Y1 (`ASPI_5D_Log_Return_Pct`) is built from market prices up to 5 trading days
    *after* each event. Splits are by event index, not calendar date, so a training
    event that lands within ~5 trading days of the first test event can have a label
    computed from prices dated on/after that test event's own reference date -- a
    fold-boundary embargo violation (see Lopez de Prado, "Advances in Financial
    Machine Learning", ch. 7). Purge those training rows for this fold only; the test
    set and every other target are untouched.
    """
    if len(split.test_index) == 0 or len(split.train_index) == 0:
        return split
    first_test_date = event_dates.iloc[split.test_index[0]]
    train_horizon_end = horizon_end_dates.iloc[split.train_index]
    keep = train_horizon_end.isna() | (train_horizon_end < first_test_date)
    return WalkForwardSplit(split.train_index[keep.to_numpy()], split.test_index)


if __name__ == "__main__":
    dates = pd.Series(pd.date_range("2020-01-01", periods=10, freq="D"))
    horizon_end = dates + pd.Timedelta(days=3)  # each event's label reaches 3 days out
    split = WalkForwardSplit(np.arange(0, 6), np.arange(6, 10))
    purged = purge_horizon_overlap(split, dates, horizon_end)
    # events 3,4,5 (0-indexed) have horizon_end >= dates[6] (first test date) -> dropped
    assert list(purged.train_index) == [0, 1, 2], purged.train_index
    assert list(purged.test_index) == list(split.test_index)
    # no-overlap case: nothing purged
    horizon_end_tight = dates + pd.Timedelta(hours=1)
    purged2 = purge_horizon_overlap(split, dates, horizon_end_tight)
    assert list(purged2.train_index) == list(split.train_index)
    print("walk_forward.py self-check passed")
