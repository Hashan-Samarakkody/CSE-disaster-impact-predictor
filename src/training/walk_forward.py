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
    mode: str = "sliding",
) -> Iterator[WalkForwardSplit]:
    """Yield chronological train/test windows for time-series cross-validation.

    `mode` is "sliding", where the training block is a fixed width and the earliest events
    leave it, or "expanding", where it starts at the first event and grows. The test block
    and the step behave identically either way. Sliding is the default, so every existing
    caller is unchanged (Revision 2, T9).
    """
    if mode not in {"sliding", "expanding"}:
        raise ValueError(f"mode must be 'sliding' or 'expanding', got {mode!r}")
    start = 0
    while start + train_window + test_window <= n_samples:
        train_start = 0 if mode == "expanding" else start
        train_idx = np.arange(train_start, start + train_window)
        test_idx = np.arange(start + train_window, start + train_window + test_window)
        yield WalkForwardSplit(train_idx, test_idx)
        start += step


def purge_horizon_overlap(
    split: WalkForwardSplit,
    event_dates: pd.Series,
    horizon_end_dates: pd.Series,
) -> WalkForwardSplit:
    """Drop training rows whose label horizon reaches into the test period."""
    if len(split.test_index) == 0 or len(split.train_index) == 0:
        return split
    first_test_date = event_dates.iloc[split.test_index[0]]
    train_horizon_end = horizon_end_dates.iloc[split.train_index]
    keep = train_horizon_end.isna() | (train_horizon_end < first_test_date)
    return WalkForwardSplit(split.train_index[keep.to_numpy()], split.test_index)


# Feature columns that carry a real gap (a paired `_available`/`_observed` flag exists
MEDIAN_IMPUTE_COLS = [
    # financial_damage NaN propagates into these two DERIVED columns (log1p, then a
    "financial_damage", "log_financial_damage", "log_damage_x_flood",
    "total_deaths", "no_homeless", "mag_area_km2", "mag_wind_kph",
    "vol_ratio_1_30", "vol_ratio_5_30", "vol_ratio_10_30", "vol_cv_30", "log_vol_change_1",
    "garch_cond_vol", "gdp_growth_pct", "inflation_cpi_pct", "damage_to_gdp",
]


def median_impute_from_train(
    train_df: pd.DataFrame, *other_dfs: pd.DataFrame, cols: list[str] = MEDIAN_IMPUTE_COLS,
) -> tuple[pd.DataFrame, ...]:
    """Median of `cols`, computed from `train_df` (real training rows for THIS fold and"""
    present = [c for c in cols if c in train_df.columns]
    medians = train_df[present].median().fillna(0.0)
    dfs = (train_df,) + other_dfs
    return tuple(d.assign(**{c: d[c].fillna(medians[c]) for c in present if c in d.columns})
                 for d in dfs)


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

    train = pd.DataFrame({"financial_damage": [1.0, 3.0, np.nan, 5.0], "other": [1, 2, 3, 4]})
    test = pd.DataFrame({"financial_damage": [np.nan, 9.0], "other": [5, 6]})
    tr_filled, te_filled = median_impute_from_train(train, test, cols=["financial_damage"])
    # median of [1,3,5] (the NaN is excluded from the median itself) is 3.0
    assert tr_filled["financial_damage"].tolist() == [1.0, 3.0, 3.0, 5.0]
    assert te_filled["financial_damage"].tolist() == [3.0, 9.0]
    assert tr_filled["other"].tolist() == [1, 2, 3, 4]        # untouched, not in cols
    # a column entirely NaN in train falls back to 0.0, not NaN
    all_nan = pd.DataFrame({"financial_damage": [np.nan, np.nan]})
    filled, = median_impute_from_train(all_nan, cols=["financial_damage"])
    assert filled["financial_damage"].tolist() == [0.0, 0.0]

    print("walk_forward.py self-check passed")
