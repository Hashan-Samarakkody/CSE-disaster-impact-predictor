"""Purged inner cross validation for hyperparameter search.

The outer walk forward purge only protects the outer test period. During a
hyperparameter search an inner training row's label can still reach into the
inner validation window, which is the same leak one level deeper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def purged_inner_cv(dates_real, label_end_real, n_splits: int = 3):
    """Time ordered inner splits with each split purged against its own validation origin."""
    n_rows = len(dates_real)
    splitter = TimeSeriesSplit(n_splits=max(2, min(n_splits, n_rows - 1)))
    dates = pd.Series(dates_real).reset_index(drop=True)
    label_end = pd.Series(label_end_real).reset_index(drop=True)

    raw_splits = list(splitter.split(np.arange(n_rows)))
    purged = []
    for train_idx, val_idx in raw_splits:
        if len(val_idx) == 0 or len(train_idx) == 0:
            continue
        first_val_date = dates.iloc[val_idx[0]]
        train_end = label_end.iloc[train_idx]
        keep = (train_end.isna() | (train_end < first_val_date)).to_numpy()
        if keep.any():
            purged.append((train_idx[keep], val_idx))

    if not purged:
        print("purged_inner_cv: every inner split lost its entire training side, "
              "falling back to the unpurged splits for this fold and target.")
        return raw_splits
    return purged


def purged_inner_splits(dates, label_end, n_splits: int = 3, min_train_rows: int = 5):
    """Stricter variant: reduce the split count rather than ever dropping the purge.

    Returns an empty list when no purged configuration survives, which tells the
    caller to fall back to pre specified default hyperparameters instead.
    """
    n_rows = len(dates)
    dates = pd.Series(dates).reset_index(drop=True)
    label_end = pd.Series(label_end).reset_index(drop=True)

    for count in range(min(n_splits, n_rows - 1), 1, -1):
        purged = []
        for train_idx, val_idx in TimeSeriesSplit(n_splits=count).split(np.arange(n_rows)):
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            first_val_date = dates.iloc[val_idx[0]]
            train_end = label_end.iloc[train_idx]
            keep = (train_end.isna() | (train_end < first_val_date)).to_numpy()
            if keep.sum() >= min_train_rows:
                purged.append((train_idx[keep], val_idx))
        if len(purged) >= 2:
            return purged
    return []
