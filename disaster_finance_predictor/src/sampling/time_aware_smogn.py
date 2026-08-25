"""Time-aware SMOGN style oversampling utilities for regression events."""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_aware_smogn(
    X: pd.DataFrame,
    y: pd.DataFrame,
    minority_mask: pd.Series,
    noise_scale: float = 0.01,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic minority samples while preserving chronological order."""
    rng = np.random.default_rng(random_state)
    X_syn = X.copy()
    y_syn = y.copy()

    minority_idx = X.index[minority_mask]
    if len(minority_idx) < 2:
        return X_syn, y_syn

    synthetic_rows_X = []
    synthetic_rows_y = []

    for idx in minority_idx:
        row_x = X.loc[idx]
        row_y = y.loc[idx]
        noisy_x = row_x + rng.normal(0.0, noise_scale, size=len(row_x))
        noisy_y = row_y + rng.normal(0.0, noise_scale, size=len(row_y))
        synthetic_rows_X.append(noisy_x)
        synthetic_rows_y.append(noisy_y)

    syn_x_df = pd.DataFrame(synthetic_rows_X, columns=X.columns)
    syn_y_df = pd.DataFrame(synthetic_rows_y, columns=y.columns)

    X_syn = pd.concat([X_syn, syn_x_df], ignore_index=True)
    y_syn = pd.concat([y_syn, syn_y_df], ignore_index=True)

    return X_syn, y_syn
